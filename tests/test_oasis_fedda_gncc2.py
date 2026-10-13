"""Squared GNCC formulas, FeDDA protocol, legacy safety and three-loss report."""
import copy
import json
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
sys.path.insert(0, str(ROOT/'scripts/reports'))
import fedda_gncc2_training as g
import fedda_gncc_training as signed
import fedda_training as native
import run_oasis_fedda_training as old_runner
import run_oasis_fedda_gncc2_training as runner
import run_oasis_fedda_gncc2_eval as evaluator
import launch_oasis_fedda_gncc2 as launcher
import oasis_fedda_three_ncc_report as report


class SquaredGNCC(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def tensors(self):
        return (torch.randn(2, 1, 3, 4, 5, dtype=torch.double),
                torch.randn(2, 1, 3, 4, 5, dtype=torch.double, requires_grad=True))

    def test_formula_and_gradients(self):
        a, b = self.tensors()
        expected = torch.stack([1-torch.corrcoef(torch.stack((x.flatten(), y.flatten())))[0, 1].square()
                                for x, y in zip(a, b)]).mean()
        value = g.GlobalNCCSquared()(a, b)
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        actual_grad, = torch.autograd.grad(value, b, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, b)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)

    def test_squaring_before_batch_mean_and_sign(self):
        a, _ = self.tensors()
        b = torch.cat([a[:1], -a[1:]])
        loss = g.GlobalNCCSquared()
        self.assertAlmostEqual(float(loss(a, b)), 0., places=12)
        self.assertAlmostEqual(float(1-loss.correlation(a, b).mean().square()), 1., places=12)
        self.assertAlmostEqual(float(signed.GlobalNCC()(a, -a)), 2., places=12)

    def test_gradient_multiplier(self):
        a, b = self.tensors()
        a, b = a[:1], b[:1]
        rho = g.GlobalNCCSquared().correlation(a, b).detach()
        squared_grad, = torch.autograd.grad(g.GlobalNCCSquared()(a, b), b, retain_graph=True)
        signed_grad, = torch.autograd.grad(signed.GlobalNCC()(a, b), b)
        torch.testing.assert_close(squared_grad, 2*rho*signed_grad, rtol=1e-10, atol=1e-12)

    def test_constant_floor_and_low_variance(self):
        a = torch.ones(1, 1, 3, 4, 5, dtype=torch.double)
        b = torch.ones_like(a, requires_grad=True)
        value = g.GlobalNCCSquared()(a, b)
        self.assertEqual(value.item(), 1.)
        self.assertTrue(torch.equal(torch.autograd.grad(value, b)[0], torch.zeros_like(b)))
        a, b = self.tensors()
        a, b = a*1e-4, b*1e-4
        ac, bc = a-a.mean((1, 2, 3, 4), keepdim=True), b-b.mean((1, 2, 3, 4), keepdim=True)
        expected = (1-(ac*bc).mean((1, 2, 3, 4)).square()/g.PROTOCOL['variance_product_epsilon']).mean()
        torch.testing.assert_close(g.GlobalNCCSquared()(a, b), expected, rtol=0, atol=1e-15)

    def test_no_moment_cache_or_batch_coupling(self):
        a, b = self.tensors(); loss = g.GlobalNCCSquared()
        before = loss(a, b)
        loss(3*a+5, 2*b-4)
        torch.testing.assert_close(loss(a, b), before, rtol=0, atol=0)
        torch.testing.assert_close(loss(a, b), (loss(a[:1], b[:1])+loss(a[1:], b[1:]))/2)
        self.assertEqual(len(loss.recent_rho), 2)

    def test_shape_dtype_guards(self):
        a, b = self.tensors()
        for x, y in ((a.long(), b.long()), (a, b.float()), (a[0], b[0]), (a, b[..., :-1])):
            with self.assertRaises(ValueError): g.GlobalNCCSquared()(x, y)

    def test_protocol_and_names(self):
        self.assertEqual(g.LAMBDAS, signed.LAMBDAS)
        self.assertEqual(g.LAMBDAS['1over3'], 1/3)
        self.assertEqual(len({g.run_name(k) for k in g.LAMBDAS}), 4)
        self.assertFalse(set(g.run_name(k) for k in g.LAMBDAS)&set(signed.run_name(k) for k in signed.LAMBDAS))
        for key in ('0.3', '10'):
            with self.assertRaises(ValueError): g.run_name(key)
        self.assertEqual(g.PROTOCOL['loss'], '1-rho^2')
        for key in signed.PROTOCOL.keys()-{'loss_key', 'loss'}:
            self.assertEqual(g.PROTOCOL[key], signed.PROTOCOL[key])
        with self.assertRaises(RuntimeError):
            g.require_protocol({'gncc_variant': signed.VARIANT, 'loss_protocol': signed.PROTOCOL})
        self.assertIs(old_runner.s, native)
        self.assertEqual(signed.PROTOCOL['loss'], '1-rho')

    def test_two_sequential_updates_and_rho(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.1))
        moving, fixed = torch.randn(1, 1, 3, 4, 5), torch.randn(1, 1, 3, 4, 5)
        seen = []
        def predict(src, tgt, model, bidir=False):
            seen.append(float(model.weight.detach()))
            return src+model.weight*torch.roll(src, 1, -1), model.weight.expand(1, 3, 3, 4, 5)
        model = Toy(); optimizer = g.optimizer_for(model)
        rows = g.train_pair(types.SimpleNamespace(model_predict_v2=predict), model, None, moving, fixed,
                            g.GlobalNCCSquared(), lambda f, _: f.square().mean(), optimizer, 1.)
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual([r['direction'] for r in rows], ['forward', 'reverse'])
        self.assertTrue(all(int(v['step']) == 2 for v in optimizer.state.values()))
        for row in rows:
            self.assertAlmostEqual(row['similarity'], 1-row['signed_rho_per_image'][0]**2, places=6)
            self.assertAlmostEqual(row['loss'], row['similarity']+row['diffusion'], places=6)

    def test_cli_and_output_guards(self):
        args = ['--mode', 'train', '--gpu', '3', '--lambda-key', '1', '--wavereg-root', '/tmp/native',
                '--data-root', '/tmp/data', '--reference-checkpoint', '/tmp/reference/checkpoint/best',
                '--reference-field', '/tmp/reference/field/field', '--out', '/tmp/new_gncc2/run',
                '--baseline-preflight', '/tmp/old_preflight', '--baseline-training-root', '/tmp/old_train']
        with self.assertRaises(SystemExit): runner.parse_args(args)
        self.assertEqual(runner.parse_args(args+['--preflight', '/tmp/gate']).gpu, 3)
        with self.assertRaises(SystemExit): runner.parse_args(args+['--preflight', '/tmp/gate', '--out', '/tmp/data'])
        with self.assertRaises(SystemExit): launcher.parse_args(['--preflight', '/tmp/gate', '--gpus', '0,1,2'])
        parsed = launcher.parse_args(['--preflight', '/tmp/gate'])
        self.assertEqual(parsed.review_root.name, g.REPORT_VARIANT)
        with self.assertRaises(SystemExit): evaluator.parse_args(['--training-dir', '/tmp/train', '--out', '/tmp/train/out', '--gpu', '3'])

    def fixture(self):
        pairs = report.c.parse_pair_file(report.c.DEFAULT_PAIR_FILE)
        return [dict(loss=loss, lambda_key=key, alpha=alpha, rank=r, fixed_id=pairs[r][0], moving_id=pairs[r][1],
                     **{m: .3+.0001*r for m in report.METRICS})
                for loss, groups in report.SERIES.items() for key, alpha in groups for r in range(100)]

    def test_complete_report_and_paired_contrasts(self):
        rows = self.fixture(); means = report.summarize(rows)
        self.assertEqual((len(rows), len(means), len(report.paired_rows(rows))), (1300, 13, 1200))
        self.assertEqual(sum(r['historical'] for r in means), 1)
        self.assertTrue(all(r['n'] == 100 for r in means))
        with self.assertRaises(RuntimeError): report.summarize(rows[:-1])
        rows[0]['fixed_id'] = 'wrong'
        with self.assertRaises(RuntimeError): report.summarize(rows)

    def test_plot_two_by_three_top_lines_bottom_ellipses(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        means = report.summarize(self.fixture())
        fig, axes = report.build_figure(means)
        self.assertEqual(axes.shape, (2, 3))
        self.assertTrue(all(not any(isinstance(p, Ellipse) for p in ax.patches) for ax in axes[0]))
        self.assertTrue(all(sum(isinstance(p, Ellipse) for p in ax.patches) == 13 for ax in axes[1]))
        plt.close(fig)

    def test_report_pipeline_with_synthetic_complete_records(self):
        # Exercise the complete reporting writer without pretending synthetic
        # records are experimental evidence; all outputs are ephemeral fixtures.
        rows = self.fixture()
        references = [{**r, 'field_path': '/fixture/reference', 'field_sha256': 'fixture'}
                      for r in rows if r['loss'] != 'gncc2']
        files = {}
        protocol = {'gncc_variant': g.VARIANT, 'loss_protocol': g.PROTOCOL}
        pairs = report.c.parse_pair_file(report.c.DEFAULT_PAIR_FILE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); train_root, eval_root = root/'train', root/'eval'
            out = root/'report'; out.mkdir()
            baseline = root/'repo_experiments/baseline'
            for gpu, (key, alpha) in enumerate(g.LAMBDAS.items()):
                train, evaluation = train_root/g.run_name(key), eval_root/g.run_name(key)
                files[str(train/'manifest.json')] = dict(protocol, source_sha256={}, model_key='fedda_1mf',
                    updates_per_pair=2, physical_gpu=gpu, arguments={'data_root': '/fixture/data'})
                files[str(train/'training_summary.json')] = dict(protocol, epoch=200, adam_updates=20000,
                    best_epoch=200, best_val_dice=.81, elapsed_seconds=1., pair_stream_sha256='pairs')
                files[str(train/'initialization.json')] = {'state_sha256': 'initial'}
                files[str(evaluation/'manifest.json')] = dict(protocol, source_sha256={}, native_source_sha256={},
                    alpha=alpha, variant=g.EVAL_VARIANT, smoke=False, foreground='fixed_raw_image_positive',
                    jacobian='canonical_LUMIR_crop2_positive_J_log_std',
                    asd_definition='pooled_surfel_area_weighted_bidirectional_mean', best_epoch=200,
                    best_val_dice=.81, checkpoint='/fixture/checkpoint', checkpoint_sha256='checkpoint')
                cases, labels = [], []
                for rank in range(100):
                    fixed, moving = pairs[rank]; value = .3+.0001*rank
                    case = dict(rank=rank, fixed_id=fixed, moving_id=moving, direction='moving_to_fixed',
                        alpha=alpha, checkpoint_sha256='checkpoint', field_path='/fixture/field', field_sha256='fixture',
                        diffusion_l2_full=value, inference_seconds=0., physical_gpu=gpu, best_epoch=200,
                        **{m: value for m in report.SOURCE})
                    details = [dict(rank=rank, label_id=label, included_in_case_mean=True,
                                    dice=value, hd95_mm=value, asd_symmetric_mm=value) for label in range(1, 36)]
                    cases.append(case); labels.extend(details)
                    files[str(evaluation/'cases'/f'rank{rank:03d}_{fixed}_{moving}'/'metrics.json')] = dict(summary=case, per_label=details)
                files[str(evaluation/'per_case.json')] = cases
                files[str(evaluation/'per_label_per_case.json')] = labels
                files[str(evaluation/'summary.json')] = dict(protocol, case_count=100, label_rows=3500, status='complete',
                    metrics={m: {'mean': report.statistics.mean(r[m] for r in cases),
                                 'sample_sd': report.statistics.stdev(r[m] for r in cases)} for m in report.SOURCE})
                for directory in (train, root/'runs/oasis_fedda_gncc_diffusion_training_lambda_sweep_v1'/f'fedda_1mf_gncc_lambda_{key}_seed42'):
                    directory.mkdir(parents=True)
                    (directory/'epochs.jsonl').write_text('\n'.join(json.dumps({'epoch': e, 'validation': {'dice': .8}}) for e in range(1, 201)))
            def diffusion(records, *args):
                return [{**r, 'diffusion_l2_fg': r['dice'], 'diffusion_l2_full': r['dice']} for r in records]
            prior = {'training': [{'initial_state_sha256': 'initial', 'pair_stream_sha256': 'pairs'}]}
            with patch.object(report, 'reference_rows', return_value=(references, prior)), \
                 patch.object(report, 'read', side_effect=lambda path: files[str(path)]), \
                 patch.object(report.g, 'verify_hashes'), patch.object(report.g, 'sha256', return_value='fixture'), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(report, 'augment_diffusion', side_effect=diffusion), patch.object(report, 'plot'):
                report.write_report(train_root, eval_root, baseline, out, {})
            accepted = json.loads((out/'ACCEPTED.json').read_text())
            self.assertEqual((accepted['settings'], accepted['cases'], accepted['new_label_rows']), (13, 1300, 14000))
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(result['validation_selected']['lambda_key'], '0.1')
            self.assertEqual(len(result['paired']), 1200)


if __name__ == '__main__':
    unittest.main()
