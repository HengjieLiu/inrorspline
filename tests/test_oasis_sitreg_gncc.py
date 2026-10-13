"""GNCC training adapter contracts, with legacy sources left untouched."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
sys.path.insert(0, str(ROOT/'scripts/reports'))
import sitreg_training as native
import sitreg_gncc_training as g
import run_oasis_sitreg_training as legacy
import run_oasis_sitreg_first100_eval as legacy_eval
from run_oasis_sitreg_gncc_training import parse_args
import oasis_sitreg_gncc_report as report
from gncc_dense import GNCC


class GNCCTrainingContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_exact_grid_names_and_legacy(self):
        self.assertEqual(g.LAMBDAS, {'0.1': .1, '1over3': 1/3, '1': 1., '3': 3.})
        self.assertEqual(native.LAMBDAS, {'0.1': .1, '1over3': 1/3, '3': 3., '10': 10.})
        self.assertEqual(len({g.run_name(k) for k in g.LAMBDAS}), 4)
        self.assertIs(legacy.s, native); self.assertIs(legacy_eval.s, native)
        with self.assertRaises(ValueError):
            g.run_name('0.3')

    def test_cli(self):
        args = ['--mode', 'train', '--gpu', '0', '--lambda-key', '1', '--wavereg-root', '/tmp/native',
                '--data-root', '/tmp/data', '--reference-checkpoint', '/tmp/checkpoint', '--reference-field', '/tmp/field',
                '--out', '/tmp/new_gncc', '--baseline-preflight', '/tmp/old/preflight']
        with self.assertRaises(SystemExit):
            parse_args(args)
        self.assertEqual(parse_args(args+['--preflight', '/tmp/gate']).lambda_key, '1')
        with self.assertRaises(SystemExit):
            parse_args(args+['--preflight', '/tmp/gate', '--lambda-key', '0.3'])

    def test_pearson_value_and_gradient(self):
        a = torch.randn(1, 1, 3, 4, 5, dtype=torch.double)
        b = torch.randn_like(a).requires_grad_()
        value = g.GlobalNCC()(a, b)
        expected = 1-torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1]
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        grad, = torch.autograd.grad(value, b, retain_graph=True)
        ref, = torch.autograd.grad(expected, b)
        torch.testing.assert_close(grad, ref, rtol=1e-10, atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(g.GlobalNCC(), (a, b)))

    def test_matches_completed_gncc_definition(self):
        a, b = torch.rand(1, 1, 4, 5, 6), torch.rand(1, 1, 4, 5, 6, requires_grad=True)
        v, ref = g.GlobalNCC()(a, b), GNCC(a)(b)
        torch.testing.assert_close(v, ref, rtol=0, atol=1e-7)
        p, = torch.autograd.grad(v, b, retain_graph=True); q, = torch.autograd.grad(ref, b)
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)

    def test_unsquared_sign_affine(self):
        a = torch.randn(1, 1, 3, 4, 5, dtype=torch.double)
        loss = g.GlobalNCC()
        self.assertAlmostEqual(float(loss(a, a)), 0, places=12)
        self.assertAlmostEqual(float(loss(a, -a)), 2, places=12)
        self.assertAlmostEqual(float(loss(a, 3*a+7)), 0, places=12)

    def test_constant_stability(self):
        for factor in (0., 1e-10):
            a = torch.ones(1, 1, 3, 4, 5)
            b = (a+factor*torch.randn_like(a)).requires_grad_()
            loss = g.GlobalNCC()(a, b)
            grad, = torch.autograd.grad(loss, b)
            self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(float(loss), 1.)

    def test_stateless_per_batch_not_cross_image_correlation(self):
        loss = g.GlobalNCC()
        a, b = torch.randn(2, 1, 3, 4, 5), torch.randn(2, 1, 3, 4, 5)
        a[1] = a[1]*.2+100
        b[1] = b[1]*2-20
        torch.testing.assert_close(loss(a, b), (loss(a[:1], b[:1])+loss(a[1:], b[1:]))/2)
        original = loss(a, b).clone()
        loss(torch.randn_like(a), torch.randn_like(b))
        torch.testing.assert_close(loss(a, b), original, rtol=0, atol=0)

    def test_both_argument_gradient_paths_and_symmetry(self):
        a, b = torch.randn(1, 1, 3, 4, 5), torch.randn(1, 1, 3, 4, 5, requires_grad=True)
        one, two = g.GlobalNCC()(a, b), g.GlobalNCC()(b, a)
        p, = torch.autograd.grad(one, b); q, = torch.autograd.grad(two, b)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_native_joint_update_and_optimizer(self):
        a, b = torch.nn.Parameter(torch.randn(1, 1, 3, 4, 5)), torch.nn.Parameter(torch.randn(1, 1, 3, 4, 5))
        fixed, moving = torch.randn_like(a), torch.randn_like(b)
        sim, reg = g.GlobalNCC(), lambda x, _: x.square().mean()
        for alpha in g.LAMBDAS.values():
            total, similarity, diffusion = g.joint_loss((a, b, a, b), moving, fixed, sim, reg, alpha)
            expected = .5*(sim(fixed, a)+sim(moving, b))+.5*alpha*(reg(a, 0)+reg(b, 0))
            torch.testing.assert_close(total, expected)
            grad = torch.autograd.grad(total, (a, b))
            self.assertTrue(all(torch.isfinite(t).all() and t.abs().sum() > 0 for t in grad))
        model = torch.nn.Linear(3, 3); optimizer = g.optimizer_for(model)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertTrue(optimizer.param_groups[0]['amsgrad'])
        self.assertEqual(g.PROTOCOL['epochs']*g.PROTOCOL['pairs_per_epoch'], 10000)

    def test_checkpoint_protocol_and_pair_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            g.atomic_checkpoint(path, {'state_dict': torch.nn.Linear(2, 2).state_dict()})
            g.require_protocol(torch.load(path, weights_only=False))
            with self.assertRaises(RuntimeError):
                g.require_protocol({'loss_protocol': {'loss': '1-rho_squared'}})
            with patch.object(g, 'EXPECTED_PAIRS', {'1': 'correct'}):
                g.check_pair_hash({'epoch': 1, 'pair_stream_sha256': 'correct'})
                with self.assertRaises(RuntimeError):
                    g.check_pair_hash({'epoch': 1, 'pair_stream_sha256': 'wrong'})

    def test_report_completeness_and_layout(self):
        rows = []
        for loss, groups in [('gncc', list(g.LAMBDAS.items())), ('lncc9', list(report.OLD.values()))]:
            for key, alpha in groups:
                for rank in range(100):
                    rows.append(dict(loss=loss, lambda_key=key, alpha=alpha, rank=rank,
                                     **{m: .1+rank*.001+alpha*.0001 for m in report.METRICS}))
        means = report.summarize(rows)
        self.assertEqual(len(means), 9)
        self.assertEqual(sum(r['historical'] for r in means), 1)
        with self.assertRaises(RuntimeError):
            report.summarize(rows[:-1])
        import matplotlib.pyplot as plt
        captured = []
        with tempfile.TemporaryDirectory() as folder, patch.object(plt, 'close', lambda f: captured.append(f)):
            report.plot(means, Path(folder))
        fig = captured[-1]
        self.assertEqual(len(fig.axes), 6)
        self.assertTrue(all(len(ax.patches) == 9 for ax in fig.axes[:3]))
        self.assertTrue(all(not ax.patches for ax in fig.axes[3:]))
        for i in range(3):
            self.assertEqual(fig.axes[i].get_xlim(), fig.axes[i+3].get_xlim())
            self.assertEqual(fig.axes[i].get_ylim(), fig.axes[i+3].get_ylim())
        plt.close(fig)

    def test_complete_report_pipeline_synthetic_only(self):
        """Exercise final aggregation schemas without any real inference/training."""
        import common as c
        pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[:100]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, evaluation, baseline, out = [root/k for k in ('train', 'evaluation', 'baseline', 'report')]
            for path in (train, evaluation, baseline, out):
                path.mkdir()
            old_rows = []
            for method in report.OLD:
                for rank, (fixed, moving) in enumerate(pairs):
                    old_rows.append(dict(method=method, rank=rank, fixed_id=fixed, moving_id=moving,
                                         **{m: .3+rank*.001 for m in report.METRICS}))
            g.write_json(baseline/'results.json', {'per_case': old_rows})
            for gpu, (key, alpha) in enumerate(g.LAMBDAS.items()):
                tp, ep = train/g.run_name(key), evaluation/g.run_name(key)
                tp.mkdir(); ep.mkdir(); (tp/'checkpoints').mkdir()
                (tp/'COMPLETED').write_text('synthetic'); (ep/'COMPLETED').write_text('synthetic')
                checkpoint = tp/'checkpoints/best_dice.pth.tar'
                checkpoint.write_text('synthetic schema fixture, not a model')
                field = ep/'synthetic_field'
                field.write_text('synthetic schema fixture, not a DVF')
                digest = g.sha256(checkpoint)
                metadata = dict(best_epoch=42, best_val_dice=.7+gpu*.001)
                g.write_json(tp/'manifest.json', dict(physical_gpu=gpu, source_sha256={}))
                g.write_json(tp/'initialization.json', dict(state_sha256='same'))
                g.write_json(tp/'training_summary.json', dict(epoch=200, adam_updates=10000, elapsed_seconds=10,
                             pair_stream_sha256='same', **metadata))
                (tp/'epochs.jsonl').write_text(json.dumps({'epoch': 1, 'validation': {'dice': .7}})+'\n')
                g.write_json(ep/'manifest.json', dict(variant=g.EVAL_VARIANT, alpha=alpha, source_sha256={},
                             checkpoint=str(checkpoint), checkpoint_sha256=digest, **metadata))
                cases, all_labels = [], []
                for rank, (fixed, moving) in enumerate(pairs):
                    row = dict(rank=rank, fixed_id=fixed, moving_id=moving, direction='moving_to_fixed',
                               alpha=alpha, checkpoint_sha256=digest, field_path=str(field), field_sha256=g.sha256(field),
                               **{m: .4+rank*.001 for m in report.SOURCE})
                    labels = [dict(rank=rank, label_id=label, included_in_case_mean=True,
                                   dice=row['dice_mean'], hd95_mm=row['hd95_mean'], asd_symmetric_mm=row['asd_symmetric_mean'])
                              for label in range(1, 36)]
                    case = ep/'cases'/f'rank{rank:03d}_{fixed}_{moving}'
                    case.mkdir(parents=True)
                    (case/'COMPLETED').write_text('synthetic')
                    g.write_json(case/'metrics.json', dict(summary=row, per_label=labels))
                    cases.append(row); all_labels.extend(labels)
                g.write_json(ep/'per_case.json', cases); g.write_json(ep/'per_label_per_case.json', all_labels)
                g.write_json(ep/'summary.json', dict(status='complete', case_count=100, label_rows=3500,
                             metrics={m: report.stats([r[m] for r in cases]) for m in report.SOURCE}))
            refs = {str(baseline/'results.json'): g.sha256(baseline/'results.json')}
            report.write_report(train, evaluation, baseline, out, refs)
            self.assertTrue((out/'DONE').is_file())
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(len(result['per_case']), 900)
            self.assertEqual(result['validation_selected']['lambda_key'], '3')
            self.assertEqual(len(json.loads((out/'per_label_per_case.json').read_text())), 14000)


if __name__ == '__main__':
    unittest.main()
