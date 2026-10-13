"""FeDDA GNCC adapters, immutable legacy compatibility and complete reports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
sys.path.insert(0, str(ROOT/'scripts/reports'))
import fedda_training as native
import fedda_gncc_training as g
import sitreg_gncc_training as sg
import run_oasis_fedda_training as legacy
import run_oasis_fedda_first100_eval as legacy_eval
import run_oasis_fedda_gncc_training as runner
import run_oasis_fedda_gncc_eval as evaluator
import launch_oasis_fedda_gncc as launcher
import oasis_fedda_gncc_report as report
from certify_oasis_fedda_gncc_preflight import validate_contract


class FeDDAGNCCContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_lambda_names_and_unchanged_native(self):
        self.assertEqual(g.LAMBDAS, {'0.1': .1, '1over3': 1/3, '1': 1., '3': 3.})
        self.assertEqual(native.LAMBDAS, {'0.1': .1, '1over3': 1/3, '3': 3., '10': 10.})
        self.assertEqual(len({g.run_name(k) for k in g.LAMBDAS}), 4)
        with self.assertRaises(ValueError): g.run_name('0.3')
        with self.assertRaises(ValueError): g.run_name('10')
        self.assertIs(legacy.s, native); self.assertIs(legacy_eval.s, native)
        self.assertIs(g.train_pair, native.train_pair)
        self.assertIs(g.GlobalNCC, sg.GlobalNCC)
        self.assertEqual(g.PROTOCOL['epochs']*g.PROTOCOL['pairs_per_epoch']*g.PROTOCOL['updates_per_pair'], 20000)
        self.assertEqual(sg.PROTOCOL['updates_per_pair'], 1)
        self.assertEqual(g.PROTOCOL['validation_direction'], 'forward_only')

    def test_cli_gpu3_lambda1_and_protected_outputs(self):
        args = ['--mode', 'train', '--gpu', '3', '--lambda-key', '1', '--wavereg-root', '/tmp/native',
                '--data-root', '/tmp/data', '--reference-checkpoint', '/tmp/reference/checkpoints/best',
                '--reference-field', '/tmp/reference/fields/field', '--out', '/tmp/new_gncc/run',
                '--baseline-preflight', '/tmp/old_preflight', '--baseline-training-root', '/tmp/old_train']
        with self.assertRaises(SystemExit): runner.parse_args(args)
        parsed = runner.parse_args(args+['--preflight', '/tmp/new_gncc/gate'])
        self.assertEqual((parsed.gpu, parsed.lambda_key), (3, '1'))
        for extra in (['--lambda-key', '10'], ['--min-free-gib', '20'], ['--out', '/tmp/old_train/overwrite']):
            with self.assertRaises(SystemExit): runner.parse_args(args+['--preflight', '/tmp/gate']+extra)
        e = evaluator.parse_args(['--gpu', '3', '--training-dir', '/tmp/train', '--out', '/tmp/eval'])
        self.assertEqual(e.gpu, 3)
        with self.assertRaises(SystemExit):
            launcher.parse_args(['--preflight', '/tmp/gate', '--gpus', '0,1,2'])

    def test_signed_pearson_gradients_stateless(self):
        a, b = torch.randn(2, 1, 3, 4, 5, dtype=torch.double), torch.randn(2, 1, 3, 4, 5, dtype=torch.double, requires_grad=True)
        loss = g.GlobalNCC()
        expected = torch.stack([1-torch.corrcoef(torch.stack((x.flatten(), y.flatten())))[0, 1] for x, y in zip(a, b)]).mean()
        actual = loss(a, b)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        grad, = torch.autograd.grad(actual, b, retain_graph=True)
        ref, = torch.autograd.grad(expected, b)
        torch.testing.assert_close(grad, ref, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(float(loss(a, -a)), 2., places=12)
        before = loss(a, b).detach()
        loss(torch.randn_like(a), torch.randn_like(a))
        torch.testing.assert_close(loss(a, b), before, rtol=0, atol=0)

    def test_two_sequential_adam_updates(self):
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
                            g.GlobalNCC(), lambda field, _: field.square().mean(), optimizer, 1.)
        self.assertEqual([r['direction'] for r in rows], ['forward', 'reverse'])
        self.assertNotEqual(seen[0], seen[1])
        self.assertTrue(all(int(s['step']) == 2 for s in optimizer.state.values()))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertTrue(optimizer.param_groups[0]['amsgrad'])
        for row in rows:
            self.assertAlmostEqual(row['loss'], row['similarity']+row['diffusion'], places=6)

    def test_protocol_and_pair_hash_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            g.atomic_checkpoint(path, {'state_dict': torch.nn.Linear(2, 2).state_dict()})
            value = torch.load(path, weights_only=False)
            g.require_protocol(value)
            with self.assertRaises(RuntimeError): sg.require_protocol(value)
            with self.assertRaises(RuntimeError): g.require_protocol({'loss_protocol': sg.PROTOCOL})
        with patch.object(g, 'EXPECTED_PAIRS', {'1': 'correct'}):
            g.check_pair_hash({'epoch': 1, 'pair_stream_sha256': 'correct'})
            with self.assertRaises(RuntimeError): g.check_pair_hash({'epoch': 1, 'pair_stream_sha256': 'wrong'})

    def certificate_fixture(self):
        metadata = dict(gncc_variant=g.VARIANT, loss_protocol=copy.deepcopy(g.PROTOCOL))
        metrics = {m: .5 for m in report.SOURCE}
        directions = [dict(direction=d, loss=.5, similarity=.4, diffusion=.1, weighted_diffusion=.1)
                      for d in ('forward', 'reverse')]
        audit = dict(**metadata, status='audit_only', issues=[],
                     parity=dict(bitwise_field_reproduction=True, max_abs_voxel_error=0., rmse_voxel=0.,
                                 fixed_id='0333', moving_id='0115', actual_metrics=metrics, reference_metrics=metrics.copy()),
                     native_validation={'dice': .8141}, historical_validation_dice=.8141,
                     native_steps=[dict(lambda_key=k, **{'lambda': a}, adam_updates=2, finite_gradients=True,
                                        checkpoint_roundtrip_exact=True, directions=copy.deepcopy(directions),
                                        peak_reserved_gib=21.) for k, a in g.LAMBDAS.items()],
                     warm_memory=dict(adam_updates=6, peak_reserved_gib=21.1))
        diagnostic = dict(status='diagnostic_only', production_qualified=False, historical_parity_issues=[],
                          independent_native_vs_gncc_model=dict(bitwise_equal=True, identical_state_hash=True,
                            max_abs_voxel=0., rmse_voxel=0., native_similarity_class='NCC_vfa', native_keep_sign=False,
                            new_similarity_class='GlobalNCC', native_parameters=2854079),
                          trained_checkpoint_adam_updates=2, trained_checkpoint_directions=copy.deepcopy(directions),
                          evaluation_smokes=[dict(rank=r, label_rows=35) for r in (0, 10)])
        manifest = dict(**metadata, backend=copy.deepcopy(g.PROTOCOL['backend']), uid=1009, gid=1020,
                        physical_gpu=3, model_key='fedda_1mf', updates_per_pair=2, seed=42, use_batch_parallel=False)
        return audit, diagnostic, manifest

    def test_independent_zero_issue_audit_contract(self):
        values = self.certificate_fixture()
        validate_contract(*values)
        self.assertEqual(values[0]['status'], 'audit_only')
        self.assertFalse(values[1]['production_qualified'])

    def test_certificate_rejects_failed_or_merely_close_evidence(self):
        mutations = [lambda a, d, m: a.update(issues=['HD95 mismatch']),
                     lambda a, d, m: a['parity'].update(bitwise_field_reproduction=False, max_abs_voxel_error=.001),
                     lambda a, d, m: a['parity']['actual_metrics'].update(dice_mean=.501),
                     lambda a, d, m: a['native_validation'].update(dice=.7),
                     lambda a, d, m: d['independent_native_vs_gncc_model'].update(bitwise_equal=False),
                     lambda a, d, m: a['native_steps'].pop(),
                     lambda a, d, m: d.update(trained_checkpoint_adam_updates=1),
                     lambda a, d, m: m.update(updates_per_pair=1),
                     lambda a, d, m: m['backend'].update(benchmark=False)]
        for mutation in mutations:
            values = self.certificate_fixture(); mutation(*values)
            with self.assertRaises((ValueError, RuntimeError)): validate_contract(*values)

    def test_full_synthetic_report_and_plot(self):
        """Only temporary synthetic schemas: no real fields, inference or training."""
        import common as c
        import matplotlib.pyplot as plt
        pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[:100]
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            train, evaluation, baseline, out = [base/k for k in ('train', 'eval', 'old', 'report')]
            for path in (train, evaluation, baseline, out): path.mkdir()
            old = [dict(method=method, alpha=alpha, rank=rank, fixed_id=fixed, moving_id=moving,
                        **{m: .3+rank*.001 for m in report.METRICS})
                   for method, (_, alpha) in report.OLD.items() for rank, (fixed, moving) in enumerate(pairs)]
            g.write_json(baseline/'results.json', {'per_case': old})
            for gpu, (key, alpha) in enumerate(g.LAMBDAS.items()):
                tp, ep = train/g.run_name(key), evaluation/g.run_name(key)
                tp.mkdir(); ep.mkdir(); (tp/'checkpoints').mkdir()
                for p in (tp, ep): (p/'COMPLETED').write_text('synthetic')
                checkpoint = tp/'checkpoints/best_dice.pth.tar'; checkpoint.write_text('synthetic fixture')
                field = ep/'synthetic_field'; field.write_text('synthetic field fixture, not a DVF')
                digest = g.sha256(checkpoint)
                metadata = dict(best_epoch=42, best_val_dice=.7+gpu*.001)
                g.write_json(tp/'manifest.json', dict(physical_gpu=gpu, source_sha256={}, model_key='fedda_1mf', updates_per_pair=2))
                g.write_json(tp/'initialization.json', dict(state_sha256='same'))
                g.write_json(tp/'training_summary.json', dict(epoch=200, adam_updates=20000, elapsed_seconds=10,
                             pair_stream_sha256='same', **metadata))
                (tp/'epochs.jsonl').write_text(json.dumps({'epoch': 1, 'validation': {'dice': .7}})+'\n')
                g.write_json(ep/'manifest.json', dict(variant=g.EVAL_VARIANT, alpha=alpha, source_sha256={},
                             native_source_sha256={}, smoke=False, foreground='fixed_raw_image_positive',
                             jacobian='canonical_LUMIR_crop2_positive_J_log_std',
                             asd_definition='pooled_surfel_area_weighted_bidirectional_mean',
                             checkpoint=str(checkpoint), checkpoint_sha256=digest, **metadata))
                cases, details = [], []
                for rank, (fixed, moving) in enumerate(pairs):
                    row = dict(rank=rank, fixed_id=fixed, moving_id=moving, direction='moving_to_fixed',
                               alpha=alpha, checkpoint_sha256=digest, field_path=str(field), field_sha256=g.sha256(field),
                               **{m: .4+rank*.001 for m in report.SOURCE})
                    labels = [dict(rank=rank, label_id=label, included_in_case_mean=True,
                                   dice=row['dice_mean'], hd95_mm=row['hd95_mean'], asd_symmetric_mm=row['asd_symmetric_mean'])
                              for label in range(1, 36)]
                    case = ep/'cases'/f'rank{rank:03d}_{fixed}_{moving}'; case.mkdir(parents=True)
                    (case/'COMPLETED').write_text('synthetic')
                    g.write_json(case/'metrics.json', dict(summary=row, per_label=labels))
                    cases.append(row); details.extend(labels)
                g.write_json(ep/'per_case.json', cases); g.write_json(ep/'per_label_per_case.json', details)
                g.write_json(ep/'summary.json', dict(status='complete', case_count=100, label_rows=3500,
                             metrics={m: report.stats([r[m] for r in cases]) for m in report.SOURCE}))
            refs = {str(baseline/'results.json'): g.sha256(baseline/'results.json')}
            captured = []
            close = plt.close
            with patch.object(plt, 'close', lambda fig: captured.append(fig)):
                report.write_report(train, evaluation, baseline, out, refs)
            figures = [f for f in captured if hasattr(f, 'axes') and len(f.axes) == 6]
            self.assertEqual(len(figures), 1)
            fig = figures[0]
            self.assertTrue(all(not ax.patches for ax in fig.axes[:3]))
            self.assertTrue(all(len(ax.patches) == 9 for ax in fig.axes[3:]))
            self.assertIn('FeDDA', fig._suptitle.get_text())
            self.assertTrue(all('SITReg' not in t.get_text() for l in fig.legends for t in l.get_texts()))
            for f in captured:
                if hasattr(f, 'axes'): close(f)
            result = report.read(out/'results.json')
            self.assertEqual(len(result['per_case']), 900)
            self.assertEqual(len(report.read(out/'per_label_per_case.json')), 14000)
            self.assertTrue((out/'DONE').is_file())
            self.assertEqual(result['validation_selected']['lambda_key'], '3')
            with self.assertRaises(RuntimeError): report.summarize(result['per_case'][:-1])
            (evaluation/g.run_name('0.1')/'COMPLETED').unlink()
            with self.assertRaises(RuntimeError): report.write_report(train, evaluation, baseline, out, refs)


if __name__ == '__main__':
    unittest.main()
