"""FeDDA directional updates, dependency queue, and canonical reporting contracts."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments/oasis_alpha_sweep'))
import fedda_training as f
import launch_oasis_fedda_training as q
from run_oasis_fedda_training import parse_args
from run_oasis_fedda_first100_eval import training_ready, label_rows
from run_oasis_sitreg_first100_eval import label_rows as canonical_label_rows
from report_oasis_fedda_training import METRICS, validate_cohort
import report_oasis_fedda_training as report
from certify_oasis_fedda_preflight import validate_contract as validate_audit


class ToyNative:
    @staticmethod
    def model_predict_v2(src, tgt, model, bidir=False):
        assert bidir is False
        output = model(src)
        return output, output


class FeDDAContracts(unittest.TestCase):
    def test_two_sequential_updates_exact_native_parity(self):
        for alpha in f.LAMBDAS.values():
            torch.manual_seed(42)
            model = torch.nn.Linear(2, 2)
            reference = copy.deepcopy(model)
            opt, refopt = f.optimizer_for(model), f.optimizer_for(reference)
            src, tgt = torch.tensor([[.1, .7]]), torch.tensor([[.9, .3]])
            sim = lambda a, b: (a - b).square().mean()
            reg = lambda a, _: a.square().mean()
            rows = f.train_pair(ToyNative, model, None, src, tgt, sim, reg, opt, alpha)
            expected = []
            for a, b in ((src, tgt), (tgt, src)):
                output = reference(a)
                loss = sim(b, output) + alpha * reg(output, 0)
                refopt.zero_grad()
                loss.backward()
                refopt.step()
                expected.append(loss.item())
            self.assertEqual([r['loss'] for r in rows], expected)
            self.assertEqual([r['direction'] for r in rows], ['forward', 'reverse'])
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(all(int(s['step']) == 2 for s in opt.state.values()))

    def test_assignment_and_unique_names(self):
        self.assertEqual(q.ASSIGNMENTS, ((0, '3'), (0, '10'), (1, '0.1'), (2, '1over3')))
        self.assertEqual(len({q.run_tag(k) for k in f.LAMBDAS}), 4)
        with self.assertRaises(ValueError):
            q.run_tag('1')

    def test_cli_scope_and_gate(self):
        base = ['--wavereg-root', '/tmp/w', '--data-root', '/tmp/d', '--reference-checkpoint', '/tmp/c',
                '--reference-field', '/tmp/f', '--out', '/tmp/o']
        for key in f.LAMBDAS:
            args = parse_args([*base, '--mode', 'train', '--gpu', '0', '--preflight', '/tmp/p', '--lambda-key', key])
            self.assertEqual(args.lambda_key, key)
            self.assertEqual(args.min_free_gib, 24)
        for bad in (['--mode', 'train', '--gpu', '0'],
                    ['--mode', 'preflight', '--gpu', '1'],
                    ['--mode', 'audit', '--gpu', '1'],
                    ['--mode', 'preflight', '--gpu', '0', '--min-free-gib', '20'],
                    ['--mode', 'preflight', '--gpu', '3']):
            with self.assertRaises(SystemExit):
                parse_args([*base, *bad])
        self.assertEqual(parse_args([*base, '--mode', 'audit', '--gpu', '0']).mode, 'audit')

    def audit_fixture(self):
        metrics = {key: .5 for key in ('dice_mean', 'hd95_mean', 'asd_symmetric_mean',
                                     'log_jdet_std_fg', 'jdet_le0_pct_fg')}
        result = {'status': 'audit_only', 'issues': [], 'parity': {
            'bitwise_field_reproduction': True, 'max_abs_voxel_error': 0, 'rmse_voxel': 0,
            'fixed_id': '0333', 'moving_id': '0115', 'actual_metrics': metrics.copy(),
            'reference_metrics': metrics.copy()},
            'native_validation': {'dice': .814109}, 'historical_validation_dice': .814110,
            'warm_memory': {'adam_updates': 6, 'peak_reserved_gib': 21.25},
            'min_train_free_gib': 24,
            'native_steps': [{'lambda_key': key, 'lambda': alpha, 'adam_updates': 2,
                              'finite_gradients': True, 'checkpoint_roundtrip_exact': True,
                              'peak_reserved_gib': 21.25,
                              'directions': [{'direction': direction, 'loss': .5, 'similarity': .4,
                                              'diffusion': .1, 'weighted_diffusion': alpha * .1}
                                             for direction in ('forward', 'reverse')]}
                             for key, alpha in f.LAMBDAS.items()]}
        manifest = {'backend': {'benchmark': True, 'deterministic': False,
                               'cudnn_tf32': True, 'matmul_tf32': False},
                    'uid': 1009, 'gid': 1020, 'physical_gpu': 0, 'model_key': 'fedda_1mf',
                    'seed': 42, 'updates_per_pair': 2, 'use_batch_parallel': False}
        return result, manifest

    def test_completed_audit_certificate_contract(self):
        result, manifest = self.audit_fixture()
        validate_audit(result, manifest)
        self.assertEqual(result['status'], 'audit_only')  # Does not alter original evidence.

    def test_failed_or_merely_approximate_audit_cannot_be_certified(self):
        for mutation in (
            lambda r, m: r.update(issues=['HD95 mismatch']),
            lambda r, m: r['parity'].update(bitwise_field_reproduction=False, max_abs_voxel_error=.001),
            lambda r, m: r['parity']['actual_metrics'].update(hd95_mean=.501),
            lambda r, m: r['native_validation'].update(dice=.7),
            lambda r, m: r['native_steps'].pop(),
            lambda r, m: r.update(min_train_free_gib=22),
            lambda r, m: m['backend'].update(benchmark=False),
            lambda r, m: m.update(uid=0),
        ):
            result, manifest = self.audit_fixture()
            mutation(result, manifest)
            with self.assertRaises(ValueError):
                validate_audit(result, manifest)

    def test_memory_gate(self):
        row = {'free_mib': 39513, 'total_mib': 49140}
        self.assertAlmostEqual(f.memory_fraction(row), (39513 - 1024) / 49140)
        with self.assertRaises(ValueError):
            f.memory_fraction(row, min_free_gib=20)
        with self.assertRaises(RuntimeError):
            f.memory_fraction(row, min_free_gib=40)

    def test_dependencies_require_evaluation_not_just_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            train, evaluation = path / 'train', path / 'eval'
            train.mkdir()
            evaluation.mkdir()
            dep = {'lambda_key': '0.1', 'status': 'complete', 'train_dir': str(train), 'eval_dir': str(evaluation),
                   'attempts': [{'stage': 'eval', 'returncode': 0}]}
            self.assertEqual(q.dependency_state({'lambda_key': '3'}, [], {}), 'ready')
            self.assertEqual(q.dependency_state({'lambda_key': '0.1'}, [], {'jobs': [dep]}), 'waiting_for_dependency')
            (train / 'COMPLETED').touch()
            self.assertFalse(q.clean_evaluation(dep))
            (evaluation / 'COMPLETED').touch()
            self.assertTrue(q.clean_evaluation(dep))
            self.assertEqual(q.dependency_state({'lambda_key': '0.1'}, [], {'jobs': [dep]}), 'ready')
            own = {**dep, 'lambda_key': '3'}
            self.assertEqual(q.dependency_state({'lambda_key': '10'}, [own], {}), 'ready')
            own['status'] = 'failed'
            self.assertEqual(q.dependency_state({'lambda_key': '10'}, [own], {}), 'dependency_failed')
            dep['attempts'][-1]['returncode'] = 1
            self.assertFalse(q.clean_evaluation(dep))

    def test_training_ready_requires_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ('TRAINING_COMPLETED', 'COMPLETED', 'checkpoints/best_dice.pth.tar', 'first10/per_case.json'):
                item = p / name
                item.parent.mkdir(parents=True, exist_ok=True)
                item.touch()
            summary = p / 'training_summary.json'
            summary.write_text(json.dumps({'epoch': 200, 'adam_updates': 10000}))
            with self.assertRaises(ValueError):
                training_ready(p)
            summary.write_text(json.dumps({'epoch': 200, 'adam_updates': 20000}))
            self.assertEqual(training_ready(p), (True, 'ready'))

    def test_exact_canonical_label_function(self):
        self.assertIs(label_rows, canonical_label_rows)

    def test_original_model_and_checkpoint_identity(self):
        self.assertTrue(str(f.MODEL_CONFIG_REL).endswith('exp_midl_0407/fedda_1mf'))
        self.assertEqual(f.CHECKPOINT_SHA, 'fa9ba93e5ea9296502f1e144accf715738fe9602cfeee32a41a81b1393a5c01e')

    def test_report_rejects_partial_duplicate_or_mismatched_cohort(self):
        pairs = [(f'{i:04d}', f'{i+100:04d}') for i in range(100)]
        rows = [{'rank': i, 'fixed_id': a, 'moving_id': b, **{k: .5 for k in METRICS}}
                for i, (a, b) in enumerate(pairs)]
        validate_cohort(rows, pairs)
        for bad in (rows[:-1], rows[:-1] + [rows[0]], [{**rows[0], 'fixed_id': 'bad'}, *rows[1:]],
                    [{**rows[0], 'hd95': float('nan')}, *rows[1:]]):
            with self.assertRaises(ValueError):
                validate_cohort(bad, pairs)

    def test_complete_report_exports_seven_groups_and_curves(self):
        import common as c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
            baseline = [{'method': 'FeDDA-1MF lambda1 historical', 'alpha': 1.0, 'rank': i,
                         'fixed_id': a, 'moving_id': b, **{k: .5 + i * .0001 for k in METRICS}}
                        for i, (a, b) in enumerate(pairs[:100])]
            reference = root / 'references.csv'
            refs = [{**r, 'method': name} for name in ('VFA (original)', 'SITReg λ=1 (original)') for r in baseline]
            c.write_csv(reference, refs)
            checkpoint, field = root / 'checkpoint', root / 'field'
            checkpoint.write_bytes(b'synthetic checkpoint fixture')
            field.write_bytes(b'synthetic field fixture')
            jobs = []
            for gpu, key in q.ASSIGNMENTS:
                out = root / key
                out.mkdir()
                (out / 'COMPLETED').touch()
                rows = [{**{k: r[k] for k in ('rank', 'fixed_id', 'moving_id')},
                         **{v: r[k] for k, v in METRICS.items()},
                         'field_path': str(field), 'field_sha256': f.sha256(field)} for r in baseline]
                f.write_json(out / 'per_case.json', rows)
                f.write_json(out / 'per_label_per_case.json', [{}] * 3500)
                f.write_json(out / 'manifest.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': f.sha256(checkpoint),
                            'source_sha256': {}, 'native_source_sha256': {}, 'best_epoch': 100, 'best_val_dice': .8})
                jobs.append({'status': 'complete', 'eval_dir': str(out), 'lambda_key': key, 'gpu': gpu})
            f.write_json(root / 'queue.json', {'jobs': jobs})
            output = root / 'report'
            argv = ['report', '--training-root', str(root), '--out', str(output), '--reference-root', str(root), '--data-root', str(root)]
            with patch.object(report, 'baseline', return_value=baseline), patch.object(report, 'REFERENCE_TABLE', reference), patch.object(sys, 'argv', argv):
                report.main()
            accepted = json.loads((output / 'ACCEPTED.json').read_text())
            self.assertEqual((accepted['case_rows'], accepted['groups']), (700, 7))
            for ext in ('png', 'svg', 'pdf'):
                self.assertTrue((output / f'accuracy_regularity_lines.{ext}').is_file())
            with (output / 'mean_std.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 7)


if __name__ == '__main__':
    unittest.main()
