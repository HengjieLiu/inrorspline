"""Native VFA directional updates, approved placement, and evaluation contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments/oasis_alpha_sweep'))
import vfa_training as v
import launch_oasis_vfa_training as queue
from run_oasis_vfa_training import parse_args
from run_oasis_vfa_first100_eval import training_ready, label_rows
from run_oasis_sitreg_first100_eval import label_rows as canonical_label_rows


class ToyNative:
    @staticmethod
    def model_predict_v2(src, tgt, model, model_key, spatial):
        assert model_key == 'bs3_vfa_v0'
        x = model(src)
        return x, x


class VFAContracts(unittest.TestCase):
    def test_two_sequential_updates_match_native_and_reverse_sees_new_weights(self):
        for alpha in v.LAMBDAS.values():
            torch.manual_seed(42)
            model = torch.nn.Linear(2, 2)
            native = copy.deepcopy(model)
            opt, other = v.optimizer_for(model), v.optimizer_for(native)
            src, tgt = torch.tensor([[.1, .7]]), torch.tensor([[.9, .3]])
            sim = lambda a, b: (a - b).square().mean()
            reg = lambda a, _: a.square().mean()
            rows = v.train_pair(ToyNative, model, None, src, tgt, sim, reg, opt, alpha)
            expected = []
            for a, b in ((src, tgt), (tgt, src)):
                warped = native(a)
                loss = sim(b, warped) + alpha * reg(warped, 0)
                other.zero_grad()
                loss.backward()
                other.step()
                expected.append(loss.item())
            self.assertEqual([r['direction'] for r in rows], ['forward', 'reverse'])
            self.assertEqual([r['loss'] for r in rows], expected)
            for a, b in zip(model.parameters(), native.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(all(int(state['step']) == 2 for state in opt.state.values()))

    def test_exact_gpu_assignment(self):
        self.assertEqual(queue.ASSIGNMENTS, ((0, '10'), (1, '0.1'), (2, '1over3'), (3, '3')))

    def test_only_one_approved_gpu0_oom_retry(self):
        job = {'lambda_key': '10', 'gpu': 0, 'retried_on_gpu3': False}
        self.assertTrue(queue.retry_allowed(job, oom=True))
        self.assertFalse(queue.retry_allowed(job, oom=False))
        for override in ({'gpu': 3}, {'retried_on_gpu3': True}, {'lambda_key': '0.1'}):
            self.assertFalse(queue.retry_allowed({**job, **override}, oom=True))

    def test_oom_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertFalse(queue.is_cuda_oom(path))
            (path / 'progress.json').write_text(json.dumps({'failure_kind': 'cuda_oom'}))
            self.assertTrue(queue.is_cuda_oom(path))

    def test_memory_override(self):
        row = {'free_mib': 39513, 'total_mib': 49140}
        with self.assertRaises(RuntimeError):
            v.memory_fraction(row)
        self.assertAlmostEqual(v.memory_fraction(row, allow_tight=True), (39513 - 1024) / 49140)

    def test_cli_override_restricted(self):
        base = ['--mode', 'train', '--preflight', '/tmp/gate', '--wavereg-root', '/tmp/w',
                '--data-root', '/tmp/data', '--reference-checkpoint', '/tmp/ckpt',
                '--reference-field', '/tmp/field', '--out', '/tmp/out']
        a = parse_args([*base, '--gpu', '0', '--lambda-key', '10', '--allow-tight-memory'])
        self.assertTrue(a.allow_tight_memory)
        with self.assertRaises(SystemExit):
            parse_args([*base, '--gpu', '2', '--lambda-key', '10', '--allow-tight-memory'])

    def test_vfa_evaluation_requires_20000_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ('TRAINING_COMPLETED', 'COMPLETED', 'checkpoints/best_dice.pth.tar', 'first10/per_case.json'):
                p = path / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            summary = path / 'training_summary.json'
            summary.write_text(json.dumps({'epoch': 200, 'adam_updates': 10000}))
            with self.assertRaises(ValueError):
                training_ready(path)
            summary.write_text(json.dumps({'epoch': 200, 'adam_updates': 20000}))
            self.assertEqual(training_ready(path), (True, 'ready'))

    def test_reuses_identical_per_label_pipeline(self):
        self.assertIs(label_rows, canonical_label_rows)

    def test_sitreg_dependency_needs_clean_exit_and_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = {'jobs': [{'gpu': 0, 'status': 'complete', 'returncode': None, 'out': tmp}]}
            self.assertFalse(queue.sitreg_finished(q, 0))
            q['jobs'][0]['returncode'] = 0
            self.assertFalse(queue.sitreg_finished(q, 0))
            (Path(tmp) / 'COMPLETED').touch()
            self.assertTrue(queue.sitreg_finished(q, 0))


if __name__ == '__main__':
    unittest.main()
