"""Contracts for the approval-scoped native SITReg training adapter."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments/oasis_alpha_sweep'))
import sitreg_training as s
from run_oasis_sitreg_training import parse_args


class TrainingContracts(unittest.TestCase):
    def test_lambda_grid(self):
        self.assertEqual(s.LAMBDAS, {'0.1': .1, '1over3': 1 / 3, '3': 3., '10': 10.})
        self.assertEqual(len(set(f'sitreg_lambda_{k}' for k in s.LAMBDAS)), 4)

    def test_cli_gate(self):
        base = ['--gpu', '3', '--wavereg-root', '/tmp/w', '--data-root', '/tmp/d',
                '--reference-checkpoint', '/tmp/c', '--reference-field', '/tmp/f', '--out', '/tmp/o']
        self.assertEqual(parse_args(['--mode', 'preflight', *base]).gpu, 3)
        with self.assertRaises(SystemExit):
            parse_args(['--mode', 'train', *base])
        self.assertEqual(parse_args(['--mode', 'train', '--preflight', '/tmp/p',
                                    '--lambda-key', '1over3', *base]).lambda_key, '1over3')

    def test_joint_native_loss_and_one_update(self):
        for alpha in s.LAMBDAS.values():
            model = torch.nn.Linear(2, 2)
            native = copy.deepcopy(model)
            opt, native_opt = s.optimizer_for(model), s.optimizer_for(native)
            src, tgt = torch.tensor([[.1, .7]]), torch.tensor([[.9, .3]])
            sim = lambda a, b: (a - b).square().mean()
            reg = lambda a, _: a.square().mean()
            a, b = model(src), model(tgt)
            loss, _, _ = s.joint_loss((a, b, a, b), src, tgt, sim, reg, alpha)
            x, y = native(src), native(tgt)
            expected = (sim(tgt, x) + sim(src, y)) * .5 + (reg(x, 0) + reg(y, 0)) * alpha * .5
            torch.testing.assert_close(loss, expected, rtol=0, atol=1e-7)
            opt.zero_grad(); loss.backward(); opt.step()
            native_opt.zero_grad(); expected.backward(); native_opt.step()
            for p, q in zip(model.parameters(), native.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=1e-7)
            self.assertTrue(all(v['step'] == 1 for v in opt.state.values()))
            self.assertTrue(opt.param_groups[0]['amsgrad'])

    def test_persistent_iterator(self):
        loader = list(range(6))
        iterator = iter(loader)
        epochs = []
        for _ in range(4):
            epoch = []
            for _ in range(2):
                value, iterator = s.next_pair(iterator, loader)
                epoch.append(value)
            epochs.append(epoch)
        self.assertEqual(epochs, [[0, 1], [2, 3], [4, 5], [0, 1]])

    def test_memory_guard(self):
        self.assertAlmostEqual(s.memory_fraction({'free_mib': 38334, 'total_mib': 49140}), 37310 / 49140)
        with self.assertRaises(RuntimeError):
            s.memory_fraction({'free_mib': 36 * 1024, 'total_mib': 49140})

    def test_checkpoint_roundtrip_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            state = torch.nn.Linear(2, 1).state_dict()
            s.atomic_checkpoint(path, {'state_dict': state})
            loaded = torch.load(path, weights_only=False)['state_dict']
            self.assertEqual(s.state_hash(state), s.state_hash(loaded))
            s.verify_hashes({str(path): s.sha256(path)})
            with self.assertRaises(RuntimeError):
                s.verify_hashes({str(path): 'invalid'})


if __name__ == '__main__':
    unittest.main()
