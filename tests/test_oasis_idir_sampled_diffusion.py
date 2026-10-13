import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_alpha_sweep"))
import common as c
import idir_sampled_diffusion as p
import run_oasis_idir_sampled_diffusion_one as worker
import launch_oasis_idir_sampled_diffusion as launch


class SampledContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_grid_is_70_not_2d(self):
        specs = p.cohort(p.INITIAL_EXPONENTS)
        self.assertEqual(len(specs), 70)
        self.assertEqual({s.omega for s in specs}, {32})
        self.assertEqual(len({s.name() for s in specs}), 70)
        self.assertEqual({s.alpha for s in specs}, {.001, .01, .1, 1., 10., 100., 1000.})
        self.assertEqual({s.updates for s in specs}, {2500})
        self.assertEqual(p.SLOTS, (1, 1, 3, 3))

    def test_extension_names_and_smokes_disjoint(self):
        specs = p.cohort(range(-5, 6))
        self.assertEqual(len({s.name() for s in specs}), 110)
        self.assertNotIn(p.FitSpec(0, 0, 32, 100).name(), {s.name() for s in specs})
        for spec in (p.FitSpec(10, 0), p.FitSpec(0, 0, 30), p.FitSpec(0, -38), p.FitSpec(0, 0, 32, 200)):
            with self.assertRaises(ValueError): spec.validate()

    def test_exact10000_random_without_replacement_same_sequence(self):
        pool = torch.arange(12001 * 3).reshape(12001, 3).float()
        torch.manual_seed(1)
        coords, indices = p.sample_coords(pool)
        self.assertEqual(coords.shape, (10000, 3)); self.assertTrue(coords.requires_grad)
        self.assertEqual(len(indices.unique()), 10000)
        self.assertTrue(torch.equal(coords.detach(), pool[indices]))
        _, next_indices = p.sample_coords(pool)
        self.assertFalse(torch.equal(indices, next_indices))
        torch.manual_seed(1)
        self.assertTrue(torch.equal(indices, p.sample_coords(pool)[1]))

    def test_global_one_minus_ncc_only_constant_shift(self):
        torch.manual_seed(1)
        fixed, moving = torch.rand(10000), torch.rand(10000, requires_grad=True)
        old = c.ncc_loss(fixed, moving)
        new = 1 + c.ncc_loss(fixed, moving)
        self.assertAlmostEqual(float(new - old), 1., places=6)
        a, = torch.autograd.grad(old, moving)
        b, = torch.autograd.grad(new, moving)
        self.assertTrue(torch.equal(a, b))

    def test_affine_diffusion_units_values_and_training_gradients(self):
        torch.manual_seed(2)
        coords = torch.rand(19, 3, requires_grad=True)
        matrix = torch.nn.Parameter(torch.randn(3, 3) * .02)
        bias = torch.nn.Parameter(torch.randn(3))
        displacement = coords @ matrix.T + bias
        shape = (11, 21, 31)
        scale = torch.tensor([15., 10., 5.])
        factor = scale[:, None] / scale[None, :]
        actual = p.analytic_diffusion(coords, displacement, shape)
        expected = (matrix * factor).square().mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(matrix.grad, 2 * matrix.detach() * factor.square() / 9)
        self.assertIsNone(bias.grad)

    def test_translation_zero_diffusion_and_no_extra_batch_division(self):
        for n in (10, 10000):
            coords = torch.rand(n, 3, requires_grad=True)
            zero = p.analytic_diffusion(coords, coords * 0 + .2)
            identity_displacement = p.analytic_diffusion(coords, coords)
            self.assertEqual(float(zero), 0.)
            self.assertAlmostEqual(float(identity_displacement), 1 / 3, places=6)

    def test_only_10000_network_inputs_for_one_update(self):
        torch.manual_seed(3)
        model = c.build_siren(16, 3, 32)
        seen = []
        handle = model.register_forward_pre_hook(lambda _model, args: seen.append(args[0].shape))
        coords = (torch.rand(10000, 3) * 1.8 - .9).requires_grad_()
        fixed, moving = torch.rand(8, 9, 10), torch.rand(8, 9, 10)
        opt = p.optimizer(model)
        total, data, reg = p.objective(model, coords, fixed, moving, .1)
        total.backward(); opt.step(); handle.remove()
        self.assertEqual(seen, [torch.Size([10000, 3])])
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters()))
        self.assertEqual({int(s["step"]) for s in opt.state.values()}, {1})
        self.assertEqual(opt.param_groups[0]["lr"], .0001)

    def test_diffusion_matches_posthoc_analytic_for_native_shape(self):
        torch.manual_seed(4)
        model = c.build_siren(16, 3, 32)
        coords = torch.rand(21, 3, requires_grad=True)
        actual = p.analytic_diffusion(coords, model(coords))
        _, values = c.idir_autograd_diffusion_density(model, coords)
        torch.testing.assert_close(actual, values.mean(), atol=1e-7, rtol=1e-5)

    def mean_row(self, exponent, dice, omega=32):
        return {"omega": omega, "exponent": exponent, "n": 10, "alpha": 10. ** exponent,
                "dice_mean_mean": dice, "hd95_mean_mean": 2., "asd_symmetric_mean_mean": .4}

    def test_boundary_extension_only_after_complete_cohorts(self):
        exps = list(p.INITIAL_EXPONENTS)
        means = [self.mean_row(e, .8 - abs(e) * .01) for e in exps]
        decision = p.boundary_decision(means, exps)
        self.assertIsNone(decision["extend_exponent"])
        self.assertEqual(decision["selected"]["exponent"], 0)
        means[0]["dice_mean_mean"] = .9
        self.assertEqual(p.boundary_decision(means, exps)["extend_exponent"], -4)
        means[-1]["dice_mean_mean"] = .95
        self.assertEqual(p.boundary_decision(means, exps)["extend_exponent"], 4)
        with self.assertRaises(ValueError): p.boundary_decision(means[:-1], exps)

    def test_ties_and_exact_plateau_do_not_extend_forever(self):
        means = [self.mean_row(e, .8) for e in range(-4, 4)]
        self.assertEqual(p.select(means)["exponent"], 0)
        means[1]["asd_symmetric_mean_mean"] = .3
        self.assertEqual(p.select(means)["exponent"], -3)

    def test_incomplete_cohorts_have_no_mean(self):
        rows = [{"rank": rank, "omega": 32, "exponent": 0, **{k: .1 for k in p.METRICS}} for rank in range(9)]
        self.assertEqual(p.aggregate(rows), [])
        rows.append({**rows[0], "rank": 9})
        means = p.aggregate(rows)
        self.assertEqual(len(means), 1)
        self.assertAlmostEqual(means[0]["dice_mean_mean"], .1)

    def test_cli_requires_two_slots_per_requested_gpu(self):
        args = worker.parse_args(["--rank", "0", "--exponent", "-3", "--gpu", "3", "--slot", "2",
                                  "--oasis-root", "/tmp/input", "--out-dir", "/tmp/output"])
        self.assertEqual(worker.spec_from_args(args), p.FitSpec(0, -3))
        args = launch.parse_args(["--oasis-root", "/tmp/input"])
        self.assertEqual(args.gpu_slots, "1,1,3,3")


if __name__ == "__main__":
    unittest.main()
