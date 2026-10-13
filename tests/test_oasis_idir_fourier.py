from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_alpha_sweep"))
import common as c
import idir_sampled_diffusion as old
import idir_fourier_diffusion as p
import run_oasis_idir_fourier_one as worker
import launch_oasis_idir_fourier as launch


class FourierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_grid_230_new_plus10_reused(self):
        specs = p.grid()
        self.assertEqual(len(specs), 240)
        self.assertEqual(len({s.name() for s in specs}), 240)
        self.assertEqual(sum(s.family == "fourier" for s in specs), 180)
        self.assertEqual(sum(not (s.family == "plain" and s.width == 256 and s.alpha == 1) for s in specs), 230)
        self.assertEqual(p.GPUS, (0, 1, 2, 3))

    def test_names_types_and_validation(self):
        self.assertEqual(p.FitSpec(0, 1).name(), p.FitSpec(0, 1.).name())
        self.assertNotEqual(p.FitSpec(0, 1, 1).name(), p.FitSpec(0, 1, 1, smoke_steps=100).name())
        for spec in (p.FitSpec(10, 1), p.FitSpec(0, 0), p.FitSpec(0, 1, 128), p.FitSpec(0, 1, width=128), p.FitSpec(0, 1, 1, width=286)):
            with self.assertRaises(ValueError): spec.validate()

    def test_parameter_counts_and_native_fan_initialization(self):
        for spec, count in ((p.FitSpec(0, 1), 133379), (p.FitSpec(0, 1, 1), 166147), (p.FitSpec(0, 1, width=286), 166169)):
            torch.manual_seed(1); model = p.build_model(spec)
            self.assertEqual(spec.param_count, count)
            self.assertEqual(c.count_trainable_params(model), count)
            network = model.network if spec.sigma else model
            self.assertLessEqual(float(network.layers[0].weight.abs().max()), 1/spec.input_dim)

    def test_encoding_formula_and_frozen_matrix(self):
        torch.manual_seed(1); model = p.build_model(p.FitSpec(0, 1, 2))
        q = torch.tensor([[-1., 0., 1.], [.2, -.3, .5]], requires_grad=True)
        phase = 2*torch.pi*2*((q+1)/2 @ model.B0.T)
        expected = torch.cat((q, phase.sin(), phase.cos()), -1)
        torch.testing.assert_close(model.encode(q), expected)
        self.assertEqual(model.encode(q).shape, (2, 131))
        self.assertFalse(model.B0.requires_grad)
        self.assertNotIn("B0", dict(model.named_parameters()))

    def test_matrix_reused_across_bandwidths_nested_feature_counts(self):
        a = p.build_model(p.FitSpec(0, 1, .5))
        b = p.build_model(p.FitSpec(9, 3, 16))
        d = p.build_model(p.FitSpec(0, 1, 1, features=128))
        torch.testing.assert_close(a.B0, b.B0, rtol=0, atol=0)
        torch.testing.assert_close(a.B0, d.B0[:64], rtol=0, atol=0)

    def test_encoding_does_not_consume_global_random_state(self):
        torch.manual_seed(10); state = torch.get_rng_state()
        p.FourierSiren(p.FitSpec(0, 1, 1))
        actual = torch.get_rng_state()
        torch.set_rng_state(state); c.build_siren(256, 3, 32, input_dim=131)
        self.assertTrue(torch.equal(actual, torch.get_rng_state()))

    def test_sampler_reproduces_legacy_and_isolated_from_architecture(self):
        pool = torch.arange(12005*3).reshape(-1, 3).float()
        torch.manual_seed(1)
        expected = [old.sample_coords(pool)[1] for _ in range(3)]
        generator = torch.Generator().manual_seed(1)
        p.build_model(p.FitSpec(0, 1, 4))
        for indices in expected:
            coords, actual = p.sample_coords(pool, generator)
            self.assertTrue(torch.equal(actual, indices))
            self.assertEqual(coords.shape, (10000, 3)); self.assertTrue(coords.requires_grad)

    def test_original_coordinate_derivative_numerical(self):
        for sigma in (.5, 16.):
            torch.manual_seed(1)
            model = p.build_model(p.FitSpec(0, 1, sigma))
            self.assertTrue(p.derivative_check(model, torch.device("cpu"))["passed"])

    def test_diffusion_includes_fourier_chain_and_parameter_gradient(self):
        torch.manual_seed(2)
        model = p.build_model(p.FitSpec(0, 1, 4)).double()
        q = torch.tensor([[.2, -.1, .3], [-.4, .1, .6]], dtype=torch.float64, requires_grad=True)
        u = model(q)
        analytic = old.analytic_diffusion(q, u)
        _, density = c.idir_autograd_diffusion_density(model, q)
        torch.testing.assert_close(analytic, density.mean())
        weight = model.network.layers[0].weight
        gradient, = torch.autograd.grad(analytic, weight)
        base = float(weight[0, 0]); eps = 1e-6
        with torch.no_grad(): weight[0, 0] = base+eps
        plus = float(old.analytic_diffusion(q, model(q)).detach())
        with torch.no_grad(): weight[0, 0] = base-eps
        minus = float(old.analytic_diffusion(q, model(q)).detach())
        with torch.no_grad(): weight[0, 0] = base
        self.assertAlmostEqual(float(gradient[0, 0]), (plus-minus)/(2*eps), delta=1e-7)

    def test_one_step_exact10000_with_frozen_B0(self):
        torch.manual_seed(1); model = p.build_model(p.FitSpec(0, 1, 2))
        q = (torch.rand(10000, 3)*1.8-.9).requires_grad_()
        fixed, moving = torch.rand(8, 9, 10), torch.rand(8, 9, 10)
        seen = []; handle = model.register_forward_pre_hook(lambda _m, x:seen.append(x[0].shape))
        before = model.B0.clone(); opt = old.optimizer(model)
        total, _, _ = old.objective(model, q, fixed, moving, 1.)
        total.backward(); opt.step(); handle.remove()
        self.assertEqual(seen, [torch.Size((10000, 3))])
        self.assertTrue(torch.equal(before, model.B0))
        self.assertEqual({int(v["step"]) for v in opt.state.values()}, {1})

    def test_diagnostics_no_parameter_or_gradient_change(self):
        torch.manual_seed(1); model = p.build_model(p.FitSpec(0, 1, .5))
        q = torch.rand(100, 3, requires_grad=True)
        digest = old.state_hash(model)
        values = p.initial_diagnostics(model, q, torch.rand(8, 9, 10), torch.rand(8, 9, 10), 1.)
        self.assertEqual(old.state_hash(model), digest)
        self.assertTrue(all(v.grad is None for v in model.parameters()))
        self.assertGreater(values["data_grad_norm"], 0)

    def means(self):
        rows = []
        for spec in p.grid():
            dice = .8 - abs(spec.sigma-2)*.001 - abs(spec.alpha-1)*.001
            rows.append({"spec": asdict(spec), "rank": spec.rank, **{k: dice if k == "dice_mean" else .5 for k in p.METRICS}})
        return p.aggregate(rows)

    def test_complete_cohort_boundary_logic(self):
        means = self.means()
        decision = p.boundary_decision(means, p.SIGMAS, p.LAMBDAS)
        self.assertIsNone(decision["extra_sigma"]); self.assertEqual(decision["extra_lambdas"], [])
        with self.assertRaises(ValueError): p.boundary_decision(means[:-1], p.SIGMAS, p.LAMBDAS)
        last = next(r for r in means if r["sigma"] == 16 and r["alpha"] == 3)
        last["dice_mean_mean"] = .99
        decision = p.boundary_decision(means, p.SIGMAS, p.LAMBDAS)
        self.assertEqual(decision["extra_sigma"], 32)
        self.assertEqual(decision["extra_lambdas"], [30.])

    def test_followups_are70_unique_and_gate_regularities(self):
        decision = p.boundary_decision(self.means(), p.SIGMAS, p.LAMBDAS)
        self.assertTrue(p.promising(decision))
        jobs = p.followups(decision)
        self.assertEqual(len(jobs), 70); self.assertEqual(len({s.name() for s in jobs}), 70)
        decision["selected_fourier"]["jdet_le0_pct_fg_mean"] = 1.
        self.assertFalse(p.promising(decision))

    def test_incomplete_not_aggregated(self):
        self.assertEqual(p.aggregate([{"spec":asdict(p.FitSpec(0, 1)), "rank":0, **{k:0.5 for k in p.METRICS}}]), [])

    def test_cli_all4gpus_and_serialization(self):
        for gpu in p.GPUS:
            args = worker.parse_args(["--spec-json", json.dumps(asdict(p.FitSpec(0, 1, 2))), "--gpu", str(gpu),
                                      "--oasis-root", "/tmp/source", "--out-dir", "/tmp/output"])
            self.assertEqual(args.spec, p.FitSpec(0, 1, 2))
        self.assertEqual(launch.parse_args(["--oasis-root", "/tmp/source"]).gpus, "0,1,2,3")


if __name__ == "__main__": unittest.main()
