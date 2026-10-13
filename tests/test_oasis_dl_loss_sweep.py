import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
sys.path.insert(0, str(ROOT/"scripts/reports"))
import common as c
import dl_loss_sweep as d
import dl_ncc_diffusion as geometry
import run_oasis_dl_loss_one as worker
import launch_oasis_dl_loss_sweep as launch
import oasis_dl_loss_first10_arc as report


class SweepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_matrix_and_names(self):
        self.assertEqual(len(d.specs()), 350)
        self.assertEqual(len(set(s.name() for s in d.specs())), 350)
        self.assertEqual(len(d.specs(True)), 5)
        self.assertFalse(set(s.name() for s in d.specs()) & set(s.name() for s in d.specs(True)))
        for loss in d.LOSSES:
            self.assertEqual(len([s for s in d.specs() if s.loss == loss]), 70)

    def test_native_scale_centers_and_thirds(self):
        for loss, coefficient in zip(d.LOSSES, (.005, 1., 1., .2, .2)):
            spec = d.FitSpec(0, loss, 1.)
            self.assertAlmostEqual(spec.alpha*spec.scale, coefficient)
            self.assertAlmostEqual(d.FitSpec(0, loss, 1/3).alpha, d.CENTERS[loss]/3, places=16)

    def test_schedule(self):
        stages = d.stages(d.FitSpec(0, "mse", 1.))
        self.assertEqual([s.cps for s in stages], [8, 4, 2, None])
        self.assertEqual([s.factor for s in stages], [1]*4)
        self.assertEqual([s.lr for s in stages], [.03, .01, .003, .001])
        self.assertEqual([s.max_epochs for s in stages], [400, 800, 1300, 2000])
        self.assertEqual([s.patience for s in stages], [25, 35, 50, 100])

    def test_invalid_specs(self):
        for spec in (d.FitSpec(10, "mse", 1.), d.FitSpec(0, "vfa", 1.),
                     d.FitSpec(0, "mse", .33), d.FitSpec(0, "mse", float("nan")),
                     d.FitSpec(0, "mse", 1., seed=2)):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_cli_and_legacy_compatibility(self):
        args = worker.parse_args(["--case-rank", "0", "--loss", "femse", "--multiplier", repr(1/3),
                                  "--gpu", "1", "--oasis-root", "/tmp/data", "--wavereg-code-root", "/tmp/code",
                                  "--vfa-root", "/tmp/vfa", "--out-dir", "/tmp/output", "--source-revision", "test"])
        self.assertEqual(worker.spec_from_args(args), d.FitSpec(0, "femse", 1/3))
        import run_one
        with patch.object(sys, "argv", ["run_one.py", "--method", "direct_cp_8_4_2_be_bspline_analytic_multistage", "--alpha", "1000"]):
            old = run_one.parse_args()
        self.assertEqual(old.stage_alpha, "")
        self.assertEqual(old.dense_stage_lr, .001)

    def test_full_support_and_voxel_diffusion(self):
        shape = (8, 10, 12)
        fixed, moving = torch.rand(shape), torch.rand(shape)
        spec = d.FitSpec(0, "mind", 1.)
        similarity = lambda warped: (warped-fixed[None, None]).square().mean()
        obj = d.Objective(fixed, moving, similarity, spec, d.stages(spec)[-1])
        field = (torch.randn(*shape, 3)*.002).requires_grad_()
        loss, data, reg = obj(field)
        expected_reg = c.forward_valid_diffusion((field*geometry.voxel_scale(shape, device="cpu")).permute(3, 0, 1, 2)[None], None)
        self.assertTrue(torch.equal(reg, expected_reg))
        self.assertTrue(torch.equal(loss, data+reg))
        self.assertGreater(float(torch.autograd.grad(loss, field)[0].abs().sum()), 0.)
        # Every background voxel is retained, not gated by a foreground mask.
        zeros = torch.zeros(shape)
        obj = d.Objective(zeros, torch.ones(shape), lambda x: x.square().mean(),
                          d.FitSpec(0, "mse", 1.), d.stages(spec)[-1])
        self.assertEqual(float(obj(torch.zeros(*shape, 3))[1]), 1.)

    def test_frozen_encoder_after_warp_and_gradient(self):
        encoder = torch.nn.Conv3d(1, 2, 3, padding=1)
        for p in encoder.parameters():
            p.requires_grad_(False)
        fixed, moving = torch.rand(8, 10, 12), torch.rand(8, 10, 12)
        similarity = d.Similarity("femse", SimpleNamespace(first_level_encoder=encoder), fixed[None, None])
        spec = d.FitSpec(0, "femse", 1.)
        obj = d.Objective(fixed, moving, similarity, spec, d.stages(spec)[-1])
        field = torch.zeros(8, 10, 12, 3, requires_grad=True)
        obj(field)[0].backward()
        self.assertGreater(float(field.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_dense_initialization_and_stored_loss_decomposition(self):
        shape = (8, 16, 8)
        shapes = c.build_nested_cp_shapes(shape, 8, [8, 4, 2])
        parameter = torch.randn(*shapes[8], 3)*.001
        for cps0, cps1 in ((8, 4), (4, 2)):
            before = geometry.AlignedBSpline(shapes[cps0], shape, shape, cps0, "cpu")(parameter)
            parameter = c.cubic_bspline_dyadic_knot_insert_3d(parameter, shapes[cps1])
            after = geometry.AlignedBSpline(shapes[cps1], shape, shape, cps1, "cpu")(parameter)
            self.assertTrue(torch.allclose(before, after, atol=1e-7))
        dense = after.detach().clone()
        self.assertTrue(torch.equal(after, dense))
        spec = d.FitSpec(0, "rc", 1., smoke=True)
        fixed, moving = torch.rand(shape), torch.rand(shape)
        objective = d.Objective(fixed, moving, lambda x: (x-fixed[None, None]).square().mean(), spec, d.stages(spec)[-1])
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/"stage3"
            best, summary = d.optimize_stage(dense, objective, d.stages(spec)[-1], out)
            import json
            rows = [json.loads(line) for line in (out/"history.jsonl").read_text().splitlines()]
            for row in rows:
                self.assertAlmostEqual(row["raw_similarity"]*10, row["weighted_similarity"])
                self.assertAlmostEqual(row["loss"], row["weighted_similarity"]+row["weighted_reg_loss"], places=5)
            saved = torch.load(out/"best.pt", weights_only=False)
            self.assertTrue(torch.equal(best, saved["parameter_norm_xyz"]))
            self.assertEqual(summary, saved["summary"])

    def test_incomplete_report_rejected(self):
        with self.assertRaises(RuntimeError):
            report.summarize([])
        with self.assertRaises(RuntimeError):
            report.write_report([], [], Path("/tmp/no_write"), {})

    def test_complete_summary_and_plot_exports(self):
        rows = []
        keys = [(loss, d.CENTERS[loss]*m) for loss in d.LOSSES for m in d.MULTIPLIERS]
        keys += [("ncc", x) for x in (.01, .1, .3, 1., 3., 10., 100.)]+[("VFA", None), ("SITReg", None)]
        for loss, alpha in keys:
            for rank in range(10):
                rows.append({"loss": loss, "effective_lambda": alpha, "case_rank": rank,
                             "similarity_scale": d.SCALES.get(loss, 1.), "actual_regularizer_coefficient": alpha,
                             **{key: .1+.01*rank for key in report.METRICS}})
        means = report.summarize(rows)
        self.assertEqual(len(means), 44)
        self.assertAlmostEqual(means[0]["dice_mean"], .145)
        import numpy as np
        self.assertAlmostEqual(means[0]["dice_std"], np.std([.1+.01*r for r in range(10)], ddof=1))
        with tempfile.TemporaryDirectory() as root:
            report.plot(means, Path(root))
            report.plot(means, Path(root), folding=True)
            self.assertEqual(len(list(Path(root).iterdir())), 6)
        rows[0]["case_rank"] = 1
        with self.assertRaises(RuntimeError):
            report.summarize(rows)


if __name__ == "__main__":
    unittest.main()
