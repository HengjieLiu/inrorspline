import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
sys.path.insert(0, str(ROOT/"scripts/reports"))
import common as c
import gncc_dense as d
import dl_ncc_diffusion as base
import run_oasis_gncc_dense_one as worker
import launch_oasis_gncc_dense as launch
import oasis_gncc_dense_report as report


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(1)

    def test_matrix(self):
        self.assertEqual(len(d.specs()), 140)
        self.assertEqual(len({s.name() for s in d.specs()}), 140)
        self.assertEqual(len(d.specs(True)), 3)
        self.assertFalse({s.name() for s in d.specs()} & {s.name() for s in d.specs(True)})
        self.assertEqual({s.alpha for s in d.specs()}, set(base.LAMBDAS))

    def test_cli(self):
        args = worker.parse_args(["--case-rank", "0", "--loss", "gncc2", "--alpha", ".3", "--gpu", "3",
                                  "--oasis-root", "/tmp/data", "--out-dir", "/tmp/out"])
        self.assertEqual(args.alpha, .3)
        self.assertEqual(launch.parse_args(["--oasis-root", "/tmp/data", "--dry-run"]).max_concurrent, 4)

    def test_invalid(self):
        for spec in (d.FitSpec(10, "gncc", 1), d.FitSpec(0, "gncc", float("nan")),
                     d.FitSpec(0, "gncc", 1, seed=2), d.FitSpec(1, "lncc", 1), d.FitSpec(0, "mse", 1)):
            with self.assertRaises(ValueError): spec.validate()

    def test_schedule_and_native_legacy_unchanged(self):
        spec = d.FitSpec(0, "gncc", .3)
        stages = d.stages(spec)
        self.assertEqual([s.cps for s in stages], [8, 4, 2, None])
        self.assertEqual([s.factor for s in stages], [1]*4)
        self.assertEqual([s.lr for s in stages], [.03, .01, .003, .001])
        self.assertEqual([s.patience for s in stages], [25, 35, 50, 100])
        self.assertEqual([s.max_epochs for s in stages], [400, 800, 1300, 2000])
        self.assertFalse(base.PROTOCOL["initial_loss_normalization"])
        self.assertEqual(base.PROTOCOL["loss"], "1_mean_squared_local_correlation_plus_lambda_diffusion")

    def test_pearson_value_and_gradient(self):
        fixed = torch.randn(10000, dtype=torch.float64)
        moving = torch.randn(10000, dtype=torch.float64, requires_grad=True)
        expected = torch.corrcoef(torch.stack((fixed, moving)))[0, 1]
        actual = d.GNCC(fixed).correlation(moving)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12))
        ga, = torch.autograd.grad(actual, moving, retain_graph=True)
        ge, = torch.autograd.grad(expected, moving)
        self.assertTrue(torch.allclose(ga, ge, atol=1e-12))

    def test_identical_anticorrelated_and_affine(self):
        fixed = torch.randn(8, 10, 12)
        for squared in (False, True):
            loss = d.GNCC(fixed, squared)
            self.assertAlmostEqual(float(loss(fixed)), 0, places=6)
            self.assertAlmostEqual(float(loss(2*fixed+3)), 0, places=6)
            self.assertAlmostEqual(float(loss(-fixed)), 0 if squared else 2, places=6)

    def test_constant_and_near_constant_finite(self):
        for fixed in (torch.zeros(8, 10, 12), torch.ones(8, 10, 12), torch.ones(8, 10, 12)+torch.randn(8, 10, 12)*1e-6):
            for squared in (False, True):
                moving = fixed.clone().requires_grad_()
                loss = d.GNCC(fixed, squared)(moving)
                grad, = torch.autograd.grad(loss, moving)
                self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(grad).all())

    def test_squared_chain_rule_and_gradcheck(self):
        fixed = torch.randn(2, 3, 4, dtype=torch.float64)
        moving = (fixed*.5+torch.randn_like(fixed)).requires_grad_()
        a, b = d.GNCC(fixed), d.GNCC(fixed, True)
        ga, = torch.autograd.grad(a(moving), moving, retain_graph=True)
        gb, = torch.autograd.grad(b(moving), moving)
        self.assertTrue(torch.allclose(gb, 2*a.correlation(moving).detach()*ga, atol=1e-12))
        self.assertTrue(torch.autograd.gradcheck(b, (moving,), eps=1e-6, atol=1e-5))

    def test_no_legacy_std_and_full_volume(self):
        fixed = torch.zeros(8, 10, 12); fixed[:4] = torch.rand(4, 10, 12)
        moving = fixed.clone(); changed = moving.clone(); changed[4:] = 2
        with patch.object(c.StableStd, "apply", side_effect=AssertionError("Legacy Std used")):
            self.assertGreater(float(d.GNCC(fixed)(changed)), float(d.GNCC(fixed)(moving)))

    def test_objective_units_and_lncc_parity(self):
        shape = (8, 16, 8)
        fixed, moving = torch.rand(shape), torch.rand(shape)
        field = (torch.randn(*shape, 3)*.001).requires_grad_()
        spec = d.FitSpec(0, "gncc", .3)
        obj = d.Objective(fixed, moving, d.similarity(spec.loss, fixed[None, None]), spec, d.stages(spec)[-1])
        loss, data, reg = obj(field)
        expected = c.forward_valid_diffusion((field*base.voxel_scale(shape, device="cpu")).permute(3, 0, 1, 2)[None], None)
        self.assertTrue(torch.equal(reg, expected)); self.assertTrue(torch.equal(loss, data+.3*reg))
        for cps in (8, None):
            control_shape = c.build_nested_cp_shapes(shape, 8, [8, 4, 2])[8] if cps else None
            parameter = (torch.randn(*(control_shape or shape), 3)*.001).requires_grad_()
            spec = d.FitSpec(0, "lncc", 1., smoke=True)
            stage = d.stages(spec)[0 if cps else -1]
            a = d.Objective(fixed, moving, d.similarity("lncc", fixed[None, None]), spec, stage, control_shape)
            b = base.StageObjective(fixed, moving, 1., stage, control_shape)
            va, vb = a(parameter), b(parameter)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(va, vb)))
            ga, = torch.autograd.grad(va[0], parameter); gb, = torch.autograd.grad(vb[0], parameter)
            self.assertTrue(torch.equal(ga, gb))

    def test_stage_checkpoint(self):
        shape = (8, 16, 8)
        spec = d.FitSpec(0, "gncc2", 1., smoke=True)
        fixed, moving = torch.rand(shape), torch.rand(shape)
        stage = d.stages(spec)[-1]
        obj = d.Objective(fixed, moving, d.similarity(spec.loss, fixed[None, None]), spec, stage)
        with tempfile.TemporaryDirectory() as root:
            best, summary = d.optimize_stage(torch.zeros(*shape, 3), obj, stage, Path(root)/"stage3")
            saved = torch.load(Path(root)/"stage3/best.pt", weights_only=False)
            self.assertTrue(torch.equal(best, saved["parameter_norm_xyz"]))
            self.assertEqual(summary["similarity_scale"], 1.)
            self.assertEqual(summary["loss_key"], "gncc2")

    def test_report_completeness_and_ellipse_layout(self):
        with self.assertRaises(RuntimeError): report.summarize([])
        rows = []
        for key, alpha in [(k, a) for k in report.NCC_KEYS for a in d.LAMBDAS]+[("VFA", None), ("SITReg", None)]:
            rows += [{"loss": key, "effective_lambda": alpha, "case_rank": r, **{m: .1+.01*r for m in d.METRICS}} for r in range(10)]
        means = report.summarize(rows)
        self.assertEqual(len(means), 23)
        self.assertAlmostEqual(means[0]["dice_std"], np.std([.1+.01*r for r in range(10)], ddof=1))
        with tempfile.TemporaryDirectory() as root:
            with patch.object(report.plt, "close"):
                report.plot(means, Path(root))
                fig = report.plt.gcf()
                self.assertEqual(len(fig.axes), 6)
                self.assertTrue(all(len(ax.patches) == 23 for ax in fig.axes[:3]))
                self.assertTrue(all(len(ax.patches) == 0 for ax in fig.axes[3:]))
                for i in range(3):
                    self.assertEqual(fig.axes[i].get_xlim(), fig.axes[i+3].get_xlim())
                    self.assertEqual(fig.axes[i].get_ylim(), fig.axes[i+3].get_ylim())
            report.plt.close(fig)
            self.assertEqual(len(list(Path(root).iterdir())), 3)
        rows[0]["case_rank"] = 1
        with self.assertRaises(RuntimeError): report.summarize(rows)


if __name__ == "__main__": unittest.main()
