import copy
from dataclasses import replace
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
import dl_ncc_diffusion as d
import idir_ncc_diffusion as p
import run_oasis_idir_ncc_diffusion_one as worker
import launch_oasis_idir_ncc_diffusion_sweep as launcher


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_grid_unique_names_and_exact_third(self):
        jobs = p.specs()
        self.assertEqual(len(jobs), 250)
        self.assertEqual(len({s.name() for s in jobs}), 250)
        self.assertEqual(p.LAMBDAS["1over3"], 1 / 3)
        self.assertNotEqual(p.LAMBDAS["1over3"], .3)
        self.assertFalse({s.name() for s in jobs} & {s.name() for s in launcher.smoke_specs()})
        self.assertEqual({s.omega for s in jobs}, {8, 16, 32, 64, 128})
        self.assertEqual({s.updates for s in jobs}, {2500})

    def test_invalid_specs(self):
        spec = p.FitSpec(0, 32, "1")
        for item in (replace(spec, rank=10), replace(spec, omega=30), replace(spec, lambda_key="0.3"), replace(spec, smoke_steps=2500)):
            with self.assertRaises(ValueError):
                item.validate()

    def test_cli_and_old_runner_compatibility(self):
        args = worker.parse_args(["--case-rank", "0", "--omega", "32", "--lambda-key", "1over3", "--gpu", "1",
                                  "--oasis-root", "/tmp/input", "--dl-code-root", "/tmp/dl", "--vfa-root", "/tmp/vfa", "--out-dir", "/tmp/result"])
        self.assertEqual(worker.spec_from_args(args), p.FitSpec(0, 32, "1over3"))
        import run_one
        from unittest.mock import patch
        with patch.object(sys, "argv", ["run_one.py", "--method", "idir_original", "--alpha", "1"]):
            old = run_one.parse_args()
        self.assertEqual(old.idir_omega, 32.)

    def test_initialization_native_scaling_and_fixed_seed(self):
        torch.manual_seed(1); a = c.build_siren(256, 3, 32)
        torch.manual_seed(1); b = c.build_siren(256, 3, 64)
        self.assertEqual(c.count_trainable_params(a), 133379)
        self.assertTrue(torch.equal(a.layers[0].weight, b.layers[0].weight))
        for i in range(1, 4):
            self.assertTrue(torch.equal(a.layers[i].weight / 2, b.layers[i].weight))
        for i in range(4):
            self.assertTrue(torch.equal(a.layers[i].bias, b.layers[i].bias))
        torch.manual_seed(1); again = c.build_siren(256, 3, 32)
        self.assertEqual(p.state_hash(a), p.state_hash(again))

    def test_chunked_gradients_and_adam_step_all_frequencies(self):
        shape = (5, 6, 7)
        for omega in p.OMEGAS:
            torch.manual_seed(11)
            objective = p.FullVolumeObjective(torch.rand(shape), torch.rand(shape), 1 / 3)
            model = c.build_siren(32, 3, omega)
            other = copy.deepcopy(model)
            opt_a, opt_b = p.optimizer(model), p.optimizer(other)
            q = model(objective.coords).reshape(*shape, 3)
            values = objective(q)
            values[0].backward()
            got = p.full_volume_gradients(other, objective, chunk_size=47)
            self.assertAlmostEqual(got["loss"], float(values[0]), places=6)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=5e-4)
            opt_a.step(); opt_b.step()
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)

    def test_matches_existing_fullres_objective(self):
        torch.manual_seed(9)
        shape = (6, 7, 8)
        fixed, moving = torch.rand(shape), torch.rand(shape)
        q = (torch.randn(*shape, 3) * .01).requires_grad_()
        old = d.StageObjective(fixed, moving, .1, d.StageSpec(3, None, 1, .001, 1, 1, 1, 0))
        new = p.FullVolumeObjective(fixed, moving, .1)
        a, b = old(q), new(q)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        ga, = torch.autograd.grad(a[0], q)
        gb, = torch.autograd.grad(b[0], q)
        self.assertTrue(torch.equal(ga, gb))

    def test_voxel_scaling_axis_order_and_full_support(self):
        shape = (5, 7, 9)
        obj = p.FullVolumeObjective(torch.ones(shape), torch.ones(shape), 3.)
        q = torch.zeros(*shape, 3)
        # u_x increments1voxel along x. Mean over3components and3axes =1/9.
        q[..., 0] = torch.arange(shape[2])[None, None, :] / ((shape[2] - 1) / 2)
        total, data, reg = obj(q)
        self.assertAlmostEqual(float(reg), 1 / 9, places=7)
        self.assertAlmostEqual(float(total - data), 1 / 3, places=6)
        zero = torch.zeros(*shape, 3)
        self.assertEqual(float(obj(zero)[2]), 0.)
        # No image-positive mask is consulted; constant translation has zero diffusion.
        self.assertEqual(float(obj(zero + .07)[2]), 0.)
        torch.manual_seed(4)
        bg = p.FullVolumeObjective(torch.zeros(shape), torch.rand(shape), 1.)
        self.assertAlmostEqual(float(bg(q)[2]), 1 / 9, places=7)

    def test_optimizer_budget_and_parameters(self):
        model = c.build_siren(256, 3, 32)
        group = p.optimizer(model).param_groups[0]
        self.assertEqual(group["lr"], 1e-4)
        self.assertEqual(group["betas"], (.9, .999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertFalse(group["amsgrad"])
        self.assertEqual(group["weight_decay"], 0)
        self.assertEqual({id(x) for x in group["params"]}, {id(x) for x in model.parameters()})

    def test_label_details_and_absent_handling(self):
        rows = [{"label_id": i, "included_in_case_mean": i < 35, "exclusion_reason": "" if i < 35 else "absent_in_fixed",
                 "dice": .8 if i < 35 else None, "hd95_mm": 2. if i < 35 else None,
                 "asd_symmetric_mm": .4 if i < 35 else None} for i in range(1, 36)]
        summary = dict(dice_mean=.8, hd95_mean=2., asd_symmetric_mean=.4)
        p.validate_label_records(rows, summary)
        rows[-1]["dice"] = 0.
        with self.assertRaises(ValueError):
            p.validate_label_records(rows, summary)

    def test_partial_report_never_averages_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            spec = p.FitSpec(0, 8, "0.1")
            row = {key: .1 for key in (*p.METRICS, "diffusion_l2_full", "training_seconds", "case_seconds", "terminal_loss")}
            row.update(rank=0, omega=8, lambda_key="0.1", alpha=.1, fixed_id="0333", moving_id="0115",
                       direction="moving_to_fixed", valid_label_count=35, actual_updates=2500,
                       physical_gpu=1, best_loss_epoch_diagnostic=2500, run_dir=str(run), field_sha256="test")
            p.write_json(run / "summary.json", row)
            p.write_json(run / "per_label.json", [])
            obj = object.__new__(launcher.Controller)
            obj.args = SimpleNamespace(review_root=root / "review")
            obj.state = {"results": {spec.name(): str(run)}}
            obj.baselines = [{"representation": group, "case_rank": rank, "dice": .8, "hd95": 2., "asd": .4,
                              "log_jdet_std_fg": .6, "jdet_le0_pct_fg": .1}
                             for group in ("VFA", "SITReg") for rank in range(10)]
            obj.report(final=False)
            self.assertEqual(json.loads((obj.args.review_root / "means.json").read_text()), [])
            self.assertEqual(len(json.loads((obj.args.review_root / "per_case.json").read_text())), 1)
            self.assertFalse((obj.args.review_root / "ACCEPTED.json").exists())
            with self.assertRaises(RuntimeError):
                obj.report(final=True)

    def test_lines_only_plot_with_both_reference_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            means = []
            for omega in p.OMEGAS:
                for key, alpha in p.LAMBDAS.items():
                    row = {"omega": omega, "lambda_key": key, "alpha": alpha}
                    row.update({k + "_mean": .2 + alpha / 100 for k in p.METRICS})
                    means.append(row)
            references = [{"group": group, **{k + "_mean": .5 for k in p.METRICS}} for group in ("VFA", "SITReg")]
            launcher.Controller.plot(means, references, Path(tmp))
            for extension in ("png", "svg", "pdf"):
                self.assertTrue((Path(tmp) / f"omega_lambda_arc_lines.{extension}").is_file())


if __name__ == "__main__":
    unittest.main()
