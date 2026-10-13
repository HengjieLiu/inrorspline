from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common as c  # noqa: E402
import launch_oasis_rank000_vfa_sitreg_bspline_fit as launcher  # noqa: E402
import run_oasis_saved_field_bspline_fit_one as runner  # noqa: E402


class SavedFieldBsplineFitContracts(unittest.TestCase):
    def test_target_selection_and_cps_parser(self) -> None:
        self.assertEqual(runner.target_spec("vfa").method_key, "bs3_vfa_v0")
        self.assertEqual(runner.target_spec("sitreg").method_key, "bs4_sitreg_v0")
        for target in ("vfa", "sitreg"):
            for cps in (8, 4, 2, 1):
                with self.subTest(target=target, cps=cps):
                    args = runner.parse_args(["--target", target, "--cps", str(cps), "--lr", "0.001"])
                    self.assertEqual(args.target, target)
                    self.assertEqual(args.cps, cps)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            runner.parse_args(["--target", "vfa", "--cps", "3", "--lr", "0.001"])

    def test_expected_control_shapes(self) -> None:
        expected_shapes = {
            8: (24, 32, 28),
            4: (44, 60, 52),
            2: (84, 116, 100),
            1: (164, 228, 196),
        }
        self.assertEqual(runner.EXPECTED_CONTROL_SHAPES, expected_shapes)
        for cps, expected in expected_shapes.items():
            with self.subTest(cps=cps):
                control = torch.zeros((*expected, 3))
                dense = c.expand_bspline_controls(control, expected, c.IMAGE_SHAPE, cps)
                self.assertEqual(tuple(dense.shape), (*c.IMAGE_SHAPE, 3))

    def test_unique_candidate_run_names_and_longest_first(self) -> None:
        jobs = launcher.build_jobs(["0", "1", "2", "3"])
        self.assertEqual(len(jobs), 24)
        names = [launcher.run_name(job, 0) for job in jobs]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all("fullvoxmse" in name and "attempt0" in name for name in names))
        self.assertEqual([job.cps for job in jobs[:6]], [1] * 6)

    def test_voxel_loss_component_scaling_and_full_volume(self) -> None:
        shape = (3, 5, 7)
        pred = torch.zeros((*shape, 3), dtype=torch.float32)
        target = torch.zeros_like(pred)
        pred[..., 0] = 1.0
        pred[..., 1] = 2.0
        pred[..., 2] = 3.0
        expected = ((1.0 * 3.0) ** 2 + (2.0 * 2.0) ** 2 + (3.0 * 1.0) ** 2) / 3.0
        self.assertAlmostEqual(float(runner.full_voxel_mse(pred, target)), expected, places=6)
        pred[0, 0, 0, 0] = 2.0
        manual = runner.voxel_residual(pred - target, shape).square().mean()
        self.assertAlmostEqual(float(runner.full_voxel_mse(pred, target)), float(manual), places=6)

    def test_field_metrics_full_objective_and_secondary_mask(self) -> None:
        shape = (3, 4, 5)
        pred = torch.zeros((*shape, 3))
        target = torch.zeros_like(pred)
        pred[..., 0] = 1.0
        mask = torch.zeros(shape)
        mask[0] = 1
        metrics = runner.field_error_metrics(pred, target, mask)
        self.assertEqual(metrics["objective_mask"], "none")
        self.assertEqual(metrics["objective_regularizer"], "none")
        self.assertEqual(metrics["objective_voxel_count"], int(np.prod(shape)))
        self.assertEqual(metrics["objective_scalar_count"], int(np.prod(shape) * 3))
        self.assertEqual(metrics["fixed_positive_voxel_count"], 20)
        self.assertAlmostEqual(metrics["full_voxel_mse"], metrics["fixed_positive_voxel_mse"])

    def test_synthetic_bspline_representation_recovery_all_cps(self) -> None:
        shape = (8, 9, 10)
        for cps in (8, 4, 2, 1):
            with self.subTest(cps=cps):
                control_shape = c.bspline_control_shape(shape, cps)
                generator = torch.Generator().manual_seed(cps)
                control = torch.randn((*control_shape, 3), generator=generator) * 0.01
                target = c.expand_bspline_controls(control, control_shape, shape, cps)
                recovered = c.expand_bspline_controls(control.clone(), control_shape, shape, cps)
                self.assertEqual(float(runner.full_voxel_mse(recovered, target)), 0.0)

    def test_zero_target_one_adam_step_remains_exact_all_cps(self) -> None:
        shape = (8, 9, 10)
        for cps in (8, 4, 2, 1):
            with self.subTest(cps=cps):
                control_shape = c.bspline_control_shape(shape, cps)
                control = torch.nn.Parameter(torch.zeros((*control_shape, 3)))
                target = torch.zeros((*shape, 3))
                optimizer = torch.optim.Adam([control], lr=1e-2)
                pred = c.expand_bspline_controls(control, control_shape, shape, cps)
                loss = runner.full_voxel_mse(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.assertEqual(float(loss), 0.0)
                self.assertEqual(torch.count_nonzero(control).item(), 0)

    def test_plateau_contract(self) -> None:
        rows = [{"epoch": epoch, "mse_voxel": 1.0} for epoch in range(1, 501)]
        stop, reason = runner.plateau_decision(
            rows,
            min_epochs=500,
            check_interval=50,
            median_window=50,
            patience=200,
            relative_threshold=1e-6,
        )
        self.assertTrue(stop)
        self.assertTrue(reason.startswith("adaptive_median_plateau"))

    def test_lr_selection_tie_breaks(self) -> None:
        rows = []
        for target in launcher.TARGETS:
            for cps in launcher.CPS_VALUES:
                for lr, norm in ((1e-2, 2.0), (3e-3, 1.0), (1e-3, 1.0)):
                    rows.append(
                        {
                            "target": target,
                            "cps": cps,
                            "lr": lr,
                            "full_voxel_mse": 1.0,
                            "full_normalized_grid_mse": norm,
                        }
                    )
        selected = launcher.select_rows(rows)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(float(row["lr"]) == 1e-3 for row in selected))

    def test_nested_space_tolerance(self) -> None:
        rows = []
        for target in launcher.TARGETS:
            for cps, mse in zip(launcher.CPS_VALUES, (4.0, 3.0, 2.0, 2.001)):
                rows.append({"target": target, "cps": cps, "full_voxel_mse": mse})
        self.assertTrue(launcher.monotonicity_audit(rows, 0.001)["passed"])
        rows[-1]["full_voxel_mse"] = 2.01
        self.assertFalse(launcher.monotonicity_audit(rows, 0.001)["passed"])


if __name__ == "__main__":
    unittest.main()
