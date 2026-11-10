import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_polyaffine_fieldfit"))
from basis import create_basis
from diagnostics import global_affine_residual_normalization, blending_derivative_diagnostics, inverse_consistency, optimize_sampled_d0


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_affine_reference_residual_is_undefined(self):
        shape = (8, 9, 10)
        points = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1).astype(float)
        reference = points @ np.diag([.02, -.1, .3]) + [21, -4, 7]
        target = dict(field=reference, mask=np.ones(shape, bool), affine=np.eye(4))
        result = global_affine_residual_normalization(reference, target)
        self.assertLess(result["reference_affine_residual_rms_mm"], 1e-12)
        self.assertIsNone(result["relative_rmse_pct"])
        json.dumps(result, allow_nan=False)

    def test_nonlinear_affine_residual_normalization_and_chunking(self):
        shape = (8, 9, 10)
        points = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1).astype(float)
        reference = .03 * points ** 2
        target = dict(field=reference, mask=np.ones(shape, bool), affine=np.eye(4))
        prediction = reference + .1
        result = global_affine_residual_normalization(prediction, target, chunk_depth=2)
        full = global_affine_residual_normalization(prediction, target, chunk_depth=8)
        self.assertAlmostEqual(result["prediction_endpoint_rmse_mm"], np.sqrt(.03))
        self.assertAlmostEqual(result["relative_rmse_pct"], full["relative_rmse_pct"], places=10)
        self.assertEqual(result["status"], "defined")

    def test_affine_reproduction_has_zero_blending_derivative(self):
        torch.manual_seed(4)
        points = torch.rand(37, 3, dtype=torch.float64) * 10
        for family in ("gaussian_translation", "gaussian_affine", "tree_affine_uniform", "tree_translation_uniform"):
            basis = create_basis(family, 768, [0, 0, 0], [10, 10, 10], min_box_size_mm=.1)
            coeff = basis.coefficients_for_affine(torch.eye(3, dtype=torch.float64) * .1,
                                                   torch.tensor([20., 7., -9.], dtype=torch.float64))
            row = blending_derivative_diagnostics(basis, coeff, points)
            self.assertLess(row["blending_term"]["frobenius"]["maximum"], 1e-12)
            self.assertAlmostEqual(row["total_derivative"]["induced_infinity"]["mean"], .1)
            json.dumps(row, allow_nan=False)

    def test_translation_contrasts_have_nonzero_blending(self):
        basis = create_basis("tree_translation_uniform", 24, [0, 0, 0], [10, 10, 10], min_box_size_mm=.1)
        coeff = torch.zeros(basis.feature_count, 3, dtype=torch.float64)
        coeff[4:, 0] = 1
        row = blending_derivative_diagnostics(basis, coeff, torch.tensor([[5., 5., 5.]], dtype=torch.float64))
        self.assertGreater(row["blending_term"]["frobenius"]["maximum"], 0)
        self.assertEqual(row["local_slope_term"]["frobenius"]["maximum"], 0)

    def test_inverse_residual_reports_boundary_coverage(self):
        forward = torch.zeros(8, 9, 10, 3, dtype=torch.float64); forward[..., 0] = 3
        inverse = -forward
        result = inverse_consistency(forward, inverse, torch.eye(4, dtype=torch.float64))
        for name in ("inverse_after_forward", "forward_after_inverse"):
            full = result[name]["full"]
            self.assertAlmostEqual(full["valid_intermediate_fraction"], 5 / 8)
            self.assertLess(full["valid_residual_mm"]["maximum"], 1e-12)
        json.dumps(result, allow_nan=False)

    def test_sampled_optimizer_reaches_affine_ls_and_respects_cap(self):
        torch.manual_seed(3)
        basis = create_basis("global_affine", 12, [0, 0, 0], [10, 10, 10])
        points = torch.rand(100, 3, dtype=torch.float64) * 10
        expected = torch.randn(4, 3, dtype=torch.float64)
        targets = basis.evaluate(points, expected)
        final, report = optimize_sampled_d0(basis, points, targets, torch.zeros_like(expected),
                                            max_steps=10, lbfgs_evals=30)
        self.assertLess(report["final_endpoint_mse_mm2"], 1e-10)
        self.assertLessEqual(report["lbfgs_objective_evaluations"], 30)
        self.assertLessEqual(report["final_endpoint_mse_mm2"], report["adam_endpoint_mse_mm2"])
        torch.testing.assert_close(final, expected, rtol=1e-5, atol=1e-5)
        json.dumps(report, allow_nan=False)

    def test_sampled_optimizer_conditioning_is_independent_of_fit_roi(self):
        basis = create_basis("bspline", 192, [0, 0, 0], [10, 10, 10])
        initial = torch.zeros(basis.feature_count, 3, dtype=torch.float64)
        config = {"conditioning_samples_per_axis": 8, "conditioning_min_scale": .02}
        reports = []
        for offset in (0., 7.):
            points = torch.tensor([[offset + .1, .1, .1], [offset + .2, .2, .2]], dtype=torch.float64)
            final, report = optimize_sampled_d0(basis, points, torch.zeros_like(points), initial,
                max_steps=0, lbfgs_evals=0, conditioning_config=config)
            torch.testing.assert_close(final, initial)
            self.assertEqual(report["conditioning"]["kind"], "full_declared_box_rms_floor")
            self.assertEqual(report["sample_count"], 2)
            reports.append(report["conditioning"])
        self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
