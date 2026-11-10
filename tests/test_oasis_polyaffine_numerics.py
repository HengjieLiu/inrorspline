"""Scientific gates for P81 physical field fitting (small CPU fixtures)."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_polyaffine_fieldfit"))
import numerics as n
from basis import create_basis


class AffineBasis:
    def features(self, points):
        return torch.cat((torch.ones_like(points[:, :1]), points), -1)


class NearlyUnsupportedBasis:
    domain_min = [0., 0., 0.]
    domain_max = [1., 1., 1.]

    def features(self, points):
        return torch.stack((torch.ones_like(points[:, 0]), points[:, 0]**8), -1)


class NumericsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.basis = AffineBasis()
        self.h = torch.eye(4, dtype=torch.float64)
        self.shape = (7, 8, 9)

    def test_world_grid_axes_and_anisotropic_affine(self):
        h = self.h.clone()
        h[:3, :3] = torch.tensor([[0, 0, -2.], [0, 3., 0], [4., 0, 0]], dtype=torch.float64)
        h[:3, 3] = torch.tensor([7., -2., 9.])
        points = n.world_points(self.shape, h)
        grid = n.world_to_grid(points, self.shape, h).reshape(*self.shape, 3)
        torch.testing.assert_close(grid, n.normalized_identity(self.shape, device="cpu", dtype=torch.float64))
        field = points.reshape(*self.shape, 3)
        selected = torch.tensor([0, 12, 155, math.prod(self.shape)-1])
        sampled = n.sample_field_points(field, points[selected], h)
        torch.testing.assert_close(sampled, points[selected], atol=1e-12, rtol=1e-12)

    def test_dense_render_coefficient_gradient_matches_explicit_design(self):
        coeff = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        field = n.dense_basis_field(self.basis, coeff, self.shape, self.h, chunk=27)
        gradient = torch.randn_like(field)
        actual, = torch.autograd.grad((field*gradient).sum(), coeff)
        points = n.world_points(self.shape, self.h)
        expected = self.basis.features(points).T @ gradient.reshape(-1, 3)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

    def test_real_separable_basis_dispatch_and_gradient(self):
        for family in ("bspline", "gaussian_translation", "gaussian_affine",
                       "tree_translation_uniform", "tree_affine_uniform"):
            with self.subTest(family=family):
                basis = create_basis(family, 192, [0.,0.,0.], [6.,7.,8.])
                coeff = (torch.randn(basis.feature_count,3,dtype=torch.float64)*.1).requires_grad_(True)
                field = n.dense_basis_field(basis,coeff,self.shape,self.h,chunk=27)
                points = n.world_points(self.shape,self.h)
                explicit = basis.features(points)
                torch.testing.assert_close(field.reshape(-1,3),explicit@coeff,atol=1e-10,rtol=1e-10)
                gradient = torch.randn_like(field)
                actual, = torch.autograd.grad((field*gradient).sum(),coeff)
                torch.testing.assert_close(actual,explicit.T@gradient.reshape(-1,3),atol=1e-9,rtol=1e-9)

    def test_linear_operator_adjoint_and_lsqr_reference(self):
        points = torch.randn(123, 3, dtype=torch.float64)
        operator = n.SampledLinearOperator(self.basis, points, chunk=17, cache_bytes=0)
        x = torch.randn(4, dtype=torch.float64)
        y = torch.randn(123, dtype=torch.float64)
        torch.testing.assert_close(torch.dot(operator.forward(x), y), torch.dot(x, operator.adjoint(y)))
        truth = torch.randn(4, 3, dtype=torch.float64)
        target = self.basis.features(points) @ truth
        actual, info = n.solve_d0(self.basis, points, target,
                                   config={"atol": 1e-12, "btol": 1e-12, "chunk": 17, "cache_bytes": 0})
        self.assertTrue(info["converged"])
        torch.testing.assert_close(actual, truth, atol=1e-10, rtol=1e-10)
        self.assertFalse(info["cached_sample_design"])

    def test_full_box_scaling_is_foreground_independent(self):
        basis = NearlyUnsupportedBasis()
        points = torch.rand(97, 3, dtype=torch.float64)
        points[:, 0] *= .1
        scales, meta = n.conditioning_scales(basis, points)
        alternative = points.clone(); alternative[:, 0] += .8
        other, other_meta = n.conditioning_scales(basis, alternative)
        torch.testing.assert_close(scales, other, atol=0, rtol=0)
        self.assertEqual(meta["scale_sha256"], other_meta["scale_sha256"])
        old = basis.features(points).square().mean(0).sqrt().clamp_min(1e-6)
        self.assertGreater(float(scales[1]/old[1]), 100000)
        self.assertEqual(meta["sample_count"], 32**3)
        self.assertGreaterEqual(float(scales.min()), .01)
        self.assertEqual(meta["domain_source"], "declared_basis_box")
        with self.assertRaises(ValueError):
            n.conditioning_scales(basis, points, {"conditioning_min_scale": 1e-6})

    def test_nearly_unsupported_svf_column_no_longer_explodes(self):
        basis = NearlyUnsupportedBasis()
        points = torch.rand(32, 3, dtype=torch.float64)
        points[:, 0] *= .1
        h = self.h.clone(); h[0, 0] = 1/16; h[1, 1] = .5; h[2, 2] = .5
        initial = torch.zeros(2, 3, dtype=torch.float64)
        target = torch.zeros_like(points); target[:, 0] = .05
        coeff, report = n.optimize_nonlinear(basis, initial, (17,3,3), h,
            fit_targets_mm=target, fit_points_mm=points, mode="svf",
            config={"max_steps": 3, "min_steps": 0, "steps": 4, "lr": .03, "check_interval": 1})
        self.assertLess(float(coeff.abs().max()), 1.)
        field = n.render_field(basis, coeff, (17,3,3), h, "svf", steps=4)
        self.assertTrue(torch.isfinite(field).all())
        self.assertLess(float(field.abs().max()), 1.)
        self.assertLess(report["best_training_loss_mm2"], .05**2)
        self.assertEqual(report["conditioning"]["kind"], "full_declared_box_rms_floor")

    def test_lsqr_noisy_least_squares_matches_svd(self):
        points = torch.randn(51, 3, dtype=torch.float64)
        target = torch.randn(51, 3, dtype=torch.float64)
        actual, info = n.solve_d0(self.basis, points, target, config={"atol": 1e-12, "btol": 1e-12})
        reference = torch.linalg.lstsq(self.basis.features(points), target).solution
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
        self.assertTrue(info["converged"])

    def test_identity_large_translation_and_bounds(self):
        zero = torch.zeros((*self.shape, 3), dtype=torch.float64)
        self.assertEqual(float(n.grid_lipschitz_bound(zero, self.h)), 0.)
        translation = zero + torch.tensor([500., -100., 20.], dtype=torch.float64)
        torch.testing.assert_close(n.exp_velocity(translation, self.h), translation)
        bounded, meta = n.bounded_displacement(translation, self.h, return_metadata=True)
        torch.testing.assert_close(bounded, translation)
        self.assertEqual(meta["alpha"], 1.)

    def test_bound_correct_physical_jacobian_and_domain(self):
        h = self.h.clone()
        h[:3, :3] = torch.tensor([[0., 0., -2.], [0., 3., 0.], [4., 0., 0.]], dtype=torch.float64)
        a = torch.tensor([[.1, -.2, .3], [.01, .02, .04], [-.3, .2, .1]], dtype=torch.float64)
        field = (n.world_points(self.shape, h) @ a.T).reshape(*self.shape, 3)
        bound, meta = n.grid_lipschitz_bound(field, h, return_metadata=True, chunk_slices=2)
        self.assertAlmostEqual(bound, float(a.abs().sum(-1).max()), places=12)
        self.assertEqual(meta["cell_count"], 6*7*8)
        bounded = n.bounded_displacement(field*4, h, rho=.9)
        final_bound, _ = n.grid_lipschitz_bound(bounded, h, return_metadata=True)
        self.assertLessEqual(final_bound, .9)

    def test_bound_contains_interior_derivatives(self):
        field = torch.randn(3, 4, 5, 3, dtype=torch.float64)*.1
        bound = float(n.grid_lipschitz_bound(field, self.h))
        points = torch.rand(40, 3, dtype=torch.float64)*torch.tensor([1.8, 2.8, 3.8])+.1
        # Away from cell boundaries for finite-difference consistency.
        eps = 1e-5
        columns = []
        for axis in range(3):
            shift = torch.zeros(3, dtype=torch.float64); shift[axis] = eps
            columns.append((n.sample_field_points(field, points+shift, self.h)
                            - n.sample_field_points(field, points-shift, self.h))/(2*eps))
        jacobian = torch.stack(columns, -1)
        self.assertLessEqual(float(jacobian.abs().sum(-1).max()), bound+1e-8)

    def test_grid_bound_gradient(self):
        field = (torch.randn(3, 4, 5, 3, dtype=torch.float64)*.07).requires_grad_(True)
        bound = n.grid_lipschitz_bound(field, self.h)
        derivative, = torch.autograd.grad(bound, field)
        perturb = torch.randn_like(field)
        eps = 1e-6
        numeric = (n.grid_lipschitz_bound(field.detach()+eps*perturb, self.h)
                   - n.grid_lipschitz_bound(field.detach()-eps*perturb, self.h))/(2*eps)
        torch.testing.assert_close(numeric, (derivative*perturb).sum(), atol=1e-7, rtol=1e-7)

    def test_bounded_renderer_gradient_with_active_constraint(self):
        field = (torch.randn(3, 4, 5, 3, dtype=torch.float64)*2).requires_grad_(True)
        weight = torch.randn_like(field)
        def objective(values):
            return (n.bounded_displacement(values, self.h)*weight).sum()
        gradient, = torch.autograd.grad(objective(field), field)
        perturb = torch.randn_like(field)
        eps = 1e-6
        numeric = (objective(field.detach()+eps*perturb)-objective(field.detach()-eps*perturb))/(2*eps)
        torch.testing.assert_close(numeric, (gradient*perturb).sum(), atol=1e-7, rtol=1e-7)

    def test_composition_noncommuting_affines(self):
        points = n.world_points(self.shape, self.h).reshape(*self.shape, 3)
        outer = torch.zeros_like(points); outer[..., 0] = .1*points[..., 1]
        inner = torch.zeros_like(points); inner[..., 1] = .1*points[..., 0]
        ab = n.compose_displacements(outer, inner, self.h)
        ba = n.compose_displacements(inner, outer, self.h)
        p = points[1:-2, 1:-2, 1:-2]
        expected = torch.stack((.1*p[..., 1]+.01*p[..., 0], .1*p[..., 0], torch.zeros_like(p[..., 0])), -1)
        torch.testing.assert_close(ab[1:-2, 1:-2, 1:-2], expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(float((ab-ba).abs().max()), .01)

    def test_affine_velocity_matches_homogeneous_exponential_interior(self):
        shape = (17, 18, 19)
        h = self.h.clone(); h[:3, 3] = torch.tensor([-8., -8.5, -9.])
        points = n.world_points(shape, h)
        a = torch.tensor([[.01, -.007, 0.], [.007, .005, 0.], [0., 0., -.006]], dtype=torch.float64)
        t = torch.tensor([.02, -.01, .03], dtype=torch.float64)
        velocity = (points @ a.T+t).reshape(*shape, 3)
        field = n.exp_velocity(velocity, h, steps=9)
        homogeneous = np.zeros((4, 4)); homogeneous[:3, :3] = a.numpy(); homogeneous[:3, 3] = t.numpy()
        transform = torch.from_numpy(expm(homogeneous))
        reference = (points @ transform[:3, :3].T+transform[:3, 3]-points).reshape(*shape, 3)
        torch.testing.assert_close(field[3:-3, 3:-3, 3:-3], reference[3:-3, 3:-3, 3:-3], atol=3e-6, rtol=1e-4)

    def test_svf_coefficient_gradient_and_nonlinear_smoke(self):
        shape = (4, 5, 6)
        coefficients = (torch.randn(4, 3, dtype=torch.float64)*.001).requires_grad_(True)
        field = n.render_field(self.basis, coefficients, shape, self.h, mode="svf", steps=3, chunk=19)
        weight = torch.randn_like(field)
        gradient, = torch.autograd.grad((field*weight).sum(), coefficients)
        delta = torch.randn_like(coefficients)
        eps = 1e-6
        def objective(c):
            return (n.render_field(self.basis, c, shape, self.h, mode="svf", steps=3, chunk=19)*weight).sum()
        numeric = (objective(coefficients.detach()+eps*delta)-objective(coefficients.detach()-eps*delta))/(2*eps)
        torch.testing.assert_close(numeric, (gradient*delta).sum(), atol=1e-6, rtol=1e-5)
        indices = torch.arange(math.prod(shape))
        target = torch.zeros(len(indices), 3, dtype=torch.float64)+.05
        result, meta = n.optimize_nonlinear(self.basis, torch.zeros(4, 3, dtype=torch.float64),
            shape, self.h, indices, target, mode="svf",
            config={"max_steps": 10, "min_steps": 0, "check_interval": 2, "steps": 3, "chunk": 29, "lr": .01})
        self.assertLess(meta["best_training_loss_mm2"], .003)
        self.assertTrue(torch.isfinite(result).all())

    def test_coarse_grid_physical_sample_loss(self):
        shape = (3, 4, 5)
        h = self.h.clone(); h[:3, :3] *= 2
        points = torch.rand(17, 3, dtype=torch.float64)*torch.tensor([4., 6., 8.])
        target = torch.zeros(17, 3, dtype=torch.float64)+.03
        result, meta = n.optimize_nonlinear(self.basis, torch.zeros(4, 3, dtype=torch.float64),
            shape, h, fit_targets_mm=target, fit_points_mm=points, mode="svf",
            config={"max_steps": 1, "min_steps": 0, "steps": 2, "lr": .005})
        self.assertEqual(meta["sampling"], "physical_points")
        self.assertEqual(meta["actual_steps"], 1)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
