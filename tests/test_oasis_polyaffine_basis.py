import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_polyaffine_fieldfit"))
from basis import Basis, FAMILIES, create_basis


class BasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(5)
        self.lo, self.hi = [-79., -112., -95.], [80., 111., 96.]
        self.points = torch.rand(173, 3, dtype=torch.float64) * torch.tensor([159., 223., 191.]) + torch.tensor(self.lo)
        self.matrix = torch.tensor([[.06, .01, -.02], [.03, -.1, .01], [.01, .07, .02]], dtype=torch.float64)
        self.translation = torch.tensor([31., -9., 4.], dtype=torch.float64)

    def make(self, family, budget=768):
        return create_basis(family, budget, self.lo, self.hi,
                            start_root=family.endswith("adaptive"))

    def test_affine_reproduction_and_serialization(self):
        corners = torch.tensor([[a, b, c] for a in [self.lo[0], self.hi[0]]
                                for b in [self.lo[1], self.hi[1]] for c in [self.lo[2], self.hi[2]]], dtype=torch.float64)
        points = torch.cat((self.points, corners))
        for family in FAMILIES:
            with self.subTest(family=family):
                basis = self.make(family)
                coeff = basis.coefficients_for_affine(self.matrix, self.translation)
                expected = points @ self.matrix.T + self.translation
                torch.testing.assert_close(basis.evaluate(points, coeff), expected, rtol=1e-11, atol=1e-11)
                loaded = Basis.from_dict(json.loads(json.dumps(basis.to_dict())))
                torch.testing.assert_close(loaded.features(points), basis.features(points), rtol=0, atol=0)
                self.assertLessEqual(basis.actual_motion_params, 768)

    def test_exact_tree_parameter_counts(self):
        for budget, ka, kt in [(768, 64, 253), (1536, 128, 509), (3072, 256, 1021)]:
            for family, leaves in [("tree_affine_uniform", ka), ("tree_translation_uniform", kt)]:
                basis = self.make(family, budget)
                self.assertEqual(basis.leaf_count, leaves)
                self.assertEqual(basis.actual_motion_params, budget)

    def test_all_tree_splits_preserve_arbitrary_field(self):
        for family in [x for x in FAMILIES if x.startswith("tree")]:
            basis = create_basis(family, 1536, self.lo, self.hi, start_root=True)
            coeff = torch.randn(4, 3, dtype=torch.float64)
            for _ in range(8):
                old_basis, old_coeff = basis, coeff
                before = basis.evaluate(self.points, coeff)
                residual = torch.ones_like(self.points)
                basis, coeff, event = basis.split(coeff, self.points, residual)
                torch.testing.assert_close(basis.features(self.points)[:, :old_basis.feature_count],
                                           old_basis.features(self.points), rtol=0, atol=0)
                torch.testing.assert_close(basis.evaluate(self.points, coeff), before, rtol=1e-14, atol=1e-14)
                self.assertTrue(event["split_count"] > 0)
                # Introduce nonzero contrast coefficients before the next split.
                coeff = coeff + .01 * torch.randn_like(coeff)
                self.assertEqual(old_coeff.shape[0], old_basis.feature_count)
            weights = basis.leaf_weights(self.points)
            torch.testing.assert_close(weights.sum(1), torch.ones(len(self.points), dtype=torch.float64), rtol=1e-14, atol=1e-14)
            self.assertTrue((weights >= 0).all())

    def test_adaptive_choice_and_tie_are_deterministic(self):
        basis = create_basis("tree_translation_adaptive", 768, self.lo, self.hi, start_root=True)
        coeff = torch.zeros(4, 3, dtype=torch.float64)
        basis, coeff, _ = basis.split(coeff, self.points, torch.zeros_like(self.points))
        # Longest initial axis is y; high-y error should select the right child.
        points = torch.tensor([[0., 108., 0.], [0., 100., 0.]], dtype=torch.float64)
        result, _, event = basis.split(coeff, points, torch.ones_like(points))
        self.assertEqual(event["split_leaf_ids"], [2])
        _, _, tie = basis.split(coeff, self.points, torch.zeros_like(self.points))
        self.assertEqual(tie["split_leaf_ids"], [1])
        self.assertEqual(result.leaf_count, 3)

    def test_physical_derivatives_and_operator_adjoint(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                basis = self.make(family)
                points = self.points[:6].clone().requires_grad_(True)
                coeff = basis.coefficients_for_affine(self.matrix, self.translation)
                values = basis.evaluate(points, coeff)
                jac = torch.stack([torch.autograd.grad(values[:, i].sum(), points, retain_graph=True)[0]
                                   for i in range(3)], dim=1)
                torch.testing.assert_close(jac, self.matrix[None].expand(len(points), -1, -1), rtol=1e-10, atol=1e-11)
                phi = basis.features(self.points)
                theta = torch.randn(basis.feature_count, 3, dtype=torch.float64)
                rhs = torch.randn(len(self.points), 3, dtype=torch.float64)
                left = ((phi @ theta) * rhs).sum()
                right = (theta * (phi.T @ rhs)).sum()
                torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-10)

    def test_nonlinear_features_derivative_matches_finite_difference(self):
        for family in ["bspline", "gaussian_affine", "tree_affine_uniform"]:
            basis = self.make(family)
            coeff = torch.randn(basis.feature_count, 3, dtype=torch.float64)
            points = self.points[:3].clone().requires_grad_(True)
            values = basis.evaluate(points, coeff)
            jac = torch.stack([torch.autograd.grad(values[:, i].sum(), points, retain_graph=True)[0]
                               for i in range(3)], dim=1)
            columns = []
            for axis in range(3):
                delta = torch.zeros_like(points); delta[:, axis] = 1e-4
                columns.append((basis.evaluate(points + delta, coeff) - basis.evaluate(points - delta, coeff)) / 2e-4)
            finite = torch.stack(columns, -1)
            torch.testing.assert_close(jac, finite, rtol=1e-6, atol=1e-8)

    def test_invalid_geometry_and_adaptive_build_fail_closed(self):
        with self.assertRaises(ValueError):
            create_basis("tree_affine_adaptive", 768, self.lo, self.hi)
        with self.assertRaises(ValueError):
            create_basis("bspline", 12, self.lo, self.hi)
        with self.assertRaises(ValueError):
            create_basis("global_affine", 12, self.lo, self.lo)

    def test_native_tensor_contractions_match_features_and_adjoint(self):
        shape = (7, 9, 8)
        affine = torch.eye(4, dtype=torch.float64)
        affine[:3, :3] = torch.diag(torch.tensor([(b - a) / (n - 1)
            for a, b, n in zip(self.lo, self.hi, shape)], dtype=torch.float64))
        affine[:3, 3] = torch.tensor(self.lo)
        indices = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij"), -1)
        points = (indices @ affine[:3, :3].T + affine[:3, 3]).reshape(-1, 3)
        for family in FAMILIES:
            with self.subTest(family=family):
                basis = self.make(family)
                coeff = torch.randn(basis.feature_count, 3, dtype=torch.float64)
                if basis.is_adaptive:
                    for _ in range(5):
                        basis, coeff, _ = basis.split(coeff, points, torch.randn_like(points))
                    coeff = torch.randn(basis.feature_count, 3, dtype=torch.float64)
                expected = basis.evaluate(points, coeff).reshape(*shape, 3)
                actual = basis.dense_forward(coeff, shape, affine)
                torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
                gradient = torch.randn_like(actual)
                expected_adjoint = basis.features(points).T @ gradient.reshape(-1, 3)
                actual_adjoint = basis.dense_adjoint(gradient, shape, affine)
                torch.testing.assert_close(actual_adjoint, expected_adjoint, rtol=2e-11, atol=2e-10)
                torch.testing.assert_close((actual * gradient).sum(), (coeff * actual_adjoint).sum(), rtol=1e-12, atol=1e-9)

    def test_native_tensor_float32_and_negative_axis_affine(self):
        shape = (13, 11, 9)
        affine = torch.eye(4)
        affine[:3, :3] = torch.diag(torch.tensor([-(b - a) / (n - 1)
            for a, b, n in zip(self.lo, self.hi, shape)]))
        affine[:3, 3] = torch.tensor(self.hi)
        indices = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij"), -1)
        points = (indices @ affine[:3, :3].T + affine[:3, 3]).reshape(-1, 3)
        for family in ["bspline", "gaussian_affine", "tree_translation_uniform", "tree_affine_uniform"]:
            basis = self.make(family)
            coeff = torch.randn(basis.feature_count, 3)
            torch.testing.assert_close(basis.dense_forward(coeff, shape, affine),
                basis.evaluate(points, coeff).reshape(*shape, 3), rtol=2e-5, atol=2e-5)

    def test_native_fallback_for_oblique_grid(self):
        basis = self.make("tree_affine_uniform")
        affine = torch.eye(4); affine[0, 1] = .1
        coeff = torch.zeros(basis.feature_count, 3)
        self.assertIsNone(basis.dense_forward(coeff, (8, 9, 10), affine))
        self.assertIsNone(basis.dense_adjoint(torch.zeros(8, 9, 10, 3), (8, 9, 10), affine))


if __name__ == "__main__":
    unittest.main()
