"""Exact family-independent global-affine SVF initialization checks."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_polyaffine_fieldfit"))
from basis import create_basis
from initialization import fit_global_affine_initialization


class InitializationTests(unittest.TestCase):
    def test_affine_fit_embeds_identically_in_all_families(self):
        torch.set_num_threads(2); torch.manual_seed(13)
        lo, hi = [-4., -5., -6.], [6., 7., 8.]
        points = torch.rand(257, 3, dtype=torch.float64) * torch.tensor([10., 12., 14.]) + torch.tensor(lo)
        matrix = torch.randn(3, 3, dtype=torch.float64)*.1
        translation = torch.tensor([2., -1., 3.], dtype=torch.float64)
        values = points @ matrix.T + translation
        for family in ("bspline", "gaussian_translation", "gaussian_affine",
                       "tree_translation_uniform", "tree_translation_adaptive",
                       "tree_affine_uniform", "tree_affine_adaptive"):
            with self.subTest(family=family):
                basis = create_basis(family, 192, lo, hi, start_root=family.endswith('_adaptive'))
                coefficients, meta = fit_global_affine_initialization(
                    basis, points, values, {"atol":1e-12, "btol":1e-12})
                torch.testing.assert_close(basis.evaluate(points, coefficients), values, atol=1e-9, rtol=1e-9)
                self.assertTrue(meta['solver']['converged'])
                self.assertFalse(meta['affine_transform_logarithm'])
                self.assertTrue(meta['changes_initialization_only'])


if __name__ == '__main__': unittest.main()
