import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/reports"))
import oasis_matched_depth_diffusion_arc as m


class DiffusionArcTests(unittest.TestCase):
    def test_heterogeneous_rows_keep_all_columns(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            m.write_csv(path, [{"depth":"842", "old":1}, {"depth":"8421", "new":2}])
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["old"], "1")
            self.assertEqual(rows[1]["new"], "2")

    def test_zero_and_translation(self):
        mask = np.ones((4, 5, 6), bool)
        for value in (0., 3.):
            result = m.fd.diffusion(np.full((3, 4, 5, 6), value, np.float32), mask)
            self.assertEqual(result["diffusion_l2_fg"], 0.)
            self.assertEqual(result["diffusion_l2_full"], 0.)

    def test_affine_component_axis_reduction(self):
        z, y, x = np.indices((4, 5, 6), dtype=np.float32)
        field = np.stack((2*x, 3*y, 4*z))
        result = m.fd.diffusion(field, np.ones(z.shape, bool))
        self.assertAlmostEqual(result["diffusion_l2_fg"], (4+9+16)/9)
        self.assertAlmostEqual(result["diffusion_l2_full"], (4+9+16)/9)

    def test_both_endpoints_excludes_boundary_jump(self):
        mask = np.zeros((4, 5, 6), bool)
        mask[1:3, 1:4, 1:5] = True
        field = np.zeros((3, *mask.shape), np.float32)
        field[:, ~mask] = 10
        result = m.fd.diffusion(field, mask)
        self.assertEqual(result["diffusion_l2_fg"], 0.)
        self.assertGreater(result["diffusion_l2_full"], 0.)

    def test_normalized_xyz_scaling(self):
        q = np.ones((5, 7, 9, 3), np.float32)
        v = m.normalized_to_voxel(q)
        np.testing.assert_array_equal(v[:, 0, 0, 0], [4, 3, 2])
        self.assertEqual(v.shape, (3, 5, 7, 9))

    def test_component_permutation_energy_invariant(self):
        field = np.random.default_rng(1).normal(size=(3, 4, 5, 6)).astype(np.float32)
        mask = np.ones(field.shape[1:], bool)
        for key in ("diffusion_l2_fg", "diffusion_l2_full"):
            self.assertAlmostEqual(m.fd.diffusion(field, mask)[key], m.fd.diffusion(field[::-1], mask)[key], places=14)

    def test_parity_with_training_formula(self):
        import torch
        spec = importlib.util.spec_from_file_location("diffusion_common_test", ROOT / "scripts/experiments/oasis_alpha_sweep/common.py")
        common = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = common
        spec.loader.exec_module(common)
        field = np.random.default_rng(2).normal(size=(3, 4, 5, 6)).astype(np.float32)
        mask = np.ones(field.shape[1:], bool)
        mask[0] = False
        result = m.fd.diffusion(field, mask)
        for support, key in ((None, "diffusion_l2_full"), (mask, "diffusion_l2_fg")):
            expected = common.forward_valid_diffusion(torch.from_numpy(field)[None], support).item()
            self.assertAlmostEqual(result[key], expected, places=6)

    def test_invalid_fields_rejected(self):
        with self.assertRaises(ValueError):
            m.fd.diffusion(np.full((3, 4, 5, 6), np.nan), np.ones((4, 5, 6), bool))
        with self.assertRaises(ValueError):
            m.fd.diffusion(np.zeros((3, 4, 5, 6)), np.zeros((4, 5, 6), bool))

    def test_complete_groups_and_sample_sd(self):
        rows = [dict(depth="842", family="displacement", alpha=1., case_rank=i,
                     **{k:float(i) for k in m.MEASURES}) for i in range(100)]
        r = m.summarize(rows)[0]
        self.assertEqual(r["dice_mean"], 49.5)
        self.assertAlmostEqual(r["dice_sd"], np.std(np.arange(100), ddof=1))
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                m.summarize(bad)

    def test_figures_have_shared_axes_and_no_top_ellipses(self):
        means = []
        for family in m.arc.STYLES:
            for a in (m.arc.DL_LAMBDAS if family in ("SITReg", "VFA") else m.arc.OPT_LAMBDAS):
                means.append(dict(family=family, alpha=a, **{k+"_mean":.8 if k=="dice" else .4 for k in m.MEASURES},
                                  **{k+"_sd":.02 for k in m.MEASURES}))
        limits = [{"ylim": list(m.arc.TOP[k][:2])} for _ in range(2) for k in m.arc.Y_LABELS]
        diagnostics = []
        for depth in ("842", "8421"):
            fig, axes = m.build_figure(depth, means, limits, (0, 1))
            diagnostics.append(axes)
            self.assertEqual(len(fig.axes), 6)
            for i, ax in enumerate(fig.axes):
                self.assertEqual(len(ax.lines), 5)
                self.assertEqual(len(ax.patches), 0 if i<3 else 31)
            self.assertIn("VELOCITY", fig.texts[-1].get_text())
            m.arc.plt.close(fig)
        self.assertEqual(*diagnostics)


if __name__ == "__main__":
    unittest.main()
