import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_first100_five_family_dual_arc as arc


class ArcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = [(f"{r:04d}", f"{r+100:04d}") for r in range(100)]
        cls.rows = [{"family": family, "alpha": alpha, "case_rank": rank,
                     "fixed_id": pair[0], "moving_id": pair[1], "direction": "moving_to_fixed",
                     "dice": .8+rank*.0001, "hd95": 2+rank*.001, "asd": .4+rank*.0001,
                     "log_jdet_std_fg": .6+rank*.001, "jdet_le0_pct_fg": rank*.01}
                    for family in arc.STYLES for alpha in
                    (arc.DL_LAMBDAS if family in ("SITReg", "VFA") else arc.OPT_LAMBDAS)
                    for rank, pair in enumerate(cls.pairs)]

    def test_matrix_and_sample_sd(self):
        means = arc.summarize(self.rows, self.pairs)
        self.assertEqual((len(self.rows), len(means)), (3100, 31))
        self.assertAlmostEqual(means[0]["dice_sd"], np.std([.8+r*.0001 for r in range(100)], ddof=1))
        self.assertAlmostEqual(means[0]["folding_ratio_fg_mean"], means[0]["jdet_le0_pct_fg_mean"]/100)
        self.assertAlmostEqual(means[0]["folding_ratio_fg_sd"], means[0]["jdet_le0_pct_fg_sd"]/100)

    def test_reject_incomplete_duplicate_and_wrong_pair(self):
        for rows in (self.rows[:-1], self.rows+[self.rows[0]]):
            with self.assertRaises(ValueError):
                arc.summarize(rows, self.pairs)
        rows = copy.deepcopy(self.rows)
        rows[0]["fixed_id"] = "9999"
        with self.assertRaises(ValueError):
            arc.summarize(rows, self.pairs)

    def test_reject_nonfinite_and_invalid_ratio(self):
        for key, value in (("dice", float("nan")), ("jdet_le0_pct_fg", -1), ("jdet_le0_pct_fg", 101)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                arc.summarize(rows, self.pairs)

    def test_floor_does_not_change_data(self):
        means = [{"folding_ratio_fg_mean": v} for v in (0, 5.8e-8, .1)]
        before = copy.deepcopy(means)
        self.assertEqual(arc.folding_floor(means), 1e-9)
        self.assertEqual(means, before)
        with self.assertRaises(ValueError):
            arc.folding_floor([means[0]])

    def test_contour_clipping_and_unclipped_extents(self):
        p = arc.sd_contour(1e-5, .8, 2e-5, .02, 1e-8)
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(p[:, 0].min(), 1e-8)
        self.assertLessEqual(p[:, 0].max(), 3.000000001e-5)
        p = arc.sd_contour(1, 2, .1, .2, 1e-8)
        np.testing.assert_allclose(p.min(axis=0), [.9, 1.8], atol=1e-12)
        np.testing.assert_allclose(p.max(axis=0), [1.1, 2.2], atol=1e-12)

    def test_figures_share_y_limits_and_complete_lines(self):
        means = arc.summarize(self.rows, self.pairs)
        means[0]["folding_ratio_fg_mean"] = 0
        means[0]["folding_ratio_fg_sd"] = 0
        linear, a = arc.figure(means)
        log, b = arc.figure(means, True)
        self.assertEqual(len(a), 6)
        self.assertEqual([r["ylim"] for r in a], [r["ylim"] for r in b])
        self.assertEqual(len({tuple(r["xlim"]) for r in a}), 1)
        self.assertEqual(len({tuple(r["xlim"]) for r in b}), 1)
        self.assertTrue(all(r["scale"] == "log" for r in b))
        for axis in linear.axes:
            self.assertEqual(len(axis.lines), 5)
            self.assertEqual(sum(len(line.get_xdata()) for line in axis.lines), 31)
        self.assertEqual(len(linear.axes[0].patches), 0)
        self.assertEqual(len(linear.axes[3].patches), 31)
        arc.plt.close(linear)
        arc.plt.close(log)


if __name__ == "__main__":
    unittest.main()
