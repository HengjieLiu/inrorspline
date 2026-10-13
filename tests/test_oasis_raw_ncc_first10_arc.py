import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/reports"))
import oasis_raw_ncc_first10_arc as arc


def synthetic_rows():
    return [{"group": group, "case_rank": rank, "alpha": 30,
             **{key: float(rank) for key in arc.METRICS}}
            for group in arc.GROUPS for rank in range(10)]


class ArcTests(unittest.TestCase):
    def test_sample_sd_and_cohort_means(self):
        rows = arc.summarize(synthetic_rows())
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["n"], 10)
        self.assertEqual(rows[0]["dice_mean"], 4.5)
        self.assertAlmostEqual(rows[0]["dice_std"], np.std(np.arange(10), ddof=1))
        self.assertEqual(rows[6]["n"], 9)
        self.assertEqual(rows[6]["dice_mean"], 5.)

    def test_missing_duplicate_and_nonfinite_rejected(self):
        rows = synthetic_rows()
        with self.assertRaises(RuntimeError):
            arc.summarize(rows[:-1])
        with self.assertRaises(RuntimeError):
            arc.summarize(rows + [rows[0]])
        rows[0]["asd"] = float("nan")
        with self.assertRaises(ValueError):
            arc.summarize(rows)

    def test_axis_aligned_two_sd_ellipse(self):
        fig, ax = arc.plt.subplots()
        row = {"x_mean": .6, "y_mean": .8, "x_std": .1, "y_std": .02}
        ellipse = arc.add_sd_ellipse(ax, row, "x", "y", "#0072B2")
        self.assertEqual(ellipse.center, (.6, .8))
        self.assertEqual(ellipse.width, .2)
        self.assertEqual(ellipse.height, .04)
        self.assertEqual(ellipse.angle, 0)
        arc.plt.close(fig)

    def test_plot_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            means = arc.summarize(synthetic_rows())
            arc.make_plot(means, Path(directory))
            for suffix in ("png", "svg", "pdf"):
                self.assertGreater((Path(directory) / f"arc_sdlogj_first10.{suffix}").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
