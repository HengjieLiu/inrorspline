import copy
import math
from pathlib import Path
import statistics
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_first10_family_arc as a


def fixtures():
    rows = []
    for panel, series, alpha in sorted(a.expected_groups()):
        for rank in range(10):
            rows.append(dict(panel=panel, series=series, alpha=alpha, case_rank=rank,
                             dice=.8 + rank * .001, hd95=1.8 + rank * .01,
                             asd=.35 + rank * .002, log_jdet_std_fg=.5 + rank * .004,
                             jdet_le0_pct_fg=rank * .001))
    return rows


class CohortTests(unittest.TestCase):
    def test_exact_matrix(self):
        rows = fixtures()
        self.assertEqual(len(rows), 700)
        summary = a.summarize(rows)
        self.assertEqual(len(summary), 70)
        self.assertEqual(dict(a.Counter(r["panel"] for r in summary)), dict(A=15, B=21, C=3, D=3, E=28))

    def test_sample_sd_and_mean(self):
        result = a.summarize(fixtures())[0]
        vals = [.8 + rank * .001 for rank in range(10)]
        self.assertEqual(result["dice_mean"], statistics.mean(vals))
        self.assertEqual(result["dice_sd"], statistics.stdev(vals))
        self.assertNotEqual(result["dice_sd"], statistics.pstdev(vals))

    def test_missing_duplicate_wrong_weight_and_nonfinite_rejected(self):
        rows = fixtures()
        with self.assertRaises(ValueError): a.summarize(rows[:-1])
        duplicate = copy.deepcopy(rows); duplicate[-1]["case_rank"] = 0
        with self.assertRaises(ValueError): a.summarize(duplicate)
        wrong = copy.deepcopy(rows); wrong[0]["alpha"] = .2
        with self.assertRaises(ValueError): a.summarize(wrong)
        bad = copy.deepcopy(rows); bad[0]["asd"] = math.nan
        with self.assertRaises(ValueError): a.summarize(bad)

    def test_pyramid_only(self):
        row = dict(resolution="pyramid", representation="8421", alpha="1")
        self.assertEqual(a.classify("direct", row), ("B", "8421", 1.))
        row["resolution"] = "fullres"
        self.assertIsNone(a.classify("direct", row))

    def test_matched_composition_only(self):
        self.assertEqual(a.classify("compose", dict(depth="4", config_key="unbounded_1")), ("D", "unbounded_1", 1.))
        for depth, config in [(2, "unbounded_1"), (4, "unbounded_2_compose"), (4, "bounded_3_compose")]:
            self.assertIsNone(a.classify("compose", dict(depth=depth, config_key=config)))

    def test_dl_uniform_series_and_exact_third(self):
        self.assertEqual(a.classify("dl", {"method": "VFA (original)"}), ("A", "vfa", 1.))
        self.assertEqual(a.classify("dl", {"method": "VFA λ=3"}), ("A", "vfa", 3.))
        self.assertEqual(a.classify("dl", {"method": "FeDDA-1MF lambda0.333333"}), ("A", "fedda", 1/3))
        self.assertFalse(any("histor" in a.STYLES[k][0] or "original" in a.STYLES[k][0] for k in a.FIGURES["A"][1]))

    def test_svf_four_arms(self):
        for schedule in ("842", "8421"):
            for penalty in ("velocity", "displacement"):
                row = dict(source_kind="new_svf", schedule=schedule, regularize=penalty, alpha=".3")
                self.assertEqual(a.classify("svf", row), ("E", schedule+"_"+penalty, .3))

    def test_sd_extent_guard(self):
        rows = a.summarize(fixtures())
        a.check_extents(rows)
        rows[0]["hd95_sd"] = 8.
        with self.assertRaises(ValueError): a.check_extents(rows)

    def test_canonical_pair_file(self):
        pairs = a.case_pairs(a.REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt")
        self.assertEqual(pairs[0], ("0333", "0115"))
        self.assertEqual(pairs[4], ("0084", "0108"))
        self.assertEqual(pairs[-1], ("0173", "0136"))


class PlotTests(unittest.TestCase):
    def test_all_figures_axes_rows_ellipses_and_fonts(self):
        summary = a.summarize(fixtures())
        for panel in a.FIGURES:
            with self.subTest(panel=panel):
                fig, metadata = a.make_figure(panel, summary)
                try:
                    self.assertEqual(len(fig.axes), 6)
                    self.assertEqual(len(metadata), 6)
                    for i, ax in enumerate(fig.axes):
                        metric = ("dice", "hd95", "asd")[i % 3]
                        self.assertEqual(tuple(ax.get_xlim()), a.AXES["x"]["limits"])
                        self.assertEqual(tuple(ax.get_ylim()), a.AXES[metric]["limits"])
                        self.assertEqual(tuple(ax.get_xticks()), a.AXES["x"]["ticks"])
                        self.assertEqual(tuple(ax.get_yticks()), a.AXES[metric]["ticks"])
                        self.assertEqual(ax.xaxis.label.get_fontsize(), 19)
                        self.assertTrue(all(t.get_fontsize() == 16 for t in ax.get_xticklabels()))
                        for line in ax.lines:
                            self.assertEqual(len(line.get_xdata()), 1 if panel in "CD" else 5 if panel == "A" else 7)
                            if panel in "CD": self.assertEqual(line.get_linestyle(), "None")
                        if i < 3:
                            self.assertFalse(ax.patches)
                        else:
                            expected = [r for s in a.FIGURES[panel][1] for r in sorted(summary, key=lambda r: r["alpha"])
                                        if r["panel"] == panel and r["series"] == s]
                            self.assertEqual(len(ax.patches), len(expected))
                            for ellipse, r in zip(ax.patches, expected):
                                self.assertEqual(ellipse.width, 2*r["log_jdet_std_fg_sd"])
                                self.assertEqual(ellipse.height, 2*r[metric+"_sd"])
                                self.assertEqual(ellipse.angle, 0)
                finally:
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main()
