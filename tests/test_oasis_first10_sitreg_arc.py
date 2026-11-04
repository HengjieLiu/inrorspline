import copy
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_first10_family_arc as base
import oasis_first10_combined_arc as old
import oasis_first10_sitreg_arc as arc
from test_oasis_first10_combined_arc import fixtures


def source():
    rows, pairs = fixtures()
    return old.select(rows), pairs


class SitregRevisionTests(unittest.TestCase):
    def test_exact_subset_and_source_mapping(self):
        rows, pairs = source()
        saved = copy.deepcopy(rows)
        selected = arc.select(rows)
        self.assertEqual(rows, saved)
        self.assertEqual(len(selected), 400)
        self.assertEqual(len(arc.summarize(selected, pairs)), 40)
        self.assertEqual({r["series"] for r in selected}, set(arc.SERIES))
        self.assertEqual({r["source_series"] for r in selected if r["series"] == "svf"}, {"842_velocity", "8421_velocity"})
        self.assertEqual({r["source_series"] for r in selected if r["series"] == "displacement"}, {"842", "8421"})
        for row in selected:
            original = next(r for r in rows if all(r[k] == row[k] for k in ("panel", "series", "alpha", "case_rank")))
            self.assertTrue(all(original[m] == row[m] for m in base.METRICS))

    def test_validation(self):
        rows, pairs = source(); rows = arc.select(rows)
        with self.assertRaises(ValueError): arc.summarize(rows[:-1], pairs)
        for key, value in (("dice", float("nan")), ("case_rank", 99),
                           ("direction", "fixed_to_moving"), ("alpha", 2.),
                           ("source_series", "8421_displacement")):
            bad = copy.deepcopy(rows); bad[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                arc.summarize(bad, pairs)
        bad = copy.deepcopy(rows); bad[-1] = copy.deepcopy(bad[-2])
        with self.assertRaises(ValueError): arc.summarize(bad, pairs)

    def test_styles(self):
        self.assertEqual(len({s[1] for s in arc.STYLES.values()}), 4)
        self.assertEqual(len({s[2] for s in arc.STYLES.values()}), 4)
        self.assertEqual(len({s[3] for s in arc.STYLES.values()}), 4)
        self.assertEqual(arc.STYLES["sitreg"], base.STYLES["sitreg"])
        self.assertEqual(arc.STYLES["bounded2"][3], "none")

    def test_tighter_top_axes_not_cropped_means(self):
        rows, pairs = source(); summary = arc.summarize(arc.select(rows), pairs)
        for metric in arc.TOP_AXES:
            self.assertGreater(arc.TOP_AXES[metric]["limits"][0], base.AXES[metric]["limits"][0])
            self.assertLess(arc.TOP_AXES[metric]["limits"][1], base.AXES[metric]["limits"][1])
        for panel in arc.PANELS:
            fig, diagnostics = arc.figure(panel, summary)
            try:
                self.assertEqual(len(fig.axes), 6)
                self.assertEqual([r["ellipse_count"] for r in diagnostics], [0, 0, 0, 20, 20, 20])
                self.assertTrue(all(r["point_count"] == 20 for r in diagnostics))
                for i, ax in enumerate(fig.axes):
                    metric = ("dice", "hd95", "asd")[i % 3]
                    spec = arc.TOP_AXES[metric] if i < 3 else base.AXES[metric]
                    self.assertEqual(tuple(ax.get_ylim()), spec["limits"])
                    self.assertEqual(tuple(ax.get_yticks()), spec["ticks"])
                    self.assertEqual(tuple(ax.get_xlim()), base.AXES["x"]["limits"])
                    self.assertEqual([len(line.get_xdata()) for line in ax.lines], [5, 7, 1, 7])
                    self.assertEqual(ax.lines[2].get_linestyle(), "None")
                    self.assertGreater(ax.lines[2].get_markersize(), ax.lines[0].get_markersize())
                for a, b in zip(fig.axes[:3], fig.axes[3:]):
                    for al, bl in zip(a.lines, b.lines):
                        self.assertEqual(list(al.get_xdata()), list(bl.get_xdata()))
                        self.assertEqual(list(al.get_ydata()), list(bl.get_ydata()))
                labels = [t.get_text() for t in fig.legends[0].get_texts()]
                self.assertEqual(labels, [arc.STYLES[s][0] for s in arc.SERIES])
            finally:
                plt.close(fig)
        bad = copy.deepcopy(summary); bad[0]["dice_mean"] = .5
        with self.assertRaises(ValueError): arc.check_top_extents(bad)


if __name__ == "__main__":
    unittest.main()
