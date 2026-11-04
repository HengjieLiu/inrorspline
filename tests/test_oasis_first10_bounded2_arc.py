import copy
from pathlib import Path
import sys
import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_first10_bounded2_sweep_arc as arc
import oasis_first10_combined_arc as combined
import oasis_first10_sitreg_arc as old
from test_oasis_first10_combined_arc import fixtures


def samples():
    original, pairs = fixtures()
    fg = old.select(combined.select(original))
    sweep = [{**r, "alpha": a} for r in original if r["panel"] == "C" and r["series"] == "bounded_2_compose" for a in arc.base.LAMBDAS]
    return arc.select(original, fg, sweep), pairs


class BoundedArcTests(unittest.TestCase):
    def test_scope_and_no_four_level_substitution(self):
        rows, pairs = samples()
        summaries = arc.summarize(rows, pairs)
        self.assertEqual(len(rows), 550); self.assertEqual(len(summaries), 55)
        self.assertEqual([sum(r["panel"] == p for r in summaries) for p in "CFG"], [9, 26, 20])
        self.assertEqual({r["alpha"] for r in rows if r["panel"] == "G" and r["series"] == "bounded2"}, {1.})
        self.assertEqual({r["plot_source"] for r in rows if r["panel"] == "G" and r["series"] == "bounded2"}, {"P75_FG_sitreg_v2"})
        self.assertFalse(any(r["series"] in ("vfa", "fedda") for r in rows))
        with self.assertRaises(ValueError): arc.summarize(rows[:-1], pairs)
        bad = copy.deepcopy(rows); bad[-1]["case_rank"] = 0
        with self.assertRaises(ValueError): arc.summarize(bad, pairs)

    def test_axes_extents_and_styles(self):
        rows, pairs = samples(); summary = arc.summarize(rows, pairs)
        axes, top, changes = arc.derive_axes(summary)
        self.assertFalse(changes)
        self.assertEqual(axes, arc.base.AXES); self.assertEqual(top, old.TOP_AXES)
        self.assertEqual(arc.panel_styles("F")["bounded2"][3], ":")
        self.assertEqual(arc.panel_styles("G")["bounded2"][3], "none")
        bad = copy.deepcopy(summary); next(r for r in bad if r["panel"] == "F")["dice_mean"] = .9
        enlarged, tight, changes = arc.derive_axes(bad)
        self.assertGreater(tight["dice"]["limits"][1], .9)
        self.assertTrue(changes)

    def test_three_figures(self):
        rows, pairs = samples(); summary = arc.summarize(rows, pairs)
        axes, top, _ = arc.derive_axes(summary)
        for panel, count in (("C", 9), ("F", 26), ("G", 20)):
            fig, diagnostics = arc.figure(panel, summary, axes, top)
            try:
                self.assertTrue(all(a["point_count"] == count for a in diagnostics))
                self.assertEqual([a["ellipse_count"] for a in diagnostics], [0, 0, 0, count, count, count])
                for i, ax in enumerate(fig.axes):
                    metric = ("dice", "hd95", "asd")[i % 3]
                    spec = top[metric] if i < 3 and panel != "C" else axes[metric]
                    self.assertEqual(tuple(ax.get_ylim()), tuple(spec["limits"]))
            finally:
                plt.close(fig)


if __name__ == "__main__": unittest.main()
