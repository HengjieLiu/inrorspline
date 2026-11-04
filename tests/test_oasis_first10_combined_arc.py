import copy
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_first10_family_arc as base
import oasis_first10_combined_arc as arc


def fixtures():
    pairs = [(f"{r:04d}", f"{r+10:04d}") for r in range(10)]
    rows = []
    for panel, series, alpha in base.expected_groups():
        for rank, (fixed, moving) in enumerate(pairs):
            rows.append(dict(panel=panel, series=series, alpha=alpha, case_rank=rank,
                             fixed_id=fixed, moving_id=moving, direction="moving_to_fixed",
                             dice=.8+rank*.001, hd95=1.8+rank*.01, asd=.35+rank*.002,
                             log_jdet_std_fg=.5+rank*.004, jdet_le0_pct_fg=.001*rank))
    return rows, pairs


class CombinedTests(unittest.TestCase):
    def test_scope_and_pairs(self):
        original, pairs = fixtures()
        rows = arc.select(original); summaries = arc.summarize(rows, pairs)
        self.assertEqual(len(rows), 600); self.assertEqual(len(summaries), 60)
        self.assertEqual(len({(r['source_panel'],r['source_series'],r['alpha'],r['case_rank']) for r in rows}), 450)
        self.assertFalse(any(r['source_series'] in ('842_dense','842_displacement','8421_displacement') for r in rows))
        self.assertEqual({r['source_series'] for r in rows if r['series']=='svf'}, {'842_velocity','8421_velocity'})
        self.assertEqual({r['alpha'] for r in rows if r['series']=='bounded2'}, {1.})

    def test_missing_duplicate_nonfinite(self):
        source, pairs = fixtures(); rows = arc.select(source)
        with self.assertRaises(ValueError): arc.summarize(rows[:-1], pairs)
        bad = copy.deepcopy(rows); bad[-1]['case_rank'] = 0
        with self.assertRaises(ValueError): arc.summarize(bad, pairs)
        bad = copy.deepcopy(rows); bad[-1]['dice'] = float('nan')
        with self.assertRaises(ValueError): arc.summarize(bad, pairs)

    def test_six_styles_and_bounded_marker(self):
        self.assertEqual(len({s[1] for s in arc.STYLES.values()}), 6)
        self.assertEqual(arc.STYLES['bounded2'][3], 'none')
        for s in arc.SERIES[:3]: self.assertEqual(arc.STYLES[s], base.STYLES[s])

    def test_rendered_axes_and_top_bottom(self):
        source, pairs = fixtures(); summary = arc.summarize(arc.select(source), pairs)
        for panel in arc.PANELS:
            fig, diagnostics = arc.figure(panel, summary)
            try:
                self.assertEqual(len(fig.axes), 6)
                self.assertTrue(all(r['point_count'] == 30 for r in diagnostics))
                self.assertEqual([r['ellipse_count'] for r in diagnostics], [0,0,0,30,30,30])
                for i, ax in enumerate(fig.axes):
                    metric = ('dice','hd95','asd')[i%3]
                    self.assertEqual(tuple(ax.get_xlim()), base.AXES['x']['limits'])
                    self.assertEqual(tuple(ax.get_ylim()), base.AXES[metric]['limits'])
                    self.assertEqual(tuple(ax.get_xticks()), base.AXES['x']['ticks'])
                    self.assertEqual(tuple(ax.get_yticks()), base.AXES[metric]['ticks'])
                    self.assertEqual([len(line.get_xdata()) for line in ax.lines], [5,5,5,7,1,7])
                    self.assertEqual(ax.lines[4].get_linestyle(), 'None')
            finally: plt.close(fig)


if __name__ == '__main__': unittest.main()
