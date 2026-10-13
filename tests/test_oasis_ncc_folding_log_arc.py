from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/reports'))
import oasis_ncc_folding_log_arc as plot


class LogFoldingTests(unittest.TestCase):
    def test_labels_and_display_floor(self):
        self.assertEqual(plot.label_value(0), '0%')
        self.assertNotEqual(plot.label_value(4.62e-7), '0%')
        self.assertEqual(plot.display_floor([{plot.X+'_mean': v} for v in [0, 4.62e-7, 3.9]]), 1e-8)
        for value in [-1, float('nan')]:
            with self.assertRaises(ValueError):
                plot.label_value(value)

    def test_clipped_arithmetic_sd_not_log_sd(self):
        p = plot.sd_contour(.01, .8, .02, .03, 1e-8)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all(p[:, 0] >= 1e-8))
        self.assertAlmostEqual(p[:, 0].max(), .03)
        self.assertAlmostEqual(p[:, 1].min(), .77)
        self.assertAlmostEqual(p[:, 1].max(), .83)

    def test_both_real_figures_keep_all_actual_values(self):
        # Synthetic nine-point fixture, including exact zeros and tiny nonzeros.
        means = {}
        values = [4.62e-7, .000035, .00053, .0099, 0, 0, .0005, .013, .059]
        index = 0
        for loss, group in plot.base.old.SERIES.items():
            for key, alpha in group:
                r = dict(loss=loss, lambda_key=key, alpha=alpha, historical=loss=='lncc9' and key=='1')
                for m in plot.base.METRICS:
                    r[m+'_mean'], r[m+'_std'] = .8, .03
                r[plot.X+'_mean'], r[plot.X+'_std'] = values[index], values[index] / 2
                means[loss, key] = r
                index += 1
        original = [r[plot.X+'_mean'] for r in means.values()]
        for model in ['sitreg', 'fedda']:
            fig, axes = plot.build_figure(model, means, list(means.values()), 1e-8)
            try:
                for ax in axes.flat:
                    self.assertEqual(ax.get_xscale(), 'log')
                    self.assertEqual(len(ax.texts), 9)
                    self.assertEqual(sum('0%†' in t.get_text() for t in ax.texts), 2)
                    self.assertTrue(all('%' in t.get_text() and 'λ=' in t.get_text() for t in ax.texts))
                self.assertEqual(len(axes[0, 0].patches), 0)
                self.assertEqual(len(axes[1, 0].patches), 7)
                self.assertEqual(original, [r[plot.X+'_mean'] for r in means.values()])
            finally:
                plot.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
