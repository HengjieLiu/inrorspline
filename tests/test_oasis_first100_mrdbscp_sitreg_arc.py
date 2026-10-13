import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_first100_mrdbscp_sitreg_arc as plot


class FigureTests(unittest.TestCase):
    def fixture(self):
        names = [n for g in plot.SERIES.values() for _, n in g]+["VFA (original)"]
        rows = [dict(method=n, rank=i, **{m:.2+i*.001 for m in plot.source.METRICS}) for n in names for i in range(100)]
        return {"per_case":rows, "means":plot.source.summarize(rows)}

    def test_validation_and_layout(self):
        means = plot.validated_means(self.fixture())
        self.assertEqual(len(means), 9)
        fig, axes = plot.build_figure(means)
        try:
            self.assertEqual(axes.shape, (2,3))
            for col in range(3):
                self.assertEqual(len(axes[0,col].patches), 0)
                self.assertEqual(len(axes[1,col].patches), 9)
                self.assertEqual(axes[0,col].get_ylim(), axes[1,col].get_ylim())
                ellipse = axes[1,col].patches[0]
                first = means[plot.SERIES["MR-D-BSCP"][0][1]]
                self.assertEqual(ellipse.width, 2*first[plot.X+"_std"])
                self.assertEqual(ellipse.height, 2*first[plot.Y[col][0]+"_std"])
        finally:
            plot.plt.close(fig)

    def test_changed_statistics_rejected(self):
        data = self.fixture()
        data["means"][0]["dice_std"] += .1
        with self.assertRaises(ValueError):
            plot.validated_means(data)


if __name__ == "__main__":
    unittest.main()
