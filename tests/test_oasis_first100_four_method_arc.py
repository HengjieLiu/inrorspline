import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_first100_four_method_arc as plot


class PlotTests(unittest.TestCase):
    def test_four_series_and_ellipse_geometry(self):
        means=[]
        for family in plot.COLORS:
            for alpha in ([3.,1.,.3,.1] if family=="MR-D-BSCP" else [10.,3.,1.,1/3,.1]):
                row=dict(family=family,alpha=alpha,historical=family!="MR-D-BSCP" and alpha==1.)
                for m in plot.s.METRICS:
                    row[m+"_mean"]=.8-alpha*.04
                    row[m+"_std"]=.03
                means.append(row)
        fig,axes=plot.build_figure(means)
        try:
            self.assertEqual(axes.shape,(2,3))
            for col in range(3):
                self.assertEqual(len(axes[0,col].patches),0)
                self.assertEqual(len(axes[1,col].patches),19)
                self.assertEqual(axes[1,col].patches[0].width,.06)
                self.assertEqual(axes[0,col].get_ylim(),axes[1,col].get_ylim())
                self.assertEqual(sum(l.get_marker()=="D" for l in axes[0,col].lines),3)
        finally:
            plot.plt.close(fig)

    def test_cohort_checks(self):
        pairs=[(str(i),str(i+1)) for i in range(100)]
        rows=[dict(method="test",rank=i,fixed_id=str(i),moving_id=str(i+1),
                   **{m:.1 for m in plot.s.METRICS}) for i in range(100)]
        plot.check_rows(rows,pairs)
        summary=plot.s.summarize(rows)
        plot.check_summary(rows,summary)
        summary[0]["dice_std"]+=.01
        with self.assertRaises(ValueError): plot.check_summary(rows,summary)
        rows[0]["fixed_id"]="wrong"
        with self.assertRaises(ValueError): plot.check_rows(rows,pairs)


if __name__=="__main__": unittest.main()
