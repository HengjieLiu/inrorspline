"""Saved-table validation tests; no model, image or field evaluation imports."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/"scripts/reports/oasis_test100_label_arc.py"
spec = importlib.util.spec_from_file_location("label_arc", SOURCE)
arc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arc)


class LabelArcTests(unittest.TestCase):
    def fixtures(self):
        pairs=[("0001","0002"),("0003","0004"),("0005","0006")]
        cases=[]
        labels=[]
        for rank,(fixed,moving) in enumerate(pairs):
            cases.append(dict(config_id="a",rank=rank,fixed_id=fixed,moving_id=moving,
                              direction="moving_to_fixed",dice=.5+rank*.1,hd95=2.,asd=1.,
                              log_jdet_std_fg=float(rank),jdet_le0_pct_fg=0.))
            for label in range(1,36):
                valid=not(label==34 and rank==2)
                labels.append(dict(config_id="a",rank=rank,fixed_id=fixed,moving_id=moving,label_id=label,
                                   label_name=f"Label{label}",included=valid,dice=.5+rank*.1 if valid else None,
                                   hd95=2. if valid else None,asd=1. if valid else None))
        return pairs,cases,labels

    def test_group_reconstructs_and_absence(self):
        pairs,cases,labels=self.fixtures()
        arc.validate_group(cases,labels,pairs)

    def test_duplicate_missing_rank_and_label(self):
        pairs,cases,labels=self.fixtures()
        for bad in (labels[:-1],labels+[labels[0]],labels[:-1]+[labels[0]]):
            with self.assertRaises(ValueError):arc.validate_group(cases,bad,pairs)
        with self.assertRaises(ValueError):arc.validate_group(cases+[cases[0]],labels,pairs)

    def test_wrong_identity_direction(self):
        pairs,cases,labels=self.fixtures()
        cases[0]["direction"]="fixed_to_moving"
        with self.assertRaises(ValueError):arc.validate_group(cases,labels,pairs)
        pairs,cases,labels=self.fixtures()
        labels[0]["fixed_id"]="9999"
        with self.assertRaises(ValueError):arc.validate_group(cases,labels,pairs)

    def test_case_reconstruction_failure(self):
        pairs,cases,labels=self.fixtures()
        cases[0]["dice"]+=.01
        with self.assertRaises(ValueError):arc.validate_group(cases,labels,pairs)

    def test_common_support_applies_to_both_axes(self):
        _,cases,labels=self.fixtures()
        config=dict(config_id="a",family="VFA",alpha=1.,lambda_key="1",historical=True,main=True)
        rows,support=arc.summarize([config],cases,labels)
        row=next(r for r in rows if r["label_id"]==34)
        self.assertEqual(row["n"],2)
        self.assertEqual(row["log_jdet_std_fg_mean"],.5)
        self.assertAlmostEqual(row["log_jdet_std_fg_sd"],math.sqrt(.5))
        self.assertEqual(next(r for r in support if r["label_id"]==34)["excluded_ranks"],[2])

    def test_intersection_not_available_case_per_method(self):
        _,cases,labels=self.fixtures()
        configs=[dict(config_id=c,family="VFA",alpha=1.,lambda_key="1",historical=True,main=True) for c in ("a","b")]
        other=[{**r,"config_id":"b"} for r in labels]
        for r in other:
            if r["label_id"]==1 and r["rank"]==0:r.update(included=False,dice=None,hd95=None,asd=None)
        rows,_=arc.summarize(configs,cases+[{**r,"config_id":"b"} for r in cases],labels+other)
        for r in rows:
            if r["label_id"]==1:
                self.assertEqual(r["n"],2)
                self.assertEqual(r["log_jdet_std_fg_mean"],1.5)

    def test_null_nan_not_infinity(self):
        for x in (None,"",float("nan"),"nan"):self.assertIsNone(arc.number(x))
        with self.assertRaises(ValueError):arc.number(float("inf"))

    def test_official_label_map_complete_unique(self):
        text="\n".join(f"{i} Name-{i} 1 2 3 0" for i in range(36))
        self.assertEqual(arc.label_names(text)[34],"Name-34")
        with self.assertRaises(ValueError):arc.label_names(text+"\n34 duplicate 0 0 0 0")
        with self.assertRaises(ValueError):arc.label_names("1 Name 0 0 0 0")

    def test_input_changes_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.json"
            path.write_text("{}")
            audit=arc.Audit()
            audit.read(path)
            path.write_text("{\"changed\":true}")
            with self.assertRaises(ValueError):audit.finish()

    def test_lambda_not_regularity_sorting(self):
        rows=[]
        for a,x in ((.1,.1),(1.,.9),(3.,.5)):
            rows.append(dict(family="VFA",alpha=a,historical=False,lambda_key=str(a),
                             log_jdet_std_fg_mean=x,log_jdet_std_fg_sd=.1,dice_mean=.8,dice_sd=.01))
        fig,ax=arc.plt.subplots()
        arc.plot_curve(ax,rows,"dice")
        self.assertEqual(list(ax.lines[0].get_xdata()),[.5,.9])
        arc.plt.close(fig)


if __name__ == "__main__":unittest.main()
