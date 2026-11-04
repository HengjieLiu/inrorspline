import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_first10_lambda1_table as t
import oasis_first10_family_arc as a


class LambdaOneTests(unittest.TestCase):
    def fixtures(self):
        pairs = [(f"{i:04d}", f"{i+10:04d}") for i in range(10)]
        rows = []
        for panel, series, _, _ in t.ORDER:
            for rank, (fixed, moving) in enumerate(pairs):
                rows.append(dict(panel=panel, series=series, alpha=1., case_rank=rank,
                                 fixed_id=fixed, moving_id=moving, direction="moving_to_fixed",
                                 **{m: .3 + .01*rank for m in t.METRICS}))
        return a.summarize(rows, complete=False), rows, pairs

    def test_order_and_scope(self):
        rows, cases = t.select(*self.fixtures())
        self.assertEqual(len(rows), 11)
        self.assertEqual(len(cases), 110)
        self.assertEqual([r["method"] for r in rows[:3]], ["SITReg", "VFA", "FeDDA-1MF"])
        self.assertEqual([r["source_panel"] for r in rows], list("AAABBCDCDEE"))
        self.assertNotIn("842_dense", [r["source_series"] for r in rows])
        self.assertEqual(rows[-1]["source_series"], "8421_velocity")
        self.assertEqual(rows[-1]["stage_weights"], "1/1/1/1")

    def test_missing_duplicate_wrong_direction_and_statistics(self):
        summary, cases, pairs = self.fixtures()
        with self.assertRaises(ValueError): t.select(summary, cases[:-1], pairs)
        with self.assertRaises(ValueError): t.select(summary, cases + [cases[0]], pairs)
        bad = copy.deepcopy(cases); bad[0]["direction"] = "fixed_to_moving"
        with self.assertRaises(ValueError): t.select(summary, bad, pairs)
        bad_summary = copy.deepcopy(summary); bad_summary[0]["dice_sd"] = 0.
        with self.assertRaises(ValueError): t.select(bad_summary, cases, pairs)

    def test_lambda_filter_and_folding_precision(self):
        summary, cases, pairs = self.fixtures()
        extra = [{**r, "alpha": 3.} for r in cases]
        rows, selected = t.select(summary, cases+extra, pairs)
        self.assertEqual(len(selected), 110)
        rows[0]["jdet_le0_pct_fg_mean"] = .000028
        self.assertIn("0.000028 ±", t.markdown(rows))


if __name__ == "__main__": unittest.main()
