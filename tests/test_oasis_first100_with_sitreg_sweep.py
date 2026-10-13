import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_first100_with_sitreg_sweep as report


class SummaryTests(unittest.TestCase):
    def rows(self):
        return [dict(method="test", rank=r, **{m:float(r) for m in report.METRICS}) for r in range(100)]

    def test_sample_sd(self):
        row = report.summarize(self.rows())[0]
        self.assertEqual(row["dice_mean"], 49.5)
        self.assertAlmostEqual(row["dice_std"], (100*101/12)**0.5)

    def test_incomplete_duplicate_nonfinite(self):
        for rows in (self.rows()[:-1], self.rows()+[self.rows()[0]]):
            with self.assertRaises(ValueError):
                report.summarize(rows)
        rows = self.rows()
        rows[0]["dice"] = float("nan")
        with self.assertRaises(ValueError):
            report.summarize(rows)


if __name__ == "__main__":
    unittest.main()
