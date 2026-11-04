from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import report_oasis_bounded2_842_sweep as report


class ReportTests(unittest.TestCase):
    def test_matrix_rejects_missing_duplicate_and_wrong_lambda(self):
        rows = [{"case_rank": rank, "alpha": alpha} for alpha in report.r.LAMBDAS for rank in range(10)]
        report.require_matrix(rows)
        for bad in (rows[:-1], rows[:-1] + [rows[0]], rows[:-1] + [{"case_rank": 9, "alpha": 2.}]):
            with self.assertRaises(ValueError): report.require_matrix(bad)

    def test_sample_statistics_and_cohort(self):
        rows = [{"case_rank": rank, **{k: rank/10. for k in report.METRICS}} for rank in range(10)]
        summary = report.summarize(rows, "test", .1)
        values = np.arange(10)/10.
        self.assertAlmostEqual(summary["dice_mean"], values.mean(), places=12)
        self.assertAlmostEqual(summary["dice_sd"], values.std(ddof=1), places=12)
        with self.assertRaises(ValueError): report.summarize(rows[:-1], "test", .1)
        rows[-1]["dice"] = float("nan")
        with self.assertRaises(ValueError): report.summarize(rows, "test", .1)

    def test_history_weight_update_and_best_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            rows = [{"epoch": i, "optimizer_updates_completed": i, "data_loss": .8-i*.01,
                     "reg_loss": .01, "weighted_reg_loss": .03, "loss": .83-i*.01,
                     "lr": .003} for i in range(3)]
            report.write_csv(path / "history.csv", rows)
            block = {"alpha": 3., "model_path": str(path / "best.pt"),
                     "optimizer": {"actual_epochs": 2, "best_epoch": 2, "stage": {"lr": .003}}}
            self.assertEqual(report.verify_history(block), 3)
            block["alpha"] = 1.
            with self.assertRaises(ValueError): report.verify_history(block)


if __name__ == "__main__": unittest.main()
