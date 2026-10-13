import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_dl_loss_lines_only as plot


class LineOnlyTests(unittest.TestCase):
    def fixture(self):
        pairs = plot.pairs()
        keys = [(loss, alpha) for loss in plot.LOSSES for alpha in plot.GRIDS[loss]]
        keys += [(ref, None) for ref in plot.REFERENCES]
        rows, means = [], []
        for loss, alpha in keys:
            for rank in range(10):
                rows.append({"loss": loss, "effective_lambda": alpha, "case_rank": rank,
                             "fixed_id": pairs[rank][0], "moving_id": pairs[rank][1],
                             "direction": "moving_to_fixed", "resolution": "full", "representation": "842_dense",
                             **{metric: .1+.01*rank for metric in plot.METRICS}})
            means.append({"loss": loss, "effective_lambda": alpha, "n": 10,
                          **{f"{metric}_{suffix}": value for metric in plot.METRICS for suffix, value in
                             (("mean", .145), ("std", float(np.std(np.arange(10)*.01+.1, ddof=1))))}})
        data = {"complete": True, "variant": plot.VARIANT, "case_metrics": rows,
                "means": means, "candidates": [None]*350}
        accepted = {"technical_completion": True, "candidate_count": 350, "configuration_count": 35,
                    "historical_ncc_count": 70, "original_reference_count": 20}
        return data, accepted, pairs

    def test_complete_cohorts_and_recomputed_values(self):
        self.assertEqual(len(plot.validate_payload(*self.fixture())), 44)

    def test_incomplete_duplicate_changed_and_nonfinite_rejected(self):
        for kind in ("incomplete", "duplicate", "mean", "nonfinite", "case", "ncc_branch"):
            data, accepted, pairs = copy.deepcopy(self.fixture())
            if kind == "incomplete":
                accepted["technical_completion"] = False
            elif kind == "duplicate":
                data["case_metrics"][0]["case_rank"] = 1
            elif kind == "mean":
                data["means"][0]["dice_mean"] += .01
            elif kind == "nonfinite":
                data["case_metrics"][0]["dice"] = float("nan")
            elif kind == "case":
                data["case_metrics"][0]["fixed_id"] = "9999"
            else:
                data["case_metrics"][0]["representation"] = "8421"
            with self.assertRaises(ValueError):
                plot.validate_payload(data, accepted, pairs)

    def test_exactly_six_plain_lines_two_points_and_no_ellipses(self):
        means = plot.validate_payload(*self.fixture())
        fig, axes = plot.build_figure(means)
        try:
            self.assertEqual(len(axes), 3)
            for axis in axes:
                self.assertEqual(len(axis.lines), 8)
                self.assertEqual(len(axis.patches), 0)
                self.assertEqual(len(axis.collections), 0)
                self.assertEqual(len(axis.containers), 0)
                for line in axis.lines[:6]:
                    self.assertEqual(len(line.get_xdata()), 7)
                    self.assertIn(line.get_marker(), (None, "None", ""))
                    self.assertEqual(line.get_linestyle(), "-")
                for line in axis.lines[6:]:
                    self.assertEqual(len(line.get_xdata()), 1)
                    self.assertEqual(line.get_linestyle(), "None")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
