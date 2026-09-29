"""CPU-only contracts for the OASIS100 fixed-50 workflow."""
import math
import unittest

from .full100 import candidates
from .full100_report import bootstrap_ci, mean_sd


class Full100Contracts(unittest.TestCase):
    def test_candidates_are_exact(self):
        specs = candidates()
        self.assertEqual(
            [spec["name"] for spec in specs],
            ["fixed_50_50_50_cached_batch", "fixed_50_50_50_amsgrad"],
        )
        self.assertEqual(specs[0]["features"], ["cache", "batch", "cpbe"])
        self.assertEqual(specs[0]["optimizer"], "adam")
        self.assertEqual(specs[1]["features"], ["cpbe"])
        self.assertEqual(specs[1]["optimizer"], "amsgrad")
        for spec in specs:
            self.assertEqual(spec["overrides"]["stage_epochs"], "50,50,50")
            self.assertEqual(spec["overrides"]["stage_lr"], "0.03,0.01,0.003")
            self.assertEqual(spec["overrides"]["early_stop_mode"], "none")
            self.assertEqual(spec["overrides"]["stage_full_cps"], "8,4,2")
            self.assertEqual(spec["overrides"]["stage_downsample"], "4,2,1")

    def test_summary_math_ignores_nonfinite(self):
        mean, sd, count = mean_sd([1, 2, math.nan])
        self.assertEqual(count, 2)
        self.assertEqual(mean, 1.5)
        self.assertAlmostEqual(sd, 2 ** -0.5)

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_ci([1, 2, 3], 7), bootstrap_ci([1, 2, 3], 7))


if __name__ == "__main__":
    unittest.main()
