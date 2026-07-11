#!/usr/bin/env python
"""Regression tests for OASIS absent-label metric handling."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/experiments/oasis_alpha_sweep"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common as c  # noqa: E402


class OasisLabelMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_labels = list(c.OASIS_LABELS)
        c.OASIS_LABELS = [1, 2]

    def tearDown(self) -> None:
        c.OASIS_LABELS = self.original_labels

    def test_dice_nan_when_fixed_moving_or_warped_label_absent(self) -> None:
        fixed = torch.zeros((4, 4, 4), dtype=torch.float32)
        moving = torch.zeros_like(fixed)
        warped = torch.zeros_like(fixed)
        fixed[0:2, 0:2, 0:2] = 1
        moving[0:2, 0:2, 0:2] = 1
        warped[0:2, 0:2, 0:2] = 1
        fixed[3, 3, 3] = 2
        moving[3, 3, 3] = 2

        values = c.dice_per_label(warped, fixed, moving)

        self.assertGreater(values[0], 0.99)
        self.assertTrue(math.isnan(values[1]))
        self.assertGreater(c.safe_nanmean(values), 0.99)

    def test_surface_metrics_nan_when_label_absent(self) -> None:
        if c.surface_distance_metrics is None:
            self.skipTest("surface_distance is not installed")
        fixed = torch.zeros((5, 5, 5), dtype=torch.float32)
        moving = torch.zeros_like(fixed)
        warped = torch.zeros_like(fixed)
        fixed[1:3, 1:3, 1:3] = 1
        moving[1:3, 1:3, 1:3] = 1
        warped[1:3, 1:3, 1:3] = 1
        fixed[4, 4, 4] = 2
        moving[4, 4, 4] = 2

        metrics = c.surface_metrics_per_label(warped, fixed, moving)

        self.assertTrue(math.isnan(metrics["hd95"][1]))
        self.assertTrue(math.isnan(metrics["asd_symmetric"][1]))

    def test_safe_nanmean_uses_finite_values_only(self) -> None:
        self.assertEqual(c.safe_nanmean([1.0, float("nan"), 3.0]), 2.0)
        self.assertTrue(math.isnan(c.safe_nanmean([float("nan")])))


if __name__ == "__main__":
    unittest.main()
