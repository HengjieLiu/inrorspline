#!/usr/bin/env python
"""Run production-path DirectCP 8-4-2 DVF-only timing jobs."""

from __future__ import annotations

from pathlib import Path

import common as c
import run_directcp842_dvf_only_timing as runner


runner.VARIANT = "oasis_seed3_directcp842_production_dvf_timing_alpha1000_v1"
runner.DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_production_dvf_timing_alpha1000_v1")
runner.DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_production_dvf_timing"


if __name__ == "__main__":
    runner.main()
