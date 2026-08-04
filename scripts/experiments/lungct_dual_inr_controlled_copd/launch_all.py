#!/usr/bin/env python3
"""Launch the COPD controlled N0/N1/P0/P1 cohort with the shared GPU queue."""

from __future__ import annotations

import sys

from scripts.experiments.lungct_dual_inr_controlled import launch_all as core


def main() -> None:
    core.RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_controlled_copd.run_one"
    # The six-arm report is built after the dependent A cohort completes.
    # Callers can still run the legacy N/P-only aggregate explicitly if needed.
    core.AGGREGATE_MODULE = "scripts.experiments.lungct_dual_inr_copd.aggregate"
    if "--skip-aggregate" not in sys.argv:
        sys.argv.append("--skip-aggregate")
    core.main()


if __name__ == "__main__":
    main()
