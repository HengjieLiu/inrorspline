#!/usr/bin/env python
"""Launch the OASIS first10 DirectCP cps2 completion alpha add-on."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import launch_direct_cp_cps2_prelim as launcher  # noqa: E402


DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay"
VARIANT = "oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1"


def has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def main() -> None:
    user_args = sys.argv[1:]
    defaults = [
        ("--gpus", "1,2,3"),
        ("--alphas", "0,10,100000"),
        ("--out-root", str(DEFAULT_OUT_ROOT)),
        ("--review-root", str(DEFAULT_REVIEW_ROOT)),
        ("--variant", VARIANT),
    ]
    injected: list[str] = []
    for option, value in defaults:
        if not has_option(user_args, option):
            injected.extend([option, value])
    if not has_option(user_args, "--skip-aggregate"):
        injected.append("--skip-aggregate")
    sys.argv = [sys.argv[0], *injected, *user_args]
    launcher.main()


if __name__ == "__main__":
    main()
