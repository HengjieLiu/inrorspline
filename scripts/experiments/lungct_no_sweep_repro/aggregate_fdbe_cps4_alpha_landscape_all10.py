#!/usr/bin/env python
"""Aggregate the LungCT unmasked FD-BE cps4 alpha-landscape runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_fdbe_cps4_alpha_extension_all10 as base  # noqa: E402


DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/lungct_fdbe_cps4_alpha_landscape_all10_v1")
DEFAULT_REVIEW_DIR = base.REPO_ROOT / "experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10"
LANDSCAPE_ALPHAS = (10000.0, 20000.0, 50000.0, 200000.0, 500000.0)


def make_lanes() -> tuple[base.LaneSpec, ...]:
    lanes: list[base.LaneSpec] = []
    for alpha in LANDSCAPE_ALPHAS:
        lanes.extend(
            [
                base.LaneSpec(
                    f"inr_bscp_a{base.c.format_float(alpha)}",
                    "inr_bscp",
                    "INR-BSCP cps4 (FD-BE)",
                    "sinr",
                    "sinr_cps4_global_ncc_fd_be",
                    1.0e-4,
                    alpha,
                ),
                base.LaneSpec(
                    f"d_bscp_a{base.c.format_float(alpha)}",
                    "d_bscp",
                    "D-BSCP cps4 (FD-BE)",
                    "directcp",
                    "direct_cp_cps4_global_ncc_fd_be",
                    1.0e-3,
                    alpha,
                ),
            ]
        )
    return tuple(lanes)


def aggregate_mean_rows(case_rows: Sequence[dict[str, Any]], source_path: str) -> list[dict[str, Any]]:
    rows = _ORIGINAL_AGGREGATE_MEAN_ROWS(case_rows, source_path)
    for row in rows:
        row["source_group"] = "lungct_fdbe_cps4_alpha_landscape_all10"
    return rows


def write_report(path: Path, manifest: dict[str, Any], mean_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# LungCT FD-BE cps4 alpha-landscape aggregation",
        "",
        f"- Updated: `{manifest['updated_at_utc']}`",
        "- Scope: unmasked full-crop global NCC + dense FD-BE, LungCT cases 01-10.",
        "- New landscape alphas in this run: `1e4, 2e4, 5e4, 2e5, 5e5`.",
        "- Fixed T00 lung mask is metric-only in these rows.",
        f"- Acceptance: `{manifest['accepted']}` (`{manifest['completed_valid_rows']}/{manifest['expected_rows']}` valid rows).",
        "",
        "| Method | alpha | Cases | TRE | SDlogJ | Folding % | Full 1-NCC | Fixed-mask 1-NCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append(
            "| {method} | {alpha:g} | {cases} | {tre:.3f} | {logj:.4f} | {fold:.4f} | {full:.5f} | {mask:.5f} |".format(
                method=row["display_method"],
                alpha=float(row["reg_value"]),
                cases=int(row["cases"]),
                tre=float(row["original_snap_tre_mean_mm_mean"]),
                logj=float(row["fg_logj_std_mean"]),
                fold=float(row["fg_jdet_le0_pct_mean"]),
                full=float(row["full_crop_one_minus_ncc_mean"]),
                mask=float(row["fixed_t00_lung_mask_one_minus_ncc_mean"]),
            )
        )
    if manifest["validation_errors"]:
        lines.extend(["", "## Validation Errors", ""])
        for error in manifest["validation_errors"][:200]:
            lines.append(f"- `{error}`")
    path.write_text("\n".join(lines) + "\n")


_ORIGINAL_AGGREGATE_MEAN_ROWS = base.aggregate_mean_rows
base.DEFAULT_RUN_ROOT = DEFAULT_RUN_ROOT
base.DEFAULT_REVIEW_DIR = DEFAULT_REVIEW_DIR
base.LANES = make_lanes()
base.aggregate_mean_rows = aggregate_mean_rows
base.write_report = write_report


if __name__ == "__main__":
    base.main()
