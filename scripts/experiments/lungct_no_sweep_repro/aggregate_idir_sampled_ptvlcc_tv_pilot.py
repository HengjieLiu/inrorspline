#!/usr/bin/env python
"""Aggregate LungCT IDIR sampled pTV-LCC + TV weight-0.11 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import aggregate_report as base
import common as c
from launch_idir_sampled_ptvlcc_tv_w0p11_all10 import DEFAULT_OUT_ROOT
from run_one import METHOD_IDIR, METHOD_IDIR_SAMPLED_PTV_TV, METHOD_DIRECT_CP_32_PTV


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_lungct_idir_sampled_ptvlcc_tv_w0p11_all10"
EXPECTED_CASES = list(range(1, 11))
METRIC_KEYS = [
    "original_snap_tre_mean_mm",
    "original_subvoxel_tre_mean_mm",
    "reg1mm_snap_tre_mean_mm",
    "reg1mm_subvoxel_tre_mean_mm",
    "fg_logj_std",
    "fg_jdet_le0_pct",
    "fg_bending_energy",
    "fg_diffusion_l2",
    "elapsed_sec",
    "peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--expected-cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    return parser.parse_args()


def method_label(method: str) -> str:
    if method == METHOD_IDIR_SAMPLED_PTV_TV:
        return "IDIR sampled pTV-LCC + TV"
    if method == METHOD_IDIR:
        return "IDIR BE"
    if method == METHOD_DIRECT_CP_32_PTV:
        return "DirectCP 4-stage pTV-LCC + isoTV"
    return method


def row_from_summary(path: Path, payload: Dict[str, object]) -> Dict[str, object]:
    flat = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    row = {
        "method": str(flat.get("method")),
        "method_label": method_label(str(flat.get("method"))),
        "case_id": int(flat.get("case_id") or payload.get("case_id")),
        "status": flat.get("status"),
        "ptv_isotv_weight": training.get("ptv_isotv_weight"),
        "sampled_lcc_centers": training.get("sampled_lcc_centers"),
        "sampled_lcc_offsets": training.get("sampled_lcc_offsets"),
        "train_coords": training.get("train_coords"),
        "summary_path": str(path),
        "run_name": payload.get("run_name"),
    }
    row.update(
        {
            "original_identity_tre_mean_mm": flat.get("native_crop_identity_tre_mean_mm"),
            "original_snap_tre_mean_mm": flat.get("native_crop_snap_tre_mean_mm"),
            "original_snap_tre_std_mm": flat.get("native_crop_snap_tre_std_mm"),
            "original_snap_tre_median_mm": flat.get("native_crop_snap_tre_median_mm"),
            "original_subvoxel_tre_mean_mm": flat.get("native_crop_sub_tre_mean_mm"),
            "original_subvoxel_tre_std_mm": flat.get("native_crop_sub_tre_std_mm"),
            "original_subvoxel_tre_median_mm": flat.get("native_crop_sub_tre_median_mm"),
            "reg1mm_identity_tre_mean_mm": flat.get("reg1mm_identity_tre_mean_mm"),
            "reg1mm_snap_tre_mean_mm": flat.get("reg1mm_snap_tre_mean_mm"),
            "reg1mm_snap_tre_std_mm": flat.get("reg1mm_snap_tre_std_mm"),
            "reg1mm_snap_tre_median_mm": flat.get("reg1mm_snap_tre_median_mm"),
            "reg1mm_subvoxel_tre_mean_mm": flat.get("reg1mm_sub_tre_mean_mm"),
            "reg1mm_subvoxel_tre_std_mm": flat.get("reg1mm_sub_tre_std_mm"),
            "reg1mm_subvoxel_tre_median_mm": flat.get("reg1mm_sub_tre_median_mm"),
            "fg_logj_std": flat.get("fg_log_jdet_std"),
            "fg_jdet_le0_pct": flat.get("fg_jdet_le0_pct"),
            "fg_bending_energy": flat.get("fg_bending_energy"),
            "fg_diffusion_l2": flat.get("fg_diffusion_l2"),
            "elapsed_sec": flat.get("elapsed_sec"),
            "peak_mem_gb": flat.get("peak_mem_gb"),
        }
    )
    return row


def collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((root / METHOD_IDIR_SAMPLED_PTV_TV).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        rows.append(row_from_summary(summary_path, payload))
    return rows


def completion_errors(rows: Sequence[Dict[str, object]], expected_cases: Sequence[int]) -> List[str]:
    errors: List[str] = []
    by_case = {}
    for row in rows:
        case_id = int(row["case_id"])
        if case_id in by_case:
            errors.append(f"duplicate completed case {case_id:02d}")
        by_case[case_id] = row
    got = sorted(by_case)
    expected = sorted(int(v) for v in expected_cases)
    if got != expected:
        errors.append(f"completed cases {got}, expected {expected}")
    return errors


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    out: Dict[str, object] = {
        "method": METHOD_IDIR_SAMPLED_PTV_TV,
        "method_label": method_label(METHOD_IDIR_SAMPLED_PTV_TV),
        "cases": len(rows),
        "ptv_isotv_weight": rows[0].get("ptv_isotv_weight"),
    }
    for key in METRIC_KEYS:
        mean, std = base.mean_std([row.get(key) for row in rows])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return [out]


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return base.markdown_table(headers, rows)


def main() -> None:
    args = parse_args()
    expected_cases = c.parse_int_list(args.expected_cases)
    rows = sorted(collect_rows(args.out_root), key=lambda row: int(row["case_id"]))
    errors = completion_errors(rows, expected_cases)
    mean_rows = aggregate_rows(rows)

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    c.write_csv(args.review_root / "mean_rows.csv", mean_rows)
    c.write_json(args.review_root / "mean_rows.json", mean_rows)
    c.write_json(args.review_root / "aggregation_status.json", {"errors": errors, "completed_rows": len(rows)})

    mean_table_rows = [
        [
            row["method_label"],
            row["cases"],
            base.fmt(row.get("original_snap_tre_mean_mm_mean")),
            base.fmt(row.get("original_subvoxel_tre_mean_mm_mean")),
            base.fmt(row.get("reg1mm_snap_tre_mean_mm_mean")),
            base.fmt(row.get("reg1mm_subvoxel_tre_mean_mm_mean")),
            base.fmt(row.get("fg_logj_std_mean"), 6),
            base.fmt(row.get("fg_jdet_le0_pct_mean"), 4),
            base.fmt(row.get("fg_bending_energy_mean"), 6),
            base.fmt(row.get("fg_diffusion_l2_mean"), 6),
            base.fmt(row.get("elapsed_sec_mean"), 1),
            base.fmt(row.get("peak_mem_gb_mean"), 2),
        ]
        for row in mean_rows
    ]
    case_table_rows = [
        [
            f"{int(row['case_id']):02d}",
            base.fmt(row.get("original_identity_tre_mean_mm")),
            base.fmt(row.get("original_snap_tre_mean_mm")),
            base.fmt(row.get("original_subvoxel_tre_mean_mm")),
            base.fmt(row.get("reg1mm_snap_tre_mean_mm")),
            base.fmt(row.get("reg1mm_subvoxel_tre_mean_mm")),
            base.fmt(row.get("fg_logj_std"), 6),
            base.fmt(row.get("fg_jdet_le0_pct"), 4),
            base.fmt(row.get("elapsed_sec"), 1),
            base.fmt(row.get("peak_mem_gb"), 2),
        ]
        for row in rows
    ]
    report = "\n".join(
        [
            "# LungCT IDIR Sampled pTV-LCC + TV 0.11 Pilot",
            "",
            f"Output root: `{args.out_root}`",
            "",
            f"Aggregation errors: `{errors}`",
            "",
            "## Mean Row",
            "",
            table(
                [
                    "Method",
                    "Cases",
                    "Original Snap TRE",
                    "Original Sub TRE",
                    "1mm Snap TRE",
                    "1mm Sub TRE",
                    "FG logJ std",
                    "FG J<=0%",
                    "BE fg",
                    "Diff fg",
                    "Time sec",
                    "Peak GB",
                ],
                mean_table_rows,
            ),
            "",
            "## Per-Case Rows",
            "",
            table(
                [
                    "Case",
                    "Original Identity TRE",
                    "Original Snap TRE",
                    "Original Sub TRE",
                    "1mm Snap TRE",
                    "1mm Sub TRE",
                    "FG logJ std",
                    "FG J<=0%",
                    "Time sec",
                    "Peak GB",
                ],
                case_table_rows,
            ),
            "",
            "## Notes",
            "",
            "- This is an IDIR-specific sampled approximation to pTV-LCC+isoTV, not the dense pTVReg/control-grid objective.",
            "- Training samples pTV 1mm border-crop coordinates; fixed T00 lung mask is used only for foreground metrics.",
            "- Original Snap TRE is the pTVReg-comparable native/crop snap-to-voxel convention.",
        ]
    )
    (args.review_root / "review_report.md").write_text(report + "\n")
    print(
        json.dumps(
            {
                "completed_rows": len(rows),
                "errors": errors,
                "review_root": str(args.review_root),
                "case_csv": str(args.review_root / "case_rows.csv"),
                "mean_csv": str(args.review_root / "mean_rows.csv"),
                "report": str(args.review_root / "review_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
