#!/usr/bin/env python
"""Aggregate COPD ``ptvreg-python`` fixed-parameter summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths
import run_copd_case as runner


EXPERIMENT_KEY = "2026-08-28_ptvreg_python_copd_fixed_all10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=runner.PROTOCOL_CHOICES, default="canonical_1mm")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--case-ids", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.out_root = runner.resolve_run_root(args.out_root)
    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=runner.REPO_ROOT,
    )
    return args


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def mean_std(values: Sequence[object]) -> tuple[float, float]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def fmt(value: object, digits: int = 3) -> str:
    f = safe_float(value)
    if math.isnan(f):
        return "n/a"
    return f"{f:.{digits}f}"


def metric(payload: Dict[str, object], key: str) -> Dict[str, object]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def flatten_summary(path: Path, payload: Dict[str, object]) -> Dict[str, object]:
    native_snap = metric(payload, "native_crop_tre_snap_to_voxel")
    native_sub = metric(payload, "native_crop_tre_subvoxel")
    native_identity = metric(payload, "native_crop_identity_tre")
    reg_snap = metric(payload, "tre_snap_to_voxel")
    reg_sub = metric(payload, "tre_subvoxel")
    reg_identity = metric(payload, "identity_tre")
    regularity = metric(payload, "regularity_fg_registration_crop")
    opts = metric(payload, "opts")
    source = metric(payload, "source_provenance")
    input_metadata = metric(payload, "input_metadata")
    return {
        "case_id": int(payload.get("case_id", -1)),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "run_name": payload.get("run_name"),
        "status": payload.get("status"),
        "protocol": payload.get("protocol"),
        "protocol_tag": payload.get("protocol_tag"),
        "summary_path": str(path),
        "input_shape_zyx": payload.get("input_shape_zyx"),
        "dense_field_shape_zyx": payload.get("dense_field_shape_zyx"),
        "dense_field_shape_matches_input": payload.get("dense_field_shape_matches_input"),
        "dense_field_all_finite": payload.get("dense_field_all_finite"),
        "dense_field_nan_count": payload.get("dense_field_nan_count"),
        "native_crop_identity_tre_mean_mm": native_identity.get("tre_mean_mm"),
        "native_crop_snap_tre_mean_mm": native_snap.get("tre_mean_mm"),
        "native_crop_snap_tre_std_mm": native_snap.get("tre_std_mm"),
        "native_crop_snap_tre_median_mm": native_snap.get("tre_median_mm"),
        "native_crop_sub_tre_mean_mm": native_sub.get("tre_mean_mm"),
        "native_crop_sub_tre_std_mm": native_sub.get("tre_std_mm"),
        "native_crop_sub_tre_median_mm": native_sub.get("tre_median_mm"),
        "reg1mm_identity_tre_mean_mm": reg_identity.get("tre_mean_mm"),
        "reg1mm_snap_tre_mean_mm": reg_snap.get("tre_mean_mm"),
        "reg1mm_snap_tre_std_mm": reg_snap.get("tre_std_mm"),
        "reg1mm_snap_tre_median_mm": reg_snap.get("tre_median_mm"),
        "reg1mm_sub_tre_mean_mm": reg_sub.get("tre_mean_mm"),
        "reg1mm_sub_tre_std_mm": reg_sub.get("tre_std_mm"),
        "reg1mm_sub_tre_median_mm": reg_sub.get("tre_median_mm"),
        "registration_grid_snap_tre_mean_mm": reg_snap.get("tre_mean_mm"),
        "registration_grid_sub_tre_mean_mm": reg_sub.get("tre_mean_mm"),
        "registration_grid_spacing_zyx_mm": payload.get("registration_grid_spacing_zyx_mm"),
        "regularity_spacing_basis": payload.get("regularity_spacing_basis", "registration_grid_voxel_derivatives"),
        "fg_log_jdet_std": regularity.get("log_jdet_std_fg"),
        "fg_jdet_le0_pct": regularity.get("jdet_le0_pct_fg"),
        "fg_bending_energy": regularity.get("bending_energy_fg"),
        "fg_diffusion_l2": regularity.get("diffusion_l2_fg"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "peak_mem_gb": payload.get("peak_mem_gb"),
        "external_ptvreg_python_commit": source.get("external_ptvreg_python_commit"),
        "repo_commit": source.get("repo_commit"),
        "repo_dirty_tracked": source.get("repo_dirty_tracked"),
        "input_protocol": input_metadata.get("input_protocol"),
        "grid_spacing": opts.get("grid_spacing"),
        "isoTV": opts.get("isoTV"),
        "metric": opts.get("metric"),
        "metric_param": opts.get("metric_param"),
        "opt_method": opts.get("opt_method"),
        "max_iters": opts.get("max_iters"),
        "admm_max_outer": opts.get("admm_max_outer"),
        "admm_lbfgs_iters": opts.get("admm_lbfgs_iters"),
    }


def collect_rows(out_root: Path, protocol: str, requested_cases: Sequence[int]) -> List[Dict[str, object]]:
    tag = runner.protocol_tag(protocol)
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((Path(out_root) / tag).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        if payload.get("dataset") != "copd" or payload.get("method") != runner.METHOD:
            continue
        if payload.get("protocol") != protocol:
            continue
        case_id = int(payload.get("case_id", -1))
        if case_id not in requested_cases:
            continue
        rows.append(flatten_summary(summary_path, payload))
    rows.sort(key=lambda row: int(row["case_id"]))
    return rows


def acceptance_from_rows(rows: Sequence[Dict[str, object]], requested_cases: Sequence[int]) -> Dict[str, object]:
    present = {int(row["case_id"]) for row in rows}
    missing = [int(case_id) for case_id in requested_cases if int(case_id) not in present]
    metric_keys = [
        "native_crop_snap_tre_mean_mm",
        "native_crop_sub_tre_mean_mm",
        "reg1mm_snap_tre_mean_mm",
        "reg1mm_sub_tre_mean_mm",
        "fg_log_jdet_std",
        "fg_jdet_le0_pct",
        "fg_bending_energy",
        "fg_diffusion_l2",
    ]
    invalid: List[str] = []
    for row in rows:
        case_id = int(row["case_id"])
        for key in metric_keys:
            if not math.isfinite(safe_float(row.get(key))):
                invalid.append(f"case{case_id:02d}:{key}")
        if row.get("dense_field_shape_matches_input") is not True:
            invalid.append(f"case{case_id:02d}:dense_field_shape_matches_input")
        if row.get("dense_field_all_finite") is not True:
            invalid.append(f"case{case_id:02d}:dense_field_all_finite")
        if int(row.get("dense_field_nan_count") or 0) != 0:
            invalid.append(f"case{case_id:02d}:dense_field_nan_count")
    passed = len(missing) == 0 and len(invalid) == 0
    return {
        "passed": bool(passed),
        "n_completed": len(rows),
        "n_requested": len(requested_cases),
        "missing_cases": missing,
        "invalid_items": invalid,
        "required_tre_conventions": [
            "native_crop_tre_snap_to_voxel",
            "native_crop_tre_subvoxel",
            "tre_snap_to_voxel",
            "tre_subvoxel",
        ],
        "headline_metric": "native_crop_tre_snap_to_voxel",
    }


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    requested_cases = runner.c.parse_int_list(args.case_ids)
    rows = collect_rows(args.out_root, args.protocol, requested_cases)
    acceptance = acceptance_from_rows(rows, requested_cases)
    if not acceptance["passed"] and not args.allow_missing:
        raise SystemExit(json.dumps(acceptance, indent=2))

    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)
    prefix = f"ptvreg_python_copd_{args.protocol}"
    runner.c.write_csv(review_root / f"{prefix}_case_rows.csv", rows)
    runner.c.write_json(review_root / f"{prefix}_case_rows.json", rows)

    native_snap_mean, native_snap_std = mean_std([row.get("native_crop_snap_tre_mean_mm") for row in rows])
    native_sub_mean, native_sub_std = mean_std([row.get("native_crop_sub_tre_mean_mm") for row in rows])
    reg_snap_mean, reg_snap_std = mean_std([row.get("reg1mm_snap_tre_mean_mm") for row in rows])
    reg_sub_mean, reg_sub_std = mean_std([row.get("reg1mm_sub_tre_mean_mm") for row in rows])
    registration_grid_label = (
        "Native registration-grid" if str(args.protocol) == "paper_hu_native" else "Cropped-1mm"
    )
    logj_mean, logj_std = mean_std([row.get("fg_log_jdet_std") for row in rows])
    jle0_mean, _ = mean_std([row.get("fg_jdet_le0_pct") for row in rows])
    be_mean, _ = mean_std([row.get("fg_bending_energy") for row in rows])
    diff_mean, _ = mean_std([row.get("fg_diffusion_l2") for row in rows])
    elapsed_mean, _ = mean_std([row.get("elapsed_sec") for row in rows])
    peak_mean, _ = mean_std([row.get("peak_mem_gb") for row in rows])

    mean_row = {
        "dataset": "copd",
        "method": runner.METHOD,
        "protocol": str(args.protocol),
        "protocol_tag": runner.protocol_tag(args.protocol),
        "display_name": f"ptvreg-python ADMM ({args.protocol})",
        "n_completed": len(rows),
        "requested_cases": ",".join(str(v) for v in requested_cases),
        "missing_cases": ",".join(str(v) for v in acceptance["missing_cases"]),
        "acceptance_passed": bool(acceptance["passed"]),
        "native_crop_snap_tre_mean_mm": native_snap_mean,
        "native_crop_snap_tre_std_across_cases_mm": native_snap_std,
        "native_crop_sub_tre_mean_mm": native_sub_mean,
        "native_crop_sub_tre_std_across_cases_mm": native_sub_std,
        "reg1mm_snap_tre_mean_mm": reg_snap_mean,
        "reg1mm_snap_tre_std_across_cases_mm": reg_snap_std,
        "reg1mm_sub_tre_mean_mm": reg_sub_mean,
        "reg1mm_sub_tre_std_across_cases_mm": reg_sub_std,
        "fg_log_jdet_std_mean": logj_mean,
        "fg_log_jdet_std_across_cases": logj_std,
        "fg_jdet_le0_pct_mean": jle0_mean,
        "fg_bending_energy_mean": be_mean,
        "fg_diffusion_l2_mean": diff_mean,
        "elapsed_sec_mean": elapsed_mean,
        "peak_mem_gb_mean": peak_mean,
    }
    runner.c.write_csv(review_root / f"{prefix}_mean_row.csv", [mean_row])
    runner.c.write_json(review_root / f"{prefix}_mean_row.json", mean_row)
    runner.c.write_json(review_root / f"{prefix}_acceptance.json", acceptance)

    case_table = markdown_table(
        [
            "Case",
            "Native Snap TRE",
            "Native Sub TRE",
            f"{registration_grid_label} Snap TRE",
            f"{registration_grid_label} Sub TRE",
            "SDlogJ",
            "J<=0 %",
            "Elapsed s",
        ],
        [
            [
                int(row["case_id"]),
                fmt(row.get("native_crop_snap_tre_mean_mm")),
                fmt(row.get("native_crop_sub_tre_mean_mm")),
                fmt(row.get("reg1mm_snap_tre_mean_mm")),
                fmt(row.get("reg1mm_sub_tre_mean_mm")),
                fmt(row.get("fg_log_jdet_std"), 4),
                fmt(row.get("fg_jdet_le0_pct"), 4),
                fmt(row.get("elapsed_sec"), 1),
            ]
            for row in rows
        ],
    )
    report = [
        f"# ptvreg-python COPD Fixed All10: {args.protocol}",
        "",
        f"Completed summaries: `{len(rows)}/{len(requested_cases)}`.",
        f"Acceptance passed: `{acceptance['passed']}`.",
        f"Run root: `{args.out_root}`.",
        "",
        "Headline metric: `native_crop_tre_snap_to_voxel` in mm.",
        "",
        "## Mean Row",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Native-crop snap TRE mean", f"{fmt(native_snap_mean)} +/- {fmt(native_snap_std)}"],
                ["Native-crop subvoxel TRE mean", f"{fmt(native_sub_mean)} +/- {fmt(native_sub_std)}"],
                [f"{registration_grid_label} snap TRE mean", f"{fmt(reg_snap_mean)} +/- {fmt(reg_snap_std)}"],
                [f"{registration_grid_label} subvoxel TRE mean", f"{fmt(reg_sub_mean)} +/- {fmt(reg_sub_std)}"],
                ["Mean SDlogJ", fmt(logj_mean, 4)],
                ["Mean J<=0 %", fmt(jle0_mean, 4)],
                ["Mean bending energy", fmt(be_mean, 6)],
                ["Mean diffusion L2", fmt(diff_mean, 6)],
                ["Mean elapsed sec", fmt(elapsed_mean, 1)],
                ["Mean peak GB", fmt(peak_mean, 2)],
            ],
        ),
        "",
        "## Per Case",
        "",
        case_table,
        "",
        "## Acceptance",
        "",
        markdown_table(
            ["Check", "Value"],
            [
                ["completed", f"{acceptance['n_completed']}/{acceptance['n_requested']}"],
                ["missing_cases", acceptance["missing_cases"]],
                ["invalid_items", acceptance["invalid_items"]],
            ],
        ),
        "",
        "## Files",
        "",
        f"- `{prefix}_case_rows.csv`",
        f"- `{prefix}_mean_row.csv`",
        f"- `{prefix}_case_rows.json`",
        f"- `{prefix}_mean_row.json`",
        f"- `{prefix}_acceptance.json`",
        "",
    ]
    (review_root / f"{prefix}_acceptance_report.md").write_text("\n".join(report))
    print(
        json.dumps(
            {
                "review_root": str(review_root),
                "protocol": str(args.protocol),
                "n_completed": len(rows),
                "acceptance_passed": acceptance["passed"],
                "missing": acceptance["missing_cases"],
                "native_crop_snap_tre_mean_mm": native_snap_mean,
                "native_crop_snap_tre_std_across_cases_mm": native_snap_std,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
