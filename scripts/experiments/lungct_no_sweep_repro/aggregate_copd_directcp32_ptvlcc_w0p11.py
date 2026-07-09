#!/usr/bin/env python
"""Aggregate COPD DirectCP32 pTV-LCC + isoTV sanity summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import common as c
from run_one import METHOD_DIRECT_CP_32_PTV


DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/copd_directcp32_ptvlcc_w0p11_sanity_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--case-ids", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--title", type=str, default="COPD DirectCP32 pTV-LCC + isoTV Sanity Run")
    parser.add_argument("--display-name", type=str, default="MR-D-BSCP (pTVReg loss)")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def metric(payload: Dict[str, object], key: str) -> Dict[str, object]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


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
    reg = metric(payload, "regularity_fg_registration_crop")
    training = metric(payload, "training")
    return {
        "case_id": int(payload.get("case_id", -1)),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "run_name": payload.get("run_name"),
        "status": payload.get("status"),
        "summary_path": str(path),
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
        "fg_log_jdet_std": reg.get("log_jdet_std_fg"),
        "fg_jdet_le0_pct": reg.get("jdet_le0_pct_fg"),
        "fg_bending_energy": reg.get("bending_energy_fg"),
        "fg_diffusion_l2": reg.get("diffusion_l2_fg"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "peak_mem_gb": payload.get("peak_mem_gb"),
        "seed": payload.get("seed"),
        "stage_epochs": ",".join(str(v) for v in training.get("stage_epochs", []))
        if isinstance(training.get("stage_epochs"), list)
        else "",
        "stage_full_cps": ",".join(str(v) for v in training.get("stage_full_cps", []))
        if isinstance(training.get("stage_full_cps"), list)
        else "",
        "stage_downsample": ",".join(str(v) for v in training.get("stage_downsample", []))
        if isinstance(training.get("stage_downsample"), list)
        else "",
        "lr": training.get("stage_lr", [None])[0]
        if isinstance(training.get("stage_lr"), list) and training.get("stage_lr")
        else training.get("lr"),
        "ptv_lcc_sigma_mm": training.get("ptv_lcc_sigma_mm"),
        "ptv_isotv_weight": training.get("ptv_isotv_weight"),
        "ptv_csqrt": training.get("ptv_csqrt"),
    }


def collect_rows(out_root: Path, requested_cases: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((out_root / METHOD_DIRECT_CP_32_PTV).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        if payload.get("dataset") != "copd":
            continue
        case_id = int(payload.get("case_id", -1))
        if case_id not in requested_cases:
            continue
        rows.append(flatten_summary(summary_path, payload))
    rows.sort(key=lambda row: int(row["case_id"]))
    return rows


def main() -> None:
    args = parse_args()
    requested_cases = c.parse_int_list(args.case_ids)
    rows = collect_rows(args.out_root, requested_cases)
    present = {int(row["case_id"]) for row in rows}
    missing = [case_id for case_id in requested_cases if case_id not in present]
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing completed COPD summaries for case(s): {missing}")

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "copd_case_rows.csv", rows)

    native_snap_mean, native_snap_std = mean_std([row.get("native_crop_snap_tre_mean_mm") for row in rows])
    native_sub_mean, native_sub_std = mean_std([row.get("native_crop_sub_tre_mean_mm") for row in rows])
    reg_snap_mean, reg_snap_std = mean_std([row.get("reg1mm_snap_tre_mean_mm") for row in rows])
    reg_sub_mean, reg_sub_std = mean_std([row.get("reg1mm_sub_tre_mean_mm") for row in rows])
    logj_mean, logj_std = mean_std([row.get("fg_log_jdet_std") for row in rows])
    jle0_mean, _ = mean_std([row.get("fg_jdet_le0_pct") for row in rows])
    be_mean, _ = mean_std([row.get("fg_bending_energy") for row in rows])
    diff_mean, _ = mean_std([row.get("fg_diffusion_l2") for row in rows])
    elapsed_mean, _ = mean_std([row.get("elapsed_sec") for row in rows])
    peak_mean, _ = mean_std([row.get("peak_mem_gb") for row in rows])

    mean_row = {
        "dataset": "copd",
        "method": METHOD_DIRECT_CP_32_PTV,
        "display_name": args.display_name,
        "n_completed": len(rows),
        "requested_cases": ",".join(str(v) for v in requested_cases),
        "missing_cases": ",".join(str(v) for v in missing),
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
    c.write_csv(args.review_root / "copd_mean_row.csv", [mean_row])
    c.write_json(args.review_root / "copd_case_rows.json", rows)
    c.write_json(args.review_root / "copd_mean_row.json", mean_row)

    case_table = markdown_table(
        [
            "Case",
            "Orig Snap TRE",
            "Orig Sub TRE",
            "Reg1mm Snap TRE",
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
                fmt(row.get("fg_log_jdet_std"), 4),
                fmt(row.get("fg_jdet_le0_pct"), 4),
                fmt(row.get("elapsed_sec"), 1),
            ]
            for row in rows
        ],
    )
    report = [
        f"# {args.title}",
        "",
        f"Display name: `{args.display_name}`.",
        "",
        f"Completed summaries: `{len(rows)}/{len(requested_cases)}`.",
        f"Run root: `{args.out_root}`.",
        "",
        "Headline metric: native-crop/original-domain snap-to-voxel TRE in mm.",
        "",
        "## Mean Row",
        "",
        markdown_table(
            [
                "Metric",
                "Value",
            ],
            [
                ["Orig snap TRE mean", f"{fmt(native_snap_mean)} +/- {fmt(native_snap_std)}"],
                ["Orig sub TRE mean", f"{fmt(native_sub_mean)} +/- {fmt(native_sub_std)}"],
                ["Reg1mm snap TRE mean", f"{fmt(reg_snap_mean)} +/- {fmt(reg_snap_std)}"],
                ["Reg1mm sub TRE mean", f"{fmt(reg_sub_mean)} +/- {fmt(reg_sub_std)}"],
                ["Mean SDlogJ", fmt(logj_mean, 4)],
                ["Mean J<=0 %", fmt(jle0_mean, 4)],
                ["Mean elapsed sec", fmt(elapsed_mean, 1)],
            ],
        ),
        "",
        "## Per Case",
        "",
        case_table,
        "",
        "## Files",
        "",
        "- `copd_case_rows.csv`",
        "- `copd_mean_row.csv`",
        "- `copd_case_rows.json`",
        "- `copd_mean_row.json`",
        "",
    ]
    if missing:
        report.insert(6, f"Missing cases: `{missing}`.")
        report.insert(7, "")
    (args.review_root / "review_report.md").write_text("\n".join(report))
    print(json.dumps({"review_root": str(args.review_root), "n_completed": len(rows), "missing": missing}, indent=2))
    print(f"Native crop snap TRE mean: {fmt(native_snap_mean)} +/- {fmt(native_snap_std)} mm")


if __name__ == "__main__":
    main()
