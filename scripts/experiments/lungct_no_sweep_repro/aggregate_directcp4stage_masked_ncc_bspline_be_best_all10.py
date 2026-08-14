#!/usr/bin/env python
"""Aggregate corrected LungCT MR-D-BSCP IDIR-loss all-case test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import aggregate_report as base
import common as c
from launch_directcp4stage_masked_ncc_bspline_be_best_all10 import DEFAULT_RUN_ROOT
from run_one import METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10"
DEFAULT_FDBE_MEAN_ROWS = (
    c.REPO_ROOT
    / "experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/fdbe_mean_rows.csv"
)
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
    "fg_disp_mag_mean_vox",
    "fg_disp_mag_p95_vox",
    "fg_disp_mag_max_vox",
    "elapsed_sec",
    "peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--fdbe-mean-rows", type=Path, default=DEFAULT_FDBE_MEAN_ROWS)
    parser.add_argument("--alpha", type=float, default=10000.0)
    return parser.parse_args()


def row_from_summary(path: Path, payload: Dict[str, object]) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    stage_summaries = payload.get("stage_summaries", [])
    stage_train_coords = ""
    stage_batch_sizes = ""
    if isinstance(stage_summaries, list):
        stage_train_coords = ",".join(str(int(s.get("masked_train_coords", 0))) for s in stage_summaries if isinstance(s, dict))
        stage_batch_sizes = ",".join(str(int(s.get("masked_batch_size", 0))) for s in stage_summaries if isinstance(s, dict))
    out: Dict[str, object] = {
        "method": METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
        "case_id": int(row.get("case_id") or payload.get("case_id")),
        "alpha": float(training.get("alpha", float("nan"))),
        "summary_path": str(path),
        "loss": training.get("loss", ""),
        "image_loss": training.get("image_loss", ""),
        "regularizer": training.get("regularizer", ""),
        "sample_domain": training.get("sample_domain", ""),
        "stage_masked_train_coords": stage_train_coords,
        "stage_masked_batch_sizes": stage_batch_sizes,
    }
    out.update(
        {
            "original_snap_tre_mean_mm": row.get("native_crop_snap_tre_mean_mm"),
            "original_snap_tre_std_mm": row.get("native_crop_snap_tre_std_mm"),
            "original_snap_tre_median_mm": row.get("native_crop_snap_tre_median_mm"),
            "original_subvoxel_tre_mean_mm": row.get("native_crop_sub_tre_mean_mm"),
            "original_subvoxel_tre_std_mm": row.get("native_crop_sub_tre_std_mm"),
            "original_subvoxel_tre_median_mm": row.get("native_crop_sub_tre_median_mm"),
            "reg1mm_snap_tre_mean_mm": row.get("reg1mm_snap_tre_mean_mm"),
            "reg1mm_snap_tre_std_mm": row.get("reg1mm_snap_tre_std_mm"),
            "reg1mm_snap_tre_median_mm": row.get("reg1mm_snap_tre_median_mm"),
            "reg1mm_subvoxel_tre_mean_mm": row.get("reg1mm_sub_tre_mean_mm"),
            "reg1mm_subvoxel_tre_std_mm": row.get("reg1mm_sub_tre_std_mm"),
            "reg1mm_subvoxel_tre_median_mm": row.get("reg1mm_sub_tre_median_mm"),
            "fg_logj_std": row.get("fg_log_jdet_std"),
            "fg_jdet_le0_pct": row.get("fg_jdet_le0_pct"),
            "fg_bending_energy": row.get("fg_bending_energy"),
            "fg_diffusion_l2": row.get("fg_diffusion_l2"),
            "fg_disp_mag_mean_vox": row.get("fg_disp_mag_mean_vox"),
            "fg_disp_mag_p95_vox": row.get("fg_disp_mag_p95_vox"),
            "fg_disp_mag_max_vox": row.get("fg_disp_mag_max_vox"),
            "elapsed_sec": row.get("elapsed_sec"),
            "peak_mem_gb": row.get("peak_mem_gb"),
        }
    )
    analytic = payload.get("analytic_bspline_regularity_fg_registration_crop", {})
    if isinstance(analytic, dict):
        out["be_bspline_analytic_fg_norm"] = analytic.get("be_bspline_analytic_fg_norm")
        out["diffusion_bspline_analytic_fg_norm"] = analytic.get("diffusion_bspline_analytic_fg_norm")
    return out


def collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((root / METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        rows.append(row_from_summary(summary_path, payload))
    return sorted(rows, key=lambda row: int(row["case_id"]))


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "method": METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
        "alpha": float(rows[0]["alpha"]) if rows else "",
        "cases": len(rows),
    }
    for key in METRIC_KEYS + ["be_bspline_analytic_fg_norm", "diffusion_bspline_analytic_fg_norm"]:
        mean, std = base.mean_std([row.get(key) for row in rows])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def read_fdbe_best(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [row for row in rows if abs(float(row.get("reg_value", "nan")) - 10000.0) < 1e-9]
    if not candidates:
        raise RuntimeError(f"Could not find FD-BE alpha=10000 row in {path}")
    return candidates[0]


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return base.markdown_table(list(headers), [[str(v) for v in row] for row in rows])


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.run_root)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    mean_row = aggregate_rows(rows)
    c.write_csv(args.review_root / "mean_row.csv", [mean_row])
    c.write_json(args.review_root / "mean_row.json", mean_row)

    errors: List[str] = []
    got_cases = sorted(int(row["case_id"]) for row in rows)
    if got_cases != EXPECTED_CASES:
        errors.append(f"got cases {got_cases}, expected {EXPECTED_CASES}")
    for row in rows:
        if abs(float(row["alpha"]) - float(args.alpha)) > 1e-9:
            errors.append(f"case {int(row['case_id']):02d}: alpha {row['alpha']} != {args.alpha}")

    fdbe = read_fdbe_best(args.fdbe_mean_rows)
    comparison_rows = [
        {
            "label": "Current MR-D-BSCP full-crop NCC + dense FD-BE",
            "method": "direct_cp_32_16_8_4_global_ncc_fd_be",
            "alpha": 10000.0,
            "cases": fdbe.get("cases", "10"),
            "original_snap_tre_mean_mm": fdbe.get("original_snap_tre_mean_mm_mean"),
            "original_snap_tre_std_mm": fdbe.get("original_snap_tre_mean_mm_std"),
            "fg_logj_std_mean": fdbe.get("fg_logj_std_mean"),
            "fg_logj_std_std": fdbe.get("fg_logj_std_std"),
            "fg_jdet_le0_pct_mean": fdbe.get("fg_jdet_le0_pct_mean"),
            "fg_bending_energy_mean": fdbe.get("fg_bending_energy_mean"),
            "fg_diffusion_l2_mean": fdbe.get("fg_diffusion_l2_mean"),
        },
        {
            "label": "Corrected MR-D-BSCP fixed-mask NCC + analytic B-spline BE",
            "method": METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
            "alpha": mean_row.get("alpha"),
            "cases": mean_row.get("cases"),
            "original_snap_tre_mean_mm": mean_row.get("original_snap_tre_mean_mm_mean"),
            "original_snap_tre_std_mm": mean_row.get("original_snap_tre_mean_mm_std"),
            "fg_logj_std_mean": mean_row.get("fg_logj_std_mean"),
            "fg_logj_std_std": mean_row.get("fg_logj_std_std"),
            "fg_jdet_le0_pct_mean": mean_row.get("fg_jdet_le0_pct_mean"),
            "fg_bending_energy_mean": mean_row.get("fg_bending_energy_mean"),
            "fg_diffusion_l2_mean": mean_row.get("fg_diffusion_l2_mean"),
        },
    ]
    c.write_csv(args.review_root / "comparison_rows.csv", comparison_rows)
    c.write_json(args.review_root / "comparison_rows.json", comparison_rows)

    case_table = []
    for row in rows:
        case_table.append(
            [
                f"{int(row['case_id']):02d}",
                fmt(row["original_snap_tre_mean_mm"]),
                fmt(row["fg_logj_std"], 6),
                fmt(row["fg_jdet_le0_pct"], 4),
                fmt(row["elapsed_sec"], 1),
                fmt(row["peak_mem_gb"], 2),
            ]
        )

    lines = [
        "# LungCT Corrected MR-D-BSCP IDIR-Loss Test",
        "",
        "## Protocol",
        "",
        f"- Method: `{METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE}`.",
        "- Schedule: `32,16,8,4`; downsample `8,4,2,1`; stage epochs `200,400,800,1000`.",
        f"- Alpha: `{float(args.alpha):g}`; LR `1e-3`; seed `1`; sampled batch size `10000`.",
        "- Loss: fixed T00 lung-mask sampled global NCC plus sampled analytic cubic B-spline BE.",
        "- Inputs: canonical pTVReg-style 1mm landmark crop, not lung-masked images.",
        "- Metrics: original/native-crop snap TRE and fixed T00 lung-mask foreground regularity.",
        "",
        "## Completion",
        "",
        f"- Completed rows: `{len(rows)}`.",
        f"- Completion errors: `{errors}`.",
        "",
        "## Mean Comparison",
        "",
        markdown_table(
            ["Row", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0 %", "FG BE", "FG diffusion"],
            [
                [
                    "Current FD-BE",
                    f"{fmt(fdbe.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(fdbe.get('original_snap_tre_mean_mm_std'))}",
                    f"{fmt(fdbe.get('fg_logj_std_mean'), 6)} +/- {fmt(fdbe.get('fg_logj_std_std'), 6)}",
                    fmt(fdbe.get("fg_jdet_le0_pct_mean"), 4),
                    fmt(fdbe.get("fg_bending_energy_mean"), 6),
                    fmt(fdbe.get("fg_diffusion_l2_mean"), 6),
                ],
                [
                    "Corrected analytic-mask",
                    f"{fmt(mean_row.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(mean_row.get('original_snap_tre_mean_mm_std'))}",
                    f"{fmt(mean_row.get('fg_logj_std_mean'), 6)} +/- {fmt(mean_row.get('fg_logj_std_std'), 6)}",
                    fmt(mean_row.get("fg_jdet_le0_pct_mean"), 4),
                    fmt(mean_row.get("fg_bending_energy_mean"), 6),
                    fmt(mean_row.get("fg_diffusion_l2_mean"), 6),
                ],
            ],
        ),
        "",
        "## Corrected Case Rows",
        "",
        markdown_table(["Case", "TRE", "FG stdLogJ", "FG J<=0 %", "Elapsed sec", "Peak GB"], case_table),
        "",
        "## Artifacts",
        "",
        "- `case_rows.csv`, `case_rows.json`",
        "- `mean_row.csv`, `mean_row.json`",
        "- `comparison_rows.csv`, `comparison_rows.json`",
    ]
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "run_root": str(args.run_root),
        "review_root": str(args.review_root),
        "fdbe_mean_rows": str(args.fdbe_mean_rows),
        "errors": errors,
    }
    c.write_json(args.review_root / "manifest.json", manifest)
    print(json.dumps({"rows": len(rows), "errors": errors, "review_root": str(args.review_root)}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
