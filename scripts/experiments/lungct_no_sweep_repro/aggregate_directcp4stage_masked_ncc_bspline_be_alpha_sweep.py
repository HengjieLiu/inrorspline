#!/usr/bin/env python
"""Aggregate LungCT DirectCP fixed-mask NCC + analytic BE alpha sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import aggregate_report as base
import common as c
from launch_directcp4stage_masked_ncc_bspline_be_alpha_sweep import DEFAULT_RUN_ROOT
from run_one import METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep"
DEFAULT_ANCHOR_ROOT = Path("/database/inrorspline/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10_v1")
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


def parse_alpha_list(value: str) -> List[float]:
    vals = [float(v) for v in value.replace(" ", "").split(",") if v]
    if not vals:
        raise ValueError(f"Expected comma-separated alphas, got {value!r}")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--fdbe-mean-rows", type=Path, default=DEFAULT_FDBE_MEAN_ROWS)
    parser.add_argument("--alphas", type=str, default="100,1000,10000,100000,1000000")
    parser.add_argument("--no-anchor", action="store_true")
    return parser.parse_args()


def row_from_summary(path: Path, payload: Dict[str, object], source: str) -> Dict[str, object]:
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
        "source": source,
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


def collect_rows(root: Path, source: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((root / METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        rows.append(row_from_summary(summary_path, payload, source))
    return rows


def aggregate_rows(alpha: float, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "method": METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
        "alpha": float(alpha),
        "cases": len(rows),
        "sources": ",".join(sorted({str(row.get("source", "")) for row in rows})),
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


def next_scale_hint(mean_rows: Sequence[Dict[str, object]]) -> str:
    full = [row for row in mean_rows if int(row.get("cases", 0)) == len(EXPECTED_CASES)]
    if not full:
        return "No full 10-case alpha point is available yet."
    best = min(full, key=lambda row: float(row["original_snap_tre_mean_mm_mean"]))
    alphas = sorted(float(row["alpha"]) for row in full)
    best_alpha = float(best["alpha"])
    if best_alpha == alphas[0]:
        return f"Best completed alpha is the low endpoint `{best_alpha:g}`; expand downward or test `/3` and `x3` around it."
    if best_alpha == alphas[-1]:
        return f"Best completed alpha is the high endpoint `{best_alpha:g}`; expand upward or test `/3` and `x3` around it."
    return f"Best completed alpha is `{best_alpha:g}`; next local checks should be about `{best_alpha / 3:g}` and `{best_alpha * 3:g}`."


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    requested_alphas = parse_alpha_list(args.alphas)
    rows = collect_rows(args.run_root, "sweep")
    if not args.no_anchor:
        rows.extend(collect_rows(args.anchor_root, "anchor_alpha10000"))
    rows = sorted(rows, key=lambda row: (float(row["alpha"]), int(row["case_id"]), str(row["source"])))
    rows = [row for row in rows if any(abs(float(row["alpha"]) - alpha) < 1e-9 for alpha in requested_alphas)]

    dedup: Dict[Tuple[float, int], Dict[str, object]] = {}
    for row in rows:
        key = (float(row["alpha"]), int(row["case_id"]))
        if key not in dedup or str(row.get("source")) == "sweep":
            dedup[key] = row
    case_rows = [dedup[key] for key in sorted(dedup)]
    c.write_csv(args.review_root / "case_rows.csv", case_rows)
    c.write_json(args.review_root / "case_rows.json", case_rows)

    errors: List[str] = []
    mean_rows: List[Dict[str, object]] = []
    for alpha in requested_alphas:
        alpha_rows = sorted(
            [row for row in case_rows if abs(float(row["alpha"]) - alpha) < 1e-9],
            key=lambda row: int(row["case_id"]),
        )
        got_cases = sorted(int(row["case_id"]) for row in alpha_rows)
        if got_cases != EXPECTED_CASES:
            errors.append(f"alpha {alpha:g}: got cases {got_cases}, expected {EXPECTED_CASES}")
        if alpha_rows:
            mean_rows.append(aggregate_rows(alpha, alpha_rows))
    c.write_csv(args.review_root / "mean_rows.csv", mean_rows)
    c.write_json(args.review_root / "mean_rows.json", mean_rows)

    fdbe = read_fdbe_best(args.fdbe_mean_rows)
    comparison_rows: List[Dict[str, object]] = [
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
        }
    ]
    for row in mean_rows:
        comparison_rows.append(
            {
                "label": "Corrected MR-D-BSCP fixed-mask NCC + analytic B-spline BE",
                "method": METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
                "alpha": row.get("alpha"),
                "cases": row.get("cases"),
                "original_snap_tre_mean_mm": row.get("original_snap_tre_mean_mm_mean"),
                "original_snap_tre_std_mm": row.get("original_snap_tre_mean_mm_std"),
                "fg_logj_std_mean": row.get("fg_logj_std_mean"),
                "fg_logj_std_std": row.get("fg_logj_std_std"),
                "fg_jdet_le0_pct_mean": row.get("fg_jdet_le0_pct_mean"),
                "fg_bending_energy_mean": row.get("fg_bending_energy_mean"),
                "fg_diffusion_l2_mean": row.get("fg_diffusion_l2_mean"),
            }
        )
    c.write_csv(args.review_root / "comparison_rows.csv", comparison_rows)
    c.write_json(args.review_root / "comparison_rows.json", comparison_rows)

    full_rows = [row for row in mean_rows if int(row.get("cases", 0)) == len(EXPECTED_CASES)]
    best = min(full_rows, key=lambda row: float(row["original_snap_tre_mean_mm_mean"])) if full_rows else None
    mean_table = []
    for row in mean_rows:
        mean_table.append(
            [
                fmt(row["alpha"], 6),
                row["cases"],
                f"{fmt(row.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(row.get('original_snap_tre_mean_mm_std'))}",
                f"{fmt(row.get('fg_logj_std_mean'), 6)} +/- {fmt(row.get('fg_logj_std_std'), 6)}",
                fmt(row.get("fg_jdet_le0_pct_mean"), 4),
                fmt(row.get("fg_bending_energy_mean"), 6),
                fmt(row.get("fg_diffusion_l2_mean"), 6),
                row.get("sources", ""),
            ]
        )

    lines = [
        "# LungCT Corrected MR-D-BSCP IDIR-Loss Alpha Sweep",
        "",
        "## Protocol",
        "",
        f"- Method: `{METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE}`.",
        "- Schedule: `32,16,8,4`; downsample `8,4,2,1`; stage epochs `200,400,800,1000`.",
        "- LR `1e-3`; seed `1`; sampled batch size `10000`.",
        "- Loss: fixed T00 lung-mask sampled global NCC plus sampled analytic cubic B-spline BE.",
        "- Inputs: canonical pTVReg-style 1mm landmark crop, not lung-masked images.",
        "- Metrics: original/native-crop snap TRE and fixed T00 lung-mask foreground regularity.",
        f"- Sweep alphas: `{', '.join(fmt(v, 6) for v in requested_alphas)}`.",
        "- The alpha `10000` row is imported from the completed anchor run unless `--no-anchor` is used.",
        "",
        "## Completion",
        "",
        f"- Completed case rows: `{len(case_rows)}`.",
        f"- Completion errors: `{errors}`.",
        "",
        "## Mean Rows",
        "",
        markdown_table(
            ["Alpha", "Cases", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0 %", "FG BE", "FG diffusion", "Sources"],
            mean_table,
        ),
        "",
        "## Current Reference",
        "",
        markdown_table(
            ["Row", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0 %", "FG BE", "FG diffusion"],
            [
                [
                    "Current full-crop NCC + dense FD-BE, alpha 10000",
                    f"{fmt(fdbe.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(fdbe.get('original_snap_tre_mean_mm_std'))}",
                    f"{fmt(fdbe.get('fg_logj_std_mean'), 6)} +/- {fmt(fdbe.get('fg_logj_std_std'), 6)}",
                    fmt(fdbe.get("fg_jdet_le0_pct_mean"), 4),
                    fmt(fdbe.get("fg_bending_energy_mean"), 6),
                    fmt(fdbe.get("fg_diffusion_l2_mean"), 6),
                ]
            ],
        ),
        "",
        "## Provisional Decision",
        "",
        f"- Best corrected full point by TRE: alpha `{fmt(best['alpha'], 6)}` with TRE `{fmt(best['original_snap_tre_mean_mm_mean'])}` and FG stdLogJ `{fmt(best['fg_logj_std_mean'], 6)}`." if best else "- No complete corrected point yet.",
        f"- {next_scale_hint(mean_rows)}",
        "",
        "## Artifacts",
        "",
        "- `case_rows.csv`, `case_rows.json`",
        "- `mean_rows.csv`, `mean_rows.json`",
        "- `comparison_rows.csv`, `comparison_rows.json`",
    ]
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "run_root": str(args.run_root),
        "anchor_root": None if args.no_anchor else str(args.anchor_root),
        "review_root": str(args.review_root),
        "fdbe_mean_rows": str(args.fdbe_mean_rows),
        "requested_alphas": requested_alphas,
        "errors": errors,
    }
    c.write_json(args.review_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "case_rows": len(case_rows),
                "mean_rows": len(mean_rows),
                "errors": errors,
                "best_alpha": None if best is None else best.get("alpha"),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
