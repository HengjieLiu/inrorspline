#!/usr/bin/env python
"""Aggregate LungCT DirectCP32 full-crop NCC + dense forward-difference BE sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aggregate_report as base
import common as c
from launch_directcp32_global_ncc_fdbe_sweep import DEFAULT_ALPHAS, DEFAULT_SWEEP_ROOT
from run_one import METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    return parser.parse_args()


def parse_alphas(value: str) -> List[float]:
    return [float(v) for v in value.replace(" ", "").split(",") if v]


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def row_from_summary(path: Path, payload: Dict[str, object]) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    return {
        "method": row.get("method"),
        "case_id": int(row.get("case_id") or payload.get("case_id")),
        "alpha": float(training.get("alpha", float("nan"))),
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
        "summary_path": str(path),
    }


def collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((root / METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        rows.append(row_from_summary(summary_path, payload))
    return sorted(rows, key=lambda row: (int(row["case_id"]), float(row["alpha"])))


def completion_errors(rows: Sequence[Dict[str, object]], expected_alphas: Sequence[float]) -> List[str]:
    errors: List[str] = []
    for case_id in (1, 8):
        got = sorted(float(row["alpha"]) for row in rows if int(row["case_id"]) == case_id)
        want = sorted(float(v) for v in expected_alphas)
        if len(got) != len(want) or any(abs(a - b) > 1e-9 for a, b in zip(got, want)):
            errors.append(f"case {case_id:02d}: got alphas {got}, expected {want}")
    return errors


def write_plots(rows: Sequence[Dict[str, object]], plot_dir: Path, alpha_order: Sequence[float]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for case_id in sorted({int(row["case_id"]) for row in rows}):
        case_rows = [row for row in rows if int(row["case_id"]) == case_id]
        by_alpha = {float(row["alpha"]): row for row in case_rows}
        ordered = [by_alpha[a] for a in alpha_order if a in by_alpha]
        if not ordered:
            continue

        y = np.asarray([base.safe_float(row["original_snap_tre_mean_mm"]) for row in ordered], dtype=np.float64)
        labels = [f"{float(row['alpha']):g}" for row in ordered]
        x = np.arange(len(ordered), dtype=np.float64)
        keep = np.isfinite(y)
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
        ax.plot(x[keep], y[keep], marker="o", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Original snap TRE mean (mm) ↓")
        ax.set_title(f"Case {case_id:02d}: DirectCP32 global NCC + fd-BE")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"case{case_id:02d}_original_snap_tre_vs_alpha.png")
        plt.close(fig)

        xs = np.asarray([base.safe_float(row["fg_logj_std"]) for row in ordered], dtype=np.float64)
        keep = np.isfinite(xs) & np.isfinite(y)
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
        ax.plot(xs[keep], y[keep], marker="o", linewidth=1.2)
        for row, xval, yval in zip(ordered, xs, y):
            if math.isfinite(float(xval)) and math.isfinite(float(yval)):
                ax.annotate(f"a={float(row['alpha']):g}", (xval, yval), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("FG stdLogJ ↓")
        ax.set_ylabel("Original snap TRE mean (mm) ↓")
        ax.set_title(f"Case {case_id:02d}: TRE vs regularity")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"case{case_id:02d}_original_snap_tre_vs_fg_logj_std.png")
        plt.close(fig)


def markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                f"{int(row['case_id']):02d}",
                fmt(row["alpha"], 3),
                f"{fmt(row['original_snap_tre_mean_mm'])} +/- {fmt(row['original_snap_tre_std_mm'])}",
                f"{fmt(row['original_subvoxel_tre_mean_mm'])} +/- {fmt(row['original_subvoxel_tre_std_mm'])}",
                f"{fmt(row['reg1mm_snap_tre_mean_mm'])} +/- {fmt(row['reg1mm_snap_tre_std_mm'])}",
                f"{fmt(row['reg1mm_subvoxel_tre_mean_mm'])} +/- {fmt(row['reg1mm_subvoxel_tre_std_mm'])}",
                fmt(row["fg_logj_std"], 6),
                fmt(row["fg_jdet_le0_pct"], 4),
                fmt(row["fg_bending_energy"], 6),
                fmt(row["fg_diffusion_l2"], 6),
                fmt(row["fg_disp_mag_mean_vox"], 3),
                fmt(row["fg_disp_mag_p95_vox"], 3),
                fmt(row["fg_disp_mag_max_vox"], 3),
                fmt(row["elapsed_sec"], 1),
                fmt(row["peak_mem_gb"], 2),
            ]
        )
    return base.markdown_table(
        [
            "Case",
            "Alpha",
            "Original snap TRE",
            "Original sub TRE",
            "Reg1mm snap TRE",
            "Reg1mm sub TRE",
            "FG stdLogJ",
            "FG J<=0 %",
            "FG BE fd",
            "FG diffusion fd",
            "Disp mean",
            "Disp p95",
            "Disp max",
            "Elapsed sec",
            "Peak GB",
        ],
        table_rows,
    )


def best_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for case_id in sorted({int(row["case_id"]) for row in rows}):
        case_rows = [
            row
            for row in rows
            if int(row["case_id"]) == case_id and math.isfinite(base.safe_float(row["original_snap_tre_mean_mm"]))
        ]
        if not case_rows:
            continue
        best = min(case_rows, key=lambda row: float(row["original_snap_tre_mean_mm"]))
        lines.append(
            "- Case "
            f"{case_id:02d}: best original snap TRE at alpha `{float(best['alpha']):g}` = "
            f"`{fmt(best['original_snap_tre_mean_mm'])} mm`, FG stdLogJ `{fmt(best['fg_logj_std'], 6)}`, "
            f"FG J<=0 `{fmt(best['fg_jdet_le0_pct'], 4)}%`."
        )
    return lines


def main() -> None:
    args = parse_args()
    expected_alphas = parse_alphas(args.alphas)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.sweep_root)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    write_plots(rows, args.review_root / "plots", expected_alphas)
    errors = completion_errors(rows, expected_alphas)
    lines = [
        "# LungCT DirectCP32 Global NCC + Dense fd-BE Sweep",
        "",
        "## Summary",
        "",
        f"- Method: `{METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE}`.",
        "- Loss: full-crop global NCC plus `alpha *` dense forward-difference BE.",
        "- Cases: `01` and `08`; cropped 1mm pTV-prep inputs.",
        "- Primary metric: `original_snap` TRE; all four TRE variants are retained.",
        "- Foreground regularity uses the fixed T00 lung mask on the cropped 1mm grid.",
        "",
        "## Completion",
        "",
        f"- Completed rows: `{len(rows)}`.",
        f"- Expected rows: `{2 * len(expected_alphas)}`.",
    ]
    if errors:
        lines.append("- Completion errors:")
        lines.extend(f"  - {err}" for err in errors)
    else:
        lines.append("- Completion check: passed.")
    lines.extend(["", "## Best Per Case", ""])
    lines.extend(best_lines(rows) or ["- No finite TRE rows found."])
    lines.extend(["", "## Case Rows", "", markdown_table(rows), "", "## Artifacts", ""])
    lines.extend(
        [
            "- CSV: `case_rows.csv`.",
            "- JSON: `case_rows.json`.",
            "- Plots: `plots/case*_original_snap_tre_vs_alpha.png` and `plots/case*_original_snap_tre_vs_fg_logj_std.png`.",
        ]
    )
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "errors": errors, "review_root": str(args.review_root)}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
