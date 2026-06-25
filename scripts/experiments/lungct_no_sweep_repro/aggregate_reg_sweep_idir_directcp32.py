#!/usr/bin/env python
"""Aggregate LungCT IDIR and DirectCP 32-16-8-4 regularization sweep."""

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
from launch_reg_sweep_idir_directcp32 import (
    BASE_DIRECTCP_ISOTV,
    BASE_IDIR_ALPHA,
    DEFAULT_SWEEP_ROOT,
    METHOD_DIRECTCP32,
    METHOD_IDIR,
    SWEEP_METHODS,
)


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--methods", type=str, default=",".join(SWEEP_METHODS))
    return parser.parse_args()


def method_label(method: str) -> str:
    if method == METHOD_IDIR:
        return "IDIR BE default"
    if method == METHOD_DIRECTCP32:
        return "DirectCP 32-16-8-4 pTV LCC + isoTV"
    return method


def reg_value_and_factor(method: str, payload: Dict[str, object]) -> tuple[float, float]:
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    if method == METHOD_IDIR:
        value = float(training.get("alpha_bending", BASE_IDIR_ALPHA))
        return value, value / BASE_IDIR_ALPHA
    if method == METHOD_DIRECTCP32:
        value = float(training.get("ptv_isotv_weight", BASE_DIRECTCP_ISOTV))
        return value, value / BASE_DIRECTCP_ISOTV
    raise ValueError(f"Unsupported sweep method: {method}")


def row_from_summary(path: Path, payload: Dict[str, object], source: str) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    method = str(row.get("method"))
    reg_value, factor = reg_value_and_factor(method, payload)
    return {
        "method": method,
        "method_label": method_label(method),
        "case_id": int(row.get("case_id") or payload.get("case_id")),
        "source": source,
        "reg_factor": float(factor),
        "reg_value": float(reg_value),
        "primary_original_snap_tre_mean_mm": row.get("native_crop_snap_tre_mean_mm"),
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


def collect_rows(root: Path, methods: Sequence[str], source: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in methods:
        for summary_path in sorted((root / method).glob("*/summary.json")):
            payload = json.loads(summary_path.read_text())
            if payload.get("status") != "completed":
                continue
            rows.append(row_from_summary(summary_path, payload, source))
    return rows


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def finite_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        val = base.safe_float(row.get(key))
        if math.isfinite(val):
            values.append(val)
    return values


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    groups = sorted(
        {(str(row["method"]), float(row["reg_factor"]), float(row["reg_value"])) for row in rows},
        key=lambda item: (item[0], item[1]),
    )
    metric_keys = [
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
    for method, factor, value in groups:
        group_rows = [
            row
            for row in rows
            if str(row["method"]) == method and abs(float(row["reg_factor"]) - factor) < 1e-10
        ]
        agg: Dict[str, object] = {
            "method": method,
            "method_label": method_label(method),
            "reg_factor": factor,
            "reg_value": value,
            "cases": len(group_rows),
        }
        for key in metric_keys:
            mean, std = base.mean_std([row.get(key) for row in group_rows])
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def write_plots(agg_rows: Sequence[Dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("fg_logj_std_mean", "FG logJ std", "fg_logj_std"),
        ("fg_jdet_le0_pct_mean", "FG J <= 0 %", "fg_jdet_le0_pct"),
        ("fg_bending_energy_mean", "FG bending energy", "fg_bending_energy"),
        ("fg_diffusion_l2_mean", "FG diffusion energy", "fg_diffusion_l2"),
    ]
    for method in SWEEP_METHODS:
        rows = [row for row in agg_rows if row.get("method") == method]
        for x_key, x_label, slug in metric_specs:
            xs = np.asarray([base.safe_float(row.get(x_key)) for row in rows], dtype=np.float64)
            ys = np.asarray([base.safe_float(row.get("original_snap_tre_mean_mm_mean")) for row in rows], dtype=np.float64)
            factors = [float(row["reg_factor"]) for row in rows]
            keep = np.isfinite(xs) & np.isfinite(ys)
            if not bool(np.any(keep)):
                continue
            fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
            ax.plot(xs[keep], ys[keep], marker="o", linewidth=1.2)
            for x, y, factor in zip(xs[keep], ys[keep], np.asarray(factors)[keep]):
                ax.annotate(f"{factor:g}x", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Original snap TRE mean (mm)")
            ax.set_title(method_label(method))
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{method}_original_snap_vs_{slug}.png")
            plt.close(fig)


def markdown_mean_table(agg_rows: Sequence[Dict[str, object]]) -> str:
    rows = []
    for row in agg_rows:
        rows.append(
            [
                row["method_label"],
                fmt(row["reg_factor"], 2),
                fmt(row["reg_value"], 6),
                row["cases"],
                f"{fmt(row['original_snap_tre_mean_mm_mean'])} +/- {fmt(row['original_snap_tre_mean_mm_std'])}",
                f"{fmt(row['original_subvoxel_tre_mean_mm_mean'])} +/- {fmt(row['original_subvoxel_tre_mean_mm_std'])}",
                f"{fmt(row['reg1mm_snap_tre_mean_mm_mean'])} +/- {fmt(row['reg1mm_snap_tre_mean_mm_std'])}",
                f"{fmt(row['reg1mm_subvoxel_tre_mean_mm_mean'])} +/- {fmt(row['reg1mm_subvoxel_tre_mean_mm_std'])}",
                fmt(row["fg_logj_std_mean"], 6),
                fmt(row["fg_jdet_le0_pct_mean"], 4),
                fmt(row["fg_bending_energy_mean"], 6),
                fmt(row["fg_diffusion_l2_mean"], 6),
            ]
        )
    return base.markdown_table(
        [
            "Method",
            "Factor",
            "Reg value",
            "Cases",
            "Original Snap",
            "Original Sub",
            "1mm Snap",
            "1mm Sub",
            "FG logJ std",
            "FG J<=0%",
            "BE fg",
            "Diff fg",
        ],
        rows,
    )


def best_rows(agg_rows: Sequence[Dict[str, object]]) -> List[List[object]]:
    out: List[List[object]] = []
    for method in SWEEP_METHODS:
        rows = [row for row in agg_rows if row.get("method") == method]
        rows = [row for row in rows if math.isfinite(base.safe_float(row.get("original_snap_tre_mean_mm_mean")))]
        if not rows:
            continue
        best = min(rows, key=lambda row: float(row["original_snap_tre_mean_mm_mean"]))
        out.append(
            [
                method_label(method),
                fmt(best["reg_factor"], 2),
                fmt(best["reg_value"], 6),
                fmt(best["original_snap_tre_mean_mm_mean"]),
                fmt(best["fg_logj_std_mean"], 6),
                fmt(best["fg_jdet_le0_pct_mean"], 4),
                fmt(best["fg_bending_energy_mean"], 6),
                fmt(best["fg_diffusion_l2_mean"], 6),
            ]
        )
    return out


def main() -> None:
    args = parse_args()
    methods = [m for m in args.methods.replace(" ", "").split(",") if m]
    unknown = sorted(set(methods) - set(SWEEP_METHODS))
    if unknown:
        raise ValueError(f"Unsupported sweep method(s): {unknown}")
    rows = collect_rows(args.baseline_root, methods, "baseline_1e0")
    rows.extend(collect_rows(args.sweep_root, methods, "sweep"))
    rows = sorted(rows, key=lambda row: (str(row["method"]), float(row["reg_factor"]), int(row["case_id"])))
    agg_rows = aggregate_rows(rows)

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "sweep_case_rows.csv", rows)
    c.write_json(args.review_root / "sweep_case_rows.json", rows)
    c.write_csv(args.review_root / "sweep_mean_rows.csv", agg_rows)
    c.write_json(args.review_root / "sweep_mean_rows.json", agg_rows)
    write_plots(agg_rows, args.review_root / "plots")

    counts = base.markdown_table(
        ["Method", "Factor", "Cases"],
        [[row["method_label"], fmt(row["reg_factor"], 2), row["cases"]] for row in agg_rows],
    )
    best = base.markdown_table(
        ["Method", "Best factor", "Reg value", "Original Snap", "FG logJ std", "FG J<=0%", "BE fg", "Diff fg"],
        best_rows(agg_rows),
    )
    report = "\n".join(
        [
            "# LungCT IDIR + DirectCP 32-16-8-4 Regularization Sweep",
            "",
            "Original Snap TRE is the primary comparison metric. Original means native/crop-space TRE after resizing the saved 1mm displacement field back to the corrected pTVReg native landmark crop. The three other TRE columns are retained as diagnostics.",
            "",
            "## Mean Rows",
            "",
            markdown_mean_table(agg_rows),
            "",
            "## Best Original Snap By Method",
            "",
            best,
            "",
            "## Case Counts",
            "",
            counts,
            "",
            "## Artifacts",
            "",
            "- Per-case rows: `sweep_case_rows.csv` and `sweep_case_rows.json`.",
            "- Mean rows: `sweep_mean_rows.csv` and `sweep_mean_rows.json`.",
            "- Pareto plots: `plots/*original_snap_vs*.png`.",
        ]
    )
    (args.review_root / "review_report.md").write_text(report + "\n")
    print(
        json.dumps(
            {
                "case_rows": len(rows),
                "mean_rows": len(agg_rows),
                "review_root": str(args.review_root),
                "report": str(args.review_root / "review_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
