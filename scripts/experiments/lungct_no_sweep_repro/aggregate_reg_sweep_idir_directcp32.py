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
from matplotlib.patches import Ellipse
import numpy as np

import aggregate_report as base
import common as c
from launch_reg_sweep_idir_directcp32 import (
    BASE_DIRECTCP_ISOTV,
    BASE_IDIR_ALPHA,
    DEFAULT_STAGE_EPOCHS,
    DEFAULT_SWEEP_ROOT,
    METHOD_DIRECTCP32,
    METHOD_IDIR,
    SWEEP_METHODS,
)


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"
PLOT_METRIC_SPECS = [
    ("fg_logj_std_mean", "FG logJ std", "fg_logj_std"),
    ("fg_jdet_le0_pct_mean", "FG J <= 0 %", "fg_jdet_le0_pct"),
    ("fg_bending_energy_mean", "FG bending energy", "fg_bending_energy"),
    ("fg_diffusion_l2_mean", "FG diffusion energy", "fg_diffusion_l2"),
]
EXPECTED_FACTORS = [0.01, 0.1, 1.0, 10.0, 100.0]


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


def is_expected_factor(row: Dict[str, object]) -> bool:
    factor = float(row["reg_factor"])
    return any(abs(factor - expected) < 1e-10 for expected in EXPECTED_FACTORS)


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def fmt_min(value_sec: object, digits: int = 2) -> str:
    value = base.safe_float(value_sec)
    if not math.isfinite(value):
        return "NA"
    return f"{value / 60.0:.{digits}f}"


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


def epoch_config_table() -> str:
    return base.markdown_table(
        ["Method", "Epoch setting", "LR", "Schedule", "Default regularization"],
        [
            [
                method_label(METHOD_IDIR),
                "2500 epochs",
                "1e-4",
                "single stage",
                f"alpha_bending={BASE_IDIR_ALPHA:g}",
            ],
            [
                method_label(METHOD_DIRECTCP32),
                f"stage epochs {DEFAULT_STAGE_EPOCHS} (total {sum(c.parse_stage_ints(DEFAULT_STAGE_EPOCHS, 4))})",
                "1e-3",
                "CPS 32,16,8,4; downsample 8,4,2,1",
                f"ptv_isotv_weight={BASE_DIRECTCP_ISOTV:g}",
            ],
        ],
    )


def baseline_cost_table(rows: Sequence[Dict[str, object]]) -> str:
    table_rows: List[List[object]] = []
    for case_id in range(1, 11):
        row_out: List[object] = [f"{case_id:02d}"]
        for method in SWEEP_METHODS:
            match = [
                row
                for row in rows
                if str(row["method"]) == method
                and int(row["case_id"]) == case_id
                and abs(float(row["reg_factor"]) - 1.0) < 1e-10
            ]
            if not match:
                row_out.extend(["NA", "NA"])
                continue
            row = match[0]
            row_out.extend([fmt_min(row["elapsed_sec"]), fmt(row["peak_mem_gb"], 2)])
        table_rows.append(row_out)
    return base.markdown_table(
        ["Case", "IDIR time min", "IDIR peak GB", "DirectCP time min", "DirectCP peak GB"],
        table_rows,
    )


def method_cost_summary_table(rows: Sequence[Dict[str, object]], baseline_only: bool) -> str:
    table_rows: List[List[object]] = []
    for method in SWEEP_METHODS:
        method_rows = [row for row in rows if str(row["method"]) == method]
        if baseline_only:
            method_rows = [row for row in method_rows if abs(float(row["reg_factor"]) - 1.0) < 1e-10]
        elapsed_values = [row.get("elapsed_sec") for row in method_rows]
        memory_values = [row.get("peak_mem_gb") for row in method_rows]
        elapsed_mean, elapsed_std = base.mean_std(elapsed_values)
        memory_mean, memory_std = base.mean_std(memory_values)
        elapsed_finite = finite_values(method_rows, "elapsed_sec")
        memory_finite = finite_values(method_rows, "peak_mem_gb")
        table_rows.append(
            [
                method_label(method),
                len(method_rows),
                f"{fmt_min(elapsed_mean)} +/- {fmt_min(elapsed_std)}",
                f"{fmt_min(min(elapsed_finite))}-{fmt_min(max(elapsed_finite))}" if elapsed_finite else "NA",
                f"{fmt(memory_mean, 2)} +/- {fmt(memory_std, 2)}",
                f"{fmt(min(memory_finite), 2)}-{fmt(max(memory_finite), 2)}" if memory_finite else "NA",
            ]
        )
    scope = "default 1x" if baseline_only else "all factors"
    return base.markdown_table(
        ["Method", f"Rows ({scope})", "Elapsed min mean +/- SD", "Elapsed min range", "Peak GB mean +/- SD", "Peak GB range"],
        table_rows,
    )


def write_plots(agg_rows: Sequence[Dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for method in SWEEP_METHODS:
        rows = [row for row in agg_rows if row.get("method") == method]
        rows = sorted(rows, key=lambda row: float(row["reg_factor"]))
        for x_key, x_label, slug in PLOT_METRIC_SPECS:
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

    styles = {
        METHOD_IDIR: {"marker": "o", "color": "#1f77b4", "linestyle": "-"},
        METHOD_DIRECTCP32: {"marker": "s", "color": "#ff7f0e", "linestyle": "--"},
    }
    for x_key, x_label, slug in PLOT_METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
        wrote_any = False
        x_std_key = x_key.removesuffix("_mean") + "_std"
        y_key = "original_snap_tre_mean_mm_mean"
        y_std_key = "original_snap_tre_mean_mm_std"
        x_extent: List[float] = []
        y_extent: List[float] = []
        for method in SWEEP_METHODS:
            rows = [row for row in agg_rows if row.get("method") == method]
            rows = sorted(rows, key=lambda row: float(row["reg_factor"]))
            xs = np.asarray([base.safe_float(row.get(x_key)) for row in rows], dtype=np.float64)
            ys = np.asarray([base.safe_float(row.get(y_key)) for row in rows], dtype=np.float64)
            x_stds = np.asarray([base.safe_float(row.get(x_std_key)) for row in rows], dtype=np.float64)
            y_stds = np.asarray([base.safe_float(row.get(y_std_key)) for row in rows], dtype=np.float64)
            keep = np.isfinite(xs) & np.isfinite(ys)
            if not bool(np.any(keep)):
                continue
            style = styles.get(method, {"marker": "o"})
            color = style.get("color")
            for x, y, x_std, y_std in zip(xs[keep], ys[keep], x_stds[keep], y_stds[keep]):
                x_extent.append(float(x))
                y_extent.append(float(y))
                if math.isfinite(float(x_std)) and math.isfinite(float(y_std)) and (x_std > 0 or y_std > 0):
                    x_radius = max(float(x_std), 0.0)
                    y_radius = max(float(y_std), 0.0)
                    ellipse = Ellipse(
                        (float(x), float(y)),
                        width=2.0 * x_radius,
                        height=2.0 * y_radius,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.14,
                        zorder=1,
                    )
                    ax.add_patch(ellipse)
                    x_extent.extend([float(x) - x_radius, float(x) + x_radius])
                    y_extent.extend([float(y) - y_radius, float(y) + y_radius])
            ax.plot(
                xs[keep],
                ys[keep],
                marker=style["marker"],
                color=color,
                linestyle=style.get("linestyle", "-"),
                linewidth=1.2,
                label=method_label(method),
                zorder=3,
            )
            wrote_any = True
        if not wrote_any:
            plt.close(fig)
            continue
        if x_extent and y_extent:
            x_min, x_max = min(x_extent), max(x_extent)
            y_min, y_max = min(y_extent), max(y_extent)
            x_pad = 0.05 * max(x_max - x_min, 1e-6)
            y_pad = 0.05 * max(y_max - y_min, 1e-6)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
        ax.set_xlabel(f"{x_label} ↓")
        ax.set_ylabel("Original snap TRE mean (mm) ↓")
        ax.set_title("IDIR vs DirectCP 32-16-8-4")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"combined_original_snap_vs_{slug}.png")
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
                fmt(row["fg_disp_mag_mean_vox_mean"], 3),
                fmt(row["fg_disp_mag_p95_vox_mean"], 3),
                fmt(row["fg_disp_mag_max_vox_mean"], 3),
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
            "Disp mean",
            "Disp p95",
            "Disp max",
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
                fmt(best["fg_disp_mag_mean_vox_mean"], 3),
                fmt(best["fg_disp_mag_p95_vox_mean"], 3),
                fmt(best["fg_disp_mag_max_vox_mean"], 3),
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
    rows = [row for row in rows if is_expected_factor(row)]
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
        [
            "Method",
            "Best factor",
            "Reg value",
            "Original Snap",
            "FG logJ std",
            "FG J<=0%",
            "BE fg",
            "Diff fg",
            "Disp mean",
            "Disp p95",
            "Disp max",
        ],
        best_rows(agg_rows),
    )
    report = "\n".join(
        [
            "# LungCT IDIR + DirectCP 32-16-8-4 Regularization Sweep",
            "",
            "Original Snap TRE is the primary comparison metric. Original means native/crop-space TRE after resizing the saved 1mm displacement field back to the corrected pTVReg native landmark crop. The three other TRE columns are retained as diagnostics.",
            "",
            "Combined Pareto plots include translucent axis-aligned ellipses at each factor point. Each ellipse is centered at the mean point, with horizontal radius equal to one standard deviation of the x-axis regularity metric across cases and vertical radius equal to one standard deviation of original-snap TRE across cases. Points are connected in increasing regularization-factor order: 0.01x, 0.1x, 1x, 10x, 100x.",
            "",
            "## Mean Rows",
            "",
            markdown_mean_table(agg_rows),
            "",
            "## Best Original Snap By Method",
            "",
            best,
            "",
            "## Compute Cost And Epochs",
            "",
            "Runtime and memory are from completed `summary.json` files. The per-case table below uses the reused default `1x` baseline rows, because those are the primary comparison points. The all-factor summary includes all 5 factors per method.",
            "",
            "### Epoch Settings",
            "",
            epoch_config_table(),
            "",
            "### Default 1x Per-Case Cost",
            "",
            baseline_cost_table(rows),
            "",
            "### Default 1x Summary",
            "",
            method_cost_summary_table(rows, baseline_only=True),
            "",
            "### All-Factor Summary",
            "",
            method_cost_summary_table(rows, baseline_only=False),
            "",
            "## Case Counts",
            "",
            counts,
            "",
            "## Artifacts",
            "",
            "- Per-case rows: `sweep_case_rows.csv` and `sweep_case_rows.json`.",
            "- Mean rows: `sweep_mean_rows.csv` and `sweep_mean_rows.json`.",
            "- Pareto plots: method-specific `plots/*original_snap_vs*.png` and combined `plots/combined_original_snap_vs*.png`.",
            "- Training-curve analysis: `training_curve_analysis/training_curve_epoch_analysis.md` and `plots/training_curves/`.",
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
