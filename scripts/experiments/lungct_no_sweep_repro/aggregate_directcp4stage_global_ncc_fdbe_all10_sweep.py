#!/usr/bin/env python
"""Aggregate all-case LungCT DirectCP 4-stage global NCC + dense FD-BE sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import aggregate_report as base
import common as c
from launch_directcp4stage_global_ncc_fdbe_all10_sweep import DEFAULT_SWEEP_ROOT
from run_one import METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE


METHOD_IDIR = "idir_be_default_lungmask"
METHOD_PTV_DIRECTCP = "direct_cp_32_16_8_4_ptv_lcc_isotv"
METHOD_FDBE = METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE

DEFAULT_SELECTED_ROOT = Path("/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep"
DEFAULT_REG_SWEEP_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"
DEFAULT_REG_SWEEP_CASE_ROWS = DEFAULT_REG_SWEEP_REVIEW_ROOT / "sweep_case_rows.csv"

EXPECTED_CASES = list(range(1, 11))
EXPECTED_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
EXPECTED_FACTORS = [0.01, 0.1, 1.0, 10.0, 100.0]
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
PLOT_METRICS = [
    ("fg_logj_std_mean", "FG stdLogJ", "fg_logj_std"),
    ("fg_jdet_le0_pct_mean", "FG J <= 0 %", "fg_jdet_le0_pct"),
    ("fg_bending_energy_mean", "FG bending energy", "fg_bending_energy"),
    ("fg_diffusion_l2_mean", "FG diffusion energy", "fg_diffusion_l2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--missing-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--reg-sweep-case-rows", type=Path, default=DEFAULT_REG_SWEEP_CASE_ROWS)
    parser.add_argument("--plot-dir-name", type=str, default="plots_compare_ellipses")
    return parser.parse_args()


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def fmt_reg(value: float) -> str:
    return f"{float(value):g}"


def method_label(method: str) -> str:
    if method == METHOD_IDIR:
        return "IDIR BE"
    if method == METHOD_PTV_DIRECTCP:
        return "DirectCP 4-stage pTV LCC + isoTV"
    if method == METHOD_FDBE:
        return "DirectCP 4-stage global NCC + FD-BE"
    return method


def fdbe_row_from_summary(path: Path, payload: Dict[str, object], source: str) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    alpha = float(training.get("alpha", float("nan")))
    out = {
        "method": METHOD_FDBE,
        "method_label": method_label(METHOD_FDBE),
        "case_id": int(row.get("case_id") or payload.get("case_id")),
        "source": source,
        "reg_sort": alpha,
        "reg_value": alpha,
        "reg_label": f"a={fmt_reg(alpha)}",
        "alpha": alpha,
        "summary_path": str(path),
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
    return out


def collect_fdbe_rows(root: Path, source: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted((root / METHOD_FDBE).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        row = fdbe_row_from_summary(summary_path, payload, source)
        if int(row["case_id"]) not in EXPECTED_CASES:
            continue
        if not any(abs(float(row["alpha"]) - alpha) < 1e-9 for alpha in EXPECTED_ALPHAS):
            continue
        rows.append(row)
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_reg_sweep_row(row: Dict[str, str]) -> Dict[str, object]:
    method = str(row["method"])
    reg_factor = float(row["reg_factor"])
    reg_value = float(row["reg_value"])
    if method == METHOD_IDIR:
        label = f"a={fmt_reg(reg_value)}"
    elif method == METHOD_PTV_DIRECTCP:
        label = f"w={fmt_reg(reg_value)}"
    else:
        label = f"{fmt_reg(reg_factor)}x"
    out: Dict[str, object] = {
        "method": method,
        "method_label": method_label(method),
        "case_id": int(row["case_id"]),
        "source": row.get("source", "reg_sweep"),
        "reg_sort": reg_factor,
        "reg_factor": reg_factor,
        "reg_value": reg_value,
        "reg_label": label,
        "summary_path": row.get("summary_path", ""),
    }
    for key in METRIC_KEYS:
        out[key] = row.get(key, "")
    return out


def collect_reg_sweep_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_csv_rows(path):
        method = str(row.get("method", ""))
        if method not in {METHOD_IDIR, METHOD_PTV_DIRECTCP}:
            continue
        case_id = int(row["case_id"])
        factor = float(row["reg_factor"])
        if case_id not in EXPECTED_CASES:
            continue
        if not any(abs(factor - expected) < 1e-10 for expected in EXPECTED_FACTORS):
            continue
        rows.append(normalize_reg_sweep_row(row))
    return rows


def dedupe_fdbe_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    by_key: Dict[Tuple[int, float], Dict[str, object]] = {}
    errors: List[str] = []
    for row in rows:
        key = (int(row["case_id"]), float(row["alpha"]))
        if key in by_key:
            errors.append(f"duplicate FD-BE row for case {key[0]:02d}, alpha {key[1]:g}")
            continue
        by_key[key] = dict(row)
    return [by_key[key] for key in sorted(by_key)], errors


def completion_errors(fdbe_rows: Sequence[Dict[str, object]], reg_rows: Sequence[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    fd_keys = {(int(row["case_id"]), float(row["alpha"])) for row in fdbe_rows}
    for case_id in EXPECTED_CASES:
        got = sorted(alpha for c, alpha in fd_keys if c == case_id)
        if got != EXPECTED_ALPHAS:
            errors.append(f"FD-BE case {case_id:02d}: got alphas {got}, expected {EXPECTED_ALPHAS}")
    for method in (METHOD_IDIR, METHOD_PTV_DIRECTCP):
        for factor in EXPECTED_FACTORS:
            got_cases = sorted(int(row["case_id"]) for row in reg_rows if row["method"] == method and abs(float(row["reg_sort"]) - factor) < 1e-10)
            if got_cases != EXPECTED_CASES:
                errors.append(f"{method} factor {factor:g}: got cases {got_cases}, expected {EXPECTED_CASES}")
    return errors


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    groups = sorted(
        {(str(row["method"]), float(row["reg_sort"]), str(row["reg_label"]), float(row["reg_value"])) for row in rows},
        key=lambda item: (item[0], item[1]),
    )
    for method, reg_sort, reg_label, reg_value in groups:
        group_rows = [
            row
            for row in rows
            if str(row["method"]) == method
            and abs(float(row["reg_sort"]) - reg_sort) < 1e-10
            and str(row["reg_label"]) == reg_label
        ]
        agg: Dict[str, object] = {
            "method": method,
            "method_label": method_label(method),
            "reg_sort": reg_sort,
            "reg_value": reg_value,
            "reg_label": reg_label,
            "cases": len(group_rows),
        }
        for key in METRIC_KEYS:
            mean, std = base.mean_std([row.get(key) for row in group_rows])
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def write_ellipse_plots(agg_rows: Sequence[Dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    styles = {
        METHOD_IDIR: {"marker": "o", "color": "#1f77b4", "linestyle": "-"},
        METHOD_PTV_DIRECTCP: {"marker": "s", "color": "#ff7f0e", "linestyle": "--"},
        METHOD_FDBE: {"marker": "^", "color": "#2ca02c", "linestyle": "-."},
    }
    for x_key, x_label, slug in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7.0, 4.9), dpi=160)
        wrote_any = False
        x_std_key = x_key.removesuffix("_mean") + "_std"
        y_key = "original_snap_tre_mean_mm_mean"
        y_std_key = "original_snap_tre_mean_mm_std"
        x_extent: List[float] = []
        y_extent: List[float] = []
        for method in (METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE):
            rows = [row for row in agg_rows if row.get("method") == method]
            rows = sorted(rows, key=lambda row: float(row["reg_sort"]))
            xs = np.asarray([base.safe_float(row.get(x_key)) for row in rows], dtype=np.float64)
            ys = np.asarray([base.safe_float(row.get(y_key)) for row in rows], dtype=np.float64)
            x_stds = np.asarray([base.safe_float(row.get(x_std_key)) for row in rows], dtype=np.float64)
            y_stds = np.asarray([base.safe_float(row.get(y_std_key)) for row in rows], dtype=np.float64)
            keep = np.isfinite(xs) & np.isfinite(ys)
            if not bool(np.any(keep)):
                continue
            style = styles[method]
            color = style["color"]
            for row, x, y, x_std, y_std in zip(rows, xs, ys, x_stds, y_stds):
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    continue
                ax.annotate(str(row["reg_label"]), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
                x_extent.append(float(x))
                y_extent.append(float(y))
                if math.isfinite(float(x_std)) and math.isfinite(float(y_std)) and (x_std > 0 or y_std > 0):
                    ellipse = Ellipse(
                        (float(x), float(y)),
                        width=2.0 * max(float(x_std), 0.0),
                        height=2.0 * max(float(y_std), 0.0),
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.14,
                        zorder=1,
                    )
                    ax.add_patch(ellipse)
                    x_extent.extend([float(x) - float(x_std), float(x) + float(x_std)])
                    y_extent.extend([float(y) - float(y_std), float(y) + float(y_std)])
            ax.plot(
                xs[keep],
                ys[keep],
                marker=style["marker"],
                color=color,
                linestyle=style["linestyle"],
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
            ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
            ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
        ax.set_xlabel(f"{x_label} ↓")
        ax.set_ylabel("Original snap TRE mean (mm) ↓")
        ax.set_title("All-case LungCT Pareto with 1-SD ellipses")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"all10_original_snap_vs_{slug}_ellipses.png")
        plt.close(fig)


def markdown_mean_table(rows: Sequence[Dict[str, object]], method: str) -> str:
    table_rows = []
    for row in sorted([row for row in rows if row["method"] == method], key=lambda row: float(row["reg_sort"])):
        table_rows.append(
            [
                row["reg_label"],
                f"{fmt(row['original_snap_tre_mean_mm_mean'])} +/- {fmt(row['original_snap_tre_mean_mm_std'])}",
                f"{fmt(row['original_subvoxel_tre_mean_mm_mean'])} +/- {fmt(row['original_subvoxel_tre_mean_mm_std'])}",
                fmt(row["fg_logj_std_mean"], 6),
                fmt(row["fg_jdet_le0_pct_mean"], 4),
                fmt(row["fg_bending_energy_mean"], 6),
                fmt(row["fg_diffusion_l2_mean"], 6),
            ]
        )
    return base.markdown_table(
        ["Reg", "Original snap TRE", "Original sub TRE", "FG stdLogJ", "FG J<=0 %", "FG BE", "FG diffusion"],
        table_rows,
    )


def best_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method in (METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE):
        method_rows = [
            row
            for row in rows
            if row["method"] == method and math.isfinite(base.safe_float(row.get("original_snap_tre_mean_mm_mean")))
        ]
        if not method_rows:
            continue
        best = min(method_rows, key=lambda row: float(row["original_snap_tre_mean_mm_mean"]))
        lines.append(
            "- "
            f"{method_label(method)}: best `{best['reg_label']}` with original snap TRE "
            f"`{fmt(best['original_snap_tre_mean_mm_mean'])} +/- {fmt(best['original_snap_tre_mean_mm_std'])} mm`, "
            f"FG stdLogJ `{fmt(best['fg_logj_std_mean'], 6)}`, FG J<=0 `{fmt(best['fg_jdet_le0_pct_mean'], 4)}%`."
        )
    return lines


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    fd_rows_raw = collect_fdbe_rows(args.selected_root, "selected_case_root") + collect_fdbe_rows(
        args.missing_root, "missing_case_root"
    )
    fd_rows, duplicate_errors = dedupe_fdbe_rows(fd_rows_raw)
    reg_rows = collect_reg_sweep_rows(args.reg_sweep_case_rows)
    errors = duplicate_errors + completion_errors(fd_rows, reg_rows)
    comparison_case_rows = reg_rows + fd_rows
    fd_mean_rows = aggregate_rows(fd_rows)
    comparison_mean_rows = aggregate_rows(comparison_case_rows)

    c.write_csv(args.review_root / "fdbe_case_rows.csv", fd_rows)
    c.write_json(args.review_root / "fdbe_case_rows.json", fd_rows)
    c.write_csv(args.review_root / "fdbe_mean_rows.csv", fd_mean_rows)
    c.write_json(args.review_root / "fdbe_mean_rows.json", fd_mean_rows)
    c.write_csv(args.review_root / "comparison_case_rows.csv", comparison_case_rows)
    c.write_json(args.review_root / "comparison_case_rows.json", comparison_case_rows)
    c.write_csv(args.review_root / "comparison_mean_rows.csv", comparison_mean_rows)
    c.write_json(args.review_root / "comparison_mean_rows.json", comparison_mean_rows)
    write_ellipse_plots(comparison_mean_rows, args.review_root / args.plot_dir_name)

    lines = [
        "# LungCT DirectCP 4-stage Global NCC + Dense FD-BE All-Case Sweep",
        "",
        "## Summary",
        "",
        "- FD-BE method: `direct_cp_32_16_8_4_global_ncc_fd_be`.",
        "- Stage schedule: `32,16,8,4`; downsample schedule: `8,4,2,1`; stage epochs: `200,400,800,1000`.",
        "- FD-BE alpha grid: `0,10,100,1000,10000,100000`.",
        "- FD-BE rows merge existing cases `01,08` with new missing-case runs.",
        "- Comparison figures use all 10 cases for IDIR, pTV DirectCP, and FD-BE.",
        "- Ellipses are axis-aligned 1-SD ellipses across cases.",
        "- Ellipse centers are method/reg-weight means; horizontal radius is the across-case SD of the plotted regularity metric, vertical radius is the across-case SD of original-snap TRE, and covariance is not used.",
        "",
        "## Completion",
        "",
        f"- FD-BE rows: `{len(fd_rows)}` expected `60`.",
        f"- Comparison rows: `{len(comparison_case_rows)}`.",
    ]
    if errors:
        lines.append("- Completion errors:")
        lines.extend(f"  - {err}" for err in errors)
    else:
        lines.append("- Completion check: passed.")
    lines.extend(["", "## Best Mean Points", ""])
    lines.extend(best_lines(comparison_mean_rows) or ["- No finite rows found."])
    lines.extend(["", "## FD-BE Mean Rows", "", markdown_mean_table(fd_mean_rows, METHOD_FDBE)])
    lines.extend(["", "## Comparison Mean Rows", ""])
    for method in (METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE):
        lines.extend(["", f"### {method_label(method)}", "", markdown_mean_table(comparison_mean_rows, method)])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- FD-BE CSV/JSON: `fdbe_case_rows.*`, `fdbe_mean_rows.*`.",
            "- Comparison CSV/JSON: `comparison_case_rows.*`, `comparison_mean_rows.*`.",
            f"- Ellipse plots: `{args.plot_dir_name}/all10_original_snap_vs_*_ellipses.png`.",
        ]
    )
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"fdbe_rows": len(fd_rows), "comparison_rows": len(comparison_case_rows), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
