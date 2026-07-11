#!/usr/bin/env python
"""Generate OASIS first10 Pareto plots with saved DL/Greedy baseline dots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


DEFAULT_INPUT = (
    c.REPO_ROOT
    / "experiments/2026-06-25_oasis_seed3_first10_saved_baselines/population_summary_with_saved_baselines.csv"
)
DEFAULT_REVIEW_ROOT = (
    c.REPO_ROOT
    / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_saved_baselines_paper_v1"
)

SWEEP_METHODS = [
    "idir_original_dense_siren_be_autograd",
    "sinr_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
]
BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]
BASELINE_SPECS = {
    "bs1_vxm_v0_2": {"label": "BS1 VoxelMorph", "dot_label": "BS1", "color": "#8c564b"},
    "bs2_tm_v0": {"label": "BS2 TransMorph", "dot_label": "BS2", "color": "#e377c2"},
    "bs3_vfa_v0": {"label": "BS3 VFA", "dot_label": "BS3", "color": "#7f7f7f"},
    "bs4_sitreg_v0": {"label": "BS4 SITReg", "dot_label": "BS4", "color": "#bcbd22"},
    "greedy": {"label": "Greedy", "dot_label": "Greedy", "color": "#17becf"},
}
METHOD_SPECS: Dict[str, Dict[str, object]] = {**base.METHOD_SPECS, **BASELINE_SPECS}

ACCURACY_METRICS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "Mean Dice", True),
    ("hd95_mean", "Mean HD95", False),
    ("asd_warped_to_fixed_mean", "ASD warped-to-fixed", False),
    ("asd_symmetric_mean", "ASD symmetric", False),
]
FD_REGULARITY_METRICS: List[Tuple[str, str]] = [
    ("be_fd_forward_fixed_fg_voxel", "FG BE forward FD"),
    ("diffusion_fd_forward_fixed_fg_voxel", "FG diffusion forward FD"),
    ("diffusion_l2_fg", "FG diffusion L2"),
    ("jdet_le0_pct_fg", "FG Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "FG std(log Jdet)"),
    ("ndv_frac_fg", "FG NDV fraction"),
]
FD_PRIMARY_PANELS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "jdet_le0_pct_fg", True),
    ("dice_mean", "log_jdet_std_fg", True),
    ("dice_mean", "be_fd_forward_fixed_fg_voxel", True),
    ("dice_mean", "diffusion_fd_forward_fixed_fg_voxel", True),
]
OUTPUT_FILES = [
    "population_primary_line_segments_with_saved_baselines.pdf",
    "population_primary_line_segments_with_saved_baselines.png",
    "population_contact_sheet_line_segments_with_saved_baselines.pdf",
    "population_contact_sheet_line_segments_with_saved_baselines.png",
    "population_primary_ellipses_with_saved_baselines.pdf",
    "population_primary_ellipses_with_saved_baselines.png",
    "population_contact_sheet_ellipses_with_saved_baselines.pdf",
    "population_contact_sheet_ellipses_with_saved_baselines.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def metric_mean(metric: str) -> str:
    return f"{metric}_mean"


def metric_std(metric: str) -> str:
    return f"{metric}_std"


def method_sort_key(row: Mapping[str, object]) -> Tuple[int, float]:
    method = str(row.get("method_key"))
    order = SWEEP_METHODS + BASELINE_METHODS
    method_index = order.index(method) if method in order else len(order)
    return method_index, as_float(row.get("alpha"))


def validate_rows(rows: Sequence[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    expected_counts = {method: 6 for method in SWEEP_METHODS}
    expected_counts.update({method: 1 for method in BASELINE_METHODS})
    observed_counts = {method: 0 for method in expected_counts}
    for row in rows:
        method = str(row.get("method_key"))
        if method in observed_counts:
            observed_counts[method] += 1
        else:
            errors.append(f"Unexpected method row: {method}")
        if int(as_float(row.get("n_cases"))) != 10:
            errors.append(f"{method}: n_cases {row.get('n_cases')} != 10")
    if len(rows) != 29:
        errors.append(f"Expected 29 population rows, found {len(rows)}")
    for method, expected in expected_counts.items():
        if observed_counts.get(method, 0) != expected:
            errors.append(f"{method}: expected {expected} rows, found {observed_counts.get(method, 0)}")

    required_metric_pairs = set()
    for acc_key, _acc_label, _higher in ACCURACY_METRICS:
        for reg_key, _reg_label in FD_REGULARITY_METRICS:
            required_metric_pairs.add((acc_key, reg_key))
    for acc_key, reg_key, _higher in FD_PRIMARY_PANELS:
        required_metric_pairs.add((acc_key, reg_key))

    for acc_key, reg_key in sorted(required_metric_pairs):
        for suffix_key in [metric_mean(acc_key), metric_std(acc_key), metric_mean(reg_key), metric_std(reg_key)]:
            if rows and suffix_key not in rows[0]:
                errors.append(f"Missing required column: {suffix_key}")
                continue
            bad = [str(row.get("method_key")) for row in rows if not is_finite(row.get(suffix_key))]
            if bad:
                errors.append(f"Column {suffix_key} has non-finite values for: {', '.join(bad[:10])}")
    return errors


def should_symlog(reg_key: str) -> bool:
    return "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key


def panel_specs(grid: bool) -> Tuple[List[Tuple[str, str, bool, str, str]], int, int]:
    if grid:
        specs = [
            (acc_key, acc_label, higher, reg_key, reg_label)
            for acc_key, acc_label, higher in ACCURACY_METRICS
            for reg_key, reg_label in FD_REGULARITY_METRICS
        ]
        return specs, len(ACCURACY_METRICS), len(FD_REGULARITY_METRICS)
    label_by_acc = {key: label for key, label, _higher in ACCURACY_METRICS}
    label_by_reg = dict(FD_REGULARITY_METRICS)
    specs = [(acc, label_by_acc[acc], higher, reg, label_by_reg[reg]) for acc, reg, higher in FD_PRIMARY_PANELS]
    return specs, 2, 2


def add_ellipse(ax: plt.Axes, row: Mapping[str, object], x: float, y: float, reg_key: str, acc_key: str, color: str) -> None:
    x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
    y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
    if not math.isfinite(x_std) or not math.isfinite(y_std) or (x_std == 0.0 and y_std == 0.0):
        return
    ellipse = Ellipse(
        (x, y),
        width=2.0 * x_std,
        height=2.0 * y_std,
        facecolor=color,
        edgecolor=color,
        linewidth=0.8,
        alpha=0.14,
        zorder=1,
    )
    ax.add_patch(ellipse)


def collect_extents(
    rows: Sequence[Mapping[str, object]],
    reg_key: str,
    acc_key: str,
    *,
    ellipses: bool,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = as_float(row.get(metric_mean(reg_key)))
        y = as_float(row.get(metric_mean(acc_key)))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)
        if ellipses:
            x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
            y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
            if math.isfinite(x_std):
                xs.extend([x - x_std, x + x_std])
            if math.isfinite(y_std):
                ys.extend([y - y_std, y + y_std])
    return xs, ys


def set_limits(ax: plt.Axes, rows: Sequence[Mapping[str, object]], reg_key: str, acc_key: str, *, ellipses: bool) -> None:
    xs, ys = collect_extents(rows, reg_key, acc_key, ellipses=ellipses)
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = 0.05 * max(x_max - x_min, 1e-8)
    y_pad = 0.05 * max(y_max - y_min, 1e-8)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


def plot_population(
    rows: Sequence[Dict[str, object]],
    path_base: Path,
    title: str,
    *,
    grid: bool,
    ellipses: bool,
) -> None:
    specs, n_rows, n_cols = panel_specs(grid)
    figsize = (4.1 * n_cols, 3.3 * n_rows) if grid else (13.0, 10.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    all_ordered_rows = sorted(rows, key=method_sort_key)

    for ax, (acc_key, acc_label, _higher, reg_key, reg_label) in zip(axes.ravel(), specs):
        acc_mean = metric_mean(acc_key)
        reg_mean = metric_mean(reg_key)
        for method in SWEEP_METHODS:
            group = [
                row
                for row in all_ordered_rows
                if str(row.get("method_key")) == method and is_finite(row.get(acc_mean)) and is_finite(row.get(reg_mean))
            ]
            group.sort(key=lambda row: as_float(row.get("alpha")))
            if not group:
                continue
            spec = METHOD_SPECS[method]
            color = str(spec["color"])
            xs = [as_float(row.get(reg_mean)) for row in group]
            ys = [as_float(row.get(acc_mean)) for row in group]
            if ellipses:
                for row, x, y in zip(group, xs, ys):
                    add_ellipse(ax, row, x, y, reg_key, acc_key, color)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.35,
                markersize=4.8,
                color=color,
                label=str(spec["label"]),
                zorder=3,
            )
            for row, x, y in zip(group, xs, ys):
                ax.annotate(f"{as_float(row.get('alpha')):g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")

        for method in BASELINE_METHODS:
            group = [
                row
                for row in all_ordered_rows
                if str(row.get("method_key")) == method and is_finite(row.get(acc_mean)) and is_finite(row.get(reg_mean))
            ]
            if not group:
                continue
            row = group[0]
            spec = METHOD_SPECS[method]
            color = str(spec["color"])
            x = as_float(row.get(reg_mean))
            y = as_float(row.get(acc_mean))
            if ellipses:
                add_ellipse(ax, row, x, y, reg_key, acc_key, color)
            ax.scatter([x], [y], marker="D", s=44, color=color, label=str(spec["label"]), zorder=4)
            ax.annotate(str(spec["dot_label"]), (x, y), fontsize=6, xytext=(4, 3), textcoords="offset points")

        set_limits(ax, all_ordered_rows, reg_key, acc_key, ellipses=ellipses)
        if should_symlog(reg_key):
            ax.set_xscale("symlog", linthresh=1e-8)
        ax.set_xlabel(reg_label)
        ax.set_ylabel(acc_label)
        ax.grid(True, alpha=0.25)

    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if not grid else 0.985))
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), dpi=180)
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def write_report(args: argparse.Namespace, rows: Sequence[Dict[str, object]], output_paths: Sequence[Path]) -> None:
    lines = [
        "# OASIS Seed-3 First10 Saved-Baseline Pareto Plot Styles",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Input population CSV: `{args.input}`",
        f"- Review root: `{args.review_root}`",
        f"- Population rows: `{len(rows)}`",
        "- Styles: line-segment-only and LungCT-style axis-aligned 1-SD ellipses.",
        "- Baselines are plotted as standalone diamond dots and are not connected by lines.",
        "- Ellipse radii use across-case standard deviations from the population CSV for both sweep and saved-baseline points.",
        "- DirectCP cps2 preliminary rows are intentionally excluded.",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in output_paths)
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.input))
    errors = validate_rows(rows)
    if errors:
        raise RuntimeError("Input validation failed:\n" + "\n".join(errors[:80]))

    args.review_root.mkdir(parents=True, exist_ok=True)
    plot_population(
        rows,
        args.review_root / "population_primary_line_segments_with_saved_baselines",
        "OASIS seed-3 first10: primary FD/Jacobian ARC curves with saved baselines",
        grid=False,
        ellipses=False,
    )
    plot_population(
        rows,
        args.review_root / "population_contact_sheet_line_segments_with_saved_baselines",
        "OASIS seed-3 first10: accuracy vs FD/Jacobian regularity with saved baselines",
        grid=True,
        ellipses=False,
    )
    plot_population(
        rows,
        args.review_root / "population_primary_ellipses_with_saved_baselines",
        "OASIS seed-3 first10: primary FD/Jacobian ARC curves with 1-SD ellipses",
        grid=False,
        ellipses=True,
    )
    plot_population(
        rows,
        args.review_root / "population_contact_sheet_ellipses_with_saved_baselines",
        "OASIS seed-3 first10: accuracy vs FD/Jacobian regularity with 1-SD ellipses",
        grid=True,
        ellipses=True,
    )

    output_paths = [args.review_root / name for name in OUTPUT_FILES]
    missing = [str(path) for path in output_paths if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError("Missing or empty plot output(s):\n" + "\n".join(missing))
    manifest = {
        "generated_at": c.now_text(),
        "input": str(args.input),
        "review_root": str(args.review_root),
        "row_count": len(rows),
        "sweep_methods": SWEEP_METHODS,
        "baseline_methods": BASELINE_METHODS,
        "primary_panels": FD_PRIMARY_PANELS,
        "full_regularities": FD_REGULARITY_METRICS,
        "output_files": [str(path) for path in output_paths],
        "validation": "passed",
    }
    c.write_json(args.review_root / "plot_manifest.json", manifest)
    write_report(args, rows, output_paths)
    print(json.dumps({"review_root": str(args.review_root), "output_files": len(output_paths)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
