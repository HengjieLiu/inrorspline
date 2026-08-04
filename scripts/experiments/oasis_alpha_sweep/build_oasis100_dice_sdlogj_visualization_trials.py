#!/usr/bin/env python
"""Try alternative OASIS100 Dice-vs-SDlogJ reviewer-response visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines as oasis_plot  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as oasis_plus  # noqa: E402


DIAGNOSTIC_ROOT = (
    c.REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100/"
    "reviewer_response/oasis100_dice_sdlogj"
)
DEFAULT_OUTPUT_ROOT = DIAGNOSTIC_ROOT / "innovative_visualization_trials"
DEFAULT_CASE_PAIRS = DIAGNOSTIC_ROOT / "covariance_diagnostics/case_pairs.csv"
DEFAULT_COVARIANCE_SUMMARY = DIAGNOSTIC_ROOT / "covariance_diagnostics/covariance_summary.csv"

ACC_KEY = "dice_mean"
REG_KEY = "log_jdet_std_fg"
ACC_LABEL = "Mean Dice"
REG_LABEL = "Foreground SDlogJ"

SWEEP_METHODS = [
    "idir_original_dense_siren_be_autograd",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    f100.DCP842_METHOD,
]
BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]
SELECTED_ALPHAS = {
    "idir_original_dense_siren_be_autograd": 1.0,
    "sinr_cps4_be_bspline_analytic": 1000.0,
    "direct_cp_cps4_be_bspline_analytic": 1000.0,
    "sinr_cps2_be_bspline_analytic": 1000.0,
    "direct_cp_cps2_be_bspline_analytic": 1000.0,
    f100.DCP842_METHOD: 1000.0,
}
ALL_ALPHA_LABELS = ["0", "0.01", "0.1", "1", "10", "100", "1000", "10000", "100000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-pairs", type=Path, default=DEFAULT_CASE_PAIRS)
    parser.add_argument("--covariance-summary", type=Path, default=DEFAULT_COVARIANCE_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def alpha_text(value: object) -> str:
    value_float = as_float(value)
    return f"{value_float:g}" if math.isfinite(value_float) else "baseline"


def method_order(method: str) -> int:
    order = SWEEP_METHODS + BASELINE_METHODS
    return order.index(method) if method in order else len(order)


def group_sort_key(row: Mapping[str, object]) -> Tuple[int, float]:
    method = str(row["method_key"])
    alpha = as_float(row.get("alpha"))
    return method_order(method), alpha if math.isfinite(alpha) else math.inf


def display_label(method: str) -> str:
    spec = oasis_plot.METHOD_SPECS.get(method, {})
    return str(spec.get("label", method))


def color_for(method: str) -> str:
    spec = oasis_plot.METHOD_SPECS.get(method, {})
    return str(spec.get("color", "#555555"))


def group_label(method: str, alpha: object) -> str:
    if method in BASELINE_METHODS:
        return display_label(method)
    return f"{display_label(method)} a={alpha_text(alpha)}"


def repo_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(c.REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def write_readme(path: Path, title: str, body: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", *body]
    path.write_text("\n".join(lines).rstrip() + "\n")


def output_status(paths: Sequence[Path]) -> Dict[str, object]:
    return {
        "expected_files": [repo_path(path) for path in paths],
        "missing_or_empty": [repo_path(path) for path in paths if not path.exists() or path.stat().st_size <= 0],
    }


def selected_rows(case_pairs: Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
    out: List[Mapping[str, str]] = []
    for row in case_pairs:
        method = str(row["method_key"])
        if method in BASELINE_METHODS:
            out.append(row)
            continue
        selected = SELECTED_ALPHAS.get(method)
        if selected is not None and math.isclose(as_float(row.get("alpha")), selected, rel_tol=0.0, abs_tol=1e-9):
            out.append(row)
    return out


def group_rows(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], List[Mapping[str, str]]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        method = str(row["method_key"])
        alpha = "baseline" if method in BASELINE_METHODS else alpha_text(row.get("alpha"))
        groups[(method, alpha)].append(row)
    return dict(groups)


def group_xy(rows: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([as_float(row[REG_KEY]) for row in rows], dtype=np.float64)
    y = np.asarray([as_float(row[ACC_KEY]) for row in rows], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def regression_line(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    if x.size < 2 or float(np.std(x)) <= 0.0:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    return xs, slope * xs + intercept


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid)
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    bw = 1.06 * std * values.size ** (-1 / 5) if std > 0.0 else 0.03
    bw = max(bw, 1e-4)
    diff = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * diff * diff).sum(axis=1) / (values.size * bw * math.sqrt(2.0 * math.pi))
    peak = float(np.max(density))
    return density / peak if peak > 0.0 else density


def plot_small_multiple_case_clouds(case_pairs: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    groups = group_rows(selected_rows(case_pairs))
    keys = sorted(groups, key=lambda key: (method_order(key[0]), as_float(key[1])))
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 10.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, key in zip(axes_flat, keys):
        method, alpha = key
        rows = groups[key]
        x, y = group_xy(rows)
        color = color_for(method)
        ax.scatter(x, y, s=12, color=color, alpha=0.42, edgecolor="none")
        ax.scatter([np.mean(x)], [np.mean(y)], s=54, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        line = regression_line(x, y)
        if line is not None:
            ax.plot(line[0], line[1], color="#202020", linewidth=1.0, alpha=0.8)
        ax.set_title(group_label(method, alpha), fontsize=9)
        ax.text(
            0.03,
            0.05,
            f"r={pearson(x, y):.2f}\nrho={spearman(x, y):.2f}",
            transform=ax.transAxes,
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.grid(True, alpha=0.22)
    for ax in axes_flat[len(keys) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(REG_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(ACC_LABEL)
    fig.suptitle("Small-multiple case clouds: selected OASIS100 operating points", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path = out_dir / "small_multiple_case_clouds_selected_points.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Small-Multiple Case Clouds",
        [
            "One panel per selected method/reference point. Each dot is one OASIS case; the larger dot is the group mean.",
            "",
            "This view is useful when the main question is whether high-SDlogJ cases are also low-Dice cases without putting every method into one crowded axis.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def plot_correlation_slope_panels(summary_rows: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    alpha_positions = {label: idx for idx, label in enumerate(ALL_ALPHA_LABELS)}
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.4), sharex=True)
    for method in SWEEP_METHODS:
        rows = [row for row in summary_rows if str(row["method_key"]) == method]
        rows.sort(key=lambda row: alpha_positions.get(alpha_text(row["alpha"]), 999))
        xs = [alpha_positions[alpha_text(row["alpha"])] for row in rows]
        pear = [as_float(row["pearson_r"]) for row in rows]
        spear = [as_float(row["spearman_rho"]) for row in rows]
        color = color_for(method)
        axes[0].plot(xs, pear, marker="o", color=color, linewidth=1.4, label=display_label(method))
        axes[1].plot(xs, spear, marker="o", color=color, linewidth=1.4, label=display_label(method))
    axes[0].set_ylabel("Pearson r")
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_xlabel("alpha")
    for ax in axes:
        ax.axhline(0.0, color="#202020", linewidth=0.8, alpha=0.7)
        ax.set_ylim(-1.0, 0.1)
        ax.grid(True, alpha=0.24)
    axes[0].legend(fontsize=8, ncol=2, frameon=True)
    axes[1].set_xticks(range(len(ALL_ALPHA_LABELS)))
    axes[1].set_xticklabels(ALL_ALPHA_LABELS, rotation=35, ha="right")
    fig.suptitle("Correlation between case-level SDlogJ and Dice across alpha", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path = out_dir / "correlation_vs_alpha_lines.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Correlation Slope Panels",
        [
            "Pearson and Spearman correlations are plotted as method-wise trends over alpha.",
            "",
            "This is the cleanest view for asking whether the SDlogJ-Dice covariance changes with regularization strength.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def plot_ridge_marginals(case_pairs: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    outputs: List[Path] = []
    for selected, suffix, height in [(True, "selected_points", 8.8), (False, "full_alpha_grid", 17.0)]:
        groups = group_rows(selected_rows(case_pairs) if selected else case_pairs)
        keys = sorted(groups, key=lambda key: (method_order(key[0]), as_float(key[1])))
        fig, axes = plt.subplots(1, 2, figsize=(13.6, height), sharey=True)
        for metric_idx, (ax, metric, label) in enumerate([(axes[0], ACC_KEY, ACC_LABEL), (axes[1], REG_KEY, REG_LABEL)]):
            all_values = np.asarray([as_float(row[metric]) for rows in groups.values() for row in rows], dtype=np.float64)
            grid = np.linspace(float(np.nanmin(all_values)), float(np.nanmax(all_values)), 240)
            for pos, key in enumerate(keys):
                method, alpha = key
                values = np.asarray([as_float(row[metric]) for row in groups[key]], dtype=np.float64)
                density = kde(values, grid)
                y = pos + density * 0.72
                color = color_for(method)
                ax.fill_between(grid, pos, y, color=color, alpha=0.38, linewidth=0)
                ax.plot(grid, y, color=color, linewidth=0.9, alpha=0.9)
                ax.plot([float(np.nanmedian(values)), float(np.nanmedian(values))], [pos, pos + 0.45], color="#202020", linewidth=0.85)
            ax.set_xlabel(label)
            ax.grid(True, axis="x", alpha=0.22)
            if metric_idx == 0:
                ax.set_yticks(range(len(keys)))
                ax.set_yticklabels([group_label(key[0], key[1]) for key in keys], fontsize=7.0 if not selected else 8.0)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
        fig.suptitle(f"Ridgeline marginal distributions: {suffix.replace('_', ' ')}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out_path = out_dir / f"ridge_marginals_{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    write_readme(
        out_dir / "README.md",
        "Ridge Marginals",
        [
            "Horizontal ridgelines show the one-dimensional Dice and SDlogJ distributions.",
            "",
            "Compared with full-grid violins, this keeps group labels readable on the y-axis and makes many groups easier to scan.",
        ],
    )
    return outputs + [out_dir / "README.md"]


def plot_raincloud_strip(case_pairs: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    rng = np.random.default_rng(20260804)
    groups = group_rows(selected_rows(case_pairs))
    keys = sorted(groups, key=lambda key: (method_order(key[0]), as_float(key[1])))
    labels = [group_label(key[0], key[1]) for key in keys]
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 9.2), sharex=True)
    for ax, metric, ylabel in [(axes[0], ACC_KEY, ACC_LABEL), (axes[1], REG_KEY, REG_LABEL)]:
        values = [np.asarray([as_float(row[metric]) for row in groups[key]], dtype=np.float64) for key in keys]
        parts = ax.violinplot(values, positions=np.arange(len(keys)), widths=0.72, showmedians=True, showextrema=False)
        for body, key in zip(parts["bodies"], keys):
            color = color_for(key[0])
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#202020")
            parts["cmedians"].set_linewidth(1.0)
        for idx, key in enumerate(keys):
            color = color_for(key[0])
            vals = values[idx]
            jitter = rng.uniform(-0.22, 0.22, size=vals.size)
            ax.scatter(np.full(vals.size, idx) + jitter, vals, s=8, color=color, alpha=0.32, edgecolor="none")
            ax.scatter([idx + 0.34], [float(np.mean(vals))], s=30, color="#202020", marker="_", linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.24)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(labels, rotation=72, ha="right")
    fig.suptitle("Selected-point raincloud/strip diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path = out_dir / "selected_point_raincloud_strip.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Selected-Point Raincloud Strip",
        [
            "This combines distribution shape, individual case points, medians, and mean markers for selected operating points.",
            "",
            "It is a manuscript-friendly candidate if the goal is to show marginal spread without the full alpha-grid clutter.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def plot_paired_rank_dumbbell(case_pairs: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    rows = [row for row in selected_rows(case_pairs) if str(row["method_key"]) in SWEEP_METHODS]
    by_case: Dict[int, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[int(as_float(row["case_rank"]))].append(row)
    case_order = sorted(
        by_case,
        key=lambda rank: float(np.mean([as_float(row[REG_KEY]) for row in by_case[rank]])),
    )
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 8.2), sharex=True)
    for ax, metric, ylabel in [(axes[0], REG_KEY, REG_LABEL), (axes[1], ACC_KEY, ACC_LABEL)]:
        for method in SWEEP_METHODS:
            method_rows = {
                int(as_float(row["case_rank"])): as_float(row[metric])
                for row in rows
                if str(row["method_key"]) == method
            }
            values = np.asarray([method_rows[rank] for rank in case_order], dtype=np.float64)
            ax.plot(range(len(case_order)), values, color=color_for(method), linewidth=0.9, alpha=0.78, label=display_label(method))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
    axes[0].legend(fontsize=8, ncol=3, frameon=True)
    axes[1].set_xlabel("OASIS cases sorted by selected-method mean SDlogJ")
    tick_positions = list(range(0, len(case_order), 10))
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([str(case_order[idx]) for idx in tick_positions])
    fig.suptitle("Paired-rank case traces across selected methods", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "paired_rank_case_traces_selected_methods.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Paired-Rank Dumbbell/Trace Plot",
        [
            "Cases are sorted by mean SDlogJ across selected method points, then SDlogJ and Dice traces are shown for each method.",
            "",
            "This view is useful for seeing whether the same difficult cases drive high deformation irregularity and lower Dice.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def plot_hexbin_density(case_pairs: Sequence[Mapping[str, str]], summary_rows: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, method in zip(axes_flat, SWEEP_METHODS):
        method_rows = [row for row in case_pairs if str(row["method_key"]) == method]
        x, y = group_xy(method_rows)
        hb = ax.hexbin(x, y, gridsize=25, cmap="viridis", mincnt=1, linewidths=0.0)
        mean_rows = [row for row in summary_rows if str(row["method_key"]) == method]
        mean_rows.sort(key=lambda row: as_float(row["alpha"]))
        ax.plot(
            [as_float(row["sdlogj_mean"]) for row in mean_rows],
            [as_float(row["dice_mean"]) for row in mean_rows],
            color="white",
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            [as_float(row["sdlogj_mean"]) for row in mean_rows],
            [as_float(row["dice_mean"]) for row in mean_rows],
            color=color_for(method),
            edgecolor="white",
            linewidth=0.6,
            s=32,
            zorder=4,
        )
        ax.set_title(display_label(method), fontsize=10)
        ax.grid(True, alpha=0.14)
    for ax in axes[-1, :]:
        ax.set_xlabel(REG_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(ACC_LABEL)
    cbar = fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("case/alpha observations")
    fig.suptitle("Hexbin density by method with mean Pareto curve overlay", fontsize=14)
    out_path = out_dir / "hexbin_density_by_method_with_mean_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Hexbin Density",
        [
            "Each panel pools all case-level method/alpha observations for one method, then overlays the mean Pareto curve.",
            "",
            "This reduces overplotting and shows where case/alpha observations concentrate in the Dice-SDlogJ plane.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def draw_joint_panel(fig: plt.Figure, outer: gridspec.SubplotSpec, rows: Sequence[Mapping[str, str]], title: str, color: str) -> None:
    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer,
        width_ratios=[4, 1.05],
        height_ratios=[1.05, 4],
        wspace=0.04,
        hspace=0.04,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    x, y = group_xy(rows)
    ax_joint.scatter(x, y, s=13, color=color, alpha=0.45, edgecolor="none")
    line = regression_line(x, y)
    if line is not None:
        ax_joint.plot(line[0], line[1], color="#202020", linewidth=1.0)
    ax_histx.hist(x, bins=18, color=color, alpha=0.5)
    ax_histy.hist(y, bins=18, orientation="horizontal", color=color, alpha=0.5)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelsize=7)
    ax_histx.tick_params(axis="y", labelsize=7)
    ax_joint.grid(True, alpha=0.22)
    ax_joint.set_xlabel(REG_LABEL, fontsize=8)
    ax_joint.set_ylabel(ACC_LABEL, fontsize=8)
    ax_joint.tick_params(labelsize=7)
    ax_histx.set_title(title, fontsize=9)
    ax_joint.text(
        0.04,
        0.05,
        f"r={pearson(x, y):.2f}",
        transform=ax_joint.transAxes,
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )


def plot_joint_corner(case_pairs: Sequence[Mapping[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    groups = group_rows(selected_rows(case_pairs))
    keys = [(method, alpha_text(SELECTED_ALPHAS[method])) for method in SWEEP_METHODS]
    fig = plt.figure(figsize=(13.8, 8.8))
    outer = gridspec.GridSpec(2, 3, figure=fig, wspace=0.28, hspace=0.32)
    for idx, key in enumerate(keys):
        draw_joint_panel(fig, outer[idx], groups[key], group_label(key[0], key[1]), color_for(key[0]))
    fig.suptitle("Joint/corner-style selected method diagnostics", fontsize=14)
    out_path = out_dir / "joint_corner_selected_methods.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_readme(
        out_dir / "README.md",
        "Joint Corner Plots",
        [
            "Each selected method panel combines a case-level SDlogJ-vs-Dice scatter plot with marginal histograms.",
            "",
            "This is bulkier than the main Pareto plot, but it is very readable for a small set of key method comparisons.",
        ],
    )
    return [out_path, out_dir / "README.md"]


def write_root_readme(output_root: Path, manifest: Mapping[str, object]) -> None:
    write_readme(
        output_root / "README.md",
        "OASIS100 Dice-SDlogJ Innovative Visualization Trials",
        [
            "This exploratory package tries several alternatives to the dense ellipse Figure 2 style.",
            "",
            "All plots are derived from the existing paired OASIS100 case-level Dice/SDlogJ rows; no registrations or raw data are rerun.",
            "",
            "## Subfolders",
            "",
            "- `small_multiple_case_clouds/`: one selected method/reference panel at a time.",
            "- `correlation_slope_panels/`: Pearson/Spearman correlation trends over alpha.",
            "- `ridge_marginals/`: horizontal 1D distribution ridgelines.",
            "- `selected_point_raincloud_strip/`: selected-point marginal distributions plus case dots.",
            "- `paired_rank_dumbbell/`: cases sorted by SDlogJ with paired method traces.",
            "- `hexbin_density/`: pooled case/alpha density by method with mean curves.",
            "- `joint_corner_plots/`: selected method joint scatter plus marginal histograms.",
            "",
            "## Quick Take",
            "",
            "The most manuscript-friendly candidates are likely the selected raincloud/strip plot and small-multiple case clouds. The correlation slope panel is the most compact quantitative answer to the covariance question.",
            "",
            f"- Missing/empty output files: `{len(manifest['outputs']['missing_or_empty'])}`.",
        ],
    )


def main() -> None:
    args = parse_args()
    oasis_plus.configure_plot_module(float(f100.DCP842_LR))
    case_pairs = read_csv_rows(args.case_pairs)
    summary_rows = read_csv_rows(args.covariance_summary)
    output_root = Path(args.output_root)

    outputs: List[Path] = []
    outputs.extend(plot_small_multiple_case_clouds(case_pairs, output_root / "small_multiple_case_clouds", args.dpi))
    outputs.extend(plot_correlation_slope_panels(summary_rows, output_root / "correlation_slope_panels", args.dpi))
    outputs.extend(plot_ridge_marginals(case_pairs, output_root / "ridge_marginals", args.dpi))
    outputs.extend(plot_raincloud_strip(case_pairs, output_root / "selected_point_raincloud_strip", args.dpi))
    outputs.extend(plot_paired_rank_dumbbell(case_pairs, output_root / "paired_rank_dumbbell", args.dpi))
    outputs.extend(plot_hexbin_density(case_pairs, summary_rows, output_root / "hexbin_density", args.dpi))
    outputs.extend(plot_joint_corner(case_pairs, output_root / "joint_corner_plots", args.dpi))

    manifest = {
        "generated": c.now_text(),
        "artifact": "oasis100_dice_sdlogj_innovative_visualization_trials",
        "source_case_pairs": repo_path(Path(args.case_pairs)),
        "source_covariance_summary": repo_path(Path(args.covariance_summary)),
        "output_root": repo_path(output_root),
        "row_counts": {
            "case_pairs": len(case_pairs),
            "covariance_groups": len(summary_rows),
            "selected_case_pairs": len(selected_rows(case_pairs)),
        },
        "subfolders": [
            "small_multiple_case_clouds",
            "correlation_slope_panels",
            "ridge_marginals",
            "selected_point_raincloud_strip",
            "paired_rank_dumbbell",
            "hexbin_density",
            "joint_corner_plots",
        ],
        "outputs": output_status(outputs + [output_root / "README.md", output_root / "manifest.json"]),
    }
    write_json(output_root / "manifest.json", manifest)
    write_root_readme(output_root, manifest)
    manifest["outputs"] = output_status(outputs + [output_root / "README.md", output_root / "manifest.json"])
    write_json(output_root / "manifest.json", manifest)
    write_root_readme(output_root, manifest)
    print(json.dumps(manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
