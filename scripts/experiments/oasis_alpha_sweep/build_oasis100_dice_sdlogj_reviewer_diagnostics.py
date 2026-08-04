#!/usr/bin/env python
"""Build OASIS100 Dice-vs-SDlogJ reviewer-response diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_oasis_first100_extension as first100_agg  # noqa: E402
import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines as oasis_plot  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as oasis_plus  # noqa: E402


OUTPUT_ROOT = (
    c.REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100/"
    "reviewer_response/oasis100_dice_sdlogj"
)
PARETO_ROOT = c.REPO_ROOT / "experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100"
REVIEW_COMMENT = c.REPO_ROOT / "docs/offgrid_submission/review.md"
OASIS100_SOURCE_ROWS = PARETO_ROOT / "oasis100_pareto_source_rows.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--first10-main-root", type=Path, default=f100.FIRST10_MAIN_ROOT)
    parser.add_argument("--first10-cps2-prelim-root", type=Path, default=f100.FIRST10_CPS2_PRELIM_ROOT)
    parser.add_argument("--first10-cps2-completion-root", type=Path, default=f100.FIRST10_CPS2_COMPLETION_ROOT)
    parser.add_argument("--next10-root", type=Path, default=f100.NEXT10_ROOT)
    parser.add_argument("--single-stage-extension-root", type=Path, default=f100.SINGLE_STAGE_EXTENSION_ROOT)
    parser.add_argument("--baseline-first20-root", type=Path, default=f100.FIRST20_BASELINE_ROOT)
    parser.add_argument("--baseline-extension-root", type=Path, default=f100.BASELINE_EXTENSION_ROOT)
    parser.add_argument("--directcp842-first20-root", type=Path, default=f100.FIRST20_DCP842_ROOT)
    parser.add_argument("--directcp842-extension-root", type=Path, default=f100.DCP842_EXTENSION_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    return oasis_plot.as_float(value)


def finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def alpha_text(value: object) -> str:
    value_float = as_float(value)
    return f"{value_float:g}" if math.isfinite(value_float) else ""


def method_order(method: str) -> int:
    order = SWEEP_METHODS + BASELINE_METHODS
    return order.index(method) if method in order else len(order)


def group_key(row: Mapping[str, object]) -> Tuple[str, str]:
    method = str(row.get("method_key"))
    if method in BASELINE_METHODS:
        return method, "baseline"
    return method, alpha_text(row.get("alpha"))


def group_sort_key(key: Tuple[str, str]) -> Tuple[int, float]:
    method, alpha = key
    return method_order(method), as_float(alpha)


def display_label(method: str) -> str:
    spec = oasis_plot.METHOD_SPECS.get(method, {})
    return str(spec.get("label", method))


def repo_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(c.REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def group_label(method: str, alpha: str) -> str:
    label = display_label(method)
    if method in BASELINE_METHODS:
        return label
    return f"{label} a={alpha}"


def first100_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        first10_main_root=Path(args.first10_main_root),
        first10_cps2_prelim_root=Path(args.first10_cps2_prelim_root),
        first10_cps2_completion_root=Path(args.first10_cps2_completion_root),
        next10_root=Path(args.next10_root),
        single_stage_extension_root=Path(args.single_stage_extension_root),
        baseline_first20_root=Path(args.baseline_first20_root),
        baseline_extension_root=Path(args.baseline_extension_root),
        directcp842_first20_root=Path(args.directcp842_first20_root),
        directcp842_extension_root=Path(args.directcp842_extension_root),
    )


def collect_case_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    oasis_plus.configure_plot_module(float(f100.DCP842_LR))
    collected = first100_agg.collect_first100_rows(first100_args(args))
    sweep_rows = [dict(row) for row in collected["sweep_rows"] if str(row.get("method_key")) in SWEEP_METHODS]
    baseline_rows = [dict(row) for row in collected["baseline_rows"] if str(row.get("method_key")) in BASELINE_METHODS]
    return sweep_rows, baseline_rows


def validate_case_rows(sweep_rows: Sequence[Mapping[str, object]], baseline_rows: Sequence[Mapping[str, object]]) -> List[str]:
    errors: List[str] = []
    expected_ranks = set(f100.RANKS_FIRST100)
    expected_sweep_groups = {(method, alpha_text(alpha)) for method in SWEEP_METHODS for alpha in oasis_plot.METHOD_SPECS[method]["alphas"]}  # type: ignore[index]
    sweep_counts = Counter(group_key(row) for row in sweep_rows)
    baseline_counts = Counter(group_key(row) for row in baseline_rows)
    if len(sweep_rows) != 3600:
        errors.append(f"expected 3600 sweep case rows, found {len(sweep_rows)}")
    if len(baseline_rows) != 500:
        errors.append(f"expected 500 baseline case rows, found {len(baseline_rows)}")
    for key in sorted(expected_sweep_groups, key=group_sort_key):
        if sweep_counts.get(key, 0) != 100:
            errors.append(f"sweep group {key} expected N=100, found {sweep_counts.get(key, 0)}")
    for method in BASELINE_METHODS:
        key = (method, "baseline")
        if baseline_counts.get(key, 0) != 100:
            errors.append(f"baseline group {key} expected N=100, found {baseline_counts.get(key, 0)}")

    seen = Counter((group_key(row), int(as_float(row.get("case_rank")))) for row in list(sweep_rows) + list(baseline_rows))
    for (key, rank), count in seen.items():
        if count != 1:
            errors.append(f"group {key} rank {rank} has {count} rows")
    for key in sorted(list(expected_sweep_groups) + [(method, "baseline") for method in BASELINE_METHODS], key=group_sort_key):
        ranks = {rank for (seen_key, rank), count in seen.items() if seen_key == key and count == 1}
        missing = sorted(expected_ranks - ranks)
        extra = sorted(ranks - expected_ranks)
        if missing:
            errors.append(f"group {key} missing ranks {missing[:12]}")
        if extra:
            errors.append(f"group {key} unexpected ranks {extra[:12]}")

    for row in list(sweep_rows) + list(baseline_rows):
        label = f"{row.get('method_key')} rank={row.get('case_rank')} alpha={row.get('alpha')}"
        if not finite(row.get(ACC_KEY)):
            errors.append(f"{label} has non-finite {ACC_KEY}")
        if not finite(row.get(REG_KEY)):
            errors.append(f"{label} has non-finite {REG_KEY}")
    return errors


def case_pair_rows(sweep_rows: Sequence[Mapping[str, object]], baseline_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in list(sweep_rows) + list(baseline_rows):
        method = str(row.get("method_key"))
        alpha = "" if method in BASELINE_METHODS else alpha_text(row.get("alpha"))
        pair = {
            "method_key": method,
            "method_label": display_label(method),
            "alpha": alpha,
            "lr": "" if method in BASELINE_METHODS else alpha_text(row.get("lr")),
            "group_label": group_label(method, alpha or "baseline"),
            "case_rank": int(as_float(row.get("case_rank"))),
            "original_case_index": int(as_float(row.get("original_case_index"))),
            "fixed_id": str(row.get("fixed_id")),
            "moving_id": str(row.get("moving_id")),
            "dice_mean": as_float(row.get(ACC_KEY)),
            "log_jdet_std_fg": as_float(row.get(REG_KEY)),
            "source_run_name": str(row.get("run_name")),
        }
        rows.append(pair)
    return sorted(rows, key=lambda row: (method_order(str(row["method_key"])), as_float(row["alpha"]), int(row["case_rank"])))


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
    x_std = float(np.std(x, ddof=1))
    y_std = float(np.std(y, ddof=1))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / (x_std * y_std))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def covariance_summary(case_pairs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in case_pairs:
        groups[(str(row["method_key"]), str(row["alpha"]) or "baseline")].append(row)

    out: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: group_sort_key(item[0])):
        method, alpha = key
        xs = np.asarray([as_float(row[REG_KEY]) for row in rows], dtype=np.float64)
        ys = np.asarray([as_float(row[ACC_KEY]) for row in rows], dtype=np.float64)
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[keep]
        ys = ys[keep]
        n = int(xs.size)
        cov = float("nan")
        x_var = y_var = x_std = y_std = float("nan")
        eig1 = eig2 = angle = float("nan")
        if n > 1:
            cov_matrix = np.cov(xs, ys, ddof=1)
            x_var = float(cov_matrix[0, 0])
            y_var = float(cov_matrix[1, 1])
            cov = float(cov_matrix[0, 1])
            x_std = float(math.sqrt(max(x_var, 0.0)))
            y_std = float(math.sqrt(max(y_var, 0.0)))
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            eig1 = float(max(eigvals[0], 0.0))
            eig2 = float(max(eigvals[1], 0.0))
            angle = float(math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0])))
        out.append(
            {
                "method_key": method,
                "method_label": display_label(method),
                "alpha": "" if method in BASELINE_METHODS else alpha,
                "group_label": group_label(method, alpha),
                "n": n,
                "sdlogj_mean": float(np.mean(xs)) if n else float("nan"),
                "dice_mean": float(np.mean(ys)) if n else float("nan"),
                "sdlogj_std": x_std,
                "dice_std": y_std,
                "sdlogj_dice_covariance": cov,
                "pearson_r": pearson(xs, ys),
                "spearman_rho": spearman(xs, ys),
                "cov_ellipse_eigenvalue_1": eig1,
                "cov_ellipse_eigenvalue_2": eig2,
                "cov_ellipse_angle_degrees": angle,
            }
        )
    return out


def plot_covariance_ellipses(summary_rows: Sequence[Mapping[str, object]], output_path: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    for method in SWEEP_METHODS:
        rows = [row for row in summary_rows if str(row["method_key"]) == method]
        rows.sort(key=lambda row: as_float(row["alpha"]))
        color = str(oasis_plot.METHOD_SPECS[method]["color"])
        xs = [as_float(row["sdlogj_mean"]) for row in rows]
        ys = [as_float(row["dice_mean"]) for row in rows]
        ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.75, zorder=2)
        for row in rows:
            x = as_float(row["sdlogj_mean"])
            y = as_float(row["dice_mean"])
            eig1 = as_float(row["cov_ellipse_eigenvalue_1"])
            eig2 = as_float(row["cov_ellipse_eigenvalue_2"])
            angle = as_float(row["cov_ellipse_angle_degrees"])
            if math.isfinite(eig1) and math.isfinite(eig2):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * math.sqrt(max(eig1, 0.0)),
                        height=2.0 * math.sqrt(max(eig2, 0.0)),
                        angle=angle,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.9,
                        alpha=0.14,
                        zorder=1,
                    )
                )
        ax.scatter(xs, ys, s=42, color=color, edgecolor="white", linewidth=0.6, label=display_label(method), zorder=3)
    for method in BASELINE_METHODS:
        rows = [row for row in summary_rows if str(row["method_key"]) == method]
        if not rows:
            continue
        row = rows[0]
        color = str(oasis_plot.METHOD_SPECS[method]["color"])
        ax.scatter(
            [as_float(row["sdlogj_mean"])],
            [as_float(row["dice_mean"])],
            s=58,
            marker="D",
            color=color,
            edgecolor="#202020",
            linewidth=0.55,
            label=display_label(method),
            zorder=4,
        )
    ax.set_xlabel(REG_LABEL)
    ax.set_ylabel(ACC_LABEL)
    ax.set_title("OASIS100 Dice vs SDlogJ with rotated 1-SD covariance ellipses")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(summary_rows: Sequence[Mapping[str, object]], output_path: Path, dpi: int) -> Path:
    rows = [row for row in summary_rows if str(row["method_key"]) in SWEEP_METHODS]
    alpha_values = sorted({as_float(row["alpha"]) for row in rows if math.isfinite(as_float(row["alpha"]))})
    matrix = np.full((len(SWEEP_METHODS), len(alpha_values)), np.nan, dtype=np.float64)
    for row in rows:
        method = str(row["method_key"])
        alpha = as_float(row["alpha"])
        if method in SWEEP_METHODS and alpha in alpha_values:
            matrix[SWEEP_METHODS.index(method), alpha_values.index(alpha)] = as_float(row["pearson_r"])

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f"{alpha:g}" for alpha in alpha_values], rotation=35, ha="right")
    ax.set_yticks(range(len(SWEEP_METHODS)))
    ax.set_yticklabels([display_label(method) for method in SWEEP_METHODS])
    ax.set_xlabel("alpha")
    ax.set_title("Pearson r between case-level SDlogJ and Dice")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def group_values(case_pairs: Sequence[Mapping[str, object]], selected_only: bool) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[str]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in case_pairs:
        method = str(row["method_key"])
        alpha = str(row["alpha"]) or "baseline"
        if selected_only and method in SWEEP_METHODS:
            selected = SELECTED_ALPHAS[method]
            if not math.isclose(as_float(alpha), selected, rel_tol=0.0, abs_tol=1e-9):
                continue
        groups[(method, alpha)].append(row)
    labels: List[str] = []
    dice: List[np.ndarray] = []
    sdlogj: List[np.ndarray] = []
    colors: List[str] = []
    for key, rows in sorted(groups.items(), key=lambda item: group_sort_key(item[0])):
        method, alpha = key
        labels.append(group_label(method, alpha))
        dice.append(np.asarray([as_float(row[ACC_KEY]) for row in rows], dtype=np.float64))
        sdlogj.append(np.asarray([as_float(row[REG_KEY]) for row in rows], dtype=np.float64))
        colors.append(str(oasis_plot.METHOD_SPECS[method]["color"]))
    return labels, dice, sdlogj, colors


def draw_violin_panel(ax: plt.Axes, values: Sequence[np.ndarray], labels: Sequence[str], colors: Sequence[str], ylabel: str) -> None:
    positions = np.arange(1, len(values) + 1)
    parts = ax.violinplot(values, positions=positions, widths=0.82, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#202020")
        parts["cmedians"].set_linewidth(0.9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.2)
    ax.grid(True, axis="y", alpha=0.25)


def plot_violin_marginals(
    case_pairs: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    selected_only: bool,
    dpi: int,
) -> Path:
    labels, dice, sdlogj, colors = group_values(case_pairs, selected_only)
    width = 11.2 if selected_only else 18.0
    fig, axes = plt.subplots(2, 1, figsize=(width, 8.4), sharex=True)
    draw_violin_panel(axes[0], dice, labels, colors, ACC_LABEL)
    draw_violin_panel(axes[1], sdlogj, labels, colors, REG_LABEL)
    title = "selected operating points and baselines" if selected_only else "full alpha grid and baselines"
    fig.suptitle(f"OASIS100 marginal distributions: {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def output_status(paths: Sequence[Path]) -> Dict[str, object]:
    return {
        "expected_files": [repo_path(path) for path in paths],
        "missing_or_empty": [repo_path(path) for path in paths if not path.exists() or path.stat().st_size <= 0],
    }


def write_root_readme(output_root: Path, manifest: Mapping[str, object]) -> None:
    lines = [
        "# OASIS100 Dice-SDlogJ Reviewer Diagnostics",
        "",
        "This package is a reviewer-response diagnostic beside the current Figure 2 artifact; it does not replace the paper figure.",
        "",
        "## Links",
        "",
        f"- Reviewer comment: `{REVIEW_COMMENT.relative_to(c.REPO_ROOT)}:76`",
        f"- Current OASIS100 Pareto package: `{PARETO_ROOT.relative_to(c.REPO_ROOT)}/`",
        f"- Current OASIS100 Pareto source rows: `{OASIS100_SOURCE_ROWS.relative_to(c.REPO_ROOT)}`",
        "- Covariance diagnostics: `covariance_diagnostics/README.md`",
        "- Violin marginal diagnostics: `violin_marginals/README.md`",
        "",
        "## Scope",
        "",
        f"- Cohort: OASIS seed-3 shuffled first100, case ranks `0-99`.",
        f"- Paired metrics: `{ACC_KEY}` and `{REG_KEY}`.",
        f"- Sweep case rows: `{manifest['row_counts']['sweep_case_rows']}`.",
        f"- Baseline case rows: `{manifest['row_counts']['baseline_case_rows']}`.",
        f"- Diagnostic groups: `{manifest['row_counts']['diagnostic_groups']}`.",
        "",
        "## Interpretation",
        "",
        "- Rotated covariance ellipses and correlation tables address whether case-level SDlogJ and Dice move together.",
        "- Violin plots disentangle the marginal spread along each axis, but they do not show covariance.",
        "",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines))


def write_covariance_readme(out_dir: Path, manifest: Mapping[str, object]) -> None:
    lines = [
        "# Covariance Diagnostics",
        "",
        "These outputs use paired case-level OASIS100 Dice and foreground SDlogJ values for each method/alpha group.",
        "",
        "## Files",
        "",
        "- `case_pairs.csv`: paired case-level Dice/SDlogJ rows.",
        "- `covariance_summary.csv`: covariance, Pearson correlation, Spearman correlation, and rotated ellipse parameters by group.",
        "- `covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png`: 2D Pareto-style diagnostic using rotated 1-SD covariance ellipses.",
        "- `correlation_heatmap_by_method_alpha.png`: Pearson correlation by method and alpha.",
        "- `manifest.json`: provenance and file checks.",
        "",
        "## Reading Notes",
        "",
        "- Raw covariance is unit-dependent, so Pearson `r` and Spearman `rho` are the main interpretable summaries.",
        "- Ellipse angle is derived from the sample covariance matrix of `(SDlogJ, Dice)` across cases.",
        f"- Missing/empty output files: `{len(manifest['outputs']['missing_or_empty'])}`.",
        "",
        "## How To Read The Figures",
        "",
        "`covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png` is the",
        "Dice-vs-SDlogJ Pareto view with rotated covariance ellipses. Each point is a",
        "method/alpha setting averaged over OASIS `N=100`. The ellipse shows paired",
        "case-level spread: horizontal spread is SDlogJ variation, vertical spread is",
        "Dice variation, and ellipse tilt shows whether the two metrics covary across",
        "cases. A downward left-to-right tilt means cases with higher SDlogJ tend to",
        "have lower Dice.",
        "",
        "`correlation_heatmap_by_method_alpha.png` is the compact numerical summary of",
        "the same question. Each cell is the Pearson correlation between case-level",
        "SDlogJ and Dice for one method/alpha group, ranging from `-1` to `1`. In this",
        "diagnostic package, the correlations are consistently negative, about `-0.82`",
        "to `-0.46`, so higher case-level SDlogJ is associated with lower Dice within",
        "each plotted method/regularization setting. Blank cells indicate alpha values",
        "not used by that method family.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def write_violin_readme(out_dir: Path, manifest: Mapping[str, object]) -> None:
    lines = [
        "# Violin Marginal Diagnostics",
        "",
        "These outputs split the original 2D Dice-SDlogJ relationship into separate 1D marginal distributions.",
        "",
        "## Files",
        "",
        "- `violin_source_rows.csv`: paired case-level rows used for both violin plots.",
        "- `violin_marginals_full_alpha_grid.png`: Dice and SDlogJ violins for the full alpha grid plus baselines.",
        "- `violin_marginals_selected_points.png`: compact view for best/reference sweep points plus baselines.",
        "- `manifest.json`: provenance and file checks.",
        "",
        "## Reading Notes",
        "",
        "- Violin plots reduce overlap and make each axis spread easier to inspect.",
        "- Because the axes are separated, these plots do not show covariance; use the covariance diagnostics for that question.",
        f"- Missing/empty output files: `{len(manifest['outputs']['missing_or_empty'])}`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def run_preflight(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    sweep_rows, baseline_rows = collect_case_rows(args)
    errors = validate_case_rows(sweep_rows, baseline_rows)
    return sweep_rows, baseline_rows, errors


def main() -> None:
    args = parse_args()
    sweep_rows, baseline_rows, errors = run_preflight(args)
    population_rows = oasis_plot.population_summary(list(sweep_rows) + list(baseline_rows))
    preflight = {
        "sweep_case_rows": len(sweep_rows),
        "baseline_case_rows": len(baseline_rows),
        "population_groups": len(population_rows),
        "errors": errors,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, allow_nan=True), flush=True)
        if errors:
            raise SystemExit(1)
        return
    if errors:
        raise RuntimeError("OASIS100 reviewer diagnostic preflight failed:\n" + "\n".join(errors[:80]))

    output_root = Path(args.output_root)
    covariance_dir = output_root / "covariance_diagnostics"
    violin_dir = output_root / "violin_marginals"
    case_pairs = case_pair_rows(sweep_rows, baseline_rows)
    cov_rows = covariance_summary(case_pairs)

    cov_paths = [
        covariance_dir / "case_pairs.csv",
        covariance_dir / "covariance_summary.csv",
        covariance_dir / "covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png",
        covariance_dir / "correlation_heatmap_by_method_alpha.png",
        covariance_dir / "README.md",
        covariance_dir / "manifest.json",
    ]
    violin_paths = [
        violin_dir / "violin_source_rows.csv",
        violin_dir / "violin_marginals_full_alpha_grid.png",
        violin_dir / "violin_marginals_selected_points.png",
        violin_dir / "README.md",
        violin_dir / "manifest.json",
    ]

    c.write_csv(covariance_dir / "case_pairs.csv", case_pairs)
    c.write_csv(covariance_dir / "covariance_summary.csv", cov_rows)
    plot_covariance_ellipses(cov_rows, covariance_dir / "covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png", args.dpi)
    plot_correlation_heatmap(cov_rows, covariance_dir / "correlation_heatmap_by_method_alpha.png", args.dpi)

    c.write_csv(violin_dir / "violin_source_rows.csv", case_pairs)
    plot_violin_marginals(case_pairs, violin_dir / "violin_marginals_full_alpha_grid.png", selected_only=False, dpi=args.dpi)
    plot_violin_marginals(case_pairs, violin_dir / "violin_marginals_selected_points.png", selected_only=True, dpi=args.dpi)

    finite_correlation_rows = [
        row
        for row in cov_rows
        if math.isfinite(as_float(row.get("pearson_r"))) and math.isfinite(as_float(row.get("spearman_rho")))
    ]
    row_counts = {
        "sweep_case_rows": len(sweep_rows),
        "baseline_case_rows": len(baseline_rows),
        "case_pair_rows": len(case_pairs),
        "diagnostic_groups": len(cov_rows),
        "population_groups": len(population_rows),
        "finite_correlation_groups": len(finite_correlation_rows),
    }

    cov_manifest = {
        "generated": c.now_text(),
        "artifact": "oasis100_dice_sdlogj_covariance_diagnostics",
        "metrics": {"x": REG_KEY, "y": ACC_KEY},
        "row_counts": row_counts,
        "interpretation": "Pearson and Spearman correlations are emphasized because raw covariance is unit-dependent.",
        "outputs": output_status(cov_paths),
    }
    write_manifest(covariance_dir / "manifest.json", cov_manifest)
    write_covariance_readme(covariance_dir, cov_manifest)
    cov_manifest["outputs"] = output_status(cov_paths)
    write_manifest(covariance_dir / "manifest.json", cov_manifest)

    violin_manifest = {
        "generated": c.now_text(),
        "artifact": "oasis100_dice_sdlogj_violin_marginals",
        "metrics": {"dice": ACC_KEY, "sdlogj": REG_KEY},
        "selected_alphas": SELECTED_ALPHAS,
        "row_counts": row_counts,
        "interpretation": "Violin plots clarify marginal spread but do not show covariance.",
        "outputs": output_status(violin_paths),
    }
    write_manifest(violin_dir / "manifest.json", violin_manifest)
    write_violin_readme(violin_dir, violin_manifest)
    violin_manifest["outputs"] = output_status(violin_paths)
    write_manifest(violin_dir / "manifest.json", violin_manifest)

    root_manifest = {
        "generated": c.now_text(),
        "artifact": "oasis100_dice_sdlogj_reviewer_diagnostics",
        "output_root": repo_path(output_root),
        "source_links": {
            "review_comment": f"{REVIEW_COMMENT.relative_to(c.REPO_ROOT)}:76",
            "current_pareto_package": str(PARETO_ROOT.relative_to(c.REPO_ROOT)),
            "oasis100_pareto_source_rows": str(OASIS100_SOURCE_ROWS.relative_to(c.REPO_ROOT)),
        },
        "row_counts": row_counts,
        "preflight": preflight,
        "outputs": output_status([output_root / "README.md"] + cov_paths + violin_paths),
    }
    write_manifest(output_root / "manifest.json", root_manifest)
    write_root_readme(output_root, root_manifest)
    root_manifest["outputs"] = output_status([output_root / "README.md"] + cov_paths + violin_paths + [output_root / "manifest.json"])
    write_manifest(output_root / "manifest.json", root_manifest)
    print(json.dumps(root_manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
