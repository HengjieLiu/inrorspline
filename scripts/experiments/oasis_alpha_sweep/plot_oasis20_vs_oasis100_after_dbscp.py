#!/usr/bin/env python
"""Build an OASIS20/OASIS100 2x2 Pareto comparison after D-BSCP cps4 completion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

PAPER_DIR = THIS_DIR.parents[2] / "experiments" / "paper_artifacts"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import aggregate_oasis_first100_extension as first100_agg  # noqa: E402
import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines as oasis_plot  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as oasis_plus  # noqa: E402
from paper_artifact_common import display_name, load_nicknames, repo_relative, sanitize_public_value  # noqa: E402


PAPER_PARETO_ROOT = c.REPO_ROOT / "experiments/paper_artifacts/pareto_figure/v002_candidate"
METHOD_SET_CONFIGS = {
    "after-dbscp-no-inrbscp-cps4": {
        "artifact_slug": "oasis20_vs_oasis100_after_dbscp_no_inrbscp_cps4",
        "pointer_parent": f100.REVIEW_ROOT / "partial_figures",
        "sweep_methods": [
            "idir_original_dense_siren_be_autograd",
            "direct_cp_cps4_be_bspline_analytic",
            "sinr_cps2_be_bspline_analytic",
            "direct_cp_cps2_be_bspline_analytic",
            f100.DCP842_METHOD,
        ],
        "excluded_methods": ["sinr_cps4_be_bspline_analytic"],
        "output_filenames": {
            "no_annotations": "oasis20_vs_oasis100_pareto_2x2_ellipses_no_annotations.png",
            "annotated_review": "oasis20_vs_oasis100_pareto_2x2_ellipses_annotated_review.png",
        },
        "source_rows_filename": "oasis20_vs_oasis100_source_rows.csv",
        "manifest_filename": "oasis20_vs_oasis100_pareto_2x2_manifest.json",
        "report_note": "- INR-BSCP cps4 is excluded from both cohorts.",
    },
    "complete-all": {
        "artifact_slug": "oasis20_vs_oasis100_complete_all_methods",
        "pointer_parent": f100.REVIEW_ROOT / "final_completion_figures",
        "sweep_methods": [
            "idir_original_dense_siren_be_autograd",
            "sinr_cps4_be_bspline_analytic",
            "direct_cp_cps4_be_bspline_analytic",
            "sinr_cps2_be_bspline_analytic",
            "direct_cp_cps2_be_bspline_analytic",
            f100.DCP842_METHOD,
        ],
        "excluded_methods": [],
        "output_filenames": {
            "no_annotations": "oasis20_vs_oasis100_complete_pareto_2x2_ellipses_no_annotations.png",
            "annotated_review": "oasis20_vs_oasis100_complete_pareto_2x2_ellipses_annotated_review.png",
        },
        "source_rows_filename": "oasis20_vs_oasis100_complete_source_rows.csv",
        "manifest_filename": "oasis20_vs_oasis100_complete_pareto_2x2_manifest.json",
        "report_note": "- INR-BSCP cps4 is included because the full first100 run is complete.",
    },
}
ARTIFACT_SLUG = str(METHOD_SET_CONFIGS["after-dbscp-no-inrbscp-cps4"]["artifact_slug"])
SOURCE_ROWS_FILENAME = str(METHOD_SET_CONFIGS["after-dbscp-no-inrbscp-cps4"]["source_rows_filename"])
MANIFEST_FILENAME = str(METHOD_SET_CONFIGS["after-dbscp-no-inrbscp-cps4"]["manifest_filename"])
REPORT_METHOD_NOTE = str(METHOD_SET_CONFIGS["after-dbscp-no-inrbscp-cps4"]["report_note"])
DEFAULT_OASIS20_SOURCE = (
    c.REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v002_candidate/oasis20_pareto_source_rows.csv"
)
DEFAULT_NICKNAME_CSV = c.REPO_ROOT / "experiments/paper_artifacts/method_nicknames.csv"

EXCLUDED_METHODS = ["sinr_cps4_be_bspline_analytic"]
INCLUDED_SWEEP_METHODS = [
    "idir_original_dense_siren_be_autograd",
    "direct_cp_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    f100.DCP842_METHOD,
]
INCLUDED_BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]
EXPECTED_POPULATION_ROWS = 35
LOGJ_LABEL = "SDlogJ"
PANELS = [
    {
        "cohort": "OASIS20",
        "title": f"Dice vs {LOGJ_LABEL} (OASIS N=20)",
        "acc_key": "dice_mean",
    },
    {
        "cohort": "OASIS20",
        "title": f"HD95 vs {LOGJ_LABEL} (OASIS N=20)",
        "acc_key": "hd95_mean",
    },
    {
        "cohort": "OASIS100",
        "title": f"Dice vs {LOGJ_LABEL} (OASIS N=100)",
        "acc_key": "dice_mean",
    },
    {
        "cohort": "OASIS100",
        "title": f"HD95 vs {LOGJ_LABEL} (OASIS N=100)",
        "acc_key": "hd95_mean",
    },
]
FIGURE_STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 18,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 8.5,
    "legend_frame": True,
    "ellipse_alpha": 0.14,
    "grid_alpha": 0.25,
    "wspace": 0.18,
    "hspace": 0.30,
}
OUTPUTS = {
    "no_annotations": {
        "filename": "oasis20_vs_oasis100_pareto_2x2_ellipses_no_annotations.png",
        "annotate": False,
    },
    "annotated_review": {
        "filename": "oasis20_vs_oasis100_pareto_2x2_ellipses_annotated_review.png",
        "annotate": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method-set",
        choices=sorted(METHOD_SET_CONFIGS),
        default="after-dbscp-no-inrbscp-cps4",
        help="Method inclusion set for the comparison figure.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pointer-dir", type=Path, default=None)
    parser.add_argument("--oasis20-source", type=Path, default=DEFAULT_OASIS20_SOURCE)
    parser.add_argument("--nickname-csv", type=Path, default=DEFAULT_NICKNAME_CSV)
    parser.add_argument("--first10-main-root", type=Path, default=f100.FIRST10_MAIN_ROOT)
    parser.add_argument("--first10-cps2-prelim-root", type=Path, default=f100.FIRST10_CPS2_PRELIM_ROOT)
    parser.add_argument("--first10-cps2-completion-root", type=Path, default=f100.FIRST10_CPS2_COMPLETION_ROOT)
    parser.add_argument("--next10-root", type=Path, default=f100.NEXT10_ROOT)
    parser.add_argument("--single-stage-extension-root", type=Path, default=f100.SINGLE_STAGE_EXTENSION_ROOT)
    parser.add_argument("--baseline-first20-root", type=Path, default=f100.FIRST20_BASELINE_ROOT)
    parser.add_argument("--baseline-extension-root", type=Path, default=f100.BASELINE_EXTENSION_ROOT)
    parser.add_argument("--directcp842-first20-root", type=Path, default=f100.FIRST20_DCP842_ROOT)
    parser.add_argument("--directcp842-extension-root", type=Path, default=f100.DCP842_EXTENSION_ROOT)
    parser.add_argument("--dpi", type=int, default=int(FIGURE_STYLE["dpi"]))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def apply_method_set(args: argparse.Namespace) -> None:
    global ARTIFACT_SLUG
    global SOURCE_ROWS_FILENAME
    global MANIFEST_FILENAME
    global REPORT_METHOD_NOTE
    global EXCLUDED_METHODS
    global INCLUDED_SWEEP_METHODS
    global EXPECTED_POPULATION_ROWS
    config = METHOD_SET_CONFIGS[str(args.method_set)]
    ARTIFACT_SLUG = str(config["artifact_slug"])
    SOURCE_ROWS_FILENAME = str(config["source_rows_filename"])
    MANIFEST_FILENAME = str(config["manifest_filename"])
    REPORT_METHOD_NOTE = str(config["report_note"])
    EXCLUDED_METHODS = list(config["excluded_methods"])  # type: ignore[arg-type]
    INCLUDED_SWEEP_METHODS = list(config["sweep_methods"])  # type: ignore[arg-type]
    EXPECTED_POPULATION_ROWS = len(INCLUDED_BASELINE_METHODS) + 6 * len(INCLUDED_SWEEP_METHODS)
    output_filenames = config["output_filenames"]  # type: ignore[assignment]
    for mode_name, mode in OUTPUTS.items():
        mode["filename"] = str(output_filenames[mode_name])  # type: ignore[index]
    if args.output_dir is None:
        args.output_dir = PAPER_PARETO_ROOT / ARTIFACT_SLUG
    if args.pointer_dir is None:
        args.pointer_dir = Path(config["pointer_parent"]) / ARTIFACT_SLUG


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with Path(path).open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def alpha_text(value: object) -> str:
    return oasis_plot.alpha_id(value)


def method_sort_key(method: str) -> int:
    order = INCLUDED_SWEEP_METHODS + INCLUDED_BASELINE_METHODS
    return order.index(method) if method in order else len(order)


def population_sort_key(row: Mapping[str, object]) -> Tuple[int, float]:
    method = str(row.get("method_key"))
    alpha = oasis_plot.as_float(row.get("alpha"))
    return method_sort_key(method), alpha if math.isfinite(alpha) else math.inf


def configure_oasis_plotting(nickname_csv: Path) -> None:
    oasis_plus.configure_plot_module(float(f100.DCP842_LR))
    oasis_plot.SWEEP_METHODS[:] = list(INCLUDED_SWEEP_METHODS)
    oasis_plot.BASELINE_METHODS[:] = list(INCLUDED_BASELINE_METHODS)
    oasis_plot.ACCURACY_METRICS = [
        ("dice_mean", "Dice", True),
        ("hd95_mean", "HD95", False),
    ]
    oasis_plot.REGULARITY_METRICS = [("log_jdet_std_fg", LOGJ_LABEL)]
    oasis_plot.SPOTLIGHT_PANELS = [("dice_mean", "log_jdet_std_fg"), ("hd95_mean", "log_jdet_std_fg")]
    nicknames = load_nicknames(nickname_csv)
    for method_id, spec in oasis_plot.METHOD_SPECS.items():
        original_label = str(spec.get("label", method_id))
        original_dot = str(spec.get("dot_label", original_label))
        spec["label"] = display_name(method_id, original_label, nicknames)
        spec["dot_label"] = display_name(method_id, original_dot, nicknames)


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


def filter_population_rows(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    allowed = set(INCLUDED_SWEEP_METHODS + INCLUDED_BASELINE_METHODS)
    out = [dict(row) for row in rows if str(row.get("method_key")) in allowed]
    return sorted(out, key=population_sort_key)


def filter_first100_case_rows(rows: Mapping[str, Sequence[Dict[str, object]]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    sweep = [dict(row) for row in rows["sweep_rows"] if str(row.get("method_key")) in INCLUDED_SWEEP_METHODS]
    baselines = [dict(row) for row in rows["baseline_rows"] if str(row.get("method_key")) in INCLUDED_BASELINE_METHODS]
    return sweep, baselines


def validate_oasis20_rows(rows: Sequence[Mapping[str, object]]) -> List[str]:
    errors: List[str] = []
    observed = Counter(str(row.get("method_key")) for row in rows)
    if len(rows) != EXPECTED_POPULATION_ROWS:
        errors.append(f"OASIS20: expected {EXPECTED_POPULATION_ROWS} population rows, found {len(rows)}")
    for method in INCLUDED_SWEEP_METHODS:
        if observed.get(method, 0) != 6:
            errors.append(f"OASIS20: method {method} expected 6 rows, found {observed.get(method, 0)}")
    for method in INCLUDED_BASELINE_METHODS:
        if observed.get(method, 0) != 1:
            errors.append(f"OASIS20: baseline {method} expected 1 row, found {observed.get(method, 0)}")
    for method in EXCLUDED_METHODS:
        if observed.get(method, 0):
            errors.append(f"OASIS20: excluded method {method} appears in output rows")
    for row in rows:
        if int(oasis_plot.as_float(row.get("n_cases"))) != 20:
            errors.append(f"OASIS20: {row.get('method_key')} n_cases {row.get('n_cases')} != 20")
    errors.extend(validate_plot_metrics(rows, "OASIS20"))
    return errors


def validate_first100_case_rows(
    sweep_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[str], Dict[str, object]]:
    errors: List[str] = []
    expected_ranks = set(f100.RANKS_FIRST100)
    sweep_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    dcp4_counts: Dict[str, int] = {}
    duplicate_sweeps = Counter(
        (
            str(row.get("method_key")),
            int(oasis_plot.as_float(row.get("case_rank"))),
            alpha_text(row.get("alpha")),
        )
        for row in sweep_rows
    )
    for combo, count in duplicate_sweeps.items():
        if count != 1:
            errors.append(f"OASIS100: duplicate sweep case row {combo}: {count}")
    grouped_sweeps: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    for row in sweep_rows:
        grouped_sweeps[(str(row.get("method_key")), alpha_text(row.get("alpha")))].add(
            int(oasis_plot.as_float(row.get("case_rank")))
        )
    for method in INCLUDED_SWEEP_METHODS:
        for alpha in oasis_plot.METHOD_SPECS[method]["alphas"]:  # type: ignore[index]
            key = alpha_text(alpha)
            ranks = grouped_sweeps.get((method, key), set())
            sweep_counts[method][key] = len(ranks)
            if method == "direct_cp_cps4_be_bspline_analytic":
                dcp4_counts[key] = len(ranks)
            missing = sorted(expected_ranks - ranks)
            extra = sorted(ranks - expected_ranks)
            if missing:
                errors.append(f"OASIS100: {method} alpha {key} missing ranks {missing[:12]}")
            if extra:
                errors.append(f"OASIS100: {method} alpha {key} has unexpected ranks {extra[:12]}")
    duplicate_baselines = Counter(
        (str(row.get("method_key")), int(oasis_plot.as_float(row.get("case_rank"))))
        for row in baseline_rows
    )
    for combo, count in duplicate_baselines.items():
        if count != 1:
            errors.append(f"OASIS100: duplicate baseline case row {combo}: {count}")
    grouped_baselines: Dict[str, set[int]] = defaultdict(set)
    for row in baseline_rows:
        grouped_baselines[str(row.get("method_key"))].add(int(oasis_plot.as_float(row.get("case_rank"))))
    baseline_counts: Dict[str, int] = {}
    for method in INCLUDED_BASELINE_METHODS:
        ranks = grouped_baselines.get(method, set())
        baseline_counts[method] = len(ranks)
        missing = sorted(expected_ranks - ranks)
        extra = sorted(ranks - expected_ranks)
        if missing:
            errors.append(f"OASIS100: baseline {method} missing ranks {missing[:12]}")
        if extra:
            errors.append(f"OASIS100: baseline {method} has unexpected ranks {extra[:12]}")
    summary = {
        "sweep_counts": dict(sweep_counts),
        "baseline_counts": baseline_counts,
        "direct_cp_cps4_counts": dcp4_counts,
    }
    return errors, summary


def validate_oasis100_population(rows: Sequence[Mapping[str, object]]) -> List[str]:
    errors: List[str] = []
    observed = Counter(str(row.get("method_key")) for row in rows)
    if len(rows) != EXPECTED_POPULATION_ROWS:
        errors.append(f"OASIS100: expected {EXPECTED_POPULATION_ROWS} population rows, found {len(rows)}")
    for method in INCLUDED_SWEEP_METHODS:
        if observed.get(method, 0) != 6:
            errors.append(f"OASIS100: method {method} expected 6 rows, found {observed.get(method, 0)}")
    for method in INCLUDED_BASELINE_METHODS:
        if observed.get(method, 0) != 1:
            errors.append(f"OASIS100: baseline {method} expected 1 row, found {observed.get(method, 0)}")
    for method in EXCLUDED_METHODS:
        if observed.get(method, 0):
            errors.append(f"OASIS100: excluded method {method} appears in output rows")
    for row in rows:
        if int(oasis_plot.as_float(row.get("n_cases"))) != 100:
            errors.append(f"OASIS100: {row.get('method_key')} n_cases {row.get('n_cases')} != 100")
    errors.extend(validate_plot_metrics(rows, "OASIS100"))
    return errors


def validate_plot_metrics(rows: Sequence[Mapping[str, object]], cohort: str) -> List[str]:
    errors: List[str] = []
    required = [
        "dice_mean_mean",
        "dice_mean_std",
        "hd95_mean_mean",
        "hd95_mean_std",
        "log_jdet_std_fg_mean",
        "log_jdet_std_fg_std",
    ]
    for row in rows:
        label = f"{row.get('method_key')} alpha={row.get('alpha')}"
        for key in required:
            if not oasis_plot.is_finite(row.get(key)):
                errors.append(f"{cohort}: {label} has non-finite {key}")
    return errors


def build_oasis100_population(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    collected = first100_agg.collect_first100_rows(first100_args(args))
    sweep_rows, baseline_rows = filter_first100_case_rows(collected)
    coverage_errors, coverage_summary = validate_first100_case_rows(sweep_rows, baseline_rows)
    if coverage_errors:
        raise RuntimeError(
            "OASIS100 preflight failed; one or more included first100 groups are incomplete:\n"
            + "\n".join(coverage_errors[:80])
        )
    population_rows = filter_population_rows(oasis_plot.population_summary(sweep_rows + baseline_rows))
    pop_errors = validate_oasis100_population(population_rows)
    if pop_errors:
        raise RuntimeError("OASIS100 population validation failed:\n" + "\n".join(pop_errors[:80]))
    summary = {
        "case_sweep_rows": len(sweep_rows),
        "case_baseline_rows": len(baseline_rows),
        "population_rows": len(population_rows),
        "coverage": coverage_summary,
    }
    return population_rows, summary


def build_oasis20_population(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = filter_population_rows(read_csv_rows(Path(args.oasis20_source)))
    errors = validate_oasis20_rows(rows)
    if errors:
        raise RuntimeError("OASIS20 source validation failed:\n" + "\n".join(errors[:80]))
    return rows, {"population_rows": len(rows), "source": str(args.oasis20_source)}


def patch_oasis_ellipse_alpha(alpha: float) -> None:
    def add_ellipse(ax: plt.Axes, row: Mapping[str, object], x: float, y: float, reg_key: str, acc_key: str, color: str) -> None:
        x_std = max(oasis_plot.as_float(row.get(oasis_plot.metric_std(reg_key))), 0.0)
        y_std = max(oasis_plot.as_float(row.get(oasis_plot.metric_std(acc_key))), 0.0)
        if not math.isfinite(x_std) or not math.isfinite(y_std) or (x_std <= 0 and y_std <= 0):
            return
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * x_std,
                height=2.0 * y_std,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=alpha,
                zorder=1,
            )
        )

    oasis_plot.add_ellipse = add_ellipse  # type: ignore[assignment]


def apply_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=int(FIGURE_STYLE["tick_label_fontsize"]))
    ax.xaxis.label.set_size(int(FIGURE_STYLE["axis_label_fontsize"]))
    ax.yaxis.label.set_size(int(FIGURE_STYLE["axis_label_fontsize"]))
    ax.grid(True, alpha=float(FIGURE_STYLE["grid_alpha"]))


def normalize_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="best",
        ncol=2,
        fontsize=float(FIGURE_STYLE["legend_fontsize"]),
        frameon=bool(FIGURE_STYLE["legend_frame"]),
    )


def draw_oasis_panel(
    ax: plt.Axes,
    rows: Sequence[Dict[str, object]],
    *,
    title: str,
    acc_key: str,
    annotate: bool,
) -> None:
    oasis_plot.draw_panel(
        ax,
        rows,
        acc_key=acc_key,
        reg_key="log_jdet_std_fg",
        style="ellipses",
        show_legend=True,
        annotate=annotate,
    )
    ax.set_title(title, fontsize=int(FIGURE_STYLE["title_fontsize"]))
    apply_axis_style(ax)
    normalize_legend(ax)


def padded_limits(values: Sequence[float], *, lower_zero: bool) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return (0.0, 1.0)
    low = min(finite)
    high = max(finite)
    pad = 0.06 * max(high - low, 1e-8)
    low = low - pad
    high = high + pad
    if lower_zero:
        low = max(0.0, low)
    return low, high


def extent_values(
    rows_by_cohort: Sequence[Sequence[Mapping[str, object]]],
    *,
    acc_key: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for rows in rows_by_cohort:
        row_xs, row_ys = oasis_plot.collect_extents(rows, "log_jdet_std_fg", acc_key, include_std=True)
        xs.extend(row_xs)
        ys.extend(row_ys)
    return xs, ys


def apply_shared_limits(
    axes: Sequence[Sequence[plt.Axes]],
    oasis20_rows: Sequence[Mapping[str, object]],
    oasis100_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    x_dice, y_dice = extent_values([oasis20_rows, oasis100_rows], acc_key="dice_mean")
    x_hd95, y_hd95 = extent_values([oasis20_rows, oasis100_rows], acc_key="hd95_mean")
    xlim = padded_limits(x_dice + x_hd95, lower_zero=True)
    dice_ylim = padded_limits(y_dice, lower_zero=True)
    hd95_ylim = padded_limits(y_hd95, lower_zero=True)
    for row_axes in axes:
        for ax in row_axes:
            ax.set_xlim(*xlim)
    axes[0][0].set_ylim(*dice_ylim)
    axes[1][0].set_ylim(*dice_ylim)
    axes[0][1].set_ylim(*hd95_ylim)
    axes[1][1].set_ylim(*hd95_ylim)
    return {"xlim": xlim, "dice_ylim": dice_ylim, "hd95_ylim": hd95_ylim}


def write_source_rows(path: Path, oasis20_rows: Sequence[Mapping[str, object]], oasis100_rows: Sequence[Mapping[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for cohort, cohort_rows in [("oasis20", oasis20_rows), ("oasis100", oasis100_rows)]:
        for row in cohort_rows:
            row_out = {"cohort": cohort}
            row_out.update(dict(row))
            rows.append(row_out)
    c.write_csv(path, rows)


def render_figures(
    args: argparse.Namespace,
    oasis20_rows: Sequence[Dict[str, object]],
    oasis100_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Path], Dict[str, object]]:
    patch_oasis_ellipse_alpha(float(FIGURE_STYLE["ellipse_alpha"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_outputs: List[Path] = []
    limits: Dict[str, object] = {}
    rows_by_name = {"OASIS20": list(oasis20_rows), "OASIS100": list(oasis100_rows)}
    for _mode_name, mode in OUTPUTS.items():
        fig, axes = plt.subplots(2, 2, figsize=tuple(FIGURE_STYLE["figsize"]), squeeze=False)
        for ax, panel in zip(axes.ravel(), PANELS):
            draw_oasis_panel(
                ax,
                rows_by_name[str(panel["cohort"])],
                title=str(panel["title"]),
                acc_key=str(panel["acc_key"]),
                annotate=bool(mode["annotate"]),
            )
        limits = apply_shared_limits(axes, oasis20_rows, oasis100_rows)
        fig.subplots_adjust(
            wspace=float(FIGURE_STYLE["wspace"]),
            hspace=float(FIGURE_STYLE["hspace"]),
        )
        output_path = output_dir / str(mode["filename"])
        fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        saved_outputs.append(output_path)
    return saved_outputs, limits


def assert_outputs(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not Path(path).exists() or Path(path).stat().st_size <= 0]


def write_review_report(
    output_dir: Path,
    pointer_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    lines = [
        "# OASIS20 vs OASIS100 2x2 Pareto Comparison",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- Layout: OASIS20 on top, OASIS100 on bottom; Dice left, HD95 right.",
        "- Style: ellipse Pareto panels matched to the v002 candidate paper artifacts.",
        "- OASIS100 includes only complete 100-case groups.",
        REPORT_METHOD_NOTE,
        "",
        "## Counts",
        "",
        f"- OASIS20 population rows: `{manifest['row_counts']['oasis20_population_rows']}`",
        f"- OASIS100 population rows: `{manifest['row_counts']['oasis100_population_rows']}`",
        f"- Expected rows per cohort: `{EXPECTED_POPULATION_ROWS}`",
        "",
        "## Outputs",
        "",
    ]
    for output in manifest["outputs"]:  # type: ignore[index]
        lines.append(f"- `{output}`")
    lines.extend(
        [
            f"- Pointer manifest: `{repo_relative(pointer_dir / MANIFEST_FILENAME)}`",
            "",
        ]
    )
    (output_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def build_manifest(
    args: argparse.Namespace,
    outputs: Sequence[Path],
    limits: Mapping[str, object],
    oasis20_summary: Mapping[str, object],
    oasis100_summary: Mapping[str, object],
) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    source_rows_path = output_dir / SOURCE_ROWS_FILENAME
    manifest = {
        "artifact": ARTIFACT_SLUG,
        "source_version": "v002_candidate",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "oasis20_population": repo_relative(Path(args.oasis20_source)),
            "oasis100_first10_main_sweeps": str(args.first10_main_root),
            "oasis100_first10_cps2_prelim_sweeps": str(args.first10_cps2_prelim_root),
            "oasis100_first10_cps2_completion_sweeps": str(args.first10_cps2_completion_root),
            "oasis100_next10_sweeps": str(args.next10_root),
            "oasis100_single_stage_extension": str(args.single_stage_extension_root),
            "oasis100_baseline_first20": str(args.baseline_first20_root),
            "oasis100_baseline_extension": str(args.baseline_extension_root),
            "oasis100_directcp842_first20": str(args.directcp842_first20_root),
            "oasis100_directcp842_extension": str(args.directcp842_extension_root),
            "method_nicknames": repo_relative(Path(args.nickname_csv)),
        },
        "outputs": [repo_relative(path) for path in outputs],
        "source_rows": repo_relative(source_rows_path),
        "output_formats": ["png"],
        "annotation_modes": {
            name: {"annotate": bool(spec["annotate"]), "filename": spec["filename"]}
            for name, spec in OUTPUTS.items()
        },
        "panels": PANELS,
        "style": FIGURE_STYLE | {"dpi": int(args.dpi)},
        "included_methods": {
            "sweeps": list(INCLUDED_SWEEP_METHODS),
            "baselines": list(INCLUDED_BASELINE_METHODS),
        },
        "excluded_methods": list(EXCLUDED_METHODS),
        "row_counts": {
            "expected_population_rows_per_cohort": EXPECTED_POPULATION_ROWS,
            "oasis20_population_rows": int(oasis20_summary["population_rows"]),
            "oasis100_population_rows": int(oasis100_summary["population_rows"]),
            "oasis100_case_sweep_rows": int(oasis100_summary["case_sweep_rows"]),
            "oasis100_case_baseline_rows": int(oasis100_summary["case_baseline_rows"]),
        },
        "axis_limits": limits,
        "oasis100_coverage": oasis100_summary["coverage"],
        "preflight": {
            "all_included_first100_groups_complete": all(
                count == 100
                for method_counts in oasis100_summary["coverage"]["sweep_counts"].values()  # type: ignore[index]
                for count in method_counts.values()
            )
            and all(
                count == 100
                for count in oasis100_summary["coverage"]["baseline_counts"].values()  # type: ignore[index]
            ),
            "direct_cp_cps4_complete": all(
                count == 100
                for count in oasis100_summary["coverage"]["direct_cp_cps4_counts"].values()  # type: ignore[index]
            ),
            "inr_bscp_cps4_included": "sinr_cps4_be_bspline_analytic" in INCLUDED_SWEEP_METHODS,
            "inr_bscp_cps4_excluded": "sinr_cps4_be_bspline_analytic" in EXCLUDED_METHODS,
        },
    }
    return sanitize_public_value(manifest)


def main() -> None:
    args = parse_args()
    apply_method_set(args)
    configure_oasis_plotting(Path(args.nickname_csv))
    oasis20_rows, oasis20_summary = build_oasis20_population(args)
    oasis100_rows, oasis100_summary = build_oasis100_population(args)
    if args.preflight_only:
        payload = sanitize_public_value(
            {
                "status": "preflight_ok",
                "oasis20": oasis20_summary,
                "oasis100": oasis100_summary,
                "included_methods": {
                    "sweeps": INCLUDED_SWEEP_METHODS,
                    "baselines": INCLUDED_BASELINE_METHODS,
                },
                "excluded_methods": EXCLUDED_METHODS,
            }
        )
        print(json.dumps(payload, indent=2, allow_nan=True))
        return

    output_dir = Path(args.output_dir)
    pointer_dir = Path(args.pointer_dir)
    source_rows_path = output_dir / SOURCE_ROWS_FILENAME
    write_source_rows(source_rows_path, oasis20_rows, oasis100_rows)
    outputs, limits = render_figures(args, oasis20_rows, oasis100_rows)
    missing_outputs = assert_outputs(outputs + [source_rows_path])
    if missing_outputs:
        raise RuntimeError("Missing or empty comparison outputs:\n" + "\n".join(missing_outputs))

    manifest = build_manifest(args, outputs, limits, oasis20_summary, oasis100_summary)
    manifest_path = output_dir / MANIFEST_FILENAME
    c.write_json(manifest_path, manifest)
    write_review_report(output_dir, pointer_dir, manifest)

    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer_manifest = {
        "artifact": ARTIFACT_SLUG,
        "generated_utc": manifest["generated_utc"],
        "paper_artifact_manifest": repo_relative(manifest_path),
        "paper_artifact_output_dir": repo_relative(output_dir),
        "outputs": manifest["outputs"],
        "source_rows": manifest["source_rows"],
    }
    c.write_json(pointer_dir / MANIFEST_FILENAME, pointer_manifest)
    print(json.dumps(manifest, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
