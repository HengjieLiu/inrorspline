#!/usr/bin/env python
"""Render population overlay diagnostic trial figures.

The figures are source-backed by the schedule-matched camera-ready index.  They
overlay all ten case trajectories for each method and add population mean curves
with one-sample-standard-deviation bands computed in log10 metric space.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from plot_case_diagnostics import (
    METHOD_ORDER,
    METHOD_STYLE,
    DiagnosticCurve,
    endpoint_matches,
    normalize_terminal_endpoint,
)
from render_schedule_matched_diagnostics import (
    AXIS_LIMITS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POPULATION_RUN_ROOT,
    DEFAULT_REFRESH_RUN_ROOT,
    REFRESH_TOKEN,
    STAGE_BOUNDARIES,
    load_merged_curve,
    sha256,
)


AGGREGATE_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=float)
EXPECTED_GROUP_CASES = 10
OUTPUT_FIGURE = "population_overlay_mean_std_2x2.png"
OUTPUT_2X4 = "population_overlay_plus_lungct_cases_2x4.png"
OUTPUT_2X4_TRE = "population_overlay_plus_lungct_cases_2x4_tre_labels.png"
OUTPUT_2X3 = "population_overlay_plus_lungct_cases_2x3.png"
OUTPUT_SUMMARY = "population_overlay_mean_std_summary.csv"
OUTPUT_MANIFEST = "population_overlay_mean_std_manifest.json"
OUTPUT_REPORT = "population_overlay_mean_std.md"
OUTPUT_DESIGN_MANIFEST = "population_overlay_design_trials_manifest.json"
OUTPUT_DESIGN_REPORT = "population_overlay_design_trials.md"
OVERLAY_DATASET_LABEL = {"oasis": "OASIS N=10", "dirlab": "DIR-LAB/LungCT N=10"}
LUNGCT_CASE_STYLES = {
    "case01": {"label": "DIR-LAB case01", "linestyle": "solid"},
    "case08": {"label": "DIR-LAB case08", "linestyle": (0, (5, 2.4))},
}
TRE_LABEL_POSITIONS = {
    "case01": {
        "idir": (1740, 0.070),
        "inr_bscp": (1740, 0.0105),
        "d_bscp": (1740, 0.0061),
        "mr_d_bscp": (1740, 0.0035),
    },
    "case08": {
        "idir": (1740, 0.115),
        "inr_bscp": (1740, 0.0125),
        "d_bscp": (1740, 0.0190),
        "mr_d_bscp": (1740, 0.0043),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-index",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "camera_ready_source_index.csv",
    )
    parser.add_argument(
        "--population-run-root",
        type=Path,
        default=DEFAULT_POPULATION_RUN_ROOT,
    )
    parser.add_argument(
        "--refresh-run-root",
        type=Path,
        default=DEFAULT_REFRESH_RUN_ROOT,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return path.name


def image_record(path: Path, output_root: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": os.path.relpath(path, output_root),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def resolve_summary_path(
    value: Any,
    population_root: Path,
    refresh_root: Path,
    data_root: Path,
) -> Path:
    text = str(value or "")
    replacements = {
        "${POPULATION_RUN_ROOT}": str(population_root),
        "${DIRLAB_REFRESH_RUN_ROOT}": str(refresh_root),
        "${DATA_ROOT}": str(data_root),
        "${RUN_ROOT}": str(population_root),
    }
    for token, root in replacements.items():
        text = text.replace(token, root)
    return Path(text)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def validate_source_index(index: pd.DataFrame) -> None:
    if len(index) != 160:
        raise RuntimeError(f"Expected 160 source-index records, found {len(index)}")
    if index["job_id"].duplicated().any():
        raise RuntimeError("Source index contains duplicate job_id values")
    duplicate_key = index.duplicated(["dataset", "case_key", "panel", "method_slug"])
    if duplicate_key.any():
        raise RuntimeError("Source index contains duplicate dataset/case/panel/method rows")

    counts = index.groupby(["dataset", "panel", "method_slug"])["case_key"].nunique()
    expected = pd.MultiIndex.from_product(
        [["oasis", "dirlab"], ["A", "B"], METHOD_ORDER],
        names=["dataset", "panel", "method_slug"],
    )
    missing = sorted(set(expected) - set(counts.index))
    if missing:
        raise RuntimeError(f"Missing expected source groups: {missing}")
    bad_counts = counts.reindex(expected)
    if not bad_counts.eq(EXPECTED_GROUP_CASES).all():
        raise RuntimeError(f"Every group must have ten cases:\n{bad_counts}")

    refresh = index[
        index["dataset"].eq("dirlab")
        & index["panel"].eq("A")
        & index["method_slug"].eq("mr_d_bscp")
    ].copy()
    if len(refresh) != 10:
        raise RuntimeError(f"Expected ten DIR-LAB Panel A MR-D-BSCP refresh rows, found {len(refresh)}")
    checks = {
        "refresh_paths": refresh["source_path"].astype(str).str.startswith(REFRESH_TOKEN + "/").all(),
        "stage_epochs": refresh["stage_epochs"].astype(str).eq("400,400,800,900").all(),
        "stage_downsample": refresh["stage_downsample"].astype(str).eq("8,4,2,1").all(),
        "control_spacing": refresh["cps"].astype(str).eq("32,16,8,4").all(),
        "lr": np.allclose(refresh["lr"].astype(float), 0.001),
        "alpha": np.allclose(refresh["alpha"].astype(float), 10000.0),
        "seed": refresh["seed"].astype(int).eq(1).all(),
        "epochs": refresh["epochs"].astype(int).eq(2500).all(),
        "diagnostic_interval": refresh["diagnostic_interval"].astype(int).eq(10).all(),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError("DIR-LAB refresh source validation failed: " + ", ".join(failed))


def load_validated_curves(
    index: pd.DataFrame,
    population_root: Path,
    refresh_root: Path,
    data_root: Path,
) -> tuple[list[DiagnosticCurve], list[dict[str, Any]]]:
    curves: list[DiagnosticCurve] = []
    validation_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in index.iterrows():
        try:
            raw_curve = load_merged_curve(row, population_root, refresh_root, data_root)
            curve, raw_endpoint, raw_difference, endpoint_replaced = normalize_terminal_endpoint(
                raw_curve
            )
            matches, plotted_endpoint, difference = endpoint_matches(curve)
            if not matches:
                raise RuntimeError(
                    f"plotted={plotted_endpoint:.12g}, accepted={curve.accepted_endpoint:.12g}"
                )
            if not np.isfinite(curve.values).all() or not (curve.values > 0.0).all():
                raise RuntimeError("curve contains nonfinite or nonpositive values")
            curves.append(curve)
            validation_rows.append(
                {
                    "job_id": row["job_id"],
                    "dataset": curve.dataset,
                    "case_key": curve.case_key,
                    "case_order": curve.case_order,
                    "panel": curve.panel,
                    "method_slug": curve.method_slug,
                    "points": int(len(curve.epochs)),
                    "epoch_min": float(np.min(curve.epochs)),
                    "epoch_max": float(np.max(curve.epochs)),
                    "accepted_endpoint": float(curve.accepted_endpoint),
                    "plotted_endpoint": plotted_endpoint,
                    "endpoint_difference": difference,
                    "endpoint_matches": matches,
                    "raw_terminal_value": raw_endpoint,
                    "raw_terminal_difference": raw_difference,
                    "terminal_endpoint_replaced": endpoint_replaced,
                    "source_path": row["source_path"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report all validation failures together.
            failures.append(f"{row.get('job_id', '<missing-job>')}: {exc}")
    if failures:
        raise RuntimeError("Curve validation failed:\n" + "\n".join(failures))

    refresh_curves = [
        curve
        for curve in curves
        if curve.dataset == "dirlab" and curve.panel == "A" and curve.method_slug == "mr_d_bscp"
    ]
    if len(refresh_curves) != 10:
        raise RuntimeError("Expected ten loaded DIR-LAB Panel A MR-D-BSCP refresh curves")
    expected_epochs = AGGREGATE_EPOCHS.tolist()
    for curve in refresh_curves:
        if len(curve.epochs) != 251 or curve.epochs.astype(float).tolist() != expected_epochs:
            raise RuntimeError(
                f"{curve.case_key}: expected 251 refresh diagnostics through epoch 2500"
            )
    return curves, validation_rows


def group_curves(
    curves: Sequence[DiagnosticCurve], dataset: str, panel: str, method: str
) -> list[DiagnosticCurve]:
    matches = sorted(
        [
            curve
            for curve in curves
            if curve.dataset == dataset and curve.panel == panel and curve.method_slug == method
        ],
        key=lambda curve: curve.case_order,
    )
    if len(matches) != EXPECTED_GROUP_CASES:
        raise RuntimeError(
            f"{dataset}/Panel{panel}/{method}: expected 10 curves, found {len(matches)}"
        )
    return matches


def aggregate_log_space(
    curves: Sequence[DiagnosticCurve],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_values: list[np.ndarray] = []
    for curve in curves:
        if float(np.min(curve.epochs)) > 1.0 or float(np.max(curve.epochs)) < 2500.0:
            raise RuntimeError(f"{curve.case_key}/{curve.method_slug}: missing aggregate epoch span")
        log_curve = np.log10(curve.values)
        log_values.append(np.interp(AGGREGATE_EPOCHS, curve.epochs, log_curve))
    matrix = np.vstack(log_values)
    mean_log = np.mean(matrix, axis=0)
    sd_log = np.std(matrix, axis=0, ddof=1)
    return (
        mean_log,
        sd_log,
        np.power(10.0, mean_log),
        np.power(10.0, mean_log - sd_log),
        np.power(10.0, mean_log + sd_log),
        matrix,
    )


def style_axis(
    axis: plt.Axes,
    dataset: str,
    panel: str,
    *,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    for boundary in STAGE_BOUNDARIES[(dataset, panel)]:
        axis.axvline(
            boundary,
            color="#777777",
            linewidth=0.75,
            linestyle=(0, (2, 2)),
            alpha=0.55,
            zorder=1,
        )
    axis.set_yscale("log")
    axis.set_ylim(*AXIS_LIMITS[panel])
    axis.set_xlim(0, 2500)
    axis.set_xticks(np.arange(0, 2501, 500))
    axis.grid(True, which="both", color="#D5DCE3", linewidth=0.65, alpha=0.75)
    axis.tick_params(axis="both", labelsize=8.5, labelbottom=True, labelleft=True)
    if show_ylabel:
        axis.set_ylabel("1-NCC" if panel == "A" else "rRMSE (%)", fontsize=10.5)
    if panel == "B" and show_xlabel:
        axis.set_xlabel("Epoch", fontsize=10.5)


def draw_population_overlay(
    axis: plt.Axes,
    curves: Sequence[DiagnosticCurve],
    dataset: str,
    panel: str,
    summary_rows: list[dict[str, Any]],
    *,
    write_summary: bool,
) -> None:
    for method in METHOD_ORDER:
        group = group_curves(curves, dataset, panel, method)
        style = METHOD_STYLE[method]
        for curve in group:
            axis.plot(
                curve.epochs,
                curve.values,
                color=style["color"],
                linewidth=0.65,
                alpha=0.18,
                zorder=2,
            )
        mean_log, sd_log, mean, lower, upper, matrix = aggregate_log_space(group)
        axis.fill_between(
            AGGREGATE_EPOCHS,
            lower,
            upper,
            color=style["color"],
            alpha=0.11,
            linewidth=0.0,
            zorder=3,
        )
        axis.plot(
            AGGREGATE_EPOCHS,
            mean,
            color=style["color"],
            linewidth=2.25,
            zorder=METHOD_STYLE[method]["zorder"] + 8,
            label=style["label"],
        )
        if write_summary:
            for epoch, mean_l, sd_l, mean_v, low_v, high_v, column_values in zip(
                    AGGREGATE_EPOCHS,
                    mean_log,
                    sd_log,
                    mean,
                    lower,
                    upper,
                    matrix.T,
                    strict=True,
            ):
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "panel": panel,
                        "method_slug": method,
                        "display_method": style["label"],
                        "epoch": int(epoch),
                        "n_cases": len(group),
                        "mean_log10": float(mean_l),
                        "sd_log10": float(sd_l),
                        "mean_value": float(mean_v),
                        "lower_mean_minus_1sd_logspace": float(low_v),
                        "upper_mean_plus_1sd_logspace": float(high_v),
                        "min_case_value": float(np.power(10.0, np.min(column_values))),
                        "max_case_value": float(np.power(10.0, np.max(column_values))),
                    }
                )


def case_curve(
    curves: Sequence[DiagnosticCurve], case_key: str, panel: str, method: str
) -> DiagnosticCurve:
    matches = [
        curve
        for curve in curves
        if curve.dataset == "dirlab"
        and curve.case_key == case_key
        and curve.panel == panel
        and curve.method_slug == method
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"DIR-LAB {case_key}/Panel{panel}/{method}: expected one curve, found {len(matches)}"
        )
    return matches[0]


def validate_design_cases(curves: Sequence[DiagnosticCurve]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_key in LUNGCT_CASE_STYLES:
        count = sum(
            1
            for curve in curves
            if curve.dataset == "dirlab" and curve.case_key == case_key
        )
        if count != 8:
            raise RuntimeError(f"DIR-LAB {case_key}: expected 8 curves, found {count}")
        counts[case_key] = count
    return counts


def load_case_panel_a_tre_labels(
    index: pd.DataFrame,
    population_root: Path,
    refresh_root: Path,
    data_root: Path,
) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for case_key in LUNGCT_CASE_STYLES:
        labels[case_key] = {}
        for method in METHOD_ORDER:
            matches = index[
                index["dataset"].eq("dirlab")
                & index["case_key"].eq(case_key)
                & index["panel"].eq("A")
                & index["method_slug"].eq(method)
            ]
            if len(matches) != 1:
                failures.append(f"{case_key}/{method}: expected one Panel A source row")
                continue
            row = matches.iloc[0]
            summary_path = resolve_summary_path(
                row["summary_path"],
                population_root,
                refresh_root,
                data_root,
            )
            if not summary_path.is_file():
                failures.append(f"{case_key}/{method}: missing summary {row['summary_path']}")
                continue
            payload = json.loads(summary_path.read_text())
            tre = finite(nested(payload, "tre_snap_to_voxel", "tre_mean_mm"))
            if tre is None:
                failures.append(f"{case_key}/{method}: missing finite tre_snap_to_voxel.tre_mean_mm")
                continue
            labels[case_key][method] = tre
    if failures:
        raise RuntimeError("TRE label validation failed:\n" + "\n".join(failures))
    return labels


def draw_single_case(axis: plt.Axes, curves: Sequence[DiagnosticCurve], panel: str, case_key: str) -> None:
    for method in METHOD_ORDER:
        curve = case_curve(curves, case_key, panel, method)
        style = METHOD_STYLE[method]
        axis.plot(
            curve.epochs,
            curve.values,
            color=style["color"],
            linewidth=1.95,
            alpha=0.95,
            zorder=style["zorder"] + 5,
        )


def add_tre_labels(
    axis: plt.Axes,
    curves: Sequence[DiagnosticCurve],
    case_key: str,
    tre_labels: dict[str, dict[str, float]],
) -> None:
    for method in METHOD_ORDER:
        curve = case_curve(curves, case_key, "A", method)
        style = METHOD_STYLE[method]
        endpoint = float(curve.values[-1])
        x_text, y_text = TRE_LABEL_POSITIONS[case_key][method]
        label = f"TRE {tre_labels[case_key][method]:.2f} mm"
        axis.annotate(
            label,
            xy=(float(curve.epochs[-1]), endpoint),
            xytext=(x_text, y_text),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9.4,
            weight="bold",
            color=style["color"],
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": style["color"],
                "linewidth": 0.75,
                "alpha": 0.65,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            clip_on=False,
            zorder=30,
        )


def draw_case_comparison(axis: plt.Axes, curves: Sequence[DiagnosticCurve], panel: str) -> None:
    for case_key, case_style in LUNGCT_CASE_STYLES.items():
        for method in METHOD_ORDER:
            curve = case_curve(curves, case_key, panel, method)
            style = METHOD_STYLE[method]
            axis.plot(
                curve.epochs,
                curve.values,
                color=style["color"],
                linestyle=case_style["linestyle"],
                linewidth=1.95 if case_key == "case01" else 2.05,
                alpha=0.95,
                zorder=style["zorder"] + (8 if case_key == "case08" else 5),
            )


def method_legend_handles(linewidth: float = 2.8) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=linewidth,
            label=METHOD_STYLE[method]["label"],
        )
        for method in METHOD_ORDER
    ]


def case_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=style["linestyle"],
            linewidth=2.1,
            label=style["label"],
        )
        for style in LUNGCT_CASE_STYLES.values()
    ]


def render_figure(
    curves: Sequence[DiagnosticCurve],
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), sharex=False, sharey="row")
    for column, dataset in enumerate(["oasis", "dirlab"]):
        for row_index, panel in enumerate(["A", "B"]):
            axis = axes[row_index, column]
            style_axis(axis, dataset, panel)
            axis.set_title(
                f"{OVERLAY_DATASET_LABEL[dataset]} - "
                f"{'registration' if panel == 'A' else 'supervised fitting'}",
                fontsize=12.5,
                weight="bold",
            )
            draw_population_overlay(
                axis,
                curves,
                dataset,
                panel,
                summary_rows,
                write_summary=True,
            )

    fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        handlelength=2.8,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Population diagnostic overlay: thin lines are cases, thick lines are means, bands are +/- 1 SD in log10 space",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.105, hspace=0.29, wspace=0.18)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_design_2x4(
    curves: Sequence[DiagnosticCurve],
    output_path: Path,
    dpi: int,
    *,
    tre_labels: dict[str, dict[str, float]] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(23.0, 8.8), sharex=False, sharey="row")
    columns = [
        ("population", "oasis", "OASIS N=10 population"),
        ("population", "dirlab", "DIR-LAB/LungCT N=10 population"),
        ("case", "case01", "DIR-LAB case01"),
        ("case", "case08", "DIR-LAB case08"),
    ]
    for column_index, (kind, key, title) in enumerate(columns):
        for row_index, panel in enumerate(["A", "B"]):
            axis = axes[row_index, column_index]
            dataset = key if kind == "population" else "dirlab"
            style_axis(axis, dataset, panel, show_ylabel=column_index == 0)
            axis.set_title(
                f"{title} - {'registration' if panel == 'A' else 'supervised fitting'}",
                fontsize=11.7,
                weight="bold",
            )
            if kind == "population":
                draw_population_overlay(
                    axis,
                    curves,
                    key,
                    panel,
                    [],
                    write_summary=False,
                )
            else:
                draw_single_case(axis, curves, panel, key)
                if panel == "A" and tre_labels is not None:
                    add_tre_labels(axis, curves, key, tre_labels)
    fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        handlelength=2.8,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Population overlays with representative LungCT case diagnostics",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.105, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_design_2x3(
    curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 8.8), sharex=False, sharey="row")
    columns = [
        ("population", "oasis", "OASIS N=10 population"),
        ("population", "dirlab", "DIR-LAB/LungCT N=10 population"),
        ("compare", "case01_case08", "DIR-LAB case01 vs case08"),
    ]
    for column_index, (kind, key, title) in enumerate(columns):
        for row_index, panel in enumerate(["A", "B"]):
            axis = axes[row_index, column_index]
            dataset = key if kind == "population" else "dirlab"
            style_axis(axis, dataset, panel, show_ylabel=column_index == 0)
            axis.set_title(
                f"{title} - {'registration' if panel == 'A' else 'supervised fitting'}",
                fontsize=11.9,
                weight="bold",
            )
            if kind == "population":
                draw_population_overlay(
                    axis,
                    curves,
                    key,
                    panel,
                    [],
                    write_summary=False,
                )
            else:
                draw_case_comparison(axis, curves, panel)
    method_legend = fig.legend(
        handles=method_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.43, 0.012),
        ncol=4,
        frameon=False,
        fontsize=11.0,
        handlelength=2.7,
        columnspacing=1.7,
    )
    fig.add_artist(method_legend)
    fig.legend(
        handles=case_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.83, 0.012),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=2.8,
        columnspacing=1.3,
    )
    fig.suptitle(
        "Population overlays plus easy/hard LungCT representatives",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.068, right=0.99, top=0.90, bottom=0.115, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_report(
    path: Path,
    figure_path: Path,
    summary_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    figure_rel = os.path.relpath(figure_path, path.parent)
    summary_rel = os.path.relpath(summary_path, path.parent)
    manifest_rel = os.path.relpath(manifest_path, path.parent)
    lines = [
        "# Population Overlay Mean/Std Trial Figure",
        "",
        "This is an exploratory supplementary/reviewer diagnostic, not the compact main-paper figure.",
        "",
        f"![Population overlay mean/std trial]({figure_rel})",
        "",
        "## Encoding",
        "",
        "- Columns compare OASIS N=10 and DIR-LAB/LungCT N=10.",
        "- Rows compare Panel A registration `1-NCC` and Panel B supervised fitting `rRMSE (%)`.",
        "- Thin colored lines are the ten individual case trajectories for a method.",
        "- Thick colored lines are population means.",
        "- Shaded bands are mean plus/minus one sample standard deviation computed in `log10(metric)` space and then back-transformed.",
        "",
        "## Validation",
        "",
        f"- Source trajectories loaded: `{manifest['source_records_loaded']}/160`.",
        f"- Dataset/panel/method groups with ten cases: `{manifest['groups_with_ten_cases']}/16`.",
        f"- Endpoint-matched trajectories: `{manifest['endpoint_matched_curves']}/160`.",
        f"- DIR-LAB Panel A MR-D-BSCP refresh curves: `{manifest['dirlab_panelA_mrdbscp_refresh_curves']}/10`, all with 251 points through epoch 2500.",
        f"- Aggregate grid points per group: `{manifest['aggregate_epoch_count']}`.",
        "",
        "## Files",
        "",
        f"- Aggregate curve table: [`{summary_path.name}`]({summary_rel})",
        f"- Machine-readable manifest: [`{manifest_path.name}`]({manifest_rel})",
        "",
        "## Reading note",
        "",
        "This view is expected to be most interpretable for OASIS, where the ten cases are more visually coherent. For LungCT, the same figure is useful as a heterogeneity check, but method conclusions should still come from casewise and paired statistics.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_design_report(
    path: Path,
    figure_2x4: Path,
    figure_2x3: Path,
    figure_2x4_tre: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    figure_2x4_rel = os.path.relpath(figure_2x4, path.parent)
    figure_2x3_rel = os.path.relpath(figure_2x3, path.parent)
    figure_2x4_tre_rel = os.path.relpath(figure_2x4_tre, path.parent)
    manifest_rel = os.path.relpath(manifest_path, path.parent)
    lines = [
        "# Population Overlay Design Trials",
        "",
        "These are exploratory supplementary/reviewer design candidates, not the locked compact main-paper figure.",
        "",
        "## 2x4: population plus separate LungCT cases",
        "",
        f"![2x4 population overlay with separate LungCT cases]({figure_2x4_rel})",
        "",
        "This wider design prioritizes readability: OASIS population, LungCT population, DIR-LAB case01, and DIR-LAB case08 each receive their own column.",
        "",
        "## 2x4 with case TRE labels",
        "",
        f"![2x4 population overlay with LungCT case TRE labels]({figure_2x4_tre_rel})",
        "",
        "This version adds final registration TRE labels only to the DIR-LAB case01 and case08 registration panels. The labels use common `tre_snap_to_voxel.tre_mean_mm` for all four methods, are displayed with two decimals, and are anchored in nearby whitespace with leader lines to the final curve endpoints.",
        "",
        "## 2x3: population plus combined LungCT cases",
        "",
        f"![2x3 population overlay with combined LungCT cases]({figure_2x3_rel})",
        "",
        "This compact design prioritizes space: the final column overlays case01 and case08, using method color and case line style.",
        "",
        "## Encoding",
        "",
        "- Rows are Panel A registration `1-NCC` and Panel B supervised fitting `rRMSE (%)`.",
        "- Population columns use thin individual case curves, thick log-space means, and log-space +/-1 SD bands.",
        "- Case-specific columns use the same four method colors and method order as the existing diagnostic figures.",
        "- TRE labels, when shown, are method-colored and attached only to the case01/case08 registration panels.",
        "- In the 2x3 combined column, case01 is solid and case08 is dashed.",
        "",
        "## Validation",
        "",
        f"- Source trajectories loaded: `{manifest['source_records_loaded']}/160`.",
        f"- Dataset/panel/method groups with ten cases: `{manifest['groups_with_ten_cases']}/16`.",
        f"- Endpoint-matched trajectories: `{manifest['endpoint_matched_curves']}/160`.",
        f"- DIR-LAB case01 curves: `{manifest['design_case_curve_counts']['case01']}/8`.",
        f"- DIR-LAB case08 curves: `{manifest['design_case_curve_counts']['case08']}/8`.",
        f"- DIR-LAB case01/case08 registration TRE labels: `{manifest['tre_label_count']}/8`.",
        f"- DIR-LAB Panel A MR-D-BSCP refresh curves: `{manifest['dirlab_panelA_mrdbscp_refresh_curves']}/10`.",
        "",
        "## Files",
        "",
        f"- Machine-readable design manifest: [`{manifest_path.name}`]({manifest_rel})",
        "",
        "## Reading note",
        "",
        "The 2x4 is probably better for supplement/reviewer explanation. The 2x3 is probably the more realistic compact candidate if the main paper needs one figure panel for this story.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source_index = Path(args.source_index).resolve()
    output_root = Path(args.output_root).resolve()
    population_root = Path(args.population_run_root).resolve()
    refresh_root = Path(args.refresh_run_root).resolve()
    data_root = Path(args.data_root).resolve()

    index = pd.read_csv(source_index)
    validate_source_index(index)
    curves, validation_rows = load_validated_curves(
        index,
        population_root,
        refresh_root,
        data_root,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / OUTPUT_FIGURE
    figure_2x4_path = output_root / OUTPUT_2X4
    figure_2x4_tre_path = output_root / OUTPUT_2X4_TRE
    figure_2x3_path = output_root / OUTPUT_2X3
    summary_path = output_root / OUTPUT_SUMMARY
    manifest_path = output_root / OUTPUT_MANIFEST
    report_path = output_root / OUTPUT_REPORT
    design_manifest_path = output_root / OUTPUT_DESIGN_MANIFEST
    design_report_path = output_root / OUTPUT_DESIGN_REPORT

    summary_rows: list[dict[str, Any]] = []
    render_figure(curves, summary_rows, figure_path, args.dpi)
    design_case_counts = validate_design_cases(curves)
    tre_labels = load_case_panel_a_tre_labels(index, population_root, refresh_root, data_root)
    render_design_2x4(curves, figure_2x4_path, args.dpi)
    render_design_2x4(curves, figure_2x4_tre_path, args.dpi, tre_labels=tre_labels)
    render_design_2x3(curves, figure_2x3_path, args.dpi)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    group_counts = (
        pd.DataFrame(validation_rows)
        .groupby(["dataset", "panel", "method_slug"])["case_key"]
        .nunique()
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "artifact_role": "exploratory supplementary/reviewer diagnostic; not compact main-paper figure",
        "generator": repo_relative(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source_index": repo_relative(source_index),
        "source_index_sha256": sha256(source_index),
        "inputs": {
            "population_run_root": "${POPULATION_RUN_ROOT}",
            "refresh_run_root": "${DIRLAB_REFRESH_RUN_ROOT}",
            "data_root": "${DATA_ROOT}",
        },
        "source_records_loaded": len(curves),
        "endpoint_matched_curves": int(sum(row["endpoint_matches"] for row in validation_rows)),
        "terminal_endpoint_replacements": int(
            sum(row["terminal_endpoint_replaced"] for row in validation_rows)
        ),
        "groups_total": 16,
        "groups_with_ten_cases": int(group_counts.eq(EXPECTED_GROUP_CASES).sum()),
        "aggregate_epoch_count": len(AGGREGATE_EPOCHS),
        "aggregate_epochs": AGGREGATE_EPOCHS.astype(int).tolist(),
        "aggregation": {
            "individual_curves": "raw accepted trajectories",
            "mean": "arithmetic mean in log10(metric) space, back-transformed",
            "band": "sample standard deviation in log10(metric) space, back-transformed as mean +/- 1 SD",
        },
        "axis_limits": {
            "x_epoch": [0, 2500],
            "x_ticks": [0, 500, 1000, 1500, 2000, 2500],
            "A": list(AXIS_LIMITS["A"]),
            "B": list(AXIS_LIMITS["B"]),
        },
        "stage_boundaries": {
            "oasis": {"A": STAGE_BOUNDARIES[("oasis", "A")], "B": STAGE_BOUNDARIES[("oasis", "B")]},
            "dirlab": {"A": STAGE_BOUNDARIES[("dirlab", "A")], "B": STAGE_BOUNDARIES[("dirlab", "B")]},
        },
        "dirlab_panelA_mrdbscp_refresh_curves": int(
            sum(
                row["dataset"] == "dirlab"
                and row["panel"] == "A"
                and row["method_slug"] == "mr_d_bscp"
                and str(row["source_path"]).startswith(REFRESH_TOKEN + "/")
                and row["points"] == 251
                and row["epoch_max"] == 2500.0
                for row in validation_rows
            )
        ),
        "outputs": {
            "figure": image_record(figure_path, output_root),
            "summary_csv": os.path.relpath(summary_path, output_root),
            "manifest_json": os.path.relpath(manifest_path, output_root),
            "report_md": os.path.relpath(report_path, output_root),
        },
    }
    if manifest["source_records_loaded"] != 160:
        raise RuntimeError("Loaded curve count changed after rendering")
    if manifest["groups_with_ten_cases"] != 16:
        raise RuntimeError("Not every dataset/panel/method group has ten cases")
    if manifest["endpoint_matched_curves"] != 160:
        raise RuntimeError("Not every plotted endpoint matched the source index")
    if manifest["dirlab_panelA_mrdbscp_refresh_curves"] != 10:
        raise RuntimeError("DIR-LAB refresh curve validation failed after rendering")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(report_path, figure_path, summary_path, manifest_path, manifest)
    design_manifest = {
        "generated_at": manifest["generated_at"],
        "artifact_role": "exploratory supplementary/reviewer design candidates; not compact main-paper figure",
        "generator": manifest["generator"],
        "generator_sha256": manifest["generator_sha256"],
        "source_index": manifest["source_index"],
        "source_index_sha256": manifest["source_index_sha256"],
        "inputs": manifest["inputs"],
        "source_records_loaded": manifest["source_records_loaded"],
        "endpoint_matched_curves": manifest["endpoint_matched_curves"],
        "groups_total": manifest["groups_total"],
        "groups_with_ten_cases": manifest["groups_with_ten_cases"],
        "dirlab_panelA_mrdbscp_refresh_curves": manifest[
            "dirlab_panelA_mrdbscp_refresh_curves"
        ],
        "design_case_curve_counts": design_case_counts,
        "designs": {
            "2x4": {
                "role": "wider readability candidate",
                "layout": [
                    "OASIS N=10 population",
                    "DIR-LAB/LungCT N=10 population",
                    "DIR-LAB case01",
                    "DIR-LAB case08",
                ],
                "figure": image_record(figure_2x4_path, output_root),
            },
            "2x3": {
                "role": "compact candidate",
                "layout": [
                    "OASIS N=10 population",
                    "DIR-LAB/LungCT N=10 population",
                    "DIR-LAB case01 vs case08",
                ],
                "case_line_styles": {
                    "case01": "solid",
                    "case08": "dashed",
                },
                "figure": image_record(figure_2x3_path, output_root),
            },
            "2x4_tre_labels": {
                "role": "wider readability candidate with final TRE labels on case registration panels",
                "layout": [
                    "OASIS N=10 population",
                    "DIR-LAB/LungCT N=10 population",
                    "DIR-LAB case01 with Panel A TRE labels",
                    "DIR-LAB case08 with Panel A TRE labels",
                ],
                "tre_label_metric": "tre_snap_to_voxel.tre_mean_mm",
                "tre_label_display_format": "TRE {value:.2f} mm",
                "tre_label_positions_data_coordinates": TRE_LABEL_POSITIONS,
                "tre_labels_mm": tre_labels,
                "tre_label_count": sum(len(values) for values in tre_labels.values()),
                "figure": image_record(figure_2x4_tre_path, output_root),
            },
        },
        "tre_label_metric_note": (
            "Final TRE labels use common tre_snap_to_voxel.tre_mean_mm for all "
            "four methods, limited to DIR-LAB case01 and case08 Panel A."
        ),
        "tre_label_count": sum(len(values) for values in tre_labels.values()),
        "axis_limits": manifest["axis_limits"],
        "stage_boundaries": manifest["stage_boundaries"],
        "report_md": os.path.relpath(design_report_path, output_root),
    }
    design_manifest_path.write_text(json.dumps(design_manifest, indent=2) + "\n")
    write_design_report(
        design_report_path,
        figure_2x4_path,
        figure_2x3_path,
        figure_2x4_tre_path,
        design_manifest_path,
        design_manifest,
    )
    print(json.dumps(design_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
