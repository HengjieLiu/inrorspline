#!/usr/bin/env python
"""Build standardized final-paper Pareto artifacts."""

from __future__ import annotations

import json
import math
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd

from paper_artifact_common import (
    artifact_output_dir,
    display_name,
    format_lr,
    load_nicknames,
    load_standard,
    markdown_table,
    repo_path,
    repo_relative,
    safe_float,
    sanitize_public_value,
    selected_hypervariant_records,
    validate_lr_records,
    validation_report_markdown,
)


OUTPUT_FORMATS = ("png",)

OASIS_INCLUDE_DIRECTCP842 = True
OASIS_DATASET = "oasis100"
OASIS_COHORT = "first100"
OASIS_PANEL_N = 100
OASIS_DPI = 300
OASIS_TICK_LABEL_SIZE = 12
OASIS_AXIS_LABEL_SIZE = 16
OASIS_LEGEND_FONT_SIZE = 12
OASIS_ANNOTATED_DIRNAME = "oasis100"
OASIS_PAPER_NO_ANNOTATIONS_DIRNAME = "oasis100_paper_no_annotations"
OASIS_SOURCE_ROWS_FILENAME = f"{OASIS_DATASET}_pareto_source_rows.csv"
OASIS_SOURCE_WITH_DIRECTCP842 = repo_path(
    "experiments/2026-07-04_oasis_seed3_first100_extension/"
    "first100_artifacts/oasis100/population_summary_oasis100_with_saved_baselines.csv"
)
OASIS_SOURCE_FIVE_SWEEP = repo_path(
    "experiments/2026-07-04_oasis_seed3_first100_extension/"
    "first100_artifacts/oasis100/population_summary_oasis100_with_saved_baselines.csv"
)
OASIS_SOURCE_OVERRIDE_KEY = "oasis100_population_summary"
COMPOSITE_DIRNAME = "four_panel_composite"
COMPOSITE_OUTPUTS = {
    "no_annotations": {
        "filename": "pareto_four_panel_composite_no_annotations.png",
        "annotate": False,
    },
    "annotated_review": {
        "filename": "pareto_four_panel_composite_annotated_review.png",
        "annotate": True,
    },
}
COMPOSITE_STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 18,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 8.5,
    "legend_frame": True,
    "line_width": 1.55,
    "marker_size": 4.8,
    "lung_marker_size": 50,
    "ptvreg_marker_size": 120,
    "ellipse_alpha": 0.14,
    "lung_ellipse_alpha": 0.12,
    "grid_alpha": 0.25,
    "wspace": 0.18,
    "hspace": 0.30,
}
OASIS_ACCURACY_METRICS = [
    ("dice_mean", "Mean Dice", True),
    ("hd95_mean", "Mean HD95", False),
    ("asd_warped_to_fixed_mean", "ASD warped-to-fixed", False),
    ("asd_symmetric_mean", "ASD symmetric", False),
]
OASIS_REGULARITY_METRICS = [
    ("be_fd_forward_fixed_fg_voxel", "Foreground BE forward FD"),
    ("diffusion_fd_forward_fixed_fg_voxel", "Foreground diffusion forward FD"),
    ("diffusion_l2_fg", "Foreground diffusion L2"),
    ("jdet_le0_pct_fg", "Foreground Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "Foreground std(log Jdet)"),
    ("ndv_frac_fg", "Foreground NDV fraction"),
]
OASIS_SPOTLIGHT_PANELS = [
    ("dice_mean", "log_jdet_std_fg"),
    ("hd95_mean", "log_jdet_std_fg"),
]
OASIS_STYLES = ("line_segments", "line_segments_errorbars", "ellipses")

LUNG_MEAN_SOURCE = repo_path(
    "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/"
    "pareto_visual_review_midratio_extension/combined_mean_rows_with_ptvreg.csv"
)
LUNG_NO_SWEEP_SOURCE = repo_path(
    "experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.csv"
)
LUNGCT_DPI = 220
LUNGCT_FIGSIZE = (8.2, 5.8)
LUNGCT_TICK_LABEL_SIZE = 12
LUNGCT_AXIS_LABEL_SIZE = 16
LUNGCT_LEGEND_FONT_SIZE = 12
LUNGCT_SUBSET_NAME = "default_exclusions"
LUNGCT_ANNOTATED_DIRNAME = "lungct10"
LUNGCT_PAPER_NO_ANNOTATIONS_DIRNAME = "lungct10_paper_no_annotations"
LUNGCT_EXCLUDE_BY_RATIO = {
    "idir_be_default_lungmask": (),
    "direct_cp_32_16_8_4_ptv_lcc_isotv": (0.01, 10.0, 100.0),
    "direct_cp_32_16_8_4_global_ncc_fd_be": (0.0, 0.001, 0.01),
}
LUNGCT_EXCLUDE_BY_WEIGHT = {
    "idir_be_default_lungmask": (),
    "direct_cp_32_16_8_4_ptv_lcc_isotv": (),
    "direct_cp_32_16_8_4_global_ncc_fd_be": (),
}
LUNGCT_ACCURACY_METRICS = [
    ("original_snap_tre_mean_mm", "Original snap TRE (mm)", False),
]
LUNGCT_REGULARITY_METRICS = [
    ("fg_logj_std", "Foreground stdLogJ"),
    ("fg_jdet_le0_pct", "Foreground J <= 0 (%)"),
    ("fg_bending_energy", "Foreground bending energy"),
    ("fg_diffusion_l2", "Foreground diffusion energy"),
]
LUNGCT_PLOT_METHOD_ORDER = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]
LUNGCT_METHOD_LABELS = {
    "idir_be_default_lungmask": "IDIR BE",
    "sinr_cps4_ptv_lcc_isotv": "SINR cps4 pTV-LCC+isoTV",
    "direct_cp_cps4_ptv_lcc_isotv": "DirectCP cps4 pTV-LCC+isoTV",
    "direct_cp_32_16_8_4_ptv_lcc_isotv": "DirectCP 4stage pTV-LCC+isoTV",
    "direct_cp_32_16_8_4_global_ncc_fd_be": "DirectCP 4stage global-NCC+FD-BE",
    "ptvreg_reference": "pTVReg",
}
LUNGCT_METHOD_COLORS = {
    "idir_be_default_lungmask": "#0072B2",
    "sinr_cps4_ptv_lcc_isotv": "#CC79A7",
    "direct_cp_cps4_ptv_lcc_isotv": "#B23A48",
    "direct_cp_32_16_8_4_ptv_lcc_isotv": "#D55E00",
    "direct_cp_32_16_8_4_global_ncc_fd_be": "#009E73",
    "ptvreg_reference": "#111111",
}
LUNGCT_METHOD_MARKERS = {
    "idir_be_default_lungmask": "o",
    "sinr_cps4_ptv_lcc_isotv": "D",
    "direct_cp_cps4_ptv_lcc_isotv": "P",
    "direct_cp_32_16_8_4_ptv_lcc_isotv": "s",
    "direct_cp_32_16_8_4_global_ncc_fd_be": "^",
    "ptvreg_reference": "*",
}

METHOD_FAMILY_BY_ID = {
    "idir_original_dense_siren_be_autograd": "idir",
    "sinr_cps2_be_bspline_analytic": "sinr",
    "sinr_cps4_be_bspline_analytic": "sinr",
    "direct_cp_cps2_be_bspline_analytic": "directcp",
    "direct_cp_cps4_be_bspline_analytic": "directcp",
    "direct_cp_8_4_2_be_bspline_analytic_multistage": "directcp_multistage",
    "greedy": "reference",
    "bs1_vxm_v0_2": "reference",
    "bs2_tm_v0": "reference",
    "bs3_vfa_v0": "reference",
    "bs4_sitreg_v0": "reference",
    "idir_be_default_lungmask": "idir",
    "sinr_cps4_ptv_lcc_isotv": "sinr",
    "direct_cp_cps4_ptv_lcc_isotv": "directcp",
    "direct_cp_32_16_8_4_ptv_lcc_isotv": "directcp_multistage",
    "direct_cp_32_16_8_4_global_ncc_fd_be": "directcp_multistage",
    "ptvreg_reference": "reference",
}


def _mean_std(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _method_family(method_id: str) -> str:
    return METHOD_FAMILY_BY_ID.get(str(method_id), "reference")


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "map"):
        return df.map(sanitize_public_value)
    return df.applymap(sanitize_public_value)


def _display_name_mapping(*frames: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for frame in frames:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            dataset = str(row.get("dataset", ""))
            method_id = str(row.get("method_id") or row.get("method_key") or row.get("method"))
            display = str(row.get("display_method", ""))
            key = (dataset, method_id, display)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "dataset": dataset,
                    "method_id": method_id,
                    "original_name": str(row.get("method_label", "")),
                    "display_name": display,
                    "method_family": str(row.get("method_family", _method_family(method_id))),
                }
            )
    return sorted(rows, key=lambda item: (str(item["dataset"]), str(item["method_family"]), str(item["method_id"])))


def _oasis_source(standard: Mapping[str, Any] | None = None) -> Path:
    if standard is not None:
        override = standard.get("artifact_source_overrides", {}).get(OASIS_SOURCE_OVERRIDE_KEY)
        if override:
            return repo_path(str(override))
    return OASIS_SOURCE_WITH_DIRECTCP842 if OASIS_INCLUDE_DIRECTCP842 else OASIS_SOURCE_FIVE_SWEEP


def _source_lr(row: Mapping[str, Any], standard_records: list[dict[str, Any]]) -> float:
    lr = safe_float(row.get("lr"))
    if math.isfinite(lr):
        return lr
    method_id = str(row.get("method_id") or row.get("method_key") or row.get("method"))
    dataset = str(row.get("dataset", ""))
    for record in standard_records:
        if str(record.get("method_id")) == method_id and str(record.get("dataset")) == dataset:
            return safe_float(record.get("lr"))
    return float("nan")


def _metric_mean(metric: str) -> str:
    return f"{metric}_mean"


def _metric_std(metric: str) -> str:
    return f"{metric}_std"


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def _close(value: float, target: float) -> bool:
    return math.isfinite(value) and abs(value - target) <= 1e-9 * max(1.0, abs(target))


def _relative_paths(paths: Sequence[Path], output_dir: Path) -> list[str]:
    out = []
    for path in paths:
        try:
            out.append(str(path.relative_to(output_dir)))
        except ValueError:
            out.append(repo_relative(path))
    return sorted(out)


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in [
        output_dir / OASIS_ANNOTATED_DIRNAME,
        output_dir / OASIS_PAPER_NO_ANNOTATIONS_DIRNAME,
        output_dir / LUNGCT_ANNOTATED_DIRNAME,
        output_dir / LUNGCT_PAPER_NO_ANNOTATIONS_DIRNAME,
        output_dir / COMPOSITE_DIRNAME,
        output_dir / ".ipynb_checkpoints",
    ]:
        if child.exists():
            shutil.rmtree(child)
    for suffix in ("*.png", "*.pdf"):
        for path in output_dir.glob(suffix):
            path.unlink()


def load_oasis_rows(
    nicknames: dict[str, dict[str, str]],
    standard_records: list[dict[str, Any]],
    standard: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    source = _oasis_source(standard)
    rows = pd.read_csv(source)
    rows["dataset"] = OASIS_DATASET
    rows["method_id"] = rows["method_key"].astype(str)
    rows["display_method"] = [
        display_name(method_id, original_name, nicknames)
        for method_id, original_name in zip(rows["method_id"], rows["method_label"].astype(str))
    ]
    rows["method_family"] = rows["method_id"].map(_method_family)
    rows["selected_lr"] = [_source_lr(row, standard_records) for row in rows.to_dict("records")]
    rows["selected_lr_label"] = rows["selected_lr"].map(format_lr)
    rows["source_path"] = repo_relative(source)
    return rows


def _add_no_sweep_metric(no_sweep_row: dict[str, Any], metric: str, group: pd.DataFrame, source_col: str) -> None:
    mean, std = _mean_std(group[source_col])
    no_sweep_row[_metric_mean(metric)] = mean
    no_sweep_row[_metric_std(metric)] = std


def load_lung_rows(nicknames: dict[str, dict[str, str]], standard_records: list[dict[str, Any]]) -> pd.DataFrame:
    mean_rows = pd.read_csv(LUNG_MEAN_SOURCE).copy()
    mean_rows["dataset"] = "lungct10"
    mean_rows["method_id"] = mean_rows["method"].astype(str)
    mean_rows["display_method"] = [
        display_name(method_id, original_name, nicknames)
        for method_id, original_name in zip(mean_rows["method_id"], mean_rows["method_label"].astype(str))
    ]
    mean_rows["method_family"] = mean_rows["method_id"].map(_method_family)
    mean_rows["selected_lr"] = [_source_lr(row, standard_records) for row in mean_rows.to_dict("records")]
    mean_rows["selected_lr_label"] = mean_rows["selected_lr"].map(format_lr)
    mean_rows["source_path"] = repo_relative(LUNG_MEAN_SOURCE)

    case_rows = pd.read_csv(LUNG_NO_SWEEP_SOURCE)
    no_sweep_rows: list[dict[str, Any]] = []
    metric_sources = {
        "original_snap_tre_mean_mm": "native_crop_snap_tre_mean_mm",
        "original_subvoxel_tre_mean_mm": "native_crop_sub_tre_mean_mm",
        "reg1mm_snap_tre_mean_mm": "reg1mm_snap_tre_mean_mm",
        "reg1mm_subvoxel_tre_mean_mm": "reg1mm_sub_tre_mean_mm",
        "fg_logj_std": "fg_log_jdet_std",
        "fg_jdet_le0_pct": "fg_jdet_le0_pct",
        "fg_bending_energy": "fg_bending_energy",
        "fg_diffusion_l2": "fg_diffusion_l2",
    }
    for method_id in ["sinr_cps4_ptv_lcc_isotv", "direct_cp_cps4_ptv_lcc_isotv"]:
        group = case_rows[case_rows["method"].astype(str) == method_id]
        if group.empty:
            continue
        lr = safe_float(group["lr"].dropna().iloc[0]) if group["lr"].notna().any() else float("nan")
        row: dict[str, Any] = {
            "method": method_id,
            "method_id": method_id,
            "dataset": "lungct10",
            "method_label": LUNGCT_METHOD_LABELS.get(method_id, method_id),
            "display_method": display_name(method_id, LUNGCT_METHOD_LABELS.get(method_id, method_id), nicknames),
            "method_family": _method_family(method_id),
            "reg_value": float("nan"),
            "reg_label": "default",
            "ratio_to_best_tre_weight": float("nan"),
            "cases": int(group["case_id"].nunique()),
            "selected_lr": lr,
            "selected_lr_label": format_lr(lr),
            "source_path": repo_relative(LUNG_NO_SWEEP_SOURCE),
        }
        for metric, source_col in metric_sources.items():
            _add_no_sweep_metric(row, metric, group, source_col)
        no_sweep_rows.append(row)
    if no_sweep_rows:
        mean_rows = pd.concat([mean_rows, pd.DataFrame(no_sweep_rows)], ignore_index=True, sort=False)
    return mean_rows


def _configure_oasis_plot_module(
    nicknames: dict[str, dict[str, str]],
    *,
    annotate: bool,
) -> Any:
    script_dir = repo_path("scripts/experiments/oasis_alpha_sweep")
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import plot_first10_first20_alpha5_saved_baselines as oasis_plot  # type: ignore

    if OASIS_INCLUDE_DIRECTCP842:
        import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as oasis_plus  # type: ignore

        oasis_plus.configure_plot_module(1e-3)

    oasis_plot.ACCURACY_METRICS = list(OASIS_ACCURACY_METRICS)
    oasis_plot.REGULARITY_METRICS = list(OASIS_REGULARITY_METRICS)
    oasis_plot.SPOTLIGHT_PANELS = list(OASIS_SPOTLIGHT_PANELS)
    oasis_plot.STYLES = list(OASIS_STYLES)

    for method_id, spec in oasis_plot.METHOD_SPECS.items():
        original_label = str(spec.get("label", method_id))
        original_dot = str(spec.get("dot_label", original_label))
        spec["label"] = display_name(method_id, original_label, nicknames)
        spec["dot_label"] = display_name(method_id, original_dot, nicknames)

    def apply_oasis_font_sizes(ax: plt.Axes) -> None:
        ax.tick_params(axis="both", labelsize=OASIS_TICK_LABEL_SIZE)
        ax.xaxis.label.set_size(OASIS_AXIS_LABEL_SIZE)
        ax.yaxis.label.set_size(OASIS_AXIS_LABEL_SIZE)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(OASIS_LEGEND_FONT_SIZE)

    def save_figure_png_only(fig: plt.Figure, path_base: Path, dpi: int) -> list[Path]:
        path_base.parent.mkdir(parents=True, exist_ok=True)
        outputs = []
        for fmt in OUTPUT_FORMATS:
            path = path_base.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi)
            outputs.append(path)
        plt.close(fig)
        return outputs

    def plot_small_panel(
        rows: Sequence[dict[str, object]],
        out_dir: Path,
        *,
        cohort: str,
        style: str,
        acc_key: str,
        reg_key: str,
        dpi: int,
    ) -> list[Path]:
        fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.4))
        oasis_plot.draw_panel(
            ax,
            rows,
            acc_key=acc_key,
            reg_key=reg_key,
            style=style,
            show_legend=True,
            annotate=annotate,
        )
        apply_oasis_font_sizes(ax)
        ax.set_title(
            f"{cohort}: {oasis_plot.accuracy_label(acc_key)} vs "
            f"{oasis_plot.label_for_metric(reg_key, oasis_plot.REGULARITY_METRICS)}",
            fontsize=10,
        )
        fig.tight_layout()
        return oasis_plot.save_figure(fig, out_dir / f"{style}__{acc_key}_vs_{reg_key}", dpi)

    def plot_contact_sheet(
        rows: Sequence[dict[str, object]],
        out_dir: Path,
        *,
        cohort: str,
        style: str,
        dpi: int,
    ) -> list[Path]:
        n_rows = len(oasis_plot.ACCURACY_METRICS)
        n_cols = len(oasis_plot.REGULARITY_METRICS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.55 * n_rows), squeeze=False)
        for row_idx, (acc_key, acc_label, _higher) in enumerate(oasis_plot.ACCURACY_METRICS):
            for col_idx, (reg_key, reg_label) in enumerate(oasis_plot.REGULARITY_METRICS):
                ax = axes[row_idx][col_idx]
                oasis_plot.draw_panel(
                    ax,
                    rows,
                    acc_key=acc_key,
                    reg_key=reg_key,
                    style=style,
                    show_legend=(row_idx == 0 and col_idx == 0),
                    annotate=annotate,
                )
                apply_oasis_font_sizes(ax)
                ax.set_title(f"{acc_label} vs {reg_label}", fontsize=8.5)
        sweep_text = "six alpha sweeps" if OASIS_INCLUDE_DIRECTCP842 else "five alpha sweeps"
        fig.suptitle(
            f"OASIS seed-3 {cohort}: {sweep_text} with DL/Greedy references ({style})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        return oasis_plot.save_figure(fig, out_dir / f"contact_sheet_{style}", dpi)

    def plot_spotlight(
        rows: Sequence[dict[str, object]],
        out_dir: Path,
        *,
        cohort: str,
        style: str,
        acc_key: str,
        reg_key: str,
        dpi: int,
    ) -> list[Path]:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
        oasis_plot.draw_panel(
            ax,
            rows,
            acc_key=acc_key,
            reg_key=reg_key,
            style=style,
            show_legend=True,
            annotate=annotate,
        )
        apply_oasis_font_sizes(ax)
        ax.set_title(
            f"OASIS seed-3 {cohort}: {oasis_plot.accuracy_label(acc_key)} vs "
            f"{oasis_plot.label_for_metric(reg_key, oasis_plot.REGULARITY_METRICS)}",
            fontsize=11,
        )
        fig.tight_layout()
        return oasis_plot.save_figure(fig, out_dir / f"spotlight_{style}__{acc_key}_vs_{reg_key}", dpi)

    oasis_plot.save_figure = save_figure_png_only
    oasis_plot.plot_small_panel = plot_small_panel
    oasis_plot.plot_contact_sheet = plot_contact_sheet
    oasis_plot.plot_spotlight = plot_spotlight
    return oasis_plot


def generate_oasis_plots(
    output_dir: Path,
    oasis: pd.DataFrame,
    nicknames: dict[str, dict[str, str]],
    *,
    dirname: str,
    annotate: bool,
) -> list[Path]:
    oasis_dir = output_dir / dirname
    if oasis_dir.exists():
        shutil.rmtree(oasis_dir)
    plot_module = _configure_oasis_plot_module(nicknames, annotate=annotate)
    return plot_module.generate_plots_for_cohort(
        oasis_dir,
        cohort=OASIS_COHORT,
        pop_rows=oasis.to_dict("records"),
        dpi=OASIS_DPI,
    )


def _should_exclude_lung(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method_id") or row.get("method"))
    ratio = safe_float(row.get("ratio_to_best_tre_weight"))
    weight = safe_float(row.get("reg_value"))
    for value in LUNGCT_EXCLUDE_BY_RATIO.get(method, ()):
        if _close(ratio, float(value)):
            return True
    for value in LUNGCT_EXCLUDE_BY_WEIGHT.get(method, ()):
        if _close(weight, float(value)):
            return True
    return False


def _lung_display_label(method_id: str, nicknames: dict[str, dict[str, str]]) -> str:
    return display_name(method_id, LUNGCT_METHOD_LABELS.get(method_id, method_id), nicknames)


def _annotation_text(row: Mapping[str, Any]) -> str:
    method = str(row.get("method_id") or row.get("method"))
    if method == "ptvreg_reference":
        return "pTVReg"
    ratio = safe_float(row.get("ratio_to_best_tre_weight"))
    if math.isfinite(ratio):
        return f"{ratio:g}x"
    label = str(row.get("reg_label") or "")
    return label if label else "default"


def _finite_pair_rows(rows: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = rows.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out = out[out[x_col].notna() & out[y_col].notna()].copy()
    if out.empty:
        return out
    out["sort_ratio"] = pd.to_numeric(out.get("ratio_to_best_tre_weight"), errors="coerce")
    out["sort_reg"] = pd.to_numeric(out.get("reg_value"), errors="coerce")
    return out.sort_values(["sort_ratio", "sort_reg"], na_position="last")


def plot_lung_metric_pair(
    rows: pd.DataFrame,
    output_path: Path,
    *,
    acc_key: str,
    acc_label: str,
    acc_higher_is_better: bool,
    reg_key: str,
    reg_label: str,
    nicknames: dict[str, dict[str, str]],
    annotate: bool,
) -> Path:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": LUNGCT_AXIS_LABEL_SIZE,
            "xtick.labelsize": LUNGCT_TICK_LABEL_SIZE,
            "ytick.labelsize": LUNGCT_TICK_LABEL_SIZE,
            "legend.fontsize": LUNGCT_LEGEND_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=LUNGCT_FIGSIZE)
    x_col = _metric_mean(reg_key)
    y_col = _metric_mean(acc_key)
    x_std_col = _metric_std(reg_key)
    y_std_col = _metric_std(acc_key)

    for method in LUNGCT_PLOT_METHOD_ORDER:
        method_rows = rows[rows["method_id"].astype(str) == method].copy()
        if method_rows.empty:
            continue
        method_rows = _finite_pair_rows(method_rows, x_col, y_col)
        if method_rows.empty:
            continue
        color = LUNGCT_METHOD_COLORS.get(method, "#444444")
        marker = LUNGCT_METHOD_MARKERS.get(method, "o")
        xs = method_rows[x_col].to_numpy(dtype=float)
        ys = method_rows[y_col].to_numpy(dtype=float)
        if method != "ptvreg_reference" and len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75, zorder=2)
        for _, row in method_rows.iterrows():
            x = safe_float(row[x_col])
            y = safe_float(row[y_col])
            xe = max(safe_float(row.get(x_std_col)), 0.0)
            ye = max(safe_float(row.get(y_std_col)), 0.0)
            if math.isfinite(xe) and math.isfinite(ye) and (xe > 0 or ye > 0):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * xe,
                        height=2.0 * ye,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.12 if method != "ptvreg_reference" else 0.18,
                        linewidth=1.0,
                        zorder=1,
                    )
                )
        size = 120 if method == "ptvreg_reference" else 50
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=_lung_display_label(method, nicknames),
            zorder=3,
        )
        if annotate:
            for _, row in method_rows.iterrows():
                x = safe_float(row[x_col])
                y = safe_float(row[y_col])
                if math.isfinite(x) and math.isfinite(y):
                    ax.annotate(
                        _annotation_text(row),
                        (x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
    ax.set_xlabel(reg_label)
    ax.set_ylabel(acc_label)
    ax.set_title(f"{LUNGCT_SUBSET_NAME.replace('_', ' ')}: {acc_label} vs {reg_label}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LUNGCT_LEGEND_FONT_SIZE, frameon=True)
    if acc_higher_is_better:
        ax.invert_yaxis()
    ax.autoscale_view()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=LUNGCT_DPI)
    plt.close(fig)
    return output_path


def generate_lung_plots(
    output_dir: Path,
    lung: pd.DataFrame,
    nicknames: dict[str, dict[str, str]],
    *,
    dirname: str,
    annotate: bool,
) -> list[Path]:
    lung_dir = output_dir / dirname / LUNGCT_SUBSET_NAME
    if lung_dir.parent.exists():
        shutil.rmtree(lung_dir.parent)
    subset = lung.loc[[not _should_exclude_lung(row) for _, row in lung.iterrows()]].copy()
    paths: list[Path] = []
    for acc_key, acc_label, acc_higher in LUNGCT_ACCURACY_METRICS:
        for reg_key, reg_label in LUNGCT_REGULARITY_METRICS:
            path = lung_dir / f"{_slug(acc_key)}_vs_{_slug(reg_key)}.png"
            paths.append(
                plot_lung_metric_pair(
                    subset,
                    path,
                    acc_key=acc_key,
                    acc_label=acc_label,
                    acc_higher_is_better=acc_higher,
                    reg_key=reg_key,
                    reg_label=reg_label,
                    nicknames=nicknames,
                    annotate=annotate,
                )
            )
    return paths


def _patch_oasis_composite_ellipse(oasis_plot: Any, alpha: float) -> None:
    def add_ellipse(ax: plt.Axes, row: Mapping[str, Any], x: float, y: float, reg_key: str, acc_key: str, color: str) -> None:
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

    oasis_plot.add_ellipse = add_ellipse


def _apply_composite_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=COMPOSITE_STYLE["tick_label_fontsize"])
    ax.xaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.yaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.grid(True, alpha=COMPOSITE_STYLE["grid_alpha"])


def _normalize_composite_legend(ax: plt.Axes, *, ncol: int) -> None:
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
        ncol=ncol,
        fontsize=COMPOSITE_STYLE["legend_fontsize"],
        frameon=COMPOSITE_STYLE["legend_frame"],
    )


def _draw_composite_oasis_panel(
    ax: plt.Axes,
    oasis_plot: Any,
    oasis: pd.DataFrame,
    *,
    title: str,
    acc_key: str,
    annotate: bool,
) -> None:
    oasis_plot.REGULARITY_METRICS = [("log_jdet_std_fg", "SDlogJ")]
    _patch_oasis_composite_ellipse(oasis_plot, float(COMPOSITE_STYLE["ellipse_alpha"]))
    oasis_plot.draw_panel(
        ax,
        oasis.to_dict("records"),
        acc_key=acc_key,
        reg_key="log_jdet_std_fg",
        style="ellipses",
        show_legend=True,
        annotate=annotate,
    )
    ax.set_title(title, fontsize=COMPOSITE_STYLE["title_fontsize"])
    _apply_composite_axis_style(ax)
    _normalize_composite_legend(ax, ncol=2)


def _draw_composite_lung_panel(
    ax: plt.Axes,
    lung_subset: pd.DataFrame,
    *,
    title: str,
    acc_key: str,
    acc_label: str,
    acc_higher_is_better: bool,
    reg_key: str,
    reg_label: str,
    nicknames: dict[str, dict[str, str]],
    annotate: bool,
) -> None:
    x_col = _metric_mean(reg_key)
    y_col = _metric_mean(acc_key)
    x_std_col = _metric_std(reg_key)
    y_std_col = _metric_std(acc_key)
    for method in LUNGCT_PLOT_METHOD_ORDER:
        method_rows = lung_subset[lung_subset["method_id"].astype(str) == method].copy()
        if method_rows.empty:
            continue
        method_rows = _finite_pair_rows(method_rows, x_col, y_col)
        if method_rows.empty:
            continue
        color = LUNGCT_METHOD_COLORS.get(method, "#444444")
        marker = LUNGCT_METHOD_MARKERS.get(method, "o")
        xs = method_rows[x_col].to_numpy(dtype=float)
        ys = method_rows[y_col].to_numpy(dtype=float)
        if method != "ptvreg_reference" and len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=COMPOSITE_STYLE["line_width"], alpha=0.75, zorder=2)
        for _, row in method_rows.iterrows():
            x = safe_float(row[x_col])
            y = safe_float(row[y_col])
            xe = max(safe_float(row.get(x_std_col)), 0.0)
            ye = max(safe_float(row.get(y_std_col)), 0.0)
            if math.isfinite(xe) and math.isfinite(ye) and (xe > 0 or ye > 0):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * xe,
                        height=2.0 * ye,
                        facecolor=color,
                        edgecolor=color,
                        alpha=COMPOSITE_STYLE["lung_ellipse_alpha"],
                        linewidth=1.0,
                        zorder=1,
                    )
                )
        size = COMPOSITE_STYLE["ptvreg_marker_size"] if method == "ptvreg_reference" else COMPOSITE_STYLE["lung_marker_size"]
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=_lung_display_label(method, nicknames),
            zorder=3,
        )
        if annotate:
            for _, row in method_rows.iterrows():
                x = safe_float(row[x_col])
                y = safe_float(row[y_col])
                if math.isfinite(x) and math.isfinite(y):
                    ax.annotate(
                        _annotation_text(row),
                        (x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=max(COMPOSITE_STYLE["legend_fontsize"] - 2, 6),
                        color=color,
                    )
    ax.set_xlabel(reg_label)
    ax.set_ylabel(acc_label)
    ax.set_title(title, fontsize=COMPOSITE_STYLE["title_fontsize"])
    if acc_higher_is_better:
        ax.invert_yaxis()
    ax.autoscale_view()
    _apply_composite_axis_style(ax)
    _normalize_composite_legend(ax, ncol=1)


def generate_four_panel_composite(
    output_dir: Path,
    oasis: pd.DataFrame,
    lung: pd.DataFrame,
    nicknames: dict[str, dict[str, str]],
) -> list[Path]:
    composite_dir = output_dir / COMPOSITE_DIRNAME
    if composite_dir.exists():
        shutil.rmtree(composite_dir)
    composite_dir.mkdir(parents=True, exist_ok=True)
    lung_subset = lung.loc[[not _should_exclude_lung(row) for _, row in lung.iterrows()]].copy()
    panels = [
        {
            "dataset": "oasis",
            "title": f"Dice vs SDlogJ (OASIS N={OASIS_PANEL_N})",
            "acc_key": "dice_mean",
            "reg_key": "log_jdet_std_fg",
            "style": "ellipses",
            "legend": {"show": True, "loc": "best", "ncol": 2},
        },
        {
            "dataset": "oasis",
            "title": f"HD95 vs SDlogJ (OASIS N={OASIS_PANEL_N})",
            "acc_key": "hd95_mean",
            "reg_key": "log_jdet_std_fg",
            "style": "ellipses",
            "legend": {"show": True, "loc": "best", "ncol": 2},
        },
        {
            "dataset": "lungct",
            "title": "TRE vs SDlogJ (4DCT N=10)",
            "acc_key": "original_snap_tre_mean_mm",
            "acc_label": "TRE (mm)",
            "acc_higher_is_better": False,
            "reg_key": "fg_logj_std",
            "reg_label": "Foreground SDlogJ",
            "legend": {"show": True, "loc": "best", "ncol": 1},
        },
        {
            "dataset": "lungct",
            "title": "TRE vs bending energy (4DCT N=10)",
            "acc_key": "original_snap_tre_mean_mm",
            "acc_label": "TRE (mm)",
            "acc_higher_is_better": False,
            "reg_key": "fg_bending_energy",
            "reg_label": "Foreground bending energy",
            "legend": {"show": True, "loc": "best", "ncol": 1},
        },
    ]
    saved_outputs: list[Path] = []
    for _mode_name, mode in COMPOSITE_OUTPUTS.items():
        fig, axes = plt.subplots(2, 2, figsize=COMPOSITE_STYLE["figsize"], squeeze=False)
        oasis_plot = _configure_oasis_plot_module(nicknames, annotate=bool(mode["annotate"]))
        for ax, panel in zip(axes.ravel(), panels):
            if panel["dataset"] == "oasis":
                _draw_composite_oasis_panel(
                    ax,
                    oasis_plot,
                    oasis,
                    title=str(panel["title"]),
                    acc_key=str(panel["acc_key"]),
                    annotate=bool(mode["annotate"]),
                )
            else:
                _draw_composite_lung_panel(
                    ax,
                    lung_subset,
                    title=str(panel["title"]),
                    acc_key=str(panel["acc_key"]),
                    acc_label=str(panel["acc_label"]),
                    acc_higher_is_better=bool(panel["acc_higher_is_better"]),
                    reg_key=str(panel["reg_key"]),
                    reg_label=str(panel["reg_label"]),
                    nicknames=nicknames,
                    annotate=bool(mode["annotate"]),
                )
        fig.subplots_adjust(wspace=COMPOSITE_STYLE["wspace"], hspace=COMPOSITE_STYLE["hspace"])
        output_path = composite_dir / str(mode["filename"])
        fig.savefig(output_path, dpi=COMPOSITE_STYLE["dpi"], bbox_inches="tight")
        plt.close(fig)
        saved_outputs.append(output_path)
    manifest = {
        "artifact": "pareto_four_panel_composite",
        "source_version": f"{standard_version_for_manifest(output_dir)}",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            OASIS_DATASET: repo_relative(output_dir / OASIS_SOURCE_ROWS_FILENAME),
            "lungct10": repo_relative(output_dir / "lungct10_pareto_source_rows.csv"),
            "method_nicknames": "experiments/paper_artifacts/method_nicknames.csv",
        },
        "outputs": [repo_relative(path) for path in saved_outputs],
        "output_formats": list(OUTPUT_FORMATS),
        "annotation_modes": {
            name: {"annotate": bool(spec["annotate"]), "filename": spec["filename"]}
            for name, spec in COMPOSITE_OUTPUTS.items()
        },
        "panels": panels,
        "style": COMPOSITE_STYLE,
        "label_conventions": {
            "logj_label_style": "compact",
            "logj_label": "SDlogJ",
            "lungct_y_label": "TRE (mm)",
        },
        "lungct_exclusion_rule": {
            "exclude_by_ratio": LUNGCT_EXCLUDE_BY_RATIO,
            "exclude_by_weight": LUNGCT_EXCLUDE_BY_WEIGHT,
        },
    }
    (composite_dir / "pareto_four_panel_composite_manifest.json").write_text(
        json.dumps(sanitize_public_value(manifest), indent=2) + "\n"
    )
    return saved_outputs


def standard_version_for_manifest(output_dir: Path) -> str:
    return output_dir.name


def write_selected_hypervariants(output_dir: Path, standard: Mapping[str, Any]) -> None:
    records = selected_hypervariant_records("pareto_figure", standard)
    summary_rows = [
        {
            "dataset": row.get("dataset", ""),
            "method_id": row.get("method_id", ""),
            "lr": format_lr(row.get("lr")),
            "variant": row.get("variant", ""),
        }
        for row in records
    ]
    (output_dir / "selected_hypervariants.md").write_text(
        "# Pareto Figure Selected Hypervariants\n\n"
        + markdown_table(["dataset", "method_id", "lr", "variant"], summary_rows)
        + "\n"
    )


def write_manifest(
    output_dir: Path,
    standard: Mapping[str, Any],
    validation_rows: list[dict[str, Any]],
    *,
    standard_path: str,
    nickname_path: str,
    oasis: pd.DataFrame,
    lung: pd.DataFrame,
    oasis_paths: Sequence[Path],
    oasis_paper_paths: Sequence[Path],
    lung_paths: Sequence[Path],
    lung_paper_paths: Sequence[Path],
    composite_paths: Sequence[Path],
) -> None:
    records = selected_hypervariant_records("pareto_figure", standard)
    outputs = {
        f"{OASIS_DATASET}_annotated": _relative_paths(oasis_paths, output_dir),
        f"{OASIS_DATASET}_paper_no_annotations": _relative_paths(oasis_paper_paths, output_dir),
        "lungct10_annotated": _relative_paths(lung_paths, output_dir),
        "lungct10_paper_no_annotations": _relative_paths(lung_paper_paths, output_dir),
        "four_panel_composite": _relative_paths(composite_paths, output_dir),
    }
    manifest = {
        "artifact": "pareto_figure",
        "version": standard["artifact_versions"]["pareto_figure"]["current"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "standard_path": standard_path,
        "nickname_path": nickname_path,
        "output_formats": list(OUTPUT_FORMATS),
        "source_files": {
            OASIS_DATASET: repo_relative(_oasis_source(standard)),
            "lungct10_mean_rows": repo_relative(LUNG_MEAN_SOURCE),
            "lungct10_single_stage_rows": repo_relative(LUNG_NO_SWEEP_SOURCE),
        },
        "plot_counts": {
            f"{OASIS_DATASET}_annotated_png": len(oasis_paths),
            f"{OASIS_DATASET}_paper_no_annotations_png": len(oasis_paper_paths),
            "lungct10_annotated_png": len(lung_paths),
            "lungct10_paper_no_annotations_png": len(lung_paper_paths),
            "four_panel_composite_png": len(composite_paths),
            "total_png": len(oasis_paths) + len(oasis_paper_paths) + len(lung_paths) + len(lung_paper_paths) + len(composite_paths),
        },
        "metric_pair_coverage": {
            OASIS_DATASET: {
                "accuracy_metrics": [item[0] for item in OASIS_ACCURACY_METRICS],
                "regularity_metrics": [item[0] for item in OASIS_REGULARITY_METRICS],
                "styles": list(OASIS_STYLES),
                "spotlight_panels": [list(item) for item in OASIS_SPOTLIGHT_PANELS],
                "include_directcp842": OASIS_INCLUDE_DIRECTCP842,
                "annotated_dir": OASIS_ANNOTATED_DIRNAME,
                "paper_no_annotations_dir": OASIS_PAPER_NO_ANNOTATIONS_DIRNAME,
            },
            "lungct10": {
                "subset": LUNGCT_SUBSET_NAME,
                "accuracy_metrics": [item[0] for item in LUNGCT_ACCURACY_METRICS],
                "regularity_metrics": [item[0] for item in LUNGCT_REGULARITY_METRICS],
                "exclude_by_ratio": LUNGCT_EXCLUDE_BY_RATIO,
                "exclude_by_weight": LUNGCT_EXCLUDE_BY_WEIGHT,
                "annotated_dir": LUNGCT_ANNOTATED_DIRNAME,
                "paper_no_annotations_dir": LUNGCT_PAPER_NO_ANNOTATIONS_DIRNAME,
            },
        },
        "method_styles": {
            "lungct10_plot_method_order": LUNGCT_PLOT_METHOD_ORDER,
            "lungct10_colors": LUNGCT_METHOD_COLORS,
            "lungct10_markers": LUNGCT_METHOD_MARKERS,
        },
        "font_sizes": {
            OASIS_DATASET: {
                "tick_labels": OASIS_TICK_LABEL_SIZE,
                "axis_labels": OASIS_AXIS_LABEL_SIZE,
                "legend": OASIS_LEGEND_FONT_SIZE,
            },
            "lungct10": {
                "tick_labels": LUNGCT_TICK_LABEL_SIZE,
                "axis_labels": LUNGCT_AXIS_LABEL_SIZE,
                "legend": LUNGCT_LEGEND_FONT_SIZE,
            },
        },
        "display_name_mapping": _display_name_mapping(oasis, lung),
        "outputs": outputs,
        "selected_hypervariants": records,
        "validation_status_counts": pd.Series([row["status"] for row in validation_rows]).value_counts().to_dict(),
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_review_report(
    output_dir: Path,
    validation_rows: list[dict[str, Any]],
    *,
    version: str,
    oasis_paths: Sequence[Path],
    oasis_paper_paths: Sequence[Path],
    lung_paths: Sequence[Path],
    lung_paper_paths: Sequence[Path],
    composite_paths: Sequence[Path],
) -> None:
    lines = [
        f"# Pareto Figure {version} Plot Package",
        "",
        f"- Output formats: `{', '.join(OUTPUT_FORMATS)}`",
        f"- OASIS annotated PNGs: `{len(oasis_paths)}`",
        f"- OASIS paper no-annotation PNGs: `{len(oasis_paper_paths)}`",
        f"- LungCT annotated PNGs: `{len(lung_paths)}`",
        f"- LungCT paper no-annotation PNGs: `{len(lung_paper_paths)}`",
        f"- Four-panel composite PNGs: `{len(composite_paths)}`",
        f"- OASIS DirectCP 8-4-2 included: `{OASIS_INCLUDE_DIRECTCP842}`",
        f"- LungCT subset: `{LUNGCT_SUBSET_NAME}`",
        f"- LungCT plotted methods: `{LUNGCT_PLOT_METHOD_ORDER}`",
        f"- OASIS fonts: tick labels `{OASIS_TICK_LABEL_SIZE}`, axis labels `{OASIS_AXIS_LABEL_SIZE}`, legend `{OASIS_LEGEND_FONT_SIZE}`",
        f"- LungCT fonts: tick labels `{LUNGCT_TICK_LABEL_SIZE}`, axis labels `{LUNGCT_AXIS_LABEL_SIZE}`, legend `{LUNGCT_LEGEND_FONT_SIZE}`",
        f"- Blocking LR validation issues: `{sum(row['status'] == 'blocking_issue' for row in validation_rows)}`",
        "",
        "## LungCT Exclusions",
        "",
        f"- By ratio: `{LUNGCT_EXCLUDE_BY_RATIO}`",
        f"- By weight: `{LUNGCT_EXCLUDE_BY_WEIGHT}`",
        "",
        "## Files",
        "",
        f"- `{OASIS_ANNOTATED_DIRNAME}/small_plots/`: annotated OASIS accuracy-regularity pairs.",
        f"- `{OASIS_ANNOTATED_DIRNAME}/big_plots/`: annotated OASIS contact sheets and spotlight plots.",
        f"- `{OASIS_PAPER_NO_ANNOTATIONS_DIRNAME}/small_plots/`: OASIS pairs without alpha/reference text annotations.",
        f"- `{OASIS_PAPER_NO_ANNOTATIONS_DIRNAME}/big_plots/`: OASIS contact sheets and spotlight plots without alpha/reference text annotations.",
        f"- `{LUNGCT_ANNOTATED_DIRNAME}/{LUNGCT_SUBSET_NAME}/`: annotated original snap TRE against each LungCT regularity metric.",
        f"- `{LUNGCT_PAPER_NO_ANNOTATIONS_DIRNAME}/{LUNGCT_SUBSET_NAME}/`: original snap TRE plots without ratio/pTVReg text annotations.",
        f"- `{COMPOSITE_DIRNAME}/`: combined OASIS100/LungCT paper panels.",
        "",
    ]
    (output_dir / "review_report.md").write_text("\n".join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-path", type=Path, default=None)
    parser.add_argument("--nickname-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    standard = load_standard(args.standard_path) if args.standard_path is not None else load_standard()
    nicknames = load_nicknames(args.nickname_path) if args.nickname_path is not None else load_nicknames()
    output_dir = artifact_output_dir("pareto_figure", standard)
    _clean_output_dir(output_dir)

    standard_records = selected_hypervariant_records("pareto_figure", standard)
    oasis = load_oasis_rows(nicknames, standard_records, standard)
    lung = load_lung_rows(nicknames, standard_records)
    _sanitize_dataframe(oasis).to_csv(output_dir / OASIS_SOURCE_ROWS_FILENAME, index=False)
    _sanitize_dataframe(lung).to_csv(output_dir / "lungct10_pareto_source_rows.csv", index=False)

    oasis_paths = generate_oasis_plots(
        output_dir,
        oasis,
        nicknames,
        dirname=OASIS_ANNOTATED_DIRNAME,
        annotate=True,
    )
    oasis_paper_paths = generate_oasis_plots(
        output_dir,
        oasis,
        nicknames,
        dirname=OASIS_PAPER_NO_ANNOTATIONS_DIRNAME,
        annotate=False,
    )
    lung_paths = generate_lung_plots(
        output_dir,
        lung,
        nicknames,
        dirname=LUNGCT_ANNOTATED_DIRNAME,
        annotate=True,
    )
    lung_paper_paths = generate_lung_plots(
        output_dir,
        lung,
        nicknames,
        dirname=LUNGCT_PAPER_NO_ANNOTATIONS_DIRNAME,
        annotate=False,
    )
    composite_paths = generate_four_panel_composite(output_dir, oasis, lung, nicknames)

    validation_rows = validate_lr_records(standard_records, artifact="pareto_figure", standard=standard)
    validation_rows = [sanitize_public_value(row) for row in validation_rows]
    pd.DataFrame(validation_rows).to_csv(output_dir / "standard_validation.csv", index=False)
    (output_dir / "standard_validation_report.md").write_text(
        validation_report_markdown("Pareto Figure LR Validation", validation_rows)
    )
    write_selected_hypervariants(output_dir, standard)
    write_manifest(
        output_dir,
        standard,
        validation_rows,
        standard_path=repo_relative(args.standard_path) if args.standard_path is not None else "experiments/paper_artifacts/artifact_standard.json",
        nickname_path=repo_relative(args.nickname_path) if args.nickname_path is not None else "experiments/paper_artifacts/method_nicknames.csv",
        oasis=oasis,
        lung=lung,
        oasis_paths=oasis_paths,
        oasis_paper_paths=oasis_paper_paths,
        lung_paths=lung_paths,
        lung_paper_paths=lung_paper_paths,
        composite_paths=composite_paths,
    )
    write_review_report(
        output_dir,
        validation_rows,
        version=str(standard["artifact_versions"]["pareto_figure"]["current"]),
        oasis_paths=oasis_paths,
        oasis_paper_paths=oasis_paper_paths,
        lung_paths=lung_paths,
        lung_paper_paths=lung_paper_paths,
        composite_paths=composite_paths,
    )
    print(
        f"wrote {repo_relative(output_dir)} "
        f"({len(oasis_paths)} OASIS annotated PNGs, "
        f"{len(oasis_paper_paths)} OASIS paper PNGs, "
        f"{len(lung_paths)} LungCT annotated PNGs, {len(lung_paper_paths)} LungCT paper PNGs, "
        f"{len(composite_paths)} composite PNGs)"
    )


if __name__ == "__main__":
    main()
