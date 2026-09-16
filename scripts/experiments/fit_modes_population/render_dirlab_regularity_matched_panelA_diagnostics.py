#!/usr/bin/env python
"""Render v004 fit-mode diagnostics with DIR-LAB regularity-matched Panel A.

v004 starts from the v003 fixed-domain source index and replaces only the
DIR-LAB Panel-A single-stage BSCP rows:

* INR-BSCP cps4: alpha 10000 -> 20000
* D-BSCP cps4: alpha 10000 -> 50000

The replacement curves are exact-compatible reuses from the accepted all-10
FD-BE cps4 alpha-landscape experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import render_lungmask_panelA_population_diagnostics as base
import render_masked_panelA_both_datasets_diagnostics as v003
from plot_case_diagnostics import DiagnosticCurve, METHOD_ORDER, METHOD_STYLE, normalize_terminal_endpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


EXPERIMENT_KEY = "paper_artifacts/fit_modes_population_diagnostics/v004_dirlab_regularity_matched_panelA"
ALPHA_RUN_KEY = "lungct_fdbe_cps4_alpha_landscape_all10_v1"
POPULATION_RUN_KEY = "fit_modes_population_n10_v1"
REFRESH_RUN_KEY = "fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
LUNGMASK_RUN_KEY = "dirlab_case01_case08_panelA_lungmask_diag_curves_v1"
OASIS_MASKED_RUN_KEY = "oasis_panelA_fixedmask_diag_curves_v1"
ALPHA_TOKEN = "${DIRLAB_FDBE_CPS4_ALPHA_LANDSCAPE_RUN_ROOT}"
EXPECTED_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=int)
V004_AXIS_LIMITS = {"A": (1.0e-2, 1.0e0), "B": (1.0e-1, 1.0e3)}
TRE_LABEL_ORDER = ["inr_bscp", "d_bscp", "idir", "mr_d_bscp"]
FIG3_CAMERA_READY_OUTPUT = "population_overlay_plus_lungct_cases_2x4_tre_labels_camera_ready_fig3.png"
FIG3_CAMERA_READY_FONTS = {
    "tick": 12.0,
    "xlabel": 14.0,
    "legend": 16.0,
    "subtitle": 16.0,
    "panel_title": 20.0,
    "ylabel": 16.0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v003-root", type=Path, required=True)
    parser.add_argument("--alpha-case-rows", type=Path, required=True)
    parser.add_argument("--alpha-run-root", type=Path, default=None)
    parser.add_argument("--population-run-root", type=Path, default=None)
    parser.add_argument("--refresh-run-root", type=Path, default=None)
    parser.add_argument("--lungmask-run-root", type=Path, default=None)
    parser.add_argument("--oasis-masked-run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Use only external stores for the v004 output and generated inputs."""

    args.v003_root = artifact_paths.resolve_external_input_directory(
        args.v003_root,
        label="--v003-root",
        repo_root=REPO_ROOT,
    )
    args.alpha_case_rows = artifact_paths.resolve_external_input_file(
        args.alpha_case_rows,
        label="--alpha-case-rows",
        repo_root=REPO_ROOT,
    )
    args.alpha_run_root = artifact_paths.resolve_run_root(ALPHA_RUN_KEY, args.alpha_run_root, repo_root=REPO_ROOT)
    args.population_run_root = artifact_paths.resolve_run_root(POPULATION_RUN_KEY, args.population_run_root, repo_root=REPO_ROOT)
    args.refresh_run_root = artifact_paths.resolve_run_root(REFRESH_RUN_KEY, args.refresh_run_root, repo_root=REPO_ROOT)
    args.lungmask_run_root = artifact_paths.resolve_run_root(LUNGMASK_RUN_KEY, args.lungmask_run_root, repo_root=REPO_ROOT)
    args.oasis_masked_run_root = artifact_paths.resolve_run_root(OASIS_MASKED_RUN_KEY, args.oasis_masked_run_root, repo_root=REPO_ROOT)
    args.output_root = artifact_paths.resolve_review_root(EXPERIMENT_KEY, args.output_root, repo_root=REPO_ROOT)
    return args


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable_alpha_path(value: Any, alpha_root: Path) -> str:
    text = str(value or "")
    if text.startswith("${RUN_ROOT}/"):
        return text.replace("${RUN_ROOT}", ALPHA_TOKEN, 1)
    if text.startswith(str(alpha_root)):
        return text.replace(str(alpha_root), ALPHA_TOKEN, 1)
    return text


def resolve_path(value: Any, roots: dict[str, Path]) -> Path:
    return base.resolve_portable(value, roots)


def selected_alpha_rows(alpha_rows: pd.DataFrame) -> pd.DataFrame:
    target = (
        (
            alpha_rows["method_id"].eq("sinr_cps4_global_ncc_fd_be")
            & np.isclose(alpha_rows["alpha"].astype(float), 20000.0)
        )
        | (
            alpha_rows["method_id"].eq("direct_cp_cps4_global_ncc_fd_be")
            & np.isclose(alpha_rows["alpha"].astype(float), 50000.0)
        )
    )
    selected = alpha_rows[target].copy()
    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 selected regularity-matched rows, found {len(selected)}")
    duplicate = selected.duplicated(["case_key", "method_slug"])
    if duplicate.any():
        raise RuntimeError("Selected alpha rows contain duplicate case/method keys")
    return selected


def validate_alpha_curve(row: pd.Series, alpha_root: Path) -> dict[str, Any]:
    roots = {ALPHA_TOKEN: alpha_root}
    curve_path = resolve_path(portable_alpha_path(row["curve_path"], alpha_root), roots)
    summary_path = resolve_path(portable_alpha_path(row["summary_path"], alpha_root), roots)
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    curve = pd.read_csv(curve_path)
    required = ["global_epoch", "global_ncc_value", "lung_mask_ncc_value", "lung_mask_one_minus_ncc"]
    missing = [column for column in required if column not in curve.columns]
    if missing:
        raise RuntimeError(f"{curve_path}: missing columns {missing}")
    epochs = curve["global_epoch"].astype(int).to_numpy()
    if not np.array_equal(epochs, EXPECTED_EPOCHS):
        raise RuntimeError(f"{curve_path}: expected 251 diagnostics ending at 2500")
    values = curve[required[1:]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{curve_path}: nonfinite NCC diagnostic values")
    if not (curve["lung_mask_one_minus_ncc"].to_numpy(float) > 0.0).all():
        raise RuntimeError(f"{curve_path}: nonpositive lung-mask 1-NCC")
    summary = base.load_json(summary_path)
    if summary.get("status") != "completed" and not (summary_path.parent / "DONE").is_file():
        raise RuntimeError(f"{summary_path}: summary status is not completed")
    final = curve.iloc[-1]
    endpoint = float(final["lung_mask_one_minus_ncc"])
    expected_endpoint = float(row["fixed_t00_lung_mask_one_minus_ncc"])
    if not np.isclose(endpoint, expected_endpoint, rtol=2.0e-5, atol=2.0e-8):
        raise RuntimeError(
            f"{curve_path}: endpoint mismatch curve={endpoint}, case_rows={expected_endpoint}"
        )
    if int(row["diagnostic_rows"]) != 251:
        raise RuntimeError(f"{row['case_key']}/{row['method_slug']}: case_rows diagnostic count mismatch")
    return {
        "curve_path": curve_path,
        "summary_path": summary_path,
        "point_count": int(len(curve)),
        "epoch_max": int(epochs[-1]),
        "endpoint_value": endpoint,
        "diagnostic_ncc": float(final["lung_mask_ncc_value"]),
        "full_crop_one_minus_ncc": 1.0 - float(final["global_ncc_value"]),
        "curve_min": float(np.min(curve["lung_mask_one_minus_ncc"].to_numpy(float))),
        "curve_max": float(np.max(curve["lung_mask_one_minus_ncc"].to_numpy(float))),
        "curve_sha256": base.sha256(curve_path),
        "summary_sha256": base.sha256(summary_path),
    }


def build_v004_source_index(
    v003_index: pd.DataFrame,
    alpha_rows: pd.DataFrame,
    alpha_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    if len(v003_index) != 160:
        raise RuntimeError(f"Expected 160 v003 rows, found {len(v003_index)}")
    selected = selected_alpha_rows(alpha_rows)
    selected_by_key = {
        (str(row["case_key"]), str(row["method_slug"])): row for _, row in selected.iterrows()
    }
    target = (
        v003_index["dataset"].eq("dirlab")
        & v003_index["panel"].eq("A")
        & v003_index["method_slug"].isin(["inr_bscp", "d_bscp"])
    )
    if int(target.sum()) != 20:
        raise RuntimeError(f"Expected 20 replaceable v003 rows, found {int(target.sum())}")

    v004 = v003_index.copy()
    audit: list[dict[str, Any]] = []
    for row_index, old in v003_index[target].iterrows():
        key = (str(old["case_key"]), str(old["method_slug"]))
        if key not in selected_by_key:
            raise RuntimeError(f"Missing selected alpha row for {key}")
        new = selected_by_key[key]
        check = validate_alpha_curve(new, alpha_root)
        expected_alpha = 20000.0 if key[1] == "inr_bscp" else 50000.0
        expected_lr = 0.0001 if key[1] == "inr_bscp" else 0.001
        if not np.isclose(float(new["alpha"]), expected_alpha):
            raise RuntimeError(f"{key}: alpha mismatch")
        if not np.isclose(float(new["lr"]), expected_lr):
            raise RuntimeError(f"{key}: lr mismatch")
        if int(float(new["cps"])) != 4 or int(float(new["epochs"])) != 2500 or int(float(new["seed"])) != 1:
            raise RuntimeError(f"{key}: cps/epochs/seed mismatch")
        updates = {
            "status": "completed_reused",
            "source_kind": "dirlab_regularity_matched_panelA_alpha_landscape_reuse",
            "source_path": portable_alpha_path(new["curve_path"], alpha_root),
            "summary_path": portable_alpha_path(new["summary_path"], alpha_root),
            "endpoint_name": "lung_mask_one_minus_ncc",
            "endpoint_value": check["endpoint_value"],
            "diagnostic_ncc": check["diagnostic_ncc"],
            "runtime_sec": float(new["elapsed_sec"]),
            "peak_memory_gb": float(new["peak_mem_gb"]),
            "endpoint_source": "fdbe_cps4_alpha_landscape:lung_mask_one_minus_ncc",
            "lr": float(new["lr"]),
            "alpha": float(new["alpha"]),
            "cps": "4",
            "stage_downsample": np.nan,
            "stage_epochs": np.nan,
            "epochs": 2500,
            "seed": 1,
            "diagnostic_interval": 10,
            "acceptance_checks": (
                "regularity_matched_panelA=true;exact_compatible_reuse=true;"
                "curve_rows=251;endpoint_source=lung_mask_one_minus_ncc"
            ),
        }
        for field, value in updates.items():
            v004.at[row_index, field] = value
        audit.append(
            {
                "job_id": old["job_id"],
                "case_key": old["case_key"],
                "method_slug": old["method_slug"],
                "display_method": old["display_method"],
                "old_v003_alpha": float(old["alpha"]),
                "new_v004_alpha": float(new["alpha"]),
                "old_v003_endpoint": float(old["endpoint_value"]),
                "new_v004_endpoint": check["endpoint_value"],
                "endpoint_delta": check["endpoint_value"] - float(old["endpoint_value"]),
                "tre_mean_mm": float(new["original_snap_tre_mean_mm"]),
                "foreground_sdlogj": float(new["fg_logj_std"]),
                "full_crop_one_minus_ncc": check["full_crop_one_minus_ncc"],
                "point_count": check["point_count"],
                "epoch_max": check["epoch_max"],
                "curve_min": check["curve_min"],
                "curve_max": check["curve_max"],
                "curve_sha256": check["curve_sha256"],
                "summary_sha256": check["summary_sha256"],
                "source_path": updates["source_path"],
                "summary_path": updates["summary_path"],
            }
        )

    pd.testing.assert_frame_equal(v003_index.loc[~target, v003_index.columns], v004.loc[~target, v003_index.columns])
    return v004, audit, {
        "v003_rows": int(len(v003_index)),
        "v004_rows": int(len(v004)),
        "replacements": int(target.sum()),
        "unchanged_rows": int((~target).sum()),
    }


def validate_v004_index(index: pd.DataFrame) -> None:
    v003.validate_v003_index(index)
    replacements = index[index["source_kind"].eq("dirlab_regularity_matched_panelA_alpha_landscape_reuse")]
    if len(replacements) != 20:
        raise RuntimeError(f"Expected 20 v004 replacements, found {len(replacements)}")
    if not (replacements["dataset"].eq("dirlab") & replacements["panel"].eq("A")).all():
        raise RuntimeError("v004 replacements must be DIR-LAB Panel A rows")
    expected = {
        "inr_bscp": ("sinr_cps4_global_ncc_fd_be", 20000.0, 0.0001),
        "d_bscp": ("direct_cp_cps4_global_ncc_fd_be", 50000.0, 0.001),
    }
    for method_slug, (implementation_id, alpha, lr) in expected.items():
        rows = replacements[replacements["method_slug"].eq(method_slug)]
        if len(rows) != 10:
            raise RuntimeError(f"Expected ten {method_slug} replacements, found {len(rows)}")
        checks = [
            rows["implementation_id"].eq(implementation_id).all(),
            np.allclose(rows["alpha"].astype(float), alpha),
            np.allclose(rows["lr"].astype(float), lr),
            rows["cps"].astype(str).eq("4").all(),
            rows["epochs"].astype(int).eq(2500).all(),
            rows["seed"].astype(int).eq(1).all(),
            rows["diagnostic_interval"].astype(int).eq(10).all(),
            rows["endpoint_name"].eq("lung_mask_one_minus_ncc").all(),
            rows["source_path"].astype(str).str.startswith(ALPHA_TOKEN + "/").all(),
        ]
        if not all(bool(check) for check in checks):
            raise RuntimeError(f"{method_slug}: v004 replacement metadata failed validation")
    mr = index[index["dataset"].eq("dirlab") & index["panel"].eq("A") & index["method_slug"].eq("mr_d_bscp")]
    if len(mr) != 10:
        raise RuntimeError("Expected ten DIR-LAB MR-D-BSCP Panel A rows")
    if not mr["stage_epochs"].astype(str).eq("400,400,800,900").all():
        raise RuntimeError("DIR-LAB MR-D-BSCP schedule is not 400/400/800/900")


def preserve_or_write_source_index(index_csv: Path, index_json: Path, index: pd.DataFrame) -> None:
    """Write the v004 source index only for first creation; preserve it on style-only rerenders."""
    if index_csv.exists():
        existing = pd.read_csv(index_csv)
        pd.testing.assert_frame_equal(
            existing,
            index,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    else:
        index.to_csv(index_csv, index=False)

    if index_json.exists():
        existing_json = json.loads(index_json.read_text())
        if len(existing_json) != len(index):
            raise RuntimeError(
                f"Existing v004 source-index JSON has {len(existing_json)} rows; expected {len(index)}"
            )
    else:
        index_json.write_text(index.to_json(orient="records", indent=2) + "\n")


def load_and_validate_curves(index: pd.DataFrame, roots: dict[str, Path]) -> tuple[list[DiagnosticCurve], list[dict[str, Any]]]:
    curves, rows = v003.load_and_validate_curves(index, roots)
    panel_a = [curve for curve in curves if curve.panel == "A"]
    below = [
        {
            "dataset": curve.dataset,
            "case_key": curve.case_key,
            "method_slug": curve.method_slug,
            "minimum": float(np.min(curve.values)),
        }
        for curve in panel_a
        if float(np.min(curve.values)) < V004_AXIS_LIMITS["A"][0]
    ]
    if below:
        raise RuntimeError(f"Panel A curve values fall below v004 axis floor {V004_AXIS_LIMITS['A'][0]}: {below[:5]}")
    return curves, rows


def load_tre_labels(index: pd.DataFrame, roots: dict[str, Path]) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    panel_a = index[index["dataset"].eq("dirlab") & index["panel"].eq("A")]
    for _, row in panel_a.iterrows():
        case_key = str(row["case_key"])
        method = str(row["method_slug"])
        try:
            summary_path = resolve_path(row["summary_path"], roots)
            summary = base.load_json(summary_path)
            tre = base.finite(summary.get("tre_snap_to_voxel", {}).get("tre_mean_mm"))
            if tre is None:
                raise RuntimeError("missing tre_snap_to_voxel.tre_mean_mm")
            labels.setdefault(case_key, {})[method] = tre
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case_key}/{method}: {exc}")
    for case_id in range(1, 11):
        case_key = f"case{case_id:02d}"
        if set(labels.get(case_key, {})) != set(METHOD_ORDER):
            failures.append(f"{case_key}: missing TRE labels {labels.get(case_key, {})}")
    if failures:
        raise RuntimeError("TRE label validation failed:\n" + "\n".join(failures))
    return labels


def legend_handles(linewidth: float = 2.4) -> list[Line2D]:
    return [
        Line2D([0], [0], color=METHOD_STYLE[method]["color"], linewidth=linewidth, label=METHOD_STYLE[method]["label"])
        for method in METHOD_ORDER
    ]


def apply_axis_style(axis: plt.Axes, dataset: str, panel: str, show_xlabel: bool = True) -> None:
    base.style_axis(axis, dataset, panel, show_xlabel=show_xlabel)
    axis.set_ylim(*V004_AXIS_LIMITS[panel])


def tre_label_anchors(case_curves: Sequence[DiagnosticCurve], case_key: str) -> dict[str, tuple[float, float]]:
    endpoints = {
        method: float(base.method_curve(case_curves, "A", method).values[-1])
        for method in METHOD_ORDER
    }
    y_inr_bscp = min(0.78, max(0.18, endpoints["inr_bscp"] * 2.35))
    y_d_bscp = min(y_inr_bscp / 1.55, max(0.105, endpoints["d_bscp"] * 1.65))
    y_idir = max(0.015, min(0.072, endpoints["idir"] * 0.72))
    y_mr = max(0.0125, min(y_idir / 1.42, endpoints["mr_d_bscp"] * 0.68))
    if case_key == "case08":
        y_idir = max(0.022, min(0.034, y_idir * 0.72))
        y_mr = max(0.014, min(y_idir / 1.55, y_mr * 0.72))
    return {
        "inr_bscp": (1640.0, y_inr_bscp),
        "d_bscp": (1640.0, y_d_bscp),
        "idir": (1640.0, y_idir),
        "mr_d_bscp": (1640.0, y_mr),
    }


def tre_label_text(method: str, value: float, label_style: str) -> str:
    if label_style == "compact":
        compact = {
            "inr_bscp": "INR-BSCP",
            "d_bscp": "D-BSCP",
            "idir": "INR-Dense",
            "mr_d_bscp": "MR-D-BSCP",
        }
        return f"{compact[method]} {value:.2f}"
    return f"{METHOD_STYLE[method]['label']}: {value:.2f} mm"


def add_tre_inside_labels(
    axis: plt.Axes,
    case_curves: Sequence[DiagnosticCurve],
    case_key: str,
    tre_labels: dict[str, dict[str, float]],
    *,
    fontsize: float,
    label_style: str = "full",
    arrow_linewidth: float = 0.72,
) -> None:
    anchors = tre_label_anchors(case_curves, case_key)
    for method in TRE_LABEL_ORDER:
        curve = base.method_curve(case_curves, "A", method)
        color = METHOD_STYLE[method]["color"]
        x_text, y_text = anchors[method]
        axis.annotate(
            tre_label_text(method, tre_labels[case_key][method], label_style),
            xy=(float(curve.epochs[-1]), float(curve.values[-1])),
            xytext=(x_text, y_text),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": arrow_linewidth,
                "alpha": 0.70,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            zorder=20,
            clip_on=False,
        )


def render_case_figure_v004(
    case_curves: Sequence[DiagnosticCurve],
    output_path: Path,
    dpi: int,
    tre_labels: dict[str, dict[str, float]],
) -> None:
    dataset = case_curves[0].dataset
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=False)
    for axis, panel in zip(axes, ["A", "B"]):
        base.draw_curves(axis, case_curves, panel)
        show_tre = dataset == "dirlab" and panel == "A"
        apply_axis_style(axis, dataset, panel, show_xlabel=True)
        if show_tre:
            add_tre_inside_labels(axis, case_curves, case_curves[0].case_key, tre_labels, fontsize=8.0)
        axis.set_ylabel(base.PANEL_Y_LABEL[panel], fontsize=9.4)
        axis.set_title(("A. " if panel == "A" else "B. ") + base.PANEL_TITLE[panel], loc="left", fontsize=10.5, weight="bold")
    fig.suptitle(case_curves[0].case_label, fontsize=12.5, weight="bold", y=0.995)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=4, frameon=False, fontsize=8.6, handlelength=2.2, columnspacing=1.1)
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_composite_v004(
    dataset: str,
    case_group_list: Sequence[Sequence[DiagnosticCurve]],
    output_path: Path,
    dpi: int,
    tre_labels: dict[str, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(20.0, 13.0), sharex=False, sharey=False)
    for block in range(2):
        for column in range(5):
            curves = case_group_list[block * 5 + column]
            for panel_offset, panel in enumerate(["A", "B"]):
                row = block * 2 + panel_offset
                axis = axes[row, column]
                base.draw_curves(axis, curves, panel)
                show_tre = dataset == "dirlab" and panel == "A"
                apply_axis_style(axis, dataset, panel, show_xlabel=panel == "B")
                if show_tre:
                    add_tre_inside_labels(
                        axis,
                        curves,
                        curves[0].case_key,
                        tre_labels,
                        fontsize=4.7,
                        label_style="compact",
                        arrow_linewidth=0.48,
                    )
                if column == 0:
                    axis.set_ylabel(base.PANEL_Y_LABEL[panel], fontsize=9.4, weight="bold")
                if panel == "A":
                    axis.set_title(base.short_case_title(curves[0]), fontsize=10.3, weight="bold")
    title_suffix = (
        "Panel A uses fixed image>0 mask"
        if dataset == "oasis"
        else "Panel A uses fixed T00 lung mask; single-stage BSCP alphas are SDlogJ-matched"
    )
    fig.suptitle(f"{base.DATASET_LABEL[dataset]} diagnostics (v004; {title_suffix})", fontsize=17, weight="bold", y=0.992)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=11, handlelength=2.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.95, bottom=0.075, hspace=0.37, wspace=0.32)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_design_2x4_v004(curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int, tre_labels: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.8, 8.4), sharex=False, sharey="row")
    columns = [
        ("population", "oasis", "OASIS N=10 population"),
        ("population", "dirlab", "DIR-LAB/LungCT N=10 population"),
        ("case", "case01", "DIR-LAB case01"),
        ("case", "case08", "DIR-LAB case08"),
    ]
    for row, panel in enumerate(["A", "B"]):
        for column, (kind, key, title) in enumerate(columns):
            axis = axes[row, column]
            if kind == "population":
                base.draw_population_overlay(axis, curves, key, panel)
                dataset = key
            else:
                case_curves = [curve for curve in curves if curve.dataset == "dirlab" and curve.case_key == key]
                base.draw_single_case(axis, case_curves, panel)
                dataset = "dirlab"
                if panel == "A":
                    add_tre_inside_labels(axis, case_curves, key, tre_labels, fontsize=8.2)
            apply_axis_style(axis, dataset, panel)
            axis.set_title(title, fontsize=12.0, weight="bold")
            if column == 0:
                axis.set_ylabel(("A. " if panel == "A" else "B. ") + base.PANEL_Y_LABEL[panel], fontsize=11.0, weight="bold")
    fig.legend(handles=legend_handles(3.0), loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, frameon=False, fontsize=10.4)
    fig.suptitle("Population diagnostics with DIR-LAB representative cases", fontsize=15.0, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.025, 0.055, 1.0, 0.955))
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def apply_fig3_camera_ready_axis_style(axis: plt.Axes, dataset: str, panel: str) -> None:
    apply_axis_style(axis, dataset, panel)
    axis.tick_params(axis="both", labelsize=FIG3_CAMERA_READY_FONTS["tick"])
    axis.set_xlabel("Step", fontsize=FIG3_CAMERA_READY_FONTS["xlabel"])


def render_design_2x4_v004_camera_ready_fig3(
    curves: Sequence[DiagnosticCurve],
    output_path: Path,
    dpi: int,
    tre_labels: dict[str, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(21.5, 10.4), sharex=False, sharey="row")
    columns = [
        ("population", "oasis", "OASIS N=10 population"),
        ("population", "dirlab", "DIR-LAB 4DCT N=10 population"),
        ("case", "case01", "DIR-LAB 4DCT case01"),
        ("case", "case08", "DIR-LAB 4DCT case08"),
    ]
    for row, panel in enumerate(["A", "B"]):
        for column, (kind, key, title) in enumerate(columns):
            axis = axes[row, column]
            if kind == "population":
                base.draw_population_overlay(axis, curves, key, panel)
                dataset = key
            else:
                case_curves = [curve for curve in curves if curve.dataset == "dirlab" and curve.case_key == key]
                base.draw_single_case(axis, case_curves, panel)
                dataset = "dirlab"
                if panel == "A":
                    add_tre_inside_labels(axis, case_curves, key, tre_labels, fontsize=8.2)
            apply_fig3_camera_ready_axis_style(axis, dataset, panel)
            axis.set_title(title, fontsize=FIG3_CAMERA_READY_FONTS["subtitle"], weight="bold")
            axis.set_ylabel(
                "Similarity (1-NCC)" if panel == "A" else "rRMSE (%)",
                fontsize=FIG3_CAMERA_READY_FONTS["ylabel"],
            )
    fig.legend(
        handles=legend_handles(3.0),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=False,
        fontsize=FIG3_CAMERA_READY_FONTS["legend"],
    )
    fig.text(
        0.075,
        0.968,
        "A. Similarity metric during unsupervised registration optimization",
        ha="left",
        va="top",
        fontsize=FIG3_CAMERA_READY_FONTS["panel_title"],
        weight="bold",
    )
    fig.text(
        0.075,
        0.505,
        "B. Target field fitting error during supervised fitting",
        ha="left",
        va="top",
        fontsize=FIG3_CAMERA_READY_FONTS["panel_title"],
        weight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.895, bottom=0.145, hspace=0.78, wspace=0.30)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_reports(
    output_root: Path,
    counts: dict[str, int],
    audit: Sequence[dict[str, Any]],
    curve_summary: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]],
    tre_labels: dict[str, dict[str, float]],
) -> None:
    inr = [row for row in audit if row["method_slug"] == "inr_bscp"]
    direct = [row for row in audit if row["method_slug"] == "d_bscp"]
    label_tre_mean = {
        method: float(np.mean([tre_labels[f"case{case_id:02d}"][method] for case_id in range(1, 11)]))
        for method in ["inr_bscp", "d_bscp"]
    }
    payload = {
        "created_at_utc": utc_text(),
        "status": "passed",
        "counts": counts,
        "axis_limits": V004_AXIS_LIMITS,
        "replacement_summary": {
            "inr_bscp_alpha": 20000.0,
            "d_bscp_alpha": 50000.0,
            "inr_bscp_mean_pareto_tre_mm": float(np.mean([row["tre_mean_mm"] for row in inr])),
            "d_bscp_mean_pareto_tre_mm": float(np.mean([row["tre_mean_mm"] for row in direct])),
            "inr_bscp_mean_label_tre_snap_to_voxel_mm": label_tre_mean["inr_bscp"],
            "d_bscp_mean_label_tre_snap_to_voxel_mm": label_tre_mean["d_bscp"],
            "inr_bscp_mean_sdlogj": float(np.mean([row["foreground_sdlogj"] for row in inr])),
            "d_bscp_mean_sdlogj": float(np.mean([row["foreground_sdlogj"] for row in direct])),
        },
        "tre_label_metric": "tre_snap_to_voxel.tre_mean_mm",
        "tre_label_count": sum(len(values) for values in tre_labels.values()),
    }
    (output_root / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_root / "validation_report.md").write_text(
        "\n".join(
            [
                "# v004 DIR-LAB Regularity-Matched Panel A Validation Report",
                "",
                "- Status: `passed`",
                f"- Source index rows: `{counts['v004_rows']}`",
                f"- DIR-LAB Panel A regularity-matched replacements: `{counts['replacements']}/20`",
                f"- Unchanged v003 rows: `{counts['unchanged_rows']}`",
                "- Replacements are exact-compatible reuse from the accepted FD-BE cps4 alpha-landscape run.",
                "- INR-BSCP cps4 uses `a=20000`; D-BSCP cps4 uses `a=50000`.",
                "- DIR-LAB MR-D-BSCP remains `400/400/800/900`, cps `32/16/8/4`, `a=10000`.",
                "- OASIS remains unchanged from v003 except for the shared Panel A axis scale.",
                "- Panel A axis: `1e-2–1e0`; Panel B axis: `1e-1–1e3`.",
                f"- DIR-LAB Panel A TRE labels: `{payload['tre_label_count']}/40`.",
                "- TRE labels use inside-plot leader lines; INR-BSCP and D-BSCP labels are placed above the curves, while INR-Dense and MR-D-BSCP labels are placed below.",
                f"- Figure 3 camera-ready 2x4 variant: `{FIG3_CAMERA_READY_OUTPUT}`; uses no super-title, `DIR-LAB 4DCT` subtitles, `Step` x-axis labels, and enlarged tick/axis/legend typography.",
                "- TRE labels use `tre_snap_to_voxel.tre_mean_mm` from each run summary; the regularity-selection comparison table reports the Pareto source TRE used with foreground SDlogJ.",
            ]
        )
        + "\n"
    )
    (output_root / "diagnostic_comparison.md").write_text(
        "\n".join(
            [
                "# v004 Diagnostic Comparison",
                "",
                "v004 starts from v003 and changes only DIR-LAB Panel A single-stage BSCP operating points. "
                "The change makes the DIR-LAB registration diagnostics regularity-matched by foreground `SD(log J)` rather than using the same nominal FD-BE weight for both single-stage methods.",
                "",
                "| Method | v003 alpha | v004 alpha | Mean Pareto TRE (mm) | Mean label TRE (mm) | Mean foreground SDlogJ |",
                "|---|---:|---:|---:|---:|---:|",
                f"| INR-BSCP cps4 | 10000 | 20000 | {np.mean([row['tre_mean_mm'] for row in inr]):.3f} | {label_tre_mean['inr_bscp']:.3f} | {np.mean([row['foreground_sdlogj'] for row in inr]):.3f} |",
                f"| D-BSCP cps4 | 10000 | 50000 | {np.mean([row['tre_mean_mm'] for row in direct]):.3f} | {label_tre_mean['d_bscp']:.3f} | {np.mean([row['foreground_sdlogj'] for row in direct]):.3f} |",
                "",
                "`Mean label TRE` is computed from `tre_snap_to_voxel.tre_mean_mm`, the same metric printed on the DIR-LAB Panel A figures. `Mean Pareto TRE` is retained from the alpha-landscape source rows because that is the value paired with foreground `SD(log J)` in the Pareto selection table.",
                "",
                "All other records are unchanged from v003. v003 remains the fixed-domain but not DIR-LAB-regularity-matched comparison artifact.",
            ]
        )
        + "\n"
    )
    (output_root / "metric_domain_summary.md").write_text(
        "\n".join(
            [
                "# v004 Camera-Ready Diagnostic Metric Domain Summary",
                "",
                "v004 is the camera-ready Figure 3 candidate. It preserves the v003 fixed-domain metric convention and additionally regularity-matches the DIR-LAB Panel A single-stage BSCP operating points.",
                "",
                "| Dataset | Panel | Plotted metric | Evaluation domain | v004 change |",
                "|---|---|---|---|---|",
                "| OASIS | A: unsupervised registration | `1−NCC` | Fixed raw `image > 0` mask | unchanged from v003 |",
                "| OASIS | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed raw `image > 0` mask | unchanged from v003 |",
                "| DIR-LAB/LungCT | A: unsupervised registration | `1−NCC` | Fixed T00 lung mask on canonical 1 mm crop | INR-BSCP `a=20000`; D-BSCP `a=50000` |",
                "| DIR-LAB/LungCT | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed T00 lung mask on canonical 1 mm crop | unchanged from v003 |",
                "",
                "Panel A uses global NCC: `ncc_loss = -corr(fixed, warped)`, plotted as `1 + ncc_loss`. The plotted DIR-LAB Panel A metric is evaluated inside the fixed T00 lung mask for all four methods.",
                "",
                "For DIR-LAB Panel A, the single-stage BSCP alphas are selected from the Pareto sweep to approximately match foreground `SD(log J)` to IDIR and MR-D-BSCP, rather than keeping identical nominal alpha values across parameterizations.",
                "",
                "Safe claim: v004 compares DIR-LAB Panel A methods at approximately matched deformation regularity and a common fixed-lung evaluation domain. Caveat: the optimization/training domains are still not identical across all methods.",
            ]
        )
        + "\n"
    )
    write_csv(output_root / "dirlab_regularity_matched_panelA_replacement_audit.csv", audit)


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    base.AXIS_LIMITS["A"] = V004_AXIS_LIMITS["A"]
    base.AXIS_LIMITS["B"] = V004_AXIS_LIMITS["B"]
    v003.AXIS_LIMITS["A"] = V004_AXIS_LIMITS["A"]
    v003.AXIS_LIMITS["B"] = V004_AXIS_LIMITS["B"]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = output_root / "case_figures"
    case_dir.mkdir(parents=True, exist_ok=True)

    v003_csv = args.v003_root / "camera_ready_source_index_masked_panelA_both_datasets.csv"
    v003_index = pd.read_csv(v003_csv)
    alpha_rows = pd.read_csv(args.alpha_case_rows)
    v004_index, audit, counts = build_v004_source_index(v003_index, alpha_rows, args.alpha_run_root.resolve())
    validate_v004_index(v004_index)

    index_csv = output_root / "camera_ready_source_index_dirlab_regularity_matched_panelA.csv"
    index_json = output_root / "camera_ready_source_index_dirlab_regularity_matched_panelA.json"
    preserve_or_write_source_index(index_csv, index_json, v004_index)

    roots = {
        base.POPULATION_TOKEN: args.population_run_root.resolve(),
        base.REFRESH_TOKEN: args.refresh_run_root.resolve(),
        base.LUNGMASK_TOKEN: args.lungmask_run_root.resolve(),
        v003.OASIS_MASKED_TOKEN: args.oasis_masked_run_root.resolve(),
        base.DATA_TOKEN: args.data_root.resolve(),
        ALPHA_TOKEN: args.alpha_run_root.resolve(),
    }
    curves, curve_summary = load_and_validate_curves(v004_index, roots)
    tre_labels = load_tre_labels(v004_index, roots)
    write_csv(output_root / "curve_summary.csv", curve_summary)

    images: list[dict[str, Any]] = []
    for dataset in ["oasis", "dirlab"]:
        groups = base.case_groups(curves, dataset)
        composite = output_root / f"{dataset}_population_diagnostics_4x5.png"
        render_composite_v004(dataset, groups, composite, args.dpi, tre_labels)
        images.append(base.image_record(composite, output_root, f"{dataset}_4x5"))
        for group in groups:
            case_key = group[0].case_key
            path = case_dir / f"{dataset}_{case_key}_diagnostics.png"
            render_case_figure_v004(group, path, args.dpi, tre_labels)
            images.append(base.image_record(path, output_root, "case_figure"))

    overlay_summary = base.render_overlay_2x2(curves, output_root / "population_overlay_mean_std_2x2.png", args.dpi)
    write_csv(output_root / "population_overlay_mean_std_summary.csv", overlay_summary)
    images.append(base.image_record(output_root / "population_overlay_mean_std_2x2.png", output_root, "overlay_2x2"))
    base.render_design_2x4(curves, output_root / "population_overlay_plus_lungct_cases_2x4.png", args.dpi)
    images.append(base.image_record(output_root / "population_overlay_plus_lungct_cases_2x4.png", output_root, "design_2x4"))
    base.render_design_2x3(curves, output_root / "population_overlay_plus_lungct_cases_2x3.png", args.dpi)
    images.append(base.image_record(output_root / "population_overlay_plus_lungct_cases_2x3.png", output_root, "design_2x3"))
    render_design_2x4_v004(curves, output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", args.dpi, tre_labels)
    images.append(base.image_record(output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", output_root, "design_2x4_tre"))
    render_design_2x4_v004_camera_ready_fig3(curves, output_root / FIG3_CAMERA_READY_OUTPUT, args.dpi, tre_labels)
    images.append(base.image_record(output_root / FIG3_CAMERA_READY_OUTPUT, output_root, "design_2x4_tre_camera_ready_fig3"))

    write_reports(output_root, counts, audit, curve_summary, images, tre_labels)
    manifest = {
        "created_at_utc": utc_text(),
        "version": "v004_dirlab_regularity_matched_panelA",
        "v003_source_index": "v003_masked_panelA_both_datasets/camera_ready_source_index_masked_panelA_both_datasets.csv",
        "alpha_case_rows": f"external --alpha-case-rows/{args.alpha_case_rows.name}",
        "run_root_tokens": sorted(roots),
        "axis_limits": V004_AXIS_LIMITS,
        "stage_boundaries": {f"{dataset}_{panel}": values for (dataset, panel), values in base.STAGE_BOUNDARIES.items()},
        "counts": counts,
        "replacement_policy": "exact-compatible reuse; rerun only failed or incompatible records",
        "tre_label_metric": "tre_snap_to_voxel.tre_mean_mm",
        "tre_label_layout": {
            "dirlab_case_figures": "inside-plot leader-line labels; INR-BSCP/D-BSCP above curves and INR-Dense/MR-D-BSCP below curves",
            "dirlab_4x5": "compact inside-plot leader-line labels with the same high-TRE-above, low-TRE-below ordering",
            "representative_2x4": "inside-plot leader-line labels following the v003 style without right-side gutter expansion",
            "camera_ready_fig3_2x4": "inside-plot leader-line labels; case08 INR-Dense/MR-D-BSCP anchors are moved downward to avoid overlap with the MR-D-BSCP curve",
        },
        "camera_ready_fig3_variant": {
            "filename": FIG3_CAMERA_READY_OUTPUT,
            "super_title": "removed",
            "dirlab_label": "DIR-LAB 4DCT",
            "x_axis_label": "Step",
            "fonts": FIG3_CAMERA_READY_FONTS,
        },
        "tre_labels_mm": tre_labels,
        "images": images,
        "checksums": {
            "source_index_csv": base.sha256(index_csv),
            "source_index_json": base.sha256(index_json),
            "curve_summary_csv": base.sha256(output_root / "curve_summary.csv"),
            "validation_report_json": base.sha256(output_root / "validation_report.json"),
        },
    }
    (output_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output_root": str(output_root), "images": len(images)}, indent=2))


if __name__ == "__main__":
    main()
