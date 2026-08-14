#!/usr/bin/env python
"""Render v002 population diagnostics with DIR-LAB Panel A lung-mask NCC.

This renderer is intentionally versioned and non-destructive.  It reads the
v001 schedule-matched source index, replaces only DIR-LAB Panel A trajectories
with paired-diagnostic runs whose plotted metric is fixed-T00-lung-mask
``1-NCC``, and writes a new artifact family under
``v002_lungmask_panelA``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from plot_case_diagnostics import (
    DiagnosticCurve,
    METHOD_ORDER,
    METHOD_STYLE,
    load_curve,
    normalize_terminal_endpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V001_ROOT = (
    REPO_ROOT / "experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/paper_artifacts/fit_modes_population_diagnostics/v002_lungmask_panelA"
)
DEFAULT_POPULATION_RUN_ROOT = Path("/database/inrorspline/runs/fit_modes_population_n10_v1")
DEFAULT_REFRESH_RUN_ROOT = Path(
    "/database/inrorspline/runs/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
)
DEFAULT_LUNGMASK_RUN_ROOT = Path(
    "/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1"
)
DEFAULT_DATA_ROOT = Path("/database/inrorspline")

POPULATION_TOKEN = "${POPULATION_RUN_ROOT}"
REFRESH_TOKEN = "${DIRLAB_REFRESH_RUN_ROOT}"
LUNGMASK_TOKEN = "${DIRLAB_LUNGMASK_RUN_ROOT}"
DATA_TOKEN = "${DATA_ROOT}"
AGGREGATE_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=float)
AXIS_LIMITS = {"A": (1.0e-3, 5.0e-1), "B": (1.0e-1, 1.0e3)}
STAGE_BOUNDARIES = {
    ("oasis", "A"): [400, 1200],
    ("oasis", "B"): [400, 1200],
    ("dirlab", "A"): [400, 800, 1600],
    ("dirlab", "B"): [400, 800, 1600],
}
DATASET_LABEL = {"oasis": "OASIS N=10", "dirlab": "DIR-LAB/LungCT N=10"}
PANEL_TITLE = {"A": "Registration: 1−NCC", "B": "Supervised fitting: rRMSE (%)"}
PANEL_Y_LABEL = {"A": "1−NCC", "B": "rRMSE (%)"}

METHODS = {
    "idir": {
        "impl": "idir_be_default_lungmask",
        "run_name": "lungmask_curves_case{case_id:02d}_panelA_idir_e2500_lr1em4_seed1_diag10",
    },
    "inr_bscp": {
        "impl": "sinr_cps4_global_ncc_fd_be",
        "run_name": "lungmask_curves_case{case_id:02d}_panelA_inr_bscp_e2500_lr1em4_seed1_diag10",
    },
    "d_bscp": {
        "impl": "direct_cp_cps4_global_ncc_fd_be",
        "run_name": "lungmask_curves_case{case_id:02d}_panelA_d_bscp_e2500_lr1em3_seed1_diag10",
    },
    "mr_d_bscp": {
        "impl": "direct_cp_32_16_8_4_global_ncc_fd_be",
        "run_name": (
            "lungmask_curves_case{case_id:02d}_panelA_mr_d_bscp_"
            "sched32-16-8-4_ds8-4-2-1_e400-400-800-900_a10000_lr0p001_seed1_diag10"
        ),
    },
}

LUNGCT_CASE_STYLES = {
    "case01": {"label": "DIR-LAB case01", "linestyle": "solid"},
    "case08": {"label": "DIR-LAB case08", "linestyle": (0, (5, 2.4))},
}
TRE_LABEL_POSITIONS = {
    "case01": {
        "idir": (1760, 0.070),
        "inr_bscp": (1760, 0.0105),
        "d_bscp": (1760, 0.0060),
        "mr_d_bscp": (1760, 0.0035),
    },
    "case08": {
        "idir": (1760, 0.135),
        "d_bscp": (1760, 0.050),
        "inr_bscp": (1760, 0.032),
        "mr_d_bscp": (1760, 0.0042),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v001-root", type=Path, default=DEFAULT_V001_ROOT)
    parser.add_argument("--population-run-root", type=Path, default=DEFAULT_POPULATION_RUN_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument("--lungmask-run-root", type=Path, default=DEFAULT_LUNGMASK_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable_from_root(path: Path, root: Path, token: str) -> str:
    text = str(path)
    root_text = str(root)
    if text == root_text or text.startswith(root_text + "/"):
        return text.replace(root_text, token, 1)
    raise ValueError(f"Path is outside {token}: {path}")


def resolve_portable(value: Any, roots: dict[str, Path]) -> Path:
    text = str(value or "")
    for token, root in roots.items():
        text = text.replace(token, str(root))
    return Path(text)


def case_label(row: pd.Series) -> str:
    if row["dataset"] == "oasis":
        return f"OASIS r{int(row['case_rank'])}: M{int(row['moving_id']):04d}→F{int(row['fixed_id']):04d}"
    return f"DIR-LAB case{int(row['case_id']):02d}: T50→T00"


def lungmask_paths(case_id: int, method_slug: str, lungmask_root: Path) -> tuple[Path, Path]:
    spec = METHODS[method_slug]
    run_name = spec["run_name"].format(case_id=case_id)
    run_dir = lungmask_root / spec["impl"] / run_name
    return run_dir / "global_ncc_diagnostic.csv", run_dir / "summary.json"


def validate_lungmask_curve(curve_path: Path, summary_path: Path, case_id: int, method_slug: str) -> dict[str, Any]:
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    curve = pd.read_csv(curve_path)
    summary = load_json(summary_path)
    required = ["global_epoch", "global_ncc_value", "lung_mask_one_minus_ncc", "lung_mask_ncc_value"]
    missing = [column for column in required if column not in curve.columns]
    if missing:
        raise RuntimeError(f"{curve_path}: missing columns {missing}")
    epochs = curve["global_epoch"].astype(int).tolist()
    if epochs != AGGREGATE_EPOCHS.astype(int).tolist():
        raise RuntimeError(f"{curve_path}: expected 251 diagnostics ending at 2500")
    values = curve[["global_ncc_value", "lung_mask_one_minus_ncc", "lung_mask_ncc_value"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{curve_path}: nonfinite paired diagnostic")
    if not (curve["lung_mask_one_minus_ncc"].to_numpy(float) > 0.0).all():
        raise RuntimeError(f"{curve_path}: nonpositive lung-mask 1-NCC")
    if summary.get("status") != "completed" or int(summary.get("case_id", -1)) != case_id:
        raise RuntimeError(f"{summary_path}: summary status/case mismatch")
    training = summary.get("training", {})
    if int(training.get("global_ncc_diagnostic_interval", -1)) != 10:
        raise RuntimeError(f"{summary_path}: diagnostic interval is not 10")
    if training.get("lung_mask_ncc_diagnostic_name") != "fixed_t00_lung_mask_global_ncc":
        raise RuntimeError(f"{summary_path}: missing lung-mask diagnostic metadata")
    if method_slug == "mr_d_bscp":
        expected = {
            "stage_epochs": [400, 400, 800, 900],
            "stage_full_cps": [32, 16, 8, 4],
            "stage_downsample": [8, 4, 2, 1],
        }
        for field, wanted in expected.items():
            if training.get(field) != wanted:
                raise RuntimeError(f"{summary_path}: {field} mismatch: {training.get(field)}")
    final = curve.iloc[-1]
    full_crop_endpoint = 1.0 - float(final["global_ncc_value"])
    lung_endpoint = float(final["lung_mask_one_minus_ncc"])
    return {
        "case_id": case_id,
        "method_slug": method_slug,
        "points": int(len(curve)),
        "epoch_max": int(final["global_epoch"]),
        "lung_mask_voxels": int(final.get("lung_mask_voxels", summary.get("fixed_lung_mask_voxels", -1))),
        "full_crop_one_minus_ncc": full_crop_endpoint,
        "lung_mask_one_minus_ncc": lung_endpoint,
        "runtime_sec": finite(summary.get("elapsed_sec")),
        "peak_memory_gb": finite(summary.get("peak_mem_gb")),
        "tre_mean_mm": finite(summary.get("tre_snap_to_voxel", {}).get("tre_mean_mm")),
        "curve_sha256": sha256(curve_path),
        "summary_sha256": sha256(summary_path),
    }


def build_v002_source_index(
    v001: pd.DataFrame, lungmask_root: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if len(v001) != 160 or v001["job_id"].duplicated().any():
        raise RuntimeError("v001 source index must contain 160 unique records")
    target = v001["dataset"].eq("dirlab") & v001["panel"].eq("A")
    if int(target.sum()) != 40:
        raise RuntimeError(f"Expected 40 DIR-LAB Panel A rows, found {int(target.sum())}")
    v002 = v001.copy()
    audit: list[dict[str, Any]] = []
    for row_index, row in v001[target].iterrows():
        case_id = int(row["case_id"])
        method_slug = str(row["method_slug"])
        curve_path, summary_path = lungmask_paths(case_id, method_slug, lungmask_root)
        check = validate_lungmask_curve(curve_path, summary_path, case_id, method_slug)
        old_endpoint = float(row["endpoint_value"])
        full_diff = check["full_crop_one_minus_ncc"] - old_endpoint
        updates = {
            "status": "completed" if case_id not in (1, 8) else "completed_existing",
            "source_kind": "lungmask_panelA_dual_diagnostic",
            "source_path": portable_from_root(curve_path, lungmask_root, LUNGMASK_TOKEN),
            "summary_path": portable_from_root(summary_path, lungmask_root, LUNGMASK_TOKEN),
            "endpoint_name": "lung_mask_one_minus_ncc",
            "endpoint_value": check["lung_mask_one_minus_ncc"],
            "diagnostic_ncc": 1.0 - check["lung_mask_one_minus_ncc"],
            "runtime_sec": check["runtime_sec"],
            "peak_memory_gb": check["peak_memory_gb"],
            "endpoint_source": "lungmask_panelA:lung_mask_one_minus_ncc",
            "diagnostic_interval": 10,
            "acceptance_checks": (
                "lungmask_panelA=true;dual_diagnostic=true;curve_rows=251;"
                "endpoint_source=lung_mask_one_minus_ncc"
            ),
        }
        if method_slug == "mr_d_bscp":
            updates.update(
                {
                    "cps": "32,16,8,4",
                    "stage_downsample": "8,4,2,1",
                    "stage_epochs": "400,400,800,900",
                    "epochs": 2500,
                    "lr": 0.001,
                    "alpha": 10000.0,
                    "seed": 1,
                }
            )
        for field, value in updates.items():
            v002.at[row_index, field] = value
        audit.append(
            {
                **check,
                "job_id": row["job_id"],
                "case_key": row["case_key"],
                "display_method": row["display_method"],
                "old_v001_endpoint_full_crop": old_endpoint,
                "new_v002_endpoint_lung_mask": check["lung_mask_one_minus_ncc"],
                "full_crop_endpoint_repro_diff_vs_v001": full_diff,
                "source_path": updates["source_path"],
                "summary_path": updates["summary_path"],
                "reused_existing_case01_case08": case_id in (1, 8),
            }
        )
    untouched_cols = [column for column in v001.columns if column in v002.columns]
    pd.testing.assert_frame_equal(v001.loc[~target, untouched_cols], v002.loc[~target, untouched_cols])
    return v002, audit, {
        "v001_rows": int(len(v001)),
        "v002_rows": int(len(v002)),
        "replacements": int(target.sum()),
        "unchanged_rows": int((~target).sum()),
    }


def load_v002_curve(row: pd.Series, roots: dict[str, Path]) -> DiagnosticCurve:
    source = resolve_portable(row["source_path"], roots)
    if not source.is_file():
        raise FileNotFoundError(source)
    if row["dataset"] == "dirlab" and row["panel"] == "A":
        frame = pd.read_csv(source)
        epochs = frame["global_epoch"].to_numpy(float)
        values = frame["lung_mask_one_minus_ncc"].to_numpy(float)
        keep = np.isfinite(epochs) & np.isfinite(values) & (values > 0.0)
        epochs = epochs[keep]
        values = values[keep]
        order = np.argsort(epochs, kind="stable")
        accepted_endpoint = finite(row["endpoint_value"])
        if accepted_endpoint is None:
            raise RuntimeError(f"{row['job_id']}: missing endpoint")
        return DiagnosticCurve(
            dataset=str(row["dataset"]),
            case_key=str(row["case_key"]),
            case_order=int(row["case_id"]),
            case_label=case_label(row),
            panel="A",
            method_slug=str(row["method_slug"]),
            method_label=str(row["display_method"]),
            epochs=epochs[order],
            values=values[order],
            source_path=source,
            source_metric="lung_mask_one_minus_ncc",
            target_power=None,
            accepted_endpoint=accepted_endpoint,
        )
    adjusted = row.copy()
    source_text = str(row["source_path"])
    if source_text.startswith(POPULATION_TOKEN):
        adjusted["source_path"] = source_text.replace(POPULATION_TOKEN, "${RUN_ROOT}", 1)
        return load_curve(adjusted, roots[POPULATION_TOKEN], roots[DATA_TOKEN])
    if source_text.startswith(REFRESH_TOKEN):
        adjusted["source_path"] = source_text.replace(REFRESH_TOKEN, "${RUN_ROOT}", 1)
        return load_curve(adjusted, roots[REFRESH_TOKEN], roots[DATA_TOKEN])
    if source_text.startswith(DATA_TOKEN):
        return load_curve(adjusted, roots[POPULATION_TOKEN], roots[DATA_TOKEN])
    raise ValueError(f"Unrecognized source path: {source_text}")


def validate_v002_index(index: pd.DataFrame) -> None:
    if len(index) != 160:
        raise RuntimeError(f"Expected 160 records, found {len(index)}")
    if index.duplicated(["dataset", "case_key", "panel", "method_slug"]).any():
        raise RuntimeError("Duplicate dataset/case/panel/method records")
    counts = index.groupby(["dataset", "panel", "method_slug"])["case_key"].nunique()
    expected = pd.MultiIndex.from_product(
        [["oasis", "dirlab"], ["A", "B"], METHOD_ORDER],
        names=["dataset", "panel", "method_slug"],
    )
    bad = counts.reindex(expected)
    if not bad.eq(10).all():
        raise RuntimeError(f"Expected 10 cases per dataset/panel/method:\n{bad}")


def load_and_validate_curves(index: pd.DataFrame, roots: dict[str, Path]) -> tuple[list[DiagnosticCurve], list[dict[str, Any]]]:
    curves: list[DiagnosticCurve] = []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in index.iterrows():
        try:
            raw = load_v002_curve(row, roots)
            curve, raw_endpoint, raw_difference, endpoint_replaced = normalize_terminal_endpoint(raw)
            plotted = float(curve.values[-1])
            difference = plotted - float(curve.accepted_endpoint)
            if not np.isclose(plotted, curve.accepted_endpoint, rtol=2.0e-5, atol=2.0e-8):
                raise RuntimeError(f"endpoint mismatch: plotted={plotted}, accepted={curve.accepted_endpoint}")
            if not np.isfinite(curve.values).all() or not (curve.values > 0.0).all():
                raise RuntimeError("nonfinite or nonpositive plotted curve")
            curves.append(curve)
            rows.append(
                {
                    "job_id": row["job_id"],
                    "dataset": curve.dataset,
                    "case_key": curve.case_key,
                    "case_order": curve.case_order,
                    "panel": curve.panel,
                    "method_slug": curve.method_slug,
                    "source_metric": curve.source_metric,
                    "points": int(len(curve.epochs)),
                    "epoch_min": float(np.min(curve.epochs)),
                    "epoch_max": float(np.max(curve.epochs)),
                    "value_min": float(np.min(curve.values)),
                    "value_max": float(np.max(curve.values)),
                    "accepted_endpoint": float(curve.accepted_endpoint),
                    "plotted_endpoint": plotted,
                    "endpoint_difference": difference,
                    "raw_terminal_value": raw_endpoint,
                    "raw_terminal_difference": raw_difference,
                    "terminal_endpoint_replaced": endpoint_replaced,
                    "source_path": row["source_path"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row.get('job_id', '<unknown>')}: {exc}")
    if failures:
        raise RuntimeError("Curve validation failed:\n" + "\n".join(failures))
    lungmask = [
        curve for curve in curves if curve.dataset == "dirlab" and curve.panel == "A"
    ]
    if len(lungmask) != 40:
        raise RuntimeError(f"Expected 40 DIR-LAB Panel A lung-mask curves, found {len(lungmask)}")
    for curve in lungmask:
        if len(curve.epochs) != 251 or float(curve.epochs[-1]) != 2500.0:
            raise RuntimeError(f"{curve.case_key}/{curve.method_slug}: invalid lung-mask diagnostic length")
    return curves, rows


def method_curve(curves: Sequence[DiagnosticCurve], panel: str, method: str) -> DiagnosticCurve:
    matches = [curve for curve in curves if curve.panel == panel and curve.method_slug == method]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {panel}/{method} curve, found {len(matches)}")
    return matches[0]


def legend_handles(linewidth: float = 2.4) -> list[Line2D]:
    return [
        Line2D([0], [0], color=METHOD_STYLE[method]["color"], linewidth=linewidth, label=METHOD_STYLE[method]["label"])
        for method in METHOD_ORDER
    ]


def style_axis(axis: plt.Axes, dataset: str, panel: str, show_xlabel: bool = True) -> None:
    for boundary in STAGE_BOUNDARIES[(dataset, panel)]:
        axis.axvline(boundary, color="#777777", linewidth=0.72, linestyle=(0, (2, 2)), alpha=0.55, zorder=1)
    axis.set_yscale("log")
    axis.set_ylim(*AXIS_LIMITS[panel])
    axis.set_xlim(0, 2500)
    axis.set_xticks(np.arange(0, 2501, 500))
    axis.grid(True, which="both", color="#D5DCE3", linewidth=0.62, alpha=0.75)
    axis.tick_params(axis="both", labelsize=8.1, labelbottom=True, labelleft=True)
    axis.set_xlabel("Epoch" if show_xlabel else "", fontsize=9.0)


def draw_curves(axis: plt.Axes, curves: Sequence[DiagnosticCurve], panel: str, linewidth: float = 1.65) -> None:
    for method in METHOD_ORDER:
        curve = method_curve(curves, panel, method)
        style = METHOD_STYLE[method]
        axis.plot(curve.epochs, curve.values, color=style["color"], linewidth=linewidth, zorder=style["zorder"], label=style["label"])


def short_case_title(curve: DiagnosticCurve) -> str:
    if curve.dataset == "oasis":
        return curve.case_label.replace("OASIS ", "")
    return curve.case_label.replace("DIR-LAB ", "")


def render_case_figure(curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int) -> None:
    dataset = curves[0].dataset
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=False)
    for axis, panel in zip(axes, ["A", "B"]):
        draw_curves(axis, curves, panel)
        style_axis(axis, dataset, panel)
        axis.set_ylabel(PANEL_Y_LABEL[panel], fontsize=9.4)
        axis.set_title(("A. " if panel == "A" else "B. ") + PANEL_TITLE[panel], loc="left", fontsize=10.5, weight="bold")
    fig.suptitle(curves[0].case_label, fontsize=12.5, weight="bold", y=0.995)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=4, frameon=False, fontsize=8.6, handlelength=2.2, columnspacing=1.1)
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_composite(dataset: str, case_groups: Sequence[Sequence[DiagnosticCurve]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(20.0, 13.0), sharex=False, sharey=False)
    for block in range(2):
        for column in range(5):
            curves = case_groups[block * 5 + column]
            for panel_offset, panel in enumerate(["A", "B"]):
                row = block * 2 + panel_offset
                axis = axes[row, column]
                draw_curves(axis, curves, panel)
                style_axis(axis, dataset, panel, show_xlabel=panel == "B")
                if column == 0:
                    axis.set_ylabel(PANEL_Y_LABEL[panel], fontsize=9.4, weight="bold")
                if panel == "A":
                    axis.set_title(short_case_title(curves[0]), fontsize=10.3, weight="bold")
    fig.suptitle(f"{DATASET_LABEL[dataset]} diagnostics (v002; DIR-LAB Panel A uses fixed T00 lung mask)", fontsize=17, weight="bold", y=0.992)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=11, handlelength=2.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.95, bottom=0.075, hspace=0.37, wspace=0.28)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def case_groups(curves: Sequence[DiagnosticCurve], dataset: str) -> list[list[DiagnosticCurve]]:
    groups: dict[str, list[DiagnosticCurve]] = {}
    order: dict[str, int] = {}
    for curve in curves:
        if curve.dataset != dataset:
            continue
        groups.setdefault(curve.case_key, []).append(curve)
        order[curve.case_key] = curve.case_order
    result = [groups[key] for key in sorted(groups, key=lambda value: order[value])]
    if len(result) != 10:
        raise RuntimeError(f"{dataset}: expected ten case groups, found {len(result)}")
    return result


def aggregate_group(curves: Sequence[DiagnosticCurve]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = []
    for curve in curves:
        values = np.interp(AGGREGATE_EPOCHS, curve.epochs, curve.values)
        if not np.isfinite(values).all() or not (values > 0).all():
            raise RuntimeError("Invalid aggregate curve values")
        stacked.append(np.log10(values))
    logs = np.vstack(stacked)
    mean_log = np.mean(logs, axis=0)
    std_log = np.std(logs, axis=0, ddof=1)
    return 10.0**mean_log, 10.0 ** (mean_log - std_log), 10.0 ** (mean_log + std_log)


def draw_population_overlay(axis: plt.Axes, curves: Sequence[DiagnosticCurve], dataset: str, panel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_curves = [curve for curve in curves if curve.dataset == dataset and curve.panel == panel and curve.method_slug == method]
        if len(method_curves) != 10:
            raise RuntimeError(f"{dataset}/Panel {panel}/{method}: expected ten curves")
        style = METHOD_STYLE[method]
        for curve in method_curves:
            axis.plot(curve.epochs, curve.values, color=style["color"], linewidth=0.72, alpha=0.22, zorder=style["zorder"])
        mean, low, high = aggregate_group(method_curves)
        axis.fill_between(AGGREGATE_EPOCHS, low, high, color=style["color"], alpha=0.13, linewidth=0, zorder=style["zorder"] - 1)
        axis.plot(AGGREGATE_EPOCHS, mean, color=style["color"], linewidth=2.6, alpha=0.98, zorder=style["zorder"] + 3)
        rows.append(
            {
                "dataset": dataset,
                "panel": panel,
                "method_slug": method,
                "n_cases": len(method_curves),
                "endpoint_log_mean": float(mean[-1]),
                "endpoint_log_minus_1sd": float(low[-1]),
                "endpoint_log_plus_1sd": float(high[-1]),
            }
        )
    return rows


def render_overlay_2x2(curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=False, sharey="row")
    summary: list[dict[str, Any]] = []
    for row, panel in enumerate(["A", "B"]):
        for column, dataset in enumerate(["oasis", "dirlab"]):
            axis = axes[row, column]
            summary.extend(draw_population_overlay(axis, curves, dataset, panel))
            style_axis(axis, dataset, panel)
            axis.set_title(DATASET_LABEL[dataset], fontsize=12.2, weight="bold")
            if column == 0:
                axis.set_ylabel(("A. " if panel == "A" else "B. ") + PANEL_Y_LABEL[panel], fontsize=11.0, weight="bold")
    fig.legend(handles=legend_handles(3.0), loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, frameon=False, fontsize=10.4)
    fig.suptitle("Population overlay diagnostics: individual cases, log-space mean, and ±1 SD", fontsize=15.0, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.025, 0.055, 1.0, 0.955))
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return summary


def draw_single_case(axis: plt.Axes, curves: Sequence[DiagnosticCurve], panel: str, linestyle: Any = "solid", alpha: float = 1.0) -> None:
    for method in METHOD_ORDER:
        curve = method_curve(curves, panel, method)
        style = METHOD_STYLE[method]
        axis.plot(curve.epochs, curve.values, color=style["color"], linewidth=2.1, linestyle=linestyle, alpha=alpha, zorder=style["zorder"] + 3)


def render_design_2x4(curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int, tre_labels: bool = False) -> None:
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
                draw_population_overlay(axis, curves, key, panel)
                dataset = key
            else:
                case_curves = [curve for curve in curves if curve.dataset == "dirlab" and curve.case_key == key]
                draw_single_case(axis, case_curves, panel)
                dataset = "dirlab"
                if tre_labels and panel == "A":
                    add_tre_labels(axis, case_curves, key)
            style_axis(axis, dataset, panel)
            axis.set_title(title, fontsize=12.0, weight="bold")
            if column == 0:
                axis.set_ylabel(("A. " if panel == "A" else "B. ") + PANEL_Y_LABEL[panel], fontsize=11.0, weight="bold")
    fig.legend(handles=legend_handles(3.0), loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, frameon=False, fontsize=10.4)
    fig.suptitle("Population diagnostics with DIR-LAB representative cases", fontsize=15.0, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.025, 0.055, 1.0, 0.955))
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_design_2x3(curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.3), sharex=False, sharey="row")
    for row, panel in enumerate(["A", "B"]):
        for column, dataset in enumerate(["oasis", "dirlab"]):
            axis = axes[row, column]
            draw_population_overlay(axis, curves, dataset, panel)
            style_axis(axis, dataset, panel)
            axis.set_title(DATASET_LABEL[dataset] + " population", fontsize=12.0, weight="bold")
            if column == 0:
                axis.set_ylabel(("A. " if panel == "A" else "B. ") + PANEL_Y_LABEL[panel], fontsize=11.0, weight="bold")
        axis = axes[row, 2]
        for case_key, case_style in LUNGCT_CASE_STYLES.items():
            case_curves = [curve for curve in curves if curve.dataset == "dirlab" and curve.case_key == case_key]
            draw_single_case(axis, case_curves, panel, linestyle=case_style["linestyle"], alpha=0.96)
        style_axis(axis, "dirlab", panel)
        axis.set_title("DIR-LAB case01 vs case08", fontsize=12.0, weight="bold")
    method_handles = legend_handles(2.8)
    case_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.2, linestyle=LUNGCT_CASE_STYLES[key]["linestyle"], label=LUNGCT_CASE_STYLES[key]["label"])
        for key in ["case01", "case08"]
    ]
    fig.legend(handles=method_handles + case_handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=6, frameon=False, fontsize=9.6)
    fig.suptitle("Compact population and representative-case diagnostic trial", fontsize=15.0, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.025, 0.062, 1.0, 0.955))
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def summary_for_curve(curve: DiagnosticCurve) -> dict[str, Any]:
    path = curve.source_path.parent / "summary.json"
    if not path.is_file():
        return {}
    return load_json(path)


def add_tre_labels(axis: plt.Axes, case_curves: Sequence[DiagnosticCurve], case_key: str) -> None:
    positions = TRE_LABEL_POSITIONS[case_key]
    for method in METHOD_ORDER:
        curve = method_curve(case_curves, "A", method)
        summary = summary_for_curve(curve)
        tre = finite(summary.get("tre_snap_to_voxel", {}).get("tre_mean_mm"))
        if tre is None:
            continue
        x_text, y_text = positions[method]
        style = METHOD_STYLE[method]
        axis.annotate(
            f"TRE {tre:.2f} mm",
            xy=(float(curve.epochs[-1]), float(curve.values[-1])),
            xytext=(x_text, y_text),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9.5,
            color=style["color"],
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": style["color"], "linewidth": 0.82, "alpha": 0.75},
            zorder=10,
        )


def image_record(path: Path, output_root: Path, kind: str) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "kind": kind,
        "path": os.path.relpath(path, output_root),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def write_report(
    output_root: Path,
    audit: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    lines = [
        "# v002 Lung-Mask Panel A Population Diagnostics",
        "",
        "This is a no-overwrite candidate figure family. v001 remains the full-crop/schedule-matched historical comparison; v002 changes only DIR-LAB Panel A to fixed T00 lung-mask `1−NCC`.",
        "",
        "## Main artifacts",
        "",
    ]
    for image in images:
        if image["kind"] in {"dirlab_4x5", "overlay_2x2", "design_2x4", "design_2x3", "design_2x4_tre"}:
            lines.append(f"- [{image['path']}]({image['path']})")
    lines.extend(
        [
            "",
            "## Validation summary",
            "",
            f"- v002 source records: `{counts['v002_rows']}`.",
            f"- DIR-LAB Panel A lung-mask replacements: `{counts['replacements']}`.",
            f"- Unchanged non-DIR-LAB-Panel-A records: `{counts['unchanged_rows']}`.",
            "- Axis definitions are locked to Panel A `1e-3–5e-1`, Panel B `1e-1–1e3`, epochs `0–2500`.",
            "- Case01/case08 dual diagnostics are exact-compatible reused; cases02–07/09/10 are newly launched records when absent.",
            "- MR-D-BSCP Panel A schedule is `400/400/800/900` for all DIR-LAB cases.",
            "",
            "## Before/after endpoint comparison",
            "",
            "| Case | Method | v001 full-crop 1−NCC | v002 lung-mask 1−NCC | Difference | TRE mean mm |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(audit, key=lambda item: (item["case_id"], METHOD_ORDER.index(item["method_slug"]))):
        lines.append(
            f"| case{row['case_id']:02d} | {row['display_method']} | "
            f"{row['old_v001_endpoint_full_crop']:.6g} | {row['new_v002_endpoint_lung_mask']:.6g} | "
            f"{row['new_v002_endpoint_lung_mask'] - row['old_v001_endpoint_full_crop']:.6g} | "
            f"{row['tre_mean_mm']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Panel B is unchanged from v001 because the accepted supervised fitting curves already use the fixed-mask rRMSE convention.",
            "- OASIS is regenerated/copied into v002 for side-by-side completeness but is intentionally unchanged.",
        ]
    )
    (output_root / "diagnostic_comparison.md").write_text("\n".join(lines) + "\n")


def write_validation_report(output_root: Path, counts: dict[str, Any], audit: Sequence[dict[str, Any]], curve_rows: Sequence[dict[str, Any]]) -> None:
    failures = []
    if counts["v002_rows"] != 160 or counts["replacements"] != 40:
        failures.append("source-index-counts")
    if len(audit) != 40:
        failures.append("audit-count")
    if sum(row["reused_existing_case01_case08"] for row in audit) != 8:
        failures.append("case01-case08-reuse-count")
    max_repro_diff = max(abs(row["full_crop_endpoint_repro_diff_vs_v001"]) for row in audit)
    if max_repro_diff > 2.0e-5:
        failures.append("full-crop-repro-diff")
    lung_rows = [
        row for row in curve_rows if row["dataset"] == "dirlab" and row["panel"] == "A"
    ]
    if len(lung_rows) != 40 or any(row["points"] != 251 or row["epoch_max"] != 2500 for row in lung_rows):
        failures.append("lung-curve-length")
    payload = {
        "created_at_utc": utc_text(),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "counts": counts,
        "max_full_crop_reproduction_abs_diff_vs_v001": max_repro_diff,
        "dirlab_panelA_dual_diagnostic_records": len(audit),
        "dirlab_panelA_reused_case01_case08_records": sum(row["reused_existing_case01_case08"] for row in audit),
        "dirlab_panelA_new_records_expected": 32,
    }
    (output_root / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# v002 Lung-Mask Panel A Validation Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source index rows: `{counts['v002_rows']}`",
        f"- Replacements: `{counts['replacements']}`",
        f"- Unchanged rows: `{counts['unchanged_rows']}`",
        f"- DIR-LAB Panel A dual-diagnostic records: `{len(audit)}/40`",
        f"- Reused case01/case08 records: `{payload['dirlab_panelA_reused_case01_case08_records']}/8`",
        f"- Max full-crop endpoint reproduction abs diff vs v001: `{max_repro_diff:.3e}`",
    ]
    if failures:
        lines.extend(["", "## Failures", "", *[f"- `{failure}`" for failure in failures]])
    (output_root / "validation_report.md").write_text("\n".join(lines) + "\n")
    if failures:
        raise RuntimeError(f"Validation failed: {failures}")


def sanitize_launcher_sidecars(output_root: Path) -> list[str]:
    """Replace private/container paths in launcher sidecars with portable tokens."""

    replacements = [
        ("/homebase/code_sync/inrorspline", "${REPO_ROOT}"),
        ("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm", "${LUNGCT_PTVREG_ISO1MM}"),
        (
            "/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1",
            "${DIRLAB_LUNGMASK_RUN_ROOT}",
        ),
        (
            "/database/inrorspline/runs/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1",
            "${DIRLAB_REFRESH_RUN_ROOT}",
        ),
        ("/database/inrorspline/runs/fit_modes_population_n10_v1", "${POPULATION_RUN_ROOT}"),
        ("/database/inrorspline", "${DATA_ROOT}"),
    ]
    sanitized: list[str] = []
    for filename in [
        "lungmask_before_after_curves_launcher_state.json",
        "lungmask_before_after_curves_job_manifest.csv",
        "lungmask_before_after_curves_monitor.md",
    ]:
        path = output_root / filename
        if not path.is_file():
            continue
        text = path.read_text()
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)
            sanitized.append(filename)
    return sanitized


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = output_root / "case_figures"
    case_dir.mkdir(parents=True, exist_ok=True)

    v001_index_path = args.v001_root / "camera_ready_source_index.csv"
    v001 = pd.read_csv(v001_index_path)
    v002, audit, counts = build_v002_source_index(v001, args.lungmask_run_root.resolve())
    validate_v002_index(v002)
    v002_csv = output_root / "camera_ready_source_index_lungmask_panelA.csv"
    v002_json = output_root / "camera_ready_source_index_lungmask_panelA.json"
    v002.to_csv(v002_csv, index=False)
    v002_json.write_text(v002.to_json(orient="records", indent=2) + "\n")

    roots = {
        POPULATION_TOKEN: args.population_run_root.resolve(),
        REFRESH_TOKEN: args.refresh_run_root.resolve(),
        LUNGMASK_TOKEN: args.lungmask_run_root.resolve(),
        DATA_TOKEN: args.data_root.resolve(),
    }
    curves, curve_summary = load_and_validate_curves(v002, roots)
    write_csv(output_root / "curve_summary.csv", curve_summary)

    images: list[dict[str, Any]] = []
    for dataset in ["oasis", "dirlab"]:
        groups = case_groups(curves, dataset)
        composite = output_root / f"{dataset}_population_diagnostics_4x5.png"
        render_composite(dataset, groups, composite, args.dpi)
        images.append(image_record(composite, output_root, f"{dataset}_4x5"))
        for group in groups:
            case_key = group[0].case_key
            path = case_dir / f"{dataset}_{case_key}_diagnostics.png"
            render_case_figure(group, path, args.dpi)
            images.append(image_record(path, output_root, "case_figure"))

    overlay_summary = render_overlay_2x2(curves, output_root / "population_overlay_mean_std_2x2.png", args.dpi)
    images.append(image_record(output_root / "population_overlay_mean_std_2x2.png", output_root, "overlay_2x2"))
    write_csv(output_root / "population_overlay_mean_std_summary.csv", overlay_summary)

    render_design_2x4(curves, output_root / "population_overlay_plus_lungct_cases_2x4.png", args.dpi)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x4.png", output_root, "design_2x4"))
    render_design_2x3(curves, output_root / "population_overlay_plus_lungct_cases_2x3.png", args.dpi)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x3.png", output_root, "design_2x3"))
    render_design_2x4(curves, output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", args.dpi, tre_labels=True)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", output_root, "design_2x4_tre"))

    write_report(output_root, audit, images, counts)
    write_validation_report(output_root, counts, audit, curve_summary)
    manifest = {
        "created_at_utc": utc_text(),
        "version": "v002_lungmask_panelA",
        "v001_source_index": "v001_schedule_matched/camera_ready_source_index.csv",
        "run_root_tokens": sorted(roots),
        "axis_limits": AXIS_LIMITS,
        "stage_boundaries": {f"{dataset}_{panel}": values for (dataset, panel), values in STAGE_BOUNDARIES.items()},
        "counts": counts,
        "images": images,
        "checksums": {
            "source_index_csv": sha256(v002_csv),
            "source_index_json": sha256(v002_json),
            "curve_summary_csv": sha256(output_root / "curve_summary.csv"),
            "validation_report_json": sha256(output_root / "validation_report.json"),
        },
    }
    (output_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(output_root / "lungmask_panelA_replacement_audit.csv", audit)
    sanitized = sanitize_launcher_sidecars(output_root)
    print(
        json.dumps(
            {
                "status": "passed",
                "output_root": str(output_root),
                "images": len(images),
                "sanitized_sidecars": sanitized,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
