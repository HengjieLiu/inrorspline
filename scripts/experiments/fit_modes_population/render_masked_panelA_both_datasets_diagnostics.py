#!/usr/bin/env python
"""Render v003 diagnostics with fixed-domain Panel A metrics for both datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from PIL import Image

from plot_case_diagnostics import DiagnosticCurve, METHOD_ORDER, normalize_terminal_endpoint
from render_lungmask_panelA_population_diagnostics import (
    AXIS_LIMITS,
    DATA_TOKEN,
    DATASET_LABEL,
    DEFAULT_DATA_ROOT,
    LUNGMASK_TOKEN,
    PANEL_Y_LABEL,
    POPULATION_TOKEN,
    REFRESH_TOKEN,
    STAGE_BOUNDARIES,
    case_groups,
    case_label,
    draw_curves,
    finite,
    legend_handles,
    load_json,
    load_v002_curve,
    portable_from_root,
    render_case_figure,
    render_design_2x3,
    render_design_2x4,
    render_overlay_2x2,
    resolve_portable,
    sha256,
    short_case_title,
    style_axis,
    validate_v002_index,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


EXPERIMENT_KEY = "paper_artifacts/fit_modes_population_diagnostics/v003_masked_panelA_both_datasets"
OASIS_MASKED_RUN_KEY = "oasis_panelA_fixedmask_diag_curves_v1"
POPULATION_RUN_KEY = "fit_modes_population_n10_v1"
REFRESH_RUN_KEY = "fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
LUNGMASK_RUN_KEY = "dirlab_case01_case08_panelA_lungmask_diag_curves_v1"
OASIS_MASKED_TOKEN = "${OASIS_MASKED_PANELA_RUN_ROOT}"
EXPECTED_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=float)

OASIS_METHODS = {
    "idir": {
        "impl": "idir_original_dense_siren_be_autograd",
        "run_name": "oasis_fixedmask_rank{rank:03d}_panelA_idir_e2500_a1_lr1em4_seed1_diag10",
    },
    "inr_bscp": {
        "impl": "sinr_cps2_be_bspline_analytic",
        "run_name": "oasis_fixedmask_rank{rank:03d}_panelA_inr_bscp_e2500_a1000_lr1em4_seed1_diag10",
    },
    "d_bscp": {
        "impl": "direct_cp_cps2_be_bspline_analytic",
        "run_name": "oasis_fixedmask_rank{rank:03d}_panelA_d_bscp_e2500_a1000_lr1em3_seed1_diag10",
    },
    "mr_d_bscp": {
        "impl": "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "run_name": (
            "oasis_fixedmask_rank{rank:03d}_panelA_mr_d_bscp_"
            "sched8-4-2_ds4-2-1_e400-800-1300_a1000_lr0p001_seed1_diag10"
        ),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v002-root", type=Path, required=True)
    parser.add_argument("--oasis-masked-run-root", type=Path, default=None)
    parser.add_argument("--population-run-root", type=Path, default=None)
    parser.add_argument("--refresh-run-root", type=Path, default=None)
    parser.add_argument("--lungmask-run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Route this versioned renderer and all migrated generated inputs externally."""

    args.v002_root = artifact_paths.resolve_external_input_directory(
        args.v002_root,
        label="--v002-root",
        repo_root=REPO_ROOT,
    )
    args.oasis_masked_run_root = artifact_paths.resolve_run_root(
        OASIS_MASKED_RUN_KEY,
        args.oasis_masked_run_root,
        repo_root=REPO_ROOT,
    )
    args.population_run_root = artifact_paths.resolve_run_root(
        POPULATION_RUN_KEY,
        args.population_run_root,
        repo_root=REPO_ROOT,
    )
    args.refresh_run_root = artifact_paths.resolve_run_root(
        REFRESH_RUN_KEY,
        args.refresh_run_root,
        repo_root=REPO_ROOT,
    )
    args.lungmask_run_root = artifact_paths.resolve_run_root(
        LUNGMASK_RUN_KEY,
        args.lungmask_run_root,
        repo_root=REPO_ROOT,
    )
    args.output_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.output_root,
        repo_root=REPO_ROOT,
    )
    return args


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


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


def oasis_paths(rank: int, method_slug: str, run_root: Path) -> tuple[Path, Path]:
    spec = OASIS_METHODS[method_slug]
    run_dir = run_root / spec["impl"] / spec["run_name"].format(rank=rank)
    return run_dir / "global_ncc_diagnostic.csv", run_dir / "summary.json"


def validate_oasis_curve(curve_path: Path, summary_path: Path, row: pd.Series) -> dict[str, Any]:
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    curve = pd.read_csv(curve_path)
    summary = load_json(summary_path)
    required = [
        "global_epoch",
        "global_ncc_value",
        "fixed_image_positive_ncc_value",
        "fixed_image_positive_one_minus_ncc",
    ]
    missing = [column for column in required if column not in curve.columns]
    if missing:
        raise RuntimeError(f"{curve_path}: missing columns {missing}")
    epochs = curve["global_epoch"].astype(int).to_numpy()
    if not np.array_equal(epochs, EXPECTED_EPOCHS.astype(int)):
        raise RuntimeError(f"{curve_path}: expected 251 diagnostics ending at 2500")
    values = curve[["global_ncc_value", "fixed_image_positive_one_minus_ncc", "fixed_image_positive_ncc_value"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{curve_path}: nonfinite paired diagnostic")
    if not (curve["fixed_image_positive_one_minus_ncc"].to_numpy(float) > 0.0).all():
        raise RuntimeError(f"{curve_path}: nonpositive fixed-mask 1-NCC")
    if summary.get("status") != "completed" and not (summary_path.parent / "DONE").is_file():
        raise RuntimeError(f"{summary_path}: summary is not completed")
    if int(summary.get("case_rank", -1)) != int(row["case_rank"]):
        raise RuntimeError(f"{summary_path}: case rank mismatch")
    if str(summary.get("fixed_id")).zfill(4) != str(int(row["fixed_id"])).zfill(4):
        raise RuntimeError(f"{summary_path}: fixed id mismatch")
    if str(summary.get("moving_id")).zfill(4) != str(int(row["moving_id"])).zfill(4):
        raise RuntimeError(f"{summary_path}: moving id mismatch")
    if int(summary.get("global_ncc_diagnostic_interval", -1)) != 10:
        raise RuntimeError(f"{summary_path}: diagnostic interval is not 10")
    if summary.get("fixed_image_positive_ncc_diagnostic_name") != "fixed_image_positive_global_ncc":
        raise RuntimeError(f"{summary_path}: fixed-image-positive diagnostic metadata missing")
    if str(row["method_slug"]) == "mr_d_bscp":
        if summary.get("stage_full_cps") != [8, 4, 2]:
            raise RuntimeError(f"{summary_path}: MR-D-BSCP cps mismatch")
        if summary.get("stage_downsample") != [4, 2, 1]:
            raise RuntimeError(f"{summary_path}: MR-D-BSCP downsample mismatch")
        if summary.get("stage_epochs") != [400, 800, 1300]:
            raise RuntimeError(f"{summary_path}: MR-D-BSCP stage epochs mismatch")
    final = curve.iloc[-1]
    full_one_minus = 1.0 - float(final["global_ncc_value"])
    masked_one_minus = float(final["fixed_image_positive_one_minus_ncc"])
    return {
        "case_key": row["case_key"],
        "case_rank": int(row["case_rank"]),
        "method_slug": row["method_slug"],
        "old_v002_endpoint_full_image": float(row["endpoint_value"]),
        "new_v003_endpoint_fixed_image_positive": masked_one_minus,
        "full_image_endpoint_repro_diff_vs_v002": full_one_minus - float(row["endpoint_value"]),
        "fixed_image_positive_ncc_value": float(final["fixed_image_positive_ncc_value"]),
        "fixed_image_positive_mask_voxels": int(final.get("fixed_image_positive_mask_voxels", summary.get("fixed_image_positive_voxels", -1))),
        "point_count": int(len(curve)),
        "epoch_max": int(epochs[-1]),
        "runtime_sec": finite(summary.get("training_elapsed_sec")),
        "peak_memory_gb": finite(summary.get("training_peak_mem_gb")),
        "curve_sha256": sha256(curve_path),
        "summary_sha256": sha256(summary_path),
    }


def build_v003_source_index(v002: pd.DataFrame, oasis_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    v003 = v002.copy()
    target = v003["dataset"].eq("oasis") & v003["panel"].eq("A")
    audit: list[dict[str, Any]] = []
    for row_index, row in v003[target].iterrows():
        rank = int(row["case_rank"])
        method_slug = str(row["method_slug"])
        curve_path, summary_path = oasis_paths(rank, method_slug, oasis_root)
        check = validate_oasis_curve(curve_path, summary_path, row)
        updates = {
            "status": "completed",
            "source_kind": "oasis_fixedmask_panelA_dual_diagnostic",
            "source_path": portable_from_root(curve_path, oasis_root, OASIS_MASKED_TOKEN),
            "summary_path": portable_from_root(summary_path, oasis_root, OASIS_MASKED_TOKEN),
            "endpoint_name": "fixed_image_positive_one_minus_ncc",
            "endpoint_value": check["new_v003_endpoint_fixed_image_positive"],
            "diagnostic_ncc": check["fixed_image_positive_ncc_value"],
            "runtime_sec": check["runtime_sec"],
            "peak_memory_gb": check["peak_memory_gb"],
            "endpoint_source": "oasis_fixedmask_panelA:fixed_image_positive_one_minus_ncc",
            "diagnostic_interval": 10,
            "acceptance_checks": (
                "oasis_fixedmask_panelA=true;dual_diagnostic=true;curve_rows=251;"
                "endpoint_source=fixed_image_positive_one_minus_ncc"
            ),
        }
        for field, value in updates.items():
            v003.at[row_index, field] = value
        audit.append({**check, "job_id": row["job_id"], "source_path": updates["source_path"], "summary_path": updates["summary_path"]})
    untouched_cols = [column for column in v002.columns if column in v003.columns]
    pd.testing.assert_frame_equal(v002.loc[~target, untouched_cols], v003.loc[~target, untouched_cols])
    return v003, audit, {
        "v002_rows": int(len(v002)),
        "v003_rows": int(len(v003)),
        "replacements": int(target.sum()),
        "unchanged_rows": int((~target).sum()),
    }


def load_v003_curve(row: pd.Series, roots: dict[str, Path]) -> DiagnosticCurve:
    source_text = str(row["source_path"])
    if row["dataset"] == "oasis" and row["panel"] == "A" and source_text.startswith(OASIS_MASKED_TOKEN):
        source = resolve_portable(source_text, {OASIS_MASKED_TOKEN: roots[OASIS_MASKED_TOKEN]})
        frame = pd.read_csv(source)
        epochs = frame["global_epoch"].to_numpy(float)
        values = frame["fixed_image_positive_one_minus_ncc"].to_numpy(float)
        keep = np.isfinite(epochs) & np.isfinite(values) & (values > 0.0)
        epochs = epochs[keep]
        values = values[keep]
        order = np.argsort(epochs, kind="stable")
        endpoint = finite(row["endpoint_value"])
        if endpoint is None:
            raise RuntimeError(f"{row['job_id']}: missing endpoint")
        return DiagnosticCurve(
            dataset="oasis",
            case_key=str(row["case_key"]),
            case_order=int(row["case_rank"]),
            case_label=case_label(row),
            panel="A",
            method_slug=str(row["method_slug"]),
            method_label=str(row["display_method"]),
            epochs=epochs[order],
            values=values[order],
            source_path=source,
            source_metric="fixed_image_positive_one_minus_ncc",
            target_power=None,
            accepted_endpoint=endpoint,
        )
    return load_v002_curve(row, roots)


def validate_v003_index(index: pd.DataFrame) -> None:
    validate_v002_index(index)
    replacements = index[index["source_kind"].eq("oasis_fixedmask_panelA_dual_diagnostic")]
    if len(replacements) != 40:
        raise RuntimeError(f"Expected 40 OASIS Panel A replacements, found {len(replacements)}")
    if not (replacements["dataset"].eq("oasis") & replacements["panel"].eq("A")).all():
        raise RuntimeError("Non-OASIS Panel A row marked as v003 replacement")
    dirlab_panel_a = index[index["dataset"].eq("dirlab") & index["panel"].eq("A")]
    if not dirlab_panel_a["endpoint_name"].eq("lung_mask_one_minus_ncc").all():
        raise RuntimeError("DIR-LAB Panel A must remain the v002 lung-mask metric")


def load_and_validate_curves(index: pd.DataFrame, roots: dict[str, Path]) -> tuple[list[DiagnosticCurve], list[dict[str, Any]]]:
    curves: list[DiagnosticCurve] = []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in index.iterrows():
        try:
            raw = load_v003_curve(row, roots)
            curve, raw_endpoint, raw_difference, endpoint_replaced = normalize_terminal_endpoint(raw)
            plotted = float(curve.values[-1])
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
                    "endpoint_difference": plotted - float(curve.accepted_endpoint),
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
    oasis_masked = [curve for curve in curves if curve.dataset == "oasis" and curve.panel == "A"]
    if len(oasis_masked) != 40:
        raise RuntimeError(f"Expected 40 OASIS Panel A fixed-mask curves, found {len(oasis_masked)}")
    for curve in oasis_masked:
        if len(curve.epochs) != 251 or float(curve.epochs[-1]) != 2500.0:
            raise RuntimeError(f"{curve.case_key}/{curve.method_slug}: invalid OASIS fixed-mask diagnostic length")
    return curves, rows


def render_composite_v003(dataset: str, case_group_list: Sequence[Sequence[DiagnosticCurve]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(20.0, 13.0), sharex=False, sharey=False)
    for block in range(2):
        for column in range(5):
            curves = case_group_list[block * 5 + column]
            for panel_offset, panel in enumerate(["A", "B"]):
                row = block * 2 + panel_offset
                axis = axes[row, column]
                draw_curves(axis, curves, panel)
                style_axis(axis, dataset, panel, show_xlabel=panel == "B")
                if column == 0:
                    axis.set_ylabel(PANEL_Y_LABEL[panel], fontsize=9.4, weight="bold")
                if panel == "A":
                    axis.set_title(short_case_title(curves[0]), fontsize=10.3, weight="bold")
    title_suffix = "Panel A uses fixed image>0 mask" if dataset == "oasis" else "Panel A uses fixed T00 lung mask"
    fig.suptitle(f"{DATASET_LABEL[dataset]} diagnostics (v003; {title_suffix})", fontsize=17, weight="bold", y=0.992)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=11, handlelength=2.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.95, bottom=0.075, hspace=0.37, wspace=0.28)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_reports(output_root: Path, counts: dict[str, int], audit: Sequence[dict[str, Any]], curve_summary: Sequence[dict[str, Any]], images: Sequence[dict[str, Any]]) -> None:
    max_repro = max(abs(float(row["full_image_endpoint_repro_diff_vs_v002"])) for row in audit)
    payload = {
        "created_at_utc": utc_text(),
        "status": "passed",
        "counts": counts,
        "oasis_panelA_fixedmask_records": len(audit),
        "max_full_image_reproduction_abs_diff_vs_v002": max_repro,
        "axis_limits": AXIS_LIMITS,
    }
    (output_root / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_root / "validation_report.md").write_text(
        "\n".join(
            [
                "# v003 Fixed-Domain Panel A Validation Report",
                "",
                "- Status: `passed`",
                f"- Source index rows: `{counts['v003_rows']}`",
                f"- OASIS Panel A replacements: `{counts['replacements']}/40`",
                f"- Unchanged v002 rows: `{counts['unchanged_rows']}`",
                f"- Max OASIS full-image endpoint reproduction abs diff vs v002: `{max_repro:.3e}`",
                "- DIR-LAB Panel A remains the accepted v002 fixed-T00-lung-mask metric.",
                "- All Panel B rows are preserved from v002.",
            ]
        )
        + "\n"
    )
    (output_root / "diagnostic_comparison.md").write_text(
        "\n".join(
            [
                "# v003 Diagnostic Comparison",
                "",
                "v003 replaces only OASIS Panel A. The new plotted OASIS Panel A metric is fixed raw `image > 0` masked `1−NCC`.",
                "DIR-LAB Panel A remains the v002 fixed T00 lung-mask `1−NCC`; both datasets now use fixed-domain Panel A metrics.",
                "",
                f"- Replacements: `{counts['replacements']}`",
                f"- Unchanged rows: `{counts['unchanged_rows']}`",
                f"- Generated images: `{len(images)}`",
                "",
                "The full-image OASIS `1−NCC` column is retained in each new curve CSV as a reproducibility cross-check.",
            ]
        )
        + "\n"
    )
    (output_root / "metric_domain_summary.md").write_text(
        "\n".join(
            [
                "# v003 Camera-Ready Diagnostic Metric Domain Summary",
                "",
                "v003 is the camera-ready diagnostic candidate where both datasets use anatomically appropriate fixed-domain masks for Panel A and Panel B.",
                "",
                "| Dataset | Panel | Plotted metric | Evaluation domain |",
                "|---|---|---|---|",
                "| OASIS | A: unsupervised registration | `1−NCC` | Fixed raw `image > 0` mask |",
                "| OASIS | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed raw `image > 0` mask |",
                "| DIR-LAB/LungCT | A: unsupervised registration | `1−NCC` | Fixed T00 lung mask on canonical 1 mm crop |",
                "| DIR-LAB/LungCT | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed T00 lung mask on canonical 1 mm crop |",
                "",
                "Panel A uses the same global NCC formula in both datasets: `ncc_loss = -corr(fixed, warped)`, and the plotted value is `1 + ncc_loss`. The difference is the fixed-domain mask: OASIS uses fixed raw `image > 0`; DIR-LAB uses the fixed T00 lung mask.",
                "",
                "Panel B is unchanged from v002. It plots `100 × sqrt(masked MSE / (masked target power + 1e-12))`, using SITReg normalized-grid targets for OASIS and pTVReg normalized-grid targets for DIR-LAB.",
                "",
                "Training objectives are not changed by v003. The refresh changes the diagnostic/evaluation domain for OASIS Panel A only.",
                "",
                "Safe claim: the plotted Panel A metric is now method-consistent within each dataset and uses each dataset's fixed anatomical domain. Safe caveat: OASIS and DIR-LAB use different anatomical masks, and Panel A training domains are still not identical across all methods.",
            ]
        )
        + "\n"
    )
    write_csv(output_root / "oasis_fixedmask_panelA_replacement_audit.csv", audit)


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = output_root / "case_figures"
    case_dir.mkdir(parents=True, exist_ok=True)

    v002_csv = args.v002_root / "camera_ready_source_index_lungmask_panelA.csv"
    v002 = pd.read_csv(v002_csv)
    v003, audit, counts = build_v003_source_index(v002, args.oasis_masked_run_root.resolve())
    validate_v003_index(v003)
    v003_csv = output_root / "camera_ready_source_index_masked_panelA_both_datasets.csv"
    v003_json = output_root / "camera_ready_source_index_masked_panelA_both_datasets.json"
    v003.to_csv(v003_csv, index=False)
    v003_json.write_text(v003.to_json(orient="records", indent=2) + "\n")

    roots = {
        POPULATION_TOKEN: args.population_run_root.resolve(),
        REFRESH_TOKEN: args.refresh_run_root.resolve(),
        LUNGMASK_TOKEN: args.lungmask_run_root.resolve(),
        OASIS_MASKED_TOKEN: args.oasis_masked_run_root.resolve(),
        DATA_TOKEN: args.data_root.resolve(),
    }
    curves, curve_summary = load_and_validate_curves(v003, roots)
    write_csv(output_root / "curve_summary.csv", curve_summary)

    images: list[dict[str, Any]] = []
    for dataset in ["oasis", "dirlab"]:
        groups = case_groups(curves, dataset)
        composite = output_root / f"{dataset}_population_diagnostics_4x5.png"
        render_composite_v003(dataset, groups, composite, args.dpi)
        images.append(image_record(composite, output_root, f"{dataset}_4x5"))
        for group in groups:
            case_key = group[0].case_key
            path = case_dir / f"{dataset}_{case_key}_diagnostics.png"
            render_case_figure(group, path, args.dpi)
            images.append(image_record(path, output_root, "case_figure"))

    overlay_summary = render_overlay_2x2(curves, output_root / "population_overlay_mean_std_2x2.png", args.dpi)
    write_csv(output_root / "population_overlay_mean_std_summary.csv", overlay_summary)
    images.append(image_record(output_root / "population_overlay_mean_std_2x2.png", output_root, "overlay_2x2"))
    render_design_2x4(curves, output_root / "population_overlay_plus_lungct_cases_2x4.png", args.dpi)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x4.png", output_root, "design_2x4"))
    render_design_2x3(curves, output_root / "population_overlay_plus_lungct_cases_2x3.png", args.dpi)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x3.png", output_root, "design_2x3"))
    render_design_2x4(curves, output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", args.dpi, tre_labels=True)
    images.append(image_record(output_root / "population_overlay_plus_lungct_cases_2x4_tre_labels.png", output_root, "design_2x4_tre"))

    write_reports(output_root, counts, audit, curve_summary, images)
    manifest = {
        "created_at_utc": utc_text(),
        "version": "v003_masked_panelA_both_datasets",
        "v002_source_index": "v002_lungmask_panelA/camera_ready_source_index_lungmask_panelA.csv",
        "run_root_tokens": sorted(roots),
        "axis_limits": AXIS_LIMITS,
        "stage_boundaries": {f"{dataset}_{panel}": values for (dataset, panel), values in STAGE_BOUNDARIES.items()},
        "counts": counts,
        "images": images,
        "checksums": {
            "source_index_csv": sha256(v003_csv),
            "source_index_json": sha256(v003_json),
            "curve_summary_csv": sha256(output_root / "curve_summary.csv"),
            "validation_report_json": sha256(output_root / "validation_report.json"),
        },
    }
    (output_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output_root": str(output_root), "images": len(images)}, indent=2))


if __name__ == "__main__":
    main()
