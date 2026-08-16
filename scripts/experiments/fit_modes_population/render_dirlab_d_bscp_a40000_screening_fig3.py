#!/usr/bin/env python
"""Render a v004 Figure-3 screening variant with D-BSCP a=40000.

This is intentionally *not* a new camera-ready source index.  It keeps the
accepted v004 population context and swaps only the DIR-LAB case01/case08
representative Panel-A D-BSCP curves/TRE labels to the focused a=40000
screening runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import render_dirlab_regularity_matched_panelA_diagnostics as v004  # noqa: E402
import render_lungmask_panelA_population_diagnostics as base  # noqa: E402
import render_masked_panelA_both_datasets_diagnostics as v003  # noqa: E402


SCREEN_TOKEN = "${DIRLAB_DBSCP_A40000_CASE01_CASE08_SCREEN_RUN_ROOT}"
DEFAULT_SCREEN_RUN_ROOT = Path(
    "/database/inrorspline/runs/lungct_fdbe_cps4_alpha40000_case01_case08_screen_v1"
)
DEFAULT_OUTPUT_NAME = (
    "population_overlay_plus_lungct_cases_2x4_tre_labels_"
    "camera_ready_fig3_d_bscp_a40000_case01_case08_screen.png"
)
SUMMARY_NAME = "d_bscp_a40000_case01_case08_screen_summary.csv"
MANIFEST_NAME = "d_bscp_a40000_case01_case08_screen_manifest.json"
REPORT_NAME = "d_bscp_a40000_case01_case08_screen.md"
EXPECTED_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v004-root", type=Path, default=v004.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--screen-run-root", type=Path, default=DEFAULT_SCREEN_RUN_ROOT)
    parser.add_argument("--population-run-root", type=Path, default=base.DEFAULT_POPULATION_RUN_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=base.DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument("--lungmask-run-root", type=Path, default=base.DEFAULT_LUNGMASK_RUN_ROOT)
    parser.add_argument("--oasis-masked-run-root", type=Path, default=v003.DEFAULT_OASIS_MASKED_RUN_ROOT)
    parser.add_argument("--alpha-run-root", type=Path, default=v004.DEFAULT_ALPHA_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def screen_run_dir(screen_root: Path, case_key: str) -> Path:
    return (
        screen_root
        / "direct_cp_cps4_global_ncc_fd_be"
        / f"direct_cp_cps4_global_ncc_fd_be_{case_key}_e2500_lr0p001_a40000_seed1"
    )


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite value: {value}")
    return result


def load_screen_curve(screen_root: Path, case_key: str, original_curve: Any) -> tuple[Any, dict[str, Any]]:
    run_dir = screen_run_dir(screen_root, case_key)
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "global_ncc_diagnostic.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    summary = base.load_json(summary_path)
    curve = pd.read_csv(curve_path)
    training = summary.get("training", {})

    checks = {
        "status_completed": summary.get("status") == "completed",
        "method": summary.get("method") == "direct_cp_cps4_global_ncc_fd_be",
        "case": int(summary.get("case_id")) == int(case_key[-2:]),
        "epochs": int(training.get("epochs")) == 2500,
        "lr": np.isclose(float(training.get("lr")), 0.001),
        "alpha": np.isclose(float(training.get("alpha")), 40000.0),
        "cps": int(training.get("cps")) == 4,
        "seed": int(training.get("seed")) == 1,
        "diagnostic_interval": int(training.get("global_ncc_diagnostic_interval")) == 10,
    }
    failed = [name for name, ok in checks.items() if not bool(ok)]
    if failed:
        raise RuntimeError(f"{case_key}: screening run failed metadata checks: {failed}")

    epochs = curve["global_epoch"].to_numpy(float)
    values = curve["lung_mask_one_minus_ncc"].to_numpy(float)
    if len(curve) != 251 or not np.array_equal(epochs, EXPECTED_EPOCHS):
        raise RuntimeError(f"{case_key}: expected 251 points at epoch 1 and every 10 through 2500")
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise RuntimeError(f"{case_key}: nonfinite or nonpositive lung-mask 1-NCC values")
    tre = finite(summary.get("tre_snap_to_voxel", {}).get("tre_mean_mm"))
    sdlogj = finite(summary.get("regularity_fg_registration_crop", {}).get("log_jdet_std_fg"))
    endpoint = finite(values[-1])
    if not np.isclose(endpoint, 1.0 - float(curve["lung_mask_ncc_value"].iloc[-1]), rtol=2.0e-5, atol=2.0e-8):
        raise RuntimeError(f"{case_key}: final 1-NCC is inconsistent with final lung-mask NCC")

    replacement = replace(
        original_curve,
        epochs=epochs,
        values=values,
        source_path=curve_path,
        source_metric="lung_mask_one_minus_ncc",
        accepted_endpoint=endpoint,
    )
    row = {
        "case_key": case_key,
        "method_slug": "d_bscp",
        "screen_alpha": 40000.0,
        "screen_lr": 0.001,
        "screen_cps": 4,
        "screen_epochs": 2500,
        "screen_seed": 1,
        "screen_diagnostic_points": int(len(curve)),
        "screen_final_lung_mask_one_minus_ncc": endpoint,
        "screen_tre_snap_to_voxel_mean_mm": tre,
        "screen_foreground_sdlogj": sdlogj,
        "screen_elapsed_sec": finite(summary.get("elapsed_sec")),
        "screen_peak_mem_gb": finite(summary.get("peak_mem_gb")),
        "screen_curve_path": str(curve_path).replace(str(screen_root), SCREEN_TOKEN, 1),
        "screen_summary_path": str(summary_path).replace(str(screen_root), SCREEN_TOKEN, 1),
        "screen_curve_sha256": base.sha256(curve_path),
        "screen_summary_sha256": base.sha256(summary_path),
    }
    return replacement, row


def method_curve(curves: Sequence[Any], dataset: str, case_key: str, panel: str, method: str) -> Any:
    matches = [
        curve
        for curve in curves
        if curve.dataset == dataset
        and curve.case_key == case_key
        and curve.panel == panel
        and curve.method_slug == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one curve for {dataset}/{case_key}/{panel}/{method}, found {len(matches)}")
    return matches[0]


def screening_case_curves(curves: Sequence[Any], replacements: dict[str, Any], case_key: str, panel: str) -> list[Any]:
    case_curves = [
        curve
        for curve in curves
        if curve.dataset == "dirlab" and curve.case_key == case_key and curve.panel == panel
    ]
    if panel == "A":
        case_curves = [replacements[case_key] if curve.method_slug == "d_bscp" else curve for curve in case_curves]
    return case_curves


def apply_fig3_camera_ready_axis_style(axis: plt.Axes, dataset: str, panel: str) -> None:
    v004.apply_fig3_camera_ready_axis_style(axis, dataset, panel)


def render_screening_fig3(
    curves: Sequence[Any],
    replacements: dict[str, Any],
    tre_labels: dict[str, dict[str, float]],
    output_path: Path,
    dpi: int,
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
                case_curves = screening_case_curves(curves, replacements, key, panel)
                base.draw_single_case(axis, case_curves, panel)
                dataset = "dirlab"
                if panel == "A":
                    v004.add_tre_inside_labels(axis, case_curves, key, tre_labels, fontsize=8.2)
            apply_fig3_camera_ready_axis_style(axis, dataset, panel)
            axis.set_title(title, fontsize=v004.FIG3_CAMERA_READY_FONTS["subtitle"], weight="bold")
            axis.set_ylabel(
                "Similarity (1-NCC)" if panel == "A" else "rRMSE (%)",
                fontsize=v004.FIG3_CAMERA_READY_FONTS["ylabel"],
            )
    fig.legend(
        handles=v004.legend_handles(3.0),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=False,
        fontsize=v004.FIG3_CAMERA_READY_FONTS["legend"],
    )
    fig.text(
        0.075,
        0.968,
        "A. Similarity metric during unsupervised registration optimization",
        ha="left",
        va="top",
        fontsize=v004.FIG3_CAMERA_READY_FONTS["panel_title"],
        weight="bold",
    )
    fig.text(
        0.075,
        0.505,
        "B. Target field fitting error during supervised fitting",
        ha="left",
        va="top",
        fontsize=v004.FIG3_CAMERA_READY_FONTS["panel_title"],
        weight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.895, bottom=0.145, hspace=0.78, wspace=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.v004_root.resolve()
    index_path = output_root / "camera_ready_source_index_dirlab_regularity_matched_panelA.csv"
    index = pd.read_csv(index_path)
    v004.validate_v004_index(index)

    roots = {
        base.POPULATION_TOKEN: args.population_run_root.resolve(),
        base.REFRESH_TOKEN: args.refresh_run_root.resolve(),
        base.LUNGMASK_TOKEN: args.lungmask_run_root.resolve(),
        v003.OASIS_MASKED_TOKEN: args.oasis_masked_run_root.resolve(),
        base.DATA_TOKEN: args.data_root.resolve(),
        v004.ALPHA_TOKEN: args.alpha_run_root.resolve(),
    }
    curves, _ = v004.load_and_validate_curves(index, roots)
    baseline_tre_labels = v004.load_tre_labels(index, roots)
    screen_tre_labels = json.loads(json.dumps(baseline_tre_labels))

    replacements: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for case_key in ["case01", "case08"]:
        original = method_curve(curves, "dirlab", case_key, "A", "d_bscp")
        replacement, row = load_screen_curve(args.screen_run_root.resolve(), case_key, original)
        row["baseline_alpha"] = 50000.0
        row["baseline_final_lung_mask_one_minus_ncc"] = float(original.values[-1])
        row["baseline_tre_snap_to_voxel_mean_mm"] = float(baseline_tre_labels[case_key]["d_bscp"])
        row["delta_final_lung_mask_one_minus_ncc"] = (
            row["screen_final_lung_mask_one_minus_ncc"] - row["baseline_final_lung_mask_one_minus_ncc"]
        )
        row["delta_tre_snap_to_voxel_mean_mm"] = (
            row["screen_tre_snap_to_voxel_mean_mm"] - row["baseline_tre_snap_to_voxel_mean_mm"]
        )
        replacements[case_key] = replacement
        screen_tre_labels[case_key]["d_bscp"] = row["screen_tre_snap_to_voxel_mean_mm"]
        rows.append(row)

    output_path = output_root / args.output_name
    render_screening_fig3(curves, replacements, screen_tre_labels, output_path, args.dpi)
    summary_path = output_root / SUMMARY_NAME
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    manifest = {
        "created_at_utc": utc_text(),
        "status": "passed",
        "artifact_type": "screening_only_not_camera_ready_source_index",
        "source_index": "camera_ready_source_index_dirlab_regularity_matched_panelA.csv",
        "screening_change": "Only DIR-LAB representative case01/case08 Panel-A D-BSCP curves and TRE labels use a=40000.",
        "population_columns": "Unchanged v004 population context; not an all-10 a=40000 population.",
        "baseline_alpha": 50000.0,
        "screen_alpha": 40000.0,
        "run_root_token": SCREEN_TOKEN,
        "axis_limits": v004.V004_AXIS_LIMITS,
        "figure": output_path.name,
        "summary_csv": summary_path.name,
        "figure_sha256": base.sha256(output_path),
        "summary_sha256": base.sha256(summary_path),
        "screening_rows": rows,
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = output_root / REPORT_NAME
    report_path.write_text(
        "\n".join(
            [
                "# D-BSCP a=40000 Case01/Case08 Screening Figure",
                "",
                "This is a focused screening artifact, not a replacement for the v004 camera-ready source index.",
                "",
                f"- Figure: `{output_path.name}`",
                f"- Summary CSV: `{summary_path.name}`",
                f"- Manifest: `{manifest_path.name}`",
                "- Changed only: DIR-LAB 4DCT representative case01/case08 Panel-A D-BSCP curves and TRE labels.",
                "- Unchanged: OASIS, DIR-LAB population context, all Panel B curves, all non-D-BSCP methods.",
                "- Baseline comparison uses the current v004 D-BSCP `a=50000` rows.",
                "",
                "| Case | Baseline a=50000 TRE (mm) | Screen a=40000 TRE (mm) | Delta TRE (mm) | Baseline final 1-NCC | Screen final 1-NCC | Delta 1-NCC | Screen SDlogJ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    (
                        f"| {row['case_key']} | {row['baseline_tre_snap_to_voxel_mean_mm']:.3f} | "
                        f"{row['screen_tre_snap_to_voxel_mean_mm']:.3f} | {row['delta_tre_snap_to_voxel_mean_mm']:.3f} | "
                        f"{row['baseline_final_lung_mask_one_minus_ncc']:.5f} | {row['screen_final_lung_mask_one_minus_ncc']:.5f} | "
                        f"{row['delta_final_lung_mask_one_minus_ncc']:.5f} | {row['screen_foreground_sdlogj']:.3f} |"
                    )
                    for row in rows
                ],
                "",
                "Interpretation guardrail: the population columns are retained as v004 context. Because only case01 and case08 were rerun at `a=40000`, this figure should be used for alpha screening only.",
            ]
        )
        + "\n"
    )
    print(json.dumps({"status": "passed", "figure": str(output_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
