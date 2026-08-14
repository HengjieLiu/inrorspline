#!/usr/bin/env python
"""Render curve-style before/after LungCT metric-domain diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from plot_case_diagnostics import METHOD_ORDER, METHOD_STYLE


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/2026-08-14_dirlab_case01_case08_lungmask_diagnostics"
DEFAULT_SOURCE_INDEX = (
    REPO_ROOT
    / "experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched"
    / "camera_ready_source_index.csv"
)
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1")
DEFAULT_DATA_ROOT = Path("/database/inrorspline")
DEFAULT_REFRESH_RUN_ROOT = Path(
    "/database/inrorspline/runs/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
)
TARGET_CASES = ("case01", "case08")
AXIS_LIMITS = {"A": (1.0e-3, 5.0e-1), "B": (1.0e-1, 1.0e3)}
STAGE_BOUNDARIES = [400, 800, 1600]
EXPECTED_EPOCHS = np.array([1, *range(10, 2501, 10)], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_record(path: Path, output_dir: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": os.path.relpath(path, output_dir),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def resolve_path(value: Any, *, data_root: Path, refresh_run_root: Path) -> Path:
    text = str(value or "")
    replacements = {
        "${DATA_ROOT}": str(data_root),
        "${DIRLAB_REFRESH_RUN_ROOT}": str(refresh_run_root),
        "${POPULATION_RUN_ROOT}": str(data_root / "runs/fit_modes_population_n10_v1"),
        "${RUN_ROOT}": str(refresh_run_root),
        "${REPO_ROOT}": str(REPO_ROOT),
    }
    for token, root in replacements.items():
        text = text.replace(token, root)
    return Path(text)


def portable_path(path: Path, *, run_root: Path, data_root: Path, refresh_run_root: Path) -> str:
    text = str(path)
    for root, token in [
        (run_root, "${LUNGMASK_CURVE_RUN_ROOT}"),
        (refresh_run_root, "${DIRLAB_REFRESH_RUN_ROOT}"),
        (data_root, "${DATA_ROOT}"),
        (REPO_ROOT, "${REPO_ROOT}"),
    ]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return path.name


def run_name(case_key: str, method_slug: str) -> tuple[str, str]:
    case_id = int(case_key.replace("case", ""))
    mapping = {
        "idir": (
            "idir_be_default_lungmask",
            f"lungmask_curves_case{case_id:02d}_panelA_idir_e2500_lr1em4_seed1_diag10",
        ),
        "inr_bscp": (
            "sinr_cps4_global_ncc_fd_be",
            f"lungmask_curves_case{case_id:02d}_panelA_inr_bscp_e2500_lr1em4_seed1_diag10",
        ),
        "d_bscp": (
            "direct_cp_cps4_global_ncc_fd_be",
            f"lungmask_curves_case{case_id:02d}_panelA_d_bscp_e2500_lr1em3_seed1_diag10",
        ),
        "mr_d_bscp": (
            "direct_cp_32_16_8_4_global_ncc_fd_be",
            f"lungmask_curves_case{case_id:02d}_panelA_mr_d_bscp_sched32-16-8-4_ds8-4-2-1_e400-400-800-900_a10000_lr0p001_seed1_diag10",
        ),
    }
    return mapping[method_slug]


def epoch_column(frame: pd.DataFrame) -> str:
    if "global_epoch" in frame and frame["global_epoch"].notna().any():
        return "global_epoch"
    if "epoch" in frame and frame["epoch"].notna().any():
        return "epoch"
    raise KeyError("No epoch/global_epoch column")


def panel_b_values(frame: pd.DataFrame, target_power: float) -> tuple[np.ndarray, np.ndarray, str]:
    column = next(
        (
            name
            for name in [
                "finest_field_mse_objective_normalized_grid",
                "finest_data_loss",
                "field_mse_objective_normalized_grid",
                "data_loss",
                "loss",
            ]
            if name in frame and frame[name].notna().any()
        ),
        None,
    )
    if column is None:
        raise KeyError("No Panel-B MSE column")
    epochs = frame[epoch_column(frame)].to_numpy(float)
    values = 100.0 * np.sqrt(frame[column].to_numpy(float) / (float(target_power) + 1.0e-12))
    keep = np.isfinite(epochs) & np.isfinite(values) & (values > 0)
    return epochs[keep], values[keep], column


def load_panel_a_curve(run_root: Path, case_key: str, method_slug: str) -> dict[str, Any]:
    impl, name = run_name(case_key, method_slug)
    curve_path = run_root / impl / name / "global_ncc_diagnostic.csv"
    summary_path = run_root / impl / name / "summary.json"
    if not curve_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing Panel-A lung-mask rerun for {case_key}/{method_slug}")
    frame = pd.read_csv(curve_path)
    epochs = frame[epoch_column(frame)].to_numpy(float)
    full = 1.0 - frame["global_ncc_value"].to_numpy(float)
    lung = frame["lung_mask_one_minus_ncc"].to_numpy(float)
    keep = np.isfinite(epochs) & np.isfinite(full) & np.isfinite(lung) & (full > 0) & (lung > 0)
    epochs, full, lung = epochs[keep], full[keep], lung[keep]
    order = np.argsort(epochs, kind="stable")
    return {
        "epochs": epochs[order],
        "full_crop": full[order],
        "lung_mask": lung[order],
        "curve_path": curve_path,
        "summary_path": summary_path,
        "point_count": int(len(epochs)),
    }


def validate_expected_epochs(epochs: np.ndarray) -> bool:
    return len(epochs) == len(EXPECTED_EPOCHS) and np.array_equal(epochs.astype(int), EXPECTED_EPOCHS.astype(int))


def render_panel_a_before_after_figure(
    output_path: Path,
    panel_a: dict[tuple[str, str], dict[str, Any]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), sharex=False)
    for col, case_key in enumerate(TARGET_CASES):
        ax = axes[0, col]
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            curve = panel_a[(case_key, method)]
            ax.plot(
                curve["epochs"],
                curve["full_crop"],
                color=style["color"],
                linestyle="-",
                linewidth=2.35,
                alpha=0.96,
            )
        ax.set_title(f"DIR-LAB {case_key}: before fix", fontsize=13.0, weight="bold")
        ax.set_ylabel("Panel A full-crop 1−NCC", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(*AXIS_LIMITS["A"])
        ax.set_xlim(0, 2500)
        ax.set_xticks(np.arange(0, 2501, 500))
        for boundary in STAGE_BOUNDARIES:
            ax.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)

        ax = axes[1, col]
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            curve = panel_a[(case_key, method)]
            ax.plot(
                curve["epochs"],
                curve["lung_mask"],
                color=style["color"],
                linestyle="-",
                linewidth=2.35,
                alpha=0.96,
            )
        ax.set_title(f"DIR-LAB {case_key}: after fix", fontsize=13.0, weight="bold")
        ax.set_ylabel("Panel A lung-mask 1−NCC", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(*AXIS_LIMITS["A"])
        ax.set_xlim(0, 2500)
        ax.set_xticks(np.arange(0, 2501, 500))
        for boundary in STAGE_BOUNDARIES:
            ax.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#555555")

    method_handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], linewidth=2.4, label=METHOD_STYLE[m]["label"])
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11.0,
        handlelength=2.5,
        columnspacing=1.7,
    )
    fig.suptitle("Panel A metric-domain before/after curves for DIR-LAB case01 and case08", fontsize=15.5, weight="bold", y=0.992)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.91, bottom=0.15, wspace=0.24, hspace=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_context_overlap_figure(
    output_path: Path,
    panel_a: dict[tuple[str, str], dict[str, Any]],
    panel_b: dict[tuple[str, str], dict[str, Any]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), sharex=False)
    for col, case_key in enumerate(TARGET_CASES):
        ax = axes[0, col]
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            curve = panel_a[(case_key, method)]
            ax.plot(curve["epochs"], curve["full_crop"], color=style["color"], linestyle=(0, (4, 2.2)), linewidth=1.9, alpha=0.48)
            ax.plot(curve["epochs"], curve["lung_mask"], color=style["color"], linestyle="-", linewidth=2.35, alpha=0.96)
        ax.set_title(f"DIR-LAB {case_key}: registration", fontsize=13.0, weight="bold")
        ax.set_ylabel("Panel A 1−NCC", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(*AXIS_LIMITS["A"])
        ax.set_xlim(0, 2500)
        ax.set_xticks(np.arange(0, 2501, 500))
        for boundary in STAGE_BOUNDARIES:
            ax.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)

        ax = axes[1, col]
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            curve = panel_b[(case_key, method)]
            ax.plot(curve["epochs"], curve["values"], color=style["color"], linestyle=(0, (4, 2.2)), linewidth=1.8, alpha=0.36)
            ax.plot(curve["epochs"], curve["values"], color=style["color"], linestyle="-", linewidth=2.2, alpha=0.94)
        ax.set_title(f"DIR-LAB {case_key}: supervised fitting", fontsize=13.0, weight="bold")
        ax.set_ylabel("Panel B rRMSE (%)", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(*AXIS_LIMITS["B"])
        ax.set_xlim(0, 2500)
        ax.set_xticks(np.arange(0, 2501, 500))
        for boundary in STAGE_BOUNDARIES:
            ax.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)
        ax.text(0.02, 0.95, "before/after overlap: already fixed-mask", transform=ax.transAxes, ha="left", va="top", fontsize=9.6, bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86})

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#555555")

    handles = [
        *[Line2D([0], [0], color=METHOD_STYLE[m]["color"], linewidth=2.4, label=METHOD_STYLE[m]["label"]) for m in METHOD_ORDER],
        Line2D([0], [0], color="#333333", linestyle=(0, (4, 2.2)), linewidth=2.1, alpha=0.55, label="before: full-crop Panel A"),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.4, alpha=0.95, label="after: fixed T00 lung mask"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10.2, handlelength=2.2, columnspacing=1.2)
    fig.suptitle("Metric-domain before/after curves for DIR-LAB case01 and case08", fontsize=15.5, weight="bold", y=0.992)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.91, bottom=0.15, wspace=0.24, hspace=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.source_index)
    selected = index[
        index["dataset"].eq("dirlab")
        & index["case_key"].isin(TARGET_CASES)
        & index["method_slug"].isin(METHOD_ORDER)
    ].copy()
    if len(selected) != 16:
        raise RuntimeError(f"Expected 16 selected source rows, found {len(selected)}")

    panel_a: dict[tuple[str, str], dict[str, Any]] = {}
    panel_b: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    validation = {
        "panel_a_curves": 0,
        "panel_a_expected_epoch_curves": 0,
        "panel_a_finite_dual_domain_curves": 0,
        "panel_b_curves": 0,
        "panel_b_endpoint_matches_source": 0,
        "mrdbscp_panelA_updated_schedule_records": 0,
        "max_panelA_full_crop_endpoint_abs_diff_vs_source": 0.0,
    }

    for case_key in TARGET_CASES:
        for method in METHOD_ORDER:
            key = (case_key, method)
            row_a = selected[selected["case_key"].eq(case_key) & selected["panel"].eq("A") & selected["method_slug"].eq(method)].iloc[0]
            curve_a = load_panel_a_curve(args.run_root, case_key, method)
            panel_a[key] = curve_a
            endpoint_full = float(curve_a["full_crop"][-1])
            endpoint_lung = float(curve_a["lung_mask"][-1])
            source_endpoint = float(row_a["endpoint_value"])
            endpoint_diff = abs(endpoint_full - source_endpoint)
            validation["max_panelA_full_crop_endpoint_abs_diff_vs_source"] = max(
                float(validation["max_panelA_full_crop_endpoint_abs_diff_vs_source"]), endpoint_diff
            )
            validation["panel_a_curves"] += 1
            validation["panel_a_expected_epoch_curves"] += int(validate_expected_epochs(curve_a["epochs"]))
            validation["panel_a_finite_dual_domain_curves"] += int(
                np.isfinite(curve_a["full_crop"]).all() and np.isfinite(curve_a["lung_mask"]).all()
            )
            validation["mrdbscp_panelA_updated_schedule_records"] += int(
                method == "mr_d_bscp" and str(row_a.get("stage_epochs", "")) == "400,400,800,900"
            )
            rows.append(
                {
                    "case_key": case_key,
                    "panel": "A",
                    "method_slug": method,
                    "display_method": METHOD_STYLE[method]["label"],
                    "before_metric": "full_crop_one_minus_ncc",
                    "after_metric": "fixed_t00_lung_mask_one_minus_ncc",
                    "before_endpoint": endpoint_full,
                    "after_endpoint": endpoint_lung,
                    "accepted_source_endpoint": source_endpoint,
                    "before_endpoint_abs_diff_vs_source": endpoint_diff,
                    "point_count": len(curve_a["epochs"]),
                    "expected_epochs": validate_expected_epochs(curve_a["epochs"]),
                    "curve_path": portable_path(
                        curve_a["curve_path"],
                        run_root=args.run_root,
                        data_root=args.data_root,
                        refresh_run_root=args.refresh_run_root,
                    ),
                }
            )

            row_b = selected[selected["case_key"].eq(case_key) & selected["panel"].eq("B") & selected["method_slug"].eq(method)].iloc[0]
            curve_path_b = resolve_path(row_b["source_path"], data_root=args.data_root, refresh_run_root=args.refresh_run_root)
            frame_b = pd.read_csv(curve_path_b)
            target_power = finite(row_b.get("field_target_power_masked_normalized_grid"))
            if target_power is None or target_power <= 0:
                raise RuntimeError(f"Invalid Panel-B target power for {case_key}/{method}")
            epochs_b, values_b, source_metric_b = panel_b_values(frame_b, target_power)
            panel_b[key] = {"epochs": epochs_b, "values": values_b, "curve_path": curve_path_b, "source_metric": source_metric_b}
            endpoint_b = float(values_b[-1])
            source_endpoint_b = float(row_b["endpoint_value"])
            validation["panel_b_curves"] += 1
            validation["panel_b_endpoint_matches_source"] += int(abs(endpoint_b - source_endpoint_b) < 1.0e-3)
            rows.append(
                {
                    "case_key": case_key,
                    "panel": "B",
                    "method_slug": method,
                    "display_method": METHOD_STYLE[method]["label"],
                    "before_metric": "fixed_t00_lung_mask_rrmse_percent",
                    "after_metric": "fixed_t00_lung_mask_rrmse_percent",
                    "before_endpoint": source_endpoint_b,
                    "after_endpoint": source_endpoint_b,
                    "raw_curve_endpoint": endpoint_b,
                    "raw_curve_endpoint_abs_diff_vs_source": abs(endpoint_b - source_endpoint_b),
                    "point_count": len(epochs_b),
                    "curve_path": portable_path(
                        curve_path_b,
                        run_root=args.run_root,
                        data_root=args.data_root,
                        refresh_run_root=args.refresh_run_root,
                    ),
                }
            )

    figure_path = args.output_dir / "lungmask_before_after_curves_2x2.png"
    render_panel_a_before_after_figure(figure_path, panel_a, args.dpi)
    context_figure_path = args.output_dir / "lungmask_before_after_curves_with_panelB_context_2x2.png"
    render_context_overlap_figure(context_figure_path, panel_a, panel_b, args.dpi)
    summary_path = args.output_dir / "lungmask_before_after_curves_summary.csv"
    write_csv(summary_path, rows)
    validation["pass"] = bool(
        validation["panel_a_curves"] == 8
        and validation["panel_a_expected_epoch_curves"] == 8
        and validation["panel_a_finite_dual_domain_curves"] == 8
        and validation["panel_b_curves"] == 8
        and validation["panel_b_endpoint_matches_source"] == 8
        and validation["mrdbscp_panelA_updated_schedule_records"] == 2
        and float(validation["max_panelA_full_crop_endpoint_abs_diff_vs_source"]) < 1.0e-3
    )
    manifest = {
        "generated_at_utc": utc_text(),
        "script": "scripts/experiments/fit_modes_population/render_dirlab_lungmask_before_after_curves.py",
        "source_index": str(args.source_index),
        "run_root_token": "${LUNGMASK_CURVE_RUN_ROOT}",
        "figure": image_record(figure_path, args.output_dir),
        "context_figure": image_record(context_figure_path, args.output_dir),
        "summary_csv": summary_path.name,
        "validation": validation,
        "axis_limits": AXIS_LIMITS,
        "stage_boundaries": STAGE_BOUNDARIES,
        "panel_a_before": "full-crop 1-NCC from the new dual-diagnostic reruns",
        "panel_a_after": "fixed T00 lung-mask 1-NCC from the same reruns",
        "main_figure_layout": "2x2: columns case01/case08; top row before-fix Panel A full-crop; bottom row after-fix Panel A fixed-lung-mask",
        "context_figure_layout": "2x2 overlap/context figure with Panel A overlaid before/after and Panel B unchanged",
    }
    manifest_path = args.output_dir / "lungmask_before_after_curves_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    report_path = args.output_dir / "lungmask_before_after_curves.md"
    report_path.write_text(
        "\n".join(
            [
                "# DIR-LAB case01/case08 before/after curve diagnostics",
                "",
                f"Generated: `{manifest['generated_at_utc']}`",
                "",
                f"![Before/after curves]({figure_path.name})",
                "",
                "Main figure layout: columns are DIR-LAB case01 and case08; the top row is before-fix Panel A full-crop `1−NCC`; the bottom row is after-fix Panel A fixed T00 lung-mask `1−NCC`.",
                "",
                f"A secondary context version that includes Panel B is kept as [`{context_figure_path.name}`]({context_figure_path.name}).",
                "",
                "## Validation",
                "",
                *[f"- `{key}`: `{value}`" for key, value in validation.items()],
                "",
                f"Curve summary: [`{summary_path.name}`]({summary_path.name})",
                f"Manifest: [`{manifest_path.name}`]({manifest_path.name})",
            ]
        )
        + "\n"
    )
    print(json.dumps({"figure": str(figure_path), "validation": validation}, indent=2, sort_keys=True))
    if not validation["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
