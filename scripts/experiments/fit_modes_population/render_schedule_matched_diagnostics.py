#!/usr/bin/env python
"""Render schedule-matched OASIS10 and DIR-LAB10 fit-modes diagnostics.

The historical population index remains immutable.  This renderer creates a
versioned 160-record source index by replacing only the ten DIR-LAB Panel-A
MR-D-BSCP rows with the accepted 400/400/800/900 diagnostic refresh.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
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
BASE_EXPERIMENT = REPO_ROOT / "experiments/2026-08-02_fit_modes_population_n10"
REFRESH_EXPERIMENT = (
    REPO_ROOT / "experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched"
)
DEFAULT_POPULATION_RUN_ROOT = Path(
    os.environ.get(
        "POPULATION_RUN_ROOT",
        "/workspace/scratch/runs/fit_modes_population_n10_v1",
    )
)
DEFAULT_REFRESH_RUN_ROOT = Path(
    os.environ.get(
        "DIRLAB_REFRESH_RUN_ROOT",
        "/workspace/scratch/runs/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1",
    )
)
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))

POPULATION_TOKEN = "${POPULATION_RUN_ROOT}"
REFRESH_TOKEN = "${DIRLAB_REFRESH_RUN_ROOT}"
METHOD = "direct_cp_32_16_8_4_global_ncc_fd_be"
EXPECTED_CASES = list(range(1, 11))
EXPECTED_REFRESH_EPOCHS = [1, *range(10, 2501, 10)]
AXIS_LIMITS = {"A": (1.0e-3, 5.0e-1), "B": (1.0e-1, 1.0e3)}
STAGE_BOUNDARIES = {
    ("oasis", "A"): [400, 1200],
    ("oasis", "B"): [400, 1200],
    ("dirlab", "A"): [400, 800, 1600],
    ("dirlab", "B"): [400, 800, 1600],
}
PANEL_Y_LABEL = {"A": "A. Registration: 1−NCC", "B": "B. Field fitting: rRMSE (%)"}
DATASET_LABEL = {"oasis": "OASIS N=10", "dirlab": "DIR-LAB N=10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-index",
        type=Path,
        default=BASE_EXPERIMENT / "optimization_index.csv",
    )
    parser.add_argument(
        "--replacement-index",
        type=Path,
        default=REFRESH_EXPERIMENT / "replacement_index.csv",
    )
    parser.add_argument("--population-run-root", type=Path, default=DEFAULT_POPULATION_RUN_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def git_output(*args: str) -> str:
    command = ["git", *args]
    repo_stat = REPO_ROOT.stat()
    if os.geteuid() == 0 and repo_stat.st_uid != 0:
        command = [
            "setpriv",
            f"--reuid={repo_stat.st_uid}",
            f"--regid={repo_stat.st_gid}",
            "--clear-groups",
            *command,
        ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def refresh_portable_path(value: Any) -> str:
    text = str(value)
    if not text.startswith("${RUN_ROOT}/"):
        raise ValueError(f"Replacement path is not portable: {text}")
    return text.replace("${RUN_ROOT}", REFRESH_TOKEN, 1)


def population_portable_path(value: Any) -> str:
    text = str(value)
    if text.startswith("${RUN_ROOT}/"):
        return text.replace("${RUN_ROOT}", POPULATION_TOKEN, 1)
    if text.startswith("${DATA_ROOT}/"):
        return text
    raise ValueError(f"Population path is not portable: {text}")


def resolve_refresh_path(value: Any, refresh_root: Path) -> Path:
    text = str(value)
    if not text.startswith(REFRESH_TOKEN + "/"):
        raise ValueError(f"Expected {REFRESH_TOKEN} path, got {text}")
    return Path(text.replace(REFRESH_TOKEN, str(refresh_root), 1))


def canonical_rows_hash(frame: pd.DataFrame) -> str:
    return sha256_text(frame.to_json(orient="records", double_precision=15))


def validate_replacement_metadata(replacement: pd.DataFrame) -> None:
    if len(replacement) != 10 or replacement["case_id"].astype(int).tolist() != EXPECTED_CASES:
        raise RuntimeError("Replacement index must contain ordered DIR-LAB cases 01-10")
    checks = {
        "dataset": replacement["dataset"].eq("dirlab").all(),
        "panel": replacement["panel"].eq("A").all(),
        "method": replacement["method_slug"].eq("mr_d_bscp").all(),
        "implementation": replacement["implementation_id"].eq(METHOD).all(),
        "status": replacement["status"].eq("completed").all(),
        "lr": np.allclose(replacement["lr"].astype(float), 0.001),
        "alpha": np.allclose(replacement["alpha"].astype(float), 10000.0),
        "control_spacing": replacement["stage_full_cps"].astype(str).eq("32,16,8,4").all(),
        "downsampling": replacement["stage_downsample"].astype(str).eq("8,4,2,1").all(),
        "schedule": replacement["stage_epochs"].astype(str).eq("400,400,800,900").all(),
        "epochs": replacement["epochs"].astype(int).eq(2500).all(),
        "seed": replacement["seed"].astype(int).eq(1).all(),
        "diagnostic_interval": replacement["diagnostic_interval"].astype(int).eq(10).all(),
        "diagnostic_rows": replacement["diagnostic_rows"].astype(int).eq(251).all(),
        "no_validation_errors": replacement["validation_errors"].fillna("").eq("").all(),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError("Replacement metadata failed: " + ", ".join(failed))


def build_source_index(
    base: pd.DataFrame, replacement: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str]:
    if len(base) != 160 or base["job_id"].duplicated().any():
        raise RuntimeError("Historical population index must contain 160 unique jobs")
    target = (
        base["dataset"].eq("dirlab")
        & base["panel"].eq("A")
        & base["method_slug"].eq("mr_d_bscp")
    )
    if int(target.sum()) != 10:
        raise RuntimeError(f"Expected ten replaceable rows, found {int(target.sum())}")
    before_unchanged = base.loc[~target].copy()
    merged = base.copy()
    audit: list[dict[str, Any]] = []
    for _, new in replacement.iterrows():
        case_id = int(new["case_id"])
        matches = target & merged["case_id"].fillna(-1).astype(int).eq(case_id)
        if int(matches.sum()) != 1:
            raise RuntimeError(f"case{case_id:02d}: expected one historical target row")
        row_index = merged.index[matches][0]
        old = merged.loc[row_index].copy()
        updates = {
            "implementation_id": new["implementation_id"],
            "status": "completed",
            "source_kind": new["source_kind"],
            "source_path": refresh_portable_path(new["source_path"]),
            "summary_path": refresh_portable_path(new["summary_path"]),
            "endpoint_name": new["endpoint_name"],
            "endpoint_value": float(new["endpoint_value"]),
            "diagnostic_ncc": float(new["diagnostic_ncc"]),
            "runtime_sec": float(new["runtime_sec"]),
            "peak_memory_gb": float(new["peak_memory_gb"]),
            "endpoint_source": "schedule_refresh:global_ncc_value",
            "lr": float(new["lr"]),
            "alpha": float(new["alpha"]),
            "cps": str(new["stage_full_cps"]),
            "stage_downsample": str(new["stage_downsample"]),
            "stage_epochs": str(new["stage_epochs"]),
            "epochs": int(new["epochs"]),
            "seed": int(new["seed"]),
            "diagnostic_interval": int(new["diagnostic_interval"]),
            "acceptance_checks": (
                "schedule_refresh_accepted=true;curve_rows=251;endpoint_matches=true;"
                "schedule=400,400,800,900"
            ),
        }
        for field, value in updates.items():
            merged.at[row_index, field] = value
        audit.append(
            {
                "case_id": case_id,
                "job_id": old["job_id"],
                "old_source_path": population_portable_path(old["source_path"]),
                "new_source_path": updates["source_path"],
                "old_stage_epochs": old["stage_epochs"],
                "new_stage_epochs": updates["stage_epochs"],
                "old_epochs": int(old["epochs"]),
                "new_epochs": updates["epochs"],
                "old_endpoint": float(old["endpoint_value"]),
                "new_endpoint": updates["endpoint_value"],
            }
        )
    after_unchanged = merged.loc[~target].copy()
    pd.testing.assert_frame_equal(before_unchanged, after_unchanged, check_exact=True)
    before_hash = canonical_rows_hash(before_unchanged)
    after_hash = canonical_rows_hash(after_unchanged)
    if before_hash != after_hash:
        raise RuntimeError("The 150 nonreplacement records changed")
    if len(merged) != 160 or merged["job_id"].duplicated().any():
        raise RuntimeError("Merged source index is not 160 unique records")
    for column in ("source_path", "summary_path"):
        merged.loc[~target, column] = merged.loc[~target, column].map(
            population_portable_path
        )
    return merged, audit, before_hash, after_hash


def validate_refresh_sources(
    replacement: pd.DataFrame, refresh_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in replacement.iterrows():
        case_id = int(row["case_id"])
        curve_path = Path(
            str(row["source_path"]).replace("${RUN_ROOT}", str(refresh_root), 1)
        )
        summary_path = Path(
            str(row["summary_path"]).replace("${RUN_ROOT}", str(refresh_root), 1)
        )
        if not curve_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"case{case_id:02d}: missing refresh source")
        if sha256(curve_path) != str(row["curve_sha256"]):
            raise RuntimeError(f"case{case_id:02d}: curve checksum mismatch")
        curve = pd.read_csv(curve_path)
        summary = load_json(summary_path)
        training = summary.get("training", {})
        epochs = curve["global_epoch"].astype(int).tolist()
        final = curve.iloc[-1]
        checks = {
            "curve_rows": len(curve) == 251,
            "epoch_set": epochs == EXPECTED_REFRESH_EPOCHS,
            "curve_finite": np.isfinite(
                curve[["global_ncc_loss", "global_ncc_value"]].to_numpy(float)
            ).all(),
            "domain": curve["eval_domain"].astype(str).eq("full_1mm_crop").all(),
            "final_identity": (
                int(final["global_epoch"]) == 2500
                and int(final["stage_idx"]) == 3
                and int(final["full_cps"]) == 4
                and int(final["downsample"]) == 1
            ),
            "summary_status": summary.get("status") == "completed",
            "summary_method": summary.get("method") == METHOD,
            "summary_case": int(summary.get("case_id", -1)) == case_id,
            "schedule": training.get("stage_epochs") == [400, 400, 800, 900],
            "control_spacing": training.get("stage_full_cps") == [32, 16, 8, 4],
            "downsampling": training.get("stage_downsample") == [8, 4, 2, 1],
            "lr": training.get("stage_lr") == [0.001, 0.001, 0.001, 0.001],
            "alpha": abs(float(training.get("alpha", math.nan)) - 10000.0) < 1e-8,
            "seed": int(training.get("seed", -1)) == 1,
            "diagnostic": int(training.get("global_ncc_diagnostic_interval", -1)) == 10,
            "diagnostic_nontraining": training.get("global_ncc_diagnostic_affects_training")
            is False,
            "mask_rule": summary.get("mask_rule") == "fixed_t00_lung_mask_metric_only",
            "direction": str(summary.get("direction", "")).replace(" ", "") == "T50->T00",
        }
        failed = [name for name, passed in checks.items() if not bool(passed)]
        if failed:
            raise RuntimeError(f"case{case_id:02d}: " + ", ".join(failed))
        records.append(
            {
                "case_id": case_id,
                "curve_rows": len(curve),
                "epoch_min": int(curve["global_epoch"].min()),
                "epoch_max": int(curve["global_epoch"].max()),
                "final_ncc": float(final["global_ncc_value"]),
                "final_one_minus_ncc": 1.0 - float(final["global_ncc_value"]),
                "curve_sha256": sha256(curve_path),
                "all_checks_pass": True,
            }
        )
    return records


def load_merged_curve(
    row: pd.Series,
    population_root: Path,
    refresh_root: Path,
    data_root: Path,
) -> DiagnosticCurve:
    if str(row["source_path"]).startswith(REFRESH_TOKEN + "/"):
        adjusted = row.copy()
        adjusted["source_path"] = str(row["source_path"]).replace(
            REFRESH_TOKEN, "${RUN_ROOT}", 1
        )
        return load_curve(adjusted, refresh_root, data_root)
    if str(row["source_path"]).startswith(POPULATION_TOKEN + "/"):
        adjusted = row.copy()
        adjusted["source_path"] = str(row["source_path"]).replace(
            POPULATION_TOKEN, "${RUN_ROOT}", 1
        )
        return load_curve(adjusted, population_root, data_root)
    if str(row["source_path"]).startswith("${DATA_ROOT}/"):
        return load_curve(row, population_root, data_root)
    raise ValueError(f"Unrecognized portable source path: {row['source_path']}")


def method_curve(
    curves: Sequence[DiagnosticCurve], panel: str, method: str
) -> DiagnosticCurve:
    matches = [
        curve for curve in curves if curve.panel == panel and curve.method_slug == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Panel {panel}/{method}: expected one curve, found {len(matches)}")
    return matches[0]


def style_axis(axis: plt.Axes, dataset: str, panel: str, show_xlabel: bool) -> None:
    for boundary in STAGE_BOUNDARIES[(dataset, panel)]:
        axis.axvline(
            boundary,
            color="#777777",
            linewidth=0.7,
            linestyle=(0, (2, 2)),
            alpha=0.55,
            zorder=1,
        )
    axis.set_yscale("log")
    axis.set_ylim(*AXIS_LIMITS[panel])
    axis.set_xlim(0, 2500)
    axis.set_xticks(np.arange(0, 2501, 500))
    axis.grid(True, which="both", color="#D5DCE3", linewidth=0.6, alpha=0.75)
    axis.tick_params(axis="both", labelsize=7.4, labelbottom=True, labelleft=True)
    axis.set_xlabel("Epoch" if show_xlabel else "", fontsize=8.3)


def draw_curves(axis: plt.Axes, curves: Sequence[DiagnosticCurve], panel: str) -> None:
    for method in METHOD_ORDER:
        curve = method_curve(curves, panel, method)
        style = METHOD_STYLE[method]
        axis.plot(
            curve.epochs,
            curve.values,
            color=style["color"],
            linewidth=1.55,
            zorder=style["zorder"],
            label=style["label"],
        )


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=2.2,
            label=METHOD_STYLE[method]["label"],
        )
        for method in METHOD_ORDER
    ]


def short_case_title(curve: DiagnosticCurve) -> str:
    if curve.dataset == "oasis":
        return curve.case_label.replace("OASIS ", "")
    return curve.case_label.replace("DIR-LAB ", "")


def render_case_figure(
    curves: Sequence[DiagnosticCurve], output_path: Path, dpi: int
) -> None:
    dataset = curves[0].dataset
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=False)
    for axis, panel in zip(axes, ["A", "B"]):
        draw_curves(axis, curves, panel)
        style_axis(axis, dataset, panel, show_xlabel=True)
        axis.set_ylabel("1−NCC" if panel == "A" else "rRMSE (%)", fontsize=9)
        axis.set_title(
            "A. Registration" if panel == "A" else "B. Supervised field fitting",
            loc="left",
            fontsize=10,
            weight="bold",
        )
    fig.suptitle(curves[0].case_label, fontsize=12.5, weight="bold", y=0.995)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.004),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.97))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_composite(
    dataset: str,
    case_groups: Sequence[Sequence[DiagnosticCurve]],
    output_path: Path,
    dpi: int,
) -> None:
    if len(case_groups) != 10:
        raise RuntimeError(f"{dataset}: expected ten case groups")
    fig, axes = plt.subplots(4, 5, figsize=(20.0, 13.0), sharex=False, sharey=False)
    for block in range(2):
        for column in range(5):
            case_index = block * 5 + column
            curves = case_groups[case_index]
            for panel_offset, panel in enumerate(["A", "B"]):
                row = block * 2 + panel_offset
                axis = axes[row, column]
                draw_curves(axis, curves, panel)
                style_axis(axis, dataset, panel, show_xlabel=panel == "B")
                if column == 0:
                    axis.set_ylabel(PANEL_Y_LABEL[panel], fontsize=9.2, weight="bold")
                if panel == "A":
                    axis.set_title(short_case_title(curves[0]), fontsize=10.2, weight="bold")
    fig.suptitle(
        f"{DATASET_LABEL[dataset]}: registration and supervised field-fitting diagnostics",
        fontsize=17,
        weight="bold",
        y=0.992,
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.95, bottom=0.075, hspace=0.37, wspace=0.28)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def image_record(path: Path, output_root: Path, kind: str, dataset: str) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "kind": kind,
        "dataset": dataset,
        "path": os.path.relpath(path, output_root),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def endpoint_table(curves: Sequence[DiagnosticCurve], dataset: str, panel: str) -> list[str]:
    relevant = sorted(
        [curve for curve in curves if curve.dataset == dataset and curve.panel == panel],
        key=lambda curve: (curve.case_order, METHOD_ORDER.index(curve.method_slug)),
    )
    by_case: dict[str, dict[str, DiagnosticCurve]] = {}
    labels: dict[str, str] = {}
    orders: dict[str, int] = {}
    for curve in relevant:
        by_case.setdefault(curve.case_key, {})[curve.method_slug] = curve
        labels[curve.case_key] = curve.case_label
        orders[curve.case_key] = curve.case_order
    lines = [
        "| Case | INR-Dense | INR-BSCP | D-BSCP | MR-D-BSCP |",
        "|---|---:|---:|---:|---:|",
    ]
    for case_key in sorted(by_case, key=lambda value: orders[value]):
        values = []
        for method in METHOD_ORDER:
            endpoint = float(by_case[case_key][method].values[-1])
            values.append(f"{endpoint:.4g}" if panel == "A" else f"{endpoint:.3f}")
        lines.append(f"| {labels[case_key]} | " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    curves: Sequence[DiagnosticCurve],
    image_records: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Schedule-Matched Population Diagnostic Figures",
        "",
        "These source-backed figures are supplementary/reviewer diagnostics, not the final compact main-paper figure.",
        "",
        "The camera-ready source index preserves 150 accepted population records and replaces only the ten DIR-LAB Panel A MR-D-BSCP records with the accepted `400/400/800/900` schedule refresh.",
        "",
        "## Shared axes",
        "",
        "- Panel A: registration `1−NCC`, log range `1e-3–5e-1`.",
        "- Panel B: supervised target-field fitting `rRMSE (%)`, log range `1e-1–1e3`.",
        "- Every subplot uses epochs `0–2500` with 500-epoch ticks.",
        "- The same Panel A and Panel B definitions are used across both datasets and all cases.",
        "- Gray dotted lines indicate MR-D-BSCP stage transitions only.",
        "",
    ]
    for dataset in ["oasis", "dirlab"]:
        composite = next(
            row for row in image_records if row["kind"] == "composite" and row["dataset"] == dataset
        )
        lines.extend(
            [
                f"## {DATASET_LABEL[dataset]}",
                "",
                f"![{DATASET_LABEL[dataset]} schedule-matched diagnostics]({composite['path']})",
                "",
                "### Panel A endpoints: 1−NCC",
                "",
                *endpoint_table(curves, dataset, "A"),
                "",
                "### Panel B endpoints: rRMSE (%)",
                "",
                *endpoint_table(curves, dataset, "B"),
                "",
            ]
        )
    lines.extend(
        [
            "## Validation and provenance",
            "",
            "- `160/160` trajectories passed endpoint agreement.",
            "- `10/10` refreshed DIR-LAB curves contain 251 finite full-crop diagnostics ending at epoch 2500.",
            "- `150/150` nonreplacement source-index rows are unchanged.",
            "- Individual case figures are under `case_figures/`.",
            "- Machine-readable sources and checks are in `camera_ready_source_index.csv`, `curve_summary.csv`, `replacement_audit.csv`, `validation_report.json`, and `figure_manifest.json`.",
            "",
            "The historical 2400-epoch index and figures remain immutable sensitivity evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_index = Path(args.base_index).resolve()
    replacement_index = Path(args.replacement_index).resolve()
    population_root = Path(args.population_run_root).resolve()
    refresh_root = Path(args.refresh_run_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    base = pd.read_csv(base_index)
    replacement = pd.read_csv(replacement_index)
    validate_replacement_metadata(replacement)
    merged, replacement_audit, unchanged_before, unchanged_after = build_source_index(
        base, replacement
    )
    refresh_checks = validate_refresh_sources(replacement, refresh_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "source_records": len(merged),
                    "replacement_records": len(replacement_audit),
                    "unchanged_records": len(merged) - len(replacement_audit),
                    "refresh_curves_valid": len(refresh_checks),
                    "output_root": str(output_root),
                },
                indent=2,
            )
        )
        return
    if output_root.exists():
        raise FileExistsError(
            f"Versioned output already exists: {output_root}. Use a new version path."
        )
    output_root.mkdir(parents=True)
    case_output = output_root / "case_figures"
    case_output.mkdir()

    source_index_csv = output_root / "camera_ready_source_index.csv"
    merged.to_csv(source_index_csv, index=False)
    source_index_json = output_root / "camera_ready_source_index.json"
    source_index_json.write_text(
        json.dumps(json.loads(merged.to_json(orient="records")), indent=2) + "\n"
    )
    write_csv(output_root / "replacement_audit.csv", replacement_audit)
    write_csv(output_root / "refresh_source_validation.csv", refresh_checks)

    curves: list[DiagnosticCurve] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in merged.iterrows():
        raw_curve = load_merged_curve(row, population_root, refresh_root, data_root)
        try:
            curve, raw_endpoint, raw_difference, endpoint_replaced = normalize_terminal_endpoint(
                raw_curve
            )
        except RuntimeError as error:
            failures.append(str(error))
            continue
        plotted = float(curve.values[-1])
        difference = plotted - curve.accepted_endpoint
        matches = bool(
            np.isclose(plotted, curve.accepted_endpoint, rtol=2.0e-5, atol=2.0e-8)
        )
        if not matches:
            failures.append(
                f"{curve.dataset}/{curve.case_key}/Panel{curve.panel}/{curve.method_slug}: "
                f"plotted={plotted:.12g}, accepted={curve.accepted_endpoint:.12g}"
            )
            continue
        curves.append(curve)
        summary_rows.append(
            {
                "job_id": row["job_id"],
                "dataset": curve.dataset,
                "case_key": curve.case_key,
                "case_order": curve.case_order,
                "case_label": curve.case_label,
                "panel": curve.panel,
                "method_slug": curve.method_slug,
                "display_method": curve.method_label,
                "points": len(curve.epochs),
                "epoch_min": float(np.min(curve.epochs)),
                "epoch_max": float(np.max(curve.epochs)),
                "value_start": float(curve.values[0]),
                "value_end": plotted,
                "value_min": float(np.min(curve.values)),
                "value_max": float(np.max(curve.values)),
                "raw_terminal_value": raw_endpoint,
                "raw_terminal_difference": raw_difference,
                "terminal_endpoint_replaced": endpoint_replaced,
                "accepted_endpoint": curve.accepted_endpoint,
                "endpoint_difference": difference,
                "endpoint_matches": matches,
                "source_metric": curve.source_metric,
                "target_power": curve.target_power,
                "source_path": row["source_path"],
            }
        )
    if failures:
        raise RuntimeError("Curve validation failed:\n" + "\n".join(failures))
    if len(curves) != 160:
        raise RuntimeError(f"Expected 160 validated curves, found {len(curves)}")

    image_records: list[dict[str, Any]] = []
    for dataset in ["oasis", "dirlab"]:
        case_keys = sorted(
            {curve.case_key for curve in curves if curve.dataset == dataset},
            key=lambda key: next(
                curve.case_order
                for curve in curves
                if curve.dataset == dataset and curve.case_key == key
            ),
        )
        case_groups = [
            [curve for curve in curves if curve.dataset == dataset and curve.case_key == case_key]
            for case_key in case_keys
        ]
        for case_key, group in zip(case_keys, case_groups):
            output_path = case_output / f"{dataset}_{case_key}_diagnostics.png"
            render_case_figure(group, output_path, args.dpi)
            image_records.append(image_record(output_path, output_root, "case", dataset))
        composite_path = output_root / f"{dataset}_population_diagnostics_4x5.png"
        render_composite(dataset, case_groups, composite_path, args.dpi)
        image_records.append(image_record(composite_path, output_root, "composite", dataset))

    curve_summary = output_root / "curve_summary.csv"
    pd.DataFrame(summary_rows).sort_values(
        ["dataset", "case_order", "panel", "method_slug"]
    ).to_csv(curve_summary, index=False)
    validation = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "accepted": True,
        "source_records": len(merged),
        "validated_curves": len(curves),
        "validated_endpoints": sum(bool(row["endpoint_matches"]) for row in summary_rows),
        "replacement_records": len(replacement_audit),
        "unchanged_records": 150,
        "unchanged_hash_before": unchanged_before,
        "unchanged_hash_after": unchanged_after,
        "refresh_curves_valid": len(refresh_checks),
        "refresh_points_per_curve": 251,
        "generated_case_figures": sum(row["kind"] == "case" for row in image_records),
        "generated_composites": sum(row["kind"] == "composite" for row in image_records),
        "problems": [],
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    (output_root / "validation_report.md").write_text(
        "\n".join(
            [
                "# Schedule-Matched Diagnostic Validation",
                "",
                "Acceptance: `PASS`",
                "",
                "- Source index: `160/160` unique records.",
                "- Replacements: `10/10` DIR-LAB Panel A MR-D-BSCP records.",
                "- Unchanged records: `150/150`, identical canonical hash.",
                "- Curves/endpoints: `160/160` validated.",
                "- Refresh diagnostics: `10/10`, 251 finite points each through epoch 2500.",
                "- Figures: 20 individual cases and two 4×5 composites.",
                "- Problems: none.",
            ]
        )
        + "\n"
    )
    manifest = {
        "generated_at": validation["generated_at"],
        "artifact_role": "supplementary/reviewer diagnostic; not compact main-paper figure",
        "generator": "scripts/experiments/fit_modes_population/render_schedule_matched_diagnostics.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "repo_commit": git_output("rev-parse", "HEAD"),
        "repo_branch": git_output("branch", "--show-current"),
        "container_image": "${DOCKER_IMAGE}",
        "inputs": {
            "historical_index": "experiments/2026-08-02_fit_modes_population_n10/optimization_index.csv",
            "historical_index_sha256": sha256(base_index),
            "replacement_index": "experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh/replacement_index.csv",
            "replacement_index_sha256": sha256(replacement_index),
            "population_run_root": "${POPULATION_RUN_ROOT}",
            "refresh_run_root": REFRESH_TOKEN,
            "data_root": "${DATA_ROOT}",
        },
        "source_index": source_index_csv.name,
        "source_records": 160,
        "replacements": 10,
        "unchanged_records": 150,
        "metrics": {"A": "1-NCC", "B": "100*sqrt(MSE/(target_power+1e-12))"},
        "scale": "logarithmic",
        "axis_limits": {
            "x_epoch": [0, 2500],
            "x_ticks": list(range(0, 2501, 500)),
            "A": list(AXIS_LIMITS["A"]),
            "B": list(AXIS_LIMITS["B"]),
            "scope": "identical within each panel across both datasets and every case",
        },
        "layout": {
            "rows": 4,
            "columns": 5,
            "row_1": "Panel A cases 1-5",
            "row_2": "Panel B cases 1-5",
            "row_3": "Panel A cases 6-10",
            "row_4": "Panel B cases 6-10",
        },
        "stage_boundaries": {
            "oasis": {"A": [400, 1200], "B": [400, 1200]},
            "dirlab": {"A": [400, 800, 1600], "B": [400, 800, 1600]},
        },
        "figures": image_records,
        "curve_summary": curve_summary.name,
        "validation_report": "validation_report.json",
        "comparison_report": "diagnostic_comparison.md",
    }
    (output_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_report(output_root / "diagnostic_comparison.md", curves, image_records)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
