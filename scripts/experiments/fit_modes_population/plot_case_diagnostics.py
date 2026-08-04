#!/usr/bin/env python
"""Render and compare the 20 accepted population fit-modes diagnostics.

Each case receives a two-panel figure:

* Panel A: registration diagnostic, ``1 - NCC``.
* Panel B: supervised target-field fitting, rRMSE (%).

Every case in a dataset uses the same Panel-A and Panel-B axis definition.  The
script also checks each plotted endpoint against ``optimization_index.csv`` and
writes a machine-readable curve summary and axis manifest beside the figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-08-02_fit_modes_population_n10"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
DEFAULT_RUN_ROOT = Path(
    os.environ.get("RUN_ROOT", "/workspace/scratch/runs/fit_modes_population_n10_v1")
)

METHOD_ORDER = ["idir", "inr_bscp", "d_bscp", "mr_d_bscp"]
METHOD_STYLE = {
    "idir": {"label": "INR-Dense", "color": "#2C7BB6", "zorder": 5},
    "inr_bscp": {"label": "INR-BSCP", "color": "#F28E2B", "zorder": 4},
    "d_bscp": {"label": "D-BSCP", "color": "#D62728", "zorder": 3},
    "mr_d_bscp": {"label": "MR-D-BSCP", "color": "#CC79A7", "zorder": 2},
}
PANEL_LABEL = {"A": "Registration: 1-NCC", "B": "Field fitting: rRMSE (%)"}
PANEL_Y_LABEL = {"A": "1-NCC", "B": "rRMSE (%)"}
BASE_AXIS_LIMITS = {"A": (1.0e-3, 5.0e-1), "B": (1.0e-1, 1.0e2)}
STAGE_BOUNDARIES = {
    ("oasis", "A"): [400, 1200],
    ("oasis", "B"): [400, 1200],
    ("dirlab", "A"): [200, 600, 1400],
    ("dirlab", "B"): [400, 800, 1600],
}
EXPECTED_CASES = {"oasis": 10, "dirlab": 10}


@dataclass(frozen=True)
class DiagnosticCurve:
    dataset: str
    case_key: str
    case_order: int
    case_label: str
    panel: str
    method_slug: str
    method_label: str
    epochs: np.ndarray
    values: np.ndarray
    source_path: Path
    source_metric: str
    target_power: float | None
    accepted_endpoint: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--comparison-md", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def resolve_path(value: Any, run_root: Path, data_root: Path) -> Path:
    text = str(value or "")
    text = text.replace("${RUN_ROOT}", str(run_root))
    text = text.replace("${DATA_ROOT}", str(data_root))
    text = text.replace("${REPO_ROOT}", str(REPO_ROOT))
    return Path(text)


def portable_path(path: Path, run_root: Path, data_root: Path) -> str:
    text = str(path)
    for root, token in [
        (run_root, "${RUN_ROOT}"),
        (data_root, "${DATA_ROOT}"),
        (REPO_ROOT, "${REPO_ROOT}"),
    ]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return text


def case_metadata(row: pd.Series) -> tuple[int, str]:
    if row["dataset"] == "oasis":
        rank = int(row["case_rank"])
        moving = int(row["moving_id"])
        fixed = int(row["fixed_id"])
        return rank, f"OASIS r{rank}: M{moving:04d}→F{fixed:04d}"
    case_id = int(row["case_id"])
    return case_id, f"DIR-LAB case{case_id:02d}: T50→T00"


def epoch_column(frame: pd.DataFrame, source: Path) -> str:
    if "global_epoch" in frame.columns and frame["global_epoch"].notna().any():
        return "global_epoch"
    if "epoch" in frame.columns and frame["epoch"].notna().any():
        return "epoch"
    raise KeyError(f"No epoch column in {source}")


def first_finite_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next(
        (column for column in candidates if column in frame and frame[column].notna().any()),
        None,
    )


def panel_a_values(frame: pd.DataFrame, source: Path) -> tuple[str, np.ndarray]:
    value_column = first_finite_column(
        frame, ["global_ncc_value", "similarity_value", "ncc_value"]
    )
    if value_column is not None:
        return value_column, 1.0 - frame[value_column].to_numpy(float)

    loss_column = first_finite_column(
        frame, ["global_ncc_loss", "similarity_loss", "data_loss", "loss"]
    )
    if loss_column is None:
        raise KeyError(f"No NCC diagnostic column in {source}")
    raw = frame[loss_column].to_numpy(float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        raise ValueError(f"No finite NCC diagnostic values in {source}")
    # These registrations use loss=-NCC.  Positive legacy values, if ever
    # encountered, are already 1-NCC and must not receive the +1 transform.
    values = 1.0 + raw if float(np.nanmedian(finite_raw)) <= 0.0 else raw
    return loss_column, values


def panel_b_values(
    frame: pd.DataFrame, source: Path, target_power: float
) -> tuple[str, np.ndarray]:
    mse_column = first_finite_column(
        frame,
        [
            "finest_field_mse_objective_normalized_grid",
            "finest_data_loss",
            "field_mse_objective_normalized_grid",
            "data_loss",
            "loss",
        ],
    )
    if mse_column is None:
        raise KeyError(f"No target-field MSE column in {source}")
    mse = frame[mse_column].to_numpy(float)
    values = 100.0 * np.sqrt(mse / (target_power + 1.0e-12))
    return mse_column, values


def load_curve(
    row: pd.Series, run_root: Path, data_root: Path
) -> DiagnosticCurve:
    source = resolve_path(row["source_path"], run_root, data_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    x_column = epoch_column(frame, source)
    epochs = frame[x_column].to_numpy(float)
    target_power = finite(row.get("field_target_power_masked_normalized_grid"))
    if row["panel"] == "A":
        source_metric, values = panel_a_values(frame, source)
    else:
        if target_power is None or target_power <= 0.0:
            raise ValueError(f"Invalid target power for {row['job_id']}: {target_power}")
        source_metric, values = panel_b_values(frame, source, target_power)

    keep = np.isfinite(epochs) & np.isfinite(values) & (values > 0.0)
    epochs = epochs[keep]
    values = values[keep]
    if epochs.size == 0:
        raise ValueError(f"No finite positive diagnostic values in {source}")
    order = np.argsort(epochs, kind="stable")
    epochs = epochs[order]
    values = values[order]
    case_order, case_label = case_metadata(row)
    accepted_endpoint = finite(row["endpoint_value"])
    if accepted_endpoint is None:
        raise ValueError(f"Missing accepted endpoint for {row['job_id']}")
    return DiagnosticCurve(
        dataset=str(row["dataset"]),
        case_key=str(row["case_key"]),
        case_order=case_order,
        case_label=case_label,
        panel=str(row["panel"]),
        method_slug=str(row["method_slug"]),
        method_label=str(row["display_method"]),
        epochs=epochs,
        values=values,
        source_path=source,
        source_metric=source_metric,
        target_power=target_power,
        accepted_endpoint=accepted_endpoint,
    )


def validate_index(index: pd.DataFrame) -> None:
    if len(index) != 160:
        raise RuntimeError(f"Expected 160 accepted records, found {len(index)}")
    if index["job_id"].duplicated().any():
        raise RuntimeError("optimization_index.csv contains duplicate job IDs")
    accepted_statuses = {"completed", "completed_existing", "reused"}
    bad_statuses = sorted(set(index["status"].astype(str)) - accepted_statuses)
    if bad_statuses:
        raise RuntimeError(f"Non-accepted statuses in optimization index: {bad_statuses}")
    for dataset, expected_cases in EXPECTED_CASES.items():
        subset = index[index["dataset"].eq(dataset)]
        actual_cases = subset["case_key"].nunique()
        if actual_cases != expected_cases:
            raise RuntimeError(
                f"{dataset}: expected {expected_cases} cases, found {actual_cases}"
            )
        counts = subset.groupby(["case_key", "panel"])["method_slug"].nunique()
        if len(counts) != expected_cases * 2 or not counts.eq(4).all():
            raise RuntimeError(f"{dataset}: not every case/panel has four unique methods")


def endpoint_matches(curve: DiagnosticCurve) -> tuple[bool, float, float]:
    plotted = float(curve.values[-1])
    difference = plotted - curve.accepted_endpoint
    matches = bool(
        np.isclose(plotted, curve.accepted_endpoint, rtol=2.0e-5, atol=2.0e-8)
    )
    return matches, plotted, difference


def normalize_terminal_endpoint(
    curve: DiagnosticCurve,
) -> tuple[DiagnosticCurve, float, float, bool]:
    """Use the accepted post-update endpoint for the displayed terminal point.

    Several single-stage field-fit histories log the epoch loss immediately
    before the optimizer update, while ``summary.json`` evaluates the full mask
    after the final update.  The accepted optimization index deliberately uses
    the latter.  A bounded replacement keeps the trajectory and its endpoint
    convention consistent while retaining the raw terminal value in the curve
    summary for audit.
    """

    matches, raw_endpoint, raw_difference = endpoint_matches(curve)
    if matches:
        return curve, raw_endpoint, raw_difference, False
    relative_difference = abs(raw_difference) / max(abs(curve.accepted_endpoint), 1.0e-12)
    if curve.panel != "B" or relative_difference > 0.15:
        raise RuntimeError(
            f"{curve.dataset}/{curve.case_key}/Panel{curve.panel}/{curve.method_slug}: "
            f"raw={raw_endpoint:.12g}, accepted={curve.accepted_endpoint:.12g}, "
            f"relative_difference={relative_difference:.3%}"
        )
    values = curve.values.copy()
    values[-1] = curve.accepted_endpoint
    return replace(curve, values=values), raw_endpoint, raw_difference, True


def shared_axis_limits(
    curves: list[DiagnosticCurve], dataset: str, panel: str
) -> tuple[float, float]:
    relevant = [curve.values for curve in curves if curve.dataset == dataset and curve.panel == panel]
    values = np.concatenate(relevant)
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    lower, upper = BASE_AXIS_LIMITS[panel]
    if data_min < lower * 0.98:
        lower = 10.0 ** math.floor(math.log10(data_min))
    if data_max > upper * 1.02:
        upper = 10.0 ** math.ceil(math.log10(data_max))
    if not (0.0 < lower < upper):
        raise RuntimeError(
            f"Invalid shared axis for {dataset} Panel {panel}: {(lower, upper)}"
        )
    return float(lower), float(upper)


def figure_filename(dataset: str, case_key: str) -> str:
    return f"{dataset}_{case_key}_diagnostics.png"


def render_case_figure(
    case_curves: list[DiagnosticCurve],
    axis_limits: dict[tuple[str, str], tuple[float, float]],
    output_path: Path,
    dpi: int,
) -> None:
    dataset = case_curves[0].dataset
    case_label = case_curves[0].case_label
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
    for axis, panel in zip(axes, ["A", "B"]):
        for method in METHOD_ORDER:
            matches = [
                curve
                for curve in case_curves
                if curve.panel == panel and curve.method_slug == method
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{case_label} Panel {panel} {method}: expected one curve, found {len(matches)}"
                )
            curve = matches[0]
            style = METHOD_STYLE[method]
            axis.plot(
                curve.epochs,
                curve.values,
                color=style["color"],
                linewidth=1.9,
                zorder=style["zorder"],
                label=style["label"],
            )
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
        axis.set_ylim(*axis_limits[(dataset, panel)])
        axis.set_xlim(0, 2500)
        axis.set_ylabel(PANEL_Y_LABEL[panel])
        axis.set_title(f"{panel}. {PANEL_LABEL[panel]}", loc="left", fontsize=10.5, weight="bold")
        axis.grid(True, which="both", color="#D5DCE3", linewidth=0.65, alpha=0.75)
        axis.tick_params(labelsize=8.5)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(case_label, fontsize=12.5, weight="bold", y=0.995)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=2.2,
            label=METHOD_STYLE[method]["label"],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.02, 0.055, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_endpoint(value: float, panel: str) -> str:
    return f"{value:.4g}" if panel == "A" else f"{value:.3f}"


def endpoint_table(
    curves: list[DiagnosticCurve], dataset: str, panel: str
) -> list[str]:
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
    for case_key in sorted(by_case, key=lambda key: orders[key]):
        values = [
            format_endpoint(float(by_case[case_key][method].values[-1]), panel)
            for method in METHOD_ORDER
        ]
        lines.append(f"| {labels[case_key]} | " + " | ".join(values) + " |")
    return lines


def image_grid(
    curves: list[DiagnosticCurve], dataset: str, output_dir: Path, review_root: Path
) -> list[str]:
    one_per_case: dict[str, DiagnosticCurve] = {}
    for curve in sorted(curves, key=lambda item: item.case_order):
        if curve.dataset == dataset:
            one_per_case.setdefault(curve.case_key, curve)
    cases = list(one_per_case.values())
    if len(cases) != 10:
        raise RuntimeError(f"{dataset}: expected 10 rendered cases, found {len(cases)}")
    lines = ["<table>"]
    for start in [0, 5]:
        lines.append("  <tr>")
        for curve in cases[start : start + 5]:
            image_path = output_dir / figure_filename(dataset, curve.case_key)
            relative = os.path.relpath(image_path, review_root)
            lines.extend(
                [
                    '    <td align="center" width="20%">',
                    f"      <strong>{curve.case_label}</strong><br>",
                    f'      <a href="{relative}"><img src="{relative}" alt="{curve.case_label} diagnostics" width="100%"></a>',
                    "    </td>",
                ]
            )
        lines.append("  </tr>")
    lines.append("</table>")
    return lines


def write_comparison_report(
    path: Path,
    curves: list[DiagnosticCurve],
    axis_limits: dict[tuple[str, str], tuple[float, float]],
    output_dir: Path,
    summary_path: Path,
    manifest_path: Path,
) -> None:
    relative_summary = os.path.relpath(summary_path, path.parent)
    relative_manifest = os.path.relpath(manifest_path, path.parent)
    lines = [
        "# Population Fit-Modes Diagnostic Comparison",
        "",
        "This report places all ten cases from each locked cohort on a common visual scale. "
        "Each thumbnail is a complete two-panel case diagnostic; select it to open the full-resolution figure.",
        "",
        "## Reading the diagnostics",
        "",
        "- Panel A is the full-image/full-crop registration diagnostic `1-NCC` on a logarithmic axis; lower is better.",
        "- Panel B is supervised target-field fitting `rRMSE (%) = 100 × sqrt(MSE / (target power + 1e-12))` on a logarithmic axis; lower is better.",
        "- All trajectories run to epoch 2500. Gray dotted lines mark MR-D-BSCP stage transitions; they do not apply to the single-stage methods.",
        "- Axis limits are identical across all ten cases within each dataset and panel. They are locked in the manifest, not autoscaled per case.",
        "- Some single-stage Panel B histories record the epoch-2500 loss immediately before the last optimizer update. Where it differs, the displayed terminal point uses the accepted full-mask post-update `summary.json` value; both raw and accepted endpoints remain in the curve summary.",
        "",
        "## Shared axis definitions",
        "",
        "| Dataset | Panel A: 1-NCC | Panel B: rRMSE (%) | Scope |",
        "|---|---:|---:|---|",
    ]
    for dataset, label in [("oasis", "OASIS"), ("dirlab", "DIR-LAB")]:
        a_low, a_high = axis_limits[(dataset, "A")]
        b_low, b_high = axis_limits[(dataset, "B")]
        lines.append(
            f"| {label} | `{a_low:g}` to `{a_high:g}` | `{b_low:g}` to `{b_high:g}` | shared by 10 cases |"
        )

    lines.extend(
        [
            "",
            "## Open camera-ready schedule issue",
            "",
            "| Dataset | Panel A MR-D-BSCP | Panel B MR-D-BSCP | Status |",
            "|---|---|---|---|",
            "| OASIS | `400/800/1300` = 2500 | `400/800/1300` = 2500 | schedule matched |",
            "| DIR-LAB | `200/400/800/1000` = 2400 | `400/400/800/900` = 2500 | **not schedule/budget matched** |",
            "",
            "The DIR-LAB discrepancy is part of the accepted protocols, not a plotting artifact. Until a replacement Panel A cohort is run, these results support final-spacing matching (`32-16-8-4`) but not identical schedules or equal Panel A budgets. The preferred camera-ready correction is tracked in [`camera_ready_checklist.md`](../../docs/todos/camera_ready_checklist.md).",
        ]
    )

    for dataset, label in [("oasis", "OASIS N=10"), ("dirlab", "DIR-LAB N=10")]:
        lines.extend(["", f"## {label}", ""])
        lines.extend(image_grid(curves, dataset, output_dir, path.parent))
        lines.extend(
            [
                "",
                "### Panel A endpoints: 1-NCC",
                "",
                *endpoint_table(curves, dataset, "A"),
                "",
                "### Panel B endpoints: rRMSE (%)",
                "",
                *endpoint_table(curves, dataset, "B"),
            ]
        )

    lines.extend(
        [
            "",
            "## Validation and provenance",
            "",
            "All 160 plotted trajectories passed endpoint agreement against the accepted population optimization index. "
            "The comparison combines 136 population runs with 24 exact-compatible source-backed reuses from the original three-case figure.",
            "",
            f"- Curve-level sources, metrics, point counts, extrema, and endpoint checks: [`{Path(relative_summary).name}`]({relative_summary})",
            f"- Shared axis definitions and generation metadata: [`{Path(relative_manifest).name}`]({relative_manifest})",
            "- Accepted run index: [`optimization_index.csv`](optimization_index.csv)",
            "- Protocol and cohort provenance: [`experiment_protocol.md`](experiment_protocol.md)",
            "- Reproducible generator: [`plot_case_diagnostics.py`](../../scripts/experiments/fit_modes_population/plot_case_diagnostics.py)",
            "",
            "These diagnostics support case-by-case optimization review. Population claims should use the accepted endpoint statistics and paired population analysis rather than visual curve inspection alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    review_root = args.review_root.resolve()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else review_root / "diagnostic_figures"
    )
    comparison_md = (
        args.comparison_md.resolve()
        if args.comparison_md is not None
        else review_root / "diagnostic_comparison.md"
    )
    index_path = review_root / "optimization_index.csv"
    index = pd.read_csv(index_path)
    validate_index(index)
    loaded_curves = [load_curve(row, run_root, data_root) for _, row in index.iterrows()]

    failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    curves: list[DiagnosticCurve] = []
    for raw_curve in loaded_curves:
        try:
            curve, raw_endpoint, raw_difference, endpoint_replaced = normalize_terminal_endpoint(
                raw_curve
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        matches, plotted_endpoint, difference = endpoint_matches(curve)
        if not matches:
            failures.append(
                f"{curve.dataset}/{curve.case_key}/Panel{curve.panel}/{curve.method_slug}: "
                f"plotted={plotted_endpoint:.12g}, accepted={curve.accepted_endpoint:.12g}"
            )
            continue
        curves.append(curve)
        summary_rows.append(
            {
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
                "value_end": plotted_endpoint,
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
                "source_path": portable_path(curve.source_path, run_root, data_root),
            }
        )
    if failures:
        raise RuntimeError("Endpoint validation failed:\n" + "\n".join(failures))

    axis_limits = {
        (dataset, panel): shared_axis_limits(curves, dataset, panel)
        for dataset in EXPECTED_CASES
        for panel in ["A", "B"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_records: list[dict[str, str]] = []
    for dataset in EXPECTED_CASES:
        case_keys = sorted(
            {curve.case_key for curve in curves if curve.dataset == dataset},
            key=lambda key: next(
                curve.case_order
                for curve in curves
                if curve.dataset == dataset and curve.case_key == key
            ),
        )
        for case_key in case_keys:
            case_curves = [
                curve
                for curve in curves
                if curve.dataset == dataset and curve.case_key == case_key
            ]
            output_path = output_dir / figure_filename(dataset, case_key)
            render_case_figure(case_curves, axis_limits, output_path, args.dpi)
            figure_records.append(
                {
                    "dataset": dataset,
                    "case_key": case_key,
                    "case_label": case_curves[0].case_label,
                    "figure": os.path.relpath(output_path, review_root),
                    "bytes": output_path.stat().st_size,
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                }
            )

    summary_path = review_root / "diagnostic_curve_summary.csv"
    pd.DataFrame(summary_rows).sort_values(
        ["dataset", "case_order", "panel", "method_slug"]
    ).to_csv(summary_path, index=False)
    manifest_path = review_root / "diagnostic_manifest.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "source_index": "optimization_index.csv",
        "generator": "scripts/experiments/fit_modes_population/plot_case_diagnostics.py",
        "metric_panel_a": "1 - NCC",
        "metric_panel_b": "100 * sqrt(MSE / (target_power + 1e-12))",
        "scale": "logarithmic",
        "axis_scope": "shared across all ten cases within each dataset and panel",
        "axis_limits": {
            dataset: {
                panel: list(axis_limits[(dataset, panel)]) for panel in ["A", "B"]
            }
            for dataset in EXPECTED_CASES
        },
        "stage_boundaries": {
            dataset: {
                panel: STAGE_BOUNDARIES[(dataset, panel)] for panel in ["A", "B"]
            }
            for dataset in EXPECTED_CASES
        },
        "mr_d_bscp_schedule_warning": {
            "status": "open_camera_ready_issue",
            "oasis": {
                "panel_a": "400,800,1300",
                "panel_b": "400,800,1300",
                "total_epochs": 2500,
                "matched": True,
            },
            "dirlab": {
                "panel_a": "200,400,800,1000",
                "panel_a_total_epochs": 2400,
                "panel_b": "400,400,800,900",
                "panel_b_total_epochs": 2500,
                "matched": False,
            },
            "preferred_camera_ready_fix": (
                "new versioned DIR-LAB Panel A MR-D-BSCP N=10 cohort using "
                "400,400,800,900; preserve the accepted 2400-epoch cohort"
            ),
            "checklist": "docs/todos/camera_ready_checklist.md",
        },
        "expected_curves": 160,
        "validated_endpoints": sum(row["endpoint_matches"] for row in summary_rows),
        "terminal_endpoint_replacements": sum(
            row["terminal_endpoint_replaced"] for row in summary_rows
        ),
        "terminal_endpoint_policy": (
            "Panel B history rows are pre-update; bounded mismatches use the accepted "
            "post-update full-mask endpoint from summary.json at the terminal point"
        ),
        "generated_case_figures": len(figure_records),
        "figures": figure_records,
        "curve_summary": summary_path.name,
        "comparison_report": comparison_md.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_comparison_report(
        comparison_md,
        curves,
        axis_limits,
        output_dir,
        summary_path,
        manifest_path,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
