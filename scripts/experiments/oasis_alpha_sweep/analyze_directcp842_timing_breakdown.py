#!/usr/bin/env python
"""Analyze DirectCP 8-4-2 detailed timing breakdown runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_timing_breakdown_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_timing_breakdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--phase", type=str, default="rank0")
    parser.add_argument("--expected-case-ranks", type=str, default="")
    parser.add_argument("--expected-candidates", type=str, default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_tokens(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def event_seconds(timing: Dict[str, object]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    events = timing.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            totals[str(event.get("name", ""))] += as_float(event.get("elapsed_sec"), 0.0)
            category = str(event.get("category", ""))
            if category:
                totals[f"category:{category}"] += as_float(event.get("elapsed_sec"), 0.0)
    return dict(totals)


def event_by_name(timing: Dict[str, object], name: str) -> Optional[Dict[str, object]]:
    events = timing.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and str(event.get("name")) == str(name):
                return event
    return None


def flatten_reconciliation(
    command_rows: Sequence[Dict[str, str]],
    summaries: Dict[str, Dict[str, object]],
    timings: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cmd_row in command_rows:
        run_dir = str(cmd_row.get("run_dir", ""))
        summary = summaries.get(run_dir, {})
        timing = timings.get(run_dir, {})
        external_wall = as_float(cmd_row.get("external_wall_sec"), 0.0)
        script_total = as_float(timing.get("script_total_sec"), math.nan)
        external_uninstrumented = external_wall - script_total if math.isfinite(script_total) else math.nan
        external_residual = external_wall - script_total - external_uninstrumented if math.isfinite(script_total) else math.nan
        rows.append(
            {
                "row_type": "external_wall",
                "phase": cmd_row.get("phase", ""),
                "candidate": cmd_row.get("candidate", ""),
                "case_rank": cmd_row.get("case_rank", ""),
                "run_dir": run_dir,
                "name": "external_wall",
                "elapsed_sec": external_wall,
                "child_elapsed_sec": script_total,
                "self_unaccounted_sec": external_uninstrumented,
                "reconciliation_residual_sec": external_residual,
                "tolerance_sec": max(1.0, 0.01 * max(abs(external_wall), 1e-9)),
                "within_tolerance": bool(
                    math.isfinite(external_residual)
                    and abs(external_residual) <= max(1.0, 0.01 * max(abs(external_wall), 1e-9))
                ),
                "summary_total_elapsed_sec": summary.get("total_elapsed_sec"),
                "script_total_sec": script_total,
                "external_wall_sec": external_wall,
            }
        )
        rec_rows = timing.get("reconciliation", [])
        if isinstance(rec_rows, list):
            for rec in rec_rows:
                if not isinstance(rec, dict):
                    continue
                elapsed = as_float(rec.get("elapsed_sec"), 0.0)
                child = as_float(rec.get("child_elapsed_sec"), 0.0)
                self_unaccounted = as_float(rec.get("self_unaccounted_sec"), 0.0)
                residual = elapsed - child - self_unaccounted
                tolerance = max(0.5, 0.01 * max(abs(elapsed), 1e-9))
                rows.append(
                    {
                        "row_type": "script_parent",
                        "phase": cmd_row.get("phase", ""),
                        "candidate": cmd_row.get("candidate", ""),
                        "case_rank": cmd_row.get("case_rank", ""),
                        "run_dir": run_dir,
                        "name": rec.get("name"),
                        "elapsed_sec": elapsed,
                        "child_elapsed_sec": child,
                        "self_unaccounted_sec": self_unaccounted,
                        "reconciliation_residual_sec": residual,
                        "tolerance_sec": tolerance,
                        "within_tolerance": bool(abs(residual) <= tolerance),
                        "self_unaccounted_within_tolerance": rec.get("within_tolerance"),
                        "child_count": rec.get("child_count"),
                        "script_total_sec": script_total,
                        "external_wall_sec": external_wall,
                    }
                )
    return rows


def phase_summary_map(timing: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    output: Dict[int, Dict[str, object]] = {}
    rows = timing.get("stage_phase_summaries", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                output[as_int(row.get("stage_idx"))] = row
    return output


def flatten_stage_rows(
    command_rows: Sequence[Dict[str, str]],
    summaries: Dict[str, Dict[str, object]],
    timings: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cmd_row in command_rows:
        run_dir = str(cmd_row.get("run_dir", ""))
        summary = summaries.get(run_dir, {})
        timing = timings.get(run_dir, {})
        phase_by_stage = phase_summary_map(timing)
        event_totals = event_seconds(timing)
        prolong_by_stage: Dict[int, float] = defaultdict(float)
        prolong_rows = summary.get("prolongation_diagnostics", [])
        if isinstance(prolong_rows, list):
            for row in prolong_rows:
                if isinstance(row, dict):
                    prolong_by_stage[as_int(row.get("stage_idx"))] += as_float(row.get("timing_prolongation_sec"), 0.0)
        stage_summaries = summary.get("stage_summaries", [])
        if not isinstance(stage_summaries, list):
            continue
        for stage in stage_summaries:
            if not isinstance(stage, dict):
                continue
            stage_idx = as_int(stage.get("stage_idx"))
            timing_breakdown = stage.get("timing_breakdown", {})
            if not isinstance(timing_breakdown, dict):
                timing_breakdown = {}
            phase_summary = phase_by_stage.get(stage_idx, {})
            phase_totals = phase_summary.get("phase_totals_sec", {})
            if not isinstance(phase_totals, dict):
                phase_totals = {}
            actual_epochs = as_int(stage.get("actual_epochs"))
            optimizer_loop_sec = as_float(timing_breakdown.get("optimizer_loop_sec"), 0.0)
            fixed_known = (
                as_float(timing_breakdown.get("loss_volume_setup_sec"), 0.0)
                + as_float(timing_breakdown.get("stage_setup_sec"), 0.0)
                + prolong_by_stage.get(stage_idx, 0.0)
                + as_float(timing_breakdown.get("final_dense_expand_sec"), 0.0)
                + as_float(timing_breakdown.get("control_summary_stats_sec"), 0.0)
            )
            rows.append(
                {
                    "phase": cmd_row.get("phase", ""),
                    "candidate": cmd_row.get("candidate", ""),
                    "case_rank": as_int(cmd_row.get("case_rank")),
                    "run_dir": run_dir,
                    "stage_idx": stage_idx,
                    "full_cps": stage.get("full_cps"),
                    "downsample": stage.get("downsample"),
                    "lr": stage.get("lr"),
                    "requested_epochs": as_int(stage.get("requested_epochs")),
                    "actual_epochs": actual_epochs,
                    "stop_reason": stage.get("stop_reason"),
                    "stage_elapsed_legacy_sec": stage.get("elapsed_sec"),
                    "stage_total_sec": as_float(timing_breakdown.get("stage_total_sec"), 0.0),
                    "fixed_stage_known_sec": fixed_known,
                    "loss_volume_setup_sec": as_float(timing_breakdown.get("loss_volume_setup_sec"), 0.0),
                    "stage_setup_sec": as_float(timing_breakdown.get("stage_setup_sec"), 0.0),
                    "prolongation_sec": prolong_by_stage.get(stage_idx, 0.0),
                    "optimizer_loop_sec": optimizer_loop_sec,
                    "final_dense_expand_sec": as_float(timing_breakdown.get("final_dense_expand_sec"), 0.0),
                    "control_summary_stats_sec": as_float(timing_breakdown.get("control_summary_stats_sec"), 0.0),
                    "optimizer_loop_sec_per_epoch": optimizer_loop_sec / actual_epochs if actual_epochs > 0 else math.nan,
                    "phase_control_grid_sec": as_float(phase_totals.get("control_grid"), 0.0),
                    "phase_expand_sec": as_float(phase_totals.get("expand"), 0.0),
                    "phase_warp_ncc_sec": as_float(phase_totals.get("warp_ncc"), 0.0),
                    "phase_regularizer_sec": as_float(phase_totals.get("regularizer"), 0.0),
                    "phase_backward_step_sec": as_float(phase_totals.get("backward_step"), 0.0),
                    "phase_logging_early_stop_wall_sec": as_float(
                        phase_totals.get("logging_early_stop_wall_sec"),
                        0.0,
                    ),
                    "script_total_sec": as_float(timing.get("script_total_sec"), math.nan),
                    "training_event_sec": event_totals.get("training", math.nan),
                }
            )
    return rows


def flatten_method_rows(
    command_rows: Sequence[Dict[str, str]],
    summaries: Dict[str, Dict[str, object]],
    timings: Dict[str, Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    stage_by_run: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        stage_by_run[str(row.get("run_dir"))].append(row)

    rows: List[Dict[str, object]] = []
    for cmd_row in command_rows:
        run_dir = str(cmd_row.get("run_dir", ""))
        summary = summaries.get(run_dir, {})
        timing = timings.get(run_dir, {})
        events = event_seconds(timing)
        run_stage_rows = stage_by_run.get(run_dir, [])
        optimizer_loop_sec = sum(as_float(row.get("optimizer_loop_sec"), 0.0) for row in run_stage_rows)
        fixed_stage_known_sec = sum(as_float(row.get("fixed_stage_known_sec"), 0.0) for row in run_stage_rows)
        post_eval_sec = events.get("post_train_evaluation", 0.0)
        write_sec = events.get("write_outputs", 0.0)
        setup_sec = events.get("setup", 0.0)
        sampler_save_sec = events.get("sampler_save", 0.0)
        script_total_sec = as_float(timing.get("script_total_sec"), math.nan)
        external_wall_sec = as_float(cmd_row.get("external_wall_sec"), math.nan)
        total_fixed_known_sec = fixed_stage_known_sec + post_eval_sec + write_sec + setup_sec + sampler_save_sec
        rows.append(
            {
                "phase": cmd_row.get("phase", ""),
                "candidate": cmd_row.get("candidate", ""),
                "case_rank": as_int(cmd_row.get("case_rank")),
                "original_case_index": cmd_row.get("original_case_index"),
                "fixed_id": cmd_row.get("fixed_id"),
                "moving_id": cmd_row.get("moving_id"),
                "run_dir": run_dir,
                "returncode": as_int(cmd_row.get("returncode")),
                "done_exists": cmd_row.get("done_exists"),
                "timing_exists": cmd_row.get("timing_exists"),
                "external_wall_sec": external_wall_sec,
                "script_total_sec": script_total_sec,
                "external_uninstrumented_sec": external_wall_sec - script_total_sec
                if math.isfinite(external_wall_sec) and math.isfinite(script_total_sec)
                else math.nan,
                "summary_total_elapsed_sec": summary.get("total_elapsed_sec"),
                "training_elapsed_sec": summary.get("training_elapsed_sec"),
                "method_execution_sec": events.get("method_execution", math.nan),
                "training_event_sec": events.get("training", math.nan),
                "setup_sec": setup_sec,
                "optimizer_loop_sec": optimizer_loop_sec,
                "fixed_stage_known_sec": fixed_stage_known_sec,
                "post_train_evaluation_sec": post_eval_sec,
                "write_outputs_sec": write_sec,
                "sampler_save_sec": sampler_save_sec,
                "known_step_proportional_sec": optimizer_loop_sec,
                "known_fixed_or_io_sec": total_fixed_known_sec,
                "other_script_sec": script_total_sec - optimizer_loop_sec - total_fixed_known_sec
                if math.isfinite(script_total_sec)
                else math.nan,
                "actual_epochs": summary.get("actual_epochs"),
                "stage_actual_epochs": summary.get("stage_actual_epochs"),
                "requested_epochs": summary.get("requested_epochs"),
                "dice_mean": summary.get("dice_mean"),
                "hd95_mean": summary.get("hd95_mean"),
                "asd_symmetric_mean": summary.get("asd_symmetric_mean"),
                "training_peak_mem_gb": summary.get("training_peak_mem_gb"),
                "non_target_gpu_apps_pre_count": cmd_row.get("non_target_gpu_apps_pre_count"),
                "non_target_gpu_apps_post_count": cmd_row.get("non_target_gpu_apps_post_count"),
            }
        )
    return rows


def fit_linear(rows: Sequence[Dict[str, object]], x_key: str, y_key: str) -> Dict[str, object]:
    x = np.asarray([as_float(row.get(x_key)) for row in rows], dtype=np.float64)
    y = np.asarray([as_float(row.get(y_key)) for row in rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.ptp(x)) <= 1e-9:
        return {"n": int(x.size), "intercept_sec": math.nan, "slope_sec_per_epoch": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {
        "n": int(x.size),
        "intercept_sec": float(intercept),
        "slope_sec_per_epoch": float(slope),
        "r2": float(r2),
    }


def grouped(rows: Iterable[Dict[str, object]], keys: Sequence[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    output: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        output[tuple(row.get(key) for key in keys)].append(row)
    return output


def mean_value(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def comparison_rows(method_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    by_rank = grouped(method_rows, ["case_rank"])
    for (case_rank,), rows in sorted(by_rank.items(), key=lambda item: item[0]):
        by_candidate = {str(row.get("candidate")): row for row in rows}
        original = by_candidate.get("original_mrdbscp")
        fast = by_candidate.get("A_fast_decay")
        if original is None or fast is None:
            continue
        output.append(
            {
                "case_rank": case_rank,
                "original_external_wall_sec": original.get("external_wall_sec"),
                "fast_external_wall_sec": fast.get("external_wall_sec"),
                "external_wall_savings_sec": as_float(original.get("external_wall_sec")) - as_float(fast.get("external_wall_sec")),
                "external_wall_speedup": as_float(original.get("external_wall_sec")) / as_float(fast.get("external_wall_sec"))
                if as_float(fast.get("external_wall_sec")) > 0
                else math.nan,
                "original_script_total_sec": original.get("script_total_sec"),
                "fast_script_total_sec": fast.get("script_total_sec"),
                "script_total_savings_sec": as_float(original.get("script_total_sec")) - as_float(fast.get("script_total_sec")),
                "original_optimizer_loop_sec": original.get("optimizer_loop_sec"),
                "fast_optimizer_loop_sec": fast.get("optimizer_loop_sec"),
                "optimizer_loop_savings_sec": as_float(original.get("optimizer_loop_sec")) - as_float(fast.get("optimizer_loop_sec")),
                "original_actual_epochs": original.get("actual_epochs"),
                "fast_actual_epochs": fast.get("actual_epochs"),
                "epoch_reduction": as_float(original.get("actual_epochs")) - as_float(fast.get("actual_epochs")),
                "original_dice_mean": original.get("dice_mean"),
                "fast_dice_mean": fast.get("dice_mean"),
                "dice_delta_fast_minus_original": as_float(fast.get("dice_mean")) - as_float(original.get("dice_mean")),
                "original_hd95_mean": original.get("hd95_mean"),
                "fast_hd95_mean": fast.get("hd95_mean"),
                "hd95_delta_fast_minus_original": as_float(fast.get("hd95_mean")) - as_float(original.get("hd95_mean")),
            }
        )
    return output


def write_linear_fits(review_root: Path, stage_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (stage_idx,), grouped_rows in sorted(grouped(stage_rows, ["stage_idx"]).items(), key=lambda item: item[0]):
        fit = fit_linear(grouped_rows, "actual_epochs", "stage_total_sec")
        rows.append(
            {
                "fit_scope": "all_candidates",
                "stage_idx": stage_idx,
                "elapsed_name": "stage_total_sec",
                **fit,
            }
        )
        loop_fit = fit_linear(grouped_rows, "actual_epochs", "optimizer_loop_sec")
        rows.append(
            {
                "fit_scope": "all_candidates",
                "stage_idx": stage_idx,
                "elapsed_name": "optimizer_loop_sec",
                **loop_fit,
            }
        )
    c.write_csv(review_root / "timing_stage_linear_fits.csv", rows)
    return rows


def format_sec(value: object) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.2f}s"


def format_float(value: object, digits: int = 4) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def report_markdown(
    *,
    args: argparse.Namespace,
    command_rows: Sequence[Dict[str, str]],
    method_rows: Sequence[Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
    reconciliation_rows: Sequence[Dict[str, object]],
    paired_rows: Sequence[Dict[str, object]],
    fit_rows: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# DirectCP 8-4-2 Timing Breakdown")
    lines.append("")
    lines.append(f"- Phase: `{args.phase}`")
    lines.append(f"- Output root: `{args.out_root}`")
    lines.append(f"- Completed command rows: {len(command_rows)}")
    lines.append(f"- Timing JSON rows: {sum(1 for row in method_rows if str(row.get('timing_exists')).lower() == 'true')}/{len(method_rows)}")
    failures = [row for row in method_rows if as_int(row.get("returncode")) != 0]
    lines.append(f"- Failed subprocesses: {len(failures)}")
    non_target = sum(
        as_int(row.get("non_target_gpu_apps_pre_count")) + as_int(row.get("non_target_gpu_apps_post_count"))
        for row in method_rows
    )
    lines.append(f"- Non-target GPU compute-app observations: {non_target}")
    lines.append("")

    bad_recon = [
        row
        for row in reconciliation_rows
        if not bool(row.get("within_tolerance", False))
    ]
    large_self = [
        row
        for row in reconciliation_rows
        if row.get("row_type") == "script_parent"
        and as_int(row.get("child_count")) > 0
        and str(row.get("self_unaccounted_within_tolerance")) == "False"
    ]
    lines.append("## Reconciliation")
    lines.append("")
    lines.append(f"- Residual failures: {len(bad_recon)}")
    lines.append(f"- Parent intervals with large self-unaccounted time: {len(large_self)}")
    external_uninstrumented = [as_float(row.get("external_uninstrumented_sec")) for row in method_rows]
    external_uninstrumented = [value for value in external_uninstrumented if math.isfinite(value)]
    if external_uninstrumented:
        lines.append(f"- External uninstrumented wall mean: {float(np.mean(external_uninstrumented)):.2f}s")
    lines.append("")

    method_groups = grouped(method_rows, ["candidate"])
    method_means: Dict[str, Dict[str, float]] = {}
    for (candidate,), rows in method_groups.items():
        method_means[str(candidate)] = {
            "external_wall_sec": mean_value(rows, "external_wall_sec"),
            "script_total_sec": mean_value(rows, "script_total_sec"),
            "external_uninstrumented_sec": mean_value(rows, "external_uninstrumented_sec"),
            "optimizer_loop_sec": mean_value(rows, "optimizer_loop_sec"),
            "known_fixed_or_io_sec": mean_value(rows, "known_fixed_or_io_sec"),
            "other_script_sec": mean_value(rows, "other_script_sec"),
            "training_elapsed_sec": mean_value(rows, "training_elapsed_sec"),
            "fixed_stage_known_sec": mean_value(rows, "fixed_stage_known_sec"),
            "setup_sec": mean_value(rows, "setup_sec"),
            "post_train_evaluation_sec": mean_value(rows, "post_train_evaluation_sec"),
            "write_outputs_sec": mean_value(rows, "write_outputs_sec"),
            "sampler_save_sec": mean_value(rows, "sampler_save_sec"),
        }

    lines.append("## Timing Hierarchy And Definitions")
    lines.append("")
    lines.append("Yes: `optimizer_loop_sec` is a strict subset of `training_elapsed_sec`.")
    lines.append("")
    lines.append("But `known_fixed_or_io_sec` is not a single parent interval. It is a derived roll-up across several places in the run.")
    lines.append("")
    lines.append("Conceptual hierarchy:")
    lines.append("")
    lines.append("```text")
    lines.append("external_wall")
    lines.append("  script_total")
    lines.append("    setup")
    lines.append("    method_execution")
    lines.append("      initial_dense_grid / small setup")
    lines.append("      training")
    lines.append("        stage/prolongation fixed work")
    lines.append("        optimizer_loop")
    lines.append("      post_train_evaluation")
    lines.append("      small metric/model assembly")
    lines.append("    sampler_save")
    lines.append("    write_outputs")
    lines.append("    other_script")
    lines.append("  external_uninstrumented")
    lines.append("```")
    lines.append("")
    lines.append("Meaning of the major intervals:")
    lines.append("")
    lines.append("- `external_wall_sec`: launcher-measured subprocess wall time, including Python startup/import, argument parsing, script execution, and process exit.")
    lines.append("- `script_total_sec`: in-script wall time from `run_one.py` timing instrumentation. It is the reconciled parent for recorded script intervals.")
    lines.append("- `method_execution_sec`: the method call itself; for DirectCP this includes training, post-training evaluation, and small model/metric assembly.")
    lines.append("- `training_elapsed_sec`: DirectCP multistage training block only. It includes per-stage fixed work, prolongations, optimizer loops, and final dense expansions, but excludes post-training image/label evaluation and output writing.")
    lines.append("- `optimizer_loop_sec`: strict subset of `training_elapsed_sec`; summed over the per-stage epoch loops.")
    lines.append("- `known_fixed_or_io_sec`: derived roll-up, not a single parent interval. It sums known fixed stage work plus setup, post-training evaluation, output writing, and sampler save.")
    lines.append("- `other_script_sec`: measured script time not assigned to `optimizer_loop_sec` or `known_fixed_or_io_sec`, such as initial dense grid setup, small model/metric assembly, and timing bookkeeping.")
    lines.append("- `external_uninstrumented_sec`: `external_wall_sec - script_total_sec`; mostly process startup/import/argparse/exit and timing-artifact write overhead outside the script timer.")
    lines.append("")
    lines.append("Important accounting relationships:")
    lines.append("")
    lines.append("```text")
    lines.append("external_wall_sec ~= script_total_sec + external_uninstrumented_sec")
    lines.append("script_total_sec ~= optimizer_loop_sec + known_fixed_or_io_sec + other_script_sec")
    lines.append("training_elapsed_sec ~= optimizer_loop_sec + non_loop_training_work")
    lines.append("```")
    lines.append("")
    lines.append("Because `known_fixed_or_io_sec` mixes intervals inside and outside training, it should not be added to `training_elapsed_sec`. It is designed to explain `script_total_sec` together with `optimizer_loop_sec` and `other_script_sec`.")
    lines.append("")

    lines.append("## What Is In Fixed/IO Known")
    lines.append("")
    lines.append("Defined in `scripts/experiments/oasis_alpha_sweep/analyze_directcp842_timing_breakdown.py` around line 204 as:")
    lines.append("")
    lines.append("```text")
    lines.append("fixed/io known =")
    lines.append("  fixed_stage_known")
    lines.append("  + setup")
    lines.append("  + post_train_evaluation")
    lines.append("  + write_outputs")
    lines.append("  + sampler_save")
    lines.append("")
    lines.append("fixed_stage_known = sum over stages:")
    lines.append("  loss_volume_setup")
    lines.append("  + stage_setup")
    lines.append("  + prolongation")
    lines.append("  + final_dense_expand")
    lines.append("  + control_summary_stats")
    lines.append("```")
    lines.append("")
    lines.append("So `fixed/io known` includes:")
    lines.append("")
    lines.append("- CUDA/device setup, random seeds, case load, case validation/print, and sampler initialization.")
    lines.append("- Per-stage loss-volume creation.")
    lines.append("- Per-stage grid/control/optimizer setup.")
    lines.append("- Stage-to-stage prolongation.")
    lines.append("- Final dense expansion after each stage.")
    lines.append("- Control summary statistics.")
    lines.append("- Post-training image warp, segmentation warp, Dice, HD95/ASD, and dense NCC.")
    lines.append("- Regularity metrics.")
    lines.append("- Dense displacement save.")
    lines.append("- Model/control saves.")
    lines.append("- CSV/JSON/summary/DONE writing.")
    lines.append("")
    lines.append("It does not include:")
    lines.append("")
    lines.append("- Optimizer-loop epochs.")
    lines.append("- Small residual script work, reported as `other_script_sec`.")
    lines.append("- Python process startup/import/argument parsing/process exit, reported as `external_uninstrumented_sec`.")
    lines.append("- The `timing.json` write itself, which is effectively outside `script_total_sec`.")
    lines.append("")

    if method_means:
        lines.append("## Worked Accounting Examples")
        lines.append("")
        for candidate, label in [
            ("original_mrdbscp", "original"),
            ("A_fast_decay", "A_fast_decay"),
        ]:
            means = method_means.get(candidate)
            if not means:
                continue
            script_total = means["script_total_sec"]
            optimizer_loop = means["optimizer_loop_sec"]
            fixed_io = means["known_fixed_or_io_sec"]
            other_script = means["other_script_sec"]
            wall = means["external_wall_sec"]
            external_uninstrumented = means["external_uninstrumented_sec"]
            training = means["training_elapsed_sec"]
            training_non_loop = training - optimizer_loop
            lines.append(f"{label}:")
            lines.append("")
            lines.append("```text")
            lines.append(f"script_total {script_total:.2f}")
            lines.append(f"  = optimizer_loop {optimizer_loop:.2f}")
            lines.append(f"  + fixed/io known {fixed_io:.2f}")
            lines.append(f"  + other_script {other_script:.2f}")
            lines.append("")
            lines.append(f"wall {wall:.2f}")
            lines.append(f"  = script_total {script_total:.2f}")
            lines.append(f"  + external_uninstrumented {external_uninstrumented:.2f}")
            lines.append("")
            lines.append(f"training {training:.2f}")
            lines.append(f"  = optimizer_loop {optimizer_loop:.2f}")
            lines.append(f"  + non_loop_training_work {training_non_loop:.2f}")
            lines.append("```")
            lines.append("")

    if paired_rows:
        optimizer_saving = mean_value(paired_rows, "optimizer_loop_savings_sec")
        wall_saving = mean_value(paired_rows, "external_wall_savings_sec")
        epoch_reduction = mean_value(paired_rows, "epoch_reduction")
        fast_fixed_io = method_means.get("A_fast_decay", {}).get("known_fixed_or_io_sec", math.nan)
        lines.append("The key reason epoch reduction does not translate linearly is:")
        lines.append("")
        lines.append("```text")
        lines.append(f"A_fast_decay removes about {optimizer_saving:.2f}s from optimizer_loop,")
        lines.append(f"but the A_fast_decay fixed/io known bucket is still about {fast_fixed_io:.2f}s per case.")
        lines.append(f"That is why a {epoch_reduction:.1f} average epoch reduction gives only {wall_saving:.2f}s wall saving per case.")
        lines.append("```")
        lines.append("")

    lines.append("## Method Means")
    lines.append("")
    lines.append("| candidate | n | wall | script | training | optimizer loop | fixed/io known | epochs | Dice | HD95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (candidate,), rows in sorted(grouped(method_rows, ["candidate"]).items(), key=lambda item: str(item[0])):
        lines.append(
            "| "
            + str(candidate)
            + f" | {len(rows)}"
            + f" | {format_sec(mean_value(rows, 'external_wall_sec'))}"
            + f" | {format_sec(mean_value(rows, 'script_total_sec'))}"
            + f" | {format_sec(mean_value(rows, 'training_elapsed_sec'))}"
            + f" | {format_sec(mean_value(rows, 'optimizer_loop_sec'))}"
            + f" | {format_sec(mean_value(rows, 'known_fixed_or_io_sec'))}"
            + f" | {format_float(mean_value(rows, 'actual_epochs'), 1)}"
            + f" | {format_float(mean_value(rows, 'dice_mean'), 5)}"
            + f" | {format_float(mean_value(rows, 'hd95_mean'), 4)}"
            + " |"
        )
    lines.append("")

    lines.append("## Accounting Means")
    lines.append("")
    lines.append("| candidate | wall | script | external uninstrumented | optimizer loop | fixed/io known | other script | training | training non-loop |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (candidate,), rows in sorted(grouped(method_rows, ["candidate"]).items(), key=lambda item: str(item[0])):
        optimizer_loop = mean_value(rows, "optimizer_loop_sec")
        training = mean_value(rows, "training_elapsed_sec")
        training_non_loop = training - optimizer_loop if math.isfinite(training) and math.isfinite(optimizer_loop) else math.nan
        lines.append(
            "| "
            + str(candidate)
            + f" | {format_sec(mean_value(rows, 'external_wall_sec'))}"
            + f" | {format_sec(mean_value(rows, 'script_total_sec'))}"
            + f" | {format_sec(mean_value(rows, 'external_uninstrumented_sec'))}"
            + f" | {format_sec(optimizer_loop)}"
            + f" | {format_sec(mean_value(rows, 'known_fixed_or_io_sec'))}"
            + f" | {format_sec(mean_value(rows, 'other_script_sec'))}"
            + f" | {format_sec(training)}"
            + f" | {format_sec(training_non_loop)}"
            + " |"
        )
    lines.append("")

    lines.append("## Fixed/IO Known Components")
    lines.append("")
    lines.append("`fixed/io known` is computed as:")
    lines.append("")
    lines.append("```text")
    lines.append("fixed/io known = fixed_stage_known + setup + post_train_evaluation + write_outputs + sampler_save")
    lines.append("fixed_stage_known = sum_stages(loss_volume_setup + stage_setup + prolongation + final_dense_expand + control_summary_stats)")
    lines.append("```")
    lines.append("")
    lines.append("| candidate | fixed stage known | setup | post-train eval | write outputs | sampler save | fixed/io known |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for (candidate,), rows in sorted(grouped(method_rows, ["candidate"]).items(), key=lambda item: str(item[0])):
        lines.append(
            "| "
            + str(candidate)
            + f" | {format_sec(mean_value(rows, 'fixed_stage_known_sec'))}"
            + f" | {format_sec(mean_value(rows, 'setup_sec'))}"
            + f" | {format_sec(mean_value(rows, 'post_train_evaluation_sec'))}"
            + f" | {format_sec(mean_value(rows, 'write_outputs_sec'))}"
            + f" | {format_sec(mean_value(rows, 'sampler_save_sec'))}"
            + f" | {format_sec(mean_value(rows, 'known_fixed_or_io_sec'))}"
            + " |"
        )
    lines.append("")

    if paired_rows:
        lines.append("## Original vs A_fast_decay")
        lines.append("")
        lines.append("| n pairs | wall savings | wall speedup | optimizer-loop savings | epoch reduction | Dice delta | HD95 delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {len(paired_rows)}"
            f" | {format_sec(mean_value(paired_rows, 'external_wall_savings_sec'))}"
            f" | {format_float(mean_value(paired_rows, 'external_wall_speedup'), 3)}x"
            f" | {format_sec(mean_value(paired_rows, 'optimizer_loop_savings_sec'))}"
            f" | {format_float(mean_value(paired_rows, 'epoch_reduction'), 1)}"
            f" | {format_float(mean_value(paired_rows, 'dice_delta_fast_minus_original'), 5)}"
            f" | {format_float(mean_value(paired_rows, 'hd95_delta_fast_minus_original'), 4)}"
            " |"
        )
        lines.append("")

    lines.append("## Stage Linear Fits")
    lines.append("")
    lines.append("| stage | elapsed | n | intercept | slope/epoch | R2 |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in fit_rows:
        lines.append(
            f"| {row.get('stage_idx')}"
            f" | {row.get('elapsed_name')}"
            f" | {row.get('n')}"
            f" | {format_sec(row.get('intercept_sec'))}"
            f" | {format_sec(row.get('slope_sec_per_epoch'))}"
            f" | {format_float(row.get('r2'), 3)}"
            " |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Epoch reduction only converts into wall-clock savings through the optimizer-loop bucket. "
        "Loss-volume setup, grid/control setup, prolongation, final dense expansion, dense image/label "
        "evaluation, regularity metrics, and serialization remain largely fixed per case. The `known_fixed_or_io_sec` "
        "column is therefore the main explanation when a large epoch reduction produces a modest total-wall reduction."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `timing_reconciliation.csv`")
    lines.append("- `timing_stage_breakdown.csv`")
    lines.append("- `timing_method_comparison.csv`")
    lines.append("- `timing_stage_linear_fits.csv`")
    lines.append("- `command_timing_<phase>.csv`")
    lines.append("")
    return "\n".join(lines)


def load_run_payloads(command_rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], List[str]]:
    summaries: Dict[str, Dict[str, object]] = {}
    timings: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []
    for row in command_rows:
        run_dir = Path(str(row.get("run_dir", "")))
        summary_path = run_dir / "summary.json"
        timing_path = run_dir / "timing.json"
        if summary_path.exists():
            payload = c.read_json(summary_path)
            if isinstance(payload, dict):
                summaries[str(run_dir)] = payload
        else:
            missing.append(str(summary_path))
        if timing_path.exists():
            payload = c.read_json(timing_path)
            if isinstance(payload, dict):
                timings[str(run_dir)] = payload
        else:
            missing.append(str(timing_path))
    return summaries, timings, missing


def check_expected(args: argparse.Namespace, command_rows: Sequence[Dict[str, str]], missing_payloads: Sequence[str]) -> None:
    expected_ranks = parse_tokens(str(args.expected_case_ranks))
    expected_candidates = parse_tokens(str(args.expected_candidates))
    if expected_ranks and expected_candidates:
        expected_count = len(expected_ranks) * len(expected_candidates)
        if len(command_rows) != expected_count and not args.allow_incomplete:
            raise RuntimeError(f"Expected {expected_count} command rows, found {len(command_rows)}.")
    failures = [row for row in command_rows if as_int(row.get("returncode")) != 0]
    missing_done = [row for row in command_rows if str(row.get("done_exists")).lower() != "true"]
    if (failures or missing_done or missing_payloads) and not args.allow_incomplete:
        raise RuntimeError(
            "Incomplete timing analysis inputs: "
            f"failures={len(failures)}, missing_done={len(missing_done)}, missing_payloads={len(missing_payloads)}"
        )


def main() -> None:
    args = parse_args()
    command_path = Path(args.out_root) / f"command_timing_{args.phase}.csv"
    command_rows = read_csv(command_path)
    if not command_rows:
        raise RuntimeError(f"No command timing rows found at {command_path}")
    summaries, timings, missing_payloads = load_run_payloads(command_rows)
    check_expected(args, command_rows, missing_payloads)

    reconciliation_rows = flatten_reconciliation(command_rows, summaries, timings)
    stage_rows = flatten_stage_rows(command_rows, summaries, timings)
    method_rows = flatten_method_rows(command_rows, summaries, timings, stage_rows)
    paired_rows = comparison_rows(method_rows)

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(Path(args.review_root) / "timing_reconciliation.csv", reconciliation_rows)
    c.write_csv(Path(args.review_root) / "timing_stage_breakdown.csv", stage_rows)
    c.write_csv(Path(args.review_root) / "timing_method_comparison.csv", method_rows)
    c.write_csv(Path(args.review_root) / "timing_paired_comparison.csv", paired_rows)
    fit_rows = write_linear_fits(Path(args.review_root), stage_rows)
    report = report_markdown(
        args=args,
        command_rows=command_rows,
        method_rows=method_rows,
        stage_rows=stage_rows,
        reconciliation_rows=reconciliation_rows,
        paired_rows=paired_rows,
        fit_rows=fit_rows,
    )
    (Path(args.review_root) / "timing_report.md").write_text(report + "\n")
    c.write_json(
        Path(args.review_root) / "analysis_summary.json",
        {
            "completed_at": c.now_text(),
            "phase": str(args.phase),
            "command_row_count": len(command_rows),
            "method_row_count": len(method_rows),
            "stage_row_count": len(stage_rows),
            "reconciliation_row_count": len(reconciliation_rows),
            "paired_row_count": len(paired_rows),
            "missing_payloads": list(missing_payloads),
        },
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
