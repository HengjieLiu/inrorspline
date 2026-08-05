#!/usr/bin/env python
"""Analyze DirectCP 8-4-2 DVF-only timing runs."""

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
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_dvf_only_timing"
DEFAULT_PREVIOUS_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_dvf_only_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--phase", type=str, default="rank0")
    parser.add_argument("--expected-case-ranks", type=str, default="")
    parser.add_argument("--expected-candidates", type=str, default="")
    parser.add_argument("--previous-review-root", type=Path, default=DEFAULT_PREVIOUS_REVIEW_ROOT)
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


def grouped(rows: Iterable[Dict[str, object]], keys: Sequence[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    out: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key) for key in keys)].append(row)
    return out


def mean_value(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def format_sec(value: object) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    if number != 0.0 and abs(number) < 1.0:
        return f"{number * 1000.0:.1f}ms"
    return f"{number:.2f}s"


def format_float(value: object, digits: int = 4) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def event_seconds(timing: Dict[str, object]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    events = timing.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            elapsed = as_float(event.get("elapsed_sec"), 0.0)
            name = str(event.get("name", ""))
            category = str(event.get("category", ""))
            totals[name] += elapsed
            if category:
                totals[f"category:{category}"] += elapsed
    return dict(totals)


def event_by_name(timing: Dict[str, object], name: str) -> Optional[Dict[str, object]]:
    events = timing.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and str(event.get("name")) == str(name):
                return event
    return None


def event_count_matching(timing: Dict[str, object], patterns: Sequence[str]) -> int:
    events = timing.get("events", [])
    count = 0
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            name = str(event.get("name", ""))
            category = str(event.get("category", ""))
            if any(pattern in name or pattern in category for pattern in patterns):
                count += 1
    return count


def mark_offsets(timing: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    marks = timing.get("marks", [])
    if isinstance(marks, list):
        for mark in marks:
            if isinstance(mark, dict):
                out[str(mark.get("name", ""))] = as_float(mark.get("offset_sec"))
    return out


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


def command_timing_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / f"command_timing_dvf_only_{args.phase}.csv"


def check_expected(args: argparse.Namespace, command_rows: Sequence[Dict[str, str]], missing_payloads: Sequence[str]) -> None:
    expected_ranks = parse_tokens(str(args.expected_case_ranks))
    expected_candidates = parse_tokens(str(args.expected_candidates))
    if expected_ranks and expected_candidates:
        expected_count = len(expected_ranks) * len(expected_candidates)
        if len(command_rows) != expected_count and not args.allow_incomplete:
            raise RuntimeError(f"Expected {expected_count} command rows, found {len(command_rows)}.")
    failures = [row for row in command_rows if as_int(row.get("returncode")) != 0]
    missing_done = [row for row in command_rows if str(row.get("done_exists")).lower() != "true"]
    missing_dvf = [row for row in command_rows if str(row.get("dense_dvf_exists")).lower() != "true"]
    if (failures or missing_done or missing_dvf or missing_payloads) and not args.allow_incomplete:
        raise RuntimeError(
            "Incomplete DVF-only timing analysis inputs: "
            f"failures={len(failures)}, missing_done={len(missing_done)}, "
            f"missing_dvf={len(missing_dvf)}, missing_payloads={len(missing_payloads)}"
        )


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
                "reconciliation_residual_sec": 0.0 if math.isfinite(script_total) else math.nan,
                "tolerance_sec": max(1.0, 0.01 * max(abs(external_wall), 1e-9)),
                "within_tolerance": math.isfinite(script_total),
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


def prolongation_by_target_stage(summary: Dict[str, object]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = summary.get("prolongation_diagnostics", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                stage_idx = as_int(row.get("stage_idx"))
                out[stage_idx]["total"] += as_float(row.get("timing_prolongation_sec"), 0.0)
                out[stage_idx]["pure_knot"] += as_float(row.get("timing_pure_knot_insertion_sec"), 0.0)
                out[stage_idx]["preserve_dense"] += as_float(
                    row.get("timing_prolongation_preserve_dense_expand_sec"),
                    0.0,
                )
                out[stage_idx]["preserve_metrics"] += as_float(
                    row.get("timing_prolongation_preserve_metrics_sec"),
                    0.0,
                )
    return {stage_idx: dict(values) for stage_idx, values in out.items()}


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
        stage_summaries = summary.get("stage_summaries", [])
        if not isinstance(stage_summaries, list):
            continue
        incoming_prolong = prolongation_by_target_stage(summary)
        outgoing_prolong = {stage_idx - 1: values for stage_idx, values in incoming_prolong.items() if stage_idx > 0}
        last_stage_idx = max([as_int(stage.get("stage_idx")) for stage in stage_summaries if isinstance(stage, dict)] or [0])
        for stage in stage_summaries:
            if not isinstance(stage, dict):
                continue
            stage_idx = as_int(stage.get("stage_idx"))
            tb = stage.get("timing_breakdown", {})
            if not isinstance(tb, dict):
                tb = {}
            loss_volume = as_float(tb.get("loss_volume_setup_sec"), 0.0)
            stage_setup = as_float(tb.get("stage_setup_sec"), 0.0)
            optimizer_loop = as_float(tb.get("optimizer_loop_sec"), 0.0)
            intermediate_dense = as_float(tb.get("intermediate_dense_expand_sec"), 0.0)
            final_output_dense = as_float(tb.get("final_output_dense_expand_sec"), 0.0)
            legacy_dense = as_float(tb.get("final_dense_expand_sec"), 0.0)
            control_stats = as_float(tb.get("control_summary_stats_sec"), 0.0)
            incoming_values = incoming_prolong.get(stage_idx, {})
            outgoing_values = outgoing_prolong.get(stage_idx, {})
            incoming = as_float(incoming_values.get("total"), 0.0)
            outgoing_total = as_float(outgoing_values.get("total"), 0.0)
            pure_knot = as_float(outgoing_values.get("pure_knot"), 0.0)
            preserve_dense = as_float(outgoing_values.get("preserve_dense"), 0.0)
            preserve_metrics = as_float(outgoing_values.get("preserve_metrics"), 0.0)
            pre = loss_volume + stage_setup
            diagnostic_post = intermediate_dense + preserve_dense + preserve_metrics
            post_required_to_dvf = pure_knot + final_output_dense
            post_all = post_required_to_dvf + diagnostic_post + control_stats
            post_to_dvf = post_required_to_dvf + diagnostic_post + (
                0.0 if stage_idx == last_stage_idx else control_stats
            )
            rows.append(
                {
                    "phase": cmd_row.get("phase", ""),
                    "candidate": cmd_row.get("candidate", ""),
                    "case_rank": as_int(cmd_row.get("case_rank")),
                    "case_sequence_index": cmd_row.get("case_sequence_index"),
                    "warm_status": cmd_row.get("warm_status"),
                    "persistent_process": cmd_row.get("persistent_process"),
                    "run_dir": run_dir,
                    "stage_number": int(stage_idx) + 1,
                    "stage_idx": stage_idx,
                    "full_cps": stage.get("full_cps"),
                    "downsample": stage.get("downsample"),
                    "lr": stage.get("lr"),
                    "requested_epochs": as_int(stage.get("requested_epochs")),
                    "actual_epochs": as_int(stage.get("actual_epochs")),
                    "stop_reason": stage.get("stop_reason"),
                    "raw_stage_total_sec": as_float(tb.get("stage_total_sec"), 0.0),
                    "pre_stage_setup_sec": pre,
                    "loss_volume_setup_sec": loss_volume,
                    "stage_setup_sec": stage_setup,
                    "incoming_prolongation_raw_sec": incoming,
                    "optimizer_loop_sec": optimizer_loop,
                    "post_stage_processing_sec": post_all,
                    "post_stage_processing_to_dvf_sec": post_to_dvf,
                    "post_stage_after_dvf_metadata_sec": control_stats if stage_idx == last_stage_idx else 0.0,
                    "outgoing_prolongation_or_knot_insertion_sec": outgoing_total,
                    "pure_knot_insertion_sec": pure_knot,
                    "prolongation_preserve_dense_expand_sec": preserve_dense,
                    "prolongation_preserve_metrics_sec": preserve_metrics,
                    "intermediate_dense_expand_sec": intermediate_dense,
                    "final_output_dense_expand_sec": final_output_dense,
                    "final_dense_expansion_sec": legacy_dense,
                    "control_summary_stats_sec": control_stats,
                    "diagnostic_dense_overhead_sec": diagnostic_post,
                    "non_loop_training_sec": pre + post_all,
                    "stage_grouped_total_sec": pre + optimizer_loop + post_all,
                    "stage_grouped_to_dvf_sec": pre + optimizer_loop + post_to_dvf,
                    "optimizer_loop_sec_per_epoch": optimizer_loop / as_int(stage.get("actual_epochs"))
                    if as_int(stage.get("actual_epochs")) > 0
                    else math.nan,
                    "script_total_sec": as_float(timing.get("script_total_sec"), math.nan),
                }
            )
    return rows


def flatten_method_rows(
    command_rows: Sequence[Dict[str, str]],
    summaries: Dict[str, Dict[str, object]],
    timings: Dict[str, Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    by_run: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in stage_rows:
        by_run[str(row.get("run_dir"))].append(row)

    rows: List[Dict[str, object]] = []
    for cmd_row in command_rows:
        run_dir = str(cmd_row.get("run_dir", ""))
        summary = summaries.get(run_dir, {})
        timing = timings.get(run_dir, {})
        events = event_seconds(timing)
        marks = mark_offsets(timing)
        case_load_complete = marks.get("case_load_complete", math.nan)
        dvf_mem = marks.get("dvf_in_memory_ready", math.nan)
        dvf_file = marks.get("dvf_file_ready", math.nan)
        run_stage_rows = by_run.get(run_dir, [])
        total_optimizer = sum(as_float(row.get("optimizer_loop_sec"), 0.0) for row in run_stage_rows)
        training = events.get("training", as_float(summary.get("training_elapsed_sec"), math.nan))
        total_non_loop_training = training - total_optimizer if math.isfinite(training) else math.nan
        external_wall = as_float(cmd_row.get("external_wall_sec"), math.nan)
        script_total = as_float(timing.get("script_total_sec"), math.nan)
        full_file_minus_mem = dvf_file - dvf_mem if math.isfinite(dvf_file) and math.isfinite(dvf_mem) else math.nan
        dvf_file_write = events.get("method_execution.dvf_file_write", 0.0)
        rows.append(
            {
                "phase": cmd_row.get("phase", ""),
                "candidate": cmd_row.get("candidate", ""),
                "case_rank": as_int(cmd_row.get("case_rank")),
                "case_sequence_index": cmd_row.get("case_sequence_index"),
                "warm_status": cmd_row.get("warm_status"),
                "persistent_process": cmd_row.get("persistent_process"),
                "original_case_index": cmd_row.get("original_case_index"),
                "fixed_id": cmd_row.get("fixed_id"),
                "moving_id": cmd_row.get("moving_id"),
                "run_dir": run_dir,
                "returncode": as_int(cmd_row.get("returncode")),
                "done_exists": cmd_row.get("done_exists"),
                "summary_exists": cmd_row.get("summary_exists"),
                "timing_exists": cmd_row.get("timing_exists"),
                "dense_dvf_exists": cmd_row.get("dense_dvf_exists"),
                "dense_dvf_path": cmd_row.get("dense_dvf_path"),
                "external_wall_sec": external_wall,
                "persistent_case_wall_sec": cmd_row.get("persistent_case_wall_sec"),
                "script_total_sec": script_total,
                "external_uninstrumented_sec": external_wall - script_total
                if math.isfinite(external_wall) and math.isfinite(script_total)
                else math.nan,
                "time_before_case_load_complete_sec": case_load_complete,
                "full_script_to_dvf_in_memory_sec": dvf_mem,
                "loaded_images_to_dvf_in_memory_sec": dvf_mem - case_load_complete
                if math.isfinite(dvf_mem) and math.isfinite(case_load_complete)
                else math.nan,
                "full_script_to_dvf_file_sec": dvf_file,
                "loaded_images_to_dvf_file_sec": dvf_file - case_load_complete
                if math.isfinite(dvf_file) and math.isfinite(case_load_complete)
                else math.nan,
                "dvf_file_write_sec": dvf_file_write,
                "small_file_prep_and_post_dvf_overhead_sec": full_file_minus_mem - dvf_file_write
                if math.isfinite(full_file_minus_mem)
                else math.nan,
                "setup_sec": events.get("setup", math.nan),
                "case_load_sec": events.get("setup.case_load", math.nan),
                "method_execution_sec": events.get("method_execution", math.nan),
                "initial_dense_grid_sec": events.get("training.initial_dense_grid", math.nan),
                "training_sec": training,
                "post_dvf_metadata_write_sec": events.get("post_dvf_metadata_write", 0.0),
                "dvf_file_prep_sec": events.get("method_execution.dvf_file_prep", 0.0),
                "total_optimizer_loop_sec": total_optimizer,
                "total_non_loop_training_sec": total_non_loop_training,
                "total_pre_stage_setup_sec": sum(as_float(row.get("pre_stage_setup_sec"), 0.0) for row in run_stage_rows),
                "total_post_stage_processing_sec": sum(as_float(row.get("post_stage_processing_sec"), 0.0) for row in run_stage_rows),
                "total_post_stage_processing_to_dvf_sec": sum(
                    as_float(row.get("post_stage_processing_to_dvf_sec"), 0.0) for row in run_stage_rows
                ),
                "total_prolongation_or_knot_insertion_sec": sum(
                    as_float(row.get("outgoing_prolongation_or_knot_insertion_sec"), 0.0) for row in run_stage_rows
                ),
                "total_pure_knot_insertion_sec": sum(
                    as_float(row.get("pure_knot_insertion_sec"), 0.0) for row in run_stage_rows
                ),
                "total_prolongation_preserve_dense_expand_sec": sum(
                    as_float(row.get("prolongation_preserve_dense_expand_sec"), 0.0) for row in run_stage_rows
                ),
                "total_prolongation_preserve_metrics_sec": sum(
                    as_float(row.get("prolongation_preserve_metrics_sec"), 0.0) for row in run_stage_rows
                ),
                "total_intermediate_dense_expand_sec": sum(
                    as_float(row.get("intermediate_dense_expand_sec"), 0.0) for row in run_stage_rows
                ),
                "total_final_output_dense_expand_sec": sum(
                    as_float(row.get("final_output_dense_expand_sec"), 0.0) for row in run_stage_rows
                ),
                "total_final_dense_expansion_sec": sum(
                    as_float(row.get("final_dense_expansion_sec"), 0.0) for row in run_stage_rows
                ),
                "total_diagnostic_dense_overhead_sec": sum(
                    as_float(row.get("diagnostic_dense_overhead_sec"), 0.0) for row in run_stage_rows
                ),
                "total_dvf_file_write_sec": dvf_file_write,
                "actual_epochs": summary.get("actual_epochs"),
                "stage_actual_epochs": summary.get("stage_actual_epochs"),
                "requested_epochs": summary.get("requested_epochs"),
                "training_peak_mem_gb": summary.get("training_peak_mem_gb"),
                "evaluation_event_count": event_count_matching(timing, ["post_train_evaluation", "dice", "hd95", "asd", "dense_ncc"]),
                "full_write_outputs_event_count": event_count_matching(timing, ["write_outputs"]),
                "non_target_gpu_apps_pre_count": cmd_row.get("non_target_gpu_apps_pre_count"),
                "non_target_gpu_apps_post_count": cmd_row.get("non_target_gpu_apps_post_count"),
            }
        )
    return rows


def grouped_breakdown_rows(method_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = [
        "total_optimizer_loop_sec",
        "total_non_loop_training_sec",
        "total_pre_stage_setup_sec",
        "total_post_stage_processing_sec",
        "total_post_stage_processing_to_dvf_sec",
        "total_prolongation_or_knot_insertion_sec",
        "total_pure_knot_insertion_sec",
        "total_prolongation_preserve_dense_expand_sec",
        "total_prolongation_preserve_metrics_sec",
        "total_intermediate_dense_expand_sec",
        "total_final_output_dense_expand_sec",
        "total_final_dense_expansion_sec",
        "total_diagnostic_dense_overhead_sec",
        "total_dvf_file_write_sec",
        "dvf_file_prep_sec",
        "post_dvf_metadata_write_sec",
    ]
    rows: List[Dict[str, object]] = []
    for row in method_rows:
        for key in keys:
            rows.append(
                {
                    "phase": row.get("phase"),
                    "candidate": row.get("candidate"),
                    "case_rank": row.get("case_rank"),
                    "run_dir": row.get("run_dir"),
                    "group": key.removesuffix("_sec"),
                    "elapsed_sec": row.get(key),
                }
            )
    return rows


def previous_conservative_rows(
    previous_review_root: Path,
    current_method_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    path = Path(previous_review_root) / "dvf_only_method_comparison.csv"
    previous = read_csv(path)
    if not previous:
        return []
    prev_by_candidate = {str(row.get("candidate")): row for row in previous}
    rows: List[Dict[str, object]] = []
    for current in current_method_rows:
        candidate = str(current.get("candidate"))
        prev = prev_by_candidate.get(candidate)
        if prev is None:
            continue
        for metric in [
            "full_script_to_dvf_in_memory_sec",
            "loaded_images_to_dvf_in_memory_sec",
            "full_script_to_dvf_file_sec",
            "loaded_images_to_dvf_file_sec",
            "total_non_loop_training_sec",
            "total_optimizer_loop_sec",
            "external_wall_sec",
        ]:
            rows.append(
                {
                    "candidate": candidate,
                    "metric": metric.removesuffix("_sec"),
                    "previous_conservative_sec": prev.get(metric),
                    "current_production_sec": current.get(metric),
                    "previous_minus_current_sec": as_float(prev.get(metric)) - as_float(current.get(metric)),
                    "previous_review_root": str(previous_review_root),
                }
            )
    return rows


def paired_savings(method_rows: Sequence[Dict[str, object]], stage_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_case = grouped(method_rows, ["case_rank"])
    for (case_rank,), case_rows in by_case.items():
        by_candidate = {str(row.get("candidate")): row for row in case_rows}
        original = by_candidate.get("original_mrdbscp")
        fast = by_candidate.get("A_fast_decay")
        if original is None or fast is None:
            continue
        for metric in [
            "full_script_to_dvf_in_memory_sec",
            "loaded_images_to_dvf_in_memory_sec",
            "full_script_to_dvf_file_sec",
            "loaded_images_to_dvf_file_sec",
            "dvf_file_write_sec",
            "external_wall_sec",
            "external_uninstrumented_sec",
            "total_optimizer_loop_sec",
            "total_non_loop_training_sec",
            "total_pure_knot_insertion_sec",
            "total_prolongation_preserve_dense_expand_sec",
            "total_prolongation_preserve_metrics_sec",
            "total_intermediate_dense_expand_sec",
            "total_final_output_dense_expand_sec",
            "total_diagnostic_dense_overhead_sec",
        ]:
            rows.append(
                {
                    "case_rank": case_rank,
                    "component": metric.removesuffix("_sec"),
                    "original_sec": original.get(metric),
                    "A_fast_decay_sec": fast.get(metric),
                    "original_minus_A_fast_decay_sec": as_float(original.get(metric)) - as_float(fast.get(metric)),
                }
            )
    stage_by_key = grouped(stage_rows, ["case_rank", "stage_idx"])
    for (case_rank, stage_idx), rows_for_stage in stage_by_key.items():
        by_candidate = {str(row.get("candidate")): row for row in rows_for_stage}
        original = by_candidate.get("original_mrdbscp")
        fast = by_candidate.get("A_fast_decay")
        if original is None or fast is None:
            continue
        for metric in [
            "pre_stage_setup_sec",
            "optimizer_loop_sec",
            "pure_knot_insertion_sec",
            "prolongation_preserve_dense_expand_sec",
            "prolongation_preserve_metrics_sec",
            "intermediate_dense_expand_sec",
            "final_output_dense_expand_sec",
            "post_stage_processing_to_dvf_sec",
            "post_stage_processing_sec",
        ]:
            rows.append(
                {
                    "case_rank": case_rank,
                    "component": f"stage_{int(stage_idx) + 1}_{metric.removesuffix('_sec')}",
                    "original_sec": original.get(metric),
                    "A_fast_decay_sec": fast.get(metric),
                    "original_minus_A_fast_decay_sec": as_float(original.get(metric)) - as_float(fast.get(metric)),
                }
            )
    return rows


def endpoint_residual(row: Dict[str, object], stage_rows: Sequence[Dict[str, object]]) -> float:
    run_dir = str(row.get("run_dir"))
    stages = [stage for stage in stage_rows if str(stage.get("run_dir")) == run_dir]
    stage_sum = sum(as_float(stage.get("stage_grouped_to_dvf_sec"), 0.0) for stage in stages)
    components = as_float(row.get("setup_sec"), 0.0) + as_float(row.get("initial_dense_grid_sec"), 0.0) + stage_sum
    return as_float(row.get("full_script_to_dvf_in_memory_sec")) - components


def stage_rows_for_run(stage_rows: Sequence[Dict[str, object]], run_dir: object) -> List[Dict[str, object]]:
    return sorted(
        [row for row in stage_rows if str(row.get("run_dir")) == str(run_dir)],
        key=lambda row: as_int(row.get("stage_idx")),
    )


def worked_example_lines(row: Dict[str, object], stage_rows: Sequence[Dict[str, object]]) -> List[str]:
    stages = stage_rows_for_run(stage_rows, row.get("run_dir"))
    residual = endpoint_residual(row, stage_rows)
    lines = [
        "```text",
        f"{row.get('candidate')} full_script_to_dvf_in_memory",
        f"  = setup {as_float(row.get('setup_sec')):.2f}",
        f"  + initial_dense_grid/small setup {as_float(row.get('initial_dense_grid_sec')):.2f}",
    ]
    for stage in stages:
        lines.extend(
            [
                f"  + stage {stage.get('stage_number')} pre {as_float(stage.get('pre_stage_setup_sec')):.2f}",
                f"  + stage {stage.get('stage_number')} optimizer {as_float(stage.get('optimizer_loop_sec')):.2f}",
                f"  + stage {stage.get('stage_number')} post-to-DVF {as_float(stage.get('post_stage_processing_to_dvf_sec')):.2f}",
            ]
        )
    lines.append(f"  + other pre-DVF measured residual {residual:.2f}")
    lines.append(f"  = {as_float(row.get('full_script_to_dvf_in_memory_sec')):.2f}")
    lines.append("")
    file_overhead = (
        as_float(row.get("full_script_to_dvf_file_sec"))
        - as_float(row.get("full_script_to_dvf_in_memory_sec"))
        - as_float(row.get("dvf_file_write_sec"))
    )
    lines.append(f"{row.get('candidate')} full_script_to_dvf_file")
    lines.append(f"  = full_script_to_dvf_in_memory {as_float(row.get('full_script_to_dvf_in_memory_sec')):.2f}")
    lines.append(f"  + dvf_file_write {as_float(row.get('dvf_file_write_sec')):.2f}")
    lines.append(f"  + small file-prep/post-DVF overhead before file-ready {file_overhead:.2f}")
    lines.append(f"  = {as_float(row.get('full_script_to_dvf_file_sec')):.2f}")
    lines.append("```")
    return lines


def report_markdown(
    *,
    args: argparse.Namespace,
    command_rows: Sequence[Dict[str, str]],
    method_rows: Sequence[Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
    grouped_rows: Sequence[Dict[str, object]],
    reconciliation_rows: Sequence[Dict[str, object]],
    savings_rows: Sequence[Dict[str, object]],
    previous_rows: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# OASIS DVF-Only Runtime Timing")
    lines.append("")
    lines.append(f"- Phase: `{args.phase}`")
    lines.append(f"- Output root: `{args.out_root}`")
    lines.append(f"- Review root: `{args.review_root}`")
    lines.append(f"- Completed command rows: {len(command_rows)}")
    lines.append(f"- Timing JSON rows: {sum(1 for row in method_rows if str(row.get('timing_exists')).lower() == 'true')}/{len(method_rows)}")
    lines.append(f"- Dense DVF files present: {sum(1 for row in method_rows if str(row.get('dense_dvf_exists')).lower() == 'true')}/{len(method_rows)}")
    failures = [row for row in method_rows if as_int(row.get("returncode")) != 0]
    lines.append(f"- Failed subprocesses: {len(failures)}")
    non_target = sum(
        as_int(row.get("non_target_gpu_apps_pre_count")) + as_int(row.get("non_target_gpu_apps_post_count"))
        for row in method_rows
    )
    lines.append(f"- Non-target GPU compute-app observations: {non_target}")
    eval_events = sum(as_int(row.get("evaluation_event_count")) + as_int(row.get("full_write_outputs_event_count")) for row in method_rows)
    lines.append(f"- Evaluation/full-output events counted in DVF-only runs: {eval_events}")
    lines.append("")

    bad_recon = [row for row in reconciliation_rows if not bool(row.get("within_tolerance", False))]
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
    lines.append("")

    lines.append("## Timing Hierarchy And Definitions")
    lines.append("")
    lines.append("Conceptual hierarchy:")
    lines.append("")
    lines.append("```text")
    lines.append("external_wall")
    lines.append("  script_total")
    lines.append("    setup")
    lines.append("      device")
    lines.append("      seed")
    lines.append("      case_load")
    lines.append("      case_validate_print")
    lines.append("      sampler_init")
    lines.append("    method_execution")
    lines.append("      initial_dense_grid / small setup")
    lines.append("      training")
    lines.append("        stage 1")
    lines.append("          pre_stage_setup")
    lines.append("          optimizer_loop")
    lines.append("          post_stage_processing")
    lines.append("        stage 2")
    lines.append("          pre_stage_setup")
    lines.append("          optimizer_loop")
    lines.append("          post_stage_processing")
    lines.append("        stage 3")
    lines.append("          pre_stage_setup")
    lines.append("          optimizer_loop")
    lines.append("          post_stage_processing")
    lines.append("            final_dense_expand = dvf_in_memory_ready")
    lines.append("      dvf_file_write = dvf_file_ready")
    lines.append("      small metadata assembly")
    lines.append("    post_dvf_metadata_write")
    lines.append("  external_uninstrumented")
    lines.append("```")
    lines.append("")
    lines.append("Definitions:")
    lines.append("")
    lines.append("- `external_wall`: launcher-measured subprocess wall time.")
    lines.append("- `script_total`: in-script measured wall time from `run_one.py` timing start to timing finalization.")
    lines.append("- `full_script_to_dvf_in_memory`: script timer start to the final dense displacement tensor being ready in memory.")
    lines.append("- `loaded_images_to_dvf_in_memory`: after case/image load completes to the in-memory dense DVF endpoint.")
    lines.append("- `full_script_to_dvf_file`: script timer start to the dense DVF `.npy` being written.")
    lines.append("- `loaded_images_to_dvf_file`: after case/image load completes to the dense DVF `.npy` being written.")
    lines.append("- `dvf_file_write`: the timed `numpy.save` write of the normalized-grid dense displacement array only.")
    lines.append("- `optimizer_loop`: epoch-loop work. It is a strict subset of `training`.")
    lines.append("- `training`: the optimization block, including non-loop stage work; DirectCP uses three stages and SINR uses one.")
    lines.append("- `non_loop_training`: stage setup, pure knot insertion, optional diagnostic dense checks/saves, final output dense expansion, control stats, and other non-epoch work.")
    lines.append("- `external_uninstrumented`: process startup/import/argparse/exit and timing-artifact overhead outside `script_total`.")
    lines.append("- For SINR cps4/cps2, the hierarchy collapses to one fixed-cps stage: no prolongation, no knot insertion, and no intermediate dense diagnostic expansion.")
    lines.append("")
    lines.append("Inclusion/exclusion rules:")
    lines.append("")
    lines.append("- DVF-only headline times include setup, image/case load for full-script starts, method optimization, final dense expansion, and optionally dense DVF file write.")
    lines.append("- DVF-only headline times do not include image warp evaluation, segmentation warp, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, or full CSV packaging.")
    lines.append("- Previous `fixed/io known` is not the DVF-only headline because it included evaluation and regularity/output packaging that are intentionally skipped here.")
    lines.append("- Intermediate dense expansion is not required to start the next stage. Production DVF timing skips it by default.")
    lines.append("- `pure_knot_insertion` means control-grid knot insertion only. Optional preservation dense expansion and metrics are reported separately.")
    lines.append("- SINR has no multistage knot insertion/prolongation; its post-to-DVF component is final model output plus final dense B-spline expansion.")
    lines.append("- Dense expansion and knot insertion are already PyTorch tensor operations. In normal GPU DirectCP runs they operate on CUDA tensors, so the main avoidable cost here is diagnostic dense expansion/preservation checking, not missing GPU acceleration.")
    lines.append("- The final-stage `dvf_in_memory_ready` mark is recorded immediately after final output dense expansion. Final control-summary stats may occur after that mark, so they are part of total `non_loop_training` but not part of the in-memory DVF endpoint.")
    lines.append("")
    lines.append("Accounting formulas:")
    lines.append("")
    lines.append("```text")
    lines.append("optimizer_loop is a strict subset of training.")
    lines.append("")
    lines.append("training =")
    lines.append("  total_optimizer_loop")
    lines.append("  + total_non_loop_training")
    lines.append("")
    lines.append("full_script_to_dvf_in_memory =")
    lines.append("  setup")
    lines.append("  + initial_dense_grid / small setup")
    lines.append("  + training_to_final_dense_expand")
    lines.append("")
    lines.append("full_script_to_dvf_file =")
    lines.append("  full_script_to_dvf_in_memory")
    lines.append("  + dvf_file_write")
    lines.append("  + small file-prep overhead")
    lines.append("")
    lines.append("loaded_images_to_dvf_in_memory =")
    lines.append("  full_script_to_dvf_in_memory")
    lines.append("  - time_before_case_load_complete")
    lines.append("")
    lines.append("loaded_images_to_dvf_file =")
    lines.append("  full_script_to_dvf_file")
    lines.append("  - time_before_case_load_complete")
    lines.append("")
    lines.append("external_wall =")
    lines.append("  script_total")
    lines.append("  + external_uninstrumented")
    lines.append("```")
    lines.append("")

    lines.append("## Headline Times")
    lines.append("")
    lines.append("| candidate | full script -> DVF memory | loaded images -> DVF memory | full script -> DVF file | loaded images -> DVF file | DVF file write | external wall | external uninstrumented | epochs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(method_rows, key=lambda item: str(item.get("candidate"))):
        lines.append(
            "| "
            + str(row.get("candidate"))
            + f" | {format_sec(row.get('full_script_to_dvf_in_memory_sec'))}"
            + f" | {format_sec(row.get('loaded_images_to_dvf_in_memory_sec'))}"
            + f" | {format_sec(row.get('full_script_to_dvf_file_sec'))}"
            + f" | {format_sec(row.get('loaded_images_to_dvf_file_sec'))}"
            + f" | {format_sec(row.get('dvf_file_write_sec'))}"
            + f" | {format_sec(row.get('external_wall_sec'))}"
            + f" | {format_sec(row.get('external_uninstrumented_sec'))}"
            + f" | {format_float(row.get('actual_epochs'), 1)}"
            + " |"
        )
    lines.append("")

    lines.append("## Grouped Totals")
    lines.append("")
    lines.append("| candidate | optimizer loop | non-loop training | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | DVF file write |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(method_rows, key=lambda item: str(item.get("candidate"))):
        lines.append(
            "| "
            + str(row.get("candidate"))
            + f" | {format_sec(row.get('total_optimizer_loop_sec'))}"
            + f" | {format_sec(row.get('total_non_loop_training_sec'))}"
            + f" | {format_sec(row.get('total_pure_knot_insertion_sec'))}"
            + f" | {format_sec(row.get('total_prolongation_preserve_dense_expand_sec'))}"
            + f" | {format_sec(row.get('total_prolongation_preserve_metrics_sec'))}"
            + f" | {format_sec(row.get('total_intermediate_dense_expand_sec'))}"
            + f" | {format_sec(row.get('total_final_output_dense_expand_sec'))}"
            + f" | {format_sec(row.get('total_dvf_file_write_sec'))}"
            + " |"
        )
    lines.append("")

    lines.append("## Worked Numeric Examples")
    lines.append("")
    for row in sorted(method_rows, key=lambda item: str(item.get("candidate"))):
        lines.extend(worked_example_lines(row, stage_rows))
        lines.append("")

    if savings_rows:
        lines.append("## Direct Savings")
        lines.append("")
        lines.append("Savings are `original_mrdbscp - A_fast_decay`; positive values mean `A_fast_decay` is faster.")
        lines.append("")
        lines.append("| component | savings |")
        lines.append("|---|---:|")
        for row in savings_rows:
            lines.append(f"| {row.get('component')} | {format_sec(row.get('original_minus_A_fast_decay_sec'))} |")
        lines.append("")

    if previous_rows:
        lines.append("## Previous Conservative Timing Comparison")
        lines.append("")
        lines.append("Positive values mean the new production path is faster than the previous conservative DVF-only timing.")
        lines.append("Endpoint rows are the primary comparison. Component rows can shift because skipping earlier dense expansions also changes where CUDA warmup/autotune cost appears.")
        lines.append("")
        lines.append("| candidate | metric | previous | production | previous - production |")
        lines.append("|---|---|---:|---:|---:|")
        for row in previous_rows:
            lines.append(
                "| "
                + str(row.get("candidate"))
                + f" | {row.get('metric')}"
                + f" | {format_sec(row.get('previous_conservative_sec'))}"
                + f" | {format_sec(row.get('current_production_sec'))}"
                + f" | {format_sec(row.get('previous_minus_current_sec'))}"
                + " |"
            )
        lines.append("")

    lines.append("## Stage Breakdown")
    lines.append("")
    lines.append("| candidate | stage | pre | optimizer | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | post-to-DVF | epochs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(stage_rows, key=lambda item: (str(item.get("candidate")), as_int(item.get("stage_idx")))):
        lines.append(
            "| "
            + str(row.get("candidate"))
            + f" | {row.get('stage_number')}"
            + f" | {format_sec(row.get('pre_stage_setup_sec'))}"
            + f" | {format_sec(row.get('optimizer_loop_sec'))}"
            + f" | {format_sec(row.get('pure_knot_insertion_sec'))}"
            + f" | {format_sec(row.get('prolongation_preserve_dense_expand_sec'))}"
            + f" | {format_sec(row.get('prolongation_preserve_metrics_sec'))}"
            + f" | {format_sec(row.get('intermediate_dense_expand_sec'))}"
            + f" | {format_sec(row.get('final_output_dense_expand_sec'))}"
            + f" | {format_sec(row.get('post_stage_processing_to_dvf_sec'))}"
            + f" | {format_float(row.get('actual_epochs'), 1)}"
            + " |"
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `dvf_only_timing_report.md`")
    lines.append("- `dvf_only_method_comparison.csv`")
    lines.append("- `dvf_only_stage_breakdown.csv`")
    lines.append("- `dvf_only_grouped_breakdown.csv`")
    lines.append("- `dvf_only_reconciliation.csv`")
    lines.append(f"- `command_timing_dvf_only_{args.phase}.csv`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    command_path = command_timing_path(args)
    command_rows = read_csv(command_path)
    if not command_rows:
        raise RuntimeError(f"No DVF-only command timing rows found at {command_path}")
    summaries, timings, missing_payloads = load_run_payloads(command_rows)
    check_expected(args, command_rows, missing_payloads)

    reconciliation_rows = flatten_reconciliation(command_rows, summaries, timings)
    stage_rows = flatten_stage_rows(command_rows, summaries, timings)
    method_rows = flatten_method_rows(command_rows, summaries, timings, stage_rows)
    grouped_rows = grouped_breakdown_rows(method_rows)
    savings_rows = paired_savings(method_rows, stage_rows)
    previous_rows = previous_conservative_rows(Path(args.previous_review_root), method_rows)

    eval_events = sum(as_int(row.get("evaluation_event_count")) + as_int(row.get("full_write_outputs_event_count")) for row in method_rows)
    if eval_events and not args.allow_incomplete:
        raise RuntimeError(f"DVF-only runs unexpectedly contain evaluation/full-output events: {eval_events}")

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(Path(args.review_root) / "dvf_only_reconciliation.csv", reconciliation_rows)
    c.write_csv(Path(args.review_root) / "dvf_only_stage_breakdown.csv", stage_rows)
    c.write_csv(Path(args.review_root) / "dvf_only_method_comparison.csv", method_rows)
    c.write_csv(Path(args.review_root) / "dvf_only_grouped_breakdown.csv", grouped_rows)
    c.write_csv(Path(args.review_root) / "dvf_only_direct_savings.csv", savings_rows)
    c.write_csv(Path(args.review_root) / "dvf_only_previous_conservative_comparison.csv", previous_rows)
    report = report_markdown(
        args=args,
        command_rows=command_rows,
        method_rows=method_rows,
        stage_rows=stage_rows,
        grouped_rows=grouped_rows,
        reconciliation_rows=reconciliation_rows,
        savings_rows=savings_rows,
        previous_rows=previous_rows,
    )
    (Path(args.review_root) / "dvf_only_timing_report.md").write_text(report + "\n")
    c.write_json(
        Path(args.review_root) / "dvf_only_analysis_summary.json",
        {
            "completed_at": c.now_text(),
            "phase": str(args.phase),
            "command_row_count": len(command_rows),
            "method_row_count": len(method_rows),
            "stage_row_count": len(stage_rows),
            "grouped_row_count": len(grouped_rows),
            "reconciliation_row_count": len(reconciliation_rows),
            "savings_row_count": len(savings_rows),
            "previous_conservative_row_count": len(previous_rows),
            "missing_payloads": list(missing_payloads),
            "evaluation_event_count": eval_events,
        },
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
