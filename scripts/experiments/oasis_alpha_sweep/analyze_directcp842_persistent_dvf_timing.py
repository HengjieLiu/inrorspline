#!/usr/bin/env python
"""Summarize persistent-process DirectCP 8-4-2 DVF-only timing."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing"
METRICS = [
    "full_script_to_dvf_in_memory_sec",
    "loaded_images_to_dvf_in_memory_sec",
    "full_script_to_dvf_file_sec",
    "loaded_images_to_dvf_file_sec",
    "external_wall_sec",
    "total_optimizer_loop_sec",
    "total_non_loop_training_sec",
    "actual_epochs",
]


def event_seconds(timing: Dict[str, object]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    events = timing.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            name = str(event.get("name", ""))
            totals[name] = totals.get(name, 0.0) + as_float(event.get("elapsed_sec"), 0.0)
    return totals


def load_timing(row: Dict[str, str]) -> Dict[str, object]:
    run_dir = Path(str(row.get("run_dir", "")))
    timing_path = run_dir / "timing.json"
    if not timing_path.exists():
        return {}
    payload = c.read_json(timing_path)
    return payload if isinstance(payload, dict) else {}


def phase_value(timing: Dict[str, object], stage_idx: int, name: str) -> float:
    summaries = timing.get("stage_phase_summaries", [])
    if not isinstance(summaries, list):
        return math.nan
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        if as_int(summary.get("stage_idx"), -1) != int(stage_idx):
            continue
        phases = summary.get("phase_totals_sec", {})
        if isinstance(phases, dict):
            return as_float(phases.get(name), 0.0)
    return math.nan


def phase_sum(timing: Dict[str, object], name: str) -> float:
    summaries = timing.get("stage_phase_summaries", [])
    total = 0.0
    found = False
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            phases = summary.get("phase_totals_sec", {})
            if isinstance(phases, dict) and name in phases:
                total += as_float(phases.get(name), 0.0)
                found = True
    return total if found else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--phase", type=str, default="first10")
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


def format_sec(value: object) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    if number != 0.0 and abs(number) < 1.0:
        return f"{number * 1000.0:.1f}ms"
    return f"{number:.2f}s"


def format_float(value: object, digits: int = 2) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
    }


def rows_for(rows: Sequence[Dict[str, str]], candidate: str, group: str) -> List[Dict[str, str]]:
    selected = [row for row in rows if str(row.get("candidate")) == str(candidate)]
    if group == "cold_rank0":
        return [row for row in selected if as_int(row.get("case_rank")) == 0 or str(row.get("warm_status")) == "cold_rank0"]
    if group == "warm_ranks1_9":
        return [row for row in selected if as_int(row.get("case_rank")) != 0]
    if group == "amortized_ranks0_9":
        return selected
    raise ValueError(group)


def build_summary_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    candidates = sorted({str(row.get("candidate")) for row in rows if row.get("candidate")})
    groups = ["cold_rank0", "warm_ranks1_9", "amortized_ranks0_9"]
    out: List[Dict[str, object]] = []
    for candidate in candidates:
        cold = rows_for(rows, candidate, "cold_rank0")
        warm = rows_for(rows, candidate, "warm_ranks1_9")
        for group in groups:
            group_rows = rows_for(rows, candidate, group)
            base: Dict[str, object] = {
                "candidate": candidate,
                "group": group,
                "case_count": len(group_rows),
                "case_ranks": ",".join(str(as_int(row.get("case_rank"))) for row in group_rows),
            }
            for metric in METRICS:
                stats = mean_std(as_float(row.get(metric)) for row in group_rows)
                base[f"{metric}_mean"] = stats["mean"]
                base[f"{metric}_std"] = stats["std"]
            out.append(base)
        if cold and warm:
            delta: Dict[str, object] = {
                "candidate": candidate,
                "group": "cold_minus_warm_mean",
                "case_count": len(cold),
                "case_ranks": "0_minus_1-9",
            }
            for metric in METRICS:
                cold_value = as_float(cold[0].get(metric))
                warm_stats = mean_std(as_float(row.get(metric)) for row in warm)
                delta[f"{metric}_mean"] = cold_value - warm_stats["mean"]
                delta[f"{metric}_std"] = math.nan
            out.append(delta)
    return out


def build_savings_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_group: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_group.setdefault(str(row.get("group")), {})[str(row.get("candidate"))] = dict(row)
    out: List[Dict[str, object]] = []
    for group, candidates in by_group.items():
        if "original_mrdbscp" not in candidates or "A_fast_decay" not in candidates:
            continue
        original = candidates["original_mrdbscp"]
        fast = candidates["A_fast_decay"]
        for metric in METRICS:
            key = f"{metric}_mean"
            out.append(
                {
                    "group": group,
                    "metric": metric,
                    "original_mean": original.get(key),
                    "A_fast_decay_mean": fast.get(key),
                    "original_minus_A_fast_decay": as_float(original.get(key)) - as_float(fast.get(key)),
                }
            )
    return out


def build_rank_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "candidate": row.get("candidate"),
                "case_rank": as_int(row.get("case_rank")),
                "original_case_index": row.get("original_case_index"),
                "fixed_id": row.get("fixed_id"),
                "moving_id": row.get("moving_id"),
                "warm_status": row.get("warm_status") or ("cold_rank0" if as_int(row.get("case_rank")) == 0 else "warm_persistent"),
                "full_script_to_dvf_in_memory_sec": row.get("full_script_to_dvf_in_memory_sec"),
                "loaded_images_to_dvf_in_memory_sec": row.get("loaded_images_to_dvf_in_memory_sec"),
                "full_script_to_dvf_file_sec": row.get("full_script_to_dvf_file_sec"),
                "loaded_images_to_dvf_file_sec": row.get("loaded_images_to_dvf_file_sec"),
                "external_wall_sec": row.get("external_wall_sec"),
                "total_optimizer_loop_sec": row.get("total_optimizer_loop_sec"),
                "total_non_loop_training_sec": row.get("total_non_loop_training_sec"),
                "actual_epochs": row.get("actual_epochs"),
                "stage_actual_epochs": row.get("stage_actual_epochs"),
                "evaluation_event_count": row.get("evaluation_event_count"),
                "dense_dvf_exists": row.get("dense_dvf_exists"),
            }
        )
    return sorted(out, key=lambda item: (str(item["candidate"]), as_int(item["case_rank"])))


def component_value(row: Dict[str, str], timing: Dict[str, object], component: str) -> float:
    events = event_seconds(timing)
    if component == "full_script_to_dvf_in_memory":
        return as_float(row.get("full_script_to_dvf_in_memory_sec"))
    if component == "loaded_images_to_dvf_in_memory":
        return as_float(row.get("loaded_images_to_dvf_in_memory_sec"))
    if component == "external_wall":
        return as_float(row.get("external_wall_sec"))
    if component == "setup":
        return as_float(row.get("setup_sec"))
    if component == "case_load":
        return as_float(row.get("case_load_sec"))
    if component == "method_execution":
        return as_float(row.get("method_execution_sec"))
    if component == "training":
        return as_float(row.get("training_sec"))
    if component == "optimizer_loop":
        return as_float(row.get("total_optimizer_loop_sec"))
    if component == "non_loop_training":
        return as_float(row.get("total_non_loop_training_sec"))
    if component == "loss_volume_setup_total":
        return sum(events.get(f"stage_{stage_idx}.loss_volume_setup", 0.0) for stage_idx in range(3))
    if component == "total_expand_phase":
        return phase_sum(timing, "expand")
    if component == "total_generator_forward_phase":
        return phase_sum(timing, "generator_forward")
    if component == "total_backward_step_phase":
        return phase_sum(timing, "backward_step")
    if component == "total_regularizer_phase":
        return phase_sum(timing, "regularizer")
    if component == "total_warp_ncc_phase":
        return phase_sum(timing, "warp_ncc")
    if component.startswith("stage_"):
        parts = component.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            stage_idx = int(parts[1]) - 1
            phase_name = "_".join(parts[2:])
            if phase_name == "optimizer_loop":
                return events.get(f"stage_{stage_idx}.optimizer_loop", math.nan)
            return phase_value(timing, stage_idx, phase_name)
    if component == "actual_epochs":
        return as_float(row.get("actual_epochs"))
    return math.nan


def build_cold_warm_breakdown_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    component_specs = [
        ("full_script_to_dvf_in_memory", "Full script -> DVF memory", "sec"),
        ("loaded_images_to_dvf_in_memory", "Loaded images -> DVF memory", "sec"),
        ("external_wall", "External wall", "sec"),
        ("setup", "Setup", "sec"),
        ("case_load", "Case/image load", "sec"),
        ("method_execution", "Method execution", "sec"),
        ("training", "Training", "sec"),
        ("optimizer_loop", "Raw optimizer loop", "sec"),
        ("non_loop_training", "Non-loop training", "sec"),
        ("loss_volume_setup_total", "Loss-volume setup total", "sec"),
        ("total_generator_forward_phase", "Total generator forward phase", "sec"),
        ("total_expand_phase", "Total optimizer expand phase", "sec"),
        ("total_backward_step_phase", "Total backward/step phase", "sec"),
        ("total_regularizer_phase", "Total regularizer phase", "sec"),
        ("total_warp_ncc_phase", "Total warp/NCC phase", "sec"),
        ("stage_1_optimizer_loop", "Stage 1 optimizer loop", "sec"),
        ("stage_1_generator_forward", "Stage 1 generator forward phase", "sec"),
        ("stage_1_expand", "Stage 1 expand phase", "sec"),
        ("stage_1_backward_step", "Stage 1 backward/step phase", "sec"),
        ("stage_1_regularizer", "Stage 1 regularizer phase", "sec"),
        ("stage_2_optimizer_loop", "Stage 2 optimizer loop", "sec"),
        ("stage_2_expand", "Stage 2 expand phase", "sec"),
        ("stage_2_backward_step", "Stage 2 backward/step phase", "sec"),
        ("stage_2_regularizer", "Stage 2 regularizer phase", "sec"),
        ("stage_3_optimizer_loop", "Stage 3 optimizer loop", "sec"),
        ("stage_3_expand", "Stage 3 expand phase", "sec"),
        ("stage_3_backward_step", "Stage 3 backward/step phase", "sec"),
        ("stage_3_regularizer", "Stage 3 regularizer phase", "sec"),
        ("actual_epochs", "Actual epochs", "count"),
    ]
    candidates = sorted({str(row.get("candidate")) for row in rows if row.get("candidate")})
    out: List[Dict[str, object]] = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if str(row.get("candidate")) == candidate]
        cold_rows = [row for row in candidate_rows if as_int(row.get("case_rank")) == 0]
        warm_rows = sorted(
            [row for row in candidate_rows if as_int(row.get("case_rank")) != 0],
            key=lambda item: as_int(item.get("case_rank")),
        )
        if not cold_rows or not warm_rows:
            continue
        cold = cold_rows[0]
        warm = warm_rows[0]
        cold_timing = load_timing(cold)
        warm_timing = load_timing(warm)
        for key, label, unit in component_specs:
            cold_value = component_value(cold, cold_timing, key)
            warm_value = component_value(warm, warm_timing, key)
            if not math.isfinite(cold_value) and not math.isfinite(warm_value):
                continue
            ratio = cold_value / warm_value if math.isfinite(cold_value) and math.isfinite(warm_value) and warm_value != 0 else math.nan
            out.append(
                {
                    "candidate": candidate,
                    "component": key,
                    "label": label,
                    "unit": unit,
                    "cold_rank": as_int(cold.get("case_rank")),
                    "warm_rank": as_int(warm.get("case_rank")),
                    "cold_value": cold_value,
                    "warm_value": warm_value,
                    "cold_minus_warm": cold_value - warm_value
                    if math.isfinite(cold_value) and math.isfinite(warm_value)
                    else math.nan,
                    "cold_div_warm": ratio,
                }
            )
    return out


def build_pairwise_runtime_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_group: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_group.setdefault(str(row.get("group")), {})[str(row.get("candidate"))] = dict(row)
    rows: List[Dict[str, object]] = []
    groups = ["cold_rank0", "warm_ranks1_9", "amortized_ranks0_9"]
    directcp_candidates = ["original_mrdbscp", "A_fast_decay"]
    sinr_candidates = ["sinr_cps4", "sinr_cps2"]
    metrics = [
        "full_script_to_dvf_in_memory_sec",
        "loaded_images_to_dvf_in_memory_sec",
        "full_script_to_dvf_file_sec",
        "external_wall_sec",
        "total_optimizer_loop_sec",
        "actual_epochs",
    ]
    for group in groups:
        candidates = by_group.get(group, {})
        for sinr in sinr_candidates:
            if sinr not in candidates:
                continue
            for directcp in directcp_candidates:
                if directcp not in candidates:
                    continue
                sinr_row = candidates[sinr]
                directcp_row = candidates[directcp]
                for metric in metrics:
                    key = f"{metric}_mean"
                    rows.append(
                        {
                            "group": group,
                            "sinr_candidate": sinr,
                            "directcp_candidate": directcp,
                            "metric": metric,
                            "sinr_mean": sinr_row.get(key),
                            "directcp_mean": directcp_row.get(key),
                            "sinr_minus_directcp": as_float(sinr_row.get(key)) - as_float(directcp_row.get(key)),
                        }
                    )
    return rows


def format_component(value: object, unit: str) -> str:
    if unit == "count":
        return format_float(value, 1)
    return format_sec(value)


def report_lines(
    rows: Sequence[Dict[str, str]],
    summary_rows: Sequence[Dict[str, object]],
    savings_rows: Sequence[Dict[str, object]],
    cold_warm_rows: Sequence[Dict[str, object]],
    pairwise_rows: Sequence[Dict[str, object]],
    phase: str,
) -> List[str]:
    candidates = sorted({str(row.get("candidate")) for row in rows if row.get("candidate")})
    lines: List[str] = []
    lines.append("# OASIS Persistent-Process DVF Timing: DirectCP + SINR")
    lines.append("")
    lines.append(f"- Phase: `{phase}`")
    lines.append(f"- Case rows: {len(rows)}")
    lines.append("- Process model: one long-lived Python worker subprocess per candidate/method.")
    lines.append("- Cold case: rank `0` inside each candidate worker. This includes CUDA/PyTorch/cuDNN first-use overhead for that worker.")
    lines.append("- Warm cases: ranks `1-9` inside the same candidate worker. These estimate persistent-process throughput after first-use overhead has been paid.")
    lines.append("- Amortized first10: ranks `0-9`, including the cold first case.")
    lines.append("- DVF-only production mode excludes image/segmentation evaluation, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, and full packaging.")
    lines.append("- SINR rows use a single fixed cps stage with no multistage knot insertion/prolongation; DirectCP `8,4,2` rows keep their existing three-stage breakdown.")
    lines.append("")
    lines.append("Important interpretation:")
    lines.append("")
    lines.append("- `cold_minus_warm_mean` estimates first-case overhead plus any rank0-specific case/runtime difference. It is not a pure hardware constant because rank0 is a different image pair.")
    lines.append("- Warm ranks use the same schedule/shapes in the same process, so cuDNN algorithm choices and PyTorch/CUDA caches can be reused when applicable.")
    lines.append("- Compare candidates primarily within the same group: cold vs cold, warm mean vs warm mean, and amortized mean vs amortized mean.")
    lines.append("")
    lines.append("## Why Cold Rank0 Is Much Slower")
    lines.append("")
    lines.append("This is **not** registration warm-start. Each case still starts from its own newly initialized method parameters, its own optimizer, its own image pair, and its own seed reset inside `run_one.py`. For DirectCP this means a fresh control grid; for SINR this means a fresh network/generator and spline output path. No deformation, control field, gradients, or optimizer state are reused from rank0 to later ranks.")
    lines.append("")
    lines.append("What is warm is the long-lived Python/GPU process: CUDA context, PyTorch/CUDA allocator state, cuDNN/PyTorch kernel and algorithm-selection caches, and library/kernel first-use state for the same OASIS tensor shapes. Because ranks within a candidate use the same method, cps/schedule, and image shape, later ranks can reuse those process-level choices/caches where PyTorch supports it.")
    lines.append("")
    lines.append("The cold-vs-warm gap is larger than expected because first-use overhead is not confined to `setup`. It is charged across several first GPU operations, including loss-volume blur/downsampling and the first optimizer-loop expansion/backward paths for each stage. The table below compares cold rank0 with the first warm rank1 in the same candidate worker.")
    lines.append("")
    lines.append("| candidate | component | cold rank0 | warm rank1 | cold - warm | cold/warm |")
    lines.append("|---|---|---:|---:|---:|---:|")
    selected_components = {
        "full_script_to_dvf_in_memory",
        "loaded_images_to_dvf_in_memory",
        "external_wall",
        "setup",
        "training",
        "optimizer_loop",
        "loss_volume_setup_total",
        "total_generator_forward_phase",
        "total_expand_phase",
        "total_backward_step_phase",
        "stage_1_expand",
        "stage_1_generator_forward",
        "stage_2_expand",
        "stage_3_expand",
        "stage_3_optimizer_loop",
        "actual_epochs",
    }
    for row in cold_warm_rows:
        if str(row.get("component")) not in selected_components:
            continue
        unit = str(row.get("unit", "sec"))
        ratio = as_float(row.get("cold_div_warm"))
        ratio_text = f"{ratio:.1f}x" if math.isfinite(ratio) else "NA"
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('label')} | "
            + f"{format_component(row.get('cold_value'), unit)} | "
            + f"{format_component(row.get('warm_value'), unit)} | "
            + f"{format_component(row.get('cold_minus_warm'), unit)} | "
            + f"{ratio_text} |"
        )
    lines.append("")
    lines.append("SINR-specific interpretation:")
    lines.append("")
    lines.append("- `sinr_cps4` and `sinr_cps2` are single-stage 2500-epoch runs at alpha `1000` and lr `1e-4`.")
    lines.append("- SINR has no stage-to-stage knot insertion, no prolongation, and no intermediate dense diagnostic expansion in this DVF-only run.")
    lines.append("- For SINR, the stage post-to-DVF component is final network output plus final dense B-spline expansion; the final `.npy` write is reported separately.")
    lines.append("")
    lines.append("For `A_fast_decay`, the headline `144.55s` cold rank0 versus `16.02s` warm rank1 is therefore not explained by epochs: cold rank0 used `600` epochs and warm rank1 used `575` epochs. The large gap mainly comes from process-level first-use costs distributed through setup, loss-volume setup, expansion phases, and backward/step phases.")
    lines.append("")
    lines.append("For practical reporting, use:")
    lines.append("")
    lines.append("- cold single-case latency: rank0 in a fresh worker process;")
    lines.append("- warm persistent-process throughput: mean of ranks `1-9`;")
    lines.append("- amortized first10 latency: mean of ranks `0-9`, including the cold first case.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| candidate | group | cases | full script -> DVF memory | loaded images -> DVF memory | full script -> DVF file | wall/case | optimizer | epochs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('group')} | {row.get('case_count')} | "
            + f"{format_sec(row.get('full_script_to_dvf_in_memory_sec_mean'))} +/- {format_sec(row.get('full_script_to_dvf_in_memory_sec_std'))} | "
            + f"{format_sec(row.get('loaded_images_to_dvf_in_memory_sec_mean'))} +/- {format_sec(row.get('loaded_images_to_dvf_in_memory_sec_std'))} | "
            + f"{format_sec(row.get('full_script_to_dvf_file_sec_mean'))} +/- {format_sec(row.get('full_script_to_dvf_file_sec_std'))} | "
            + f"{format_sec(row.get('external_wall_sec_mean'))} +/- {format_sec(row.get('external_wall_sec_std'))} | "
            + f"{format_sec(row.get('total_optimizer_loop_sec_mean'))} +/- {format_sec(row.get('total_optimizer_loop_sec_std'))} | "
            + f"{format_float(row.get('actual_epochs_mean'), 1)} +/- {format_float(row.get('actual_epochs_std'), 1)} |"
        )
    lines.append("")
    lines.append("## Original Minus A_fast_decay")
    lines.append("")
    lines.append("Positive values mean `A_fast_decay` is faster/lower for that metric.")
    lines.append("")
    lines.append("| group | metric | original | A_fast_decay | original - A_fast_decay |")
    lines.append("|---|---|---:|---:|---:|")
    wanted_metrics = {
        "full_script_to_dvf_in_memory_sec",
        "loaded_images_to_dvf_in_memory_sec",
        "full_script_to_dvf_file_sec",
        "external_wall_sec",
        "total_optimizer_loop_sec",
        "actual_epochs",
    }
    for row in savings_rows:
        if str(row.get("metric")) not in wanted_metrics:
            continue
        metric = str(row.get("metric"))
        formatter = format_float if metric == "actual_epochs" else format_sec
        lines.append(
            "| "
            + f"{row.get('group')} | {metric} | "
            + f"{formatter(row.get('original_mean'))} | "
            + f"{formatter(row.get('A_fast_decay_mean'))} | "
            + f"{formatter(row.get('original_minus_A_fast_decay'))} |"
        )
    lines.append("")
    if pairwise_rows:
        lines.append("## SINR Minus DirectCP Runtime")
        lines.append("")
        lines.append("Positive values mean the DirectCP candidate is faster/lower than the SINR candidate for that metric.")
        lines.append("")
        lines.append("| group | SINR | DirectCP | metric | SINR | DirectCP | SINR - DirectCP |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in pairwise_rows:
            metric = str(row.get("metric"))
            if metric not in wanted_metrics:
                continue
            formatter = format_float if metric == "actual_epochs" else format_sec
            lines.append(
                "| "
                + f"{row.get('group')} | {row.get('sinr_candidate')} | {row.get('directcp_candidate')} | {metric} | "
                + f"{formatter(row.get('sinr_mean'))} | "
                + f"{formatter(row.get('directcp_mean'))} | "
                + f"{formatter(row.get('sinr_minus_directcp'))} |"
            )
        lines.append("")
    lines.append("## Per-Rank Case Rows")
    lines.append("")
    lines.append("| candidate | rank | fixed | moving | status | full script -> DVF memory | loaded images -> DVF memory | wall | optimizer | epochs |")
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    for row in build_rank_rows(rows):
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('case_rank')} | {row.get('fixed_id')} | {row.get('moving_id')} | {row.get('warm_status')} | "
            + f"{format_sec(row.get('full_script_to_dvf_in_memory_sec'))} | "
            + f"{format_sec(row.get('loaded_images_to_dvf_in_memory_sec'))} | "
            + f"{format_sec(row.get('external_wall_sec'))} | "
            + f"{format_sec(row.get('total_optimizer_loop_sec'))} | "
            + f"{format_float(row.get('actual_epochs'), 0)} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `persistent_process_timing_report.md`")
    lines.append("- `persistent_process_case_timing.csv`")
    lines.append("- `persistent_process_summary.csv`")
    lines.append("- `persistent_process_savings.csv`")
    lines.append("- `persistent_process_pairwise_runtime_comparison.csv`")
    lines.append("- `persistent_process_cold_warm_breakdown.csv`")
    lines.append("- `dvf_only_method_comparison.csv`")
    lines.append("- `dvf_only_stage_breakdown.csv`")
    return lines


def main() -> None:
    args = parse_args()
    method_rows = read_csv(Path(args.review_root) / "dvf_only_method_comparison.csv")
    if not method_rows:
        raise RuntimeError(f"No method comparison rows found under {args.review_root}")
    summary_rows = build_summary_rows(method_rows)
    savings_rows = build_savings_rows(summary_rows)
    cold_warm_rows = build_cold_warm_breakdown_rows(method_rows)
    pairwise_rows = build_pairwise_runtime_rows(summary_rows)
    case_rows = build_rank_rows(method_rows)
    c.write_csv(Path(args.review_root) / "persistent_process_case_timing.csv", case_rows)
    c.write_csv(Path(args.review_root) / "persistent_process_summary.csv", summary_rows)
    c.write_csv(Path(args.review_root) / "persistent_process_savings.csv", savings_rows)
    c.write_csv(Path(args.review_root) / "persistent_process_pairwise_runtime_comparison.csv", pairwise_rows)
    c.write_csv(Path(args.review_root) / "persistent_process_cold_warm_breakdown.csv", cold_warm_rows)
    report = "\n".join(report_lines(method_rows, summary_rows, savings_rows, cold_warm_rows, pairwise_rows, str(args.phase))) + "\n"
    (Path(args.review_root) / "persistent_process_timing_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
