#!/usr/bin/env python
"""Run OASIS paper-facing cold/warm DVF-only timing."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from launch_first10_be_2500e import parse_case_ranks  # noqa: E402
import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402
import run_one  # noqa: E402

EXPERIMENTS_DIR = c.REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


VARIANT = "oasis_paper_cold_warm_timing_alpha1000_v1"
RUN_KEY = "oasis_paper_cold_warm_timing_alpha1000_v1"
EXPERIMENT_KEY = "2026-08-15_oasis_paper_cold_warm_timing"
DEFAULT_OUT_ROOT = None
DEFAULT_REVIEW_ROOT = None

FAST_CANDIDATES = [
    "MR_A_fast_decay",
    "D2_A_bal_lr10x",
    "D4_A_bal_lr10x",
    "S2_A_bal_lr10x",
    "S4_A_bal_lr10x",
]
BASELINE_CANDIDATES = [
    "MR_baseline_lr1x_full",
    "D2_baseline_lr1x_full",
    "D4_baseline_lr1x_full",
    "S2_baseline_lr1x_full",
    "S4_baseline_lr1x_full",
]
BASELINE_FOR_FAST = {
    "MR_A_fast_decay": "MR_baseline_lr1x_full",
    "D2_A_bal_lr10x": "D2_baseline_lr1x_full",
    "D4_A_bal_lr10x": "D4_baseline_lr1x_full",
    "S2_A_bal_lr10x": "S2_baseline_lr1x_full",
    "S4_A_bal_lr10x": "S4_baseline_lr1x_full",
}


@dataclass(frozen=True)
class WorkItem:
    name: str
    phase_group: str
    candidate_name: str
    gpu: str


@dataclass(frozen=True)
class CaseStep:
    timing_condition: str
    rank: int
    warm_sequence_index: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "fast", "baseline", "all", "aggregate"], default="all")
    parser.add_argument("--worker-candidate", type=str, default="")
    parser.add_argument("--worker-phase-group", type=str, default="")
    parser.add_argument("--worker-name", type=str, default="")
    parser.add_argument("--worker-gpu", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--timing-detail", choices=["off", "coarse", "deep"], default="coarse")
    parser.add_argument("--step-override", type=int, default=0)
    parser.add_argument("--monitor-interval-sec", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve paper-timing outputs under the external artifact store."""

    args.out_root = artifact_paths.resolve_run_root(
        RUN_KEY,
        args.out_root,
        repo_root=c.REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=c.REPO_ROOT,
    )
    return args


def parse_gpus(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ]:
        env[key] = threads
    return env


def selected_candidate(name: str) -> unified.Candidate:
    known = unified.candidate_map()
    if name not in known:
        raise ValueError(f"Unknown candidate {name!r}; known={sorted(known)}")
    return known[name]


def paper_method_label(label: str) -> str:
    return (
        str(label)
        .replace("MR-D-BSCP 8-4-2", "MR-D-BSCP (8-4-2)")
        .replace("D-BSCP cps2", "D-BSCP (cps2)")
        .replace("D-BSCP cps4", "D-BSCP (cps4)")
        .replace("SINR cps2", "INR-BSCP (cps2)")
        .replace("SINR cps4", "INR-BSCP (cps4)")
    )


def is_smoke_phase(phase_group: str) -> bool:
    return str(phase_group).startswith("smoke")


def effective_step_override(args: argparse.Namespace, phase_group: str) -> int:
    if is_smoke_phase(phase_group) and int(args.step_override) <= 0:
        return 1
    return int(args.step_override)


def effective_candidate(args: argparse.Namespace, candidate: unified.Candidate, phase_group: str) -> unified.Candidate:
    ns = SimpleNamespace(epoch_override=effective_step_override(args, phase_group))
    return unified.phase_candidate(candidate, ns)


def work_items_for_stage(stage: str, gpus: Sequence[str]) -> List[WorkItem]:
    if not gpus:
        raise ValueError("At least one GPU identifier is required.")
    items: List[WorkItem] = []
    if stage == "smoke":
        names = FAST_CANDIDATES + BASELINE_CANDIDATES
        for idx, name in enumerate(names):
            phase_group = "smoke_fast" if name in FAST_CANDIDATES else "smoke_baseline"
            items.append(WorkItem(f"{phase_group}_{name}", phase_group, name, gpus[idx % len(gpus)]))
        return items
    if stage in {"fast", "all"}:
        for idx, name in enumerate(FAST_CANDIDATES):
            items.append(WorkItem(f"fast_{name}", "fast", name, gpus[idx % len(gpus)]))
    if stage in {"baseline", "all"}:
        for idx, name in enumerate(BASELINE_CANDIDATES):
            items.append(WorkItem(f"baseline_{name}", "baseline", name, gpus[idx % len(gpus)]))
    return items


def candidate_names_for_stage(stage: str) -> List[str]:
    if stage == "fast":
        return FAST_CANDIDATES
    if stage == "baseline":
        return BASELINE_CANDIDATES
    return FAST_CANDIDATES + BASELINE_CANDIDATES


def case_steps(phase_group: str) -> List[CaseStep]:
    if is_smoke_phase(phase_group):
        return [CaseStep("cold", 0, None), CaseStep("warm", 0, 0)]
    return [CaseStep("cold", 0, None)] + [CaseStep("warm", rank, idx) for idx, rank in enumerate(range(10))]


def timing_run_name(
    *,
    candidate: unified.Candidate,
    phase_group: str,
    condition: str,
    rank: int,
    warm_sequence_index: Optional[int],
    alpha: float,
) -> str:
    if condition == "cold":
        timing_suffix = f"cold_rank{int(rank):03d}"
    else:
        if warm_sequence_index is None:
            raise ValueError("warm_sequence_index is required for warm runs")
        timing_suffix = f"warm_seq{int(warm_sequence_index):02d}_rank{int(rank):03d}"
    return (
        f"{candidate.method}_paper_coldwarm_{phase_group}_{candidate.name}_"
        f"{timing_suffix}_a{c.format_float(float(alpha))}"
    )


def run_dir(args: argparse.Namespace, candidate: unified.Candidate, phase_group: str, step: CaseStep) -> Path:
    effective = effective_candidate(args, candidate, phase_group)
    return (
        Path(args.out_root)
        / effective.method
        / timing_run_name(
            candidate=effective,
            phase_group=phase_group,
            condition=step.timing_condition,
            rank=int(step.rank),
            warm_sequence_index=step.warm_sequence_index,
            alpha=float(args.alpha),
        )
    )


def build_run_one_argv(
    args: argparse.Namespace,
    *,
    candidate: unified.Candidate,
    phase_group: str,
    step: CaseStep,
    gpu: str,
) -> List[str]:
    effective = effective_candidate(args, candidate, phase_group)
    case = c.resolve_case(rank=int(step.rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    run_name = timing_run_name(
        candidate=effective,
        phase_group=phase_group,
        condition=step.timing_condition,
        rank=int(step.rank),
        warm_sequence_index=step.warm_sequence_index,
        alpha=float(args.alpha),
    )
    argv = [
        "run_one.py",
        "--method",
        effective.method,
        "--alpha",
        f"{float(args.alpha):g}",
        "--lr",
        f"{float(effective.lr):g}",
        "--gpu",
        str(gpu),
        "--case-rank",
        str(int(step.rank)),
        "--epochs",
        str(int(effective.epochs)),
        "--eval-interval",
        str(1 if is_smoke_phase(phase_group) else int(args.eval_interval)),
        "--seed",
        str(int(args.seed)),
        "--runtime-candidate",
        effective.name,
        "--timing-detail",
        str(args.timing_detail),
        "--output-mode",
        "dvf_only",
        "--variant",
        str(args.variant),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--run-name",
        run_name,
        "--expected-original-case-index",
        str(case.original_case_index),
        "--expected-fixed-id",
        str(case.fixed_id),
        "--expected-moving-id",
        str(case.moving_id),
        "--stage-early-stop-mode",
        effective.early_stop_mode,
    ]
    if effective.method == unified.MR_METHOD:
        argv.extend(
            [
                "--stage-full-cps",
                effective.stage_full_cps,
                "--stage-downsample",
                effective.stage_downsample,
                "--stage-epochs",
                effective.stage_epochs,
                "--stage-lr",
                effective.stage_lr,
                "--stage-patience",
                effective.stage_patience,
                "--stage-check-interval",
                effective.stage_check_interval,
                "--stage-rel-delta-total",
                effective.stage_rel_delta_total,
                "--stage-rel-delta-data",
                effective.stage_rel_delta_data,
            ]
        )
        if effective.stage_min_epochs:
            argv.extend(["--stage-min-epochs", effective.stage_min_epochs])
    elif effective.early_stop_mode == "adaptive":
        argv.extend(
            [
                "--stage-min-epochs",
                str(int(effective.min_epochs)),
                "--stage-patience",
                str(int(effective.patience)),
                "--stage-check-interval",
                "1",
                "--stage-min-delta",
                f"{float(effective.min_delta):g}",
            ]
        )
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def sample_gpu_state(event: str, gpu: str = "") -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        gpu_cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.STDOUT, timeout=15)
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "timestamp": c.now_text(),
                        "event": event,
                        "target_gpu": str(gpu),
                        "gpu_index": parts[0],
                        "gpu_name": parts[1],
                        "memory_total_mib": parts[2],
                        "memory_used_mib": parts[3],
                        "utilization_gpu_pct": parts[4],
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "timestamp": c.now_text(),
                "event": event,
                "target_gpu": str(gpu),
                "gpu_index": "",
                "gpu_name": "",
                "memory_total_mib": "",
                "memory_used_mib": "",
                "utilization_gpu_pct": "",
                "error": str(exc)[:500],
            }
        )
    return rows


def finite_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return default


def timing_offsets(timing_path: Path) -> Dict[str, float]:
    out = {
        "case_load_complete_sec": float("nan"),
        "dvf_in_memory_ready_sec": float("nan"),
        "dvf_file_ready_sec": float("nan"),
        "script_total_sec": float("nan"),
    }
    if not Path(timing_path).exists():
        return out
    try:
        payload = c.read_json(Path(timing_path))
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    out["script_total_sec"] = finite_float(payload.get("script_total_sec"))
    marks = payload.get("marks", [])
    if isinstance(marks, list):
        for mark in marks:
            if not isinstance(mark, dict):
                continue
            name = str(mark.get("name", ""))
            offset = finite_float(mark.get("offset_sec"))
            if name == "case_load_complete":
                out["case_load_complete_sec"] = offset
            elif name == "dvf_in_memory_ready":
                out["dvf_in_memory_ready_sec"] = offset
            elif name == "dvf_file_ready":
                out["dvf_file_ready_sec"] = offset
    return out


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0 if len(vals) == 1 else float("nan")


def read_progress_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(Path(root).glob("worker_progress_*.csv")):
        try:
            rows.extend(c.read_csv(path))
        except Exception:
            continue
    return rows


def status_for_candidate(rows: List[Dict[str, str]], *, expected_warm: int) -> str:
    if any(str(row.get("status")) == "failed" for row in rows):
        return "failed"
    cold_done = sum(1 for row in rows if row.get("timing_condition") == "cold" and str(row.get("done")).lower() == "true")
    warm_done = sum(1 for row in rows if row.get("timing_condition") == "warm" and str(row.get("done")).lower() == "true")
    if cold_done >= 1 and warm_done >= expected_warm:
        return "completed"
    if rows:
        return "running"
    return "pending"


def monitor_rows(stage: str, review_root: Path) -> List[Dict[str, object]]:
    progress = read_progress_rows(review_root)
    names = candidate_names_for_stage(stage)
    out_rows: List[Dict[str, object]] = []
    for name in names:
        rows = [row for row in progress if row.get("candidate") == name]
        candidate = selected_candidate(name)
        phase_group = "fast" if name in FAST_CANDIDATES else "baseline"
        if stage == "smoke":
            phase_group = "smoke_fast" if name in FAST_CANDIDATES else "smoke_baseline"
        rows = [row for row in rows if row.get("phase_group") == phase_group]
        expected_warm = 1 if stage == "smoke" else 10
        cold_rows = [row for row in rows if row.get("timing_condition") == "cold" and str(row.get("done")).lower() == "true"]
        warm_rows = [row for row in rows if row.get("timing_condition") == "warm" and str(row.get("done")).lower() == "true"]
        latest = rows[-1] if rows else {}
        out_rows.append(
            {
                "phase_group": phase_group,
                "candidate": name,
                "method": candidate.method,
                "method_label": paper_method_label(candidate.method_label),
                "gpu": latest.get("gpu", ""),
                "status": status_for_candidate(rows, expected_warm=expected_warm),
                "cold_done": f"{len(cold_rows)}/1",
                "warm_done": f"{len(warm_rows)}/{expected_warm}",
                "cold_loaded_images_to_dvf_file_sec": mean(finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in cold_rows),
                "warm_loaded_images_to_dvf_file_sec_mean": mean(finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in warm_rows),
                "warm_loaded_images_to_dvf_file_sec_std": std(finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in warm_rows),
                "cold_full_script_to_dvf_file_sec": mean(finite_float(row.get("full_script_to_dvf_file_sec")) for row in cold_rows),
                "warm_full_script_to_dvf_file_sec_mean": mean(finite_float(row.get("full_script_to_dvf_file_sec")) for row in warm_rows),
                "actual_steps_mean": mean(finite_float(row.get("actual_steps")) for row in cold_rows + warm_rows),
                "peak_mem_gb_max": max(
                    [finite_float(row.get("training_peak_mem_gb")) for row in cold_rows + warm_rows]
                    or [float("nan")]
                ),
                "latest_finished_case": latest.get("case_rank", ""),
                "latest_condition": latest.get("timing_condition", ""),
                "latest_run_dir": latest.get("run_dir", ""),
            }
        )
    return out_rows


def format_sec(value: object) -> str:
    val = finite_float(value)
    return "" if not math.isfinite(val) else f"{val:.1f}"


def write_live_monitor(args: argparse.Namespace, *, stage: Optional[str] = None) -> None:
    active_stage = stage or str(args.stage)
    if active_stage == "all":
        active_stage = "all"
    rows = monitor_rows(active_stage, Path(args.review_root))
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / "live_monitor.csv", rows)
    lines = [
        "# OASIS Paper Cold/Warm Timing Live Monitor",
        "",
        f"- Updated: `{c.now_text()}`",
        f"- Stage view: `{active_stage}`",
        f"- Timing endpoint: loaded input images to saved dense DVF.",
        "",
        "| Phase | Candidate | Method | GPU | Status | Cold | Warm | Cold DVF s | Warm DVF s | Warm SD | Steps Mean | Peak GB | Latest |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {phase_group} | `{candidate}` | {method_label} | {gpu} | {status} | {cold_done} | {warm_done} | "
            "{cold} | {warm} | {warm_sd} | {steps} | {peak} | {latest} |".format(
                phase_group=row.get("phase_group", ""),
                candidate=row.get("candidate", ""),
                method_label=row.get("method_label", ""),
                gpu=row.get("gpu", ""),
                status=row.get("status", ""),
                cold_done=row.get("cold_done", ""),
                warm_done=row.get("warm_done", ""),
                cold=format_sec(row.get("cold_loaded_images_to_dvf_file_sec")),
                warm=format_sec(row.get("warm_loaded_images_to_dvf_file_sec_mean")),
                warm_sd=format_sec(row.get("warm_loaded_images_to_dvf_file_sec_std")),
                steps=format_sec(row.get("actual_steps_mean")),
                peak=format_sec(row.get("peak_mem_gb_max")),
                latest=f"{row.get('latest_condition', '')} rank {row.get('latest_finished_case', '')}",
            )
        )
    text = "\n".join(lines) + "\n"
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.review_root) / "live_monitor.md").write_text(text)
    (Path(args.out_root) / "live_monitor.md").write_text(text)


def summarize_run(
    *,
    args: argparse.Namespace,
    item: WorkItem,
    candidate: unified.Candidate,
    effective: unified.Candidate,
    step: CaseStep,
    target_dir: Path,
    log_path: Path,
    status: str,
    error: str,
    elapsed: float,
) -> Dict[str, object]:
    case = c.resolve_case(rank=int(step.rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    summary_path = target_dir / "summary.json"
    timing_path = target_dir / "timing.json"
    summary: Dict[str, object] = {}
    if summary_path.exists():
        try:
            loaded = c.read_json(summary_path)
            if isinstance(loaded, dict):
                summary = loaded
        except Exception as exc:
            error = (error + "\n" + f"summary_read_error={exc}")[:2000]
    offsets = timing_offsets(timing_path)
    full_script_to_dvf_file_sec = finite_float(summary.get("full_script_to_dvf_file_sec"))
    if not math.isfinite(full_script_to_dvf_file_sec):
        full_script_to_dvf_file_sec = offsets["dvf_file_ready_sec"]
    loaded_images_to_dvf_file_sec = finite_float(summary.get("loaded_images_to_dvf_file_sec"))
    if not math.isfinite(loaded_images_to_dvf_file_sec) and math.isfinite(full_script_to_dvf_file_sec):
        case_load_complete = offsets["case_load_complete_sec"]
        if math.isfinite(case_load_complete):
            loaded_images_to_dvf_file_sec = full_script_to_dvf_file_sec - case_load_complete
    dvf_file_path = str(summary.get("dvf_file_path", ""))
    row = {
        "phase_group": item.phase_group,
        "candidate": candidate.name,
        "effective_candidate": effective.name,
        "method": effective.method,
        "method_label": paper_method_label(effective.method_label),
        "family": effective.family,
        "template": effective.template,
        "gpu": str(item.gpu),
        "timing_condition": step.timing_condition,
        "case_rank": int(step.rank),
        "warm_sequence_index": "" if step.warm_sequence_index is None else int(step.warm_sequence_index),
        "original_case_index": int(case.original_case_index),
        "fixed_id": case.fixed_id,
        "moving_id": case.moving_id,
        "status": status,
        "done": bool((target_dir / "DONE").exists()),
        "summary_exists": bool(summary_path.exists()),
        "timing_json_exists": bool(timing_path.exists()),
        "dvf_file_exists": bool(dvf_file_path and Path(dvf_file_path).exists()),
        "output_mode": summary.get("output_mode", ""),
        "no_evaluation": summary.get("no_evaluation", ""),
        "evaluation_skipped": ",".join(summary.get("evaluation_skipped", []))
        if isinstance(summary.get("evaluation_skipped"), list)
        else summary.get("evaluation_skipped", ""),
        "requested_steps": summary.get("requested_epochs", summary.get("epochs", "")),
        "actual_steps": summary.get("actual_epochs", ""),
        "loaded_images_to_dvf_file_sec": loaded_images_to_dvf_file_sec,
        "full_script_to_dvf_file_sec": full_script_to_dvf_file_sec,
        "case_load_complete_sec": offsets["case_load_complete_sec"],
        "dvf_in_memory_ready_sec": offsets["dvf_in_memory_ready_sec"],
        "dvf_file_ready_sec": offsets["dvf_file_ready_sec"],
        "total_elapsed_sec": summary.get("total_elapsed_sec", ""),
        "training_elapsed_sec": summary.get("training_elapsed_sec", ""),
        "training_peak_mem_gb": summary.get("training_peak_mem_gb", ""),
        "persistent_case_wall_sec": float(elapsed),
        "run_dir": str(target_dir),
        "summary_path": str(summary_path),
        "timing_path": str(timing_path),
        "dvf_file_path": dvf_file_path,
        "log_path": str(log_path),
        "error": error[:2000],
    }
    return row


def write_worker_progress(args: argparse.Namespace, item: WorkItem, rows: Sequence[Dict[str, object]]) -> None:
    name = f"worker_progress_{item.phase_group}_{item.candidate_name}.csv"
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / name, rows)


def write_gpu_observations(args: argparse.Namespace, item: WorkItem, rows: Sequence[Dict[str, object]]) -> None:
    name = f"gpu_observations_{item.phase_group}_{item.candidate_name}.csv"
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / name, rows)


def run_worker(args: argparse.Namespace) -> None:
    item = WorkItem(
        name=str(args.worker_name),
        phase_group=str(args.worker_phase_group),
        candidate_name=str(args.worker_candidate),
        gpu=str(args.worker_gpu),
    )
    candidate = selected_candidate(item.candidate_name)
    effective = effective_candidate(args, candidate, item.phase_group)
    log_dir = Path(args.out_root) / "logs" / "paper_cold_warm_worker" / item.name
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    gpu_rows: List[Dict[str, object]] = []
    gpu_rows.extend(sample_gpu_state("worker_start", item.gpu))
    for step_index, step in enumerate(case_steps(item.phase_group)):
        effective = effective_candidate(args, candidate, item.phase_group)
        target_dir = run_dir(args, candidate, item.phase_group, step)
        log_path = log_dir / f"{target_dir.name}.log"
        status = "completed"
        error = ""
        start = time.perf_counter()
        if (target_dir / "DONE").exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            argv = build_run_one_argv(args, candidate=candidate, phase_group=item.phase_group, step=step, gpu=item.gpu)
            with log_path.open("w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                old_argv = sys.argv[:]
                try:
                    sys.argv = argv
                    run_one.main()
                except Exception:
                    status = "failed"
                    error = traceback.format_exc()
                    print(error, flush=True)
                finally:
                    sys.argv = old_argv
        elapsed = time.perf_counter() - start
        gpu_rows.extend(sample_gpu_state(f"after_{step.timing_condition}_{step_index}", item.gpu))
        row = summarize_run(
            args=args,
            item=item,
            candidate=candidate,
            effective=effective,
            step=step,
            target_dir=target_dir,
            log_path=log_path,
            status=status,
            error=error,
            elapsed=elapsed,
        )
        rows.append(row)
        if str(row["status"]) == "failed" or not bool(row["done"]):
            failures.append(row)
        write_worker_progress(args, item, rows)
        write_gpu_observations(args, item, gpu_rows)
        write_live_monitor(args)
        gc.collect()
    payload = {
        "completed_at": c.now_text(),
        "worker_name": item.name,
        "phase_group": item.phase_group,
        "candidate": item.candidate_name,
        "gpu": item.gpu,
        "case_count": len(rows),
        "failures": failures,
        "rows": rows,
    }
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_json(root / f"worker_summary_{item.phase_group}_{item.candidate_name}.json", payload)
    if failures:
        raise RuntimeError(f"Worker {item.name} had {len(failures)} failures")


def write_manifest(args: argparse.Namespace, items: Sequence[WorkItem], *, stage: str) -> None:
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    job_rows = []
    for item in items:
        candidate = selected_candidate(item.candidate_name)
        effective = effective_candidate(args, candidate, item.phase_group)
        candidate_rows.append(
            {
                "stage": stage,
                "phase_group": item.phase_group,
                "candidate": item.candidate_name,
                "method": effective.method,
                "method_label": paper_method_label(effective.method_label),
                "family": effective.family,
                "template": effective.template,
                "gpu": item.gpu,
                "case_sequence": "cold rank0, then warm rank0-9" if not is_smoke_phase(item.phase_group) else "cold rank0, then warm rank0",
                "output_mode": "dvf_only",
                "timing_detail": str(args.timing_detail),
                "step_override": effective_step_override(args, item.phase_group),
            }
        )
        for step in case_steps(item.phase_group):
            case = c.resolve_case(rank=int(step.rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
            job_rows.append(
                {
                    "stage": stage,
                    "phase_group": item.phase_group,
                    "candidate": item.candidate_name,
                    "method": effective.method,
                    "method_label": paper_method_label(effective.method_label),
                    "gpu": item.gpu,
                    "timing_condition": step.timing_condition,
                    "case_rank": int(step.rank),
                    "warm_sequence_index": "" if step.warm_sequence_index is None else int(step.warm_sequence_index),
                    "original_case_index": int(case.original_case_index),
                    "fixed_id": case.fixed_id,
                    "moving_id": case.moving_id,
                    "run_name": timing_run_name(
                        candidate=effective,
                        phase_group=item.phase_group,
                        condition=step.timing_condition,
                        rank=int(step.rank),
                        warm_sequence_index=step.warm_sequence_index,
                        alpha=float(args.alpha),
                    ),
                }
            )
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / f"paper_cold_warm_candidate_manifest_{stage}.csv", candidate_rows)
        c.write_json(root / f"paper_cold_warm_candidate_manifest_{stage}.json", candidate_rows)
        c.write_csv(root / f"paper_cold_warm_job_manifest_{stage}.csv", job_rows)
        c.write_json(
            root / f"paper_cold_warm_job_manifest_{stage}.json",
            {
                "created_at": c.now_text(),
                "stage": stage,
                "candidate_count": len(candidate_rows),
                "registration_count": len(job_rows),
                "expected_full_registration_count": 110 if stage == "all" else len(job_rows),
                "output_mode": "dvf_only",
                "timing_endpoint": "loaded input images to saved dense DVF",
                "jobs": job_rows,
            },
        )


def launch_worker(args: argparse.Namespace, item: WorkItem) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-candidate",
        item.candidate_name,
        "--worker-phase-group",
        item.phase_group,
        "--worker-name",
        item.name,
        "--worker-gpu",
        item.gpu,
        "--stage",
        str(args.stage),
        "--alpha",
        f"{float(args.alpha):g}",
        "--variant",
        str(args.variant),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--cpu-threads",
        str(args.cpu_threads),
        "--timing-detail",
        str(args.timing_detail),
        "--step-override",
        str(args.step_override),
        "--monitor-interval-sec",
        str(args.monitor_interval_sec),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    log_dir = Path(args.out_root) / "logs" / "paper_cold_warm_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "item": item,
        "cmd": cmd,
        "log_path": log_dir / f"worker_{item.name}_gpu{item.gpu}.log",
    }


def run_items(args: argparse.Namespace, items: Sequence[WorkItem], *, label: str) -> Dict[str, object]:
    if not items:
        return {"label": label, "completed_workers": [], "failed_workers": [], "elapsed_sec": 0.0}
    gpus = parse_gpus(str(args.gpus))
    max_concurrent = min(max(1, int(args.max_concurrent)), len(gpus), len(items))
    pending = list(items)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    env = thread_env(int(args.cpu_threads))
    started = time.time()
    last_monitor = 0.0
    while pending or running:
        while pending and len(running) < max_concurrent:
            busy = {str(item["item"].gpu) for item in running}
            free_gpus = [gpu for gpu in gpus if gpu not in busy]
            if not free_gpus:
                break
            base_item = pending.pop(0)
            item = WorkItem(base_item.name, base_item.phase_group, base_item.candidate_name, free_gpus[0])
            worker = launch_worker(args, item)
            print(json.dumps({"launch_worker": item.__dict__, "cmd": worker["cmd"], "log": str(worker["log_path"])}), flush=True)
            log_f = Path(worker["log_path"]).open("w")
            proc = subprocess.Popen(worker["cmd"], stdout=log_f, stderr=subprocess.STDOUT, env=env)
            worker.update({"proc": proc, "log_file": log_f, "started_epoch": time.time()})
            running.append(worker)
        still: List[Dict[str, object]] = []
        for worker in running:
            proc = worker["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still.append(worker)
                continue
            worker["log_file"].close()
            item = worker["item"]
            assert isinstance(item, WorkItem)
            row = {
                "label": label,
                "worker_name": item.name,
                "phase_group": item.phase_group,
                "candidate": item.candidate_name,
                "gpu": item.gpu,
                "returncode": int(code),
                "log_path": str(worker["log_path"]),
                "elapsed_sec": time.time() - float(worker["started_epoch"]),
            }
            if code == 0:
                completed.append(row)
            else:
                failures.append(row)
            print(json.dumps({"worker_done": row}), flush=True)
        running = still
        now = time.time()
        if now - last_monitor >= float(args.monitor_interval_sec):
            write_live_monitor(args)
            last_monitor = now
        if pending or running:
            time.sleep(5.0)
    write_live_monitor(args)
    payload = {
        "completed_at": c.now_text(),
        "label": label,
        "worker_count": len(items),
        "completed_workers": completed,
        "failed_workers": failures,
        "elapsed_sec": time.time() - started,
    }
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(root / f"launch_summary_paper_cold_warm_{label}.json", payload)
    if failures and not args.allow_incomplete:
        raise RuntimeError("Paper cold/warm timing workers failed: " + json.dumps(failures, indent=2))
    return payload


def analyze(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_oasis_paper_cold_warm_timing.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--variant",
        str(args.variant),
    ]
    if args.allow_incomplete:
        cmd.append("--allow-incomplete")
    print(json.dumps({"analyze_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    if args.worker_candidate:
        run_worker(args)
        return
    if args.stage == "aggregate":
        analyze(args)
        return
    gpus = parse_gpus(str(args.gpus))
    items = work_items_for_stage(str(args.stage), gpus)
    write_manifest(args, items, stage=str(args.stage))
    write_live_monitor(args)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "stage": str(args.stage),
                "candidate_count": len(items),
                "registration_count": sum(len(case_steps(item.phase_group)) for item in items),
                "gpus": gpus,
                "max_concurrent": min(max(1, int(args.max_concurrent)), len(gpus), len(items) if items else 1),
                "output_mode": "dvf_only",
                "dry_run": bool(args.dry_run),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for item in items:
            worker = launch_worker(args, item)
            print(json.dumps({"launch_worker": item.__dict__, "cmd": worker["cmd"], "log": str(worker["log_path"])}), flush=True)
        return
    if args.stage == "all":
        fast_items = work_items_for_stage("fast", gpus)
        baseline_items = work_items_for_stage("baseline", gpus)
        run_items(args, fast_items, label="fast")
        run_items(args, baseline_items, label="baseline")
    else:
        run_items(args, items, label=str(args.stage))
    if not args.skip_analyze and str(args.stage) in {"all", "fast", "baseline"}:
        analyze(args)


if __name__ == "__main__":
    main()
