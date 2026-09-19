#!/usr/bin/env python
"""Run and aggregate the focused MR-D-BSCP cuDNN cold/warm ablation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import artifact_paths  # noqa: E402
import common as c  # noqa: E402
import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402


VARIANT = "oasis_seed3_first5_mrdbscp_cudnn_ablation_v1"
CANDIDATE_NAME = "MR_A_fast_decay"
METHOD = unified.MR_METHOD
EXPERIMENT_KEY = "2026-09-01_oasis_mrdbscp_cudnn_ablation"
RUN_KEY = "oasis_mrdbscp_cudnn_ablation_20260901_v1"
TABLE1_FAST_CANDIDATES: Sequence[str] = (
    "MR_A_fast_decay",
    "D2_A_bal_lr10x",
    "D4_A_bal_lr10x",
    "S2_A_bal_lr10x",
    "S4_A_bal_lr10x",
)
TABLE1_BASE_FAST_CANDIDATES: Sequence[str] = (
    "MR_baseline_lr1x_full",
    "MR_A_fast_decay",
    "D2_baseline_lr1x_full",
    "D2_A_bal_lr10x",
    "D4_baseline_lr1x_full",
    "D4_A_bal_lr10x",
    "S2_baseline_lr1x_full",
    "S2_A_bal_lr10x",
    "S4_baseline_lr1x_full",
    "S4_A_bal_lr10x",
)


@dataclass(frozen=True)
class Config:
    slug: str
    label: str
    benchmark: bool
    deterministic: bool


@dataclass(frozen=True)
class TimingStep:
    rank: int
    timing_condition: str
    warm_sequence_index: Optional[int]


CONFIGS: Sequence[Config] = (
    Config("benchmark_true_deterministic_false", "benchmark=True, deterministic=False", True, False),
    Config("benchmark_false_deterministic_false", "benchmark=False, deterministic=False", False, False),
    Config("benchmark_true_deterministic_true", "benchmark=True, deterministic=True", True, True),
    Config("benchmark_false_deterministic_true", "benchmark=False, deterministic=True", False, True),
)
CONFIG_BY_SLUG = {config.slug: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", choices=sorted(CONFIG_BY_SLUG), default="")
    parser.add_argument("--worker-candidate", type=str, default="")
    parser.add_argument("--worker-attempt", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--case-ranks", type=str, default="0-4")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--repeat-cold-rank", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--container-name", type=str, default="py310sitreg_v1_test1")
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--timing-detail", choices=["deep"], default="deep")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-monitor-interval", type=float, default=1.0)
    parser.add_argument("--max-gpu-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dice", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.out_root = artifact_paths.resolve_run_root(
        RUN_KEY,
        args.out_root,
        repo_root=REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    return args


def parse_case_ranks(text: str) -> List[int]:
    ranks: List[int] = []
    for token in str(text).replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            ranks.extend(range(int(start), int(end) + 1))
        else:
            ranks.append(int(token))
    if not ranks or ranks != sorted(set(ranks)):
        raise ValueError(f"Invalid case ranks: {text!r}")
    return ranks


def selected_configs(args: argparse.Namespace) -> List[Config]:
    requested = str(getattr(args, "configs", "")).strip()
    if not requested:
        return list(CONFIGS)
    slugs = [token.strip() for token in requested.split(",") if token.strip()]
    if not slugs or len(slugs) != len(set(slugs)):
        raise ValueError(f"Invalid or duplicate configuration selection: {requested!r}")
    missing = [slug for slug in slugs if slug not in CONFIG_BY_SLUG]
    if missing:
        raise ValueError(f"Unknown configurations {missing}; known={sorted(CONFIG_BY_SLUG)}")
    return [CONFIG_BY_SLUG[slug] for slug in slugs]


def selected_candidates(args: argparse.Namespace) -> List[unified.Candidate]:
    requested = str(getattr(args, "candidates", "")).strip()
    names = [CANDIDATE_NAME] if not requested else [token.strip() for token in requested.split(",") if token.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid or duplicate candidate selection: {requested!r}")
    known = unified.candidate_map()
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"Unknown candidates {missing}; known={sorted(known)}")
    return [known[name] for name in names]


def timing_steps(args: argparse.Namespace) -> List[TimingStep]:
    ranks = parse_case_ranks(str(args.case_ranks))
    if bool(getattr(args, "smoke", False)):
        return [TimingStep(rank=ranks[0], timing_condition="cold", warm_sequence_index=None)]
    steps = [TimingStep(rank=ranks[0], timing_condition="cold", warm_sequence_index=None)]
    warm_ranks = ranks
    if not bool(getattr(args, "repeat_cold_rank", False)):
        warm_ranks = ranks[1:]
    for warm_index, rank in enumerate(warm_ranks, start=1):
        steps.append(
            TimingStep(
                rank=rank,
                timing_condition="warm",
                warm_sequence_index=warm_index,
            )
        )
    return steps


def unique_case_ranks(args: argparse.Namespace) -> List[int]:
    return list(dict.fromkeys(step.rank for step in timing_steps(args)))


def canonical_case_steps(args: argparse.Namespace) -> List[TimingStep]:
    """Return one timing step per unique case for posthoc metric evaluation."""

    seen = set()
    steps: List[TimingStep] = []
    for step in timing_steps(args):
        if step.rank in seen:
            continue
        seen.add(step.rank)
        steps.append(step)
    return steps


def config_out_root(args: argparse.Namespace, config: Config) -> Path:
    return Path(args.out_root) / config.slug


def config_review_root(args: argparse.Namespace, config: Config) -> Path:
    return Path(args.review_root) / config.slug


def candidate_out_root(args: argparse.Namespace, config: Config, candidate: unified.Candidate) -> Path:
    root = config_out_root(args, config)
    return root / candidate.name if len(selected_candidates(args)) > 1 else root


def candidate_review_root(args: argparse.Namespace, config: Config, candidate: unified.Candidate) -> Path:
    root = config_review_root(args, config)
    return root / candidate.name if len(selected_candidates(args)) > 1 else root


def runner_namespace(args: argparse.Namespace, config: Config, candidate: unified.Candidate) -> argparse.Namespace:
    return SimpleNamespace(
        alpha=float(args.alpha),
        seed=int(args.seed),
        phase="primary",
        timing_detail="deep",
        variant=str(args.variant),
        out_root=candidate_out_root(args, config, candidate),
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
        eval_interval=1 if args.smoke else int(args.eval_interval),
        epoch_override=1 if args.smoke else 0,
        overwrite=bool(args.overwrite),
    )


def timing_run_name(
    args: argparse.Namespace,
    candidate: unified.Candidate,
    *,
    rank: int,
    timing_condition: str,
    warm_sequence_index: Optional[int],
    attempt: int = 0,
) -> str:
    base = unified.run_name(candidate, int(rank), float(args.alpha), "primary")
    if bool(getattr(args, "smoke", False)):
        name = f"{base}_smoke"
    elif not bool(getattr(args, "repeat_cold_rank", False)):
        name = base
    elif timing_condition == "cold":
        name = f"{base}_cold"
    else:
        if warm_sequence_index is None:
            raise ValueError("Warm timing requires a warm sequence index")
        name = f"{base}_warm{int(warm_sequence_index):02d}"
    if int(attempt) > 0:
        name += f"_retry{int(attempt):02d}"
    return name


def timing_step_key(step: TimingStep) -> str:
    warm_index = "" if step.warm_sequence_index is None else str(int(step.warm_sequence_index))
    return f"{step.timing_condition}:{int(step.rank)}:{warm_index}"


def primary_attempt_path(args: argparse.Namespace, config: Config, candidate: unified.Candidate) -> Path:
    return candidate_review_root(args, config, candidate) / "primary_attempts.json"


def load_primary_attempts(args: argparse.Namespace, config: Config, candidate: unified.Candidate) -> Dict[str, int]:
    path = primary_attempt_path(args, config, candidate)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return {str(key): int(value) for key, value in payload.items()}


def write_primary_attempts(
    args: argparse.Namespace,
    config: Config,
    candidate: unified.Candidate,
    attempts: Mapping[str, int],
) -> None:
    write_json(primary_attempt_path(args, config, candidate), dict(attempts))


def selected_attempt(args: argparse.Namespace, config: Config, candidate: unified.Candidate, step: TimingStep) -> int:
    return int(load_primary_attempts(args, config, candidate).get(timing_step_key(step), 0))


def build_run_one_argv(
    args: argparse.Namespace,
    config: Config,
    step: TimingStep,
    candidate: unified.Candidate | None = None,
    attempt: int = 0,
) -> List[str]:
    candidate = candidate or selected_candidates(args)[0]
    argv = unified.run_one_argv(runner_namespace(args, config, candidate), candidate, int(step.rank), str(args.gpu))
    run_name = timing_run_name(
        args,
        candidate,
        rank=step.rank,
        timing_condition=step.timing_condition,
        warm_sequence_index=step.warm_sequence_index,
        attempt=attempt,
    )
    run_name_index = argv.index("--run-name")
    argv[run_name_index + 1] = run_name
    argv.extend(
        [
            "--output-mode",
            "dvf_only",
            "--timing-detail",
            "deep",
            "--cudnn-benchmark",
            str(config.benchmark).lower(),
            "--cudnn-deterministic",
            str(config.deterministic).lower(),
        ]
    )
    return argv


def run_dir(
    args: argparse.Namespace,
    config: Config,
    rank: int,
    *,
    timing_condition: str = "cold",
    warm_sequence_index: Optional[int] = None,
    candidate: unified.Candidate | None = None,
    attempt: Optional[int] = None,
) -> Path:
    candidate = candidate or selected_candidates(args)[0]
    step = TimingStep(int(rank), str(timing_condition), warm_sequence_index)
    if attempt is None:
        attempt = selected_attempt(args, config, candidate, step)
    return candidate_out_root(args, config, candidate) / candidate.method / timing_run_name(
        args,
        candidate,
        rank=int(rank),
        timing_condition=timing_condition,
        warm_sequence_index=warm_sequence_index,
        attempt=int(attempt),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def event_total(timing: Mapping[str, object], name: str) -> float:
    events = timing.get("events", [])
    if not isinstance(events, list):
        return float("nan")
    values = [
        float(event.get("elapsed_sec", 0.0))
        for event in events
        if isinstance(event, dict) and event.get("name") == name
    ]
    return float(sum(values)) if values else 0.0


def phase_total(timing: Mapping[str, object], name: str) -> float:
    summaries = timing.get("stage_phase_summaries", [])
    if not isinstance(summaries, list):
        return float("nan")
    return float(
        sum(
            float(stage.get("phase_totals_sec", {}).get(name, 0.0))
            for stage in summaries
            if isinstance(stage, dict) and isinstance(stage.get("phase_totals_sec", {}), dict)
        )
    )


def mark_offset(timing: Mapping[str, object], name: str) -> float:
    marks = timing.get("marks", [])
    if not isinstance(marks, list):
        return float("nan")
    for mark in marks:
        if isinstance(mark, dict) and mark.get("name") == name:
            return float(mark["offset_sec"])
    return float("nan")


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def paper_method_label(label: str) -> str:
    return (
        str(label)
        .replace("MR-D-BSCP 8-4-2", "MR-D-BSCP (8-4-2)")
        .replace("D-BSCP cps2", "D-BSCP (cps2)")
        .replace("D-BSCP cps4", "D-BSCP (cps4)")
        .replace("SINR cps2", "INR-BSCP (cps2)")
        .replace("SINR cps4", "INR-BSCP (cps4)")
    )


def summarize_case(
    args: argparse.Namespace,
    config: Config,
    step: TimingStep,
    sequence_index: int,
    candidate: unified.Candidate | None = None,
    attempt: Optional[int] = None,
) -> Dict[str, object]:
    candidate = candidate or selected_candidates(args)[0]
    target = run_dir(
        args,
        config,
        step.rank,
        timing_condition=step.timing_condition,
        warm_sequence_index=step.warm_sequence_index,
        candidate=candidate,
        attempt=attempt,
    )
    summary_path = target / "summary.json"
    timing_path = target / "timing.json"
    row: Dict[str, object] = {
        "config": config.slug,
        "config_label": config.label,
        "cudnn_benchmark": config.benchmark,
        "cudnn_deterministic": config.deterministic,
        "gpu": str(args.gpu),
        "candidate": candidate.name,
        "method": candidate.method,
        "method_label": paper_method_label(candidate.method_label),
        "case_rank": int(step.rank),
        "case_sequence_index": int(sequence_index),
        "timing_condition": step.timing_condition,
        "warm_sequence_index": step.warm_sequence_index,
        "attempt": int(attempt if attempt is not None else selected_attempt(args, config, candidate, step)),
        "run_dir": str(target),
        "summary_path": str(summary_path),
        "timing_path": str(timing_path),
        "status": "missing",
        "done": False,
    }
    if not summary_path.exists() or not timing_path.exists():
        return row
    try:
        summary = json.loads(summary_path.read_text())
        timing = json.loads(timing_path.read_text())
        if not isinstance(summary, dict) or not isinstance(timing, dict):
            raise ValueError("summary.json or timing.json is not an object")
        load_offset = mark_offset(timing, "case_load_complete")
        ready_offset = mark_offset(timing, "dvf_file_ready")
        endpoint = ready_offset - load_offset
        phase_names = ["control_grid", "expand", "generator_forward", "regularizer", "warp_ncc", "backward_step"]
        phase_sum = sum(phase_total(timing, name) for name in phase_names)
        initial_grid = event_total(timing, "training.initial_dense_grid")
        loss_setup = sum(event_total(timing, f"stage_{idx}.loss_volume_setup") for idx in range(3))
        training = event_total(timing, "training")
        dvf_output = event_total(timing, "method_execution.dvf_file_prep") + event_total(
            timing, "method_execution.dvf_file_write"
        )
        row.update(
            {
                "status": "completed" if (target / "DONE").exists() else "incomplete",
                "done": bool((target / "DONE").exists()),
                "fixed_id": summary.get("fixed_id"),
                "moving_id": summary.get("moving_id"),
                "original_case_index": summary.get("original_case_index"),
                "dvf_file_path": summary.get("dvf_file_path"),
                "loaded_images_to_dvf_file_sec": endpoint,
                "full_script_to_dvf_file_sec": ready_offset,
                "setup_device_sec": event_total(timing, "setup.device"),
                "setup_case_load_sec": event_total(timing, "setup.case_load"),
                "training_sec": training,
                "initial_dense_grid_sec": initial_grid,
                "loss_volume_setup_sec": loss_setup,
                "training_phase_sum_sec": phase_sum,
                "training_phase_unaccounted_sec": training - initial_grid - loss_setup - phase_sum,
                "dvf_output_sec": dvf_output,
                "actual_epochs": summary.get("actual_epochs"),
                "training_peak_mem_gb": summary.get("training_peak_mem_gb"),
                "cuda_backend": timing.get("cuda_backend", {}),
            }
        )
        for name in [
            "control_grid",
            "expand",
            "generator_forward",
            "regularizer",
            "warp_ncc",
            "backward_step",
            "logging_early_stop_wall_sec",
        ]:
            row[f"phase_{name}_sec"] = phase_total(timing, name)
    except Exception as exc:
        row["status"] = "invalid"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]:
        env[key] = threads
    return env


def execute_worker_step(
    args: argparse.Namespace,
    config: Config,
    candidate: unified.Candidate,
    step: TimingStep,
    sequence_index: int,
    attempt: int,
    monitor_log: Path,
    known_worker_gpu_pids: set[int],
) -> Dict[str, object]:
    monitor = GpuProcessMonitor(
        str(args.gpu),
        float(args.gpu_monitor_interval),
        monitor_log,
        known_worker_gpu_pids=known_worker_gpu_pids,
    )
    start = time.perf_counter()
    error = ""
    status = "completed"
    old_argv = sys.argv[:]
    monitor.start()
    try:
        sys.argv = build_run_one_argv(args, config, step, candidate, attempt=attempt)
        import run_one  # noqa: PLC0415

        run_one.main()
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        sys.argv = old_argv
        monitor.stop()
    elapsed = time.perf_counter() - start
    row = summarize_case(args, config, step, sequence_index, candidate, attempt=attempt)
    row.update(
        {
            "worker_status": status,
            "worker_wall_sec": elapsed,
            "error": error[:4000],
            "gpu_monitor_clean": monitor.clean,
            "gpu_monitor_unexpected_count": len(monitor.unexpected),
            "gpu_monitor_error_count": len(monitor.errors),
            "gpu_monitor_log": str(monitor_log),
        }
    )
    return row


def worker(args: argparse.Namespace) -> int:
    config = CONFIG_BY_SLUG[str(args.worker_config)]
    candidate_name = str(getattr(args, "worker_candidate", "") or selected_candidates(args)[0].name)
    candidate = unified.candidate_map()[candidate_name]
    steps = timing_steps(args)
    progress: List[Dict[str, object]] = []
    log_root = candidate_review_root(args, config, candidate) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    worker_log = log_root / "worker.log"
    failure = False
    contaminated_cold = False
    primary_attempts = load_primary_attempts(args, config, candidate)
    known_worker_gpu_pids: set[int] = set()
    with worker_log.open("w") as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
        print(
            json.dumps(
                {
                    "worker_start": asdict(config),
                    "candidate": asdict(candidate),
                    "gpu": str(args.gpu),
                    "worker_attempt": int(args.worker_attempt),
                    "timing_steps": [asdict(step) for step in steps],
                }
            ),
            flush=True,
        )
        for sequence_index, step in enumerate(steps):
            retry_index = 0
            while True:
                attempt = int(args.worker_attempt) + retry_index
                monitor_log = (
                    candidate_review_root(args, config, candidate)
                    / "logs"
                    / f"gpu_monitor_{step.timing_condition}_rank{step.rank:03d}_attempt{attempt:02d}.jsonl"
                )
                row = execute_worker_step(
                    args,
                    config,
                    candidate,
                    step,
                    sequence_index,
                    attempt,
                    monitor_log,
                    known_worker_gpu_pids,
                )
                progress.append(row)
                write_csv(candidate_review_root(args, config, candidate) / "worker_progress.csv", progress)
                print(json.dumps({"case_complete": row}, allow_nan=True, default=str), flush=True)
                gc.collect()
                if row["worker_status"] != "completed" or not bool(row.get("done")):
                    failure = True
                    break
                if bool(row.get("gpu_monitor_clean")):
                    primary_attempts[timing_step_key(step)] = attempt
                    write_primary_attempts(args, config, candidate, primary_attempts)
                    break
                if step.timing_condition == "cold":
                    # A clean cold timing requires a new Python/CUDA worker.
                    contaminated_cold = True
                    break
                retry_index += 1
                if retry_index > int(args.max_gpu_retries):
                    failure = True
                    break
                wait_for_gpu_idle(
                    args,
                    candidate_review_root(args, config, candidate) / "logs" / "gpu_idle_wait.jsonl",
                    allow_current_process=True,
                    allow_gpu_pids=known_worker_gpu_pids,
                )
            if failure or contaminated_cold:
                break
        write_json(
            candidate_review_root(args, config, candidate) / "worker_summary.json",
            {
                "config": asdict(config),
                "candidate": asdict(candidate),
                "gpu": str(args.gpu),
                "case_ranks": unique_case_ranks(args),
                "timing_steps": [asdict(step) for step in steps],
                "completed_count": sum(bool(row.get("done")) for row in progress),
                "failed": failure,
                "contaminated_cold": contaminated_cold,
                "worker_attempt": int(args.worker_attempt),
                "primary_attempts": primary_attempts,
                "rows": progress,
            },
        )
        print(
            json.dumps(
                {
                    "worker_done": not failure and not contaminated_cold,
                    "completed_count": len(progress),
                    "contaminated_cold": contaminated_cold,
                }
            ),
            flush=True,
        )
    if contaminated_cold:
        return 2
    return 1 if failure else 0


def git_metadata() -> Dict[str, object]:
    def command(argv: Sequence[str]) -> str:
        try:
            return subprocess.check_output(list(argv), cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    status = command(["git", "status", "--short", "--untracked-files=normal"])
    return {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_branch": command(["git", "branch", "--show-current"]),
        "git_dirty": bool(status),
        "git_status_short": status,
    }


def nvidia_smi(argv: Sequence[str]) -> str:
    return subprocess.check_output(["nvidia-smi", *argv], text=True, stderr=subprocess.STDOUT).strip()


def parse_compute_apps(raw: str) -> List[Dict[str, object]]:
    text = str(raw).strip()
    if not text:
        return []
    rows: List[Dict[str, object]] = []
    for line in text.splitlines():
        if "No running processes found" in line:
            continue
        values = next(csv.reader([line], skipinitialspace=True), [])
        if not values:
            continue
        try:
            pid = int(str(values[0]).strip())
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "process_name": str(values[1]).strip() if len(values) > 1 else "",
                "used_memory_mb": str(values[2]).strip() if len(values) > 2 else "",
            }
        )
    return rows


def process_tree_pids(root_pid: Optional[int] = None) -> set[int]:
    root = int(root_pid if root_pid is not None else os.getpid())
    pids = {root}
    changed = True
    while changed:
        changed = False
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in pids:
                continue
            try:
                fields = Path("/proc") .joinpath(entry, "stat").read_text().split()
                parent_pid = int(fields[3])
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                continue
            if parent_pid in pids:
                pids.add(pid)
                changed = True
    return pids


class GpuProcessMonitor:
    """Poll GPU compute processes and identify non-worker interference."""

    def __init__(
        self,
        gpu: str,
        interval_sec: float,
        log_path: Path,
        known_worker_gpu_pids: Optional[set[int]] = None,
    ) -> None:
        self.gpu = str(gpu)
        self.interval_sec = max(0.2, float(interval_sec))
        self.log_path = Path(log_path)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self.records: List[Dict[str, object]] = []
        self.unexpected: List[Dict[str, object]] = []
        self.errors: List[str] = []
        self.known_worker_gpu_pids = known_worker_gpu_pids if known_worker_gpu_pids is not None else set()

    @property
    def clean(self) -> bool:
        return not self.unexpected and not self.errors

    def _snapshot(self, event: str) -> Dict[str, object]:
        record: Dict[str, object] = {
            "timestamp": c.now_text(),
            "event": event,
            "gpu": self.gpu,
            "worker_pid": os.getpid(),
            "allowed_pids": sorted(process_tree_pids()),
            "known_worker_gpu_pids": sorted(self.known_worker_gpu_pids),
        }
        try:
            raw = nvidia_smi(
                [
                    f"--id={self.gpu}",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            apps = parse_compute_apps(raw)
            app_pids = {int(app["pid"]) for app in apps}
            if self.known_worker_gpu_pids:
                unexpected = [app for app in apps if int(app["pid"]) not in self.known_worker_gpu_pids]
            elif event == "start":
                # A fresh worker must start with no GPU process. Any process
                # already present is external; a later singleton is learned
                # as the worker despite Docker/host PID namespace differences.
                unexpected = list(apps)
            elif len(apps) == 1:
                self.known_worker_gpu_pids.update(app_pids)
                unexpected = []
            else:
                unexpected = list(apps)
            record.update({"apps": apps, "unexpected": unexpected, "monitor_error": ""})
            if unexpected:
                self.unexpected.extend(unexpected)
        except Exception as exc:  # pragma: no cover - exercised on unavailable GPU hosts
            error = f"{type(exc).__name__}: {exc}"
            self.errors.append(error)
            record.update({"apps": [], "unexpected": [], "monitor_error": error})
        self.records.append(record)
        if self._handle is not None:
            self._handle.write(json.dumps(record, default=str) + "\n")
            self._handle.flush()
        return record

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._snapshot("poll")

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a")
        self._snapshot("start")
        self._thread = threading.Thread(target=self._poll, name="gpu-process-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2.0 * self.interval_sec))
        self._snapshot("stop")
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def wait_for_gpu_idle(
    args: argparse.Namespace,
    log_path: Optional[Path] = None,
    *,
    allow_current_process: bool = False,
    allow_gpu_pids: Optional[set[int]] = None,
) -> None:
    """Wait for a clear GPU before a fresh worker is started."""

    interval = max(0.2, float(getattr(args, "gpu_monitor_interval", 1.0)))
    deadline = time.monotonic() + max(60.0, float(getattr(args, "gpu_idle_timeout", 3600.0)))
    while True:
        raw = nvidia_smi(
            [
                f"--id={args.gpu}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        apps = parse_compute_apps(raw)
        if allow_gpu_pids:
            apps = [app for app in apps if int(app["pid"]) not in allow_gpu_pids]
        if allow_current_process:
            allowed = process_tree_pids()
            apps = [app for app in apps if int(app["pid"]) not in allowed]
        if not apps:
            return
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps({"timestamp": c.now_text(), "apps": apps}) + "\n")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU {args.gpu} did not become idle before retry")
        time.sleep(interval)


def environment_preflight(args: argparse.Namespace) -> Dict[str, object]:
    oasis_root = Path(args.oasis_root)
    pair_file = Path(args.pair_file)
    if not oasis_root.is_dir():
        raise FileNotFoundError(f"OASIS root does not exist inside the execution environment: {oasis_root}")
    sorted_pair_file = oasis_root / "pair_test_200.txt"
    if not sorted_pair_file.is_file():
        raise FileNotFoundError(f"OASIS sorted pair file is missing: {sorted_pair_file}")
    if not pair_file.is_file():
        raise FileNotFoundError(f"Requested shuffled pair file is missing: {pair_file}")
    pairs = c.parse_pair_file(pair_file)
    ranks = unique_case_ranks(args)
    if max(ranks) >= len(pairs):
        raise IndexError(f"Requested case rank {max(ranks)} outside pair file length {len(pairs)}")
    missing_inputs: List[str] = []
    for rank in ranks:
        fixed_id, moving_id = pairs[rank]
        for subject_id in (fixed_id, moving_id):
            for prefix in ("img", "seg"):
                path = oasis_root / "test" / f"{prefix}{subject_id}.nii.gz"
                if not path.is_file():
                    missing_inputs.append(str(path))
    if missing_inputs:
        preview = ", ".join(missing_inputs[:6])
        suffix = " ..." if len(missing_inputs) > 6 else ""
        raise FileNotFoundError(f"Missing OASIS inputs ({len(missing_inputs)}): {preview}{suffix}")
    gpu_id = str(args.gpu)
    gpu_query = nvidia_smi([f"--id={gpu_id}", "--query-gpu=index,name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"])
    compute_apps = nvidia_smi([f"--id={gpu_id}", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if compute_apps and "No running processes found" not in compute_apps:
        raise RuntimeError(f"GPU {gpu_id} is not idle; refusing to start the ablation:\n{compute_apps}")
    import torch

    return {
        **git_metadata(),
        "created_at": c.now_text(),
        "gpu_requested": str(args.gpu),
        "gpu_query": gpu_query,
        "compute_apps_gpu": compute_apps,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "container": str(args.container_name),
        "image": "hengjieliu96/dl_dir:py310sitreg_v1",
        "oasis_root": str(oasis_root),
        "pair_file": str(pair_file),
        "sorted_pair_file": str(sorted_pair_file),
        "pair_count": len(pairs),
        "checked_case_ranks": ranks,
    }


def worker_command(
    args: argparse.Namespace,
    config: Config,
    candidate_name: str | None = None,
    *,
    worker_attempt: int = 0,
) -> List[str]:
    candidate_name = str(candidate_name or getattr(args, "worker_candidate", "") or selected_candidates(args)[0].name)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        config.slug,
        "--worker-candidate",
        candidate_name,
        "--worker-attempt",
        str(int(worker_attempt)),
        "--gpu",
        str(args.gpu),
        "--case-ranks",
        str(args.case_ranks),
        "--configs",
        str(args.configs),
        "--candidates",
        str(getattr(args, "candidates", "")),
        "--container-name",
        str(args.container_name),
        "--alpha",
        f"{float(args.alpha):g}",
        "--seed",
        str(int(args.seed)),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--cpu-threads",
        str(int(args.cpu_threads)),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--variant",
        str(args.variant),
        "--gpu-monitor-interval",
        str(float(args.gpu_monitor_interval)),
        "--max-gpu-retries",
        str(int(args.max_gpu_retries)),
    ] + (["--repeat-cold-rank"] if args.repeat_cold_rank else []) + (["--smoke"] if args.smoke else []) + (["--overwrite"] if args.overwrite else [])


def write_manifest(
    args: argparse.Namespace,
    environment: Mapping[str, object],
    configs: Sequence[Config],
) -> None:
    steps = timing_steps(args)
    candidates = selected_candidates(args)
    payload = {
        "created_at": c.now_text(),
        "variant": str(args.variant),
        "candidates": [asdict(candidate) for candidate in candidates],
        "gpu": str(args.gpu),
        "case_ranks": unique_case_ranks(args),
        "timing_steps": [asdict(step) for step in steps],
        "case_sequence": "rank 0 cold; rank 0 rerun plus ranks 1-9 warm in one persistent worker" if args.repeat_cold_rank else "first rank cold; remaining ranks warm in one persistent worker",
        "configuration_count": len(configs),
        "expected_timing_runs": len(candidates) * len(configs) * len(steps),
        "configs": [asdict(config) for config in configs],
        "cudnn_settings": {
            "requested_matrix": [asdict(config) for config in configs],
            "effective_values_recorded_per_run": True,
            "effective_value_sources": ["timing.json.cuda_backend", "summary.json.cuda_backend"],
        },
        "gpu_monitor": {
            "interval_sec": float(args.gpu_monitor_interval),
            "max_retries": int(args.max_gpu_retries),
            "unexpected_process_policy": "retry_contaminated_warm; restart_worker_for_contaminated_cold",
        },
        "environment": dict(environment),
        "worker_commands": {
            f"{candidate.name}/{config.slug}": worker_command(args, config, candidate.name)
            for candidate in candidates
            for config in configs
        },
    }
    write_json(Path(args.review_root) / "execution_manifest.json", payload)
    write_json(Path(args.out_root) / "execution_manifest.json", payload)


def finite_values(rows: Iterable[Mapping[str, object]], key: str) -> List[float]:
    values = [finite(row.get(key)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def aggregate_timing(args: argparse.Namespace) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    steps = timing_steps(args)
    case_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    phase_names = ["expand", "backward_step", "regularizer", "warp_ncc", "logging_early_stop_wall_sec"]
    for candidate in selected_candidates(args):
        for config in selected_configs(args):
            rows = [
                summarize_case(
                    args,
                    config,
                    step,
                    sequence_index,
                    candidate,
                    attempt=selected_attempt(args, config, candidate, step),
                )
                for sequence_index, step in enumerate(steps)
            ]
            case_rows.extend(rows)
            cold = [row for row in rows if row["timing_condition"] == "cold" and row.get("status") == "completed"]
            warm = [row for row in rows if row["timing_condition"] == "warm" and row.get("status") == "completed"]
            cold_endpoint = finite_values(cold, "loaded_images_to_dvf_file_sec")
            warm_endpoint = finite_values(warm, "loaded_images_to_dvf_file_sec")
            row: Dict[str, object] = {
                "candidate": candidate.name,
                "method": candidate.method,
                "method_label": paper_method_label(candidate.method_label),
                "config": config.slug,
                "config_label": config.label,
                "cudnn_benchmark": config.benchmark,
                "cudnn_deterministic": config.deterministic,
                "gpu": str(args.gpu),
                "cold_count": len(cold),
                "warm_count": len(warm),
                "cold_endpoint_sec": cold_endpoint[0] if cold_endpoint else float("nan"),
                "warm_endpoint_mean_sec": mean(warm_endpoint) if warm_endpoint else float("nan"),
                "warm_endpoint_sd_sec": stdev(warm_endpoint) if len(warm_endpoint) > 1 else float("nan"),
            }
            row["cold_minus_warm_gap_sec"] = finite(row["cold_endpoint_sec"]) - finite(row["warm_endpoint_mean_sec"])
            aggregate_keys = [
                "setup_device_sec",
                "training_sec",
                "initial_dense_grid_sec",
                "loss_volume_setup_sec",
                "training_phase_sum_sec",
                "training_phase_unaccounted_sec",
                "dvf_output_sec",
                "actual_epochs",
                "training_peak_mem_gb",
            ] + [f"phase_{name}_sec" for name in phase_names]
            for key in aggregate_keys:
                cold_values = finite_values(cold, key)
                warm_values = finite_values(warm, key)
                row[f"cold_{key}"] = cold_values[0] if cold_values else float("nan")
                row[f"warm_{key}_mean"] = mean(warm_values) if warm_values else float("nan")
                row[f"gap_{key}"] = finite(row[f"cold_{key}"]) - finite(row[f"warm_{key}_mean"])
            summary_rows.append(row)

    case_fields = list(case_rows[0].keys()) if case_rows else []
    write_csv(Path(args.review_root) / "timing_case_rows.csv", case_rows, case_fields)
    write_csv(Path(args.review_root) / "timing_summary.csv", summary_rows)
    write_json(Path(args.review_root) / "timing_summary.json", summary_rows)

    phase_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        for name in phase_names:
            phase_rows.append(
                {
                    "config": row["config"],
                    "candidate": row["candidate"],
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "config_label": row["config_label"],
                    "phase": name,
                    "cold_sec": row.get(f"cold_phase_{name}_sec"),
                    "warm_mean_sec": row.get(f"warm_phase_{name}_sec_mean"),
                    "cold_minus_warm_sec": row.get(f"gap_phase_{name}_sec"),
                }
            )
    write_csv(Path(args.review_root) / "timing_phase_breakdown.csv", phase_rows)
    return case_rows, summary_rows


def load_dice_summary(review_root: Path) -> List[Dict[str, object]]:
    path = Path(review_root) / "dice_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_workers(args: argparse.Namespace, configs: Sequence[Config]) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    env = thread_env(int(args.cpu_threads))
    candidates = selected_candidates(args)
    for candidate in candidates:
        for config in configs:
            worker_attempt = 0
            while True:
                log_path = (
                    candidate_review_root(args, config, candidate)
                    / "logs"
                    / f"controller_attempt{worker_attempt:02d}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                wait_for_gpu_idle(
                    args,
                    candidate_review_root(args, config, candidate) / "logs" / "controller_gpu_idle_wait.jsonl",
                )
                command = worker_command(args, config, candidate.name, worker_attempt=worker_attempt)
                started = time.time()
                with log_path.open("w") as log_handle:
                    completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=False)
                result = {
                    "candidate": candidate.name,
                    "method": candidate.method,
                    "config": config.slug,
                    "config_label": config.label,
                    "worker_attempt": worker_attempt,
                    "returncode": int(completed.returncode),
                    "elapsed_sec": time.time() - started,
                    "log_path": str(log_path),
                }
                results.append(result)
                if completed.returncode == 2 and worker_attempt < int(args.max_gpu_retries):
                    result["retry_reason"] = "contaminated_cold"
                    worker_attempt += 1
                    continue
                break
            if results[-1]["returncode"] != 0:
                break
        if results and results[-1]["returncode"] != 0:
            break
    payload = {"completed_at": c.now_text(), "sequential": True, "candidate_order": [c.name for c in candidates], "results": results}
    write_json(Path(args.review_root) / "timing_launch_summary.json", payload)
    write_json(Path(args.out_root) / "timing_launch_summary.json", payload)
    return payload


def write_report(
    args: argparse.Namespace,
    environment: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, object]],
    dice_summary: Sequence[Mapping[str, object]] | None,
) -> None:
    candidates = selected_candidates(args)
    configs = selected_configs(args)
    candidate_labels = {candidate.name: paper_method_label(candidate.method_label) for candidate in candidates}
    candidate_display = lambda name: f"{candidate_labels.get(str(name), name)} (`{name}`)"
    expected_timing_jsons = len(candidates) * len(configs) * len(timing_steps(args))
    lines = [
        "# OASIS Table-1 Base/Fast F/T Cold/Warm Rerun",
        "",
        f"- Candidates: {', '.join(f'{candidate_labels[candidate.name]} (`{candidate.name}`)' for candidate in candidates)}; protocol: seed-3 shuffled OASIS ranks {args.case_ranks}.",
        f"- GPU: physical GPU {args.gpu}; {len(candidates)} candidates and {len(configs)} configurations were run sequentially in candidate-major order.",
        "- This is a first-10 diagnostic and is not a publication-ready replacement for the OASIS-100 Dice column.",
        "- Timing endpoint: loaded images resident on the selected device through saved dense DVF .npy; evaluation is excluded.",
        f"- Cold/warm convention: {len([step for step in timing_steps(args) if step.timing_condition == 'cold'])} cold and {len([step for step in timing_steps(args) if step.timing_condition == 'warm'])} warm measurements per configuration.",
        "",
        "## Timing summary",
        "",
        "| Candidate | Configuration | Cold (s) | Warm mean (s) | Warm SD (s) | Gap (s) | setup.device cold/warm (s) | Expand gap (s) | Backward gap (s) | Dice mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dice_by_config = {(str(row.get("candidate")), str(row.get("config"))): row for row in (dice_summary or [])}
    for row in summary_rows:
        dice = dice_by_config.get((str(row.get("candidate")), str(row.get("config"))), {})
        dice_value = finite(dice.get("dice_mean"))
        dice_text = f"{dice_value:.4f}" if math.isfinite(dice_value) else "n/a"
        lines.append(
            "| {candidate} | {label} | {cold:.2f} | {warm:.2f} | {sd:.2f} | {gap:.2f} | {setup_cold:.2f}/{setup_warm:.2f} | {expand:.2f} | {backward:.2f} | {dice} |".format(
                candidate=candidate_display(row.get("candidate")),
                label=row.get("config_label"),
                cold=finite(row.get("cold_endpoint_sec")),
                warm=finite(row.get("warm_endpoint_mean_sec")),
                sd=finite(row.get("warm_endpoint_sd_sec")),
                gap=finite(row.get("cold_minus_warm_gap_sec")),
                setup_cold=finite(row.get("cold_setup_device_sec")),
                setup_warm=finite(row.get("warm_setup_device_sec_mean")),
                expand=finite(row.get("gap_phase_expand_sec")),
                backward=finite(row.get("gap_phase_backward_step_sec")),
                dice=dice_text,
            )
        )
    lines.extend(
        [
            "",
            "## Detailed timing breakdown",
            "",
            "| Candidate | Configuration | Loss-volume setup gap (s) | Regularizer gap (s) | Warp/NCC gap (s) | Logging gap (s) | Training residual gap (s) | Optimizer steps cold/warm | Peak memory cold/warm (GB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {candidate} | {label} | {loss:.2f} | {regularizer:.2f} | {warp:.2f} | {logging:.2f} | {residual:.2f} | {steps_cold:.1f}/{steps_warm:.1f} | {mem_cold:.2f}/{mem_warm:.2f} |".format(
                candidate=candidate_display(row.get("candidate")),
                label=row.get("config_label"),
                loss=finite(row.get("gap_loss_volume_setup_sec")),
                regularizer=finite(row.get("gap_phase_regularizer_sec")),
                warp=finite(row.get("gap_phase_warp_ncc_sec")),
                logging=finite(row.get("gap_phase_logging_early_stop_wall_sec_sec")),
                residual=finite(row.get("gap_training_phase_unaccounted_sec")),
                steps_cold=finite(row.get("cold_actual_epochs")),
                steps_warm=finite(row.get("warm_actual_epochs_mean")),
                mem_cold=finite(row.get("cold_training_peak_mem_gb")),
                mem_warm=finite(row.get("warm_training_peak_mem_gb_mean")),
            )
        )
    by_candidate_config = {(str(row.get("candidate")), str(row.get("config"))): row for row in summary_rows}
    reference_slug = "benchmark_true_deterministic_false"
    if reference_slug in {config.slug for config in configs}:
        lines.extend(
            [
                "",
                "## Comparison to paired reference",
                "",
                "Changes are reported as arm minus the `benchmark=True, deterministic=False` reference for the same candidate; percentages are relative to that paired reference.",
                "",
                "| Candidate | Configuration | Cold change (s, %) | Warm mean change (s, %) | Gap change (s, %) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for candidate in candidates:
            ref = by_candidate_config.get((candidate.name, reference_slug))
            if ref is None:
                continue
            for config_slug in [
                "benchmark_false_deterministic_true",
                "benchmark_false_deterministic_false",
            ]:
                row = by_candidate_config.get((candidate.name, config_slug))
                if row is None:
                    continue
                cells = []
                for key in ["cold_endpoint_sec", "warm_endpoint_mean_sec", "cold_minus_warm_gap_sec"]:
                    reference_value = finite(ref.get(key))
                    arm_value = finite(row.get(key))
                    delta = arm_value - reference_value
                    percent = 100.0 * delta / reference_value if math.isfinite(delta) and reference_value != 0.0 else float("nan")
                    cells.append(f"{delta:+.2f} ({percent:+.1f}%)")
                lines.append(f"| {candidate_display(candidate.name)} | {row.get('config_label')} | {cells[0]} | {cells[1]} | {cells[2]} |")
    else:
        lines.extend(["", "## Comparison to paired reference", "", "No benchmark-enabled paired reference was run in this diagnostic."])
    comparison_lines: List[str] = []
    for candidate in candidates:
        ref = by_candidate_config.get((candidate.name, reference_slug))
        no_benchmark = by_candidate_config.get((candidate.name, "benchmark_false_deterministic_false"))
        if ref is not None and no_benchmark is not None:
            ref_gap = finite(ref.get("cold_minus_warm_gap_sec"))
            no_benchmark_gap = finite(no_benchmark.get("cold_minus_warm_gap_sec"))
            if math.isfinite(ref_gap) and math.isfinite(no_benchmark_gap) and ref_gap != 0.0:
                reduction = 100.0 * (ref_gap - no_benchmark_gap) / ref_gap
                comparison_lines.append(
                    f"- {candidate.name}: with deterministic=False, disabling benchmark changed the cold/warm gap from {ref_gap:.2f} s to {no_benchmark_gap:.2f} s ({reduction:.1f}% reduction)."
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All gap values are cold minus the mean of warm cases. The expand phase is dense B-spline expansion; backward_step includes zeroing gradients, autograd backward, and the optimizer update.",
            "The setup.device block is reported separately because it occurs before the loaded-images timing endpoint. DVF output preparation and writing are also reported separately.",
            *comparison_lines,
            "",
            "## Environment",
            "",
            f"- PyTorch: {environment.get('torch_version')}; CUDA runtime: {environment.get('cuda_runtime_version')}; cuDNN: {environment.get('cudnn_version')}.",
            f"- GPU query: {environment.get('gpu_query')}.",
            f"- Git commit: {environment.get('git_commit')}; dirty tree at launch: {environment.get('git_dirty')}.",
            "",
            "## Backend metadata audit",
            "",
            f"Requested and effective cuDNN flags were audited in {expected_timing_jsons} primary timing JSON files; the effective values were read from each run's `cuda_backend` metadata.",
            "",
            "| Configuration | Requested benchmark | Requested deterministic | Effective benchmark | Effective deterministic |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {config.label} | {str(config.benchmark).lower()} | {str(config.deterministic).lower()} | {str(config.benchmark).lower()} | {str(config.deterministic).lower()} |"
                for config in configs
            ],
        ]
    )
    if dice_summary:
        lines.extend(["", "## Dice evaluation", "", f"Dice was evaluated after timing aggregation from {len(candidates) * len(configs) * len(unique_case_ranks(args))} unique saved DVFs. Evaluation time is not included in the timing table. This first-10 result is diagnostic and does not replace OASIS-100 accuracy."])
    report_name = "oasis_table1_candidates_cudnn_report.md" if len(candidates) > 1 else "mrdbscp_cudnn_ablation_report.md"
    report_path = Path(args.review_root) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = resolve_artifact_paths(parse_args())
    candidates = selected_candidates(args)
    configs = selected_configs(args)
    steps = timing_steps(args)
    if args.worker_config:
        return worker(args)

    environment = environment_preflight(args) if not args.dry_run else git_metadata()
    write_manifest(args, environment, configs)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "candidates": [asdict(candidate) for candidate in candidates],
                    "configs": [asdict(config) for config in configs],
                    "expected_timing_runs": len(candidates) * len(configs) * len(steps),
                },
                indent=2,
            ),
            flush=True,
        )
        for candidate in candidates:
            for config in configs:
                print(json.dumps({"candidate": candidate.name, "config": config.slug, "command": worker_command(args, config, candidate.name)}), flush=True)
        return 0

    launch = run_workers(args, configs)
    _, summary_rows = aggregate_timing(args)
    successful_launches = {
        (str(item.get("candidate")), str(item.get("config")))
        for item in launch.get("results", [])
        if int(item.get("returncode", 1)) == 0
    }
    expected_launches = {
        (candidate.name, config.slug)
        for candidate in candidates
        for config in configs
    }
    all_timing_complete = len(summary_rows) == len(candidates) * len(configs) and all(
        int(row.get("cold_count", 0)) == 1
        and int(row.get("warm_count", 0)) == len([step for step in steps if step.timing_condition == "warm"])
        for row in summary_rows
    ) and successful_launches == expected_launches
    if not all_timing_complete:
        write_report(args, environment, summary_rows, None)
        raise RuntimeError("Timing ablation incomplete; Dice evaluation was not started.")

    dice_summary = None
    if not args.skip_dice:
        evaluator = [
            sys.executable,
            str(THIS_DIR / "evaluate_mrdbscp_cudnn_ablation.py"),
            "--gpu",
            str(args.gpu),
            "--case-ranks",
            str(args.case_ranks),
            "--configs",
            str(args.configs),
            "--candidates",
            str(getattr(args, "candidates", "")),
            "--alpha",
            f"{float(args.alpha):g}",
            "--out-root",
            str(args.out_root),
            "--review-root",
            str(args.review_root),
            "--oasis-root",
            str(args.oasis_root),
            "--pair-file",
            str(args.pair_file),
            "--variant",
            str(args.variant),
        ]
        if args.repeat_cold_rank:
            evaluator.append("--repeat-cold-rank")
        eval_log = Path(args.review_root) / "logs" / "dice_evaluation.log"
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        with eval_log.open("w") as log_handle:
            completed = subprocess.run(
                evaluator,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=thread_env(int(args.cpu_threads)),
            )
        if completed.returncode != 0:
            write_report(args, environment, summary_rows, None)
            raise RuntimeError(f"Posthoc Dice evaluation failed; see {eval_log}")
        dice_summary = load_dice_summary(Path(args.review_root))
    write_report(args, environment, summary_rows, dice_summary)
    print(
        json.dumps(
            {"completed": True, "timing_runs": len(candidates) * len(configs) * len(steps), "dice_evaluated": not args.skip_dice},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
