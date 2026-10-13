#!/usr/bin/env python
"""Run and aggregate the rank-0 OASIS MR-D-BSCP 8-4-2-1 lambda sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


METHOD = "direct_cp_8_4_2_1_be_bspline_analytic_multistage"
VARIANT = "oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1"
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
DEFAULT_OUT_ROOT = ARTIFACT_ROOT / "runs" / VARIANT
DEFAULT_REVIEW_ROOT = ARTIFACT_ROOT / "repo_experiments" / VARIANT
DEFAULT_OASIS_ROOT = Path(os.environ["BRAIN_DATA"]) if os.environ.get("BRAIN_DATA") else None
STAGE_CPS = [8, 4, 2, 1]
STAGE_DOWNSAMPLE = [4, 2, 1, 1]
STAGE_EPOCHS = [400, 800, 1300, 1300]
STAGE_LR = [3e-2, 1e-2, 3e-3, 1e-3]
STAGE_MIN_EPOCHS = [40, 80, 130, 130]
STAGE_PATIENCE = [20, 35, 50, 50]
STAGE_CHECK_INTERVAL = [25, 25, 25, 25]
STAGE_REL_DELTA_TOTAL = [1e-4] * 4
STAGE_REL_DELTA_DATA = [5e-5] * 4
PREFIX_ALPHA = 1000.0
CASE_RANK = 0
EXPECTED_FIXED = "0333"
EXPECTED_MOVING = "0115"
ALLOWED_GPUS = {"0", "1", "2", "3"}
EXPECTED_DVF_SHAPE = (*c.IMAGE_SHAPE, 3)

# The order implements the requested initial assignment. The final entry queues.
DEFAULT_JOBS: Sequence[Tuple[float, Optional[str]]] = (
    (10.0, "0"),
    (100.0, "1"),
    (10000.0, "2"),
    (1000.0, "3"),
    (100000.0, None),
)

BASELINES: Sequence[Mapping[str, object]] = (
    {
        "method": "MR-D-BSCP 8-4-2",
        "kind": "existing_baseline",
        "dice": 0.7607446419879308,
        "hd95": 2.6523419316255900,
        "asd": 0.5925002179651616,
        "source": "previously verified OASIS shuffled rank-0 result",
    },
    {
        "method": "Greedy",
        "kind": "existing_baseline",
        "dice": 0.7712877733642253,
        "hd95": 2.1963895289869924,
        "asd": 0.5075983977968538,
        "source": "previously verified OASIS shuffled rank-0 result",
    },
    {
        "method": "VFA",
        "kind": "existing_baseline",
        "dice": 0.7968203408706900,
        "hd95": 2.0485631555197186,
        "asd": 0.4400145870843727,
        "source": "previously verified OASIS shuffled rank-0 result",
    },
    {
        "method": "SITReg",
        "kind": "existing_baseline",
        "dice": 0.7987193557676338,
        "hd95": 2.0613193369724470,
        "asd": 0.42306412147236566,
        "source": "previously verified OASIS shuffled rank-0 result",
    },
)


@dataclass(frozen=True)
class Job:
    job_index: int
    cps1_alpha: float
    stage_alpha: Tuple[float, float, float, float]
    preferred_gpu: Optional[str]
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str


def csv_text(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def command_output(argv: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(argv), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # provenance must not prevent the experiment
        return f"unavailable: {type(exc).__name__}: {exc}"


def source_metadata() -> Dict[str, object]:
    status = command_output(["git", "status", "--short", "--untracked-files=normal"])
    commit = command_output(["git", "rev-parse", "HEAD"])
    branch = command_output(["git", "branch", "--show-current"])
    dirty = bool(status) and not status.startswith("unavailable:")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "git_status_short": status,
        "source_revision": f"{commit}-dirty" if dirty else commit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--prefix-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    if args.oasis_root is None:
        parser.error("--oasis-root or BRAIN_DATA is required")
    return args


def parse_gpus(text: str) -> List[str]:
    gpus = [token.strip() for token in str(text).split(",") if token.strip()]
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list contains duplicates: {gpus}")
    unknown = set(gpus) - ALLOWED_GPUS
    if unknown:
        raise ValueError(f"Expected physical GPUs {sorted(ALLOWED_GPUS)}, got {gpus}")
    if not gpus:
        raise ValueError("At least one GPU is required")
    return gpus


def build_jobs(args: argparse.Namespace) -> List[Job]:
    case = c.resolve_case(rank=CASE_RANK, pair_file=args.pair_file, oasis_root=args.oasis_root)
    if case.fixed_id != EXPECTED_FIXED or case.moving_id != EXPECTED_MOVING:
        raise RuntimeError(
            f"Rank 0 changed: observed {case.fixed_id}/{case.moving_id}, "
            f"expected {EXPECTED_FIXED}/{EXPECTED_MOVING}"
        )
    return [
        Job(
            job_index=index,
            cps1_alpha=float(cps1_alpha),
            stage_alpha=(PREFIX_ALPHA, PREFIX_ALPHA, PREFIX_ALPHA, float(cps1_alpha)),
            preferred_gpu=preferred_gpu,
            case_rank=CASE_RANK,
            original_case_index=int(case.original_case_index),
            fixed_id=str(case.fixed_id),
            moving_id=str(case.moving_id),
        )
        for index, (cps1_alpha, preferred_gpu) in enumerate(DEFAULT_JOBS)
    ]


def alpha_key(value: float) -> str:
    return c.format_float(float(value))


def run_name(job: Job) -> str:
    stage_alpha = "-".join(alpha_key(value) for value in job.stage_alpha)
    return (
        f"{METHOD}_rank{job.case_rank:03d}_sched8-4-2-1_ds4-2-1-1"
        f"_e400-800-1300-1300_sa{stage_alpha}_lr3em2-1em2-3em3-1em3"
        "_esadaptive_min40-80-130-130_pat20-35-50-50_seed1"
    )


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / METHOD / run_name(job)


def done_path(args: argparse.Namespace, job: Job) -> Path:
    return run_dir(args, job) / "DONE"


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        env[key] = threads
    return env


def build_command(args: argparse.Namespace, job: Job, gpu: str, revision: str) -> List[str]:
    command = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method", METHOD,
        "--alpha", f"{PREFIX_ALPHA:g}",
        "--stage-alpha", csv_text(job.stage_alpha),
        "--gpu", str(gpu),
        "--case-rank", str(job.case_rank),
        "--epochs", str(sum(STAGE_EPOCHS)),
        "--eval-interval", str(args.eval_interval),
        "--lr", "0.01",
        "--stage-full-cps", csv_text(STAGE_CPS),
        "--stage-downsample", csv_text(STAGE_DOWNSAMPLE),
        "--stage-epochs", csv_text(STAGE_EPOCHS),
        "--stage-lr", csv_text(STAGE_LR),
        "--stage-early-stop-mode", "adaptive",
        "--stage-min-epochs", csv_text(STAGE_MIN_EPOCHS),
        "--stage-patience", csv_text(STAGE_PATIENCE),
        "--stage-check-interval", csv_text(STAGE_CHECK_INTERVAL),
        "--stage-rel-delta-total", csv_text(STAGE_REL_DELTA_TOTAL),
        "--stage-rel-delta-data", csv_text(STAGE_REL_DELTA_DATA),
        "--seed", str(args.seed),
        "--cudnn-benchmark", "false",
        "--cudnn-deterministic", "true",
        "--prolongation-preserve-check",
        "--variant", VARIANT,
        "--runtime-candidate", f"cps1_alpha_{alpha_key(job.cps1_alpha)}",
        "--source-revision", str(revision),
        "--out-root", str(args.out_root),
        "--oasis-root", str(args.oasis_root),
        "--pair-file", str(args.pair_file),
        "--run-name", run_name(job),
        "--expected-original-case-index", str(job.original_case_index),
        "--expected-fixed-id", job.fixed_id,
        "--expected-moving-id", job.moving_id,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def gpu_snapshot(event: str, target_gpu: Optional[str] = None) -> Dict[str, object]:
    argv = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows: List[Dict[str, object]] = []
    error: Optional[str] = None
    try:
        output = subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=20)
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            rows.append(
                {
                    "gpu": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_pct": int(parts[5]),
                }
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "timestamp": c.now_text(),
        "event": str(event),
        "target_gpu": target_gpu,
        "rows": rows,
        "error": error,
    }


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, allow_nan=True, default=str) + "\n")


def select_freest_gpu(gpus: Sequence[str]) -> str:
    snapshot = gpu_snapshot("retry_gpu_selection")
    free_by_gpu = {
        str(row["gpu"]): int(row["memory_free_mib"])
        for row in snapshot["rows"]  # type: ignore[index]
        if str(row["gpu"]) in set(gpus)
    }
    if not free_by_gpu:
        return str(gpus[0])
    return max(gpus, key=lambda gpu: free_by_gpu.get(str(gpu), -1))


def write_manifest(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": c.now_text(),
        "variant": VARIANT,
        "method": METHOD,
        "case_rank": CASE_RANK,
        "fixed_id": EXPECTED_FIXED,
        "moving_id": EXPECTED_MOVING,
        "direction": "moving_to_fixed",
        "gpus": list(gpus),
        "max_concurrent": min(int(args.max_concurrent), len(gpus)),
        "stage_full_cps": STAGE_CPS,
        "stage_downsample": STAGE_DOWNSAMPLE,
        "stage_epochs": STAGE_EPOCHS,
        "stage_lr": STAGE_LR,
        "stage_min_epochs": STAGE_MIN_EPOCHS,
        "stage_patience": STAGE_PATIENCE,
        "stage_check_interval": STAGE_CHECK_INTERVAL,
        "stage_rel_delta_total": STAGE_REL_DELTA_TOTAL,
        "stage_rel_delta_data": STAGE_REL_DELTA_DATA,
        "stage_early_stop_mode": "adaptive",
        "seed": int(args.seed),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "oasis_root": str(args.oasis_root),
        "pair_file": str(args.pair_file),
        "source": dict(source),
        "jobs": [{**asdict(job), "run_name": run_name(job), "run_dir": str(run_dir(args, job))} for job in jobs],
        "initial_gpu_snapshot": gpu_snapshot("manifest_start"),
    }
    c.write_json(args.out_root / "job_manifest.json", payload)
    c.write_csv(args.out_root / "job_manifest.csv", [asdict(job) for job in jobs])


def is_oom_log(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="replace").lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def launch_jobs(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    max_concurrent = min(max(1, int(args.max_concurrent)), len(gpus))
    env = thread_env(int(args.cpu_threads))
    revision = str(source["source_revision"])
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_log = args.out_root / "gpu_observations.jsonl"
    event_log = args.out_root / "launch_events.jsonl"
    pending = [job for job in jobs if args.overwrite or not done_path(args, job).exists()]
    skipped = [job for job in jobs if job not in pending]
    running: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    retry_waiting: List[Dict[str, object]] = []
    successful_completions = 0

    if args.dry_run:
        for job in pending:
            gpu = job.preferred_gpu or str(gpus[0])
            print(json.dumps({"gpu": gpu, "job": asdict(job), "command": build_command(args, job, gpu, revision)}))
        return

    append_jsonl(event_log, {"event": "launcher_start", "timestamp": c.now_text(), "skipped": len(skipped)})
    while pending or running or retry_waiting:
        busy_gpus = {str(item["gpu"]) for item in running}
        free_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]

        promoted: List[Dict[str, object]] = []
        for retry in retry_waiting:
            if successful_completions >= int(retry["eligible_after_successes"]):
                promoted.append(retry)
        for retry in promoted:
            retry_waiting.remove(retry)
            pending.append(retry["job"])  # type: ignore[arg-type]

        while pending and free_gpus and len(running) < max_concurrent:
            selected_index: Optional[int] = None
            for index, candidate in enumerate(pending):
                if candidate.preferred_gpu is None or candidate.preferred_gpu in free_gpus:
                    selected_index = index
                    break
            if selected_index is None:
                break
            job = pending.pop(selected_index)
            retry_count = int(getattr(job, "_retry_attempt", 0))
            if retry_count:
                gpu = select_freest_gpu(free_gpus)
            else:
                gpu = job.preferred_gpu if job.preferred_gpu in free_gpus else free_gpus[0]
            attempt = int(retry_count)
            log_path = log_dir / f"cps1_alpha_{alpha_key(job.cps1_alpha)}_attempt{attempt}_gpu{gpu}.log"
            while log_path.exists():
                attempt += 1
                log_path = log_dir / f"cps1_alpha_{alpha_key(job.cps1_alpha)}_attempt{attempt}_gpu{gpu}.log"
            snapshot = gpu_snapshot("before_launch", str(gpu))
            append_jsonl(gpu_log, {**snapshot, "cps1_alpha": job.cps1_alpha, "attempt": attempt})
            command = build_command(args, job, str(gpu), revision)
            log_handle = log_path.open("w")
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            item = {
                "job": job,
                "gpu": str(gpu),
                "attempt": attempt,
                "retry_count": retry_count,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "started_monotonic": time.monotonic(),
            }
            running.append(item)
            append_jsonl(
                event_log,
                {
                    "event": "job_start",
                    "timestamp": c.now_text(),
                    "cps1_alpha": job.cps1_alpha,
                    "gpu": str(gpu),
                    "attempt": attempt,
                    "retry_count": retry_count,
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                },
            )
            free_gpus.remove(str(gpu))
            if float(args.launch_stagger_sec) > 0:
                time.sleep(float(args.launch_stagger_sec))

        for item in list(running):
            process = item["process"]
            assert isinstance(process, subprocess.Popen)
            returncode = process.poll()
            if returncode is None:
                continue
            handle = item["log_handle"]
            handle.close()  # type: ignore[union-attr]
            running.remove(item)
            job = item["job"]
            assert isinstance(job, Job)
            log_path = Path(item["log_path"])
            elapsed = time.monotonic() - float(item["started_monotonic"])
            append_jsonl(
                gpu_log,
                {
                    **gpu_snapshot("after_exit", str(item["gpu"])),
                    "cps1_alpha": job.cps1_alpha,
                    "attempt": int(item["attempt"]),
                    "retry_count": int(item["retry_count"]),
                    "returncode": int(returncode),
                },
            )
            append_jsonl(
                event_log,
                {
                    "event": "job_exit",
                    "timestamp": c.now_text(),
                    "cps1_alpha": job.cps1_alpha,
                    "gpu": item["gpu"],
                    "attempt": int(item["attempt"]),
                    "retry_count": int(item["retry_count"]),
                    "returncode": int(returncode),
                    "elapsed_sec": elapsed,
                },
            )
            if returncode == 0 and done_path(args, job).exists():
                successful_completions += 1
                continue
            if is_oom_log(log_path) and int(item["retry_count"]) == 0:
                retry_job = Job(
                    job_index=job.job_index,
                    cps1_alpha=job.cps1_alpha,
                    stage_alpha=job.stage_alpha,
                    preferred_gpu=None,
                    case_rank=job.case_rank,
                    original_case_index=job.original_case_index,
                    fixed_id=job.fixed_id,
                    moving_id=job.moving_id,
                )
                object.__setattr__(retry_job, "_retry_attempt", 1)
                retry_waiting.append(
                    {
                        "job": retry_job,
                        "eligible_after_successes": successful_completions + (1 if running else 0),
                    }
                )
            else:
                failures.append(
                    {
                        "cps1_alpha": job.cps1_alpha,
                        "gpu": item["gpu"],
                        "attempt": item["attempt"],
                        "retry_count": item["retry_count"],
                        "returncode": returncode,
                        "log_path": str(log_path),
                    }
                )

        if running or retry_waiting:
            time.sleep(max(0.25, float(args.poll_sec)))
        elif pending:
            # Pending preferred jobs can only reach this branch with a restricted GPU list.
            for index, job in enumerate(pending):
                if job.preferred_gpu not in gpus:
                    pending[index] = Job(**{**asdict(job), "preferred_gpu": None})

    c.write_json(
        args.out_root / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "successful_count": sum(done_path(args, job).exists() for job in jobs),
            "skipped_existing_count": len(skipped),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} sweep jobs failed; see {args.out_root / 'launch_summary.json'}")


def read_stage_tail(run_path: Path, stage_idx: int) -> Dict[str, float]:
    with (run_path / "stage_history.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(float(row["stage_idx"])) == int(stage_idx)]
    if not rows:
        raise RuntimeError(f"Missing stage {stage_idx} history in {run_path}")
    last = rows[-1]
    return {
        "epoch": float(last["epoch"]),
        "loss": float(last["loss"]),
        "data_loss": float(last["data_loss"]),
        "reg_loss": float(last["reg_loss"]),
    }


def validate_absent_labels(summary: Mapping[str, object]) -> None:
    for label in range(1, 36):
        values = [
            float(summary[f"dice_label_{label:02d}"]),
            float(summary[f"hd95_label_{label:02d}"]),
            float(summary[f"asd_symmetric_label_{label:02d}"]),
        ]
        finite = [math.isfinite(value) for value in values]
        if any(finite) and not all(finite):
            raise RuntimeError(f"Inconsistent absent-label metrics for label {label}: {values}")


def validate_candidate(args: argparse.Namespace, job: Job) -> Tuple[Dict[str, object], Dict[str, object]]:
    path = run_dir(args, job)
    if not (path / "DONE").exists():
        raise RuntimeError(f"Missing completion marker: {path}")
    summary = c.read_json(path / "summary.json")
    if not isinstance(summary, dict):
        raise RuntimeError(f"Invalid summary object: {path / 'summary.json'}")
    expected = {
        "method": METHOD,
        "case_rank": CASE_RANK,
        "original_case_index": job.original_case_index,
        "fixed_id": EXPECTED_FIXED,
        "moving_id": EXPECTED_MOVING,
        "seed": int(args.seed),
        "cps": 1,
        "stage_early_stop_mode": "adaptive",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"{path}: {key}={summary.get(key)!r}, expected {value!r}")
    runtime_flags = {
        "cudnn_benchmark": summary.get("cudnn_benchmark"),
        "cudnn_deterministic": summary.get("cudnn_deterministic"),
    }
    if any(value is None for value in runtime_flags.values()):
        import torch

        state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=False)
        saved_args = state.get("args", {})
        runtime_flags = {
            "cudnn_benchmark": bool(saved_args.get("cudnn_benchmark")),
            "cudnn_deterministic": bool(saved_args.get("cudnn_deterministic")),
        }
        del state
    if runtime_flags != {"cudnn_benchmark": False, "cudnn_deterministic": True}:
        raise RuntimeError(f"{path}: unexpected backend flags {runtime_flags}")
    list_expected = {
        "stage_full_cps": STAGE_CPS,
        "stage_downsample": STAGE_DOWNSAMPLE,
        "stage_requested_epochs": STAGE_EPOCHS,
        "stage_lr": STAGE_LR,
        "stage_alpha": list(job.stage_alpha),
        "stage_min_epochs": STAGE_MIN_EPOCHS,
        "stage_patience": STAGE_PATIENCE,
        "stage_check_interval": STAGE_CHECK_INTERVAL,
        "stage_rel_delta_total": STAGE_REL_DELTA_TOTAL,
        "stage_rel_delta_data": STAGE_REL_DELTA_DATA,
    }
    for key, value in list_expected.items():
        observed = summary.get(key)
        if observed is None or not np.allclose(np.asarray(observed, dtype=float), np.asarray(value, dtype=float)):
            raise RuntimeError(f"{path}: {key}={observed!r}, expected {value!r}")
    for key in ("dice_mean", "hd95_mean", "asd_symmetric_mean"):
        if not math.isfinite(float(summary[key])):
            raise RuntimeError(f"{path}: non-finite {key}={summary[key]!r}")
    if summary.get("asd_backend") != "google_deepmind_surface_distance_area_weighted":
        raise RuntimeError(f"{path}: unexpected ASD backend {summary.get('asd_backend')!r}")
    validate_absent_labels(summary)
    dvf_path = Path(str(summary.get("dvf_file_path") or ""))
    if not dvf_path.exists():
        candidates = list(path.glob("disp_norm_*.npy"))
        if len(candidates) != 1:
            raise RuntimeError(f"{path}: cannot resolve saved DVF")
        dvf_path = candidates[0]
    dense = np.load(dvf_path, mmap_mode="r")
    if tuple(dense.shape) != EXPECTED_DVF_SHAPE:
        raise RuntimeError(f"{path}: DVF shape {dense.shape}, expected {EXPECTED_DVF_SHAPE}")
    if not np.isfinite(dense).all():
        raise RuntimeError(f"{path}: DVF contains non-finite values")
    prefix = {str(stage): read_stage_tail(path, stage) for stage in range(3)}
    return summary, {
        "run_dir": str(path),
        "dvf_path": str(dvf_path),
        "dvf_shape": list(dense.shape),
        "dvf_finite": True,
        "prefix_stage_tails": prefix,
    }


def prefix_validation(
    candidates: Sequence[Tuple[Job, Mapping[str, object], Mapping[str, object]]],
    tolerance: float,
) -> Dict[str, object]:
    reference_job, _reference_summary, reference_audit = candidates[0]
    reference = reference_audit["prefix_stage_tails"]
    comparisons: List[Dict[str, object]] = []
    passed = True
    for job, _summary, audit in candidates[1:]:
        stage_rows: List[Dict[str, object]] = []
        for stage in range(3):
            ref = reference[str(stage)]  # type: ignore[index]
            observed = audit["prefix_stage_tails"][str(stage)]  # type: ignore[index]
            deltas = {key: abs(float(observed[key]) - float(ref[key])) for key in ("loss", "data_loss", "reg_loss")}
            epoch_match = int(float(observed["epoch"])) == int(float(ref["epoch"]))
            stage_passed = epoch_match and all(delta <= float(tolerance) for delta in deltas.values())
            passed = passed and stage_passed
            stage_rows.append(
                {
                    "stage_idx": stage,
                    "epoch_match": epoch_match,
                    "deltas": deltas,
                    "passed": stage_passed,
                }
            )
        comparisons.append(
            {
                "reference_cps1_alpha": reference_job.cps1_alpha,
                "candidate_cps1_alpha": job.cps1_alpha,
                "stages": stage_rows,
            }
        )
    return {"passed": passed, "tolerance": float(tolerance), "comparisons": comparisons}


def candidate_row(job: Job, summary: Mapping[str, object], audit: Mapping[str, object]) -> Dict[str, object]:
    return {
        "method": f"MR-D-BSCP 8-4-2-1 (cps1 λ={job.cps1_alpha:g})",
        "kind": "new_candidate",
        "cps1_alpha": job.cps1_alpha,
        "stage_alpha": list(job.stage_alpha),
        "dice": float(summary["dice_mean"]),
        "hd95": float(summary["hd95_mean"]),
        "asd": float(summary["asd_symmetric_mean"]),
        "actual_epochs": int(summary["actual_epochs"]),
        "stage_actual_epochs": summary["stage_actual_epochs"],
        "stage_stop_reasons": summary["stage_stop_reasons"],
        "training_elapsed_sec": float(summary["training_elapsed_sec"]),
        "total_elapsed_sec": float(summary["total_elapsed_sec"]),
        "gpu": summary.get("gpu_requested"),
        "run_dir": audit["run_dir"],
        "source": summary.get("source_revision"),
    }


def fmt(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def markdown_report(
    combined_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    best: Mapping[str, Mapping[str, object]],
    overall: Mapping[str, object],
    prefix: Mapping[str, object],
) -> str:
    lines = [
        "# OASIS rank-0 MR-D-BSCP 8-4-2-1 lambda sweep",
        "",
        f"Fixed `{EXPECTED_FIXED}`, moving `{EXPECTED_MOVING}` (moving to fixed).",
        "",
        "## Metrics",
        "",
        "| Method | cps1 λ | Dice ↑ | HD95 ↓ | ASD ↓ | GPU | Actual epochs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined_rows:
        lines.append(
            "| {method} | {alpha} | {dice} | {hd95} | {asd} | {gpu} | {epochs} |".format(
                method=row["method"],
                alpha="" if row.get("cps1_alpha") is None else f"{float(row['cps1_alpha']):g}",
                dice=fmt(row["dice"]),
                hd95=fmt(row["hd95"]),
                asd=fmt(row["asd"]),
                gpu=row.get("gpu", ""),
                epochs=row.get("actual_epochs", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Sweep selection",
            "",
            f"- Best Dice: cps1 λ={float(best['dice']['cps1_alpha']):g} ({fmt(best['dice']['dice'])}).",
            f"- Best HD95: cps1 λ={float(best['hd95']['cps1_alpha']):g} ({fmt(best['hd95']['hd95'])}).",
            f"- Best ASD: cps1 λ={float(best['asd']['cps1_alpha']):g} ({fmt(best['asd']['asd'])}).",
            f"- Overall selection (Dice, then ASD, then HD95): cps1 λ={float(overall['cps1_alpha']):g}.",
            f"- Deterministic 8-4-2 prefix validation: {'PASS' if prefix['passed'] else 'FAIL'}.",
            "",
            "## Candidate diagnostics",
            "",
            "| cps1 λ | Stage epochs | Training seconds | Total seconds | Stop reasons |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in sorted(candidate_rows, key=lambda item: float(item["cps1_alpha"])):
        lines.append(
            f"| {float(row['cps1_alpha']):g} | `{json.dumps(row['stage_actual_epochs'])}` | "
            f"{fmt(row['training_elapsed_sec'], 3)} | {fmt(row['total_elapsed_sec'], 3)} | "
            f"`{json.dumps(row['stage_stop_reasons'])}` |"
        )
    return "\n".join(lines) + "\n"


def aggregate(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    validated: List[Tuple[Job, Dict[str, object], Dict[str, object]]] = []
    for job in jobs:
        summary, audit = validate_candidate(args, job)
        validated.append((job, summary, audit))
    prefix = prefix_validation(validated, float(args.prefix_tolerance))
    if not bool(prefix["passed"]):
        c.write_json(args.review_root / "prefix_validation.json", prefix)
        raise RuntimeError("The deterministic 8-4-2 prefix comparison failed")
    new_rows = [candidate_row(job, summary, audit) for job, summary, audit in validated]
    new_rows.sort(key=lambda row: float(row["cps1_alpha"]))
    best = {
        "dice": max(new_rows, key=lambda row: float(row["dice"])),
        "hd95": min(new_rows, key=lambda row: float(row["hd95"])),
        "asd": min(new_rows, key=lambda row: float(row["asd"])),
    }
    overall = sorted(
        new_rows,
        key=lambda row: (-float(row["dice"]), float(row["asd"]), float(row["hd95"])),
    )[0]
    combined_rows = [dict(row) for row in BASELINES] + new_rows
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "comparison.csv", combined_rows)
    c.write_json(args.review_root / "prefix_validation.json", prefix)
    c.write_json(
        args.review_root / "comparison.json",
        {
            "created_at": c.now_text(),
            "variant": VARIANT,
            "case_rank": CASE_RANK,
            "fixed_id": EXPECTED_FIXED,
            "moving_id": EXPECTED_MOVING,
            "direction": "moving_to_fixed",
            "rows": combined_rows,
            "best_candidate_by_metric": best,
            "overall_selected_candidate": overall,
            "prefix_validation": prefix,
            "candidate_audits": [audit for _job, _summary, audit in validated],
        },
    )
    (args.review_root / "report.md").write_text(
        markdown_report(combined_rows, new_rows, best, overall, prefix)
    )
    c.write_json(
        args.review_root / "ACCEPTED.json",
        {
            "accepted_at": c.now_text(),
            "candidate_count": len(new_rows),
            "all_complete": True,
            "all_dvfs_finite": True,
            "prefix_validation_passed": True,
            "overall_cps1_alpha": overall["cps1_alpha"],
        },
    )
    print(json.dumps({"report": str(args.review_root / "report.md"), "overall": overall}, indent=2))


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(str(args.gpus))
    jobs = build_jobs(args)
    source = source_metadata()
    if not args.aggregate_only:
        write_manifest(args, jobs, gpus, source)
        launch_jobs(args, jobs, gpus, source)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args, jobs)


if __name__ == "__main__":
    main()
