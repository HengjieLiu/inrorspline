#!/usr/bin/env python
"""Launch, validate, recover, and report the rank-0 saved-field fit matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_oasis_saved_field_bspline_fit_one as fit_runner  # noqa: E402


VARIANT = fit_runner.VARIANT
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
DEFAULT_OUT_ROOT = ARTIFACT_ROOT / "runs" / VARIANT
DEFAULT_REVIEW_ROOT = ARTIFACT_ROOT / "repo_experiments" / VARIANT
DEFAULT_OASIS_ROOT = Path(os.environ["BRAIN_DATA"]) if os.environ.get("BRAIN_DATA") else None
DEFAULT_DL_ROOT = (Path(os.environ["INRORSPLINE_OASIS_DL_ROOT"])
                   if os.environ.get("INRORSPLINE_OASIS_DL_ROOT") else None)
TARGETS = ("vfa", "sitreg")
CPS_VALUES = (8, 4, 2, 1)
LEARNING_RATES = (1e-2, 3e-3, 1e-3)
ALLOWED_GPUS = ("0", "1", "2", "3")
CASE_RANK = 0
EXPECTED_ORIGINAL_INDEX = 158
EXPECTED_FIXED = "0333"
EXPECTED_MOVING = "0115"
EXPECTED_DENSE_SHAPE = (*c.IMAGE_SHAPE, 3)
EXPECTED_UID = 1009
EXPECTED_GID = 1020


@dataclass(frozen=True)
class Job:
    job_index: int
    target: str
    cps: int
    lr: float
    preferred_gpu: Optional[str]
    case_rank: int = CASE_RANK
    original_case_index: int = EXPECTED_ORIGINAL_INDEX
    fixed_id: str = EXPECTED_FIXED
    moving_id: str = EXPECTED_MOVING


@dataclass(frozen=True)
class RecoveryJob:
    target: str
    cps: int
    lr: float
    source_run_dir: str
    source_best_epoch: int
    recovery_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(ALLOWED_GPUS))
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--poll-sec", type=float, default=3.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--dl-root", type=Path, default=DEFAULT_DL_ROOT)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--min-epochs", type=int, default=500)
    parser.add_argument("--check-interval", type=int, default=50)
    parser.add_argument("--median-window", type=int, default=50)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--relative-mse-threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nested-tolerance", type=float, default=0.001)
    parser.add_argument("--expected-uid", type=int, default=EXPECTED_UID)
    parser.add_argument("--expected-gid", type=int, default=EXPECTED_GID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    args = parser.parse_args()
    if args.oasis_root is None:
        parser.error("--oasis-root or BRAIN_DATA is required")
    if args.dl_root is None:
        parser.error("--dl-root or INRORSPLINE_OASIS_DL_ROOT is required")
    return args


def parse_gpus(text: str) -> List[str]:
    values = [token.strip() for token in str(text).split(",") if token.strip()]
    if not values or len(values) != len(set(values)) or any(value not in ALLOWED_GPUS for value in values):
        raise ValueError(f"Expected unique GPUs from {ALLOWED_GPUS}, got {values}")
    return values


def command_output(argv: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(argv), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata() -> Dict[str, object]:
    status = command_output(["git", "status", "--short", "--untracked-files=normal"])
    commit = command_output(["git", "rev-parse", "HEAD"])
    source_files = [Path(__file__).resolve(), (THIS_DIR / "run_oasis_saved_field_bspline_fit_one.py").resolve()]
    return {
        "git_commit": commit,
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_dirty": bool(status) and not status.startswith("unavailable:"),
        "git_status_short": status,
        "source_revision": f"{commit}-dirty" if status and not status.startswith("unavailable:") else commit,
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
    }


def build_jobs(gpus: Sequence[str]) -> List[Job]:
    # cps1 is largest, so launch it first; retain target/LR determinism within CPS.
    ordered: List[Tuple[str, int, float]] = []
    for cps in sorted(CPS_VALUES):
        for target in TARGETS:
            for lr in LEARNING_RATES:
                ordered.append((target, cps, lr))
    return [
        Job(index, target, cps, lr, str(gpus[index]) if index < len(gpus) else None)
        for index, (target, cps, lr) in enumerate(ordered)
    ]


def lr_key(value: float) -> str:
    return c.format_float(float(value))


def job_key(job: Job) -> str:
    return f"{job.target}_cps{job.cps}_lr{lr_key(job.lr)}"


def run_name(job: Job, attempt: int = 0) -> str:
    return (
        f"fieldfit_{job.target}_rank000_cps{job.cps}_fullvoxmse_lr{lr_key(job.lr)}"
        "_max5000_min500_pat200_ci50_win50_rt1em6_seed1"
        f"_attempt{int(attempt)}"
    )


def run_dir(args: argparse.Namespace, job: Job, attempt: int = 0) -> Path:
    return Path(args.out_root) / job.target / f"cps{job.cps}" / run_name(job, attempt)


def completed_attempt(args: argparse.Namespace, job: Job) -> Optional[int]:
    attempts = []
    for attempt in range(2):
        if (run_dir(args, job, attempt) / "DONE").is_file():
            attempts.append(attempt)
    if len(attempts) > 1:
        raise RuntimeError(f"Multiple accepted attempts for {job_key(job)}: {attempts}")
    return attempts[0] if attempts else None


def build_command(
    args: argparse.Namespace,
    job: Job,
    *,
    gpu: str,
    attempt: int,
    source_revision: str,
) -> List[str]:
    command = [
        sys.executable,
        str(THIS_DIR / "run_oasis_saved_field_bspline_fit_one.py"),
        "--target",
        job.target,
        "--cps",
        str(job.cps),
        "--lr",
        f"{job.lr:.12g}",
        "--gpu",
        str(gpu),
        "--case-rank",
        str(job.case_rank),
        "--max-epochs",
        str(args.max_epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--check-interval",
        str(args.check_interval),
        "--median-window",
        str(args.median_window),
        "--patience",
        str(args.patience),
        "--relative-mse-threshold",
        f"{args.relative_mse_threshold:.12g}",
        "--seed",
        str(args.seed),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--dl-root",
        str(args.dl_root),
        "--expected-original-case-index",
        str(job.original_case_index),
        "--expected-fixed-id",
        job.fixed_id,
        "--expected-moving-id",
        job.moving_id,
        "--source-revision",
        source_revision,
        "--run-name",
        run_name(job, attempt),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def recovery_name(job: RecoveryJob) -> str:
    return (
        f"fieldfit_{job.target}_rank000_cps{job.cps}_fullvoxmse_recovery{job.recovery_index}"
        f"_lr{lr_key(job.lr)}_extra5000_seed1"
    )


def recovery_run_dir(args: argparse.Namespace, job: RecoveryJob) -> Path:
    return Path(args.out_root) / job.target / f"cps{job.cps}" / recovery_name(job)


def build_recovery_command(
    args: argparse.Namespace,
    job: RecoveryJob,
    *,
    gpu: str,
    source_revision: str,
) -> List[str]:
    checkpoint = Path(job.source_run_dir) / "best_checkpoint.pt"
    return [
        sys.executable,
        str(THIS_DIR / "run_oasis_saved_field_bspline_fit_one.py"),
        "--target",
        job.target,
        "--cps",
        str(job.cps),
        "--lr",
        f"{job.lr:.12g}",
        "--gpu",
        str(gpu),
        "--case-rank",
        str(CASE_RANK),
        "--max-epochs",
        "5000",
        "--min-epochs",
        str(args.min_epochs),
        "--check-interval",
        str(args.check_interval),
        "--median-window",
        str(args.median_window),
        "--patience",
        str(args.patience),
        "--relative-mse-threshold",
        f"{args.relative_mse_threshold:.12g}",
        "--seed",
        str(args.seed),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--dl-root",
        str(args.dl_root),
        "--expected-original-case-index",
        str(EXPECTED_ORIGINAL_INDEX),
        "--expected-fixed-id",
        EXPECTED_FIXED,
        "--expected-moving-id",
        EXPECTED_MOVING,
        "--source-revision",
        source_revision,
        "--run-name",
        recovery_name(job),
        "--init-control",
        str(checkpoint),
        "--recovery-of",
        job.source_run_dir,
        "--initial-updates",
        str(job.source_best_epoch),
    ]


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        env[key] = str(max(1, int(cpu_threads)))
    return env


def gpu_snapshot(event: str, target_gpu: Optional[str] = None) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    error: Optional[str] = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6:
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
        "event": event,
        "target_gpu": target_gpu,
        "rows": rows,
        "error": error,
    }


def select_freest_gpu(gpus: Sequence[str]) -> str:
    snapshot = gpu_snapshot("select_freest")
    free = {str(row["gpu"]): int(row["memory_free_mib"]) for row in snapshot["rows"]}  # type: ignore[index]
    return max(gpus, key=lambda gpu: free.get(str(gpu), -1)) if free else str(gpus[0])


def append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(row), allow_nan=True, default=str) + "\n")


def write_manifest(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    case = c.resolve_case(rank=CASE_RANK, pair_file=args.pair_file, oasis_root=args.oasis_root)
    if (case.original_case_index, case.fixed_id, case.moving_id) != (
        EXPECTED_ORIGINAL_INDEX,
        EXPECTED_FIXED,
        EXPECTED_MOVING,
    ):
        raise RuntimeError(f"Rank-0 case identity changed: {case}")
    manifest = {
        "created_at": c.now_text(),
        "variant": VARIANT,
        "method": fit_runner.METHOD,
        "case": asdict(case),
        "direction": "moving_to_fixed",
        "targets": list(TARGETS),
        "cps": list(CPS_VALUES),
        "learning_rates": list(LEARNING_RATES),
        "candidate_count": len(jobs),
        "objective": {
            "name": "full_volume_voxel_mm_mse",
            "mask": "none",
            "regularizer": "none",
            "image_loss": "none",
            "voxel_count": int(np.prod(c.IMAGE_SHAPE)),
            "scalar_count": int(np.prod(c.IMAGE_SHAPE) * 3),
            "xyz_scales": [
                (c.IMAGE_SHAPE[2] - 1) / 2,
                (c.IMAGE_SHAPE[1] - 1) / 2,
                (c.IMAGE_SHAPE[0] - 1) / 2,
            ],
        },
        "optimization": {
            "optimizer": "Adam",
            "max_epochs": int(args.max_epochs),
            "min_epochs": int(args.min_epochs),
            "check_interval": int(args.check_interval),
            "median_window": int(args.median_window),
            "patience": int(args.patience),
            "relative_mse_threshold": float(args.relative_mse_threshold),
            "initialization": "independent_exact_zero",
            "seed": int(args.seed),
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "selection": "lowest full voxel MSE; normalized MSE; lower LR",
            "recovery": "offending selected nested-space fit, best checkpoint, LR/10, up to 5000 extra epochs",
        },
        "control_shapes": {str(key): list(value) for key, value in fit_runner.EXPECTED_CONTROL_SHAPES.items()},
        "gpus": list(gpus),
        "max_concurrent": min(int(args.max_concurrent), len(gpus)),
        "one_new_worker_per_gpu": True,
        "launcher_uid": os.getuid(),
        "launcher_gid": os.getgid(),
        "required_uid": int(args.expected_uid),
        "required_gid": int(args.expected_gid),
        "source": dict(source),
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "oasis_root": str(args.oasis_root),
        "pair_file": str(args.pair_file),
        "dl_root": str(args.dl_root),
        "jobs": [
            {
                **asdict(job),
                "job_key": job_key(job),
                "run_name_attempt0": run_name(job, 0),
                "run_dir_attempt0": str(run_dir(args, job, 0)),
            }
            for job in jobs
        ],
        "initial_gpu_snapshot": gpu_snapshot("manifest_start"),
    }
    c.write_json(args.out_root / "job_manifest.json", manifest)
    c.write_csv(args.out_root / "job_manifest.csv", [asdict(job) for job in jobs])


def is_oom_log(path: Path) -> bool:
    return path.is_file() and "out of memory" in path.read_text(errors="replace").lower()


def launch_jobs(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    pending: List[Tuple[Job, int, int]] = []
    for job in jobs:
        if args.overwrite or completed_attempt(args, job) is None:
            pending.append((job, 0, 0))
    running: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    completed_count = sum(completed_attempt(args, job) is not None for job in jobs)
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.out_root / "launch_events.jsonl"
    gpu_path = args.out_root / "gpu_observations.jsonl"
    source_revision = str(source["source_revision"])
    max_concurrent = min(max(1, int(args.max_concurrent)), len(gpus))

    if args.dry_run:
        for job, attempt, _wait_for in pending:
            gpu = job.preferred_gpu or gpus[job.job_index % len(gpus)]
            print(json.dumps({"job": asdict(job), "gpu": gpu, "command": build_command(
                args, job, gpu=gpu, attempt=attempt, source_revision=source_revision
            )}))
        return

    while pending or running:
        busy = {str(item["gpu"]) for item in running}
        free = [gpu for gpu in gpus if gpu not in busy]
        launchable = [item for item in pending if completed_count >= item[2]]
        while launchable and free and len(running) < max_concurrent:
            selected = next(
                (
                    item
                    for item in launchable
                    if item[0].preferred_gpu is None or item[0].preferred_gpu in free or item[1] > 0
                ),
                launchable[0],
            )
            pending.remove(selected)
            job, attempt, _wait_for = selected
            if attempt > 0:
                gpu = select_freest_gpu(free)
            elif job.preferred_gpu in free:
                gpu = str(job.preferred_gpu)
            else:
                gpu = str(free[0])
            command = build_command(
                args,
                job,
                gpu=gpu,
                attempt=attempt,
                source_revision=source_revision,
            )
            log_path = log_dir / f"{job_key(job)}_attempt{attempt}_gpu{gpu}.log"
            snapshot = gpu_snapshot("before_launch", gpu)
            append_jsonl(gpu_path, {**snapshot, "job_key": job_key(job), "attempt": attempt})
            handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=thread_env(args.cpu_threads),
            )
            running.append(
                {
                    "job": job,
                    "attempt": attempt,
                    "gpu": gpu,
                    "command": command,
                    "log_path": log_path,
                    "handle": handle,
                    "process": process,
                    "started": time.monotonic(),
                }
            )
            append_jsonl(
                event_path,
                {
                    "event": "candidate_start",
                    "timestamp": c.now_text(),
                    "job_key": job_key(job),
                    "attempt": attempt,
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                },
            )
            free.remove(gpu)
            if args.launch_stagger_sec > 0:
                time.sleep(float(args.launch_stagger_sec))
            launchable = [item for item in pending if completed_count >= item[2]]

        for item in list(running):
            process = item["process"]
            assert isinstance(process, subprocess.Popen)
            returncode = process.poll()
            if returncode is None:
                continue
            item["handle"].close()  # type: ignore[union-attr]
            running.remove(item)
            job = item["job"]
            assert isinstance(job, Job)
            attempt = int(item["attempt"])
            log_path = Path(item["log_path"])
            append_jsonl(
                gpu_path,
                {
                    **gpu_snapshot("after_exit", str(item["gpu"])),
                    "job_key": job_key(job),
                    "attempt": attempt,
                    "returncode": returncode,
                },
            )
            append_jsonl(
                event_path,
                {
                    "event": "candidate_exit",
                    "timestamp": c.now_text(),
                    "job_key": job_key(job),
                    "attempt": attempt,
                    "gpu": item["gpu"],
                    "returncode": returncode,
                    "elapsed_sec": time.monotonic() - float(item["started"]),
                },
            )
            if returncode == 0 and (run_dir(args, job, attempt) / "DONE").is_file():
                completed_count += 1
            elif attempt == 0 and is_oom_log(log_path):
                # Preserve attempt0 and wait for a different sweep worker to finish if one remains.
                wait_for = completed_count + (1 if running else 0)
                retry_job = Job(
                    job.job_index,
                    job.target,
                    job.cps,
                    job.lr,
                    None,
                    job.case_rank,
                    job.original_case_index,
                    job.fixed_id,
                    job.moving_id,
                )
                pending.append((retry_job, 1, wait_for))
            else:
                failures.append(
                    {
                        "job_key": job_key(job),
                        "attempt": attempt,
                        "gpu": item["gpu"],
                        "returncode": returncode,
                        "log_path": str(log_path),
                    }
                )
        if running or pending:
            if pending and not running and not any(completed_count >= item[2] for item in pending):
                # A failed peer cannot satisfy an OOM retry's wait gate. Release the
                # retry after all other workers have exited so the launcher terminates.
                pending = [(job, attempt, completed_count) for job, attempt, _wait in pending]
            time.sleep(max(0.25, float(args.poll_sec)))

    success = sum(completed_attempt(args, job) is not None for job in jobs)
    c.write_json(
        args.out_root / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "successful_candidates": success,
            "required_candidates": len(jobs),
            "failures": failures,
        },
    )
    if failures or success != len(jobs):
        raise RuntimeError(f"Candidate matrix incomplete: {success}/{len(jobs)}; failures={len(failures)}")


def validate_absent_labels(summary: Mapping[str, object], prefix: str) -> None:
    for label in range(1, 36):
        keys = (
            f"{prefix}_dice_label_{label:02d}",
            f"{prefix}_hd95_label_{label:02d}",
            f"{prefix}_asd_symmetric_label_{label:02d}",
        )
        values = [float(summary[key]) for key in keys]
        finite = [math.isfinite(value) for value in values]
        if any(finite) and not all(finite):
            raise RuntimeError(f"Inconsistent absent-label metrics for {prefix}, label {label}: {values}")


def validate_summary(
    path: Path,
    *,
    target: str,
    cps: int,
    expected_lr: float,
    args: argparse.Namespace,
    independent: bool,
) -> Dict[str, object]:
    if not (path / "DONE").is_file():
        raise RuntimeError(f"Missing DONE: {path}")
    summary = c.read_json(path / "summary.json")
    if not isinstance(summary, dict):
        raise RuntimeError(f"Invalid summary: {path}")
    expected: Dict[str, object] = {
        "variant": VARIANT,
        "method": fit_runner.METHOD,
        "target": target,
        "case_rank": CASE_RANK,
        "original_case_index": EXPECTED_ORIGINAL_INDEX,
        "fixed_id": EXPECTED_FIXED,
        "moving_id": EXPECTED_MOVING,
        "direction": "moving_to_fixed",
        "cps": int(cps),
        "bspline_order": 3,
        "regularizer": "none",
        "image_loss": "none",
        "training_mask": "none_full_volume",
        "objective_voxel_count": int(np.prod(c.IMAGE_SHAPE)),
        "objective_scalar_count": int(np.prod(c.IMAGE_SHAPE) * 3),
        "seed": int(args.seed),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "independent_fit": independent,
        "coarse_to_fine_initialization": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"{path}: {key}={summary.get(key)!r}, expected {value!r}")
    if tuple(summary.get("control_shape", [])) != fit_runner.EXPECTED_CONTROL_SHAPES[int(cps)]:
        raise RuntimeError(f"{path}: wrong control shape")
    if not math.isclose(float(summary["lr"]), float(expected_lr), rel_tol=0, abs_tol=1e-15):
        raise RuntimeError(f"{path}: wrong LR {summary['lr']} != {expected_lr}")
    if int(summary.get("uid", -1)) != int(args.expected_uid) or int(summary.get("gid", -1)) != int(args.expected_gid):
        raise RuntimeError(f"{path}: wrong UID/GID {summary.get('uid')}:{summary.get('gid')}")
    audit = summary.get("target_audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{path}: missing target audit")
    if audit.get("source_component_order") != "zyx_voxel" or audit.get("training_layout") != (
        "DHW3 normalized-grid xyz displacement"
    ):
        raise RuntimeError(f"{path}: target convention audit failed")
    finite_keys = (
        "full_voxel_mse",
        "full_voxel_rmse",
        "full_voxel_relative_rmse",
        "full_epe_mm_mean",
        "full_epe_mm_median",
        "full_epe_mm_p95",
        "full_epe_mm_max",
        "full_normalized_grid_mse",
        "fixed_positive_voxel_mse",
        "fixed_positive_voxel_rmse",
        "fixed_positive_voxel_relative_rmse",
        "fixed_positive_epe_mm_mean",
        "fitted_dice_mean",
        "fitted_hd95_mean",
        "fitted_asd_symmetric_mean",
        "source_dice_mean",
        "source_hd95_mean",
        "source_asd_symmetric_mean",
    )
    if any(not math.isfinite(float(summary[key])) for key in finite_keys):
        raise RuntimeError(f"{path}: non-finite scalar metric")
    validate_absent_labels(summary, "fitted")
    validate_absent_labels(summary, "source")
    dvfs = list(path.glob("disp_norm_*.npy"))
    if len(dvfs) != 1:
        raise RuntimeError(f"{path}: expected one dense field")
    dense = np.load(dvfs[0], mmap_mode="r")
    if tuple(dense.shape) != EXPECTED_DENSE_SHAPE or not np.isfinite(dense).all():
        raise RuntimeError(f"{path}: invalid dense field shape/values")
    if not (path / "best_checkpoint.pt").is_file() or not (path / "history.csv").is_file():
        raise RuntimeError(f"{path}: missing checkpoint/history")
    return summary


def initial_candidate_rows(args: argparse.Namespace, jobs: Sequence[Job]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    source_hashes: Dict[str, str] = {}
    source_metrics: Dict[str, Tuple[float, float, float]] = {}
    for job in jobs:
        attempt = completed_attempt(args, job)
        if attempt is None:
            raise RuntimeError(f"Missing successful attempt for {job_key(job)}")
        path = run_dir(args, job, attempt)
        summary = validate_summary(
            path,
            target=job.target,
            cps=job.cps,
            expected_lr=job.lr,
            args=args,
            independent=True,
        )
        audit = summary["target_audit"]
        assert isinstance(audit, dict)
        source_hash = str(audit["source_sha256"])
        source_metric = (
            float(summary["source_dice_mean"]),
            float(summary["source_hd95_mean"]),
            float(summary["source_asd_symmetric_mean"]),
        )
        if job.target in source_hashes and source_hashes[job.target] != source_hash:
            raise RuntimeError(f"{job.target}: target source hash changed across candidates")
        if job.target in source_metrics and not np.allclose(
            np.asarray(source_metrics[job.target]), np.asarray(source_metric), rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"{job.target}: source registration metrics changed across candidates")
        source_hashes[job.target] = source_hash
        source_metrics[job.target] = source_metric
        rows.append(summary_to_row(summary, path=path, candidate_kind="initial", attempt=attempt))
    return rows


def summary_to_row(
    summary: Mapping[str, object],
    *,
    path: Path,
    candidate_kind: str,
    attempt: int = 0,
) -> Dict[str, object]:
    keys = (
        "target",
        "target_label",
        "cps",
        "lr",
        "full_voxel_mse",
        "full_voxel_rmse",
        "full_voxel_relative_rmse",
        "full_voxel_target_power",
        "full_normalized_grid_mse",
        "full_normalized_grid_rmse",
        "full_normalized_grid_relative_rmse",
        "full_normalized_grid_target_power",
        "full_epe_mm_mean",
        "full_epe_mm_median",
        "full_epe_mm_p95",
        "full_epe_mm_max",
        "full_voxel_mse_x",
        "full_voxel_mse_y",
        "full_voxel_mse_z",
        "fixed_positive_voxel_mse",
        "fixed_positive_voxel_rmse",
        "fixed_positive_voxel_relative_rmse",
        "fixed_positive_voxel_target_power",
        "fixed_positive_normalized_grid_mse",
        "fixed_positive_normalized_grid_rmse",
        "fixed_positive_normalized_grid_relative_rmse",
        "fixed_positive_normalized_grid_target_power",
        "fixed_positive_epe_mm_mean",
        "fixed_positive_epe_mm_median",
        "fixed_positive_epe_mm_p95",
        "fixed_positive_epe_mm_max",
        "fixed_positive_voxel_mse_x",
        "fixed_positive_voxel_mse_y",
        "fixed_positive_voxel_mse_z",
        "fixed_positive_normalized_grid_mse_x",
        "fixed_positive_normalized_grid_mse_y",
        "fixed_positive_normalized_grid_mse_z",
        "full_normalized_grid_mse_x",
        "full_normalized_grid_mse_y",
        "full_normalized_grid_mse_z",
        "full_voxel_count",
        "full_scalar_count",
        "fixed_positive_voxel_count",
        "fixed_positive_scalar_count",
        "fitted_dice_mean",
        "fitted_hd95_mean",
        "fitted_asd_symmetric_mean",
        "source_dice_mean",
        "source_hd95_mean",
        "source_asd_symmetric_mean",
        "delta_dice_mean",
        "delta_hd95_mean",
        "delta_asd_symmetric_mean",
        "best_epoch",
        "optimizer_steps",
        "cumulative_optimizer_steps",
        "stopping_reason",
        "training_elapsed_sec",
        "total_elapsed_sec",
        "parameter_count",
        "gpu_requested",
        "source_revision",
        "recovery_of",
    )
    row = {key: summary.get(key) for key in keys}
    row.update(
        {
            "candidate_kind": candidate_kind,
            "attempt": int(attempt),
            "run_dir": str(path),
            "source_sha256": summary["target_audit"]["source_sha256"],  # type: ignore[index]
        }
    )
    return row


def select_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for target in TARGETS:
        for cps in CPS_VALUES:
            candidates = [row for row in rows if row["target"] == target and int(row["cps"]) == cps]
            if not candidates:
                raise RuntimeError(f"No candidates for {target}/cps{cps}")
            winner = min(
                candidates,
                key=lambda row: (
                    float(row["full_voxel_mse"]),
                    float(row["full_normalized_grid_mse"]),
                    float(row["lr"]),
                ),
            )
            selected.append(dict(winner))
    return selected


def monotonicity_audit(
    selected: Sequence[Mapping[str, object]], tolerance: float
) -> Dict[str, object]:
    comparisons: List[Dict[str, object]] = []
    offending: List[Dict[str, object]] = []
    for target in TARGETS:
        by_cps = {int(row["cps"]): row for row in selected if row["target"] == target}
        for coarse, fine in zip(CPS_VALUES, CPS_VALUES[1:]):
            coarse_mse = float(by_cps[coarse]["full_voxel_mse"])
            fine_mse = float(by_cps[fine]["full_voxel_mse"])
            passed = fine_mse <= coarse_mse * (1.0 + float(tolerance))
            row = {
                "target": target,
                "coarse_cps": coarse,
                "fine_cps": fine,
                "coarse_mse": coarse_mse,
                "fine_mse": fine_mse,
                "ratio_fine_over_coarse": fine_mse / max(coarse_mse, 1e-30),
                "tolerance_fraction": float(tolerance),
                "passed": passed,
            }
            comparisons.append(row)
            if not passed:
                offending.append({**row, "selected_fine": dict(by_cps[fine])})
    return {"passed": not offending, "comparisons": comparisons, "offending": offending}


def launch_recovery_batch(
    args: argparse.Namespace,
    recoveries: Sequence[RecoveryJob],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    pending = list(recoveries)
    running: List[Dict[str, object]] = []
    log_dir = args.out_root / "logs"
    event_path = args.out_root / "launch_events.jsonl"
    gpu_path = args.out_root / "gpu_observations.jsonl"
    failures: List[Dict[str, object]] = []
    while pending or running:
        busy = {str(item["gpu"]) for item in running}
        free = [gpu for gpu in gpus if gpu not in busy]
        while pending and free:
            job = pending.pop(0)
            out = recovery_run_dir(args, job)
            if (out / "DONE").is_file() and not args.overwrite:
                continue
            gpu = select_freest_gpu(free)
            command = build_recovery_command(
                args, job, gpu=gpu, source_revision=str(source["source_revision"])
            )
            log_path = log_dir / f"recovery_{job.target}_cps{job.cps}_{job.recovery_index}_gpu{gpu}.log"
            append_jsonl(gpu_path, {**gpu_snapshot("before_recovery", gpu), "recovery": asdict(job)})
            handle = log_path.open("w")
            process = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=thread_env(args.cpu_threads)
            )
            running.append(
                {
                    "job": job,
                    "gpu": gpu,
                    "process": process,
                    "handle": handle,
                    "log_path": log_path,
                    "started": time.monotonic(),
                }
            )
            append_jsonl(
                event_path,
                {
                    "event": "recovery_start",
                    "timestamp": c.now_text(),
                    "recovery": asdict(job),
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                },
            )
            free.remove(gpu)
        for item in list(running):
            process = item["process"]
            assert isinstance(process, subprocess.Popen)
            returncode = process.poll()
            if returncode is None:
                continue
            item["handle"].close()  # type: ignore[union-attr]
            running.remove(item)
            job = item["job"]
            assert isinstance(job, RecoveryJob)
            append_jsonl(
                event_path,
                {
                    "event": "recovery_exit",
                    "timestamp": c.now_text(),
                    "recovery": asdict(job),
                    "gpu": item["gpu"],
                    "returncode": returncode,
                    "elapsed_sec": time.monotonic() - float(item["started"]),
                },
            )
            if returncode != 0 or not (recovery_run_dir(args, job) / "DONE").is_file():
                failures.append(
                    {
                        "recovery": asdict(job),
                        "returncode": returncode,
                        "log_path": str(item["log_path"]),
                    }
                )
        if pending or running:
            time.sleep(max(0.25, float(args.poll_sec)))
    if failures:
        c.write_json(args.out_root / "recovery_failures.json", failures)
        raise RuntimeError(f"{len(failures)} monotonicity recovery fits failed")


def recover_monotonicity(
    args: argparse.Namespace,
    initial_rows: Sequence[Mapping[str, object]],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    all_rows = [dict(row) for row in initial_rows]
    recovery_rows: List[Dict[str, object]] = []
    recovered_pairs: set[Tuple[str, int]] = set()
    for round_index in range(1, 7):
        selected = select_rows(all_rows)
        audit = monotonicity_audit(selected, float(args.nested_tolerance))
        if audit["passed"]:
            return all_rows, selected, audit
        if args.skip_recovery:
            return all_rows, selected, audit
        recoveries: List[RecoveryJob] = []
        for offense in audit["offending"]:  # type: ignore[index]
            target = str(offense["target"])
            cps = int(offense["fine_cps"])
            key = (target, cps)
            if key in recovered_pairs:
                continue
            selected_fine = offense["selected_fine"]
            assert isinstance(selected_fine, dict)
            recoveries.append(
                RecoveryJob(
                    target=target,
                    cps=cps,
                    lr=float(selected_fine["lr"]) / 10.0,
                    source_run_dir=str(selected_fine["run_dir"]),
                    source_best_epoch=int(selected_fine["best_epoch"]),
                    recovery_index=round_index,
                )
            )
            recovered_pairs.add(key)
        if not recoveries:
            return all_rows, selected, audit
        launch_recovery_batch(args, recoveries, gpus, source)
        for recovery in recoveries:
            path = recovery_run_dir(args, recovery)
            summary = validate_summary(
                path,
                target=recovery.target,
                cps=recovery.cps,
                expected_lr=recovery.lr,
                args=args,
                independent=False,
            )
            row = summary_to_row(summary, path=path, candidate_kind="recovery")
            recovery_rows.append(row)
            all_rows.append(row)
    selected = select_rows(all_rows)
    return all_rows, selected, monotonicity_audit(selected, float(args.nested_tolerance))


def matrix_rows(selected: Sequence[Mapping[str, object]], metric: str) -> List[Dict[str, object]]:
    rows = []
    for target in TARGETS:
        row: Dict[str, object] = {"target": target}
        for cps in CPS_VALUES:
            match = next(item for item in selected if item["target"] == target and int(item["cps"]) == cps)
            row[f"cps{cps}"] = match[metric]
        rows.append(row)
    return rows


def fmt(value: object, digits: int = 8) -> str:
    return f"{float(value):.{digits}g}"


def markdown_matrix(title: str, rows: Sequence[Mapping[str, object]]) -> List[str]:
    lines = [f"### {title}", "", "| Target | cps8 | cps4 | cps2 | cps1 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {str(row['target']).upper()} | {fmt(row['cps8'])} | {fmt(row['cps4'])} "
            f"| {fmt(row['cps2'])} | {fmt(row['cps1'])} |"
        )
    return lines


def markdown_report(
    initial_rows: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    recovery_rows: Sequence[Mapping[str, object]],
    monotonicity: Mapping[str, object],
) -> str:
    mse = matrix_rows(selected, "full_voxel_mse")
    rmse = matrix_rows(selected, "full_voxel_rmse")
    rrmse = matrix_rows(selected, "full_voxel_relative_rmse")
    lines = [
        "# OASIS rank-0 VFA/SITReg cubic B-spline approximation benchmark",
        "",
        f"Fixed `{EXPECTED_FIXED}`, moving `{EXPECTED_MOVING}` (moving → fixed). Each cps fit starts independently from zero; the training objective is unmasked full-volume voxel/mm MSE with no regularizer.",
        "",
        "## Headline selected fits",
        "",
        *markdown_matrix("Voxel/mm MSE ↓", mse),
        "",
        *markdown_matrix("Voxel/mm RMSE ↓", rmse),
        "",
        *markdown_matrix("Relative RMSE ↓", rrmse),
        "",
        "## Selected fit details",
        "",
        "| Target | CPS | LR | MSE | RMSE | rRMSE | EPE mean | EPE p95 | Dice | HD95 | ASD | ΔDice | ΔHD95 | ΔASD | Best epoch | GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (str(item["target"]), -int(item["cps"]))):
        lines.append(
            f"| {str(row['target']).upper()} | {int(row['cps'])} | {float(row['lr']):g} "
            f"| {fmt(row['full_voxel_mse'])} | {fmt(row['full_voxel_rmse'])} "
            f"| {fmt(row['full_voxel_relative_rmse'])} | {fmt(row['full_epe_mm_mean'])} "
            f"| {fmt(row['full_epe_mm_p95'])} | {fmt(row['fitted_dice_mean'], 7)} "
            f"| {fmt(row['fitted_hd95_mean'], 7)} | {fmt(row['fitted_asd_symmetric_mean'], 7)} "
            f"| {float(row['delta_dice_mean']):+.6f} | {float(row['delta_hd95_mean']):+.6f} "
            f"| {float(row['delta_asd_symmetric_mean']):+.6f} | {int(row['best_epoch'])} "
            f"| {row['gpu_requested']} |"
        )
    lines += [
        "",
        "## All 24 independent candidates",
        "",
        "| Target | CPS | LR | Full MSE | Masked MSE | Norm MSE | Best epoch | Stop | Runtime s | GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in sorted(initial_rows, key=lambda item: (str(item["target"]), -int(item["cps"]), -float(item["lr"]))):
        lines.append(
            f"| {str(row['target']).upper()} | {int(row['cps'])} | {float(row['lr']):g} "
            f"| {fmt(row['full_voxel_mse'])} | {fmt(row['fixed_positive_voxel_mse'])} "
            f"| {fmt(row['full_normalized_grid_mse'])} | {int(row['best_epoch'])} "
            f"| {row['stopping_reason']} | {float(row['training_elapsed_sec']):.2f} | {row['gpu_requested']} |"
        )
    lines += [
        "",
        "## Validation",
        "",
        f"- Initial candidates complete: `{len(initial_rows)}/24`.",
        f"- Selected fits: `{len(selected)}/8`.",
        f"- Nested-space ordering with 0.1% tolerance: `{'PASS' if monotonicity['passed'] else 'FAIL'}`.",
        f"- Recovery continuations used: `{len(recovery_rows)}`.",
        "- Canonical registration metrics use the shared absent-label and symmetric area-weighted ASD pipeline.",
    ]
    return "\n".join(lines) + "\n"


def write_curves(review_root: Path, selected: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    curve_rows = []
    for row in selected:
        curve_rows.append(
            {
                "target": row["target"],
                "cps": row["cps"],
                "voxel_mse": row["full_voxel_mse"],
                "voxel_rmse": row["full_voxel_rmse"],
                "relative_rmse": row["full_voxel_relative_rmse"],
            }
        )
    c.write_csv(review_root / "error_vs_cps.csv", curve_rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        specs = (
            ("full_voxel_mse", "Voxel MSE"),
            ("full_voxel_rmse", "Voxel RMSE"),
            ("full_voxel_relative_rmse", "Relative RMSE"),
        )
        for target in TARGETS:
            rows = sorted(
                [row for row in selected if row["target"] == target], key=lambda item: -int(item["cps"])
            )
            x = [int(row["cps"]) for row in rows]
            for axis, (metric, label) in zip(axes, specs):
                axis.plot(x, [float(row[metric]) for row in rows], marker="o", label=target.upper())
                axis.set_xlabel("Control-point spacing (voxels)")
                axis.set_ylabel(label)
                axis.set_xticks([8, 4, 2, 1])
                axis.invert_xaxis()
                axis.grid(True, alpha=0.25)
        axes[0].set_yscale("log")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(review_root / "error_vs_cps.png", dpi=180)
        fig.savefig(review_root / "error_vs_cps.pdf")
        plt.close(fig)
        return {"passed": True, "error": None}
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def aggregate(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    source: Mapping[str, object],
) -> None:
    initial_rows = initial_candidate_rows(args, jobs)
    all_rows, selected, monotonicity = recover_monotonicity(args, initial_rows, gpus, source)
    recovery_rows = [row for row in all_rows if row["candidate_kind"] == "recovery"]
    if len(initial_rows) != 24 or len(selected) != 8:
        raise RuntimeError(f"Wrong result cardinality: initial={len(initial_rows)}, selected={len(selected)}")

    args.review_root.mkdir(parents=True, exist_ok=True)
    mse_matrix = matrix_rows(selected, "full_voxel_mse")
    rmse_matrix = matrix_rows(selected, "full_voxel_rmse")
    rrmse_matrix = matrix_rows(selected, "full_voxel_relative_rmse")
    c.write_csv(args.review_root / "all_24_candidates.csv", initial_rows)
    c.write_csv(args.review_root / "selected_8_fits.csv", selected)
    c.write_csv(args.review_root / "recovery_candidates.csv", recovery_rows)
    c.write_csv(args.review_root / "headline_voxel_mse.csv", mse_matrix)
    c.write_csv(args.review_root / "headline_voxel_rmse.csv", rmse_matrix)
    c.write_csv(args.review_root / "headline_relative_rmse.csv", rrmse_matrix)
    c.write_json(args.review_root / "nested_space_validation.json", monotonicity)
    curve_status = write_curves(args.review_root, selected)
    payload = {
        "created_at": c.now_text(),
        "variant": VARIANT,
        "case": {
            "rank": CASE_RANK,
            "original_index": EXPECTED_ORIGINAL_INDEX,
            "fixed_id": EXPECTED_FIXED,
            "moving_id": EXPECTED_MOVING,
            "direction": "moving_to_fixed",
        },
        "selection_rule": "minimum full voxel MSE; exact ties by normalized-grid MSE then lower LR",
        "all_24_candidates": initial_rows,
        "recovery_candidates": recovery_rows,
        "selected_8_fits": selected,
        "headline_voxel_mse": mse_matrix,
        "headline_voxel_rmse": rmse_matrix,
        "headline_relative_rmse": rrmse_matrix,
        "nested_space_validation": monotonicity,
        "curve_status": curve_status,
        "source": dict(source),
    }
    c.write_json(args.review_root / "comparison.json", payload)
    (args.review_root / "report.md").write_text(
        markdown_report(initial_rows, selected, recovery_rows, monotonicity)
    )
    accepted = bool(monotonicity["passed"]) and bool(curve_status["passed"])
    acceptance = {
        "accepted_at": c.now_text() if accepted else None,
        "accepted": accepted,
        "initial_candidates": len(initial_rows),
        "selected_fits": len(selected),
        "finite_dense_fields": 24,
        "full_volume_objective": True,
        "no_regularization": True,
        "nested_space_ordering_passed": bool(monotonicity["passed"]),
        "curve_generation_passed": bool(curve_status["passed"]),
        "recovery_count": len(recovery_rows),
    }
    c.write_json(args.review_root / ("ACCEPTED.json" if accepted else "NOT_ACCEPTED.json"), acceptance)
    if not accepted:
        raise RuntimeError(f"Benchmark failed acceptance: {acceptance}")
    print(json.dumps({"report": str(args.review_root / "report.md"), "acceptance": acceptance}, indent=2))


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    if not args.dry_run and not args.aggregate_only:
        if os.getuid() != int(args.expected_uid) or os.getgid() != int(args.expected_gid):
            raise RuntimeError(
                f"Launch must run as UID:GID {args.expected_uid}:{args.expected_gid}; "
                f"observed {os.getuid()}:{os.getgid()}"
            )
    jobs = build_jobs(gpus)
    source = source_metadata()
    if not args.aggregate_only:
        write_manifest(args, jobs, gpus, source)
        launch_jobs(args, jobs, gpus, source)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args, jobs, gpus, source)


if __name__ == "__main__":
    main()
