#!/usr/bin/env python
"""Run and aggregate the rank-0 OASIS 8-4-2 -> dense-field lambda sweep."""

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


METHOD = "direct_cp_8_4_2_then_dense_fd_be_multistage"
VARIANT = "oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1"
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
DEFAULT_OUT_ROOT = ARTIFACT_ROOT / "runs" / VARIANT
DEFAULT_REVIEW_ROOT = ARTIFACT_ROOT / "repo_experiments" / VARIANT
DEFAULT_BSPLINE_REVIEW_ROOT = ARTIFACT_ROOT / "repo_experiments" / (
    "oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1"
)
DEFAULT_OASIS_ROOT = Path(os.environ["BRAIN_DATA"]) if os.environ.get("BRAIN_DATA") else None
PREFIX_CPS = [8, 4, 2]
PREFIX_DOWNSAMPLE = [4, 2, 1]
PREFIX_EPOCHS = [400, 800, 1300]
PREFIX_LR = [3e-2, 1e-2, 3e-3]
PREFIX_ALPHA = [1000.0, 1000.0, 1000.0]
PREFIX_MIN_EPOCHS = [40, 80, 130]
PREFIX_PATIENCE = [20, 35, 50]
PREFIX_CHECK_INTERVAL = [25, 25, 25]
PREFIX_REL_DELTA_TOTAL = [1e-4] * 3
PREFIX_REL_DELTA_DATA = [5e-5] * 3
DENSE_LR = 1e-3
DENSE_EPOCHS = 1300
DENSE_MIN_EPOCHS = 130
DENSE_PATIENCE = 50
DENSE_CHECK_INTERVAL = 25
DENSE_REL_DELTA_TOTAL = 1e-4
DENSE_REL_DELTA_DATA = 5e-5
CASE_RANK = 0
EXPECTED_FIXED = "0333"
EXPECTED_MOVING = "0115"
EXPECTED_DVF_SHAPE = (*c.IMAGE_SHAPE, 3)
ALLOWED_GPUS = {"0", "1", "2", "3"}

# Four requested initial workers followed by the queued fifth candidate.
DEFAULT_JOBS: Sequence[Tuple[float, Optional[str]]] = (
    (10.0, "0"),
    (100.0, "1"),
    (10000.0, "2"),
    (1000.0, "3"),
    (100000.0, None),
)

BASELINES: Sequence[Mapping[str, object]] = (
    {"method": "MR-D-BSCP 8-4-2", "kind": "existing_baseline", "dice": 0.7607446419879308,
     "hd95": 2.6523419316255900, "asd": 0.5925002179651616},
    {"method": "Greedy", "kind": "existing_baseline", "dice": 0.7712877733642253,
     "hd95": 2.1963895289869924, "asd": 0.5075983977968538},
    {"method": "VFA", "kind": "existing_baseline", "dice": 0.7968203408706900,
     "hd95": 2.0485631555197186, "asd": 0.4400145870843727},
    {"method": "SITReg", "kind": "existing_baseline", "dice": 0.7987193557676338,
     "hd95": 2.0613193369724470, "asd": 0.42306412147236566},
)


@dataclass(frozen=True)
class Job:
    job_index: int
    dense_alpha: float
    preferred_gpu: Optional[str]
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str


def csv_text(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def alpha_key(value: float) -> str:
    return c.format_float(float(value))


def command_output(argv: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(argv), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def source_metadata() -> Dict[str, object]:
    status = command_output(["git", "status", "--short", "--untracked-files=normal"])
    commit = command_output(["git", "rev-parse", "HEAD"])
    return {
        "git_commit": commit,
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_dirty": bool(status) and not status.startswith("unavailable:"),
        "git_status_short": status,
        "source_revision": f"{commit}-dirty" if status and not status.startswith("unavailable:") else commit,
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
    parser.add_argument("--bspline-review-root", type=Path, default=DEFAULT_BSPLINE_REVIEW_ROOT)
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
    if not gpus or len(gpus) != len(set(gpus)) or set(gpus) - ALLOWED_GPUS:
        raise ValueError(f"Expected unique physical GPUs from {sorted(ALLOWED_GPUS)}, got {gpus}")
    return gpus


def build_jobs(args: argparse.Namespace) -> List[Job]:
    case = c.resolve_case(rank=CASE_RANK, pair_file=args.pair_file, oasis_root=args.oasis_root)
    if (case.fixed_id, case.moving_id) != (EXPECTED_FIXED, EXPECTED_MOVING):
        raise RuntimeError(
            f"Rank 0 changed: observed {case.fixed_id}/{case.moving_id}, "
            f"expected {EXPECTED_FIXED}/{EXPECTED_MOVING}"
        )
    return [
        Job(index, float(alpha), gpu, CASE_RANK, int(case.original_case_index),
            str(case.fixed_id), str(case.moving_id))
        for index, (alpha, gpu) in enumerate(DEFAULT_JOBS)
    ]


def run_name(job: Job) -> str:
    return (
        f"{METHOD}_rank{job.case_rank:03d}_sched8-4-2-dense_ds4-2-1-1"
        f"_e400-800-1300-1300_sa1000-1000-1000-da{alpha_key(job.dense_alpha)}"
        "_lr3em2-1em2-3em3-1em3_esadaptive_min40-80-130-130"
        "_pat20-35-50-50_ci25-25-25-25_rt1em4_rd5em5_seed1"
    )


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / METHOD / run_name(job)


def done_path(args: argparse.Namespace, job: Job) -> Path:
    return run_dir(args, job) / "DONE"


def build_command(args: argparse.Namespace, job: Job, gpu: str, revision: str) -> List[str]:
    command = [
        sys.executable, str(THIS_DIR / "run_one.py"),
        "--method", METHOD,
        "--alpha", "1000",
        "--stage-alpha", csv_text(PREFIX_ALPHA),
        "--dense-stage-alpha", f"{job.dense_alpha:g}",
        "--gpu", str(gpu),
        "--case-rank", str(job.case_rank),
        "--epochs", str(sum(PREFIX_EPOCHS) + DENSE_EPOCHS),
        "--eval-interval", str(args.eval_interval),
        "--lr", "0.01",
        "--stage-full-cps", csv_text(PREFIX_CPS),
        "--stage-downsample", csv_text(PREFIX_DOWNSAMPLE),
        "--stage-epochs", csv_text(PREFIX_EPOCHS),
        "--stage-lr", csv_text(PREFIX_LR),
        "--stage-early-stop-mode", "adaptive",
        "--stage-min-epochs", csv_text(PREFIX_MIN_EPOCHS),
        "--stage-patience", csv_text(PREFIX_PATIENCE),
        "--stage-check-interval", csv_text(PREFIX_CHECK_INTERVAL),
        "--stage-rel-delta-total", csv_text(PREFIX_REL_DELTA_TOTAL),
        "--stage-rel-delta-data", csv_text(PREFIX_REL_DELTA_DATA),
        "--dense-stage-lr", str(DENSE_LR),
        "--dense-stage-epochs", str(DENSE_EPOCHS),
        "--dense-stage-early-stop-mode", "adaptive",
        "--dense-stage-min-epochs", str(DENSE_MIN_EPOCHS),
        "--dense-stage-patience", str(DENSE_PATIENCE),
        "--dense-stage-check-interval", str(DENSE_CHECK_INTERVAL),
        "--dense-stage-rel-delta-total", str(DENSE_REL_DELTA_TOTAL),
        "--dense-stage-rel-delta-data", str(DENSE_REL_DELTA_DATA),
        "--seed", str(args.seed),
        "--cudnn-benchmark", "false",
        "--cudnn-deterministic", "true",
        "--prolongation-preserve-check",
        "--variant", VARIANT,
        "--runtime-candidate", f"dense_alpha_{alpha_key(job.dense_alpha)}",
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


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        env[key] = str(max(1, int(cpu_threads)))
    return env


def gpu_snapshot(event: str, target_gpu: Optional[str] = None) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    error: Optional[str] = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6:
                rows.append({"gpu": parts[0], "name": parts[1], "memory_total_mib": int(parts[2]),
                             "memory_used_mib": int(parts[3]), "memory_free_mib": int(parts[4]),
                             "utilization_gpu_pct": int(parts[5])})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"timestamp": c.now_text(), "event": event, "target_gpu": target_gpu,
            "rows": rows, "error": error}


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, allow_nan=True, default=str) + "\n")


def select_freest_gpu(gpus: Sequence[str]) -> str:
    rows = gpu_snapshot("retry_gpu_selection")["rows"]
    free = {str(row["gpu"]): int(row["memory_free_mib"]) for row in rows}  # type: ignore[union-attr]
    return max(gpus, key=lambda gpu: free.get(gpu, -1)) if free else str(gpus[0])


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str],
                   source: Mapping[str, object]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    c.write_json(args.out_root / "job_manifest.json", {
        "created_at": c.now_text(), "variant": VARIANT, "method": METHOD,
        "case_rank": CASE_RANK, "fixed_id": EXPECTED_FIXED, "moving_id": EXPECTED_MOVING,
        "direction": "moving_to_fixed", "launcher_uid": os.getuid(), "launcher_gid": os.getgid(),
        "gpus": list(gpus), "max_concurrent": min(int(args.max_concurrent), len(gpus)),
        "prefix": {"stage_full_cps": PREFIX_CPS, "stage_downsample": PREFIX_DOWNSAMPLE,
                   "stage_epochs": PREFIX_EPOCHS, "stage_lr": PREFIX_LR,
                   "stage_alpha": PREFIX_ALPHA, "stage_min_epochs": PREFIX_MIN_EPOCHS,
                   "stage_patience": PREFIX_PATIENCE, "stage_check_interval": PREFIX_CHECK_INTERVAL,
                   "stage_rel_delta_total": PREFIX_REL_DELTA_TOTAL,
                   "stage_rel_delta_data": PREFIX_REL_DELTA_DATA,
                   "representation": "cubic_bspline_control_displacement",
                   "regularizer": "analytic_cubic_bspline_be"},
        "dense_stage": {"representation": "dense_voxel", "alpha_sweep": sorted(job.dense_alpha for job in jobs),
                        "lr": DENSE_LR, "epochs": DENSE_EPOCHS, "min_epochs": DENSE_MIN_EPOCHS,
                        "patience": DENSE_PATIENCE, "check_interval": DENSE_CHECK_INTERVAL,
                        "rel_delta_total": DENSE_REL_DELTA_TOTAL,
                        "rel_delta_data": DENSE_REL_DELTA_DATA,
                        "image_loss": "full_image_global_ncc",
                        "regularizer": "unmasked_forward_valid_dense_fd_be",
                        "initialization": "expanded_cps2_absolute_field_detach_clone"},
        "seed": int(args.seed), "cudnn_benchmark": False, "cudnn_deterministic": True,
        "out_root": str(args.out_root), "review_root": str(args.review_root),
        "bspline_review_root": str(args.bspline_review_root), "source": dict(source),
        "jobs": [{**asdict(job), "run_name": run_name(job), "run_dir": str(run_dir(args, job))}
                 for job in jobs],
        "initial_gpu_snapshot": gpu_snapshot("manifest_start"),
    })
    c.write_csv(args.out_root / "job_manifest.csv", [asdict(job) for job in jobs])


def is_oom_log(path: Path) -> bool:
    return path.exists() and "out of memory" in path.read_text(errors="replace").lower()


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str],
                source: Mapping[str, object]) -> None:
    max_concurrent = min(max(1, int(args.max_concurrent)), len(gpus))
    pending: List[Tuple[Job, int]] = [
        (job, 0) for job in jobs if args.overwrite or not done_path(args, job).exists()
    ]
    running: List[Dict[str, object]] = []
    retry_waiting: List[Tuple[Job, int, int]] = []
    failures: List[Dict[str, object]] = []
    successful_completions = 0
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.out_root / "launch_events.jsonl"
    gpu_log = args.out_root / "gpu_observations.jsonl"
    revision = str(source["source_revision"])
    if args.dry_run:
        for job, _attempt in pending:
            gpu = job.preferred_gpu or str(gpus[0])
            print(json.dumps({"gpu": gpu, "job": asdict(job),
                              "command": build_command(args, job, gpu, revision)}))
        return

    while pending or running or retry_waiting:
        for retry in list(retry_waiting):
            if successful_completions >= retry[2]:
                retry_waiting.remove(retry)
                pending.append((retry[0], retry[1]))
        busy = {str(item["gpu"]) for item in running}
        free = [gpu for gpu in gpus if gpu not in busy]
        while pending and free and len(running) < max_concurrent:
            selected = next((i for i, (job, _retry) in enumerate(pending)
                             if job.preferred_gpu is None or job.preferred_gpu in free), None)
            if selected is None:
                break
            job, retry_count = pending.pop(selected)
            gpu = select_freest_gpu(free) if retry_count else (
                job.preferred_gpu if job.preferred_gpu in free else free[0]
            )
            attempt = retry_count
            log_path = log_dir / f"dense_alpha_{alpha_key(job.dense_alpha)}_attempt{attempt}_gpu{gpu}.log"
            while log_path.exists():
                attempt += 1
                log_path = log_dir / f"dense_alpha_{alpha_key(job.dense_alpha)}_attempt{attempt}_gpu{gpu}.log"
            snapshot = gpu_snapshot("before_launch", gpu)
            append_jsonl(gpu_log, {**snapshot, "dense_alpha": job.dense_alpha, "attempt": attempt})
            command = build_command(args, job, gpu, revision)
            handle = log_path.open("w")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=thread_env(args.cpu_threads))
            running.append({"job": job, "gpu": gpu, "retry_count": retry_count,
                            "attempt": attempt, "process": process, "handle": handle,
                            "log_path": log_path, "start": time.monotonic()})
            append_jsonl(event_log, {"event": "job_start", "timestamp": c.now_text(),
                                     "dense_alpha": job.dense_alpha, "gpu": gpu, "attempt": attempt,
                                     "pid": process.pid, "command": command, "log_path": str(log_path)})
            free.remove(gpu)
            if args.launch_stagger_sec > 0:
                time.sleep(args.launch_stagger_sec)

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
            log_path = Path(item["log_path"])
            append_jsonl(gpu_log, {**gpu_snapshot("after_exit", str(item["gpu"])),
                                   "dense_alpha": job.dense_alpha, "attempt": item["attempt"],
                                   "returncode": returncode})
            append_jsonl(event_log, {"event": "job_exit", "timestamp": c.now_text(),
                                     "dense_alpha": job.dense_alpha, "gpu": item["gpu"],
                                     "attempt": item["attempt"], "returncode": returncode,
                                     "elapsed_sec": time.monotonic() - float(item["start"])})
            if returncode == 0 and done_path(args, job).exists():
                successful_completions += 1
            elif is_oom_log(log_path) and int(item["retry_count"]) == 0:
                retry_job = Job(job.job_index, job.dense_alpha, None, job.case_rank,
                                job.original_case_index, job.fixed_id, job.moving_id)
                retry_waiting.append((retry_job, 1, successful_completions + (1 if running else 0)))
            else:
                failures.append({"dense_alpha": job.dense_alpha, "gpu": item["gpu"],
                                 "attempt": item["attempt"], "returncode": returncode,
                                 "log_path": str(log_path)})
        if running or retry_waiting:
            time.sleep(max(0.25, float(args.poll_sec)))
        elif pending:
            pending = [(Job(job.job_index, job.dense_alpha, None, job.case_rank,
                            job.original_case_index, job.fixed_id, job.moving_id), retry)
                       if job.preferred_gpu not in gpus else (job, retry) for job, retry in pending]

    c.write_json(args.out_root / "launch_summary.json", {
        "completed_at": c.now_text(), "successful_count": sum(done_path(args, job).exists() for job in jobs),
        "failures": failures,
    })
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see {args.out_root / 'launch_summary.json'}")


def read_stage_tail(path: Path, stage_idx: int) -> Dict[str, float]:
    with (path / "stage_history.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(float(row["stage_idx"])) == stage_idx]
    if not rows:
        raise RuntimeError(f"Missing stage {stage_idx} history in {path}")
    return {key: float(rows[-1][key]) for key in ("epoch", "loss", "data_loss", "reg_loss")}


def validate_absent_labels(summary: Mapping[str, object]) -> None:
    for label in range(1, 36):
        values = [float(summary[f"dice_label_{label:02d}"]),
                  float(summary[f"hd95_label_{label:02d}"]),
                  float(summary[f"asd_symmetric_label_{label:02d}"])]
        finite = [math.isfinite(value) for value in values]
        if any(finite) and not all(finite):
            raise RuntimeError(f"Inconsistent absent-label metrics for label {label}: {values}")


def validate_candidate(args: argparse.Namespace, job: Job) -> Tuple[Dict[str, object], Dict[str, object]]:
    path = run_dir(args, job)
    if not (path / "DONE").exists():
        raise RuntimeError(f"Missing completion marker: {path}")
    summary = c.read_json(path / "summary.json")
    assert isinstance(summary, dict)
    expected = {"method": METHOD, "case_rank": CASE_RANK,
                "original_case_index": job.original_case_index, "fixed_id": EXPECTED_FIXED,
                "moving_id": EXPECTED_MOVING, "seed": int(args.seed), "cps": None,
                "final_field_representation": "dense_voxel_displacement",
                "final_field_is_single_bspline": False, "dense_stage_uses_cps1": False,
                "dense_stage_uses_analytic_bspline_be": False,
                "be_training_reduction":
                    "dense valid-stencil finite-difference BE over the unmasked full voxel field; "
                    "spatial mean plus component mean"}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"{path}: {key}={summary.get(key)!r}, expected {value!r}")
    if summary.get("stage_full_cps") != [8, 4, 2, None]:
        raise RuntimeError(f"{path}: wrong stage_full_cps {summary.get('stage_full_cps')!r}")
    numeric_expected = {
        "stage_downsample": [4, 2, 1, 1],
        "stage_requested_epochs": [400, 800, 1300, 1300],
        "stage_lr": [*PREFIX_LR, DENSE_LR],
        "stage_alpha": [*PREFIX_ALPHA, job.dense_alpha],
        "stage_min_epochs": [*PREFIX_MIN_EPOCHS, DENSE_MIN_EPOCHS],
        "stage_patience": [*PREFIX_PATIENCE, DENSE_PATIENCE],
        "stage_check_interval": [*PREFIX_CHECK_INTERVAL, DENSE_CHECK_INTERVAL],
        "stage_rel_delta_total": [*PREFIX_REL_DELTA_TOTAL, DENSE_REL_DELTA_TOTAL],
        "stage_rel_delta_data": [*PREFIX_REL_DELTA_DATA, DENSE_REL_DELTA_DATA],
    }
    for key, value in numeric_expected.items():
        if not np.allclose(np.asarray(summary.get(key), dtype=float), np.asarray(value, dtype=float)):
            raise RuntimeError(f"{path}: {key}={summary.get(key)!r}, expected {value!r}")
    if summary.get("stage_representation") != ["bspline_control"] * 3 + ["dense_voxel"]:
        raise RuntimeError(f"{path}: incorrect stage representations")
    audit = summary.get("dense_initialization_audit")
    if not isinstance(audit, dict) or any(float(value) != 0.0 for value in audit.values()):
        raise RuntimeError(f"{path}: dense initialization was not exact: {audit!r}")
    if (path / "control_output.pt").exists():
        raise RuntimeError(f"{path}: dense final method must not emit final B-spline control_output.pt")
    if (summary.get("cudnn_benchmark"), summary.get("cudnn_deterministic")) != (False, True):
        raise RuntimeError(f"{path}: unexpected cuDNN flags")
    for key in ("dice_mean", "hd95_mean", "asd_symmetric_mean"):
        if not math.isfinite(float(summary[key])):
            raise RuntimeError(f"{path}: non-finite {key}")
    if summary.get("asd_backend") != "google_deepmind_surface_distance_area_weighted":
        raise RuntimeError(f"{path}: unexpected ASD backend")
    validate_absent_labels(summary)
    dvfs = list(path.glob("disp_norm_*.npy"))
    if len(dvfs) != 1:
        raise RuntimeError(f"{path}: expected one serialized DVF, got {len(dvfs)}")
    dense = np.load(dvfs[0], mmap_mode="r")
    if tuple(dense.shape) != EXPECTED_DVF_SHAPE or not np.isfinite(dense).all():
        raise RuntimeError(f"{path}: invalid DVF shape/finite status: {dense.shape}")
    return summary, {"run_dir": str(path), "dvf_path": str(dvfs[0]),
                     "dvf_shape": list(dense.shape), "dvf_finite": True,
                     "prefix_stage_tails": {str(stage): read_stage_tail(path, stage) for stage in range(3)}}


def prefix_validation(validated: Sequence[Tuple[Job, Mapping[str, object], Mapping[str, object]]],
                      tolerance: float) -> Dict[str, object]:
    reference = validated[0][2]["prefix_stage_tails"]
    comparisons: List[Dict[str, object]] = []
    passed = True
    for job, _summary, audit in validated[1:]:
        stages = []
        for stage in range(3):
            ref = reference[str(stage)]  # type: ignore[index]
            observed = audit["prefix_stage_tails"][str(stage)]  # type: ignore[index]
            deltas = {key: abs(float(observed[key]) - float(ref[key]))
                      for key in ("loss", "data_loss", "reg_loss")}
            stage_passed = (int(observed["epoch"]) == int(ref["epoch"]) and
                            all(delta <= tolerance for delta in deltas.values()))
            passed = passed and stage_passed
            stages.append({"stage_idx": stage, "epoch_match": int(observed["epoch"]) == int(ref["epoch"]),
                           "deltas": deltas, "passed": stage_passed})
        comparisons.append({"reference_dense_alpha": validated[0][0].dense_alpha,
                            "candidate_dense_alpha": job.dense_alpha, "stages": stages})
    return {"passed": passed, "tolerance": tolerance, "comparisons": comparisons}


def load_bspline_rows(path: Path) -> Dict[float, Mapping[str, object]]:
    payload = c.read_json(path / "comparison.json")
    assert isinstance(payload, dict)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Missing prior B-spline rows in {path / 'comparison.json'}")
    result = {float(row["cps1_alpha"]): row for row in rows
              if isinstance(row, dict) and row.get("kind") == "new_candidate"}
    expected = {10.0, 100.0, 1000.0, 10000.0, 100000.0}
    if set(result) != expected:
        raise RuntimeError(f"Prior B-spline sweep lambdas {sorted(result)} != {sorted(expected)}")
    return result


def fmt(value: object, digits: int = 6) -> str:
    return "" if value is None or value == "" else f"{float(value):.{digits}f}"


def markdown_report(rows: Sequence[Mapping[str, object]], candidates: Sequence[Mapping[str, object]],
                    paired: Sequence[Mapping[str, object]], best: Mapping[str, Mapping[str, object]],
                    overall: Mapping[str, object], prefix: Mapping[str, object]) -> str:
    lines = ["# OASIS rank-0 MR-D-BSCP 8-4-2 → dense-field lambda sweep", "",
             f"Fixed `{EXPECTED_FIXED}`, moving `{EXPECTED_MOVING}` (moving → fixed).", "",
             "## Metrics", "",
             "| Method | Dense λ | Dice ↑ | HD95 ↓ | ASD ↓ | GPU | Stage epochs |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        dense_alpha = "" if row.get("dense_alpha") is None else f"{float(row['dense_alpha']):g}"
        stage_epochs = row.get("stage_actual_epochs")
        stage_epochs_text = f"`{json.dumps(stage_epochs)}`" if stage_epochs else ""
        lines.append(f"| {row['method']} | {dense_alpha} "
                     f"| {fmt(row['dice'])} | {fmt(row['hd95'])} | {fmt(row['asd'])} | "
                     f"{row.get('gpu', '')} | {stage_epochs_text} |")
    lines += ["", "## Prior cps1 B-spline sweep", "",
              "| cps1 λ | Dice ↑ | HD95 ↓ | ASD ↓ |", "|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {float(row['lambda']):g} | {float(row['bspline_dice']):.6f} | "
                     f"{float(row['bspline_hd95']):.6f} | {float(row['bspline_asd']):.6f} |")
    lines += ["", "## Dense minus cps1-B-spline at matched λ", "",
              "| λ | Δ Dice | Δ HD95 | Δ ASD |", "|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {float(row['lambda']):g} | {float(row['delta_dice']):+.6f} | "
                     f"{float(row['delta_hd95']):+.6f} | {float(row['delta_asd']):+.6f} |")
    lines += ["", "## Selection and verification", "",
              f"- Best Dice: dense λ={float(best['dice']['dense_alpha']):g} ({fmt(best['dice']['dice'])}).",
              f"- Best HD95: dense λ={float(best['hd95']['dense_alpha']):g} ({fmt(best['hd95']['hd95'])}).",
              f"- Best ASD: dense λ={float(best['asd']['dense_alpha']):g} ({fmt(best['asd']['asd'])}).",
              f"- Overall (Dice, then ASD, then HD95): dense λ={float(overall['dense_alpha']):g}.",
              f"- Independent 8-4-2 prefix equality at 1e-6: {'PASS' if prefix['passed'] else 'FAIL'}.", "",
              "## Runtime diagnostics", "",
              "| Dense λ | Training seconds | Total seconds | Stop reasons |",
              "|---:|---:|---:|---|"]
    for row in sorted(candidates, key=lambda item: float(item["dense_alpha"])):
        lines.append(f"| {float(row['dense_alpha']):g} | {fmt(row['training_elapsed_sec'], 3)} | "
                     f"{fmt(row['total_elapsed_sec'], 3)} | `{json.dumps(row['stage_stop_reasons'])}` |")
    return "\n".join(lines) + "\n"


def aggregate(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    validated = [(job, *validate_candidate(args, job)) for job in jobs]
    prefix = prefix_validation(validated, float(args.prefix_tolerance))
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_json(args.review_root / "prefix_validation.json", prefix)
    if not prefix["passed"]:
        raise RuntimeError("The deterministic 8-4-2 prefix comparison failed")
    candidates: List[Dict[str, object]] = []
    for job, summary, audit in validated:
        candidates.append({"method": f"MR-D-BSCP 8-4-2 → dense (λ={job.dense_alpha:g})",
                           "kind": "dense_candidate", "dense_alpha": job.dense_alpha,
                           "dice": float(summary["dice_mean"]), "hd95": float(summary["hd95_mean"]),
                           "asd": float(summary["asd_symmetric_mean"]),
                           "actual_epochs": int(summary["actual_epochs"]),
                           "stage_actual_epochs": summary["stage_actual_epochs"],
                           "stage_stop_reasons": summary["stage_stop_reasons"],
                           "training_elapsed_sec": float(summary["training_elapsed_sec"]),
                           "total_elapsed_sec": float(summary["total_elapsed_sec"]),
                           "gpu": summary.get("gpu_requested"), "run_dir": audit["run_dir"],
                           "source": summary.get("source_revision")})
    candidates.sort(key=lambda row: float(row["dense_alpha"]))
    prior = load_bspline_rows(args.bspline_review_root)
    paired = []
    bspline_rows = []
    for row in candidates:
        alpha = float(row["dense_alpha"])
        old = prior[alpha]
        paired.append({"lambda": alpha, "dense_dice": row["dice"], "bspline_dice": old["dice"],
                       "delta_dice": float(row["dice"]) - float(old["dice"]),
                       "dense_hd95": row["hd95"], "bspline_hd95": old["hd95"],
                       "delta_hd95": float(row["hd95"]) - float(old["hd95"]),
                       "dense_asd": row["asd"], "bspline_asd": old["asd"],
                       "delta_asd": float(row["asd"]) - float(old["asd"])})
        bspline_rows.append({"method": f"MR-D-BSCP 8-4-2-1 B-spline (λ={alpha:g})",
                             "kind": "prior_bspline_candidate", "cps1_alpha": alpha,
                             "dice": old["dice"], "hd95": old["hd95"], "asd": old["asd"],
                             "gpu": old.get("gpu"), "stage_actual_epochs": old.get("stage_actual_epochs")})
    best = {"dice": max(candidates, key=lambda row: float(row["dice"])),
            "hd95": min(candidates, key=lambda row: float(row["hd95"])),
            "asd": min(candidates, key=lambda row: float(row["asd"]))}
    overall = sorted(candidates, key=lambda row: (-float(row["dice"]), float(row["asd"]),
                                                  float(row["hd95"])))[0]
    rows = [dict(row) for row in BASELINES] + bspline_rows + candidates
    c.write_csv(args.review_root / "comparison.csv", rows)
    c.write_csv(args.review_root / "paired_dense_minus_bspline.csv", paired)
    payload = {"created_at": c.now_text(), "variant": VARIANT, "case_rank": CASE_RANK,
               "fixed_id": EXPECTED_FIXED, "moving_id": EXPECTED_MOVING,
               "direction": "moving_to_fixed", "rows": rows,
               "paired_dense_minus_bspline": paired, "best_dense_candidate_by_metric": best,
               "overall_selected_dense_candidate": overall, "prefix_validation": prefix,
               "candidate_audits": [audit for _job, _summary, audit in validated]}
    c.write_json(args.review_root / "comparison.json", payload)
    (args.review_root / "report.md").write_text(
        markdown_report([dict(row) for row in BASELINES] + candidates, candidates, paired,
                        best, overall, prefix)
    )
    c.write_json(args.review_root / "ACCEPTED.json", {
        "accepted_at": c.now_text(), "candidate_count": len(candidates), "all_complete": True,
        "all_dvfs_finite": True, "prefix_validation_passed": True,
        "overall_dense_alpha": overall["dense_alpha"],
    })
    print(json.dumps({"report": str(args.review_root / "report.md"), "overall": overall}, indent=2))


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    jobs = build_jobs(args)
    source = source_metadata()
    if not args.aggregate_only:
        write_manifest(args, jobs, gpus, source)
        launch_jobs(args, jobs, gpus, source)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args, jobs)


if __name__ == "__main__":
    main()
