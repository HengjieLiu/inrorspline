#!/usr/bin/env python
"""Launch the OASIS rank-9 image-registration vs Greedy-field fit LR sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


VARIANT = "oasis_rank009_fit_modes_lr_sweep_v1"
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep"
CASE_RANK = 9
EXPECTED_ORIGINAL_INDEX = 88
EXPECTED_FIXED_ID = "0173"
EXPECTED_MOVING_ID = "0136"
LRS = [1e-2, 1e-3, 1e-4]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    method_key: str
    field_method_key: str
    label: str
    alpha: float


@dataclass(frozen=True)
class Job:
    job_index: int
    objective: str
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method: str
    method_key: str
    alpha: float
    lr: float


METHOD_SPECS = [
    MethodSpec("idir_original", "idir_original_dense_siren_be_autograd", "idir_original_dense_siren_field_mse", "IDIR", 1.0),
    MethodSpec("sinr_cps4_be_bspline_analytic", "sinr_cps4_be_bspline_analytic", "sinr_cps4_field_mse", "SINR cps4", 1000.0),
    MethodSpec("direct_cp_cps4_be_bspline_analytic", "direct_cp_cps4_be_bspline_analytic", "direct_cp_cps4_field_mse", "DirectCP cps4", 1000.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--field-chunk-size", type=int, default=200000)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


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


def expected_case(args: argparse.Namespace) -> c.CaseInfo:
    case = c.resolve_case(rank=CASE_RANK, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    if int(case.original_case_index) != EXPECTED_ORIGINAL_INDEX:
        raise RuntimeError(f"Rank {CASE_RANK} original index {case.original_case_index} != {EXPECTED_ORIGINAL_INDEX}")
    if case.fixed_id != EXPECTED_FIXED_ID or case.moving_id != EXPECTED_MOVING_ID:
        raise RuntimeError(f"Rank {CASE_RANK} pair {case.fixed_id}/{case.moving_id} != {EXPECTED_FIXED_ID}/{EXPECTED_MOVING_ID}")
    return case


def build_jobs(args: argparse.Namespace) -> List[Job]:
    case = expected_case(args)
    jobs: List[Job] = []
    for objective in ["image_registration", "greedy_field_fit"]:
        for spec in METHOD_SPECS:
            method_key = spec.method_key if objective == "image_registration" else spec.field_method_key
            for lr in LRS:
                jobs.append(
                    Job(
                        job_index=len(jobs),
                        objective=objective,
                        case_rank=int(case.rank),
                        original_case_index=int(case.original_case_index),
                        fixed_id=str(case.fixed_id),
                        moving_id=str(case.moving_id),
                        method=spec.method,
                        method_key=method_key,
                        alpha=float(spec.alpha),
                        lr=float(lr),
                    )
                )
    return jobs


def run_name_for_job(args: argparse.Namespace, job: Job) -> str:
    run_name_prefix = "image_reg" if job.objective == "image_registration" else "greedy_field_fit"
    return (
        f"{run_name_prefix}_{job.method_key}_rank{job.case_rank:03d}"
        f"_a{c.format_float(job.alpha)}_e{args.epochs}_lr{c.format_float(job.lr)}"
    )


def run_dir_for_job(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / run_name_for_job(args, job)


def build_cmd(args: argparse.Namespace, job: Job, gpu: str) -> List[str]:
    script = "run_one.py" if job.objective == "image_registration" else "run_rank009_greedy_field_fit_one.py"
    run_name = run_name_for_job(args, job)
    cmd = [
        sys.executable,
        str(THIS_DIR / script),
        "--method",
        job.method,
        "--alpha",
        f"{job.alpha:g}",
        "--lr",
        f"{job.lr:g}",
        "--gpu",
        gpu,
        "--case-rank",
        str(job.case_rank),
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--variant",
        VARIANT,
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--expected-original-case-index",
        str(job.original_case_index),
        "--expected-fixed-id",
        str(job.fixed_id),
        "--expected-moving-id",
        str(job.moving_id),
        "--run-name",
        run_name,
    ]
    if job.objective == "greedy_field_fit":
        cmd.extend(
            [
                "--coords-batch-size",
                str(args.coords_batch_size),
                "--field-chunk-size",
                str(args.field_chunk_size),
                "--eval-chunk-size",
                str(args.eval_chunk_size),
            ]
        )
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(job: Job, gpu: str) -> str:
    return (
        f"{job.objective}_rank{job.case_rank:03d}_{job.method_key}"
        f"_a{c.format_float(job.alpha)}_lr{c.format_float(job.lr)}_gpu{gpu}.log"
    )


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(job) for job in jobs]
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_json(
        args.out_root / "job_manifest.json",
        {
            "created_at": c.now_text(),
            "variant": VARIANT,
            "job_count": len(jobs),
            "case_rank": CASE_RANK,
            "original_case_index": EXPECTED_ORIGINAL_INDEX,
            "fixed_id": EXPECTED_FIXED_ID,
            "moving_id": EXPECTED_MOVING_ID,
            "epochs": int(args.epochs),
            "eval_interval": int(args.eval_interval),
            "seed": int(args.seed),
            "learning_rates": LRS,
            "field_fit_domain": "fixed_image_positive",
            "field_fit_mask_definition": "fixed raw image > 0 after standard OASIS reorientation",
            "idir_field_fit_coords_batch_size": int(args.coords_batch_size),
            "out_root": str(args.out_root),
            "review_root": str(args.review_root),
            "oasis_root": str(args.oasis_root),
            "pair_file": str(args.pair_file),
            "method_specs": [asdict(spec) for spec in METHOD_SPECS],
            "jobs": rows,
        },
    )


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    gpus = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if len(gpus) < 1:
        raise RuntimeError("At least one GPU is required.")
    max_concurrent = min(int(args.max_concurrent), len(gpus))
    if max_concurrent < 1:
        raise RuntimeError(f"Invalid max_concurrent={args.max_concurrent}")
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = thread_env(args.cpu_threads)
    pending = list(jobs)
    running: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    started = time.time()

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % max_concurrent]
            cmd = build_cmd(args, job, gpu)
            log_path = log_dir / log_name(job, gpu)
            print(json.dumps({"launch": cmd, "log": str(log_path)}), flush=True)
        return

    while pending or running:
        while pending and len(running) < max_concurrent:
            busy_gpus = {str(item["gpu"]) for item in running}
            free_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]
            if not free_gpus:
                break
            job = pending.pop(0)
            done_path = run_dir_for_job(args, job) / "DONE"
            if done_path.exists() and not args.overwrite:
                print(json.dumps({"skip_existing_done": asdict(job), "done_path": str(done_path)}), flush=True)
                continue
            gpu = free_gpus[0]
            cmd = build_cmd(args, job, gpu)
            log_path = log_dir / log_name(job, gpu)
            print(json.dumps({"launch": cmd, "log": str(log_path)}), flush=True)
            log_f = log_path.open("w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "gpu": gpu, "proc": proc, "log_file": log_f, "log_path": log_path, "cmd": cmd})
            time.sleep(float(args.launch_stagger_sec))

        still_running: List[Dict[str, object]] = []
        for item in running:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still_running.append(item)
                continue
            log_f = item["log_file"]
            log_f.close()
            job = item["job"]
            assert isinstance(job, Job)
            if code != 0:
                failures.append(
                    {
                        "job": asdict(job),
                        "gpu": item["gpu"],
                        "returncode": code,
                        "log_path": str(item["log_path"]),
                        "cmd": item["cmd"],
                    }
                )
            else:
                print(json.dumps({"completed": asdict(job), "gpu": item["gpu"]}), flush=True)
        running = still_running
        if running or pending:
            time.sleep(5.0)

    c.write_json(
        args.out_root / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "job_count": len(jobs),
            "failures": failures,
            "elapsed_sec": time.time() - started,
        },
    )
    if failures:
        raise RuntimeError("One or more rank-9 fit-mode jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_rank009_fit_modes_lr_sweep.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--expected-epochs",
        str(args.epochs),
        "--expected-eval-interval",
        str(args.eval_interval),
        "--expected-seed",
        str(args.seed),
    ]
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "launch",
                "variant": VARIANT,
                "job_count": len(jobs),
                "gpus": str(args.gpus),
                "max_concurrent": int(args.max_concurrent),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    launch_jobs(args, jobs)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
