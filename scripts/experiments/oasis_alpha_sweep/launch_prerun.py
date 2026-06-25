#!/usr/bin/env python
"""Launch the 12-run OASIS seed-3 case-0 pre-run inside Docker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


@dataclass(frozen=True)
class Job:
    method: str
    alpha: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=c.DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-original-case-index", type=int, default=158)
    parser.add_argument("--expected-fixed-id", type=str, default="0333")
    parser.add_argument("--expected-moving-id", type=str, default="0115")
    parser.add_argument("--case-label", type=str, default="OASIS seed-3 rank 0")
    parser.add_argument("--figure-title", type=str, default="OASIS seed-3 rank 0 pre-run")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def jobs() -> List[Job]:
    return [
        *(Job("idir_original", alpha, 1e-4) for alpha in [0.01, 0.1, 1.0]),
        *(Job("sinr_cps4_be_bspline_analytic", alpha, 1e-4) for alpha in [100.0, 1000.0, 10000.0]),
        *(Job("sinr_cps2_be_bspline_analytic", alpha, 1e-4) for alpha in [100.0, 1000.0, 10000.0]),
        *(Job("direct_cp_cps4_be_bspline_analytic", alpha, 1e-3) for alpha in [100.0, 1000.0, 10000.0]),
    ]


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


def build_cmd(args: argparse.Namespace, job: Job, gpu: str) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method",
        job.method,
        "--alpha",
        f"{job.alpha:g}",
        "--lr",
        f"{job.lr:g}",
        "--gpu",
        gpu,
        "--case-rank",
        str(args.case_rank),
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--expected-original-case-index",
        str(args.expected_original_case_index),
        "--expected-fixed-id",
        str(args.expected_fixed_id),
        "--expected-moving-id",
        str(args.expected_moving_id),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(job: Job, gpu: str) -> str:
    return f"{job.method}_a{c.format_float(job.alpha)}_lr{c.format_float(job.lr)}_gpu{gpu}.log"


def run_validate_case(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method",
        "idir_original",
        "--alpha",
        "0.01",
        "--case-rank",
        str(args.case_rank),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--expected-original-case-index",
        str(args.expected_original_case_index),
        "--expected-fixed-id",
        str(args.expected_fixed_id),
        "--expected-moving-id",
        str(args.expected_moving_id),
        "--validate-case-only",
    ]
    print(json.dumps({"validate_case_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def launch_jobs(args: argparse.Namespace) -> None:
    gpus = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if len(gpus) < 4:
        raise RuntimeError(f"Expected 4 GPUs for this pre-run, got {gpus}")
    max_concurrent = min(int(args.max_concurrent), len(gpus))
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = thread_env(args.cpu_threads)
    pending = list(jobs())
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
            if code != 0:
                failures.append(
                    {
                        "job": item["job"].__dict__,
                        "gpu": item["gpu"],
                        "returncode": code,
                        "log_path": str(item["log_path"]),
                        "cmd": item["cmd"],
                    }
                )
            else:
                print(json.dumps({"completed": item["job"].__dict__, "gpu": item["gpu"]}), flush=True)
        running = still_running
        if running or pending:
            time.sleep(5.0)

    c.write_json(args.out_root / "launch_summary.json", {"failures": failures, "elapsed_sec": time.time() - started})
    if failures:
        raise RuntimeError("One or more pre-run jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_prerun.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--expected-original-case-index",
        str(args.expected_original_case_index),
        "--expected-fixed-id",
        str(args.expected_fixed_id),
        "--expected-moving-id",
        str(args.expected_moving_id),
        "--case-label",
        str(args.case_label),
        "--figure-title",
        str(args.figure_title),
    ]
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "jobs": [job.__dict__ for job in jobs()],
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "gpus": args.gpus,
                "epochs": args.epochs,
                "eval_interval": args.eval_interval,
                "case_label": args.case_label,
                "expected_original_case_index": args.expected_original_case_index,
                "expected_fixed_id": args.expected_fixed_id,
                "expected_moving_id": args.expected_moving_id,
            },
            indent=2,
        ),
        flush=True,
    )
    run_validate_case(args)
    launch_jobs(args)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
