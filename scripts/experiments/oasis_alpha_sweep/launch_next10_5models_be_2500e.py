#!/usr/bin/env python
"""Launch the OASIS seed-3 next10 five-model BE alpha sweep."""

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
from launch_first10_be_2500e import parse_case_ranks  # noqa: E402


DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_seed3_next10_alpha_sweep"
VARIANT = "oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1"
IDIR_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SPLINE_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    method_key: str
    lr: float
    alphas: Sequence[float]


@dataclass(frozen=True)
class Job:
    job_index: int
    phase_index: int
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method: str
    method_key: str
    alpha: float
    lr: float


METHOD_SPECS = [
    MethodSpec("sinr_cps2_be_bspline_analytic", "sinr_cps2_be_bspline_analytic", 1e-4, SPLINE_ALPHAS),
    MethodSpec("sinr_cps4_be_bspline_analytic", "sinr_cps4_be_bspline_analytic", 1e-4, SPLINE_ALPHAS),
    MethodSpec("idir_original", "idir_original_dense_siren_be_autograd", 1e-4, IDIR_ALPHAS),
    MethodSpec("direct_cp_cps2_be_bspline_analytic", "direct_cp_cps2_be_bspline_analytic", 1e-3, SPLINE_ALPHAS),
    MethodSpec("direct_cp_cps4_be_bspline_analytic", "direct_cp_cps4_be_bspline_analytic", 1e-3, SPLINE_ALPHAS),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--case-ranks", type=str, default="10-19")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
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


def build_jobs_by_method(args: argparse.Namespace) -> Dict[str, List[Job]]:
    cases = [
        c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in parse_case_ranks(str(args.case_ranks))
    ]
    by_method: Dict[str, List[Job]] = {}
    job_index = 0
    for phase_index, spec in enumerate(METHOD_SPECS):
        jobs: List[Job] = []
        for alpha in spec.alphas:
            for case in cases:
                jobs.append(
                    Job(
                        job_index=job_index,
                        phase_index=phase_index,
                        case_rank=int(case.rank),
                        original_case_index=int(case.original_case_index),
                        fixed_id=str(case.fixed_id),
                        moving_id=str(case.moving_id),
                        method=spec.method,
                        method_key=spec.method_key,
                        alpha=float(alpha),
                        lr=float(spec.lr),
                    )
                )
                job_index += 1
        by_method[spec.method_key] = jobs
    return by_method


def run_name(args: argparse.Namespace, job: Job) -> str:
    return (
        f"{job.method_key}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}"
        f"_e{args.epochs}_lr{c.format_float(job.lr)}"
    )


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / run_name(args, job)


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
        str(job.case_rank),
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--variant",
        str(args.variant),
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
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(args: argparse.Namespace, job: Job, gpu: str) -> str:
    return (
        f"phase{job.phase_index:02d}_rank{job.case_rank:03d}_{job.method}"
        f"_a{c.format_float(job.alpha)}_lr{c.format_float(job.lr)}_gpu{gpu}.log"
    )


def write_manifest(args: argparse.Namespace, by_method: Dict[str, List[Job]]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs = [job for method_key in [spec.method_key for spec in METHOD_SPECS] for job in by_method[method_key]]
    rows = [asdict(job) for job in jobs]
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_json(
        args.out_root / "job_manifest.json",
        {
            "created_at": c.now_text(),
            "job_count": len(jobs),
            "case_ranks": parse_case_ranks(str(args.case_ranks)),
            "epochs": int(args.epochs),
            "eval_interval": int(args.eval_interval),
            "seed": int(args.seed),
            "variant": str(args.variant),
            "out_root": str(args.out_root),
            "review_root": str(args.review_root),
            "oasis_root": str(args.oasis_root),
            "pair_file": str(args.pair_file),
            "method_specs": [asdict(spec) for spec in METHOD_SPECS],
            "phase_order": [spec.method_key for spec in METHOD_SPECS],
            "jobs": rows,
        },
    )


def launch_phase(args: argparse.Namespace, method_key: str, jobs: Sequence[Job], env: Dict[str, str]) -> List[Dict[str, object]]:
    gpus = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if len(gpus) < 4:
        raise RuntimeError(f"Expected 4 GPUs for the next10 sweep, got {gpus}")
    max_concurrent = min(int(args.max_concurrent), len(gpus))
    pending = list(jobs)
    running: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.time()
    print(json.dumps({"phase_start": method_key, "job_count": len(jobs), "started_at": c.now_text()}), flush=True)

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % max_concurrent]
            print(json.dumps({"launch": build_cmd(args, job, gpu), "log": str(log_dir / log_name(args, job, gpu))}), flush=True)
        return failures

    while pending or running:
        while pending and len(running) < max_concurrent:
            busy_gpus = {str(item["gpu"]) for item in running}
            free_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]
            if not free_gpus:
                break
            job = pending.pop(0)
            gpu = free_gpus[0]
            done_path = run_dir(args, job) / "DONE"
            if done_path.exists() and not args.overwrite:
                print(json.dumps({"skip_existing": str(done_path.parent), "job": asdict(job)}), flush=True)
                continue
            cmd = build_cmd(args, job, gpu)
            log_path = log_dir / log_name(args, job, gpu)
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
            item["log_file"].close()
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
        if failures:
            break
        if running or pending:
            time.sleep(5.0)

    print(
        json.dumps(
            {
                "phase_end": method_key,
                "completed_at": c.now_text(),
                "elapsed_sec": time.time() - phase_started,
                "failures": failures,
            },
            default=str,
        ),
        flush=True,
    )
    return failures


def launch_jobs(args: argparse.Namespace, by_method: Dict[str, List[Job]]) -> None:
    env = thread_env(args.cpu_threads)
    started = time.time()
    failures: List[Dict[str, object]] = []
    for spec in METHOD_SPECS:
        failures.extend(launch_phase(args, spec.method_key, by_method[spec.method_key], env))
        if failures:
            break
    c.write_json(
        args.out_root / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "job_count": sum(len(jobs) for jobs in by_method.values()),
            "phase_order": [spec.method_key for spec in METHOD_SPECS],
            "failures": failures,
            "elapsed_sec": time.time() - started,
        },
    )
    if failures:
        raise RuntimeError("One or more next10 sweep jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_first10_population.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--case-ranks",
        str(args.case_ranks),
        "--expected-epochs",
        str(args.epochs),
        "--expected-eval-interval",
        str(args.eval_interval),
        "--expected-seed",
        str(args.seed),
        "--expected-variant",
        str(args.variant),
        "--include-direct-cp-cps2",
    ]
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    by_method = build_jobs_by_method(args)
    write_manifest(args, by_method)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "launch",
                "job_count": sum(len(jobs) for jobs in by_method.values()),
                "case_ranks": parse_case_ranks(str(args.case_ranks)),
                "method_specs": [asdict(spec) for spec in METHOD_SPECS],
                "phase_order": [spec.method_key for spec in METHOD_SPECS],
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "variant": str(args.variant),
                "gpus": args.gpus,
                "epochs": args.epochs,
                "eval_interval": args.eval_interval,
                "seed": args.seed,
            },
            indent=2,
        ),
        flush=True,
    )
    launch_jobs(args, by_method)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
