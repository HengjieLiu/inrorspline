#!/usr/bin/env python
"""Launch the OASIS first10 DirectCP cps2 preliminary alpha add-on."""

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


DEFAULT_ORIGINAL_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay"
VARIANT = "oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1"
METHOD = "direct_cp_cps2_be_bspline_analytic"
ALPHAS = [100.0, 1000.0, 10000.0]
LR = 1e-3


@dataclass(frozen=True)
class Job:
    job_index: int
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method: str
    method_key: str
    alpha: float
    lr: float
    gpu: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="1,2,3")
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--alphas", type=str, default="100,1000,10000")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def parse_alphas(text: str) -> List[float]:
    alphas: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            alphas.append(float(token))
    if not alphas:
        raise ValueError(f"No alphas parsed from {text!r}")
    return alphas


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


def build_jobs(args: argparse.Namespace) -> List[Job]:
    gpus = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    alphas = parse_alphas(str(args.alphas))
    if len(gpus) != len(alphas):
        raise RuntimeError(f"Expected one GPU per alpha, got gpus={gpus}, alphas={alphas}")
    jobs: List[Job] = []
    for rank in parse_case_ranks(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for alpha, gpu in zip(alphas, gpus):
            jobs.append(
                Job(
                    job_index=len(jobs),
                    case_rank=int(case.rank),
                    original_case_index=int(case.original_case_index),
                    fixed_id=str(case.fixed_id),
                    moving_id=str(case.moving_id),
                    method=METHOD,
                    method_key=METHOD,
                    alpha=float(alpha),
                    lr=float(args.lr),
                    gpu=str(gpu),
                )
            )
    return jobs


def run_name(args: argparse.Namespace, job: Job) -> str:
    return f"{job.method_key}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}_e{args.epochs}_lr{c.format_float(job.lr)}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / run_name(args, job)


def build_cmd(args: argparse.Namespace, job: Job) -> List[str]:
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
        job.gpu,
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


def log_name(args: argparse.Namespace, job: Job) -> str:
    return (
        f"rank{job.case_rank:03d}_{job.method}_a{c.format_float(job.alpha)}"
        f"_lr{c.format_float(job.lr)}_gpu{job.gpu}.log"
    )


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
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
            "out_root": str(args.out_root),
            "original_root": str(args.original_root),
            "review_root": str(args.review_root),
            "oasis_root": str(args.oasis_root),
            "pair_file": str(args.pair_file),
            "method": METHOD,
            "alphas": parse_alphas(str(args.alphas)),
            "lr": float(args.lr),
            "variant": str(args.variant),
            "gpu_alpha_map": {str(job.alpha): job.gpu for job in jobs[: len(parse_alphas(str(args.alphas)))]},
            "jobs": rows,
        },
    )


def launch_wave(args: argparse.Namespace, wave: Sequence[Job], env: Dict[str, str]) -> List[Dict[str, object]]:
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for job in wave:
        done_path = run_dir(args, job) / "DONE"
        if done_path.exists() and not args.overwrite:
            print(json.dumps({"skip_existing": str(done_path.parent), "job": asdict(job)}), flush=True)
            continue
        cmd = build_cmd(args, job)
        log_path = log_dir / log_name(args, job)
        print(json.dumps({"launch": cmd, "log": str(log_path)}), flush=True)
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        running.append({"job": job, "proc": proc, "log_file": log_f, "log_path": log_path, "cmd": cmd})
        time.sleep(float(args.launch_stagger_sec))

    while running:
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
                        "returncode": code,
                        "log_path": str(item["log_path"]),
                        "cmd": item["cmd"],
                    }
                )
            else:
                print(json.dumps({"completed": asdict(job)}), flush=True)
        running = still_running
        if running:
            time.sleep(5.0)
    return failures


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"launch": build_cmd(args, job), "log": str(Path(args.out_root) / "logs" / log_name(args, job))}), flush=True)
        return

    started = time.time()
    failures: List[Dict[str, object]] = []
    by_case: Dict[int, List[Job]] = {}
    for job in jobs:
        by_case.setdefault(int(job.case_rank), []).append(job)
    env = thread_env(args.cpu_threads)
    for case_rank in sorted(by_case):
        wave = sorted(by_case[case_rank], key=lambda job: parse_alphas(str(args.alphas)).index(float(job.alpha)))
        print(json.dumps({"case_rank_wave": case_rank, "jobs": [asdict(job) for job in wave]}), flush=True)
        failures.extend(launch_wave(args, wave, env))
        if failures:
            break

    c.write_json(
        Path(args.out_root) / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "job_count": len(jobs),
            "failures": failures,
            "elapsed_sec": time.time() - started,
        },
    )
    if failures:
        raise RuntimeError("One or more DirectCP cps2 preliminary jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_direct_cp_cps2_overlay.py"),
        "--original-root",
        str(args.original_root),
        "--addon-root",
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
                "job_count": len(jobs),
                "case_ranks": parse_case_ranks(str(args.case_ranks)),
                "method": METHOD,
                "alphas": parse_alphas(str(args.alphas)),
                "lr": float(args.lr),
                "out_root": str(args.out_root),
                "original_root": str(args.original_root),
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
    launch_jobs(args, jobs)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
