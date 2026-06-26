#!/usr/bin/env python
"""Launch selected OASIS DirectCP 8-4-2 alpha=1000 preliminary ranks."""

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


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
VARIANT = "oasis_seed3_first10_directcp842_alpha1000_prelim_v1"
DEFAULT_OUT_ROOT = Path(f"/database/inrorspline/runs/{VARIANT}")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim"
DEFAULT_REFERENCE_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay"
ALLOWED_GPUS = {"0", "1", "2", "3"}
ALPHA = 1000.0
LR = 1e-2
STAGE_FULL_CPS = "8,4,2"
STAGE_DOWNSAMPLE = "4,2,1"
STAGE_EPOCHS = "400,800,1300"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stage-full-cps", type=str, default=STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=STAGE_DOWNSAMPLE)
    parser.add_argument("--stage-epochs", type=str, default=STAGE_EPOCHS)
    parser.add_argument("--epochs", type=int, default=sum(c.parse_int_list(STAGE_EPOCHS)))
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=2)
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


def build_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []
    for rank in parse_case_ranks(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        jobs.append(
            Job(
                job_index=len(jobs),
                case_rank=int(case.rank),
                original_case_index=int(case.original_case_index),
                fixed_id=str(case.fixed_id),
                moving_id=str(case.moving_id),
                method=METHOD,
                method_key=METHOD,
                alpha=float(args.alpha),
                lr=float(args.lr),
            )
        )
    return jobs


def build_run_name(args: argparse.Namespace, job: Job) -> str:
    return (
        f"{job.method_key}_rank{job.case_rank:03d}_sched{args.stage_full_cps.replace(',', '-')}"
        f"_ds{args.stage_downsample.replace(',', '-')}_e{args.stage_epochs.replace(',', '-')}"
        f"_a{c.format_float(job.alpha)}_lr{c.format_float(job.lr)}"
    )


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / build_run_name(args, job)


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
        str(gpu),
        "--case-rank",
        str(job.case_rank),
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
        "--variant",
        VARIANT,
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--run-name",
        build_run_name(args, job),
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
        f"rank{job.case_rank:03d}_{job.method}_a{c.format_float(job.alpha)}"
        f"_lr{c.format_float(job.lr)}_gpu{gpu}.log"
    )


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(job) for job in jobs]
    ranks = parse_case_ranks(str(args.case_ranks))
    manifest_stem = f"job_manifest_ranks{min(ranks):03d}-{max(ranks):03d}"
    c.write_csv(args.out_root / f"{manifest_stem}.csv", rows)
    c.write_json(
        args.out_root / f"{manifest_stem}.json",
        {
            "created_at": c.now_text(),
            "job_count": len(jobs),
            "case_ranks": ranks,
            "method": METHOD,
            "alpha": float(args.alpha),
            "lr": float(args.lr),
            "stage_full_cps": str(args.stage_full_cps),
            "stage_downsample": str(args.stage_downsample),
            "stage_epochs": str(args.stage_epochs),
            "epochs": int(args.epochs),
            "eval_interval": int(args.eval_interval),
            "seed": int(args.seed),
            "out_root": str(args.out_root),
            "review_root": str(args.review_root),
            "reference_root": str(args.reference_root),
            "oasis_root": str(args.oasis_root),
            "pair_file": str(args.pair_file),
            "jobs": rows,
        },
    )
    c.write_json(
        args.out_root / "job_manifest_latest.json",
        {
            "created_at": c.now_text(),
            "manifest": manifest_stem,
            "job_count": len(jobs),
            "case_ranks": ranks,
            "gpus": str(args.gpus),
            "max_concurrent": int(args.max_concurrent),
        },
    )


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    gpus = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if not gpus:
        raise RuntimeError("At least one GPU is required.")
    unknown_gpus = set(gpus) - ALLOWED_GPUS
    if unknown_gpus:
        raise RuntimeError(f"This preliminary launcher only knows local GPUs {sorted(ALLOWED_GPUS)}; got {gpus}")
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
            print(json.dumps({"launch": cmd, "log": str(log_dir / log_name(args, job, gpu))}), flush=True)
        return

    while pending or running:
        while pending and len(running) < max_concurrent:
            busy_gpus = {str(item["gpu"]) for item in running}
            free_gpus = [gpu for gpu in gpus[:max_concurrent] if gpu not in busy_gpus]
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
        raise RuntimeError("One or more DirectCP 8-4-2 preliminary jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_directcp842_alpha1000_prelim.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--reference-root",
        str(args.reference_root),
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
                "alpha": float(args.alpha),
                "lr": float(args.lr),
                "stage_full_cps": str(args.stage_full_cps),
                "stage_downsample": str(args.stage_downsample),
                "stage_epochs": str(args.stage_epochs),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "gpus": args.gpus,
                "max_concurrent": int(args.max_concurrent),
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
