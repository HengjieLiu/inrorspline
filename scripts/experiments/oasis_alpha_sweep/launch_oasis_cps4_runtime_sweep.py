#!/usr/bin/env python
"""Launch OASIS cps4 runtime candidates for D-BSCP and SINR."""

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


VARIANT = "oasis_seed3_first10_cps4_single_stage_runtime_sweep_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep"
ALPHA = 1000.0
DIRECT_CP_CPS4 = "direct_cp_cps4_be_bspline_analytic"
SINR_CPS4 = "sinr_cps4_be_bspline_analytic"


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    method_label: str
    family: str
    lr: float
    epochs: int
    early_stop_mode: str = "none"
    min_epochs: int = 0
    patience: int = 0
    min_delta: float = 0.0
    description: str = ""
    is_baseline: bool = False


@dataclass(frozen=True)
class Job:
    job_index: int
    candidate: str
    method: str
    method_label: str
    family: str
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    alpha: float
    lr: float
    epochs: int
    early_stop_mode: str
    min_epochs: int
    patience: int
    min_delta: float
    is_baseline: bool


CANDIDATES: Sequence[Candidate] = [
    Candidate(
        "D_baseline_lr1e-3_e2500",
        DIRECT_CP_CPS4,
        "D-BSCP cps4",
        "D-BSCP",
        1e-3,
        2500,
        description="D-BSCP cps4 baseline; lr=1e-3, 2500 epochs",
        is_baseline=True,
    ),
    Candidate(
        "S_baseline_lr1e-4_e2500",
        SINR_CPS4,
        "SINR cps4",
        "SINR",
        1e-4,
        2500,
        description="SINR cps4 baseline; lr=1e-4, 2500 epochs",
        is_baseline=True,
    ),
    Candidate("D_F1000_lr3e-3", DIRECT_CP_CPS4, "D-BSCP cps4", "D-BSCP", 3e-3, 1000, description="fixed 1000 epochs"),
    Candidate("D_F600_lr1e-2", DIRECT_CP_CPS4, "D-BSCP cps4", "D-BSCP", 1e-2, 600, description="fixed 600 epochs"),
    Candidate("D_F400_lr3e-2", DIRECT_CP_CPS4, "D-BSCP cps4", "D-BSCP", 3e-2, 400, description="fixed 400 epochs"),
    Candidate(
        "D_A_lr1e-2",
        DIRECT_CP_CPS4,
        "D-BSCP cps4",
        "D-BSCP",
        1e-2,
        2500,
        early_stop_mode="adaptive",
        min_epochs=300,
        patience=150,
        min_delta=1e-4,
        description="adaptive total-loss plateau; min=300, patience=150, min_delta=1e-4",
    ),
    Candidate(
        "D_A_lr3e-2",
        DIRECT_CP_CPS4,
        "D-BSCP cps4",
        "D-BSCP",
        3e-2,
        2500,
        early_stop_mode="adaptive",
        min_epochs=200,
        patience=100,
        min_delta=1e-4,
        description="adaptive total-loss plateau; min=200, patience=100, min_delta=1e-4",
    ),
    Candidate("S_F1500_lr3e-4", SINR_CPS4, "SINR cps4", "SINR", 3e-4, 1500, description="fixed 1500 epochs"),
    Candidate("S_F1000_lr1e-3", SINR_CPS4, "SINR cps4", "SINR", 1e-3, 1000, description="fixed 1000 epochs"),
    Candidate("S_F600_lr3e-3", SINR_CPS4, "SINR cps4", "SINR", 3e-3, 600, description="fixed 600 epochs"),
    Candidate(
        "S_A_lr1e-3",
        SINR_CPS4,
        "SINR cps4",
        "SINR",
        1e-3,
        2500,
        early_stop_mode="adaptive",
        min_epochs=500,
        patience=200,
        min_delta=1e-4,
        description="adaptive total-loss plateau; min=500, patience=200, min_delta=1e-4",
    ),
    Candidate(
        "S_A_lr3e-3",
        SINR_CPS4,
        "SINR cps4",
        "SINR",
        3e-3,
        2500,
        early_stop_mode="adaptive",
        min_epochs=350,
        patience=150,
        min_delta=1e-4,
        description="adaptive total-loss plateau; min=350, patience=150, min_delta=1e-4",
    ),
]


def candidate_map() -> Dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def parse_candidate_names(text: str) -> List[str]:
    requested = str(text).strip()
    if requested in {"", "all"}:
        return [candidate.name for candidate in CANDIDATES]
    if requested == "screen":
        return [candidate.name for candidate in CANDIDATES if not candidate.is_baseline]
    names = [token.strip() for token in requested.split(",") if token.strip()]
    known = candidate_map()
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"Unknown candidates {missing}; known={sorted(known)}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--epoch-override", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--allow-incomplete-aggregate", action="store_true")
    return parser.parse_args()


def resolve_gpus(text: str) -> List[str]:
    requested = str(text).strip().lower()
    if requested not in {"auto", "all"}:
        return [token.strip() for token in str(text).split(",") if token.strip()]
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not detect GPUs with nvidia-smi; pass --gpus explicitly.") from exc
    gpus = [line.strip() for line in raw.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("nvidia-smi did not report any visible GPUs.")
    return gpus


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


def effective_epochs(args: argparse.Namespace, candidate: Candidate) -> int:
    if int(args.epoch_override) > 0:
        return int(args.epoch_override)
    return int(candidate.epochs)


def effective_min_epochs(args: argparse.Namespace, candidate: Candidate) -> int:
    if str(candidate.early_stop_mode) != "adaptive":
        return 0
    if int(args.epoch_override) > 0:
        return min(int(candidate.min_epochs), int(args.epoch_override))
    return int(candidate.min_epochs)


def build_jobs(args: argparse.Namespace) -> List[Job]:
    selected = [candidate_map()[name] for name in parse_candidate_names(str(args.candidates))]
    jobs: List[Job] = []
    for rank in parse_case_ranks(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for candidate in selected:
            jobs.append(
                Job(
                    job_index=len(jobs),
                    candidate=str(candidate.name),
                    method=str(candidate.method),
                    method_label=str(candidate.method_label),
                    family=str(candidate.family),
                    case_rank=int(case.rank),
                    original_case_index=int(case.original_case_index),
                    fixed_id=str(case.fixed_id),
                    moving_id=str(case.moving_id),
                    alpha=float(args.alpha),
                    lr=float(candidate.lr),
                    epochs=int(effective_epochs(args, candidate)),
                    early_stop_mode=str(candidate.early_stop_mode),
                    min_epochs=int(effective_min_epochs(args, candidate)),
                    patience=int(candidate.patience),
                    min_delta=float(candidate.min_delta),
                    is_baseline=bool(candidate.is_baseline),
                )
            )
    return jobs


def build_run_name(job: Job) -> str:
    return f"{job.method}_runtime_{job.candidate}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method / build_run_name(job)


def build_cmd(args: argparse.Namespace, job: Job, gpu: str) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method",
        str(job.method),
        "--alpha",
        f"{float(job.alpha):g}",
        "--lr",
        f"{float(job.lr):g}",
        "--gpu",
        str(gpu),
        "--case-rank",
        str(job.case_rank),
        "--epochs",
        str(job.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--stage-early-stop-mode",
        str(job.early_stop_mode),
        "--runtime-candidate",
        str(job.candidate),
        "--variant",
        str(args.variant),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--run-name",
        build_run_name(job),
        "--expected-original-case-index",
        str(job.original_case_index),
        "--expected-fixed-id",
        str(job.fixed_id),
        "--expected-moving-id",
        str(job.moving_id),
    ]
    if str(job.early_stop_mode) == "adaptive":
        cmd.extend(
            [
                "--stage-min-epochs",
                str(job.min_epochs),
                "--stage-patience",
                str(job.patience),
                "--stage-check-interval",
                "1",
                "--stage-min-delta",
                f"{float(job.min_delta):g}",
            ]
        )
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(job: Job, gpu: str) -> str:
    return f"{job.candidate}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}_gpu{gpu}.log"


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str], max_concurrent: int) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(job) for job in jobs]
    selected_names = parse_candidate_names(str(args.candidates))
    selected_candidates = [asdict(candidate_map()[name]) for name in selected_names]
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_csv(args.review_root / "job_manifest.csv", rows)
    c.write_csv(args.review_root / "candidate_manifest.csv", selected_candidates)
    manifest = {
        "created_at": c.now_text(),
        "job_count": len(jobs),
        "case_ranks": parse_case_ranks(str(args.case_ranks)),
        "candidates": selected_names,
        "candidate_definitions": selected_candidates,
        "alpha": float(args.alpha),
        "variant": str(args.variant),
        "eval_interval": int(args.eval_interval),
        "seed": int(args.seed),
        "epoch_override": int(args.epoch_override),
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "oasis_root": str(args.oasis_root),
        "pair_file": str(args.pair_file),
        "gpus": list(gpus),
        "max_concurrent": int(max_concurrent),
        "jobs": rows,
    }
    c.write_json(args.out_root / "job_manifest.json", manifest)
    c.write_json(args.review_root / "job_manifest.json", manifest)


def count_done(args: argparse.Namespace, jobs: Sequence[Job] | None = None) -> int:
    if jobs is None:
        return len(list(Path(args.out_root).glob("*/*/DONE")))
    return sum(1 for job in jobs if (run_dir(args, job) / "DONE").exists())


def write_state(
    args: argparse.Namespace,
    *,
    started_epoch: float,
    jobs: Sequence[Job],
    pending: Sequence[Job],
    running: Sequence[Dict[str, object]],
    completed: Sequence[Dict[str, object]],
    skipped: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    gpus: Sequence[str],
    max_concurrent: int,
) -> None:
    state = {
        "updated_at": c.now_text(),
        "updated_epoch": time.time(),
        "started_epoch": float(started_epoch),
        "elapsed_sec": time.time() - float(started_epoch),
        "job_count": len(jobs),
        "done_count": count_done(args, jobs),
        "completed_in_launcher": len(completed),
        "skipped_existing": len(skipped),
        "pending_count": len(pending),
        "running_count": len(running),
        "failure_count": len(failures),
        "gpus": list(gpus),
        "max_concurrent": int(max_concurrent),
        "running": [
            {
                "job": asdict(item["job"]),
                "gpu": item["gpu"],
                "log_path": str(item["log_path"]),
                "cmd": item["cmd"],
            }
            for item in running
        ],
        "failures": list(failures),
    }
    c.write_json(args.out_root / "launcher_state.json", state)
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_json(args.review_root / "launcher_state.json", state)


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str], max_concurrent: int) -> None:
    if not gpus:
        raise RuntimeError("At least one GPU is required.")
    max_concurrent = min(int(max_concurrent), len(gpus))
    if max_concurrent < 1:
        raise RuntimeError(f"Invalid max_concurrent={max_concurrent}")
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = thread_env(args.cpu_threads)
    pending = list(jobs)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    started = time.time()

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % max_concurrent]
            cmd = build_cmd(args, job, gpu)
            print(json.dumps({"launch": cmd, "log": str(log_dir / log_name(job, gpu))}), flush=True)
        return

    write_state(
        args,
        started_epoch=started,
        jobs=jobs,
        pending=pending,
        running=running,
        completed=completed,
        skipped=skipped,
        failures=failures,
        gpus=gpus,
        max_concurrent=max_concurrent,
    )

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
                skipped.append(asdict(job))
                print(json.dumps({"skip_existing": str(done_path.parent), "job": asdict(job)}), flush=True)
                continue
            cmd = build_cmd(args, job, gpu)
            log_path = log_dir / log_name(job, gpu)
            print(json.dumps({"launch": cmd, "log": str(log_path)}), flush=True)
            log_f = log_path.open("w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "gpu": gpu, "proc": proc, "log_file": log_f, "log_path": log_path, "cmd": cmd})
            write_state(
                args,
                started_epoch=started,
                jobs=jobs,
                pending=pending,
                running=running,
                completed=completed,
                skipped=skipped,
                failures=failures,
                gpus=gpus,
                max_concurrent=max_concurrent,
            )
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
                completed.append(asdict(job))
                print(json.dumps({"completed": asdict(job), "gpu": item["gpu"]}), flush=True)
        running = still_running
        write_state(
            args,
            started_epoch=started,
            jobs=jobs,
            pending=pending,
            running=running,
            completed=completed,
            skipped=skipped,
            failures=failures,
            gpus=gpus,
            max_concurrent=max_concurrent,
        )
        if running or pending:
            time.sleep(5.0)

    summary = {
        "completed_at": c.now_text(),
        "job_count": len(jobs),
        "done_count": count_done(args, jobs),
        "completed_in_launcher": len(completed),
        "skipped_existing": len(skipped),
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }
    c.write_json(args.out_root / "launch_summary.json", summary)
    c.write_json(args.review_root / "launch_summary.json", summary)
    if failures:
        raise RuntimeError("One or more cps4 runtime-sweep jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_cps4_runtime_sweep.py"),
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
        "--candidates",
        str(args.candidates),
        "--alpha",
        f"{float(args.alpha):g}",
        "--variant",
        str(args.variant),
        "--expected-eval-interval",
        str(args.eval_interval),
        "--expected-seed",
        str(args.seed),
        "--epoch-override",
        str(args.epoch_override),
    ]
    if args.allow_incomplete_aggregate:
        cmd.append("--allow-incomplete")
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    gpus = resolve_gpus(str(args.gpus))
    max_concurrent = int(args.max_concurrent) if int(args.max_concurrent) > 0 else len(gpus)
    jobs = build_jobs(args)
    write_manifest(args, jobs, gpus, max_concurrent)
    print(
        json.dumps(
            {
                "job_count": len(jobs),
                "gpus": gpus,
                "max_concurrent": max_concurrent,
                "dry_run": bool(args.dry_run),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "candidates": parse_candidate_names(str(args.candidates)),
                "epoch_override": int(args.epoch_override),
            },
            indent=2,
        ),
        flush=True,
    )
    launch_jobs(args, jobs, gpus, max_concurrent)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
