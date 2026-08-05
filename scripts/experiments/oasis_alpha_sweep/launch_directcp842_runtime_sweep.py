#!/usr/bin/env python
"""Launch OASIS DirectCP 8-4-2 runtime optimization candidates."""

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
VARIANT = "oasis_seed3_first10_directcp842_runtime_sweep_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_runtime_sweep_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_runtime_sweep"
ALPHA = 1000.0
STAGE_FULL_CPS = "8,4,2"
STAGE_DOWNSAMPLE = "4,2,1"


@dataclass(frozen=True)
class Candidate:
    name: str
    lr: float
    stage_epochs: str
    stage_lr: str = ""
    early_stop_mode: str = "none"
    stage_min_epochs: str = ""
    stage_patience: str = "100"
    stage_check_interval: str = "25"
    stage_rel_delta_total: str = "5e-5"
    stage_rel_delta_data: str = "2e-5"
    description: str = ""


@dataclass(frozen=True)
class Job:
    job_index: int
    candidate: str
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method: str
    method_key: str
    alpha: float
    lr: float
    stage_epochs: str
    stage_lr: str
    early_stop_mode: str
    stage_min_epochs: str
    stage_patience: str
    stage_check_interval: str
    stage_rel_delta_total: str
    stage_rel_delta_data: str


CANDIDATES: Sequence[Candidate] = [
    Candidate("F1550_lr3e-3", 3e-3, "250,500,800", description="fixed 1550 epochs"),
    Candidate("F1550_lr1e-2", 1e-2, "250,500,800", description="fixed 1550 epochs"),
    Candidate("F1550_lr3e-2", 3e-2, "250,500,800", description="fixed 1550 epochs"),
    Candidate("F950_lr1e-2", 1e-2, "150,300,500", description="fixed 950 epochs"),
    Candidate("F950_lr3e-2", 3e-2, "150,300,500", description="fixed 950 epochs"),
    Candidate("F950_lr1e-1", 1e-1, "150,300,500", description="fixed 950 epochs"),
    Candidate("F950_decay", 1e-2, "150,300,500", stage_lr="3e-2,1e-2,3e-3", description="fixed 950 epochs, decayed stage LR"),
    Candidate("F600_lr1e-2", 1e-2, "100,200,300", description="fixed 600 epochs"),
    Candidate("F600_lr3e-2", 3e-2, "100,200,300", description="fixed 600 epochs"),
    Candidate("F600_lr1e-1", 1e-1, "100,200,300", description="fixed 600 epochs"),
    Candidate("F600_decay", 1e-2, "100,200,300", stage_lr="3e-2,1e-2,3e-3", description="fixed 600 epochs, decayed stage LR"),
    Candidate(
        "A_safe_lr1e-2",
        1e-2,
        "400,800,1300",
        early_stop_mode="adaptive",
        stage_min_epochs="150,300,500",
        stage_patience="100,150,200",
        stage_rel_delta_total="2e-5",
        stage_rel_delta_data="1e-5",
        description="adaptive conservative plateau",
    ),
    Candidate(
        "A_safe_lr3e-2",
        3e-2,
        "400,800,1300",
        early_stop_mode="adaptive",
        stage_min_epochs="150,300,500",
        stage_patience="100,150,200",
        stage_rel_delta_total="2e-5",
        stage_rel_delta_data="1e-5",
        description="adaptive conservative plateau",
    ),
    Candidate(
        "A_bal_lr1e-2",
        1e-2,
        "400,800,1300",
        early_stop_mode="adaptive",
        stage_min_epochs="100,200,300",
        stage_patience="75,100,150",
        stage_rel_delta_total="5e-5",
        stage_rel_delta_data="2e-5",
        description="adaptive balanced plateau",
    ),
    Candidate(
        "A_bal_lr3e-2",
        3e-2,
        "400,800,1300",
        early_stop_mode="adaptive",
        stage_min_epochs="100,200,300",
        stage_patience="75,100,150",
        stage_rel_delta_total="5e-5",
        stage_rel_delta_data="2e-5",
        description="adaptive balanced plateau",
    ),
    Candidate(
        "A_fast_decay",
        1e-2,
        "400,800,1300",
        stage_lr="3e-2,1e-2,3e-3",
        early_stop_mode="adaptive",
        stage_min_epochs="75,150,250",
        stage_patience="50,75,100",
        stage_rel_delta_total="1e-4",
        stage_rel_delta_data="5e-5",
        description="adaptive fast plateau, decayed stage LR",
    ),
]


def candidate_map() -> Dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def parse_candidate_names(text: str) -> List[str]:
    requested = str(text).strip()
    if requested in {"", "all"}:
        return [candidate.name for candidate in CANDIDATES]
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
    parser.add_argument("--stage-full-cps", type=str, default=STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=STAGE_DOWNSAMPLE)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
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


def candidate_epochs(candidate: Candidate) -> int:
    return int(sum(c.parse_int_list(candidate.stage_epochs)))


def build_jobs(args: argparse.Namespace) -> List[Job]:
    selected = [candidate_map()[name] for name in parse_candidate_names(str(args.candidates))]
    jobs: List[Job] = []
    for rank in parse_case_ranks(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for candidate in selected:
            jobs.append(
                Job(
                    job_index=len(jobs),
                    candidate=candidate.name,
                    case_rank=int(case.rank),
                    original_case_index=int(case.original_case_index),
                    fixed_id=str(case.fixed_id),
                    moving_id=str(case.moving_id),
                    method=METHOD,
                    method_key=METHOD,
                    alpha=float(args.alpha),
                    lr=float(candidate.lr),
                    stage_epochs=str(candidate.stage_epochs),
                    stage_lr=str(candidate.stage_lr),
                    early_stop_mode=str(candidate.early_stop_mode),
                    stage_min_epochs=str(candidate.stage_min_epochs),
                    stage_patience=str(candidate.stage_patience),
                    stage_check_interval=str(candidate.stage_check_interval),
                    stage_rel_delta_total=str(candidate.stage_rel_delta_total),
                    stage_rel_delta_data=str(candidate.stage_rel_delta_data),
                )
            )
    return jobs


def build_run_name(job: Job) -> str:
    return f"{METHOD}_runtime_{job.candidate}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / METHOD / build_run_name(job)


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
        str(candidate_epochs(candidate_map()[job.candidate])),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(job.stage_epochs),
        "--stage-early-stop-mode",
        str(job.early_stop_mode),
        "--stage-patience",
        str(job.stage_patience),
        "--stage-check-interval",
        str(job.stage_check_interval),
        "--stage-rel-delta-total",
        str(job.stage_rel_delta_total),
        "--stage-rel-delta-data",
        str(job.stage_rel_delta_data),
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
    if job.stage_lr:
        cmd.extend(["--stage-lr", str(job.stage_lr)])
    if job.stage_min_epochs:
        cmd.extend(["--stage-min-epochs", str(job.stage_min_epochs)])
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
        "method": METHOD,
        "variant": str(args.variant),
        "alpha": float(args.alpha),
        "stage_full_cps": str(args.stage_full_cps),
        "stage_downsample": str(args.stage_downsample),
        "eval_interval": int(args.eval_interval),
        "seed": int(args.seed),
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
        return len(list((Path(args.out_root) / METHOD).glob("*/DONE")))
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
        raise RuntimeError("One or more runtime-sweep jobs failed: " + json.dumps(failures, indent=2, default=str))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_directcp842_runtime_sweep.py"),
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
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "launch",
                "job_count": len(jobs),
                "case_ranks": parse_case_ranks(str(args.case_ranks)),
                "candidates": parse_candidate_names(str(args.candidates)),
                "method": METHOD,
                "variant": str(args.variant),
                "alpha": float(args.alpha),
                "stage_full_cps": str(args.stage_full_cps),
                "stage_downsample": str(args.stage_downsample),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "gpus": gpus,
                "max_concurrent": int(max_concurrent),
                "eval_interval": args.eval_interval,
                "seed": args.seed,
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
