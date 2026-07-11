#!/usr/bin/env python
"""Launch saved DL/Greedy baseline evaluation for OASIS seed-3 ranks 20-99."""

from __future__ import annotations

import argparse
import json
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
import evaluate_saved_baselines_first10 as saved  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402


@dataclass(frozen=True)
class Job:
    job_index: int
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method_key: str
    method_label: str
    family: str
    source_kind: str
    selector: str
    disp_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--case-ranks", type=str, default="20-99")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(spec.method_key for spec in saved.BASELINE_SPECS),
    )
    parser.add_argument("--variant", type=str, default=f100.BASELINE_EXTENSION_VARIANT)
    parser.add_argument("--out-root", type=Path, default=f100.BASELINE_EXTENSION_ROOT)
    parser.add_argument("--review-root", type=Path, default=f100.REVIEW_ROOT / "saved_baselines")
    parser.add_argument("--dl-root", type=Path, default=saved.DEFAULT_DL_ROOT)
    parser.add_argument("--greedy-root", type=Path, default=saved.DEFAULT_GREEDY_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--launch-stagger-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_specs(methods_text: str) -> List[saved.BaselineSpec]:
    return saved.selected_specs(methods_text)


def build_jobs(args: argparse.Namespace) -> List[Job]:
    specs = selected_specs(str(args.methods))
    jobs: List[Job] = []
    for rank in f100.parse_int_ranges(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for spec in specs:
            jobs.append(
                Job(
                    job_index=len(jobs),
                    case_rank=int(case.rank),
                    original_case_index=int(case.original_case_index),
                    fixed_id=str(case.fixed_id),
                    moving_id=str(case.moving_id),
                    method_key=spec.method_key,
                    method_label=spec.method_label,
                    family=spec.family,
                    source_kind=spec.source_kind,
                    selector=spec.selector,
                    disp_path=str(saved.disp_path(args, spec, case.fixed_id, case.moving_id)),
                )
            )
    return jobs


def run_name(job: Job) -> str:
    return f"{job.method_key}_rank{job.case_rank:03d}_{job.selector}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / run_name(job)


def preflight(jobs: Sequence[Job]) -> None:
    missing = [job.disp_path for job in jobs if not Path(job.disp_path).exists()]
    if missing:
        raise FileNotFoundError("Missing saved displacement files:\n" + "\n".join(missing[:80]))


def build_cmd(args: argparse.Namespace, job: Job, gpu: str) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "evaluate_saved_baselines_first10.py"),
        "--case-ranks",
        str(job.case_rank),
        "--methods",
        str(job.method_key),
        "--gpu",
        str(gpu),
        "--device",
        "cuda",
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--variant",
        str(args.variant),
        "--dl-root",
        str(args.dl_root),
        "--greedy-root",
        str(args.greedy_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--skip-aggregate",
        "--no-manifest",
        "--no-evaluation-summary",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(job: Job, gpu: str) -> str:
    return f"rank{job.case_rank:03d}_{job.method_key}_gpu{gpu}.log"


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str], max_concurrent: int) -> None:
    rows = [asdict(job) for job in jobs]
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": c.now_text(),
        "job_count": len(rows),
        "case_ranks": f100.parse_int_ranges(str(args.case_ranks)),
        "methods": [spec.method_key for spec in selected_specs(str(args.methods))],
        "variant": str(args.variant),
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "dl_root": str(args.dl_root),
        "greedy_root": str(args.greedy_root),
        "oasis_root": str(args.oasis_root),
        "pair_file": str(args.pair_file),
        "gpus": list(gpus),
        "max_concurrent": int(max_concurrent),
        "jobs": rows,
    }
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_json(args.out_root / "job_manifest.json", manifest)
    c.write_csv(args.review_root / "job_manifest.csv", rows)
    c.write_json(args.review_root / "job_manifest.json", manifest)


def count_done(args: argparse.Namespace) -> int:
    return len(list(Path(args.out_root).glob("*/*/DONE")))


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
        "started_epoch": float(started_epoch),
        "elapsed_sec": time.time() - float(started_epoch),
        "job_count": len(jobs),
        "done_count": count_done(args),
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
    c.write_json(Path(args.out_root) / "launcher_state.json", state)
    c.write_json(Path(args.review_root) / "launcher_state.json", state)


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str], max_concurrent: int) -> None:
    max_concurrent = min(int(max_concurrent), len(gpus))
    if max_concurrent < 1:
        raise RuntimeError(f"Invalid max_concurrent={max_concurrent}")
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = f100.thread_env(args.cpu_threads)
    pending = list(jobs)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    started = time.time()

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % max_concurrent]
            print(json.dumps({"launch": build_cmd(args, job, gpu), "log": str(log_dir / log_name(job, gpu))}), flush=True)
        return

    write_state(args, started_epoch=started, jobs=jobs, pending=pending, running=running, completed=completed, skipped=skipped, failures=failures, gpus=gpus, max_concurrent=max_concurrent)
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
            write_state(args, started_epoch=started, jobs=jobs, pending=pending, running=running, completed=completed, skipped=skipped, failures=failures, gpus=gpus, max_concurrent=max_concurrent)
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
                failures.append({"job": asdict(job), "gpu": item["gpu"], "returncode": code, "log_path": str(item["log_path"]), "cmd": item["cmd"]})
            else:
                completed.append(asdict(job))
                print(json.dumps({"completed": asdict(job), "gpu": item["gpu"]}), flush=True)
        running = still_running
        write_state(args, started_epoch=started, jobs=jobs, pending=pending, running=running, completed=completed, skipped=skipped, failures=failures, gpus=gpus, max_concurrent=max_concurrent)
        if running or pending:
            time.sleep(5.0)

    summary = {
        "completed_at": c.now_text(),
        "job_count": len(jobs),
        "done_count": count_done(args),
        "completed_in_launcher": len(completed),
        "skipped_existing": len(skipped),
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }
    c.write_json(Path(args.out_root) / "launch_summary.json", summary)
    c.write_json(Path(args.review_root) / "launch_summary.json", summary)
    if failures:
        raise RuntimeError("One or more first100 saved baseline jobs failed: " + json.dumps(failures, indent=2, default=str))


def main() -> None:
    args = parse_args()
    gpus = f100.resolve_gpus(str(args.gpus))
    max_concurrent = int(args.max_concurrent) if int(args.max_concurrent) > 0 else len(gpus)
    jobs = build_jobs(args)
    write_manifest(args, jobs, gpus, max_concurrent)
    preflight(jobs)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "launch",
                "job_count": len(jobs),
                "case_ranks": f100.parse_int_ranges(str(args.case_ranks)),
                "methods": [spec.method_key for spec in selected_specs(str(args.methods))],
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "gpus": gpus,
                "max_concurrent": int(max_concurrent),
            },
            indent=2,
        ),
        flush=True,
    )
    launch_jobs(args, jobs, gpus, max_concurrent)


if __name__ == "__main__":
    main()

