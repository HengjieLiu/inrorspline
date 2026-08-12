#!/usr/bin/env python
"""Launch the OASIS MR-SINR 8/4/2 one-case pilot."""

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
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_mr_sinr842_revision_pilot_v1")
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-08-07_oasis_mr_sinr842_revision"
VARIANT = "oasis_mr_sinr842_revision_rank0_alpha1000_pilot_v1"


@dataclass(frozen=True)
class PilotJob:
    method: str
    gpu: str
    alpha: float = 1000.0
    case_rank: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--stage-full-cps", type=str, default="8,4,2")
    parser.add_argument("--stage-downsample", type=str, default="4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="400,800,1300")
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    return parser.parse_args()


def jobs(args: argparse.Namespace) -> List[PilotJob]:
    gpu_values = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if not gpu_values:
        gpu_values = ["0"]
    methods = ["mr_sinr842_composed", "mr_sinr842_requery"]
    return [
        PilotJob(method=method, gpu=gpu_values[idx % len(gpu_values)], alpha=float(args.alpha))
        for idx, method in enumerate(methods)
    ]


def run_name(job: PilotJob, args: argparse.Namespace) -> str:
    return f"{job.method}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}_pilot"


def command_for_job(job: PilotJob, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method",
        job.method,
        "--alpha",
        str(job.alpha),
        "--gpu",
        str(job.gpu),
        "--case-rank",
        str(job.case_rank),
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
        "--out-root",
        str(args.out_root),
        "--pair-file",
        str(c.DEFAULT_PAIR_FILE),
        "--expected-original-case-index",
        "158",
        "--expected-fixed-id",
        "0333",
        "--expected-moving-id",
        "0115",
        "--run-name",
        run_name(job, args),
        "--variant",
        str(args.variant),
    ]
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def aggregate_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_mr_sinr842_pilot.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
    ]


def log_path(args: argparse.Namespace, job: PilotJob) -> Path:
    return Path(args.out_root) / "logs" / f"{run_name(job, args)}.log"


def run_job(args: argparse.Namespace, job: PilotJob) -> Dict[str, object]:
    cmd = command_for_job(job, args)
    path = log_path(args, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with path.open("w") as log_f:
        log_f.write(json.dumps({"command": cmd, "started_at": c.now_text()}) + "\n")
        log_f.flush()
        code = subprocess.call(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return {
        "method": job.method,
        "gpu": job.gpu,
        "run_name": run_name(job, args),
        "log_path": str(path),
        "returncode": int(code),
        "elapsed_sec": float(time.time() - start),
    }


def run_jobs(args: argparse.Namespace, launch_jobs: Sequence[PilotJob]) -> List[Dict[str, object]]:
    if not bool(args.parallel):
        return [run_job(args, job) for job in launch_jobs]

    records: List[Dict[str, object]] = []
    processes: List[tuple[PilotJob, subprocess.Popen[object], object, float, Path]] = []
    for job in launch_jobs:
        cmd = command_for_job(job, args)
        path = log_path(args, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_f = path.open("w")
        log_f.write(json.dumps({"command": cmd, "started_at": c.now_text()}) + "\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((job, proc, log_f, time.time(), path))
    for job, proc, log_f, start, path in processes:
        code = proc.wait()
        log_f.close()
        records.append(
            {
                "method": job.method,
                "gpu": job.gpu,
                "run_name": run_name(job, args),
                "log_path": str(path),
                "returncode": int(code),
                "elapsed_sec": float(time.time() - start),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": c.now_text(),
        "variant": str(args.variant),
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "case_rank": 0,
        "expected_original_case_index": 158,
        "expected_fixed_id": "0333",
        "expected_moving_id": "0115",
        "alpha": float(args.alpha),
        "epochs": int(args.epochs),
        "stage_full_cps": str(args.stage_full_cps),
        "stage_downsample": str(args.stage_downsample),
        "stage_epochs": str(args.stage_epochs),
        "jobs": [
            {
                **asdict(job),
                "run_name": run_name(job, args),
                "command": command_for_job(job, args),
            }
            for job in jobs(args)
        ],
        "aggregate_command": aggregate_command(args),
        "launched": bool(args.launch),
        "parallel": bool(args.parallel),
    }
    c.write_json(args.review_root / "job_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if not bool(args.launch):
        return
    launch_records = run_jobs(args, jobs(args))
    c.write_json(args.review_root / "launch_summary.json", {"jobs": launch_records, "created_at": c.now_text()})
    failures = [record for record in launch_records if int(record["returncode"]) != 0]
    if failures:
        raise RuntimeError(f"Pilot job failures: {failures}")
    if bool(args.aggregate_after):
        subprocess.check_call(aggregate_command(args))


if __name__ == "__main__":
    main()
