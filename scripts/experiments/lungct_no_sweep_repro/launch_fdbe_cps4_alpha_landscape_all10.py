#!/usr/bin/env python
"""Launch the LungCT unmasked FD-BE cps4 alpha-landscape runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import launch_fdbe_cps4_alpha_extension_all10 as base  # noqa: E402

EXPERIMENTS_DIR = base.REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


RUN_KEY = "lungct_fdbe_cps4_alpha_landscape_all10_v1"
SMOKE_RUN_KEY = "lungct_fdbe_cps4_alpha_landscape_all10_smoke_v1"
EXPERIMENT_KEY = "2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10"
DEFAULT_RUN_ROOT = None
DEFAULT_SMOKE_RUN_ROOT = None
DEFAULT_REVIEW_DIR = None
DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
DEFAULT_CASES = base.DEFAULT_CASES

METHOD_LANES = (
    {
        "method_slug": "inr_bscp",
        "display_method": "INR-BSCP cps4 (FD-BE)",
        "implementation_id": "sinr_cps4_global_ncc_fd_be",
        "lr": 1.0e-4,
    },
    {
        "method_slug": "d_bscp",
        "display_method": "D-BSCP cps4 (FD-BE)",
        "implementation_id": "direct_cp_cps4_global_ncc_fd_be",
        "lr": 1.0e-3,
    },
)
WAVE_ALPHAS = {
    "wave1": (20000.0, 50000.0, 200000.0, 500000.0),
    "wave2": (10000.0,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=sorted(WAVE_ALPHAS), default="wave1")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPUs, one per alpha lane.")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--epoch-override", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--global-ncc-diagnostic-interval", type=int, default=10)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--gpu-check-timeout-sec", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args()
    if args.epoch_override is not None and int(args.epoch_override) < 1:
        raise ValueError("--epoch-override must be >= 1")
    if args.gpus is None:
        args.gpus = "0,1,2,3" if args.wave == "wave1" else "0,1"
    if args.smoke:
        if str(args.cases) == DEFAULT_CASES:
            args.cases = "1"
        if args.epoch_override is None:
            args.epoch_override = 1
        args.eval_interval = 1
        args.global_ncc_diagnostic_interval = 1
    return resolve_artifact_paths(args)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve landscape run, log, and review outputs externally."""

    run_key = SMOKE_RUN_KEY if args.smoke else RUN_KEY
    args.run_root = artifact_paths.resolve_run_root(run_key, args.run_root, repo_root=base.REPO_ROOT)
    review_parts = [EXPERIMENT_KEY]
    if args.smoke:
        review_parts.append("smoke")
    review_parts.append(str(args.wave))
    review_key = "/".join(review_parts)
    requested_review_dir = None if args.review_dir is None else Path(args.review_dir)
    if requested_review_dir is not None:
        if args.smoke:
            requested_review_dir = requested_review_dir / "smoke"
        requested_review_dir = requested_review_dir / str(args.wave)
    args.review_dir = artifact_paths.resolve_review_root(
        review_key, requested_review_dir, repo_root=base.REPO_ROOT
    )
    return args


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_gpus(text: str, expected: int) -> tuple[str, ...]:
    gpus = tuple(token.strip() for token in str(text).split(",") if token.strip())
    if len(gpus) != expected:
        raise ValueError(f"Expected {expected} GPU ids for this wave, got {gpus}")
    return gpus


def expected_gpu_count(wave: str) -> int:
    return len(METHOD_LANES) if wave == "wave2" else len(WAVE_ALPHAS[wave])


def make_method_lane(method: dict[str, Any], alpha: float, gpu: str) -> base.LaneSpec:
    return base.LaneSpec(
        lane_slug=f"{method['method_slug']}_a{base.c.format_float(alpha)}",
        method_slug=str(method["method_slug"]),
        display_method=str(method["display_method"]),
        implementation_id=str(method["implementation_id"]),
        lr=float(method["lr"]),
        alpha=float(alpha),
        gpu=str(gpu),
    )


def build_lane_groups(args: argparse.Namespace, gpus: Sequence[str]) -> list[dict[str, Any]]:
    cases = base.parse_cases(args.cases)
    groups: list[dict[str, Any]] = []
    if args.wave == "wave2":
        alpha = WAVE_ALPHAS[args.wave][0]
        for method, gpu in zip(METHOD_LANES, gpus):
            group_jobs: list[dict[str, Any]] = []
            method_lane = make_method_lane(method, alpha, str(gpu))
            for case_id in cases:
                job = base.build_job(args, int(case_id), method_lane)
                job["gpu_lane_slug"] = f"gpu{gpu}_{method['method_slug']}_a{base.c.format_float(alpha)}"
                job["wave"] = args.wave
                group_jobs.append(job)
            groups.append(
                {
                    "gpu_lane_slug": f"gpu{gpu}_{method['method_slug']}_a{base.c.format_float(alpha)}",
                    "gpu": str(gpu),
                    "alpha": float(alpha),
                    "jobs": group_jobs,
                }
            )
        return groups
    for alpha, gpu in zip(WAVE_ALPHAS[args.wave], gpus):
        group_jobs: list[dict[str, Any]] = []
        for method in METHOD_LANES:
            method_lane = make_method_lane(method, alpha, str(gpu))
            for case_id in cases:
                job = base.build_job(args, int(case_id), method_lane)
                job["gpu_lane_slug"] = f"gpu{gpu}_a{base.c.format_float(alpha)}"
                job["wave"] = args.wave
                group_jobs.append(job)
        groups.append(
            {
                "gpu_lane_slug": f"gpu{gpu}_a{base.c.format_float(alpha)}",
                "gpu": str(gpu),
                "alpha": float(alpha),
                "jobs": group_jobs,
            }
        )
    return groups


def flatten(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [job for group in groups for job in group["jobs"]]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_state(review_dir: Path, jobs: Sequence[dict[str, Any]], started_at: float) -> None:
    statuses = sorted({str(job.get("status", "")) for job in jobs})
    counts = {status: sum(str(job.get("status", "")) == status for job in jobs) for status in statuses}
    payload = {
        "updated_at_utc": utc_text(),
        "elapsed_sec": time.time() - started_at,
        "counts": counts,
        "jobs": jobs,
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "launcher_state.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    write_csv(review_dir / "job_manifest.csv", jobs)
    lines = [
        "# LungCT FD-BE cps4 alpha-landscape launcher",
        "",
        f"- Updated: `{payload['updated_at_utc']}`",
        f"- Counts: `{counts}`",
        "",
        "| Job | Status | Wave | GPU | Alpha | Method | Elapsed sec | Note |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for job in jobs:
        lines.append(
            f"| `{job['job_id']}` | `{job.get('status', '')}` | `{job.get('wave', '')}` | "
            f"`{job.get('gpu', '')}` | `{job.get('alpha', '')}` | `{job.get('method_slug', '')}` | "
            f"`{job.get('elapsed_sec', '')}` | `{job.get('failure_note', '')}` |"
        )
    (review_dir / "launcher_monitor.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus, expected_gpu_count(args.wave))
    lane_groups = build_lane_groups(args, gpus)
    jobs = flatten(lane_groups)
    started_at = time.time()
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            ok, note = base.compatible_complete(job)
            if ok:
                job["status"] = "completed_existing"
                job["failure_note"] = ""
            else:
                job["failure_note"] = note
        write_state(review_dir, jobs, started_at)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "wave": args.wave,
                    "jobs": len(jobs),
                    "alpha_lanes": [group["alpha"] for group in lane_groups],
                    "cases": list(base.parse_cases(args.cases)),
                    "gpus": list(gpus),
                    "pending": sum(job["status"] == "pending" for job in jobs),
                    "completed_existing": sum(job["status"] == "completed_existing" for job in jobs),
                    "review_dir": str(review_dir),
                    "run_root": str(args.run_root),
                },
                indent=2,
            )
        )
        return

    if not args.skip_gpu_check:
        gpu_state = base.check_gpus(
            gpus,
            min_free_gb=float(args.min_free_gb),
            timeout_sec=float(args.gpu_check_timeout_sec),
        )
        (review_dir / "gpu_preflight.json").write_text(json.dumps(gpu_state, indent=2) + "\n")

    write_state(review_dir, jobs, started_at)
    base.write_state = write_state
    lock = threading.Lock()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for group in lane_groups:
        thread = threading.Thread(
            target=base.worker,
            kwargs={
                "lane_slug": str(group["gpu_lane_slug"]),
                "lane_jobs": list(group["jobs"]),
                "all_jobs": jobs,
                "review_dir": review_dir,
                "lock": lock,
                "stop_event": stop_event,
                "started_at": started_at,
                "resume": bool(args.resume),
                "overwrite": bool(args.overwrite),
            },
            daemon=False,
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    write_state(review_dir, jobs, started_at)
    failed = [job["job_id"] for job in jobs if job["status"] == "failed"]
    print(json.dumps({"wave": args.wave, "jobs": len(jobs), "failed": failed, "review_dir": str(review_dir)}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
