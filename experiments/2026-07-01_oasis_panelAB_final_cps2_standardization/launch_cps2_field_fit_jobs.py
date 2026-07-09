#!/usr/bin/env python
"""Launch OASIS rank-9 final-cps cps2 field-fit add-on jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_ROOT = Path(__file__).resolve().parent
RUNNER_ROOT = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep"
DATA_RUN_ROOT = Path("/") / "database" / "inrorspline" / "runs"


@dataclass(frozen=True)
class Job:
    target: str
    method: str
    method_key: str
    lr: str
    gpu: str

    @property
    def prefix(self) -> str:
        return "sitreg_field_fit" if self.target == "sitreg" else "greedy_field_fit"

    @property
    def lr_tag(self) -> str:
        return {"1e-4": "0p0001", "1e-3": "0p001"}[self.lr]

    def run_name(self, epochs: int) -> str:
        return f"{self.prefix}_{self.method_key}_rank009_a1000_e{epochs}_lr{self.lr_tag}"

    def script(self) -> Path:
        name = "run_rank009_sitreg_field_fit_one.py" if self.target == "sitreg" else "run_rank009_greedy_field_fit_one.py"
        return RUNNER_ROOT / name


JOBS = [
    Job("sitreg", "sinr_cps2_be_bspline_analytic", "sinr_cps2_field_mse", "1e-4", "0"),
    Job("sitreg", "direct_cp_cps2_be_bspline_analytic", "direct_cp_cps2_field_mse", "1e-3", "1"),
    Job("greedy", "sinr_cps2_be_bspline_analytic", "sinr_cps2_field_mse", "1e-4", "2"),
    Job("greedy", "direct_cp_cps2_be_bspline_analytic", "direct_cp_cps2_field_mse", "1e-3", "3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    cmd = [
        sys.executable,
        str(job.script()),
        "--method",
        job.method,
        "--alpha",
        "1000",
        "--lr",
        job.lr,
        "--gpu",
        job.gpu,
        "--case-rank",
        "9",
        "--epochs",
        str(args.epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        "1",
        "--variant",
        args.variant,
        "--out-root",
        str(args.out_root),
        "--expected-original-case-index",
        "88",
        "--expected-fixed-id",
        "0173",
        "--expected-moving-id",
        "0136",
        "--run-name",
        job.run_name(args.epochs),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def sanitize_review_value(value):
    if isinstance(value, str):
        value = value.replace(str(REPO_ROOT), "${REPO_ROOT_IN_CONTAINER}")
        value = value.replace(str(DATA_RUN_ROOT), "${DATA_ROOT}/runs")
        return value
    if isinstance(value, list):
        return [sanitize_review_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_review_value(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    log_dir = REVIEW_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    procs = []
    for job in JOBS:
        log_path = log_dir / f"{args.mode}_{job.target}_{job.method_key}_lr{job.lr_tag}_gpu{job.gpu}.log"
        cmd = build_cmd(args, job)
        manifest.append(
            {
                **asdict(job),
                "run_name": job.run_name(args.epochs),
                "log": str(log_path.relative_to(REVIEW_ROOT)),
                "cmd": sanitize_review_value(cmd),
            }
        )
        log_file = log_path.open("w")
        procs.append(
            {
                "job": job,
                "proc": subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT),
                "log_file": log_file,
                "log": log_path,
            }
        )

    (REVIEW_ROOT / f"{args.mode}_job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.time()
    while True:
        completed = [item for item in procs if item["proc"].poll() is not None]
        failed = [item for item in completed if item["proc"].returncode not in (0, None)]
        running = [item for item in procs if item["proc"].poll() is None]
        elapsed = time.time() - start
        completed_count = len(completed)
        jobs_per_hour = completed_count / (elapsed / 3600.0) if elapsed > 0 else 0.0
        remaining = len(procs) - completed_count
        eta_sec = remaining / jobs_per_hour * 3600.0 if jobs_per_hour > 0 else None
        newest_log_mtime = None
        for item in procs:
            try:
                mtime = item["log"].stat().st_mtime
            except FileNotFoundError:
                continue
            newest_log_mtime = mtime if newest_log_mtime is None else max(newest_log_mtime, mtime)

        snapshot = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_sec": elapsed,
            "completed_jobs": completed_count,
            "failed_jobs": len(failed),
            "running_jobs": len(running),
            "remaining_jobs": remaining,
            "jobs_per_hour": jobs_per_hour,
            "eta_sec": eta_sec,
            "newest_log_update_time": (
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(newest_log_mtime))
                if newest_log_mtime
                else None
            ),
            "failure_logs": [str(item["log"].relative_to(REVIEW_ROOT)) for item in failed],
        }
        with (REVIEW_ROOT / f"{args.mode}_monitor_snapshots.jsonl").open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
        eta_text = "unknown" if eta_sec is None else f"{eta_sec / 60.0:.1f} min"
        (REVIEW_ROOT / f"{args.mode}_monitor_latest.md").write_text(
            "\n".join(
                [
                    f"# Latest {args.mode} PanelAB cps2 monitor",
                    "",
                    f"- Time: {snapshot['time']}",
                    f"- Completed: {completed_count}/{len(procs)}",
                    f"- Failed: {len(failed)}",
                    f"- Running: {len(running)}",
                    f"- Remaining: {remaining}",
                    f"- Elapsed minutes: {elapsed / 60.0:.1f}",
                    f"- Jobs/hour: {jobs_per_hour:.2f}",
                    f"- ETA: {eta_text}",
                    f"- Newest log update: {snapshot['newest_log_update_time']}",
                    f"- Failure logs: {', '.join(snapshot['failure_logs']) if snapshot['failure_logs'] else 'none'}",
                    "",
                ]
            )
        )
        print(json.dumps(snapshot), flush=True)
        if completed_count == len(procs):
            break
        time.sleep(max(1, int(args.poll_sec)))

    failures = []
    for item in procs:
        rc = item["proc"].wait()
        item["log_file"].close()
        if rc != 0:
            failures.append(
                {
                    **asdict(item["job"]),
                    "returncode": rc,
                    "log": str(item["log"].relative_to(REVIEW_ROOT)),
                }
            )
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        raise SystemExit(1)
    print(
        json.dumps(
            {"status": "ok", "mode": args.mode, "out_root": sanitize_review_value(str(args.out_root))},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
