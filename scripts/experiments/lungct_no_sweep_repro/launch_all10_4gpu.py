#!/usr/bin/env python
"""Launch LungCT no-sweep reproduction jobs across four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import common as c
from run_one import METHODS, default_lr, run_name


RUNNER = Path(__file__).resolve().parent / "run_one.py"
DEFAULT_LOG_ROOT = c.REPO_ROOT / "logs/lungct_ptvprep_no_sweep_repro_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-dense-dvf", action="store_true")
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = c.parse_int_list(args.cases)
    methods = [m for m in args.methods.replace(" ", "").split(",") if m]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")
    jobs: List[Dict[str, object]] = []
    for method in methods:
        lr = default_lr(method)
        for case_id in cases:
            fake = argparse.Namespace(
                run_name="",
                method=method,
                case_id=case_id,
                seed=args.seed,
                lr=None,
                epochs=None,
                alpha_bending=10.0,
                ptv_isotv_weight=0.11,
                stage_epochs="200,400,800,1000",
            )
            name = run_name(fake, method, lr)
            summary_path = args.out_root / method / name / "summary.json"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--method",
                method,
                "--case-id",
                str(case_id),
                "--data-root",
                str(args.data_root),
                "--out-root",
                str(args.out_root),
                "--seed",
                str(args.seed),
            ]
            if args.force:
                cmd.append("--overwrite")
            if args.no_save_dense_dvf:
                cmd.append("--no-save-dense-dvf")
            jobs.append(
                {
                    "method": method,
                    "case_id": case_id,
                    "run_name": name,
                    "cmd": cmd,
                    "summary_path": summary_path,
                    "log_path": args.log_root / method / f"{name}.log",
                }
            )
    return jobs


def is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "completed"


def main() -> None:
    args = parse_args()
    gpus = [str(v) for v in c.parse_int_list(args.gpus)]
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    queued = [job for job in jobs if args.force or not is_completed(Path(job["summary_path"]))]
    skipped = [job for job in jobs if job not in queued]
    manifest = {
        "requested_jobs": len(jobs),
        "queued_jobs": len(queued),
        "skipped_completed": len(skipped),
        "gpus": gpus,
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for job in queued:
            print(json.dumps({"dry_run_job": job["run_name"], "cmd": job["cmd"]}), flush=True)
        return

    active: List[Dict[str, object]] = []
    pending = list(queued)
    failures: List[Dict[str, object]] = []
    while pending or active:
        while pending and len(active) < len(gpus):
            used = {str(item["gpu"]) for item in active}
            free = [gpu for gpu in gpus if gpu not in used]
            if not free:
                break
            gpu = free[0]
            job = pending.pop(0)
            cmd = list(job["cmd"]) + ["--gpu", gpu]
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=c.REPO_ROOT,
                env=os.environ.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({"proc": proc, "log_file": log_file, "gpu": gpu, "job": job, "start": time.time()})
            print(
                json.dumps(
                    {
                        "launched": job["run_name"],
                        "method": job["method"],
                        "case_id": job["case_id"],
                        "gpu": gpu,
                        "pid": proc.pid,
                        "remaining": len(pending),
                        "log": str(log_path),
                    }
                ),
                flush=True,
            )

        time.sleep(30)
        still_active: List[Dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still_active.append(item)
                continue
            log_file = item["log_file"]
            log_file.close()
            job = item["job"]
            elapsed = time.time() - float(item["start"])
            payload = {"finished": job["run_name"], "returncode": code, "elapsed_sec": elapsed, "gpu": item["gpu"]}
            print(json.dumps(payload), flush=True)
            if code != 0:
                failures.append(payload)
        active = still_active
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()
