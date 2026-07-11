#!/usr/bin/env python
"""Launch LungCT IDIR sampled pTV-LCC + TV weight-0.11 pilot jobs."""

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
from run_one import METHOD_IDIR_SAMPLED_PTV_TV, default_lr, run_name


RUNNER = Path(__file__).resolve().parent / "run_one.py"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1")
DEFAULT_LOG_ROOT = c.REPO_ROOT / "logs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1"
DEFAULT_WEIGHT = 0.11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ptv-isotv-weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--sampled-lcc-centers", type=int, default=512)
    parser.add_argument("--sampled-lcc-offsets", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "completed"


def build_job(args: argparse.Namespace, case_id: int) -> Dict[str, object]:
    lr = float(args.lr if args.lr is not None else default_lr(METHOD_IDIR_SAMPLED_PTV_TV))
    fake = argparse.Namespace(
        run_name="",
        method=METHOD_IDIR_SAMPLED_PTV_TV,
        case_id=int(case_id),
        seed=int(args.seed),
        lr=lr,
        epochs=int(args.epochs),
        alpha_bending=10.0,
        ptv_isotv_weight=float(args.ptv_isotv_weight),
        stage_epochs="200,400,800,1000",
    )
    name = run_name(fake, METHOD_IDIR_SAMPLED_PTV_TV, lr)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--method",
        METHOD_IDIR_SAMPLED_PTV_TV,
        "--case-id",
        str(int(case_id)),
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(args.out_root),
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(lr),
        "--ptv-isotv-weight",
        str(float(args.ptv_isotv_weight)),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--sampled-lcc-centers",
        str(int(args.sampled_lcc_centers)),
        "--sampled-lcc-offsets",
        str(int(args.sampled_lcc_offsets)),
    ]
    if args.force:
        cmd.append("--overwrite")
    return {
        "method": METHOD_IDIR_SAMPLED_PTV_TV,
        "case_id": int(case_id),
        "run_name": name,
        "weight": float(args.ptv_isotv_weight),
        "cmd": cmd,
        "summary_path": args.out_root / METHOD_IDIR_SAMPLED_PTV_TV / name / "summary.json",
        "log_path": args.log_root / METHOD_IDIR_SAMPLED_PTV_TV / f"{name}.log",
    }


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    return [build_job(args, case_id) for case_id in c.parse_int_list(args.cases)]


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
        "method": METHOD_IDIR_SAMPLED_PTV_TV,
        "ptv_isotv_weight": float(args.ptv_isotv_weight),
        "epochs": int(args.epochs),
        "lr": float(args.lr if args.lr is not None else default_lr(METHOD_IDIR_SAMPLED_PTV_TV)),
        "sampled_lcc_centers": int(args.sampled_lcc_centers),
        "sampled_lcc_offsets": int(args.sampled_lcc_offsets),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for job in queued:
            print(
                json.dumps(
                    {
                        "dry_run_job": job["run_name"],
                        "method": job["method"],
                        "case_id": job["case_id"],
                        "weight": job["weight"],
                        "cmd": job["cmd"],
                    }
                ),
                flush=True,
            )
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
            payload = {
                "finished": job["run_name"],
                "case_id": job["case_id"],
                "returncode": code,
                "elapsed_sec": elapsed,
                "gpu": item["gpu"],
            }
            print(json.dumps(payload), flush=True)
            if code != 0:
                failures.append(payload)
        active = still_active
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()
