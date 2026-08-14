#!/usr/bin/env python
"""Launch LungCT 400-400-800-900 schedule-standardization pilot jobs."""

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
from run_one import (
    METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
    METHOD_DIRECT_CP_32_PTV,
    default_lr,
    run_name,
)


RUNNER = Path(__file__).resolve().parent / "run_one.py"
PILOT_PTV = "ptvreg_w0p11"
PILOT_MASKED = "masked_alpha33333"
PILOTS = [PILOT_PTV, PILOT_MASKED]

DEFAULT_CASES = "1,2,3,4,5,6,7,8,9,10"
DEFAULT_STAGE_FULL_CPS = "32,16,8,4"
DEFAULT_STAGE_DOWNSAMPLE = "8,4,2,1"
DEFAULT_STAGE_EPOCHS = "400,400,800,900"
DEFAULT_ALPHA = 33333.333333333336
DEFAULT_PTV_WEIGHT = 0.11

DEFAULT_PTV_RUN_ROOT = Path("/database/inrorspline/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1")
DEFAULT_MASKED_RUN_ROOT = Path(
    "/database/inrorspline/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10_v1"
)
DEFAULT_LOG_ROOT = c.REPO_ROOT / "logs/lungct_2500_schedule_standardization_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilots", type=str, default=",".join(PILOTS), help=f"Comma list from {PILOTS}")
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--ptv-out-root", type=Path, default=DEFAULT_PTV_RUN_ROOT)
    parser.add_argument("--masked-out-root", type=Path, default=DEFAULT_MASKED_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ptv-isotv-weight", type=float, default=DEFAULT_PTV_WEIGHT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--stage-full-cps", type=str, default=DEFAULT_STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=DEFAULT_STAGE_DOWNSAMPLE)
    parser.add_argument("--stage-epochs", type=str, default=DEFAULT_STAGE_EPOCHS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_pilots(value: str) -> List[str]:
    pilots = [v for v in value.replace(" ", "").split(",") if v]
    unknown = sorted(set(pilots) - set(PILOTS))
    if unknown:
        raise ValueError(f"Unknown pilot(s): {unknown}; expected subset of {PILOTS}")
    if not pilots:
        raise ValueError("At least one pilot is required.")
    return pilots


def is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "completed"


def build_job(args: argparse.Namespace, pilot: str, case_id: int) -> Dict[str, object]:
    if pilot == PILOT_PTV:
        method = METHOD_DIRECT_CP_32_PTV
        out_root = args.ptv_out_root
        lr = float(args.lr if args.lr is not None else default_lr(method))
        fake = argparse.Namespace(
            run_name="",
            method=method,
            dataset="4dct",
            case_id=case_id,
            seed=args.seed,
            lr=lr,
            epochs=None,
            alpha_bending=10.0,
            alpha=args.alpha,
            ptv_isotv_weight=args.ptv_isotv_weight,
            stage_full_cps=args.stage_full_cps,
            stage_downsample=args.stage_downsample,
            stage_epochs=args.stage_epochs,
        )
        extra_args = ["--ptv-isotv-weight", str(float(args.ptv_isotv_weight))]
        reg_value = float(args.ptv_isotv_weight)
    elif pilot == PILOT_MASKED:
        method = METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE
        out_root = args.masked_out_root
        lr = float(args.lr if args.lr is not None else default_lr(method))
        fake = argparse.Namespace(
            run_name="",
            method=method,
            dataset="4dct",
            case_id=case_id,
            seed=args.seed,
            lr=lr,
            epochs=None,
            alpha_bending=10.0,
            alpha=args.alpha,
            ptv_isotv_weight=args.ptv_isotv_weight,
            stage_full_cps=args.stage_full_cps,
            stage_downsample=args.stage_downsample,
            stage_epochs=args.stage_epochs,
        )
        extra_args = ["--alpha", str(float(args.alpha)), "--batch-size", str(int(args.batch_size))]
        reg_value = float(args.alpha)
    else:
        raise ValueError(f"Unknown pilot: {pilot}")

    name = run_name(fake, method, lr)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--method",
        method,
        "--case-id",
        str(int(case_id)),
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(out_root),
        "--seed",
        str(int(args.seed)),
        "--lr",
        str(lr),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
        *extra_args,
    ]
    if args.force:
        cmd.append("--overwrite")
    return {
        "pilot": pilot,
        "method": method,
        "case_id": int(case_id),
        "reg_value": reg_value,
        "run_name": name,
        "cmd": cmd,
        "summary_path": out_root / method / name / "summary.json",
        "log_path": args.log_root / pilot / f"{name}.log",
    }


def main() -> None:
    args = parse_args()
    pilots = parse_pilots(args.pilots)
    cases = c.parse_int_list(args.cases)
    gpus = [str(v) for v in c.parse_int_list(args.gpus)]
    args.ptv_out_root.mkdir(parents=True, exist_ok=True)
    args.masked_out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    jobs = [build_job(args, pilot, case_id) for pilot in pilots for case_id in cases]
    queued = [job for job in jobs if args.force or not is_completed(Path(job["summary_path"]))]
    manifest = {
        "requested_jobs": len(jobs),
        "queued_jobs": len(queued),
        "skipped_completed": len(jobs) - len(queued),
        "pilots": pilots,
        "cases": cases,
        "gpus": gpus,
        "data_root": str(args.data_root),
        "ptv_out_root": str(args.ptv_out_root),
        "masked_out_root": str(args.masked_out_root),
        "log_root": str(args.log_root),
        "stage_full_cps": args.stage_full_cps,
        "stage_downsample": args.stage_downsample,
        "stage_epochs": args.stage_epochs,
        "ptv_isotv_weight": float(args.ptv_isotv_weight),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for idx, job in enumerate(queued):
            gpu = gpus[idx % len(gpus)]
            print(
                json.dumps(
                    {
                        "dry_run_job": job["run_name"],
                        "pilot": job["pilot"],
                        "method": job["method"],
                        "case_id": job["case_id"],
                        "reg_value": job["reg_value"],
                        "assigned_gpu": gpu,
                        "cmd": list(job["cmd"]) + ["--gpu", gpu],
                    }
                ),
                flush=True,
            )
        return

    pending = list(queued)
    active: List[Dict[str, object]] = []
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
                        "pilot": job["pilot"],
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
            item["log_file"].close()
            job = item["job"]
            elapsed = time.time() - float(item["start"])
            payload = {
                "finished": job["run_name"],
                "returncode": code,
                "elapsed_sec": elapsed,
                "gpu": item["gpu"],
                "pilot": job["pilot"],
                "case_id": job["case_id"],
            }
            print(json.dumps(payload), flush=True)
            if code != 0:
                failures.append(payload)
        active = still_active
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()
