#!/usr/bin/env python
"""Launch COPD DirectCP32 pTV-LCC + isoTV sanity jobs across visible GPUs."""

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
from run_one import METHOD_DIRECT_CP_32_PTV, default_lr, run_name


RUNNER = Path(__file__).resolve().parent / "run_one.py"
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/copd_directcp32_ptvlcc_w0p11_sanity_v1")
DEFAULT_LOG_ROOT = c.REPO_ROOT / "logs/copd_directcp32_ptvlcc_w0p11_sanity_v1"
DEFAULT_CASES = "1,2,3,4,5,6,7,8,9,10"
DEFAULT_STAGE_EPOCHS = "200,400,800,1000"
DEFAULT_STAGE_CPS = "32,16,8,4"
DEFAULT_STAGE_DOWNSAMPLE = "8,4,2,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stage-full-cps", type=str, default=DEFAULT_STAGE_CPS)
    parser.add_argument("--stage-downsample", type=str, default=DEFAULT_STAGE_DOWNSAMPLE)
    parser.add_argument("--stage-epochs", type=str, default=DEFAULT_STAGE_EPOCHS)
    parser.add_argument("--ptv-lcc-sigma-mm", type=float, default=2.1)
    parser.add_argument("--ptv-isotv-weight", type=float, default=0.11)
    parser.add_argument("--ptv-csqrt", type=float, default=5e-3)
    parser.add_argument("--border-mask-voxels", type=int, default=5)
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
    method = METHOD_DIRECT_CP_32_PTV
    lr = float(args.lr if args.lr is not None else default_lr(method))
    fake = argparse.Namespace(
        run_name="",
        dataset="copd",
        method=method,
        case_id=case_id,
        seed=args.seed,
        lr=lr,
        epochs=None,
        alpha_bending=10.0,
        alpha=100.0,
        ptv_isotv_weight=args.ptv_isotv_weight,
        stage_full_cps=args.stage_full_cps,
        stage_downsample=args.stage_downsample,
        stage_epochs=args.stage_epochs,
    )
    name = run_name(fake, method, lr)
    summary_path = args.out_root / method / name / "summary.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--dataset",
        "copd",
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
        "--lr",
        str(lr),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
        "--ptv-lcc-sigma-mm",
        str(args.ptv_lcc_sigma_mm),
        "--ptv-isotv-weight",
        str(args.ptv_isotv_weight),
        "--ptv-csqrt",
        str(args.ptv_csqrt),
        "--border-mask-voxels",
        str(args.border_mask_voxels),
        "--save-dense-dvf",
    ]
    if args.force:
        cmd.append("--overwrite")
    return {
        "method": method,
        "case_id": int(case_id),
        "run_name": name,
        "cmd": cmd,
        "summary_path": summary_path,
        "log_path": args.log_root / f"{name}.log",
    }


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    return [build_job(args, int(case_id)) for case_id in c.parse_int_list(args.cases)]


def main() -> None:
    args = parse_args()
    gpus = [str(v) for v in c.parse_int_list(args.gpus)]
    if not gpus:
        raise SystemExit("At least one GPU is required.")
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
        "dataset": "copd",
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "method": METHOD_DIRECT_CP_32_PTV,
        "cases": c.parse_int_list(args.cases),
        "stage_full_cps": args.stage_full_cps,
        "stage_downsample": args.stage_downsample,
        "stage_epochs": args.stage_epochs,
        "ptv_lcc_sigma_mm": float(args.ptv_lcc_sigma_mm),
        "ptv_isotv_weight": float(args.ptv_isotv_weight),
        "ptv_csqrt": float(args.ptv_csqrt),
        "border_mask_voxels": int(args.border_mask_voxels),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for idx, job in enumerate(queued):
            gpu = gpus[idx % len(gpus)]
            print(
                json.dumps(
                    {
                        "dry_run_job": job["run_name"],
                        "case_id": job["case_id"],
                        "assigned_gpu": gpu,
                        "cmd": list(job["cmd"]) + ["--gpu", gpu],
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
                "returncode": code,
                "elapsed_sec": elapsed,
                "gpu": item["gpu"],
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
