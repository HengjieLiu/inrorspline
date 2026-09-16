#!/usr/bin/env python
"""Launch LungCT Figure 2 MR-D-BSCP 2500-step schedule refresh jobs."""

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

EXPERIMENTS_DIR = c.REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths
from run_one import (
    METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE,
    METHOD_DIRECT_CP_32_PTV,
    default_lr,
    run_name,
)


RUNNER = Path(__file__).resolve().parent / "run_one.py"
FAMILY_IDIR_FDBE = "idir_fdbe"
FAMILY_PTV = "ptv"
FAMILIES = [FAMILY_IDIR_FDBE, FAMILY_PTV]

DEFAULT_CASES = "1,2,3,4,5,6,7,8,9,10"
DEFAULT_ALPHAS = "1000,3000,10000,30000,100000"
DEFAULT_PTV_WEIGHTS = "0.011,0.033,0.055,0.11,0.33"
DEFAULT_STAGE_FULL_CPS = "32,16,8,4"
DEFAULT_STAGE_DOWNSAMPLE = "8,4,2,1"
DEFAULT_STAGE_EPOCHS = "400,400,800,900"
IDIR_RUN_KEY = "lungct_fig2_mrdbscp_idirloss_fdbe_e400_400_800_900_v1"
PTV_RUN_KEY = "lungct_fig2_mrdbscp_ptvloss_e400_400_800_900_v1"
LOG_EXPERIMENT_KEY = "2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/logs"
DEFAULT_IDIR_OUT_ROOT = None
DEFAULT_PTV_OUT_ROOT = None
DEFAULT_PTV_ANCHOR_ROOT = Path("/database/inrorspline/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1")
DEFAULT_LOG_ROOT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=str, default=",".join(FAMILIES), help=f"Comma list from {FAMILIES}")
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--ptv-weights", type=str, default=DEFAULT_PTV_WEIGHTS)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--idir-out-root", type=Path, default=None)
    parser.add_argument("--ptv-out-root", type=Path, default=None)
    parser.add_argument("--ptv-anchor-root", type=Path, default=DEFAULT_PTV_ANCHOR_ROOT)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stage-full-cps", type=str, default=DEFAULT_STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=DEFAULT_STAGE_DOWNSAMPLE)
    parser.add_argument("--stage-epochs", type=str, default=DEFAULT_STAGE_EPOCHS)
    parser.add_argument("--rerun-ptv-anchor", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve all generated outputs under the external artifact store."""

    args.idir_out_root = artifact_paths.resolve_run_root(
        IDIR_RUN_KEY, args.idir_out_root, repo_root=c.REPO_ROOT
    )
    args.ptv_out_root = artifact_paths.resolve_run_root(
        PTV_RUN_KEY, args.ptv_out_root, repo_root=c.REPO_ROOT
    )
    args.log_root = artifact_paths.resolve_review_root(
        LOG_EXPERIMENT_KEY, args.log_root, repo_root=c.REPO_ROOT
    )
    return args


def parse_families(value: str) -> List[str]:
    families = [v for v in value.replace(" ", "").split(",") if v]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"Unknown family value(s): {unknown}; expected subset of {FAMILIES}")
    if not families:
        raise ValueError("At least one family is required.")
    return families


def parse_float_list(value: str) -> List[float]:
    vals = [float(v) for v in value.replace(" ", "").split(",") if v]
    if not vals:
        raise ValueError(f"Expected comma-separated numeric values, got {value!r}")
    return vals


def is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "completed"


def fake_args(args: argparse.Namespace, method: str, case_id: int, reg_value: float) -> argparse.Namespace:
    lr = float(args.lr if args.lr is not None else default_lr(method))
    return argparse.Namespace(
        run_name="",
        method=method,
        dataset="4dct",
        case_id=int(case_id),
        seed=int(args.seed),
        lr=lr,
        epochs=None,
        alpha=float(reg_value) if method == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE else 10.0,
        alpha_bending=10.0,
        ptv_isotv_weight=float(reg_value) if method == METHOD_DIRECT_CP_32_PTV else 0.11,
        stage_full_cps=args.stage_full_cps,
        stage_downsample=args.stage_downsample,
        stage_epochs=args.stage_epochs,
    )


def build_idir_job(args: argparse.Namespace, case_id: int, alpha: float) -> Dict[str, object]:
    method = METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE
    lr = float(args.lr if args.lr is not None else default_lr(method))
    fake = fake_args(args, method, case_id, alpha)
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
        str(args.idir_out_root),
        "--seed",
        str(int(args.seed)),
        "--lr",
        str(lr),
        "--alpha",
        str(float(alpha)),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
    ]
    if args.force:
        cmd.append("--overwrite")
    return {
        "family": FAMILY_IDIR_FDBE,
        "method": method,
        "case_id": int(case_id),
        "reg_name": "alpha",
        "reg_value": float(alpha),
        "run_name": name,
        "cmd": cmd,
        "summary_path": args.idir_out_root / method / name / "summary.json",
        "log_path": args.log_root / FAMILY_IDIR_FDBE / f"{name}.log",
        "reused_anchor": False,
    }


def build_ptv_job(args: argparse.Namespace, case_id: int, weight: float) -> Dict[str, object]:
    method = METHOD_DIRECT_CP_32_PTV
    lr = float(args.lr if args.lr is not None else default_lr(method))
    fake = fake_args(args, method, case_id, weight)
    name = run_name(fake, method, lr)
    anchor_summary = args.ptv_anchor_root / method / name / "summary.json"
    reuse_anchor = abs(float(weight) - 0.11) <= 1e-12 and not args.rerun_ptv_anchor and is_completed(anchor_summary)
    out_root = args.ptv_anchor_root if reuse_anchor else args.ptv_out_root
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
        str(args.ptv_out_root),
        "--seed",
        str(int(args.seed)),
        "--lr",
        str(lr),
        "--ptv-isotv-weight",
        str(float(weight)),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--stage-epochs",
        str(args.stage_epochs),
    ]
    if args.force:
        cmd.append("--overwrite")
    return {
        "family": FAMILY_PTV,
        "method": method,
        "case_id": int(case_id),
        "reg_name": "ptv_isotv_weight",
        "reg_value": float(weight),
        "run_name": name,
        "cmd": cmd,
        "summary_path": out_root / method / name / "summary.json",
        "log_path": args.log_root / FAMILY_PTV / f"{name}.log",
        "reused_anchor": reuse_anchor,
    }


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    cases = c.parse_int_list(args.cases)
    families = parse_families(args.families)
    if FAMILY_IDIR_FDBE in families:
        jobs.extend(build_idir_job(args, case_id, alpha) for alpha in parse_float_list(args.alphas) for case_id in cases)
    if FAMILY_PTV in families:
        jobs.extend(build_ptv_job(args, case_id, weight) for weight in parse_float_list(args.ptv_weights) for case_id in cases)
    return jobs


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    gpus = [str(v) for v in c.parse_int_list(args.gpus)]
    args.idir_out_root.mkdir(parents=True, exist_ok=True)
    args.ptv_out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    queued = [
        job
        for job in jobs
        if not bool(job["reused_anchor"]) and (args.force or not is_completed(Path(job["summary_path"])))
    ]
    manifest = {
        "requested_jobs": len(jobs),
        "queued_jobs": len(queued),
        "skipped_completed_or_reused": len(jobs) - len(queued),
        "reused_ptv_anchor_jobs": sum(1 for job in jobs if bool(job["reused_anchor"])),
        "families": parse_families(args.families),
        "cases": c.parse_int_list(args.cases),
        "alphas": parse_float_list(args.alphas),
        "ptv_weights": parse_float_list(args.ptv_weights),
        "gpus": gpus,
        "data_root": str(args.data_root),
        "idir_out_root": str(args.idir_out_root),
        "ptv_out_root": str(args.ptv_out_root),
        "ptv_anchor_root": str(args.ptv_anchor_root),
        "log_root": str(args.log_root),
        "stage_full_cps": args.stage_full_cps,
        "stage_downsample": args.stage_downsample,
        "stage_epochs": args.stage_epochs,
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
                        "family": job["family"],
                        "method": job["method"],
                        "case_id": job["case_id"],
                        "reg_name": job["reg_name"],
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
                        "family": job["family"],
                        "case_id": job["case_id"],
                        "reg_name": job["reg_name"],
                        "reg_value": job["reg_value"],
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
                "family": job["family"],
                "case_id": job["case_id"],
                "reg_name": job["reg_name"],
                "reg_value": job["reg_value"],
            }
            print(json.dumps(payload), flush=True)
            if code != 0:
                failures.append(payload)
        active = still_active
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()
