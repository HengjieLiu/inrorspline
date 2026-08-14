#!/usr/bin/env python
"""Launch DIR-LAB Panel-A runs with paired full-crop/lung-mask NCC curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/2026-08-14_dirlab_case01_case08_lungmask_diagnostics"
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1")
DEFAULT_DATA_ROOT = Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm")
DEFAULT_CASES = tuple(range(1, 11))
METHODS = [
    {
        "slug": "idir",
        "label": "INR-Dense",
        "impl": "idir_be_default_lungmask",
        "lr": "1e-4",
        "alpha_bending": "3",
        "epochs": "2500",
    },
    {
        "slug": "inr_bscp",
        "label": "INR-BSCP",
        "impl": "sinr_cps4_global_ncc_fd_be",
        "lr": "1e-4",
        "alpha": "10000",
        "epochs": "2500",
    },
    {
        "slug": "d_bscp",
        "label": "D-BSCP",
        "impl": "direct_cp_cps4_global_ncc_fd_be",
        "lr": "1e-3",
        "alpha": "10000",
        "epochs": "2500",
    },
    {
        "slug": "mr_d_bscp",
        "label": "MR-D-BSCP",
        "impl": "direct_cp_32_16_8_4_global_ncc_fd_be",
        "lr": "1e-3",
        "alpha": "10000",
        "stage_full_cps": "32,16,8,4",
        "stage_downsample": "8,4,2,1",
        "stage_epochs": "400,400,800,900",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--cases",
        default=",".join(str(case_id) for case_id in DEFAULT_CASES),
        help="Comma-separated DIR-LAB case ids, e.g. 1,8 or 1,2,3,4,5,6,7,8,9,10.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_gpus(value: str) -> list[str]:
    gpus = [token.strip() for token in value.split(",") if token.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def parse_cases(value: str) -> tuple[int, ...]:
    cases: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        case_id = int(token)
        if case_id < 1 or case_id > 10:
            raise ValueError(f"DIR-LAB case id out of range: {case_id}")
        cases.append(case_id)
    unique_cases = tuple(dict.fromkeys(cases))
    if not unique_cases:
        raise ValueError("--cases cannot be empty")
    return unique_cases


def run_name(case_id: int, spec: dict[str, str]) -> str:
    if spec["slug"] == "mr_d_bscp":
        return (
            f"lungmask_curves_case{case_id:02d}_panelA_{spec['slug']}_"
            "sched32-16-8-4_ds8-4-2-1_e400-400-800-900_a10000_lr0p001_seed1_diag10"
        )
    lr_tag = spec["lr"].replace("-", "m").replace(".", "p")
    return f"lungmask_curves_case{case_id:02d}_panelA_{spec['slug']}_e2500_lr{lr_tag}_seed1_diag10"


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case_id in parse_cases(args.cases):
        for spec in METHODS:
            name = run_name(case_id, spec)
            run_dir = args.run_root / spec["impl"] / name
            command = [
                sys.executable,
                str(RUNNER),
                "--method",
                spec["impl"],
                "--case-id",
                str(case_id),
                "--data-root",
                str(args.data_root),
                "--out-root",
                str(args.run_root),
                "--run-name",
                name,
                "--gpu",
                "__GPU__",
                "--seed",
                "1",
                "--lr",
                spec["lr"],
                "--eval-interval",
                "250",
                "--global-ncc-diagnostic-interval",
                "10",
            ]
            if spec["slug"] == "idir":
                command.extend(["--epochs", spec["epochs"], "--alpha-bending", spec["alpha_bending"]])
            elif spec["slug"] == "mr_d_bscp":
                command.extend(
                    [
                        "--alpha",
                        spec["alpha"],
                        "--stage-full-cps",
                        spec["stage_full_cps"],
                        "--stage-downsample",
                        spec["stage_downsample"],
                        "--stage-epochs",
                        spec["stage_epochs"],
                    ]
                )
            else:
                command.extend(["--epochs", spec["epochs"], "--alpha", spec["alpha"]])
            jobs.append(
                {
                    "job_id": f"case{case_id:02d}_panelA_{spec['slug']}",
                    "case_key": f"case{case_id:02d}",
                    "case_id": case_id,
                    "method_slug": spec["slug"],
                    "display_method": spec["label"],
                    "implementation_id": spec["impl"],
                    "run_name": name,
                    "run_dir": str(run_dir),
                    "summary_path": str(run_dir / "summary.json"),
                    "curve_path": str(run_dir / "global_ncc_diagnostic.csv"),
                    "log_path": str(args.run_root / "logs" / f"{name}.log"),
                    "command": command,
                    "status": "pending",
                    "gpu": "",
                    "pid": "",
                    "elapsed_sec": "",
                    "returncode": "",
                    "failure_note": "",
                }
            )
    return jobs


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def compatible_complete(job: dict[str, Any]) -> tuple[bool, str]:
    summary_path = Path(job["summary_path"])
    curve_path = Path(job["curve_path"])
    summary = load_json(summary_path)
    if summary.get("status") != "completed":
        return False, "summary_missing_or_not_completed"
    training = summary.get("training")
    if not isinstance(training, dict):
        return False, "training_missing"
    if int(training.get("global_ncc_diagnostic_interval", -1)) != 10:
        return False, "diagnostic_interval_not_10"
    if training.get("lung_mask_ncc_diagnostic_name") != "fixed_t00_lung_mask_global_ncc":
        return False, "lung_mask_diagnostic_missing"
    if job["method_slug"] == "mr_d_bscp":
        if training.get("stage_epochs") != [400, 400, 800, 900]:
            return False, "mrdbscp_stage_epochs_mismatch"
        if training.get("stage_full_cps") != [32, 16, 8, 4]:
            return False, "mrdbscp_cps_mismatch"
        if training.get("stage_downsample") != [8, 4, 2, 1]:
            return False, "mrdbscp_downsample_mismatch"
    if not curve_path.is_file():
        return False, "curve_missing"
    with curve_path.open() as handle:
        header = handle.readline().strip().split(",")
    for required in ("global_ncc_value", "lung_mask_ncc_value", "lung_mask_one_minus_ncc"):
        if required not in header:
            return False, f"{required}_missing"
    return True, ""


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (list, dict)):
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_state(output_dir: Path, jobs: Sequence[dict[str, Any]], started_at: float) -> None:
    counts = {status: sum(job["status"] == status for job in jobs) for status in sorted({job["status"] for job in jobs})}
    payload = {
        "updated_at_utc": utc_text(),
        "elapsed_sec": time.time() - started_at,
        "counts": counts,
        "jobs": jobs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lungmask_before_after_curves_launcher_state.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    lines = [
        "# Lung-mask before/after curve launcher state",
        "",
        f"- Updated: `{payload['updated_at_utc']}`",
        f"- Elapsed sec: `{payload['elapsed_sec']:.1f}`",
        f"- Counts: `{counts}`",
        "",
        "| Job | Status | GPU | Elapsed sec | Note |",
        "|---|---|---:|---:|---|",
    ]
    for job in jobs:
        lines.append(
            f"| `{job['job_id']}` | `{job['status']}` | `{job.get('gpu', '')}` | "
            f"`{job.get('elapsed_sec', '')}` | `{job.get('failure_note', '')}` |"
        )
    (output_dir / "lungmask_before_after_curves_monitor.md").write_text("\n".join(lines) + "\n")
    write_csv(output_dir / "lungmask_before_after_curves_job_manifest.csv", jobs)


def launch_job(job: dict[str, Any], gpu: str) -> subprocess.Popen[str]:
    command = [gpu if token == "__GPU__" else token for token in job["command"]]
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    job["gpu"] = gpu
    job["status"] = "running"
    job["started_at"] = time.time()
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    job["pid"] = process.pid
    job["_log_handle"] = handle
    job["_process"] = process
    return process


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    jobs = build_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for job in jobs:
            ok, note = compatible_complete(job)
            if ok:
                job["status"] = "completed_existing"
                job["failure_note"] = "exact compatible existing output available"
            else:
                job["failure_note"] = note
        write_state(args.output_dir, jobs, time.time())
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "jobs": len(jobs),
                    "cases": list(parse_cases(args.cases)),
                    "completed_existing": sum(job["status"] == "completed_existing" for job in jobs),
                    "pending": sum(job["status"] == "pending" for job in jobs),
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
        return
    started_at = time.time()
    pending = []
    for job in jobs:
        ok, note = compatible_complete(job)
        if ok and args.resume:
            job["status"] = "completed_existing"
            job["failure_note"] = ""
        else:
            if ok and not args.resume:
                job["status"] = "completed_existing"
                job["failure_note"] = "exact compatible existing output reused"
            else:
                job["status"] = "pending"
                job["failure_note"] = note
                pending.append(job)
    running: dict[str, dict[str, Any]] = {}
    idle = list(gpus)
    write_state(args.output_dir, jobs, started_at)
    while pending or running:
        while pending and idle:
            gpu = idle.pop(0)
            job = pending.pop(0)
            launch_job(job, gpu)
            running[gpu] = job
        write_state(args.output_dir, jobs, started_at)
        time.sleep(max(1, int(args.poll_sec)))
        for gpu, job in list(running.items()):
            process = job["_process"]
            rc = process.poll()
            if rc is None:
                continue
            job["_log_handle"].close()
            job["returncode"] = rc
            job["elapsed_sec"] = f"{time.time() - float(job['started_at']):.3f}"
            ok, note = compatible_complete(job)
            if rc == 0 and ok:
                job["status"] = "completed"
                job["failure_note"] = ""
            else:
                job["status"] = "failed"
                job["failure_note"] = note or f"returncode={rc}"
            del job["_process"]
            del job["_log_handle"]
            del running[gpu]
            idle.append(gpu)
        if any(job["status"] == "failed" for job in jobs):
            write_state(args.output_dir, jobs, started_at)
            raise SystemExit(2)
    write_state(args.output_dir, jobs, started_at)
    print(json.dumps({"completed": len(jobs), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
