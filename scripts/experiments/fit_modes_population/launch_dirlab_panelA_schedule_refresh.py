#!/usr/bin/env python
"""Launch the DIR-LAB Panel-A MR-D-BSCP 2500-epoch diagnostic refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


RUNNER = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"
COMMON = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/common.py"
METHOD = "direct_cp_32_16_8_4_global_ncc_fd_be"
EXPERIMENT_ID = "fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
EXPERIMENT_KEY = "2026-08-13_fit_modes_dirlab_panelA_schedule_refresh"
DEFAULT_DATA_ROOT = Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm")
DEFAULT_CASES = tuple(range(1, 11))
SMOKE_CASES = (2, 8)
FULL_STAGE_EPOCHS = (400, 400, 800, 900)
SMOKE_STAGE_EPOCHS = (1, 1, 1, 1)
STAGE_FULL_CPS = (32, 16, 8, 4)
STAGE_DOWNSAMPLE = (8, 4, 2, 1)
HISTORICAL_MEDIAN_RUNTIME_SEC = 308.9180957078934
TERMINAL_SUCCESS = {"completed", "completed_existing"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--smoke-run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cases", type=str, default=",".join(map(str, DEFAULT_CASES)))
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-free-gpus", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve generated logs, run data, and review reports below ARTIFACT_ROOT."""

    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    args.run_root = artifact_paths.resolve_run_root(
        EXPERIMENT_ID,
        args.run_root,
        repo_root=REPO_ROOT,
    )
    args.smoke_run_root = artifact_paths.resolve_run_root(
        f"{EXPERIMENT_ID}_smoke",
        args.smoke_run_root,
        repo_root=REPO_ROOT,
    )
    return args


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_ints(value: str) -> list[int]:
    values = parse_int_sequence(value)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected a nonempty unique integer list, got {value!r}")
    return values


def parse_int_sequence(value: str) -> list[int]:
    values = [int(token) for token in value.replace(" ", "").split(",") if token]
    if not values:
        raise ValueError(f"Expected a nonempty integer list, got {value!r}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    command = ["git", *args]
    repo_stat = REPO_ROOT.stat()
    if os.geteuid() == 0 and repo_stat.st_uid != 0:
        command = [
            "setpriv",
            f"--reuid={repo_stat.st_uid}",
            f"--regid={repo_stat.st_gid}",
            "--clear-groups",
            *command,
        ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def portable_path(path: Path, run_root: Path, data_root: Path) -> str:
    text = str(path)
    replacements = [
        (str(run_root), "${RUN_ROOT}"),
        (str(data_root), "${DIRLAB_PTVREG_ISO1MM_ROOT}"),
        (str(REPO_ROOT), "${REPO_ROOT}"),
    ]
    for root, token in replacements:
        if text == root or text.startswith(root + "/"):
            return text.replace(root, token, 1)
    return text


def portable_command(command: Sequence[str], run_root: Path, data_root: Path) -> list[str]:
    return [portable_path(Path(value), run_root, data_root) if value.startswith("/") else value for value in command]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_name(case_id: int, smoke: bool) -> str:
    epochs = SMOKE_STAGE_EPOCHS if smoke else FULL_STAGE_EPOCHS
    diagnostic = 1 if smoke else 10
    epoch_tag = "-".join(map(str, epochs))
    return (
        f"fit_modes_dirlab_case{case_id:02d}_panelA_mrdbscp_"
        f"sched32-16-8-4_ds8-4-2-1_e{epoch_tag}_a10000_lr0p001_"
        f"seed1_globalnccdiag{diagnostic}"
    )


def build_job(case_id: int, run_root: Path, data_root: Path, smoke: bool) -> dict[str, Any]:
    epochs = SMOKE_STAGE_EPOCHS if smoke else FULL_STAGE_EPOCHS
    diagnostic = 1 if smoke else 10
    name = run_name(case_id, smoke)
    run_dir = run_root / METHOD / name
    command = [
        sys.executable,
        str(RUNNER),
        "--method",
        METHOD,
        "--case-id",
        str(case_id),
        "--data-root",
        str(data_root),
        "--out-root",
        str(run_root),
        "--run-name",
        name,
        "--seed",
        "1",
        "--lr",
        "0.001",
        "--alpha",
        "10000",
        "--stage-full-cps",
        ",".join(map(str, STAGE_FULL_CPS)),
        "--stage-downsample",
        ",".join(map(str, STAGE_DOWNSAMPLE)),
        "--stage-epochs",
        ",".join(map(str, epochs)),
        "--eval-interval",
        "1" if smoke else "250",
        "--global-ncc-diagnostic-interval",
        str(diagnostic),
    ]
    if smoke:
        command.append("--no-save-dense-dvf")
    return {
        "job_id": f"dirlab_case{case_id:02d}_panelA_mr_d_bscp_schedule_refresh",
        "case_id": case_id,
        "method": METHOD,
        "objective": "image_registration_global_ncc_plus_dense_fd_be",
        "lr": 0.001,
        "alpha": 10000.0,
        "seed": 1,
        "stage_full_cps": ",".join(map(str, STAGE_FULL_CPS)),
        "stage_downsample": ",".join(map(str, STAGE_DOWNSAMPLE)),
        "stage_epochs": ",".join(map(str, epochs)),
        "total_epochs": sum(epochs),
        "diagnostic_interval": diagnostic,
        "run_name": name,
        "run_dir": run_dir,
        "summary_path": run_dir / "summary.json",
        "curve_path": run_dir / "global_ncc_diagnostic.csv",
        "log_path": run_root / "logs" / f"{name}.log",
        "command": command,
        "status": "pending",
        "gpu": "",
        "pid": "",
        "attempts": 0,
        "elapsed_sec": "",
        "returncode": "",
        "failure_note": "",
    }


def completed_compatible(job: dict[str, Any]) -> tuple[bool, str]:
    summary_path = Path(job["summary_path"])
    curve_path = Path(job["curve_path"])
    payload = load_json(summary_path)
    if payload.get("status") != "completed":
        return False, "summary_missing_or_not_completed"
    training = payload.get("training")
    if not isinstance(training, dict):
        return False, "training_metadata_missing"
    expected = {
        "method": payload.get("method") == METHOD,
        "case": int(payload.get("case_id", -1)) == int(job["case_id"]),
        "stage_full_cps": training.get("stage_full_cps") == parse_ints(str(job["stage_full_cps"])),
        "stage_downsample": training.get("stage_downsample") == parse_ints(str(job["stage_downsample"])),
        "stage_epochs": training.get("stage_epochs")
        == parse_int_sequence(str(job["stage_epochs"])),
        "lr": all(abs(float(value) - 0.001) < 1e-12 for value in training.get("stage_lr", [])),
        "alpha": abs(float(training.get("alpha", math.nan)) - 10000.0) < 1e-8,
        "seed": int(training.get("seed", -1)) == 1,
        "diagnostic_interval": int(training.get("global_ncc_diagnostic_interval", -1))
        == int(job["diagnostic_interval"]),
        "curve": curve_path.is_file(),
    }
    failed = [name for name, passed in expected.items() if not passed]
    return not failed, "" if not failed else "incompatible:" + ",".join(failed)


def gpu_snapshot(gpus: Sequence[int]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    wanted = set(gpus)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        index = int(fields[0])
        if index in wanted:
            rows.append(
                {
                    "index": index,
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_pct": int(fields[4]),
                }
            )
    if sorted(row["index"] for row in rows) != sorted(wanted):
        raise RuntimeError(f"Could not resolve requested GPUs {sorted(wanted)}")
    return rows


def require_free(snapshot: Sequence[dict[str, Any]]) -> None:
    # This server retains roughly 1.1 GiB per display-attached GPU with no
    # compute process. Treat that measured baseline as idle while still
    # rejecting a material allocation or sustained utilization.
    busy = [
        row
        for row in snapshot
        if int(row["memory_used_mib"]) > 2048 or int(row["utilization_pct"]) > 20
    ]
    if busy:
        raise RuntimeError(f"Requested GPUs are not free: {busy}")


def public_job(job: dict[str, Any], run_root: Path, data_root: Path) -> dict[str, Any]:
    row = dict(job)
    for key in ["run_dir", "summary_path", "curve_path", "log_path"]:
        row[key] = portable_path(Path(row[key]), run_root, data_root)
    row["command"] = portable_command(row["command"], run_root, data_root)
    return row


def write_job_manifests(
    review_root: Path, jobs: Sequence[dict[str, Any]], run_root: Path, data_root: Path
) -> None:
    public = [public_job(job, run_root, data_root) for job in jobs]
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "job_manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    csv_rows = []
    for row in public:
        flat = dict(row)
        flat["command"] = json.dumps(flat["command"])
        csv_rows.append(flat)
    write_csv(review_root / "job_manifest.csv", csv_rows)


def provenance(run_root: Path, data_root: Path, snapshot: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "experiment_id": EXPERIMENT_ID,
        "repo_commit": git_output("rev-parse", "HEAD"),
        "repo_branch": git_output("branch", "--show-current"),
        "dirty_paths": git_output("status", "--short").splitlines(),
        "source_sha256": {
            "runner": sha256(RUNNER),
            "common": sha256(COMMON),
            "launcher": sha256(Path(__file__).resolve()),
        },
        "container_name": "py310sitreg_v1_test1",
        "container_image": "hengjieliu96/dl_dir:py310sitreg_v1",
        "dataset_protocol": "ptv_landmark_crop_1mm; DIR-LAB T50->T00",
        "data_root": portable_path(data_root, run_root, data_root),
        "run_root": "${RUN_ROOT}",
        "gpus": list(snapshot),
    }


def eta_seconds(jobs: Sequence[dict[str, Any]], active_gpu_count: int) -> float:
    remaining = sum(job["status"] in {"pending", "running"} for job in jobs)
    if remaining == 0:
        return 0.0
    observed = [
        float(job["elapsed_sec"])
        for job in jobs
        if job["status"] in TERMINAL_SUCCESS and job["elapsed_sec"] not in {"", None}
    ]
    per_job = statistics.median(observed) if observed else HISTORICAL_MEDIAN_RUNTIME_SEC
    return float(per_job * math.ceil(remaining / max(active_gpu_count, 1)))


def write_monitor(
    review_root: Path,
    jobs: Sequence[dict[str, Any]],
    started_at: datetime,
    gpus: Sequence[int],
) -> None:
    counts = {
        status: sum(job["status"] == status for job in jobs)
        for status in ["completed", "completed_existing", "running", "pending", "failed"]
    }
    eta = eta_seconds(jobs, len(gpus))
    payload = {
        "updated_at": iso_now(),
        "elapsed_sec": (utc_now() - started_at).total_seconds(),
        "counts": counts,
        "active_jobs": {
            str(job["gpu"]): job["job_id"] for job in jobs if job["status"] == "running"
        },
        "eta_sec": eta,
        "estimated_completion": (utc_now() + timedelta(seconds=eta)).strftime(
            "%Y-%m-%d %H:%M:%S +0000"
        ),
        "recent_failures": [
            {"job_id": job["job_id"], "note": job["failure_note"]}
            for job in jobs
            if job["status"] == "failed"
        ],
    }
    (review_root / "monitor_status.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Schedule Refresh Monitor",
        "",
        f"Updated: `{payload['updated_at']}`",
        f"Elapsed: `{payload['elapsed_sec'] / 60.0:.1f} min`",
        f"ETA: `{eta / 60.0:.1f} min` (`{payload['estimated_completion']}`)",
        "",
        "| Completed | Reused | Running | Pending | Failed |",
        "|---:|---:|---:|---:|---:|",
        f"| {counts['completed']} | {counts['completed_existing']} | {counts['running']} | {counts['pending']} | {counts['failed']} |",
        "",
        "## Active GPUs",
        "",
    ]
    lines.extend(
        [f"- GPU `{gpu}`: `{payload['active_jobs'].get(str(gpu), 'idle')}`" for gpu in gpus]
    )
    if payload["recent_failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{row['job_id']}`: {row['note']}" for row in payload["recent_failures"]
        )
    (review_root / "monitor_status.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    gpus = parse_ints(args.gpus)
    if gpus != [0, 1]:
        raise ValueError(f"This locked refresh requires GPUs 0,1; got {gpus}")
    if args.poll_sec < 5 or args.poll_sec > 60:
        raise ValueError("--poll-sec must be between 5 and 60 seconds")
    cases = list(SMOKE_CASES) if args.smoke else parse_ints(args.cases)
    if not args.smoke and cases != list(DEFAULT_CASES):
        raise ValueError(f"Full refresh requires cases 1-10 in order; got {cases}")
    review_root = Path(args.review_root) / "smoke" if args.smoke else Path(args.review_root)
    run_root = Path(args.smoke_run_root) if args.smoke else Path(args.run_root)
    data_root = Path(args.data_root)
    jobs = [build_job(case_id, run_root, data_root, args.smoke) for case_id in cases]
    if len(jobs) != (2 if args.smoke else 10):
        raise RuntimeError(f"Unexpected job count: {len(jobs)}")

    snapshot = gpu_snapshot(gpus)
    if args.require_free_gpus and not args.dry_run:
        require_free(snapshot)
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "provenance.json").write_text(
        json.dumps(provenance(run_root, data_root, snapshot), indent=2) + "\n"
    )

    for job in jobs:
        run_dir = Path(job["run_dir"])
        if not run_dir.exists():
            continue
        compatible, note = completed_compatible(job)
        if compatible and args.resume:
            job["status"] = "completed_existing"
            job["failure_note"] = "exact-compatible completed output"
        elif compatible:
            raise RuntimeError(
                f"Completed output exists for {job['job_id']}; use --resume to reuse it"
            )
        else:
            raise RuntimeError(
                f"Non-compatible or partial output exists for {job['job_id']}: {note}. "
                "Use a new versioned run root; this launcher will not overwrite it."
            )
    write_job_manifests(review_root, jobs, run_root, data_root)
    started_at = utc_now()
    write_monitor(review_root, jobs, started_at, gpus)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "smoke" if args.smoke else "full",
                    "jobs": len(jobs),
                    "pending": sum(job["status"] == "pending" for job in jobs),
                    "completed_existing": sum(
                        job["status"] == "completed_existing" for job in jobs
                    ),
                    "gpus": gpus,
                    "stage_epochs": jobs[0]["stage_epochs"],
                    "diagnostic_interval": jobs[0]["diagnostic_interval"],
                },
                indent=2,
            )
        )
        return

    pending = [job for job in jobs if job["status"] == "pending"]
    active: list[dict[str, Any]] = []
    while pending or active:
        while pending and len(active) < len(gpus):
            used = {int(item["job"]["gpu"]) for item in active}
            gpu = next(value for value in gpus if value not in used)
            job = pending.pop(0)
            command = [*job["command"], "--gpu", str(gpu)]
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job.update(
                {
                    "status": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "attempts": int(job["attempts"]) + 1,
                }
            )
            active.append(
                {
                    "job": job,
                    "process": process,
                    "log_handle": log_handle,
                    "started": time.monotonic(),
                }
            )
            print(f"launched {job['job_id']} on GPU {gpu} pid={process.pid}", flush=True)

        time.sleep(args.poll_sec)
        for item in list(active):
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            item["log_handle"].close()
            job = item["job"]
            job["elapsed_sec"] = time.monotonic() - item["started"]
            job["returncode"] = returncode
            compatible, note = completed_compatible(job)
            if returncode == 0 and compatible:
                job["status"] = "completed"
                print(
                    f"completed {job['job_id']} GPU {job['gpu']} in {float(job['elapsed_sec']) / 60.0:.1f} min",
                    flush=True,
                )
            else:
                job["status"] = "failed"
                job["failure_note"] = f"returncode={returncode}; {note}"
                print(f"FAILED {job['job_id']}: {job['failure_note']}", flush=True)
            active.remove(item)
        write_job_manifests(review_root, jobs, run_root, data_root)
        write_monitor(review_root, jobs, started_at, gpus)

    state = {
        "finished_at": iso_now(),
        "mode": "smoke" if args.smoke else "full",
        "status_counts": {
            status: sum(job["status"] == status for job in jobs)
            for status in sorted({str(job["status"]) for job in jobs})
        },
        "accepted_terminal_records": sum(job["status"] in TERMINAL_SUCCESS for job in jobs),
        "expected_records": len(jobs),
    }
    (review_root / "launcher_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2), flush=True)
    if state["accepted_terminal_records"] != state["expected_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
