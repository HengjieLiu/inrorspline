#!/usr/bin/env python
"""Launch the LungCT case01/case08 masked-NCC FD-BE quick A/B."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


RUNNER = THIS_DIR / "run_one.py"
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_v1")
DEFAULT_DATA_ROOT = Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm")
DEFAULT_REVIEW_DIR = REPO_ROOT / "experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick"


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    impl: str
    lr: float
    alpha: float
    gpu: str
    epochs: int | None = None
    stage_full_cps: str | None = None
    stage_downsample: str | None = None
    stage_epochs: str | None = None


METHODS = (
    MethodSpec(
        slug="inr_bscp",
        label="INR-BSCP",
        impl="sinr_cps4_masked_ncc_fd_be",
        lr=1.0e-4,
        alpha=10000.0,
        epochs=2500,
        gpu="1",
    ),
    MethodSpec(
        slug="d_bscp",
        label="D-BSCP",
        impl="direct_cp_cps4_masked_ncc_fd_be",
        lr=1.0e-3,
        alpha=10000.0,
        epochs=2500,
        gpu="2",
    ),
    MethodSpec(
        slug="mr_d_bscp",
        label="MR-D-BSCP",
        impl="direct_cp_32_16_8_4_masked_ncc_fd_be",
        lr=1.0e-3,
        alpha=10000.0,
        gpu="3",
        stage_full_cps="32,16,8,4",
        stage_downsample="8,4,2,1",
        stage_epochs="400,400,800,900",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cases", default="1,8")
    parser.add_argument(
        "--gpus",
        default="1,2,3",
        help="Comma-separated GPUs assigned to SINR, D-BSCP, MR-D-BSCP in that order.",
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--gpu-check-timeout-sec", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_cases(text: str) -> tuple[int, ...]:
    cases: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        case_id = int(token)
        if case_id < 1 or case_id > 10:
            raise ValueError(f"DIR-Lab case id out of range: {case_id}")
        cases.append(case_id)
    result = tuple(dict.fromkeys(cases))
    if not result:
        raise ValueError("--cases cannot be empty")
    return result


def parse_gpus(text: str) -> tuple[str, str, str]:
    gpus = tuple(token.strip() for token in str(text).split(",") if token.strip())
    if len(gpus) != len(METHODS):
        raise ValueError(f"Expected {len(METHODS)} GPUs, got {gpus}")
    return gpus  # type: ignore[return-value]


def method_specs_for_gpus(gpus: Sequence[str]) -> tuple[MethodSpec, ...]:
    return tuple(
        MethodSpec(**{**asdict(spec), "gpu": str(gpu)})
        for spec, gpu in zip(METHODS, gpus)
    )


def run_name(case_id: int, spec: MethodSpec) -> str:
    case_token = f"case{case_id:02d}"
    if spec.stage_epochs is not None:
        stage_full = str(spec.stage_full_cps).replace(",", "-")
        stage_down = str(spec.stage_downsample).replace(",", "-")
        stage_epochs = str(spec.stage_epochs).replace(",", "-")
        return (
            f"{spec.impl}_{case_token}_sched{stage_full}_ds{stage_down}_"
            f"e{stage_epochs}_lr{c.format_float(spec.lr)}_a{c.format_float(spec.alpha)}_seed1"
        )
    return (
        f"{spec.impl}_{case_token}_e{int(spec.epochs or 2500)}_"
        f"lr{c.format_float(spec.lr)}_a{c.format_float(spec.alpha)}_seed1"
    )


def build_job(args: argparse.Namespace, case_id: int, spec: MethodSpec) -> dict[str, Any]:
    name = run_name(case_id, spec)
    run_dir = Path(args.run_root) / spec.impl / name
    command = [
        sys.executable,
        str(RUNNER),
        "--method",
        spec.impl,
        "--case-id",
        str(case_id),
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(args.run_root),
        "--gpu",
        spec.gpu,
        "--seed",
        "1",
        "--lr",
        f"{spec.lr:g}",
        "--alpha",
        f"{spec.alpha:g}",
        "--eval-interval",
        "250",
        "--global-ncc-diagnostic-interval",
        "10",
    ]
    if args.overwrite:
        command.append("--overwrite")
    if spec.stage_epochs is not None:
        command.extend(
            [
                "--stage-full-cps",
                str(spec.stage_full_cps),
                "--stage-downsample",
                str(spec.stage_downsample),
                "--stage-epochs",
                str(spec.stage_epochs),
            ]
        )
    else:
        command.extend(["--epochs", str(int(spec.epochs or 2500)), "--cps", "4"])
    return {
        "job_id": f"case{case_id:02d}_{spec.slug}",
        "case_key": f"case{case_id:02d}",
        "case_id": case_id,
        "method_slug": spec.slug,
        "display_method": spec.label,
        "implementation_id": spec.impl,
        "gpu": spec.gpu,
        "run_name": name,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "curve_path": str(run_dir / "global_ncc_diagnostic.csv"),
        "log_path": str(Path(args.run_root) / "logs" / f"{name}_gpu{spec.gpu}.log"),
        "command": command,
        "status": "pending",
        "returncode": "",
        "elapsed_sec": "",
        "failure_note": "",
    }


def build_jobs(args: argparse.Namespace, specs: Sequence[MethodSpec]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        for case_id in parse_cases(args.cases):
            jobs.append(build_job(args, case_id, spec))
    return jobs


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def compatible_complete(job: dict[str, Any]) -> tuple[bool, str]:
    summary = load_json(Path(job["summary_path"]))
    if summary.get("status") != "completed":
        return False, "summary_missing_or_not_completed"
    if summary.get("method") != job["implementation_id"]:
        return False, "method_mismatch"
    if summary.get("loss_domain") != "fixed_t00_lung_mask_global_ncc":
        return False, "loss_domain_mismatch"
    if summary.get("mask_rule") != "fixed_t00_lung_mask_loss_and_metric":
        return False, "mask_rule_mismatch"
    training = summary.get("training")
    if not isinstance(training, dict):
        return False, "training_missing"
    if training.get("image_loss") != "fixed_t00_lung_mask_global_ncc":
        return False, "image_loss_mismatch"
    if training.get("regularizer") != "dense_forward_difference_be":
        return False, "regularizer_mismatch"
    if float(training.get("alpha", float("nan"))) != 10000.0:
        return False, "alpha_mismatch"
    if int(training.get("global_ncc_diagnostic_interval", -1)) != 10:
        return False, "diagnostic_interval_mismatch"
    if job["method_slug"] == "mr_d_bscp":
        if training.get("stage_full_cps") != [32, 16, 8, 4]:
            return False, "stage_full_cps_mismatch"
        if training.get("stage_downsample") != [8, 4, 2, 1]:
            return False, "stage_downsample_mismatch"
        if training.get("stage_epochs") != [400, 400, 800, 900]:
            return False, "stage_epochs_mismatch"
    else:
        if int(training.get("epochs", -1)) != 2500:
            return False, "epochs_mismatch"
        if int(training.get("cps", -1)) != 4:
            return False, "cps_mismatch"
    if not Path(job["curve_path"]).is_file():
        return False, "global_ncc_diagnostic_missing"
    return True, ""


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
    counts = {status: sum(job.get("status") == status for job in jobs) for status in sorted({job.get("status") for job in jobs})}
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
        "# LungCT masked-NCC FD-BE quick launcher",
        "",
        f"- Updated: `{payload['updated_at_utc']}`",
        f"- Counts: `{counts}`",
        "",
        "| Job | Status | GPU | Elapsed sec | Note |",
        "|---|---|---:|---:|---|",
    ]
    for job in jobs:
        lines.append(
            f"| `{job['job_id']}` | `{job.get('status', '')}` | `{job.get('gpu', '')}` | "
            f"`{job.get('elapsed_sec', '')}` | `{job.get('failure_note', '')}` |"
        )
    (review_dir / "launcher_monitor.md").write_text("\n".join(lines) + "\n")


def check_gpus(gpus: Sequence[str], *, min_free_gb: float, timeout_sec: float) -> dict[str, dict[str, float]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(timeout_sec),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with returncode {completed.returncode}: {completed.stdout.strip()}")
    state: dict[str, dict[str, float]] = {}
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 3:
            continue
        index, used_mb, total_mb = parts[:3]
        used = float(used_mb)
        total = float(total_mb)
        state[index] = {
            "used_mb": used,
            "total_mb": total,
            "free_gb": (total - used) / 1024.0,
        }
    missing = [gpu for gpu in gpus if str(gpu) not in state]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report requested GPUs: {missing}")
    tight = [gpu for gpu in gpus if state[str(gpu)]["free_gb"] < float(min_free_gb)]
    if tight:
        free = {gpu: state[str(gpu)]["free_gb"] for gpu in tight}
        raise RuntimeError(f"Requested GPUs below min free memory {min_free_gb:g} GB: {free}")
    return state


def launch_job(job: dict[str, Any]) -> int:
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(token) for token in job["command"]]
    with log_path.open("w") as stream:
        stream.write(f"started_at_utc={utc_text()}\n")
        stream.write(f"gpu={job['gpu']}\n")
        stream.write("command=" + " ".join(command) + "\n\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
        stream.write(f"\nfinished_at_utc={utc_text()}\n")
        stream.write(f"returncode={completed.returncode}\n")
    return int(completed.returncode)


def worker(
    *,
    method_slug: str,
    jobs: list[dict[str, Any]],
    review_dir: Path,
    lock: threading.Lock,
    stop_event: threading.Event,
    started_at: float,
    resume: bool,
    overwrite: bool,
) -> None:
    for job in jobs:
        if stop_event.is_set():
            return
        ok, note = compatible_complete(job)
        if ok and (resume or not overwrite):
            with lock:
                job["status"] = "completed_existing"
                job["failure_note"] = ""
                write_state(review_dir, all_jobs, started_at)
            continue
        with lock:
            job["status"] = "running"
            job["failure_note"] = note
            job["started_at"] = time.time()
            write_state(review_dir, all_jobs, started_at)
        rc = launch_job(job)
        ok, note = compatible_complete(job)
        with lock:
            job["returncode"] = rc
            job["elapsed_sec"] = f"{time.time() - float(job['started_at']):.3f}"
            if rc == 0 and ok:
                job["status"] = "completed"
                job["failure_note"] = ""
            else:
                job["status"] = "failed"
                job["failure_note"] = note or f"returncode={rc}"
                stop_event.set()
            write_state(review_dir, all_jobs, started_at)
        if rc != 0:
            return


all_jobs: list[dict[str, Any]] = []


def main() -> None:
    global all_jobs
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    specs = method_specs_for_gpus(gpus)
    all_jobs = build_jobs(args, specs)
    started_at = time.time()
    args.review_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in all_jobs:
            ok, note = compatible_complete(job)
            if ok:
                job["status"] = "completed_existing"
                job["failure_note"] = ""
            else:
                job["failure_note"] = note
        write_state(args.review_dir, all_jobs, started_at)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "jobs": len(all_jobs),
                    "cases": list(parse_cases(args.cases)),
                    "gpus": list(gpus),
                    "pending": sum(job["status"] == "pending" for job in all_jobs),
                    "completed_existing": sum(job["status"] == "completed_existing" for job in all_jobs),
                    "review_dir": str(args.review_dir),
                },
                indent=2,
            )
        )
        return

    if not args.skip_gpu_check:
        gpu_state = check_gpus(gpus, min_free_gb=float(args.min_free_gb), timeout_sec=float(args.gpu_check_timeout_sec))
        (args.review_dir / "gpu_preflight.json").write_text(json.dumps(gpu_state, indent=2) + "\n")

    write_state(args.review_dir, all_jobs, started_at)
    lock = threading.Lock()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for spec in specs:
        method_jobs = [job for job in all_jobs if job["method_slug"] == spec.slug]
        thread = threading.Thread(
            target=worker,
            kwargs={
                "method_slug": spec.slug,
                "jobs": method_jobs,
                "review_dir": args.review_dir,
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
    write_state(args.review_dir, all_jobs, started_at)
    failed = [job["job_id"] for job in all_jobs if job["status"] == "failed"]
    print(json.dumps({"jobs": len(all_jobs), "failed": failed, "review_dir": str(args.review_dir)}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
