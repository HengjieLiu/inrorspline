#!/usr/bin/env python
"""Launch the LungCT unmasked FD-BE cps4 alpha-extension runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


RUNNER = THIS_DIR / "run_one.py"
RUN_KEY = "lungct_fdbe_cps4_alpha1e5_1e6_all10_v1"
SMOKE_RUN_KEY = "lungct_fdbe_cps4_alpha1e5_1e6_all10_smoke_v1"
EXPERIMENT_KEY = "2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10"
DEFAULT_RUN_ROOT = None
DEFAULT_SMOKE_RUN_ROOT = None
DEFAULT_DATA_ROOT = Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm")
DEFAULT_REVIEW_DIR = None
DEFAULT_CASES = "1,2,3,4,5,6,7,8,9,10"


@dataclass(frozen=True)
class LaneSpec:
    lane_slug: str
    method_slug: str
    display_method: str
    implementation_id: str
    lr: float
    alpha: float
    gpu: str


LANES = (
    LaneSpec(
        lane_slug="inr_bscp_a100000",
        method_slug="inr_bscp",
        display_method="INR-BSCP cps4 (FD-BE)",
        implementation_id="sinr_cps4_global_ncc_fd_be",
        lr=1.0e-4,
        alpha=100000.0,
        gpu="0",
    ),
    LaneSpec(
        lane_slug="inr_bscp_a1000000",
        method_slug="inr_bscp",
        display_method="INR-BSCP cps4 (FD-BE)",
        implementation_id="sinr_cps4_global_ncc_fd_be",
        lr=1.0e-4,
        alpha=1000000.0,
        gpu="1",
    ),
    LaneSpec(
        lane_slug="d_bscp_a100000",
        method_slug="d_bscp",
        display_method="D-BSCP cps4 (FD-BE)",
        implementation_id="direct_cp_cps4_global_ncc_fd_be",
        lr=1.0e-3,
        alpha=100000.0,
        gpu="2",
    ),
    LaneSpec(
        lane_slug="d_bscp_a1000000",
        method_slug="d_bscp",
        display_method="D-BSCP cps4 (FD-BE)",
        implementation_id="direct_cp_cps4_global_ncc_fd_be",
        lr=1.0e-3,
        alpha=1000000.0,
        gpu="3",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPUs assigned to INR-a1e5, INR-a1e6, D-a1e5, D-a1e6.",
    )
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
    if args.smoke:
        if str(args.cases) == DEFAULT_CASES:
            args.cases = "1"
        if args.epoch_override is None:
            args.epoch_override = 1
        args.eval_interval = 1
        args.global_ncc_diagnostic_interval = 1
    return resolve_artifact_paths(args)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve run and review outputs beneath the external artifact store."""

    run_key = SMOKE_RUN_KEY if args.smoke else RUN_KEY
    args.run_root = artifact_paths.resolve_run_root(run_key, args.run_root, repo_root=REPO_ROOT)
    review_key = f"{EXPERIMENT_KEY}/smoke" if args.smoke else EXPERIMENT_KEY
    requested_review_dir = None if args.review_dir is None else Path(args.review_dir)
    if requested_review_dir is not None and args.smoke:
        requested_review_dir = requested_review_dir / "smoke"
    args.review_dir = artifact_paths.resolve_review_root(
        review_key, requested_review_dir, repo_root=REPO_ROOT
    )
    return args


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


def parse_gpus(text: str) -> tuple[str, str, str, str]:
    gpus = tuple(token.strip() for token in str(text).split(",") if token.strip())
    if len(gpus) != len(LANES):
        raise ValueError(f"Expected {len(LANES)} GPU ids, got {gpus}")
    return gpus  # type: ignore[return-value]


def lanes_for_gpus(gpus: Sequence[str]) -> tuple[LaneSpec, ...]:
    return tuple(LaneSpec(**{**asdict(lane), "gpu": str(gpu)}) for lane, gpu in zip(LANES, gpus))


def expected_epochs(args: argparse.Namespace) -> int:
    return int(args.epoch_override if args.epoch_override is not None else args.epochs)


def run_name(case_id: int, lane: LaneSpec, epochs: int) -> str:
    return (
        f"{lane.implementation_id}_case{case_id:02d}_e{int(epochs)}_"
        f"lr{c.format_float(lane.lr)}_a{c.format_float(lane.alpha)}_seed1"
    )


def build_job(args: argparse.Namespace, case_id: int, lane: LaneSpec) -> dict[str, Any]:
    epochs = expected_epochs(args)
    name = run_name(case_id, lane, epochs)
    run_dir = Path(args.run_root) / lane.implementation_id / name
    command = [
        sys.executable,
        str(RUNNER),
        "--method",
        lane.implementation_id,
        "--case-id",
        str(case_id),
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(args.run_root),
        "--run-name",
        name,
        "--gpu",
        lane.gpu,
        "--seed",
        "1",
        "--lr",
        f"{lane.lr:g}",
        "--alpha",
        f"{lane.alpha:g}",
        "--cps",
        "4",
        "--epochs",
        str(epochs),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--global-ncc-diagnostic-interval",
        str(int(args.global_ncc_diagnostic_interval)),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.smoke:
        command.append("--no-save-dense-dvf")
    return {
        "job_id": f"case{case_id:02d}_{lane.lane_slug}",
        "case_key": f"case{case_id:02d}",
        "case_id": int(case_id),
        "lane_slug": lane.lane_slug,
        "method_slug": lane.method_slug,
        "display_method": lane.display_method,
        "implementation_id": lane.implementation_id,
        "lr": float(lane.lr),
        "alpha": float(lane.alpha),
        "cps": 4,
        "epochs": int(epochs),
        "seed": 1,
        "diagnostic_interval": int(args.global_ncc_diagnostic_interval),
        "gpu": lane.gpu,
        "run_name": name,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "curve_path": str(run_dir / "global_ncc_diagnostic.csv"),
        "log_path": str(Path(args.run_root) / "logs" / f"{name}_gpu{lane.gpu}.log"),
        "command": command,
        "command_text": " ".join(str(part) for part in command),
        "status": "pending",
        "returncode": "",
        "elapsed_sec": "",
        "failure_note": "",
    }


def build_jobs(args: argparse.Namespace, lanes: Sequence[LaneSpec]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for lane in lanes:
        for case_id in parse_cases(args.cases):
            jobs.append(build_job(args, case_id, lane))
    return jobs


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def close(left: Any, right: float, tol: float = 1.0e-8) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value - float(right)) <= tol * max(1.0, abs(float(right)))


def last_curve_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    raw = rows[-1].get("global_epoch", rows[-1].get("epoch"))
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def compatible_complete(job: dict[str, Any]) -> tuple[bool, str]:
    summary = load_json(Path(job["summary_path"]))
    if summary.get("status") != "completed":
        return False, "summary_missing_or_not_completed"
    method = str(summary.get("method", ""))
    if method != job["implementation_id"]:
        return False, "method_mismatch"
    if "masked" in method.lower():
        return False, "masked_method_id"
    if summary.get("dataset") != "4dct":
        return False, "dataset_mismatch"
    if int(summary.get("case_id", -1)) != int(job["case_id"]):
        return False, "case_id_mismatch"
    if summary.get("mask_rule") != "fixed_t00_lung_mask_metric_only":
        return False, "mask_rule_mismatch"
    if summary.get("loss_domain") != "full_crop_global_ncc":
        return False, "loss_domain_mismatch"
    if "masked" in str(summary.get("protocol_tag", "")).lower():
        return False, "masked_protocol_tag"
    training = summary.get("training")
    if not isinstance(training, dict):
        return False, "training_missing"
    checks = {
        "epochs": int(training.get("epochs", -1)) == int(job["epochs"]),
        "lr": close(training.get("lr"), float(job["lr"])),
        "alpha": close(training.get("alpha"), float(job["alpha"])),
        "cps": int(training.get("cps", -1)) == 4,
        "seed": int(training.get("seed", -1)) == 1,
        "image_loss": training.get("image_loss") == "full_crop_global_ncc",
        "regularizer": training.get("regularizer") == "dense_forward_difference_be",
        "diagnostic_interval": int(training.get("global_ncc_diagnostic_interval", -1))
        == int(job["diagnostic_interval"]),
        "global_diagnostic_nontraining": training.get("global_ncc_diagnostic_affects_training") is False,
        "lung_mask_diagnostic_nontraining": training.get("lung_mask_ncc_diagnostic_affects_training") is False,
    }
    for name, passed in checks.items():
        if not passed:
            return False, f"{name}_mismatch"
    curve_path = Path(job["curve_path"])
    if not curve_path.is_file():
        return False, "global_ncc_diagnostic_missing"
    if last_curve_epoch(curve_path) != int(job["epochs"]):
        return False, "global_ncc_final_epoch_mismatch"
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
        "# LungCT FD-BE cps4 alpha-extension launcher",
        "",
        f"- Updated: `{payload['updated_at_utc']}`",
        f"- Counts: `{counts}`",
        "",
        "| Job | Status | GPU | Alpha | Elapsed sec | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for job in jobs:
        lines.append(
            f"| `{job['job_id']}` | `{job.get('status', '')}` | `{job.get('gpu', '')}` | "
            f"`{job.get('alpha', '')}` | `{job.get('elapsed_sec', '')}` | `{job.get('failure_note', '')}` |"
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
        stream.write(f"command={' '.join(command)}\n\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
        stream.write(f"\nfinished_at_utc={utc_text()}\n")
        stream.write(f"returncode={completed.returncode}\n")
    return int(completed.returncode)


def worker(
    *,
    lane_slug: str,
    lane_jobs: list[dict[str, Any]],
    all_jobs: list[dict[str, Any]],
    review_dir: Path,
    lock: threading.Lock,
    stop_event: threading.Event,
    started_at: float,
    resume: bool,
    overwrite: bool,
) -> None:
    for job in lane_jobs:
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


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    lanes = lanes_for_gpus(gpus)
    jobs = build_jobs(args, lanes)
    started_at = time.time()
    Path(args.review_dir).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            ok, note = compatible_complete(job)
            if ok:
                job["status"] = "completed_existing"
                job["failure_note"] = ""
            else:
                job["failure_note"] = note
        write_state(Path(args.review_dir), jobs, started_at)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "jobs": len(jobs),
                    "lanes": [lane.lane_slug for lane in lanes],
                    "cases": list(parse_cases(args.cases)),
                    "gpus": list(gpus),
                    "pending": sum(job["status"] == "pending" for job in jobs),
                    "completed_existing": sum(job["status"] == "completed_existing" for job in jobs),
                    "review_dir": str(args.review_dir),
                    "run_root": str(args.run_root),
                },
                indent=2,
            )
        )
        return

    if not args.skip_gpu_check:
        gpu_state = check_gpus(gpus, min_free_gb=float(args.min_free_gb), timeout_sec=float(args.gpu_check_timeout_sec))
        (Path(args.review_dir) / "gpu_preflight.json").write_text(json.dumps(gpu_state, indent=2) + "\n")

    write_state(Path(args.review_dir), jobs, started_at)
    lock = threading.Lock()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for lane in lanes:
        lane_jobs = [job for job in jobs if job["lane_slug"] == lane.lane_slug]
        thread = threading.Thread(
            target=worker,
            kwargs={
                "lane_slug": lane.lane_slug,
                "lane_jobs": lane_jobs,
                "all_jobs": jobs,
                "review_dir": Path(args.review_dir),
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
    write_state(Path(args.review_dir), jobs, started_at)
    failed = [job["job_id"] for job in jobs if job["status"] == "failed"]
    print(json.dumps({"jobs": len(jobs), "failed": failed, "review_dir": str(args.review_dir)}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
