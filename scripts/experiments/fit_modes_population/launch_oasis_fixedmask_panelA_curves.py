#!/usr/bin/env python
"""Launch OASIS Panel-A runs with paired full-image/fixed-foreground NCC curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep/run_one.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/2026-08-14_oasis_panelA_fixedmask_diagnostics"
DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/oasis_panelA_fixedmask_diag_curves_v1")
DEFAULT_OASIS_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_OASIS_ROOT",
        os.environ.get(
            "OASIS_ROOT",
            "/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1",
        ),
    )
)
DEFAULT_PAIR_FILE = REPO_ROOT / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
ORIGINAL_FIRST10_INDICES = [158, 62, 14, 69, 47, 137, 110, 6, 116, 88]
EXPECTED_EPOCHS = [1, *range(10, 2501, 10)]
VARIANT = "oasis_panelA_fixedmask_diagnostic_v003"

METHODS: dict[str, dict[str, Any]] = {
    "idir": {
        "label": "INR-Dense",
        "impl": "idir_original",
        "out_key": "idir_original_dense_siren_be_autograd",
        "lr": "1e-4",
        "alpha": "1",
        "epochs": "2500",
        "estimated_sec": 220.0,
    },
    "inr_bscp": {
        "label": "INR-BSCP",
        "impl": "sinr_cps2_be_bspline_analytic",
        "out_key": "sinr_cps2_be_bspline_analytic",
        "lr": "1e-4",
        "alpha": "1000",
        "epochs": "2500",
        "cps": "2",
        "estimated_sec": 305.0,
    },
    "d_bscp": {
        "label": "D-BSCP",
        "impl": "direct_cp_cps2_be_bspline_analytic",
        "out_key": "direct_cp_cps2_be_bspline_analytic",
        "lr": "1e-3",
        "alpha": "1000",
        "epochs": "2500",
        "cps": "2",
        "estimated_sec": 110.0,
    },
    "mr_d_bscp": {
        "label": "MR-D-BSCP",
        "impl": "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "out_key": "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "lr": "1e-3",
        "alpha": "1000",
        "stage_full_cps": "8,4,2",
        "stage_downsample": "4,2,1",
        "stage_epochs": "400,800,1300",
        "estimated_sec": 195.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_csv_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        result.append(int(token))
    unique = tuple(dict.fromkeys(result))
    if not unique:
        raise ValueError("No integer values were provided.")
    return unique


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(token.strip() for token in value.split(",") if token.strip())
    if len(gpus) != 4:
        raise ValueError("This launcher expects exactly four GPU ids.")
    return gpus


def parse_pair_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ids = re.findall(r"\d+", line)
        if ids:
            if len(ids) != 2:
                raise ValueError(f"Cannot parse OASIS pair line: {line!r}")
            pairs.append((ids[0].zfill(4), ids[1].zfill(4)))
    if len(pairs) < 10:
        raise RuntimeError(f"Expected at least 10 OASIS pairs in {path}, found {len(pairs)}")
    return pairs


def fmt(value: str) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("+", "")


def run_name(rank: int, method_slug: str, smoke: bool) -> str:
    spec = METHODS[method_slug]
    suffix = "_smoke" if smoke else ""
    if method_slug == "mr_d_bscp":
        stages = "1-1-1" if smoke else "400-800-1300"
        return (
            f"oasis_fixedmask_rank{rank:03d}_panelA_{method_slug}_"
            f"sched8-4-2_ds4-2-1_e{stages}_a1000_lr0p001_seed1_diag10{suffix}"
        )
    epochs = "2" if smoke else spec["epochs"]
    return (
        f"oasis_fixedmask_rank{rank:03d}_panelA_{method_slug}_"
        f"e{epochs}_a{fmt(spec['alpha'])}_lr{fmt(spec['lr'])}_seed1_diag10{suffix}"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def curve_has_fixedmask_columns(path: Path, *, smoke: bool) -> tuple[bool, str]:
    if not path.is_file():
        return False, "curve_missing"
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception as exc:  # noqa: BLE001
        return False, f"curve_unreadable:{exc}"
    if not rows:
        return False, "curve_empty"
    required = {
        "global_epoch",
        "global_ncc_value",
        "fixed_image_positive_ncc_value",
        "fixed_image_positive_one_minus_ncc",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        return False, "missing_columns:" + ",".join(missing)
    try:
        epochs = [int(float(row["global_epoch"])) for row in rows]
        values = [float(row["fixed_image_positive_one_minus_ncc"]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        return False, f"curve_parse_error:{exc}"
    if not all(math.isfinite(value) and value > 0 for value in values):
        return False, "nonfinite_or_nonpositive_fixedmask_values"
    if not smoke and epochs != EXPECTED_EPOCHS:
        return False, "unexpected_epoch_grid"
    if smoke and epochs[-1] < 1:
        return False, "unexpected_smoke_epoch_grid"
    return True, ""


def compatible_complete(job: dict[str, Any], *, smoke: bool) -> tuple[bool, str]:
    summary = read_json(Path(job["summary_path"]))
    if summary.get("status") != "completed" and not (Path(job["run_dir"]) / "DONE").is_file():
        return False, "summary_missing_or_not_completed"
    if summary.get("fixed_image_positive_ncc_diagnostic_name") != "fixed_image_positive_global_ncc":
        return False, "fixed_image_positive_diagnostic_metadata_missing"
    if int(summary.get("global_ncc_diagnostic_interval", -1)) != (1 if smoke else 10):
        return False, "diagnostic_interval_mismatch"
    if int(summary.get("case_rank", -1)) != int(job["case_rank"]):
        return False, "case_rank_mismatch"
    method_slug = str(job["method_slug"])
    if method_slug == "mr_d_bscp":
        if summary.get("stage_full_cps") != [8, 4, 2]:
            return False, "stage_full_cps_mismatch"
        if summary.get("stage_downsample") != [4, 2, 1]:
            return False, "stage_downsample_mismatch"
        expected = [1, 1, 1] if smoke else [400, 800, 1300]
        if summary.get("stage_epochs") != expected:
            return False, "stage_epochs_mismatch"
    ok, note = curve_has_fixedmask_columns(Path(job["curve_path"]), smoke=smoke)
    if not ok:
        return False, note
    return True, ""


def versioned_paths(args: argparse.Namespace, rank: int, method_slug: str) -> tuple[str, Path]:
    spec = METHODS[method_slug]
    base_name = run_name(rank, method_slug, args.smoke)
    root = Path(args.run_root) / spec["out_key"]
    candidate = root / base_name
    version = 1
    while candidate.exists():
        probe = {
            "run_dir": str(candidate),
            "summary_path": str(candidate / "summary.json"),
            "curve_path": str(candidate / "global_ncc_diagnostic.csv"),
            "case_rank": rank,
            "method_slug": method_slug,
        }
        ok, _note = compatible_complete(probe, smoke=args.smoke)
        if ok:
            return candidate.name, candidate
        version += 1
        candidate = root / f"{base_name}_v{version:02d}"
    return candidate.name, candidate


def build_command(args: argparse.Namespace, rank: int, method_slug: str, run_name_value: str, case: dict[str, Any]) -> list[str]:
    spec = METHODS[method_slug]
    epochs = "2" if args.smoke and method_slug != "mr_d_bscp" else spec.get("epochs", "2500")
    diagnostic = "1" if args.smoke else "10"
    eval_interval = "1" if args.smoke else "250"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--method",
        spec["impl"],
        "--alpha",
        spec["alpha"],
        "--lr",
        spec["lr"],
        "--gpu",
        "__GPU__",
        "--case-rank",
        str(rank),
        "--epochs",
        str(epochs),
        "--eval-interval",
        eval_interval,
        "--global-ncc-diagnostic-interval",
        diagnostic,
        "--seed",
        "1",
        "--variant",
        VARIANT,
        "--out-root",
        str(args.run_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--expected-original-case-index",
        str(case["original_case_index"]),
        "--expected-fixed-id",
        str(case["fixed_id"]),
        "--expected-moving-id",
        str(case["moving_id"]),
        "--run-name",
        run_name_value,
    ]
    if method_slug == "mr_d_bscp":
        cmd.extend(
            [
                "--stage-full-cps",
                spec["stage_full_cps"],
                "--stage-downsample",
                spec["stage_downsample"],
                "--stage-epochs",
                "1,1,1" if args.smoke else spec["stage_epochs"],
            ]
        )
    if args.smoke:
        cmd.append("--no-save-final-dense")
    return cmd


def build_cases(pair_file: Path, ranks: Sequence[int]) -> dict[int, dict[str, Any]]:
    pairs = parse_pair_file(pair_file)
    cases: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        if rank < 0 or rank >= 10:
            raise ValueError(f"This launcher is locked to OASIS first10 ranks; got {rank}")
        fixed_id, moving_id = pairs[rank]
        cases[int(rank)] = {
            "case_rank": int(rank),
            "original_case_index": int(ORIGINAL_FIRST10_INDICES[rank]),
            "fixed_id": fixed_id,
            "moving_id": moving_id,
            "case_key": f"rank{rank:03d}",
        }
    return cases


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    ranks = (0,) if args.smoke else parse_csv_ints(args.ranks)
    cases = build_cases(Path(args.pair_file), ranks)
    method_order = ["idir", "inr_bscp", "d_bscp", "mr_d_bscp"] if args.smoke else ["inr_bscp", "d_bscp", "idir", "mr_d_bscp"]
    jobs: list[dict[str, Any]] = []
    for method_slug in method_order:
        for rank in ranks:
            name, run_dir = versioned_paths(args, rank, method_slug)
            spec = METHODS[method_slug]
            command = build_command(args, rank, method_slug, name, cases[rank])
            jobs.append(
                {
                    "job_id": f"oasis_rank{rank:03d}_panelA_{method_slug}",
                    "dataset": "oasis",
                    "case_key": f"rank{rank:03d}",
                    "case_rank": rank,
                    "original_case_index": cases[rank]["original_case_index"],
                    "fixed_id": cases[rank]["fixed_id"],
                    "moving_id": cases[rank]["moving_id"],
                    "panel": "A",
                    "method_slug": method_slug,
                    "display_method": spec["label"],
                    "implementation_id": spec["impl"],
                    "run_name": name,
                    "run_dir": str(run_dir),
                    "summary_path": str(run_dir / "summary.json"),
                    "curve_path": str(run_dir / "global_ncc_diagnostic.csv"),
                    "command": command,
                    "estimated_sec": float(spec["estimated_sec"]) * (0.05 if args.smoke else 1.0),
                    "status": "pending",
                    "gpu": "",
                    "pid": "",
                    "elapsed_sec": "",
                    "returncode": "",
                    "failure_note": "",
                    "source_kind": "new_run",
                }
            )
    expected = 4 if args.smoke else 40
    if len(jobs) != expected or len({job["job_id"] for job in jobs}) != expected:
        raise RuntimeError(f"Expected {expected} unique jobs, got {len(jobs)}")
    return jobs


def worker_queues(jobs: Sequence[dict[str, Any]], gpus: Sequence[str], *, smoke: bool) -> dict[str, list[dict[str, Any]]]:
    if smoke:
        return {gpu: [job] for gpu, job in zip(gpus, jobs)}
    queues = {gpu: [] for gpu in gpus}
    for job in jobs:
        rank = int(job["case_rank"])
        method = str(job["method_slug"])
        if method in {"inr_bscp", "d_bscp"}:
            gpu = gpus[0] if rank <= 4 else gpus[1]
        else:
            gpu = gpus[2] if rank <= 4 else gpus[3]
        queues[gpu].append(job)
    order = {"inr_bscp": 0, "d_bscp": 1, "idir": 0, "mr_d_bscp": 1}
    for gpu, queue in queues.items():
        queue.sort(key=lambda item: (order[str(item["method_slug"])], int(item["case_rank"])))
    return queues


def portable(value: Any, args: argparse.Namespace) -> Any:
    replacements = [
        (str(args.run_root), "${OASIS_MASKED_PANELA_RUN_ROOT}"),
        (str(args.oasis_root), "${OASIS_ROOT}"),
        (str(args.pair_file), "${OASIS_PAIR_FILE}"),
        (str(REPO_ROOT), "${REPO_ROOT}"),
    ]
    if isinstance(value, dict):
        return {key: portable(item, args) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [portable(item, args) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("_") or isinstance(value, (list, dict)):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def status_payload(args: argparse.Namespace, jobs: Sequence[dict[str, Any]], queues: dict[str, list[dict[str, Any]]], started: float) -> dict[str, Any]:
    counts = {status: sum(job["status"] == status for job in jobs) for status in sorted({job["status"] for job in jobs})}
    running = {str(job["gpu"]): job for job in jobs if job["status"] == "running"}
    remaining = 0.0
    for job in jobs:
        if job["status"] in {"pending", "running"}:
            elapsed = float(job.get("elapsed_sec") or 0.0)
            remaining += max(float(job["estimated_sec"]) - elapsed, float(job["estimated_sec"]) * 0.1)
    active_gpus = [gpu for gpu in queues if queues[gpu] or gpu in running]
    eta = remaining / max(len(active_gpus), 1)
    return portable(
        {
            "updated_at_utc": utc_text(),
            "elapsed_sec": time.time() - started,
            "counts": counts,
            "active_gpus": {
                gpu: {
                    "job_id": running[gpu]["job_id"],
                    "elapsed_sec": running[gpu].get("elapsed_sec", ""),
                    "log_path": running[gpu].get("log_path", ""),
                }
                for gpu in sorted(running)
            },
            "pending_by_gpu": {gpu: [job["job_id"] for job in queue] for gpu, queue in queues.items()},
            "eta_seconds": eta,
            "estimated_completion_utc": datetime.fromtimestamp(time.time() + eta, timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        },
        args,
    )


def write_state(args: argparse.Namespace, jobs: Sequence[dict[str, Any]], queues: dict[str, list[dict[str, Any]]], started: float) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_jobs = [portable(dict(job), args) for job in jobs]
    write_csv(output / "job_manifest.csv", safe_jobs)
    (output / "job_manifest.json").write_text(json.dumps(safe_jobs, indent=2, default=str) + "\n")
    payload = status_payload(args, jobs, queues, started)
    (output / "launcher_state.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    lines = [
        "# OASIS fixed-mask Panel A launcher state",
        "",
        f"- Updated: `{payload['updated_at_utc']}`",
        f"- Elapsed sec: `{float(payload['elapsed_sec']):.1f}`",
        f"- ETA sec: `{float(payload['eta_seconds']):.1f}`",
        f"- Estimated completion: `{payload['estimated_completion_utc']}`",
        f"- Counts: `{payload['counts']}`",
        "",
        "| GPU | Active job | Pending |",
        "|---:|---|---|",
    ]
    for gpu in sorted(queues):
        active = payload["active_gpus"].get(gpu, {})
        lines.append(f"| {gpu} | `{active.get('job_id', 'idle')}` | `{len(payload['pending_by_gpu'].get(gpu, []))}` |")
    (output / "monitor_status.md").write_text("\n".join(lines) + "\n")
    (output / "monitor_status.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def launch(job: dict[str, Any], gpu: str) -> subprocess.Popen[str]:
    command = [gpu if token == "__GPU__" else token for token in job["command"]]
    log_path = Path(job["run_dir"]).parents[1] / "logs" / f"{job['job_id']}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    handle.write(json.dumps({"started_at_utc": utc_text(), "job_id": job["job_id"], "gpu": gpu, "command": command}) + "\n")
    handle.flush()
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    job["status"] = "running"
    job["gpu"] = gpu
    job["pid"] = process.pid
    job["started_at"] = time.time()
    job["log_path"] = str(log_path)
    job["_process"] = process
    job["_log_handle"] = handle
    return process


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    if args.poll_sec < 1 or args.poll_sec > 60:
        raise ValueError("--poll-sec must be in [1, 60]")
    jobs = build_jobs(args)
    for job in jobs:
        ok, note = compatible_complete(job, smoke=args.smoke)
        if ok:
            job["status"] = "completed_existing"
            job["source_kind"] = "exact_compatible_existing"
        else:
            job["failure_note"] = note
    queues = worker_queues([job for job in jobs if job["status"] == "pending"], gpus, smoke=args.smoke)
    started = time.time()
    write_state(args, jobs, queues, started)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "jobs": len(jobs), "pending": sum(job["status"] == "pending" for job in jobs), "output_dir": str(args.output_dir)}, indent=2))
        return

    running: dict[str, dict[str, Any]] = {}
    while any(queues.values()) or running:
        for gpu, queue in queues.items():
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            launch(job, gpu)
            running[gpu] = job
            print(json.dumps({"started": job["job_id"], "gpu": gpu, "remaining_on_gpu": len(queue)}), flush=True)
        time.sleep(int(args.poll_sec))
        for gpu, job in list(running.items()):
            process = job["_process"]
            job["elapsed_sec"] = f"{time.time() - float(job['started_at']):.3f}"
            returncode = process.poll()
            if returncode is None:
                continue
            job["_log_handle"].close()
            job["returncode"] = returncode
            ok, note = compatible_complete(job, smoke=args.smoke)
            if returncode == 0 and ok:
                job["status"] = "completed"
                job["failure_note"] = ""
                print(json.dumps({"completed": job["job_id"], "gpu": gpu, "elapsed_sec": job["elapsed_sec"]}), flush=True)
            else:
                job["status"] = "failed"
                job["failure_note"] = note or f"returncode={returncode}"
                print(json.dumps({"failed": job["job_id"], "gpu": gpu, "note": job["failure_note"], "log": job.get("log_path", "")}), flush=True)
            del job["_process"]
            del job["_log_handle"]
            del running[gpu]
        write_state(args, jobs, queues, started)
        if any(job["status"] == "failed" for job in jobs):
            raise SystemExit(1)
    write_state(args, jobs, queues, started)
    print(json.dumps({"status": "completed", "jobs": len(jobs), "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
