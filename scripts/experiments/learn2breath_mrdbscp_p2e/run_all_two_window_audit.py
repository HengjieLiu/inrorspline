#!/usr/bin/env python3
"""Launch and aggregate the dedicated four-GPU Learn2Breath P2e window audit.

The script has no hidden scheduler state.  It writes a fixed 40-row manifest
with one persistent worker per physical GPU.  Each worker executes its first
case as cold, waits at a controller barrier, then runs its remaining nine
cases warm in the same Python/CUDA process.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
PROFILES = ("p2e_equiv_hu", "l2b_lung_hu")
COPD_CASES = tuple(str(case_id) for case_id in range(104, 114))
VALIDATION_CASES = tuple(str(case_id) for case_id in range(1, 11))
LANES = (
    {"lane": 0, "gpu": 0, "dataset": "copd", "intensity_profile": "p2e_equiv_hu", "cases": COPD_CASES},
    {"lane": 1, "gpu": 1, "dataset": "copd", "intensity_profile": "l2b_lung_hu", "cases": COPD_CASES},
    {"lane": 2, "gpu": 2, "dataset": "validation", "intensity_profile": "p2e_equiv_hu", "cases": VALIDATION_CASES},
    {"lane": 3, "gpu": 3, "dataset": "validation", "intensity_profile": "l2b_lung_hu", "cases": VALIDATION_CASES},
)


class AuditError(RuntimeError):
    """Raised when the audit manifest or a run artifact is invalid."""


def runner_module():
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    return importlib.import_module("run_pairlocal")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in records:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (dict, list, tuple)):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in records)


def case_label(dataset: str, case_id: str) -> str:
    return f"case_{int(case_id)}" if dataset == "copd" else f"NLST_{int(case_id):04d}"


def manifest_path(out_root: Path) -> Path:
    return out_root / "manifest" / "audit_manifest.json"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if tuple(int(value) for value in args.gpus.split(",")) != (0, 1, 2, 3):
        raise AuditError("Dedicated audit requires --gpus 0,1,2,3 in that order")
    jobs: list[dict[str, Any]] = []
    for lane in LANES:
        for sequence_index, case_id in enumerate(lane["cases"]):
            label = case_label(str(lane["dataset"]), str(case_id))
            jobs.append(
                {
                    "job_id": f"gpu{lane['gpu']}_{lane['dataset']}_{lane['intensity_profile']}_{label}",
                    "lane": int(lane["lane"]),
                    "gpu": int(lane["gpu"]),
                    "sequence_index": sequence_index,
                    "temperature": "cold" if sequence_index == 0 else "warm",
                    "dataset": str(lane["dataset"]),
                    "case_id": str(case_id),
                    "case_label": label,
                    "intensity_profile": str(lane["intensity_profile"]),
                    "data_root": args.copd_data_root if lane["dataset"] == "copd" else args.validation_data_root,
                    "r231_root": args.r231_root if lane["dataset"] == "copd" else None,
                    "learn2breath_repo": args.learn2breath_repo,
                    "out_root": args.out_root / "runs" / str(lane["dataset"]) / str(lane["intensity_profile"]) / label,
                }
            )
    if len(jobs) != 40 or sorted(sum((list(lane["cases"]) for lane in LANES), [])) is None:
        raise AssertionError("Internal dedicated audit manifest construction failed")
    manifest = {
        "schema_version": 1,
        "experiment_id": "2026-08-30_learn2breath_mrdbscp_p2e_pairlocal_all20_two_windows",
        "profiles": list(PROFILES),
        "workers": [{key: value for key, value in lane.items() if key != "cases"} for lane in LANES],
        "job_count": len(jobs),
        "jobs": jobs,
    }
    destination = manifest_path(args.out_root)
    if destination.exists() and not args.force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != json_ready(manifest):
            raise AuditError(f"Refusing to replace a different manifest: {destination}")
        return existing
    write_json(destination, manifest)
    write_csv(args.out_root / "manifest" / "audit_manifest.csv", jobs)
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("job_count") != 40 or len(manifest.get("jobs", [])) != 40:
        raise AuditError("Expected exactly 40 jobs in the dedicated audit manifest")
    return manifest


def preflight_one(job: dict[str, Any]) -> dict[str, Any]:
    runner = runner_module()
    case = runner.load_pair(job["dataset"], Path(job["data_root"]), job["case_id"], job["intensity_profile"])
    report = runner.preflight(
        case,
        data_root=Path(job["data_root"]),
        r231_root=Path(job["r231_root"]) if job["r231_root"] is not None else None,
        case_id=job["case_id"],
        learn2breath_repo=Path(job["learn2breath_repo"]),
    )
    return report


def preflight_all(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        started = time.perf_counter()
        try:
            report = preflight_one(job)
            out_path = args.out_root / "preflight" / job["dataset"] / job["intensity_profile"] / job["case_label"] / "preflight.json"
            write_json(out_path, report)
            row = {**job, "status": "passed", "preflight_elapsed_sec": time.perf_counter() - started, "preflight_path": out_path}
        except Exception as exc:  # noqa: BLE001 - preserve all gate diagnostics.
            row = {**job, "status": "failed", "preflight_elapsed_sec": time.perf_counter() - started, "error": repr(exc)}
            failures.append(row)
        rows.append(row)
    write_csv(args.out_root / "preflight" / "preflight_rows.csv", rows)
    write_json(args.out_root / "preflight" / "preflight_summary.json", {"rows": rows, "failures": failures})
    if failures:
        raise AuditError(f"Preflight failed for {len(failures)}/40 jobs")
    return 0


def runner_args(job: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=job["dataset"],
        data_root=Path(job["data_root"]),
        case_id=str(job["case_id"]),
        intensity_profile=job["intensity_profile"],
        out_root=Path(job["out_root"]),
        r231_root=Path(job["r231_root"]) if job["r231_root"] is not None else None,
        learn2breath_repo=Path(job["learn2breath_repo"]),
        device="cuda:0",
        log_every=100,
        force=False,
    )


def assert_completed_summary(summary: dict[str, Any], job: dict[str, Any]) -> None:
    if summary.get("status") != "completed" or summary.get("dataset") != job["dataset"]:
        raise AuditError("Runner did not return a completed summary for the requested job")
    profile = summary.get("intensity_profile", {}).get("name")
    if profile != job["intensity_profile"]:
        raise AuditError(f"Summary profile mismatch: {profile!r} != {job['intensity_profile']!r}")
    if not summary.get("field_quality", {}).get("finite"):
        raise AuditError("Non-finite registration field")
    folding = summary.get("field_quality", {}).get("challenge_folding_percent_whole_grid")
    if not np.isfinite(folding):
        raise AuditError("Challenge-style whole-grid folding is non-finite")
    if job["dataset"] == "copd":
        tre = summary.get("tre") or {}
        if tre.get("landmark_count") != 300 or not np.isfinite(tre.get("registered_mean_tre_mm", np.nan)):
            raise AuditError("COPD TRE result is missing or non-finite")
    else:
        evaluation = summary.get("validation_evaluation") or {}
        dice = evaluation.get("dice_per_label") or {}
        if set(dice) != {str(label) for label in (8, 16, 32, 64, 128)} or not all(np.isfinite(value) for value in dice.values()):
            raise AuditError("Validation per-lobe Dice is missing or non-finite")
        if not np.isfinite(evaluation.get("folding_percent_whole_grid", np.nan)):
            raise AuditError("Validation challenge folding is non-finite")


def update_summary_worker_timing(summary_path: Path, job: dict[str, Any], worker_case_elapsed: float, worker_started: float) -> None:
    runner = runner_module()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("timing", {}).update(
        {
            "worker_case_elapsed_sec": float(worker_case_elapsed),
            "worker_process_start_to_case_complete_sec": float(time.perf_counter() - worker_started),
            "gpu": int(job["gpu"]),
            "lane": int(job["lane"]),
            "lane_sequence_index": int(job["sequence_index"]),
            "temperature": job["temperature"],
        }
    )
    runner.write_json(summary_path, summary)


def wait_for_controller(control_root: Path) -> None:
    release = control_root / "release_warm.json"
    abort = control_root / "abort.json"
    while True:
        if abort.is_file():
            raise AuditError(f"Controller aborted warm launch: {abort.read_text(encoding='utf-8')}")
        if release.is_file():
            return
        time.sleep(0.5)


def worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    manifest = read_manifest(args.manifest)
    jobs = [job for job in manifest["jobs"] if int(job["lane"]) == int(args.lane)]
    if len(jobs) != 10 or [job["sequence_index"] for job in jobs] != list(range(10)):
        raise AuditError(f"Lane {args.lane} must have exactly ten ordered jobs")
    runner = runner_module()  # Keep imports and CUDA initialization in this persistent worker.
    control_root = args.out_root / "control"
    progress: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        job_started = worker_started if index == 0 else time.perf_counter()
        try:
            summary = runner.run_registration(runner_args(job))
            assert_completed_summary(summary, job)
            summary_path = Path(job["out_root"]) / "summary.json"
            worker_case_elapsed = time.perf_counter() - job_started
            update_summary_worker_timing(summary_path, job, worker_case_elapsed, worker_started)
            row = {
                **job,
                "status": "completed",
                "worker_case_elapsed_sec": worker_case_elapsed,
                "worker_process_start_to_case_complete_sec": time.perf_counter() - worker_started,
                "summary_path": summary_path,
            }
        except Exception as exc:  # noqa: BLE001 - keep per-case failure evidence.
            row = {
                **job,
                "status": "failed",
                "worker_case_elapsed_sec": time.perf_counter() - job_started,
                "worker_process_start_to_case_complete_sec": time.perf_counter() - worker_started,
                "error": repr(exc),
            }
            progress.append(row)
            write_csv(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_progress.csv", progress)
            write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_cold_gate.json", row if index == 0 else {"status": "failed", "row": row})
            raise
        progress.append(row)
        write_csv(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_progress.csv", progress)
        if index == 0:
            write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_cold_gate.json", row)
            wait_for_controller(control_root)
    worker_summary = {
        "lane": int(args.lane),
        "gpu": int(jobs[0]["gpu"]),
        "status": "completed",
        "job_count": len(progress),
        "cold_count": sum(row["temperature"] == "cold" for row in progress),
        "warm_count": sum(row["temperature"] == "warm" for row in progress),
        "worker_total_elapsed_sec": time.perf_counter() - worker_started,
        "rows": progress,
    }
    write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_worker_summary.json", worker_summary)
    return 0


def launch(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    control_root = args.out_root / "control"
    if control_root.exists() and any(control_root.iterdir()):
        raise AuditError(f"Control directory is not empty; refuse timing-contaminated relaunch: {control_root}")
    control_root.mkdir(parents=True, exist_ok=True)
    for job in manifest["jobs"]:
        if (Path(job["out_root"]) / "summary.json").exists():
            raise AuditError(f"Existing output would contaminate cold/warm timing: {job['out_root']}")
    started = time.perf_counter()
    processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    for lane in LANES:
        gpu = int(lane["gpu"])
        log_path = args.out_root / "logs" / f"gpu{gpu}_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        process = subprocess.Popen(
            [sys.executable, str(THIS_FILE), "worker", "--manifest", str(args.manifest), "--out-root", str(args.out_root), "--lane", str(lane["lane"])],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((lane, process, handle))
    cold_paths = [args.out_root / "workers" / f"gpu{lane['gpu']}_cold_gate.json" for lane in LANES]
    failure: str | None = None
    while not all(path.is_file() for path in cold_paths):
        for lane, process, _ in processes:
            if process.poll() is not None:
                failure = f"GPU {lane['gpu']} worker exited before cold gate with code {process.returncode}"
                break
        if failure is not None:
            break
        time.sleep(0.5)
    if failure is None:
        cold_rows = [json.loads(path.read_text(encoding="utf-8")) for path in cold_paths]
        failed_cold = [row for row in cold_rows if row.get("status") != "completed"]
        if failed_cold:
            failure = f"Cold gate failed for {[row.get('job_id') for row in failed_cold]}"
        else:
            write_json(control_root / "release_warm.json", {"status": "released", "cold_rows": cold_rows})
    if failure is not None:
        write_json(control_root / "abort.json", {"status": "aborted", "reason": failure})
    return_codes: dict[str, int] = {}
    for lane, process, handle in processes:
        return_codes[str(lane["gpu"])] = process.wait()
        handle.close()
    summary = {
        "status": "completed" if failure is None and all(code == 0 for code in return_codes.values()) else "failed",
        "overall_makespan_sec": time.perf_counter() - started,
        "worker_return_codes": return_codes,
        "failure": failure,
    }
    write_json(args.out_root / "launch_summary.json", summary)
    if summary["status"] != "completed":
        raise AuditError(f"Dedicated launch failed: {summary}")
    return 0


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"n": 0, "mean": None, "std": None, "median": None, "p95": None, "sum": None}
    return {
        "n": len(finite),
        "mean": float(statistics.mean(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "median": float(statistics.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "sum": float(sum(finite)),
    }


def read_completed_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        path = Path(job["out_root"]) / "summary.json"
        if not path.is_file():
            raise AuditError(f"Missing summary for {job['job_id']}: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert_completed_summary(summary, job)
        timing = summary.get("timing") or {}
        if timing.get("temperature") != job["temperature"] or timing.get("gpu") != job["gpu"]:
            raise AuditError(f"Worker timing metadata mismatch for {job['job_id']}")
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "case_id": job["case_id"],
            "case_label": job["case_label"],
            "intensity_profile": job["intensity_profile"],
            "gpu": job["gpu"],
            "lane": job["lane"],
            "lane_sequence_index": job["sequence_index"],
            "temperature": job["temperature"],
            "optimizer_elapsed_sec": timing["optimizer_elapsed_sec"],
            "full_case_elapsed_sec": timing["full_case_elapsed_sec"],
            "worker_case_elapsed_sec": timing["worker_case_elapsed_sec"],
            "peak_memory_gb": summary["training"]["peak_memory_gb"],
            "challenge_folding_percent_whole_grid": summary["field_quality"]["challenge_folding_percent_whole_grid"],
            "p2e_folding_percent_whole_grid": summary["field_quality"]["folding_percent_whole_grid"],
            "summary_path": path,
        }
        if job["dataset"] == "copd":
            tre = summary["tre"]
            row.update({
                "baseline_mean_tre_mm": tre["baseline_mean_tre_mm"],
                "registered_mean_tre_mm": tre["registered_mean_tre_mm"],
                "registered_median_tre_mm": tre["registered_median_tre_mm"],
                "registered_p95_tre_mm": tre["registered_p95_tre_mm"],
                "registered_max_tre_mm": tre["registered_max_tre_mm"],
            })
        else:
            evaluation = summary["validation_evaluation"]
            row.update({
                "mean_lobe_dice": evaluation["mean_lobe_dice"],
                "challenge_folding_percent_fixed_lobes": evaluation["folding_percent_fixed_lobes"],
            })
            for label, value in evaluation["dice_per_label"].items():
                row[f"dice_lobe_{label}"] = value
        rows.append(row)
    return rows


def aggregate(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    rows = read_completed_rows(manifest)
    if len(rows) != 40:
        raise AuditError("Aggregate requires exactly 40 completed rows")
    copd_rows = [row for row in rows if row["dataset"] == "copd"]
    validation_rows = [row for row in rows if row["dataset"] == "validation"]
    write_csv(args.out_root / "reports" / "copd_per_case.csv", copd_rows)
    write_csv(args.out_root / "reports" / "validation_per_case.csv", validation_rows)
    write_csv(args.out_root / "reports" / "timing_per_case.csv", rows)
    group_rows: list[dict[str, Any]] = []
    for dataset in ("copd", "validation"):
        for profile in PROFILES:
            selected = [row for row in rows if row["dataset"] == dataset and row["intensity_profile"] == profile]
            if len(selected) != 10 or sum(row["temperature"] == "cold" for row in selected) != 1:
                raise AuditError(f"Expected one cold and nine warm rows for {dataset}/{profile}")
            cold = [row for row in selected if row["temperature"] == "cold"]
            warm = [row for row in selected if row["temperature"] == "warm"]
            result: dict[str, Any] = {"dataset": dataset, "intensity_profile": profile, "case_count": len(selected)}
            for field in ("optimizer_elapsed_sec", "full_case_elapsed_sec", "worker_case_elapsed_sec"):
                cold_stats = finite_summary([row[field] for row in cold])
                warm_stats = finite_summary([row[field] for row in warm])
                all_stats = finite_summary([row[field] for row in selected])
                result[f"cold_{field}"] = cold_stats["mean"]
                result[f"warm_{field}_mean"] = warm_stats["mean"]
                result[f"warm_{field}_std"] = warm_stats["std"]
                result[f"warm_{field}_median"] = warm_stats["median"]
                result[f"warm_{field}_p95"] = warm_stats["p95"]
                result[f"warm_{field}_total"] = warm_stats["sum"]
                result[f"all_{field}_total"] = all_stats["sum"]
            result["challenge_folding_mean_pct"] = finite_summary([row["challenge_folding_percent_whole_grid"] for row in selected])["mean"]
            if dataset == "copd":
                for field in ("baseline_mean_tre_mm", "registered_mean_tre_mm", "registered_median_tre_mm", "registered_p95_tre_mm"):
                    stats = finite_summary([row[field] for row in selected])
                    result[f"{field}_mean"] = stats["mean"]
                    result[f"{field}_std"] = stats["std"]
            else:
                result["mean_lobe_dice_mean"] = finite_summary([row["mean_lobe_dice"] for row in selected])["mean"]
                result["mean_lobe_dice_std"] = finite_summary([row["mean_lobe_dice"] for row in selected])["std"]
                result["challenge_folding_fixed_lobes_mean_pct"] = finite_summary(
                    [row["challenge_folding_percent_fixed_lobes"] for row in selected]
                )["mean"]
                for label in (8, 16, 32, 64, 128):
                    result[f"dice_lobe_{label}_mean"] = finite_summary([row[f"dice_lobe_{label}"] for row in selected])["mean"]
            worker_summary_path = args.out_root / "workers" / f"gpu{selected[0]['gpu']}_worker_summary.json"
            worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
            result["worker_total_elapsed_sec"] = worker_summary["worker_total_elapsed_sec"]
            group_rows.append(result)
    write_csv(args.out_root / "reports" / "cohort_profile_summary.csv", group_rows)
    launch_summary_path = args.out_root / "launch_summary.json"
    launch_summary = json.loads(launch_summary_path.read_text(encoding="utf-8")) if launch_summary_path.is_file() else {}
    report = {"case_rows": rows, "cohort_profile_summary": group_rows, "launch_summary": launch_summary}
    write_json(args.out_root / "reports" / "aggregate_report.json", report)
    lines = ["# Dedicated four-GPU two-window P2e audit", "", "## Cohort/profile summary", ""]
    lines.append("| Cohort | Profile | COPD TRE / validation Dice | Challenge folding % | Cold full s | Warm full s mean ± SD | Worker total s |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in group_rows:
        metric = row.get("registered_mean_tre_mm_mean") if row["dataset"] == "copd" else row.get("mean_lobe_dice_mean")
        lines.append(
            "| {dataset} | {intensity_profile} | {metric:.5f} | {fold:.5f} | {cold:.3f} | {warm:.3f} ± {std:.3f} | {worker:.3f} |".format(
                dataset=row["dataset"], intensity_profile=row["intensity_profile"], metric=metric,
                fold=row["challenge_folding_mean_pct"], cold=row["cold_full_case_elapsed_sec"],
                warm=row["warm_full_case_elapsed_sec_mean"], std=row["warm_full_case_elapsed_sec_std"],
                worker=row["worker_total_elapsed_sec"],
            )
        )
    if launch_summary:
        lines.extend(["", "## Overall wall time", "", f"- Four-worker makespan: {launch_summary.get('overall_makespan_sec'):.3f} s."])
    report_path = args.out_root / "reports" / "aggregate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def common_paths(parser: argparse.ArgumentParser, *, include_gpus: bool = False) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--copd-data-root", type=Path, required=True)
    parser.add_argument("--validation-data-root", type=Path, required=True)
    parser.add_argument("--r231-root", type=Path, required=True)
    parser.add_argument("--learn2breath-repo", type=Path, required=True)
    if include_gpus:
        parser.add_argument("--gpus", default="0,1,2,3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-manifest")
    common_paths(make, include_gpus=True)
    make.add_argument("--force", action="store_true")
    preflight_parser = sub.add_parser("preflight-all")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument("--out-root", type=Path, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--out-root", type=Path, required=True)
    worker_parser.add_argument("--lane", type=int, required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--manifest", type=Path, required=True)
    launch_parser.add_argument("--out-root", type=Path, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "make-manifest":
        manifest = build_manifest(args)
        print(json.dumps({"manifest": str(manifest_path(args.out_root)), "job_count": manifest["job_count"]}, indent=2))
        return 0
    if args.command == "preflight-all":
        return preflight_all(args)
    if args.command == "worker":
        return worker(args)
    if args.command == "launch":
        return launch(args)
    if args.command == "aggregate":
        return aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
