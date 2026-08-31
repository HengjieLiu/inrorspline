#!/usr/bin/env python3
"""Schedule the FireANTs CT-only two-window audit and merge locked P2e rows.

Each of four persistent workers owns exactly one cohort/window condition.  A
controller releases their warm queues only after all four cold fields have
passed finite DVF and cohort-metric gates.  The merger is read-only with
respect to the completed P2e output root.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np


THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
RUNNER_PATH = THIS_DIR / "run_pairlocal.py"
PROFILES = ("p2e_equiv_hu", "l2b_lung_hu")
COPD_CASES = tuple(str(value) for value in range(104, 114))
VALIDATION_CASES = tuple(str(value) for value in range(1, 11))


class AuditError(RuntimeError):
    """Raised when the immutable manifest, gates, or reports are invalid."""


def runner_module() -> ModuleType:
    name = "learn2breath_fireants_pairlocal_runner"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AuditError(f"Cannot load FireANTs runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    fields: list[str] = []
    for row in records:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list, tuple)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def case_label(dataset: str, case_id: str) -> str:
    return f"case_{int(case_id)}" if dataset == "copd" else f"NLST_{int(case_id):04d}"


def lanes(gpus: tuple[int, int, int, int]) -> tuple[dict[str, Any], ...]:
    return (
        {"lane": 0, "gpu": gpus[0], "dataset": "copd", "intensity_profile": PROFILES[0], "cases": COPD_CASES},
        {"lane": 1, "gpu": gpus[1], "dataset": "copd", "intensity_profile": PROFILES[1], "cases": COPD_CASES},
        {"lane": 2, "gpu": gpus[2], "dataset": "validation", "intensity_profile": PROFILES[0], "cases": VALIDATION_CASES},
        {"lane": 3, "gpu": gpus[3], "dataset": "validation", "intensity_profile": PROFILES[1], "cases": VALIDATION_CASES},
    )


def parse_gpu_ids(value: str) -> tuple[int, int, int, int]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 4 or len(set(parsed)) != 4 or any(item < 0 for item in parsed):
        raise AuditError("--gpus must name four distinct non-negative GPU IDs, e.g. 0,1,2,3")
    return parsed  # type: ignore[return-value]


def manifest_path(out_root: Path) -> Path:
    return out_root / "manifest" / "fireants_audit_manifest.json"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    gpu_ids = parse_gpu_ids(args.gpus)
    jobs: list[dict[str, Any]] = []
    for lane in lanes(gpu_ids):
        data_root = args.copd_data_root if lane["dataset"] == "copd" else args.validation_data_root
        for sequence_index, case_id in enumerate(lane["cases"]):
            label = case_label(lane["dataset"], case_id)
            jobs.append(
                {
                    "job_id": f"fireants_{lane['dataset']}_{lane['intensity_profile']}_{label}",
                    "method": "fireants_greedy_ct_only_ncc",
                    "dataset": lane["dataset"],
                    "case_id": case_id,
                    "case_label": label,
                    "intensity_profile": lane["intensity_profile"],
                    "gpu": int(lane["gpu"]),
                    "lane": int(lane["lane"]),
                    "sequence_index": sequence_index,
                    "temperature": "cold" if sequence_index == 0 else "warm",
                    "data_root": str(data_root),
                    "r231_root": str(args.r231_root) if lane["dataset"] == "copd" else None,
                    "learn2breath_repo": str(args.learn2breath_repo),
                    "out_root": str(args.out_root / "runs" / lane["dataset"] / lane["intensity_profile"] / label),
                }
            )
    manifest = {
        "experiment": "learn2breath_fireants_greedy_ct_only_two_window_audit",
        "job_count": len(jobs),
        "profiles": list(PROFILES),
        "scales": [16, 8, 4, 2, 1],
        "iterations": [200, 200, 200, 200, 200],
        "ct_only": True,
        "p2e_results_root_read_only": str(args.p2e_results_root),
        "jobs": jobs,
    }
    if len(jobs) != 40:
        raise AuditError(f"Expected exactly 40 jobs, got {len(jobs)}")
    destination = manifest_path(args.out_root)
    if destination.exists() and not args.force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != json_ready(manifest):
            raise AuditError(f"Refusing to replace a different manifest: {destination}")
        return existing
    write_json(destination, manifest)
    write_csv(args.out_root / "manifest" / "fireants_audit_manifest.csv", jobs)
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    if manifest.get("job_count") != 40 or len(jobs) != 40:
        raise AuditError("The FireANTs audit manifest must contain exactly 40 jobs")
    expected = {(dataset, profile) for dataset in ("copd", "validation") for profile in PROFILES}
    observed = {(job["dataset"], job["intensity_profile"]) for job in jobs}
    if observed != expected:
        raise AuditError("Manifest does not contain exactly the requested cohort/window conditions")
    return manifest


def preflight_all(args: argparse.Namespace) -> int:
    runner = runner_module()
    manifest = read_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        started = time.perf_counter()
        try:
            report = runner.preflight_job(
                dataset=job["dataset"],
                case_id=job["case_id"],
                intensity_profile=job["intensity_profile"],
                data_root=Path(job["data_root"]),
                r231_root=Path(job["r231_root"]) if job["r231_root"] else None,
                learn2breath_repo=Path(job["learn2breath_repo"]),
                device="cuda:0",
            )
            destination = args.out_root / "preflight" / job["dataset"] / job["intensity_profile"] / job["case_label"] / "preflight.json"
            write_json(destination, report)
            row = {**job, "status": "passed", "preflight_elapsed_sec": time.perf_counter() - started, "preflight_path": destination}
        except Exception as exc:  # noqa: BLE001 - preserve all gate failures.
            row = {**job, "status": "failed", "preflight_elapsed_sec": time.perf_counter() - started, "error": repr(exc)}
            failures.append(row)
        rows.append(row)
    write_csv(args.out_root / "preflight" / "preflight_rows.csv", rows)
    write_json(args.out_root / "preflight" / "preflight_summary.json", {"rows": rows, "failures": failures})
    if failures:
        raise AuditError(f"Preflight failed for {len(failures)}/40 FireANTs jobs")
    return 0


def runner_args(job: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=job["dataset"],
        case_id=str(job["case_id"]),
        intensity_profile=job["intensity_profile"],
        data_root=Path(job["data_root"]),
        r231_root=Path(job["r231_root"]) if job["r231_root"] else None,
        learn2breath_repo=Path(job["learn2breath_repo"]),
        out_root=Path(job["out_root"]),
        device="cuda:0",
        force=False,
        progress=False,
    )


def assert_completed(summary: dict[str, Any], job: dict[str, Any]) -> None:
    if not summary.get("completed") or summary.get("dataset") != job["dataset"]:
        raise AuditError(f"Missing completed result for {job['job_id']}")
    if summary.get("intensity_profile") != job["intensity_profile"]:
        raise AuditError(f"Intensity profile mismatch for {job['job_id']}")
    quality = summary.get("field_quality") or {}
    if not quality.get("finite") or not np.isfinite(float(quality.get("challenge_folding_percent_whole_grid", np.nan))):
        raise AuditError(f"Nonfinite or unevaluable field for {job['job_id']}")
    if job["dataset"] == "copd":
        tre = summary.get("tre") or {}
        if tre.get("landmark_count") != 300 or not np.isfinite(float(tre.get("registered_mean_tre_mm", np.nan))):
            raise AuditError(f"Missing physical TRE for {job['job_id']}")
    else:
        metrics = summary.get("validation") or {}
        dice = metrics.get("dice_per_label") or {}
        if set(dice) != {str(label) for label in (8, 16, 32, 64, 128)}:
            raise AuditError(f"Missing per-lobe Dice for {job['job_id']}")
        if not all(np.isfinite(float(value)) for value in dice.values()):
            raise AuditError(f"Nonfinite per-lobe Dice for {job['job_id']}")


def update_worker_timing(summary_path: Path, job: dict[str, Any], worker_started: float, worker_case_elapsed: float) -> None:
    runner = runner_module()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["timing"] = {
        "worker_case_elapsed_sec": float(worker_case_elapsed),
        "worker_process_start_to_case_complete_sec": float(time.perf_counter() - worker_started),
        "gpu": int(job["gpu"]),
        "lane": int(job["lane"]),
        "lane_sequence_index": int(job["sequence_index"]),
        "temperature": job["temperature"],
    }
    runner.p2e_module().write_json(summary_path, summary)


def wait_for_warm_release(control_root: Path) -> None:
    while True:
        abort = control_root / "abort.json"
        if abort.is_file():
            raise AuditError(f"Warm queue aborted: {abort.read_text(encoding='utf-8')}")
        if (control_root / "release_warm.json").is_file():
            return
        time.sleep(0.5)


def worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    manifest = read_manifest(args.manifest)
    jobs = [job for job in manifest["jobs"] if int(job["lane"]) == int(args.lane)]
    if len(jobs) != 10 or [job["sequence_index"] for job in jobs] != list(range(10)):
        raise AuditError(f"Lane {args.lane} must contain its ten ordered cases")
    runner = runner_module()  # Persistent import/CUDA process for cold/warm accounting.
    control_root = args.out_root / "control"
    progress: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        case_started = started if index == 0 else time.perf_counter()
        try:
            summary = runner.run_registration(runner_args(job))
            assert_completed(summary, job)
            summary_path = Path(job["out_root"]) / "summary.json"
            elapsed = time.perf_counter() - case_started
            update_worker_timing(summary_path, job, started, elapsed)
            row = {**job, "status": "completed", "worker_case_elapsed_sec": elapsed, "summary_path": summary_path}
        except Exception as exc:  # noqa: BLE001
            row = {**job, "status": "failed", "worker_case_elapsed_sec": time.perf_counter() - case_started, "error": repr(exc)}
            progress.append(row)
            write_csv(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_progress.csv", progress)
            write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_cold_gate.json", row if index == 0 else {"status": "failed", "row": row})
            write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_worker_failure.json", row)
            raise
        progress.append(row)
        write_csv(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_progress.csv", progress)
        if index == 0:
            write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_cold_gate.json", row)
            wait_for_warm_release(control_root)
    worker_summary = {
        "lane": int(args.lane),
        "gpu": int(jobs[0]["gpu"]),
        "status": "completed",
        "job_count": len(progress),
        "cold_count": 1,
        "warm_count": 9,
        "worker_total_elapsed_sec": time.perf_counter() - started,
        "rows": progress,
    }
    write_json(args.out_root / "workers" / f"gpu{jobs[0]['gpu']}_worker_summary.json", worker_summary)
    return 0


def launch(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    unique_lanes = sorted({(int(job["lane"]), int(job["gpu"])) for job in manifest["jobs"]})
    control = args.out_root / "control"
    if control.exists() and any(control.iterdir()):
        raise AuditError(f"Control directory is not empty: {control}")
    if any((Path(job["out_root"]) / "summary.json").exists() for job in manifest["jobs"]):
        raise AuditError("Existing case outputs would contaminate cold/warm timing; use a new output root")
    control.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processes: list[tuple[int, int, subprocess.Popen[Any], Any]] = []
    for lane, gpu in unique_lanes:
        log_path = args.out_root / "logs" / f"gpu{gpu}_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        process = subprocess.Popen(
            [sys.executable, str(THIS_FILE), "worker", "--manifest", str(args.manifest), "--out-root", str(args.out_root), "--lane", str(lane)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((lane, gpu, process, handle))
    cold_paths = [args.out_root / "workers" / f"gpu{gpu}_cold_gate.json" for _, gpu in unique_lanes]
    failure: str | None = None
    while not all(path.is_file() for path in cold_paths):
        exited = [(gpu, process.returncode) for _, gpu, process, _ in processes if process.poll() is not None]
        if exited:
            failure = f"Worker exited before cold gate: {exited}"
            break
        time.sleep(0.5)
    if failure is None:
        cold_rows = [json.loads(path.read_text(encoding="utf-8")) for path in cold_paths]
        failed = [row for row in cold_rows if row.get("status") != "completed"]
        if failed:
            failure = f"Cold gate failures: {[row.get('job_id') for row in failed]}"
        else:
            write_json(control / "release_warm.json", {"status": "released", "cold_rows": cold_rows})
    if failure is not None:
        write_json(control / "abort.json", {"status": "aborted", "reason": failure})
    return_codes: dict[str, int] = {}
    for _, gpu, process, handle in processes:
        return_codes[str(gpu)] = process.wait()
        handle.close()
    summary = {
        "status": "completed" if failure is None and all(code == 0 for code in return_codes.values()) else "failed",
        "overall_makespan_sec": time.perf_counter() - started,
        "worker_return_codes": return_codes,
        "failure": failure,
    }
    write_json(args.out_root / "launch_summary.json", summary)
    if summary["status"] != "completed":
        raise AuditError(f"Four-lane FireANTs launch failed: {summary}")
    return 0


def coordinate_external(args: argparse.Namespace) -> int:
    """Coordinate four externally launched, single-GPU persistent workers.

    This keeps the existing cold/warm gate intact when a host/runtime issue
    requires different, explicitly recorded container bases for individual
    physical GPUs. Workers still share the same immutable manifest and output
    root; only process placement is external to this controller.
    """

    manifest = read_manifest(args.manifest)
    unique_lanes = sorted({(int(job["lane"]), int(job["gpu"])) for job in manifest["jobs"]})
    control = args.out_root / "control"
    if control.exists() and any(control.iterdir()):
        raise AuditError(f"Control directory is not empty: {control}")
    if any((Path(job["out_root"]) / "summary.json").exists() for job in manifest["jobs"]):
        raise AuditError("Existing case outputs would contaminate cold/warm timing; use a new output root")
    control.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(control / "external_coordination_started.json", {
        "mode": "external_single_gpu_persistent_workers",
        "lanes": [{"lane": lane, "gpu": gpu} for lane, gpu in unique_lanes],
    })
    cold_paths = [args.out_root / "workers" / f"gpu{gpu}_cold_gate.json" for _, gpu in unique_lanes]
    failure_paths = [args.out_root / "workers" / f"gpu{gpu}_worker_failure.json" for _, gpu in unique_lanes]
    worker_paths = [args.out_root / "workers" / f"gpu{gpu}_worker_summary.json" for _, gpu in unique_lanes]
    failure: str | None = None
    while not all(path.is_file() for path in cold_paths):
        failures = [path for path in failure_paths if path.is_file()]
        if failures:
            failure = f"Worker failure before cold gate: {failures}"
            break
        time.sleep(0.5)
    if failure is None:
        cold_rows = [json.loads(path.read_text(encoding="utf-8")) for path in cold_paths]
        failed = [row for row in cold_rows if row.get("status") != "completed"]
        if failed:
            failure = f"Cold gate failures: {[row.get('job_id') for row in failed]}"
        else:
            write_json(control / "release_warm.json", {"status": "released", "cold_rows": cold_rows})
    if failure is not None:
        write_json(control / "abort.json", {"status": "aborted", "reason": failure})
    else:
        while not all(path.is_file() for path in worker_paths):
            failures = [path for path in failure_paths if path.is_file()]
            if failures:
                failure = f"Worker failure after warm release: {failures}"
                write_json(control / "abort.json", {"status": "aborted", "reason": failure})
                break
            time.sleep(1.0)
    summary = {
        "status": "completed" if failure is None and all(path.is_file() for path in worker_paths) else "failed",
        "overall_makespan_sec": time.perf_counter() - started,
        "coordination_mode": "external_single_gpu_persistent_workers",
        "worker_return_codes": "external_workers_reported_completed" if failure is None else None,
        "failure": failure,
    }
    write_json(args.out_root / "launch_summary.json", summary)
    if summary["status"] != "completed":
        raise AuditError(f"External four-lane FireANTs launch failed: {summary}")
    return 0


def finite_stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if value not in (None, "") and np.isfinite(float(value))]
    if not numbers:
        return {"n": 0, "mean": None, "std": None, "median": None, "p95": None, "sum": None}
    return {
        "n": len(numbers),
        "mean": float(statistics.mean(numbers)),
        "std": float(statistics.stdev(numbers)) if len(numbers) > 1 else 0.0,
        "median": float(statistics.median(numbers)),
        "p95": float(np.percentile(numbers, 95)),
        "sum": float(sum(numbers)),
    }


def read_fireants_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        summary_path = Path(job["out_root"]) / "summary.json"
        if not summary_path.is_file():
            raise AuditError(f"Missing FireANTs summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert_completed(summary, job)
        timing = summary.get("timing") or {}
        quality = summary["field_quality"]
        row: dict[str, Any] = {
            "job_id": job["job_id"], "dataset": job["dataset"], "case_id": job["case_id"], "case_label": job["case_label"],
            "method": "FireANTs Greedy NCC", "method_id": "fireants_greedy_ct_only_ncc", "intensity_profile": job["intensity_profile"],
            "runtime_environment": summary.get("runtime_environment", "fireants_runtime_unspecified"), "result_provenance": "new_fireants_ct_only_run",
            "gpu": timing.get("gpu"), "lane": timing.get("lane"), "lane_sequence_index": timing.get("lane_sequence_index"),
            "temperature": timing.get("temperature"), "optimizer_elapsed_sec": summary["optimizer_elapsed_sec"],
            "full_case_elapsed_sec": summary["full_case_elapsed_sec"], "worker_case_elapsed_sec": timing.get("worker_case_elapsed_sec"),
            "peak_memory_gb": summary["peak_memory_gb"], "challenge_folding_percent_whole_grid": quality.get("challenge_folding_percent_whole_grid"),
            "field_path": summary["field_path"], "summary_path": summary_path,
        }
        if job["dataset"] == "copd":
            row.update(summary["tre"])
            row["p2e_folding_percent_whole_grid"] = None
        else:
            metrics = summary["validation"]
            row.update({
                "mean_lobe_dice": metrics["mean_lobe_dice"],
                "challenge_folding_percent_fixed_lobes": metrics["folding_percent_fixed_lobes"],
            })
            for label, dice in metrics["dice_per_label"].items():
                row[f"dice_lobe_{label}"] = dice
        rows.append(row)
    return rows


def aggregate_fireants(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    rows = read_fireants_rows(manifest)
    if len(rows) != 40:
        raise AuditError("FireANTs aggregation expected 40 rows")
    write_csv(args.out_root / "reports" / "fireants_copd_per_case.csv", [row for row in rows if row["dataset"] == "copd"])
    write_csv(args.out_root / "reports" / "fireants_validation_per_case.csv", [row for row in rows if row["dataset"] == "validation"])
    write_csv(args.out_root / "reports" / "fireants_timing_per_case.csv", rows)
    write_json(args.out_root / "reports" / "fireants_case_rows.json", rows)
    return 0


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalise_p2e_rows(rows: list[dict[str, Any]], *, dataset: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["method"] = "MR-D-BSCP P2e"
        row["method_id"] = "mr_d_bscp_p2e"
        row["runtime_environment"] = "learn2breath26_dev_cuda11.8"
        row["result_provenance"] = "locked_completed_p2e_audit_read_only"
        row["dataset"] = dataset
        result.append(row)
    return result


def required_combined_keys(rows: list[dict[str, Any]], dataset: str) -> None:
    if len(rows) != 40:
        raise AuditError(f"Expected 40 merged {dataset} rows, got {len(rows)}")
    expected = {(str(case), method, profile) for case in (COPD_CASES if dataset == "copd" else VALIDATION_CASES)
                for method in ("FireANTs Greedy NCC", "MR-D-BSCP P2e") for profile in PROFILES}
    observed = {(str(int(row["case_id"])), row["method"], row["intensity_profile"]) for row in rows}
    if observed != expected:
        raise AuditError(f"Combined {dataset} table is incomplete, duplicated, or has wrong profiles")
    if dataset == "validation":
        for row in rows:
            for label in (8, 16, 32, 64, 128):
                if not np.isfinite(float(row[f"dice_lobe_{label}"])):
                    raise AuditError(f"Missing per-lobe Dice in combined validation row: {row}")


def grouped_summary(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in ("FireANTs Greedy NCC", "MR-D-BSCP P2e"):
        for profile in PROFILES:
            selected = [row for row in rows if row["method"] == method and row["intensity_profile"] == profile]
            entry: dict[str, Any] = {"dataset": dataset, "method": method, "intensity_profile": profile, "case_count": len(selected)}
            for field in ("optimizer_elapsed_sec", "full_case_elapsed_sec", "worker_case_elapsed_sec"):
                cold = finite_stats(row.get(field) for row in selected if row.get("temperature") == "cold")
                warm = finite_stats(row.get(field) for row in selected if row.get("temperature") == "warm")
                total = finite_stats(row.get(field) for row in selected)
                entry[f"cold_{field}"] = cold["mean"]
                entry[f"warm_{field}_mean"] = warm["mean"]
                entry[f"warm_{field}_std"] = warm["std"]
                entry[f"warm_{field}_median"] = warm["median"]
                entry[f"warm_{field}_p95"] = warm["p95"]
                entry[f"sum_{field}"] = total["sum"]
            entry["challenge_folding_mean_pct"] = finite_stats(row.get("challenge_folding_percent_whole_grid") for row in selected)["mean"]
            if dataset == "copd":
                entry["baseline_mean_tre_mm_mean"] = finite_stats(row.get("baseline_mean_tre_mm") for row in selected)["mean"]
                entry["registered_mean_tre_mm_mean"] = finite_stats(row.get("registered_mean_tre_mm") for row in selected)["mean"]
                entry["registered_mean_tre_mm_std"] = finite_stats(row.get("registered_mean_tre_mm") for row in selected)["std"]
            else:
                entry["mean_lobe_dice_mean"] = finite_stats(row.get("mean_lobe_dice") for row in selected)["mean"]
                entry["mean_lobe_dice_std"] = finite_stats(row.get("mean_lobe_dice") for row in selected)["std"]
                entry["challenge_folding_fixed_lobes_mean_pct"] = finite_stats(row.get("challenge_folding_percent_fixed_lobes") for row in selected)["mean"]
                for label in (8, 16, 32, 64, 128):
                    entry[f"dice_lobe_{label}_mean"] = finite_stats(row.get(f"dice_lobe_{label}") for row in selected)["mean"]
            result.append(entry)
    return result


def markdown_table(rows: list[dict[str, Any]], headers: list[tuple[str, str]], float_columns: set[str]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values: list[str] = []
        for key, _ in headers:
            value = row.get(key, "")
            if key in float_columns and value not in (None, ""):
                values.append(f"{float(value):.5f}")
            else:
                values.append(str(value if value is not None else ""))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def aggregate_combined(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    fireants = read_fireants_rows(manifest)
    p2e_root = Path(manifest["p2e_results_root_read_only"])
    p2e_copd = normalise_p2e_rows(read_csv(p2e_root / "reports" / "copd_per_case.csv"), dataset="copd")
    p2e_validation = normalise_p2e_rows(read_csv(p2e_root / "reports" / "validation_per_case.csv"), dataset="validation")
    copd = [row for row in fireants if row["dataset"] == "copd"] + p2e_copd
    validation = [row for row in fireants if row["dataset"] == "validation"] + p2e_validation
    copd.sort(key=lambda row: (int(row["case_id"]), row["method"], row["intensity_profile"]))
    validation.sort(key=lambda row: (int(row["case_id"]), row["method"], row["intensity_profile"]))
    required_combined_keys(copd, "copd")
    required_combined_keys(validation, "validation")
    timing = copd + validation
    summaries = grouped_summary(copd, "copd") + grouped_summary(validation, "validation")
    output = args.out_root / "reports"
    write_csv(output / "combined_copd_per_case.csv", copd)
    write_csv(output / "combined_validation_per_case.csv", validation)
    write_csv(output / "combined_timing_per_case.csv", timing)
    write_csv(output / "combined_grouped_summary.csv", summaries)
    launch = json.loads((args.out_root / "launch_summary.json").read_text(encoding="utf-8")) if (args.out_root / "launch_summary.json").is_file() else {}
    write_json(output / "combined_report.json", {
        "policy": "reported_only_no_winner_no_submission",
        "p2e_results_root_read_only": p2e_root,
        "runtime_caveat": "P2e CUDA11.8 and FireANTs CUDA12.1 timings are reported separately and are not a direct cross-container ranking.",
        "copd_rows": copd, "validation_rows": validation, "grouped_summary": summaries, "fireants_launch": launch,
    })
    lines = ["# FireANTs CT-only and locked P2e two-window audit", "", "The P2e rows are read-only provenance from the completed audit. FireANTs is CT-only; masks, lobes, and landmarks are evaluation/QC-only. No method or window is selected, and no submission is generated.", "", "## COPD detailed per-case TRE and folding", ""]
    lines.extend(markdown_table(copd, [
        ("case_label", "Case"), ("method", "Method"), ("intensity_profile", "Profile"), ("baseline_mean_tre_mm", "Baseline TRE mm"),
        ("registered_mean_tre_mm", "Registered TRE mm"), ("registered_median_tre_mm", "Median TRE mm"),
        ("registered_p95_tre_mm", "p95 TRE mm"), ("registered_max_tre_mm", "Max TRE mm"), ("challenge_folding_percent_whole_grid", "Whole folding %"),
    ], {"baseline_mean_tre_mm", "registered_mean_tre_mm", "registered_median_tre_mm", "registered_p95_tre_mm", "registered_max_tre_mm", "challenge_folding_percent_whole_grid"}))
    lines.extend(["", "## Validation detailed per-case Dice and folding", ""])
    lines.extend(markdown_table(validation, [
        ("case_label", "Case"), ("method", "Method"), ("intensity_profile", "Profile"), ("dice_lobe_8", "Dice 8"), ("dice_lobe_16", "Dice 16"),
        ("dice_lobe_32", "Dice 32"), ("dice_lobe_64", "Dice 64"), ("dice_lobe_128", "Dice 128"), ("mean_lobe_dice", "Mean Dice"),
        ("challenge_folding_percent_whole_grid", "Whole folding %"), ("challenge_folding_percent_fixed_lobes", "Lobe folding %"),
    ], {"dice_lobe_8", "dice_lobe_16", "dice_lobe_32", "dice_lobe_64", "dice_lobe_128", "mean_lobe_dice", "challenge_folding_percent_whole_grid", "challenge_folding_percent_fixed_lobes"}))
    lines.extend(["", "## Timing and environment caveat", "", "P2e ran in the prior CUDA 11.8 development container; FireANTs ran in the CUDA 12.1 FireANTs container. Per-method cold/warm timing is retained, but timings are not interpreted as a direct cross-container runtime comparison.", ""])
    lines.extend(markdown_table(summaries, [
        ("dataset", "Cohort"), ("method", "Method"), ("intensity_profile", "Profile"), ("case_count", "N"), ("cold_full_case_elapsed_sec", "Cold full s"),
        ("warm_full_case_elapsed_sec_mean", "Warm full mean s"), ("warm_full_case_elapsed_sec_std", "Warm full SD s"), ("sum_optimizer_elapsed_sec", "Optimizer sum s"),
        ("sum_full_case_elapsed_sec", "Full sum s"),
    ], {"cold_full_case_elapsed_sec", "warm_full_case_elapsed_sec_mean", "warm_full_case_elapsed_sec_std", "sum_optimizer_elapsed_sec", "sum_full_case_elapsed_sec"}))
    if launch:
        lines.extend(["", f"FireANTs four-GPU makespan: {float(launch.get('overall_makespan_sec', math.nan)):.3f} s."])
    (output / "combined_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def common_paths(parser: argparse.ArgumentParser, *, create: bool = False) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    if create:
        parser.add_argument("--copd-data-root", type=Path, required=True)
        parser.add_argument("--validation-data-root", type=Path, required=True)
        parser.add_argument("--r231-root", type=Path, required=True)
        parser.add_argument("--learn2breath-repo", type=Path, required=True)
        parser.add_argument("--p2e-results-root", type=Path, required=True)
        parser.add_argument("--gpus", default="0,1,2,3")
        parser.add_argument("--force", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-manifest")
    common_paths(make, create=True)
    preflight = sub.add_parser("preflight-all")
    preflight.add_argument("--manifest", type=Path, required=True)
    common_paths(preflight)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--lane", type=int, required=True)
    common_paths(worker_parser)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--manifest", type=Path, required=True)
    common_paths(launch_parser)
    external = sub.add_parser("coordinate-external")
    external.add_argument("--manifest", type=Path, required=True)
    common_paths(external)
    fireants = sub.add_parser("aggregate-fireants")
    fireants.add_argument("--manifest", type=Path, required=True)
    common_paths(fireants)
    combined = sub.add_parser("aggregate-combined")
    combined.add_argument("--manifest", type=Path, required=True)
    common_paths(combined)
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
    if args.command == "coordinate-external":
        return coordinate_external(args)
    if args.command == "aggregate-fireants":
        return aggregate_fireants(args)
    if args.command == "aggregate-combined":
        return aggregate_combined(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
