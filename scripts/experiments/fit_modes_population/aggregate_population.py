#!/usr/bin/env python
"""Aggregate and validate the OASIS10/DIR-LAB10 fit-modes population run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


EXPERIMENT_KEY = "2026-08-02_fit_modes_population_n10"
RUN_KEY = "fit_modes_population_n10_v1"
SMOKE_RUN_KEY = "fit_modes_population_n10_smoke_v1"
FIGURE_EXPERIMENT_KEY = "paper_artifacts/fitting_figure/v002_candidate"
TERMINAL_SUCCESS = {"completed", "completed_existing", "reused"}
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--figure-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Keep aggregate reports external and resolve the current figure input safely."""

    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    args.run_root = artifact_paths.resolve_run_root(
        SMOKE_RUN_KEY if args.review_root.name == "smoke" else RUN_KEY,
        args.run_root,
        repo_root=REPO_ROOT,
    )
    args.figure_root = artifact_paths.resolve_review_root(
        FIGURE_EXPERIMENT_KEY,
        args.figure_root,
        repo_root=REPO_ROOT,
    )
    return args


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def resolve_path(value: Any, run_root: Path, data_root: Path) -> Path:
    text = str(value or "")
    text = text.replace("${RUN_ROOT}", str(run_root))
    text = text.replace("${DATA_ROOT}", str(data_root))
    text = text.replace("${REPO_ROOT}", str(REPO_ROOT))
    return Path(text)


def review_path(path: Path | None, run_root: Path, data_root: Path) -> str:
    if path is None:
        return ""
    text = str(path)
    if text == str(run_root) or text.startswith(str(run_root) + "/"):
        return text.replace(str(run_root), "${RUN_ROOT}", 1)
    if text == str(data_root) or text.startswith(str(data_root) + "/"):
        return text.replace(str(data_root), "${DATA_ROOT}", 1)
    if text == str(REPO_ROOT) or text.startswith(str(REPO_ROOT) + "/"):
        return text.replace(str(REPO_ROOT), "${REPO_ROOT}", 1)
    return text


def curve_path(job: dict[str, Any], run_root: Path, data_root: Path) -> Path | None:
    source = resolve_path(
        job.get("source_path") or job.get("curve_path") or "", run_root, data_root
    )
    if source.is_file():
        return source
    if source.is_dir():
        preferred = [
            source
            / resolve_path(job.get("curve_path", "history.csv"), run_root, data_root).name,
            source / "global_ncc_diagnostic.csv",
            source / "stage_history.csv",
            source / "history.csv",
            source / "image_similarity_eval.csv",
        ]
        return next((path for path in preferred if path.is_file()), None)
    fallback = resolve_path(job.get("curve_path") or "", run_root, data_root)
    return fallback if fallback.is_file() else None


def summary_path(
    job: dict[str, Any], curve: Path | None, run_root: Path, data_root: Path
) -> Path | None:
    direct = resolve_path(job.get("summary_path") or "", run_root, data_root)
    if direct.is_file():
        return direct
    source = resolve_path(job.get("source_path") or "", run_root, data_root)
    candidates: list[Path] = []
    if source.is_dir():
        candidates.append(source / "summary.json")
    if curve is not None:
        candidates.extend(parent / "summary.json" for parent in [curve.parent, *list(curve.parents)[:3]])
    return next((path for path in candidates if path.is_file()), None)


def last_finite_row(path: Path, fields: Iterable[str]) -> tuple[dict[str, str], str, float] | None:
    rows = read_csv(path)
    for row in reversed(rows):
        for field in fields:
            value = finite(row.get(field))
            if value is not None:
                return row, field, value
    return None


def selected_endpoint_map(figure_root: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    path = figure_root / "fit_modes_standardized_panelAB_curve_summary.csv"
    for row in read_csv(path):
        plot_case = str(row["plot_case"])
        key = (str(row["panel"]), plot_case, str(row["display_method"]))
        value = finite(row.get("value_end"))
        if value is not None:
            result[key] = value
    return result


def target_power_map(figure_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    path = figure_root / "fit_modes_standardized_target_power_summary.csv"
    for row in read_csv(path):
        value = finite(row.get("target_power_component_mean"))
        if value is not None:
            result[str(row["case"])] = value
    return result


def selected_case_label(job: dict[str, Any]) -> str:
    if job["dataset"] == "oasis":
        return f"OASIS rank-{int(job['case_rank'])}"
    return f"LungCT case{int(job['case_id']):02d}"


def selected_power_label(job: dict[str, Any]) -> str:
    if job["dataset"] == "oasis":
        return "OASIS SITReg"
    return f"LungCT case{int(job['case_id']):02d}"


def endpoint_from_job(
    job: dict[str, Any],
    summary: dict[str, Any],
    curve: Path | None,
    selected_endpoints: dict[tuple[str, str, str], float],
    powers: dict[str, float],
) -> tuple[str, float | None, float | None, float | None, str]:
    reused = job.get("source_kind") == "exact_compatible_current_figure"
    if reused:
        endpoint = selected_endpoints.get((str(job["panel"]), selected_case_label(job), str(job["display_method"])))
        if job["panel"] == "A":
            return "one_minus_ncc", endpoint, None, None, "v002_candidate_standardized_curve_summary"
        power = powers.get(selected_power_label(job))
        mse = None if endpoint is None or power is None else (endpoint / 100.0) ** 2 * power
        return "rRMSE_percent", endpoint, mse, power, "v002_candidate_standardized_curve_summary"

    if job["panel"] == "A":
        if curve is None:
            return "one_minus_ncc", None, None, None, "missing_curve"
        match = last_finite_row(curve, ["global_ncc_value", "ncc_value", "global_ncc_loss", "data_loss", "loss"])
        if match is None:
            return "one_minus_ncc", None, None, None, "no_finite_registration_endpoint"
        _, field, value = match
        if field in {"global_ncc_value", "ncc_value"}:
            endpoint = 1.0 - value
        elif field == "global_ncc_loss" and value <= 0:
            endpoint = 1.0 + value
        else:
            endpoint = value
        return "one_minus_ncc", endpoint, None, None, f"curve:{field}"

    mse = finite(summary.get("field_mse_objective_normalized_grid"))
    power = finite(summary.get("field_target_power_masked_normalized_grid"))
    rrmse = finite(summary.get("field_rrmse_objective_percent"))
    source = "summary.json"
    if mse is None and curve is not None:
        match = last_finite_row(
            curve,
            ["finest_field_mse_objective_normalized_grid", "field_mse_objective_normalized_grid", "data_loss", "loss"],
        )
        if match is not None:
            _, field, mse = match
            source = f"curve:{field}"
    if power is None:
        power = powers.get(selected_power_label(job))
        if power is not None:
            source += "+legacy_target_power"
    if rrmse is None and mse is not None and power is not None:
        rrmse = 100.0 * math.sqrt(mse / (power + 1e-12))
    return "rRMSE_percent", rrmse, mse, power, source


def runtime_values(summary: dict[str, Any], job: dict[str, Any]) -> tuple[float | None, float | None]:
    runtime = next(
        (
            value
            for key in ("training_elapsed_sec", "elapsed_sec", "total_elapsed_sec")
            if (value := finite(summary.get(key))) is not None
        ),
        finite(job.get("elapsed_sec")),
    )
    peak = next(
        (
            value
            for key in ("training_peak_mem_gb", "peak_mem_gb")
            if (value := finite(summary.get(key))) is not None
        ),
        None,
    )
    return runtime, peak


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def population_stats(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get("endpoint_value"))
        if value is not None:
            grouped[(row["dataset"], row["panel"], row["method_slug"], row["endpoint_name"])].append(value)
    output: list[dict[str, Any]] = []
    for (dataset, panel, method, endpoint), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "panel": panel,
                "method_slug": method,
                "endpoint_name": endpoint,
                "n": len(values),
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "q1": quantile(values, 0.25),
                "q3": quantile(values, 0.75),
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    review = Path(args.review_root)
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    jobs_path = review / "job_manifest.json"
    jobs = json.loads(jobs_path.read_text())
    if not isinstance(jobs, list):
        raise RuntimeError(f"Expected a list in {jobs_path}")
    selected_endpoints = selected_endpoint_map(Path(args.figure_root))
    powers = target_power_map(Path(args.figure_root))
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for job in jobs:
        curve = curve_path(job, run_root, data_root)
        summary_file = summary_path(job, curve, run_root, data_root)
        summary = load_json(summary_file) if summary_file else {}
        endpoint_name, endpoint, mse, power, endpoint_source = endpoint_from_job(
            job, summary, curve, selected_endpoints, powers
        )
        runtime, peak = runtime_values(summary, job)
        successful = str(job.get("status")) in TERMINAL_SUCCESS
        checks = {
            "terminal_success": successful,
            "curve_exists": curve is not None,
            "endpoint_finite": endpoint is not None,
            "target_power_finite": job["panel"] != "B" or power is not None,
            "runtime_finite": runtime is not None,
            "peak_memory_finite": peak is not None,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            problems.append(f"{job['job_id']}: {','.join(failed_checks)}")
        rows.append(
            {
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "case_key": job["case_key"],
                "case_rank": job.get("case_rank"),
                "original_case_index": job.get("original_case_index"),
                "fixed_id": job.get("fixed_id"),
                "moving_id": job.get("moving_id"),
                "case_id": job.get("case_id"),
                "panel": job["panel"],
                "objective": job["objective"],
                "method_slug": job["method_slug"],
                "display_method": job["display_method"],
                "implementation_id": job["implementation_id"],
                "status": job["status"],
                "source_kind": job["source_kind"],
                "source_path": review_path(curve, run_root, data_root)
                if curve
                else job.get("source_path", ""),
                "summary_path": review_path(summary_file, run_root, data_root),
                "endpoint_name": endpoint_name,
                "endpoint_value": endpoint,
                "field_mse_objective_normalized_grid": mse,
                "field_target_power_masked_normalized_grid": power,
                "diagnostic_ncc": None if job["panel"] != "A" or endpoint is None else 1.0 - endpoint,
                "runtime_sec": runtime,
                "peak_memory_gb": peak,
                "endpoint_source": endpoint_source,
                "lr": job["lr"],
                "alpha": job.get("alpha"),
                "cps": job["cps"],
                "stage_downsample": job["stage_downsample"],
                "stage_epochs": job["stage_epochs"],
                "epochs": job["epochs"],
                "seed": job["seed"],
                "diagnostic_interval": job["diagnostic_interval"],
                "acceptance_checks": ";".join(f"{key}={str(value).lower()}" for key, value in checks.items()),
            }
        )

    expected = 16 if review.name == "smoke" else 160
    duplicate_ids = len({row["job_id"] for row in rows}) != len(rows)
    if len(rows) != expected:
        problems.append(f"record_count: expected {expected}, found {len(rows)}")
    if duplicate_ids:
        problems.append("duplicate_job_ids")
    stats = population_stats(rows)
    write_csv(review / "optimization_index.csv", rows)
    (review / "optimization_index.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    write_csv(review / "population_statistics.csv", stats)
    acceptance = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "expected_records": expected,
        "observed_records": len(rows),
        "accepted": not problems,
        "problems": problems,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "population_groups_with_n10": sum(int(row["n"]) == 10 for row in stats),
        "population_groups_total": len(stats),
    }
    (review / "acceptance_report.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    lines = [
        "# Population Fit-Modes Acceptance Report",
        "",
        f"Generated: `{acceptance['generated_at']}`",
        f"Acceptance: `{'PASS' if acceptance['accepted'] else 'FAIL'}`",
        f"Records: `{len(rows)}/{expected}`",
        f"Complete population groups: `{acceptance['population_groups_with_n10']}/{acceptance['population_groups_total']}`",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in acceptance["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- `{problem}`" for problem in problems] or ["- None."])
    (review / "acceptance_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(acceptance, indent=2), flush=True)
    if problems and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
