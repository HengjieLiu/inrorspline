#!/usr/bin/env python3
"""Validate and report adjacent first/repeat LungCT shape diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

import common as c
import launch_mrdbscp_shape_pair_cache_diagnostic_gpu1 as diagnostic

diagnostic.configure_base()

from aggregate_mrdbscp_adaptive_fixed50_gpu1 import (
    finite,
    provenance_errors,
    schedule_errors,
    validate_dvf,
)


DATASETS = diagnostic.base.DATASETS
VARIANTS = diagnostic.base.VARIANTS
CASES = diagnostic.base.CASES
EXPERIMENT_ID = diagnostic.EXPERIMENT_ID
PHYSICAL_GPU = diagnostic.base.PHYSICAL_GPU
TRE_KEYS = (
    "native_crop_tre_snap_to_voxel",
    "native_crop_tre_subvoxel",
    "tre_snap_to_voxel",
    "tre_subvoxel",
)
TIMING_KEYS = (
    "loaded_images_to_dvf_file_sec",
    "loaded_images_to_dvf_in_memory_sec",
    "optimizer_elapsed_sec",
    "peak_mem_gb",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    return parser.parse_args(argv)


def mean(values: Iterable[Any]) -> float:
    numbers = [finite(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return float(statistics.mean(numbers)) if numbers else float("nan")


def sample_sd(values: Iterable[Any]) -> float:
    numbers = [finite(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return float(statistics.stdev(numbers)) if len(numbers) >= 2 else 0.0 if len(numbers) == 1 else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def accepted_records(run_root: Path, errors: list[str]) -> list[dict[str, Any]]:
    path = run_root / "accepted_arms.json"
    accepted = c.read_json(path) if path.is_file() else {}
    if not isinstance(accepted, dict):
        errors.append("accepted_arms.json is invalid")
        return []
    keys = {f"primary:{dataset}:{variant}" for dataset in DATASETS for variant in VARIANTS}
    records: list[dict[str, Any]] = []
    for key in sorted(keys):
        record = accepted.get(key)
        if not isinstance(record, dict):
            errors.append(f"missing accepted arm {key}")
            continue
        monitor = record.get("monitor", {})
        worker = record.get("worker_result", {})
        if not record.get("accepted") or not isinstance(monitor, dict) or not monitor.get("clean"):
            errors.append(f"{key}: arm/monitor is not cleanly accepted")
        if record.get("contaminated") or monitor.get("bad_samples") or finite(monitor.get("max_sample_gap_sec")) > 5.0:
            errors.append(f"{key}: contamination, query error, or excessive monitor gap")
        if not isinstance(worker, dict) or worker.get("status") != "completed" or int(worker.get("row_count", -1)) != 20:
            errors.append(f"{key}: worker did not complete 20 rows")
        else:
            records.append(record)
    return records


def read_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors = provenance_errors(run_root)
    rows: list[dict[str, Any]] = []
    for record in accepted_records(run_root, errors):
        for worker_row in record["worker_result"]["rows"]:
            dataset = str(worker_row.get("dataset", ""))
            variant = str(worker_row.get("variant", ""))
            case_id = int(worker_row.get("case_id", -1))
            condition = str(worker_row.get("timing_condition", ""))
            prefix = f"{dataset}/{variant}/case{case_id:02d}/{condition}"
            summary_path = Path(str(worker_row.get("summary_path", "")))
            dvf_path = Path(str(worker_row.get("dense_dvf_path", "")))
            if not summary_path.is_file():
                errors.append(f"{prefix}: missing summary")
                continue
            summary = c.read_json(summary_path)
            if not isinstance(summary, dict):
                errors.append(f"{prefix}: summary is invalid")
                continue
            if summary.get("status") != "completed" or summary.get("dataset") != dataset or summary.get("variant_id") != variant:
                errors.append(f"{prefix}: identity/status mismatch")
            for error in schedule_errors(summary, dataset, variant):
                errors.append(f"{prefix}: {error}")
            timing = summary.get("timing", {})
            if not isinstance(timing, dict) or timing.get("timing_condition") != condition or timing.get("timing_mode") != "input_to_dvf":
                errors.append(f"{prefix}: timing condition/mode mismatch")
                timing = {}
            training = summary.get("training", {})
            actual = training.get("actual_stage_epochs", []) if isinstance(training, dict) else []
            shape = summary.get("input_shape_zyx", [])
            row: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "case_id": case_id,
                "timing_condition": condition,
                "sequence_index": int(worker_row.get("sequence_index", -1)),
                "run_name": str(worker_row.get("run_name", "")),
                "matrix_zyx": "x".join(str(int(value)) for value in shape),
                "voxel_count": math.prod(int(value) for value in shape),
                "actual_stage_epochs": json.dumps(actual),
                "actual_optimizer_steps": sum(int(value) for value in actual),
                "summary_path": str(summary_path),
                "dense_dvf_path": str(dvf_path),
            }
            for key in TIMING_KEYS:
                row[key] = summary.get(key, timing.get(key))
                if not math.isfinite(finite(row[key])):
                    errors.append(f"{prefix}: nonfinite {key}")
            for tre_key in TRE_KEYS:
                metric = summary.get(tre_key, {})
                if not isinstance(metric, dict) or int(metric.get("n_landmarks", -1)) != 300 or not math.isfinite(finite(metric.get("tre_mean_mm"))):
                    errors.append(f"{prefix}: invalid 300-landmark {tre_key}")
                row[f"{tre_key}_mean_mm"] = metric.get("tre_mean_mm") if isinstance(metric, dict) else None
            valid, message = validate_dvf(dvf_path, shape)
            if not valid:
                errors.append(f"{prefix}: {message}")
            rows.append(row)
    expected = {
        (dataset, variant, case_id, condition)
        for dataset in DATASETS
        for variant in VARIANTS
        for case_id in CASES
        for condition in ("shape_first", "shape_repeat")
    }
    observed = {(row["dataset"], row["variant"], row["case_id"], row["timing_condition"]) for row in rows}
    if len(rows) != diagnostic.EXPECTED_PRIMARY_ROWS or observed != expected:
        errors.append(f"paired row contract mismatch: rows={len(rows)}, unique_keys={len(observed)}")
    if len({row["run_name"] for row in rows}) != len(rows) or len({row["dense_dvf_path"] for row in rows}) != len(rows):
        errors.append("run names or DVF paths are not unique")
    return rows, errors


def pair_rows(rows: Sequence[dict[str, Any]], errors: list[str]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["dataset"]), str(row["variant"]), int(row["case_id"]), str(row["timing_condition"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for case_id in CASES:
                first = indexed.get((dataset, variant, int(case_id), "shape_first"))
                repeat = indexed.get((dataset, variant, int(case_id), "shape_repeat"))
                if first is None or repeat is None:
                    continue
                if first["matrix_zyx"] != repeat["matrix_zyx"]:
                    errors.append(f"{dataset}/{variant}/case{case_id:02d}: matrix mismatch")
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "variant": variant,
                    "case_id": int(case_id),
                    "matrix_zyx": first["matrix_zyx"],
                    "voxel_count": first["voxel_count"],
                    "first_actual_optimizer_steps": first["actual_optimizer_steps"],
                    "repeat_actual_optimizer_steps": repeat["actual_optimizer_steps"],
                    "step_delta_repeat_minus_first": int(repeat["actual_optimizer_steps"]) - int(first["actual_optimizer_steps"]),
                }
                for key in TIMING_KEYS:
                    first_value = finite(first[key])
                    repeat_value = finite(repeat[key])
                    row[f"first_{key}"] = first_value
                    row[f"repeat_{key}"] = repeat_value
                    row[f"reduction_{key}"] = first_value - repeat_value
                    row[f"speedup_{key}"] = first_value / repeat_value if repeat_value > 0.0 else float("nan")
                for tre_key in TRE_KEYS:
                    metric_key = f"{tre_key}_mean_mm"
                    row[f"first_{metric_key}"] = first[metric_key]
                    row[f"repeat_{metric_key}"] = repeat[metric_key]
                    row[f"delta_{metric_key}"] = finite(repeat[metric_key]) - finite(first[metric_key])
                if variant == "fixed50" and row["step_delta_repeat_minus_first"] != 0:
                    errors.append(f"{dataset}/{variant}/case{case_id:02d}: fixed50 step mismatch")
                output.append(row)
    if len(output) != 40:
        errors.append(f"pair row count={len(output)}, expected=40")
    return output


def aggregate_pairs(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = [row for row in pairs if row["dataset"] == dataset and row["variant"] == variant]
            record: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "case_count": len(selected),
                "step_changed_case_count": sum(int(row["step_delta_repeat_minus_first"]) != 0 for row in selected),
            }
            for key in TIMING_KEYS:
                record[f"first_{key}_mean"] = mean(row[f"first_{key}"] for row in selected)
                record[f"first_{key}_sd"] = sample_sd(row[f"first_{key}"] for row in selected)
                record[f"repeat_{key}_mean"] = mean(row[f"repeat_{key}"] for row in selected)
                record[f"repeat_{key}_sd"] = sample_sd(row[f"repeat_{key}"] for row in selected)
                record[f"mean_reduction_{key}"] = mean(row[f"reduction_{key}"] for row in selected)
                repeat_mean = finite(record[f"repeat_{key}_mean"])
                record[f"ratio_of_means_{key}"] = finite(record[f"first_{key}_mean"]) / repeat_mean if repeat_mean > 0.0 else float("nan")
                record[f"median_case_speedup_{key}"] = statistics.median(finite(row[f"speedup_{key}"]) for row in selected)
            output.append(record)
    return output


def write_report(review_root: Path, pairs: Sequence[dict[str, Any]], aggregates: Sequence[dict[str, Any]], errors: Sequence[str]) -> None:
    lines = [
        "# LungCT adjacent shape-first versus shape-repeat diagnostic",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Technical acceptance: `{'PASS' if not errors else 'FAIL'}`",
        "- Each case is executed twice consecutively in the same persistent dataset/variant process; controls, optimizer, and seed reset between registrations.",
        "- `shape_first` is process-warm for cases 02-10 but the first occurrence of that matrix; `shape_repeat` immediately repeats the identical matrix.",
        "- Fixed-50 is the clean backend-state diagnostic because first/repeat optimizer steps are identical.",
        "",
        "## Aggregate primary timing",
        "",
        "| Dataset | Variant | Cases | First s | Repeat s | Reduction s | Ratio of means | Median case speedup | Step-changed cases |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['case_count']} | "
            f"{fmt(row['first_loaded_images_to_dvf_file_sec_mean'])} +/- {fmt(row['first_loaded_images_to_dvf_file_sec_sd'])} | "
            f"{fmt(row['repeat_loaded_images_to_dvf_file_sec_mean'])} +/- {fmt(row['repeat_loaded_images_to_dvf_file_sec_sd'])} | "
            f"{fmt(row['mean_reduction_loaded_images_to_dvf_file_sec'])} | "
            f"{fmt(row['ratio_of_means_loaded_images_to_dvf_file_sec'], 2)}x | "
            f"{fmt(row['median_case_speedup_loaded_images_to_dvf_file_sec'], 2)}x | {row['step_changed_case_count']} |"
        )
    for dataset in DATASETS:
        lines.extend(
            [
                "",
                f"## {dataset.upper()} casewise paired timing",
                "",
                "| Case | Matrix ZxYxX | Adaptive first/repeat s | Adaptive speedup | Adaptive steps first/repeat | Fixed-50 first/repeat s | Fixed-50 speedup |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case_id in CASES:
            adaptive = next(row for row in pairs if row["dataset"] == dataset and row["variant"] == "adaptive" and row["case_id"] == case_id)
            fixed = next(row for row in pairs if row["dataset"] == dataset and row["variant"] == "fixed50" and row["case_id"] == case_id)
            lines.append(
                f"| {case_id:02d} | {adaptive['matrix_zyx']} | {fmt(adaptive['first_loaded_images_to_dvf_file_sec'])} / {fmt(adaptive['repeat_loaded_images_to_dvf_file_sec'])} | "
                f"{fmt(adaptive['speedup_loaded_images_to_dvf_file_sec'], 2)}x | {adaptive['first_actual_optimizer_steps']}/{adaptive['repeat_actual_optimizer_steps']} | "
                f"{fmt(fixed['first_loaded_images_to_dvf_file_sec'])} / {fmt(fixed['repeat_loaded_images_to_dvf_file_sec'])} | {fmt(fixed['speedup_loaded_images_to_dvf_file_sec'], 2)}x |"
            )
    lines.extend(["", "## Acceptance", "", f"- Validation errors: `{len(errors)}`"])
    lines.extend([f"- `{error}`" for error in errors] if errors else ["- All 80 fields and 40 adjacent pairs passed the diagnostic contract."])
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows, errors = read_rows(args.run_root)
    pairs = pair_rows(rows, errors)
    aggregates = aggregate_pairs(pairs)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    c.write_csv(args.review_root / "paired_case_rows.csv", pairs)
    c.write_json(args.review_root / "paired_case_rows.json", pairs)
    c.write_csv(args.review_root / "paired_aggregate.csv", aggregates)
    c.write_json(args.review_root / "paired_aggregate.json", aggregates)
    acceptance = {
        "experiment_id": EXPERIMENT_ID,
        "passed": not errors,
        "errors": errors,
        "primary_row_count": len(rows),
        "pair_count": len(pairs),
        "physical_gpu": PHYSICAL_GPU,
        "timing_endpoint": "case_load_complete_to_dvf_file_ready",
    }
    c.write_json(args.review_root / "acceptance.json", acceptance)
    write_report(args.review_root, pairs, aggregates, errors)
    print(json.dumps(acceptance, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
