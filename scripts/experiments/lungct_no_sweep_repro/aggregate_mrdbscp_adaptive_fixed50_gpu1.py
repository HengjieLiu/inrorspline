#!/usr/bin/env python3
"""Validate and aggregate the GPU-1 adaptive versus fixed-50 LungCT study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

import numpy as np

import common as c
from launch_mrdbscp_adaptive_fixed50_gpu1 import (
    CASES,
    DATASETS,
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_WARM_ROWS,
    EXPERIMENT_ID,
    PHYSICAL_GPU,
    VARIANTS,
    variant_spec,
)


TRE_KEYS = {
    "native_crop_tre_snap_mean_mm": "native_crop_tre_snap_to_voxel",
    "native_crop_tre_subvoxel_mean_mm": "native_crop_tre_subvoxel",
    "reg1mm_tre_snap_mean_mm": "tre_snap_to_voxel",
    "reg1mm_tre_subvoxel_mean_mm": "tre_subvoxel",
}
REGULARITY_KEYS = (
    "diffusion_l2_full",
    "diffusion_l2_fg",
    "bending_energy_full",
    "bending_energy_fg",
    "log_jdet_std_full",
    "log_jdet_std_fg",
    "jdet_le0_pct_full",
    "jdet_le0_pct_fg",
)
TIMING_KEYS = (
    "loaded_images_to_dvf_file_sec",
    "loaded_images_to_dvf_in_memory_sec",
    "optimizer_elapsed_sec",
    "full_case_elapsed_sec",
    "peak_mem_gb",
)
QUALITY_KEYS = tuple(TRE_KEYS) + REGULARITY_KEYS + (
    "final_loss",
    "final_data_loss",
    "final_reg_loss",
    "actual_optimizer_steps",
    "peak_mem_gb",
)
P33_HISTORICAL = {
    "4dct": {
        "variant": "balanced_cumulative",
        "native_crop_tre_snap_mean_mm": 0.9178887,
        "cold_loaded_images_to_dvf_file_sec": 103.9,
        "warm_loaded_images_to_dvf_file_sec_mean": 87.98,
        "warm_loaded_images_to_dvf_file_sec_sd": 36.96,
    },
    "copd": {
        "variant": "balanced_ordinal",
        "native_crop_tre_snap_mean_mm": 1.0157553,
        "cold_loaded_images_to_dvf_file_sec": 168.55,
        "warm_loaded_images_to_dvf_file_sec_mean": 77.81,
        "warm_loaded_images_to_dvf_file_sec_sd": 56.61,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    return parser.parse_args(argv)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def mean(values: Iterable[Any]) -> float:
    numbers = [finite(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return float(statistics.mean(numbers)) if numbers else float("nan")


def sample_sd(values: Iterable[Any]) -> float:
    numbers = [finite(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    if len(numbers) >= 2:
        return float(statistics.stdev(numbers))
    return 0.0 if len(numbers) == 1 else float("nan")


def close_list(observed: Any, expected: Sequence[float]) -> bool:
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    return all(math.isclose(finite(left), float(right), rel_tol=1e-10, abs_tol=1e-12) for left, right in zip(observed, expected))


def add_error(errors: list[str], prefix: str, message: str) -> None:
    errors.append(f"{prefix}: {message}")


def provenance_errors(run_root: Path) -> list[str]:
    errors: list[str] = []
    required = (
        run_root / "study_manifest.json",
        run_root / "accepted_arms.json",
        run_root / "preflight_DONE.json",
        run_root / "preflight" / "loader_audit.json",
        run_root / "latest_provenance.json",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"missing provenance file: {path}")
    latest = c.read_json(run_root / "latest_provenance.json") if (run_root / "latest_provenance.json").is_file() else {}
    if isinstance(latest, dict):
        provenance_path = Path(str(latest.get("path", "")))
        manifest_path = provenance_path / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing provenance manifest: {manifest_path}")
        else:
            payload = c.read_json(manifest_path)
            for key in ("git_head", "git_status", "source_sha256", "docker_inspect", "gpu_query", "args"):
                if not isinstance(payload, dict) or key not in payload:
                    errors.append(f"provenance manifest missing {key}")
    loader_path = run_root / "preflight" / "loader_audit.json"
    if loader_path.is_file():
        loader = c.read_json(loader_path)
        if not isinstance(loader, dict) or loader.get("status") != "passed" or int(loader.get("case_count", -1)) != 20:
            errors.append("loader audit did not pass all 20 cases")
    return errors


def accepted_primary_records(run_root: Path, errors: list[str]) -> list[dict[str, Any]]:
    accepted_path = run_root / "accepted_arms.json"
    accepted = c.read_json(accepted_path) if accepted_path.is_file() else {}
    if not isinstance(accepted, dict):
        errors.append("accepted_arms.json is not a dictionary")
        return []
    records: list[dict[str, Any]] = []
    expected_keys = {f"primary:{dataset}:{variant}" for dataset in DATASETS for variant in VARIANTS}
    observed_keys = {key for key in accepted if str(key).startswith("primary:")}
    if observed_keys != expected_keys:
        errors.append(f"accepted primary arm keys mismatch: observed={sorted(observed_keys)} expected={sorted(expected_keys)}")
    for key in sorted(expected_keys):
        record = accepted.get(key)
        if not isinstance(record, dict):
            errors.append(f"missing accepted arm {key}")
            continue
        monitor = record.get("monitor", {})
        if not record.get("accepted") or not isinstance(monitor, dict) or not monitor.get("clean"):
            errors.append(f"arm {key} is not cleanly accepted")
        if record.get("contaminated") or monitor.get("bad_samples"):
            errors.append(f"arm {key} has GPU contamination/query failures")
        if finite(monitor.get("max_sample_gap_sec")) > 5.0:
            errors.append(f"arm {key} monitoring gap exceeds five seconds")
        command = record.get("command", [])
        if not isinstance(command, list) or "CUDA_VISIBLE_DEVICES=1" not in command:
            errors.append(f"arm {key} does not prove physical GPU-1 masking")
        worker = record.get("worker_result", {})
        if not isinstance(worker, dict) or worker.get("status") != "completed" or int(worker.get("row_count", -1)) != 11:
            errors.append(f"arm {key} worker result is incomplete")
        else:
            records.append(record)
    return records


def schedule_errors(summary: dict[str, Any], dataset: str, variant: str) -> list[str]:
    errors: list[str] = []
    spec = variant_spec(dataset, variant)
    training = summary.get("training", {})
    if not isinstance(training, dict):
        return ["missing training payload"]
    exact_lists = {
        "stage_full_cps": spec.stage_full_cps,
        "stage_downsample": spec.stage_downsample,
        "stage_epochs": spec.stage_epochs,
        "stage_lr": spec.stage_lr,
    }
    for key, expected in exact_lists.items():
        if not close_list(training.get(key), expected):
            errors.append(f"{key}={training.get(key)!r}, expected={list(expected)!r}")
    if str(training.get("stage_early_stop_mode")) != spec.early_stop_mode:
        errors.append(f"stage_early_stop_mode={training.get('stage_early_stop_mode')!r}")
    if str(training.get("optimizer")) != "Adam" or int(training.get("seed", -1)) != 1:
        errors.append("optimizer/seed mismatch")
    scalar_expected = {
        "ptv_lcc_sigma_mm": 2.1,
        "ptv_isotv_weight": 0.055,
        "ptv_csqrt": 0.005,
        "border_mask_voxels": 5,
    }
    for key, expected in scalar_expected.items():
        if not math.isclose(finite(training.get(key)), expected, rel_tol=1e-10, abs_tol=1e-12):
            errors.append(f"{key}={training.get(key)!r}, expected={expected}")
    if training.get("prolongation") != "exact crop-aligned cubic B-spline dyadic knot insertion":
        errors.append("exact knot-insertion provenance missing")
    actual = training.get("actual_stage_epochs", [])
    if not isinstance(actual, list) or len(actual) != len(spec.stage_epochs):
        errors.append(f"invalid actual_stage_epochs={actual!r}")
    elif variant == "fixed50" and [int(value) for value in actual] != list(spec.stage_epochs):
        errors.append(f"fixed50 did not execute exact steps: {actual!r}")
    elif variant == "adaptive" and any(
        int(value) < int(minimum) or int(value) > int(cap)
        for value, minimum, cap in zip(actual, spec.min_epochs, spec.stage_epochs)
    ):
        errors.append(f"adaptive actual steps outside min/cap: {actual!r}")
    if variant == "adaptive":
        adaptive_lists = {
            "stage_min_epochs": spec.min_epochs,
            "stage_patience": spec.patience,
            "stage_total_thresholds": spec.total_thresholds,
            "stage_data_thresholds": spec.data_thresholds,
        }
        for key, expected in adaptive_lists.items():
            if not close_list(training.get(key), expected):
                errors.append(f"{key}={training.get(key)!r}, expected={list(expected)!r}")
        if int(training.get("stage_check_interval", -1)) != 25 or int(training.get("stage_window", -1)) != 25:
            errors.append("adaptive check/window interval mismatch")
    expected_trace = [False] * (len(spec.stage_full_cps) - 1) + [True]
    trace = summary.get("full_grid_materialization_trace", training.get("full_grid_materialization_trace"))
    if trace != expected_trace:
        errors.append(f"materialization trace={trace!r}, expected={expected_trace!r}")
    return errors


def validate_dvf(path: Path, input_shape: Any) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing dense DVF"
    try:
        field = np.load(path, mmap_mode="r")
        expected_shape = tuple(int(value) for value in input_shape) + (3,)
        if field.dtype != np.float32:
            return False, f"DVF dtype={field.dtype}, expected=float32"
        if tuple(field.shape) != expected_shape:
            return False, f"DVF shape={field.shape}, expected={expected_shape}"
        if not bool(np.isfinite(field).all()):
            return False, "DVF contains nonfinite values"
    except Exception as exc:
        return False, f"DVF load failed: {exc!r}"
    return True, ""


def read_case_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors = provenance_errors(run_root)
    records = accepted_primary_records(run_root, errors)
    rows: list[dict[str, Any]] = []
    for record in records:
        worker = record["worker_result"]
        for worker_row in worker["rows"]:
            dataset = str(worker_row.get("dataset", ""))
            variant = str(worker_row.get("variant", ""))
            case_id = int(worker_row.get("case_id", -1))
            condition = str(worker_row.get("timing_condition", ""))
            prefix = f"{dataset}/{variant}/case{case_id:02d}/{condition}"
            summary_path = Path(str(worker_row.get("summary_path", "")))
            dvf_path = Path(str(worker_row.get("dense_dvf_path", "")))
            if not summary_path.is_file():
                add_error(errors, prefix, f"missing summary {summary_path}")
                continue
            summary = c.read_json(summary_path)
            if not isinstance(summary, dict):
                add_error(errors, prefix, "summary is not a dictionary")
                continue
            if summary.get("status") != "completed" or summary.get("dataset") != dataset or summary.get("variant_id") != variant:
                add_error(errors, prefix, "summary identity/status mismatch")
            for error in schedule_errors(summary, dataset, variant):
                add_error(errors, prefix, error)
            timing = summary.get("timing", {})
            if not isinstance(timing, dict):
                timing = {}
            if timing.get("timing_mode") != "input_to_dvf" or timing.get("timing_condition") != condition:
                add_error(errors, prefix, "timing mode/condition mismatch")
            if timing.get("timing_endpoint") != "loaded input images to saved dense DVF":
                add_error(errors, prefix, "timing endpoint mismatch")
            row: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "case_id": case_id,
                "case_occurrence": str(worker_row.get("case_occurrence", "")),
                "timing_condition": condition,
                "timing_sequence_index": int(worker_row.get("sequence_index", -1)),
                "run_name": str(worker_row.get("run_name", "")),
                "physical_gpu": PHYSICAL_GPU,
                "summary_path": str(summary_path),
                "dense_dvf_path": str(dvf_path),
                "worker_case_wall_sec": worker_row.get("worker_case_wall_sec"),
                "actual_stage_epochs": json.dumps(summary.get("training", {}).get("actual_stage_epochs", [])),
                "actual_optimizer_steps": sum(int(value) for value in summary.get("training", {}).get("actual_stage_epochs", [])),
                "stage_stop_reasons": json.dumps(summary.get("training", {}).get("stage_stop_reasons", [])),
                "full_grid_materialization_trace": json.dumps(summary.get("full_grid_materialization_trace", [])),
            }
            for key in TIMING_KEYS:
                row[key] = summary.get(key, timing.get(key))
            for output_key, summary_key in TRE_KEYS.items():
                metric = summary.get(summary_key, {})
                if not isinstance(metric, dict) or int(metric.get("n_landmarks", -1)) != 300:
                    add_error(errors, prefix, f"{summary_key} does not account for 300 landmarks")
                    row[output_key] = float("nan")
                else:
                    row[output_key] = metric.get("tre_mean_mm")
            regularity = summary.get("regularity_fg_registration_crop", {})
            if not isinstance(regularity, dict):
                regularity = {}
            for key in REGULARITY_KEYS:
                row[key] = regularity.get(key)
            history = summary.get("history_tail", [])
            final_history = history[-1] if isinstance(history, list) and history and isinstance(history[-1], dict) else {}
            row["final_loss"] = final_history.get("loss")
            row["final_data_loss"] = final_history.get("data_loss")
            row["final_reg_loss"] = final_history.get("reg_loss")
            for key in (*TIMING_KEYS, *TRE_KEYS, *REGULARITY_KEYS, "final_loss", "final_data_loss", "final_reg_loss"):
                if not math.isfinite(finite(row.get(key))):
                    add_error(errors, prefix, f"missing/nonfinite {key}")
            valid_dvf, dvf_error = validate_dvf(dvf_path, summary.get("input_shape_zyx", []))
            if not valid_dvf:
                add_error(errors, prefix, dvf_error)
            rows.append(row)
    return rows, errors


def cohort_errors(rows: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != EXPECTED_PRIMARY_ROWS:
        errors.append(f"primary timing row count={len(rows)}, expected={EXPECTED_PRIMARY_ROWS}")
    names = [str(row["run_name"]) for row in rows]
    paths = [str(row["dense_dvf_path"]) for row in rows]
    if len(set(names)) != len(names):
        errors.append("run names are not unique across the 44 measurements")
    if len(set(paths)) != len(paths):
        errors.append("DVF paths are not unique across the 44 measurements")
    expected_measurements = {
        (dataset, variant, case_id, condition)
        for dataset in DATASETS
        for variant in VARIANTS
        for case_id, condition in ((1, "cold"), (1, "warm"), *((case, "warm") for case in range(2, 11)))
    }
    observed_measurements = {
        (str(row["dataset"]), str(row["variant"]), int(row["case_id"]), str(row["timing_condition"]))
        for row in rows
    }
    if observed_measurements != expected_measurements:
        errors.append("cold/warm measurement keys do not match the frozen 44-row plan")
    warm = [row for row in rows if row["timing_condition"] == "warm"]
    warm_keys = {(row["dataset"], row["variant"], row["case_id"]) for row in warm}
    expected_warm_keys = {(dataset, variant, case_id) for dataset in DATASETS for variant in VARIANTS for case_id in CASES}
    if len(warm) != EXPECTED_WARM_ROWS or warm_keys != expected_warm_keys:
        errors.append(f"warm cohort is not 40 unique dataset/variant/case results: rows={len(warm)}, keys={len(warm_keys)}")
    if len([row for row in rows if row["timing_condition"] == "cold"]) != 4:
        errors.append("cold QA field count is not four")
    return errors


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    timing_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            arm = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            cold = [row for row in arm if row["timing_condition"] == "cold"]
            warm = [row for row in arm if row["timing_condition"] == "warm"]
            warm_case1 = [row for row in warm if int(row["case_id"]) == 1]
            timing: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "cold_case_count": len(cold),
                "warm_case_count": len(warm),
                "cold_warm_case01_gap_sec": finite(cold[0]["loaded_images_to_dvf_file_sec"]) - finite(warm_case1[0]["loaded_images_to_dvf_file_sec"]) if cold and warm_case1 else float("nan"),
                "cold_case01_actual_optimizer_steps": cold[0]["actual_optimizer_steps"] if cold else None,
                "warm_case01_actual_optimizer_steps": warm_case1[0]["actual_optimizer_steps"] if warm_case1 else None,
                "warm_actual_optimizer_steps_mean": mean(row["actual_optimizer_steps"] for row in warm),
                "warm_actual_optimizer_steps_sd": sample_sd(row["actual_optimizer_steps"] for row in warm),
            }
            for key in TIMING_KEYS:
                timing[f"cold_{key}"] = finite(cold[0][key]) if cold else float("nan")
                timing[f"warm_{key}_mean"] = mean(row[key] for row in warm)
                timing[f"warm_{key}_sd"] = sample_sd(row[key] for row in warm)
            timing_rows.append(timing)
            quality: dict[str, Any] = {"dataset": dataset, "variant": variant, "case_count": len(warm)}
            for key in QUALITY_KEYS:
                quality[f"{key}_mean"] = mean(row[key] for row in warm)
                quality[f"{key}_sd"] = sample_sd(row[key] for row in warm)
            quality_rows.append(quality)
            for row in arm:
                summary = c.read_json(Path(row["summary_path"]))
                training = summary.get("training", {}) if isinstance(summary, dict) else {}
                stage_summaries = summary.get("stage_summaries", []) if isinstance(summary, dict) else []
                for index, requested in enumerate(training.get("stage_epochs", [])):
                    stage = stage_summaries[index] if index < len(stage_summaries) else {}
                    stage_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "case_id": row["case_id"],
                            "timing_condition": row["timing_condition"],
                            "stage_idx": index,
                            "requested_epochs": requested,
                            "actual_epochs": training.get("actual_stage_epochs", [])[index],
                            "stop_reason": training.get("stage_stop_reasons", [])[index],
                            "full_grid_materialized": stage.get("full_grid_dvf_materialized") if isinstance(stage, dict) else None,
                        }
                    )
        adaptive_quality = next(row for row in quality_rows if row["dataset"] == dataset and row["variant"] == "adaptive")
        fixed_quality = next(row for row in quality_rows if row["dataset"] == dataset and row["variant"] == "fixed50")
        adaptive_timing = next(row for row in timing_rows if row["dataset"] == dataset and row["variant"] == "adaptive")
        fixed_timing = next(row for row in timing_rows if row["dataset"] == dataset and row["variant"] == "fixed50")
        comparison: dict[str, Any] = {
            "dataset": dataset,
            "reference": "adaptive",
            "comparison": "fixed50",
            "warm_input_to_dvf_speedup_adaptive_over_fixed50": finite(adaptive_timing["warm_loaded_images_to_dvf_file_sec_mean"]) / finite(fixed_timing["warm_loaded_images_to_dvf_file_sec_mean"]),
            "warm_input_to_dvf_delta_fixed50_minus_adaptive_sec": finite(fixed_timing["warm_loaded_images_to_dvf_file_sec_mean"]) - finite(adaptive_timing["warm_loaded_images_to_dvf_file_sec_mean"]),
        }
        for key in QUALITY_KEYS:
            comparison[f"{key}_delta_fixed50_minus_adaptive"] = finite(fixed_quality[f"{key}_mean"]) - finite(adaptive_quality[f"{key}_mean"])
        comparison_rows.append(comparison)
    return timing_rows, quality_rows, comparison_rows, stage_rows


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    review_root: Path,
    rows: Sequence[dict[str, Any]],
    timing_rows: Sequence[dict[str, Any]],
    quality_rows: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# GPU-1 LungCT MR-D-BSCP adaptive versus fixed-50",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Technical acceptance: `{'PASS' if not errors else 'FAIL'}`",
        f"- Primary measurements: `{len(rows)}/{EXPECTED_PRIMARY_ROWS}`; warm quality cohort: `{len([row for row in rows if row['timing_condition'] == 'warm'])}/{EXPECTED_WARM_ROWS}`.",
        "- Headline timing: synchronized `case_load_complete -> dvf_file_ready`, including final field materialization, CPU transfer, and DVF serialization; TRE/regularity evaluation is excluded.",
        "- Quality aggregates use the 40 warm fields only. The four cold case-01 fields are retained for QA.",
        "- Every accepted arm ran on physical GPU 1, exposed to its worker as logical CUDA device 0, with clean NVML monitoring.",
        "",
        "## Warm timing and cold/warm gap",
        "",
        "The cold/warm gap is descriptive; cold and warm case-01 optimizer-step counts are shown because adaptive stopping can differ between repeats.",
        "",
        "| Dataset | Variant | Warm cases | Warm steps | Input -> DVF s | In-memory s | Optimizer s | Peak GB | Cold/warm case01 steps | Cold case01 s | Warm case01 s | Cold-warm gap s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    case1_warm = {(row["dataset"], row["variant"]): row for row in rows if row["timing_condition"] == "warm" and int(row["case_id"]) == 1}
    for row in timing_rows:
        warm1 = case1_warm[(row["dataset"], row["variant"])]
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['warm_case_count']} | {fmt(row['warm_actual_optimizer_steps_mean'], 1)} +/- {fmt(row['warm_actual_optimizer_steps_sd'], 1)} | "
            f"{fmt(row['warm_loaded_images_to_dvf_file_sec_mean'])} +/- {fmt(row['warm_loaded_images_to_dvf_file_sec_sd'])} | "
            f"{fmt(row['warm_loaded_images_to_dvf_in_memory_sec_mean'])} +/- {fmt(row['warm_loaded_images_to_dvf_in_memory_sec_sd'])} | "
            f"{fmt(row['warm_optimizer_elapsed_sec_mean'])} +/- {fmt(row['warm_optimizer_elapsed_sec_sd'])} | {fmt(row['warm_peak_mem_gb_mean'])} | "
            f"{int(row['cold_case01_actual_optimizer_steps'])}/{int(row['warm_case01_actual_optimizer_steps'])} | "
            f"{fmt(row['cold_loaded_images_to_dvf_file_sec'])} | {fmt(warm1['loaded_images_to_dvf_file_sec'])} | {fmt(row['cold_warm_case01_gap_sec'])} |"
        )
    lines.extend([
        "",
        "## Warm-cohort quality",
        "",
        "Native-crop snap-to-voxel TRE is headline; all other TRE conventions are diagnostics.",
        "",
        "| Dataset | Variant | Cases | Steps | Native snap TRE mm | Native subvoxel mm | Reg-grid snap mm | Reg-grid subvoxel mm | FG SDlogJ | FG fold % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in quality_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['case_count']} | {fmt(row['actual_optimizer_steps_mean'], 1)} +/- {fmt(row['actual_optimizer_steps_sd'], 1)} | "
            f"{fmt(row['native_crop_tre_snap_mean_mm_mean'])} +/- {fmt(row['native_crop_tre_snap_mean_mm_sd'])} | "
            f"{fmt(row['native_crop_tre_subvoxel_mean_mm_mean'])} +/- {fmt(row['native_crop_tre_subvoxel_mean_mm_sd'])} | "
            f"{fmt(row['reg1mm_tre_snap_mean_mm_mean'])} +/- {fmt(row['reg1mm_tre_snap_mean_mm_sd'])} | "
            f"{fmt(row['reg1mm_tre_subvoxel_mean_mm_mean'])} +/- {fmt(row['reg1mm_tre_subvoxel_mean_mm_sd'])} | "
            f"{fmt(row['log_jdet_std_fg_mean'], 4)} | {fmt(row['jdet_le0_pct_fg_mean'], 4)} |"
        )
    lines.extend([
        "",
        "## Warm-cohort regularity, loss, and memory",
        "",
        "| Dataset | Variant | Diffusion full / FG | Bending full / FG | SDlogJ full / FG | Fold full / FG % | Final loss | Peak GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in quality_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {fmt(row['diffusion_l2_full_mean'], 5)} / {fmt(row['diffusion_l2_fg_mean'], 5)} | "
            f"{fmt(row['bending_energy_full_mean'], 5)} / {fmt(row['bending_energy_fg_mean'], 5)} | "
            f"{fmt(row['log_jdet_std_full_mean'], 4)} / {fmt(row['log_jdet_std_fg_mean'], 4)} | "
            f"{fmt(row['jdet_le0_pct_full_mean'], 4)} / {fmt(row['jdet_le0_pct_fg_mean'], 4)} | "
            f"{fmt(row['final_loss_mean'], 1)} | {fmt(row['peak_mem_gb_mean'], 3)} |"
        )
    lines.extend([
        "",
        "Sample SDs, final data/regularization loss components, stage counts, and casewise values are in the CSV/JSON artifacts beside this report.",
        "",
        "## Direct fixed-50 comparison",
        "",
        "Positive speedup means fixed-50 is faster. Quality deltas are fixed-50 minus adaptive.",
        "",
        "| Dataset | Warm speedup | Timing delta s | Native snap TRE delta mm | FG SDlogJ delta | FG fold delta pp |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row['warm_input_to_dvf_speedup_adaptive_over_fixed50'], 2)}x | {fmt(row['warm_input_to_dvf_delta_fixed50_minus_adaptive_sec'])} | "
            f"{fmt(row['native_crop_tre_snap_mean_mm_delta_fixed50_minus_adaptive'])} | {fmt(row['log_jdet_std_fg_delta_fixed50_minus_adaptive'], 4)} | "
            f"{fmt(row['jdet_le0_pct_fg_delta_fixed50_minus_adaptive'], 4)} |"
        )
    lines.extend([
        "",
        "## Historical P33 context (not matched timing evidence)",
        "",
        "P33 used a different cold/warm layout and mixed GPUs. Its values are context for quality/reproducibility only; this report does not claim a matched speedup against the old 2,500-step runs.",
        "",
        "| Dataset | Historical selected variant | Native snap TRE mm | Cold s | Warm s |",
        "|---|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        historical = P33_HISTORICAL[dataset]
        lines.append(
            f"| {dataset} | {historical['variant']} | {fmt(historical['native_crop_tre_snap_mean_mm'])} | {fmt(historical['cold_loaded_images_to_dvf_file_sec'])} | "
            f"{fmt(historical['warm_loaded_images_to_dvf_file_sec_mean'])} +/- {fmt(historical['warm_loaded_images_to_dvf_file_sec_sd'])} |"
        )
    lines.extend(["", "## Acceptance", "", f"- Validation errors: `{len(errors)}`"])
    lines.extend([f"- `{error}`" for error in errors] if errors else ["- All 44 timings and fields passed the frozen technical acceptance contract."])
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows, errors = read_case_rows(args.run_root)
    errors.extend(cohort_errors(rows))
    timing_rows, quality_rows, comparison_rows, stage_rows = aggregate_rows(rows)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    c.write_csv(args.review_root / "timing_aggregate.csv", timing_rows)
    c.write_json(args.review_root / "timing_aggregate.json", timing_rows)
    c.write_csv(args.review_root / "quality_aggregate.csv", quality_rows)
    c.write_json(args.review_root / "quality_aggregate.json", quality_rows)
    c.write_csv(args.review_root / "fixed50_vs_adaptive.csv", comparison_rows)
    c.write_json(args.review_root / "fixed50_vs_adaptive.json", comparison_rows)
    c.write_csv(args.review_root / "stage_rows.csv", stage_rows)
    c.write_json(args.review_root / "stage_rows.json", stage_rows)
    c.write_json(args.review_root / "p33_historical_context.json", P33_HISTORICAL)
    acceptance = {
        "experiment_id": EXPERIMENT_ID,
        "passed": not errors,
        "errors": errors,
        "primary_row_count": len(rows),
        "warm_row_count": len([row for row in rows if row["timing_condition"] == "warm"]),
        "cold_row_count": len([row for row in rows if row["timing_condition"] == "cold"]),
        "physical_gpu": PHYSICAL_GPU,
        "headline_tre": "native_crop_tre_snap_to_voxel",
        "timing_endpoint": "case_load_complete_to_dvf_file_ready",
        "historical_comparison_policy": "P33 context only; no matched speedup claim",
    }
    c.write_json(args.review_root / "acceptance.json", acceptance)
    write_report(args.review_root, rows, timing_rows, quality_rows, comparison_rows, errors)
    print(json.dumps(acceptance, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
