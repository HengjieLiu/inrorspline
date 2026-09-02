#!/usr/bin/env python
"""Aggregate casewise TRE and cold/warm DVF timing for the LungCT campaign."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import common as c
from launch_mrdbscp_acceleration_timing import CASES, DATASETS, EXPERIMENT_ID, VARIANT_IDS, variant_spec


def finite(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mean(values: Iterable[object]) -> float:
    vals = [finite(value) for value in values]
    vals = [value for value in vals if math.isfinite(value)]
    return float(statistics.mean(vals)) if vals else float("nan")


def sample_sd(values: Iterable[object]) -> float:
    vals = [finite(value) for value in values]
    vals = [value for value in vals if math.isfinite(value)]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0 if len(vals) == 1 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_rows(run_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(Path(run_root).rglob("summary.json")):
        try:
            summary = c.read_json(summary_path)
        except Exception:
            continue
        if not isinstance(summary, dict) or str(summary.get("experiment_id", "")) not in {"", EXPERIMENT_ID}:
            continue
        dataset = str(summary.get("dataset", ""))
        variant_id = str(summary.get("variant_id", ""))
        if dataset not in DATASETS or variant_id not in VARIANT_IDS:
            continue
        timing = summary.get("timing", {})
        if not isinstance(timing, dict):
            timing = {}
        training = summary.get("training", {})
        if not isinstance(training, dict):
            training = {}
        trace = summary.get("full_grid_materialization_trace", training.get("full_grid_materialization_trace"))
        expected_trace = [False] * (len(variant_spec(dataset, variant_id).stage_full_cps) - 1) + [True]
        rows.append(
            {
                "dataset": dataset,
                "variant_id": variant_id,
                "case_id": int(summary.get("case_id", -1)),
                "data_root": str(summary.get("data_root", "")),
                "status": str(summary.get("status", "")),
                "summary_path": str(summary_path),
                "timing_path": str(summary_path.parent / "timing.json"),
                "dense_dvf_path": str(summary_path.parent / "disp_norm_xyz_reg1mm.npy"),
                "timing_condition": str(timing.get("timing_condition", "unspecified")),
                "timing_sequence_index": timing.get("timing_sequence_index"),
                "loaded_images_to_dvf_file_sec": timing.get("loaded_images_to_dvf_file_sec", summary.get("loaded_images_to_dvf_file_sec")),
                "loaded_images_to_dvf_in_memory_sec": timing.get("loaded_images_to_dvf_in_memory_sec", summary.get("loaded_images_to_dvf_in_memory_sec")),
                "full_script_to_dvf_file_sec": timing.get("full_script_to_dvf_file_sec", summary.get("full_script_to_dvf_file_sec")),
                "optimizer_elapsed_sec": timing.get("optimizer_elapsed_sec", summary.get("optimizer_elapsed_sec")),
                "full_case_elapsed_sec": timing.get("full_case_elapsed_sec", summary.get("full_case_elapsed_sec")),
                "post_dvf_evaluation_sec": timing.get("post_dvf_evaluation_sec"),
                "native_crop_tre_snap_mean_mm": summary.get("native_crop_tre_snap_to_voxel", {}).get("tre_mean_mm"),
                "native_crop_tre_subvoxel_mean_mm": summary.get("native_crop_tre_subvoxel", {}).get("tre_mean_mm"),
                "reg1mm_tre_snap_mean_mm": summary.get("tre_snap_to_voxel", {}).get("tre_mean_mm"),
                "reg1mm_tre_subvoxel_mean_mm": summary.get("tre_subvoxel", {}).get("tre_mean_mm"),
                "actual_stage_epochs": json.dumps(training.get("actual_stage_epochs", [])),
                "requested_optimizer_steps": sum(int(value) for value in training.get("stage_epochs", []) if isinstance(value, (int, float))),
                "actual_optimizer_steps": sum(int(value) for value in training.get("actual_stage_epochs", []) if isinstance(value, (int, float))),
                "stage_stop_reasons": json.dumps(training.get("stage_stop_reasons", [])),
                "full_grid_materialization_trace": json.dumps(trace),
                "expected_full_grid_materialization_trace": json.dumps(expected_trace),
                "materialization_trace_pass": trace == expected_trace,
                "peak_mem_gb": summary.get("peak_mem_gb"),
            }
        )
    return rows


def add_baseline_comparisons(rows: Sequence[Dict[str, object]]) -> None:
    baseline_by_case = {
        (str(row["dataset"]), int(row["case_id"])): row
        for row in rows
        if str(row["variant_id"]) == "original_corrected"
    }
    metrics = (
        "native_crop_tre_snap_mean_mm",
        "native_crop_tre_subvoxel_mean_mm",
        "reg1mm_tre_snap_mean_mm",
        "reg1mm_tre_subvoxel_mean_mm",
        "loaded_images_to_dvf_file_sec",
        "optimizer_elapsed_sec",
        "full_case_elapsed_sec",
    )
    for row in rows:
        baseline = baseline_by_case.get((str(row["dataset"]), int(row["case_id"])))
        for metric in metrics:
            baseline_value = baseline.get(metric) if baseline is not None else float("nan")
            current = finite(row.get(metric))
            reference = finite(baseline_value)
            row[f"original_{metric}"] = reference
            row[f"delta_{metric}_vs_original"] = current - reference if math.isfinite(current) and math.isfinite(reference) else float("nan")
        current_time = finite(row.get("loaded_images_to_dvf_file_sec"))
        reference_time = finite(baseline.get("loaded_images_to_dvf_file_sec")) if baseline is not None else float("nan")
        row["input_to_dvf_speedup_vs_original"] = reference_time / current_time if math.isfinite(reference_time) and math.isfinite(current_time) and current_time > 0.0 else float("nan")


def read_stage_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    stage_rows: List[Dict[str, object]] = []
    for case_row in rows:
        summary = c.read_json(Path(str(case_row["summary_path"])))
        if not isinstance(summary, dict):
            continue
        training = summary.get("training", {})
        if not isinstance(training, dict):
            training = {}
        requested = training.get("stage_epochs", [])
        actual = training.get("actual_stage_epochs", [])
        reasons = training.get("stage_stop_reasons", [])
        summaries = summary.get("stage_summaries", [])
        if not isinstance(requested, list):
            requested = []
        if not isinstance(actual, list):
            actual = []
        if not isinstance(reasons, list):
            reasons = []
        if not isinstance(summaries, list):
            summaries = []
        for stage_idx, requested_epochs in enumerate(requested):
            stage_summary = summaries[stage_idx] if stage_idx < len(summaries) and isinstance(summaries[stage_idx], dict) else {}
            stage_rows.append(
                {
                    "dataset": case_row["dataset"],
                    "variant_id": case_row["variant_id"],
                    "case_id": case_row["case_id"],
                    "stage_idx": stage_idx,
                    "requested_epochs": int(requested_epochs),
                    "actual_epochs": int(actual[stage_idx]) if stage_idx < len(actual) else None,
                    "stop_reason": str(reasons[stage_idx]) if stage_idx < len(reasons) else "",
                    "full_grid_materialized": stage_summary.get("full_grid_dvf_materialized"),
                    "full_grid_materialization_trace": json.dumps(summary.get("full_grid_materialization_trace", [])),
                    "summary_path": case_row["summary_path"],
                }
            )
    return stage_rows


def format_shape(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "x".join(str(int(item)) for item in value)


def read_geometry_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    geometry_rows: List[Dict[str, object]] = []
    seen = set()
    for case_row in rows:
        key = (str(case_row["dataset"]), int(case_row["case_id"]))
        if key in seen:
            continue
        summary = c.read_json(Path(str(case_row["summary_path"])))
        if not isinstance(summary, dict):
            continue
        training = summary.get("training", {})
        if not isinstance(training, dict):
            continue
        stage_cps = training.get("stage_full_cps", [])
        nested_shapes = training.get("nested_control_shapes_zyx", {})
        if not isinstance(stage_cps, list) or not isinstance(nested_shapes, dict):
            continue
        row: Dict[str, object] = {
            "dataset": case_row["dataset"],
            "case_id": case_row["case_id"],
            "matrix_size_zyx": format_shape(summary.get("input_shape_zyx")),
        }
        for stage_idx, cps in enumerate(stage_cps):
            shape = nested_shapes.get(str(cps), nested_shapes.get(cps, []))
            control_points = 1
            if isinstance(shape, list) and shape:
                for dimension in shape:
                    control_points *= int(dimension)
            row[f"stage_{stage_idx}"] = f"cps={int(cps)}; grid={format_shape(shape)}; control_points={control_points}"
        geometry_rows.append(row)
        seen.add(key)
    return sorted(geometry_rows, key=lambda row: (str(row["dataset"]), int(row["case_id"])))


def read_lung_volume_rows(rows: Sequence[Dict[str, object]], data_root: Path | None = None) -> List[Dict[str, object]]:
    roots = {Path(str(row.get("data_root", ""))) for row in rows if str(row.get("data_root", ""))}
    root = Path(data_root) if data_root is not None else next(iter(roots), None)
    if root is None:
        raise RuntimeError("Cannot compute lung-mask volumes without a data root.")
    if data_root is None and len(roots) != 1:
        raise RuntimeError(f"Expected one campaign data root, found {sorted(str(item) for item in roots)}")
    volume_rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for case_id in CASES:
            case = c.load_case(root, int(case_id), dataset=dataset)
            inspiration_voxels = int(case.fixed_mask.astype(bool).sum())
            expiration_voxels = int(case.moving_mask.astype(bool).sum())
            voxel_volume_ml = 0.001
            inspiration_ml = inspiration_voxels * voxel_volume_ml
            expiration_ml = expiration_voxels * voxel_volume_ml
            difference_ml = inspiration_ml - expiration_ml
            volume_change_percent = 100.0 * difference_ml / expiration_ml if expiration_ml > 0.0 else float("nan")
            volume_rows.append(
                {
                    "dataset": dataset,
                    "case_id": int(case_id),
                    "inspiration_label_fixed": str(case.fixed_label),
                    "expiration_label_moving": str(case.moving_label),
                    "matrix_size_zyx": format_shape(case.fixed.shape),
                    "voxel_volume_ml": voxel_volume_ml,
                    "inspiration_mask_voxels": inspiration_voxels,
                    "expiration_mask_voxels": expiration_voxels,
                    "inspiration_volume_ml": inspiration_ml,
                    "expiration_volume_ml": expiration_ml,
                    "inspiration_minus_expiration_ml": difference_ml,
                    "percent_change_from_expiration_to_inspiration": volume_change_percent,
                }
            )
    return volume_rows


def validate(rows: Sequence[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    expected_keys = {(dataset, variant, case_id) for dataset in DATASETS for variant in VARIANT_IDS for case_id in CASES}
    observed_keys = {(str(row["dataset"]), str(row["variant_id"]), int(row["case_id"])) for row in rows}
    missing = sorted(expected_keys - observed_keys)
    duplicate_keys = sorted(key for key in observed_keys if sum((str(row["dataset"]), str(row["variant_id"]), int(row["case_id"])) == key for row in rows) > 1)
    if missing:
        errors.append(f"missing rows: {missing}")
    if duplicate_keys:
        errors.append(f"duplicate rows: {duplicate_keys}")
    for row in rows:
        prefix = f"{row['dataset']}/{row['variant_id']}/case{int(row['case_id']):02d}"
        if row.get("status") != "completed":
            errors.append(f"{prefix}: status={row.get('status')}")
        if not bool(row.get("materialization_trace_pass")):
            errors.append(f"{prefix}: materialization trace mismatch")
        for key in ("loaded_images_to_dvf_file_sec", "optimizer_elapsed_sec", "full_case_elapsed_sec", "native_crop_tre_snap_mean_mm"):
            if not math.isfinite(finite(row.get(key))):
                errors.append(f"{prefix}: missing/nonfinite {key}")
        if row.get("timing_condition") not in {"cold", "warm"}:
            errors.append(f"{prefix}: invalid timing condition {row.get('timing_condition')}")
        for path_key in ("timing_path", "dense_dvf_path"):
            if not Path(str(row[path_key])).exists():
                errors.append(f"{prefix}: missing {path_key}={row[path_key]}")
        try:
            actual_epochs = json.loads(str(row.get("actual_stage_epochs", "[]")))
            requested_epochs = list(variant_spec(str(row["dataset"]), str(row["variant_id"])).stage_epochs)
            if len(actual_epochs) != len(requested_epochs) or any(
                int(actual) < 1 or int(actual) > int(requested)
                for actual, requested in zip(actual_epochs, requested_epochs)
            ):
                errors.append(f"{prefix}: invalid actual stage epoch counts {actual_epochs}; requested={requested_epochs}")
        except Exception as exc:
            errors.append(f"{prefix}: invalid stage epoch payload: {exc!r}")
    return errors


def aggregate_timing(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = (
        "loaded_images_to_dvf_file_sec",
        "loaded_images_to_dvf_in_memory_sec",
        "full_script_to_dvf_file_sec",
        "optimizer_elapsed_sec",
        "full_case_elapsed_sec",
        "post_dvf_evaluation_sec",
    )
    output: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for variant_id in VARIANT_IDS:
            selected = [row for row in rows if row["dataset"] == dataset and row["variant_id"] == variant_id]
            for group, group_rows in (
                ("cold", [row for row in selected if row.get("timing_condition") == "cold"]),
                ("warm", [row for row in selected if row.get("timing_condition") == "warm"]),
                ("all_10", selected),
            ):
                record: Dict[str, object] = {
                    "dataset": dataset,
                    "variant_id": variant_id,
                    "group": group,
                    "case_count": len(group_rows),
                    "stage_epochs": ",".join(str(value) for value in variant_spec(dataset, variant_id).stage_epochs),
                }
                for metric in metrics:
                    record[f"{metric}_mean"] = mean(row.get(metric) for row in group_rows)
                    record[f"{metric}_sd"] = sample_sd(row.get(metric) for row in group_rows)
                output.append(record)
    baseline_by_group = {
        (str(row["dataset"]), str(row["group"])): row
        for row in output
        if str(row["variant_id"]) == "original_corrected"
    }
    for row in output:
        baseline = baseline_by_group[(str(row["dataset"]), str(row["group"]))]
        primary = finite(row["loaded_images_to_dvf_file_sec_mean"])
        baseline_primary = finite(baseline["loaded_images_to_dvf_file_sec_mean"])
        row["loaded_images_to_dvf_file_sec_baseline_mean"] = baseline_primary
        row["loaded_images_to_dvf_file_sec_delta_vs_original"] = primary - baseline_primary
        row["loaded_images_to_dvf_file_speedup_vs_original"] = baseline_primary / primary if math.isfinite(primary) and primary > 0.0 else float("nan")
    return output


def aggregate_tre(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = ("native_crop_tre_snap_mean_mm", "native_crop_tre_subvoxel_mean_mm", "reg1mm_tre_snap_mean_mm", "reg1mm_tre_subvoxel_mean_mm")
    output: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for variant_id in VARIANT_IDS:
            selected = [row for row in rows if row["dataset"] == dataset and row["variant_id"] == variant_id]
            record: Dict[str, object] = {
                "dataset": dataset,
                "variant_id": variant_id,
                "case_count": len(selected),
                "stage_epochs": ",".join(str(value) for value in variant_spec(dataset, variant_id).stage_epochs),
                "requested_optimizer_steps": sum(variant_spec(dataset, variant_id).stage_epochs),
                "actual_optimizer_steps_mean": mean(row.get("actual_optimizer_steps") for row in selected),
                "actual_optimizer_steps_sd": sample_sd(row.get("actual_optimizer_steps") for row in selected),
            }
            for metric in metrics:
                record[f"{metric}_mean"] = mean(row.get(metric) for row in selected)
                record[f"{metric}_sd"] = sample_sd(row.get(metric) for row in selected)
            output.append(record)
    baseline_by_dataset = {
        str(row["dataset"]): row
        for row in output
        if str(row["variant_id"]) == "original_corrected"
    }
    for row in output:
        baseline = baseline_by_dataset[str(row["dataset"])]
        for metric in metrics:
            current = finite(row[f"{metric}_mean"])
            reference = finite(baseline[f"{metric}_mean"])
            row[f"{metric}_baseline_mean"] = reference
            row[f"{metric}_delta_vs_original"] = current - reference
    return output


def fmt(value: object, digits: int = 3) -> str:
    number = finite(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    review_root: Path,
    rows: Sequence[Dict[str, object]],
    tre_rows: Sequence[Dict[str, object]],
    timing_rows: Sequence[Dict[str, object]],
    stage_rows: Sequence[Dict[str, object]],
    geometry_rows: Sequence[Dict[str, object]],
    lung_volume_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# LungCT MR-D-BSCP 1 mm acceleration and timing",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Case rows: `{len(rows)}/120`",
        "- Timing endpoint: loaded input images to saved dense DVF (`loaded_images_to_dvf_file_sec`).",
        "- Cold/warm timing: each variant/dataset worker uses case 1 as cold and cases 2–10 as warm; controls and optimizer state are reset per case.",
        "- Queue time is excluded from per-case registration timing.",
        "- Multistage optimization keeps per-epoch stage-grid expansion and disables non-final full-grid expansion.",
        "",
        "## TRE aggregate",
        "",
        "| Dataset | Variant | Cases | Steps mean +/- SD | Native snap TRE mm | Delta vs original | Reg-1mm snap TRE mm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in tre_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant_id']} | {row['case_count']} | {fmt(row['actual_optimizer_steps_mean'], 1)} +/- {fmt(row['actual_optimizer_steps_sd'], 1)} | "
            f"{fmt(row['native_crop_tre_snap_mean_mm_mean'])} +/- {fmt(row['native_crop_tre_snap_mean_mm_sd'])} | "
            f"{fmt(row['native_crop_tre_snap_mean_mm_delta_vs_original'])} | "
            f"{fmt(row['reg1mm_tre_snap_mean_mm_mean'])} +/- {fmt(row['reg1mm_tre_snap_mean_mm_sd'])} |"
        )
    lines.extend(["", "## Casewise comparison against fresh original 2500 baseline", "", "Deltas are variant minus the same-dataset, same-case original rerun; negative timing delta means faster.", ""])
    for dataset in DATASETS:
        lines.extend([
            f"### {dataset}",
            "",
            "| Case | Variant | Actual steps | Native snap TRE mm | Delta TRE | Input -> DVF s | Delta timing s | Speedup |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        selected = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: (int(row["case_id"]), VARIANT_IDS.index(str(row["variant_id"]))),
        )
        for row in selected:
            lines.append(
                f"| {int(row['case_id']):02d} | {row['variant_id']} | {int(row['actual_optimizer_steps'])} | {fmt(row['native_crop_tre_snap_mean_mm'])} | "
                f"{fmt(row['delta_native_crop_tre_snap_mean_mm_vs_original'])} | "
                f"{fmt(row['loaded_images_to_dvf_file_sec'])} | {fmt(row['delta_loaded_images_to_dvf_file_sec_vs_original'])} | "
                f"{fmt(row['input_to_dvf_speedup_vs_original'])} |"
            )
    lines.extend(["", "## Per-case matrix and control-point geometry", "", "Matrix sizes are recorded as Z x Y x X. Each stage entry gives the stage control-point spacing (`cps`), control-grid matrix, and total control-point count.", ""])
    for dataset in DATASETS:
        dataset_geometry = [row for row in geometry_rows if row["dataset"] == dataset]
        stage_columns = sorted(
            {key for row in dataset_geometry for key in row if str(key).startswith("stage_")},
            key=lambda key: int(str(key).split("_")[-1]),
        )
        lines.extend([
            f"### {dataset}",
            "",
            "| Case | Matrix size Z x Y x X | " + " | ".join(f"Stage {int(key.split('_')[-1])}" for key in stage_columns) + " |",
            "|---:|---|" + "---|" * len(stage_columns),
        ])
        for row in dataset_geometry:
            lines.append(
                "| " + str(int(row["case_id"])).zfill(2) + " | " + str(row["matrix_size_zyx"]) + " | "
                + " | ".join(str(row.get(key, "")) for key in stage_columns)
                + " |"
            )
    lines.extend(["", "## Lung-mask volumes", "", "Volumes use the 1 mm isotropic masks: inspiration is the fixed mask and expiration is the moving mask. Labels are 4DCT fixed `T00` / moving `T50` and COPD fixed `iBH` / moving `eBH`. Percentage change is `(inspiration - expiration) / expiration * 100`.", ""])
    for dataset in DATASETS:
        lines.extend([
            f"### {dataset}",
            "",
            "| Case | Matrix size Z x Y x X | Inspiration mask voxels | Expiration mask voxels | Inspiration volume mL | Expiration volume mL | Difference mL | Percentage change vs expiration |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in [item for item in lung_volume_rows if item["dataset"] == dataset]:
            lines.append(
                f"| {int(row['case_id']):02d} | {row['matrix_size_zyx']} | {row['inspiration_mask_voxels']} | {row['expiration_mask_voxels']} | "
                f"{fmt(row['inspiration_volume_ml'], 3)} | {fmt(row['expiration_volume_ml'], 3)} | "
                f"{fmt(row['inspiration_minus_expiration_ml'], 3)} | {fmt(row['percent_change_from_expiration_to_inspiration'], 2)}% |"
            )
    lines.extend(["", "## Timing aggregate", "", "Primary timing is mean +/- sample SD; delta and speedup compare against the fresh original 2500 baseline for the same dataset and timing group.", "", "| Dataset | Variant | Group | Cases | Input -> DVF s | Delta s | Speedup | Full case s | Optimizer s |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in timing_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant_id']} | {row['group']} | {row['case_count']} | "
            f"{fmt(row['loaded_images_to_dvf_file_sec_mean'])} +/- {fmt(row['loaded_images_to_dvf_file_sec_sd'])} | "
            f"{fmt(row['loaded_images_to_dvf_file_sec_delta_vs_original'])} | "
            f"{fmt(row['loaded_images_to_dvf_file_speedup_vs_original'])} | "
            f"{fmt(row['full_case_elapsed_sec_mean'])} +/- {fmt(row['full_case_elapsed_sec_sd'])} | "
            f"{fmt(row['optimizer_elapsed_sec_mean'])} +/- {fmt(row['optimizer_elapsed_sec_sd'])} |"
        )
    trace_counts = collections.Counter(str(row.get("full_grid_materialization_trace", "")) for row in rows)
    stop_counts = collections.Counter(str(row.get("stage_stop_reasons", "")) for row in rows)
    lines.extend([
        "",
        "## Stage and materialization validation",
        "",
        f"- Stage rows written: `{len(stage_rows)}`; detailed table: `stage_rows.csv`.",
        f"- Case trace counts: `{dict(trace_counts)}`.",
        f"- Case stop-reason payloads: `{len(stop_counts)}` distinct; exact stage reasons are in `stage_rows.csv`.",
        "- Expected final-only traces: 4DCT `[false,false,false,true]`; COPD `[false,false,false,false,true]`.",
        "",
        "## Validation",
        "",
        f"- Validation errors: `{len(errors)}`",
    ])
    if errors:
        lines.extend(f"- `{error}`" for error in errors)
    else:
        lines.append("- All 120 rows are complete, finite, and satisfy the final-only materialization trace.")
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.run_root)
    add_baseline_comparisons(rows)
    errors = validate(rows)
    tre_rows = aggregate_tre(rows)
    timing_rows = aggregate_timing(rows)
    stage_rows = read_stage_rows(rows)
    geometry_rows = read_geometry_rows(rows)
    lung_volume_rows = read_lung_volume_rows(rows, args.data_root)
    expected_geometry_keys = {(dataset, case_id) for dataset in DATASETS for case_id in CASES}
    observed_geometry_keys = {(str(row["dataset"]), int(row["case_id"])) for row in geometry_rows}
    if expected_geometry_keys - observed_geometry_keys:
        errors.append(f"missing geometry rows: {sorted(expected_geometry_keys - observed_geometry_keys)}")
    expected_volume_keys = {(dataset, case_id) for dataset in DATASETS for case_id in CASES}
    observed_volume_keys = {(str(row["dataset"]), int(row["case_id"])) for row in lung_volume_rows}
    if expected_volume_keys - observed_volume_keys:
        errors.append(f"missing lung-volume rows: {sorted(expected_volume_keys - observed_volume_keys)}")
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    c.write_csv(args.review_root / "tre_aggregate.csv", tre_rows)
    c.write_json(args.review_root / "tre_aggregate.json", tre_rows)
    c.write_csv(args.review_root / "timing_aggregate.csv", timing_rows)
    c.write_json(args.review_root / "timing_aggregate.json", timing_rows)
    c.write_csv(args.review_root / "stage_rows.csv", stage_rows)
    c.write_json(args.review_root / "stage_rows.json", stage_rows)
    c.write_csv(args.review_root / "geometry_rows.csv", geometry_rows)
    c.write_json(args.review_root / "geometry_rows.json", geometry_rows)
    c.write_csv(args.review_root / "lung_volume_rows.csv", lung_volume_rows)
    c.write_json(args.review_root / "lung_volume_rows.json", lung_volume_rows)
    c.write_json(args.review_root / "aggregate.json", {"experiment_id": EXPERIMENT_ID, "case_count": len(rows), "errors": errors})
    write_report(args.review_root, rows, tre_rows, timing_rows, stage_rows, geometry_rows, lung_volume_rows, errors)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "case_rows": len(rows), "errors": errors}, indent=2), flush=True)
    if errors and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
