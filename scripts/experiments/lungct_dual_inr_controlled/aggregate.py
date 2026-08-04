#!/usr/bin/env python3
"""Validate and aggregate the controlled N0/N1/P0/P1 DIR-Lab cohort.

The acceptance gate is deliberately technical.  TRE or any other scientific
performance value is never used to accept, reject, rank, or stop a record.
The only causal comparisons produced here are N1-N0 and P1-P0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (
    REPORT_METRICS,
    compute_common_regularity,
)
from scripts.experiments.lungct_dual_inr_author.portable_paths import (
    portable_review_value,
)


EXPECTED_CASES = tuple(range(1, 11))
EXPECTED_STEPS = 2500
EXPECTED_STAGE_STARTS = (0, 400, 800, 1600)
EXPECTED_STAGE_LENGTHS = (400, 400, 800, 900)
EXPECTED_FINE_START = 400
DEFAULT_SEED = 42
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PTVREG_REFERENCE = (
    REPOSITORY_ROOT
    / "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension"
    / "pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv"
)
DEFAULT_MR_REFERENCE = (
    REPOSITORY_ROOT
    / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"
    / "directcp_refined_case_rows.csv"
)
DEFAULT_MR_REFERENCE_MEANS = DEFAULT_MR_REFERENCE.with_name(
    "directcp_refined_mean_rows.csv"
)
EXPECTED_DEFAULT_MR_REG_VALUE = 0.055

REFERENCE_META: dict[str, dict[str, str]] = {
    "PTVREG": {
        "display_name": "pTVReg canonical 1 mm",
        "protocol_tag": "ptvreg_uploaded_field+canonical_ptv_crop_iso1mm+metric_only_fixed_t00_mask",
        "preprocessing": (
            "Validated uploaded pTVReg field on the fixed/moving landmark-union crop "
            "([x,y,z] margin [10,10,5]), CT [80,900] to [0,1], approximately "
            "isotropic 1 mm; no lung mask in optimization and fixed/T00 lung mask "
            "for regularity evaluation only."
        ),
    },
    "MR_D_BSCP": {
        "display_name": "MR-D-BSCP best 1 mm",
        "protocol_tag": "mr_d_bscp_32_16_8_4+ptv_crop_iso1mm+ptv_lcc_isotv+metric_only_fixed_t00_mask",
        "preprocessing": (
            "Our PyTorch pTV-style landmark crop and isotropic-1-mm path with no "
            "lung mask in optimization, a 32-16-8-4 control schedule, 8-4-2-1 "
            "image schedule, local NCC, and isoTV; fixed/T00 lung mask is used "
            "for regularity evaluation only."
        ),
    },
}

VARIANT_META: dict[str, dict[str, Any]] = {
    "N0": {
        "directory": "n0",
        "protocol": "N",
        "protocol_tag": "native_idir_fixed_t00_mask",
        "preprocessing": (
            "Native full volume, native anisotropic spacing, no crop/resampling; "
            "fixed/T00 R231 lung mask is the optimization sampling domain."
        ),
        "image_factors": (1, 1, 1, 1),
        "schedule": "four-stage full-resolution schedule control",
    },
    "N1": {
        "directory": "n1",
        "protocol": "N",
        "protocol_tag": "native_idir_fixed_t00_mask",
        "preprocessing": (
            "Native full volume, native anisotropic spacing, no crop/resampling; "
            "fixed/T00 R231 lung mask is the optimization sampling domain."
        ),
        "image_factors": (8, 4, 2, 1),
        "schedule": "MR-D-BSCP-shaped image pyramid",
    },
    "P0": {
        "directory": "p0",
        "protocol": "P",
        "protocol_tag": "ptvreg_iso1mm_landmark_crop_no_mask",
        "preprocessing": (
            "Canonical pTVReg-style landmark-union crop at isotropic 1 mm, "
            "CT [80,900] mapped to [0,1], five-voxel full-grid border exclusion, "
            "and no lung mask in optimization."
        ),
        "image_factors": (1, 1, 1, 1),
        "schedule": "four-stage full-resolution schedule control",
    },
    "P1": {
        "directory": "p1",
        "protocol": "P",
        "protocol_tag": "ptvreg_iso1mm_landmark_crop_no_mask",
        "preprocessing": (
            "Canonical pTVReg-style landmark-union crop at isotropic 1 mm, "
            "CT [80,900] mapped to [0,1], five-voxel full-grid border exclusion, "
            "and no lung mask in optimization."
        ),
        "image_factors": (8, 4, 2, 1),
        "schedule": "MR-D-BSCP-shaped image pyramid",
    },
}
VARIANTS = tuple(VARIANT_META)
PAIRINGS = (("N0", "N1", "N1_minus_N0"), ("P0", "P1", "P1_minus_P0"))

SCIENTIFIC_METRICS = (
    "headline_native_snap_tre_mm",
    "headline_native_subvoxel_tre_mm",
    "registration_grid_snap_tre_mm",
    "registration_grid_subvoxel_tre_mm",
    "direct_inr_snap_tre_mm",
    "direct_inr_subvoxel_tre_mm",
)
RESOURCE_METRICS = (
    "training_wall_seconds",
    "total_wall_seconds",
    "peak_training_allocated_bytes",
    "peak_training_reserved_bytes",
    "peak_total_allocated_bytes",
    "peak_total_reserved_bytes",
)
REGULARITY_FIELDS = tuple(f"regularity_{name}" for name in REPORT_METRICS)
POPULATION_METRICS = SCIENTIFIC_METRICS + RESOURCE_METRICS + REGULARITY_FIELDS

OUTPUT_NAMES = (
    "optimization_index.csv",
    "optimization_index.json",
    "regularity_metrics.csv",
    "regularity_metrics.json",
    "population_statistics.csv",
    "population_statistics.json",
    "paired_comparisons.csv",
    "acceptance_report.json",
    "acceptance_report.md",
    "camera_ready_readout.md",
)


class AggregationError(RuntimeError):
    """Raised for a malformed invocation or an incomplete default cohort."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return str(path)
    return "${REPO_ROOT}/" + relative.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregationError(f"expected a JSON object in {path}")
    return payload


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(payload: Mapping[str, Any], paths: Sequence[str]) -> tuple[Any, str | None]:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return value, path
    return None, None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("tre_mean_mm", "mean_mm", "mean", "value"):
            if key in value:
                value = value[key]
                break
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strict_finite(value: Any, label: str) -> float:
    result = _finite_number(value)
    if result is None:
        raise AggregationError(f"{label} is missing or non-finite: {value!r}")
    return result


def _int_sequence(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise AggregationError(f"{label} must be a sequence")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"{label} contains a non-integer") from exc


def _parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise AggregationError(f"{label} is not boolean: {value!r}")


def _history_column(
    fieldnames: Sequence[str], candidates: Sequence[str], label: str
) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise AggregationError(f"history lacks {label}; tried {list(candidates)}")


def validate_history(path: Path, variant: str) -> dict[str, Any]:
    """Apply the locked 2,500-step schedule/finiteness contract."""

    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or ())
    except OSError as exc:
        raise AggregationError(f"cannot read history {path}: {exc}") from exc
    if len(rows) != EXPECTED_STEPS:
        raise AggregationError(
            f"history has {len(rows)} rows; expected exactly {EXPECTED_STEPS}"
        )
    step_key = _history_column(
        fieldnames,
        ("zero_step", "global_zero_step", "global_step", "step", "iteration"),
        "step",
    )
    stage_key = _history_column(fieldnames, ("stage_index", "stage"), "stage index")
    factor_key = _history_column(
        fieldnames,
        ("image_factor", "resolution_factor", "pyramid_factor", "downsample_factor"),
        "image factor",
    )
    fine_key = _history_column(
        fieldnames,
        ("fine_active", "fine_branch_active", "fine_enabled", "use_fine"),
        "fine state",
    )
    loss_key = _history_column(
        fieldnames, ("loss_total", "total_loss", "loss"), "total loss"
    )

    raw_steps: list[int] = []
    for index, row in enumerate(rows):
        try:
            raw_steps.append(int(row[step_key]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(f"history row {index + 1} has invalid step") from exc
    if raw_steps == list(range(EXPECTED_STEPS)):
        zero_steps = raw_steps
    elif raw_steps == list(range(1, EXPECTED_STEPS + 1)):
        zero_steps = [value - 1 for value in raw_steps]
    else:
        raise AggregationError("history steps are not contiguous 0..2499 (or 1..2500)")

    factors_expected = tuple(
        int(value) for value in VARIANT_META[variant]["image_factors"]
    )
    stage_starts: list[int] = []
    factors_seen: list[int] = []
    fine_first: int | None = None
    total_loss_values: list[float] = []
    numeric_columns: set[str] = set()

    for key in fieldnames:
        values = [row.get(key, "").strip() for row in rows]
        if values and all(value != "" for value in values):
            try:
                converted = [float(value) for value in values]
            except ValueError:
                continue
            numeric_columns.add(key)
            if not all(math.isfinite(value) for value in converted):
                raise AggregationError(
                    f"history numeric column {key!r} contains NaN/Inf"
                )

    for index, (row, zero_step) in enumerate(zip(rows, zero_steps)):
        expected_stage = (
            0
            if zero_step < 400
            else 1 if zero_step < 800 else 2 if zero_step < 1600 else 3
        )
        try:
            stage = int(row[stage_key])
            factor = int(row[factor_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(
                f"history row {index + 1} has invalid stage/factor"
            ) from exc
        if stage != expected_stage:
            raise AggregationError(
                f"history step {zero_step} has stage {stage}; expected {expected_stage}"
            )
        if factor != factors_expected[stage]:
            raise AggregationError(
                f"history step {zero_step} has factor {factor}; expected {factors_expected[stage]}"
            )
        fine = _parse_bool(row[fine_key], f"history step {zero_step} fine state")
        if fine != (zero_step >= EXPECTED_FINE_START):
            raise AggregationError(
                f"history step {zero_step} fine state is {fine}; expected {zero_step >= EXPECTED_FINE_START}"
            )
        if fine and fine_first is None:
            fine_first = zero_step
        if zero_step in EXPECTED_STAGE_STARTS:
            stage_starts.append(zero_step)
            factors_seen.append(factor)
        total_loss_values.append(
            _strict_finite(row.get(loss_key), f"history step {zero_step} {loss_key}")
        )

    if tuple(stage_starts) != EXPECTED_STAGE_STARTS:
        raise AggregationError(f"realized stage starts are {stage_starts}")
    if tuple(factors_seen) != factors_expected:
        raise AggregationError(f"realized factors are {factors_seen}")
    if fine_first != EXPECTED_FINE_START:
        raise AggregationError(
            f"fine branch first active at {fine_first}, expected 400"
        )
    return {
        "history_rows": len(rows),
        "step_column": step_key,
        "stage_column": stage_key,
        "factor_column": factor_key,
        "fine_column": fine_key,
        "loss_column": loss_key,
        "numeric_columns_checked": sorted(numeric_columns),
        "stage_starts": stage_starts,
        "stage_lengths": list(EXPECTED_STAGE_LENGTHS),
        "image_factors": factors_seen,
        "fine_activation_step": fine_first,
        "mean_loss_total": statistics.fmean(total_loss_values),
        "final_loss_total": total_loss_values[-1],
    }


def _array_all_finite(array: np.ndarray) -> bool:
    if array.ndim == 0:
        return bool(np.isfinite(array).all())
    for start in range(0, array.shape[0], 16):
        if not bool(np.isfinite(array[start : start + 16]).all()):
            return False
    return True


def validate_arrays(case_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    field_path = case_dir / "dense_dvf_xyz_voxel.npy"
    mask_path = case_dir / "fixed_metric_mask_xyz.npy"
    try:
        field = np.load(field_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AggregationError(f"cannot load dense field/metric mask: {exc}") from exc
    if field.ndim != 4 or field.shape[-1] != 3 or min(field.shape[:3]) < 3:
        raise AggregationError(f"dense field has invalid shape {field.shape}")
    if mask.ndim != 3 or tuple(mask.shape) != tuple(field.shape[:3]):
        raise AggregationError(
            f"metric mask shape {mask.shape} does not match field {field.shape[:3]}"
        )
    if not _array_all_finite(field):
        raise AggregationError("dense displacement field contains NaN/Inf")
    if not _array_all_finite(mask):
        raise AggregationError("fixed metric mask contains NaN/Inf")
    mask_voxels = int(np.count_nonzero(mask))
    if mask_voxels <= 0:
        raise AggregationError("fixed metric mask is empty")
    summary_shape = summary.get("shape_xyz")
    if summary_shape is not None and _int_sequence(
        summary_shape, "summary shape_xyz"
    ) != tuple(field.shape[:3]):
        raise AggregationError("summary shape_xyz disagrees with dense field")
    return {
        "shape_xyz": list(field.shape[:3]),
        "field_dtype": str(field.dtype),
        "metric_mask_dtype": str(mask.dtype),
        "metric_mask_voxels": mask_voxels,
        "dense_dvf_path": str(field_path),
        "fixed_metric_mask_path": str(mask_path),
    }


def _spacing(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    value, _ = _first(
        summary,
        (
            "spacing_xyz_mm",
            "image_spacing_xyz",
            "registration_spacing_xyz_mm",
            "data.spacing_xyz_mm",
        ),
    )
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AggregationError("summary lacks three-value spacing_xyz_mm")
    spacing = tuple(_strict_finite(item, "spacing") for item in value)
    if min(spacing) <= 0:
        raise AggregationError(f"spacing must be positive, got {spacing}")
    return spacing  # type: ignore[return-value]


def validate_summary(
    summary: Mapping[str, Any], *, variant: str, case: int, seed: int
) -> tuple[float, float, float]:
    status = str(summary.get("status", "")).lower()
    if status not in {"complete", "completed", "success", "done"}:
        raise AggregationError(f"summary status is not complete: {status!r}")
    observed_variant = str(summary.get("variant", "")).upper()
    if observed_variant != variant:
        raise AggregationError(f"summary variant {observed_variant!r} != {variant}")
    try:
        observed_case = int(summary.get("case_id"))
        observed_seed = int(summary.get("seed"))
    except (TypeError, ValueError) as exc:
        raise AggregationError("summary case_id/seed is invalid") from exc
    if observed_case != case or observed_seed != seed:
        raise AggregationError(
            f"summary identity is case{observed_case:02d}/seed{observed_seed}, expected case{case:02d}/seed{seed}"
        )
    run_mode = str(summary.get("run_mode", "")).lower()
    if (
        "smoke" in run_mode
        or "pilot" in run_mode
        or run_mode not in {"full", "full_length", "production"}
    ):
        raise AggregationError(f"summary run_mode is not a full run: {run_mode!r}")
    protocol = str(summary.get("protocol", "")).upper()
    if protocol != str(VARIANT_META[variant]["protocol"]):
        raise AggregationError(
            f"summary protocol {protocol!r} does not match {variant}"
        )
    if (
        _int_sequence(summary.get("stage_starts"), "summary stage_starts")
        != EXPECTED_STAGE_STARTS
    ):
        raise AggregationError("summary stage_starts disagree with locked plan")
    if (
        _int_sequence(summary.get("stage_lengths"), "summary stage_lengths")
        != EXPECTED_STAGE_LENGTHS
    ):
        raise AggregationError("summary stage_lengths disagree with locked plan")
    expected_factors = tuple(VARIANT_META[variant]["image_factors"])
    if (
        _int_sequence(summary.get("image_factors"), "summary image_factors")
        != expected_factors
    ):
        raise AggregationError("summary image_factors disagree with locked variant")
    try:
        fine_start = int(summary.get("fine_activation_step"))
    except (TypeError, ValueError) as exc:
        raise AggregationError("summary fine_activation_step is invalid") from exc
    if fine_start != EXPECTED_FINE_START:
        raise AggregationError(f"summary fine activation is {fine_start}, expected 400")
    return _spacing(summary)


def _metric_value(
    payload: Mapping[str, Any], paths: Sequence[str]
) -> tuple[float | None, str | None]:
    value, path = _first(payload, paths)
    return _finite_number(value), path


def extract_metrics(
    summary: Mapping[str, Any], variant: str
) -> tuple[dict[str, Any], list[str]]:
    """Extract reporting values without applying a scientific acceptance gate."""

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    is_native = variant.startswith("N")
    headline_snap_paths = (
        "metrics.headline_tre",
        "metrics.native_crop_tre_snap_to_voxel",
        "metrics.native_tre_snap_to_voxel",
        "metrics.native_crop.dense_snap",
        "metrics.dense_dvf.snap",
        "native_crop_tre_snap_to_voxel",
        "landmark_metrics.native.snap",
    )
    if is_native:
        headline_snap_paths = (
            "metrics.headline_tre",
            "metrics.tre_snap_to_voxel",
            "metrics.dense_dvf.snap",
            "metrics.native_tre_snap_to_voxel",
            "landmark_metrics.dense_dvf.snap",
        )
    headline_sub_paths = (
        "metrics.native_crop_tre_subvoxel",
        "metrics.native_crop.dense_subvoxel",
        "metrics.headline_tre_subvoxel",
        "native_crop_tre_subvoxel",
        "landmark_metrics.native.subvoxel",
    )
    if is_native:
        headline_sub_paths = (
            "metrics.tre_subvoxel",
            "metrics.dense_dvf.subvoxel",
            "metrics.native_tre_subvoxel",
            "landmark_metrics.dense_dvf.subvoxel",
        )
    aliases: dict[str, Sequence[str]] = {
        "headline_native_snap_tre_mm": headline_snap_paths,
        "headline_native_subvoxel_tre_mm": headline_sub_paths,
        "registration_grid_snap_tre_mm": (
            "metrics.registration_grid_tre_snap",
            "metrics.registration_grid.dense_snap",
            "metrics.tre_snap_to_voxel",
            "metrics.dense_dvf.snap",
            "landmark_metrics.dense_dvf.snap",
        ),
        "registration_grid_subvoxel_tre_mm": (
            "metrics.registration_grid_tre_sub",
            "metrics.registration_grid_tre_subvoxel",
            "metrics.registration_grid.dense_subvoxel",
            "metrics.tre_subvoxel",
            "metrics.dense_dvf.subvoxel",
            "landmark_metrics.dense_dvf.subvoxel",
        ),
        "direct_inr_snap_tre_mm": (
            "metrics.direct_inr.snap",
            "metrics.registration_grid.direct_inr.snap",
            "metrics.direct_tre_snap_to_voxel",
            "landmark_metrics.direct_inr.snap",
        ),
        "direct_inr_subvoxel_tre_mm": (
            "metrics.direct_inr.subvoxel",
            "metrics.registration_grid.direct_inr.subvoxel",
            "metrics.direct_tre_subvoxel",
            "landmark_metrics.direct_inr.subvoxel",
        ),
    }
    record: dict[str, Any] = {}
    warnings: list[str] = []
    for name, paths in aliases.items():
        value, source = _metric_value(summary, paths)
        record[name] = value
        record[f"{name}_source"] = source
        if value is None:
            warnings.append(
                f"{name} missing or non-finite (reporting only; not an acceptance gate)"
            )

    runtime_paths = {
        "training_wall_seconds": (
            "runtime.training_wall_seconds",
            "training_wall_seconds",
        ),
        "total_wall_seconds": (
            "runtime.total_wall_seconds",
            "runtime.total_wall_seconds_through_evaluation",
            "total_wall_seconds",
            "elapsed_sec",
        ),
        "peak_training_allocated_bytes": ("cuda_peak_memory_bytes.training_allocated",),
        "peak_training_reserved_bytes": ("cuda_peak_memory_bytes.training_reserved",),
        "peak_total_allocated_bytes": ("cuda_peak_memory_bytes.total_allocated",),
        "peak_total_reserved_bytes": ("cuda_peak_memory_bytes.total_reserved",),
    }
    for name, paths in runtime_paths.items():
        value, _ = _metric_value(summary, paths)
        record[name] = value
        if value is None:
            warnings.append(f"{name} missing/non-finite (reporting only)")
    if record["peak_total_reserved_bytes"] is None:
        peak_gib, _ = _metric_value(summary, ("peak_mem_gb", "peak_memory_gib"))
        if peak_gib is not None:
            record["peak_total_reserved_bytes"] = peak_gib * 1024**3
    return record, warnings


def _regularity_path(case_dir: Path) -> Path:
    return case_dir / "common_regularity.json"


def _load_existing_regularity(
    path: Path, *, variant: str, case: int, spacing: Sequence[float]
) -> dict[str, Any]:
    payload = _load_json(path)
    if str(payload.get("status", "complete")).lower() not in {
        "complete",
        "completed",
        "success",
    }:
        raise AggregationError(f"existing regularity payload is not complete: {path}")
    if (
        str(payload.get("variant", variant)).upper() != variant
        or int(payload.get("case_id", case)) != case
    ):
        raise AggregationError(f"existing regularity identity mismatch: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AggregationError(f"existing regularity payload lacks metrics: {path}")
    for metric in REPORT_METRICS:
        _strict_finite(metrics.get(metric), f"existing regularity {metric}")
    observed_spacing = payload.get("spacing_xyz_mm", spacing)
    if not np.allclose(
        np.asarray(observed_spacing, dtype=float),
        np.asarray(spacing, dtype=float),
        rtol=0,
        atol=1e-8,
    ):
        raise AggregationError(f"existing regularity spacing mismatch: {path}")
    return payload


def compute_or_load_regularity(
    *, case_dir: Path, variant: str, case: int, spacing: Sequence[float]
) -> dict[str, Any]:
    output = _regularity_path(case_dir)
    if output.exists():
        payload = _load_existing_regularity(
            output, variant=variant, case=case, spacing=spacing
        )
        payload = dict(payload)
        payload["cache_status"] = "reused_existing"
        return payload
    field = np.load(case_dir / "dense_dvf_xyz_voxel.npy", allow_pickle=False)
    mask = np.load(case_dir / "fixed_metric_mask_xyz.npy", allow_pickle=False)
    computed = compute_common_regularity(
        disp_vox_xyz=field,
        fixed_lobe_union_xyz=mask,
        spacing_xyz_mm=spacing,
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "variant": variant,
        "case_id": case,
        "protocol": VARIANT_META[variant]["protocol"],
        "protocol_tag": VARIANT_META[variant]["protocol_tag"],
        "spacing_xyz_mm": [float(value) for value in spacing],
        "dense_dvf_xyz_voxel": "dense_dvf_xyz_voxel.npy",
        "fixed_metric_mask_xyz": "fixed_metric_mask_xyz.npy",
        "metric_scope": "full registration grid and fixed/T00 lung foreground metric mask",
        "metrics": computed["metrics"],
        "conversion": computed["conversion"],
        "interpretation": {
            "diffusion_and_bending": (
                "Voxel-coordinate finite-difference metrics; do not compare their magnitude "
                "across native anisotropic and pTVReg isotropic grids as physical energies."
            ),
            "jacobian": "Jacobian determinant statistics are dimensionless.",
            "foreground": (
                "N uses the fixed/T00 lung mask also used for optimization sampling; "
                "P uses a fixed lung mask for metric scope only, never for optimization."
            ),
        },
    }
    _atomic_json(output, payload)
    payload["cache_status"] = "computed"
    return payload


def inspect_record(
    *, run_root: Path, variant: str, case: int, seed: int
) -> dict[str, Any]:
    meta = VARIANT_META[variant]
    case_dir = run_root / str(meta["directory"]) / f"case_{case:02d}"
    record: dict[str, Any] = {
        "variant": variant,
        "case_id": case,
        "seed": seed,
        "case_dir": str(case_dir),
        "protocol": meta["protocol"],
        "protocol_tag": meta["protocol_tag"],
        "preprocessing": meta["preprocessing"],
        "schedule": meta["schedule"],
        "expected_image_factors": list(meta["image_factors"]),
        "technical_accepted": False,
        "technical_errors": [],
        "reporting_warnings": [],
        "regularity_status": "not_run",
        "regularity_error": None,
    }
    errors: list[str] = record["technical_errors"]
    if not case_dir.is_dir():
        errors.append("case directory missing")
        return record
    if not (case_dir / "DONE").is_file():
        errors.append("DONE marker missing")
        return record
    summary_path = case_dir / "summary.json"
    history_path = case_dir / "history.csv"
    for required in (
        summary_path,
        history_path,
        case_dir / "dense_dvf_xyz_voxel.npy",
        case_dir / "fixed_metric_mask_xyz.npy",
    ):
        if not required.is_file() or required.stat().st_size == 0:
            errors.append(f"required artifact missing/empty: {required.name}")
    if errors:
        return record
    try:
        summary = _load_json(summary_path)
        spacing = validate_summary(summary, variant=variant, case=case, seed=seed)
        history_audit = validate_history(history_path, variant)
        array_audit = validate_arrays(case_dir, summary)
        record.update(
            {
                "spacing_xyz_mm": list(spacing),
                "shape_xyz": array_audit["shape_xyz"],
                "metric_mask_voxels": array_audit["metric_mask_voxels"],
                "history_rows": history_audit["history_rows"],
                "realized_stage_starts": history_audit["stage_starts"],
                "realized_stage_lengths": history_audit["stage_lengths"],
                "realized_image_factors": history_audit["image_factors"],
                "fine_activation_step": history_audit["fine_activation_step"],
                "mean_loss_total": history_audit["mean_loss_total"],
                "final_loss_total": history_audit["final_loss_total"],
                "sampling_rule": summary.get("sampling_rule"),
                "protocol_tags": summary.get("protocol_tags"),
                "technical_accepted": True,
            }
        )
        extracted, warnings = extract_metrics(summary, variant)
        record.update(extracted)
        record["reporting_warnings"].extend(warnings)
    except (AggregationError, OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    return record


def add_regularity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute posthoc records sequentially and continue after every failure."""

    reports: list[dict[str, Any]] = []
    for record in records:
        if not record["technical_accepted"]:
            reports.append(
                {
                    "variant": record["variant"],
                    "case_id": record["case_id"],
                    "status": "not_run_technical_rejection",
                    "error": None,
                }
            )
            continue
        try:
            payload = compute_or_load_regularity(
                case_dir=Path(record["case_dir"]),
                variant=str(record["variant"]),
                case=int(record["case_id"]),
                spacing=record["spacing_xyz_mm"],
            )
            metrics = payload["metrics"]
            for metric in REPORT_METRICS:
                record[f"regularity_{metric}"] = _strict_finite(
                    metrics.get(metric), f"regularity {metric}"
                )
            record["regularity_status"] = "complete"
            reports.append(
                {
                    "variant": record["variant"],
                    "case_id": record["case_id"],
                    "protocol": record["protocol"],
                    "status": "complete",
                    "cache_status": payload.get("cache_status"),
                    "path": str(_regularity_path(Path(record["case_dir"]))),
                    **{metric: float(metrics[metric]) for metric in REPORT_METRICS},
                }
            )
        except Exception as exc:  # posthoc failures must not prevent remaining records
            record["regularity_status"] = "failed"
            record["regularity_error"] = f"{type(exc).__name__}: {exc}"
            reports.append(
                {
                    "variant": record["variant"],
                    "case_id": record["case_id"],
                    "protocol": record["protocol"],
                    "status": "failed",
                    "error": record["regularity_error"],
                }
            )
    return reports


def _describe(values: Iterable[Any]) -> dict[str, Any] | None:
    clean = [
        value
        for value in (_finite_number(item) for item in values)
        if value is not None
    ]
    if not clean:
        return None
    ordered = sorted(clean)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    q1 = percentile(0.25)
    q3 = percentile(0.75)
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "sd_sample": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(clean),
        "max": max(clean),
    }


def population_statistics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted = [row for row in records if row.get("technical_accepted")]
    for variant in VARIANTS:
        group = [row for row in accepted if row["variant"] == variant]
        for metric in POPULATION_METRICS:
            described = _describe(row.get(metric) for row in group)
            if described is not None:
                rows.append({"group": variant, "metric": metric, **described})
    return rows


def reference_population_statistics(
    reference_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_records = reference_bundle.get("records", {})
    if not isinstance(reference_records, Mapping):
        return rows
    metrics = SCIENTIFIC_METRICS + REGULARITY_FIELDS
    for group in ("PTVREG", "MR_D_BSCP"):
        by_case = reference_records.get(group, {})
        if not isinstance(by_case, Mapping):
            continue
        records = list(by_case.values())
        for metric in metrics:
            described = _describe(
                record.get(metric) for record in records if isinstance(record, Mapping)
            )
            if described is not None:
                rows.append(
                    {
                        "group": group,
                        "metric": metric,
                        "external_reference": True,
                        "protocol_tag": REFERENCE_META[group]["protocol_tag"],
                        **described,
                    }
                )
    return rows


def paired_comparisons(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["variant"]), int(row["case_id"])): row
        for row in records
        if row.get("technical_accepted")
    }
    rows: list[dict[str, Any]] = []
    delta_metrics = POPULATION_METRICS + ("mean_loss_total", "final_loss_total")
    for control, multires, direction in PAIRINGS:
        for case in EXPECTED_CASES:
            left = by_key.get((control, case))
            right = by_key.get((multires, case))
            row: dict[str, Any] = {
                "case_id": case,
                "protocol": VARIANT_META[control]["protocol"],
                "control": control,
                "multires": multires,
                "direction": direction,
                "paired_technical_complete": left is not None and right is not None,
            }
            if left is not None and right is not None:
                for metric in delta_metrics:
                    a = _finite_number(left.get(metric))
                    b = _finite_number(right.get(metric))
                    row[f"delta_{metric}"] = None if a is None or b is None else b - a
            rows.append(row)
    return rows


def _reference_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []

        def visit(value: Any, inherited_method: str | None = None) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item, inherited_method)
            elif isinstance(value, dict):
                method = inherited_method
                for key in ("method", "model", "variant", "name", "display_name"):
                    if key in value and isinstance(value[key], str):
                        method = value[key]
                        break
                if any(key in value for key in ("case", "case_id", "dataset_case")):
                    row = dict(value)
                    if method is not None:
                        row.setdefault("method", method)
                    rows.append(row)
                for key, item in value.items():
                    child_method = method
                    if _normalize_method(key).find("mrdbscp") >= 0:
                        child_method = key
                    if isinstance(item, (dict, list)):
                        visit(item, child_method)

        visit(payload)
        return rows
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]


def _normalize_method(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _reference_case(row: Mapping[str, Any]) -> int | None:
    value, _ = _first(row, ("case", "case_id", "dataset_case"))
    if value is None:
        return None
    match = re.search(r"(?:case[_ -]?)?(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        return None
    case = int(match.group(1))
    return case if case in EXPECTED_CASES else None


def _reference_method(row: Mapping[str, Any]) -> str:
    value, _ = _first(
        row, ("method", "model", "variant", "name", "display_name", "method_label")
    )
    return "" if value is None else str(value)


def _is_ptvreg_reference_row(row: Mapping[str, Any]) -> bool:
    tokens = {
        _normalize_method(row.get("method")),
        _normalize_method(row.get("method_label")),
        _normalize_method(row.get("source")),
    }
    return bool({"ptvreg", "ptvregreference", "ptvreguploadedmat"}.intersection(tokens))


def _is_mr_d_bscp_row(row: Mapping[str, Any]) -> bool:
    tokens = " ".join(
        _normalize_method(row.get(key))
        for key in ("method", "method_label", "model", "variant", "display_name")
    )
    return "mrdbscp" in tokens or "directcp321684ptvlccisotv" in tokens


def _reference_record(
    *,
    row: Mapping[str, Any],
    case: int,
    group: str,
    source_path: Path,
    source_row: int,
) -> dict[str, Any]:
    meta = REFERENCE_META[group]
    aliases: dict[str, Sequence[str]] = {
        "headline_native_snap_tre_mm": (
            "headline_native_snap_tre_mm",
            "native_crop_snap_tre_mean_mm",
            "original_snap_tre_mean_mm",
            "primary_original_snap_tre_mean_mm",
            "snap_tre_mm",
            "tre_mean_mm",
            "tre_mm",
        ),
        "headline_native_subvoxel_tre_mm": (
            "headline_native_subvoxel_tre_mm",
            "native_crop_sub_tre_mean_mm",
            "native_crop_subvoxel_tre_mean_mm",
            "original_subvoxel_tre_mean_mm",
        ),
        "registration_grid_snap_tre_mm": (
            "registration_grid_snap_tre_mm",
            "reg1mm_snap_tre_mean_mm",
        ),
        "registration_grid_subvoxel_tre_mm": (
            "registration_grid_subvoxel_tre_mm",
            "reg1mm_subvoxel_tre_mean_mm",
        ),
        "regularity_log_jdet_std_fg": (
            "regularity_log_jdet_std_fg",
            "log_jdet_std_fg",
            "fg_logj_std",
            "fg_log_jdet_std",
            "sdlogj_fg",
        ),
        "regularity_jdet_le0_pct_fg": (
            "regularity_jdet_le0_pct_fg",
            "jdet_le0_pct_fg",
            "fg_jdet_le0_pct",
            "folding_pct_fg",
        ),
        "regularity_diffusion_l2_fg": (
            "regularity_diffusion_l2_fg",
            "diffusion_l2_fg",
            "fg_diffusion_l2",
            "fg_diffusion",
            "diffusion_fg",
        ),
        "regularity_bending_energy_fg": (
            "regularity_bending_energy_fg",
            "bending_energy_fg",
            "fg_bending_energy",
            "fg_bending",
            "bending_fg",
        ),
        "regularity_disp_mag_mean_fg_vox": (
            "regularity_disp_mag_mean_fg_vox",
            "disp_mag_mean_fg_vox",
            "fg_disp_mag_mean_vox",
        ),
        "regularity_disp_mag_p95_fg_vox": (
            "regularity_disp_mag_p95_fg_vox",
            "disp_mag_p95_fg_vox",
            "fg_disp_mag_p95_vox",
        ),
        "regularity_disp_mag_max_fg_vox": (
            "regularity_disp_mag_max_fg_vox",
            "disp_mag_max_fg_vox",
            "fg_disp_mag_max_vox",
        ),
    }
    record: dict[str, Any] = {
        "reference_group": group,
        "display_name": meta["display_name"],
        "case_id": case,
        "method": _reference_method(row),
        "protocol_tag": meta["protocol_tag"],
        "preprocessing": meta["preprocessing"],
        "source_path": _portable_path(source_path),
        "source_row": source_row,
    }
    for name, paths in aliases.items():
        value, source = _metric_value(row, paths)
        record[name] = value
        record[f"{name}_source"] = source
    reg_value, reg_source = _metric_value(
        row, ("reg_value", "regularization_value", "ptv_isotv_weight", "weight")
    )
    record["regularization_value"] = reg_value
    record["regularization_value_source"] = reg_source
    return record


def _require_reference_cases(
    records: Mapping[int, Mapping[str, Any]], *, label: str
) -> None:
    missing = sorted(set(EXPECTED_CASES) - set(records))
    extra = sorted(set(records) - set(EXPECTED_CASES))
    if missing or extra or len(records) != len(EXPECTED_CASES):
        raise AggregationError(
            f"{label} reference must contain one accepted row for every case; "
            f"missing={missing}, extra={extra}, count={len(records)}"
        )
    required_metrics = (
        "headline_native_snap_tre_mm",
        "headline_native_subvoxel_tre_mm",
        "registration_grid_snap_tre_mm",
        "registration_grid_subvoxel_tre_mm",
        "regularity_diffusion_l2_fg",
        "regularity_bending_energy_fg",
        "regularity_log_jdet_std_fg",
        "regularity_jdet_le0_pct_fg",
    )
    for case, record in records.items():
        missing_metrics = [
            metric
            for metric in required_metrics
            if _finite_number(record.get(metric)) is None
        ]
        if missing_metrics:
            raise AggregationError(
                f"{label} case{case:02d} lacks finite TRE/regularity fields: "
                f"{missing_metrics}"
            )


def load_ptvreg_reference(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise AggregationError(f"pTVReg reference path does not exist: {path}")
    selected: dict[int, dict[str, Any]] = {}
    for source_index, row in enumerate(_reference_rows(path), start=1):
        if not _is_ptvreg_reference_row(row):
            continue
        case = _reference_case(row)
        if case is None:
            continue
        if case in selected:
            raise AggregationError(
                f"pTVReg reference has duplicate accepted rows for case{case:02d}"
            )
        selected[case] = _reference_record(
            row=row,
            case=case,
            group="PTVREG",
            source_path=path,
            source_row=source_index,
        )
    _require_reference_cases(selected, label="pTVReg")
    return selected


def _select_mr_cohort_setting(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(
            f"MR-D-BSCP cohort reference path does not exist: {path}"
        )
    candidates: list[dict[str, Any]] = []
    for source_index, row in enumerate(_reference_rows(path), start=1):
        if not _is_mr_d_bscp_row(row):
            continue
        cases, _ = _metric_value(row, ("cases", "n", "case_count"))
        if cases is not None and int(cases) != len(EXPECTED_CASES):
            continue
        reg_value, reg_source = _metric_value(
            row, ("reg_value", "regularization_value", "ptv_isotv_weight", "weight")
        )
        cohort_tre, tre_source = _metric_value(
            row,
            (
                "original_snap_tre_mean_mm_mean",
                "headline_native_snap_tre_mm_mean",
                "native_crop_snap_tre_mean_mm_mean",
                "mean_tre_mm",
            ),
        )
        if reg_value is None or cohort_tre is None:
            continue
        candidates.append(
            {
                "regularization_value": reg_value,
                "cohort_headline_native_snap_tre_mm": cohort_tre,
                "source_row": source_index,
                "regularization_value_source": reg_source,
                "tre_source": tre_source,
            }
        )
    if not candidates:
        raise AggregationError(
            f"MR-D-BSCP cohort table has no finite ten-case schedule rows: {path}"
        )
    selected = min(
        candidates, key=lambda row: row["cohort_headline_native_snap_tre_mm"]
    )
    return {
        **selected,
        "selection_rule": (
            "single cohort-level regularization setting with the lowest ten-case "
            "headline/native snap TRE; never per-case selection"
        ),
        "candidate_setting_count": len(candidates),
        "source_path": _portable_path(path),
    }


def load_mr_d_bscp_reference(
    *,
    case_path: Path | None,
    cohort_path: Path | None,
    explicit_reg_value: float | None,
    assert_default_setting: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    if case_path is None:
        return {}, None
    if not case_path.is_file():
        raise AggregationError(
            f"MR-D-BSCP case reference path does not exist: {case_path}"
        )
    selection: dict[str, Any]
    if cohort_path is not None:
        selection = _select_mr_cohort_setting(cohort_path)
        selected_reg_value = float(selection["regularization_value"])
        if explicit_reg_value is not None and not math.isclose(
            selected_reg_value, explicit_reg_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AggregationError(
                "explicit MR-D-BSCP setting disagrees with the cohort-level lowest-TRE row: "
                f"{explicit_reg_value} versus {selected_reg_value}"
            )
    elif explicit_reg_value is not None:
        selected_reg_value = float(explicit_reg_value)
        selection = {
            "regularization_value": selected_reg_value,
            "selection_rule": "explicit single cohort-level setting supplied by CLI",
            "source_path": None,
        }
    else:
        raise AggregationError(
            "MR-D-BSCP case references require --mr-reference-means or "
            "--mr-reference-reg-value; per-case lowest-TRE selection is prohibited"
        )
    if assert_default_setting and not math.isclose(
        selected_reg_value, EXPECTED_DEFAULT_MR_REG_VALUE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AggregationError(
            "default MR-D-BSCP reference bundle no longer selects expected w=0.055; "
            f"observed {selected_reg_value}"
        )

    selected: dict[int, dict[str, Any]] = {}
    matching_rows = 0
    for source_index, row in enumerate(_reference_rows(case_path), start=1):
        if not _is_mr_d_bscp_row(row):
            continue
        case = _reference_case(row)
        row_reg_value, _ = _metric_value(
            row, ("reg_value", "regularization_value", "ptv_isotv_weight", "weight")
        )
        if (
            case is None
            or row_reg_value is None
            or not math.isclose(
                row_reg_value, selected_reg_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            continue
        matching_rows += 1
        if case in selected:
            raise AggregationError(
                f"MR-D-BSCP w={selected_reg_value:g} has duplicate rows for case{case:02d}"
            )
        selected[case] = _reference_record(
            row=row,
            case=case,
            group="MR_D_BSCP",
            source_path=case_path,
            source_row=source_index,
        )
    _require_reference_cases(selected, label=f"MR-D-BSCP w={selected_reg_value:g}")
    selection = {
        **selection,
        "case_source_path": _portable_path(case_path),
        "matching_case_rows": matching_rows,
        "per_case_oracle_selection": False,
    }
    return selected, selection


def load_reference_bundle(
    *,
    ptvreg_path: Path | None,
    mr_case_path: Path | None,
    mr_cohort_path: Path | None,
    mr_reg_value: float | None,
    use_repo_defaults: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    used_default_ptv = False
    used_default_mr = False
    if use_repo_defaults:
        if ptvreg_path is None:
            if DEFAULT_PTVREG_REFERENCE.is_file():
                ptvreg_path = DEFAULT_PTVREG_REFERENCE
                used_default_ptv = True
            else:
                warnings.append(
                    f"default pTVReg reference is unavailable: {_portable_path(DEFAULT_PTVREG_REFERENCE)}"
                )
        if mr_case_path is None:
            if DEFAULT_MR_REFERENCE.is_file():
                mr_case_path = DEFAULT_MR_REFERENCE
                used_default_mr = True
            else:
                warnings.append(
                    f"default MR-D-BSCP case reference is unavailable: {_portable_path(DEFAULT_MR_REFERENCE)}"
                )
        if mr_cohort_path is None and used_default_mr:
            if DEFAULT_MR_REFERENCE_MEANS.is_file():
                mr_cohort_path = DEFAULT_MR_REFERENCE_MEANS
            elif mr_reg_value is None:
                warnings.append(
                    f"default MR-D-BSCP mean reference is unavailable: {_portable_path(DEFAULT_MR_REFERENCE_MEANS)}"
                )

    ptvreg = load_ptvreg_reference(ptvreg_path)
    mr_d_bscp, mr_selection = load_mr_d_bscp_reference(
        case_path=mr_case_path,
        cohort_path=mr_cohort_path,
        explicit_reg_value=mr_reg_value,
        assert_default_setting=used_default_mr
        and mr_cohort_path == DEFAULT_MR_REFERENCE_MEANS,
    )
    return {
        "schema_version": 1,
        "repo_defaults_enabled": use_repo_defaults,
        "warnings": warnings,
        "sources": {
            "ptvreg_case_table": (
                None if ptvreg_path is None else _portable_path(ptvreg_path)
            ),
            "mr_d_bscp_case_table": (
                None if mr_case_path is None else _portable_path(mr_case_path)
            ),
            "mr_d_bscp_cohort_table": (
                None if mr_cohort_path is None else _portable_path(mr_cohort_path)
            ),
        },
        "used_default_ptvreg": used_default_ptv,
        "used_default_mr_d_bscp": used_default_mr,
        "mr_d_bscp_selection": mr_selection,
        "records": {"PTVREG": ptvreg, "MR_D_BSCP": mr_d_bscp},
        "protocols": REFERENCE_META,
    }


def _lookup_population(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["group"]), str(row["metric"])): row for row in rows}


def _mean_sd(row: Mapping[str, Any] | None, scale: float = 1.0) -> str:
    if row is None:
        return "NA"
    return f"{float(row['mean']) / scale:.4f} ± {float(row['sd_sample']) / scale:.4f}"


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def render_acceptance_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled DUAL-INR acceptance report",
        "",
        f"Status: **{report['status']}**. Accepted `{report['accepted_records']}` of "
        f"`{report['expected_records']}` expected optimization records.",
        "",
        "Acceptance is technical only: DONE/artifacts, identity/protocol, 2,500 finite "
        "history rows, exact stages/factors, fine activation at step 400, and finite dense "
        "field/mask geometry. TRE values are never an acceptance or stopping criterion.",
        "",
        "| Variant | Accepted | Expected | Regularity complete |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["by_variant"][variant]
        lines.append(
            f"| {variant} | {item['accepted']} | {item['expected']} | {item['regularity_complete']} |"
        )
    failures = report["technical_failures"]
    lines.extend(["", "## Technical failures", ""])
    if not failures:
        lines.append("None.")
    else:
        for failure in failures:
            lines.append(
                f"- `{failure['variant']} case{int(failure['case_id']):02d}`: "
                + "; ".join(failure["errors"])
            )
    posthoc = report["regularity_failures"]
    lines.extend(["", "## Posthoc regularity failures", ""])
    if not posthoc:
        lines.append("None.")
    else:
        for failure in posthoc:
            lines.append(
                f"- `{failure['variant']} case{int(failure['case_id']):02d}`: {failure['error']}"
            )
    return "\n".join(lines) + "\n"


def render_camera_readout(
    *,
    records: Sequence[Mapping[str, Any]],
    population: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
) -> str:
    lookup = _lookup_population(population)
    by_key = {
        (str(row["variant"]), int(row["case_id"])): row
        for row in records
        if row.get("technical_accepted")
    }
    bundled_records = reference_bundle.get("records", {})
    if not isinstance(bundled_records, Mapping):
        bundled_records = {}
    ptvreg = bundled_records.get("PTVREG", {})
    mr_reference = bundled_records.get("MR_D_BSCP", {})
    if not isinstance(ptvreg, Mapping):
        ptvreg = {}
    if not isinstance(mr_reference, Mapping):
        mr_reference = {}
    selection = reference_bundle.get("mr_d_bscp_selection")
    mr_weight = (
        _finite_number(selection.get("regularization_value"))
        if isinstance(selection, Mapping)
        else None
    )
    mr_label = "MR-D-BSCP" + ("" if mr_weight is None else f" w={mr_weight:g}")

    lines = [
        "# Camera-ready controlled DUAL-INR readout",
        "",
        f"Optimization acceptance: **{acceptance['status']}** "
        f"({acceptance['accepted_records']}/{acceptance['expected_records']}). "
        "All values below are descriptive; no performance threshold was used.",
        "",
        "## Headline/native-domain TRE",
        "",
        "Across-case mean ± sample SD in mm. Sample quartiles and IQR are retained "
        "in `population_statistics.csv/json`.",
        "",
        "| Variant/reference | Preprocessing | Headline snap TRE | Headline subvoxel TRE |",
        "|---|---|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {VARIANT_META[variant]['preprocessing']} | "
            f"{_mean_sd(lookup.get((variant, 'headline_native_snap_tre_mm')))} | "
            f"{_mean_sd(lookup.get((variant, 'headline_native_subvoxel_tre_mm')))} |"
        )
    for group in ("PTVREG", "MR_D_BSCP"):
        if bundled_records.get(group):
            lines.append(
                f"| {REFERENCE_META[group]['display_name']} (reference) | "
                f"{REFERENCE_META[group]['preprocessing']} | "
                f"{_mean_sd(lookup.get((group, 'headline_native_snap_tre_mm')))} | "
                f"{_mean_sd(lookup.get((group, 'headline_native_subvoxel_tre_mm')))} |"
            )

    lines.extend(
        [
            "",
            "## Registration-grid TRE diagnostics",
            "",
            "| Variant/reference | Grid snap TRE (mm) | Grid subvoxel TRE (mm) | Direct-INR snap TRE (mm) |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {_mean_sd(lookup.get((variant, 'registration_grid_snap_tre_mm')))} | "
            f"{_mean_sd(lookup.get((variant, 'registration_grid_subvoxel_tre_mm')))} | "
            f"{_mean_sd(lookup.get((variant, 'direct_inr_snap_tre_mm')))} |"
        )
    for group in ("PTVREG", "MR_D_BSCP"):
        if bundled_records.get(group):
            lines.append(
                f"| {REFERENCE_META[group]['display_name']} (reference) | "
                f"{_mean_sd(lookup.get((group, 'registration_grid_snap_tre_mm')))} | "
                f"{_mean_sd(lookup.get((group, 'registration_grid_subvoxel_tre_mm')))} | NA |"
            )

    lines.extend(
        [
            "",
            "## Case-by-case headline snap TRE",
            "",
            f"| Case | N0 native | N1 native MR-shaped | P0 pTV 1 mm | P1 pTV 1 mm MR-shaped | pTVReg canonical 1 mm* | {mr_label} 1 mm* |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in EXPECTED_CASES:
        values = [
            _fmt(by_key.get((variant, case), {}).get("headline_native_snap_tre_mm"))
            for variant in VARIANTS
        ]
        p = ptvreg.get(case, {})
        m = mr_reference.get(case, {})
        lines.append(
            f"| {case:02d} | {' | '.join(values)} | "
            f"{_fmt(p.get('headline_native_snap_tre_mm'))} | "
            f"{_fmt(m.get('headline_native_snap_tre_mm'))} |"
        )

    if not ptvreg and not mr_reference:
        lines.extend(
            [
                "",
                "*No reference tables were available or supplied; reference cells are intentionally `NA`.",
            ]
        )
    else:
        lines.extend(["", "### Reference selection and preprocessing", ""])
        if ptvreg:
            lines.append(
                f"- pTVReg tag: `{REFERENCE_META['PTVREG']['protocol_tag']}`. "
                f"{REFERENCE_META['PTVREG']['preprocessing']} Source: "
                f"`{reference_bundle['sources']['ptvreg_case_table']}`."
            )
        if mr_reference:
            selection_rule = (
                selection.get("selection_rule", "single cohort-level setting")
                if isinstance(selection, Mapping)
                else "single cohort-level setting"
            )
            cohort_tre = (
                selection.get("cohort_headline_native_snap_tre_mm")
                if isinstance(selection, Mapping)
                else None
            )
            lines.append(
                f"- MR-D-BSCP tag: `{REFERENCE_META['MR_D_BSCP']['protocol_tag']}`. "
                f"{REFERENCE_META['MR_D_BSCP']['preprocessing']} The fixed reference "
                f"setting is `w={_fmt(mr_weight, 3)}` for all ten cases, selected by "
                f"the rule “{selection_rule}” (cohort mean TRE {_fmt(cohort_tre)} mm). "
                "There is no per-case oracle selection."
            )
        for warning in reference_bundle.get("warnings", []):
            lines.append(f"- Reference warning: {warning}")

        lines.extend(
            [
                "",
                "### Case-by-case 1 mm reference TRE diagnostics",
                "",
                f"| Case | pTVReg native snap | pTVReg native sub | pTVReg grid snap | pTVReg grid sub | {mr_label} native snap | {mr_label} native sub | {mr_label} grid snap | {mr_label} grid sub |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in EXPECTED_CASES:
            p = ptvreg.get(case, {})
            m = mr_reference.get(case, {})
            fields = (
                p.get("headline_native_snap_tre_mm"),
                p.get("headline_native_subvoxel_tre_mm"),
                p.get("registration_grid_snap_tre_mm"),
                p.get("registration_grid_subvoxel_tre_mm"),
                m.get("headline_native_snap_tre_mm"),
                m.get("headline_native_subvoxel_tre_mm"),
                m.get("registration_grid_snap_tre_mm"),
                m.get("registration_grid_subvoxel_tre_mm"),
            )
            lines.append(
                f"| {case:02d} | " + " | ".join(_fmt(value) for value in fields) + " |"
            )

        lines.extend(
            [
                "",
                "### Case-by-case 1 mm reference regularity",
                "",
                f"| Case | pTVReg diffusion FG | pTVReg bending FG | pTVReg SDlogJ FG | pTVReg J≤0 (%) | {mr_label} diffusion FG | {mr_label} bending FG | {mr_label} SDlogJ FG | {mr_label} J≤0 (%) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in EXPECTED_CASES:
            p = ptvreg.get(case, {})
            m = mr_reference.get(case, {})
            fields = (
                p.get("regularity_diffusion_l2_fg"),
                p.get("regularity_bending_energy_fg"),
                p.get("regularity_log_jdet_std_fg"),
                p.get("regularity_jdet_le0_pct_fg"),
                m.get("regularity_diffusion_l2_fg"),
                m.get("regularity_bending_energy_fg"),
                m.get("regularity_log_jdet_std_fg"),
                m.get("regularity_jdet_le0_pct_fg"),
            )
            lines.append(
                f"| {case:02d} | "
                + " | ".join(_fmt(value, 6) for value in fields)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Paired schedule effects",
            "",
            "Deltas are multiresolution minus full-resolution within the same preprocessing protocol.",
            "",
            "| Pair | Headline snap delta (mm) | Grid snap delta (mm) | SDlogJ FG delta | J≤0 FG delta (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _control, _multires, direction in PAIRINGS:
        group = [row for row in pairs if row["direction"] == direction]
        lines.append(
            f"| {direction} | "
            f"{_mean_sd(_describe(row.get('delta_headline_native_snap_tre_mm') for row in group))} | "
            f"{_mean_sd(_describe(row.get('delta_registration_grid_snap_tre_mm') for row in group))} | "
            f"{_mean_sd(_describe(row.get('delta_regularity_log_jdet_std_fg') for row in group))} | "
            f"{_mean_sd(_describe(row.get('delta_regularity_jdet_le0_pct_fg') for row in group))} |"
        )

    lines.extend(
        [
            "",
            "## Common posthoc regularity",
            "",
            "| Variant/reference | Diffusion FG | Bending FG | SDlogJ FG | J≤0 FG (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in (*VARIANTS, "PTVREG", "MR_D_BSCP"):
        if group not in VARIANTS and not bundled_records.get(group):
            continue
        label = (
            group
            if group in VARIANTS
            else REFERENCE_META[group]["display_name"] + " (reference)"
        )
        lines.append(
            f"| {label} | {_mean_sd(lookup.get((group, 'regularity_diffusion_l2_fg')))} | "
            f"{_mean_sd(lookup.get((group, 'regularity_bending_energy_fg')))} | "
            f"{_mean_sd(lookup.get((group, 'regularity_log_jdet_std_fg')))} | "
            f"{_mean_sd(lookup.get((group, 'regularity_jdet_le0_pct_fg')))} |"
        )
    lines.extend(
        [
            "",
            "Diffusion and bending are voxel-coordinate finite-difference diagnostics. Their "
            "magnitudes are not physical-coordinate energies and should not be compared directly "
            "between native anisotropic N and isotropic-1-mm P/reference preprocessing. Jacobian "
            "statistics are dimensionless. P/reference lung masks are metric-only and absent "
            "from optimization.",
            "",
            "## Runtime and peak memory",
            "",
            "| Variant | Training seconds | Total seconds | Peak reserved GiB |",
            "|---|---:|---:|---:|",
        ]
    )
    gib = float(1024**3)
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {_mean_sd(lookup.get((variant, 'training_wall_seconds')))} | "
            f"{_mean_sd(lookup.get((variant, 'total_wall_seconds')))} | "
            f"{_mean_sd(lookup.get((variant, 'peak_total_reserved_bytes')), gib)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- N0↔N1 and P0↔P1 isolate the image-pyramid schedule under their respective preprocessing protocols.",
            "- N↔P changes crop, resampling, intensity preprocessing, and sampling domain; it is a protocol comparison, not a clean method-only effect.",
            "- N1/P1 are MR-D-BSCP-shaped (MR-inspired) image pyramids. They are not literal B-spline control-point schedules and use no knot insertion.",
            "- A single frozen DUAL hyperparameter recipe was transferred across protocols. These rows are controlled baselines, not claims of protocol-specific optimal tuning.",
            "- The original/native and pTVReg registration-grid TRE conventions are both retained; only explicitly matched domains should be compared.",
            "- pTVReg and MR-D-BSCP rows are contextual 1 mm references with separately tagged preprocessing. MR-D-BSCP uses one cohort-selected setting for all cases, never case-wise best values.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    *,
    run_root: Path,
    review_root: Path,
    seed: int = DEFAULT_SEED,
    allow_partial: bool = False,
    ptvreg_reference_path: Path | None = None,
    mr_reference_path: Path | None = None,
    mr_reference_mean_path: Path | None = None,
    mr_reference_reg_value: float | None = None,
    use_repo_default_references: bool = True,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    review_root = review_root.resolve()
    if not run_root.is_dir():
        raise AggregationError(f"run root does not exist: {run_root}")
    records = [
        inspect_record(run_root=run_root, variant=variant, case=case, seed=seed)
        for variant in VARIANTS
        for case in EXPECTED_CASES
    ]
    regularity = add_regularity(records)
    pairs = paired_comparisons(records)
    reference_bundle = load_reference_bundle(
        ptvreg_path=ptvreg_reference_path,
        mr_case_path=mr_reference_path,
        mr_cohort_path=mr_reference_mean_path,
        mr_reg_value=mr_reference_reg_value,
        use_repo_defaults=use_repo_default_references,
    )
    population = population_statistics(records) + reference_population_statistics(
        reference_bundle
    )

    accepted = [row for row in records if row["technical_accepted"]]
    technical_failures = [
        {
            "variant": row["variant"],
            "case_id": row["case_id"],
            "case_dir": row["case_dir"],
            "errors": row["technical_errors"],
        }
        for row in records
        if not row["technical_accepted"]
    ]
    regularity_failures = [row for row in regularity if row["status"] == "failed"]
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        group = [row for row in records if row["variant"] == variant]
        by_variant[variant] = {
            "expected": len(EXPECTED_CASES),
            "accepted": sum(bool(row["technical_accepted"]) for row in group),
            "regularity_complete": sum(
                row["regularity_status"] == "complete" for row in group
            ),
        }
    complete = len(accepted) == len(VARIANTS) * len(EXPECTED_CASES)
    acceptance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if complete else ("partial" if allow_partial else "fail"),
        "run_root": str(run_root),
        "review_root": str(review_root),
        "seed": seed,
        "expected_records": len(VARIANTS) * len(EXPECTED_CASES),
        "accepted_records": len(accepted),
        "unique_accepted_variant_case_seed": len(
            {(row["variant"], row["case_id"], row["seed"]) for row in accepted}
        ),
        "complete_40_record_cohort": complete,
        "scientific_metric_gate": None,
        "scientific_metric_policy": (
            "TRE and regularity magnitudes never determine technical acceptance or execution continuation."
        ),
        "by_variant": by_variant,
        "technical_failures": technical_failures,
        "regularity_failures": regularity_failures,
        "regularity_complete_records": sum(
            row["status"] == "complete" for row in regularity
        ),
        "reference_sources": reference_bundle["sources"],
        "reference_warnings": reference_bundle["warnings"],
        "ptvreg_reference_cases": sorted(reference_bundle["records"]["PTVREG"]),
        "mr_d_bscp_reference_cases": sorted(reference_bundle["records"]["MR_D_BSCP"]),
        "mr_d_bscp_reference_selection": reference_bundle["mr_d_bscp_selection"],
    }
    index_payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "records": records,
        "external_reference_bundle": reference_bundle,
    }
    regularity_payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "sequential_posthoc": True,
        "metric_function": (
            "scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity."
            "compute_common_regularity"
        ),
        "records": regularity,
        "external_reference_records": reference_bundle["records"],
    }
    population_payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "statistics": (
            "Across-case mean, sample SD, median, linearly interpolated Q1/Q3, "
            "and sample IQR (Q3-Q1); no controlled-run selection by metric."
        ),
        "rows": population,
        "external_reference_bundle": reference_bundle,
    }

    portable_roots = (
        (run_root, "${RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
        (REPOSITORY_ROOT, "${REPO_ROOT}"),
    )
    review_records = portable_review_value(records, portable_roots)
    review_regularity = portable_review_value(regularity, portable_roots)
    review_population = portable_review_value(population, portable_roots)
    review_pairs = portable_review_value(pairs, portable_roots)
    review_acceptance = portable_review_value(acceptance, portable_roots)
    review_reference_bundle = portable_review_value(
        reference_bundle, portable_roots
    )

    review_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(review_root / "optimization_index.csv", review_records)
    _atomic_json(
        review_root / "optimization_index.json",
        portable_review_value(index_payload, portable_roots),
    )
    _atomic_csv(review_root / "regularity_metrics.csv", review_regularity)
    _atomic_json(
        review_root / "regularity_metrics.json",
        portable_review_value(regularity_payload, portable_roots),
    )
    _atomic_csv(review_root / "population_statistics.csv", review_population)
    _atomic_json(
        review_root / "population_statistics.json",
        portable_review_value(population_payload, portable_roots),
    )
    _atomic_csv(review_root / "paired_comparisons.csv", review_pairs)
    _atomic_json(review_root / "acceptance_report.json", review_acceptance)
    _atomic_text(
        review_root / "acceptance_report.md",
        render_acceptance_markdown(review_acceptance),
    )
    _atomic_text(
        review_root / "camera_ready_readout.md",
        render_camera_readout(
            records=review_records,
            population=review_population,
            pairs=review_pairs,
            acceptance=review_acceptance,
            reference_bundle=review_reference_bundle,
        ),
    )
    if not complete and not allow_partial:
        raise AggregationError(
            f"complete cohort gate failed: accepted {len(accepted)}/40; "
            f"see {review_root / 'acceptance_report.md'}"
        )
    return {
        "acceptance": acceptance,
        "record_count": len(records),
        "accepted_record_count": len(accepted),
        "regularity_complete_count": acceptance["regularity_complete_records"],
        "review_root": str(review_root),
        "outputs": [str(review_root / name) for name in OUTPUT_NAMES],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a monitoring snapshot without requiring all 40 accepted records.",
    )
    parser.add_argument(
        "--ptvreg-reference",
        type=Path,
        help=(
            "Optional pTVReg CSV/TSV/JSON case table. It must contain exactly one "
            "explicit pTVReg row for each DIR-Lab case."
        ),
    )
    parser.add_argument(
        "--mr-reference",
        type=Path,
        help=(
            "Optional MR-D-BSCP CSV/TSV/JSON case table. Rows are filtered to one "
            "cohort-level regularization setting; per-case oracle selection is prohibited."
        ),
    )
    parser.add_argument(
        "--mr-reference-means",
        type=Path,
        help=(
            "Optional MR-D-BSCP cohort-mean table used to select one lowest-mean-TRE "
            "regularization setting before case rows are read."
        ),
    )
    parser.add_argument(
        "--mr-reference-reg-value",
        type=float,
        help=(
            "Explicit single MR-D-BSCP regularization value for a custom case table. "
            "Use only when no cohort-mean selection table is supplied."
        ),
    )
    parser.add_argument(
        "--no-default-references",
        action="store_true",
        help=(
            "Disable the tracked repo reference bundle. By default, when present, the "
            f"pTVReg table {DEFAULT_PTVREG_REFERENCE.relative_to(REPOSITORY_ROOT)} and "
            f"MR case/mean tables below {DEFAULT_MR_REFERENCE.parent.relative_to(REPOSITORY_ROOT)} are used."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = aggregate(
            run_root=args.run_root,
            review_root=args.review_root,
            seed=args.seed,
            allow_partial=args.allow_partial,
            ptvreg_reference_path=args.ptvreg_reference,
            mr_reference_path=args.mr_reference,
            mr_reference_mean_path=args.mr_reference_means,
            mr_reference_reg_value=args.mr_reference_reg_value,
            use_repo_default_references=not args.no_default_references,
        )
    except AggregationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Accepted {result['accepted_record_count']}/40 optimization records; "
        f"regularity complete {result['regularity_complete_count']}/40; "
        f"review: {result['review_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
