#!/usr/bin/env python3
"""Aggregate the DUAL pTV-1mm fixed-lung-mask revision pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.lungct_dual_inr_author.portable_paths import (
    portable_review_value,
)
from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (
    REPORT_METRICS,
    compute_common_regularity,
)
from scripts.experiments.lungct_dual_inr_controlled.adapter import (
    AUTHOR_BLUR_SIGMAS,
    AUTHOR_BLUR_TOTAL_STEPS,
    LOCKED_STAGE_LENGTHS,
    LOCKED_STAGE_STARTS,
    LUNGMASK_REVISION_VARIANTS,
    MULTIRESOLUTION_FACTORS,
    SCHEDULE_AUTHOR_BLUR,
    variant_loss_branch,
    variant_schedule,
)


EXPECTED_STEPS = 2500
EXPECTED_FINE_START = 400
PILOT_CASES = (1, 8)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class AggregationError(RuntimeError):
    """Raised when a revision record violates the technical contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    _atomic_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
                writer.writerow(
                    {
                        key: json.dumps(_jsonable(row.get(key)), sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else "" if row.get(key) is None else row.get(key)
                        for key in fieldnames
                    }
                )
    os.replace(temporary, path)


def parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in str(text).replace(" ", "").split(",") if part)
    if not values or any(value not in range(1, 11) for value in values):
        raise argparse.ArgumentTypeError("cases must be comma-separated values in 1..10")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("cases must be unique")
    return values


def parse_variants(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().upper() for part in str(text).split(",") if part.strip())
    if not values or any(value not in LUNGMASK_REVISION_VARIANTS for value in values):
        raise argparse.ArgumentTypeError(
            f"variants must be from {','.join(LUNGMASK_REVISION_VARIANTS)}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("variants must be unique")
    return values


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregationError(f"expected JSON object in {path}")
    return payload


def finite_metric(value: Any) -> float | None:
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


def require_finite(value: Any, label: str) -> float:
    result = finite_metric(value)
    if result is None:
        raise AggregationError(f"{label} missing or non-finite: {value!r}")
    return result


def nested(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def validate_image_pyramid_history(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise AggregationError(f"cannot read history {path}: {exc}") from exc
    if len(rows) != EXPECTED_STEPS:
        raise AggregationError(f"history has {len(rows)} rows; expected {EXPECTED_STEPS}")
    stage_starts: list[int] = []
    factors_seen: list[int] = []
    fine_first: int | None = None
    loss_values: list[float] = []
    for index, row in enumerate(rows):
        try:
            zero_step = int(row.get("zero_step", index))
            stage = int(row["stage_index"])
            factor = int(row["image_factor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(f"history row {index + 1} has invalid step/stage") from exc
        expected_stage = 0 if zero_step < 400 else 1 if zero_step < 800 else 2 if zero_step < 1600 else 3
        if stage != expected_stage:
            raise AggregationError(f"step {zero_step} stage {stage} != {expected_stage}")
        expected_factor = MULTIRESOLUTION_FACTORS[stage]
        if factor != expected_factor:
            raise AggregationError(f"step {zero_step} factor {factor} != {expected_factor}")
        fine_active = str(row.get("fine_active", "")).lower() in {"true", "1", "yes"}
        if fine_active != (zero_step >= EXPECTED_FINE_START):
            raise AggregationError(f"step {zero_step} fine_active contract failed")
        if fine_active and fine_first is None:
            fine_first = zero_step
        if zero_step in LOCKED_STAGE_STARTS:
            stage_starts.append(zero_step)
            factors_seen.append(factor)
        loss_values.append(require_finite(row.get("loss_total"), f"step {zero_step} loss_total"))
        require_finite(row.get("loss_similarity"), f"step {zero_step} loss_similarity")
        require_finite(row.get("loss_spatial_weighted"), f"step {zero_step} loss_spatial_weighted")
    if tuple(stage_starts) != LOCKED_STAGE_STARTS:
        raise AggregationError(f"stage starts {stage_starts} != {LOCKED_STAGE_STARTS}")
    if tuple(factors_seen) != MULTIRESOLUTION_FACTORS:
        raise AggregationError(f"image factors {factors_seen} != {MULTIRESOLUTION_FACTORS}")
    if fine_first != EXPECTED_FINE_START:
        raise AggregationError(f"fine activation {fine_first} != {EXPECTED_FINE_START}")
    return {
        "history_rows": len(rows),
        "stage_starts": stage_starts,
        "stage_lengths": list(LOCKED_STAGE_LENGTHS),
        "image_factors": factors_seen,
        "blur_sigmas": [],
        "fine_activation_step": fine_first,
        "mean_loss_total": statistics.fmean(loss_values),
        "final_loss_total": loss_values[-1],
    }


def validate_author_blur_history(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise AggregationError(f"cannot read history {path}: {exc}") from exc
    if len(rows) != AUTHOR_BLUR_TOTAL_STEPS:
        raise AggregationError(
            f"history has {len(rows)} rows; expected {AUTHOR_BLUR_TOTAL_STEPS}"
        )
    stage_starts: list[int] = []
    sigmas_seen: list[float] = []
    fine_first: int | None = None
    loss_values: list[float] = []
    previous_stage = -1
    for index, row in enumerate(rows):
        try:
            zero_step = int(row.get("zero_step", index))
            stage = int(row["stage_index"])
            sigma = float(row["blur_sigma"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(f"history row {index + 1} has invalid step/stage/sigma") from exc
        if zero_step != index:
            raise AggregationError(f"history row {index + 1} zero_step {zero_step} != {index}")
        if stage < previous_stage or stage < 0 or stage >= len(AUTHOR_BLUR_SIGMAS):
            raise AggregationError(f"step {zero_step} invalid author blur stage {stage}")
        expected_sigma = float(AUTHOR_BLUR_SIGMAS[stage])
        if not math.isclose(sigma, expected_sigma, rel_tol=0.0, abs_tol=1e-8):
            raise AggregationError(f"step {zero_step} sigma {sigma} != {expected_sigma}")
        if stage != previous_stage:
            stage_starts.append(zero_step)
            sigmas_seen.append(sigma)
            previous_stage = stage
        fine_active = str(row.get("fine_active", "")).lower() in {"true", "1", "yes"}
        if fine_active != (stage >= 1):
            raise AggregationError(f"step {zero_step} fine_active contract failed")
        if fine_active and fine_first is None:
            fine_first = zero_step
        loss_values.append(require_finite(row.get("loss_total"), f"step {zero_step} loss_total"))
        require_finite(row.get("loss_similarity"), f"step {zero_step} loss_similarity")
        require_finite(row.get("loss_spatial_weighted"), f"step {zero_step} loss_spatial_weighted")
        mask_weighted = require_finite(row.get("loss_mask_weighted"), f"step {zero_step} loss_mask_weighted")
        if not math.isclose(mask_weighted, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise AggregationError(f"step {zero_step} mask loss is not disabled")
    if tuple(sigmas_seen) != AUTHOR_BLUR_SIGMAS:
        raise AggregationError(f"visited blur sigmas {sigmas_seen} != {AUTHOR_BLUR_SIGMAS}")
    if fine_first != stage_starts[1]:
        raise AggregationError(f"fine activation {fine_first} != stage 1 start {stage_starts[1]}")
    lengths = [
        (stage_starts[index + 1] if index + 1 < len(stage_starts) else AUTHOR_BLUR_TOTAL_STEPS)
        - start
        for index, start in enumerate(stage_starts)
    ]
    return {
        "history_rows": len(rows),
        "stage_starts": stage_starts,
        "stage_lengths": lengths,
        "image_factors": [1] * len(stage_starts),
        "blur_sigmas": sigmas_seen,
        "fine_activation_step": fine_first,
        "mean_loss_total": statistics.fmean(loss_values),
        "final_loss_total": loss_values[-1],
    }


def validate_history(path: Path, variant: str) -> dict[str, Any]:
    if variant_schedule(variant) == SCHEDULE_AUTHOR_BLUR:
        return validate_author_blur_history(path)
    return validate_image_pyramid_history(path)


def tre_values(summary: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "headline_native_snap_tre_mm": finite_metric(
            nested(summary, "metrics.native_crop_tre_snap_to_voxel")
            or nested(summary, "metrics.headline_tre")
        ),
        "headline_native_subvoxel_tre_mm": finite_metric(
            nested(summary, "metrics.native_crop_tre_subvoxel")
        ),
        "registration_grid_snap_tre_mm": finite_metric(
            nested(summary, "metrics.registration_grid.dense_snap")
            or nested(summary, "metrics.tre_snap_to_voxel")
        ),
        "registration_grid_subvoxel_tre_mm": finite_metric(
            nested(summary, "metrics.registration_grid.dense_subvoxel")
            or nested(summary, "metrics.tre_subvoxel")
        ),
        "direct_inr_snap_tre_mm": finite_metric(
            nested(summary, "metrics.registration_grid.direct_inr.snap")
        ),
        "direct_inr_subvoxel_tre_mm": finite_metric(
            nested(summary, "metrics.registration_grid.direct_inr.subvoxel")
        ),
    }


def compute_or_load_regularity(case_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    output = case_dir / "common_regularity.json"
    if output.is_file():
        payload = load_json(output)
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping) and all(finite_metric(metrics.get(m)) is not None for m in REPORT_METRICS):
            payload["cache_status"] = "loaded"
            return payload
    field = np.load(case_dir / "dense_dvf_xyz_voxel.npy", allow_pickle=False)
    mask = np.load(case_dir / "fixed_metric_mask_xyz.npy", allow_pickle=False)
    computed = compute_common_regularity(
        disp_vox_xyz=field,
        fixed_lobe_union_xyz=mask,
        spacing_xyz_mm=record["spacing_xyz_mm"],
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "variant": record["variant"],
        "case_id": record["case_id"],
        "protocol": "PL",
        "protocol_tag": "ptv_landmark_crop_1mm_fixed_t00_lungmask",
        "metrics": computed["metrics"],
        "conversion": computed["conversion"],
        "cache_status": "computed",
    }
    _atomic_json(output, payload)
    return payload


def inspect_record(*, run_root: Path, variant: str, case_id: int, seed: int) -> dict[str, Any]:
    case_dir = run_root / variant.lower() / f"case_{case_id:02d}"
    record: dict[str, Any] = {
        "variant": variant,
        "case_id": int(case_id),
        "seed": int(seed),
        "case_dir": str(case_dir),
        "protocol": "PL",
        "protocol_tag": "ptv_landmark_crop_1mm_fixed_t00_lungmask",
        "loss_branch_identifier": variant_loss_branch(variant),
        "technical_accepted": False,
        "technical_errors": [],
        "regularity_status": "not_run",
        "regularity_error": None,
    }
    errors: list[str] = record["technical_errors"]
    required = {
        "DONE": case_dir / "DONE",
        "summary": case_dir / "summary.json",
        "history": case_dir / "history.csv",
        "dense_dvf": case_dir / "dense_dvf_xyz_voxel.npy",
        "metric_mask": case_dir / "fixed_metric_mask_xyz.npy",
    }
    for label, path in required.items():
        if not path.exists():
            errors.append(f"required artifact missing/empty: {label}")
        elif label != "DONE" and path.is_file() and path.stat().st_size == 0:
            errors.append(f"required artifact missing/empty: {label}")
    if errors:
        return record
    try:
        summary = load_json(required["summary"])
        if str(summary.get("status", "")).lower() != "complete":
            raise AggregationError("summary status is not complete")
        if str(summary.get("variant", "")).upper() != variant:
            raise AggregationError("summary variant mismatch")
        if str(summary.get("protocol", "")).upper() != "PL":
            raise AggregationError("summary protocol is not PL")
        if str(summary.get("loss_branch_identifier")) != variant_loss_branch(variant):
            raise AggregationError("loss branch identifier mismatch")
        if int(summary.get("case_id")) != int(case_id) or int(summary.get("seed")) != int(seed):
            raise AggregationError("summary case/seed mismatch")
        schedule = variant_schedule(variant)
        if str(summary.get("schedule_name", schedule)) != schedule:
            raise AggregationError("summary schedule name mismatch")
        if schedule == SCHEDULE_AUTHOR_BLUR:
            if tuple(float(v) for v in summary.get("blur_sigmas", ())) != AUTHOR_BLUR_SIGMAS:
                raise AggregationError("summary blur sigmas disagree with author plan")
            if int(sum(int(v) for v in summary.get("stage_lengths", ()))) != AUTHOR_BLUR_TOTAL_STEPS:
                raise AggregationError("summary author-blur stage lengths do not sum to 3000")
            if tuple(int(v) for v in summary.get("image_factors", ())) != (1, 1, 1):
                raise AggregationError("summary author-blur image factors disagree with plan")
        else:
            if tuple(int(v) for v in summary.get("stage_lengths", ())) != LOCKED_STAGE_LENGTHS:
                raise AggregationError("summary stage lengths disagree with plan")
            if tuple(int(v) for v in summary.get("stage_starts", ())) != LOCKED_STAGE_STARTS:
                raise AggregationError("summary stage starts disagree with plan")
            if tuple(int(v) for v in summary.get("image_factors", ())) != MULTIRESOLUTION_FACTORS:
                raise AggregationError("summary image factors disagree with plan")
        if str(summary.get("sampling_rule")) != "fixed_t00_lung_mask":
            raise AggregationError("sampling rule is not fixed_t00_lung_mask")
        if str(summary.get("metric_mask_rule")) != "fixed_t00_lung_mask":
            raise AggregationError("metric mask rule is not fixed_t00_lung_mask")
        data = summary.get("data")
        if not isinstance(data, Mapping):
            raise AggregationError("summary lacks data contract")
        sampling_voxels = int(data.get("sampling_mask_voxels", 0))
        metric_voxels = int(data.get("metric_mask_voxels", 0))
        if sampling_voxels <= 0 or metric_voxels <= 0 or sampling_voxels != metric_voxels:
            raise AggregationError("sampling and metric mask voxel counts must match and be positive")
        spacing = tuple(float(value) for value in summary.get("spacing_xyz_mm", ()))
        if len(spacing) != 3 or min(spacing) <= 0 or not all(math.isfinite(v) for v in spacing):
            raise AggregationError("invalid spacing_xyz_mm")
        history = validate_history(required["history"], variant)
        field = np.load(required["dense_dvf"], mmap_mode="r", allow_pickle=False)
        mask = np.load(required["metric_mask"], mmap_mode="r", allow_pickle=False)
        if field.ndim != 4 or field.shape[-1] != 3 or tuple(field.shape[:3]) != tuple(mask.shape):
            raise AggregationError("dense field and metric mask shapes disagree")
        values = tre_values(summary)
        for key in (
            "headline_native_snap_tre_mm",
            "headline_native_subvoxel_tre_mm",
            "registration_grid_snap_tre_mm",
            "registration_grid_subvoxel_tre_mm",
        ):
            if values[key] is None:
                raise AggregationError(f"{key} is missing or non-finite")
        record.update(
            {
                "technical_accepted": True,
                "shape_xyz": list(field.shape[:3]),
                "spacing_xyz_mm": list(spacing),
                "sampling_mask_voxels": sampling_voxels,
                "metric_mask_voxels": metric_voxels,
                "run_mode": summary.get("run_mode"),
                "run_tag": summary.get("run_tag"),
                "regularization_multiplier": finite_metric(
                    summary.get("regularization_multiplier")
                ),
                "resolved_regularization_weights": summary.get(
                    "resolved_regularization_weights"
                ),
                "sampling_rule": summary.get("sampling_rule"),
                "metric_mask_rule": summary.get("metric_mask_rule"),
                "protocol_tags": summary.get("protocol_tags"),
                "source_paths": data.get("source_paths"),
                "schedule_name": schedule,
                "stage_lengths": history["stage_lengths"],
                "stage_starts": history["stage_starts"],
                "image_factors": history["image_factors"],
                "blur_sigmas": history["blur_sigmas"],
                "fine_activation_step": history["fine_activation_step"],
                "history_rows": history["history_rows"],
                "mean_loss_total": history["mean_loss_total"],
                "final_loss_total": history["final_loss_total"],
                "training_wall_seconds": finite_metric(nested(summary, "runtime.training_wall_seconds")),
                "total_wall_seconds": finite_metric(nested(summary, "runtime.total_wall_seconds")),
                "peak_total_reserved_bytes": finite_metric(nested(summary, "cuda_peak_memory_bytes.total_reserved")),
                **values,
            }
        )
    except (AggregationError, OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    return record


def add_regularity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            payload = compute_or_load_regularity(Path(record["case_dir"]), record)
            metrics = payload["metrics"]
            for metric in REPORT_METRICS:
                record[f"regularity_{metric}"] = require_finite(
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
                    "path": str(Path(record["case_dir"]) / "common_regularity.json"),
                    **{metric: float(metrics[metric]) for metric in REPORT_METRICS},
                }
            )
        except BaseException as exc:
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


def describe(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
    }


def population_statistics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "headline_native_snap_tre_mm",
        "headline_native_subvoxel_tre_mm",
        "registration_grid_snap_tre_mm",
        "registration_grid_subvoxel_tre_mm",
        "direct_inr_snap_tre_mm",
        "direct_inr_subvoxel_tre_mm",
        "training_wall_seconds",
        "total_wall_seconds",
        "regularity_diffusion_l2_fg",
        "regularity_bending_energy_fg",
        "regularity_log_jdet_std_fg",
        "regularity_jdet_le0_pct_fg",
    )
    rows: list[dict[str, Any]] = []
    accepted = [record for record in records if record["technical_accepted"]]
    for variant in LUNGMASK_REVISION_VARIANTS:
        group = [record for record in accepted if record["variant"] == variant]
        for metric in metrics:
            rows.append(
                {
                    "variant": variant,
                    "loss_branch_identifier": variant_loss_branch(variant),
                    "metric": metric,
                    **describe([record.get(metric) for record in group]),
                }
            )
    return rows


def render_acceptance(report: Mapping[str, Any]) -> str:
    lines = [
        "# Lung-Mask Revision Pilot Acceptance",
        "",
        f"- Status: `{report['status']}`",
        f"- Accepted records: `{report['accepted_records']}/{report['expected_records']}`",
        f"- Regularity complete: `{report['regularity_complete_records']}/{report['expected_records']}`",
        "- Technical acceptance ignores TRE and regularity magnitude.",
        "",
        "| Variant | Loss branch | Accepted | Expected | Regularity complete |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, item in report["by_variant"].items():
        lines.append(
            f"| {variant} | `{item['loss_branch_identifier']}` | "
            f"{item['accepted']} | {item['expected']} | {item['regularity_complete']} |"
        )
    if report["technical_failures"]:
        lines.extend(["", "## Technical Failures", ""])
        for failure in report["technical_failures"]:
            lines.append(
                f"- `{failure['variant']} case{int(failure['case_id']):02d}`: "
                + "; ".join(str(error) for error in failure["errors"])
            )
    if report["regularity_failures"]:
        lines.extend(["", "## Regularity Failures", ""])
        for failure in report["regularity_failures"]:
            lines.append(
                f"- `{failure['variant']} case{int(failure['case_id']):02d}`: {failure['error']}"
            )
    return "\n".join(lines) + "\n"


def render_pilot_report(records: Sequence[Mapping[str, Any]], population: Sequence[Mapping[str, Any]]) -> str:
    lookup = {
        (row["variant"], row["metric"]): row
        for row in population
        if row.get("n")
    }

    def fmt(row: Mapping[str, Any] | None, digits: int = 4) -> str:
        if not row or row.get("mean") is None:
            return "n/a"
        return f"{float(row['mean']):.{digits}f} +/- {float(row['sd']):.{digits}f}"

    lines = [
        "# Lung-Mask Revision Pilot Readout",
        "",
        "This report covers the pTVReg-style 1 mm DUAL-INR branch using the fixed T00 lung mask as both optimization and metric mask.",
        "",
        "| Variant | Loss branch | Native snap TRE | Reg-grid subvoxel TRE | FG SDlogJ | FG J<=0 pct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in LUNGMASK_REVISION_VARIANTS:
        lines.append(
            f"| {variant} | `{variant_loss_branch(variant)}` | "
            f"{fmt(lookup.get((variant, 'headline_native_snap_tre_mm')))} | "
            f"{fmt(lookup.get((variant, 'registration_grid_subvoxel_tre_mm')))} | "
            f"{fmt(lookup.get((variant, 'regularity_log_jdet_std_fg')), 6)} | "
            f"{fmt(lookup.get((variant, 'regularity_jdet_le0_pct_fg')), 6)} |"
        )
    lines.extend(
        [
            "",
            "## Case Rows",
            "",
            "| Variant | Case | Loss branch | Native snap TRE | Native subvoxel TRE | Sampling voxels | Metric voxels |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        if not record["technical_accepted"]:
            continue
        lines.append(
            f"| {record['variant']} | {int(record['case_id']):02d} | "
            f"`{record['loss_branch_identifier']}` | "
            f"{float(record['headline_native_snap_tre_mm']):.4f} | "
            f"{float(record['headline_native_subvoxel_tre_mm']):.4f} | "
            f"{int(record['sampling_mask_voxels'])} | {int(record['metric_mask_voxels'])} |"
        )
    lines.extend(
        [
            "",
            "Acceptance is technical only: expected artifacts, finite metrics, protocol identity, expected stages, and completed posthoc regularity. TRE and regularity magnitudes are decision inputs after inspection, not completion gates.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    *,
    run_root: Path,
    review_root: Path,
    variants: Sequence[str],
    cases: Sequence[int],
    seed: int,
    allow_partial: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    review_root = review_root.resolve()
    if not run_root.is_dir():
        raise AggregationError(f"run root does not exist: {run_root}")
    records = [
        inspect_record(run_root=run_root, variant=variant, case_id=case, seed=seed)
        for variant in variants
        for case in cases
    ]
    regularity = add_regularity(records)
    population = population_statistics(records)
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
    for variant in variants:
        group = [row for row in records if row["variant"] == variant]
        by_variant[variant] = {
            "loss_branch_identifier": variant_loss_branch(variant),
            "expected": len(cases),
            "accepted": sum(bool(row["technical_accepted"]) for row in group),
            "regularity_complete": sum(row["regularity_status"] == "complete" for row in group),
        }
    complete = len(accepted) == len(variants) * len(cases)
    acceptance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if complete else ("partial" if allow_partial else "fail"),
        "run_root": str(run_root),
        "review_root": str(review_root),
        "variants": list(variants),
        "cases": list(cases),
        "seed": int(seed),
        "expected_records": len(variants) * len(cases),
        "accepted_records": len(accepted),
        "regularity_complete_records": sum(row["status"] == "complete" for row in regularity),
        "scientific_metric_gate": None,
        "scientific_metric_policy": (
            "TRE and regularity magnitudes never determine technical acceptance or execution continuation."
        ),
        "by_variant": by_variant,
        "technical_failures": technical_failures,
        "regularity_failures": regularity_failures,
    }
    portable_roots = (
        (run_root, "${RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
        (REPOSITORY_ROOT, "${REPO_ROOT}"),
    )
    review_records = portable_review_value(records, portable_roots)
    review_regularities = portable_review_value(regularity, portable_roots)
    review_population = portable_review_value(population, portable_roots)
    review_acceptance = portable_review_value(acceptance, portable_roots)
    review_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(review_root / "optimization_index.csv", review_records)
    _atomic_json(
        review_root / "optimization_index.json",
        portable_review_value(
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "run_root": str(run_root),
                "records": records,
            },
            portable_roots,
        ),
    )
    _atomic_csv(review_root / "regularity_metrics.csv", review_regularities)
    _atomic_json(
        review_root / "regularity_metrics.json",
        portable_review_value(
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "run_root": str(run_root),
                "records": regularity,
            },
            portable_roots,
        ),
    )
    _atomic_csv(review_root / "population_statistics.csv", review_population)
    _atomic_json(
        review_root / "population_statistics.json",
        {"schema_version": 1, "created_utc": utc_now(), "rows": review_population},
    )
    _atomic_json(review_root / "acceptance_report.json", review_acceptance)
    _atomic_text(review_root / "acceptance_report.md", render_acceptance(review_acceptance))
    _atomic_text(review_root / "pilot_readout.md", render_pilot_report(review_records, review_population))
    if not complete and not allow_partial:
        raise AggregationError(
            f"complete pilot gate failed: accepted {len(accepted)}/{len(variants) * len(cases)}"
        )
    return {
        "accepted_record_count": len(accepted),
        "expected_record_count": len(variants) * len(cases),
        "regularity_complete_count": acceptance["regularity_complete_records"],
        "review_root": str(review_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--variants", type=parse_variants, default=LUNGMASK_REVISION_VARIANTS)
    parser.add_argument("--cases", type=parse_csv_ints, default=PILOT_CASES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = aggregate(
            run_root=args.run_root,
            review_root=args.review_root,
            variants=args.variants,
            cases=args.cases,
            seed=args.seed,
            allow_partial=args.allow_partial,
        )
    except AggregationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Accepted {result['accepted_record_count']}/{result['expected_record_count']} "
        f"records; regularity complete {result['regularity_complete_count']}; "
        f"review: {result['review_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
