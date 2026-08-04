#!/usr/bin/env python3
"""Aggregate a complete paired DUAL-INR-DIR A0/A1 cohort.

Aggregation is acceptance-gated: every requested case must have a strongly
validated A0 and A1 record and every A1 blur boundary must exactly match its
paired A0 boundary.  The command writes compact CSV, JSON, and Markdown review
artifacts only after the complete cohort passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.experiments.lungct_dual_inr_author.launch_author import (  # type: ignore
        CASES_BY_MODE,
        LOCKED_SEED,
        LOCKED_TOTAL_STEPS,
        SMOKE_STEPS,
        atomic_write_json,
        code_sha256_binding,
        file_sha256,
        verify_code_snapshot,
        validate_done_summary,
    )
    from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (  # type: ignore
        COMMON_SOURCE,
        POSTHOC_SOURCE,
        REPORT_METRICS,
    )
    from scripts.experiments.lungct_dual_inr_author.portable_paths import (  # type: ignore
        expand_portable_review_value,
        portable_review_value,
    )
else:
    from .launch_author import (
        CASES_BY_MODE,
        LOCKED_SEED,
        LOCKED_TOTAL_STEPS,
        SMOKE_STEPS,
        atomic_write_json,
        code_sha256_binding,
        file_sha256,
        verify_code_snapshot,
        validate_done_summary,
    )
    from .posthoc_author_regularity import COMMON_SOURCE, POSTHOC_SOURCE, REPORT_METRICS
    from .portable_paths import expand_portable_review_value, portable_review_value


VARIANTS = ("A0", "A1")
COMPONENT_FIELDS = (
    "loss_total",
    "loss_similarity",
    "loss_spatial_weighted",
    "loss_mask_raw",
    "loss_mask_weighted",
    "coarse_jacobian_raw",
    "coarse_jacobian_weighted",
    "coarse_laplacian_raw",
    "coarse_laplacian_weighted",
    "fine_jacobian_raw",
    "fine_jacobian_weighted",
    "fine_laplacian_raw",
    "fine_laplacian_weighted",
)
TRE_FIELDS = (
    "initial_snap_tre_mm",
    "initial_subvoxel_tre_mm",
    "direct_inr_snap_tre_mm",
    "direct_inr_subvoxel_tre_mm",
    "dense_snap_tre_mm",
    "dense_subvoxel_tre_mm",
    "direct_dense_snap_abs_difference_mm",
    "direct_dense_subvoxel_abs_difference_mm",
)
RUNTIME_MEMORY_FIELDS = (
    "training_wall_seconds",
    "total_wall_seconds",
    "cuda_training_allocated_bytes",
    "cuda_training_reserved_bytes",
    "cuda_total_allocated_bytes",
    "cuda_total_reserved_bytes",
)
OUTPUT_FILENAMES = (
    "dual_author_per_case.csv",
    "dual_author_paired_deltas.csv",
    "dual_author_population.csv",
    "dual_author_boundary_audit.csv",
    "dual_author_aggregate.json",
    "dual_author_report.md",
)
REGULARITY_PREFIX = "regularity_"


class AggregationError(RuntimeError):
    """Raised when the requested cohort is incomplete or internally inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregationError(f"Expected a JSON object in {path}")
    return payload


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AggregationError(f"{label} is non-finite: {value!r}")
    return result


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise AggregationError(f"Missing summary field: {'.'.join(keys)}")
        current = current[key]
    return current


def _read_history(path: Path, expected_steps: int) -> dict[str, Any]:
    sums = {field: 0.0 for field in COMPONENT_FIELDS}
    last: dict[str, str] | None = None
    stage_counts: dict[int, int] = {}
    count = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(COMPONENT_FIELDS) | {"stage_index", "step", "zero_step"}
        missing -= set(reader.fieldnames or ())
        if missing:
            raise AggregationError(f"{path} lacks columns {sorted(missing)}")
        for count, row in enumerate(reader, start=1):
            last = row
            stage = int(row["stage_index"])
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            for field in COMPONENT_FIELDS:
                sums[field] += _finite_float(row[field], f"{path.name} row {count} {field}")
    if count != expected_steps or last is None:
        raise AggregationError(f"{path} has {count} rows; expected {expected_steps}")
    metrics: dict[str, Any] = {"history_rows": count}
    for field in COMPONENT_FIELDS:
        metrics[f"mean_{field}"] = sums[field] / count
        metrics[f"final_{field}"] = _finite_float(last[field], f"final {field}")
    for stage in range(3):
        metrics[f"stage_{stage}_history_steps"] = stage_counts.get(stage, 0)
    return metrics


def _read_stage_trace(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise AggregationError(f"Empty stage trace: {path}")
    rows: list[dict[str, Any]] = []
    flat: dict[str, Any] = {"stage_count": len(raw_rows)}
    for position, row in enumerate(raw_rows):
        try:
            stage = int(row["stage_index"])
            sigma = _finite_float(row["blur_sigma"], f"stage {position} sigma")
            start = int(row["start_zero_step"])
            end = int(row["end_zero_step_inclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError(f"Invalid stage trace row in {path}: {row}") from exc
        if stage != position or end < start:
            raise AggregationError(f"Nonsequential stage trace row in {path}: {row}")
        normalized = {
            "stage_index": stage,
            "blur_sigma": sigma,
            "start_zero_step": start,
            "end_zero_step_inclusive": end,
            "steps": end - start + 1,
            "trigger": row.get("trigger", ""),
        }
        rows.append(normalized)
        flat[f"stage_{stage}_blur_sigma"] = sigma
        flat[f"stage_{stage}_start_step"] = start
        flat[f"stage_{stage}_steps"] = end - start + 1
    flat["stage_start_steps"] = [row["start_zero_step"] for row in rows]
    flat["stage_sigmas"] = [row["blur_sigma"] for row in rows]
    return rows, flat


def _case_record(
    *, variant: str, case: int, seed: int, case_dir: Path, summary: Mapping[str, Any], expected_steps: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = _nested(summary, "landmark_metrics")
    record: dict[str, Any] = {
        "case": case,
        "variant": variant,
        "seed": seed,
        "case_dir": str(case_dir),
        "run_mode": summary.get("run_mode"),
        "stage_policy": summary.get("stage_policy"),
        "initial_snap_tre_mm": _finite_float(_nested(metrics, "initial", "snap", "mean_mm"), "initial snap TRE"),
        "initial_subvoxel_tre_mm": _finite_float(_nested(metrics, "initial", "subvoxel", "mean_mm"), "initial subvoxel TRE"),
        "direct_inr_snap_tre_mm": _finite_float(_nested(metrics, "direct_inr", "snap", "mean_mm"), "direct snap TRE"),
        "direct_inr_subvoxel_tre_mm": _finite_float(_nested(metrics, "direct_inr", "subvoxel", "mean_mm"), "direct subvoxel TRE"),
        "dense_snap_tre_mm": _finite_float(_nested(metrics, "dense_dvf", "snap", "mean_mm"), "dense snap TRE"),
        "dense_subvoxel_tre_mm": _finite_float(_nested(metrics, "dense_dvf", "subvoxel", "mean_mm"), "dense subvoxel TRE"),
        "direct_dense_snap_abs_difference_mm": _finite_float(
            _nested(metrics, "direct_dense_agreement", "snap_mean_tre_absolute_difference_mm"),
            "direct/dense snap agreement",
        ),
        "direct_dense_subvoxel_abs_difference_mm": _finite_float(
            _nested(metrics, "direct_dense_agreement", "subvoxel_mean_tre_absolute_difference_mm"),
            "direct/dense subvoxel agreement",
        ),
        "five_lobe_mean_dice": _finite_float(
            _nested(summary, "five_lobe_dice", "mean_dice"), "five-lobe mean Dice"
        ),
        "training_wall_seconds": _finite_float(
            _nested(summary, "runtime", "training_wall_seconds"), "training runtime"
        ),
        "total_wall_seconds": _finite_float(
            _nested(summary, "runtime", "total_wall_seconds_through_evaluation"), "total runtime"
        ),
        "cuda_training_allocated_bytes": _finite_float(
            _nested(summary, "cuda_peak_memory_bytes", "training_allocated"), "training allocated memory"
        ),
        "cuda_training_reserved_bytes": _finite_float(
            _nested(summary, "cuda_peak_memory_bytes", "training_reserved"), "training reserved memory"
        ),
        "cuda_total_allocated_bytes": _finite_float(
            _nested(summary, "cuda_peak_memory_bytes", "total_allocated"), "total allocated memory"
        ),
        "cuda_total_reserved_bytes": _finite_float(
            _nested(summary, "cuda_peak_memory_bytes", "total_reserved"), "total reserved memory"
        ),
    }
    record.update(_read_history(case_dir / "history.csv", expected_steps))
    stage_rows, stage_flat = _read_stage_trace(case_dir / "stage_trace.csv")
    record.update(stage_flat)
    record["stage_trace"] = stage_rows
    return record, stage_rows


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int]:
    clean = [_finite_float(value, "population value") for value in values]
    if not clean:
        raise AggregationError("Cannot summarize an empty population")
    q1 = _percentile(clean, 0.25)
    q3 = _percentile(clean, 0.75)
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


def load_posthoc_regularity_gate(
    *,
    run_root: Path,
    data_root: Path,
    posthoc_root: Path,
    mode: str,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Validate, but do not recompute, the separate common-regularity gate."""

    posthoc_root = posthoc_root.resolve()
    manifest_path = posthoc_root / "posthoc_author_regularity_manifest.json"
    portable_roots = (
        (run_root, "${RUN_ROOT}"),
        (data_root, "${DATA_ROOT}"),
        (posthoc_root, "${REVIEW_ROOT}"),
        (Path(__file__).resolve().parents[3], "${REPO_ROOT}"),
    )
    manifest = expand_portable_review_value(
        _load_json(manifest_path), portable_roots
    )
    expected_cases = list(CASES_BY_MODE[mode])
    if manifest.get("schema_version") != 1:
        raise AggregationError("Unsupported posthoc regularity manifest schema")
    if manifest.get("cases") != expected_cases:
        raise AggregationError(
            f"Posthoc regularity cases {manifest.get('cases')!r} do not match {expected_cases}"
        )
    try:
        if Path(str(manifest.get("run_root"))).resolve() != run_root.resolve():
            raise AggregationError("Posthoc regularity manifest points to a different run root")
        if Path(str(manifest.get("data_root"))).resolve() != data_root.resolve():
            raise AggregationError("Posthoc regularity manifest points to a different data root")
    except (OSError, TypeError, ValueError) as exc:
        raise AggregationError("Posthoc regularity manifest has an invalid run root") from exc
    if Path(str(manifest.get("review_root"))).resolve() != posthoc_root:
        raise AggregationError("Posthoc regularity manifest points to a different review root")

    posthoc_bytes = POSTHOC_SOURCE.read_bytes()
    try:
        posthoc_relative = POSTHOC_SOURCE.relative_to(Path(__file__).resolve().parents[3]).as_posix()
    except ValueError as exc:  # pragma: no cover - repository layout invariant
        raise AggregationError("Posthoc source is outside the repository root") from exc
    expected_posthoc_source = {
        "path": str(POSTHOC_SOURCE),
        "repository_relative_path": posthoc_relative,
        "size_bytes": len(posthoc_bytes),
        "sha256": file_sha256(POSTHOC_SOURCE),
    }
    if manifest.get("posthoc_source") != expected_posthoc_source:
        raise AggregationError("Posthoc regularity generator provenance is stale or mismatched")

    common_hash = file_sha256(COMMON_SOURCE)
    if manifest.get("common_source_sha256") != common_hash:
        raise AggregationError("Posthoc regularity common-source hash is stale or mismatched")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_cases):
        raise AggregationError("Posthoc regularity manifest does not contain one output per case")
    by_case: dict[int, dict[str, Any]] = {}
    for output in outputs:
        if not isinstance(output, Mapping):
            raise AggregationError("Malformed posthoc regularity output record")
        case = int(output.get("case", -1))
        resolved_outputs: dict[str, Path] = {}
        for kind in ("csv", "json", "markdown"):
            path = Path(str(output.get(kind))).resolve()
            try:
                path.relative_to(posthoc_root)
            except ValueError as exc:
                raise AggregationError(
                    f"Posthoc regularity {kind} must remain below --posthoc-root"
                ) from exc
            if not path.is_file() or path.stat().st_size == 0:
                raise AggregationError(
                    f"Posthoc regularity {kind} is missing or empty for case{case:02d}"
                )
            resolved_outputs[kind] = path
        json_path = resolved_outputs["json"]
        payload = expand_portable_review_value(
            _load_json(json_path), portable_roots
        )
        if case not in expected_cases or case in by_case or payload.get("case") != case:
            raise AggregationError(f"Invalid or duplicate posthoc regularity case: {case}")
        if payload.get("schema_version") != 1 or payload.get("seed") != seed:
            raise AggregationError(f"Posthoc regularity identity mismatch for case{case:02d}")
        expected_run_mode = "short_smoke" if mode == "smoke" else "full_length"
        if payload.get("run_mode") != expected_run_mode:
            raise AggregationError(f"Posthoc regularity run mode mismatch for case{case:02d}")
        if payload.get("metric_source", {}).get("sha256") != common_hash:
            raise AggregationError(f"Posthoc regularity metric-source hash mismatch for case{case:02d}")
        if payload.get("posthoc_source") != expected_posthoc_source:
            raise AggregationError(
                f"Posthoc regularity generator provenance mismatch for case{case:02d}"
            )

        for variant_key in ("a0", "a1"):
            metrics = payload.get(variant_key, {}).get("metrics")
            if not isinstance(metrics, Mapping):
                raise AggregationError(
                    f"Posthoc regularity lacks {variant_key} metrics for case{case:02d}"
                )
            for metric in REPORT_METRICS:
                _finite_float(metrics.get(metric), f"case{case:02d} {variant_key} {metric}")
        deltas = payload.get("a1_minus_a0")
        if not isinstance(deltas, Mapping):
            raise AggregationError(f"Posthoc regularity lacks paired deltas for case{case:02d}")
        for metric in REPORT_METRICS:
            observed = _finite_float(deltas.get(metric), f"case{case:02d} delta {metric}")
            expected = float(payload["a1"]["metrics"][metric]) - float(
                payload["a0"]["metrics"][metric]
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-10):
                raise AggregationError(
                    f"Posthoc regularity paired delta mismatch for case{case:02d} {metric}"
                )
        expected_inputs = {
            "a0_dense_dvf": run_root / "a0" / f"case_{case:02d}" / "dense_dvf_crop_voxel_xyz.npy",
            "a1_dense_dvf": run_root / "a1" / f"case_{case:02d}" / "dense_dvf_crop_voxel_xyz.npy",
            "fixed_segmentation": data_root / f"case_{case:02d}" / "segmentations_00.nii",
        }
        for key, expected in expected_inputs.items():
            value = payload.get("inputs", {}).get(key)
            if Path(str(value)).resolve() != expected.resolve():
                raise AggregationError(
                    f"Posthoc regularity {key} does not match accepted case{case:02d} field"
                )
        by_case[case] = payload

    if sorted(by_case) != expected_cases:
        raise AggregationError("Posthoc regularity cohort is incomplete")
    gate = {
        "status": "pass",
        "computed": True,
        "manifest": str(manifest_path),
        "cases": expected_cases,
        "common_source": str(COMMON_SOURCE),
        "common_source_sha256": common_hash,
        "posthoc_source": expected_posthoc_source,
        "convention": manifest.get("conversion"),
        "metric_scope": (
            "Shared finite-difference metrics on author crop voxel fields, with full-crop and "
            "fixed/T00 five-lobe-union scopes; displacement magnitude additionally reported in mm."
        ),
        "unit_caveat": (
            "Diffusion and bending are voxel-coordinate finite-difference metrics, not "
            "physical-coordinate energies; Jacobian metrics are dimensionless."
        ),
    }
    return by_case, gate


def _numeric_delta_fields(record: Mapping[str, Any]) -> list[str]:
    prefixes = ("final_", "mean_", "stage_")
    explicit = set(TRE_FIELDS) | set(RUNTIME_MEMORY_FIELDS) | {"five_lobe_mean_dice"}
    return sorted(
        key
        for key, value in record.items()
        if isinstance(value, (int, float))
        and key not in {"case", "seed", "history_rows", "stage_count"}
        and (key in explicit or key.startswith(prefixes) or key.startswith(REGULARITY_PREFIX))
    )


def collect_complete_cohort(
    *,
    run_root: Path,
    mode: str,
    seed: int = LOCKED_SEED,
    expected_adapter_bundle_sha256: str,
    posthoc_results: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in CASES_BY_MODE:
        raise AggregationError(f"Unknown mode {mode!r}")
    if seed != LOCKED_SEED:
        raise AggregationError(f"Locked cohort requires seed {LOCKED_SEED}")
    expected_steps = SMOKE_STEPS if mode == "smoke" else LOCKED_TOTAL_STEPS
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for case in CASES_BY_MODE[mode]:
        a0_dir = run_root / "a0" / f"case_{case:02d}"
        for variant in VARIANTS:
            case_dir = run_root / variant.lower() / f"case_{case:02d}"
            validation = validate_done_summary(
                case_dir,
                variant=variant,
                case=case,
                seed=seed,
                expected_steps=expected_steps,
                a0_case_dir=a0_dir if variant == "A1" else None,
                expected_adapter_bundle_sha256=expected_adapter_bundle_sha256,
            )
            if not validation.valid or validation.summary is None:
                errors.extend(
                    f"{variant} case{case:02d}: {error}" for error in validation.errors
                )
                continue
            try:
                record, stage_rows = _case_record(
                    variant=variant,
                    case=case,
                    seed=seed,
                    case_dir=case_dir,
                    summary=validation.summary,
                    expected_steps=expected_steps,
                )
                regularity = posthoc_results[case][variant.lower()]["metrics"]
                record.update(
                    {
                        f"{REGULARITY_PREFIX}{metric}": float(regularity[metric])
                        for metric in REPORT_METRICS
                    }
                )
                records.append(record)
                if not stage_rows:
                    errors.append(f"{variant} case{case:02d}: empty normalized stage trace")
            except AggregationError as exc:
                errors.append(f"{variant} case{case:02d}: {exc}")

    expected_records = 2 * len(CASES_BY_MODE[mode])
    if errors or len(records) != expected_records:
        detail = "\n- ".join(errors or [f"found {len(records)} of {expected_records} records"])
        raise AggregationError(f"Complete paired-cohort gate failed:\n- {detail}")

    by_key = {(row["variant"], int(row["case"])): row for row in records}
    deltas: list[dict[str, Any]] = []
    boundary_audit: list[dict[str, Any]] = []
    for case in CASES_BY_MODE[mode]:
        a0 = by_key[("A0", case)]
        a1 = by_key[("A1", case)]
        a0_starts = list(a0["stage_start_steps"])
        a1_starts = list(a1["stage_start_steps"])
        identical = a0_starts == a1_starts
        boundary_audit.append(
            {
                "case": case,
                "seed": seed,
                "a0_stage_start_steps": a0_starts,
                "a1_stage_start_steps": a1_starts,
                "identical": identical,
                "a0_policy": a0["stage_policy"],
                "a1_policy": a1["stage_policy"],
            }
        )
        if not identical:
            raise AggregationError(f"A1 case{case:02d} stage boundaries differ from A0")
        fields = sorted(set(_numeric_delta_fields(a0)) & set(_numeric_delta_fields(a1)))
        delta: dict[str, Any] = {"case": case, "seed": seed, "direction": "A1_minus_A0"}
        for field in fields:
            delta[f"delta_{field}"] = float(a1[field]) - float(a0[field])
        for metric in REPORT_METRICS:
            delta_key = f"delta_{REGULARITY_PREFIX}{metric}"
            expected_delta = float(posthoc_results[case]["a1_minus_a0"][metric])
            if not math.isclose(
                float(delta[delta_key]), expected_delta, rel_tol=0.0, abs_tol=1e-10
            ):
                raise AggregationError(
                    f"Aggregated regularity delta disagrees with posthoc gate for case{case:02d} {metric}"
                )
        deltas.append(delta)

    population: list[dict[str, Any]] = []
    population_metrics = sorted(
        set(TRE_FIELDS)
        | set(RUNTIME_MEMORY_FIELDS)
        | {"five_lobe_mean_dice"}
        | {f"final_{field}" for field in COMPONENT_FIELDS}
        | {f"mean_{field}" for field in COMPONENT_FIELDS}
        | {f"stage_{stage}_steps" for stage in range(3)}
        | {f"stage_{stage}_start_step" for stage in range(3)}
        | {f"{REGULARITY_PREFIX}{metric}" for metric in REPORT_METRICS}
    )
    for variant in VARIANTS:
        group = [row for row in records if row["variant"] == variant]
        for metric in population_metrics:
            if all(metric in row for row in group):
                population.append(
                    {"group": variant, "metric": metric, **describe(float(row[metric]) for row in group)}
                )
    for metric in population_metrics:
        delta_key = f"delta_{metric}"
        if all(delta_key in row for row in deltas):
            population.append(
                {
                    "group": "A1_minus_A0",
                    "metric": metric,
                    **describe(float(row[delta_key]) for row in deltas),
                }
            )
    return records, deltas, population, boundary_audit


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AggregationError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _population_lookup(population: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["group"]), str(row["metric"])): row for row in population}


def _mean_sd(row: Mapping[str, Any] | None, *, scale: float = 1.0) -> str:
    if row is None:
        return "NA"
    return f"{float(row['mean']) / scale:.4f} ± {float(row['sd_sample']) / scale:.4f}"


def render_markdown(
    *,
    mode: str,
    cases: Sequence[int],
    population: Sequence[Mapping[str, Any]],
    boundary_audit: Sequence[Mapping[str, Any]],
    regularity_gate: Mapping[str, Any],
) -> str:
    lookup = _population_lookup(population)
    lines = [
        "# DUAL-INR-DIR Author A0/A1 Aggregate",
        "",
        f"Cohort: `{mode}`; cases: `{','.join(f'{case:02d}' for case in cases)}`; seed: `{LOCKED_SEED}`.",
        "All requested A0/A1 records passed the DONE/artifact/history/provenance gates.",
        "",
        "## TRE",
        "",
        "| Group | Dense snap TRE (mm) | Dense subvoxel TRE (mm) | Direct INR snap TRE (mm) |",
        "|---|---:|---:|---:|",
    ]
    for group in ("A0", "A1", "A1_minus_A0"):
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    _mean_sd(lookup.get((group, "dense_snap_tre_mm"))),
                    _mean_sd(lookup.get((group, "dense_subvoxel_tre_mm"))),
                    _mean_sd(lookup.get((group, "direct_inr_snap_tre_mm"))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are across-case mean ± sample SD; paired deltas are `A1 - A0`.",
            "",
            "## Training components",
            "",
            "| Group | Final total | Final similarity | Final spatial | Final mask |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in ("A0", "A1", "A1_minus_A0"):
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    _mean_sd(lookup.get((group, "final_loss_total"))),
                    _mean_sd(lookup.get((group, "final_loss_similarity"))),
                    _mean_sd(lookup.get((group, "final_loss_spatial_weighted"))),
                    _mean_sd(lookup.get((group, "final_loss_mask_weighted"))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Runtime and peak memory",
            "",
            "| Group | Training seconds | Total seconds | Peak allocated GiB | Peak reserved GiB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    gib = float(1024**3)
    for group in ("A0", "A1", "A1_minus_A0"):
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    _mean_sd(lookup.get((group, "training_wall_seconds"))),
                    _mean_sd(lookup.get((group, "total_wall_seconds"))),
                    _mean_sd(lookup.get((group, "cuda_total_allocated_bytes")), scale=gib),
                    _mean_sd(lookup.get((group, "cuda_total_reserved_bytes")), scale=gib),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Schedule replay audit",
            "",
            "| Case | A0 boundaries | A1 boundaries | Exact match |",
            "|---:|---|---|---|",
        ]
    )
    for row in boundary_audit:
        lines.append(
            f"| {int(row['case']):02d} | `{row['a0_stage_start_steps']}` | "
            f"`{row['a1_stage_start_steps']}` | {'yes' if row['identical'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Common posthoc regularity",
            "",
            f"Status: **{regularity_gate['status']}**; computed by the separate validated posthoc gate.",
            "",
            "| Group | Diffusion FG | Bending FG | SDlogJ FG | J<=0 FG (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in ("A0", "A1", "A1_minus_A0"):
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    _mean_sd(lookup.get((group, "regularity_diffusion_l2_fg"))),
                    _mean_sd(lookup.get((group, "regularity_bending_energy_fg"))),
                    _mean_sd(lookup.get((group, "regularity_log_jdet_std_fg"))),
                    _mean_sd(lookup.get((group, "regularity_jdet_le0_pct_fg"))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- Scope: {regularity_gate['metric_scope']}",
            f"- Unit caveat: {regularity_gate['unit_caveat']}",
            f"- Source hash: `{regularity_gate['common_source_sha256']}`.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "This report is descriptive. It does not establish equivalence or a method-only comparison with IDIR/pTVReg.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_cohort(
    *,
    run_root: Path,
    posthoc_root: Path,
    review_root: Path,
    mode: str,
    seed: int = LOCKED_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    posthoc_root = posthoc_root.resolve()
    review_root = review_root.resolve()
    if not run_root.is_dir():
        raise AggregationError(f"Run root does not exist: {run_root}")
    manifest_path = run_root / f"launcher_manifest_{mode}.json"
    manifest = _load_json(manifest_path)
    if not manifest.get("dry_run_passed"):
        raise AggregationError(f"Launcher dry-run manifest did not pass: {manifest_path}")
    plan = manifest.get("plan")
    if not isinstance(plan, Mapping) or plan.get("mode") != mode:
        raise AggregationError("Launcher manifest mode/plan is inconsistent")
    expected_bundle = plan.get("adapter_source_bundle_sha256")
    if not isinstance(expected_bundle, str) or len(expected_bundle) != 64:
        raise AggregationError("Launcher manifest lacks an adapter source bundle hash")
    code_binding = plan.get("orchestration_code_sha256")
    if not isinstance(code_binding, Mapping):
        raise AggregationError("Launcher manifest lacks orchestration code hashes")
    if dict(code_binding) != code_sha256_binding():
        raise AggregationError(
            "Current aggregation/orchestration sources differ from the approved code snapshot"
        )
    try:
        verify_code_snapshot(run_root / "code_snapshot", code_binding)
    except Exception as exc:
        raise AggregationError(f"Code snapshot gate failed: {exc}") from exc
    data_root_value = plan.get("data_root")
    if not isinstance(data_root_value, str):
        raise AggregationError("Launcher manifest lacks the prepared data root")
    data_root = Path(data_root_value).resolve()
    posthoc_results, regularity_gate = load_posthoc_regularity_gate(
        run_root=run_root,
        data_root=data_root,
        posthoc_root=posthoc_root,
        mode=mode,
        seed=seed,
    )
    existing = [review_root / name for name in OUTPUT_FILENAMES if (review_root / name).exists()]
    if existing and not overwrite:
        raise AggregationError(
            "Review outputs already exist; pass --overwrite after inspection: "
            + ", ".join(str(path) for path in existing)
        )

    records, deltas, population, boundary_audit = collect_complete_cohort(
        run_root=run_root,
        mode=mode,
        seed=seed,
        expected_adapter_bundle_sha256=expected_bundle,
        posthoc_results=posthoc_results,
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "dual_inr_dirlab_author_repro_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "cases": list(CASES_BY_MODE[mode]),
        "seed": seed,
        "launcher_plan_sha256": manifest.get("plan_sha256"),
        "adapter_source_bundle_sha256": expected_bundle,
        "record_count": len(records),
        "paired_case_count": len(deltas),
        "complete_paired_cohort": True,
        "all_realized_boundaries_identical": all(row["identical"] for row in boundary_audit),
        "per_case": records,
        "paired_deltas_a1_minus_a0": deltas,
        "population": population,
        "boundary_audit": boundary_audit,
        "common_posthoc_regularity_gate": regularity_gate,
    }

    portable_roots = (
        (run_root, "${RUN_ROOT}"),
        (data_root, "${DATA_ROOT}"),
        (posthoc_root, "${POSTHOC_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
        (Path(__file__).resolve().parents[3], "${REPO_ROOT}"),
    )
    review_records = portable_review_value(records, portable_roots)
    review_deltas = portable_review_value(deltas, portable_roots)
    review_population = portable_review_value(population, portable_roots)
    review_boundary_audit = portable_review_value(boundary_audit, portable_roots)
    review_regularity_gate = portable_review_value(
        regularity_gate, portable_roots
    )
    review_payload = portable_review_value(payload, portable_roots)

    review_root.mkdir(parents=True, exist_ok=True)
    write_csv(review_root / "dual_author_per_case.csv", review_records)
    write_csv(review_root / "dual_author_paired_deltas.csv", review_deltas)
    write_csv(review_root / "dual_author_population.csv", review_population)
    write_csv(
        review_root / "dual_author_boundary_audit.csv", review_boundary_audit
    )
    atomic_write_json(review_root / "dual_author_aggregate.json", review_payload)
    markdown = render_markdown(
        mode=mode,
        cases=CASES_BY_MODE[mode],
        population=review_population,
        boundary_audit=review_boundary_audit,
        regularity_gate=review_regularity_gate,
    )
    (review_root / "dual_author_report.md").write_text(markdown, encoding="utf-8")
    return review_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--posthoc-root",
        type=Path,
        required=True,
        help="Completed output root from posthoc_author_regularity.py",
    )
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(CASES_BY_MODE), required=True)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = aggregate_cohort(
        run_root=args.run_root,
        posthoc_root=args.posthoc_root,
        review_root=args.review_root,
        mode=args.mode,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"Aggregated {payload['record_count']} records / "
        f"{payload['paired_case_count']} pairs into {args.review_root.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
