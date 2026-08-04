#!/usr/bin/env python3
"""Compute common posthoc regularity metrics for paired DUAL-INR-DIR runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.experiments.lungct_dual_inr_author.external_api import (  # type: ignore
        load_external_api,
    )
    from scripts.experiments.lungct_dual_inr_author.plot_author_review import (  # type: ignore
        ReviewError,
        load_paired_case,
        parse_cases,
    )
    from scripts.experiments.lungct_dual_inr_author.portable_paths import (  # type: ignore
        portable_review_value,
    )
    from scripts.experiments.lungct_no_sweep_repro.common import (  # type: ignore
        displacement_regularities,
    )
else:
    from .external_api import load_external_api
    from .plot_author_review import ReviewError, load_paired_case, parse_cases
    from .portable_paths import portable_review_value
    from ..lungct_no_sweep_repro.common import displacement_regularities


DEFAULT_CASES = tuple(range(1, 11))
POSTHOC_SOURCE = Path(__file__).resolve()
REPOSITORY_ROOT = POSTHOC_SOURCE.parents[3]
COMMON_SOURCE = (
    POSTHOC_SOURCE.parents[1] / "lungct_no_sweep_repro" / "common.py"
)
REPORT_METRICS = (
    "diffusion_l2_full",
    "diffusion_l2_fg",
    "bending_energy_full",
    "bending_energy_fg",
    "log_jdet_std_full",
    "log_jdet_std_fg",
    "jdet_le0_pct_full",
    "jdet_le0_pct_fg",
    "jdet_mean_full",
    "jdet_mean_fg",
    "disp_mag_mean_full_vox",
    "disp_mag_p95_full_vox",
    "disp_mag_max_full_vox",
    "disp_mag_mean_fg_vox",
    "disp_mag_p95_fg_vox",
    "disp_mag_max_fg_vox",
    "disp_mag_mean_full_mm",
    "disp_mag_p95_full_mm",
    "disp_mag_max_full_mm",
    "disp_mag_mean_fg_mm",
    "disp_mag_p95_fg_mm",
    "disp_mag_max_fg_mm",
)


class RegularityError(ReviewError):
    """Raised when paired posthoc regularity inputs fail their contract."""


def _source_record(path: Path) -> dict[str, Any]:
    """Capture the exact source bytes used to generate a posthoc report."""

    resolved = path.resolve()
    data = resolved.read_bytes()
    try:
        repository_relative_path = resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        repository_relative_path = None
    return {
        "path": str(resolved),
        "repository_relative_path": repository_relative_path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def xyz_field_to_common_zyx(disp_vox_xyz: np.ndarray) -> np.ndarray:
    """Convert ``(X,Y,Z, xyz)`` into the common ``(Z,Y,X, zyx)`` convention."""

    field = np.asarray(disp_vox_xyz)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise RegularityError(
            f"Expected displacement shape (X,Y,Z,3 xyz), got {field.shape}"
        )
    if min(field.shape[:3]) < 3:
        raise RegularityError(
            f"Every displacement dimension must be >=3, got {field.shape[:3]}"
        )
    if not np.isfinite(field).all():
        raise RegularityError("Displacement contains non-finite values")
    return np.transpose(field, (2, 1, 0, 3))[..., ::-1].astype(
        np.float32, copy=False
    )


def xyz_mask_to_common_zyx(mask_xyz: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_xyz)
    if mask.ndim != 3:
        raise RegularityError(f"Expected mask shape (X,Y,Z), got {mask.shape}")
    return np.transpose(mask.astype(bool, copy=False), (2, 1, 0))


def _physical_magnitude_metrics(
    disp_vox_zyx: np.ndarray,
    spacing_zyx_mm: Sequence[float],
    foreground_mask_zyx: np.ndarray,
) -> dict[str, float]:
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise RegularityError(f"Invalid spacing_zyx_mm: {spacing_zyx_mm}")
    magnitude = np.linalg.norm(
        disp_vox_zyx.astype(np.float64) * spacing.reshape((1, 1, 1, 3)), axis=-1
    )
    foreground = magnitude[foreground_mask_zyx]
    if foreground.size == 0:
        raise RegularityError("Fixed lobe-union foreground mask is empty")

    def summarize(values: np.ndarray, scope: str) -> dict[str, float]:
        return {
            f"disp_mag_mean_{scope}_mm": float(np.mean(values)),
            f"disp_mag_p95_{scope}_mm": float(np.percentile(values, 95)),
            f"disp_mag_max_{scope}_mm": float(np.max(values)),
        }

    return {**summarize(magnitude, "full"), **summarize(foreground, "fg")}


def compute_common_regularity(
    *,
    disp_vox_xyz: np.ndarray,
    fixed_lobe_union_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
) -> dict[str, Any]:
    """Convert conventions, call the shared implementation, and add mm magnitude."""

    field_xyz = np.asarray(disp_vox_xyz)
    mask_xyz = np.asarray(fixed_lobe_union_xyz)
    if field_xyz.shape[:3] != mask_xyz.shape:
        raise RegularityError(
            f"DVF/mask shape mismatch: {field_xyz.shape[:3]} versus {mask_xyz.shape}"
        )
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if spacing_xyz.shape != (3,):
        raise RegularityError(f"Expected spacing_xyz_mm length 3, got {spacing_xyz_mm}")

    field_zyx = xyz_field_to_common_zyx(field_xyz)
    mask_zyx = xyz_mask_to_common_zyx(mask_xyz)
    spacing_zyx = spacing_xyz[::-1]
    metrics = displacement_regularities(
        field_zyx,
        foreground_mask=mask_zyx,
        log_base="e",
    )
    metrics.update(
        _physical_magnitude_metrics(field_zyx, spacing_zyx, mask_zyx)
    )
    for metric in REPORT_METRICS:
        value = float(metrics.get(metric, float("nan")))
        if not math.isfinite(value):
            raise RegularityError(f"Common regularity metric {metric} is non-finite")
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "conversion": {
            "source_shape_and_convention": "(X,Y,Z,xyz) author crop voxel displacement",
            "target_shape_and_convention": "(Z,Y,X,zyx) common voxel displacement",
            "spatial_permutation": [2, 1, 0],
            "vector_component_permutation": [2, 1, 0],
            "source_shape_xyz": list(field_xyz.shape[:3]),
            "target_shape_zyx": list(field_zyx.shape[:3]),
            "spacing_xyz_mm": [float(value) for value in spacing_xyz],
            "spacing_zyx_mm": [float(value) for value in spacing_zyx],
            "fixed_lobe_union_voxels": int(mask_zyx.sum()),
        },
    }


def _bbox_payload(bbox: Sequence[slice] | None) -> list[dict[str, int | None]] | None:
    if bbox is None:
        return None
    return [
        {"start": item.start, "stop": item.stop, "step": item.step} for item in bbox
    ]


def _normalize_bbox_payload(value: Any) -> list[dict[str, int | None]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise RegularityError(f"Malformed summary crop_bbox: {value!r}")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise RegularityError(f"Malformed summary crop_bbox item: {item!r}")
        normalized.append(
            {
                "start": None if item.get("start") is None else int(item["start"]),
                "stop": None if item.get("stop") is None else int(item["stop"]),
                "step": None if item.get("step") is None else int(item["step"]),
            }
        )
    return normalized


def validate_author_geometry(
    *, pair: Mapping[str, Any], data: Mapping[str, Any], case: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    fixed_image = np.asarray(data["images"]["fixed"])
    fixed_lobes = np.asarray(data["lung_classes_masks"]["fixed"])
    if fixed_image.ndim != 3 or fixed_lobes.shape != (5,) + fixed_image.shape:
        raise RegularityError(
            f"Case {case:02d} author crop/lobes have invalid shapes: "
            f"{fixed_image.shape}, {fixed_lobes.shape}"
        )
    if any(int(mask.sum()) <= 0 for mask in fixed_lobes):
        raise RegularityError(f"Case {case:02d} has an empty fixed lobe channel")
    fixed_union = np.any(fixed_lobes > 0.5, axis=0)
    spacing_xyz = tuple(float(value) for value in data["image_spacing"])
    actual_bbox = _bbox_payload(data.get("bbox"))
    original_shape = list(data["original_shape"])

    for variant in ("a0", "a1"):
        record = pair[variant]
        summary = record["summary"]
        if int(summary.get("fixed_phase", -1)) != 0 or int(
            summary.get("moving_phase", -1)
        ) != 5:
            raise RegularityError(f"Case {case:02d} {variant} phase contract mismatch")
        if list(summary.get("image_shape_crop", [])) != list(fixed_image.shape):
            raise RegularityError(f"Case {case:02d} {variant} crop shape mismatch")
        if list(summary.get("image_shape_original", [])) != original_shape:
            raise RegularityError(f"Case {case:02d} {variant} original shape mismatch")
        summary_spacing = tuple(float(value) for value in summary.get("image_spacing_xyz", ()))
        if len(summary_spacing) != 3 or not np.allclose(
            summary_spacing, spacing_xyz, rtol=0.0, atol=1e-8
        ):
            raise RegularityError(f"Case {case:02d} {variant} spacing mismatch")
        if _normalize_bbox_payload(summary.get("crop_bbox")) != actual_bbox:
            raise RegularityError(f"Case {case:02d} {variant} crop bbox mismatch")
        if record["dense_dvf"].shape[:3] != fixed_image.shape:
            raise RegularityError(f"Case {case:02d} {variant} DVF crop shape mismatch")
    return fixed_union, spacing_xyz


def compute_paired_case(
    *,
    api: Any,
    pair: Mapping[str, Any],
    data_root: Path,
    posthoc_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case = int(pair["case"])
    data = api.utils.load_and_crop_full_dirlab(
        data_root / f"case_{case:02d}",
        fixed_phase=0,
        moving_phase=5,
        crop=True,
    )
    fixed_union, spacing_xyz = validate_author_geometry(
        pair=pair, data=data, case=case
    )
    results = {
        variant: compute_common_regularity(
            disp_vox_xyz=pair[variant]["dense_dvf"],
            fixed_lobe_union_xyz=fixed_union,
            spacing_xyz_mm=spacing_xyz,
        )
        for variant in ("a0", "a1")
    }
    a0_metrics = results["a0"]["metrics"]
    a1_metrics = results["a1"]["metrics"]
    delta = {key: float(a1_metrics[key] - a0_metrics[key]) for key in a0_metrics}
    return {
        "schema_version": 1,
        "case": case,
        "seed": int(pair["a0"]["seed"]),
        "run_mode": pair["a0"]["run_mode"],
        "a0": results["a0"],
        "a1": results["a1"],
        "a1_minus_a0": delta,
        "posthoc_source": dict(posthoc_source or _source_record(POSTHOC_SOURCE)),
        "metric_source": {
            "path": str(COMMON_SOURCE),
            "functions": [
                "displacement_regularities",
                "diffusion_loss_vox",
                "bending_energy_vox",
                "jacobian_determinant_zyx",
            ],
            "sha256": hashlib.sha256(COMMON_SOURCE.read_bytes()).hexdigest(),
        },
        "interpretation": {
            "diffusion_and_bending": (
                "Finite-difference metrics in native crop voxel coordinates; they "
                "depend on voxel/component scaling and are not physical-coordinate "
                "energies."
            ),
            "jacobian_determinant": (
                "Dimensionless and invariant when converting voxel to physical "
                "coordinates because F_phys = S F_vox S^-1 is a similarity "
                "transform, so det(F_phys) = det(F_vox)."
            ),
            "sdlogj": "Natural-log SD over positive determinants only.",
            "jdet_border": (
                "The shared implementation excludes a two-voxel border on axes "
                "longer than four voxels."
            ),
            "foreground": "Exact fixed/T00 five-lobe union in the author crop.",
            "displacement": (
                "Shared voxel-coordinate magnitude plus an added anisotropic-spacing "
                "physical-mm magnitude diagnostic."
            ),
        },
        "inputs": {
            "a0_dense_dvf": str(
                pair["a0"]["case_dir"] / "dense_dvf_crop_voxel_xyz.npy"
            ),
            "a1_dense_dvf": str(
                pair["a1"]["case_dir"] / "dense_dvf_crop_voxel_xyz.npy"
            ),
            "fixed_segmentation": str(
                data_root / f"case_{case:02d}" / "segmentations_00.nii"
            ),
            "crop_bbox": _bbox_payload(data.get("bbox")),
            "original_shape_xyz": [int(value) for value in data["original_shape"]],
        },
    }


def _metric_unit(metric: str) -> str:
    if metric.startswith(("diffusion", "bending")):
        return "voxel-coordinate finite difference"
    if metric.startswith("log_jdet") or metric.startswith("jdet_mean"):
        return "dimensionless"
    if metric.startswith("jdet_le0_pct"):
        return "%"
    if metric.endswith("_mm"):
        return "mm"
    if metric.endswith("_vox"):
        return "voxel-coordinate magnitude"
    return "dimensionless"


def _rows_for_case(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, metrics in (
        ("A0", result["a0"]["metrics"]),
        ("A1", result["a1"]["metrics"]),
        ("A1_minus_A0", result["a1_minus_a0"]),
    ):
        row = {
            "case": int(result["case"]),
            "seed": int(result["seed"]),
            "variant": variant,
        }
        row.update({metric: float(metrics[metric]) for metric in REPORT_METRICS})
        rows.append(row)
    return rows


def _markdown_for_case(result: Mapping[str, Any]) -> str:
    lines = [
        f"# DUAL-INR-DIR common regularity: case {int(result['case']):02d}",
        "",
        f"Seed: `{int(result['seed'])}`. Paired delta is `A1 - A0`.",
        "",
        "| Metric | Unit/convention | A0 | A1 | A1 - A0 |",
        "|---|---|---:|---:|---:|",
    ]
    for metric in REPORT_METRICS:
        lines.append(
            f"| `{metric}` | {_metric_unit(metric)} | "
            f"{result['a0']['metrics'][metric]:.8g} | "
            f"{result['a1']['metrics'][metric]:.8g} | "
            f"{result['a1_minus_a0'][metric]:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Convention notes",
            "",
            f"- {result['interpretation']['diffusion_and_bending']}",
            f"- {result['interpretation']['jacobian_determinant']}",
            f"- {result['interpretation']['sdlogj']}",
            f"- {result['interpretation']['jdet_border']}",
            f"- Foreground: {result['interpretation']['foreground']}",
            "",
            "## Posthoc source provenance",
            "",
            f"- Path: `{result['posthoc_source']['path']}`",
            f"- Size: `{int(result['posthoc_source']['size_bytes'])}` bytes",
            f"- SHA-256: `{result['posthoc_source']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def generate_posthoc_reports(
    *,
    run_root: Path,
    data_root: Path,
    review_root: Path,
    cases: Sequence[int] = DEFAULT_CASES,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    data_root = data_root.resolve()
    review_root = review_root.resolve()
    if not run_root.is_dir() or not data_root.is_dir():
        raise RegularityError("run_root and data_root must both exist directories")
    if not cases:
        raise RegularityError("At least one case is required")

    posthoc_source = _source_record(POSTHOC_SOURCE)
    api = load_external_api()
    results = []
    for case in cases:
        pair = load_paired_case(
            run_root=run_root,
            data_root=data_root,
            case=int(case),
            allow_smoke=allow_smoke,
        )
        results.append(
            compute_paired_case(
                api=api,
                pair=pair,
                data_root=data_root,
                posthoc_source=posthoc_source,
            )
        )

    review_root.mkdir(parents=True, exist_ok=True)
    targets = []
    for result in results:
        stem = f"case_{int(result['case']):02d}_paired_regularity"
        targets.extend(
            [
                review_root / f"{stem}.csv",
                review_root / f"{stem}.json",
                review_root / f"{stem}.md",
            ]
        )
    manifest_path = review_root / "posthoc_author_regularity_manifest.json"
    existing = [str(path) for path in [*targets, manifest_path] if path.exists()]
    if existing:
        raise RegularityError(f"Refusing to overwrite existing reports: {existing}")

    outputs = []
    for result in results:
        stem = f"case_{int(result['case']):02d}_paired_regularity"
        csv_path = review_root / f"{stem}.csv"
        json_path = review_root / f"{stem}.json"
        markdown_path = review_root / f"{stem}.md"
        portable_roots = (
            (run_root, "${RUN_ROOT}"),
            (data_root, "${DATA_ROOT}"),
            (review_root, "${REVIEW_ROOT}"),
            (REPOSITORY_ROOT, "${REPO_ROOT}"),
        )
        review_result = portable_review_value(result, portable_roots)
        _atomic_csv(csv_path, _rows_for_case(review_result))
        _atomic_json(json_path, review_result)
        _atomic_text(markdown_path, _markdown_for_case(review_result))
        outputs.append(
            {
                "case": int(result["case"]),
                "csv": portable_review_value(str(csv_path), portable_roots),
                "json": portable_review_value(str(json_path), portable_roots),
                "markdown": portable_review_value(
                    str(markdown_path), portable_roots
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "data_root": str(data_root),
        "review_root": str(review_root),
        "cases": [int(case) for case in cases],
        "posthoc_source": posthoc_source,
        "common_source": str(COMMON_SOURCE),
        "common_source_sha256": hashlib.sha256(COMMON_SOURCE.read_bytes()).hexdigest(),
        "conversion": "(X,Y,Z,xyz) -> transpose spatial (2,1,0), reverse vectors -> (Z,Y,X,zyx)",
        "outputs": outputs,
    }
    review_manifest = portable_review_value(manifest, portable_roots)
    _atomic_json(manifest_path, review_manifest)
    return review_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=list(DEFAULT_CASES))
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Explicitly permit completed short-smoke records; full runs are required by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_posthoc_reports(
        run_root=args.run_root,
        data_root=args.data_root,
        review_root=args.review_root,
        cases=args.cases,
        allow_smoke=args.allow_smoke,
    )
    print(
        f"Wrote paired common-regularity reports for {len(manifest['outputs'])} "
        f"case(s) under {manifest['review_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
