#!/usr/bin/env python
"""Prepare a versioned T00/T50 DIR-Lab tree for unmodified DUAL-INR-DIR.

The external DUAL-INR-DIR converter expects the original ZIP archives and
converts all ten breathing phases.  This adapter consumes this repository's
canonical extracted 4DCT layout, writes only the T00/T50 phases required by
DUAL, and intentionally reproduces the author converter's intensity,
orientation, landmark, TotalSegmentator, and union-lung-crop conventions.

The output is always ``OUTPUT_PARENT/VERSION/case_XX`` and must live outside
the Git repository.  ``prepare --dry-run`` validates sources, dependencies,
and the prospective target without writing anything.  ``validate`` performs a
read-only structural, geometry, crop, landmark, and checksum audit of an
existing artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.common.lungct import (  # noqa: E402
    FourDCTCaseFiles,
    load_landmarks_xyz,
    resolve_4dct_case,
)
from scripts.data.common.volume import parse_case_ids  # noqa: E402


SCHEMA_VERSION = 1
ADAPTER_ID = "dual_author_4dct_t00t50"
AUTHOR_FIXED_PHASE = 0
AUTHOR_MOVING_PHASE = 5
AUTHOR_LUNG_CLASSES: Tuple[int, ...] = (10, 11, 12, 13, 14)
TOTALSEGMENTATOR_TOTAL_TASK_IDS: Tuple[int, ...] = (291, 292, 293, 294, 295)
AUTHOR_LUNG_CLASS_NAMES: Mapping[int, str] = {
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
}
AUTHOR_CROP_PADDING = 2
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Parsed from external/DUAL-INR-DIR/scripts/convert_dirlab_4dct.py.  These
# values are metadata-only, but retaining them makes the minimal tree match the
# author converter rather than silently dropping fields.
OBSERVER_TRE_MM: Mapping[int, Tuple[float, float]] = {
    1: (0.85, 1.24),
    2: (0.70, 0.99),
    3: (0.77, 1.01),
    4: (1.13, 1.27),
    5: (0.92, 1.16),
    6: (0.97, 1.38),
    7: (0.81, 1.32),
    8: (1.03, 2.19),
    9: (0.75, 1.09),
    10: (0.86, 1.45),
}

# Recomputed directly from the canonical extracted release using the upstream
# converter's T00/T50 landmark pairing and native spacing.  These values are
# the adapter's identity-baseline source of truth.  They differ from the
# rounded "no reg." row in the DUAL paper, retained separately below.
CANONICAL_RELEASE_IDENTITY_TRE_MEAN_MM: Mapping[int, float] = {
    1: 3.8924062198159763,
    2: 4.33784245401475,
    3: 6.942992820787408,
    4: 9.830147256419307,
    5: 7.476853258560301,
    6: 10.89102421124554,
    7: 11.026219932523533,
    8: 14.994680761939078,
    9: 7.918263838803335,
    10: 7.301381916369588,
}

PUBLISHED_NO_REG_TRE_MM: Mapping[int, Tuple[float, float]] = {
    1: (4.01, 2.91),
    2: (4.65, 4.09),
    3: (6.73, 4.21),
    4: (9.42, 4.81),
    5: (7.10, 5.15),
    6: (11.10, 6.98),
    7: (11.59, 7.87),
    8: (15.16, 9.11),
    9: (7.82, 3.99),
    10: (7.63, 6.54),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> Dict[str, Any]:
    display_path = (
        path.relative_to(relative_to).as_posix() if relative_to else str(path)
    )
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def validate_version(version: str) -> str:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            "VERSION must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'."
        )
    return version


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_location(output_parent: Path, version: str) -> Path:
    target = (output_parent / validate_version(version)).resolve()
    if is_relative_to(target, REPO_ROOT.resolve()):
        raise ValueError(f"Output artifact must live outside Git repository: {target}")
    return target


def author_image_from_raw(path: Path, shape_zyx: Sequence[int]) -> np.ndarray:
    """Apply the DUAL author converter's raw-image transform exactly.

    DIR-Lab is read as native int16, shifted by -1024 HU, clipped, reshaped
    into zyx, and flipped on the z array axis before SimpleITK serialization.
    """
    raw = np.fromfile(path, dtype=np.int16)
    expected = int(np.prod(shape_zyx))
    if raw.size != expected:
        raise ValueError(
            f"{path} has {raw.size} voxels; expected {expected} for {tuple(shape_zyx)}"
        )
    shifted = raw - 1024
    shifted = np.clip(shifted, -1024, 3071)
    return np.ascontiguousarray(np.flip(shifted.reshape(tuple(shape_zyx)), axis=0))


def author_landmarks_xyz(
    landmarks_xyz: np.ndarray, shape_xyz: Sequence[int]
) -> np.ndarray:
    """Reproduce DUAL's landmark transform, including retained x/y indexing.

    The author converter flips z with ``depth - z``.  It does *not* subtract
    one from x or y; this adapter preserves that behavior for reproduction and
    records the convention explicitly in the artifact contract.
    """
    points = np.asarray(landmarks_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 xyz landmarks, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks contain non-finite values.")
    transformed = points.copy()
    transformed[:, 2] = float(shape_xyz[2]) - transformed[:, 2]
    return transformed


def write_landmarks(path: Path, points_xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points_xyz, delimiter=",", fmt="%.3f")


def bounding_box_from_mask(
    mask: np.ndarray, padding: int = AUTHOR_CROP_PADDING
) -> Tuple[slice, ...]:
    """NumPy equivalent of DUAL ``get_bounding_box`` (half-open slices)."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask_bool.shape}")
    if not np.any(mask_bool):
        raise ValueError("Cannot compute DUAL crop from an empty lung union mask.")
    slices: List[slice] = []
    for axis in range(mask_bool.ndim):
        collapse_axes = tuple(i for i in range(mask_bool.ndim) if i != axis)
        occupied = np.flatnonzero(np.any(mask_bool, axis=collapse_axes))
        start = max(int(occupied[0]) - padding, 0)
        stop = min(int(occupied[-1]) + padding + 1, mask_bool.shape[axis])
        slices.append(slice(start, stop))
    return tuple(slices)


def dual_union_bbox_xyz(
    fixed_segmentation_zyx: np.ndarray,
    moving_segmentation_zyx: np.ndarray,
    *,
    lung_classes: Sequence[int] = AUTHOR_LUNG_CLASSES,
    padding: int = AUTHOR_CROP_PADDING,
) -> Tuple[slice, ...]:
    """Compute DUAL's crop after its SITK zyx -> NumPy xyz axis swap."""
    fixed = np.swapaxes(np.asarray(fixed_segmentation_zyx), 0, 2)
    moving = np.swapaxes(np.asarray(moving_segmentation_zyx), 0, 2)
    if fixed.shape != moving.shape:
        raise ValueError(
            f"Fixed/moving segmentation shape mismatch: {fixed.shape} vs {moving.shape}"
        )
    union = np.logical_or(np.isin(fixed, lung_classes), np.isin(moving, lung_classes))
    return bounding_box_from_mask(union, padding=padding)


def bbox_payload(bbox: Sequence[slice]) -> Dict[str, List[int]]:
    starts = [int(item.start) for item in bbox]
    stops = [int(item.stop) for item in bbox]
    return {
        "start_xyz": starts,
        "stop_xyz_exclusive": stops,
        "shape_xyz": [stop - start for start, stop in zip(starts, stops)],
    }


def points_inside_shape(
    points_xyz: np.ndarray, shape_xyz: Sequence[int]
) -> Dict[str, Any]:
    points = np.asarray(points_xyz, dtype=np.float64)
    shape = np.asarray(shape_xyz, dtype=np.float64)
    inside = np.all((points >= 0.0) & (points <= shape[None, :] - 1.0), axis=1)
    return {
        "inside_count": int(inside.sum()),
        "total_count": int(inside.size),
        "inside_fraction": float(inside.mean()) if inside.size else float("nan"),
        "all_inside": bool(np.all(inside)),
        "min_xyz": [float(v) for v in points.min(axis=0)],
        "max_xyz": [float(v) for v in points.max(axis=0)],
        "shape_xyz": [int(v) for v in shape_xyz],
    }


def crop_landmarks(points_xyz: np.ndarray, bbox: Sequence[slice]) -> np.ndarray:
    starts = np.asarray([int(item.start) for item in bbox], dtype=np.float64)
    return np.asarray(points_xyz, dtype=np.float64) - starts[None, :]


def identity_tre_mm(
    fixed_landmarks_xyz: np.ndarray,
    moving_landmarks_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
) -> Dict[str, Any]:
    fixed = np.asarray(fixed_landmarks_xyz, dtype=np.float64)
    moving = np.asarray(moving_landmarks_xyz, dtype=np.float64)
    if fixed.shape != moving.shape or fixed.ndim != 2 or fixed.shape[1] != 3:
        raise ValueError(
            f"Identity TRE requires matching Nx3 arrays, got {fixed.shape} and {moving.shape}"
        )
    distances = np.linalg.norm(
        (fixed - moving) * np.asarray(spacing_xyz_mm, dtype=np.float64)[None, :],
        axis=1,
    )
    return {
        "count": int(distances.size),
        "mean_mm": float(distances.mean()),
        "std_mm_population": float(distances.std()),
        "min_mm": float(distances.min()),
        "max_mm": float(distances.max()),
    }


def source_case_paths(case: FourDCTCaseFiles) -> Dict[str, Path]:
    values = {
        "fixed_image_t00": case.image_t00,
        "moving_image_t50": case.image_t50,
        "fixed_landmarks_t00": case.landmarks_t00,
        "moving_landmarks_t50": case.landmarks_t50,
    }
    if any(value is None for value in values.values()):
        raise ValueError(
            f"Case {case.case_id} has incomplete canonical source paths: {case.notes}"
        )
    return {key: Path(value) for key, value in values.items() if value is not None}


def validate_source_case(
    case_id: int, data_root: Path
) -> Tuple[FourDCTCaseFiles, Dict[str, Any]]:
    case = resolve_4dct_case(case_id, data_root)
    if case.status != "complete":
        raise ValueError(
            f"Case {case_id:02d} source status is {case.status}: {case.notes}"
        )
    paths = source_case_paths(case)
    expected_image_bytes = int(np.prod(case.shape_zyx)) * np.dtype(np.int16).itemsize
    image_sizes: Dict[str, int] = {}
    for key in ("fixed_image_t00", "moving_image_t50"):
        size = paths[key].stat().st_size
        if size != expected_image_bytes:
            raise ValueError(
                f"{paths[key]} has {size} bytes; expected {expected_image_bytes}"
            )
        image_sizes[key] = size

    landmark_shapes: Dict[str, List[int]] = {}
    shape_xyz = tuple(int(v) for v in case.shape_zyx[::-1])
    for key in ("fixed_landmarks_t00", "moving_landmarks_t50"):
        points = load_landmarks_xyz(paths[key])
        if points.shape != (300, 3):
            raise ValueError(
                f"{paths[key]} has landmark shape {points.shape}; expected (300, 3)"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{paths[key]} contains non-finite landmarks")
        # Canonical DIR-Lab text coordinates are treated as 1-based for this
        # source sanity check.  The author-specific transform is applied later.
        if not np.all((points >= 1.0) & (points <= np.asarray(shape_xyz)[None, :])):
            raise ValueError(
                f"{paths[key]} has landmarks outside canonical 1-based xyz bounds"
            )
        landmark_shapes[key] = [int(v) for v in points.shape]

    return case, {
        "case_id": case_id,
        "status": "pass",
        "shape_zyx": [int(v) for v in case.shape_zyx],
        "spacing_zyx_mm": [float(v) for v in case.spacing_zyx],
        "expected_image_bytes": expected_image_bytes,
        "image_sizes": image_sizes,
        "landmark_shapes": landmark_shapes,
        "source_paths": {key: str(value) for key, value in paths.items()},
    }


def require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required in the intended Docker environment."
        ) from exc
    return sitk


def image_geometry(image: Any) -> Dict[str, Any]:
    return {
        "size_xyz": [int(v) for v in image.GetSize()],
        "spacing_xyz_mm": [float(v) for v in image.GetSpacing()],
        "origin_xyz_mm": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
    }


def geometry_matches(a: Any, b: Any, atol: float = 1e-6) -> bool:
    return bool(
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0.0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0.0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0.0)
    )


def write_author_image(
    path: Path, image_zyx: np.ndarray, spacing_zyx: Sequence[float]
) -> None:
    sitk = require_sitk()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ascontiguousarray(image_zyx))
    image.SetSpacing(tuple(float(v) for v in spacing_zyx[::-1]))
    sitk.WriteImage(image, str(path))


def totalsegmentator_model_provenance(
    weights_root: Path,
    *,
    require_complete: bool,
    hash_files: bool,
    task_ids: Sequence[int] = TOTALSEGMENTATOR_TOTAL_TASK_IDS,
) -> Dict[str, Any]:
    """Resolve the exact nnU-Net configs/checkpoints used by task=total."""
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for task_id in task_ids:
        dataset_dirs = sorted(
            path for path in weights_root.glob(f"Dataset{task_id}_*") if path.is_dir()
        )
        if len(dataset_dirs) != 1:
            missing.append(
                f"task {task_id}: expected one Dataset{task_id}_* directory, found {len(dataset_dirs)}"
            )
            continue
        dataset_dir = dataset_dirs[0]
        trainer_dir = dataset_dir / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
        required = {
            "dataset_config": trainer_dir / "dataset.json",
            "plans_config": trainer_dir / "plans.json",
            "checkpoint_fold_0": trainer_dir / "fold_0" / "checkpoint_final.pth",
        }
        absent = [str(path) for path in required.values() if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        files: Dict[str, Dict[str, Any]] = {}
        for role, path in required.items():
            record: Dict[str, Any] = {
                "path_relative_to_weights_root": path.relative_to(
                    weights_root
                ).as_posix(),
                "bytes": path.stat().st_size,
            }
            if hash_files:
                record["sha256"] = sha256_file(path)
            files[role] = record
        rows.append(
            {
                "task_id": int(task_id),
                "dataset_directory": dataset_dir.name,
                "trainer_configuration": trainer_dir.name,
                "files": files,
            }
        )
    if require_complete and missing:
        raise RuntimeError(
            "TotalSegmentator task=total model cache is incomplete: "
            + "; ".join(missing)
        )
    return {
        "status": "complete" if not missing else "not_yet_complete",
        "weights_root_container": str(weights_root),
        "required_task_ids": [int(v) for v in task_ids],
        "resolved_tasks": rows,
        "missing": missing,
        "hashes_included": hash_files,
    }


def validate_totalsegmentator_installation(
    *,
    require_weights: bool = False,
    hash_weights: bool = False,
) -> Dict[str, Any]:
    if importlib.util.find_spec("totalsegmentator") is None:
        raise RuntimeError(
            "TotalSegmentator is not installed in the intended Docker environment."
        )
    try:
        from totalsegmentator.map_to_binary import class_map
    except ImportError as exc:
        raise RuntimeError("Could not import TotalSegmentator's class map.") from exc
    total_map = class_map["total"]
    observed = {label: total_map.get(label) for label in AUTHOR_LUNG_CLASSES}
    if observed != dict(AUTHOR_LUNG_CLASS_NAMES):
        raise RuntimeError(
            f"TotalSegmentator lung label map is incompatible with DUAL: {observed}; "
            f"expected {dict(AUTHOR_LUNG_CLASS_NAMES)}"
        )
    try:
        version = importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        from totalsegmentator.config import get_weights_dir
    except ImportError as exc:
        raise RuntimeError(
            "Could not resolve TotalSegmentator's weights cache."
        ) from exc
    weights = totalsegmentator_model_provenance(
        Path(get_weights_dir()),
        require_complete=require_weights,
        hash_files=hash_weights,
    )
    return {
        "package": "TotalSegmentator",
        "package_version": version,
        "python_api_configuration": {
            "task": "total",
            "task_ids": list(TOTALSEGMENTATOR_TOTAL_TASK_IDS),
            "ml": True,
            "fast": False,
            "fastest": False,
            "internal_resample_spacing_mm": 1.5,
            "trainer": "nnUNetTrainerNoMirroring",
            "model": "3d_fullres",
            "folds": [0],
        },
        "lung_class_map": observed,
        "model_cache": weights,
    }


def run_totalsegmentator(
    image_path: Path, segmentation_path: Path, device: str
) -> None:
    from totalsegmentator.python_api import totalsegmentator

    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"TotalSegmentator: {image_path.parent.parent.name}/{image_path.name} on {device}",
        file=sys.stderr,
        flush=True,
    )
    result = totalsegmentator(
        input=image_path,
        output=segmentation_path,
        ml=True,
        task="total",
        quiet=True,
        device=device,
    )
    if not segmentation_path.exists():
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError(
                "TotalSegmentator returned an image without writing the target, and nibabel is unavailable."
            ) from exc
        nib.save(result, str(segmentation_path))


def find_reusable_segmentation(root: Path, case_id: int, phase: int) -> Path:
    base = root / f"case_{case_id:02d}" / f"segmentations_{phase:02d}"
    for suffix in (".nii", ".nii.gz"):
        candidate = Path(str(base) + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No reusable segmentation found for case {case_id:02d}, phase {phase:02d} under {root}"
    )


def copy_segmentation_as_nii(source: Path, target: Path) -> None:
    sitk = require_sitk()
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.ReadImage(str(source)), str(target))


def case_required_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "fixed_image_t00": case_dir / "images" / "phase_00.nii",
        "moving_image_t50": case_dir / "images" / "phase_05.nii",
        "fixed_landmarks_t00": case_dir / "landmarks_raw" / "extreme_landmarks_0.csv",
        "moving_landmarks_t50": case_dir / "landmarks_raw" / "extreme_landmarks_5.csv",
        "fixed_segmentation_t00": case_dir / "segmentations_00.nii",
        "moving_segmentation_t50": case_dir / "segmentations_05.nii",
        "metadata": case_dir / "metadata.yaml",
    }


def segmentation_label_stats(array: np.ndarray) -> Dict[str, Any]:
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=0.0, rtol=0.0):
        raise ValueError("Segmentation contains non-integer label values.")
    labels, counts = np.unique(rounded.astype(np.int64, copy=False), return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(labels, counts)}
    return {
        "unique_labels": sorted(count_map),
        "lung_label_voxels": {
            str(label): count_map.get(label, 0) for label in AUTHOR_LUNG_CLASSES
        },
        "all_author_lung_labels_present": all(
            count_map.get(label, 0) > 0 for label in AUTHOR_LUNG_CLASSES
        ),
    }


def audit_author_case(case_dir: Path, case_id: int) -> Dict[str, Any]:
    sitk = require_sitk()
    paths = case_required_paths(case_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing DUAL author-format files: " + ", ".join(missing)
        )

    fixed_image = sitk.ReadImage(str(paths["fixed_image_t00"]))
    moving_image = sitk.ReadImage(str(paths["moving_image_t50"]))
    fixed_seg = sitk.ReadImage(str(paths["fixed_segmentation_t00"]))
    moving_seg = sitk.ReadImage(str(paths["moving_segmentation_t50"]))
    fixed_seg_array = sitk.GetArrayFromImage(fixed_seg)
    moving_seg_array = sitk.GetArrayFromImage(moving_seg)

    fixed_label_stats = segmentation_label_stats(fixed_seg_array)
    moving_label_stats = segmentation_label_stats(moving_seg_array)
    bbox = dual_union_bbox_xyz(fixed_seg_array, moving_seg_array)
    box = bbox_payload(bbox)

    fixed_landmarks = np.loadtxt(paths["fixed_landmarks_t00"], delimiter=",")
    moving_landmarks = np.loadtxt(paths["moving_landmarks_t50"], delimiter=",")
    if fixed_landmarks.shape != (300, 3) or moving_landmarks.shape != (300, 3):
        raise ValueError(
            f"Case {case_id:02d} author landmarks must both be (300, 3); got "
            f"{fixed_landmarks.shape} and {moving_landmarks.shape}"
        )
    crop_shape = box["shape_xyz"]
    fixed_crop_stats = points_inside_shape(
        crop_landmarks(fixed_landmarks, bbox), crop_shape
    )
    moving_crop_stats = points_inside_shape(
        crop_landmarks(moving_landmarks, bbox), crop_shape
    )
    full_shape = [int(v) for v in fixed_image.GetSize()]
    fixed_full_stats = points_inside_shape(fixed_landmarks, full_shape)
    moving_full_stats = points_inside_shape(moving_landmarks, full_shape)

    checks = {
        "fixed_moving_image_geometry_match": geometry_matches(
            fixed_image, moving_image
        ),
        "fixed_segmentation_geometry_match": geometry_matches(fixed_image, fixed_seg),
        "moving_segmentation_geometry_match": geometry_matches(
            moving_image, moving_seg
        ),
        "fixed_all_five_lobes_present": fixed_label_stats[
            "all_author_lung_labels_present"
        ],
        "moving_all_five_lobes_present": moving_label_stats[
            "all_author_lung_labels_present"
        ],
        "fixed_landmarks_inside_full": fixed_full_stats["all_inside"],
        "moving_landmarks_inside_full": moving_full_stats["all_inside"],
        "fixed_landmarks_inside_dual_crop": fixed_crop_stats["all_inside"],
        "moving_landmarks_inside_dual_crop": moving_crop_stats["all_inside"],
    }
    return {
        "case_id": case_id,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "image_geometry": image_geometry(fixed_image),
        "fixed_segmentation": fixed_label_stats,
        "moving_segmentation": moving_label_stats,
        "dual_union_crop_padding_voxels": AUTHOR_CROP_PADDING,
        "dual_union_crop": box,
        "landmarks": {
            "fixed_full": fixed_full_stats,
            "moving_full": moving_full_stats,
            "fixed_cropped": fixed_crop_stats,
            "moving_cropped": moving_crop_stats,
        },
    }


def write_metadata(path: Path, case: FourDCTCaseFiles) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required in the intended Docker environment."
        ) from exc
    observer_mean, observer_std = OBSERVER_TRE_MM[case.case_id]
    payload = {
        "image_shape": [int(v) for v in case.shape_zyx[::-1]],
        "image_spacing": [float(v) for v in case.spacing_zyx[::-1]],
        "observer_tre_mean": observer_mean,
        "observer_tre_std": observer_std,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def output_file_records(case_dir: Path, dataset_root: Path) -> List[Dict[str, Any]]:
    paths = list(case_required_paths(case_dir).values())
    return [file_record(path, relative_to=dataset_root) for path in sorted(paths)]


def build_case(
    case: FourDCTCaseFiles,
    dataset_root: Path,
    *,
    segmentation_mode: str,
    reuse_segmentations_root: Path | None,
    totalsegmentator_device: str,
) -> Dict[str, Any]:
    case_id = case.case_id
    print(
        f"Preparing DUAL author-format case {case_id:02d}", file=sys.stderr, flush=True
    )
    case_dir = dataset_root / f"case_{case_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    paths = case_required_paths(case_dir)
    source_paths = source_case_paths(case)
    shape_xyz = tuple(int(v) for v in case.shape_zyx[::-1])

    fixed_image = author_image_from_raw(source_paths["fixed_image_t00"], case.shape_zyx)
    moving_image = author_image_from_raw(
        source_paths["moving_image_t50"], case.shape_zyx
    )
    write_author_image(paths["fixed_image_t00"], fixed_image, case.spacing_zyx)
    write_author_image(paths["moving_image_t50"], moving_image, case.spacing_zyx)

    fixed_landmarks = author_landmarks_xyz(
        load_landmarks_xyz(source_paths["fixed_landmarks_t00"]), shape_xyz
    )
    moving_landmarks = author_landmarks_xyz(
        load_landmarks_xyz(source_paths["moving_landmarks_t50"]), shape_xyz
    )
    write_landmarks(paths["fixed_landmarks_t00"], fixed_landmarks)
    write_landmarks(paths["moving_landmarks_t50"], moving_landmarks)
    write_metadata(paths["metadata"], case)

    identity_tre = identity_tre_mm(
        fixed_landmarks,
        moving_landmarks,
        case.spacing_zyx[::-1],
    )
    expected_identity_mean = CANONICAL_RELEASE_IDENTITY_TRE_MEAN_MM[case_id]
    if not np.isclose(
        identity_tre["mean_mm"], expected_identity_mean, atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            f"Case {case_id:02d} canonical identity TRE changed: "
            f"observed {identity_tre['mean_mm']:.12f} mm, expected {expected_identity_mean:.12f} mm"
        )

    if segmentation_mode == "totalsegmentator":
        run_totalsegmentator(
            paths["fixed_image_t00"],
            paths["fixed_segmentation_t00"],
            totalsegmentator_device,
        )
        run_totalsegmentator(
            paths["moving_image_t50"],
            paths["moving_segmentation_t50"],
            totalsegmentator_device,
        )
        segmentation_sources: Dict[str, str] = {
            "fixed_t00": "generated_from_case_image",
            "moving_t50": "generated_from_case_image",
        }
    elif segmentation_mode == "reuse":
        if reuse_segmentations_root is None:
            raise ValueError(
                "--reuse-segmentations-root is required when --segmentation-mode=reuse"
            )
        fixed_source = find_reusable_segmentation(
            reuse_segmentations_root, case_id, AUTHOR_FIXED_PHASE
        )
        moving_source = find_reusable_segmentation(
            reuse_segmentations_root, case_id, AUTHOR_MOVING_PHASE
        )
        copy_segmentation_as_nii(fixed_source, paths["fixed_segmentation_t00"])
        copy_segmentation_as_nii(moving_source, paths["moving_segmentation_t50"])
        segmentation_sources = {
            "fixed_t00": str(fixed_source),
            "moving_t50": str(moving_source),
        }
    else:
        raise ValueError(f"Unknown segmentation mode: {segmentation_mode}")

    audit = audit_author_case(case_dir, case_id)
    if audit["status"] != "pass":
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"Case {case_id:02d} DUAL compatibility audit failed: {failed}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "generated_at_utc": utc_now(),
        "case_id": case_id,
        "fixed_phase": "T00 / author phase_00",
        "moving_phase": "T50 / author phase_05",
        "segmentation_mode": segmentation_mode,
        "segmentation_sources": segmentation_sources,
        "identity_tre_from_canonical_release_landmarks": identity_tre,
        "published_no_registration_tre_mm": {
            "mean": PUBLISHED_NO_REG_TRE_MM[case_id][0],
            "std": PUBLISHED_NO_REG_TRE_MM[case_id][1],
            "role": "paper metadata only; not the adapter validation source of truth",
        },
        "release_discrepancy_mm": {
            "canonical_minus_published_mean": identity_tre["mean_mm"]
            - PUBLISHED_NO_REG_TRE_MM[case_id][0],
            "status": "recorded_unresolved_release_or_convention_difference",
        },
        "source_files": {
            key: file_record(path) for key, path in sorted(source_paths.items())
        },
        "output_files": output_file_records(case_dir, dataset_root),
        "audit": audit,
    }
    manifest_path = case_dir / "adapter_case_manifest.json"
    write_json(manifest_path, manifest)
    print(
        f"Validated DUAL author-format case {case_id:02d}", file=sys.stderr, flush=True
    )
    return {
        "case_id": case_id,
        "status": "pass",
        "case_manifest": manifest_path.relative_to(dataset_root).as_posix(),
        "case_manifest_sha256": sha256_file(manifest_path),
        "dual_union_crop": audit["dual_union_crop"],
    }


def contract_payload(
    *,
    version: str,
    case_ids: Sequence[int],
    segmentation_mode: str,
    totalsegmentator_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "artifact_version": version,
        "dataset": "DIR-Lab 4DCT",
        "case_ids": [int(v) for v in case_ids],
        "phase_contract": {
            "fixed": "T00 -> phase_00 -> extreme_landmarks_0.csv",
            "moving": "T50 -> phase_05 -> extreme_landmarks_5.csv",
        },
        "image_contract": {
            "source_dtype": "native np.int16",
            "intensity": "subtract 1024, clip [-1024, 3071]",
            "source_array_order": "zyx",
            "stored_sitk_array_order": "zyx after np.flip(axis=0)",
            "dual_loaded_array_order": "xyz after np.swapaxes(zyx, 0, 2)",
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "landmark_contract": {
            "source_order": "xyz",
            "source_indexing": "DIR-Lab text convention (treated as 1-based for source validation)",
            "author_transform": "x and y retained; z_out = depth_z - z_in",
            "author_indexing_warning": "matches DUAL converter; no x/y one-to-zero-based subtraction",
            "crop_transform": "subtract half-open union-crop start_xyz",
        },
        "segmentation_contract": {
            "mode": segmentation_mode,
            "tool": "TotalSegmentator total task, multilabel output (ml=True)",
            "tool_info": dict(totalsegmentator_info),
            "dual_lung_classes": {
                str(k): v for k, v in AUTHOR_LUNG_CLASS_NAMES.items()
            },
        },
        "crop_contract": {
            "performed_by": "external DUAL-INR-DIR load_and_crop_full_dirlab",
            "adapter_behavior": "retain full volumes; independently audit exact prospective crop",
            "mask": "union of fixed and moving labels 10..14",
            "array_order": "xyz after DUAL axis swap",
            "padding_voxels": AUTHOR_CROP_PADDING,
            "bounds": "per-axis min/max foreground, half-open stop=max+padding+1, clipped to image",
        },
        "identity_landmark_baseline": {
            "source_of_truth": "computed from this canonical extracted release and upstream converter pairing",
            "canonical_release_case_mean_tre_mm": {
                str(case_id): value
                for case_id, value in CANONICAL_RELEASE_IDENTITY_TRE_MEAN_MM.items()
            },
            "published_dual_paper_no_reg_case_mean_std_tre_mm": {
                str(case_id): {"mean": values[0], "std": values[1]}
                for case_id, values in PUBLISHED_NO_REG_TRE_MM.items()
            },
            "published_dual_paper_overall_mean_std_tre_mm": {"mean": 8.52, "std": 5.57},
            "discrepancy_policy": (
                "The paper row is retained for traceability, but exact canonical-release recomputation "
                "is used for adapter validation. Differences are recorded as an unresolved release or "
                "landmark-convention discrepancy and must not be presented as a registration result."
            ),
            "source_checksum_policy": (
                "Each case manifest records SHA-256 for both source landmark files before conversion."
            ),
        },
        "reference": {
            "external_path": "external/DUAL-INR-DIR",
            "external_git_revision": read_git_revision(
                REPO_ROOT / "external" / "DUAL-INR-DIR"
            ),
            "repository_git_revision": read_git_revision(REPO_ROOT),
            "adapter_script": str(Path(__file__).relative_to(REPO_ROOT)),
            "adapter_script_sha256": sha256_file(Path(__file__)),
        },
    }


def write_checksums(dataset_root: Path) -> Path:
    checksum_path = dataset_root / "checksums.sha256"
    files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w") as handle:
        for path in files:
            handle.write(
                f"{sha256_file(path)}  {path.relative_to(dataset_root).as_posix()}\n"
            )
    return checksum_path


def verify_checksums(
    dataset_root: Path,
    *,
    require_complete_listing: bool = True,
) -> Dict[str, Any]:
    checksum_path = dataset_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checked = 0
    failures: List[Dict[str, str]] = []
    listed_paths: set[str] = set()
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, separator, relative = line.partition("  ")
        if not separator or not expected or not relative:
            failures.append({"path": relative, "reason": "malformed checksum line"})
            continue
        path = dataset_root / relative
        listed_paths.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        observed = sha256_file(path)
        checked += 1
        if observed != expected:
            failures.append({"path": relative, "reason": "sha256 mismatch"})
    if require_complete_listing:
        actual_paths = {
            path.relative_to(dataset_root).as_posix()
            for path in dataset_root.rglob("*")
            if path.is_file() and path != checksum_path
        }
        for relative in sorted(actual_paths - listed_paths):
            failures.append(
                {"path": relative, "reason": "not listed in checksums.sha256"}
            )
        for relative in sorted(listed_paths - actual_paths):
            if not any(row["path"] == relative for row in failures):
                failures.append({"path": relative, "reason": "listed file missing"})
    return {
        "status": "pass" if not failures else "fail",
        "checked_files": checked,
        "failures": failures,
    }


def permission_audit(dataset_root: Path) -> Dict[str, Any]:
    root_mode = dataset_root.stat().st_mode & 0o777
    writable_by_group_or_other: List[Dict[str, str]] = []
    for path in [dataset_root, *sorted(dataset_root.rglob("*"))]:
        mode = path.stat().st_mode & 0o777
        if mode & 0o022:
            writable_by_group_or_other.append(
                {
                    "path": (
                        "."
                        if path == dataset_root
                        else path.relative_to(dataset_root).as_posix()
                    ),
                    "mode": f"{mode:04o}",
                }
            )
    return {
        "status": (
            "pass" if root_mode == 0o755 and not writable_by_group_or_other else "fail"
        ),
        "root_mode": f"{root_mode:04o}",
        "required_root_mode": "0755",
        "policy": "owner writable; group/other readable and traversable where applicable; no group/other write",
        "group_or_other_writable_paths": writable_by_group_or_other,
    }


def install_executed_script_provenance(
    dataset_root: Path, source_script: Path
) -> Dict[str, Any]:
    target = dataset_root / "provenance" / "prepare_dual_author_4dct_executed.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_script, target)
    target.chmod(0o444)
    return {
        **file_record(target, relative_to=dataset_root),
        "mode": f"{target.stat().st_mode & 0o777:04o}",
    }


def executed_script_provenance_audit(dataset_root: Path) -> Dict[str, Any]:
    contract = json.loads((dataset_root / "preprocess_contract.json").read_text())
    expected = contract["reference"]["adapter_script_sha256"]
    path = dataset_root / "provenance" / "prepare_dual_author_4dct_executed.py"
    if not path.is_file():
        return {"status": "fail", "reason": "missing", "expected_sha256": expected}
    observed = sha256_file(path)
    mode = path.stat().st_mode & 0o777
    return {
        "status": "pass" if observed == expected and mode == 0o444 else "fail",
        "path": path.relative_to(dataset_root).as_posix(),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "mode": f"{mode:04o}",
        "required_mode": "0444",
    }


def finalize_existing_artifact(dataset_root: Path) -> Dict[str, Any]:
    """Finalize the one artifact produced before root-mode handling was encoded."""
    root = dataset_root.resolve()
    if not (root / "dataset_manifest.json").is_file():
        raise FileNotFoundError(root / "dataset_manifest.json")
    executed_audit = executed_script_provenance_audit(root)
    if executed_audit["status"] != "pass":
        raise RuntimeError(f"Executed-script provenance is not valid: {executed_audit}")
    pre_checksum_path = root / "checksums.sha256"
    pre_checksum_sha256 = sha256_file(pre_checksum_path)
    pre_mode = f"{root.stat().st_mode & 0o777:04o}"

    root.chmod(0o755)
    permissions = permission_audit(root)
    if permissions["status"] != "pass":
        raise RuntimeError(f"Permission audit failed after root chmod: {permissions}")

    provenance_path = root / "provenance" / "post_publication_finalization.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "finalized_at_utc": utc_now(),
        "operation": "post-publication provenance capture, root traversal permission correction, and checksum regeneration",
        "artifact_content_note": "No image, segmentation, landmark, metadata, case manifest, dataset manifest, or preprocess contract content was altered.",
        "root_mode_before": pre_mode,
        "root_mode_after": permissions["root_mode"],
        "permission_policy": permissions["policy"],
        "executed_adapter": executed_audit,
        "finalizer_script": {
            "repository_relative_path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__)),
            "role": "post-publication finalizer; distinct from the captured executed adapter",
        },
        "pre_finalization_checksums_sha256": pre_checksum_sha256,
        "checksum_action": "checksums.sha256 regenerated after adding provenance files, then independently verified",
    }
    write_json(provenance_path, payload)
    write_checksums(root)
    checksum_audit = verify_checksums(root)
    if checksum_audit["status"] != "pass":
        raise RuntimeError(f"Final checksum audit failed: {checksum_audit}")
    return {
        "mode": "finalize_existing",
        "status": "pass",
        "dataset_root": str(root),
        "permission_audit": permissions,
        "executed_script_provenance": executed_audit,
        "checksum_audit": checksum_audit,
        "provenance_path": provenance_path.relative_to(root).as_posix(),
    }


def dry_validate_prepare(args: argparse.Namespace) -> Dict[str, Any]:
    target = validate_output_location(args.output_parent, args.version)
    if target.exists():
        raise FileExistsError(f"Versioned target already exists: {target}")
    case_ids = parse_and_validate_case_ids(args.case_ids)
    sources = [validate_source_case(case_id, args.data_root)[1] for case_id in case_ids]
    if args.segmentation_mode == "totalsegmentator":
        tool_info = validate_totalsegmentator_installation(hash_weights=True)
        reuse = None
    else:
        if args.reuse_segmentations_root is None:
            raise ValueError(
                "--reuse-segmentations-root is required when --segmentation-mode=reuse"
            )
        reuse = {
            f"case_{case_id:02d}": {
                f"phase_{phase:02d}": str(
                    find_reusable_segmentation(
                        args.reuse_segmentations_root, case_id, phase
                    )
                )
                for phase in (AUTHOR_FIXED_PHASE, AUTHOR_MOVING_PHASE)
            }
            for case_id in case_ids
        }
        tool_info = {
            "mode": "reuse",
            "lung_class_map_expected": dict(AUTHOR_LUNG_CLASS_NAMES),
        }
    return {
        "mode": "prepare_dry_run",
        "status": "pass",
        "writes_performed": False,
        "prospective_target": str(target),
        "case_ids": case_ids,
        "source_validation": sources,
        "segmentation_mode": args.segmentation_mode,
        "segmentation_tool": tool_info,
        "reuse_segmentations": reuse,
    }


def prepare(args: argparse.Namespace) -> Path:
    target = validate_output_location(args.output_parent, args.version)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite versioned target: {target}")
    case_ids = parse_and_validate_case_ids(args.case_ids)
    resolved = [
        validate_source_case(case_id, args.data_root)[0] for case_id in case_ids
    ]
    if args.segmentation_mode == "totalsegmentator":
        # This first check pins API/class semantics. Weights may be downloaded
        # lazily by the first segmentation, so final hashed provenance is
        # resolved only after all cases are built.
        tool_info = validate_totalsegmentator_installation()
    else:
        if args.reuse_segmentations_root is None:
            raise ValueError(
                "--reuse-segmentations-root is required when --segmentation-mode=reuse"
            )
        for case_id in case_ids:
            for phase in (AUTHOR_FIXED_PHASE, AUTHOR_MOVING_PHASE):
                find_reusable_segmentation(
                    args.reuse_segmentations_root, case_id, phase
                )
        tool_info = {
            "mode": "reuse",
            "lung_class_map_expected": dict(AUTHOR_LUNG_CLASS_NAMES),
        }

    args.output_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{args.version}.staging-", dir=args.output_parent
    ) as staging_name:
        staging_root = Path(staging_name)
        rows = [
            build_case(
                case,
                staging_root,
                segmentation_mode=args.segmentation_mode,
                reuse_segmentations_root=args.reuse_segmentations_root,
                totalsegmentator_device=args.totalsegmentator_device,
            )
            for case in resolved
        ]
        if args.segmentation_mode == "totalsegmentator":
            tool_info = validate_totalsegmentator_installation(
                require_weights=True,
                hash_weights=True,
            )
        contract = contract_payload(
            version=args.version,
            case_ids=case_ids,
            segmentation_mode=args.segmentation_mode,
            totalsegmentator_info=tool_info,
        )
        write_json(staging_root / "preprocess_contract.json", contract)
        dataset_manifest = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
            "artifact_version": args.version,
            "generated_at_utc": utc_now(),
            "status": "pass",
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "rows": rows,
            "contract": "preprocess_contract.json",
            "checksums": "checksums.sha256",
        }
        write_json(staging_root / "dataset_manifest.json", dataset_manifest)
        executed_script = install_executed_script_provenance(
            staging_root, Path(__file__)
        )
        staging_root.chmod(0o755)
        finalization = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
            "finalized_at_utc": utc_now(),
            "operation": "in-process immutable artifact finalization before atomic publication",
            "permission_policy": (
                "artifact root 0755; owner writable; no group/other write; "
                "executed adapter captured read-only at 0444"
            ),
            "executed_adapter": executed_script,
            "checksum_action": "all artifact files except checksums.sha256 itself are listed and hashed",
        }
        write_json(
            staging_root / "provenance" / "artifact_finalization.json", finalization
        )
        write_checksums(staging_root)
        checksum_audit = verify_checksums(staging_root)
        if checksum_audit["status"] != "pass":
            raise RuntimeError(
                f"Checksum self-audit failed: {checksum_audit['failures']}"
            )
        staging_root.replace(target)
        permissions = permission_audit(target)
        if permissions["status"] != "pass":
            raise RuntimeError(
                f"Published artifact permission audit failed: {permissions}"
            )
    return target


def validate_existing(
    dataset_root: Path, requested_case_ids: str | None
) -> Dict[str, Any]:
    root = dataset_root.resolve()
    manifest_path = root / "dataset_manifest.json"
    contract_path = root / "preprocess_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset_manifest.json or preprocess_contract.json under {root}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("adapter_id") != ADAPTER_ID:
        raise ValueError(
            f"Unexpected adapter_id in {manifest_path}: {manifest.get('adapter_id')}"
        )
    if requested_case_ids:
        case_ids = parse_and_validate_case_ids(requested_case_ids)
    else:
        case_ids = [int(v) for v in manifest["case_ids"]]
    rows = [
        audit_author_case(root / f"case_{case_id:02d}", case_id) for case_id in case_ids
    ]
    checksum_audit = verify_checksums(root)
    permissions = permission_audit(root)
    executed_provenance = executed_script_provenance_audit(root)
    passed = bool(
        checksum_audit["status"] == "pass"
        and permissions["status"] == "pass"
        and executed_provenance["status"] == "pass"
        and all(row["status"] == "pass" for row in rows)
    )
    return {
        "mode": "validate_existing",
        "status": "pass" if passed else "fail",
        "writes_performed": False,
        "dataset_root": str(root),
        "case_ids": case_ids,
        "case_audits": rows,
        "checksum_audit": checksum_audit,
        "permission_audit": permissions,
        "executed_script_provenance": executed_provenance,
    }


def parse_and_validate_case_ids(text: str) -> List[int]:
    case_ids = parse_case_ids(text)
    if not case_ids:
        raise ValueError("At least one case id is required.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case ids are not allowed: {case_ids}")
    invalid = [case_id for case_id in case_ids if case_id < 1 or case_id > 10]
    if invalid:
        raise ValueError(f"4DCT case ids must be in [1, 10], got {invalid}")
    return case_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Create or dry-validate a versioned artifact."
    )
    prepare_parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Canonical DIR-Lab root containing raw/4DCT.",
    )
    prepare_parser.add_argument(
        "--output-parent",
        type=Path,
        required=True,
        help="Outside-Git parent for version directories.",
    )
    prepare_parser.add_argument(
        "--version",
        required=True,
        help="Immutable artifact directory name, e.g. dual_author_4dct_t00t50_v001.",
    )
    prepare_parser.add_argument("--case-ids", default="1-10")
    prepare_parser.add_argument(
        "--segmentation-mode",
        choices=("totalsegmentator", "reuse"),
        default="totalsegmentator",
    )
    prepare_parser.add_argument("--reuse-segmentations-root", type=Path, default=None)
    prepare_parser.add_argument("--totalsegmentator-device", default="gpu")
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read-only validation; create no files or directories.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Read-only audit of an existing artifact."
    )
    validate_parser.add_argument("--dataset-root", type=Path, required=True)
    validate_parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional subset; defaults to manifest case_ids.",
    )

    finalize_parser = subparsers.add_parser(
        "finalize-existing",
        help="One-time provenance/permission/checksum finalization for a just-published artifact.",
    )
    finalize_parser.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare" and args.dry_run:
            payload = dry_validate_prepare(args)
        elif args.command == "prepare":
            target = prepare(args)
            payload = {
                "mode": "prepare",
                "status": "pass",
                "dataset_root": str(target),
                "training_launched": False,
            }
        elif args.command == "validate":
            payload = validate_existing(args.dataset_root, args.case_ids)
        else:
            payload = finalize_existing_artifact(args.dataset_root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "pass":
            raise SystemExit(1)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
