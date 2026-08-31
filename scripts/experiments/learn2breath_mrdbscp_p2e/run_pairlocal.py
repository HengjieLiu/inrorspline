#!/usr/bin/env python3
"""Run the frozen MR-D-BSCP P2e adaptation on Learn2Breath-style CT pairs.

The optimisation path intentionally follows the historical COPD P2e run:
DirectCP cubic B-spline controls, pTV-style local correlation and physical
isoTV, and exact dyadic control-grid prolongation.  Only the input product is
new: a full, shared 1.5-mm NIfTI grid rather than the historic 1-mm landmark
crop.

The solver operates internally on `[Z,Y,X]` tensors because that is the
historical P2e implementation layout.  Its public field contract is instead
the Learn2Breath contract: `float32 [3,X,Y,Z]`, components `x,y,z`, in fixed
grid voxels, such that `moving(fixed + displacement)` is sampled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
HISTORICAL_DIR = THIS_DIR.parent / "lungct_no_sweep_repro"
if str(HISTORICAL_DIR) not in sys.path:
    sys.path.insert(0, str(HISTORICAL_DIR))
import common as p2e  # noqa: E402  # Historical P2e low-level implementation.


EXPERIMENT_ID = "2026-08-30_learn2breath_mrdbscp_p2e_pairlocal"
EXPECTED_SHAPE_XYZ = (224, 192, 224)
EXPECTED_SPACING_XYZ = (1.5, 1.5, 1.5)
STAGE_FULL_CPS = (64, 32, 16, 8, 4)
STAGE_DOWNSAMPLE = (16, 8, 4, 2, 1)
STAGE_EPOCHS = (400, 400, 400, 400, 900)
STAGE_LR = 1.0e-3
# Historical P2e used raw DIR-Lab stored values.  The pair-local manifest
# documents physical HU = stored_value - 1000, so historical stored [80,900]
# corresponds to physical HU [-920,-100].  Validation CT is supplied as HU,
# so profiles are always expressed in physical HU and never assume an
# undocumented stored-value offset.
P2E_RAW_TO_HU_INTERCEPT = -1000.0
P2E_WINDOW_LOW_STORED = 80.0
P2E_WINDOW_HIGH_STORED = 900.0
INTENSITY_PROFILES: dict[str, dict[str, Any]] = {
    "p2e_equiv_hu": {
        "low_hu": -920.0,
        "high_hu": -100.0,
        "description": "Historical P2e stored [80,900] expressed in physical HU.",
        "protocol_tag": "historical_p2e_stored80_900_equivalent_hu_minus920_minus100",
    },
    "l2b_lung_hu": {
        "low_hu": -1000.0,
        "high_hu": 200.0,
        "description": "Learn2Breath baseline lung window in physical HU.",
        "protocol_tag": "learn2breath_baseline_lung_window_hu_minus1000_200",
    },
}
VALIDATION_LOBE_LABELS = (8, 16, 32, 64, 128)
P2E_LCC_SIGMA_MM = 2.1
P2E_LCC_RADIUS_SIGMA = 2.0
P2E_LCC_SIGMA_FLOOR_PIX = 0.8
P2E_ISOTV_WEIGHT = 0.11
P2E_CSQRT = 5.0e-3
P2E_BORDER_VOXELS = 5
KNOWN_SHIFT_XYZ = np.asarray([1.25, -0.75, 0.5], dtype=np.float32)


class P2ePairlocalError(RuntimeError):
    """Raised when a frozen input or field contract is violated."""


@dataclass(frozen=True)
class PairCase:
    """One CT-only fixed/moving registration pair in solver and NIfTI layouts."""

    dataset: str
    case_label: str
    intensity_profile: str
    fixed_path: Path
    moving_path: Path
    fixed_hu_xyz: np.ndarray
    moving_hu_xyz: np.ndarray
    fixed_zyx: np.ndarray
    moving_zyx: np.ndarray
    fixed_affine_xyz: np.ndarray
    moving_affine_xyz: np.ndarray
    fixed_header: Any
    spacing_xyz_mm: tuple[float, float, float]
    landmark_ids: np.ndarray | None
    fixed_points_xyz: np.ndarray | None
    moving_points_xyz: np.ndarray | None

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.fixed_hu_xyz.shape)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.fixed_zyx.shape)

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        return tuple(reversed(self.spacing_xyz_mm))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in names and not isinstance(value, (list, tuple, dict)):
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in names} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_nibabel():
    import nibabel as nib  # noqa: PLC0415

    return nib


def intensity_profile_config(name: str) -> dict[str, Any]:
    """Return the explicit physical-HU normalization profile."""

    try:
        profile = INTENSITY_PROFILES[str(name)]
    except KeyError as exc:
        raise P2ePairlocalError(
            f"Unknown intensity profile {name!r}; expected one of {tuple(INTENSITY_PROFILES)}"
        ) from exc
    return {"name": str(name), **profile}


def normalize_p2e_ct(array_hu: np.ndarray, intensity_profile: str = "p2e_equiv_hu") -> np.ndarray:
    """Normalize physical-HU CT according to one locked audit profile.

    ``p2e_equiv_hu`` is mathematically identical to historical P2e's raw
    ``clip(stored, 80, 900)`` when ``HU = stored - 1000``.  Expressing both
    profiles directly in physical HU prevents applying a raw stored-value
    range to an already converted NIfTI image.
    """

    profile = intensity_profile_config(intensity_profile)
    low = np.float32(profile["low_hu"])
    high = np.float32(profile["high_hu"])
    clipped = np.clip(np.asarray(array_hu, dtype=np.float32), low, high)
    return np.ascontiguousarray((clipped - low) / (high - low))


def xyz_to_zyx(array_xyz: np.ndarray) -> np.ndarray:
    array = np.asarray(array_xyz)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D XYZ volume, got {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 1, 0), dtype=np.float32)


def zyx_to_xyz(array_zyx: np.ndarray) -> np.ndarray:
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D ZYX volume, got {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 1, 0), dtype=np.float32)


def affine_points(affine: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    return points @ np.asarray(affine, dtype=np.float64)[:3, :3].T + np.asarray(affine, dtype=np.float64)[:3, 3]


def require_points_in_bounds(points_xyz: np.ndarray, shape_xyz: Sequence[int], label: str) -> None:
    points = np.asarray(points_xyz, dtype=np.float64)
    shape = np.asarray(shape_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise P2ePairlocalError(f"{label} must be finite [N,3] XYZ coordinates, got {points.shape}")
    if np.any(points < 0.0) or np.any(points > shape[None, :] - 1.0):
        raise P2ePairlocalError(f"{label} falls outside fixed grid {tuple(shape.astype(int))}")


def read_paired_landmarks(path: Path, shape_xyz: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "id",
        "fixed_x_voxel",
        "fixed_y_voxel",
        "fixed_z_voxel",
        "moving_x_voxel",
        "moving_y_voxel",
        "moving_z_voxel",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300 or not rows or any(key not in rows[0] for key in required):
        raise P2ePairlocalError(f"Expected 300 complete paired landmarks in {path}")
    ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, 301, dtype=np.int64)):
        raise P2ePairlocalError("Paired landmark IDs must be ordered 1..300")
    fixed = np.asarray(
        [[float(row["fixed_x_voxel"]), float(row["fixed_y_voxel"]), float(row["fixed_z_voxel"])] for row in rows],
        dtype=np.float64,
    )
    moving = np.asarray(
        [[float(row["moving_x_voxel"]), float(row["moving_y_voxel"]), float(row["moving_z_voxel"])] for row in rows],
        dtype=np.float64,
    )
    require_points_in_bounds(fixed, shape_xyz, "fixed landmarks")
    require_points_in_bounds(moving, shape_xyz, "moving landmarks")
    return ids, fixed, moving


def pair_paths(dataset: str, data_root: Path, case_id: str) -> tuple[str, Path, Path, Path | None]:
    if dataset == "copd":
        token = str(int(case_id))
        root = data_root
        return (
            f"case_{token}",
            root / "imagesTs" / f"case_{token}_1.nii.gz",
            root / "imagesTs" / f"case_{token}_2.nii.gz",
            root / "landmarks" / f"case_{token}_paired.csv",
        )
    if dataset == "validation":
        token = f"NLST_{int(case_id):04d}"
        root = data_root / "validation" / "ct_data"
        return token, root / f"{token}_INSP.nii.gz", root / f"{token}_EXP.nii.gz", None
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_pair(dataset: str, data_root: Path, case_id: str, intensity_profile: str = "p2e_equiv_hu") -> PairCase:
    nib = import_nibabel()
    label, fixed_path, moving_path, landmarks_path = pair_paths(dataset, data_root, case_id)
    for path in (fixed_path, moving_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    if tuple(fixed_img.shape) != EXPECTED_SHAPE_XYZ or tuple(moving_img.shape) != EXPECTED_SHAPE_XYZ:
        raise P2ePairlocalError(f"Expected {EXPECTED_SHAPE_XYZ}, got {fixed_img.shape} / {moving_img.shape}")
    spacing = tuple(float(value) for value in fixed_img.header.get_zooms()[:3])
    if not np.allclose(spacing, EXPECTED_SPACING_XYZ, rtol=0.0, atol=1.0e-5):
        raise P2ePairlocalError(f"Expected 1.5-mm isotropic spacing, got {spacing}")
    fixed_affine = np.asarray(fixed_img.affine, dtype=np.float64)
    moving_affine = np.asarray(moving_img.affine, dtype=np.float64)
    if not np.allclose(fixed_affine, moving_affine, rtol=0.0, atol=1.0e-6):
        raise P2ePairlocalError("Fixed/moving affine mismatch")
    fixed_hu = np.asarray(fixed_img.get_fdata(dtype=np.float32), dtype=np.float32)
    moving_hu = np.asarray(moving_img.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(fixed_hu).all() or not np.isfinite(moving_hu).all():
        raise P2ePairlocalError("CT contains non-finite HU")
    ids: np.ndarray | None = None
    fixed_points: np.ndarray | None = None
    moving_points: np.ndarray | None = None
    if landmarks_path is not None:
        if not landmarks_path.is_file():
            raise FileNotFoundError(landmarks_path)
        ids, fixed_points, moving_points = read_paired_landmarks(landmarks_path, fixed_hu.shape)
    return PairCase(
        dataset=dataset,
        case_label=label,
        intensity_profile=str(intensity_profile),
        fixed_path=fixed_path,
        moving_path=moving_path,
        fixed_hu_xyz=fixed_hu,
        moving_hu_xyz=moving_hu,
        fixed_zyx=xyz_to_zyx(normalize_p2e_ct(fixed_hu, intensity_profile)),
        moving_zyx=xyz_to_zyx(normalize_p2e_ct(moving_hu, intensity_profile)),
        fixed_affine_xyz=fixed_affine,
        moving_affine_xyz=moving_affine,
        fixed_header=fixed_img.header.copy(),
        spacing_xyz_mm=spacing,
        landmark_ids=ids,
        fixed_points_xyz=fixed_points,
        moving_points_xyz=moving_points,
    )


def r231_paths(r231_root: Path, case_id: str) -> tuple[Path, Path]:
    token = str(int(case_id))
    return (
        r231_root / f"case_{token}_1_whole_lung.nii.gz",
        r231_root / f"case_{token}_2_whole_lung.nii.gz",
    )


def validate_r231_pair(case: PairCase, r231_root: Path, case_id: str) -> dict[str, Any]:
    nib = import_nibabel()
    fixed_path, moving_path = r231_paths(r231_root, case_id)
    for path in (fixed_path, moving_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixed = nib.load(str(fixed_path))
    moving = nib.load(str(moving_path))
    if tuple(fixed.shape) != case.shape_xyz or tuple(moving.shape) != case.shape_xyz:
        raise P2ePairlocalError("R231 mask shape does not match the fixed CT grid")
    if not np.allclose(fixed.affine, case.fixed_affine_xyz) or not np.allclose(moving.affine, case.fixed_affine_xyz):
        raise P2ePairlocalError("R231 mask affine does not match the fixed CT grid")
    fixed_mask = np.asarray(fixed.get_fdata(), dtype=np.float32) > 0
    moving_mask = np.asarray(moving.get_fdata(), dtype=np.float32) > 0
    if not fixed_mask.any() or not moving_mask.any():
        raise P2ePairlocalError("R231 mask is empty")
    return {
        "fixed_r231_path": fixed_path,
        "moving_r231_path": moving_path,
        "fixed_foreground_voxels": int(fixed_mask.sum()),
        "moving_foreground_voxels": int(moving_mask.sum()),
        "fixed_sha256": sha256_file(fixed_path),
        "moving_sha256": sha256_file(moving_path),
    }


def load_learn2breath_evaluator(learn2breath_repo: Path):
    evaluator_path = learn2breath_repo / "scripts" / "experiments" / "fireants_validation" / "fireants_syn_validation.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(evaluator_path)
    spec = importlib.util.spec_from_file_location("learn2breath_validation_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise P2ePairlocalError(f"Cannot import Learn2Breath evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_validation_lobes(case: PairCase, data_root: Path, case_id: str, evaluator) -> dict[str, Any]:
    """Validate the challenge lobe labels that are reserved for QC/evaluation."""

    nib = import_nibabel()
    paths = evaluator.validation_paths(data_root, int(case_id))
    fixed_path = Path(paths["insp_lobe"])
    moving_path = Path(paths["exp_lobe"])
    for path in (fixed_path, moving_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixed = nib.load(str(fixed_path))
    moving = nib.load(str(moving_path))
    if tuple(fixed.shape) != case.shape_xyz or tuple(moving.shape) != case.shape_xyz:
        raise P2ePairlocalError("Validation lobe-mask shape does not match the CT grid")
    if not np.allclose(fixed.affine, case.fixed_affine_xyz) or not np.allclose(moving.affine, case.fixed_affine_xyz):
        raise P2ePairlocalError("Validation lobe-mask affine does not match the CT grid")
    fixed_values = np.rint(np.asarray(fixed.get_fdata(dtype=np.float32))).astype(np.int16)
    moving_values = np.rint(np.asarray(moving.get_fdata(dtype=np.float32))).astype(np.int16)
    allowed = {0, *VALIDATION_LOBE_LABELS}
    fixed_labels = {int(value) for value in np.unique(fixed_values)}
    moving_labels = {int(value) for value in np.unique(moving_values)}
    if fixed_labels != allowed or moving_labels != allowed:
        raise P2ePairlocalError(
            f"Validation lobe labels must be {sorted(allowed)}, got fixed={sorted(fixed_labels)}, moving={sorted(moving_labels)}"
        )
    fixed_counts = {str(label): int((fixed_values == label).sum()) for label in VALIDATION_LOBE_LABELS}
    moving_counts = {str(label): int((moving_values == label).sum()) for label in VALIDATION_LOBE_LABELS}
    if any(count <= 0 for count in (*fixed_counts.values(), *moving_counts.values())):
        raise P2ePairlocalError("Validation lobe label is empty")
    return {
        "fixed_lobe_path": fixed_path,
        "moving_lobe_path": moving_path,
        "labels": list(VALIDATION_LOBE_LABELS),
        "fixed_label_voxels": fixed_counts,
        "moving_label_voxels": moving_counts,
        "fixed_sha256": sha256_file(fixed_path),
        "moving_sha256": sha256_file(moving_path),
    }


def _find_stored_value_to_hu(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        candidate = value.get("stored_value_to_hu")
        if isinstance(candidate, dict) and {"scale", "intercept"}.issubset(candidate):
            return {"scale": float(candidate["scale"]), "intercept": float(candidate["intercept"])}
        for child in value.values():
            found = _find_stored_value_to_hu(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_stored_value_to_hu(child)
            if found is not None:
                return found
    return None


def intensity_provenance(case: PairCase, data_root: Path, case_id: str) -> dict[str, Any]:
    profile = intensity_profile_config(case.intensity_profile)
    result: dict[str, Any] = {
        "input_units": "physical_HU",
        "intensity_profile": profile,
    }
    if case.dataset == "copd":
        manifest_path = data_root.parent / "manifests" / f"case_{int(case_id)}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_to_hu = _find_stored_value_to_hu(source)
        expected = {"scale": 1.0, "intercept": P2E_RAW_TO_HU_INTERCEPT}
        if stored_to_hu != expected:
            raise P2ePairlocalError(
                f"COPD manifest does not declare the expected stored-to-HU mapping {expected}: {stored_to_hu}"
            )
        result.update(
            {
                "source_manifest": manifest_path,
                "stored_value_to_hu": stored_to_hu,
                "p2e_equivalence": "stored [80,900] == physical HU [-920,-100]",
            }
        )
    else:
        result.update(
            {
                "source_manifest": None,
                "stored_value_to_hu": None,
                "p2e_equivalence": "Validation input is supplied as physical HU; use the explicit HU profile without inferring raw storage.",
            }
        )
    return result


def sample_field_trilinear_xyz(field_cxyz: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    field = np.asarray(field_cxyz, dtype=np.float32)
    if field.shape[0] != 3:
        raise ValueError(f"Expected [3,X,Y,Z] field, got {field.shape}")
    require_points_in_bounds(points_xyz, field.shape[1:], "field sample points")
    points = np.asarray(points_xyz, dtype=np.float64)
    lower = np.floor(points).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(field.shape[1:], dtype=np.int64) - 1)
    weights = points - lower
    samples = np.zeros((points.shape[0], 3), dtype=np.float64)
    for sx in (0, 1):
        ix = upper[:, 0] if sx else lower[:, 0]
        wx = weights[:, 0] if sx else 1.0 - weights[:, 0]
        for sy in (0, 1):
            iy = upper[:, 1] if sy else lower[:, 1]
            wy = weights[:, 1] if sy else 1.0 - weights[:, 1]
            for sz in (0, 1):
                iz = upper[:, 2] if sz else lower[:, 2]
                wz = weights[:, 2] if sz else 1.0 - weights[:, 2]
                samples += (wx * wy * wz)[:, None] * field[:, ix, iy, iz].T
    return samples


def normalized_zyx_to_field_cxyz(dense_norm_zyx_xyz: np.ndarray) -> np.ndarray:
    dense = np.asarray(dense_norm_zyx_xyz, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[-1] != 3:
        raise ValueError(f"Expected [Z,Y,X,3] normalized displacement, got {dense.shape}")
    z, y, x = (int(value) for value in dense.shape[:3])
    scales_xyz = np.asarray([(x - 1) / 2.0, (y - 1) / 2.0, (z - 1) / 2.0], dtype=np.float32)
    dense_vox_zyx_xyz = dense * scales_xyz.reshape(1, 1, 1, 3)
    dense_vox_xyz_xyz = np.ascontiguousarray(dense_vox_zyx_xyz.transpose(2, 1, 0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.moveaxis(dense_vox_xyz_xyz, -1, 0), dtype=np.float32)


def field_cxyz_to_normalized_zyx(field_cxyz: np.ndarray) -> np.ndarray:
    field = np.asarray(field_cxyz, dtype=np.float32)
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(f"Expected [3,X,Y,Z] field, got {field.shape}")
    x, y, z = (int(value) for value in field.shape[1:])
    dense_vox_xyz_xyz = np.moveaxis(field, 0, -1)
    dense_vox_zyx_xyz = dense_vox_xyz_xyz.transpose(2, 1, 0, 3)
    scales_xyz = np.asarray([(x - 1) / 2.0, (y - 1) / 2.0, (z - 1) / 2.0], dtype=np.float32)
    return np.ascontiguousarray(dense_vox_zyx_xyz / scales_xyz.reshape(1, 1, 1, 3), dtype=np.float32)


def dense_vox_zyx_for_regularity(field_cxyz: np.ndarray) -> np.ndarray:
    """Return vector components in Z,Y,X order for historical regularity helpers."""

    xyz = np.moveaxis(np.asarray(field_cxyz, dtype=np.float32), 0, -1).transpose(2, 1, 0, 3)
    return np.ascontiguousarray(xyz[..., [2, 1, 0]], dtype=np.float32)


def adapter_gates(case: PairCase) -> dict[str, Any]:
    if case.fixed_points_xyz is not None:
        points = case.fixed_points_xyz
    else:
        points = np.asarray([[0.0, 0.0, 0.0], [51.5, 71.25, 101.75], [223.0, 191.0, 223.0]])
    zero = np.zeros((3, *case.shape_xyz), dtype=np.float32)
    zero_samples = sample_field_trilinear_xyz(zero, points)
    known = np.broadcast_to(KNOWN_SHIFT_XYZ[:, None, None, None], zero.shape).copy()
    known_samples = sample_field_trilinear_xyz(known, points)
    if float(np.abs(zero_samples).max()) != 0.0 or not np.allclose(known_samples, KNOWN_SHIFT_XYZ, atol=1.0e-6):
        raise P2ePairlocalError("Zero/known-shift field adapter gate failed")
    world = affine_points(case.fixed_affine_xyz, points + known_samples)
    expected = affine_points(case.fixed_affine_xyz, points + KNOWN_SHIFT_XYZ[None, :])
    error = np.linalg.norm(world - expected, axis=1)
    if float(error.max()) > 1.0e-6:
        raise P2ePairlocalError("Known-shift physical-coordinate gate failed")
    return {
        "field_layout": "[3,X,Y,Z]",
        "component_order": "xyz",
        "units": "voxels",
        "semantics": "moving_sample_coordinate=fixed_coordinate+displacement",
        "zero_flow_max_abs_sample_vox": float(np.abs(zero_samples).max()),
        "known_shift_vox_xyz": KNOWN_SHIFT_XYZ,
        "known_shift_max_abs_sample_error_vox": float(np.abs(known_samples - KNOWN_SHIFT_XYZ).max()),
        "known_shift_max_tre_error_mm": float(error.max()),
    }


def interpolation_audit() -> dict[str, float]:
    """Quantify the documented historic-vs-evaluator boundary sampling difference."""

    import torch
    import torch.nn.functional as functional

    volume = torch.arange(7, dtype=torch.float32).reshape(1, 1, 1, 1, 7)
    coordinates = torch.linspace(-1.0, 1.0, 7, dtype=torch.float32)
    grid = torch.stack((coordinates, torch.zeros_like(coordinates), torch.zeros_like(coordinates)), dim=-1)
    grid = grid.reshape(1, 1, 1, 7, 3)
    historic = functional.grid_sample(volume, grid, align_corners=True, mode="bilinear")
    evaluator = functional.grid_sample(volume, grid, align_corners=False, mode="bilinear")
    return {
        "historical_align_corners": 1.0,
        "learn2breath_evaluator_align_corners": 0.0,
        "synthetic_max_abs_difference": float((historic - evaluator).abs().max()),
        "synthetic_center_abs_difference": float((historic[..., 3] - evaluator[..., 3]).abs().max()),
    }


def p2e_config(intensity_profile: str = "p2e_equiv_hu") -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "MR-D-BSCP_P2e_fullgrid_1p5mm",
        "ct_only": True,
        "intensity": {
            "input_units": "physical_HU",
            "historical_window_units": "raw_DIR-Lab_stored_value",
            "raw_to_hu_intercept": P2E_RAW_TO_HU_INTERCEPT,
            "low_stored": P2E_WINDOW_LOW_STORED,
            "high_stored": P2E_WINDOW_HIGH_STORED,
            "selected_profile": intensity_profile_config(intensity_profile),
            "rule": "clip_physical_HU_to_selected_profile_then_scale_to_[0,1]",
        },
        "stages": {
            "full_cps_voxels": list(STAGE_FULL_CPS),
            "image_downsample": list(STAGE_DOWNSAMPLE),
            "epochs": list(STAGE_EPOCHS),
            "lr": STAGE_LR,
            "seed": 1,
        },
        "objective": {
            "similarity": "pTV_local_CC",
            "lcc_sigma_mm": P2E_LCC_SIGMA_MM,
            "lcc_radius_sigma": P2E_LCC_RADIUS_SIGMA,
            "lcc_sigma_floor_pix": P2E_LCC_SIGMA_FLOOR_PIX,
            "border_voxels": P2E_BORDER_VOXELS,
            "regularizer": "physical_isoTV",
            "isotv_weight": P2E_ISOTV_WEIGHT,
            "csqrt": P2E_CSQRT,
            "optimizer": "Adam",
            "internal_interpolation": "grid_sample_align_corners_true",
            "prolongation": "exact_crop_aligned_cubic_bspline_dyadic_knot_insertion",
        },
    }


def preflight(
    case: PairCase,
    *,
    data_root: Path,
    r231_root: Path | None,
    case_id: str,
    learn2breath_repo: Path | None = None,
) -> dict[str, Any]:
    baseline: dict[str, float | int] | None = None
    if case.fixed_points_xyz is not None and case.moving_points_xyz is not None:
        fixed_world = affine_points(case.fixed_affine_xyz, case.fixed_points_xyz)
        moving_world = affine_points(case.moving_affine_xyz, case.moving_points_xyz)
        tre = np.linalg.norm(fixed_world - moving_world, axis=1)
        baseline = {
            "landmark_count": int(tre.size),
            "baseline_mean_tre_mm": float(tre.mean()),
            "baseline_median_tre_mm": float(np.median(tre)),
            "baseline_p95_tre_mm": float(np.percentile(tre, 95)),
        }
    r231 = validate_r231_pair(case, r231_root, case_id) if r231_root is not None else None
    evaluator = None
    lobe_qc = None
    if case.dataset == "validation":
        if learn2breath_repo is None:
            raise P2ePairlocalError("--learn2breath-repo is required for validation lobe preflight")
        evaluator = load_learn2breath_evaluator(learn2breath_repo)
        lobe_qc = validate_validation_lobes(case, data_root, case_id, evaluator)
    profile = intensity_profile_config(case.intensity_profile)
    low_hu = float(profile["low_hu"])
    high_hu = float(profile["high_hu"])
    return {
        "passed": True,
        "dataset": case.dataset,
        "case_label": case.case_label,
        "fixed_path": case.fixed_path,
        "moving_path": case.moving_path,
        "shape_xyz": list(case.shape_xyz),
        "shape_zyx_solver": list(case.shape_zyx),
        "spacing_xyz_mm": list(case.spacing_xyz_mm),
        "fixed_moving_affine_max_abs_difference": float(np.abs(case.fixed_affine_xyz - case.moving_affine_xyz).max()),
        "ct_hu": {
            "fixed_min": float(case.fixed_hu_xyz.min()),
            "fixed_p01": float(np.percentile(case.fixed_hu_xyz, 1)),
            "fixed_p99": float(np.percentile(case.fixed_hu_xyz, 99)),
            "moving_min": float(case.moving_hu_xyz.min()),
            "moving_p01": float(np.percentile(case.moving_hu_xyz, 1)),
            "moving_p99": float(np.percentile(case.moving_hu_xyz, 99)),
            "profile_low_hu": low_hu,
            "profile_high_hu": high_hu,
            "fixed_pct_le_profile_low": float((case.fixed_hu_xyz <= low_hu).mean() * 100.0),
            "moving_pct_le_profile_low": float((case.moving_hu_xyz <= low_hu).mean() * 100.0),
            "fixed_pct_gt_profile_high": float((case.fixed_hu_xyz > high_hu).mean() * 100.0),
            "moving_pct_gt_profile_high": float((case.moving_hu_xyz > high_hu).mean() * 100.0),
        },
        "intensity_provenance": intensity_provenance(case, data_root, case_id),
        "baseline": baseline,
        "r231_qc_only": r231,
        "validation_lobes_qc_only": lobe_qc,
        "adapter_gates": adapter_gates(case),
        "interpolation_audit": interpolation_audit(),
        "config": p2e_config(case.intensity_profile),
    }


def optimise(case: PairCase, *, device_name: str, log_every: int) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    import torch

    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise P2ePairlocalError("The P2e full-grid registration requires a CUDA device")
    device = torch.device(device_name)
    torch.manual_seed(1)
    np.random.seed(1)
    fixed = torch.from_numpy(case.fixed_zyx).to(device=device, dtype=torch.float32)
    moving = torch.from_numpy(case.moving_zyx).to(device=device, dtype=torch.float32)
    image_shape = tuple(int(value) for value in fixed.shape)
    border = p2e.make_border_mask(image_shape, P2E_BORDER_VOXELS, device, fixed.dtype)
    nested_shapes = p2e.build_nested_cp_shapes(image_shape, base_spacing=STAGE_FULL_CPS[0], spacings=STAGE_FULL_CPS)
    p2e.assert_nested_schedule(nested_shapes, STAGE_FULL_CPS)
    stage_rows: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    prev_control: torch.Tensor | None = None
    prev_shape: tuple[int, int, int] | None = None
    final_dense: torch.Tensor | None = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for stage_index, (full_cps, downsample, epochs) in enumerate(zip(STAGE_FULL_CPS, STAGE_DOWNSAMPLE, STAGE_EPOCHS)):
        fixed_level, blur_sigma, blur_kernel = p2e.make_loss_volume(fixed, downsample, 0.5, 3.0)
        moving_level, _, _ = p2e.make_loss_volume(moving, downsample, 0.5, 3.0)
        level_shape = tuple(int(value) for value in fixed_level.shape)
        stage_spacing_zyx = p2e.stage_pix_zyx(case.spacing_zyx_mm, image_shape, level_shape)
        level_border = p2e.make_loss_mask(border, downsample, 0.5, 3.0)
        if level_border is None:
            raise AssertionError("The frozen P2e border mask is required")
        lcc_cache = p2e.make_ptv_lcc_fixed_cache(
            fixed_level,
            stage_spacing_zyx,
            P2E_LCC_SIGMA_MM,
            P2E_LCC_RADIUS_SIGMA,
            P2E_LCC_SIGMA_FLOOR_PIX,
        )
        control_shape = tuple(int(value) for value in nested_shapes[full_cps])
        control = torch.nn.Parameter(torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32))
        if prev_control is not None and prev_shape is not None:
            coarse_grid = p2e.flat_to_grid(prev_control, prev_shape)
            refined_grid = p2e.cubic_bspline_dyadic_knot_insert_3d(coarse_grid, control_shape)
            with torch.no_grad():
                control.copy_(p2e.grid_to_flat(refined_grid))
        domain_cps = p2e.domain_cps_from_full(full_cps, downsample)
        identity = p2e.make_coordinate_grid(level_shape, device).detach()
        optimiser = torch.optim.Adam([control], lr=STAGE_LR)
        stage_start = time.perf_counter()
        print(
            json.dumps(
                {
                    "stage": stage_index,
                    "full_cps_vox": full_cps,
                    "downsample": downsample,
                    "domain_cps": domain_cps,
                    "control_shape_zyx": control_shape,
                    "level_shape_zyx": level_shape,
                    "stage_spacing_zyx_mm": stage_spacing_zyx,
                    "epochs": epochs,
                }
            ),
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            dense = p2e.expand_on_shape(control, control_shape, full_cps, level_shape, downsample)
            warped = p2e.warp_full(moving_level, identity + dense, mode="bilinear")
            data_loss = p2e.ptv_local_cc_loss_with_cache(
                warped,
                level_border,
                lcc_cache,
                stage_spacing_zyx,
                P2E_LCC_RADIUS_SIGMA,
            )
            regularity = p2e.isotv_control_loss(control, control_shape, level_shape, stage_spacing_zyx, domain_cps, P2E_CSQRT)
            loss = data_loss + P2E_ISOTV_WEIGHT * regularity
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            row = {
                "stage_index": stage_index,
                "stage_epoch": epoch,
                "global_epoch": sum(STAGE_EPOCHS[:stage_index]) + epoch,
                "full_cps_vox": full_cps,
                "downsample": downsample,
                "domain_cps": domain_cps,
                "loss": float(loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "isotv_loss": float(regularity.detach().cpu()),
                "isotv_weight": P2E_ISOTV_WEIGHT,
                "lr": STAGE_LR,
            }
            stage_rows.append(row)
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(json.dumps(row), flush=True)
        with torch.no_grad():
            final_dense = p2e.expand_on_shape(control, control_shape, full_cps, image_shape, 1)
        prev_control = control.detach()
        prev_shape = control_shape
        stage_summaries.append(
            {
                "stage_index": stage_index,
                "full_cps_vox": full_cps,
                "downsample": downsample,
                "domain_cps": domain_cps,
                "control_shape_zyx": list(control_shape),
                "level_shape_zyx": list(level_shape),
                "stage_spacing_zyx_mm": list(stage_spacing_zyx),
                "epochs": epochs,
                "elapsed_sec": time.perf_counter() - stage_start,
                "antialias_sigma": blur_sigma,
                "antialias_kernel_size": blur_kernel,
            }
        )
    if final_dense is None:
        raise AssertionError("No P2e stage produced a dense field")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    field = normalized_zyx_to_field_cxyz(final_dense.detach().cpu().numpy())
    metadata = {
        "training": {
            **p2e_config(case.intensity_profile)["stages"],
            **p2e_config(case.intensity_profile)["objective"],
            "nested_control_shapes_zyx": {str(key): list(value) for key, value in nested_shapes.items()},
            "loss_domain": "full_pairlocal_grid_with_5_voxel_border_mask_no_anatomical_mask",
            "landmarks_used_in_optimisation": False,
            "r231_used_in_optimisation": False,
            "labels_used_in_optimisation": False,
        },
        "stage_summaries": stage_summaries,
        "elapsed_sec": elapsed,
        "peak_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
    }
    return field, metadata, stage_rows


def write_field(case: PairCase, field_cxyz: np.ndarray, path: Path) -> None:
    nib = import_nibabel()
    field = np.asarray(field_cxyz, dtype=np.float32)
    if field.shape != (3, *case.shape_xyz) or not np.isfinite(field).all():
        raise P2ePairlocalError(f"Invalid standard field: {field.shape}")
    header = case.fixed_header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(field, case.fixed_affine_xyz, header=header)
    image.set_qform(case.fixed_affine_xyz, code=1)
    image.set_sform(case.fixed_affine_xyz, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    reloaded = nib.load(str(path))
    if tuple(reloaded.shape) != tuple(field.shape) or reloaded.get_data_dtype() != np.dtype(np.float32):
        raise P2ePairlocalError("Reloaded output field layout or dtype differs")
    if not np.allclose(reloaded.affine, case.fixed_affine_xyz, rtol=0.0, atol=1.0e-6):
        raise P2ePairlocalError("Reloaded output field affine differs from fixed CT")


def tre_metrics(case: PairCase, field_cxyz: np.ndarray, csv_path: Path) -> dict[str, Any] | None:
    if case.fixed_points_xyz is None or case.moving_points_xyz is None or case.landmark_ids is None:
        return None
    sampled = sample_field_trilinear_xyz(field_cxyz, case.fixed_points_xyz)
    predicted = case.fixed_points_xyz + sampled
    predicted_world = affine_points(case.fixed_affine_xyz, predicted)
    moving_world = affine_points(case.moving_affine_xyz, case.moving_points_xyz)
    tre = np.linalg.norm(predicted_world - moving_world, axis=1)
    baseline = np.linalg.norm(
        affine_points(case.fixed_affine_xyz, case.fixed_points_xyz) - moving_world,
        axis=1,
    )
    rows = []
    for index, landmark_id in enumerate(case.landmark_ids):
        rows.append(
            {
                "landmark_id": int(landmark_id),
                "fixed_x_voxel": float(case.fixed_points_xyz[index, 0]),
                "fixed_y_voxel": float(case.fixed_points_xyz[index, 1]),
                "fixed_z_voxel": float(case.fixed_points_xyz[index, 2]),
                "moving_x_voxel": float(case.moving_points_xyz[index, 0]),
                "moving_y_voxel": float(case.moving_points_xyz[index, 1]),
                "moving_z_voxel": float(case.moving_points_xyz[index, 2]),
                "disp_x_voxel": float(sampled[index, 0]),
                "disp_y_voxel": float(sampled[index, 1]),
                "disp_z_voxel": float(sampled[index, 2]),
                "baseline_tre_mm": float(baseline[index]),
                "registered_tre_mm": float(tre[index]),
            }
        )
    write_csv(csv_path, rows)
    return {
        "landmark_count": int(tre.size),
        "baseline_mean_tre_mm": float(baseline.mean()),
        "registered_mean_tre_mm": float(tre.mean()),
        "registered_std_tre_mm": float(tre.std()),
        "registered_median_tre_mm": float(np.median(tre)),
        "registered_p95_tre_mm": float(np.percentile(tre, 95)),
        "registered_max_tre_mm": float(tre.max()),
        "sampling": "trilinear_at_fixed_landmark_then_fixed_plus_u",
    }


def challenge_flow_xyzc(field_cxyz: np.ndarray) -> np.ndarray:
    field = np.asarray(field_cxyz, dtype=np.float32)
    if field.ndim != 4 or field.shape[0] != 3:
        raise P2ePairlocalError(f"Expected field [3,X,Y,Z], got {field.shape}")
    return np.ascontiguousarray(np.moveaxis(field, 0, -1), dtype=np.float32)


def field_quality(field_cxyz: np.ndarray, *, challenge_evaluator=None, fixed_lobe_mask_xyz: np.ndarray | None = None) -> dict[str, Any]:
    regularity = p2e.displacement_regularities(dense_vox_zyx_for_regularity(field_cxyz))
    field = np.asarray(field_cxyz, dtype=np.float32)
    magnitude = np.linalg.norm(field, axis=0)
    result = {
        "finite": bool(np.isfinite(field).all()),
        "component_min_vox": [float(value) for value in field.reshape(3, -1).min(axis=1)],
        "component_max_vox": [float(value) for value in field.reshape(3, -1).max(axis=1)],
        "magnitude_mean_vox": float(magnitude.mean()),
        "magnitude_p95_vox": float(np.percentile(magnitude, 95)),
        "magnitude_max_vox": float(magnitude.max()),
        "folding_percent_whole_grid": float(regularity["jdet_le0_pct_full"]),
        "jacobian_mean_whole_grid": float(regularity["jdet_mean_full"]),
        "log_jacobian_std_whole_grid": float(regularity["log_jdet_std_full"]),
        "regularity": regularity,
    }
    if challenge_evaluator is not None:
        flow = challenge_flow_xyzc(field)
        result["challenge_folding_percent_whole_grid"] = float(challenge_evaluator.folding_percent(flow))
        if fixed_lobe_mask_xyz is not None:
            result["challenge_folding_percent_fixed_lobes"] = float(
                challenge_evaluator.folding_percent(flow, np.asarray(fixed_lobe_mask_xyz) > 0)
            )
    return result


def validation_evaluate(dvf_path: Path, data_root: Path, case_id: str, evaluator, device: str) -> dict[str, Any]:
    metrics = evaluator.evaluate_case(dvf_path, data_root, int(case_id), device=device)
    metrics["labels_used_in_optimisation"] = False
    metrics["labels_used_for_evaluation_only"] = True
    return metrics


def run_registration(args: argparse.Namespace) -> dict[str, Any]:
    full_case_start = time.perf_counter()
    if args.learn2breath_repo is None:
        raise P2ePairlocalError("--learn2breath-repo is required to compute challenge-style folding")
    case = load_pair(args.dataset, args.data_root, args.case_id, args.intensity_profile)
    r231_root = args.r231_root if args.dataset == "copd" and args.r231_root is not None else None
    gate = preflight(
        case,
        data_root=args.data_root,
        r231_root=r231_root,
        case_id=args.case_id,
        learn2breath_repo=args.learn2breath_repo,
    )
    out_root = args.out_root
    if (out_root / "summary.json").exists() and not args.force:
        raise P2ePairlocalError(f"Refusing to overwrite completed run: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "preflight.json", gate)
    write_json(out_root / "config.json", p2e_config(case.intensity_profile))
    field, train_meta, history = optimise(case, device_name=args.device, log_every=args.log_every)
    field_path = out_root / "fields" / f"{case.case_label}_DVF.nii.gz"
    write_field(case, field, field_path)
    np.save(out_root / "fields" / f"{case.case_label}_field_cxyz_vox.npy", field, allow_pickle=False)
    write_csv(out_root / "training" / "stage_history.csv", history)
    tre = tre_metrics(case, field, out_root / "results" / "per_landmark_tre.csv")
    evaluator = load_learn2breath_evaluator(args.learn2breath_repo)
    validation = None
    fixed_lobe_mask = None
    if args.dataset == "validation":
        validation = validation_evaluate(field_path, args.data_root, args.case_id, evaluator, args.device)
        write_json(out_root / "results" / "validation_metrics.json", validation)
        lobe_paths = evaluator.validation_paths(args.data_root, int(args.case_id))
        nib = import_nibabel()
        fixed_lobe_mask = np.asarray(nib.load(str(lobe_paths["insp_lobe"])).get_fdata(dtype=np.float32), dtype=np.float32) > 0
    quality = field_quality(field, challenge_evaluator=evaluator, fixed_lobe_mask_xyz=fixed_lobe_mask)
    full_case_elapsed = time.perf_counter() - full_case_start
    summary = {
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "dataset": args.dataset,
        "case_label": case.case_label,
        "intensity_profile": intensity_profile_config(case.intensity_profile),
        "fixed_role": "INSP",
        "moving_role": "EXP",
        "input_shape_xyz": list(case.shape_xyz),
        "spacing_xyz_mm": list(case.spacing_xyz_mm),
        "field_path": field_path,
        "field_contract": adapter_gates(case),
        "preflight_path": out_root / "preflight.json",
        "training": train_meta,
        "timing": {
            "optimizer_elapsed_sec": float(train_meta["elapsed_sec"]),
            "full_case_elapsed_sec": float(full_case_elapsed),
            "definition": {
                "optimizer_elapsed_sec": "Synchronized P2e optimizer and image-pyramid work only.",
                "full_case_elapsed_sec": "Pair load through preflight, normalization, optimizer, DVF export/reload, evaluation, regularity, and result writes; excludes worker queue time.",
            },
        },
        "tre": tre,
        "field_quality": quality,
        "validation_evaluation": validation,
        "notes": [
            "CT-only optimisation: landmarks, R231 masks, and lobe labels were excluded from the loss.",
            f"Intensity profile: {case.intensity_profile} ({intensity_profile_config(case.intensity_profile)['description']})",
            "Challenge-style folding is evaluated with the unchanged Learn2Breath evaluator.",
            "This is a full-grid 1.5-mm P2e adaptation, not a direct numerical comparison with historic 1-mm-crop P2e TRE.",
        ],
    }
    write_json(out_root / "summary.json", summary)
    return summary


def warp_xyz_with_historical_p2e(source_xyz: np.ndarray, field_cxyz: np.ndarray, *, mode: str) -> np.ndarray:
    """Warp a XYZ volume with the historic `align_corners=True` P2e convention."""

    import torch

    source_zyx = xyz_to_zyx(np.asarray(source_xyz, dtype=np.float32))
    dense_norm = field_cxyz_to_normalized_zyx(field_cxyz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        source = torch.from_numpy(source_zyx).to(device)
        displacement = torch.from_numpy(dense_norm).to(device)
        identity = p2e.make_coordinate_grid(source_zyx.shape, device)
        warped = p2e.warp_full(source, identity + displacement, mode=mode)
    return zyx_to_xyz(warped.detach().cpu().numpy())


def _coronal_radiology_view(array_xyz: np.ndarray, y_index: int) -> np.ndarray:
    """Match the prior mask audit: S up, patient R on screen left."""

    return np.fliplr(np.asarray(array_xyz)[:, int(y_index), :].T)


def _draw_coronal_orientation_markers(axis) -> None:
    style = {
        "transform": axis.transAxes,
        "color": "white",
        "fontsize": 8,
        "fontweight": "bold",
        "bbox": {"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    }
    axis.text(0.50, 0.98, "S", ha="center", va="top", **style)
    axis.text(0.50, 0.02, "I", ha="center", va="bottom", **style)
    axis.text(0.02, 0.50, "R", ha="left", va="center", **style)
    axis.text(0.98, 0.50, "L", ha="right", va="center", **style)


def render_registration_audit(
    *,
    pairlocal_root: Path,
    r231_root: Path,
    field_path: Path,
    case_id: str = "111",
):
    """Render the requested inline 2x9 coronal CT/R231 review figure."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    case = load_pair("copd", pairlocal_root, case_id)
    nib = import_nibabel()
    fixed_mask_path, moving_mask_path = r231_paths(r231_root, case_id)
    validate_r231_pair(case, r231_root, case_id)
    field_img = nib.load(str(field_path))
    field = np.asarray(field_img.get_fdata(dtype=np.float32), dtype=np.float32)
    if field.shape != (3, *case.shape_xyz):
        raise P2ePairlocalError(f"Unexpected field for audit: {field.shape}")
    moving_mask = (np.asarray(nib.load(str(moving_mask_path)).get_fdata(), dtype=np.float32) > 0).astype(np.float32)
    fixed_mask = (np.asarray(nib.load(str(fixed_mask_path)).get_fdata(), dtype=np.float32) > 0).astype(np.float32)
    warped_ct = warp_xyz_with_historical_p2e(case.moving_hu_xyz, field, mode="bilinear")
    warped_mask = warp_xyz_with_historical_p2e(moving_mask, field, mode="nearest") > 0.5
    occupied_y = np.flatnonzero(fixed_mask.any(axis=(0, 2)))
    if occupied_y.size == 0:
        raise P2ePairlocalError("Fixed R231 mask is empty for coronal-position selection")
    posterior, anterior = int(occupied_y.min()), int(occupied_y.max())
    positions = [("25%", int(round(posterior + 0.25 * (anterior - posterior)))), ("50% / central", int(round(posterior + 0.50 * (anterior - posterior)))), ("75%", int(round(posterior + 0.75 * (anterior - posterior))))]
    figure, axes = plt.subplots(2, 9, figsize=(27, 6.5), layout="constrained")
    figure.suptitle(
        f"COPD {case.case_label}: P2e coronal registration audit (fixed R231 Y={posterior}..{anterior})",
        fontsize=15,
        fontweight="bold",
    )
    ct_panels = (("Moving EXP", case.moving_hu_xyz), ("Warped moving EXP", warped_ct), ("Fixed INSP", case.fixed_hu_xyz))
    mask_panels = (("Moving R231", moving_mask), ("Warped moving R231", warped_mask), ("Fixed R231", fixed_mask))
    for slice_index, (slice_label, y_index) in enumerate(positions):
        for state_index, (name, volume) in enumerate(ct_panels):
            axis = axes[0, 3 * slice_index + state_index]
            axis.imshow(
                _coronal_radiology_view(volume, y_index),
                cmap="gray",
                vmin=-1000,
                vmax=300,
                origin="lower",
                interpolation="nearest",
            )
            axis.set_title(f"{slice_label}: {name}\nY={y_index}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            _draw_coronal_orientation_markers(axis)
        for state_index, (name, volume) in enumerate(mask_panels):
            axis = axes[1, 3 * slice_index + state_index]
            axis.imshow(
                _coronal_radiology_view(volume, y_index),
                cmap=ListedColormap(("black", "lime")),
                vmin=0,
                vmax=1,
                origin="lower",
                interpolation="nearest",
            )
            axis.set_title(f"{slice_label}: {name}\nY={y_index}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            _draw_coronal_orientation_markers(axis)
    axes[0, 0].set_ylabel("CT (HU window −1000..300)", fontsize=10)
    axes[1, 0].set_ylabel("R231 masks (QC only)", fontsize=10)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", choices=("copd", "validation"), required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--case-id", required=True)
        command.add_argument("--intensity-profile", choices=tuple(INTENSITY_PROFILES), required=True)
        command.add_argument("--out-root", type=Path, required=True)
        command.add_argument("--r231-root", type=Path)
        command.add_argument("--learn2breath-repo", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--log-every", type=int, default=100)
        command.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = load_pair(args.dataset, args.data_root, args.case_id, args.intensity_profile)
    r231_root = args.r231_root if args.dataset == "copd" and args.r231_root is not None else None
    if args.command == "preflight":
        args.out_root.mkdir(parents=True, exist_ok=True)
        report = preflight(
            case,
            data_root=args.data_root,
            r231_root=r231_root,
            case_id=args.case_id,
            learn2breath_repo=args.learn2breath_repo,
        )
        write_json(args.out_root / "preflight.json", report)
        print(json.dumps(json_ready(report), indent=2, sort_keys=True), flush=True)
        return 0
    summary = run_registration(args)
    print(
        json.dumps(
            json_ready(
                {
                    "summary_path": args.out_root / "summary.json",
                    "tre": summary["tre"],
                    "field_quality": summary["field_quality"],
                    "timing": summary["timing"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
