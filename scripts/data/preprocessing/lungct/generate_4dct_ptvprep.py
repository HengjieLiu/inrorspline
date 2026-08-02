#!/usr/bin/env python
"""Generate canonical 4DCT pTVReg-crop 1 mm preprocessing artifacts.

This uses pTVReg's 4DCT fixed/moving, intensity, crop-index, and target-shape
conventions, while intentionally using PyTorch trilinear image resampling.
Lung masks are transformed through the same corrected crop and nearest-neighbor
resampling for QA and optional mask-aware variants; official pTVReg does not use
lung masks for optimization.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common.lungct import (  # noqa: E402
    landmark_mask_fraction,
    load_landmarks_xyz,
    load_raw_4dct_image,
    mask_root_case_dir,
    resolve_4dct_case,
)
from common.volume import mask_stats, parse_case_ids, write_json  # noqa: E402


PROTOCOL_NAME = "ptvreg_landmark_crop_1mm_trilinear_v1"
MARGIN_XYZ = np.asarray([10, 10, 5], dtype=np.int64)
MARGIN_ZYX = MARGIN_XYZ[[2, 1, 0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-ids", type=str, default="1-10")
    parser.add_argument("--modelname", type=str, default="R231")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def img_thr(vol: np.ndarray, min_val: float = 80.0, max_val: float = 900.0, norm: float = 1.0) -> np.ndarray:
    out = np.clip(vol.astype(np.float32), min_val, max_val)
    out = out - min_val
    out = out * (norm / (max_val - min_val))
    return out.astype(np.float32, copy=False)


def landmarks_xyz_one_based_to_zyx_zero_based(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Convert DIR-Lab/pTVReg landmark file coordinates to NumPy zyx indices."""
    return landmarks_xyz[:, [2, 1, 0]].astype(np.float64, copy=False) - 1.0


def ptvreg_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    """Return pTVReg crop bounds as Python 0-based inclusive zyx bounds.

    pTVReg computes crop bounds in MATLAB's 1-based xyz array convention:
    crop_v = [max(1, min_xyz - [10,10,5]), min(shape_xyz, max_xyz + [10,10,5])].
    This function converts those inclusive bounds to NumPy's 0-based zyx order.
    """
    both = np.concatenate([fixed_xyz, moving_xyz], axis=0)
    lo1_xyz = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi1_xyz = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape_xyz = np.asarray(tuple(shape_zyx)[::-1], dtype=np.int64)
    start1_xyz = np.maximum(1, lo1_xyz - MARGIN_XYZ)
    end1_xyz = np.minimum(shape_xyz, hi1_xyz + MARGIN_XYZ)
    start0_zyx = start1_xyz[[2, 1, 0]] - 1
    end0_zyx = end1_xyz[[2, 1, 0]] - 1
    return np.stack([start0_zyx, end0_zyx], axis=1).astype(np.int64)


def task4_legacy_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    """Return the legacy task4 Python crop bounds for audit comparison only."""
    fixed_zyx = fixed_xyz[:, [2, 1, 0]].astype(np.float64, copy=False)
    moving_zyx = moving_xyz[:, [2, 1, 0]].astype(np.float64, copy=False)
    both = np.concatenate([fixed_zyx, moving_zyx], axis=0)
    lo = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    start = np.maximum(0, lo - MARGIN_ZYX)
    end = np.minimum(shape - 1, hi + MARGIN_ZYX)
    return np.stack([start, end], axis=1).astype(np.int64)


def crop_zyx(vol: np.ndarray, bounds_zyx: np.ndarray) -> np.ndarray:
    return vol[
        bounds_zyx[0, 0] : bounds_zyx[0, 1] + 1,
        bounds_zyx[1, 0] : bounds_zyx[1, 1] + 1,
        bounds_zyx[2, 0] : bounds_zyx[2, 1] + 1,
    ]


def resize_image_trilinear_zyx(vol: np.ndarray, new_shape_zyx: Sequence[int], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        ten = torch.from_numpy(vol.astype(np.float32, copy=False))[None, None].to(device=device)
        out = F.interpolate(ten, size=tuple(int(v) for v in new_shape_zyx), mode="trilinear", align_corners=True)
        return out[0, 0].cpu().numpy().astype(np.float32, copy=False)


def resize_mask_nearest_zyx(mask: np.ndarray, new_shape_zyx: Sequence[int], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        ten = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None].to(device=device)
        out = F.interpolate(ten, size=tuple(int(v) for v in new_shape_zyx), mode="nearest")
        return (out[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)


def transform_landmarks_native0_to_crop(landmarks_native0_zyx: np.ndarray, bounds_zyx: np.ndarray) -> np.ndarray:
    return landmarks_native0_zyx - bounds_zyx[:, 0][None, :].astype(np.float64)


def transform_landmarks_crop_to_registration(
    landmarks_crop_zyx: np.ndarray,
    crop_shape_zyx: Sequence[int],
    reg_shape_zyx: Sequence[int],
) -> np.ndarray:
    crop_max = np.maximum(np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0, 1.0)
    reg_max = np.maximum(np.asarray(reg_shape_zyx, dtype=np.float64) - 1.0, 1.0)
    return landmarks_crop_zyx * (reg_max[None, :] / crop_max[None, :])


def points_inside_shape(points_zyx: np.ndarray, shape_zyx: Sequence[int]) -> Dict[str, object]:
    pts = np.asarray(points_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    inside = np.all((pts >= 0.0) & (pts <= (shape[None, :] - 1.0)), axis=1)
    return {
        "all_inside": bool(np.all(inside)),
        "inside_count": int(np.sum(inside)),
        "total_count": int(inside.size),
        "inside_fraction": float(np.mean(inside)) if inside.size else float("nan"),
        "min_zyx": [float(v) for v in np.min(pts, axis=0)],
        "max_zyx": [float(v) for v in np.max(pts, axis=0)],
        "shape_zyx": [int(v) for v in shape_zyx],
    }


def load_binary_mask(mask_root: Path, case_id: int, phase: str, modelname: str) -> np.ndarray:
    path = mask_root_case_dir(mask_root, case_id) / f"case{case_id:02d}_T{phase}_{modelname}_binary_zyx.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.uint8, copy=False)


def write_array(path: Path, arr: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_case_arrays(case_dir: Path, arrays: Dict[str, np.ndarray], overwrite: bool) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for name, arr in arrays.items():
        path = case_dir / f"{name}.npy"
        write_array(path, arr, overwrite)
        paths[name] = str(path)
    return paths


def audit_landmark_mask_fraction(mask: np.ndarray, landmarks_zyx: np.ndarray) -> Dict[str, object]:
    return {
        "fraction": landmark_mask_fraction(mask, landmarks_zyx),
        "landmarks": points_inside_shape(landmarks_zyx, mask.shape),
    }


def process_case(case_id: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    case = resolve_4dct_case(case_id, args.data_root)
    if case.status != "complete":
        raise RuntimeError(f"4DCT case {case_id} is {case.status}; notes={case.notes}")

    fixed_xyz = load_landmarks_xyz(case.landmarks_t00)
    moving_xyz = load_landmarks_xyz(case.landmarks_t50)
    fixed_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(fixed_xyz)
    moving_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(moving_xyz)

    crop_bounds = ptvreg_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, case.shape_zyx)
    legacy_bounds = task4_legacy_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, case.shape_zyx)
    crop_shape = tuple((crop_bounds[:, 1] - crop_bounds[:, 0] + 1).astype(int).tolist())
    spacing = tuple(float(v) for v in case.spacing_zyx)
    reg_shape = tuple(np.rint(np.asarray(crop_shape, dtype=np.float64) * np.asarray(spacing)).astype(int).tolist())

    fixed_raw = load_raw_4dct_image(case.image_t00, case.shape_zyx)
    moving_raw = load_raw_4dct_image(case.image_t50, case.shape_zyx)
    fixed_native = img_thr(fixed_raw)
    moving_native = img_thr(moving_raw)
    fixed_crop = crop_zyx(fixed_native, crop_bounds)
    moving_crop = crop_zyx(moving_native, crop_bounds)
    fixed_reg = resize_image_trilinear_zyx(fixed_crop, reg_shape, device)
    moving_reg = resize_image_trilinear_zyx(moving_crop, reg_shape, device)

    fixed_mask_native = load_binary_mask(args.mask_root, case_id, "00", args.modelname)
    moving_mask_native = load_binary_mask(args.mask_root, case_id, "50", args.modelname)
    if fixed_mask_native.shape != tuple(case.shape_zyx):
        raise ValueError(f"Fixed mask shape {fixed_mask_native.shape} != image shape {case.shape_zyx}")
    if moving_mask_native.shape != tuple(case.shape_zyx):
        raise ValueError(f"Moving mask shape {moving_mask_native.shape} != image shape {case.shape_zyx}")
    fixed_mask_crop = crop_zyx(fixed_mask_native, crop_bounds).astype(np.uint8)
    moving_mask_crop = crop_zyx(moving_mask_native, crop_bounds).astype(np.uint8)
    fixed_mask_reg = resize_mask_nearest_zyx(fixed_mask_crop, reg_shape, device)
    moving_mask_reg = resize_mask_nearest_zyx(moving_mask_crop, reg_shape, device)

    fixed_crop_zyx = transform_landmarks_native0_to_crop(fixed_native0_zyx, crop_bounds)
    moving_crop_zyx = transform_landmarks_native0_to_crop(moving_native0_zyx, crop_bounds)
    fixed_reg_zyx = transform_landmarks_crop_to_registration(fixed_crop_zyx, crop_shape, reg_shape)
    moving_reg_zyx = transform_landmarks_crop_to_registration(moving_crop_zyx, crop_shape, reg_shape)

    case_dir = args.out_root / "4DCT" / f"case{case_id:02d}"
    paths = save_case_arrays(
        case_dir,
        {
            "fixed_T00_native_ptvintensity_zyx": fixed_native,
            "moving_T50_native_ptvintensity_zyx": moving_native,
            "fixed_T00_crop_ptvintensity_zyx": fixed_crop,
            "moving_T50_crop_ptvintensity_zyx": moving_crop,
            "fixed_T00_reg1mm_trilinear_zyx": fixed_reg,
            "moving_T50_reg1mm_trilinear_zyx": moving_reg,
            "fixed_T00_lungmask_native_zyx": fixed_mask_native.astype(np.uint8),
            "moving_T50_lungmask_native_zyx": moving_mask_native.astype(np.uint8),
            "fixed_T00_lungmask_crop_zyx": fixed_mask_crop,
            "moving_T50_lungmask_crop_zyx": moving_mask_crop,
            "fixed_T00_lungmask_reg1mm_nearest_zyx": fixed_mask_reg,
            "moving_T50_lungmask_reg1mm_nearest_zyx": moving_mask_reg,
            "landmarks_fixed_T00_native0_zyx": fixed_native0_zyx,
            "landmarks_moving_T50_native0_zyx": moving_native0_zyx,
            "landmarks_fixed_T00_crop_zyx": fixed_crop_zyx,
            "landmarks_moving_T50_crop_zyx": moving_crop_zyx,
            "landmarks_fixed_T00_reg1mm_zyx": fixed_reg_zyx,
            "landmarks_moving_T50_reg1mm_zyx": moving_reg_zyx,
        },
        args.overwrite,
    )

    verification = {
        "fixed_crop_landmarks_inside": points_inside_shape(fixed_crop_zyx, crop_shape),
        "moving_crop_landmarks_inside": points_inside_shape(moving_crop_zyx, crop_shape),
        "fixed_reg_landmarks_inside": points_inside_shape(fixed_reg_zyx, reg_shape),
        "moving_reg_landmarks_inside": points_inside_shape(moving_reg_zyx, reg_shape),
        "fixed_native_landmark_mask_fraction": audit_landmark_mask_fraction(fixed_mask_native, fixed_native0_zyx),
        "moving_native_landmark_mask_fraction": audit_landmark_mask_fraction(moving_mask_native, moving_native0_zyx),
        "fixed_crop_landmark_mask_fraction": audit_landmark_mask_fraction(fixed_mask_crop, fixed_crop_zyx),
        "moving_crop_landmark_mask_fraction": audit_landmark_mask_fraction(moving_mask_crop, moving_crop_zyx),
        "fixed_reg_landmark_mask_fraction": audit_landmark_mask_fraction(fixed_mask_reg, fixed_reg_zyx),
        "moving_reg_landmark_mask_fraction": audit_landmark_mask_fraction(moving_mask_reg, moving_reg_zyx),
    }
    verification["pass"] = bool(
        verification["fixed_crop_landmarks_inside"]["all_inside"]
        and verification["moving_crop_landmarks_inside"]["all_inside"]
        and verification["fixed_reg_landmarks_inside"]["all_inside"]
        and verification["moving_reg_landmarks_inside"]["all_inside"]
        and fixed_reg.shape == fixed_mask_reg.shape
        and moving_reg.shape == moving_mask_reg.shape
        and fixed_crop.shape == fixed_mask_crop.shape
        and moving_crop.shape == moving_mask_crop.shape
    )

    row: Dict[str, object] = {
        "protocol_name": PROTOCOL_NAME,
        "dataset": "4dct",
        "case_id": case_id,
        "fixed_phase": "T00",
        "moving_phase": "T50",
        "raw_array_order": "zyx",
        "ptvreg_source_array_order": "xyz in MATLAB, represented as zyx in NumPy",
        "landmark_source_order": "xyz, pTVReg/MATLAB 1-based voxel coordinates",
        "landmark_saved_order": "zyx, Python 0-based voxel coordinates",
        "image_resampling": "PyTorch trilinear, align_corners=True",
        "mask_resampling": "corrected pTVReg crop, PyTorch nearest-neighbor",
        "intensity_rule": "clip [80, 900], subtract 80, divide by 820",
        "crop_rule": "pTVReg landmark bbox margin xyz [10,10,5], converted from MATLAB 1-based xyz to Python 0-based zyx",
        "crop_margin_xyz": [int(v) for v in MARGIN_XYZ],
        "crop_margin_zyx": [int(v) for v in MARGIN_ZYX],
        "raw_shape_zyx": [int(v) for v in case.shape_zyx],
        "native_spacing_zyx_mm": [float(v) for v in case.spacing_zyx],
        "crop_bounds_zyx_inclusive": crop_bounds.tolist(),
        "legacy_task4_crop_bounds_zyx_inclusive": legacy_bounds.tolist(),
        "legacy_minus_ptvreg_bounds_zyx": (legacy_bounds - crop_bounds).tolist(),
        "crop_shape_zyx": [int(v) for v in crop_shape],
        "registration_shape_zyx": [int(v) for v in reg_shape],
        "registration_spacing_zyx_mm": [1.0, 1.0, 1.0],
        "image_paths": {
            "fixed_t00": case.image_t00,
            "moving_t50": case.image_t50,
        },
        "landmark_paths": {
            "fixed_t00": case.landmarks_t00,
            "moving_t50": case.landmarks_t50,
        },
        "array_paths": paths,
        "mask_stats": {
            "fixed_native": mask_stats(fixed_mask_native),
            "moving_native": mask_stats(moving_mask_native),
            "fixed_crop": mask_stats(fixed_mask_crop),
            "moving_crop": mask_stats(moving_mask_crop),
            "fixed_reg": mask_stats(fixed_mask_reg),
            "moving_reg": mask_stats(moving_mask_reg),
        },
        "verification": verification,
    }
    write_json(case_dir / "preprocess_audit.json", row)
    print(
        f"case{case_id:02d}: crop={crop_shape} reg={reg_shape} "
        f"legacy_minus_ptvreg={row['legacy_minus_ptvreg_bounds_zyx']} pass={verification['pass']}",
        flush=True,
    )
    return row


def flatten_summary_row(row: Dict[str, object]) -> Dict[str, object]:
    verification = row["verification"]
    assert isinstance(verification, dict)
    mask_stats_payload = row["mask_stats"]
    assert isinstance(mask_stats_payload, dict)
    return {
        "case_id": row["case_id"],
        "protocol_name": row["protocol_name"],
        "raw_shape_zyx": row["raw_shape_zyx"],
        "native_spacing_zyx_mm": row["native_spacing_zyx_mm"],
        "crop_bounds_zyx_inclusive": row["crop_bounds_zyx_inclusive"],
        "legacy_minus_ptvreg_bounds_zyx": row["legacy_minus_ptvreg_bounds_zyx"],
        "crop_shape_zyx": row["crop_shape_zyx"],
        "registration_shape_zyx": row["registration_shape_zyx"],
        "verification_pass": verification["pass"],
        "fixed_crop_landmarks_inside_fraction": verification["fixed_crop_landmarks_inside"]["inside_fraction"],
        "moving_crop_landmarks_inside_fraction": verification["moving_crop_landmarks_inside"]["inside_fraction"],
        "fixed_reg_landmarks_inside_fraction": verification["fixed_reg_landmarks_inside"]["inside_fraction"],
        "moving_reg_landmarks_inside_fraction": verification["moving_reg_landmarks_inside"]["inside_fraction"],
        "fixed_native_landmark_mask_fraction": verification["fixed_native_landmark_mask_fraction"]["fraction"],
        "moving_native_landmark_mask_fraction": verification["moving_native_landmark_mask_fraction"]["fraction"],
        "fixed_crop_landmark_mask_fraction": verification["fixed_crop_landmark_mask_fraction"]["fraction"],
        "moving_crop_landmark_mask_fraction": verification["moving_crop_landmark_mask_fraction"]["fraction"],
        "fixed_reg_landmark_mask_fraction": verification["fixed_reg_landmark_mask_fraction"]["fraction"],
        "moving_reg_landmark_mask_fraction": verification["moving_reg_landmark_mask_fraction"]["fraction"],
        "fixed_reg_mask_voxels": mask_stats_payload["fixed_reg"]["voxels"],
        "moving_reg_mask_voxels": mask_stats_payload["moving_reg"]["voxels"],
    }


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_summary_row(row) for row in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    case_ids = parse_case_ids(args.case_ids)
    rows = [process_case(case_id, args, device) for case_id in case_ids]
    summary = {
        "protocol_name": PROTOCOL_NAME,
        "data_root": str(args.data_root),
        "mask_root": str(args.mask_root),
        "out_root": str(args.out_root),
        "case_ids": case_ids,
        "all_verification_pass": bool(all(row["verification"]["pass"] for row in rows)),  # type: ignore[index]
        "rows": rows,
    }
    write_json(args.out_root / "4DCT" / "summary.json", summary)
    write_summary_csv(args.out_root / "4DCT" / "summary.csv", rows)
    print(f"Wrote {len(rows)} pTVprep cases to {args.out_root}")
    print(f"All verification pass: {summary['all_verification_pass']}")


if __name__ == "__main__":
    main()
