#!/usr/bin/env python
"""Generate canonical COPD pTVReg-crop 1 mm preprocessing artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.lungct import load_landmarks_xyz, load_raw_copd_image, mask_root_case_dir, resolve_copd_case  # noqa: E402
from common.volume import mask_stats, parse_case_ids, write_json  # noqa: E402
from generate_4dct_ptvprep import (  # noqa: E402
    MARGIN_XYZ,
    MARGIN_ZYX,
    PROTOCOL_NAME,
    audit_landmark_mask_fraction,
    crop_zyx,
    img_thr,
    landmarks_xyz_one_based_to_zyx_zero_based,
    points_inside_shape,
    ptvreg_crop_bounds_zyx_from_landmarks_xyz,
    resize_image_trilinear_zyx,
    resize_mask_nearest_zyx,
    save_case_arrays,
    select_crop_bounds,
    require_fresh_case,
    task4_legacy_crop_bounds_zyx_from_landmarks_xyz,
    transform_landmarks_crop_to_registration,
    transform_landmarks_native0_to_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-ids", type=str, default="1-10")
    parser.add_argument("--modelname", type=str, default="R231")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--crop-convention", choices=["corrected", "legacy"], default="legacy")
    parser.add_argument("--save-hu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_binary_mask(mask_root: Path, case_id: int, phase: str, modelname: str) -> np.ndarray:
    path = mask_root_case_dir(mask_root, case_id, dataset="copd") / f"case{case_id:02d}_{phase}_{modelname}_binary_zyx.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.uint8, copy=False)


def process_case(case_id: int, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    case_dir = args.out_root / "COPD" / f"case{case_id:02d}"
    require_fresh_case(case_dir)
    case = resolve_copd_case(case_id, args.data_root)
    if case.status != "complete":
        raise RuntimeError(f"COPD case {case_id} is {case.status}; notes={case.notes}")

    fixed_xyz = load_landmarks_xyz(case.landmarks_ibh)
    moving_xyz = load_landmarks_xyz(case.landmarks_ebh)
    fixed_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(fixed_xyz)
    moving_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(moving_xyz)

    ptvreg_formula_bounds = ptvreg_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, case.shape_zyx)
    legacy_bounds = task4_legacy_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, case.shape_zyx)
    convention = getattr(args, "crop_convention", "legacy")
    crop_bounds = select_crop_bounds(ptvreg_formula_bounds, legacy_bounds, convention)
    crop_shape = tuple((crop_bounds[:, 1] - crop_bounds[:, 0] + 1).astype(int).tolist())
    spacing = tuple(float(v) for v in case.spacing_zyx)
    reg_shape = tuple(np.rint(np.asarray(crop_shape, dtype=np.float64) * np.asarray(spacing)).astype(int).tolist())

    fixed_raw = load_raw_copd_image(case.image_ibh, case.shape_zyx)
    moving_raw = load_raw_copd_image(case.image_ebh, case.shape_zyx)
    fixed_native = img_thr(fixed_raw)
    moving_native = img_thr(moving_raw)
    fixed_crop = crop_zyx(fixed_native, crop_bounds)
    moving_crop = crop_zyx(moving_native, crop_bounds)
    fixed_reg = resize_image_trilinear_zyx(fixed_crop, reg_shape, device)
    moving_reg = resize_image_trilinear_zyx(moving_crop, reg_shape, device)

    fixed_mask_native = load_binary_mask(args.mask_root, case_id, "iBH", args.modelname)
    moving_mask_native = load_binary_mask(args.mask_root, case_id, "eBH", args.modelname)
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

    case_dir = args.out_root / "COPD" / f"case{case_id:02d}"
    paths = save_case_arrays(
        case_dir,
        {
            "fixed_iBH_native_ptvintensity_zyx": fixed_native,
            "moving_eBH_native_ptvintensity_zyx": moving_native,
            "fixed_iBH_crop_ptvintensity_zyx": fixed_crop,
            "moving_eBH_crop_ptvintensity_zyx": moving_crop,
            "fixed_iBH_reg1mm_trilinear_zyx": fixed_reg,
            "moving_eBH_reg1mm_trilinear_zyx": moving_reg,
            "fixed_iBH_lungmask_native_zyx": fixed_mask_native.astype(np.uint8),
            "moving_eBH_lungmask_native_zyx": moving_mask_native.astype(np.uint8),
            "fixed_iBH_lungmask_crop_zyx": fixed_mask_crop,
            "moving_eBH_lungmask_crop_zyx": moving_mask_crop,
            "fixed_iBH_lungmask_reg1mm_nearest_zyx": fixed_mask_reg,
            "moving_eBH_lungmask_reg1mm_nearest_zyx": moving_mask_reg,
            "landmarks_fixed_iBH_native0_zyx": fixed_native0_zyx,
            "landmarks_moving_eBH_native0_zyx": moving_native0_zyx,
            "landmarks_fixed_iBH_crop_zyx": fixed_crop_zyx,
            "landmarks_moving_eBH_crop_zyx": moving_crop_zyx,
            "landmarks_fixed_iBH_reg1mm_zyx": fixed_reg_zyx,
            "landmarks_moving_eBH_reg1mm_zyx": moving_reg_zyx,
        },
        args.overwrite,
    )
    if getattr(args, "save_hu", False):
        paths.update(save_case_arrays(case_dir, {
            "fixed_iBH_reg1mm_hu_zyx": resize_image_trilinear_zyx(crop_zyx(fixed_raw, crop_bounds) - 1024.0, reg_shape, device),
            "moving_eBH_reg1mm_hu_zyx": resize_image_trilinear_zyx(crop_zyx(moving_raw, crop_bounds) - 1024.0, reg_shape, device),
        }, False))

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
        "protocol_name": f"{PROTOCOL_NAME}_{convention}",
        "hu_rule": "raw_CT - 1024; no clipping; same crop and endpoint-aligned resize" if getattr(args, "save_hu", False) else None,
        "dataset": "copd",
        "case_id": case_id,
        "fixed_phase": "iBH",
        "moving_phase": "eBH",
        "direction": "eBHCT -> iBHCT",
        "raw_array_order": "zyx",
        "ptvreg_source_array_order": "xyz in MATLAB, represented as zyx in NumPy",
        "landmark_source_order": "xyz, pTVReg/MATLAB 1-based voxel coordinates",
        "landmark_saved_order": "zyx, Python 0-based voxel coordinates",
        "image_resampling": "PyTorch trilinear, align_corners=True",
        "mask_resampling": f"{convention} crop, PyTorch nearest-neighbor",
        "intensity_rule": "clip [80, 900], subtract 80, divide by 820",
        "crop_rule": (
            f"landmark bbox margin xyz [10,10,5] using explicit {convention} bounds; "
            "exact pTVReg/MATLAB 1-based formula bounds are retained for audit"
        ),
        "crop_convention": convention,
        "crop_margin_xyz": [int(v) for v in MARGIN_XYZ],
        "crop_margin_zyx": [int(v) for v in MARGIN_ZYX],
        "raw_shape_zyx": [int(v) for v in case.shape_zyx],
        "native_spacing_zyx_mm": [float(v) for v in case.spacing_zyx],
        "crop_bounds_zyx_inclusive": crop_bounds.tolist(),
        "ptvreg_formula_crop_bounds_zyx_inclusive": ptvreg_formula_bounds.tolist(),
        "legacy_task4_crop_bounds_zyx_inclusive": legacy_bounds.tolist(),
        "used_minus_ptvreg_formula_bounds_zyx": (crop_bounds - ptvreg_formula_bounds).tolist(),
        "legacy_minus_ptvreg_bounds_zyx": (legacy_bounds - ptvreg_formula_bounds).tolist(),
        "crop_shape_zyx": [int(v) for v in crop_shape],
        "registration_shape_zyx": [int(v) for v in reg_shape],
        "registration_spacing_zyx_mm": [1.0, 1.0, 1.0],
        "registration_to_native_scale_zyx": ((np.asarray(crop_shape)-1)/(np.asarray(reg_shape)-1)).tolist(),
        "image_paths": {
            "fixed_ibh": case.image_ibh,
            "moving_ebh": case.image_ebh,
        },
        "landmark_paths": {
            "fixed_ibh": case.landmarks_ibh,
            "moving_ebh": case.landmarks_ebh,
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
        "verification_pass": verification["pass"],
    }
    write_json(case_dir / "preprocess_audit.json", row)
    print(
        f"copd case{case_id:02d}: crop={crop_shape} reg={reg_shape} "
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
        "dataset": "copd",
        "data_root": str(args.data_root),
        "mask_root": str(args.mask_root),
        "out_root": str(args.out_root),
        "case_ids": case_ids,
        "all_verification_pass": bool(all(row["verification"]["pass"] for row in rows)),  # type: ignore[index]
        "rows": rows,
    }
    write_json(args.out_root / "COPD" / "summary.json", summary)
    write_summary_csv(args.out_root / "COPD" / "summary.csv", rows)
    print(f"Wrote {len(rows)} COPD pTVprep cases to {args.out_root}")
    print(f"All verification pass: {summary['all_verification_pass']}")


if __name__ == "__main__":
    main()
