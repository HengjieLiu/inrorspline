"""Render a ten-case landmark residual atlas for LungCT pTVReg vs MR-D-BSCP P2e.

The script is intentionally posthoc: it reads completed pTVReg MATLAB fields,
canonical pTV-preprocessed landmarks, and completed MR-D-BSCP P2e fields.  It
does not run registration.

Required environment variables are documented in README.md.  All scalar TRE
comparisons use native-crop snap-to-voxel TRE.  Vector glyphs use local
physical-mm coordinates in the same native crop and show the coronal R/L and
I/S projection; their colors encode the full 3-D vector magnitude.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import map_coordinates


P2E_EXPERIMENT_ID = "2026-09-01_lungct_mrdbscp_acceleration_best1mm_v1"
P2E_RUN_PREFIX = "2026-09-01_lungct_mrdbscp_acceleration_best1mm"
P2E_METHOD_DIR = "direct_cp_32_16_8_4_ptv_lcc_isotv"
P2E_VARIANTS = {
    "4dct": "original_corrected",
    "copd": "oasis_bal_decay_ordinal",
}
DATASET_LABELS = {"4dct": "4DCT", "copd": "COPD"}
DATASET_DIRS = {"4dct": "4DCT", "copd": "COPD"}
FIXED_LABELS = {"4dct": "T00", "copd": "iBH"}
MOVING_LABELS = {"4dct": "T50", "copd": "eBH"}
CASE_IDS = tuple(range(1, 11))

METHOD_COLORS = {"pTVReg": "#0072B2", "MR-D-BSCP P2e": "#D55E00", "Initial": "#777777"}
INITIAL_CMAP = "magma"
RESIDUAL_CMAP = "viridis"
APPLIED_CMAP = "cividis"

# The canonical pTVReg crop stores volumes and landmark coordinates in raw
# NumPy zyx order.  The repository's accepted coronal inspection convention
# leaves x columns unflipped (patient R on the left, L on the right) and flips
# z rows for display (patient S on top, I on the bottom).  This is a display
# transform only; physical-mm magnitudes remain unchanged.
CORONAL_ORIENTATION = {
    "source_frame": "canonical pTVReg raw zyx crop",
    "horizontal_display_sign": 1.0,
    "vertical_display_sign": -1.0,
    "horizontal_left": "R",
    "horizontal_right": "L",
    "vertical_bottom": "I",
    "vertical_top": "S",
    "projection": "display=[+x,-z] from local physical xyz",
}


def required_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{name} points to a missing path: {path}")
    return path


def canonical_case_dir(data_root: Path, dataset: str, case_id: int) -> Path:
    return data_root / DATASET_DIRS[dataset] / f"case{case_id:02d}"


def landmark_names(dataset: str) -> tuple[str, str]:
    return (
        f"landmarks_fixed_{FIXED_LABELS[dataset]}_crop_zyx.npy",
        f"landmarks_moving_{MOVING_LABELS[dataset]}_crop_zyx.npy",
    )


def native_mask_name(dataset: str) -> str:
    return f"fixed_{FIXED_LABELS[dataset]}_lungmask_reg1mm_nearest_zyx.npy"


def reg_field_name(dataset: str) -> str:
    return f"fixed_{FIXED_LABELS[dataset]}_reg1mm_trilinear_zyx.npy"


def pTVReg_mat_path(ptv_root: Path, dataset: str, case_id: int) -> Path:
    if dataset == "4dct":
        return ptv_root / f"case{case_id:02d}_T50_to_T00_deformation.mat"
    return ptv_root / f"case{case_id:02d}" / f"case{case_id:02d}_eBH_to_iBH_deformation.mat"


def p2e_case_dir(p2e_root: Path, dataset: str, case_id: int) -> Path:
    variant = P2E_VARIANTS[dataset]
    return p2e_root / f"{P2E_RUN_PREFIX}_{dataset}_{variant}_case{case_id:02d}"


def read_scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def normalize_crop_v(crop_v: np.ndarray) -> np.ndarray:
    crop = np.asarray(crop_v, dtype=np.float64)
    if crop.shape == (3, 2):
        crop = crop.T
    if crop.shape != (2, 3):
        raise ValueError(f"Expected crop_v_saved shape (2,3) or (3,2), got {crop.shape}")
    return crop


def crop_bounds_zyx0(crop_v_xyz1: np.ndarray) -> np.ndarray:
    crop = normalize_crop_v(crop_v_xyz1)
    start0_xyz = crop[0].astype(np.int64) - 1
    end0_xyz = crop[1].astype(np.int64) - 1
    return np.stack((start0_xyz[[2, 1, 0]], end0_xyz[[2, 1, 0]]), axis=1)


def xyz_local_mm(points_xyz1: np.ndarray, crop_start_xyz1: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(points_xyz1, dtype=np.float64) - crop_start_xyz1[None, :]) * spacing_xyz[None, :]


def zyx_local_mm(points_zyx0: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    return np.asarray(points_zyx0, dtype=np.float64) * spacing_zyx[None, :]


def coronal_orientation_from_audit(audit: dict[str, Any], dataset: str, case_id: int) -> dict[str, Any]:
    """Return the explicit pTVReg crop-to-clinical coronal display contract."""

    if str(audit.get("raw_array_order", "")).lower() != "zyx":
        raise ValueError(f"{dataset} case{case_id:02d}: expected raw_array_order=zyx")
    saved_order = str(audit.get("landmark_saved_order", "")).lower()
    if "zyx" not in saved_order:
        raise ValueError(f"{dataset} case{case_id:02d}: landmark order is not zyx: {saved_order!r}")
    return dict(CORONAL_ORIENTATION)


def sample_nearest_czyx(field_czyx: np.ndarray, points_zyx: np.ndarray) -> np.ndarray:
    field = np.asarray(field_czyx)
    points = np.rint(np.asarray(points_zyx, dtype=np.float64)).astype(np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N,3] points, got {points.shape}")
    if np.any(points < 0) or np.any(points[:, 0] >= field.shape[1]) or np.any(points[:, 1] >= field.shape[2]) or np.any(points[:, 2] >= field.shape[3]):
        raise ValueError("Landmark is outside dense field bounds")
    return np.stack(
        [field[c, points[:, 0], points[:, 1], points[:, 2]] for c in range(field.shape[0])], axis=1
    ).astype(np.float64, copy=False)


def resize_field_align_corners(field_zyxc: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    """Match torch F.interpolate(..., mode='trilinear', align_corners=True)."""
    field = np.asarray(field_zyxc, dtype=np.float32)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected [z,y,x,3] displacement field, got {field.shape}")
    if tuple(field.shape[:3]) == tuple(output_shape):
        return field.copy()
    try:
        import torch
        import torch.nn.functional as F

        tensor = torch.from_numpy(np.ascontiguousarray(field)).permute(3, 0, 1, 2)[None]
        resized = F.interpolate(tensor, size=tuple(int(v) for v in output_shape), mode="trilinear", align_corners=True)
        return resized[0].permute(1, 2, 3, 0).numpy().astype(np.float32, copy=False)
    except ImportError:
        # The standard container has torch.  This fallback keeps the script
        # usable in a lightweight analysis environment.
        axes = [
            np.linspace(0.0, max(field.shape[axis] - 1, 0), int(output_shape[axis]))
            for axis in range(3)
        ]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        coords = np.stack((zz, yy, xx), axis=0)
        return np.stack(
            [map_coordinates(field[..., c], coords, order=1, mode="nearest") for c in range(3)], axis=-1
        ).astype(np.float32, copy=False)


def sample_ptvreg_field_at_common_mask(
    field_czyx: np.ndarray,
    fixed_mask: np.ndarray,
    crop_bounds_zyx: np.ndarray,
) -> np.ndarray:
    """Sample pTVReg's native full field at the canonical 1-mm fixed mask."""
    mask_indices = np.argwhere(np.asarray(fixed_mask, dtype=bool))
    common_shape = np.asarray(fixed_mask.shape, dtype=np.float64)
    start = np.asarray(crop_bounds_zyx[:, 0], dtype=np.float64)
    end = np.asarray(crop_bounds_zyx[:, 1], dtype=np.float64)
    native_points = start[None, :] + mask_indices.astype(np.float64) * (end - start)[None, :] / np.maximum(common_shape[None, :] - 1.0, 1.0)

    values = np.empty((mask_indices.shape[0], 3), dtype=np.float32)
    chunk_size = 1_000_000
    for begin in range(0, native_points.shape[0], chunk_size):
        end_index = min(begin + chunk_size, native_points.shape[0])
        coords = native_points[begin:end_index].T
        sampled_xyz = np.stack(
            [map_coordinates(field_czyx[c], coords, order=1, mode="nearest") for c in range(3)], axis=1
        )
        values[begin:end_index] = sampled_xyz[:, [2, 1, 0]].astype(np.float32, copy=False)
    return np.linalg.norm(values.astype(np.float64), axis=1).astype(np.float32, copy=False)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CaseAtlas:
    dataset: str
    case_id: int
    fixed_xyz_mm: np.ndarray
    target_xyz_mm: np.ndarray
    initial_vec_xyz_mm: np.ndarray
    ptv_pred_xyz_mm: np.ndarray
    ptv_residual_xyz_mm: np.ndarray
    p2e_pred_xyz_mm: np.ndarray
    p2e_residual_xyz_mm: np.ndarray
    initial_tre_mm: np.ndarray
    ptv_tre_mm: np.ndarray
    p2e_tre_mm: np.ndarray
    ptv_applied_mm: np.ndarray
    p2e_applied_mm: np.ndarray
    native_spacing_zyx_mm: np.ndarray
    crop_shape_zyx: tuple[int, int, int]
    coronal_orientation: dict[str, Any]
    source_refs: dict[str, str]
    validation: dict[str, float]


def load_canonical_case(data_root: Path, dataset: str, case_id: int) -> dict[str, Any]:
    case_dir = canonical_case_dir(data_root, dataset, case_id)
    fixed_name, moving_name = landmark_names(dataset)
    audit_path = case_dir / "preprocess_audit.json"
    fixed = np.load(case_dir / fixed_name).astype(np.float64, copy=False)
    moving = np.load(case_dir / moving_name).astype(np.float64, copy=False)
    fixed_mask = np.load(case_dir / native_mask_name(dataset)).astype(bool, copy=False)
    with audit_path.open() as handle:
        audit = json.load(handle)
    orientation = coronal_orientation_from_audit(audit, dataset, case_id)
    spacing_zyx = np.asarray(audit["native_spacing_zyx_mm"], dtype=np.float64)
    crop_shape = tuple(int(v) for v in audit["crop_shape_zyx"])
    crop_bounds = np.asarray(audit["crop_bounds_zyx_inclusive"], dtype=np.int64)
    if fixed.shape != (300, 3) or moving.shape != (300, 3):
        raise ValueError(f"{dataset} case{case_id:02d}: expected 300 crop landmarks, got {fixed.shape}/{moving.shape}")
    if fixed_mask.ndim != 3:
        raise ValueError(f"{dataset} case{case_id:02d}: fixed mask is not 3-D")
    if fixed_mask.shape != tuple(int(v) for v in np.load(case_dir / reg_field_name(dataset), mmap_mode="r").shape):
        raise ValueError(f"{dataset} case{case_id:02d}: fixed mask and registration grid disagree")
    if crop_bounds.shape != (3, 2):
        raise ValueError(f"{dataset} case{case_id:02d}: invalid crop bounds {crop_bounds.shape}")
    fixed_xyz_mm = zyx_local_mm(fixed[:, [0, 1, 2]], spacing_zyx)[:, [2, 1, 0]]
    target_xyz_mm = zyx_local_mm(moving[:, [0, 1, 2]], spacing_zyx)[:, [2, 1, 0]]
    return {
        "case_dir": case_dir,
        "fixed_crop_zyx": fixed,
        "moving_crop_zyx": moving,
        "fixed_xyz_mm": fixed_xyz_mm,
        "target_xyz_mm": target_xyz_mm,
        "fixed_mask": fixed_mask,
        "spacing_zyx": spacing_zyx,
        "crop_shape": crop_shape,
        "crop_bounds": crop_bounds,
        "audit_path": audit_path,
        "coronal_orientation": orientation,
    }


def load_ptvreg_case(
    data_root: Path,
    ptv_root: Path,
    dataset: str,
    case_id: int,
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], str]:
    path = pTVReg_mat_path(ptv_root, dataset, case_id)
    required = {"Tptv_rsz", "pts_fix", "pts_mov", "spc_orig_saved", "crop_v_saved", "TRE_phys", "TREstd_case"}
    with h5py.File(path, "r") as handle:
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        field = np.asarray(handle["Tptv_rsz"], dtype=np.float32)
        pts_fix_xyz1 = np.asarray(handle["pts_fix"], dtype=np.float64).T
        pts_mov_xyz1 = np.asarray(handle["pts_mov"], dtype=np.float64).T
        spacing_xyz = np.asarray(handle["spc_orig_saved"], dtype=np.float64).reshape(-1)
        crop_v = normalize_crop_v(np.asarray(handle["crop_v_saved"]))
        saved_mean = read_scalar(handle, "TRE_phys")
        saved_std = read_scalar(handle, "TREstd_case")
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(f"{path}: expected Tptv_rsz shape (3,z,y,x), got {field.shape}")
    if pts_fix_xyz1.shape != (300, 3) or pts_mov_xyz1.shape != (300, 3):
        raise ValueError(f"{path}: expected 300 [x,y,z] landmarks")
    if spacing_xyz.shape != (3,) or not np.isfinite(spacing_xyz).all():
        raise ValueError(f"{path}: invalid spacing {spacing_xyz}")

    crop_start_xyz1 = crop_v[0]
    crop_bounds = crop_bounds_zyx0(crop_v)
    canonical_crop_start_xyz1 = np.asarray(
        [canonical["crop_bounds"][2, 0] + 1, canonical["crop_bounds"][1, 0] + 1, canonical["crop_bounds"][0, 0] + 1],
        dtype=np.float64,
    )
    idx_xyz0 = np.rint(pts_fix_xyz1).astype(np.int64) - 1
    disp_xyz_mm = np.stack(
        [field[0, idx_xyz0[:, 2], idx_xyz0[:, 1], idx_xyz0[:, 0]],
         field[1, idx_xyz0[:, 2], idx_xyz0[:, 1], idx_xyz0[:, 0]],
         field[2, idx_xyz0[:, 2], idx_xyz0[:, 1], idx_xyz0[:, 0]]],
        axis=1,
    ).astype(np.float64, copy=False)
    predicted_xyz1 = np.rint(pts_fix_xyz1 + disp_xyz_mm / spacing_xyz[None, :])
    residual_xyz_mm = (pts_mov_xyz1 - predicted_xyz1) * spacing_xyz[None, :]
    tre_mm = np.linalg.norm(residual_xyz_mm, axis=1)
    # The completed COPD pTVReg reproduction uses the legacy MATLAB crop
    # origin, while the canonical pTV-prep handle uses the corrected origin.
    # Both refer to the same absolute native landmarks, so local coordinates
    # are expressed using the canonical crop origin for cross-method plotting.
    fixed_xyz_mm = xyz_local_mm(pts_fix_xyz1, canonical_crop_start_xyz1, spacing_xyz)
    target_xyz_mm = xyz_local_mm(pts_mov_xyz1, canonical_crop_start_xyz1, spacing_xyz)
    predicted_local_xyz_mm = xyz_local_mm(predicted_xyz1, canonical_crop_start_xyz1, spacing_xyz)
    initial_tre = np.linalg.norm(target_xyz_mm - fixed_xyz_mm, axis=1)
    if not np.allclose(fixed_xyz_mm, canonical["fixed_xyz_mm"], rtol=0, atol=1e-6):
        raise ValueError(f"{path}: pTVReg fixed landmarks do not match canonical crop landmarks")
    if not np.allclose(target_xyz_mm, canonical["target_xyz_mm"], rtol=0, atol=1e-6):
        raise ValueError(f"{path}: pTVReg moving landmarks do not match canonical crop landmarks")
    if not np.isclose(float(np.mean(tre_mm)), saved_mean, rtol=0, atol=1e-8):
        raise ValueError(f"{path}: recomputed TRE mean disagrees with TRE_phys")
    if not np.isclose(float(np.std(tre_mm, ddof=1)), saved_std, rtol=0, atol=1e-8):
        raise ValueError(f"{path}: recomputed TRE std disagrees with TREstd_case")

    applied_mm = sample_ptvreg_field_at_common_mask(field, canonical["fixed_mask"], canonical["crop_bounds"])
    validation = {
        "ptvreg_saved_tre_mean_mm": float(saved_mean),
        "ptvreg_recomputed_tre_mean_mm": float(np.mean(tre_mm)),
        "ptvreg_saved_tre_std_mm": float(saved_std),
        "ptvreg_recomputed_tre_std_mm": float(np.std(tre_mm, ddof=1)),
        "initial_tre_mean_mm": float(np.mean(initial_tre)),
        "ptvreg_crop_offset_z_vox": float(crop_bounds[0, 0] - canonical["crop_bounds"][0, 0]),
        "ptvreg_crop_offset_y_vox": float(crop_bounds[1, 0] - canonical["crop_bounds"][1, 0]),
        "ptvreg_crop_offset_x_vox": float(crop_bounds[2, 0] - canonical["crop_bounds"][2, 0]),
    }
    payload = {
        "fixed_xyz_mm": fixed_xyz_mm,
        "target_xyz_mm": target_xyz_mm,
        "predicted_xyz_mm": predicted_local_xyz_mm,
        "residual_xyz_mm": residual_xyz_mm,
        "initial_tre_mm": initial_tre,
        "tre_mm": tre_mm,
        "applied_mm": applied_mm,
        "native_spacing_zyx_mm": spacing_xyz[[2, 1, 0]],
        "crop_shape_zyx": tuple(int(v) for v in canonical["crop_shape"]),
        "validation": validation,
    }
    del field
    gc.collect()
    source_ref = f"PTVREG_{dataset.upper()}_ROOT/{path.relative_to(ptv_root)}"
    return payload, validation, source_ref


def read_p2e_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise ValueError(f"{path}: expected 300 landmark rows, got {len(rows)}")
    fixed = np.asarray([[float(row["fixed_z"]), float(row["fixed_y"]), float(row["fixed_x"])] for row in rows])
    target = np.asarray([[float(row["target_z"]), float(row["target_y"]), float(row["target_x"])] for row in rows])
    tre = np.asarray([float(row["snap_tre_mm"]) for row in rows], dtype=np.float64)
    if not np.isfinite(np.concatenate((fixed.ravel(), target.ravel(), tre))).all():
        raise ValueError(f"{path}: non-finite landmark data")
    return fixed, target, tre


def load_p2e_case(
    data_root: Path,
    p2e_root: Path,
    dataset: str,
    case_id: int,
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], str]:
    run_dir = p2e_case_dir(p2e_root, dataset, case_id)
    field_path = run_dir / "disp_vox_reg1mm_zyx.npy"
    csv_path = run_dir / "per_landmark_tre_native_crop.csv"
    field = np.load(field_path).astype(np.float32, copy=False)
    if field.shape != (*canonical["fixed_mask"].shape, 3):
        raise ValueError(f"{field_path}: field shape {field.shape} disagrees with canonical grid")
    csv_fixed, csv_target, csv_tre = read_p2e_csv(csv_path)
    if not np.allclose(csv_fixed, canonical["fixed_crop_zyx"], rtol=0, atol=1e-6):
        raise ValueError(f"{csv_path}: fixed landmarks disagree with canonical crop")
    if not np.allclose(csv_target, canonical["moving_crop_zyx"], rtol=0, atol=1e-6):
        raise ValueError(f"{csv_path}: target landmarks disagree with canonical crop")

    native_field_phys = resize_field_align_corners(field, canonical["crop_shape"])
    native_field = native_field_phys / canonical["spacing_zyx"][None, None, None, :]
    disp_at_landmarks_zyx = sample_nearest_czyx(native_field.transpose(3, 0, 1, 2), canonical["fixed_crop_zyx"])
    predicted_zyx = np.rint(canonical["fixed_crop_zyx"] + disp_at_landmarks_zyx)
    residual_zyx_mm = (canonical["moving_crop_zyx"] - predicted_zyx) * canonical["spacing_zyx"][None, :]
    tre_mm = np.linalg.norm(residual_zyx_mm, axis=1)
    fixed_xyz_mm = canonical["fixed_xyz_mm"]
    target_xyz_mm = canonical["target_xyz_mm"]
    predicted_xyz_mm = (predicted_zyx[:, [2, 1, 0]]) * canonical["spacing_zyx"][None, [2, 1, 0]]
    residual_xyz_mm = target_xyz_mm - predicted_xyz_mm
    initial_tre = np.linalg.norm(target_xyz_mm - fixed_xyz_mm, axis=1)
    if not np.allclose(tre_mm, csv_tre, rtol=0, atol=2e-4):
        raise ValueError(f"{csv_path}: recomputed snap TRE disagrees with saved snap_tre_mm")
    applied_mm = np.linalg.norm(field[canonical["fixed_mask"]].astype(np.float64), axis=1).astype(np.float32)
    validation = {
        "p2e_saved_tre_mean_mm": float(np.mean(csv_tre)),
        "p2e_recomputed_tre_mean_mm": float(np.mean(tre_mm)),
        "p2e_max_landmark_tre_abs_err_mm": float(np.max(np.abs(tre_mm - csv_tre))),
        "initial_tre_mean_mm": float(np.mean(initial_tre)),
    }
    payload = {
        "fixed_xyz_mm": fixed_xyz_mm,
        "target_xyz_mm": target_xyz_mm,
        "predicted_xyz_mm": predicted_xyz_mm,
        "residual_xyz_mm": residual_xyz_mm,
        "initial_tre_mm": initial_tre,
        "tre_mm": tre_mm,
        "applied_mm": applied_mm,
        "native_spacing_zyx_mm": canonical["spacing_zyx"],
        "crop_shape_zyx": tuple(int(v) for v in canonical["crop_shape"]),
        "validation": validation,
    }
    del field, native_field_phys, native_field
    gc.collect()
    source_ref = f"MRDBSCP_P2E_ACCEL_ROOT/{run_dir.relative_to(p2e_root)}"
    return payload, validation, source_ref


def build_case(data_root: Path, ptv_root: Path, p2e_root: Path, dataset: str, case_id: int) -> CaseAtlas:
    canonical = load_canonical_case(data_root, dataset, case_id)
    ptv, ptv_validation, ptv_ref = load_ptvreg_case(data_root, ptv_root, dataset, case_id, canonical)
    p2e, p2e_validation, p2e_ref = load_p2e_case(data_root, p2e_root, dataset, case_id, canonical)
    if not np.allclose(ptv["initial_tre_mm"], p2e["initial_tre_mm"], rtol=0, atol=1e-6):
        raise ValueError(f"{dataset} case{case_id:02d}: pTVReg and P2e initial TRE vectors disagree")
    return CaseAtlas(
        dataset=dataset,
        case_id=case_id,
        fixed_xyz_mm=canonical["fixed_xyz_mm"],
        target_xyz_mm=canonical["target_xyz_mm"],
        initial_vec_xyz_mm=canonical["target_xyz_mm"] - canonical["fixed_xyz_mm"],
        ptv_pred_xyz_mm=ptv["predicted_xyz_mm"],
        ptv_residual_xyz_mm=ptv["residual_xyz_mm"],
        p2e_pred_xyz_mm=p2e["predicted_xyz_mm"],
        p2e_residual_xyz_mm=p2e["residual_xyz_mm"],
        initial_tre_mm=ptv["initial_tre_mm"],
        ptv_tre_mm=ptv["tre_mm"],
        p2e_tre_mm=p2e["tre_mm"],
        ptv_applied_mm=ptv["applied_mm"],
        p2e_applied_mm=p2e["applied_mm"],
        native_spacing_zyx_mm=canonical["spacing_zyx"],
        crop_shape_zyx=tuple(int(v) for v in canonical["crop_shape"]),
        coronal_orientation=canonical["coronal_orientation"],
        source_refs={"pTVReg": ptv_ref, "MR-D-BSCP P2e": p2e_ref},
        validation={**ptv_validation, **p2e_validation},
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    return x, np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)


def finite_max(values: Iterable[np.ndarray], default: float = 1.0) -> float:
    maximum = default
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.size:
            maximum = max(maximum, float(np.nanmax(array)))
    return maximum


def vector_panel(
    axis: Any,
    start_xyz_mm: np.ndarray,
    vector_xyz_mm: np.ndarray,
    norm: Normalize,
    cmap: Any,
    orientation: dict[str, Any],
    colorbar_label: str,
) -> Any:
    """Render one vector panel and attach its own magnitude colorbar."""

    def project(points_xyz_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz_mm, dtype=np.float64)
        return np.column_stack(
            (
                float(orientation["horizontal_display_sign"]) * points[:, 0],
                float(orientation["vertical_display_sign"]) * points[:, 2],
            )
        )

    magnitude = np.linalg.norm(vector_xyz_mm, axis=1)
    colors = cmap(norm(magnitude))
    start_display = project(start_xyz_mm)
    vector_display = np.column_stack(
        (
            float(orientation["horizontal_display_sign"]) * vector_xyz_mm[:, 0],
            float(orientation["vertical_display_sign"]) * vector_xyz_mm[:, 2],
        )
    )
    axis.set_facecolor("white")
    axis.quiver(
        start_display[:, 0],
        start_display[:, 1],
        vector_display[:, 0],
        vector_display[:, 1],
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0022,
        headwidth=3.4,
        headlength=4.0,
        headaxislength=3.5,
        alpha=0.9,
        minlength=0.01,
    )
    axis.scatter(
        start_display[:, 0],
        start_display[:, 1],
        s=2.0,
        c="#222222",
        alpha=0.7,
        linewidths=0,
        zorder=3,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R  ←──────────→  L", fontsize=4.8, labelpad=1)
    axis.set_ylabel("I  ←──────────→  S", fontsize=4.8, labelpad=1)
    axis.grid(color="#bbbbbb", alpha=0.35, linewidth=0.35)
    axis.tick_params(colors="#111111", labelsize=4.8, length=2, pad=1)
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.45)
    axis.text(0.02, 0.96, "R", transform=axis.transAxes, color="#111111", fontsize=4.8, va="top", ha="left")
    axis.text(0.98, 0.96, "L", transform=axis.transAxes, color="#111111", fontsize=4.8, va="top", ha="right")
    axis.text(0.50, 0.98, "S", transform=axis.transAxes, color="#111111", fontsize=4.8, va="top", ha="center")
    axis.text(0.50, 0.02, "I", transform=axis.transAxes, color="#111111", fontsize=4.8, va="bottom", ha="center")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(magnitude)
    cbar_axis = inset_axes(
        axis,
        width="3.2%",
        height="64%",
        loc="center left",
        bbox_to_anchor=(1.015, 0.0, 1.0, 1.0),
        bbox_transform=axis.transAxes,
        borderpad=0.0,
    )
    cbar = axis.figure.colorbar(sm, cax=cbar_axis)
    cbar.ax.set_facecolor("white")
    cbar.ax.tick_params(labelsize=4.0, length=1.5, pad=1, colors="#111111")
    cbar.outline.set_edgecolor("#333333")
    cbar.outline.set_linewidth(0.35)
    cbar.set_label(colorbar_label, fontsize=4.0, labelpad=2)
    return cbar


def set_vector_limits(axes: list[Any], case: CaseAtlas, orientation: dict[str, Any]) -> None:
    points = np.concatenate(
        [case.fixed_xyz_mm, case.target_xyz_mm, case.ptv_pred_xyz_mm, case.p2e_pred_xyz_mm], axis=0
    )
    x = float(orientation["horizontal_display_sign"]) * points[:, 0]
    z = float(orientation["vertical_display_sign"]) * points[:, 2]
    x_span = max(float(np.ptp(x)), 1.0)
    z_span = max(float(np.ptp(z)), 1.0)
    x_pad = max(0.04 * x_span, 3.0)
    z_pad = max(0.04 * z_span, 3.0)
    limits = (float(x.min() - x_pad), float(x.max() + x_pad), float(z.min() - z_pad), float(z.max() + z_pad))
    for axis in axes:
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])


def style_ecdf_axis(axis: Any, x_limit: float, label: str | None = None) -> None:
    axis.set_facecolor("white")
    axis.set_xlim(0.0, max(float(x_limit), 1e-6))
    axis.set_ylim(0.0, 1.01)
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.set_yticklabels(("0", "0.5", "1"), fontsize=5)
    axis.tick_params(colors="#111111", labelsize=5, length=2, pad=1)
    axis.grid(color="#bbbbbb", alpha=0.24, linewidth=0.35)
    if label:
        axis.set_xlabel(label, fontsize=6, labelpad=2)
    axis.set_ylabel("fraction ≤ x", fontsize=5.5, labelpad=1)
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.45)


def add_ecdf_method_legend(axis: Any, include_initial: bool) -> None:
    """Keep the method-color key local to each ECDF subplot."""
    method_specs = (
        (("Initial", METHOD_COLORS["Initial"]),) if include_initial else ()
    ) + (
        ("pTVReg", METHOD_COLORS["pTVReg"]),
        ("MR-D-BSCP P2e", METHOD_COLORS["MR-D-BSCP P2e"]),
    )
    handles = [Line2D([0], [0], color=color, lw=1.4, label=label) for label, color in method_specs]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=4.2,
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.9,
        borderpad=0.25,
        labelspacing=0.15,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0.0,
    )


def plot_case_ecdfs(
    axes: tuple[Any, Any, Any],
    case: CaseAtlas,
    registered_limit: float,
    full_tre_limit: float,
    applied_limit: float,
) -> None:
    registered_axis, tre_axis, applied_axis = axes
    for values, color, label in (
        (case.ptv_tre_mm, METHOD_COLORS["pTVReg"], "pTVReg"),
        (case.p2e_tre_mm, METHOD_COLORS["MR-D-BSCP P2e"], "MR-D-BSCP P2e"),
    ):
        x, y = ecdf(values)
        registered_axis.plot(x, y, color=color, linewidth=0.9, label=label)
    x, y = ecdf(case.initial_tre_mm)
    tre_axis.plot(x, y, color=METHOD_COLORS["Initial"], linewidth=0.9, label="Initial")
    for values, color, label in (
        (case.ptv_tre_mm, METHOD_COLORS["pTVReg"], "pTVReg"),
        (case.p2e_tre_mm, METHOD_COLORS["MR-D-BSCP P2e"], "MR-D-BSCP P2e"),
    ):
        x, y = ecdf(values)
        tre_axis.plot(x, y, color=color, linewidth=0.9, label=label)
    for values, color, label in (
        (case.ptv_applied_mm, METHOD_COLORS["pTVReg"], "pTVReg"),
        (case.p2e_applied_mm, METHOD_COLORS["MR-D-BSCP P2e"], "MR-D-BSCP P2e"),
    ):
        x, y = ecdf(values)
        applied_axis.plot(x, y, color=color, linewidth=0.9, label=label)

    style_ecdf_axis(registered_axis, registered_limit, "TRE (mm)")
    style_ecdf_axis(tre_axis, full_tre_limit, "TRE (mm)")
    style_ecdf_axis(applied_axis, applied_limit, "|motion| (mm)")
    add_ecdf_method_legend(registered_axis, include_initial=False)
    add_ecdf_method_legend(tre_axis, include_initial=True)
    add_ecdf_method_legend(applied_axis, include_initial=False)
    p99 = registered_limit
    pTV_tail = int(np.sum(case.ptv_tre_mm > p99))
    p2e_tail = int(np.sum(case.p2e_tre_mm > p99))
    registered_axis.text(
        0.98,
        0.06,
        f">p99: {pTV_tail}/{p2e_tail}",
        transform=registered_axis.transAxes,
        fontsize=4.5,
        color="#444444",
        ha="right",
    )


def render_dataset(cases: list[CaseAtlas], output_path: Path) -> dict[str, Any]:
    dataset = cases[0].dataset
    label = DATASET_LABELS[dataset]
    orientation = cases[0].coronal_orientation
    if any(case.coronal_orientation != orientation for case in cases):
        raise ValueError(f"{dataset}: cases do not share one coronal orientation contract")
    initial_limit = finite_max((case.initial_tre_mm for case in cases))
    residual_limit = finite_max(
        (np.concatenate((case.ptv_tre_mm, case.p2e_tre_mm)) for case in cases)
    )
    registered_values = np.concatenate([case.ptv_tre_mm for case in cases] + [case.p2e_tre_mm for case in cases])
    registered_p99 = float(np.percentile(registered_values, 99))
    full_tre_limit = finite_max(
        (np.concatenate((case.initial_tre_mm, case.ptv_tre_mm, case.p2e_tre_mm)) for case in cases)
    ) * 1.03
    applied_limit = finite_max(
        (np.concatenate((case.ptv_applied_mm, case.p2e_applied_mm)) for case in cases)
    ) * 1.03

    fig = plt.figure(figsize=(27.0, 34.0), facecolor="white")
    grid = fig.add_gridspec(
        nrows=10,
        ncols=6,
        width_ratios=(1.0, 1.0, 1.0, 0.70, 1.08, 1.08),
        hspace=0.16,
        wspace=0.18,
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.905,
    )
    axes = np.empty((10, 6), dtype=object)
    for row in range(10):
        for col in range(6):
            axes[row, col] = fig.add_subplot(grid[row, col])

    initial_norm = Normalize(vmin=0.0, vmax=max(initial_limit, 1e-6))
    residual_norm = Normalize(vmin=0.0, vmax=max(residual_limit, 1e-6))
    initial_cmap = colormaps.get_cmap(INITIAL_CMAP)
    residual_cmap = colormaps.get_cmap(RESIDUAL_CMAP)
    colorbars: list[Any] = []

    for row, case in enumerate(cases):
        initial_axis, ptv_axis, p2e_axis = axes[row, :3]
        colorbars.append(
            vector_panel(
                initial_axis,
                case.fixed_xyz_mm,
                case.initial_vec_xyz_mm,
                initial_norm,
                initial_cmap,
                orientation,
                "initial magnitude (mm)",
            )
        )
        colorbars.append(
            vector_panel(
                ptv_axis,
                case.ptv_pred_xyz_mm,
                case.ptv_residual_xyz_mm,
                residual_norm,
                residual_cmap,
                orientation,
                "residual magnitude (mm)",
            )
        )
        colorbars.append(
            vector_panel(
                p2e_axis,
                case.p2e_pred_xyz_mm,
                case.p2e_residual_xyz_mm,
                residual_norm,
                residual_cmap,
                orientation,
                "residual magnitude (mm)",
            )
        )
        set_vector_limits([initial_axis, ptv_axis, p2e_axis], case, orientation)
        registered_axis, tre_axis, applied_axis = axes[row, 3:]
        plot_case_ecdfs((registered_axis, tre_axis, applied_axis), case, registered_p99, full_tre_limit, applied_limit)
        initial_axis.text(
            -0.30,
            0.50,
            f"{label}{case.case_id:02d}",
            transform=initial_axis.transAxes,
            fontsize=6.2,
            rotation=90,
            va="center",
            ha="center",
            fontweight="bold",
            clip_on=False,
        )

    titles = (
        "Initial TRE vectors",
        "pTVReg residual TRE",
        "MR-D-BSCP P2e residual TRE",
        "Registered TRE ECDF\n(dataset p99 x-limit)",
        "TRE ECDF",
        "Applied-motion ECDF",
    )
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9, pad=7, fontweight="bold")

    fig.suptitle(
        f"{label}: pTVReg reproduction vs MR-D-BSCP P2e — landmark residual atlas",
        fontsize=16,
        fontweight="bold",
        y=0.968,
    )
    fig.text(
        0.5,
        0.944,
        f"300 landmarks/case; native-crop snap-to-voxel TRE; arrows show the coronal R/L and I/S projection and color encodes 3-D physical-mm magnitude; P2e={P2E_VARIANTS[dataset]}",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )
    fig.text(
        0.5,
        0.927,
        f"Initial scale: 0–{initial_limit:.2f} mm shared across cases    |    Residual scale: 0–{residual_limit:.2f} mm shared by pTVReg/P2e    |    Registered p99={registered_p99:.2f} mm",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )

    if len(colorbars) != 30:
        raise AssertionError(f"Expected one colorbar per vector panel, got {len(colorbars)}")
    if any(colorbar.mappable.norm is not initial_norm and colorbar.mappable.norm is not residual_norm for colorbar in colorbars):
        raise AssertionError("A vector colorbar is not bound to the atlas normalization")
    if any(
        not tick_label.get_visible()
        for axis in axes.ravel()
        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels())
    ):
        raise AssertionError("At least one atlas subplot has hidden x/y tick labels")
    if any(not axis.get_xlabel() or not axis.get_ylabel() for axis in axes.ravel()):
        raise AssertionError("At least one atlas subplot is missing an x/y axis label")
    for row in axes:
        for axis, expected_count in zip(row[3:], (2, 3, 2)):
            legend = axis.get_legend()
            if legend is None or len(legend.get_texts()) != expected_count:
                raise AssertionError("An ECDF subplot is missing its method legend")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The grid margins already reserve all labels/colorbars.  Avoiding the
    # expensive tight-bbox traversal is important with 30 inset colorbars.
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "dataset": dataset,
        "rows": 10,
        "columns": 6,
        "vector_panel_count": 30,
        "colorbar_count": len(colorbars),
        "orientation": orientation,
        "initial_color_limit_mm": initial_limit,
        "residual_color_limit_mm": residual_limit,
        "registered_p99_limit_mm": registered_p99,
        "full_tre_x_limit_mm": full_tre_limit,
        "applied_motion_x_limit_mm": applied_limit,
        "output_path": output_path.name,
    }


def summary_rows(cases: list[CaseAtlas]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "dataset": case.dataset,
            "case_id": case.case_id,
            "landmarks": int(case.initial_tre_mm.size),
            "initial_tre_mean_mm": float(np.mean(case.initial_tre_mm)),
            "initial_tre_p95_mm": float(np.percentile(case.initial_tre_mm, 95)),
            "ptvreg_tre_mean_mm": float(np.mean(case.ptv_tre_mm)),
            "ptvreg_tre_p95_mm": float(np.percentile(case.ptv_tre_mm, 95)),
            "p2e_tre_mean_mm": float(np.mean(case.p2e_tre_mm)),
            "p2e_tre_p95_mm": float(np.percentile(case.p2e_tre_mm, 95)),
            "ptvreg_applied_motion_p95_mm": float(np.percentile(case.ptv_applied_mm, 95)),
            "p2e_applied_motion_p95_mm": float(np.percentile(case.p2e_applied_mm, 95)),
            "native_spacing_zyx_mm": ",".join(f"{v:g}" for v in case.native_spacing_zyx_mm),
            "crop_shape_zyx": "x".join(str(v) for v in case.crop_shape_zyx),
            "pTVReg_source": case.source_refs["pTVReg"],
            "P2e_source": case.source_refs["MR-D-BSCP P2e"],
        }
        row.update(case.validation)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def orientation_self_check() -> None:
    orientation = coronal_orientation_from_audit(
        {"raw_array_order": "zyx", "landmark_saved_order": "zyx, Python 0-based voxel coordinates"},
        "synthetic",
        0,
    )
    points_zyx = np.asarray([[1.0, 2.0, 3.0]])
    spacing = np.asarray([2.0, 3.0, 4.0])
    xyz = zyx_local_mm(points_zyx, spacing)[:, [2, 1, 0]]
    if not np.allclose(xyz, [[12.0, 6.0, 2.0]]):
        raise AssertionError(f"zyx-to-xyz orientation self-check failed: {xyz}")
    vector_zyx = np.asarray([[1.0, 2.0, 3.0]]) * spacing[None, :]
    vector_xyz = vector_zyx[:, [2, 1, 0]]
    if not np.allclose(vector_xyz[:, [0, 2]], [[12.0, 2.0]]):
        raise AssertionError(f"coronal x/z projection self-check failed: {vector_xyz}")
    # In the display frame, increasing local x moves from patient R to L and
    # decreasing local z moves from patient I to S (the raw z-row flip).
    patient_right = np.asarray([[0.0, 0.0, 10.0]])
    patient_left = np.asarray([[10.0, 0.0, 10.0]])
    patient_inferior = np.asarray([[0.0, 0.0, 10.0]])
    patient_superior = np.asarray([[0.0, 0.0, 0.0]])

    def project(points_xyz_mm: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                orientation["horizontal_display_sign"] * points_xyz_mm[:, 0],
                orientation["vertical_display_sign"] * points_xyz_mm[:, 2],
            )
        )

    right_xy, left_xy = project(patient_right)[0], project(patient_left)[0]
    inferior_xy, superior_xy = project(patient_inferior)[0], project(patient_superior)[0]
    if not (left_xy[0] > right_xy[0]):
        raise AssertionError(f"patient L is not on the display right: R={right_xy}, L={left_xy}")
    if not (superior_xy[1] > inferior_xy[1]):
        raise AssertionError(f"patient S is not on the display top: I={inferior_xy}, S={superior_xy}")
    transformed_vector = vector_xyz.copy()
    transformed_vector[:, 0] *= orientation["horizontal_display_sign"]
    transformed_vector[:, 2] *= orientation["vertical_display_sign"]
    if not np.isclose(np.linalg.norm(vector_xyz[0]), np.linalg.norm(transformed_vector[0])):
        raise AssertionError("orientation display transform changed vector magnitude")


def render_all(output_dir: Path | str | None = None) -> dict[str, Any]:
    orientation_self_check()
    data_root = required_env("INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT")
    ptv_roots = {"4dct": required_env("PTVREG_4DCT_ROOT"), "copd": required_env("PTVREG_COPD_ROOT")}
    p2e_root = required_env("MRDBSCP_P2E_ACCEL_ROOT")
    out_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_by_dataset: dict[str, list[CaseAtlas]] = {}
    for dataset in ("4dct", "copd"):
        cases_by_dataset[dataset] = []
        for case_id in CASE_IDS:
            cases_by_dataset[dataset].append(
                build_case(data_root, ptv_roots[dataset], p2e_root, dataset, case_id)
            )
            print(f"loaded {dataset} case{case_id:02d}", flush=True)

    atlas_info: list[dict[str, Any]] = []
    for dataset in ("4dct", "copd"):
        output_path = out_dir / f"{dataset}_ptvreg_vs_mrdbscp_p2e_landmark_atlas.png"
        atlas_info.append(render_dataset(cases_by_dataset[dataset], output_path))
        print(f"rendered {dataset} atlas", flush=True)

    rows = [row for dataset in ("4dct", "copd") for row in summary_rows(cases_by_dataset[dataset])]
    summary_path = out_dir / "casewise_landmark_atlas_summary.csv"
    write_csv(summary_path, rows)

    source_manifest: list[dict[str, Any]] = []
    for dataset in ("4dct", "copd"):
        for case in cases_by_dataset[dataset]:
            source_manifest.append(
                {
                    "dataset": dataset,
                    "case_id": case.case_id,
                    "pTVReg_source_ref": case.source_refs["pTVReg"],
                    "P2e_source_ref": case.source_refs["MR-D-BSCP P2e"],
                    "p2e_variant": P2E_VARIANTS[dataset],
                    "landmarks": 300,
                }
            )
    provenance = {
        "artifact": "landmark_residual_atlas",
        "created_by": "landmark_residual_atlas.py",
        "datasets": ["4dct", "copd"],
        "cases": list(CASE_IDS),
        "p2e_experiment_id": P2E_EXPERIMENT_ID,
        "p2e_case_run_prefix": P2E_RUN_PREFIX,
        "p2e_method_dir": P2E_METHOD_DIR,
        "p2e_variants": P2E_VARIANTS,
        "metric": {
            "headline_domain": "native_full_or_crop",
            "headline_mode": "snap_to_voxel",
            "registered_tre_ecdf_x_limit": "pooled_dataset_p99_across_pTVReg_and_P2e",
            "applied_motion_domain": "fixed_lung_mask_on_canonical_reg1mm_grid",
            "vector_projection": "coronal_display_x_plus_x_z_minus_z",
            "orientation": CORONAL_ORIENTATION,
            "vector_magnitude": "full_3d_physical_mm",
        },
        "required_environment": {
            "INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT": bool(os.environ.get("INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT")),
            "PTVREG_4DCT_ROOT": bool(os.environ.get("PTVREG_4DCT_ROOT")),
            "PTVREG_COPD_ROOT": bool(os.environ.get("PTVREG_COPD_ROOT")),
            "MRDBSCP_P2E_ACCEL_ROOT": bool(os.environ.get("MRDBSCP_P2E_ACCEL_ROOT")),
        },
        "source_manifest": source_manifest,
        "atlas_info": atlas_info,
        "summary_csv": summary_path.name,
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "cases_by_dataset": cases_by_dataset,
        "atlas_info": atlas_info,
        "summary_rows": rows,
        "summary_path": summary_path,
        "provenance_path": provenance_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args()
    result = render_all(args.output_dir)
    print(json.dumps({"atlas_info": result["atlas_info"], "summary_path": str(result["summary_path"]), "provenance_path": str(result["provenance_path"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
