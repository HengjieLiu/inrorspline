#!/usr/bin/env python
"""Build coronal motion-pattern panels for paper figure review."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
from matplotlib.collections import LineCollection

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
import torch  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
OASIS_DIR = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep"
LUNGCT_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
DEFAULT_OUTPUT_DIR = THIS_DIR / "v001"
DEFAULT_SITREG_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_OASIS_SITREG_ROOT",
        "/database/wavereg/wipmidl_0407/OASIS_v1/round1_MIDL/"
        "gpu8_OASIS_v1_bs4_sitreg_v0_run1_20250407165940/inference/inference_bestdice",
    )
)
DEFAULT_PTVREG_ROOT = Path(os.environ.get("INRORSPLINE_PTVREG_4DCT_ROOT", "/database/inrorspline/ptvreg/4DCT"))
OASIS_SPACING_ZYX_MM = (1.0, 1.0, 1.0)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oasis_common = load_module("inrorspline_oasis_common_for_motion_pattern", OASIS_DIR / "common.py")

if str(LUNGCT_DIR) not in sys.path:
    sys.path.insert(0, str(LUNGCT_DIR))
import common as lungct_common  # noqa: E402
import run_case01_ptvreg_field_fit_one as ptvreg_field_fit  # noqa: E402


PUBLIC_PATH_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/homebase/code_sync/inrorspline"), "${REPO_ROOT}"),
    (re.compile(r"/home/[^/]+/code_sync/inrorspline"), "${REPO_ROOT}"),
    (re.compile(r"/database/inrorspline"), "${DATA_ROOT}"),
    (re.compile(r"/data/[^/]+/inrorspline"), "${DATA_ROOT}"),
    (re.compile(r"/database/wavereg"), "${WAVEREG_DATA_ROOT}"),
    (re.compile(r"/data/[^/]+/wavereg"), "${WAVEREG_DATA_ROOT}"),
    (re.compile(r"/database/datasets/lungCT"), "${LUNGCT_DATA_ROOT}"),
    (re.compile(r"/data/[^/]+/datasets/lungCT"), "${LUNGCT_DATA_ROOT}"),
]


@dataclass
class DisplayForwardGrid:
    x: np.ndarray
    y: np.ndarray
    magnitude_mm: np.ndarray
    display_shape: tuple[int, int]
    step: int
    sample_count: int
    component_mapping: str
    color_max_mm: float


@dataclass
class DisplayVectorField:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    magnitude: np.ndarray
    unit: str
    color_max: float
    arrow_scale: float
    step: int
    sample_count: int
    component_mapping: str


@dataclass
class MotionCase:
    case_key: str
    row_label: str
    dataset: str
    reference: str
    fixed_slice: np.ndarray
    moving_slice: np.ndarray
    deformed_slice: np.ndarray
    backward_grid_slice: np.ndarray
    forward_grid: DisplayForwardGrid
    vectors: DisplayVectorField
    aspect: float
    summary_color_max_mm: float
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedSITRegField:
    disp_zyx: np.ndarray
    raw_shape: tuple[int, ...]
    storage_layout: str
    component_order: str
    reorientation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-step", type=int, default=4)
    parser.add_argument("--vector-step", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--oasis-root", type=Path, default=oasis_common.DEFAULT_OASIS_ROOT)
    parser.add_argument("--oasis-pair-file", type=Path, default=oasis_common.DEFAULT_PAIR_FILE)
    parser.add_argument("--sitreg-root", type=Path, default=DEFAULT_SITREG_ROOT)
    parser.add_argument("--lungct-ptvprep-root", type=Path, default=lungct_common.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--ptvreg-root", type=Path, default=DEFAULT_PTVREG_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def device_from_arg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pattern, replacement in PUBLIC_PATH_REPLACEMENTS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, Path):
        return sanitize_public_value(str(value))
    if isinstance(value, Mapping):
        return {str(k): sanitize_public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_public_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_public_value(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_public_value(payload), indent=2, allow_nan=True, default=str) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (dict, list, tuple)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize_public_value(row.get(key, "")) for key in fieldnames})


def normalize_for_display(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def display_window(images: Sequence[np.ndarray]) -> tuple[float, float]:
    finite_parts = [np.asarray(image)[np.isfinite(image)] for image in images]
    finite = np.concatenate([part.reshape(-1) for part in finite_parts if part.size])
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    return float(lo), float(hi)


def robust_color_max(values: Sequence[np.ndarray], percentile: float = 98.0) -> float:
    finite_parts = [np.asarray(value, dtype=np.float32)[np.isfinite(value)] for value in values]
    finite = np.concatenate([part.reshape(-1) for part in finite_parts if part.size])
    finite = finite[finite >= 0.0]
    if finite.size == 0:
        return 1.0
    out = float(np.percentile(finite, percentile))
    if not math.isfinite(out) or out <= 0.0:
        out = float(np.max(finite))
    if not math.isfinite(out) or out <= 0.0:
        out = 1.0
    return out


def bbox_center_axis(mask: np.ndarray, axis: int) -> int:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if not np.any(mask_bool):
        return int(mask_bool.shape[axis] // 2)
    coords = np.argwhere(mask_bool)
    return int(round(float(coords[:, axis].min() + coords[:, axis].max()) / 2.0))


def make_grid_image(shape_zyx: Sequence[int], axis_to_slice: int, grid_step: int) -> np.ndarray:
    if grid_step < 1:
        raise ValueError("--grid-step must be >= 1")
    shape = tuple(int(v) for v in shape_zyx)
    grid = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        if axis == int(axis_to_slice):
            continue
        for index in range(0, shape[axis], int(grid_step)):
            slicer = [slice(None)] * 3
            slicer[axis] = index
            grid[tuple(slicer)] = 1.0
    return grid


def lattice_axis(length: int, step: int) -> np.ndarray:
    if step < 1:
        raise ValueError("--grid-step must be >= 1")
    if length < 1:
        raise ValueError(f"Grid axis length must be positive, got {length}")
    coords = np.arange(0, int(length), int(step), dtype=np.float32)
    last = np.float32(int(length) - 1)
    if coords.size == 0 or coords[-1] != last:
        coords = np.concatenate([coords, np.asarray([last], dtype=np.float32)])
    return coords


def display_forward_grid_from_disp_zyx(
    disp_zyx: np.ndarray,
    index_y: int,
    grid_step: int,
    *,
    display_kind: str,
    spacing_zyx_mm: Sequence[float],
) -> DisplayForwardGrid:
    disp = np.asarray(disp_zyx, dtype=np.float32)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"Expected displacement shape [3,D,H,W], got {disp.shape}")
    d, h, w = (int(v) for v in disp.shape[1:])
    if not (0 <= int(index_y) < h):
        raise ValueError(f"Coronal index {index_y} outside y dimension {h}")
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float32)
    if spacing.shape != (3,):
        raise ValueError(f"Expected spacing_zyx_mm length 3, got {spacing_zyx_mm}")

    grid_z, grid_x = np.meshgrid(lattice_axis(d, grid_step), lattice_axis(w, grid_step), indexing="ij")
    grid_y = np.full_like(grid_z, float(index_y), dtype=np.float32)
    sample_coords = (
        np.clip(grid_z, 0, d - 1),
        np.clip(grid_y, 0, h - 1),
        np.clip(grid_x, 0, w - 1),
    )
    dz = map_coordinates(disp[0], sample_coords, order=0, mode="nearest")
    dy = map_coordinates(disp[1], sample_coords, order=0, mode="nearest")
    dx = map_coordinates(disp[2], sample_coords, order=0, mode="nearest")
    magnitude_mm = np.sqrt(
        np.square(dz * spacing[0]) + np.square(dy * spacing[1]) + np.square(dx * spacing[2])
    )

    if display_kind == "oasis_legacy_coronal":
        x_display = (d - 1) - (grid_z + dz)
        y_display = grid_x + dx
        display_shape = (w, d)
        component_mapping = "display_x=(D-1)-(z+disp_z), display_y=x+disp_x after np.fliplr(slice.T)"
    elif display_kind == "lungct_coronal_zflip":
        x_display = grid_x + dx
        y_display = (d - 1) - (grid_z + dz)
        display_shape = (d, w)
        component_mapping = "display_x=x+disp_x, display_y=(D-1)-(z+disp_z) after arr[:, y, :] z-flip"
    else:
        raise ValueError(f"Unknown forward-grid display kind: {display_kind}")

    return DisplayForwardGrid(
        x=np.asarray(x_display, dtype=np.float32),
        y=np.asarray(y_display, dtype=np.float32),
        magnitude_mm=np.asarray(magnitude_mm, dtype=np.float32),
        display_shape=tuple(int(v) for v in display_shape),
        step=int(grid_step),
        sample_count=int(x_display.size),
        component_mapping=component_mapping,
        color_max_mm=robust_color_max([magnitude_mm]),
    )


def self_check_forward_grid_helpers() -> None:
    shape = (9, 7, 11)
    index_y = 3
    step = 4
    zero = np.zeros((3, *shape), dtype=np.float32)
    oasis_zero = display_forward_grid_from_disp_zyx(
        zero,
        index_y,
        step,
        display_kind="oasis_legacy_coronal",
        spacing_zyx_mm=(1.0, 1.0, 1.0),
    )
    lung_zero = display_forward_grid_from_disp_zyx(
        zero,
        index_y,
        step,
        display_kind="lungct_coronal_zflip",
        spacing_zyx_mm=(1.0, 1.0, 1.0),
    )
    z_coords, x_coords = np.meshgrid(lattice_axis(shape[0], step), lattice_axis(shape[2], step), indexing="ij")
    if not np.allclose(oasis_zero.x, (shape[0] - 1) - z_coords) or not np.allclose(oasis_zero.y, x_coords):
        raise RuntimeError("OASIS forward-grid self-check failed for zero displacement.")
    if not np.allclose(lung_zero.x, x_coords) or not np.allclose(lung_zero.y, (shape[0] - 1) - z_coords):
        raise RuntimeError("LungCT forward-grid self-check failed for zero displacement.")
    const = zero.copy()
    const[0] = 1.0
    const[2] = 2.0
    oasis_const = display_forward_grid_from_disp_zyx(
        const,
        index_y,
        step,
        display_kind="oasis_legacy_coronal",
        spacing_zyx_mm=(1.0, 1.0, 1.0),
    )
    lung_const = display_forward_grid_from_disp_zyx(
        const,
        index_y,
        step,
        display_kind="lungct_coronal_zflip",
        spacing_zyx_mm=(1.0, 1.0, 1.0),
    )
    if not np.allclose(oasis_const.x - oasis_zero.x, -1.0) or not np.allclose(oasis_const.y - oasis_zero.y, 2.0):
        raise RuntimeError("OASIS forward-grid self-check failed for constant displacement.")
    if not np.allclose(lung_const.x - lung_zero.x, 2.0) or not np.allclose(lung_const.y - lung_zero.y, -1.0):
        raise RuntimeError("LungCT forward-grid self-check failed for constant displacement.")
    if not np.allclose(oasis_const.magnitude_mm, np.sqrt(5.0)):
        raise RuntimeError("OASIS forward-grid self-check failed for constant displacement magnitude.")
    if not np.allclose(lung_const.magnitude_mm, np.sqrt(5.0)):
        raise RuntimeError("LungCT forward-grid self-check failed for constant displacement magnitude.")


def make_display_vectors(
    u: np.ndarray,
    v: np.ndarray,
    magnitude: np.ndarray,
    mask: np.ndarray,
    *,
    step: int,
    unit: str,
    component_mapping: str,
) -> DisplayVectorField:
    if step < 1:
        raise ValueError("--vector-step must be >= 1")
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    magnitude = np.asarray(magnitude, dtype=np.float32)
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if u.shape != v.shape or u.shape != magnitude.shape or u.shape != mask_bool.shape:
        raise ValueError(
            "Displayed vector arrays must have matching shapes: "
            f"u={u.shape}, v={v.shape}, magnitude={magnitude.shape}, mask={mask_bool.shape}"
        )

    rows, cols = magnitude.shape
    offset = max(int(step) // 2, 0)
    row_idx = np.arange(offset, rows, int(step), dtype=np.int64)
    col_idx = np.arange(offset, cols, int(step), dtype=np.int64)
    if row_idx.size == 0:
        row_idx = np.arange(0, rows, max(rows, 1), dtype=np.int64)
    if col_idx.size == 0:
        col_idx = np.arange(0, cols, max(cols, 1), dtype=np.int64)

    rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")
    finite = np.isfinite(u[rr, cc]) & np.isfinite(v[rr, cc]) & np.isfinite(magnitude[rr, cc])
    selected = mask_bool[rr, cc] & finite
    if not np.any(selected):
        selected = finite

    x = cc[selected].astype(np.float32)
    y = rr[selected].astype(np.float32)
    u_sample = u[rr, cc][selected].astype(np.float32)
    v_sample = v[rr, cc][selected].astype(np.float32)
    mag_sample = magnitude[rr, cc][selected].astype(np.float32)
    finite_mag = mag_sample[np.isfinite(mag_sample)]
    if finite_mag.size == 0:
        color_max = 1.0
        p95 = 1.0
    else:
        color_max = float(np.percentile(finite_mag, 98.0))
        p95 = float(np.percentile(finite_mag, 95.0))
        if not math.isfinite(color_max) or color_max <= 0:
            color_max = float(np.max(finite_mag)) if finite_mag.size else 1.0
        if not math.isfinite(color_max) or color_max <= 0:
            color_max = 1.0
        if not math.isfinite(p95) or p95 <= 0:
            p95 = color_max
    target_arrow_length = max(float(step) * 0.75, 1.0)
    arrow_scale = max(1.0, p95 / target_arrow_length)
    return DisplayVectorField(
        x=x,
        y=y,
        u=u_sample,
        v=v_sample,
        magnitude=mag_sample,
        unit=unit,
        color_max=color_max,
        arrow_scale=arrow_scale,
        step=int(step),
        sample_count=int(x.size),
        component_mapping=component_mapping,
    )


def oasis_legacy_coronal_slice(volume_zyx: np.ndarray, index_y: int) -> np.ndarray:
    return np.fliplr(np.asarray(volume_zyx[:, index_y, :]).T)


def lungct_coronal_slice(volume_zyx: np.ndarray, index_y: int) -> np.ndarray:
    return np.flip(np.asarray(volume_zyx[:, index_y, :]), axis=0)


def norm_xyz_to_voxel_zyx(disp_norm_xyz: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    d, h, w = (int(v) for v in shape_zyx)
    disp = np.asarray(disp_norm_xyz, dtype=np.float32)
    expected = (d, h, w, 3)
    if disp.shape != expected:
        raise ValueError(f"Expected normalized xyz displacement shape {expected}, got {disp.shape}")
    out = np.empty((3, d, h, w), dtype=np.float32)
    out[0] = disp[..., 2] * ((d - 1) / 2.0)
    out[1] = disp[..., 1] * ((h - 1) / 2.0)
    out[2] = disp[..., 0] * ((w - 1) / 2.0)
    return out


def oasis_display_vectors(
    disp_zyx: np.ndarray,
    fixed_mask: np.ndarray,
    index_y: int,
    vector_step: int,
    spacing_zyx_mm: Sequence[float],
) -> DisplayVectorField:
    disp = np.asarray(disp_zyx, dtype=np.float32)
    expected = (1, 3, *oasis_common.IMAGE_SHAPE)
    if disp.shape != expected:
        raise ValueError(f"Expected OASIS displacement shape {expected}, got {disp.shape}")
    disp = disp[0]
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float32)
    if spacing.shape != (3,):
        raise ValueError(f"Expected spacing_zyx_mm length 3, got {spacing_zyx_mm}")
    dz = disp[0]
    dx = disp[2]
    magnitude_mm = np.sqrt(np.sum(np.square(disp * spacing[:, None, None, None]), axis=0))
    mask_slice = oasis_legacy_coronal_slice(np.asarray(fixed_mask, dtype=np.float32), index_y) > 0.5
    return make_display_vectors(
        u=oasis_legacy_coronal_slice(-dz, index_y),
        v=oasis_legacy_coronal_slice(dx, index_y),
        magnitude=oasis_legacy_coronal_slice(magnitude_mm, index_y),
        mask=mask_slice,
        step=vector_step,
        unit="mm",
        component_mapping="display_u=-disp_z_voxel, display_v=disp_x_voxel after np.fliplr(slice.T); color=3D magnitude_mm",
    )


def lungct_display_vectors(
    disp_norm_xyz: np.ndarray,
    fixed_mask: np.ndarray,
    index_y: int,
    spacing_zyx: Sequence[float],
    vector_step: int,
) -> DisplayVectorField:
    disp_zyx_voxel = norm_xyz_to_voxel_zyx(disp_norm_xyz, fixed_mask.shape)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if spacing.shape != (3,):
        raise ValueError(f"Expected spacing_zyx length 3, got {spacing_zyx}")
    dz = disp_zyx_voxel[0]
    dx = disp_zyx_voxel[2]
    magnitude_mm = np.sqrt(np.sum(np.square(disp_zyx_voxel * spacing[:, None, None, None]), axis=0))
    mask_slice = lungct_coronal_slice(np.asarray(fixed_mask, dtype=np.float32), index_y) > 0.5
    return make_display_vectors(
        u=lungct_coronal_slice(dx, index_y),
        v=lungct_coronal_slice(-dz, index_y),
        magnitude=lungct_coronal_slice(magnitude_mm, index_y),
        mask=mask_slice,
        step=vector_step,
        unit="mm",
        component_mapping="display_u=disp_x, display_v=-disp_z after arr[:, y, :] z-flip",
    )


def load_sitreg_disp(path: Path) -> LoadedSITRegField:
    if not path.exists():
        raise FileNotFoundError(f"Missing SITReg displacement field: {path}")
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    expected_shape = (3, *oasis_common.IMAGE_SHAPE)
    if arr.shape != expected_shape:
        raise ValueError(f"Expected SITReg field shape {expected_shape}, got {arr.shape} from {path}")
    return LoadedSITRegField(
        disp_zyx=arr[None],
        raw_shape=tuple(int(v) for v in arr.shape),
        storage_layout="channel_first_zyx_repo_ras",
        component_order="zyx_voxel",
        reorientation="none",
    )


def oasis_voxel_zyx_to_norm_xyz(disp_zyx: np.ndarray) -> np.ndarray:
    if disp_zyx.shape != (1, 3, *oasis_common.IMAGE_SHAPE):
        raise ValueError(f"Expected [1,3,{oasis_common.IMAGE_SHAPE}], got {disp_zyx.shape}")
    d, h, w = oasis_common.IMAGE_SHAPE
    disp = disp_zyx[0]
    out = np.empty((*oasis_common.IMAGE_SHAPE, 3), dtype=np.float32)
    out[..., 0] = disp[2] / ((w - 1) / 2.0)
    out[..., 1] = disp[1] / ((h - 1) / 2.0)
    out[..., 2] = disp[0] / ((d - 1) / 2.0)
    return out


def warp_with_module(
    module: Any,
    volume: np.ndarray,
    disp_norm_xyz: np.ndarray,
    device: torch.device,
    *,
    mode: str,
) -> np.ndarray:
    shape = tuple(int(v) for v in volume.shape)
    id_grid = module.make_coordinate_grid(shape, device)
    disp = torch.from_numpy(np.asarray(disp_norm_xyz, dtype=np.float32)).to(device=device)
    vol = torch.from_numpy(np.asarray(volume, dtype=np.float32)).to(device=device)
    with torch.no_grad():
        warped = module.warp_full(vol, id_grid + disp, mode=mode)
    return warped.detach().cpu().numpy().astype(np.float32, copy=False)


def build_oasis_case(args: argparse.Namespace, device: torch.device) -> MotionCase:
    batch = oasis_common.load_oasis_case(
        rank=9,
        device=torch.device("cpu"),
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.oasis_pair_file),
    )
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    if (fixed_id, moving_id) != ("0173", "0136"):
        raise RuntimeError(f"Unexpected OASIS rank-9 pair: fixed={fixed_id}, moving={moving_id}")
    fixed = np.asarray(batch["fixed"].cpu().numpy(), dtype=np.float32)
    moving = np.asarray(batch["moving"].cpu().numpy(), dtype=np.float32)
    fixed_mask = np.asarray(batch["fixed_image_mask"].cpu().numpy(), dtype=bool)
    sitreg_path = Path(args.sitreg_root) / f"disp_{fixed_id}_{moving_id}.nii.gz"
    loaded = load_sitreg_disp(sitreg_path)
    disp_norm = oasis_voxel_zyx_to_norm_xyz(loaded.disp_zyx)
    deformed = warp_with_module(oasis_common, moving, disp_norm, device, mode="bilinear")
    grid = make_grid_image(fixed.shape, axis_to_slice=1, grid_step=int(args.grid_step))
    deformed_grid = warp_with_module(oasis_common, grid, disp_norm, device, mode="bilinear")
    index_y = bbox_center_axis(fixed_mask, axis=1)
    lo, hi = display_window([fixed, moving, deformed])
    fixed_slice = normalize_for_display(oasis_legacy_coronal_slice(fixed, index_y), lo, hi)
    moving_slice = normalize_for_display(oasis_legacy_coronal_slice(moving, index_y), lo, hi)
    deformed_slice = normalize_for_display(oasis_legacy_coronal_slice(deformed, index_y), lo, hi)
    backward_grid_slice = oasis_legacy_coronal_slice(deformed_grid, index_y)
    forward_grid = display_forward_grid_from_disp_zyx(
        loaded.disp_zyx[0],
        index_y,
        int(args.grid_step),
        display_kind="oasis_legacy_coronal",
        spacing_zyx_mm=OASIS_SPACING_ZYX_MM,
    )
    vectors = oasis_display_vectors(loaded.disp_zyx, fixed_mask, index_y, int(args.vector_step), OASIS_SPACING_ZYX_MM)
    summary_color_max_mm = max(forward_grid.color_max_mm, vectors.color_max)
    manifest = {
        "case_key": "oasis_rank009",
        "dataset": "oasis",
        "display_label": "OASIS rank 9",
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "case_rank": int(batch["case_rank"]),
        "original_case_index": int(batch["original_case_index"]),
        "pair_file": str(batch["pair_file"]),
        "source_field_path": str(sitreg_path),
        "source_field": "bs4_sitreg_v0",
        "source_field_storage_layout": loaded.storage_layout,
        "source_field_component_order": loaded.component_order,
        "source_field_reorientation": loaded.reorientation,
        "source_field_units": "voxel",
        "source_field_direction": "fixed_to_moving_pull_field_for_warping_moving_to_fixed",
        "source_field_raw_shape": list(loaded.raw_shape),
        "field_shape_zyx": list(fixed.shape),
        "registration_spacing_zyx_mm": list(OASIS_SPACING_ZYX_MM),
        "slice_view": "coronal",
        "slice_axis_zyx": 1,
        "slice_index": int(index_y),
        "slice_selection": "fixed raw image-positive foreground bounding-box center",
        "display_orientation": "OASIS LIA-to-RAS loader output, then legacy np.fliplr(slice.T) display transform",
        "display_aspect": 1.0,
        "grid_step_voxels": int(args.grid_step),
        "backward_grid_mode": "image_warped_pull_grid",
        "backward_grid_direction": "synthetic raster grid sampled through fixed-to-moving pull field",
        "forward_grid_mode": "coordinate_lattice_identity_plus_displacement",
        "forward_grid_direction": "sampled fixed-coordinate lattice plus stored pull displacement",
        "forward_grid_component_mapping": forward_grid.component_mapping,
        "forward_grid_sample_count": forward_grid.sample_count,
        "forward_grid_color_quantity": "full 3D displacement magnitude",
        "forward_grid_color_units": "mm",
        "forward_grid_color_max_mm": forward_grid.color_max_mm,
        "vector_step_voxels": int(args.vector_step),
        "vector_arrow_components": "in-plane coronal displayed components",
        "vector_color_quantity": "full 3D displacement magnitude",
        "vector_color_units": vectors.unit,
        "vector_component_mapping": vectors.component_mapping,
        "vector_arrow_scale": vectors.arrow_scale,
        "vector_color_max": vectors.color_max,
        "vector_sample_count": vectors.sample_count,
        "summary_color_max_mm": summary_color_max_mm,
        "summary_color_scale_scope": "per_case_shared_by_forward_grid_and_quiver",
    }
    return MotionCase(
        case_key="oasis_rank009",
        row_label="OASIS rank 9\nSITReg",
        dataset="OASIS",
        reference="SITReg",
        fixed_slice=fixed_slice,
        moving_slice=moving_slice,
        deformed_slice=deformed_slice,
        backward_grid_slice=backward_grid_slice,
        forward_grid=forward_grid,
        vectors=vectors,
        aspect=1.0,
        summary_color_max_mm=summary_color_max_mm,
        manifest=manifest,
    )


def ptvreg_mat_path(root: Path, case_id: int) -> Path:
    return Path(root) / f"case{int(case_id):02d}_T50_to_T00_deformation.mat"


def build_lungct_case(args: argparse.Namespace, case_id: int, device: torch.device) -> MotionCase:
    case = lungct_common.load_case(Path(args.lungct_ptvprep_root), int(case_id))
    mat_path = ptvreg_mat_path(Path(args.ptvreg_root), int(case_id))
    target_args = SimpleNamespace(
        ptvreg_mat=mat_path,
        case_id=int(case_id),
        expected_tre_phys=None,
        expected_tre_std=None,
    )
    disp_norm, target_audit = ptvreg_field_fit.load_ptvreg_target(target_args, case)
    fixed = case.fixed.astype(np.float32, copy=False)
    moving = case.moving.astype(np.float32, copy=False)
    deformed = warp_with_module(lungct_common, moving, disp_norm, device, mode="bilinear")
    grid = make_grid_image(fixed.shape, axis_to_slice=1, grid_step=int(args.grid_step))
    deformed_grid = warp_with_module(lungct_common, grid, disp_norm, device, mode="bilinear")
    index_y = bbox_center_axis(case.fixed_mask.astype(bool), axis=1)
    lo, hi = display_window([fixed, moving, deformed])
    fixed_slice = normalize_for_display(lungct_coronal_slice(fixed, index_y), lo, hi)
    moving_slice = normalize_for_display(lungct_coronal_slice(moving, index_y), lo, hi)
    deformed_slice = normalize_for_display(lungct_coronal_slice(deformed, index_y), lo, hi)
    backward_grid_slice = lungct_coronal_slice(deformed_grid, index_y)
    spacing_zyx = case.preprocess_audit.get("registration_spacing_zyx_mm", [1.0, 1.0, 1.0])
    aspect = float(spacing_zyx[0]) / float(spacing_zyx[2])
    disp_zyx_voxel = norm_xyz_to_voxel_zyx(disp_norm, fixed.shape)
    forward_grid = display_forward_grid_from_disp_zyx(
        disp_zyx_voxel,
        index_y,
        int(args.grid_step),
        display_kind="lungct_coronal_zflip",
        spacing_zyx_mm=spacing_zyx,
    )
    vectors = lungct_display_vectors(disp_norm, case.fixed_mask.astype(bool), index_y, spacing_zyx, int(args.vector_step))
    summary_color_max_mm = max(forward_grid.color_max_mm, vectors.color_max)
    manifest = {
        "case_key": f"lungct_case{case_id:02d}",
        "dataset": "lungct_4dct",
        "display_label": f"LungCT case {case_id:02d}",
        "fixed_phase": "T00 end-inhale",
        "moving_phase": "T50 end-exhale",
        "registration_direction": "T50 moving to T00 fixed",
        "source_field_path": str(mat_path),
        "source_field": "pTVReg",
        "source_field_h5py_layout": target_audit["source_disp_h5py_layout"],
        "source_field_matlab_layout": target_audit["source_disp_matlab_layout"],
        "source_field_component_order": target_audit["source_disp_component_order"],
        "source_field_units": target_audit["source_displacement_units"],
        "source_field_direction": target_audit["source_transform_direction"],
        "source_saved_tre_phys_mean_mm": target_audit["source_saved_tre_phys_mean_mm"],
        "source_recomputed_snap_tre_mean_mm": target_audit["source_recomputed_snap_tre_mean_mm"],
        "source_tre_validation_abs_err": target_audit["source_tre_validation_abs_err"],
        "case_dir": str(case.case_dir),
        "field_shape_zyx": list(fixed.shape),
        "native_spacing_zyx_mm": case.preprocess_audit.get("native_spacing_zyx_mm"),
        "registration_spacing_zyx_mm": spacing_zyx,
        "crop_bounds_zyx_0based_inclusive": target_audit["crop_bounds_zyx_0based_inclusive"],
        "slice_view": "coronal",
        "slice_axis_zyx": 1,
        "slice_index": int(index_y),
        "slice_selection": "fixed T00 lung-mask bounding-box center",
        "display_orientation": "pTV-prep zyx array, coronal arr[:, y, :], flipped along z for superior-at-top display",
        "display_aspect": aspect,
        "grid_step_voxels": int(args.grid_step),
        "backward_grid_mode": "image_warped_pull_grid",
        "backward_grid_direction": "synthetic raster grid sampled through fixed_T00_to_moving_T50 pull field",
        "forward_grid_mode": "coordinate_lattice_identity_plus_displacement",
        "forward_grid_direction": "sampled fixed T00 crop-coordinate lattice plus pTVReg pull displacement",
        "forward_grid_component_mapping": forward_grid.component_mapping,
        "forward_grid_sample_count": forward_grid.sample_count,
        "forward_grid_color_quantity": "full 3D displacement magnitude",
        "forward_grid_color_units": "mm",
        "forward_grid_color_max_mm": forward_grid.color_max_mm,
        "vector_step_voxels": int(args.vector_step),
        "vector_arrow_components": "in-plane coronal displayed components",
        "vector_color_quantity": "full 3D displacement magnitude",
        "vector_color_units": vectors.unit,
        "vector_component_mapping": vectors.component_mapping,
        "vector_arrow_scale": vectors.arrow_scale,
        "vector_color_max": vectors.color_max,
        "vector_sample_count": vectors.sample_count,
        "summary_color_max_mm": summary_color_max_mm,
        "summary_color_scale_scope": "per_case_shared_by_forward_grid_and_quiver",
    }
    return MotionCase(
        case_key=f"lungct_case{case_id:02d}",
        row_label=f"LungCT case {case_id:02d}\npTVReg",
        dataset="LungCT",
        reference="pTVReg",
        fixed_slice=fixed_slice,
        moving_slice=moving_slice,
        deformed_slice=deformed_slice,
        backward_grid_slice=backward_grid_slice,
        forward_grid=forward_grid,
        vectors=vectors,
        aspect=aspect,
        summary_color_max_mm=summary_color_max_mm,
        manifest=manifest,
    )


def plot_cases(cases: Sequence[MotionCase], out_path: Path, *, composite: bool) -> None:
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 5, figsize=(14.8, 2.75 * nrows), squeeze=False, constrained_layout=True)
    col_titles = ["Fixed", "Moving", "Deformed", "Backward grid", "Forward grid"]
    for row, case in enumerate(cases):
        arrays = [
            case.fixed_slice,
            case.moving_slice,
            case.deformed_slice,
            np.clip(case.backward_grid_slice, 0.0, 1.0),
        ]
        cmaps = ["gray", "gray", "gray", "gray"]
        for col, (arr, cmap) in enumerate(zip(arrays, cmaps)):
            ax = axes[row, col]
            ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", aspect=case.aspect)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(case.row_label, fontsize=10, rotation=90)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax = axes[row, 4]
        grid = case.forward_grid
        ax.set_facecolor("black")
        segments = np.stack((grid.x, grid.y), axis=2)
        ax.add_collection(LineCollection(segments, colors="white", linewidths=0.55))
        ax.add_collection(LineCollection(segments.transpose(1, 0, 2), colors="white", linewidths=0.55))
        height, width = grid.display_shape
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect(case.aspect)
        if row == 0:
            ax.set_title(col_titles[4], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    title = "Reference Deformation Motion Patterns, Coronal View" if composite else cases[0].row_label.replace("\n", " ")
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def forward_grid_segments(grid: DisplayForwardGrid) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack((grid.x, grid.y), axis=2)
    mag = np.asarray(grid.magnitude_mm, dtype=np.float32)
    horizontal = np.stack((points[:, :-1], points[:, 1:]), axis=2).reshape(-1, 2, 2)
    horizontal_values = (0.5 * (mag[:, :-1] + mag[:, 1:])).reshape(-1)
    vertical = np.stack((points[:-1, :], points[1:, :]), axis=2).reshape(-1, 2, 2)
    vertical_values = (0.5 * (mag[:-1, :] + mag[1:, :])).reshape(-1)
    return np.concatenate([horizontal, vertical], axis=0), np.concatenate([horizontal_values, vertical_values], axis=0)


def set_display_axis(ax: plt.Axes, display_shape: tuple[int, int], aspect: float) -> None:
    height, width = display_shape
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_summary_cases(cases: Sequence[MotionCase], out_path: Path) -> None:
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 6, figsize=(18.0, 2.9 * nrows), squeeze=False, constrained_layout=True)
    col_titles = ["Fixed", "Moving", "Deformed", "Backward grid", "Forward grid", "Quiver"]
    cmap = "viridis"
    for row, case in enumerate(cases):
        image_arrays = [
            case.fixed_slice,
            case.moving_slice,
            case.deformed_slice,
            np.clip(case.backward_grid_slice, 0.0, 1.0),
        ]
        for col, arr in enumerate(image_arrays):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect=case.aspect)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(case.row_label, fontsize=9, rotation=90)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        norm = mcolors.Normalize(vmin=0.0, vmax=case.summary_color_max_mm)

        ax_grid = axes[row, 4]
        ax_grid.imshow(case.moving_slice, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect=case.aspect)
        segments, values = forward_grid_segments(case.forward_grid)
        line_collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=0.35,
            alpha=0.95,
        )
        line_collection.set_array(np.clip(values, 0.0, case.summary_color_max_mm))
        ax_grid.add_collection(line_collection)
        if row == 0:
            ax_grid.set_title(col_titles[4], fontsize=11)
        set_display_axis(ax_grid, case.forward_grid.display_shape, case.aspect)

        ax_quiver = axes[row, 5]
        ax_quiver.imshow(case.moving_slice, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect=case.aspect)
        vectors = case.vectors
        quiver = ax_quiver.quiver(
            vectors.x,
            vectors.y,
            vectors.u,
            vectors.v,
            np.clip(vectors.magnitude, 0.0, case.summary_color_max_mm),
            cmap=cmap,
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=vectors.arrow_scale,
            pivot="middle",
            width=0.004,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.7,
        )
        quiver.set_path_effects([path_effects.withStroke(linewidth=0.55, foreground="black")])
        colorbar = fig.colorbar(quiver, ax=ax_quiver, fraction=0.046, pad=0.02)
        colorbar.set_label("Motion amplitude (mm)", fontsize=7)
        colorbar.ax.tick_params(labelsize=7)
        if row == 0:
            ax_quiver.set_title(col_titles[5], fontsize=11)
        set_display_axis(ax_quiver, case.forward_grid.display_shape, case.aspect)

    fig.suptitle("Reference Deformation Summary, Coronal View", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_vector_cases(cases: Sequence[MotionCase], out_path: Path, *, composite: bool) -> None:
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 1, figsize=(5.8, 3.05 * nrows), squeeze=False, constrained_layout=True)
    for row, case in enumerate(cases):
        ax = axes[row, 0]
        ax.imshow(case.fixed_slice, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect=case.aspect)
        vectors = case.vectors
        quiver = ax.quiver(
            vectors.x,
            vectors.y,
            vectors.u,
            vectors.v,
            vectors.magnitude,
            cmap="viridis",
            clim=(0.0, vectors.color_max),
            angles="xy",
            scale_units="xy",
            scale=vectors.arrow_scale,
            pivot="middle",
            width=0.0045,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.7,
        )
        quiver.set_path_effects([path_effects.withStroke(linewidth=0.55, foreground="black")])
        colorbar = fig.colorbar(quiver, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label(f"3D displacement ({vectors.unit})", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        ax.set_title(case.row_label.replace("\n", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    title = "Reference Deformation Vector Fields, Coronal View" if composite else "Coronal Vector Field"
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    text = """# Motion Pattern Figure v001

Reference-only coronal motion-pattern panels for paper figure preparation.

- OASIS rank 9 uses the saved BS4 SITReg displacement field.
- LungCT case01 and case08 use saved pTVReg fields reconstructed onto the canonical pTV 1mm crop.
- Each row shows fixed, moving, deformed moving, a backward image-warped grid, and a forward coordinate-lattice grid.
- Summary panels add quiver and overlay both forward grid and quiver on the moving image with a shared per-case mm color scale.
- Vector panels overlay in-plane coronal arrows on the fixed image; arrow color shows full 3D displacement magnitude.

See `motion_pattern_manifest.json` for field conventions, source validation, slice indices, and orientation notes.
"""
    (out_dir / "README.md").write_text(text)


def write_review_notebook(out_dir: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Motion Pattern Reference Review\n",
                    "\n",
                    "This notebook displays the generated coronal motion-pattern panels and manifest summary.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "from IPython.display import Image, display, Markdown\n",
                    "\n",
                    "ROOT = Path.cwd()\n",
                    "manifest = json.loads((ROOT / 'motion_pattern_manifest.json').read_text())\n",
                    "display(Markdown(f\"Generated cases: `{len(manifest['cases'])}`\"))\n",
                    "display(Image(filename=str(ROOT / 'motion_pattern_coronal_summary.png')))\n",
                    "display(Image(filename=str(ROOT / 'motion_pattern_coronal_reference.png')))\n",
                    "display(Image(filename=str(ROOT / 'motion_pattern_coronal_vector_reference.png')))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for item in manifest['per_case_pngs']:\n",
                    "    display(Markdown(f\"## {item['case_key']}\"))\n",
                    "    display(Image(filename=str(ROOT / item['path'])))\n",
                    "for item in manifest['per_case_vector_pngs']:\n",
                    "    display(Markdown(f\"## {item['case_key']} vector field\"))\n",
                    "    display(Image(filename=str(ROOT / item['path'])))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for case in manifest['cases']:\n",
                    "    display(Markdown(\n",
                    "        f\"**{case['display_label']}**: slice axis `{case['slice_axis_zyx']}`, \"\n",
                    "        f\"slice index `{case['slice_index']}`, field `{case['source_field']}`.  \"\n",
                    "        f\"Orientation: {case['display_orientation']}  \"\n",
                    "        f\"Backward grid: `{case['backward_grid_mode']}`.  \"\n",
                    "        f\"Forward grid: `{case['forward_grid_mode']}`; {case['forward_grid_component_mapping']}\"\n",
                    "    ))\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(out_dir / "motion_pattern_reference_review.ipynb", notebook)


def case_csv_rows(cases: Sequence[MotionCase]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        m = case.manifest
        rows.append(
            {
                "case_key": m["case_key"],
                "dataset": m["dataset"],
                "display_label": m["display_label"],
                "reference_field": m["source_field"],
                "slice_view": m["slice_view"],
                "slice_axis_zyx": m["slice_axis_zyx"],
                "slice_index": m["slice_index"],
                "field_shape_zyx": "x".join(str(v) for v in m["field_shape_zyx"]),
                "source_field_units": m["source_field_units"],
                "source_field_direction": m["source_field_direction"],
                "display_orientation": m["display_orientation"],
                "grid_step_voxels": m["grid_step_voxels"],
                "backward_grid_mode": m["backward_grid_mode"],
                "forward_grid_mode": m["forward_grid_mode"],
                "forward_grid_sample_count": m["forward_grid_sample_count"],
                "forward_grid_color_max_mm": m["forward_grid_color_max_mm"],
                "vector_step_voxels": m["vector_step_voxels"],
                "vector_color_units": m["vector_color_units"],
                "vector_sample_count": m["vector_sample_count"],
                "summary_color_max_mm": m["summary_color_max_mm"],
            }
        )
    return rows


def check_outputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    empty = [str(path) for path in paths if path.exists() and path.stat().st_size <= 0]
    if missing or empty:
        raise RuntimeError(f"Output validation failed. missing={missing}, empty={empty}")


def main() -> None:
    args = parse_args()
    self_check_forward_grid_helpers()
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite to regenerate: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(str(args.device))

    cases = [
        build_oasis_case(args, device),
        build_lungct_case(args, 1, device),
        build_lungct_case(args, 8, device),
    ]

    composite_png = out_dir / "motion_pattern_coronal_reference.png"
    composite_pdf = out_dir / "motion_pattern_coronal_reference.pdf"
    summary_png = out_dir / "motion_pattern_coronal_summary.png"
    summary_pdf = out_dir / "motion_pattern_coronal_summary.pdf"
    vector_composite_png = out_dir / "motion_pattern_coronal_vector_reference.png"
    vector_composite_pdf = out_dir / "motion_pattern_coronal_vector_reference.pdf"
    plot_cases(cases, composite_png, composite=True)
    plot_cases(cases, composite_pdf, composite=True)
    plot_summary_cases(cases, summary_png)
    plot_summary_cases(cases, summary_pdf)
    plot_vector_cases(cases, vector_composite_png, composite=True)
    plot_vector_cases(cases, vector_composite_pdf, composite=True)

    per_case_pngs: list[dict[str, str]] = []
    per_case_vector_pngs: list[dict[str, str]] = []
    for case in cases:
        path = out_dir / f"{case.case_key}_coronal_reference.png"
        plot_cases([case], path, composite=False)
        per_case_pngs.append({"case_key": case.case_key, "path": path.name})
        vector_path = out_dir / f"{case.case_key}_coronal_vector_reference.png"
        plot_vector_cases([case], vector_path, composite=False)
        per_case_vector_pngs.append({"case_key": case.case_key, "path": vector_path.name})

    manifest = {
        "artifact": "motion_pattern_figure",
        "version": "v001",
        "generated_by": str(Path(__file__).relative_to(REPO_ROOT)),
        "output_dir": str(out_dir),
        "device": str(device),
        "cases": [case.manifest for case in cases],
        "outputs": {
            "composite_png": composite_png.name,
            "composite_pdf": composite_pdf.name,
            "summary_png": summary_png.name,
            "summary_pdf": summary_pdf.name,
            "vector_composite_png": vector_composite_png.name,
            "vector_composite_pdf": vector_composite_pdf.name,
            "case_csv": "motion_pattern_cases.csv",
            "review_notebook": "motion_pattern_reference_review.ipynb",
        },
        "per_case_pngs": per_case_pngs,
        "per_case_vector_pngs": per_case_vector_pngs,
        "notes": [
            "Reference-only visualization: no registration runs are launched.",
            "Backward grid panels warp a synthetic 3D binary grid using the same pull-field convention as image warping.",
            "Forward grid panels draw a sparse coordinate lattice after adding the sampled displacement at grid vertices.",
            "Summary panels overlay forward grid and quiver on moving images using a shared per-case mm color scale.",
            "Vector panels show displayed in-plane coronal pull-field components; colors show full 3D displacement magnitude.",
            "OASIS and LungCT use different source orientations; per-case display transforms are recorded above.",
        ],
    }

    write_csv(out_dir / "motion_pattern_cases.csv", case_csv_rows(cases))
    write_json(out_dir / "motion_pattern_manifest.json", manifest)
    write_readme(out_dir)
    write_review_notebook(out_dir)
    check_outputs(
        [
            composite_png,
            composite_pdf,
            summary_png,
            summary_pdf,
            vector_composite_png,
            vector_composite_pdf,
            out_dir / "motion_pattern_cases.csv",
            out_dir / "motion_pattern_manifest.json",
            out_dir / "motion_pattern_reference_review.ipynb",
            *(out_dir / item["path"] for item in per_case_pngs),
            *(out_dir / item["path"] for item in per_case_vector_pngs),
        ]
    )
    print(json.dumps({"output_dir": str(out_dir), "cases": [case.case_key for case in cases]}, indent=2))


if __name__ == "__main__":
    main()
