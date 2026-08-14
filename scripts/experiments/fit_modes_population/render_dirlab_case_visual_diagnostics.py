#!/usr/bin/env python
"""Render DIR-LAB case01/case08 visual diagnostics for Panel-A registration.

The renderer is intentionally source-backed by the schedule-matched
``camera_ready_source_index.csv``.  It reconstructs warped moving images from
the saved dense normalized displacement fields using the same ``grid_sample``
convention as the LungCT optimization/evaluation runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import torch

from plot_case_diagnostics import METHOD_ORDER, METHOD_STYLE
from render_schedule_matched_diagnostics import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFRESH_RUN_ROOT,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LUNGCT_RUNNER_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
if str(LUNGCT_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(LUNGCT_RUNNER_DIR))
import common as lung_common  # noqa: E402


SOURCE_INDEX = DEFAULT_OUTPUT_ROOT / "camera_ready_source_index.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "case_visual_diagnostics"
DEFAULT_RADIOLOGY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "radiology_step10_full_v001"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
DEFAULT_PTV_ROOT = Path(
    os.environ.get(
        "DIRLAB_PTVREG_ISO1MM_ROOT",
        "/workspace/data/lungct/processed/pTVReg_iso1mm",
    )
)
DATA_TOKEN = "${DATA_ROOT}"
REFRESH_TOKEN = "${DIRLAB_REFRESH_RUN_ROOT}"
PTV_TOKEN = "${DIRLAB_PTVREG_ISO1MM_ROOT}"
TARGET_CASES = ("case01", "case08")
CASE_IDS = {"case01": 1, "case08": 8}
LOCKED_CORONAL_Y = {"case01": 74, "case08": 102}
EXPECTED_PANEL = "A"
NCC_TOLERANCE = 1.0e-4
FIELD_TOLERANCE_VOX = 1.0e-4
DEFAULT_SLICE_STEP = 10
DEFAULT_GRID_STEP = 10
DEFAULT_GRID_SAMPLE_STEP = 2
DEFAULT_VECTOR_STEP = 12
DIFF_SCALE_PERCENTILE = 99.0
DISP_AMP_SCALE_PERCENTILE = 99.0


@dataclass
class MethodVisual:
    case_key: str
    method_slug: str
    method_label: str
    warped_volume: np.ndarray
    disp_vox_zyx: np.ndarray
    source_one_minus_ncc: float
    recomputed_one_minus_ncc: float
    ncc_abs_diff: float
    tre_mean_mm: float
    log_jdet_std_fg: float | None
    jdet_le0_pct_fg: float | None
    disp_mag_p95_fg_vox: float | None
    norm_to_vox_max_abs_diff: float
    summary_path_token: str
    source_path_token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, default=SOURCE_INDEX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument("--ptv-root", type=Path, default=DEFAULT_PTV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RADIOLOGY_OUTPUT_DIR)
    parser.add_argument("--slice-step", type=int, default=DEFAULT_SLICE_STEP)
    parser.add_argument("--grid-step", type=int, default=DEFAULT_GRID_STEP)
    parser.add_argument("--grid-sample-step", type=int, default=DEFAULT_GRID_SAMPLE_STEP)
    parser.add_argument("--vector-step", type=int, default=DEFAULT_VECTOR_STEP)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used only for reconstructing warped moving images.",
    )
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def image_record(path: Path, output_dir: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": os.path.relpath(path, output_dir),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_token_path(
    value: Any,
    *,
    data_root: Path,
    refresh_run_root: Path,
    ptv_root: Path,
) -> Path:
    text = str(value or "")
    replacements = {
        DATA_TOKEN: str(data_root),
        REFRESH_TOKEN: str(refresh_run_root),
        PTV_TOKEN: str(ptv_root),
        "${RUN_ROOT}": str(data_root),
        "${REPO_ROOT}": str(REPO_ROOT),
    }
    for token, root in replacements.items():
        text = text.replace(token, root)
    return Path(text)


def portable_path(path: Path, *, data_root: Path, refresh_run_root: Path, ptv_root: Path) -> str:
    text = str(path)
    for root, token in [
        (refresh_run_root, REFRESH_TOKEN),
        (ptv_root, PTV_TOKEN),
        (data_root, DATA_TOKEN),
        (REPO_ROOT, "${REPO_ROOT}"),
    ]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return path.name


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_rows(index: pd.DataFrame) -> pd.DataFrame:
    selected = index[
        index["dataset"].eq("dirlab")
        & index["case_key"].isin(TARGET_CASES)
        & index["panel"].eq(EXPECTED_PANEL)
        & index["method_slug"].isin(METHOD_ORDER)
    ].copy()
    if len(selected) != len(TARGET_CASES) * len(METHOD_ORDER):
        raise RuntimeError(f"Expected 8 DIR-LAB Panel-A source rows, found {len(selected)}")
    if selected.duplicated(["case_key", "method_slug"]).any():
        raise RuntimeError("Duplicate case/method rows in source index")
    selected["case_order_for_plot"] = selected["case_key"].map({case: i for i, case in enumerate(TARGET_CASES)})
    selected["method_order_for_plot"] = selected["method_slug"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    return selected.sort_values(["case_order_for_plot", "method_order_for_plot"]).reset_index(drop=True)


def display_coronal_zyx(array_zyx: np.ndarray, y_index: int) -> np.ndarray:
    """Return a coronal slice in radiology display array coordinates.

    The source arrays are zyx.  The returned 2D array is ``volume[:, y, :]``
    with the x axis flipped so the left side of the rendered panel is patient
    right.  Callers render this array with ``origin="upper"`` so the top of
    the panel is superior.
    """

    return np.flip(np.asarray(array_zyx)[:, int(y_index), :], axis=1)


def display_coronal_disp_zyx(disp_zyx: np.ndarray, y_index: int) -> np.ndarray:
    """Return a radiology-display displacement slice in zyx components."""

    return np.flip(np.asarray(disp_zyx)[:, int(y_index), :, :], axis=1)


def add_orientation_labels(axis: plt.Axes) -> None:
    props = {
        "boxstyle": "round,pad=0.16",
        "facecolor": "black",
        "edgecolor": "none",
        "alpha": 0.58,
    }
    text_kwargs = {
        "transform": axis.transAxes,
        "color": "#FFD92F",
        "fontsize": 8.2,
        "fontweight": "bold",
        "bbox": props,
        "zorder": 20,
    }
    axis.text(0.50, 0.985, "S", ha="center", va="top", **text_kwargs)
    axis.text(0.50, 0.015, "I", ha="center", va="bottom", **text_kwargs)
    axis.text(0.015, 0.50, "R", ha="left", va="center", **text_kwargs)
    axis.text(0.985, 0.50, "L", ha="right", va="center", **text_kwargs)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruct_warp_and_ncc(
    fixed: np.ndarray,
    moving: np.ndarray,
    disp_norm_xyz: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if disp_norm_xyz.shape != fixed.shape + (3,):
        raise ValueError(
            f"Normalized displacement shape {disp_norm_xyz.shape} does not match image {fixed.shape}"
        )
    fixed_t = torch.from_numpy(np.asarray(fixed, dtype=np.float32)).to(device)
    moving_t = torch.from_numpy(np.asarray(moving, dtype=np.float32)).to(device)
    disp_t = torch.from_numpy(np.asarray(disp_norm_xyz, dtype=np.float32)).to(device)
    id_grid = lung_common.make_coordinate_grid(fixed.shape, device).detach()
    with torch.no_grad():
        warped_t = lung_common.warp_full(moving_t, id_grid + disp_t, mode="bilinear")
        ncc_loss = lung_common.ncc_loss(fixed_t[None, None], warped_t[None, None])
    warped = warped_t.detach().to("cpu", torch.float32).numpy()
    one_minus_ncc = 1.0 + float(ncc_loss.detach().to("cpu"))
    del fixed_t, moving_t, disp_t, id_grid, warped_t, ncc_loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(warped).all():
        raise FloatingPointError("Reconstructed warped image contains nonfinite values")
    return warped, one_minus_ncc


def method_visual_from_row(
    row: pd.Series,
    *,
    case_fixed: np.ndarray,
    case_moving: np.ndarray,
    data_root: Path,
    refresh_run_root: Path,
    ptv_root: Path,
    device: torch.device,
) -> MethodVisual:
    summary_path = resolve_token_path(
        row["summary_path"],
        data_root=data_root,
        refresh_run_root=refresh_run_root,
        ptv_root=ptv_root,
    )
    source_path = resolve_token_path(
        row["source_path"],
        data_root=data_root,
        refresh_run_root=refresh_run_root,
        ptv_root=ptv_root,
    )
    run_dir = summary_path.parent
    required = [
        summary_path,
        source_path,
        run_dir / "disp_norm_xyz_reg1mm.npy",
        run_dir / "disp_vox_reg1mm_zyx.npy",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required run files:\n" + "\n".join(map(str, missing)))

    summary = load_json(summary_path)
    disp_norm_xyz = np.load(run_dir / "disp_norm_xyz_reg1mm.npy").astype(np.float32, copy=False)
    disp_vox_zyx = np.load(run_dir / "disp_vox_reg1mm_zyx.npy").astype(np.float32, copy=False)
    if disp_vox_zyx.shape != case_fixed.shape + (3,):
        raise ValueError(
            f"{row['case_key']}/{row['method_slug']}: voxel displacement shape "
            f"{disp_vox_zyx.shape} does not match image {case_fixed.shape}"
        )
    computed_vox = lung_common.dense_norm_xyz_to_vox_zyx(disp_norm_xyz)
    max_abs = float(np.max(np.abs(computed_vox - disp_vox_zyx)))
    if max_abs > FIELD_TOLERANCE_VOX:
        raise RuntimeError(
            f"{row['case_key']}/{row['method_slug']}: normalized-to-voxel field "
            f"check failed, max abs diff={max_abs:.6g}"
        )

    warped, recomputed_one_minus_ncc = reconstruct_warp_and_ncc(
        case_fixed,
        case_moving,
        disp_norm_xyz,
        device,
    )
    if float(np.min(warped)) < -1.0e-4 or float(np.max(warped)) > 1.0 + 1.0e-4:
        raise RuntimeError(
            f"{row['case_key']}/{row['method_slug']}: warped image range outside "
            f"expected [0,1], got [{float(np.min(warped)):.6g}, {float(np.max(warped)):.6g}]"
        )
    source_one_minus_ncc = float(row["endpoint_value"])
    ncc_abs_diff = abs(recomputed_one_minus_ncc - source_one_minus_ncc)
    if ncc_abs_diff > NCC_TOLERANCE:
        raise RuntimeError(
            f"{row['case_key']}/{row['method_slug']}: recomputed 1-NCC "
            f"{recomputed_one_minus_ncc:.8f} differs from source "
            f"{source_one_minus_ncc:.8f} by {ncc_abs_diff:.3g}"
        )
    tre = finite(nested(summary, "tre_snap_to_voxel", "tre_mean_mm"))
    if tre is None:
        raise RuntimeError(f"{row['case_key']}/{row['method_slug']}: missing TRE in summary")

    regularity = summary.get("regularity_fg_registration_crop")
    if not isinstance(regularity, dict):
        regularity = {}
    log_jdet = finite(regularity.get("log_jdet_std_fg"))
    folding = finite(regularity.get("jdet_le0_pct_fg"))
    disp_p95 = finite(regularity.get("disp_mag_p95_fg_vox"))

    return MethodVisual(
        case_key=str(row["case_key"]),
        method_slug=str(row["method_slug"]),
        method_label=METHOD_STYLE[str(row["method_slug"])]["label"],
        warped_volume=warped.astype(np.float32, copy=False),
        disp_vox_zyx=disp_vox_zyx,
        source_one_minus_ncc=source_one_minus_ncc,
        recomputed_one_minus_ncc=recomputed_one_minus_ncc,
        ncc_abs_diff=ncc_abs_diff,
        tre_mean_mm=float(tre),
        log_jdet_std_fg=log_jdet,
        jdet_le0_pct_fg=folding,
        disp_mag_p95_fg_vox=disp_p95,
        norm_to_vox_max_abs_diff=max_abs,
        summary_path_token=str(row["summary_path"]),
        source_path_token=str(row["source_path"]),
    )


def slice_indices_from_mask(mask: np.ndarray, step: int) -> tuple[list[int], dict[str, int]]:
    if step <= 0:
        raise ValueError("--slice-step must be positive")
    mask_coords = np.argwhere(np.asarray(mask).astype(bool))
    if mask_coords.size == 0:
        raise RuntimeError("Cannot determine coronal slice sweep from an empty mask")
    bbox_min = mask_coords.min(axis=0)
    bbox_max = mask_coords.max(axis=0)
    y_min = int(bbox_min[1])
    y_max = int(bbox_max[1])
    indices = list(range(y_min, y_max + 1, int(step)))
    if indices[-1] != y_max:
        indices.append(y_max)
    return indices, {
        "z_min": int(bbox_min[0]),
        "z_max": int(bbox_max[0]),
        "y_min": y_min,
        "y_max": y_max,
        "x_min": int(bbox_min[2]),
        "x_max": int(bbox_max[2]),
    }


def diff_limit_for_case(
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    visuals: Sequence[MethodVisual],
    y_indices: Sequence[int],
) -> float:
    values: list[np.ndarray] = []
    for y_index in y_indices:
        fixed_display = display_coronal_zyx(fixed, y_index)
        mask_display = display_coronal_zyx(mask.astype(np.float32), y_index) > 0.5
        mask_slice = np.asarray(mask_display).astype(bool)
        diff_slices = [
            display_coronal_zyx(moving, y_index) - fixed_display,
            *[
                display_coronal_zyx(visual.warped_volume, y_index) - fixed_display
                for visual in visuals
            ],
        ]
        for diff in diff_slices:
            arr = np.asarray(diff, dtype=np.float32)
            scoped = arr[mask_slice] if mask_slice.shape == arr.shape and mask_slice.any() else arr.reshape(-1)
            scoped = scoped[np.isfinite(scoped)]
            if scoped.size:
                values.append(scoped)
    if not values:
        return 0.1
    merged = np.concatenate(values)
    limit = float(np.percentile(np.abs(merged), DIFF_SCALE_PERCENTILE))
    if not math.isfinite(limit) or limit <= 1.0e-6:
        limit = float(np.max(np.abs(merged))) if merged.size else 0.1
    return float(np.clip(limit, 0.02, 1.0))


def coronal_disp_mag(disp_display_zyx: np.ndarray) -> np.ndarray:
    disp = np.asarray(disp_display_zyx, dtype=np.float32)
    return np.sqrt(disp[..., 0] ** 2 + disp[..., 2] ** 2)


def displacement_amplitude_limit_for_case(
    *,
    mask: np.ndarray,
    visuals: Sequence[MethodVisual],
    y_indices: Sequence[int],
) -> float:
    values: list[np.ndarray] = []
    for y_index in y_indices:
        mask_display = display_coronal_zyx(mask.astype(np.float32), y_index) > 0.5
        for visual in visuals:
            mag = coronal_disp_mag(display_coronal_disp_zyx(visual.disp_vox_zyx, y_index))
            scoped = mag[mask_display] if mask_display.shape == mag.shape and mask_display.any() else mag.reshape(-1)
            scoped = scoped[np.isfinite(scoped)]
            if scoped.size:
                values.append(scoped)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    limit = float(np.percentile(merged, DISP_AMP_SCALE_PERCENTILE))
    if not math.isfinite(limit) or limit <= 1.0e-6:
        limit = float(np.max(merged)) if merged.size else 1.0
    return max(limit, 1.0)


def diff_extrema_for_case(
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    visuals: Sequence[MethodVisual],
    y_indices: Sequence[int],
) -> dict[str, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for y_index in y_indices:
        fixed_display = display_coronal_zyx(fixed, y_index)
        arrays = [
            display_coronal_zyx(moving, y_index) - fixed_display,
            *[
                display_coronal_zyx(visual.warped_volume, y_index) - fixed_display
                for visual in visuals
            ],
        ]
        for arr in arrays:
            finite_values = np.asarray(arr, dtype=np.float32)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                mins.append(float(np.min(finite_values)))
                maxs.append(float(np.max(finite_values)))
    return {
        "min": min(mins) if mins else float("nan"),
        "max": max(maxs) if maxs else float("nan"),
    }


def displacement_amplitude_extrema_for_case(
    *,
    visuals: Sequence[MethodVisual],
    y_indices: Sequence[int],
) -> dict[str, float]:
    maxs: list[float] = []
    for y_index in y_indices:
        for visual in visuals:
            mag = coronal_disp_mag(display_coronal_disp_zyx(visual.disp_vox_zyx, y_index))
            finite_values = mag[np.isfinite(mag)]
            if finite_values.size:
                maxs.append(float(np.max(finite_values)))
    return {
        "min": 0.0,
        "max": max(maxs) if maxs else float("nan"),
    }


def add_mask_contour(axis: plt.Axes, mask_display: np.ndarray, color: str = "#66BD63") -> None:
    if np.asarray(mask_display).any():
        axis.contour(mask_display.astype(float), levels=[0.5], colors=color, linewidths=0.65, alpha=0.9)


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    add_orientation_labels(axis)


def style_field_axis(axis: plt.Axes, depth: int, width: int) -> None:
    axis.set_xlim(0, width - 1)
    axis.set_ylim(depth - 1, 0)
    axis.set_aspect("equal")
    axis.set_facecolor("white")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    add_orientation_labels(axis)


def add_axis_colorbar(
    fig: plt.Figure,
    axis: plt.Axes,
    mappable: Any,
    *,
    label: str,
    tick_size: float = 6.3,
    label_size: float = 6.8,
) -> None:
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="4.2%", pad=0.035)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=tick_size, length=2)
    cbar.set_label(label, fontsize=label_size)


def add_colored_line(
    axis: plt.Axes,
    x_values: np.ndarray,
    z_values: np.ndarray,
    mag_values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    linewidth: float,
) -> None:
    if len(x_values) < 2:
        return
    points = np.column_stack([x_values, z_values]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_mag = 0.5 * (mag_values[:-1] + mag_values[1:])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=0.98,
        zorder=2,
    )
    collection.set_array(segment_mag)
    axis.add_collection(collection)


def draw_deformed_grid(
    fig: plt.Figure,
    axis: plt.Axes,
    visual: MethodVisual,
    y_index: int,
    shape_zyx: Sequence[int],
    amp_norm: Normalize,
    *,
    grid_step: int,
    sample_step: int,
) -> None:
    depth, _, width = (int(value) for value in shape_zyx)
    disp = display_coronal_disp_zyx(visual.disp_vox_zyx, y_index)
    cmap = "viridis"
    # Identity grid, light gray.
    for x0 in range(0, width, grid_step):
        axis.plot([x0, x0], [0, depth - 1], color="#D8D8D8", linewidth=0.35, zorder=1)
    for z0 in range(0, depth, grid_step):
        axis.plot([0, width - 1], [z0, z0], color="#D8D8D8", linewidth=0.35, zorder=1)
    # Deformed lines in radiology display coordinates.  The display x axis is
    # flipped relative to voxel x, so in-plane dx becomes -dx on the plot.
    xs = np.arange(0, width, sample_step, dtype=int)
    zs = np.arange(0, depth, sample_step, dtype=int)
    for z0 in range(0, depth, grid_step):
        d = disp[z0, xs, :]
        x_def = xs.astype(np.float32) - d[:, 2]
        z_def = float(z0) + d[:, 0]
        mag = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
        add_colored_line(axis, x_def, z_def, mag, norm=amp_norm, cmap=cmap, linewidth=0.72)
    for x0 in range(0, width, grid_step):
        d = disp[zs, x0, :]
        x_def = float(x0) - d[:, 2]
        z_def = zs.astype(np.float32) + d[:, 0]
        mag = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
        add_colored_line(axis, x_def, z_def, mag, norm=amp_norm, cmap=cmap, linewidth=0.72)
    sm = plt.cm.ScalarMappable(norm=amp_norm, cmap=cmap)
    sm.set_array([])
    add_axis_colorbar(fig, axis, sm, label="|disp| vox")
    style_field_axis(axis, depth, width)
    axis.set_title("deformed grid", fontsize=9.5)


def draw_vector_plot(
    fig: plt.Figure,
    axis: plt.Axes,
    visual: MethodVisual,
    y_index: int,
    mask_display: np.ndarray,
    shape_zyx: Sequence[int],
    amp_norm: Normalize,
    *,
    vector_step: int,
) -> None:
    depth, _, width = (int(value) for value in shape_zyx)
    z_coords = np.arange(0, depth, vector_step, dtype=int)
    x_coords = np.arange(0, width, vector_step, dtype=int)
    zz, xx = np.meshgrid(z_coords, x_coords, indexing="ij")
    keep = mask_display[zz, xx].astype(bool)
    if not bool(keep.any()):
        keep = np.ones_like(zz, dtype=bool)
    disp = display_coronal_disp_zyx(visual.disp_vox_zyx, y_index)
    dz = disp[zz, xx, 0]
    dx = disp[zz, xx, 2]
    x_plot = xx[keep].astype(np.float32)
    z_plot = zz[keep].astype(np.float32)
    u_plot = -dx[keep].astype(np.float32)
    v_plot = dz[keep].astype(np.float32)
    mag = np.sqrt(u_plot**2 + v_plot**2)
    quiver = axis.quiver(
        x_plot,
        z_plot,
        u_plot,
        v_plot,
        mag,
        cmap="viridis",
        norm=amp_norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        alpha=0.9,
    )
    axis.text(
        0.02,
        0.98,
        "vector: (-dx, dz)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    add_axis_colorbar(fig, axis, quiver, label="|disp| vox")
    style_field_axis(axis, depth, width)
    axis.set_title("displacement vectors", fontsize=9.5)


def metrics_title(visual: MethodVisual) -> str:
    parts = [
        visual.method_label,
        f"1-NCC {visual.source_one_minus_ncc:.4f}",
        f"TRE {visual.tre_mean_mm:.2f} mm",
    ]
    if visual.log_jdet_std_fg is not None:
        parts.append(f"stdLogJ {visual.log_jdet_std_fg:.3f}")
    if visual.jdet_le0_pct_fg is not None:
        parts.append(f"fold {visual.jdet_le0_pct_fg:.2f}%")
    return "\n".join(parts)


def render_case_figure(
    *,
    case_key: str,
    case_id: int,
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    visuals: Sequence[MethodVisual],
    y_index: int,
    output_path: Path,
    output_root: Path,
    dpi: int,
    diff_limit: float,
    amp_limit: float,
    grid_step: int,
    grid_sample_step: int,
    vector_step: int,
    artifact_kind: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_display = display_coronal_zyx(fixed, y_index)
    moving_display = display_coronal_zyx(moving, y_index)
    mask_display = display_coronal_zyx(mask.astype(np.float32), y_index) > 0.5
    moving_diff = moving_display - fixed_display
    amp_norm = Normalize(vmin=0.0, vmax=float(amp_limit))

    fig, axes = plt.subplots(
        5,
        4,
        figsize=(19.2, 23.0),
        constrained_layout=False,
    )
    image_kwargs = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0, "origin": "upper"}
    diff_kwargs = {"cmap": "coolwarm", "vmin": -diff_limit, "vmax": diff_limit, "origin": "upper"}

    axes[0, 0].imshow(moving_display, **image_kwargs)
    axes[0, 0].set_title("moving T50", fontsize=10.5, weight="bold")
    add_mask_contour(axes[0, 0], mask_display)
    axes[0, 1].imshow(fixed_display, **image_kwargs)
    axes[0, 1].set_title("fixed T00", fontsize=10.5, weight="bold")
    add_mask_contour(axes[0, 1], mask_display)
    diff_im = axes[0, 2].imshow(moving_diff, **diff_kwargs)
    axes[0, 2].set_title("moving - fixed", fontsize=10.5, weight="bold")
    add_mask_contour(axes[0, 2], mask_display, color="#222222")
    add_axis_colorbar(fig, axes[0, 2], diff_im, label="diff")
    axes[0, 3].imshow(mask_display, cmap="Greens", vmin=0, vmax=1, origin="upper")
    axes[0, 3].set_title("fixed lung mask", fontsize=10.5, weight="bold")
    axes[0, 3].text(
        0.04,
        0.96,
        f"case{case_id:02d}\ncoronal Y={y_index}\nradiology view\nS top, R left",
        transform=axes[0, 3].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    for axis in axes[0, :]:
        style_image_axis(axis)

    for row_index, visual in enumerate(visuals, start=1):
        style = METHOD_STYLE[visual.method_slug]
        warped_slice = display_coronal_zyx(visual.warped_volume, y_index)
        diff_slice = warped_slice - fixed_display
        axes[row_index, 0].imshow(warped_slice, **image_kwargs)
        axes[row_index, 0].set_title(metrics_title(visual), fontsize=9.4, color=style["color"], weight="bold")
        add_mask_contour(axes[row_index, 0], mask_display)
        diff_im = axes[row_index, 1].imshow(diff_slice, **diff_kwargs)
        axes[row_index, 1].set_title("deformed - fixed", fontsize=9.5)
        add_mask_contour(axes[row_index, 1], mask_display, color="#222222")
        add_axis_colorbar(fig, axes[row_index, 1], diff_im, label="diff")
        draw_deformed_grid(
            fig,
            axes[row_index, 2],
            visual,
            y_index,
            fixed.shape,
            amp_norm,
            grid_step=grid_step,
            sample_step=grid_sample_step,
        )
        draw_vector_plot(
            fig,
            axes[row_index, 3],
            visual,
            y_index,
            mask_display,
            fixed.shape,
            amp_norm,
            vector_step=vector_step,
        )
        style_image_axis(axes[row_index, 0])
        style_image_axis(axes[row_index, 1])

    fig.legend(
        handles=[
            Line2D([0], [0], color=METHOD_STYLE[method]["color"], linewidth=3, label=METHOD_STYLE[method]["label"])
            for method in METHOD_ORDER
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        f"DIR-LAB case{case_id:02d} Panel-A visual diagnostic: T50→T00, pTVReg 1 mm crop, Y={y_index}",
        fontsize=15.5,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.044,
        "Radiology coronal display: top=S, bottom=I, left=patient R, right=patient L. "
        "Fields are pull fields on the fixed T00 grid; vectors show display components (-dx, dz).",
        ha="center",
        va="center",
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.025, right=0.972, top=0.945, bottom=0.07, hspace=0.34, wspace=0.20)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {
        "case_key": case_key,
        "case_id": case_id,
        "artifact_kind": artifact_kind,
        "path": os.path.relpath(output_path, output_root),
        "coronal_y_index": int(y_index),
        "display_convention": "zyx coronal volume[:, y, :] with x flip; rendered origin='upper'; top=S, left=patient R",
        "diff_scale": {
            "scope": "case-global rendered-slice fixed_t00_lung_mask",
            "symmetric_vmin_vmax": [-diff_limit, diff_limit],
            "percentile": DIFF_SCALE_PERCENTILE,
        },
        "displacement_amplitude_scale": {
            "components": "coronal in-plane sqrt(dz^2 + dx^2), displayed as (-dx,dz)",
            "vmin_vmax_vox": [0.0, amp_limit],
            "percentile": DISP_AMP_SCALE_PERCENTILE,
        },
        "image": image_record(output_path, output_root),
    }


def load_case_visuals(
    case_key: str,
    rows: pd.DataFrame,
    *,
    data_root: Path,
    refresh_run_root: Path,
    ptv_root: Path,
    device: torch.device,
) -> tuple[Any, list[MethodVisual]]:
    case_id = CASE_IDS[case_key]
    case = lung_common.load_case(ptv_root, case_id, dataset="4dct")
    y_index = LOCKED_CORONAL_Y[case_key]
    if y_index < 0 or y_index >= case.fixed.shape[1]:
        raise RuntimeError(f"{case_key}: locked coronal Y={y_index} outside shape {case.fixed.shape}")
    mask_coords = np.argwhere(case.fixed_mask.astype(bool))
    if mask_coords.size == 0:
        raise RuntimeError(f"{case_key}: fixed T00 lung mask is empty")
    bbox_min = mask_coords.min(axis=0)
    bbox_max = mask_coords.max(axis=0)
    bbox_center_y = int(round((int(bbox_min[1]) + int(bbox_max[1])) / 2.0))
    if bbox_center_y != y_index:
        raise RuntimeError(
            f"{case_key}: locked Y={y_index} does not match fixed-mask bbox center {bbox_center_y}"
        )
    visuals: list[MethodVisual] = []
    for method in METHOD_ORDER:
        match = rows[rows["method_slug"].eq(method)]
        if len(match) != 1:
            raise RuntimeError(f"{case_key}/{method}: expected one source row, found {len(match)}")
        visuals.append(
            method_visual_from_row(
                match.iloc[0],
                case_fixed=case.fixed,
                case_moving=case.moving,
                data_root=data_root,
                refresh_run_root=refresh_run_root,
                ptv_root=ptv_root,
                device=device,
            )
        )
    return case, visuals


def metrics_rows_from_visuals(
    visuals_by_case: dict[str, list[MethodVisual]],
    case_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    central_y_by_case = {
        str(record["case_key"]): int(record["central_coronal_y_index"])
        for record in case_records
    }
    rows: list[dict[str, Any]] = []
    for case_key in TARGET_CASES:
        for visual in visuals_by_case[case_key]:
            rows.append(
                {
                    "case_key": case_key,
                    "case_id": CASE_IDS[case_key],
                    "central_coronal_y_index": central_y_by_case[case_key],
                    "method_slug": visual.method_slug,
                    "display_method": visual.method_label,
                    "source_one_minus_ncc": visual.source_one_minus_ncc,
                    "recomputed_one_minus_ncc": visual.recomputed_one_minus_ncc,
                    "one_minus_ncc_abs_diff": visual.ncc_abs_diff,
                    "tre_snap_to_voxel_mean_mm": visual.tre_mean_mm,
                    "log_jdet_std_fg": visual.log_jdet_std_fg,
                    "jdet_le0_pct_fg": visual.jdet_le0_pct_fg,
                    "disp_mag_p95_fg_vox": visual.disp_mag_p95_fg_vox,
                    "norm_to_vox_max_abs_diff": visual.norm_to_vox_max_abs_diff,
                    "source_path": visual.source_path_token,
                    "summary_path": visual.summary_path_token,
                }
            )
    return rows


def make_contact_sheet(
    *,
    image_paths: Sequence[Path],
    y_indices: Sequence[int],
    output_path: Path,
    output_root: Path,
    title: str,
    columns: int = 4,
    thumb_width: int = 900,
) -> dict[str, Any]:
    if len(image_paths) != len(y_indices):
        raise ValueError("Contact-sheet image_paths and y_indices lengths differ")
    if not image_paths:
        raise ValueError("No images supplied for contact sheet")
    thumbs: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            scale = thumb_width / float(image.width)
            thumb_height = max(1, int(round(image.height * scale)))
            thumbs.append(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
    label_height = 32
    title_height = 46
    cell_width = thumb_width
    cell_height = max(thumb.height for thumb in thumbs) + label_height
    rows = int(math.ceil(len(thumbs) / float(columns)))
    sheet = Image.new("RGB", (columns * cell_width, title_height + rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for index, (thumb, y_index) in enumerate(zip(thumbs, y_indices, strict=True)):
        row = index // columns
        col = index % columns
        x0 = col * cell_width
        y0 = title_height + row * cell_height
        draw.text((x0 + 10, y0 + 7), f"coronal Y={int(y_index)}", fill=(0, 0, 0))
        sheet.paste(thumb, (x0, y0 + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return {
        "path": os.path.relpath(output_path, output_root),
        "source_count": len(image_paths),
        "columns": columns,
        "thumbnail_width_px": thumb_width,
        "image": image_record(output_path, output_root),
    }


def write_report(
    path: Path,
    case_records: Sequence[dict[str, Any]],
    figure_records: Sequence[dict[str, Any]],
    contact_records: Sequence[dict[str, Any]],
    metrics_rows: Sequence[dict[str, Any]],
    manifest_path: Path,
    metrics_path: Path,
) -> None:
    central_by_case = {
        record["case_key"]: record
        for record in figure_records
        if record.get("artifact_kind") == "central"
    }
    contact_by_case = {record["case_key"]: record for record in contact_records}
    case_by_key = {record["case_key"]: record for record in case_records}
    metric_rel = os.path.relpath(metrics_path, path.parent)
    manifest_rel = os.path.relpath(manifest_path, path.parent)
    lines = [
        "# DIR-LAB Case01/Case08 Radiology-Oriented Slice-Sweep Diagnostics",
        "",
        "These are investigative diagnostics for understanding intensity-loss/TRE disagreement; they are not final paper figures.",
        "",
        "## Central corrected figures",
        "",
    ]
    for case_key in TARGET_CASES:
        record = central_by_case[case_key]
        case_record = case_by_key[case_key]
        image_rel = record["path"]
        lines.extend(
            [
                f"### DIR-LAB {case_key}",
                "",
                f"Central coronal slice: `Y={case_record['central_coronal_y_index']}`.",
                "",
                f"![DIR-LAB {case_key} central radiology visual diagnostic]({image_rel})",
                "",
            ]
        )
    lines.extend(["## Every-10-slice contact sheets", ""])
    for case_key in TARGET_CASES:
        record = contact_by_case[case_key]
        case_record = case_by_key[case_key]
        y_indices = ", ".join(str(value) for value in case_record["slice_sweep_y_indices"])
        lines.extend(
            [
                f"### DIR-LAB {case_key}",
                "",
                f"Slice sweep: `{case_record['slice_sweep_count']}` coronal slices, y indices `{y_indices}`.",
                "",
                f"![DIR-LAB {case_key} every-10-slice contact sheet]({record['path']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Metrics",
            "",
            "| Case | Method | 1-NCC | recomputed 1-NCC | TRE (mm) | stdLogJ fg | fold fg (%) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_rows:
        lines.append(
            "| {case} | {method} | {ncc:.6f} | {ncc2:.6f} | {tre:.2f} | {std} | {fold} |".format(
                case=row["case_key"],
                method=row["display_method"],
                ncc=float(row["source_one_minus_ncc"]),
                ncc2=float(row["recomputed_one_minus_ncc"]),
                tre=float(row["tre_snap_to_voxel_mean_mm"]),
                std=(
                    f"{float(row['log_jdet_std_fg']):.3f}"
                    if row.get("log_jdet_std_fg") not in (None, "")
                    else ""
                ),
                fold=(
                    f"{float(row['jdet_le0_pct_fg']):.2f}"
                    if row.get("jdet_le0_pct_fg") not in (None, "")
                    else ""
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Conventions",
            "",
            "- Fixed image: T00; moving image: T50; registration direction: T50→T00.",
            "- Central coronal slice is the fixed T00 lung-mask bounding-box center along y.",
            "- Slice sweep covers the fixed T00 lung-mask y-bounds every 10 slices, with the max-y endpoint appended.",
            "- Radiology display convention: top=S, bottom=I, left=patient R, right=patient L.",
            "- Array convention: zyx `volume[:, y, :]`, x-flipped, rendered with `origin='upper'`.",
            "- Warped images are reconstructed with the same runner convention: `grid_sample(moving, identity + saved normalized xyz field, align_corners=True)`.",
            "- Difference-map scale is symmetric and case-global across rendered slices, computed from the fixed T00 lung mask.",
            "- Grid and vector colorbars use the same case-global in-plane displacement-amplitude scale, `sqrt(dz^2 + dx^2)` in voxel units.",
            "- Vector panels show fixed-space coronal display components as horizontal `-dx` and vertical `dz`.",
            "",
            "## Files",
            "",
            f"- Metrics CSV: [`{metrics_path.name}`]({metric_rel})",
            f"- Machine-readable manifest: [`{manifest_path.name}`]({manifest_rel})",
            "- Per-slice PNGs are under `slices/case01/` and `slices/case08/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source_index = Path(args.source_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    refresh_run_root = Path(args.refresh_run_root).resolve()
    ptv_root = Path(args.ptv_root).resolve()
    device = choose_device(str(args.device))

    index = pd.read_csv(source_index)
    selected = select_rows(index)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_records: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    visuals_by_case: dict[str, list[MethodVisual]] = {}
    case_records: list[dict[str, Any]] = []

    for case_key in TARGET_CASES:
        case_rows = selected[selected["case_key"].eq(case_key)].copy()
        case, visuals = load_case_visuals(
            case_key,
            case_rows,
            data_root=data_root,
            refresh_run_root=refresh_run_root,
            ptv_root=ptv_root,
            device=device,
        )
        visuals_by_case[case_key] = visuals
        case_id = CASE_IDS[case_key]
        sweep_y_indices, bbox = slice_indices_from_mask(case.fixed_mask, int(args.slice_step))
        central_y = LOCKED_CORONAL_Y[case_key]
        scale_y_indices = sorted(set([central_y, *sweep_y_indices]))
        diff_limit = diff_limit_for_case(
            fixed=case.fixed,
            moving=case.moving,
            mask=case.fixed_mask,
            visuals=visuals,
            y_indices=scale_y_indices,
        )
        amp_limit = displacement_amplitude_limit_for_case(
            mask=case.fixed_mask,
            visuals=visuals,
            y_indices=scale_y_indices,
        )
        diff_extrema = diff_extrema_for_case(
            fixed=case.fixed,
            moving=case.moving,
            visuals=visuals,
            y_indices=scale_y_indices,
        )
        amp_extrema = displacement_amplitude_extrema_for_case(
            visuals=visuals,
            y_indices=scale_y_indices,
        )

        central_output_path = (
            output_dir / f"{case_key}_coronal_y{central_y:03d}_visual_diagnostic_radiology.png"
        )
        figure_records.append(
            render_case_figure(
                case_key=case_key,
                case_id=case_id,
                fixed=case.fixed,
                moving=case.moving,
                mask=case.fixed_mask,
                visuals=visuals,
                y_index=central_y,
                output_path=central_output_path,
                output_root=output_dir,
                dpi=int(args.dpi),
                diff_limit=diff_limit,
                amp_limit=amp_limit,
                grid_step=int(args.grid_step),
                grid_sample_step=int(args.grid_sample_step),
                vector_step=int(args.vector_step),
                artifact_kind="central",
            )
        )

        slice_paths: list[Path] = []
        for y_index in sweep_y_indices:
            slice_output_path = output_dir / "slices" / case_key / f"{case_key}_y{y_index:03d}.png"
            figure_records.append(
                render_case_figure(
                    case_key=case_key,
                    case_id=case_id,
                    fixed=case.fixed,
                    moving=case.moving,
                    mask=case.fixed_mask,
                    visuals=visuals,
                    y_index=y_index,
                    output_path=slice_output_path,
                    output_root=output_dir,
                    dpi=int(args.dpi),
                    diff_limit=diff_limit,
                    amp_limit=amp_limit,
                    grid_step=int(args.grid_step),
                    grid_sample_step=int(args.grid_sample_step),
                    vector_step=int(args.vector_step),
                    artifact_kind="slice_sweep",
                )
            )
            slice_paths.append(slice_output_path)
            print(f"rendered {case_key} y={y_index}", flush=True)

        contact_record = make_contact_sheet(
            image_paths=slice_paths,
            y_indices=sweep_y_indices,
            output_path=output_dir / f"{case_key}_step{int(args.slice_step)}_contact_sheet.png",
            output_root=output_dir,
            title=(
                f"DIR-LAB {case_key} every-{int(args.slice_step)} coronal slice diagnostics "
                "(radiology view)"
            ),
        )
        contact_record.update(
            {
                "case_key": case_key,
                "case_id": case_id,
                "slice_step": int(args.slice_step),
                "slice_y_indices": sweep_y_indices,
            }
        )
        contact_records.append(contact_record)

        case_records.append(
            {
                "case_key": case_key,
                "case_id": case_id,
                "shape_zyx": [int(value) for value in case.shape_zyx],
                "fixed_label": case.fixed_label,
                "moving_label": case.moving_label,
                "fixed_mask_voxels": int(case.fixed_mask.sum()),
                "fixed_mask_bbox_zyx": bbox,
                "central_coronal_y_index": central_y,
                "fixed_mask_central_slice_voxels": int(case.fixed_mask[:, central_y, :].sum()),
                "slice_sweep_step": int(args.slice_step),
                "slice_sweep_y_indices": sweep_y_indices,
                "slice_sweep_count": len(sweep_y_indices),
                "scale_y_indices": scale_y_indices,
                "diff_scale": {
                    "scope": "case-global central-plus-sweep fixed_t00_lung_mask",
                    "symmetric_vmin_vmax": [-diff_limit, diff_limit],
                    "percentile": DIFF_SCALE_PERCENTILE,
                    "observed_extrema": diff_extrema,
                },
                "displacement_amplitude_scale": {
                    "components": "coronal in-plane sqrt(dz^2 + dx^2), displayed as (-dx,dz)",
                    "vmin_vmax_vox": [0.0, amp_limit],
                    "percentile": DISP_AMP_SCALE_PERCENTILE,
                    "observed_extrema_vox": amp_extrema,
                },
                "ptv_case_dir": f"{PTV_TOKEN}/4DCT/{case_key}",
                "fixed_image": f"{PTV_TOKEN}/4DCT/{case_key}/fixed_T00_reg1mm_trilinear_zyx.npy",
                "moving_image": f"{PTV_TOKEN}/4DCT/{case_key}/moving_T50_reg1mm_trilinear_zyx.npy",
                "fixed_mask": f"{PTV_TOKEN}/4DCT/{case_key}/fixed_T00_lungmask_reg1mm_nearest_zyx.npy",
            }
        )

    metrics_rows = metrics_rows_from_visuals(visuals_by_case, case_records)
    metrics_path = output_dir / f"case_visual_diagnostics_radiology_step{int(args.slice_step)}_metrics.csv"
    manifest_path = output_dir / f"case_visual_diagnostics_radiology_step{int(args.slice_step)}_manifest.json"
    report_path = output_dir / f"case_visual_diagnostics_radiology_step{int(args.slice_step)}.md"
    write_csv(metrics_path, metrics_rows)

    central_figure_count = sum(1 for record in figure_records if record.get("artifact_kind") == "central")
    slice_figure_count = sum(1 for record in figure_records if record.get("artifact_kind") == "slice_sweep")
    expected_slice_count = sum(int(record["slice_sweep_count"]) for record in case_records)
    displayed_slices_within_bbox = all(
        all(
            int(record["fixed_mask_bbox_zyx"]["y_min"]) <= int(y_index) <= int(record["fixed_mask_bbox_zyx"]["y_max"])
            for y_index in record["slice_sweep_y_indices"]
        )
        for record in case_records
    )

    manifest = {
        "generated_at": utc_text(),
        "artifact_role": "investigative diagnostic; not final paper figure",
        "artifact_version": "radiology_step10_full_v001",
        "generator": repo_relative(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source_index": repo_relative(source_index),
        "source_index_sha256": sha256(source_index),
        "inputs": {
            "data_root": DATA_TOKEN,
            "refresh_run_root": REFRESH_TOKEN,
            "dirlab_ptvreg_iso1mm_root": PTV_TOKEN,
        },
        "device": str(device),
        "selected_records": len(selected),
        "target_cases": list(TARGET_CASES),
        "method_order": METHOD_ORDER,
        "rendering": {
            "slice_step": int(args.slice_step),
            "grid_step": int(args.grid_step),
            "grid_sample_step": int(args.grid_sample_step),
            "vector_step": int(args.vector_step),
            "dpi": int(args.dpi),
            "central_figures": central_figure_count,
            "slice_sweep_figures": slice_figure_count,
            "contact_sheets": len(contact_records),
        },
        "ncc_tolerance": NCC_TOLERANCE,
        "field_tolerance_vox": FIELD_TOLERANCE_VOX,
        "warp_convention": {
            "function": "lungct_no_sweep_repro.common.warp_full",
            "expression": "grid_sample(moving, identity_grid + disp_norm_xyz_reg1mm, align_corners=True)",
            "field_file": "disp_norm_xyz_reg1mm.npy",
            "field_grid_layout": "zyx",
            "field_component_order": "xyz",
            "direction": "fixed T00 grid samples moving T50 image",
        },
        "display_convention": {
            "view": "radiology coronal",
            "central_slice_rule": "fixed T00 lung-mask bounding-box center along y",
            "sweep_slice_rule": "fixed T00 lung-mask y-bounds every slice_step slices with max-y endpoint appended",
            "array_slice": "zyx volume[:, y, :]",
            "display_coordinates": "x-flipped coronal array rendered with origin='upper'",
            "orientation_labels": "top=S, bottom=I, left=patient R, right=patient L",
            "vector_components": "horizontal=-dx, vertical=dz, y-axis inverted by radiology display",
            "grid_coloring": "coronal in-plane displacement magnitude sqrt(dz^2 + dx^2) in voxels",
            "diff_colorbars": "one colorbar next to every diff subplot",
            "field_colorbars": "one displacement-amplitude colorbar next to every grid/vector subplot",
        },
        "cases": case_records,
        "figures": figure_records,
        "contact_sheets": contact_records,
        "metrics_csv": os.path.relpath(metrics_path, output_dir),
        "report_md": os.path.relpath(report_path, output_dir),
        "validation": {
            "all_required_panel_a_records_resolved": len(selected) == 8,
            "central_figure_count": central_figure_count,
            "expected_central_figure_count": len(TARGET_CASES),
            "slice_figure_count": slice_figure_count,
            "expected_slice_figure_count": expected_slice_count,
            "contact_sheet_count": len(contact_records),
            "expected_contact_sheet_count": len(TARGET_CASES),
            "displayed_slices_within_fixed_lung_y_bbox": displayed_slices_within_bbox,
            "all_ncc_endpoints_matched": all(
                float(row["one_minus_ncc_abs_diff"]) <= NCC_TOLERANCE
                for row in metrics_rows
            ),
            "all_norm_to_vox_fields_matched": all(
                float(row["norm_to_vox_max_abs_diff"]) <= FIELD_TOLERANCE_VOX
                for row in metrics_rows
            ),
            "all_tre_values_finite": all(
                math.isfinite(float(row["tre_snap_to_voxel_mean_mm"]))
                for row in metrics_rows
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(report_path, case_records, figure_records, contact_records, metrics_rows, manifest_path, metrics_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
