"""Small volume, mask, and plotting helpers for inspection scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402


INDEX_TO_ARRAY_AXIS = {0: 2, 1: 1, 2: 0}


@dataclass(frozen=True)
class CoronalDisplayPlan:
    fixed_array_axis: int
    fixed_index_axis_xyz: int
    fixed_index: int
    row_array_axis: int
    row_index_axis_xyz: int
    col_array_axis: int
    col_index_axis_xyz: int
    flip_rows: bool
    flip_cols: bool
    row_spacing: float
    col_spacing: float
    aspect: float
    screen_top: str = "patient superior"
    screen_bottom: str = "patient inferior"
    screen_left: str = "patient right"
    screen_right: str = "patient left"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_sitk_zyx(path: Path | str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def load_sitk_zyx_with_metadata(path: Path | str) -> Tuple[np.ndarray, Dict[str, object]]:
    image = sitk.ReadImage(str(path))
    direction = tuple(float(v) for v in image.GetDirection())
    metadata: Dict[str, object] = {
        "size_xyz": [int(v) for v in image.GetSize()],
        "spacing_xyz": [float(v) for v in image.GetSpacing()],
        "origin_xyz": [float(v) for v in image.GetOrigin()],
        "direction": list(direction),
        "orientation": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction),
    }
    return sitk.GetArrayFromImage(image), metadata


def make_sitk_image(image_zyx: np.ndarray, spacing_zyx: Sequence[float]) -> sitk.Image:
    image = sitk.GetImageFromArray(image_zyx)
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    return image


def bbox_zyx(mask: np.ndarray) -> Dict[str, object]:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if not np.any(mask_bool):
        return {"empty": True, "min_zyx": None, "max_zyx": None, "shape_zyx": list(mask_bool.shape)}
    coords = np.argwhere(mask_bool)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return {
        "empty": False,
        "min_zyx": [int(v) for v in lo],
        "max_zyx": [int(v) for v in hi],
        "shape_zyx": list(mask_bool.shape),
    }


def bbox_center_y(mask: np.ndarray, fallback_shape: Sequence[int]) -> int:
    return bbox_center_axis(mask, 1, fallback_shape)


def bbox_center_axis(mask: np.ndarray, axis: int, fallback_shape: Sequence[int]) -> int:
    box = bbox_zyx(mask)
    if box["empty"]:
        return int(fallback_shape[axis] // 2)
    lo = box["min_zyx"]
    hi = box["max_zyx"]
    assert isinstance(lo, list) and isinstance(hi, list)
    return int(round((lo[axis] + hi[axis]) / 2.0))


def mask_stats(mask: np.ndarray) -> Dict[str, object]:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    voxels = int(mask_bool.sum())
    total = int(mask_bool.size)
    out = {
        "voxels": voxels,
        "total_voxels": total,
        "fraction": float(voxels / total) if total else float("nan"),
    }
    out.update(bbox_zyx(mask_bool))
    return out


def overlap_stats(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> Dict[str, float | int]:
    aa = np.asarray(a).astype(bool, copy=False)
    bb = np.asarray(b).astype(bool, copy=False)
    if aa.shape != bb.shape:
        raise ValueError(f"Mask shape mismatch: {aa.shape} vs {bb.shape}")
    a_sum = int(aa.sum())
    b_sum = int(bb.sum())
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    denom = a_sum + b_sum
    return {
        f"{a_name}_voxels": a_sum,
        f"{b_name}_voxels": b_sum,
        "intersection_voxels": inter,
        "union_voxels": union,
        "dice": float((2.0 * inter / denom) if denom else 1.0),
        "jaccard": float((inter / union) if union else 1.0),
        f"{a_name}_covered_by_{b_name}": float((inter / a_sum) if a_sum else float("nan")),
        f"{b_name}_covered_by_{a_name}": float((inter / b_sum) if b_sum else float("nan")),
        f"{a_name}_only_voxels": int(np.logical_and(aa, ~bb).sum()),
        f"{b_name}_only_voxels": int(np.logical_and(bb, ~aa).sum()),
    }


def parse_case_ids(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))
    return out


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def overlay_mask(
    ax: plt.Axes,
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    title: str,
    color: str,
    origin: str = "lower",
    aspect: float | str = "auto",
) -> None:
    ax.imshow(normalize_slice(image_slice), cmap="gray", origin=origin, aspect=aspect)
    masked = np.ma.masked_where(~mask_slice.astype(bool), mask_slice)
    ax.imshow(masked, cmap=color, alpha=0.35, origin=origin, aspect=aspect, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _dominant_index_axis(direction_matrix: np.ndarray, physical_axis: int) -> int:
    return int(np.argmax(np.abs(direction_matrix[physical_axis, :])))


def build_sitk_standard_coronal_plan(
    center_mask_zyx: np.ndarray,
    direction: Sequence[float],
    spacing_xyz: Sequence[float],
) -> CoronalDisplayPlan:
    direction_matrix = np.asarray(direction, dtype=float).reshape(3, 3)
    spacing = [float(v) for v in spacing_xyz]
    lr_index_axis = _dominant_index_axis(direction_matrix, 0)
    ap_index_axis = _dominant_index_axis(direction_matrix, 1)
    si_index_axis = _dominant_index_axis(direction_matrix, 2)
    if len({lr_index_axis, ap_index_axis, si_index_axis}) != 3:
        raise ValueError(f"Cannot derive unique patient axes from direction matrix: {direction_matrix}")

    fixed_array_axis = INDEX_TO_ARRAY_AXIS[ap_index_axis]
    row_array_axis = INDEX_TO_ARRAY_AXIS[si_index_axis]
    col_array_axis = INDEX_TO_ARRAY_AXIS[lr_index_axis]
    fixed_index = bbox_center_axis(center_mask_zyx, fixed_array_axis, center_mask_zyx.shape)
    row_spacing = spacing[si_index_axis]
    col_spacing = spacing[lr_index_axis]
    return CoronalDisplayPlan(
        fixed_array_axis=fixed_array_axis,
        fixed_index_axis_xyz=ap_index_axis,
        fixed_index=fixed_index,
        row_array_axis=row_array_axis,
        row_index_axis_xyz=si_index_axis,
        col_array_axis=col_array_axis,
        col_index_axis_xyz=lr_index_axis,
        flip_rows=bool(direction_matrix[2, si_index_axis] > 0),
        flip_cols=bool(direction_matrix[0, lr_index_axis] < 0),
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        aspect=float(row_spacing / col_spacing),
    )


def apply_coronal_display_plan(volume_zyx: np.ndarray, plan: CoronalDisplayPlan) -> np.ndarray:
    slice_2d = np.take(volume_zyx, plan.fixed_index, axis=plan.fixed_array_axis)
    remaining_axes = [axis for axis in range(3) if axis != plan.fixed_array_axis]
    row_pos = remaining_axes.index(plan.row_array_axis)
    col_pos = remaining_axes.index(plan.col_array_axis)
    oriented = np.transpose(slice_2d, (row_pos, col_pos))
    if plan.flip_rows:
        oriented = np.flip(oriented, axis=0)
    if plan.flip_cols:
        oriented = np.flip(oriented, axis=1)
    return oriented


def save_oasis_mask_figure(
    image_zyx: np.ndarray,
    seg_label_mask: np.ndarray,
    canonical_fg_mask: np.ndarray,
    subject_id: str,
    out_path: Path,
    direction: Sequence[float],
    spacing_xyz: Sequence[float],
) -> Dict[str, object]:
    view_plan = build_sitk_standard_coronal_plan(
        np.logical_or(seg_label_mask, canonical_fg_mask),
        direction,
        spacing_xyz,
    )
    image_slice = apply_coronal_display_plan(image_zyx, view_plan)
    seg_slice = apply_coronal_display_plan(seg_label_mask, view_plan)
    canonical_slice = apply_coronal_display_plan(canonical_fg_mask, view_plan)

    overlap = np.logical_and(seg_slice, canonical_slice)
    seg_only = np.logical_and(seg_slice, ~canonical_slice)
    canonical_only = np.logical_and(canonical_slice, ~seg_slice)
    rgb = np.dstack([normalize_slice(image_slice)] * 3)
    rgb[overlap] = 0.55 * rgb[overlap] + np.array([0.95, 0.85, 0.10]) * 0.45
    rgb[seg_only] = 0.55 * rgb[seg_only] + np.array([0.00, 0.90, 0.20]) * 0.45
    rgb[canonical_only] = 0.55 * rgb[canonical_only] + np.array([0.95, 0.05, 0.70]) * 0.45

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(normalize_slice(image_slice), cmap="gray", origin="upper", aspect=view_plan.aspect)
    axes[0].set_title(
        f"OASIS {subject_id}: raw image\nstandard coronal AP index={view_plan.fixed_index}",
        fontsize=9,
    )
    axes[0].axis("off")
    overlay_mask(
        axes[1],
        image_slice,
        canonical_slice,
        "canonical FG: raw image > 0",
        "Purples",
        origin="upper",
        aspect=view_plan.aspect,
    )
    overlay_mask(
        axes[2],
        image_slice,
        seg_slice,
        "seg > 0 comparator",
        "Greens",
        origin="upper",
        aspect=view_plan.aspect,
    )
    axes[3].imshow(rgb, origin="upper", aspect=view_plan.aspect)
    axes[3].set_title(
        "overlap yellow | seg-only green | canonical-only magenta\n"
        "screen L=patient R | top=superior",
        fontsize=9,
    )
    axes[3].axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return view_plan.to_dict()


def save_lungct_mask_figure(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    case_id: int,
    out_path: Path,
    previous_fixed_mask: np.ndarray | None = None,
    spacing_zyx: Sequence[float] | None = None,
) -> None:
    fixed_y = bbox_center_y(fixed_mask, fixed_zyx.shape)
    moving_y = bbox_center_y(moving_mask, moving_zyx.shape)
    fixed_slice = np.flip(fixed_zyx[:, fixed_y, :], axis=0)
    moving_slice = np.flip(moving_zyx[:, moving_y, :], axis=0)
    fixed_mask_slice = np.flip(fixed_mask[:, fixed_y, :], axis=0)
    moving_mask_slice = np.flip(moving_mask[:, moving_y, :], axis=0)
    aspect: float | str = "auto"
    if spacing_zyx is not None:
        aspect = float(spacing_zyx[0]) / float(spacing_zyx[2])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(normalize_slice(fixed_slice), cmap="gray", origin="lower", aspect=aspect)
    axes[0, 0].set_title(f"case {case_id:02d} T00 raw\ny={fixed_y}; top=superior", fontsize=9)
    axes[0, 0].axis("off")
    overlay_mask(axes[0, 1], fixed_slice, fixed_mask_slice, "T00 + generated mask", "Greens", aspect=aspect)
    if previous_fixed_mask is not None:
        old = np.flip(previous_fixed_mask[:, fixed_y, :], axis=0).astype(bool)
        new = fixed_mask_slice.astype(bool)
        rgb = np.dstack([normalize_slice(fixed_slice)] * 3)
        both = np.logical_and(old, new)
        old_only = np.logical_and(old, ~new)
        new_only = np.logical_and(new, ~old)
        rgb[both] = 0.55 * rgb[both] + np.array([0.95, 0.85, 0.10]) * 0.45
        rgb[old_only] = 0.55 * rgb[old_only] + np.array([0.95, 0.05, 0.05]) * 0.45
        rgb[new_only] = 0.55 * rgb[new_only] + np.array([0.05, 0.65, 0.95]) * 0.45
        axes[0, 2].imshow(rgb, origin="lower", aspect=aspect)
        axes[0, 2].set_title("T00 old/new check\nyellow both | red old | blue new", fontsize=9)
        axes[0, 2].axis("off")
    else:
        overlay_mask(
            axes[0, 2],
            fixed_slice,
            fixed_mask_slice,
            "T00 mask; no previous mask",
            "Greens",
            aspect=aspect,
        )

    axes[1, 0].imshow(normalize_slice(moving_slice), cmap="gray", origin="lower", aspect=aspect)
    axes[1, 0].set_title(f"case {case_id:02d} T50 raw\ny={moving_y}; top=superior", fontsize=9)
    axes[1, 0].axis("off")
    overlay_mask(axes[1, 1], moving_slice, moving_mask_slice, "T50 + generated mask", "Purples", aspect=aspect)
    axes[1, 2].imshow(normalize_slice(moving_slice), cmap="gray", origin="lower", aspect=aspect)
    axes[1, 2].contour(moving_mask_slice.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.8)
    axes[1, 2].set_title("T50 mask contour", fontsize=9)
    axes[1, 2].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
