#!/usr/bin/env python3
"""Accepted-orientation 2x9 coronal visual audit for FireANTs and P2e fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from run_pairlocal import p2e_module


P2E = p2e_module()
LOBE_LABELS = (8, 16, 32, 64, 128)
LOBE_NAMES = ("RUL", "RML", "RLL", "LUL", "LLL")


def _load_field(path: Path, case: Any) -> np.ndarray:
    nib = P2E.import_nibabel()
    image = nib.load(str(path))
    field = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if field.shape != (3, *case.shape_xyz) or not np.isfinite(field).all():
        raise RuntimeError(f"Invalid field for {case.case_label}: {path}")
    if not np.allclose(image.affine, case.fixed_affine_xyz, rtol=0.0, atol=1.0e-6):
        raise RuntimeError(f"Field affine differs from fixed CT: {path}")
    return field


def _positions(mask_xyz: np.ndarray) -> list[tuple[str, int]]:
    occupied_y = np.flatnonzero(np.asarray(mask_xyz, dtype=bool).any(axis=(0, 2)))
    if occupied_y.size == 0:
        raise RuntimeError("Fixed visual-QC mask is empty")
    posterior, anterior = int(occupied_y.min()), int(occupied_y.max())
    return [
        ("25%", int(round(posterior + 0.25 * (anterior - posterior)))),
        ("50% / central", int(round(posterior + 0.50 * (anterior - posterior)))),
        ("75%", int(round(posterior + 0.75 * (anterior - posterior)))),
    ]


def _base_figure(title: str):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 9, figsize=(27, 6.5), layout="constrained")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    return figure, axes


def _draw(axes, positions, ct_panels, mask_panels, *, mask_cmap, mask_vmax: int, mask_label: str) -> None:
    for slice_index, (label, y_index) in enumerate(positions):
        for state_index, (name, volume) in enumerate(ct_panels):
            axis = axes[0, 3 * slice_index + state_index]
            axis.imshow(P2E._coronal_radiology_view(volume, y_index), cmap="gray", vmin=-1000, vmax=300, origin="lower", interpolation="nearest")
            axis.set_title(f"{label}: {name}\nY={y_index}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            P2E._draw_coronal_orientation_markers(axis)
        for state_index, (name, volume) in enumerate(mask_panels):
            axis = axes[1, 3 * slice_index + state_index]
            axis.imshow(P2E._coronal_radiology_view(volume, y_index), cmap=mask_cmap, vmin=0, vmax=mask_vmax, origin="lower", interpolation="nearest")
            axis.set_title(f"{label}: {name}\nY={y_index}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            P2E._draw_coronal_orientation_markers(axis)
    axes[0, 0].set_ylabel("CT (HU window −1000..300)", fontsize=10)
    axes[1, 0].set_ylabel(mask_label, fontsize=10)


def _warp(volume_xyz: np.ndarray, field_cxyz: np.ndarray, *, mode: str, sampler: str, evaluator, device: str) -> np.ndarray:
    if sampler == "p2e_historical":
        return P2E.warp_xyz_with_historical_p2e(volume_xyz, field_cxyz, mode=mode)
    if sampler == "challenge":
        return evaluator.spatial_transform(volume_xyz, P2E.challenge_flow_xyzc(field_cxyz), mode=mode, device=device)
    raise ValueError(f"Unknown sampler: {sampler}")


def render_copd_case(
    *,
    pairlocal_root: Path,
    r231_root: Path,
    learn2breath_repo: Path,
    field_path: Path,
    case_id: str,
    intensity_profile: str,
    method: str,
    sampler: str,
    device: str = "cuda:0",
):
    """Render COPD CT/R231 visual QC; R231 is never an optimisation input."""

    from matplotlib.colors import ListedColormap

    case = P2E.load_pair("copd", pairlocal_root, case_id, intensity_profile)
    P2E.validate_r231_pair(case, r231_root, case_id)
    evaluator = P2E.load_learn2breath_evaluator(learn2breath_repo)
    nib = P2E.import_nibabel()
    fixed_path, moving_path = P2E.r231_paths(r231_root, case_id)
    fixed_mask = np.asarray(nib.load(str(fixed_path)).get_fdata(dtype=np.float32)) > 0
    moving_mask = np.asarray(nib.load(str(moving_path)).get_fdata(dtype=np.float32)) > 0
    field = _load_field(field_path, case)
    warped_ct = _warp(case.moving_hu_xyz, field, mode="bilinear", sampler=sampler, evaluator=evaluator, device=device)
    warped_mask = _warp(moving_mask.astype(np.float32), field, mode="nearest", sampler=sampler, evaluator=evaluator, device=device) > 0.5
    figure, axes = _base_figure(f"COPD {case.case_label}: {method} / {intensity_profile}")
    _draw(
        axes, _positions(fixed_mask),
        (("Moving EXP", case.moving_hu_xyz), ("Warped moving EXP", warped_ct), ("Fixed INSP", case.fixed_hu_xyz)),
        (("Moving R231", moving_mask), ("Warped moving R231", warped_mask), ("Fixed R231", fixed_mask)),
        mask_cmap=ListedColormap(("black", "lime")), mask_vmax=1, mask_label="R231 masks (QC only)",
    )
    return figure


def _lobe_display(array_xyz: np.ndarray) -> np.ndarray:
    labels = np.rint(np.asarray(array_xyz)).astype(np.int16)
    display = np.zeros(labels.shape, dtype=np.int16)
    for index, label in enumerate(LOBE_LABELS, start=1):
        display[labels == label] = index
    return display


def render_validation_case(
    *,
    validation_root: Path,
    learn2breath_repo: Path,
    field_path: Path,
    case_id: str,
    intensity_profile: str,
    method: str,
    device: str = "cuda:0",
):
    """Render validation CT/lobes with the challenge sampler used by Dice."""

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    case = P2E.load_pair("validation", validation_root, case_id, intensity_profile)
    evaluator = P2E.load_learn2breath_evaluator(learn2breath_repo)
    P2E.validate_validation_lobes(case, validation_root, case_id, evaluator)
    paths = evaluator.validation_paths(validation_root, int(case_id))
    nib = P2E.import_nibabel()
    fixed_lobes = np.asarray(nib.load(str(paths["insp_lobe"])).get_fdata(dtype=np.float32))
    moving_lobes = np.asarray(nib.load(str(paths["exp_lobe"])).get_fdata(dtype=np.float32))
    field = _load_field(field_path, case)
    warped_ct = _warp(case.moving_hu_xyz, field, mode="bilinear", sampler="challenge", evaluator=evaluator, device=device)
    warped_lobes = np.rint(_warp(moving_lobes, field, mode="nearest", sampler="challenge", evaluator=evaluator, device=device)).astype(np.int16)
    colours = ("black", "#3b4cc0", "#00a6ca", "#35b779", "#fde725", "#d01c8b")
    figure, axes = _base_figure(f"Validation {case.case_label}: {method} / {intensity_profile}")
    _draw(
        axes, _positions(fixed_lobes > 0),
        (("Moving EXP", case.moving_hu_xyz), ("Warped moving EXP", warped_ct), ("Fixed INSP", case.fixed_hu_xyz)),
        (("Moving lobes", _lobe_display(moving_lobes)), ("Warped moving lobes", _lobe_display(warped_lobes)), ("Fixed lobes", _lobe_display(fixed_lobes))),
        mask_cmap=ListedColormap(colours), mask_vmax=5, mask_label="Lobe masks (QC/evaluation only)",
    )
    figure.legend([Patch(facecolor=colours[index], edgecolor="none") for index in range(1, 6)], LOBE_NAMES, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.035), fontsize=9, frameon=False)
    return figure
