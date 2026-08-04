#!/usr/bin/env python3
"""Create bounded paired A0/A1 visual-review figures for DUAL-INR-DIR runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.experiments.lungct_dual_inr_author.external_api import (  # type: ignore
        AUTHOR_EXTERNAL_COMMIT,
        load_external_api,
    )
    from scripts.experiments.lungct_dual_inr_author.portable_paths import (  # type: ignore
        portable_review_value,
    )
else:
    from .external_api import AUTHOR_EXTERNAL_COMMIT, load_external_api
    from .portable_paths import portable_review_value


DEFAULT_CASES = (1, 8)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LOBE_LABELS = (10, 11, 12, 13, 14)
LOBE_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")
REQUIRED_HISTORY_FIELDS = (
    "zero_step",
    "stage_index",
    "blur_sigma",
    "loss_total",
    "loss_similarity",
    "loss_spatial_weighted",
    "loss_mask_weighted",
)
REQUIRED_RECORD_FILES = (
    "DONE",
    "RUN_STATE.json",
    "summary.json",
    "resolved_config.json",
    "provenance.json",
    "history.csv",
    "stage_transitions.json",
    "dense_dvf_crop_voxel_xyz.npy",
)


class ReviewError(RuntimeError):
    """Raised when an input pair is incomplete or violates the review contract."""


def parse_cases(value: str) -> list[int]:
    try:
        cases = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cases must be comma-separated integers") from exc
    if not cases or len(cases) != len(set(cases)):
        raise argparse.ArgumentTypeError("cases must be a nonempty list without duplicates")
    if any(case < 1 or case > 10 for case in cases):
        raise argparse.ArgumentTypeError("DIR-Lab 4DCT cases must be in 1..10")
    return cases


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewError(f"Expected a JSON object in {path}")
    return payload


def _resolved_equal(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _read_history(path: Path, expected_steps: int) -> dict[str, np.ndarray]:
    values = {field: [] for field in REQUIRED_HISTORY_FIELDS}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(REQUIRED_HISTORY_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ReviewError(f"{path} lacks columns {sorted(missing)}")
            for row_index, row in enumerate(reader):
                for field in REQUIRED_HISTORY_FIELDS:
                    try:
                        value = float(row[field])
                    except (TypeError, ValueError) as exc:
                        raise ReviewError(
                            f"{path} row {row_index + 2} has invalid {field}"
                        ) from exc
                    if not math.isfinite(value):
                        raise ReviewError(
                            f"{path} row {row_index + 2} has non-finite {field}"
                        )
                    values[field].append(value)
    except OSError as exc:
        raise ReviewError(f"Cannot read {path}: {exc}") from exc
    if len(values["zero_step"]) != expected_steps:
        raise ReviewError(
            f"{path} contains {len(values['zero_step'])} rows, expected {expected_steps}"
        )
    zero_steps = np.asarray(values["zero_step"], dtype=np.int64)
    if not np.array_equal(zero_steps, np.arange(expected_steps)):
        raise ReviewError(f"{path} has non-sequential zero_step values")
    return {
        field: np.asarray(items, dtype=np.float64) for field, items in values.items()
    }


def _validate_stage_history(
    history: Mapping[str, np.ndarray], trace: Mapping[str, Any], path: Path
) -> None:
    starts = [int(value) for value in trace["stage_start_steps"]]
    sigmas = [float(value) for value in trace["blur_sigmas"]]
    expected_stage = np.zeros(len(history["stage_index"]), dtype=np.int64)
    for stage_index, start in enumerate(starts):
        expected_stage[start:] = stage_index
    if not np.array_equal(history["stage_index"].astype(np.int64), expected_stage):
        raise ReviewError(f"{path} stage_index rows disagree with stage transitions")
    expected_sigma = np.asarray([sigmas[index] for index in expected_stage])
    if not np.allclose(history["blur_sigma"], expected_sigma, rtol=0.0, atol=1e-8):
        raise ReviewError(f"{path} blur_sigma rows disagree with stage transitions")


def _load_complete_record(
    *,
    case_dir: Path,
    variant: str,
    case: int,
    data_root: Path,
    allow_smoke: bool,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RECORD_FILES if not (case_dir / name).is_file()]
    if missing:
        raise ReviewError(f"Incomplete {variant} case {case:02d}: missing {missing}")

    state = _load_json(case_dir / "RUN_STATE.json")
    summary = _load_json(case_dir / "summary.json")
    config = _load_json(case_dir / "resolved_config.json")
    provenance = _load_json(case_dir / "provenance.json")
    trace = _load_json(case_dir / "stage_transitions.json")
    for label, payload in (("RUN_STATE", state), ("summary", summary)):
        expected = {"status": "complete", "variant": variant, "case": case}
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ReviewError(
                    f"{case_dir}/{label}.{key}={payload.get(key)!r}, expected {value!r}"
                )

    run_mode = str(summary.get("run_mode"))
    smoke_steps = summary.get("smoke_steps")
    if run_mode == "short_smoke" and not allow_smoke:
        raise ReviewError(
            f"{variant} case {case:02d} is a short smoke; pass --allow-smoke explicitly"
        )
    if run_mode not in {"full_length", "short_smoke"}:
        raise ReviewError(f"Unsupported run_mode in {case_dir}: {run_mode!r}")

    try:
        expected_steps = int(trace["total_steps"])
        starts = [int(value) for value in trace["stage_start_steps"]]
        sigmas = [float(value) for value in trace["blur_sigmas"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"Malformed stage transition manifest in {case_dir}") from exc
    if expected_steps <= 0 or not starts or starts[0] != 0:
        raise ReviewError(f"Invalid transition step range in {case_dir}")
    if starts != sorted(set(starts)) or any(step >= expected_steps for step in starts):
        raise ReviewError(f"Invalid stage start steps in {case_dir}: {starts}")
    if run_mode == "full_length" and (expected_steps != 3000 or sigmas != [4.0, 2.0, 0.0]):
        raise ReviewError(f"Full record in {case_dir} does not use the locked schedule")
    if run_mode == "full_length" and len(starts) != len(sigmas):
        raise ReviewError(f"Full record in {case_dir} did not visit every blur stage")
    if run_mode == "short_smoke" and int(smoke_steps) != expected_steps:
        raise ReviewError(f"Smoke metadata and transition length differ in {case_dir}")

    expected_mask_weight = 0.1 if variant == "A0" else 0.0
    if float(config.get("loss", {}).get("mask_weight", -1.0)) != expected_mask_weight:
        raise ReviewError(f"Unexpected {variant} mask weight in {case_dir}")
    if int(config.get("training", {}).get("total_steps", -1)) != expected_steps:
        raise ReviewError(f"Config and transition step counts differ in {case_dir}")
    if provenance.get("external_commit") != AUTHOR_EXTERNAL_COMMIT:
        raise ReviewError(f"External commit mismatch in {case_dir}")
    if provenance.get("external_git_status_porcelain") not in {None, ""}:
        raise ReviewError(f"External checkout was dirty for {case_dir}")
    if not _resolved_equal(provenance.get("data_root"), data_root):
        raise ReviewError(f"Provenance data root differs from requested root in {case_dir}")

    seed = int(summary.get("seed", -1))
    if int(trace.get("case", -1)) != case or int(trace.get("seed", -1)) != seed:
        raise ReviewError(f"Trace identity mismatch in {case_dir}")
    expected_policy = "author_dynamic_convergence" if variant == "A0" else "replay_a0"
    if trace.get("source_variant") != variant or trace.get("policy") != expected_policy:
        raise ReviewError(f"Trace source/policy mismatch in {case_dir}")

    history = _read_history(case_dir / "history.csv", expected_steps)
    _validate_stage_history(history, trace, case_dir / "history.csv")
    try:
        dense_dvf = np.load(case_dir / "dense_dvf_crop_voxel_xyz.npy")
    except (OSError, ValueError) as exc:
        raise ReviewError(f"Cannot load dense DVF from {case_dir}: {exc}") from exc
    if dense_dvf.ndim != 4 or dense_dvf.shape[-1] != 3:
        raise ReviewError(f"Dense DVF has invalid shape in {case_dir}: {dense_dvf.shape}")
    if not np.isfinite(dense_dvf).all():
        raise ReviewError(f"Dense DVF contains non-finite values in {case_dir}")
    if list(dense_dvf.shape[:3]) != list(summary.get("image_shape_crop", [])):
        raise ReviewError(f"Dense DVF and summary crop shapes differ in {case_dir}")

    return {
        "case_dir": case_dir,
        "variant": variant,
        "seed": seed,
        "run_mode": run_mode,
        "smoke_steps": smoke_steps,
        "summary": summary,
        "trace": trace,
        "history": history,
        "dense_dvf": dense_dvf.astype(np.float32, copy=False),
        "expected_steps": expected_steps,
    }


def load_paired_case(
    *,
    run_root: Path,
    data_root: Path,
    case: int,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    a0 = _load_complete_record(
        case_dir=run_root / "a0" / f"case_{case:02d}",
        variant="A0",
        case=case,
        data_root=data_root,
        allow_smoke=allow_smoke,
    )
    a1 = _load_complete_record(
        case_dir=run_root / "a1" / f"case_{case:02d}",
        variant="A1",
        case=case,
        data_root=data_root,
        allow_smoke=allow_smoke,
    )
    for key in ("seed", "run_mode", "smoke_steps", "expected_steps"):
        if a0[key] != a1[key]:
            raise ReviewError(f"A0/A1 {key} mismatch for case {case:02d}")
    if a0["trace"]["stage_start_steps"] != a1["trace"]["stage_start_steps"]:
        raise ReviewError(f"A0/A1 stage boundaries differ for case {case:02d}")
    if a0["trace"]["blur_sigmas"] != a1["trace"]["blur_sigmas"]:
        raise ReviewError(f"A0/A1 blur schedules differ for case {case:02d}")
    expected_a0_trace = (a0["case_dir"] / "stage_transitions.json").resolve()
    if not _resolved_equal(a1["summary"].get("a0_transition_trace"), expected_a0_trace):
        raise ReviewError(f"A1 does not identify its paired A0 trace for case {case:02d}")
    if not _resolved_equal(
        a1["trace"].get("replayed_manifest_path"), expected_a0_trace
    ):
        raise ReviewError(f"A1 transition manifest points to the wrong A0 trace")
    if a0["dense_dvf"].shape != a1["dense_dvf"].shape:
        raise ReviewError(f"A0/A1 dense DVF shapes differ for case {case:02d}")
    return {"case": case, "a0": a0, "a1": a1}


def _warp_with_dvf(
    *, api: Any, volume: np.ndarray, dense_dvf: np.ndarray, order: int
) -> np.ndarray:
    grid = np.indices(volume.shape, dtype=np.float32)
    coordinates = grid + np.moveaxis(dense_dvf, -1, 0)
    return api.utils.ndi.map_coordinates(
        volume.astype(np.float32, copy=False),
        coordinates,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis).T


def _overlay_lobes(
    ax: plt.Axes,
    masks: np.ndarray,
    *,
    axis: int,
    index: int,
    linestyle: str,
    linewidth: float,
) -> None:
    for channel, color in enumerate(LOBE_COLORS):
        plane = _plane(masks[channel], axis, index)
        if bool(np.any(plane > 0.5)) and bool(np.any(plane <= 0.5)):
            ax.contour(
                plane,
                levels=[0.5],
                colors=[color],
                linewidths=linewidth,
                linestyles=linestyle,
                origin="lower",
            )


def _show_anatomy(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    axis: int,
    index: int,
    vmin: float,
    vmax: float,
    title: str,
    fixed_masks: np.ndarray | None = None,
    warped_masks: np.ndarray | None = None,
) -> None:
    ax.imshow(
        _plane(image, axis, index),
        cmap="gray",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if fixed_masks is not None:
        _overlay_lobes(
            ax,
            fixed_masks,
            axis=axis,
            index=index,
            linestyle="solid",
            linewidth=1.25,
        )
    if warped_masks is not None:
        _overlay_lobes(
            ax,
            warped_masks,
            axis=axis,
            index=index,
            linestyle="dashed",
            linewidth=1.0,
        )
    ax.set_title(title)
    ax.set_axis_off()


def _stage_lines(ax: plt.Axes, trace: Mapping[str, Any]) -> None:
    starts = [int(value) for value in trace["stage_start_steps"]]
    sigmas = [float(value) for value in trace["blur_sigmas"]]
    for stage_index, start in enumerate(starts[1:], start=1):
        ax.axvline(start + 1, color="0.35", linestyle=":", linewidth=1.0)
        ax.text(
            start + 1,
            0.98,
            f"sigma={sigmas[stage_index]:g}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="0.25",
        )


def _plot_components(ax: plt.Axes, record: Mapping[str, Any], title: str) -> None:
    history = record["history"]
    steps = history["zero_step"] + 1
    curves = (
        ("loss_total", "total", "black"),
        ("loss_similarity", "similarity", "#377eb8"),
        ("loss_spatial_weighted", "regularization", "#4daf4a"),
        ("loss_mask_weighted", "semantic mask", "#e41a1c"),
    )
    for field, label, color in curves:
        ax.plot(steps, history[field], label=label, color=color, linewidth=1.0)
    _stage_lines(ax, record["trace"])
    ax.set_title(title)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("weighted loss")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, frameon=False)


def _file_input(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size)}


def _case_data_inputs(data_root: Path, case: int) -> list[dict[str, Any]]:
    case_dir = data_root / f"case_{case:02d}"
    paths = [
        case_dir / "images" / "phase_00.nii",
        case_dir / "images" / "phase_05.nii",
        case_dir / "segmentations_00.nii",
        case_dir / "segmentations_05.nii",
        case_dir / "landmarks_raw" / "extreme_landmarks_0.csv",
        case_dir / "landmarks_raw" / "extreme_landmarks_5.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ReviewError(f"Author-format case {case:02d} lacks inputs: {missing}")
    return [_file_input(path) for path in paths]


def prepare_case_review(
    *, api: Any, pair: Mapping[str, Any], data_root: Path, slice_axis: int
) -> dict[str, Any]:
    case = int(pair["case"])
    case_folder = data_root / f"case_{case:02d}"
    data = api.utils.load_and_crop_full_dirlab(
        case_folder,
        fixed_phase=0,
        moving_phase=5,
        crop=True,
    )
    fixed = data["images"]["fixed"].astype(np.float32, copy=False)
    moving = data["images"]["moving"].astype(np.float32, copy=False)
    fixed_masks = data["lung_classes_masks"]["fixed"].astype(np.float32, copy=False)
    moving_masks = data["lung_classes_masks"]["moving"].astype(np.float32, copy=False)
    if pair["a0"]["dense_dvf"].shape[:3] != fixed.shape:
        raise ReviewError(
            f"Case {case:02d} data crop {fixed.shape} differs from run DVF "
            f"{pair['a0']['dense_dvf'].shape[:3]}"
        )
    if tuple(pair["a0"]["summary"].get("image_spacing_xyz", ())) != tuple(
        data["image_spacing"]
    ):
        raise ReviewError(f"Case {case:02d} spacing differs between data and run")

    fixed_union = np.any(fixed_masks > 0.5, axis=0)
    reduction_axes = tuple(axis for axis in range(3) if axis != slice_axis)
    slice_areas = fixed_union.sum(axis=reduction_axes)
    slice_index = int(np.argmax(slice_areas))
    foreground_values = np.concatenate(
        [fixed[fixed_union], moving[np.any(moving_masks > 0.5, axis=0)]]
    )
    intensity_vmin, intensity_vmax = np.percentile(foreground_values, [1.0, 99.0])
    if intensity_vmax <= intensity_vmin:
        intensity_vmin, intensity_vmax = float(fixed.min()), float(fixed.max() + 1e-6)

    spacing = np.asarray(data["image_spacing"], dtype=np.float32)
    variants: dict[str, dict[str, Any]] = {}
    for key in ("a0", "a1"):
        dvf = pair[key]["dense_dvf"]
        warped_image = _warp_with_dvf(api=api, volume=moving, dense_dvf=dvf, order=1)
        warped_masks = np.stack(
            [
                _warp_with_dvf(api=api, volume=mask, dense_dvf=dvf, order=0)
                for mask in moving_masks
            ],
            axis=0,
        )
        magnitude_mm = np.linalg.norm(dvf * spacing.reshape((1, 1, 1, 3)), axis=-1)
        variants[key] = {
            "warped_image": warped_image,
            "warped_masks": warped_masks,
            "magnitude_mm": magnitude_mm,
        }

    magnitude_values = np.concatenate(
        [variants[key]["magnitude_mm"][fixed_union] for key in ("a0", "a1")]
    )
    magnitude_vmax = max(float(np.percentile(magnitude_values, 99.0)), 1e-6)
    magnitude_difference = variants["a1"]["magnitude_mm"] - variants["a0"][
        "magnitude_mm"
    ]
    warped_difference = variants["a1"]["warped_image"] - variants["a0"][
        "warped_image"
    ]
    magnitude_difference_limit = max(
        float(np.percentile(np.abs(magnitude_difference[fixed_union]), 99.0)), 1e-6
    )
    warped_difference_limit = max(
        float(np.percentile(np.abs(warped_difference[fixed_union]), 99.0)), 1e-6
    )
    return {
        "pair": pair,
        "fixed": fixed,
        "moving": moving,
        "fixed_masks": fixed_masks,
        "moving_masks": moving_masks,
        "variants": variants,
        "magnitude_difference": magnitude_difference,
        "warped_difference": warped_difference,
        "slice_axis": slice_axis,
        "slice_index": slice_index,
        "slice_area_voxels": int(slice_areas[slice_index]),
        "image_spacing_xyz": [float(value) for value in spacing],
        "scales": {
            "intensity_vmin": float(intensity_vmin),
            "intensity_vmax": float(intensity_vmax),
            "displacement_magnitude_vmin_mm": 0.0,
            "displacement_magnitude_vmax_mm": magnitude_vmax,
            "magnitude_difference_abs_limit_mm": magnitude_difference_limit,
            "warped_intensity_difference_abs_limit": warped_difference_limit,
        },
        "data_inputs": _case_data_inputs(data_root, case),
    }


def render_case_review(prepared: Mapping[str, Any], output_path: Path, dpi: int) -> None:
    pair = prepared["pair"]
    case = int(pair["case"])
    axis = int(prepared["slice_axis"])
    index = int(prepared["slice_index"])
    scales = prepared["scales"]
    a0_visual = prepared["variants"]["a0"]
    a1_visual = prepared["variants"]["a1"]

    figure, axes = plt.subplots(3, 4, figsize=(19, 13), constrained_layout=True)
    _show_anatomy(
        axes[0, 0],
        prepared["fixed"],
        axis=axis,
        index=index,
        vmin=scales["intensity_vmin"],
        vmax=scales["intensity_vmax"],
        title="Fixed T00 + fixed lobe boundaries",
        fixed_masks=prepared["fixed_masks"],
    )
    _show_anatomy(
        axes[0, 1],
        prepared["moving"],
        axis=axis,
        index=index,
        vmin=scales["intensity_vmin"],
        vmax=scales["intensity_vmax"],
        title="Moving T50 + moving lobe boundaries",
        warped_masks=prepared["moving_masks"],
    )
    _show_anatomy(
        axes[0, 2],
        a0_visual["warped_image"],
        axis=axis,
        index=index,
        vmin=scales["intensity_vmin"],
        vmax=scales["intensity_vmax"],
        title="A0 warped T50: fixed solid / warped dashed",
        fixed_masks=prepared["fixed_masks"],
        warped_masks=a0_visual["warped_masks"],
    )
    _show_anatomy(
        axes[0, 3],
        a1_visual["warped_image"],
        axis=axis,
        index=index,
        vmin=scales["intensity_vmin"],
        vmax=scales["intensity_vmax"],
        title="A1 warped T50: fixed solid / warped dashed",
        fixed_masks=prepared["fixed_masks"],
        warped_masks=a1_visual["warped_masks"],
    )

    for column, (visual, title) in enumerate(
        ((a0_visual, "A0 displacement magnitude"), (a1_visual, "A1 displacement magnitude"))
    ):
        image = axes[1, column].imshow(
            _plane(visual["magnitude_mm"], axis, index),
            cmap="magma",
            origin="lower",
            vmin=0.0,
            vmax=scales["displacement_magnitude_vmax_mm"],
        )
        axes[1, column].set_title(f"{title} (mm)")
        axes[1, column].set_axis_off()
        figure.colorbar(image, ax=axes[1, column], fraction=0.046, pad=0.02)

    image = axes[1, 2].imshow(
        _plane(prepared["magnitude_difference"], axis, index),
        cmap="coolwarm",
        origin="lower",
        vmin=-scales["magnitude_difference_abs_limit_mm"],
        vmax=scales["magnitude_difference_abs_limit_mm"],
    )
    axes[1, 2].set_title("A1 - A0 displacement magnitude (mm)")
    axes[1, 2].set_axis_off()
    figure.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.02)
    image = axes[1, 3].imshow(
        _plane(prepared["warped_difference"], axis, index),
        cmap="coolwarm",
        origin="lower",
        vmin=-scales["warped_intensity_difference_abs_limit"],
        vmax=scales["warped_intensity_difference_abs_limit"],
    )
    axes[1, 3].set_title("A1 - A0 warped intensity")
    axes[1, 3].set_axis_off()
    figure.colorbar(image, ax=axes[1, 3], fraction=0.046, pad=0.02)

    _plot_components(axes[2, 0], pair["a0"], "A0 loss components")
    _plot_components(axes[2, 1], pair["a1"], "A1 loss components")
    steps = pair["a0"]["history"]["zero_step"] + 1
    axes[2, 2].plot(
        steps, pair["a0"]["history"]["loss_total"], label="A0 total", color="#377eb8"
    )
    axes[2, 2].plot(
        steps, pair["a1"]["history"]["loss_total"], label="A1 total", color="#e41a1c"
    )
    _stage_lines(axes[2, 2], pair["a0"]["trace"])
    axes[2, 2].set_title("Paired total loss")
    axes[2, 2].set_xlabel("optimization step")
    axes[2, 2].set_ylabel("loss")
    axes[2, 2].grid(alpha=0.2)
    axes[2, 2].legend(frameon=False)

    difference_curves = (
        ("loss_total", "total", "black"),
        ("loss_similarity", "similarity", "#377eb8"),
        ("loss_spatial_weighted", "regularization", "#4daf4a"),
        ("loss_mask_weighted", "semantic mask", "#e41a1c"),
    )
    for field, label, color in difference_curves:
        difference = pair["a1"]["history"][field] - pair["a0"]["history"][field]
        axes[2, 3].plot(steps, difference, label=label, color=color, linewidth=1.0)
    axes[2, 3].axhline(0.0, color="0.3", linewidth=0.8)
    _stage_lines(axes[2, 3], pair["a0"]["trace"])
    axes[2, 3].set_title("A1 - A0 loss components")
    axes[2, 3].set_xlabel("optimization step")
    axes[2, 3].set_ylabel("loss difference")
    axes[2, 3].grid(alpha=0.2)
    axes[2, 3].legend(fontsize=7, ncol=2, frameon=False)

    figure.suptitle(
        f"DUAL-INR-DIR paired author review | case {case:02d} | "
        f"axis {axis}, slice {index} | seed {pair['a0']['seed']}",
        fontsize=15,
        y=1.015,
    )
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.png")
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def generate_review(
    *,
    run_root: Path,
    data_root: Path,
    review_output: Path,
    cases: Sequence[int] = DEFAULT_CASES,
    slice_axis: int = 2,
    dpi: int = 180,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    data_root = data_root.resolve()
    review_output = review_output.resolve()
    if not run_root.is_dir() or not data_root.is_dir():
        raise ReviewError("run_root and data_root must both exist as directories")
    if slice_axis not in {0, 1, 2}:
        raise ReviewError("slice_axis must be 0, 1, or 2")
    if dpi < 72 or dpi > 400:
        raise ReviewError("dpi must be in 72..400")
    if not cases:
        raise ReviewError("At least one case is required")

    api = load_external_api()
    prepared_cases = []
    for case in cases:
        pair = load_paired_case(
            run_root=run_root,
            data_root=data_root,
            case=int(case),
            allow_smoke=allow_smoke,
        )
        prepared_cases.append(
            prepare_case_review(
                api=api,
                pair=pair,
                data_root=data_root,
                slice_axis=slice_axis,
            )
        )

    review_output.mkdir(parents=True, exist_ok=True)
    manifest_path = review_output / "author_review_manifest.json"
    targets = [
        review_output / f"case_{int(prepared['pair']['case']):02d}_author_review.png"
        for prepared in prepared_cases
    ]
    existing = [str(path) for path in [*targets, manifest_path] if path.exists()]
    if existing:
        raise ReviewError(f"Refusing to overwrite existing review artifacts: {existing}")

    case_manifests = []
    for prepared, target in zip(prepared_cases, targets):
        render_case_review(prepared, target, dpi)
        pair = prepared["pair"]
        case_manifests.append(
            {
                "case": int(pair["case"]),
                "seed": int(pair["a0"]["seed"]),
                "run_mode": pair["a0"]["run_mode"],
                "figure": _file_input(target),
                "slice": {
                    "axis": int(prepared["slice_axis"]),
                    "index": int(prepared["slice_index"]),
                    "selection_rule": "argmax_fixed_lobe_union_area",
                    "fixed_lobe_area_voxels": int(prepared["slice_area_voxels"]),
                    "display_transform": "selected_numpy_plane_transposed_origin_lower",
                },
                "scales": prepared["scales"],
                "image_spacing_xyz": prepared["image_spacing_xyz"],
                "stage_start_steps": pair["a0"]["trace"]["stage_start_steps"],
                "inputs": {
                    "data": prepared["data_inputs"],
                    "a0_summary": _file_input(pair["a0"]["case_dir"] / "summary.json"),
                    "a1_summary": _file_input(pair["a1"]["case_dir"] / "summary.json"),
                    "a0_history": _file_input(pair["a0"]["case_dir"] / "history.csv"),
                    "a1_history": _file_input(pair["a1"]["case_dir"] / "history.csv"),
                    "a0_dense_dvf": _file_input(
                        pair["a0"]["case_dir"] / "dense_dvf_crop_voxel_xyz.npy"
                    ),
                    "a1_dense_dvf": _file_input(
                        pair["a1"]["case_dir"] / "dense_dvf_crop_voxel_xyz.npy"
                    ),
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_commit": api.commit,
        "run_root": str(run_root),
        "data_root": str(data_root),
        "review_output": str(review_output),
        "cases": [int(case) for case in cases],
        "dpi": int(dpi),
        "warping": {
            "direction": "moving_sampled_at_fixed_plus_fixed_to_moving_dvf",
            "image_interpolation": "scipy_map_coordinates_order1_mode_nearest",
            "lobe_interpolation": "scipy_map_coordinates_order0_mode_nearest",
            "displacement_units": "voxel_xyz_converted_to_mm_for_magnitude",
        },
        "boundary_style": {
            "fixed": "solid",
            "moving_or_warped": "dashed",
            "lobe_labels": list(LOBE_LABELS),
            "lobe_colors": list(LOBE_COLORS),
        },
        "case_reviews": case_manifests,
    }
    review_manifest = portable_review_value(
        manifest,
        (
            (run_root, "${RUN_ROOT}"),
            (data_root, "${DATA_ROOT}"),
            (review_output, "${REVIEW_ROOT}"),
            (REPOSITORY_ROOT, "${REPO_ROOT}"),
        ),
    )
    _write_manifest(manifest_path, review_manifest)
    return review_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=list(DEFAULT_CASES))
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Explicitly permit completed short-smoke records; full runs are required by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_review(
        run_root=args.run_root,
        data_root=args.data_root,
        review_output=args.review_output,
        cases=args.cases,
        slice_axis=args.slice_axis,
        dpi=args.dpi,
        allow_smoke=args.allow_smoke,
    )
    print(
        f"Wrote {len(manifest['case_reviews'])} paired review figure(s) and "
        f"{Path(manifest['review_output']) / 'author_review_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
