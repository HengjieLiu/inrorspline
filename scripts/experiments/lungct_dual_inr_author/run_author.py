#!/usr/bin/env python3
"""Run author-faithful DUAL-INR-DIR A0/A1 cases on converted DIR-Lab data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.experiments.lungct_dual_inr_author.adapter import (  # type: ignore
        build_author_config,
        load_transition_manifest,
        resolve_device,
        seed_everything,
        train_author,
    )
    from scripts.experiments.lungct_dual_inr_author.external_api import (  # type: ignore
        load_external_api,
        git_directory,
        repository_root,
    )
else:
    from .adapter import (
        build_author_config,
        load_transition_manifest,
        resolve_device,
        seed_everything,
        train_author,
    )
    from .external_api import git_directory, load_external_api, repository_root


ALL_DIRLAB_4DCT_CASES = tuple(range(1, 11))


def parse_cohort(value: str) -> list[int]:
    normalized = value.strip().lower()
    if normalized in {"all10", "all", "dirlab10"}:
        return list(ALL_DIRLAB_4DCT_CASES)
    try:
        cases = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cohort must be 'all10' or a comma-separated case list"
        ) from exc
    if not cases:
        raise argparse.ArgumentTypeError("cohort case list cannot be empty")
    invalid = [case for case in cases if case not in ALL_DIRLAB_4DCT_CASES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"DIR-Lab 4DCT cases must be 1..10, got {invalid}"
        )
    if len(cases) != len(set(cases)):
        raise argparse.ArgumentTypeError("cohort case list contains duplicates")
    return cases


def validate_case(case: str | int) -> int:
    case = int(case)
    if case not in ALL_DIRLAB_4DCT_CASES:
        raise argparse.ArgumentTypeError("DIR-Lab 4DCT case must be in 1..10")
    return case


def validate_smoke_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("smoke steps must be an integer") from exc
    if not 1 <= steps <= 50:
        raise argparse.ArgumentTypeError(
            "smoke steps must be in 1..50; full runs use the locked 3000-step default"
        )
    return steps


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_roots(data_root: Path, output_root: Path) -> tuple[Path, Path]:
    if not data_root.is_absolute():
        raise ValueError("--data-root must be an absolute path")
    if not output_root.is_absolute():
        raise ValueError("--output-root must be an explicit absolute scratch path")
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    repo_root = repository_root().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Converted DIR-Lab data root does not exist: {data_root}")
    if _is_below(output_root, repo_root):
        raise ValueError(
            f"Outputs must stay outside the Git repository ({repo_root}), got {output_root}"
        )
    if _is_below(output_root, data_root):
        raise ValueError(
            f"Outputs must stay outside the input data root ({data_root}), got {output_root}"
        )
    return data_root, output_root


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _run_git(path: Path, *arguments: str, check: bool) -> subprocess.CompletedProcess:
    path = path.resolve()
    return subprocess.run(
        [
            "git",
            f"--git-dir={git_directory(path)}",
            f"--work-tree={path}",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=check,
    )


def _git_head(path: Path) -> str | None:
    result = _run_git(path, "rev-parse", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status_porcelain(path: Path) -> str:
    result = _run_git(path, "status", "--porcelain", check=True)
    return result.stdout.strip()


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{line}\n")


def adapter_source_bundle() -> dict[str, Any]:
    """Hash the exact uncommitted-capable source files that define a case run."""

    root = repository_root().resolve()
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("adapter.py").resolve(),
        Path(__file__).with_name("external_api.py").resolve(),
    ]
    bundle_hasher = hashlib.sha256()
    records = []
    for path in sorted(source_paths, key=lambda item: item.name):
        data = path.read_bytes()
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        bundle_hasher.update(relative_path.encode("utf-8"))
        bundle_hasher.update(b"\0")
        bundle_hasher.update(str(len(data)).encode("ascii"))
        bundle_hasher.update(b"\0")
        bundle_hasher.update(data)
        bundle_hasher.update(b"\0")
        records.append(
            {
                "path": str(path),
                "repository_relative_path": relative_path,
                "size_bytes": len(data),
                "sha256": digest,
            }
        )
    status = _run_git(
        root, "status", "--porcelain=v1", "--untracked-files=all", check=True
    ).stdout
    status_lines = [line for line in status.splitlines() if line]
    return {
        "algorithm": "sha256",
        "bundle_sha256": bundle_hasher.hexdigest(),
        "bundle_definition": (
            "SHA256 over each source sorted by filename as "
            "relative_path NUL size_bytes NUL raw_bytes NUL"
        ),
        "sources": records,
        "main_worktree_snapshot": {
            "dirty": bool(status_lines),
            "status_porcelain_sha256": hashlib.sha256(
                status.encode("utf-8")
            ).hexdigest(),
            "status_entry_count": len(status_lines),
            "untracked_entry_count": sum(line.startswith("?? ") for line in status_lines),
            "limitation": (
                "The status digest identifies the path/status listing, not tracked diff "
                "contents or unrelated untracked file bytes. The executable case-run "
                "sources are independently captured byte-for-byte by this bundle."
            ),
        },
    }


def _candidate_trace_paths(root: Path, case: int) -> list[Path]:
    case_name = f"case_{case:02d}"
    return [
        root / "a0" / case_name / "stage_transitions.json",
        root / "A0" / case_name / "stage_transitions.json",
        root / case_name / "a0" / "stage_transitions.json",
        root / case_name / "A0" / "stage_transitions.json",
        root / case_name / "stage_transitions.json",
        root / "stage_transitions.json",
    ]


def resolve_case_transition_trace(
    trace_input: Path,
    *,
    case: int,
    total_steps: int,
    blur_sigmas: list[float],
    allow_direct_file: bool,
) -> tuple[Path, dict[str, Any]]:
    trace_input = trace_input.resolve()
    if trace_input.is_file():
        if not allow_direct_file:
            raise ValueError(
                "A direct trace file is only allowed for a single-case A1 run; "
                "pass a trace root for cohorts"
            )
        selected = trace_input
    elif trace_input.is_dir():
        matches = [path for path in _candidate_trace_paths(trace_input, case) if path.is_file()]
        unique_matches = list(dict.fromkeys(path.resolve() for path in matches))
        if len(unique_matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one A0 trace for case {case:02d} below {trace_input}; "
                f"found {len(unique_matches)}"
            )
        selected = unique_matches[0]
    else:
        raise FileNotFoundError(f"A0 transition trace input does not exist: {trace_input}")

    manifest = load_transition_manifest(
        selected,
        total_steps=total_steps,
        blur_sigmas=blur_sigmas,
        require_a0=True,
    )
    if "case" not in manifest:
        raise ValueError(f"A0 transition trace lacks required case identity: {selected}")
    if int(manifest["case"]) != case:
        raise ValueError(
            f"A0 transition trace case mismatch: requested {case:02d}, "
            f"trace records {int(manifest['case']):02d}"
        )
    return selected, manifest


def _metric_summary(distances: np.ndarray) -> dict[str, Any]:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0 or not np.isfinite(distances).all():
        raise FloatingPointError(
            "Landmark distance vector must be non-empty and entirely finite"
        )
    return {
        "n": int(distances.size),
        "mean_mm": float(np.mean(distances)),
        "std_mm_population": float(np.std(distances)),
        "median_mm": float(np.median(distances)),
        "min_mm": float(np.min(distances)),
        "max_mm": float(np.max(distances)),
    }


def evaluate_landmarks(
    *,
    api: Any,
    model: torch.nn.Module,
    fixed_landmarks: np.ndarray,
    moving_landmarks: np.ndarray,
    image_shape: tuple[int, int, int],
    image_spacing: tuple[float, float, float],
    dense_dvf: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    initial_snap_tre, initial_snap_warped = api.utils.compute_tre_dense_dvf(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=image_spacing,
        snap_to_voxel=True,
    )
    initial_subvoxel_tre, initial_subvoxel_warped = api.utils.compute_tre_dense_dvf(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=None,
        image_spacing=image_spacing,
        snap_to_voxel=False,
    )
    dense_snap_tre, dense_snap_warped = api.utils.compute_tre_dense_dvf(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=np.moveaxis(dense_dvf, -1, 0),
        image_spacing=image_spacing,
        snap_to_voxel=True,
    )
    dense_subvoxel_tre, dense_subvoxel_warped = api.utils.compute_tre_dense_dvf(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=np.moveaxis(dense_dvf, -1, 0),
        image_spacing=image_spacing,
        snap_to_voxel=False,
    )

    device = next(model.parameters()).device
    fixed_t = torch.as_tensor(fixed_landmarks, device=device, dtype=torch.float32)
    fixed_norm = api.utils.pixel_coords_to_norm(fixed_t, image_shape)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        direct_norm = fixed_norm + model(fixed_norm)
        direct_vox = api.utils.norm_to_voxel_coords(direct_norm, image_shape)
    if was_training:
        model.train()
    direct_subvoxel_warped = direct_vox.detach().cpu().numpy()
    direct_snap_warped = np.round(direct_subvoxel_warped)
    moving_snapped = np.round(moving_landmarks)
    direct_snap_tre = np.linalg.norm(
        (direct_snap_warped - moving_snapped) * np.asarray(image_spacing), axis=1
    )
    direct_subvoxel_tre = np.linalg.norm(
        (direct_subvoxel_warped - moving_landmarks) * np.asarray(image_spacing),
        axis=1,
    )

    metrics = {
        "coordinate_frame": "author_cropped_voxel_xyz",
        "spacing_order": "xyz",
        "headline_mode": "snap_to_voxel",
        "initial": {
            "snap": _metric_summary(initial_snap_tre),
            "subvoxel": _metric_summary(initial_subvoxel_tre),
        },
        "direct_inr": {
            "snap": _metric_summary(direct_snap_tre),
            "subvoxel": _metric_summary(direct_subvoxel_tre),
        },
        "dense_dvf": {
            "snap": _metric_summary(dense_snap_tre),
            "subvoxel": _metric_summary(dense_subvoxel_tre),
        },
        "direct_dense_agreement": {
            "snap_mean_tre_absolute_difference_mm": abs(
                float(np.mean(direct_snap_tre)) - float(np.mean(dense_snap_tre))
            ),
            "subvoxel_mean_tre_absolute_difference_mm": abs(
                float(np.mean(direct_subvoxel_tre))
                - float(np.mean(dense_subvoxel_tre))
            ),
            "subvoxel_warped_point_rmse_mm": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                (direct_subvoxel_warped - dense_subvoxel_warped)
                                * np.asarray(image_spacing)
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            ),
        },
    }
    arrays = {
        "fixed_landmarks_crop_xyz": np.asarray(fixed_landmarks),
        "moving_landmarks_crop_xyz": np.asarray(moving_landmarks),
        "initial_snap_warped_xyz": np.asarray(initial_snap_warped),
        "initial_subvoxel_warped_xyz": np.asarray(initial_subvoxel_warped),
        "direct_snap_warped_xyz": np.asarray(direct_snap_warped),
        "direct_subvoxel_warped_xyz": np.asarray(direct_subvoxel_warped),
        "dense_snap_warped_xyz": np.asarray(dense_snap_warped),
        "dense_subvoxel_warped_xyz": np.asarray(dense_subvoxel_warped),
        "initial_snap_tre_mm": np.asarray(initial_snap_tre),
        "initial_subvoxel_tre_mm": np.asarray(initial_subvoxel_tre),
        "direct_snap_tre_mm": np.asarray(direct_snap_tre),
        "direct_subvoxel_tre_mm": np.asarray(direct_subvoxel_tre),
        "dense_snap_tre_mm": np.asarray(dense_snap_tre),
        "dense_subvoxel_tre_mm": np.asarray(dense_subvoxel_tre),
    }
    return metrics, arrays


def compute_five_lobe_dice(
    *,
    api: Any,
    fixed_masks: np.ndarray,
    moving_masks: np.ndarray,
    dense_dvf: np.ndarray,
) -> dict[str, Any]:
    """Warp moving lobes into fixed space with the decoded fixed-to-moving DVF."""

    image_shape = fixed_masks.shape[1:]
    fixed_grid = np.stack(
        np.meshgrid(
            *[np.arange(size, dtype=np.float32) for size in image_shape],
            indexing="ij",
        ),
        axis=0,
    )
    sample_coords = fixed_grid + np.moveaxis(dense_dvf, -1, 0)
    per_lobe: list[dict[str, Any]] = []
    for channel, label_id in enumerate((10, 11, 12, 13, 14)):
        warped = api.utils.ndi.map_coordinates(
            moving_masks[channel].astype(np.float32, copy=False),
            sample_coords,
            order=0,
            mode="nearest",
            prefilter=False,
        ) > 0.5
        fixed = fixed_masks[channel] > 0.5
        denominator = int(fixed.sum()) + int(warped.sum())
        dice = (
            1.0
            if denominator == 0
            else float(2.0 * np.logical_and(fixed, warped).sum() / denominator)
        )
        per_lobe.append(
            {
                "label_id": label_id,
                "channel": channel,
                "dice": dice,
                "fixed_voxels": int(fixed.sum()),
                "warped_moving_voxels": int(warped.sum()),
            }
        )
    return {
        "direction": "moving_lobes_sampled_at_fixed_plus_dvf_into_fixed_crop",
        "dvf_convention": "fixed_to_moving_voxel_xyz",
        "interpolation": "nearest_neighbor_order0_mode_nearest",
        "binarization": "per_lobe_channel_greater_than_0.5",
        "domain": "entire_author_crop",
        "per_lobe": per_lobe,
        "mean_dice": float(np.mean([item["dice"] for item in per_lobe])),
    }


def _write_stage_trace_csv(
    path: Path, transition_manifest: dict[str, Any]
) -> None:
    starts = list(transition_manifest["stage_start_steps"])
    sigmas = list(transition_manifest["blur_sigmas"])
    events_by_start = {
        int(event["next_stage_start_step"]): event
        for event in transition_manifest.get("events", [])
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "stage_index",
            "blur_sigma",
            "start_zero_step",
            "end_zero_step_inclusive",
            "trigger",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for stage_index, start in enumerate(starts):
            end = (
                starts[stage_index + 1] - 1
                if stage_index + 1 < len(starts)
                else int(transition_manifest["total_steps"]) - 1
            )
            writer.writerow(
                {
                    "stage_index": stage_index,
                    "blur_sigma": sigmas[stage_index],
                    "start_zero_step": start,
                    "end_zero_step_inclusive": end,
                    "trigger": (
                        "initial"
                        if stage_index == 0
                        else events_by_start.get(start, {}).get("trigger", "unknown")
                    ),
                }
            )


def _pad_dvf_to_full(
    dvf_crop: np.ndarray,
    bbox: tuple[slice, slice, slice] | None,
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    if bbox is None:
        return dvf_crop
    full_dvf = np.zeros(full_shape + (3,), dtype=dvf_crop.dtype)
    full_dvf[bbox] = dvf_crop
    return full_dvf


def _save_author_dvf_nifti(
    *,
    api: Any,
    dvf_voxel_xyz: np.ndarray,
    image_spacing_xyz: tuple[float, float, float],
    path: Path,
) -> None:
    image = api.utils.sitk.GetImageFromArray(
        np.swapaxes(dvf_voxel_xyz, 0, 2), isVector=True
    )
    image.SetSpacing(tuple(image_spacing_xyz))
    api.utils.sitk.WriteImage(image, str(path))


def _new_case_output(output_root: Path, variant: str, case: int) -> Path:
    output_dir = output_root / variant.lower() / f"case_{case:02d}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to reuse or overwrite existing case output: {output_dir}"
        ) from exc
    return output_dir


def run_case(
    *,
    variant: str,
    case: int,
    data_root: Path,
    output_root: Path,
    device: str,
    seed: int,
    trace_input: Path | None,
    single_case_selection: bool,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    api = load_external_api()
    external_status = _git_status_porcelain(api.repository)
    if external_status:
        raise RuntimeError(
            "DUAL-INR-DIR checkout must be clean for an author run; git status:\n"
            f"{external_status}"
        )
    config = build_author_config(variant)
    run_mode = "short_smoke" if smoke_steps is not None else "full_length"
    if smoke_steps is not None:
        config["training"]["total_steps"] = int(smoke_steps)
        config["adapter"]["run_mode"] = run_mode
        config["adapter"]["smoke_steps"] = int(smoke_steps)
    config["paths"]["4dct_path"] = data_root
    resolved_device = resolve_device(device)
    seed_everything(seed)

    fixed_phase = int(config["fixed_phase"])
    moving_phase = int(config["moving_phase"])
    case_folder = data_root / f"case_{case:02d}"
    data = api.utils.load_and_crop_full_dirlab(
        case_folder,
        fixed_phase=fixed_phase,
        moving_phase=moving_phase,
        crop=True,
    )
    images = data["images"]
    masks = data["lung_classes_masks"]
    output_dir = _new_case_output(output_root, variant, case)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stderr_path.touch()
    _append_line(
        stdout_path,
        f"START variant={variant} case={case:02d} seed={seed} device={resolved_device}",
    )
    partial_output_policy = {
        "on_failure": (
            "Preserve all files already written, omit DONE, set RUN_STATE status "
            "to failed, and never treat the directory as an accepted record."
        ),
        "resume": "No in-place resume or overwrite; use a fresh output root.",
    }
    write_json(
        output_dir / "RUN_STATE.json",
        {
            "status": "running",
            "variant": variant,
            "case": case,
            "seed": seed,
            "run_mode": run_mode,
            "smoke_steps": smoke_steps,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "partial_output_policy": partial_output_policy,
        },
    )

    replay_path: Path | None = None
    replay_manifest: dict[str, Any] | None = None
    blur_sigmas = [float(value) for value in config["training"]["blur_sigmas"]]
    if variant == "A1":
        if trace_input is None:
            raise ValueError("A1 requires --replay-a0-transitions")
        replay_path, replay_manifest = resolve_case_transition_trace(
            trace_input,
            case=case,
            total_steps=int(config["training"]["total_steps"]),
            blur_sigmas=blur_sigmas,
            allow_direct_file=single_case_selection,
        )
    elif trace_input is not None:
        raise ValueError("--replay-a0-transitions is valid only for A1")

    source_bundle = adapter_source_bundle()
    config["export"] = {
        "dataset": "dirlab_4dct",
        "case": case,
        "fixed_phase": fixed_phase,
        "moving_phase": moving_phase,
        "seed": seed,
        "device_requested": str(device),
        "device_resolved": str(resolved_device),
        "output_dir": output_dir,
        "a0_transition_trace": replay_path,
        "run_mode": run_mode,
        "smoke_steps": smoke_steps,
        "adapter_bundle_sha256": source_bundle["bundle_sha256"],
    }
    write_json(output_dir / "resolved_config.json", config)
    api.utils.save_config(output_dir / "resolved_config.yaml", config)

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_repository_commit": _git_head(repository_root()),
        "external_repository": str(api.repository),
        "external_commit": api.commit,
        "external_git_status_porcelain": external_status,
        "external_package_dir": str(api.package_dir),
        "adapter_file": str(Path(__file__).resolve()),
        "adapter_source_bundle": source_bundle,
        "variant": variant,
        "case": case,
        "seed": seed,
        "run_mode": run_mode,
        "smoke_steps": smoke_steps,
        "adapter_bundle_sha256": source_bundle["bundle_sha256"],
        "data_root": str(data_root),
        "case_folder": str(case_folder),
        "output_dir": str(output_dir),
        "a0_transition_trace": str(replay_path) if replay_path else None,
        "device_requested": str(device),
        "device_resolved": str(resolved_device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "container_image_env": os.environ.get("CODEX_CONTAINER_IMAGE"),
        "optimizer_resolved": config["optimizer"],
        "partial_output_policy": partial_output_policy,
        "objective": {
            "similarity": "external inrdir.ncc.lncc sampled global NCC",
            "spatial": "external determinant hinge plus squared Laplacian per branch",
            "fine_regularization_scale": config["loss"]["fine_reg_scale"],
            "mask_weight": config["loss"]["mask_weight"],
        },
    }
    write_json(output_dir / "provenance.json", provenance)

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    training_started = time.perf_counter()
    history_path = output_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as history_stream:
        writer: csv.DictWriter | None = None

        def record_callback(record: dict[str, Any]) -> None:
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(history_stream, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            history_stream.flush()

        result = train_author(
            fixed_image=images["fixed"],
            moving_image=images["moving"],
            fixed_lung_classes_mask=masks["fixed"],
            moving_lung_classes_mask=masks["moving"],
            config=config,
            device=resolved_device,
            source_variant=variant,
            replay_transition_manifest=replay_manifest,
            record_callback=record_callback,
        )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        training_peak_allocated = int(torch.cuda.max_memory_allocated(resolved_device))
        training_peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
    else:
        training_peak_allocated = None
        training_peak_reserved = None
    training_wall_seconds = time.perf_counter() - training_started

    transition_manifest = dict(result["transition_manifest"])
    transition_manifest["case"] = case
    transition_manifest["seed"] = seed
    write_json(output_dir / "stage_transitions.json", transition_manifest)
    _write_stage_trace_csv(output_dir / "stage_trace.csv", transition_manifest)
    if smoke_steps is None and len(transition_manifest["stage_start_steps"]) != len(
        transition_manifest["blur_sigmas"]
    ):
        raise RuntimeError(
            "Full-length run did not visit every locked blur stage: "
            f"starts={transition_manifest['stage_start_steps']}, "
            f"sigmas={transition_manifest['blur_sigmas']}"
        )

    model = result["model"]
    if smoke_steps is None and not bool(model.fine_branch_is_active):
        raise RuntimeError("Full-length run never activated the fine branch")
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output_dir / "inr_dvf.pth",
    )
    model_config = copy.deepcopy(config)
    model_config["model"]["fine_branch_is_active"] = bool(
        model.fine_branch_is_active
    )
    api.utils.save_config(output_dir / "inr_dvf_config.yaml", model_config)
    dense_dvf = api.utils.inr_to_dense_dvf(
        model=model,
        image_shape=tuple(images["fixed"].shape),
        device=resolved_device,
    )
    if not np.isfinite(dense_dvf).all():
        raise FloatingPointError("Decoded dense DVF contains non-finite values")
    np.save(output_dir / "dense_dvf_crop_voxel_xyz.npy", dense_dvf)
    dense_dvf_full = _pad_dvf_to_full(
        dense_dvf,
        data["bbox"],
        tuple(data["original_shape"]),
    )
    _save_author_dvf_nifti(
        api=api,
        dvf_voxel_xyz=dense_dvf_full,
        image_spacing_xyz=tuple(data["image_spacing"]),
        path=output_dir / "dirinr_dvf.nii.gz",
    )

    metrics, landmark_arrays = evaluate_landmarks(
        api=api,
        model=model,
        fixed_landmarks=data["extreme_landmarks"]["fixed"],
        moving_landmarks=data["extreme_landmarks"]["moving"],
        image_shape=tuple(images["fixed"].shape),
        image_spacing=tuple(data["image_spacing"]),
        dense_dvf=dense_dvf,
    )
    write_json(output_dir / "landmark_metrics.json", metrics)
    np.savez_compressed(output_dir / "landmark_details.npz", **landmark_arrays)
    lobe_dice = compute_five_lobe_dice(
        api=api,
        fixed_masks=masks["fixed"],
        moving_masks=masks["moving"],
        dense_dvf=dense_dvf,
    )
    write_json(output_dir / "five_lobe_dice.json", lobe_dice)

    tre_lines = [
        (
            f"START TRE(extreme, snap) mean="
            f"{metrics['initial']['snap']['mean_mm']:.6f} mm "
            f"std={metrics['initial']['snap']['std_mm_population']:.6f} mm"
        ),
        (
            f"START TRE(extreme, subvoxel) mean="
            f"{metrics['initial']['subvoxel']['mean_mm']:.6f} mm "
            f"std={metrics['initial']['subvoxel']['std_mm_population']:.6f} mm"
        ),
        (
            f"TRE(extreme, INR, snap) mean="
            f"{metrics['direct_inr']['snap']['mean_mm']:.6f} mm "
            f"std={metrics['direct_inr']['snap']['std_mm_population']:.6f} mm"
        ),
        (
            f"TRE(extreme, INR, subvoxel) mean="
            f"{metrics['direct_inr']['subvoxel']['mean_mm']:.6f} mm "
            f"std={metrics['direct_inr']['subvoxel']['std_mm_population']:.6f} mm"
        ),
        (
            f"TRE(extreme, dense, snap) mean="
            f"{metrics['dense_dvf']['snap']['mean_mm']:.6f} mm "
            f"std={metrics['dense_dvf']['snap']['std_mm_population']:.6f} mm"
        ),
        (
            f"TRE(extreme, dense, subvoxel) mean="
            f"{metrics['dense_dvf']['subvoxel']['mean_mm']:.6f} mm "
            f"std={metrics['dense_dvf']['subvoxel']['std_mm_population']:.6f} mm"
        ),
    ]
    (output_dir / "tre_scores.txt").write_text(
        "\n".join(tre_lines) + "\n", encoding="utf-8"
    )

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        total_peak_allocated = int(torch.cuda.max_memory_allocated(resolved_device))
        total_peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
    else:
        total_peak_allocated = None
        total_peak_reserved = None
    total_wall_seconds = time.perf_counter() - case_started

    final_record = result["history"][-1]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "variant": variant,
        "case": case,
        "seed": seed,
        "run_mode": run_mode,
        "smoke_steps": smoke_steps,
        "fixed_phase": fixed_phase,
        "moving_phase": moving_phase,
        "image_shape_crop": list(images["fixed"].shape),
        "image_shape_original": list(data["original_shape"]),
        "image_spacing_xyz": list(data["image_spacing"]),
        "crop_bbox": data["bbox"],
        "optimizer_resolved": result["optimizer_settings"],
        "stage_start_steps": transition_manifest["stage_start_steps"],
        "stage_transition_events": transition_manifest["events"],
        "stage_policy": transition_manifest["policy"],
        "a0_transition_trace": str(replay_path) if replay_path else None,
        "final_training_record": final_record,
        "landmark_metrics": metrics,
        "five_lobe_dice": lobe_dice,
        "runtime": {
            "training_wall_seconds": training_wall_seconds,
            "total_wall_seconds_through_evaluation": total_wall_seconds,
        },
        "cuda_peak_memory_bytes": {
            "training_allocated": training_peak_allocated,
            "training_reserved": training_peak_reserved,
            "total_allocated": total_peak_allocated,
            "total_reserved": total_peak_reserved,
        },
        "partial_output_policy": partial_output_policy,
        "artifacts": {
            "history": "history.csv",
            "stage_trace": "stage_trace.csv",
            "transition_manifest": "stage_transitions.json",
            "config": "resolved_config.yaml",
            "config_json": "resolved_config.json",
            "provenance": "provenance.json",
            "model": "inr_dvf.pth",
            "model_config": "inr_dvf_config.yaml",
            "dense_dvf_full": "dirinr_dvf.nii.gz",
            "dense_dvf_crop": "dense_dvf_crop_voxel_xyz.npy",
            "tre_scores": "tre_scores.txt",
            "landmark_metrics": "landmark_metrics.json",
            "landmark_details": "landmark_details.npz",
            "five_lobe_dice": "five_lobe_dice.json",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "RUN_STATE.json",
        {
            "status": "complete",
            "variant": variant,
            "case": case,
            "seed": seed,
            "run_mode": run_mode,
            "smoke_steps": smoke_steps,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "training_wall_seconds": training_wall_seconds,
            "total_wall_seconds_through_evaluation": total_wall_seconds,
            "partial_output_policy": partial_output_policy,
        },
    )
    _append_line(
        stdout_path,
        (
            f"COMPLETE variant={variant} case={case:02d} "
            f"dense_snap_tre_mm={metrics['dense_dvf']['snap']['mean_mm']:.6f} "
            f"training_seconds={training_wall_seconds:.3f} "
            f"total_seconds={total_wall_seconds:.3f}"
        ),
    )
    (output_dir / "DONE").touch()
    print(
        f"{variant} case {case:02d} complete | "
        f"dense snap TRE={metrics['dense_dvf']['snap']['mean_mm']:.4f} mm | "
        f"{output_dir}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned DUAL-INR-DIR dual author baseline (A0) or its "
            "mask-loss-only ablation (A1) on converted DIR-Lab 4DCT data."
        )
    )
    parser.add_argument("--variant", type=str.upper, choices=["A0", "A1"], required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", type=validate_case)
    selection.add_argument(
        "--cohort",
        type=parse_cohort,
        help="'all10' or a comma-separated subset such as 1,2,5",
    )
    parser.add_argument("--device", default="0", help="CUDA index/device string or cpu")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Explicit absolute scratch/output root outside the repo and input data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-steps",
        type=validate_smoke_steps,
        default=None,
        help=(
            "Explicit short-smoke override (1..50). Outputs are isolated below "
            "smoke_Nsteps/; omit for the locked 3000-step pilot/production run."
        ),
    )
    parser.add_argument(
        "--replay-a0-transitions",
        type=Path,
        default=None,
        help=(
            "Required for A1: A0 stage_transitions.json for a single case, or a "
            "root containing per-case A0 traces for a cohort"
        ),
    )
    return parser


def _mark_partial_case_failed(
    *,
    output_root: Path,
    variant: str,
    case: int,
    seed: int,
    error: BaseException,
) -> None:
    output_dir = output_root / variant.lower() / f"case_{case:02d}"
    state_path = output_dir / "RUN_STATE.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("status") != "running":
        return
    failure = {
        "status": "failed",
        "variant": variant,
        "case": case,
        "seed": seed,
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error": str(error),
        "partial_outputs_preserved": True,
        "accepted_record": False,
        "resume": "No in-place resume; use a fresh output root.",
    }
    write_json(state_path, failure)
    with (output_dir / "stderr.log").open("a", encoding="utf-8") as stream:
        traceback.print_exception(type(error), error, error.__traceback__, file=stream)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = [args.case] if args.case is not None else args.cohort
    if args.variant == "A1" and args.replay_a0_transitions is None:
        raise SystemExit("A1 requires --replay-a0-transitions")
    if args.variant == "A0" and args.replay_a0_transitions is not None:
        raise SystemExit("A0 is adaptive and cannot use --replay-a0-transitions")

    data_root, output_root = validate_roots(args.data_root, args.output_root)
    execution_root = (
        output_root / f"smoke_{args.smoke_steps}steps"
        if args.smoke_steps is not None
        else output_root
    )
    summaries = []
    for case in cases:
        try:
            summary = run_case(
                variant=args.variant,
                case=case,
                data_root=data_root,
                output_root=execution_root,
                device=args.device,
                seed=args.seed,
                trace_input=args.replay_a0_transitions,
                single_case_selection=args.case is not None,
                smoke_steps=args.smoke_steps,
            )
        except Exception as exc:
            _mark_partial_case_failed(
                output_root=execution_root,
                variant=args.variant,
                case=case,
                seed=args.seed,
                error=exc,
            )
            raise
        summaries.append(summary)

    if args.cohort is not None:
        cohort_summary = {
            "schema_version": 1,
            "variant": args.variant,
            "seed": args.seed,
            "run_mode": "short_smoke" if args.smoke_steps is not None else "full_length",
            "smoke_steps": args.smoke_steps,
            "cases": cases,
            "n_complete": len(summaries),
            "case_summaries": summaries,
        }
        write_json(
            execution_root / args.variant.lower() / "cohort_summary.json",
            cohort_summary,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
