"""Training adapter for the audited DUAL-INR-DIR author implementation.

The adapter keeps the external model, interpolation, NCC, regularizer, blur,
sampling, convergence, and learning-rate implementations.  It owns only the
training loop needed to make implicit defaults explicit, log loss components,
and replay A0 blur-stage transitions in the mask-loss ablation A1.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f

from .external_api import AUTHOR_EXTERNAL_COMMIT, load_external_api


VARIANTS = ("A0", "A1")
TRANSITION_SCHEMA_VERSION = 1
AUTHOR_ADAMW_WEIGHT_DECAY = 0.01


def _assert_audited_dual_preset(config: Mapping[str, Any]) -> None:
    """Fail loudly if the pinned author preset no longer matches the audit."""

    expected = {
        ("model", "is_dual"): True,
        ("model", "layers"): [3, 256, 256, 3],
        ("model", "coarse_omega0"): 10,
        ("model", "fine_omega0"): 30,
        ("training", "total_steps"): 3000,
        ("training", "blur_sigmas"): [4.0, 2.0, 0.0],
        ("training", "fine_start_stage"): 1,
        ("training", "lr"): 0.001,
        ("training", "end_lr"): 0.00001,
        ("scheduler", "warmup_steps"): 20,
        ("loss", "laplace_weight"): 0.01,
        ("loss", "jacobian_weight"): 0.01,
        ("loss", "mask_weight"): 0.1,
        ("loss", "fine_reg_scale"): 0.1,
        ("data", "dense_points_per_batch"): 30000,
        ("data", "sampler"): "continuous",
    }
    mismatches: list[str] = []
    for (section, key), expected_value in expected.items():
        actual_value = config.get(section, {}).get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{section}.{key}: expected {expected_value!r}, found {actual_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            "The external DUAL_CONFIG differs from the audited author preset:\n- "
            + "\n- ".join(mismatches)
        )


def build_author_config(variant: str) -> dict[str, Any]:
    """Return the audited dual preset with all relied-on defaults explicit.

    ``A0`` is the unmodified full author objective. ``A1`` differs only by
    setting the lobe-mask loss weight to zero; mask-based cropping and sampling
    remain unchanged so that the objective is similarity plus regularization.
    """

    variant = variant.upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    api = load_external_api()
    config = copy.deepcopy(api.configs.DUAL_CONFIG)
    _assert_audited_dual_preset(config)

    config["model"]["coarse_scale"] = 0.1
    config["model"]["fine_scale"] = 0.05
    config["model"]["fine_branch_is_active"] = False
    config["training"]["blur_convergence_patience"] = 50
    config["training"]["blur_convergence_threshold"] = 0.005
    config["training"]["blur_convergence_threshold_mode"] = "rel"
    config["scheduler"]["reset_on_blur_change"] = True
    config["optimizer"] = {
        "type": "AdamW",
        "lr": float(config["training"]["lr"]),
        "end_lr": float(config["training"]["end_lr"]),
        "weight_decay": AUTHOR_ADAMW_WEIGHT_DECAY,
    }
    config["loss"]["mask_weight"] = 0.1 if variant == "A0" else 0.0
    config["adapter"] = {
        "variant": variant,
        "external_commit": AUTHOR_EXTERNAL_COMMIT,
        "stage_policy": (
            "author_dynamic_convergence"
            if variant == "A0"
            else "required_replay_a0"
        ),
        "loss_scope": "author_full" if variant == "A0" else "similarity_plus_regularization",
    }
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | int) -> torch.device:
    if isinstance(device, int) or str(device).isdigit():
        resolved = torch.device(f"cuda:{int(device)}")
    else:
        resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {resolved}")
    return resolved


def _float_list(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers") from exc


def validate_transition_manifest(
    manifest: Mapping[str, Any],
    *,
    total_steps: int | None = None,
    blur_sigmas: list[float] | None = None,
    require_a0: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a saved stage-transition manifest."""

    if int(manifest.get("schema_version", -1)) != TRANSITION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported transition manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    source_variant = str(manifest.get("source_variant", "")).upper()
    if require_a0 and source_variant != "A0":
        raise ValueError(
            f"A1 replay requires an A0 transition manifest, got {source_variant!r}"
        )

    saved_total_steps = int(manifest.get("total_steps", 0))
    if saved_total_steps <= 0:
        raise ValueError("transition manifest total_steps must be > 0")
    saved_sigmas = _float_list(manifest.get("blur_sigmas"), name="blur_sigmas")

    starts_raw = manifest.get("stage_start_steps")
    if not isinstance(starts_raw, list) or not starts_raw:
        raise ValueError("stage_start_steps must be a non-empty list")
    try:
        starts = [int(value) for value in starts_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("stage_start_steps must contain only integers") from exc
    if starts[0] != 0:
        raise ValueError("stage_start_steps must begin at zero")
    if starts != sorted(set(starts)):
        raise ValueError("stage_start_steps must be strictly increasing")
    if any(value < 0 or value >= saved_total_steps for value in starts):
        raise ValueError(
            "stage_start_steps must be within [0, total_steps); "
            f"got {starts} for {saved_total_steps} steps"
        )
    if len(starts) > len(saved_sigmas):
        raise ValueError(
            "stage_start_steps cannot contain more stages than blur_sigmas"
        )

    if total_steps is not None and saved_total_steps != int(total_steps):
        raise ValueError(
            f"A0 replay total_steps mismatch: saved {saved_total_steps}, "
            f"current {int(total_steps)}"
        )
    if blur_sigmas is not None:
        expected_sigmas = [float(value) for value in blur_sigmas]
        if saved_sigmas != expected_sigmas:
            raise ValueError(
                "A0 replay blur_sigmas mismatch: "
                f"saved {saved_sigmas}, current {expected_sigmas}"
            )

    normalized = dict(manifest)
    normalized.update(
        {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "source_variant": source_variant,
            "total_steps": saved_total_steps,
            "blur_sigmas": saved_sigmas,
            "stage_start_steps": starts,
        }
    )
    return normalized


def load_transition_manifest(
    path: str | Path,
    *,
    total_steps: int | None = None,
    blur_sigmas: list[float] | None = None,
    require_a0: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    normalized = validate_transition_manifest(
        manifest,
        total_steps=total_steps,
        blur_sigmas=blur_sigmas,
        require_a0=require_a0,
    )
    normalized["manifest_path"] = str(path.resolve())
    return normalized


def _validate_training_arrays(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_masks: np.ndarray,
    moving_masks: np.ndarray,
) -> None:
    if fixed_image.ndim != 3 or moving_image.ndim != 3:
        raise ValueError("fixed_image and moving_image must both have shape (D, H, W)")
    if fixed_image.shape != moving_image.shape:
        raise ValueError(
            f"image shapes differ: {fixed_image.shape} versus {moving_image.shape}"
        )
    if min(fixed_image.shape) < 2:
        raise ValueError(f"all image dimensions must be >= 2, got {fixed_image.shape}")
    expected_mask_shape = (5,) + fixed_image.shape
    if fixed_masks.shape != expected_mask_shape or moving_masks.shape != expected_mask_shape:
        raise ValueError(
            "author lobe masks must have shape (5, D, H, W); expected "
            f"{expected_mask_shape}, got {fixed_masks.shape} and {moving_masks.shape}"
        )


def _branch_regularization_components(
    *,
    regularizers_module: Any,
    coords: torch.Tensor,
    output: torch.Tensor,
    jacobian_weight: float,
    laplace_weight: float,
    branch_scale: float,
) -> dict[str, torch.Tensor]:
    det_j, laplacian = regularizers_module.compute_jacobian_and_laplacian(
        coords, output, add_identity=True
    )
    jacobian_raw = torch_f.relu(-det_j).mean()
    laplacian_raw = (laplacian**2).mean()
    jacobian_weighted = branch_scale * jacobian_weight * jacobian_raw
    laplacian_weighted = branch_scale * laplace_weight * laplacian_raw
    return {
        "jacobian_raw": jacobian_raw,
        "laplacian_raw": laplacian_raw,
        "jacobian_weighted": jacobian_weighted,
        "laplacian_weighted": laplacian_weighted,
    }


def _detached_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def train_author(
    *,
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_lung_classes_mask: np.ndarray,
    moving_lung_classes_mask: np.ndarray,
    config: dict[str, Any],
    device: str | int | torch.device,
    source_variant: str,
    replay_transition_manifest: Mapping[str, Any] | None = None,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the author computation with component logging and optional A0 replay."""

    source_variant = source_variant.upper()
    if source_variant not in VARIANTS:
        raise ValueError(f"source_variant must be one of {VARIANTS}")
    _validate_training_arrays(
        fixed_image,
        moving_image,
        fixed_lung_classes_mask,
        moving_lung_classes_mask,
    )
    api = load_external_api()
    device_t = device if isinstance(device, torch.device) else resolve_device(device)

    trainer = api.trainer.DirINR(config=config)
    trainer._cached_mask_indices = {}
    model = trainer._build_model(config).to(device_t)
    model.train()

    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]
    lr = float(optimizer_cfg["lr"])
    end_lr = float(optimizer_cfg["end_lr"])
    weight_decay = float(optimizer_cfg["weight_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    fixed_image_t = torch.as_tensor(fixed_image, device=device_t, dtype=torch.float32)
    moving_image_t = torch.as_tensor(moving_image, device=device_t, dtype=torch.float32)
    fixed_masks_t = torch.as_tensor(
        fixed_lung_classes_mask.astype(np.float32, copy=False), device=device_t
    )
    moving_masks_t = torch.as_tensor(
        moving_lung_classes_mask.astype(np.float32, copy=False), device=device_t
    )
    sampling_mask_t = torch.any(fixed_masks_t > 0.5, dim=0).to(torch.float32)

    blur_sigmas = trainer._resolve_blur_sigmas(config)
    total_steps = int(training_cfg["total_steps"])
    if total_steps <= 0:
        raise ValueError("training.total_steps must be > 0")
    fine_start_stage = min(
        max(int(training_cfg.get("fine_start_stage", 1)), 0),
        len(blur_sigmas) - 1,
    )
    scheduler_type = str(config["scheduler"].get("type", "cosine")).lower()
    warmup_steps = int(config["scheduler"]["warmup_steps"])
    reset_scheduler = bool(config["scheduler"].get("reset_on_blur_change", True))

    replay: dict[str, Any] | None = None
    if source_variant == "A1" and replay_transition_manifest is None:
        raise ValueError("A1 requires a realized per-case A0 transition manifest")
    if replay_transition_manifest is not None:
        if source_variant != "A1":
            raise ValueError("A0 must use author adaptive transitions, not replay")
        replay = validate_transition_manifest(
            replay_transition_manifest,
            total_steps=total_steps,
            blur_sigmas=blur_sigmas,
            require_a0=True,
        )
        replay_starts = list(replay["stage_start_steps"])
    else:
        replay_starts = [0]

    convergence_checker = api.scheduler.ConvergenceChecker(
        patience=int(training_cfg.get("blur_convergence_patience", 50)),
        threshold=float(training_cfg.get("blur_convergence_threshold", 0.005)),
        threshold_mode=str(
            training_cfg.get("blur_convergence_threshold_mode", "rel")
        ),
    )

    stage_index = 0
    sigma = float(blur_sigmas[stage_index])
    fixed_stage, moving_stage = trainer._get_blurred_images(
        fixed_image=fixed_image_t,
        moving_image=moving_image_t,
        sigma=sigma,
    )
    scheduler = trainer._build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        n_steps_stage=total_steps,
        end_lr=end_lr,
    )

    stage_start_steps = [0]
    transition_events: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    replay_stage_for_step = {
        int(start): index for index, start in enumerate(replay_starts)
    }

    loss_cfg = config["loss"]
    jacobian_weight = float(loss_cfg["jacobian_weight"])
    laplace_weight = float(loss_cfg["laplace_weight"])
    fine_reg_scale = float(loss_cfg["fine_reg_scale"])
    mask_weight = float(loss_cfg["mask_weight"])

    for zero_step in range(total_steps):
        if replay is not None and zero_step in replay_stage_for_step:
            requested_stage = replay_stage_for_step[zero_step]
            if requested_stage != stage_index:
                previous_stage = stage_index
                stage_index = requested_stage
                sigma = float(blur_sigmas[stage_index])
                fixed_stage, moving_stage = trainer._get_blurred_images(
                    fixed_image=fixed_image_t,
                    moving_image=moving_image_t,
                    sigma=sigma,
                )
                if reset_scheduler:
                    scheduler = trainer._build_scheduler(
                        optimizer=optimizer,
                        scheduler_type=scheduler_type,
                        warmup_steps=warmup_steps,
                        n_steps_stage=total_steps - zero_step,
                        end_lr=end_lr,
                    )
                stage_start_steps.append(zero_step)
                transition_events.append(
                    {
                        "trigger": "replay_a0",
                        "from_stage": previous_stage,
                        "to_stage": stage_index,
                        "after_zero_step": zero_step - 1,
                        "next_stage_start_step": zero_step,
                    }
                )

        if trainer.is_dual and stage_index >= fine_start_stage:
            if not model.fine_branch_is_active:
                model.turn_on_fine_branch()

        optimizer.zero_grad(None)
        dense_coords = trainer._sample_dense_batch(
            sampling_mask=sampling_mask_t,
            image_shape=fixed_image_t.shape,
            sampler=config["data"]["sampler"],
            n_points=int(config["data"]["dense_points_per_batch"]),
            device=device_t,
            cache_key="joint",
        ).requires_grad_(True)

        coarse, fine, total = trainer._predict_displacement(
            model=model,
            coords=dense_coords,
        )
        input_voxels = api.utils.get_interpolation_coords(
            dense_coords, fixed_image_t.shape
        )
        new_voxels = api.utils.get_interpolation_coords(
            dense_coords + total, fixed_image_t.shape
        )
        fixed_values = api.utils.trilinear_interpolation(fixed_stage, *input_voxels)
        moving_warped = api.utils.trilinear_interpolation(moving_stage, *new_voxels)
        fixed_mask_values = api.utils.trilinear_interpolation(
            fixed_masks_t, *input_voxels
        )
        moving_mask_warped = api.utils.trilinear_interpolation(
            moving_masks_t, *new_voxels
        )

        similarity = api.ncc.lncc(
            moving_warped.view(1, -1), fixed_values.view(1, -1)
        )
        zero = similarity.new_tensor(0.0)
        coarse_components = {
            "jacobian_raw": zero,
            "laplacian_raw": zero,
            "jacobian_weighted": zero,
            "laplacian_weighted": zero,
        }
        fine_components = dict(coarse_components)
        if jacobian_weight > 0 or laplace_weight > 0:
            coarse_components = _branch_regularization_components(
                regularizers_module=api.regularizers,
                coords=dense_coords,
                output=coarse,
                jacobian_weight=jacobian_weight,
                laplace_weight=laplace_weight,
                branch_scale=1.0,
            )
            if fine is not None:
                fine_components = _branch_regularization_components(
                    regularizers_module=api.regularizers,
                    coords=dense_coords,
                    output=fine,
                    jacobian_weight=jacobian_weight,
                    laplace_weight=laplace_weight,
                    branch_scale=fine_reg_scale,
                )

        spatial_weighted = sum(
            (
                coarse_components["jacobian_weighted"],
                coarse_components["laplacian_weighted"],
                fine_components["jacobian_weighted"],
                fine_components["laplacian_weighted"],
            ),
            start=zero,
        )
        mask_per_channel = torch_f.mse_loss(
            moving_mask_warped,
            fixed_mask_values,
            reduction="none",
        ).mean(dim=1)
        mask_raw = mask_per_channel.sum()
        mask_weighted = mask_weight * mask_raw
        total_loss = similarity + spatial_weighted + mask_weighted

        finite_components = {
            "similarity": similarity,
            "spatial_weighted": spatial_weighted,
            "mask_raw": mask_raw,
            "mask_weighted": mask_weighted,
            "coarse_jacobian_raw": coarse_components["jacobian_raw"],
            "coarse_jacobian_weighted": coarse_components["jacobian_weighted"],
            "coarse_laplacian_raw": coarse_components["laplacian_raw"],
            "coarse_laplacian_weighted": coarse_components["laplacian_weighted"],
            "fine_jacobian_raw": fine_components["jacobian_raw"],
            "fine_jacobian_weighted": fine_components["jacobian_weighted"],
            "fine_laplacian_raw": fine_components["laplacian_raw"],
            "fine_laplacian_weighted": fine_components["laplacian_weighted"],
            "total": total_loss,
        }
        nonfinite = {
            name: _detached_float(value)
            for name, value in finite_components.items()
            if not bool(torch.isfinite(value).all().item())
        }
        if nonfinite:
            raise FloatingPointError(
                f"Non-finite loss component(s) at zero_step={zero_step}, "
                f"stage={stage_index}: {nonfinite}"
            )

        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            if scheduler._step_count >= scheduler.t_total:
                scheduler = None

        record = {
            "zero_step": zero_step,
            "step": zero_step + 1,
            "stage_index": stage_index,
            "blur_sigma": sigma,
            "fine_active": bool(model.fine_branch_is_active),
            "loss_total": _detached_float(total_loss),
            "loss_similarity": _detached_float(similarity),
            "loss_spatial_weighted": _detached_float(spatial_weighted),
            "loss_mask_raw": _detached_float(mask_raw),
            "loss_mask_weighted": _detached_float(mask_weighted),
            "coarse_jacobian_raw": _detached_float(
                coarse_components["jacobian_raw"]
            ),
            "coarse_jacobian_weighted": _detached_float(
                coarse_components["jacobian_weighted"]
            ),
            "coarse_laplacian_raw": _detached_float(
                coarse_components["laplacian_raw"]
            ),
            "coarse_laplacian_weighted": _detached_float(
                coarse_components["laplacian_weighted"]
            ),
            "fine_jacobian_raw": _detached_float(fine_components["jacobian_raw"]),
            "fine_jacobian_weighted": _detached_float(
                fine_components["jacobian_weighted"]
            ),
            "fine_laplacian_raw": _detached_float(
                fine_components["laplacian_raw"]
            ),
            "fine_laplacian_weighted": _detached_float(
                fine_components["laplacian_weighted"]
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "coarse_scale": float(model.coarse_scale.detach().item()),
            "fine_scale": float(model.fine_scale.detach().item()),
        }
        history.append(record)
        if record_callback is not None:
            record_callback(record)

        if (
            replay is None
            and sigma > 0
            and stage_index < len(blur_sigmas) - 1
            and convergence_checker.update(record["loss_total"])
        ):
            convergence_checker.reset()
            previous_stage = stage_index
            stage_index += 1
            sigma = float(blur_sigmas[stage_index])
            fixed_stage, moving_stage = trainer._get_blurred_images(
                fixed_image=fixed_image_t,
                moving_image=moving_image_t,
                sigma=sigma,
            )
            next_start = zero_step + 1
            transition_events.append(
                {
                    "trigger": "author_dynamic_convergence",
                    "from_stage": previous_stage,
                    "to_stage": stage_index,
                    "after_zero_step": zero_step,
                    "next_stage_start_step": next_start,
                }
            )
            if next_start < total_steps:
                stage_start_steps.append(next_start)
            if reset_scheduler:
                remaining_steps = total_steps - next_start
                if remaining_steps <= 0:
                    scheduler = None
                else:
                    scheduler = trainer._build_scheduler(
                        optimizer=optimizer,
                        scheduler_type=scheduler_type,
                        warmup_steps=warmup_steps,
                        n_steps_stage=remaining_steps,
                        end_lr=end_lr,
                    )

    transition_manifest = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "source_variant": source_variant,
        "total_steps": total_steps,
        "blur_sigmas": [float(value) for value in blur_sigmas],
        "stage_start_steps": stage_start_steps,
        "events": transition_events,
        "policy": "replay_a0" if replay is not None else "author_dynamic_convergence",
        "replayed_manifest_path": replay.get("manifest_path") if replay else None,
    }
    validate_transition_manifest(transition_manifest)
    return {
        "model": model,
        "history": history,
        "transition_manifest": transition_manifest,
        "optimizer": optimizer,
        "optimizer_settings": {
            "type": "AdamW",
            "lr": lr,
            "end_lr": end_lr,
            "weight_decay": weight_decay,
        },
    }
