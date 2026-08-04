"""Fixed-stage DUAL-INR trainer for the controlled DIR-Lab baselines.

The released DUAL model, sampled global NCC, interpolation, and spatial
regularizers are retained.  This adapter deliberately replaces the author's
adaptive blur schedule with a preregistered, MR-inspired image pyramid.  Model
parameters persist across stages while AdamW and warmup-cosine state are reset
at every stage, matching the effective stage-local optimization used by the
MR-D-BSCP reference implementation in this repository.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f

from scripts.experiments.lungct_dual_inr_author.adapter import (
    AUTHOR_ADAMW_WEIGHT_DECAY,
    resolve_device,
    seed_everything,
)
from scripts.experiments.lungct_dual_inr_author.external_api import (
    AUTHOR_EXTERNAL_COMMIT,
    load_external_api,
)
from scripts.experiments.lungct_no_sweep_repro import common as mr_common


VARIANTS = ("N0", "N1", "P0", "P1")
FULL_RESOLUTION_FACTORS = (1, 1, 1, 1)
MULTIRESOLUTION_FACTORS = (8, 4, 2, 1)
LOCKED_STAGE_LENGTHS = (400, 400, 800, 900)
LOCKED_STAGE_STARTS = (0, 400, 800, 1600)
LOCKED_TOTAL_STEPS = 2500


def variant_protocol(variant: str) -> str:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return variant[0]


def variant_factors(variant: str) -> tuple[int, int, int, int]:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return MULTIRESOLUTION_FACTORS if variant.endswith("1") else FULL_RESOLUTION_FACTORS


def stage_starts(stage_lengths: Sequence[int]) -> list[int]:
    lengths = [int(value) for value in stage_lengths]
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError(f"stage lengths must be positive, got {lengths}")
    starts = [0]
    for length in lengths[:-1]:
        starts.append(starts[-1] + length)
    return starts


def _assert_released_preset(config: Mapping[str, Any]) -> None:
    expected = {
        ("model", "is_dual"): True,
        ("model", "layers"): [3, 256, 256, 3],
        ("model", "coarse_omega0"): 10,
        ("model", "fine_omega0"): 30,
        ("loss", "laplace_weight"): 0.01,
        ("loss", "jacobian_weight"): 0.01,
        ("loss", "fine_reg_scale"): 0.1,
        ("data", "dense_points_per_batch"): 30000,
        ("data", "sampler"): "continuous",
        ("scheduler", "warmup_steps"): 20,
    }
    mismatches: list[str] = []
    for (section, key), expected_value in expected.items():
        actual = config.get(section, {}).get(key)
        if actual != expected_value:
            mismatches.append(
                f"{section}.{key}: expected {expected_value!r}, found {actual!r}"
            )
    if mismatches:
        raise RuntimeError(
            "The pinned DUAL preset no longer matches the controlled-baseline audit:\n- "
            + "\n- ".join(mismatches)
        )


def build_controlled_config(
    variant: str,
    *,
    stage_lengths: Sequence[int] = LOCKED_STAGE_LENGTHS,
) -> dict[str, Any]:
    """Resolve all settings used by one controlled variant.

    A nonlocked ``stage_lengths`` value exists solely for isolated smoke tests;
    production callers must use the default and record their run mode.
    """

    variant = str(variant).upper()
    factors = list(variant_factors(variant))
    lengths = [int(value) for value in stage_lengths]
    if len(lengths) != len(factors) or any(value <= 0 for value in lengths):
        raise ValueError(
            f"stage_lengths must contain four positive values, got {lengths}"
        )

    api = load_external_api()
    config = copy.deepcopy(api.configs.DUAL_CONFIG)
    _assert_released_preset(config)

    config["model"].update(
        {
            "coarse_scale": 0.1,
            "fine_scale": 0.05,
            "fine_branch_is_active": False,
        }
    )
    config["training"].update(
        {
            "total_steps": int(sum(lengths)),
            "fine_start_stage": 1,
            "stage_lengths": lengths,
            "stage_starts": stage_starts(lengths),
            "image_factors": factors,
        }
    )
    config["loss"].update(
        {
            "mask_weight": 0.0,
            "laplace_weight": 0.01,
            "jacobian_weight": 0.01,
            "fine_reg_scale": 0.1,
        }
    )
    config["optimizer"] = {
        "type": "AdamW",
        "lr": 1.0e-3,
        "end_lr": 1.0e-5,
        "weight_decay": AUTHOR_ADAMW_WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "reset_each_stage": True,
    }
    config["scheduler"].update(
        {
            "type": "cosine",
            "warmup_steps": 20,
            "reset_each_stage": True,
        }
    )
    config["controlled_adapter"] = {
        "schema_version": 1,
        "variant": variant,
        "protocol": variant_protocol(variant),
        "external_commit": AUTHOR_EXTERNAL_COMMIT,
        "schedule_name": "mr_d_bscp_shaped_image_pyramid",
        "image_factors": factors,
        "stage_lengths": lengths,
        "stage_starts": stage_starts(lengths),
        "pyramid_source": "each_level_independently_from_full_resolution",
        "antialias_sigma_scale": 0.5,
        "antialias_radius_sigma": 3.0,
        "resize_mode": "trilinear_align_corners_true",
        "sampling_domain_resolution": "full_registration_grid_at_every_stage",
        "model_state_policy": "persist_across_stages",
        "optimizer_state_policy": "fresh_adamw_each_stage",
        "loss_scope": "sampled_global_ncc_plus_released_spatial_regularization",
        "semantic_mask_loss_present": False,
    }
    return config


def _validate_arrays(
    fixed_image_xyz: np.ndarray,
    moving_image_xyz: np.ndarray,
    sampling_mask_xyz: np.ndarray,
) -> None:
    arrays = {
        "fixed_image_xyz": np.asarray(fixed_image_xyz),
        "moving_image_xyz": np.asarray(moving_image_xyz),
        "sampling_mask_xyz": np.asarray(sampling_mask_xyz),
    }
    if arrays["fixed_image_xyz"].ndim != 3:
        raise ValueError("fixed_image_xyz must have shape (X,Y,Z)")
    expected = arrays["fixed_image_xyz"].shape
    for name, array in arrays.items():
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} does not match {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if min(expected) < 2:
        raise ValueError(f"all image dimensions must be >=2, got {expected}")
    mask = arrays["sampling_mask_xyz"] > 0
    if not bool(mask.any()):
        raise ValueError("sampling mask is empty")


def _branch_regularization_components(
    *,
    regularizers_module: Any,
    coords: torch.Tensor,
    output: torch.Tensor,
    jacobian_weight: float,
    laplace_weight: float,
    branch_scale: float,
) -> dict[str, torch.Tensor]:
    determinant, laplacian = regularizers_module.compute_jacobian_and_laplacian(
        coords, output, add_identity=True
    )
    jacobian_raw = torch_f.relu(-determinant).mean()
    laplacian_raw = (laplacian**2).mean()
    return {
        "jacobian_raw": jacobian_raw,
        "laplacian_raw": laplacian_raw,
        "jacobian_weighted": branch_scale * jacobian_weight * jacobian_raw,
        "laplacian_weighted": branch_scale * laplace_weight * laplacian_raw,
    }


def _number(value: torch.Tensor | float) -> float:
    return float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)


def _coordinate_fingerprint(coords: torch.Tensor) -> str:
    sample = coords.detach()[: min(512, int(coords.shape[0]))].cpu().numpy()
    return hashlib.sha256(sample.astype(np.float32, copy=False).tobytes()).hexdigest()


def _make_stage_volume(
    full_volume: torch.Tensor,
    factor: int,
    *,
    sigma_scale: float,
    radius_sigma: float,
) -> tuple[torch.Tensor, float, int]:
    return mr_common.make_loss_volume(
        full_volume,
        factor=int(factor),
        sigma_scale=float(sigma_scale),
        radius_sigma=float(radius_sigma),
    )


def train_controlled(
    *,
    fixed_image_xyz: np.ndarray,
    moving_image_xyz: np.ndarray,
    sampling_mask_xyz: np.ndarray,
    config: dict[str, Any],
    device: str | int | torch.device,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed-stage controlled DUAL model.

    The sampling mask is used only to draw full-domain coordinates.  No
    semantic or moving-image mask is accepted by this API, which makes the P
    no-lung-mask optimization contract structurally enforceable.
    """

    _validate_arrays(fixed_image_xyz, moving_image_xyz, sampling_mask_xyz)
    api = load_external_api()
    device_t = device if isinstance(device, torch.device) else resolve_device(device)
    variant = str(config["controlled_adapter"]["variant"]).upper()
    expected_factors = list(variant_factors(variant))
    factors = [int(value) for value in config["training"]["image_factors"]]
    if factors != expected_factors:
        raise ValueError(
            f"variant {variant} requires factors {expected_factors}, got {factors}"
        )
    lengths = [int(value) for value in config["training"]["stage_lengths"]]
    starts = stage_starts(lengths)
    if starts != [int(value) for value in config["training"]["stage_starts"]]:
        raise ValueError("resolved stage starts do not match stage lengths")

    trainer = api.trainer.DirINR(config=config)
    trainer._cached_mask_indices = {}
    model = trainer._build_model(config).to(device_t)
    model.train()

    fixed_full = torch.as_tensor(
        np.asarray(fixed_image_xyz, dtype=np.float32), device=device_t
    )
    moving_full = torch.as_tensor(
        np.asarray(moving_image_xyz, dtype=np.float32), device=device_t
    )
    sampling_mask = torch.as_tensor(
        np.asarray(sampling_mask_xyz) > 0, device=device_t, dtype=torch.float32
    )
    full_shape = tuple(int(value) for value in fixed_full.shape)

    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    end_lr = float(opt_cfg["end_lr"])
    weight_decay = float(opt_cfg["weight_decay"])
    betas = tuple(float(value) for value in opt_cfg.get("betas", (0.9, 0.999)))
    eps = float(opt_cfg.get("eps", 1.0e-8))
    warmup_steps = int(config["scheduler"]["warmup_steps"])
    scheduler_type = str(config["scheduler"].get("type", "cosine"))
    fine_start_stage = int(config["training"].get("fine_start_stage", 1))
    points_per_step = int(config["data"]["dense_points_per_batch"])
    sampler = str(config["data"]["sampler"])
    sigma_scale = float(config["controlled_adapter"]["antialias_sigma_scale"])
    radius_sigma = float(config["controlled_adapter"]["antialias_radius_sigma"])
    loss_cfg = config["loss"]
    if float(loss_cfg.get("mask_weight", 0.0)) != 0.0:
        raise ValueError("controlled variants prohibit semantic mask loss")
    jacobian_weight = float(loss_cfg["jacobian_weight"])
    laplace_weight = float(loss_cfg["laplace_weight"])
    fine_reg_scale = float(loss_cfg["fine_reg_scale"])

    history: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    global_step = 0
    fine_activation_step: int | None = None

    for stage_index, (factor, stage_length, start) in enumerate(
        zip(factors, lengths, starts)
    ):
        if start != global_step:
            raise RuntimeError(f"stage {stage_index} starts at {start}, current {global_step}")
        if trainer.is_dual and stage_index >= fine_start_stage:
            if not model.fine_branch_is_active:
                model.turn_on_fine_branch()
                fine_activation_step = global_step

        fixed_stage, blur_sigma, blur_kernel = _make_stage_volume(
            fixed_full,
            factor,
            sigma_scale=sigma_scale,
            radius_sigma=radius_sigma,
        )
        moving_stage, moving_sigma, moving_kernel = _make_stage_volume(
            moving_full,
            factor,
            sigma_scale=sigma_scale,
            radius_sigma=radius_sigma,
        )
        if fixed_stage.shape != moving_stage.shape:
            raise RuntimeError("fixed/moving pyramid level shape mismatch")
        if blur_sigma != moving_sigma or blur_kernel != moving_kernel:
            raise RuntimeError("fixed/moving antialias settings diverged")
        stage_shape = tuple(int(value) for value in fixed_stage.shape)

        # MR-style reset: parameters persist, optimizer moments do not.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        scheduler = trainer._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            n_steps_stage=int(stage_length),
            end_lr=end_lr,
        )
        stage_record: dict[str, Any] = {
            "stage_index": stage_index,
            "start_step": global_step,
            "end_step_exclusive": global_step + int(stage_length),
            "length": int(stage_length),
            "image_factor": int(factor),
            "full_shape_xyz": list(full_shape),
            "stage_shape_xyz": list(stage_shape),
            "antialias_sigma_vox": float(blur_sigma),
            "antialias_kernel_size": int(blur_kernel),
            "fine_active_at_stage_start": bool(model.fine_branch_is_active),
            "optimizer_reset": True,
            "scheduler_reset": True,
            "coordinate_fingerprint_sha256": None,
        }

        for local_step in range(int(stage_length)):
            optimizer.zero_grad(None)
            coords = trainer._sample_dense_batch(
                sampling_mask=sampling_mask,
                image_shape=full_shape,
                sampler=sampler,
                n_points=points_per_step,
                device=device_t,
                cache_key="full_resolution_sampling_domain",
            ).requires_grad_(True)
            if local_step == 0:
                stage_record["coordinate_fingerprint_sha256"] = _coordinate_fingerprint(
                    coords
                )

            coarse, fine, total = trainer._predict_displacement(model=model, coords=coords)
            fixed_voxels = api.utils.get_interpolation_coords(coords, stage_shape)
            moving_voxels = api.utils.get_interpolation_coords(coords + total, stage_shape)
            fixed_values = api.utils.trilinear_interpolation(
                fixed_stage, *fixed_voxels
            )
            moving_values = api.utils.trilinear_interpolation(
                moving_stage, *moving_voxels
            )
            similarity = api.ncc.lncc(
                moving_values.view(1, -1), fixed_values.view(1, -1)
            )
            zero = similarity.new_tensor(0.0)
            empty = {
                "jacobian_raw": zero,
                "laplacian_raw": zero,
                "jacobian_weighted": zero,
                "laplacian_weighted": zero,
            }
            coarse_reg = empty
            fine_reg = empty
            if jacobian_weight > 0.0 or laplace_weight > 0.0:
                coarse_reg = _branch_regularization_components(
                    regularizers_module=api.regularizers,
                    coords=coords,
                    output=coarse,
                    jacobian_weight=jacobian_weight,
                    laplace_weight=laplace_weight,
                    branch_scale=1.0,
                )
                if fine is not None:
                    fine_reg = _branch_regularization_components(
                        regularizers_module=api.regularizers,
                        coords=coords,
                        output=fine,
                        jacobian_weight=jacobian_weight,
                        laplace_weight=laplace_weight,
                        branch_scale=fine_reg_scale,
                    )
            spatial = sum(
                (
                    coarse_reg["jacobian_weighted"],
                    coarse_reg["laplacian_weighted"],
                    fine_reg["jacobian_weighted"],
                    fine_reg["laplacian_weighted"],
                ),
                start=zero,
            )
            total_loss = similarity + spatial
            finite = {
                "similarity": similarity,
                "spatial": spatial,
                "total": total_loss,
                "coarse_jacobian_raw": coarse_reg["jacobian_raw"],
                "coarse_laplacian_raw": coarse_reg["laplacian_raw"],
                "fine_jacobian_raw": fine_reg["jacobian_raw"],
                "fine_laplacian_raw": fine_reg["laplacian_raw"],
            }
            nonfinite = {
                name: _number(value)
                for name, value in finite.items()
                if not bool(torch.isfinite(value).all().item())
            }
            if nonfinite:
                raise FloatingPointError(
                    f"non-finite loss at step {global_step}, stage {stage_index}: {nonfinite}"
                )

            lr_before_update = float(optimizer.param_groups[0]["lr"])
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr_after_schedule = float(optimizer.param_groups[0]["lr"])

            record = {
                "zero_step": global_step,
                "step": global_step + 1,
                "stage_index": stage_index,
                "stage_local_step": local_step,
                "image_factor": int(factor),
                "stage_shape_x": stage_shape[0],
                "stage_shape_y": stage_shape[1],
                "stage_shape_z": stage_shape[2],
                "antialias_sigma_vox": float(blur_sigma),
                "fine_active": bool(model.fine_branch_is_active),
                "loss_total": _number(total_loss),
                "loss_similarity": _number(similarity),
                "loss_spatial_weighted": _number(spatial),
                "loss_mask_weighted": 0.0,
                "coarse_jacobian_raw": _number(coarse_reg["jacobian_raw"]),
                "coarse_jacobian_weighted": _number(
                    coarse_reg["jacobian_weighted"]
                ),
                "coarse_laplacian_raw": _number(coarse_reg["laplacian_raw"]),
                "coarse_laplacian_weighted": _number(
                    coarse_reg["laplacian_weighted"]
                ),
                "fine_jacobian_raw": _number(fine_reg["jacobian_raw"]),
                "fine_jacobian_weighted": _number(
                    fine_reg["jacobian_weighted"]
                ),
                "fine_laplacian_raw": _number(fine_reg["laplacian_raw"]),
                "fine_laplacian_weighted": _number(
                    fine_reg["laplacian_weighted"]
                ),
                "learning_rate_before_update": lr_before_update,
                "learning_rate_after_schedule": lr_after_schedule,
                "coarse_scale": float(model.coarse_scale.detach().item()),
                "fine_scale": float(model.fine_scale.detach().item()),
            }
            history.append(record)
            if record_callback is not None:
                record_callback(record)
            global_step += 1

        stage_records.append(stage_record)

    if global_step != int(config["training"]["total_steps"]):
        raise RuntimeError(
            f"executed {global_step} steps, expected {config['training']['total_steps']}"
        )
    return {
        "model": model,
        "history": history,
        "stages": stage_records,
        "fine_activation_step": fine_activation_step,
        "optimizer_settings": copy.deepcopy(config["optimizer"]),
        "scheduler_settings": copy.deepcopy(config["scheduler"]),
    }


__all__ = [
    "FULL_RESOLUTION_FACTORS",
    "LOCKED_STAGE_LENGTHS",
    "LOCKED_STAGE_STARTS",
    "LOCKED_TOTAL_STEPS",
    "MULTIRESOLUTION_FACTORS",
    "VARIANTS",
    "build_controlled_config",
    "resolve_device",
    "seed_everything",
    "stage_starts",
    "train_controlled",
    "variant_factors",
    "variant_protocol",
]
