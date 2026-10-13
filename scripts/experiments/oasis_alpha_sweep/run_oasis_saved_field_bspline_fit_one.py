#!/usr/bin/env python
"""Fit one independent cubic B-spline field to a saved OASIS displacement.

The optimization target is the full dense displacement in voxel/mm units.  No
image similarity term, foreground mask, or deformation regularizer enters the
loss.  The fixed-image-positive mask is used only for secondary error metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_one as image_runner  # noqa: E402
from evaluate_saved_baselines_first10 import (  # noqa: E402
    BASELINE_SPECS,
    DEFAULT_DL_ROOT,
    load_saved_disp_zyx,
    voxel_zyx_to_norm_dhw3,
)


VARIANT = "oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1"
METHOD = "saved_dense_field_independent_cubic_bspline_fullvoxel_mse"
TARGET_METHOD_KEYS = {"vfa": "bs3_vfa_v0", "sitreg": "bs4_sitreg_v0"}
TARGET_LABELS = {"vfa": "VFA", "sitreg": "SITReg"}
ALLOWED_CPS = (8, 4, 2, 1)
EXPECTED_CONTROL_SHAPES = {
    8: (24, 32, 28),
    4: (44, 60, 52),
    2: (84, 116, 100),
    1: (164, 228, 196),
}
EXPECTED_RAW_SHAPE = (3, *c.IMAGE_SHAPE)
EXPECTED_TARGET_SHAPE = (*c.IMAGE_SHAPE, 3)
DEFAULT_OUT_ROOT = Path(f"/database/inrorspline/runs/{VARIANT}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGET_METHOD_KEYS), required=True)
    parser.add_argument("--cps", type=int, choices=ALLOWED_CPS, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--min-epochs", type=int, default=500)
    parser.add_argument("--check-interval", type=int, default=50)
    parser.add_argument("--median-window", type=int, default=50)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--relative-mse-threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--dl-root", type=Path, default=DEFAULT_DL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--expected-original-case-index", type=int, default=158)
    parser.add_argument("--expected-fixed-id", default="0333")
    parser.add_argument("--expected-moving-id", default="0115")
    parser.add_argument("--run-name")
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--init-control", type=Path)
    parser.add_argument("--recovery-of")
    parser.add_argument("--initial-updates", type=int, default=0)
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-registration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-target-only", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if int(args.cps) not in ALLOWED_CPS:
        raise ValueError(f"cps must be one of {ALLOWED_CPS}")
    for name in ("max_epochs", "check_interval", "median_window", "patience"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.min_epochs) < 0 or int(args.min_epochs) > int(args.max_epochs):
        raise ValueError("min_epochs must be between zero and max_epochs")
    if float(args.lr) <= 0 or float(args.relative_mse_threshold) < 0:
        raise ValueError("lr must be positive and the relative threshold nonnegative")
    if int(args.patience) < int(args.median_window):
        raise ValueError("patience must be at least median_window")
    if args.init_control is None and (args.recovery_of or int(args.initial_updates)):
        raise ValueError("recovery metadata requires --init-control")


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return str(args.run_name)
    base = (
        f"fieldfit_{args.target}_rank{int(args.case_rank):03d}_cps{int(args.cps)}"
        f"_fullvoxmse_lr{c.format_float(float(args.lr))}_max{int(args.max_epochs)}"
        f"_min{int(args.min_epochs)}_pat{int(args.patience)}_ci{int(args.check_interval)}"
        f"_win{int(args.median_window)}_rt{c.format_float(float(args.relative_mse_threshold))}"
        f"_seed{int(args.seed)}"
    )
    return base + ("_recovery" if args.init_control is not None else "")


def target_spec(target: str):
    method_key = TARGET_METHOD_KEYS[str(target)]
    matches = [spec for spec in BASELINE_SPECS if spec.method_key == method_key]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one saved-baseline spec for {target}, found {len(matches)}")
    return matches[0]


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def target_path(args: argparse.Namespace, fixed_id: str, moving_id: str) -> Path:
    spec = target_spec(str(args.target))
    return Path(args.dl_root) / spec.source_root / f"disp_{fixed_id}_{moving_id}.nii.gz"


def load_target(
    args: argparse.Namespace,
    *,
    fixed_id: str,
    moving_id: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    path = target_path(args, fixed_id, moving_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {args.target} target field: {path}")
    loaded = load_saved_disp_zyx(path)
    if tuple(loaded.disp_zyx.shape) != (1, *EXPECTED_RAW_SHAPE):
        raise ValueError(f"Expected loaded [1,3,D,H,W], got {loaded.disp_zyx.shape}")
    if loaded.source_disp_raw_shape != ",".join(str(v) for v in EXPECTED_RAW_SHAPE):
        raise ValueError(f"Unexpected raw target shape: {loaded.source_disp_raw_shape}")
    if loaded.source_disp_storage_layout != "channel_first_zyx_repo_ras":
        raise ValueError(f"Unexpected saved-field layout: {loaded.source_disp_storage_layout}")
    if loaded.source_disp_component_order != "zyx_voxel":
        raise ValueError(f"Unexpected component order: {loaded.source_disp_component_order}")
    if loaded.source_disp_reorientation != "none":
        raise ValueError(f"Unexpected target reorientation: {loaded.source_disp_reorientation}")
    target_np = voxel_zyx_to_norm_dhw3(loaded.disp_zyx)
    if tuple(target_np.shape) != EXPECTED_TARGET_SHAPE or not np.isfinite(target_np).all():
        raise ValueError(f"Invalid normalized target: shape={target_np.shape}, finite={np.isfinite(target_np).all()}")
    spec = target_spec(str(args.target))
    audit: Dict[str, object] = {
        "target_key": str(args.target),
        "target_method_key": spec.method_key,
        "target_method_label": spec.method_label,
        "target_selector": spec.selector,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_raw_shape": list(EXPECTED_RAW_SHAPE),
        "source_loaded_shape": [1, *EXPECTED_RAW_SHAPE],
        "normalized_target_shape": list(EXPECTED_TARGET_SHAPE),
        "source_storage_layout": loaded.source_disp_storage_layout,
        "source_component_order": loaded.source_disp_component_order,
        "source_reorientation": loaded.source_disp_reorientation,
        "source_units": "zyx voxel displacement; 1 mm isotropic OASIS voxels",
        "training_layout": "DHW3 normalized-grid xyz displacement",
        "training_units": "voxel/mm residual after xyz scaling",
        "direction": "moving_to_fixed",
        "fixed_id": fixed_id,
        "moving_id": moving_id,
    }
    return torch.from_numpy(target_np).to(device=device, dtype=torch.float32), audit


def voxel_residual(diff_norm: torch.Tensor, image_shape: Sequence[int] = c.IMAGE_SHAPE) -> torch.Tensor:
    d, h, w = (int(v) for v in image_shape)
    scale = diff_norm.new_tensor([(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0])
    return diff_norm * scale.view(*([1] * (diff_norm.ndim - 1)), 3)


def full_voxel_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    if pred_norm.shape != target_norm.shape or pred_norm.shape[-1] != 3:
        raise ValueError(f"Field shape mismatch: {tuple(pred_norm.shape)} vs {tuple(target_norm.shape)}")
    return voxel_residual(pred_norm - target_norm, pred_norm.shape[:-1]).square().mean()


def _region_error_metrics(
    diff_norm_flat: torch.Tensor,
    target_norm_flat: torch.Tensor,
    *,
    prefix: str,
    image_shape: Sequence[int],
) -> Dict[str, float]:
    diff_vox = voxel_residual(diff_norm_flat, image_shape)
    target_vox = voxel_residual(target_norm_flat, image_shape)
    mse_norm = diff_norm_flat.square().mean()
    mse_vox = diff_vox.square().mean()
    target_power_norm = target_norm_flat.square().mean()
    target_power_vox = target_vox.square().mean()
    epe = torch.linalg.vector_norm(diff_vox, dim=-1)
    if epe.numel() == 0:
        raise RuntimeError(f"Cannot compute {prefix} metrics on an empty region")
    result = {
        f"{prefix}_normalized_grid_mse": float(mse_norm.detach().cpu()),
        f"{prefix}_normalized_grid_rmse": math.sqrt(float(mse_norm.detach().cpu())),
        f"{prefix}_normalized_grid_target_power": float(target_power_norm.detach().cpu()),
        f"{prefix}_normalized_grid_relative_rmse": math.sqrt(
            float(mse_norm.detach().cpu()) / max(float(target_power_norm.detach().cpu()), 1e-30)
        ),
        f"{prefix}_voxel_mse": float(mse_vox.detach().cpu()),
        f"{prefix}_voxel_rmse": math.sqrt(float(mse_vox.detach().cpu())),
        f"{prefix}_voxel_target_power": float(target_power_vox.detach().cpu()),
        f"{prefix}_voxel_relative_rmse": math.sqrt(
            float(mse_vox.detach().cpu()) / max(float(target_power_vox.detach().cpu()), 1e-30)
        ),
        f"{prefix}_epe_mm_mean": float(epe.mean().detach().cpu()),
        f"{prefix}_epe_mm_median": float(epe.median().detach().cpu()),
        f"{prefix}_epe_mm_p95": float(torch.quantile(epe, 0.95).detach().cpu()),
        f"{prefix}_epe_mm_max": float(epe.max().detach().cpu()),
        f"{prefix}_scalar_count": int(diff_norm_flat.numel()),
        f"{prefix}_voxel_count": int(diff_norm_flat.shape[0]),
    }
    for component, label in enumerate(("x", "y", "z")):
        result[f"{prefix}_voxel_mse_{label}"] = float(diff_vox[:, component].square().mean().detach().cpu())
        result[f"{prefix}_normalized_grid_mse_{label}"] = float(
            diff_norm_flat[:, component].square().mean().detach().cpu()
        )
    return result


def field_error_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    fixed_positive_mask: torch.Tensor,
) -> Dict[str, float]:
    diff_flat = (pred_norm.detach() - target_norm.detach()).reshape(-1, 3)
    target_flat = target_norm.detach().reshape(-1, 3)
    mask_flat = fixed_positive_mask.detach().reshape(-1).bool()
    result = _region_error_metrics(
        diff_flat, target_flat, prefix="full", image_shape=pred_norm.shape[:-1]
    )
    result.update(
        _region_error_metrics(
            diff_flat[mask_flat],
            target_flat[mask_flat],
            prefix="fixed_positive",
            image_shape=pred_norm.shape[:-1],
        )
    )
    result.update(
        {
            "objective_name": "full_volume_voxel_mm_mse",
            "objective_mask": "none",
            "objective_regularizer": "none",
            "objective_voxel_count": int(diff_flat.shape[0]),
            "objective_scalar_count": int(diff_flat.numel()),
            "objective_mse": result["full_voxel_mse"],
        }
    )
    return result


def plateau_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    min_epochs: int,
    check_interval: int,
    median_window: int,
    patience: int,
    relative_threshold: float,
) -> Tuple[bool, str]:
    if not rows:
        return False, ""
    epoch = int(rows[-1]["epoch"])
    if epoch < int(min_epochs) or epoch % int(check_interval) != 0:
        return False, ""
    window = int(median_window)
    if len(rows) < int(patience) + window:
        return False, ""
    recent = np.median([float(row["mse_voxel"]) for row in rows[-window:]])
    previous_stop = len(rows) - int(patience)
    previous_start = previous_stop - window
    previous = np.median(
        [float(row["mse_voxel"]) for row in rows[previous_start:previous_stop]]
    )
    relative = (float(previous) - float(recent)) / max(abs(float(previous)), 1e-30)
    if relative < float(relative_threshold):
        return True, (
            "adaptive_median_plateau:"
            f"epoch={epoch};patience={int(patience)};window={window};"
            f"relative_improvement={relative:.9g};threshold={float(relative_threshold):.9g}"
        )
    return False, ""


def load_initial_control(path: Path, cps: int, expected_shape: Sequence[int]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    control = payload.get("best_control", payload.get("control_disp"))
    if not isinstance(control, torch.Tensor):
        raise ValueError(f"Checkpoint lacks best_control/control_disp: {path}")
    if int(payload.get("cps", cps)) != int(cps):
        raise ValueError(f"Checkpoint cps mismatch in {path}")
    if tuple(control.shape) != (*tuple(int(v) for v in expected_shape), 3):
        raise ValueError(f"Checkpoint control shape {tuple(control.shape)} is not {(*expected_shape, 3)}")
    if not torch.isfinite(control).all():
        raise ValueError(f"Checkpoint contains non-finite controls: {path}")
    return control.detach().float()


def train_fit(
    args: argparse.Namespace,
    target_norm: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    cps = int(args.cps)
    # This benchmark deliberately uses the dyadically nested margin-four grids
    # frozen in the protocol, including the extra outer support relative to the
    # legacy independent-grid helper at cps 4/2/1.
    control_shape = EXPECTED_CONTROL_SHAPES[cps]
    initial = torch.zeros((*control_shape, 3), dtype=torch.float32)
    if args.init_control is not None:
        initial = load_initial_control(Path(args.init_control), cps, control_shape)
    control = torch.nn.Parameter(initial.to(device=device))
    optimizer = torch.optim.Adam([control], lr=float(args.lr))
    best_control = control.detach().clone()
    best_loss = float("inf")
    best_epoch = int(args.initial_updates)
    history: List[Dict[str, object]] = []
    stop_reason = "max_epochs"
    optimizer_steps = 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for local_epoch in range(1, int(args.max_epochs) + 1):
        cumulative_epoch = int(args.initial_updates) + local_epoch
        dense = c.expand_bspline_controls(control, control_shape, c.IMAGE_SHAPE, cps)
        loss = full_voxel_mse(dense, target_norm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite voxel MSE at epoch {cumulative_epoch}")
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_epoch = cumulative_epoch - 1
            best_control.copy_(control.detach())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        row: Dict[str, object] = {
            "epoch": cumulative_epoch,
            "local_epoch": local_epoch,
            "completed_updates_at_evaluation": cumulative_epoch - 1,
            "mse_voxel": value,
            "rmse_voxel": math.sqrt(value),
            "lr": float(args.lr),
        }
        history.append(row)
        should_stop, reason = plateau_decision(
            history,
            min_epochs=int(args.min_epochs),
            check_interval=int(args.check_interval),
            median_window=int(args.median_window),
            patience=int(args.patience),
            relative_threshold=float(args.relative_mse_threshold),
        )
        if local_epoch == 1 or local_epoch % int(args.check_interval) == 0 or should_stop:
            print(json.dumps({**row, "best_mse_voxel": best_loss, "stop": should_stop}), flush=True)
        if should_stop:
            stop_reason = reason
            break

    with torch.no_grad():
        dense_current = c.expand_bspline_controls(control, control_shape, c.IMAGE_SHAPE, cps)
        current_loss = float(full_voxel_mse(dense_current, target_norm).detach().cpu())
        final_cumulative_epoch = int(args.initial_updates) + optimizer_steps
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = final_cumulative_epoch
            best_control.copy_(control.detach())
        dense_best = c.expand_bspline_controls(best_control, control_shape, c.IMAGE_SHAPE, cps).detach()
        selected_loss = float(full_voxel_mse(dense_best, target_norm).detach().cpu())
    torch.cuda.synchronize()
    if abs(selected_loss - best_loss) > max(1e-10, 1e-6 * max(abs(best_loss), 1e-12)):
        raise RuntimeError(f"Best-checkpoint loss mismatch: tracked={best_loss}, evaluated={selected_loss}")
    return {
        "best_control": best_control.detach(),
        "dense_best": dense_best,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_mse_voxel": selected_loss,
        "final_state_mse_voxel": current_loss,
        "stop_reason": stop_reason,
        "optimizer_steps": int(optimizer_steps),
        "cumulative_optimizer_steps": int(args.initial_updates) + int(optimizer_steps),
        "control_shape": control_shape,
        "parameter_count": int(control.numel()),
        "training_elapsed_sec": time.monotonic() - started,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def gpu_snapshot() -> Dict[str, object]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        rows = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6:
                rows.append(
                    {
                        "physical_gpu": parts[0],
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_used_mib": int(parts[3]),
                        "memory_free_mib": int(parts[4]),
                        "utilization_gpu_pct": int(parts[5]),
                    }
                )
        return {"captured_at": c.now_text(), "rows": rows, "error": None}
    except Exception as exc:  # pragma: no cover - host utility failure is recorded, not hidden.
        return {"captured_at": c.now_text(), "rows": [], "error": f"{type(exc).__name__}: {exc}"}


def registration_row(metrics: Mapping[str, object], fixed_id: str, moving_id: str) -> Dict[str, object]:
    return c.accuracy_row(dict(metrics), fixed_id, moving_id)


def prefixed_scalars(prefix: str, row: Mapping[str, object]) -> Dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in row.items() if key not in {"fixed_id", "moving_id"}}


def write_outputs(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_name: str,
    batch: Mapping[str, object],
    target_norm: torch.Tensor,
    target_audit: Mapping[str, object],
    fit: Mapping[str, object],
    started_at: str,
    total_start_monotonic: float,
    initial_gpu_state: Mapping[str, object],
) -> Dict[str, object]:
    dense = fit["dense_best"]
    control = fit["best_control"]
    assert isinstance(dense, torch.Tensor) and isinstance(control, torch.Tensor)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    errors = field_error_metrics(dense, target_norm, fixed_mask)
    if int(errors["objective_voxel_count"]) != int(np.prod(c.IMAGE_SHAPE)):
        raise RuntimeError("The optimization objective did not cover the full image volume")
    if abs(float(errors["objective_mse"]) - float(fit["best_mse_voxel"])) > max(
        1e-10, 1e-6 * max(abs(float(fit["best_mse_voxel"])), 1e-12)
    ):
        raise RuntimeError("Serialized best field does not reproduce the selected objective")

    fit_row: Dict[str, object] = {}
    source_row: Dict[str, object] = {}
    if args.evaluate_registration:
        identity = c.make_coordinate_grid(c.IMAGE_SHAPE, dense.device)
        fit_eval = c.evaluate_from_dense_coords(identity + dense, dict(batch))
        source_eval = c.evaluate_from_dense_coords(identity + target_norm, dict(batch))
        fit_row = registration_row(fit_eval, str(batch["fixed_id"]), str(batch["moving_id"]))
        source_row = registration_row(source_eval, str(batch["fixed_id"]), str(batch["moving_id"]))
        c.write_csv(run_dir / "case_metrics.csv", [fit_row])
        c.write_csv(run_dir / "source_case_metrics.csv", [source_row])

    if args.save_final_dense:
        np.save(
            run_dir / f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy",
            dense.detach().cpu().numpy().astype(np.float32),
        )
    checkpoint = {
        "best_control": control.detach().cpu(),
        "control_disp": control.detach().cpu(),
        "control_shape": list(fit["control_shape"]),
        "cps": int(args.cps),
        "image_shape": list(c.IMAGE_SHAPE),
        "best_epoch": int(fit["best_epoch"]),
        "best_mse_voxel": float(fit["best_mse_voxel"]),
        "optimizer": "Adam",
        "lr": float(args.lr),
        "source_sha256": target_audit["source_sha256"],
        "run_name": run_name,
    }
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")
    torch.save(checkpoint, run_dir / "model_state.pt")
    c.write_csv(run_dir / "history.csv", fit["history"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "field_error_metrics.csv", [errors])
    c.write_json(run_dir / "target_audit.json", dict(target_audit))
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])

    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": VARIANT,
        "method": METHOD,
        "target": str(args.target),
        "target_label": TARGET_LABELS[str(args.target)],
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": str(batch["fixed_id"]),
        "moving_id": str(batch["moving_id"]),
        "direction": "moving_to_fixed",
        "pair_file": str(batch["pair_file"]),
        "cps": int(args.cps),
        "bspline_order": 3,
        "representation": "independent_direct_cubic_bspline_control_displacement",
        "control_shape": list(fit["control_shape"]),
        "parameter_count": int(fit["parameter_count"]),
        "initialization": "saved_best_control" if args.init_control else "exact_zero",
        "independent_fit": args.init_control is None,
        "coarse_to_fine_initialization": False,
        "optimizer": "Adam",
        "lr": float(args.lr),
        "max_epochs": int(args.max_epochs),
        "min_epochs": int(args.min_epochs),
        "check_interval": int(args.check_interval),
        "median_window": int(args.median_window),
        "patience": int(args.patience),
        "relative_mse_threshold": float(args.relative_mse_threshold),
        "best_epoch": int(fit["best_epoch"]),
        "optimizer_steps": int(fit["optimizer_steps"]),
        "cumulative_optimizer_steps": int(fit["cumulative_optimizer_steps"]),
        "stopping_reason": str(fit["stop_reason"]),
        "best_mse_voxel": float(fit["best_mse_voxel"]),
        "final_state_mse_voxel": float(fit["final_state_mse_voxel"]),
        "loss": "mean squared residual over all full-volume voxel/mm xyz components",
        "image_loss": "none",
        "regularizer": "none",
        "training_mask": "none_full_volume",
        "foreground_metric_mask": "fixed_raw_image_positive",
        "segmentation_mask_role": "registration_metrics_only",
        "objective_voxel_count": int(np.prod(c.IMAGE_SHAPE)),
        "objective_scalar_count": int(np.prod(c.IMAGE_SHAPE) * 3),
        "seed": int(args.seed),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "gpu_requested": str(args.gpu),
        "visible_cuda_device": int(torch.cuda.current_device()),
        "cuda_backend": c.cuda_backend_metadata(dense.device),
        "initial_gpu_state": dict(initial_gpu_state),
        "training_elapsed_sec": float(fit["training_elapsed_sec"]),
        "training_peak_mem_gb": float(fit["training_peak_mem_gb"]),
        "total_elapsed_sec": float(time.monotonic() - total_start_monotonic),
        "started_at": started_at,
        "completed_at": c.now_text(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "source_revision": str(args.source_revision),
        "target_audit": dict(target_audit),
        "recovery_of": args.recovery_of,
        "init_control": None if args.init_control is None else str(args.init_control),
        "initial_updates": int(args.initial_updates),
        **errors,
        **prefixed_scalars("fitted", fit_row),
        **prefixed_scalars("source", source_row),
    }
    if fit_row and source_row:
        summary.update(
            {
                "delta_dice_mean": float(fit_row["dice_mean"]) - float(source_row["dice_mean"]),
                "delta_hd95_mean": float(fit_row["hd95_mean"]) - float(source_row["hd95_mean"]),
                "delta_asd_symmetric_mean": float(fit_row["asd_symmetric_mean"])
                - float(source_row["asd_symmetric_mean"]),
            }
        )
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text(c.now_text() + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_name = build_run_name(args)
    run_dir = Path(args.out_root) / str(args.target) / f"cps{int(args.cps)}" / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite and not args.validate_target_only:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite and not args.validate_target_only:
        raise FileExistsError(f"Refusing to mix with incomplete run directory: {run_dir}")

    started_at = c.now_text()
    total_start = time.monotonic()
    initial_gpu_state = gpu_snapshot()
    device = c.set_cuda_device(
        str(args.gpu), cudnn_benchmark=False, cudnn_deterministic=True
    )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    batch = c.load_oasis_case(
        rank=int(args.case_rank),
        device=device,
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
    )
    image_runner.validate_expected_case(
        args,
        original_index=int(batch["original_case_index"]),
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
    )
    target_norm, target_audit = load_target(
        args,
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
        device=device,
    )
    target_audit.update(
        {
            "case_rank": int(batch["case_rank"]),
            "original_case_index": int(batch["original_case_index"]),
            "pair_file": str(batch["pair_file"]),
        }
    )
    if args.validate_target_only:
        print(json.dumps(target_audit, indent=2), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    c.write_json(
        run_dir / "launch_manifest.json",
        {
            "created_at": c.now_text(),
            "args": vars(args),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "source_revision": args.source_revision,
            "initial_gpu_state": initial_gpu_state,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "target": args.target,
                "cps": args.cps,
                "control_shape": EXPECTED_CONTROL_SHAPES[int(args.cps)],
                "lr": args.lr,
                "gpu": args.gpu,
                "objective": "full_volume_voxel_mm_mse",
            }
        ),
        flush=True,
    )
    fit = train_fit(args, target_norm, device)
    summary = write_outputs(
        args,
        run_dir=run_dir,
        run_name=run_name,
        batch=batch,
        target_norm=target_norm,
        target_audit=target_audit,
        fit=fit,
        started_at=started_at,
        total_start_monotonic=total_start,
        initial_gpu_state=initial_gpu_state,
    )
    print(json.dumps({"completed": str(run_dir), "summary": summary}, default=str), flush=True)


if __name__ == "__main__":
    main()
