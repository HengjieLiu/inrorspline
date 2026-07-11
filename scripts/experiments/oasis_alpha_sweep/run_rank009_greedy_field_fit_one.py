#!/usr/bin/env python
"""Fit one OASIS model directly to the canonical Greedy dense displacement."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_one as image_runner  # noqa: E402
from evaluate_saved_baselines_first10 import (  # noqa: E402
    DEFAULT_GREEDY_ROOT,
    load_saved_disp_zyx,
    voxel_zyx_to_norm_dhw3,
)


METHODS = [
    "idir_original",
    "sinr_cps2_be_bspline_analytic",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "direct_cp_8_4_2_be_bspline_analytic_multistage",
]

VARIANT = "oasis_rank009_fit_modes_lr_sweep_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Recorded operating-point alpha; not used in MSE training.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--case-rank", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--field-chunk-size", type=int, default=200000)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--stage-full-cps", type=str, default="8,4,2")
    parser.add_argument("--stage-downsample", type=str, default="4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="400,800,1300")
    parser.add_argument("--stage-lr", type=str, default="")
    parser.add_argument("--finest-mse-diagnostic", action="store_true")
    parser.add_argument("--finest-mse-every", type=int, default=10)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--greedy-root", type=Path, default=DEFAULT_GREEDY_ROOT)
    parser.add_argument("--expected-original-case-index", type=int, default=88)
    parser.add_argument("--expected-fixed-id", type=str, default="0173")
    parser.add_argument("--expected-moving-id", type=str, default="0136")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-target-only", action="store_true")
    return parser.parse_args()


def field_method_key(method: str) -> str:
    if method == "idir_original":
        return "idir_original_dense_siren_field_mse"
    if method == "sinr_cps2_be_bspline_analytic":
        return "sinr_cps2_field_mse"
    if method == "sinr_cps4_be_bspline_analytic":
        return "sinr_cps4_field_mse"
    if method == "direct_cp_cps2_be_bspline_analytic":
        return "direct_cp_cps2_field_mse"
    if method == "direct_cp_cps4_be_bspline_analytic":
        return "direct_cp_cps4_field_mse"
    if method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return "direct_cp_8_4_2_field_mse"
    raise ValueError(f"Unsupported field-fit method: {method}")


def field_method_label(method: str) -> str:
    if method == "idir_original":
        return "IDIR field MSE"
    if method == "sinr_cps2_be_bspline_analytic":
        return "SINR cps2 field MSE"
    if method == "sinr_cps4_be_bspline_analytic":
        return "SINR cps4 field MSE"
    if method == "direct_cp_cps2_be_bspline_analytic":
        return "DirectCP cps2 field MSE"
    if method == "direct_cp_cps4_be_bspline_analytic":
        return "DirectCP cps4 field MSE"
    if method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return "DirectCP 8-4-2 field MSE"
    raise ValueError(f"Unsupported field-fit method: {method}")


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return (
            f"{field_method_key(args.method)}_rank{args.case_rank:03d}"
            f"_sched{args.stage_full_cps.replace(',', '-')}"
            f"_ds{args.stage_downsample.replace(',', '-')}"
            f"_e{args.stage_epochs.replace(',', '-')}"
            f"_a{c.format_float(args.alpha)}_lr{c.format_float(args.lr)}"
        )
    return (
        f"{field_method_key(args.method)}_rank{args.case_rank:03d}"
        f"_a{c.format_float(args.alpha)}_e{args.epochs}_lr{c.format_float(args.lr)}"
    )


def validate_expected_case(args: argparse.Namespace, *, original_index: int, fixed_id: str, moving_id: str) -> None:
    image_runner.validate_expected_case(
        args,
        original_index=original_index,
        fixed_id=fixed_id,
        moving_id=moving_id,
    )


def greedy_disp_path(args: argparse.Namespace, fixed_id: str, moving_id: str) -> Path:
    return Path(args.greedy_root) / f"disp_{fixed_id}_{moving_id}.nii.gz"


def shape_text(shape: Tuple[int, ...]) -> str:
    return ",".join(str(int(v)) for v in shape)


def load_greedy_target(
    args: argparse.Namespace,
    *,
    fixed_id: str,
    moving_id: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    path = greedy_disp_path(args, fixed_id, moving_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical Greedy displacement: {path}")
    loaded = load_saved_disp_zyx(path)
    target_np = voxel_zyx_to_norm_dhw3(loaded.disp_zyx)
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    audit: Dict[str, object] = {
        "source_kind": "greedy",
        "source_family": "greedy",
        "source_selector": "canonical",
        "source_disp_path": str(path),
        "source_disp_component_order": loaded.source_disp_component_order,
        "source_disp_raw_shape": loaded.source_disp_raw_shape,
        "source_disp_shape": loaded.source_disp_eval_shape,
        "source_disp_storage_layout": loaded.source_disp_storage_layout,
        "source_disp_reorientation": loaded.source_disp_reorientation,
        "source_displacement_units": "voxel",
        "field_fit_target_units": "normalized_grid",
        "field_fit_target_shape": shape_text(tuple(target.shape)),
        "field_fit_target_extent": "full_dense_field_available",
    }
    return target, audit


def field_domain_audit(fixed_mask: torch.Tensor) -> Dict[str, object]:
    mask = fixed_mask.detach().bool()
    voxels = int(mask.sum().detach().cpu())
    return {
        "field_fit_domain": "fixed_image_positive",
        "field_fit_mask_definition": "fixed raw image > 0 after standard OASIS reorientation",
        "field_fit_domain_voxels": voxels,
        "field_fit_domain_fraction": float(voxels / np.prod(c.IMAGE_SHAPE)),
    }


def field_mse_metrics(
    pred_disp: torch.Tensor,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> Dict[str, float]:
    diff = pred_disp.detach() - target_disp.detach()
    mse_norm = float(diff.square().mean().detach().cpu())
    rmse_norm = math.sqrt(mse_norm)
    diff_vox = c.norm_grid_dhw3_to_voxel_dhw3(diff, c.IMAGE_SHAPE)
    mse_vox = float(diff_vox.square().mean().detach().cpu())
    rmse_vox = math.sqrt(mse_vox)
    mask = fixed_mask.detach().reshape(-1).bool()
    diff_masked = diff.reshape(-1, 3)[mask]
    diff_vox_masked = diff_vox.reshape(-1, 3)[mask]
    mse_masked_norm = float(diff_masked.square().mean().detach().cpu())
    rmse_masked_norm = math.sqrt(mse_masked_norm)
    mse_masked_vox = float(diff_vox_masked.square().mean().detach().cpu())
    rmse_masked_vox = math.sqrt(mse_masked_vox)
    return {
        "field_mse_full_normalized_grid": mse_norm,
        "field_rmse_full_normalized_grid": rmse_norm,
        "field_mse_full_voxel": mse_vox,
        "field_rmse_full_voxel": rmse_vox,
        "field_mse_masked_normalized_grid": mse_masked_norm,
        "field_rmse_masked_normalized_grid": rmse_masked_norm,
        "field_mse_masked_voxel": mse_masked_vox,
        "field_rmse_masked_voxel": rmse_masked_vox,
        "field_mse_objective_normalized_grid": mse_masked_norm,
        "field_rmse_objective_normalized_grid": rmse_masked_norm,
        "field_mse_objective_voxel": mse_masked_vox,
        "field_rmse_objective_voxel": rmse_masked_vox,
    }


def evaluate_idir_dense_disp(
    model: torch.nn.Module,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(model(coords))
    return torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3).contiguous()


def train_idir_field(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device).detach()
    mask_flat = fixed_mask.detach().reshape(-1).bool()
    coords_train = coords_all[mask_flat]
    target_train = target_disp.reshape(-1, 3).detach()[mask_flat]
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach()
        target = target_train[indices]
        pred = model(coords)
        loss = F.mse_loss(pred, target, reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        entry = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_sampled_normalized_grid": value,
            "coords_batch_size": float(batch_size),
            "train_coords": float(n_train),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    dense_disp = evaluate_idir_dense_disp(model, device, int(args.eval_chunk_size))
    return {
        "history": history,
        "model": model,
        "dense_disp": dense_disp,
        "training_elapsed_sec": time.time() - train_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "param_count": c.count_trainable_params(model),
        "coords_batch_size": batch_size,
        "train_coords": n_train,
        "field_chunk_size": None,
        "field_chunks": None,
    }


def train_spline_field(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    cps = image_runner.method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, cps, device)
    generator = image_runner.build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(args.lr))
    mask_flat = fixed_mask.detach().reshape(-1).bool()
    target_flat = target_disp.reshape(-1, 3).detach()
    n_train = int(mask_flat.sum().detach().cpu())
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        output = generator(control_coords, control_shape)
        dense_disp = c.expand_bspline_controls(output, control_shape, c.IMAGE_SHAPE, cps)
        loss = F.mse_loss(dense_disp.reshape(-1, 3)[mask_flat], target_flat[mask_flat], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        entry = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_masked_normalized_grid": value,
            "train_coords": float(n_train),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()

    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(final_output, control_shape, c.IMAGE_SHAPE, cps).detach()
    return {
        "history": history,
        "generator": generator,
        "control_shape": control_shape,
        "control_coords": control_coords.detach(),
        "control_disp": final_control_disp,
        "dense_disp": final_dense_disp,
        "training_elapsed_sec": time.time() - train_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "param_count": c.count_trainable_params(generator),
        "cps": cps,
        "coords_batch_size": None,
        "train_coords": n_train,
    }


def downsample_target_and_mask(
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    eval_shape: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(int(v) for v in eval_shape)
    if shape == tuple(int(v) for v in target_disp.shape[:3]):
        return target_disp.detach(), fixed_mask.detach().bool()
    target_ncdhw = target_disp.detach().permute(3, 0, 1, 2).unsqueeze(0)
    target_l = F.interpolate(target_ncdhw, size=shape, mode="trilinear", align_corners=True)
    target_l = target_l.squeeze(0).permute(1, 2, 3, 0).contiguous()
    mask_l = F.interpolate(fixed_mask.detach().float()[None, None], size=shape, mode="nearest")
    return target_l, mask_l.squeeze(0).squeeze(0).bool()


def should_finest_mse_diagnostic(args: argparse.Namespace, global_epoch: int, total_epochs: int) -> bool:
    if not bool(args.finest_mse_diagnostic):
        return False
    interval = int(args.finest_mse_every)
    if interval <= 0:
        return False
    return int(global_epoch) == 1 or int(global_epoch) % interval == 0 or int(global_epoch) == int(total_epochs)


def finest_field_mse_value(
    control: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> float:
    with torch.no_grad():
        dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
        mask_flat = fixed_mask.detach().reshape(-1).bool()
        target_flat = target_disp.detach().reshape(-1, 3)
        value = F.mse_loss(
            dense_full.reshape(-1, 3)[mask_flat],
            target_flat[mask_flat],
            reduction="mean",
        )
    return float(value.detach().cpu())


def optimize_multistage_field_stage(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    control: torch.nn.Parameter,
    control_shape: Sequence[int],
    full_cps: int,
    downsample: int,
    epochs: int,
    lr: float,
    stage_idx: int,
    global_epoch_offset: int,
    total_epochs: int,
) -> Tuple[torch.Tensor, Dict[str, object], List[Dict[str, object]]]:
    eval_shape = c.downsample_shape(c.IMAGE_SHAPE, int(downsample)) if int(downsample) > 1 else c.IMAGE_SHAPE
    domain_cps = c.domain_cps_from_full(int(full_cps), int(downsample))
    target_l, mask_l = downsample_target_and_mask(target_disp, fixed_mask, eval_shape)
    mask_flat = mask_l.reshape(-1).bool()
    target_flat = target_l.reshape(-1, 3).detach()
    n_train = int(mask_flat.sum().detach().cpu())
    optimizer = torch.optim.Adam([control], lr=float(lr))
    rows: List[Dict[str, object]] = []
    stage_start = time.time()
    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "domain_cps": int(domain_cps),
                "control_shape_zyx": list(int(v) for v in control_shape),
                "eval_shape_zyx": list(int(v) for v in eval_shape),
                "epochs": int(epochs),
                "lr": float(lr),
                "objective": "stage_domain_target_field_mse",
                "finest_mse_diagnostic": bool(args.finest_mse_diagnostic),
                "finest_mse_every": int(args.finest_mse_every),
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        dense_disp = c.expand_on_shape(control, control_shape, int(full_cps), eval_shape, int(downsample))
        loss = F.mse_loss(dense_disp.reshape(-1, 3)[mask_flat], target_flat[mask_flat], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        global_epoch = int(global_epoch_offset) + int(epoch)
        row: Dict[str, object] = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "global_epoch": float(global_epoch),
            "full_cps": float(full_cps),
            "downsample": float(downsample),
            "domain_cps": float(domain_cps),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "alpha": float(args.alpha),
            "lr": float(lr),
            "field_mse_stage_objective_normalized_grid": value,
            "field_mse_objective_normalized_grid": float("nan"),
            "finest_field_mse_objective_normalized_grid": float("nan"),
            "finest_mse_affects_training": False,
            "train_coords": float(n_train),
        }
        if should_finest_mse_diagnostic(args, global_epoch, int(total_epochs)):
            finest_mse = finest_field_mse_value(control, control_shape, int(full_cps), target_disp, fixed_mask)
            row["field_mse_objective_normalized_grid"] = finest_mse
            row["finest_field_mse_objective_normalized_grid"] = finest_mse
            print(json.dumps({"finest_field_mse_diagnostic": row}, allow_nan=True), flush=True)
        rows.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
            print(json.dumps(row, allow_nan=True), flush=True)
    with torch.no_grad():
        dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
    ctrl_abs = control.detach().abs().reshape(-1)
    summary = {
        "stage_idx": int(stage_idx),
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(int(v) for v in eval_shape),
        "n_control": int(np.prod(tuple(int(v) for v in control_shape))),
        "epochs": int(epochs),
        "lr": float(lr),
        "alpha_recorded_not_used": float(args.alpha),
        "loss_mode": "target_field_mse",
        "elapsed_sec": float(time.time() - stage_start),
        "control_abs_max_norm": float(ctrl_abs.max().detach().cpu()),
        "control_abs_p95_norm": float(torch.quantile(ctrl_abs, 0.95).detach().cpu()),
    }
    return dense_full.detach(), summary, rows


def train_multistage_field(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    if int(sum(stage_epochs)) != int(args.epochs):
        raise ValueError(f"stage_epochs sum {sum(stage_epochs)} != --epochs {args.epochs}")
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(args.lr))
    base_spacing = int(stage_cps[0])
    nested_shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, base_spacing=base_spacing, spacings=stage_cps)
    c.assert_nested_schedule(nested_shapes, stage_cps)

    stage_summaries: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    prolong_rows: List[Dict[str, object]] = []
    prev_control: Optional[torch.Tensor] = None
    prev_shape: Optional[Tuple[int, int, int]] = None
    prev_dense_full: Optional[torch.Tensor] = None
    final_dense: Optional[torch.Tensor] = None
    final_control: Optional[torch.Tensor] = None
    final_control_shape: Optional[Tuple[int, int, int]] = None
    final_cps: Optional[int] = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for stage_idx, (full_cps, downsample, epochs, stage_lr_value) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        global_epoch_offset = int(sum(stage_epochs[:stage_idx]))
        control_shape = tuple(int(v) for v in nested_shapes[int(full_cps)])
        control = torch.nn.Parameter(torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32))
        if prev_control is not None and prev_shape is not None and prev_dense_full is not None:
            init_flat, rows = image_runner.prolongation_diagnostics(
                prev_control,
                prev_shape,
                prev_dense_full,
                control_shape,
                int(full_cps),
                c.IMAGE_SHAPE,
                int(stage_idx),
            )
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        dense_full, summary, rows = optimize_multistage_field_stage(
            args,
            target_disp,
            fixed_mask,
            control,
            control_shape,
            int(full_cps),
            int(downsample),
            int(epochs),
            float(stage_lr_value),
            int(stage_idx),
            global_epoch_offset,
            int(sum(stage_epochs)),
        )
        prev_control = control.detach()
        prev_shape = control_shape
        prev_dense_full = dense_full
        final_dense = dense_full
        final_control = control.detach()
        final_control_shape = control_shape
        final_cps = int(full_cps)
        stage_summaries.append(summary)
        history.extend(rows)
    torch.cuda.synchronize()
    if final_dense is None or final_control is None or final_control_shape is None or final_cps is None:
        raise RuntimeError("Multistage field fitting produced no final control field.")
    final_control_grid = c.flat_to_grid(final_control, final_control_shape).detach()
    return {
        "history": [
            {
                **row,
                "epoch": float(row["global_epoch"]),
                "stage_epoch": float(row["epoch"]),
            }
            for row in history
        ],
        "stage_history": history,
        "stage_summaries": stage_summaries,
        "prolongation_diagnostics": prolong_rows,
        "control_shape": final_control_shape,
        "control_coords": torch.empty(0, 3, dtype=torch.float32),
        "control_disp": final_control_grid,
        "dense_disp": final_dense.detach(),
        "training_elapsed_sec": time.time() - train_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "param_count": int(final_control.numel()),
        "cps": int(final_cps),
        "stage_full_cps": stage_cps,
        "stage_downsample": stage_downsample,
        "stage_epochs": stage_epochs,
        "stage_lr": stage_lr,
        "nested_control_shapes_zyx": {str(k): list(v) for k, v in nested_shapes.items()},
        "prolongation": "exact crop-aligned cubic B-spline dyadic knot insertion",
        "coords_batch_size": None,
        "train_coords": int(fixed_mask.detach().reshape(-1).bool().sum().detach().cpu()),
    }


def write_outputs(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    target_audit: Dict[str, object],
    target_disp: torch.Tensor,
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    total_elapsed_sec: float,
) -> None:
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_disp = metrics["dense_disp"]
    assert isinstance(dense_disp, torch.Tensor)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_disp.device)
    dense_coords = id_grid + dense_disp
    eval_metrics = c.evaluate_from_dense_coords(dense_coords, batch)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    field_metrics = field_mse_metrics(dense_disp, target_disp, fixed_mask)

    control_disp = metrics.get("control_disp")
    if isinstance(control_disp, torch.Tensor):
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(
            dense_disp,
            fixed_mask.detach().bool(),
            control_disp_norm=control_disp,
            cps=int(metrics["cps"]),
        )
    else:
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp, fixed_mask.detach().bool())

    if args.save_final_dense:
        c.save_dense_disp(run_dir / f"disp_norm_{fixed_id}_{moving_id}.npy", dense_coords)

    state: Dict[str, object] = {
        "args": vars(args),
        "run_name": run_name,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "training_elapsed_sec": metrics["training_elapsed_sec"],
        "training_peak_mem_gb": metrics["training_peak_mem_gb"],
    }
    if "model" in metrics:
        state["model_state_dict"] = metrics["model"].state_dict()  # type: ignore[union-attr]
    if "generator" in metrics:
        state["generator_state_dict"] = metrics["generator"].state_dict()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    if "control_disp" in metrics and "generator" not in metrics:
        state["control_disp"] = metrics["control_disp"].detach().cpu()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    for key in (
        "stage_summaries",
        "prolongation_diagnostics",
        "stage_full_cps",
        "stage_downsample",
        "stage_epochs",
        "stage_lr",
        "nested_control_shapes_zyx",
        "prolongation",
    ):
        if key in metrics:
            state[key] = metrics[key]
    torch.save(state, run_dir / "model_state.pt")

    objective_name = str(getattr(args, "field_fit_objective", "greedy_field_fit_mse"))

    if isinstance(control_disp, torch.Tensor):
        torch.save(
            {
                "control_disp": control_disp.detach().cpu(),
                "control_coords": metrics["control_coords"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": metrics["control_shape"],
                "cps": metrics["cps"],
                "image_shape": c.IMAGE_SHAPE,
                "objective": objective_name,
                "field_fit_target_units": "normalized_grid",
                "method_key": field_method_key(args.method),
                "alpha_recorded_not_used": float(args.alpha),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )

    c.write_csv(run_dir / "history.csv", metrics["history"])  # type: ignore[arg-type]
    if "stage_history" in metrics:
        c.write_csv(run_dir / "stage_history.csv", metrics["stage_history"])  # type: ignore[arg-type]
    if "prolongation_diagnostics" in metrics and metrics["prolongation_diagnostics"]:
        c.write_csv(run_dir / "prolongation_metrics.csv", metrics["prolongation_diagnostics"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "case_metrics.csv", [c.accuracy_row(eval_metrics, fixed_id, moving_id)])
    c.write_csv(run_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(run_dir / "be_metrics.csv", be_rows)
    c.write_csv(run_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])
    c.write_json(run_dir / "target_audit.json", target_audit)

    last = metrics["history"][-1]  # type: ignore[index]
    is_idir = args.method == "idir_original"
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": str(args.variant),
        "objective": objective_name,
        "method": args.method,
        "method_key": field_method_key(args.method),
        "method_label": field_method_label(args.method),
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": int(args.epochs),
        "eval_interval": int(args.eval_interval),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "activation": "direct_control_points" if args.method.startswith("direct_cp_") else "Siren",
        "omega": float(args.idir_omega if is_idir else args.spline_omega),
        "lr": float(args.lr),
        "alpha": float(args.alpha),
        "alpha_role": "recorded_operating_point_not_used_by_mse_objective",
        "be_type": "dense_dvf_finite_difference_eval" if is_idir else "be_bspline_analytic_posthoc",
        "diffusion_type": "dense_dvf_finite_difference_eval" if is_idir else "diffusion_bspline_analytic_posthoc",
        "regularizer": "none_mse_only",
        "native_analytic_be_formula": "not_computed_for_field_fit_idir" if is_idir else "bspline_analytic_posthoc",
        "native_analytic_diffusion_formula": "not_computed_for_field_fit_idir" if is_idir else "bspline_analytic_posthoc",
        "be_training_reduction": "not_applicable_mse_only",
        "image_loss": "none_training_posthoc_sinr_ncc",
        "sample_domain": "fixed_image_positive",
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": metrics.get("cps"),
        "bspline_order": 3 if not is_idir else None,
        "bspline_mode": "disp" if not is_idir else None,
        "grid_order": "pytorch",
        "seed": int(args.seed),
        "coords_batch_size": metrics.get("coords_batch_size"),
        "train_coords": metrics.get("train_coords"),
        "field_chunk_size": metrics.get("field_chunk_size"),
        "field_chunks": metrics.get("field_chunks"),
        "param_count": metrics.get("param_count"),
        "stage_full_cps": metrics.get("stage_full_cps"),
        "stage_downsample": metrics.get("stage_downsample"),
        "stage_epochs": metrics.get("stage_epochs"),
        "stage_lr": metrics.get("stage_lr"),
        "stage_summaries": metrics.get("stage_summaries"),
        "prolongation_diagnostics": metrics.get("prolongation_diagnostics"),
        "nested_control_shapes_zyx": metrics.get("nested_control_shapes_zyx"),
        "prolongation": metrics.get("prolongation"),
        "finest_mse_diagnostic": bool(getattr(args, "finest_mse_diagnostic", False)),
        "finest_mse_every": int(getattr(args, "finest_mse_every", 0)),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": last["loss"],
        "final_data_loss": last["data_loss"],
        "final_reg_loss": last["reg_loss"],
        "dense_image_loss": eval_metrics["image_loss"],
        "dice_mean": eval_metrics["dice_mean"],
        "hd95_mean": eval_metrics["hd95_mean"],
        "asd_backend": eval_metrics["asd_backend"],
        "asd_warped_to_fixed_mean": eval_metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": eval_metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": eval_metrics["asd_symmetric_mean"],
    }
    summary.update(target_audit)
    summary.update(field_metrics)
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(c.accuracy_row(eval_metrics, fixed_id, moving_id))
    summary.update(reg_metrics)
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args)
    key = field_method_key(args.method)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return

    total_start = time.time()
    started_at = c.now_text()
    device = c.set_cuda_device(args.gpu)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    batch = c.load_oasis_case(
        rank=int(args.case_rank),
        device=device,
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
    )
    validate_expected_case(
        args,
        original_index=int(batch["original_case_index"]),
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
    )
    target_disp, target_audit = load_greedy_target(
        args,
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
        device=device,
    )
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    target_audit.update(field_domain_audit(fixed_mask))
    if args.validate_target_only:
        print(json.dumps({"case": {k: str(batch[k]) for k in ["case_rank", "original_case_index", "fixed_id", "moving_id"]}, "target_audit": target_audit}, indent=2), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "objective": "greedy_field_fit_mse",
                "method": args.method,
                "method_key": key,
                "gpu": args.gpu,
                "case_rank": batch["case_rank"],
                "original_case_index": batch["original_case_index"],
                "fixed_id": batch["fixed_id"],
                "moving_id": batch["moving_id"],
                "target_audit": target_audit,
                "alpha_recorded_not_used": args.alpha,
                "lr": args.lr,
                "epochs": args.epochs,
            },
            allow_nan=True,
        ),
        flush=True,
    )

    if args.method == "idir_original":
        metrics = train_idir_field(args, target_disp, fixed_mask, device)
    elif args.method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        metrics = train_multistage_field(args, target_disp, fixed_mask, device)
    else:
        metrics = train_spline_field(args, target_disp, fixed_mask, device)
    metrics["started_at"] = started_at
    total_elapsed_sec = time.time() - total_start
    write_outputs(args, batch, target_audit, target_disp, metrics, run_dir, run_name, total_elapsed_sec)


if __name__ == "__main__":
    main()
