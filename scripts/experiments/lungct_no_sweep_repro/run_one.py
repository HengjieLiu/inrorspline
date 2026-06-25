#!/usr/bin/env python
"""Run one LungCT no-sweep reproduction job."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import common as c


METHOD_IDIR = "idir_be_default_lungmask"
METHOD_SINR_CPS4_PTV = "sinr_cps4_ptv_lcc_isotv"
METHOD_DIRECT_CP_CPS4_PTV = "direct_cp_cps4_ptv_lcc_isotv"
METHOD_DIRECT_CP_32_PTV = "direct_cp_32_16_8_4_ptv_lcc_isotv"
METHOD_DIRECT_CP_32_GLOBAL_NCC_BE = "direct_cp_32_16_8_4_global_ncc_bspline_be"
METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE = "direct_cp_32_16_8_4_global_ncc_fd_be"
METHODS = [
    METHOD_IDIR,
    METHOD_SINR_CPS4_PTV,
    METHOD_DIRECT_CP_CPS4_PTV,
    METHOD_DIRECT_CP_32_PTV,
    METHOD_DIRECT_CP_32_GLOBAL_NCC_BE,
    METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--case-id", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--save-dense-dvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-loader", action="store_true")
    parser.add_argument("--audit-cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--log-base", choices=("e", "2"), default="e")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--alpha-bending", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--cps", type=int, default=4)

    parser.add_argument("--border-mask-voxels", type=int, default=5)
    parser.add_argument("--ptv-lcc-sigma-mm", type=float, default=2.1)
    parser.add_argument("--ptv-lcc-radius-sigma", type=float, default=2.0)
    parser.add_argument("--ptv-lcc-sigma-floor-pix", type=float, default=0.8)
    parser.add_argument("--ptv-isotv-weight", type=float, default=0.11)
    parser.add_argument("--ptv-csqrt", type=float, default=5e-3)

    parser.add_argument("--stage-full-cps", type=str, default="32,16,8,4")
    parser.add_argument("--stage-downsample", type=str, default="8,4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="200,400,800,1000")
    parser.add_argument("--stage-lr", type=str, default="1e-3")
    parser.add_argument("--antialias-sigma-scale", type=float, default=0.5)
    parser.add_argument("--antialias-radius-sigma", type=float, default=3.0)
    return parser.parse_args()


def default_epochs(method: str) -> int:
    if method in {METHOD_DIRECT_CP_32_PTV, METHOD_DIRECT_CP_32_GLOBAL_NCC_BE, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE}:
        raise ValueError("Multistage method uses --stage-epochs, not --epochs.")
    return 2500


def default_lr(method: str) -> float:
    if method in {
        METHOD_DIRECT_CP_CPS4_PTV,
        METHOD_DIRECT_CP_32_PTV,
        METHOD_DIRECT_CP_32_GLOBAL_NCC_BE,
        METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE,
    }:
        return 1e-3
    return 1e-4


def run_name(args: argparse.Namespace, method: str, lr: float) -> str:
    if args.run_name:
        return args.run_name
    if method == METHOD_DIRECT_CP_32_PTV:
        epoch_name = args.stage_epochs.replace(",", "-")
        return (
            f"{method}_case{args.case_id:02d}_sched32-16-8-4_ds8-4-2-1_"
            f"e{epoch_name}_lr{c.format_float(lr)}_w{c.format_float(args.ptv_isotv_weight)}_seed{args.seed}"
        )
    if method == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE:
        epoch_name = args.stage_epochs.replace(",", "-")
        return (
            f"{method}_case{args.case_id:02d}_sched32-16-8-4_ds8-4-2-1_"
            f"e{epoch_name}_lr{c.format_float(lr)}_a{c.format_float(args.alpha)}_seed{args.seed}"
        )
    if method == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE:
        epoch_name = args.stage_epochs.replace(",", "-")
        return (
            f"{method}_case{args.case_id:02d}_sched32-16-8-4_ds8-4-2-1_"
            f"e{epoch_name}_lr{c.format_float(lr)}_a{c.format_float(args.alpha)}_seed{args.seed}"
        )
    epochs = int(args.epochs if args.epochs is not None else default_epochs(method))
    extra = f"bend{c.format_float(args.alpha_bending)}" if method == METHOD_IDIR else f"w{c.format_float(args.ptv_isotv_weight)}"
    return f"{method}_case{args.case_id:02d}_e{epochs}_lr{c.format_float(lr)}_{extra}_seed{args.seed}"


def make_out_dir(args: argparse.Namespace, method: str, lr: float) -> Path:
    return Path(args.out_root) / method / run_name(args, method, lr)


def evaluate_idir_dense(model: torch.nn.Module, image_shape: Sequence[int], device: torch.device, chunk_size: int) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(image_shape, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(model(coords))
    return torch.cat(chunks, dim=0).reshape(*tuple(int(v) for v in image_shape), 3)


def train_idir(args: argparse.Namespace, case: c.LungCTCase, device: torch.device, out_dir: Path) -> Dict[str, object]:
    epochs = int(args.epochs if args.epochs is not None else 2500)
    lr = float(args.lr if args.lr is not None else 1e-4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
    coords_all = c.make_coordinate_tensor(case.shape_zyx, device)
    coords_train = coords_all[mask_t.reshape(-1)].contiguous()
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty IDIR training coordinate set.")

    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach().requires_grad_(True)
        disp = model(coords)
        warped = c.sample_volume(moving_t, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_t, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.idir_bending_energy(coords, disp, batch_size=batch_size)
        loss = data_loss + float(args.alpha_bending) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    dense = evaluate_idir_dense(model, case.shape_zyx, device, args.eval_chunk_size)
    c.write_csv(out_dir / "loss_history.csv", history)
    return {
        "dense_disp_norm_xyz": dense,
        "history_tail": history[-10:],
        "training": {
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "loss": "sampled NCC + IDIR bending",
            "alpha_bending": float(args.alpha_bending),
            "batch_size": batch_size,
            "train_coords": n_train,
            "seed": int(args.seed),
            "network": {
                "type": "SIREN",
                "layers": [3] + [int(args.hidden_dim)] * int(args.hidden_layers) + [3],
                "omega": float(args.idir_omega),
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            },
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def train_single_stage_spline(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
    *,
    direct_cp: bool,
) -> Dict[str, object]:
    method_epochs = int(args.epochs if args.epochs is not None else 2500)
    method_lr = float(args.lr if args.lr is not None else (1e-3 if direct_cp else 1e-4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    image_shape = tuple(int(v) for v in fixed_t.shape)
    control_coords, control_shape = c.bspline_control_coords(image_shape, args.cps, device)
    id_dense = c.make_coordinate_grid(image_shape, device).detach()
    border_mask = c.make_border_mask(image_shape, int(args.border_mask_voxels), device, fixed_t.dtype)
    ptv_cache = c.make_ptv_lcc_fixed_cache(
        fixed_t,
        (1.0, 1.0, 1.0),
        args.ptv_lcc_sigma_mm,
        args.ptv_lcc_radius_sigma,
        args.ptv_lcc_sigma_floor_pix,
    )

    if direct_cp:
        params = torch.nn.Parameter(torch.zeros(control_coords.shape[0], 3, device=device, dtype=torch.float32))
        trainable = [params]
        model_params = int(params.numel())
    else:
        model = c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega).to(device)
        trainable = list(model.parameters())
        model_params = int(sum(p.numel() for p in trainable if p.requires_grad))

    optimizer = torch.optim.Adam(trainable, lr=method_lr)
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    dense_disp: Optional[torch.Tensor] = None
    for epoch in range(1, method_epochs + 1):
        if direct_cp:
            control = params
        else:
            control = model(control_coords)
        dense_disp = c.expand_bspline_controls(control, control_shape, image_shape, args.cps)
        warped = c.warp_full(moving_t, id_dense + dense_disp, mode="bilinear")
        data_loss = c.ptv_local_cc_loss_with_cache(warped, border_mask, ptv_cache, (1.0, 1.0, 1.0), args.ptv_lcc_radius_sigma)
        reg_loss = c.isotv_control_loss(control, control_shape, image_shape, (1.0, 1.0, 1.0), args.cps, args.ptv_csqrt)
        loss = data_loss + float(args.ptv_isotv_weight) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "reg_weight": float(args.ptv_isotv_weight),
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == method_epochs:
            print(json.dumps(row), flush=True)
    if dense_disp is None:
        dense_disp = torch.zeros((*image_shape, 3), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    c.write_csv(out_dir / "loss_history.csv", history)
    return {
        "dense_disp_norm_xyz": dense_disp.detach(),
        "history_tail": history[-10:],
        "training": {
            "epochs": method_epochs,
            "lr": method_lr,
            "optimizer": "Adam",
            "loss": "pTV-style LCC + isoTV",
            "cps": int(args.cps),
            "bspline_order": 3,
            "ptv_lcc_sigma_mm": float(args.ptv_lcc_sigma_mm),
            "ptv_isotv_weight": float(args.ptv_isotv_weight),
            "ptv_csqrt": float(args.ptv_csqrt),
            "border_mask_voxels": int(args.border_mask_voxels),
            "seed": int(args.seed),
            "network": {
                "type": "direct_control_points" if direct_cp else "SIREN_control_generator",
                "trainable_parameters": model_params,
                "hidden_dim": None if direct_cp else int(args.hidden_dim),
                "hidden_layers": None if direct_cp else int(args.hidden_layers),
                "omega": None if direct_cp else float(args.spline_omega),
            },
            "control_shape_zyx": list(control_shape),
            "n_control": int(np.prod(control_shape)),
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def prolongation_diagnostics(
    prev_control: torch.Tensor,
    prev_shape: Sequence[int],
    prev_dense_full: torch.Tensor,
    target_shape: Sequence[int],
    target_cps: int,
    image_shape: Sequence[int],
    stage_idx: int,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    prev_grid = c.flat_to_grid(prev_control.detach(), prev_shape)
    exact_grid = c.cubic_bspline_dyadic_knot_insert_3d(prev_grid, target_shape)
    resize_grid = c.resize_control_grid(prev_grid, target_shape)
    rows: List[Dict[str, object]] = []
    for method, grid in (("exact_knot", exact_grid), ("trilinear_resize", resize_grid)):
        dense = c.expand_on_shape(c.grid_to_flat(grid), target_shape, target_cps, image_shape, 1)
        row: Dict[str, object] = {
            "stage_idx": int(stage_idx),
            "target_cps": int(target_cps),
            "method": method,
            "source_shape_zyx": list(int(v) for v in prev_shape),
            "target_shape_zyx": list(int(v) for v in target_shape),
        }
        row.update(c.field_difference_metrics(dense - prev_dense_full, prefix="preserve_"))
        rows.append(row)
    return c.grid_to_flat(exact_grid).detach(), rows


def optimize_stage(
    args: argparse.Namespace,
    fixed_t: torch.Tensor,
    moving_t: torch.Tensor,
    full_border_mask: torch.Tensor,
    control: torch.nn.Parameter,
    control_shape: Sequence[int],
    full_cps: int,
    downsample: int,
    epochs: int,
    lr: float,
    stage_idx: int,
) -> Tuple[torch.Tensor, Dict[str, object], List[Dict[str, object]]]:
    fixed_l, blur_sigma, blur_kernel = c.make_loss_volume(
        fixed_t, downsample, args.antialias_sigma_scale, args.antialias_radius_sigma
    )
    moving_l, _sigma_m, _kernel_m = c.make_loss_volume(
        moving_t, downsample, args.antialias_sigma_scale, args.antialias_radius_sigma
    )
    is_analytic_be_loss = str(args.method) == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE
    is_fd_be_loss = str(args.method) == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE
    is_be_loss = is_analytic_be_loss or is_fd_be_loss
    mask_l = (
        None
        if is_be_loss
        else c.make_loss_mask(full_border_mask, downsample, args.antialias_sigma_scale, args.antialias_radius_sigma)
    )
    eval_shape = tuple(int(v) for v in fixed_l.shape)
    cur_pix_zyx = c.stage_pix_zyx((1.0, 1.0, 1.0), fixed_t.shape, eval_shape)
    id_grid = c.make_coordinate_grid(eval_shape, fixed_t.device).detach()
    domain_cps = c.domain_cps_from_full(full_cps, downsample)
    ptv_cache = None
    if not is_be_loss:
        ptv_cache = c.make_ptv_lcc_fixed_cache(
            fixed_l,
            cur_pix_zyx,
            args.ptv_lcc_sigma_mm,
            args.ptv_lcc_radius_sigma,
            args.ptv_lcc_sigma_floor_pix,
        )
    optimizer = torch.optim.Adam([control], lr=lr)
    rows: List[Dict[str, object]] = []
    start = time.time()
    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "domain_cps": int(domain_cps),
                "control_shape_zyx": list(int(v) for v in control_shape),
                "eval_shape_zyx": list(eval_shape),
                "stage_pix_zyx_mm": list(cur_pix_zyx),
                "epochs": int(epochs),
                "lr": float(lr),
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        dense_disp = c.expand_on_shape(control, control_shape, full_cps, eval_shape, downsample)
        warped = c.warp_full(moving_l, id_grid + dense_disp, mode="bilinear")
        if is_be_loss:
            data_loss = c.ncc_loss(fixed_l[None, None], warped[None, None])
            reg_weight = float(args.alpha)
            if abs(reg_weight) <= 1e-12:
                reg_loss = torch.zeros((), dtype=data_loss.dtype, device=data_loss.device)
            elif is_analytic_be_loss:
                reg_loss = c.analytic_bspline_sampled_be(
                    control,
                    control_shape,
                    eval_shape,
                    domain_cps,
                    component_reduction="mean",
                )
            else:
                reg_loss = c.dense_forward_difference_be(dense_disp, component_reduction="mean")
        else:
            assert ptv_cache is not None
            data_loss = c.ptv_local_cc_loss_with_cache(warped, mask_l, ptv_cache, cur_pix_zyx, args.ptv_lcc_radius_sigma)
            reg_loss = c.isotv_control_loss(control, control_shape, eval_shape, cur_pix_zyx, domain_cps, args.ptv_csqrt)
            reg_weight = float(args.ptv_isotv_weight)
        loss = data_loss + reg_weight * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "full_cps": float(full_cps),
            "downsample": float(downsample),
            "domain_cps": float(domain_cps),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "reg_weight": float(reg_weight),
            "alpha": float(args.alpha) if is_be_loss else float("nan"),
            "lr": float(lr),
        }
        rows.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        dense_full = c.expand_on_shape(control, control_shape, full_cps, fixed_t.shape, 1)
    elapsed = time.time() - start
    ctrl_abs = control.detach().abs().reshape(-1)
    summary = {
        "stage_idx": int(stage_idx),
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(eval_shape),
        "stage_pix_zyx_mm": list(cur_pix_zyx),
        "n_control": int(np.prod(control_shape)),
        "epochs": int(epochs),
        "lr": float(lr),
        "reg_weight": float(args.alpha if is_be_loss else args.ptv_isotv_weight),
        "alpha": float(args.alpha) if is_be_loss else None,
        "loss_mode": (
            "global_ncc_bspline_be"
            if is_analytic_be_loss
            else "global_ncc_fd_be"
            if is_fd_be_loss
            else "ptv_lcc_isotv"
        ),
        "border_mask_voxels": 0 if is_be_loss else int(args.border_mask_voxels),
        "antialias_sigma": float(blur_sigma),
        "antialias_kernel_size": int(blur_kernel),
        "elapsed_sec": float(elapsed),
        "control_abs_max_norm": float(ctrl_abs.max().detach().cpu()),
        "control_abs_p95_norm": float(torch.quantile(ctrl_abs, 0.95).detach().cpu()),
    }
    return dense_full.detach(), summary, rows


def train_multistage(args: argparse.Namespace, case: c.LungCTCase, device: torch.device, out_dir: Path) -> Dict[str, object]:
    stage_cps = c.parse_int_list(args.stage_full_cps)
    stage_downsample = c.parse_stage_ints(args.stage_downsample, len(stage_cps))
    stage_epochs = c.parse_stage_ints(args.stage_epochs, len(stage_cps))
    stage_lr = c.parse_float_list(args.stage_lr, len(stage_cps), float(args.lr if args.lr is not None else 1e-3))
    base_spacing = int(stage_cps[0])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    image_shape = tuple(int(v) for v in fixed_t.shape)
    border_mask = c.make_border_mask(image_shape, int(args.border_mask_voxels), device, fixed_t.dtype)
    nested_shapes = c.build_nested_cp_shapes(image_shape, base_spacing=base_spacing, spacings=stage_cps)
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
    final_domain_cps: Optional[int] = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for stage_idx, (full_cps, downsample, epochs, lr) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        control_shape = tuple(int(v) for v in nested_shapes[int(full_cps)])
        control = torch.nn.Parameter(torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32))
        if prev_control is not None and prev_shape is not None and prev_dense_full is not None:
            init_flat, rows = prolongation_diagnostics(
                prev_control,
                prev_shape,
                prev_dense_full,
                control_shape,
                int(full_cps),
                image_shape,
                int(stage_idx),
            )
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        dense_full, summary, rows = optimize_stage(
            args,
            fixed_t,
            moving_t,
            border_mask,
            control,
            control_shape,
            int(full_cps),
            int(downsample),
            int(epochs),
            float(lr),
            int(stage_idx),
        )
        prev_control = control.detach()
        prev_shape = control_shape
        prev_dense_full = dense_full
        final_dense = dense_full
        final_control = control.detach()
        final_control_shape = control_shape
        final_domain_cps = int(full_cps)
        stage_summaries.append(summary)
        history.extend(rows)
    if final_dense is None:
        final_dense = torch.zeros((*image_shape, 3), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    c.write_csv(out_dir / "stage_history.csv", history)
    c.write_csv(out_dir / "prolongation_metrics.csv", prolong_rows)
    analytic_regularity: Dict[str, float] = {}
    if str(args.method) == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE and final_control is not None and final_control_shape is not None:
        mask_t = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
        cps = int(final_domain_cps if final_domain_cps is not None else stage_cps[-1])
        analytic_regularity = {
            "be_bspline_analytic_full_norm": float(
                c.analytic_bspline_sampled_be(final_control, final_control_shape, image_shape, cps).detach().cpu()
            ),
            "be_bspline_analytic_fg_norm": float(
                c.analytic_bspline_sampled_be(final_control, final_control_shape, image_shape, cps, mask_t).detach().cpu()
            ),
            "diffusion_bspline_analytic_full_norm": float(
                c.analytic_bspline_sampled_diffusion(final_control, final_control_shape, image_shape, cps).detach().cpu()
            ),
            "diffusion_bspline_analytic_fg_norm": float(
                c.analytic_bspline_sampled_diffusion(final_control, final_control_shape, image_shape, cps, mask_t).detach().cpu()
            ),
            "final_control_cps": float(cps),
        }
    is_analytic_be_loss = str(args.method) == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE
    is_fd_be_loss = str(args.method) == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE
    is_be_loss = is_analytic_be_loss or is_fd_be_loss
    regularizer_name = (
        "analytic_cubic_bspline_be"
        if is_analytic_be_loss
        else "dense_forward_difference_be"
        if is_fd_be_loss
        else "ptv_isotv"
    )
    return {
        "dense_disp_norm_xyz": final_dense.detach(),
        "history_tail": history[-10:],
        "stage_summaries": stage_summaries,
        "prolongation_diagnostics": prolong_rows,
        "analytic_bspline_regularity_fg_registration_crop": analytic_regularity,
        "training": {
            "stage_full_cps": stage_cps,
            "stage_downsample": stage_downsample,
            "stage_epochs": stage_epochs,
            "stage_lr": stage_lr,
            "optimizer": "Adam",
            "loss": (
                "full-crop global NCC + analytic cubic B-spline BE"
                if is_analytic_be_loss
                else "full-crop global NCC + dense forward-difference BE"
                if is_fd_be_loss
                else "pTV-style LCC + isoTV"
            ),
            "image_loss": "full_crop_global_ncc" if is_be_loss else "ptv_lcc",
            "regularizer": regularizer_name,
            "alpha": float(args.alpha) if is_be_loss else None,
            "ptv_lcc_sigma_mm": None if is_be_loss else float(args.ptv_lcc_sigma_mm),
            "ptv_isotv_weight": None if is_be_loss else float(args.ptv_isotv_weight),
            "ptv_csqrt": None if is_be_loss else float(args.ptv_csqrt),
            "border_mask_voxels": 0 if is_be_loss else int(args.border_mask_voxels),
            "prolongation": "exact crop-aligned cubic B-spline dyadic knot insertion",
            "seed": int(args.seed),
            "nested_control_shapes_zyx": {str(k): list(v) for k, v in nested_shapes.items()},
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def main() -> None:
    args = parse_args()
    if args.audit_loader:
        audit = c.loader_audit(args.data_root, c.parse_int_list(args.audit_cases))
        print(json.dumps(audit, indent=2), flush=True)
        return
    if args.method is None:
        raise SystemExit("--method is required unless --audit-loader is set.")

    method = str(args.method)
    lr = float(args.lr if args.lr is not None else default_lr(method))
    out_dir = make_out_dir(args, method, lr)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        try:
            existing = c.read_json(summary_path)
        except Exception:
            existing = {}
        if isinstance(existing, dict) and existing.get("status") == "completed":
            print(json.dumps({"skipped_completed": str(summary_path)}), flush=True)
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    device = c.set_cuda_device(args.gpu)
    case = c.load_case(args.data_root, args.case_id)
    if method == METHOD_IDIR:
        result = train_idir(args, case, device, out_dir)
        protocol_tag = "ptv_landmark_crop_1mm_with_idir_fixed_t00_lungmask_sampling"
        mask_rule = "fixed_t00_lung_mask_sampling_and_metric"
        loss_domain = "fixed_t00_lung_mask_sampled"
    elif method == METHOD_SINR_CPS4_PTV:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=False)
        protocol_tag = "ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
    elif method == METHOD_DIRECT_CP_CPS4_PTV:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=True)
        protocol_tag = "ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
    elif method == METHOD_DIRECT_CP_32_PTV:
        result = train_multistage(args, case, device, out_dir)
        protocol_tag = "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
    elif method == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE:
        result = train_multistage(args, case, device, out_dir)
        protocol_tag = (
            "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+"
            "full_crop_global_ncc_bspline_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
    elif method == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE:
        result = train_multistage(args, case, device, out_dir)
        protocol_tag = (
            "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+"
            "full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = c.final_metrics_from_dense(
        result["dense_disp_norm_xyz"],
        case,
        out_dir,
        bool(args.save_dense_dvf),
        args.log_base,
    )
    summary: Dict[str, object] = {
        "run_name": out_dir.name,
        "status": "completed",
        "method": method,
        "dataset": "4dct",
        "case_id": int(args.case_id),
        "seed": int(args.seed),
        "data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "case_dir": str(case.case_dir),
        "image_domain": "pTVReg landmark crop",
        "spacing": "isotropic_1mm",
        "fixed_phase": "T00",
        "moving_phase": "T50",
        "mask_rule": mask_rule,
        "loss_domain": loss_domain,
        "protocol_tag": protocol_tag,
        "input_shape_zyx": list(case.shape_zyx),
        "fixed_lung_mask_voxels": int(case.fixed_mask.sum()),
        "fixed_lung_mask_fraction": float(case.fixed_mask.mean()),
        "preprocess_audit": case.preprocess_audit,
        "training": result["training"],
        "elapsed_sec": float(result["elapsed_sec"]),
        "peak_mem_gb": float(result["peak_mem_gb"]),
        "history_tail": result.get("history_tail", []),
        "notes": [
            "Inputs are canonical regenerated pTV-prep cropped 1mm arrays.",
            (
                "This method uses full-crop global NCC with no border-mask or lung-mask loss weighting."
                if method in {METHOD_DIRECT_CP_32_GLOBAL_NCC_BE, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE}
                else "pTV loss methods use border-mask loss weighting, not lung-mask loss weighting."
            ),
            "Foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm grid.",
            "TRE is measured directly in cropped 1mm landmark coordinates.",
        ],
    }
    summary.update(
        {
            k: v
            for k, v in result.items()
            if k in ("stage_summaries", "prolongation_diagnostics", "analytic_bspline_regularity_fg_registration_crop")
        }
    )
    summary.update(metrics)
    c.write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "method": method,
                "case_id": int(args.case_id),
                "tre_snap_mean_mm": summary["tre_snap_to_voxel"]["tre_mean_mm"],
                "tre_subvoxel_mean_mm": summary["tre_subvoxel"]["tre_mean_mm"],
                "elapsed_sec": summary["elapsed_sec"],
                "peak_mem_gb": summary["peak_mem_gb"],
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
