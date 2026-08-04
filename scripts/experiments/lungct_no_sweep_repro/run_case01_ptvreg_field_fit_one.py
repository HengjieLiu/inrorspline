#!/usr/bin/env python
"""Fit one LungCT model directly to a saved pTVReg displacement."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_one as image_runner  # noqa: E402


DEFAULT_CASE_ID = 1
METHODS = ["idir", "sinr_cps4", "direct_cp_cps4", "direct_cp_4stage"]


def variant_for_case(case_id: int) -> str:
    return f"lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep_v1"


def default_out_root(case_id: int) -> Path:
    return Path(f"/database/inrorspline/runs/{variant_for_case(case_id)}")


def default_ptvreg_mat(case_id: int) -> Path:
    env_key = f"INRORSPLINE_PTVREG_CASE{int(case_id):02d}_MAT"
    return Path(
        os.environ.get(
            env_key,
            f"/database/inrorspline/ptvreg/4DCT/case{int(case_id):02d}_T50_to_T00_deformation.mat",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--case-id", type=int, default=DEFAULT_CASE_ID)
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--ptvreg-mat", type=Path, default=None)
    parser.add_argument("--expected-tre-phys", type=float, default=None)
    parser.add_argument("--expected-tre-std", type=float, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--cps", type=int, default=4)
    parser.add_argument("--stage-full-cps", type=str, default="32,16,8,4")
    parser.add_argument("--stage-downsample", type=str, default="8,4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="400,400,800,900")
    parser.add_argument("--stage-lr", type=str, default="")
    parser.add_argument("--finest-mse-diagnostic", action="store_true")
    parser.add_argument("--finest-mse-every", type=int, default=1)
    parser.add_argument("--save-dense-dvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-target-only", action="store_true")
    parser.add_argument("--log-base", choices=("e", "2"), default="e")
    args = parser.parse_args()
    if int(args.finest_mse_every) < 1:
        raise ValueError("--finest-mse-every must be >= 1")
    if args.ptvreg_mat is None:
        args.ptvreg_mat = default_ptvreg_mat(int(args.case_id))
    if args.out_root is None:
        args.out_root = default_out_root(int(args.case_id))
    return args


def method_key(method: str) -> str:
    if method == "idir":
        return "idir_ptvreg_field_mse"
    if method == "sinr_cps4":
        return "sinr_cps4_ptvreg_field_mse"
    if method == "direct_cp_cps4":
        return "direct_cp_cps4_ptvreg_field_mse"
    if method == "direct_cp_4stage":
        return "direct_cp_4stage_ptvreg_field_mse"
    raise ValueError(f"Unsupported method: {method}")


def method_label(method: str) -> str:
    return {
        "idir": "IDIR",
        "sinr_cps4": "SINR cps4",
        "direct_cp_cps4": "DirectCP cps4",
        "direct_cp_4stage": "DirectCP 4stage cps4",
    }[method]


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.method == "direct_cp_4stage":
        stage_epochs = args.stage_epochs.replace(",", "-")
        return (
            f"ptvreg_field_fit_{method_key(args.method)}_case{args.case_id:02d}"
            f"_sched32-16-8-4_e{stage_epochs}_lr{c.format_float(args.lr)}"
        )
    return (
        f"ptvreg_field_fit_{method_key(args.method)}_case{args.case_id:02d}"
        f"_e{args.epochs}_lr{c.format_float(args.lr)}"
    )


def scalar_dataset(handle: h5py.File, key: str) -> float:
    return float(np.asarray(handle[key]).reshape(-1)[0])


def normalize_crop_v(crop_v: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop_v, dtype=np.float64)
    if arr.shape == (3, 2):
        arr = arr.T
    if arr.shape != (2, 3):
        raise ValueError(f"Expected crop_v_saved shape (2,3) or (3,2), got {arr.shape}")
    return arr


def crop_v_xyz1_to_bounds_zyx0(crop_v_xyz1: np.ndarray) -> np.ndarray:
    start0_xyz = crop_v_xyz1[0].astype(np.int64) - 1
    end0_xyz = crop_v_xyz1[1].astype(np.int64) - 1
    return np.stack((start0_xyz[[2, 1, 0]], end0_xyz[[2, 1, 0]]), axis=1)


def matlab_snap_tre_from_h5(
    t_h5_czyx: np.ndarray,
    pts_fix_xyz1: np.ndarray,
    pts_mov_xyz1: np.ndarray,
    spc_xyz: np.ndarray,
) -> Dict[str, object]:
    field_xyz = np.asarray(t_h5_czyx).transpose(3, 2, 1, 0)
    idx = np.rint(pts_fix_xyz1).astype(np.int64) - 1
    if np.any(idx < 0) or np.any(idx[:, 0] >= field_xyz.shape[0]) or np.any(idx[:, 1] >= field_xyz.shape[1]) or np.any(idx[:, 2] >= field_xyz.shape[2]):
        raise ValueError("pTVReg fixed landmarks are outside Tptv_rsz bounds")
    disp_phys_xyz = field_xyz[idx[:, 0], idx[:, 1], idx[:, 2], :].astype(np.float64, copy=False)
    moved_xyz = np.rint(pts_fix_xyz1 + disp_phys_xyz / spc_xyz.reshape(1, 3))
    errs = np.sqrt(np.sum(((moved_xyz - pts_mov_xyz1) * spc_xyz.reshape(1, 3)) ** 2, axis=1))
    return {
        "tre_mean_mm": float(np.mean(errs)),
        "tre_std_mm": float(np.std(errs, ddof=1)),
        "tre_min_mm": float(np.min(errs)),
        "tre_max_mm": float(np.max(errs)),
        "per_landmark_tre_mm": [float(v) for v in errs],
    }


def resize_phys_crop_to_reg1mm(disp_phys_crop_zyx_xyz: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(disp_phys_crop_zyx_xyz, dtype=np.float32)).permute(3, 0, 1, 2)[None]
    resized = F.interpolate(tensor, size=tuple(int(v) for v in shape_zyx), mode="trilinear", align_corners=True)
    return resized[0].permute(1, 2, 3, 0).numpy().astype(np.float32, copy=False)


def phys_xyz_reg1mm_to_norm_xyz(disp_phys_reg_zyx_xyz: np.ndarray) -> np.ndarray:
    shape_zyx = np.asarray(disp_phys_reg_zyx_xyz.shape[:3], dtype=np.float32)
    scale_xyz = np.asarray(
        [
            2.0 / max(float(shape_zyx[2] - 1.0), 1.0),
            2.0 / max(float(shape_zyx[1] - 1.0), 1.0),
            2.0 / max(float(shape_zyx[0] - 1.0), 1.0),
        ],
        dtype=np.float32,
    )
    return disp_phys_reg_zyx_xyz * scale_xyz.reshape(1, 1, 1, 3)


def target_tre_diagnostic(target_norm: np.ndarray, case: c.LungCTCase) -> Dict[str, object]:
    disp_vox_zyx = c.dense_norm_xyz_to_vox_zyx(target_norm)
    return {
        "reg1mm_snap": c.landmark_tre_from_dense_disp(
            disp_vox_zyx,
            case.landmarks_fixed,
            case.landmarks_moving,
            sample_mode="nearest",
            snap_output=True,
        ),
        "reg1mm_subvoxel": c.landmark_tre_from_dense_disp(
            disp_vox_zyx,
            case.landmarks_fixed,
            case.landmarks_moving,
            sample_mode="trilinear",
            snap_output=False,
        ),
    }


def load_ptvreg_target(args: argparse.Namespace, case: c.LungCTCase) -> Tuple[np.ndarray, Dict[str, object]]:
    mat_path = Path(args.ptvreg_mat)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing pTVReg deformation .mat: {mat_path}")
    required = {"Tptv_rsz", "pts_fix", "pts_mov", "spc_orig_saved", "crop_v_saved", "TRE_phys", "TREstd_case"}
    with h5py.File(mat_path, "r") as f:
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(f"Missing pTVReg .mat keys: {missing}")
        t_h5 = f["Tptv_rsz"][:].astype(np.float32, copy=False)
        pts_fix_xyz1 = f["pts_fix"][:].T.astype(np.float64, copy=False)
        pts_mov_xyz1 = f["pts_mov"][:].T.astype(np.float64, copy=False)
        spc_xyz = f["spc_orig_saved"][:].reshape(-1).astype(np.float64, copy=False)
        crop_v_xyz1 = normalize_crop_v(f["crop_v_saved"][:])
        saved_tre = scalar_dataset(f, "TRE_phys")
        saved_std = scalar_dataset(f, "TREstd_case")
    if t_h5.ndim != 4 or t_h5.shape[0] != 3:
        raise ValueError(f"Expected Tptv_rsz h5py shape (3,z,y,x), got {t_h5.shape}")
    if pts_fix_xyz1.shape != pts_mov_xyz1.shape or pts_fix_xyz1.shape[1] != 3:
        raise ValueError(f"Landmark shape mismatch: {pts_fix_xyz1.shape} vs {pts_mov_xyz1.shape}")
    if spc_xyz.shape != (3,):
        raise ValueError(f"Expected 3 spacing values, got {spc_xyz.shape}")

    source_tre = matlab_snap_tre_from_h5(t_h5, pts_fix_xyz1, pts_mov_xyz1, spc_xyz)
    if abs(float(source_tre["tre_mean_mm"]) - saved_tre) > 1e-8:
        raise RuntimeError(f"pTVReg TRE mean validation failed: {source_tre['tre_mean_mm']} != {saved_tre}")
    if abs(float(source_tre["tre_std_mm"]) - saved_std) > 1e-8:
        raise RuntimeError(f"pTVReg TRE std validation failed: {source_tre['tre_std_mm']} != {saved_std}")
    if args.expected_tre_phys is not None and abs(saved_tre - float(args.expected_tre_phys)) > 1e-8:
        raise RuntimeError(f"Unexpected TRE_phys for case{int(args.case_id):02d}: {saved_tre} != {args.expected_tre_phys}")
    if args.expected_tre_std is not None and abs(saved_std - float(args.expected_tre_std)) > 1e-8:
        raise RuntimeError(f"Unexpected TREstd_case for case{int(args.case_id):02d}: {saved_std} != {args.expected_tre_std}")

    crop_bounds_zyx = crop_v_xyz1_to_bounds_zyx0(crop_v_xyz1)
    audit_bounds = np.asarray(case.preprocess_audit.get("crop_bounds_zyx_inclusive"), dtype=np.int64)
    if audit_bounds.shape != (3, 2) or not np.array_equal(crop_bounds_zyx, audit_bounds):
        raise RuntimeError(
            "pTVReg crop_v_saved does not match canonical pTV-prep audit: "
            f"{crop_bounds_zyx.tolist()} vs {audit_bounds.tolist()}"
        )

    disp_phys_native_zyx_xyz = t_h5.transpose(1, 2, 3, 0)
    z0, z1 = crop_bounds_zyx[0]
    y0, y1 = crop_bounds_zyx[1]
    x0, x1 = crop_bounds_zyx[2]
    disp_phys_crop_zyx_xyz = disp_phys_native_zyx_xyz[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1, :]
    disp_phys_reg_zyx_xyz = resize_phys_crop_to_reg1mm(disp_phys_crop_zyx_xyz, case.shape_zyx)
    target_norm_xyz = phys_xyz_reg1mm_to_norm_xyz(disp_phys_reg_zyx_xyz).astype(np.float32, copy=False)
    diag = target_tre_diagnostic(target_norm_xyz, case)

    audit: Dict[str, object] = {
        "source_kind": "ptvreg_saved_deformation_mat",
        "source_family": "ptvreg",
        "source_disp_path": str(mat_path),
        "source_disp_h5py_shape": list(int(v) for v in t_h5.shape),
        "source_disp_matlab_layout": "Tptv_rsz(x,y,z,component)",
        "source_disp_h5py_layout": "component,z,y,x",
        "source_disp_component_order": "x,y,z",
        "source_displacement_units": "physical_mm",
        "source_transform_direction": "fixed_T00_to_moving_T50",
        "source_spacing_xyz_mm": [float(v) for v in spc_xyz],
        "source_spacing_zyx_mm": [float(v) for v in spc_xyz[[2, 1, 0]]],
        "crop_v_saved_xyz_1based_inclusive": crop_v_xyz1.astype(int).tolist(),
        "crop_bounds_zyx_0based_inclusive": crop_bounds_zyx.astype(int).tolist(),
        "source_saved_tre_phys_mean_mm": float(saved_tre),
        "source_saved_trestd_case_mm": float(saved_std),
        "source_recomputed_snap_tre_mean_mm": float(source_tre["tre_mean_mm"]),
        "source_recomputed_snap_tre_std_mm": float(source_tre["tre_std_mm"]),
        "source_tre_validation_abs_err": abs(float(source_tre["tre_mean_mm"]) - float(saved_tre)),
        "field_fit_target_units": "normalized_grid",
        "field_fit_target_shape_zyx3": list(int(v) for v in target_norm_xyz.shape),
        "field_fit_target_extent": "reg1mm_crop_reconstructed_from_saved_native_full_ptvreg_field",
        "field_fit_target_interpolation": "crop_native_to_reg1mm_trilinear_align_corners_true",
        "field_fit_domain": "fixed_t00_lung_mask_reg1mm",
        "field_fit_mask_definition": "fixed T00 lung mask on canonical pTVReg 1mm crop",
        "field_fit_domain_voxels": int(case.fixed_mask.sum()),
        "field_fit_domain_fraction": float(case.fixed_mask.mean()),
        "target_reg1mm_snap_tre_mean_mm": float(diag["reg1mm_snap"]["tre_mean_mm"]),
        "target_reg1mm_subvoxel_tre_mean_mm": float(diag["reg1mm_subvoxel"]["tre_mean_mm"]),
    }
    return target_norm_xyz, audit


def evaluate_idir_dense(model: torch.nn.Module, image_shape: Sequence[int], device: torch.device, chunk_size: int) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(image_shape, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(model(coords))
    return torch.cat(chunks, dim=0).reshape(*tuple(int(v) for v in image_shape), 3).contiguous()


def field_mse_metrics(pred_disp: torch.Tensor, target_disp: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    diff = pred_disp.detach() - target_disp.detach()
    mse_norm = float(diff.square().mean().detach().cpu())
    rmse_norm = math.sqrt(mse_norm)
    diff_vox = torch.from_numpy(c.dense_norm_xyz_to_vox_zyx(diff)).to(device=diff.device, dtype=diff.dtype)
    mse_vox = float(diff_vox.square().mean().detach().cpu())
    rmse_vox = math.sqrt(mse_vox)
    mask_flat = mask.detach().reshape(-1).bool()
    diff_masked = diff.reshape(-1, 3)[mask_flat]
    diff_vox_masked = diff_vox.reshape(-1, 3)[mask_flat]
    mse_masked_norm = float(diff_masked.square().mean().detach().cpu())
    mse_masked_vox = float(diff_vox_masked.square().mean().detach().cpu())
    return {
        "field_mse_full_normalized_grid": mse_norm,
        "field_rmse_full_normalized_grid": rmse_norm,
        "field_mse_full_voxel": mse_vox,
        "field_rmse_full_voxel": rmse_vox,
        "field_mse_masked_normalized_grid": mse_masked_norm,
        "field_rmse_masked_normalized_grid": math.sqrt(mse_masked_norm),
        "field_mse_masked_voxel": mse_masked_vox,
        "field_rmse_masked_voxel": math.sqrt(mse_masked_vox),
        "field_mse_objective_normalized_grid": mse_masked_norm,
        "field_rmse_objective_normalized_grid": math.sqrt(mse_masked_norm),
        "field_mse_objective_voxel": mse_masked_vox,
        "field_rmse_objective_voxel": math.sqrt(mse_masked_vox),
    }


def field_target_power_metrics(target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float | str]:
    """Return the masked pTVReg target power used by the paper rRMSE metric."""
    mask_flat = mask.detach().reshape(-1).bool()
    target_masked = target.detach().reshape(-1, 3)[mask_flat]
    if target_masked.numel() == 0:
        raise RuntimeError("Cannot compute target power on an empty fixed-T00 mask.")
    power = float(target_masked.square().mean().detach().cpu())
    return {
        "field_target_power_masked_normalized_grid": power,
        "field_target_rms_masked_normalized_grid": math.sqrt(power),
        "field_target_power_units": "normalized_grid_displacement_squared",
        "field_target_power_reduction": "component_and_masked_voxel_mean",
    }


def train_idir_field(args: argparse.Namespace, target: torch.Tensor, mask: torch.Tensor, device: torch.device) -> Dict[str, object]:
    image_shape = tuple(int(v) for v in target.shape[:3])
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    coords_all = c.make_coordinate_tensor(image_shape, device).detach()
    mask_flat = mask.reshape(-1).bool()
    coords_train = coords_all[mask_flat]
    target_train = target.reshape(-1, 3).detach()[mask_flat]
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    history: List[Dict[str, float]] = []
    if batch_size <= 0:
        raise RuntimeError("Empty fixed lung-mask field-fit domain.")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        pred = model(coords_train[indices])
        loss = F.mse_loss(pred, target_train[indices], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        row = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_sampled_normalized_grid": value,
            "coords_batch_size": float(batch_size),
            "train_coords": float(n_train),
            "lr": float(args.lr),
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dense = evaluate_idir_dense(model, image_shape, device, int(args.eval_chunk_size))
    return {
        "history": history,
        "dense_disp_norm_xyz": dense,
        "model": model,
        "training_elapsed_sec": time.time() - start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "train_coords": n_train,
        "coords_batch_size": batch_size,
        "param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def train_single_stage_field(
    args: argparse.Namespace,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    *,
    direct_cp: bool,
) -> Dict[str, object]:
    image_shape = tuple(int(v) for v in target.shape[:3])
    control_coords, control_shape = c.bspline_control_coords(image_shape, int(args.cps), device)
    mask_flat = mask.reshape(-1).bool()
    target_flat = target.reshape(-1, 3).detach()
    if direct_cp:
        params = torch.nn.Parameter(torch.zeros(control_coords.shape[0], 3, device=device, dtype=torch.float32))
        trainable = [params]
        model = None
    else:
        model = c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega).to(device)
        trainable = list(model.parameters())
        params = None
    optimizer = torch.optim.Adam(trainable, lr=float(args.lr))
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        if direct_cp:
            assert params is not None
            control = params
        else:
            assert model is not None
            control = model(control_coords)
        dense = c.expand_bspline_controls(control, control_shape, image_shape, int(args.cps))
        loss = F.mse_loss(dense.reshape(-1, 3)[mask_flat], target_flat[mask_flat], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        row = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_masked_normalized_grid": value,
            "train_coords": float(mask_flat.sum().detach().cpu()),
            "lr": float(args.lr),
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        if direct_cp:
            assert params is not None
            final_control = params.detach()
        else:
            assert model is not None
            final_control = model(control_coords).detach()
        final_dense = c.expand_bspline_controls(final_control, control_shape, image_shape, int(args.cps)).detach()
    param_count = int(params.numel()) if direct_cp and params is not None else int(sum(p.numel() for p in trainable if p.requires_grad))
    return {
        "history": history,
        "dense_disp_norm_xyz": final_dense,
        "model": model,
        "control_disp": final_control.reshape(*control_shape, 3).detach(),
        "control_shape": control_shape,
        "control_coords": control_coords.detach(),
        "training_elapsed_sec": time.time() - start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "train_coords": int(mask_flat.sum().detach().cpu()),
        "coords_batch_size": None,
        "param_count": param_count,
        "cps": int(args.cps),
    }


def resize_target_for_stage(target: torch.Tensor, eval_shape: Sequence[int]) -> torch.Tensor:
    tensor = target.permute(3, 0, 1, 2)[None]
    resized = F.interpolate(tensor, size=tuple(int(v) for v in eval_shape), mode="trilinear", align_corners=True)
    return resized[0].permute(1, 2, 3, 0).contiguous()


def resize_mask_for_stage(mask: torch.Tensor, eval_shape: Sequence[int]) -> torch.Tensor:
    tensor = mask.to(dtype=torch.float32)[None, None]
    resized = F.interpolate(tensor, size=tuple(int(v) for v in eval_shape), mode="nearest")
    return resized[0, 0] > 0.5


def finest_field_mse_diagnostic(
    control: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    image_shape: Sequence[int],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, object]:
    dense = c.expand_on_shape(control, control_shape, int(full_cps), image_shape, 1)
    diff = dense - target
    mask_flat = mask.reshape(-1).bool()
    diff_flat = diff.reshape(-1, 3)
    masked_mse = diff_flat[mask_flat].square().mean()
    full_mse = diff.square().mean()
    return {
        "finest_loss": float(masked_mse.detach().cpu()),
        "finest_data_loss": float(masked_mse.detach().cpu()),
        "finest_reg_loss": 0.0,
        "finest_field_mse_objective_normalized_grid": float(masked_mse.detach().cpu()),
        "finest_field_mse_full_normalized_grid": float(full_mse.detach().cpu()),
        "finest_loss_domain": "full_1mm_crop_fixed_t00_lung_mask",
    }


def prolong_control(
    prev_control: torch.Tensor,
    prev_shape: Sequence[int],
    prev_dense_full: torch.Tensor,
    target_shape: Sequence[int],
    full_cps: int,
    image_shape: Sequence[int],
    stage_idx: int,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    prev_grid = c.flat_to_grid(prev_control.detach(), prev_shape)
    exact_grid = c.cubic_bspline_dyadic_knot_insert_3d(prev_grid, target_shape)
    resize_grid = c.resize_control_grid(prev_grid, target_shape)
    rows: List[Dict[str, object]] = []
    for name, grid in (("exact_knot", exact_grid), ("trilinear_resize", resize_grid)):
        dense = c.expand_on_shape(c.grid_to_flat(grid), target_shape, int(full_cps), image_shape, 1)
        row: Dict[str, object] = {
            "stage_idx": int(stage_idx),
            "target_cps": int(full_cps),
            "method": name,
            "source_shape_zyx": list(int(v) for v in prev_shape),
            "target_shape_zyx": list(int(v) for v in target_shape),
        }
        row.update(c.field_difference_metrics(dense - prev_dense_full, prefix="preserve_"))
        rows.append(row)
    return c.grid_to_flat(exact_grid).detach(), rows


def train_multistage_field(args: argparse.Namespace, target: torch.Tensor, mask: torch.Tensor, device: torch.device) -> Dict[str, object]:
    image_shape = tuple(int(v) for v in target.shape[:3])
    stage_cps = c.parse_int_list(args.stage_full_cps)
    stage_downsample = c.parse_stage_ints(args.stage_downsample, len(stage_cps))
    stage_epochs = c.parse_stage_ints(args.stage_epochs, len(stage_cps))
    stage_lr = c.parse_float_list(args.stage_lr, len(stage_cps), float(args.lr))
    nested_shapes = c.build_nested_cp_shapes(image_shape, base_spacing=int(stage_cps[0]), spacings=stage_cps)
    c.assert_nested_schedule(nested_shapes, stage_cps)
    history: List[Dict[str, object]] = []
    stage_summaries: List[Dict[str, object]] = []
    prolong_rows: List[Dict[str, object]] = []
    prev_control: Optional[torch.Tensor] = None
    prev_shape: Optional[Tuple[int, int, int]] = None
    prev_dense_full: Optional[torch.Tensor] = None
    final_control: Optional[torch.Tensor] = None
    final_shape: Optional[Tuple[int, int, int]] = None
    final_dense: Optional[torch.Tensor] = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    total_start = time.time()
    for stage_idx, (full_cps, downsample, epochs, lr) in enumerate(zip(stage_cps, stage_downsample, stage_epochs, stage_lr)):
        control_shape = tuple(int(v) for v in nested_shapes[int(full_cps)])
        control = torch.nn.Parameter(torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32))
        if prev_control is not None and prev_shape is not None and prev_dense_full is not None:
            init_flat, rows = prolong_control(prev_control, prev_shape, prev_dense_full, control_shape, int(full_cps), image_shape, int(stage_idx))
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        eval_shape = c.downsample_shape(image_shape, int(downsample))
        target_l = resize_target_for_stage(target, eval_shape).detach()
        mask_l = resize_mask_for_stage(mask, eval_shape).reshape(-1)
        target_flat = target_l.reshape(-1, 3)
        optimizer = torch.optim.Adam([control], lr=float(lr))
        stage_start = time.time()
        for epoch in range(1, int(epochs) + 1):
            dense_l = c.expand_on_shape(control, control_shape, int(full_cps), eval_shape, int(downsample))
            loss = F.mse_loss(dense_l.reshape(-1, 3)[mask_l], target_flat[mask_l], reduction="mean")
            finest_diag: Dict[str, object] = {}
            if bool(args.finest_mse_diagnostic):
                should_eval_finest = (
                    int(downsample) == 1
                    or epoch == 1
                    or epoch % int(args.finest_mse_every) == 0
                    or epoch == int(epochs)
                )
                if should_eval_finest:
                    if int(downsample) == 1:
                        diff_l = dense_l - target_l
                        full_mse_l = diff_l.square().mean()
                        finest_value = float(loss.detach().cpu())
                        finest_diag = {
                            "finest_loss": finest_value,
                            "finest_data_loss": finest_value,
                            "finest_reg_loss": 0.0,
                            "finest_field_mse_objective_normalized_grid": finest_value,
                            "finest_field_mse_full_normalized_grid": float(full_mse_l.detach().cpu()),
                            "finest_loss_domain": "full_1mm_crop_fixed_t00_lung_mask",
                        }
                    else:
                        with torch.no_grad():
                            finest_diag = finest_field_mse_diagnostic(
                                control,
                                control_shape,
                                int(full_cps),
                                image_shape,
                                target,
                                mask,
                            )
                else:
                    finest_diag = {
                        "finest_loss": float("nan"),
                        "finest_data_loss": float("nan"),
                        "finest_reg_loss": 0.0,
                        "finest_field_mse_objective_normalized_grid": float("nan"),
                        "finest_field_mse_full_normalized_grid": float("nan"),
                        "finest_loss_domain": "full_1mm_crop_fixed_t00_lung_mask",
                    }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            row = {
                "stage_idx": float(stage_idx),
                "epoch": float(epoch),
                "global_epoch": float(sum(stage_epochs[:stage_idx]) + epoch),
                "full_cps": float(full_cps),
                "downsample": float(downsample),
                "loss": value,
                "data_loss": value,
                "reg_loss": 0.0,
                "field_mse_objective_normalized_grid": value,
                "train_coords": float(mask_l.sum().detach().cpu()),
                "lr": float(lr),
            }
            row.update(finest_diag)
            history.append(row)
            if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
                print(json.dumps(row), flush=True)
        with torch.no_grad():
            dense_full = c.expand_on_shape(control, control_shape, int(full_cps), image_shape, 1).detach()
        summary = {
            "stage_idx": int(stage_idx),
            "full_cps": int(full_cps),
            "downsample": int(downsample),
            "control_shape_zyx": list(int(v) for v in control_shape),
            "eval_shape_zyx": list(int(v) for v in eval_shape),
            "epochs": int(epochs),
            "lr": float(lr),
            "loss_mode": "fixed_t00_lung_mask_ptvreg_field_mse",
            "elapsed_sec": float(time.time() - stage_start),
        }
        stage_summaries.append(summary)
        prev_control = control.detach()
        prev_shape = control_shape
        prev_dense_full = dense_full
        final_control = control.detach()
        final_shape = control_shape
        final_dense = dense_full
    if device.type == "cuda":
        torch.cuda.synchronize()
    if final_dense is None or final_control is None or final_shape is None:
        raise RuntimeError("No multistage field-fit stages were run.")
    return {
        "history": history,
        "dense_disp_norm_xyz": final_dense,
        "control_disp": final_control.reshape(*final_shape, 3).detach(),
        "control_shape": final_shape,
        "stage_summaries": stage_summaries,
        "prolongation_diagnostics": prolong_rows,
        "training_elapsed_sec": time.time() - total_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "train_coords": int(mask.sum().detach().cpu()),
        "coords_batch_size": None,
        "param_count": int(final_control.numel()),
        "cps": int(stage_cps[-1]),
        "stage_full_cps": stage_cps,
        "stage_downsample": stage_downsample,
        "stage_epochs": stage_epochs,
        "stage_lr": stage_lr,
    }


def write_outputs(
    args: argparse.Namespace,
    case: c.LungCTCase,
    target_audit: Dict[str, object],
    target: torch.Tensor,
    mask: torch.Tensor,
    result: Dict[str, object],
    run_dir: Path,
    run_name: str,
    started_at: str,
    total_elapsed_sec: float,
) -> None:
    dense = result["dense_disp_norm_xyz"]
    assert isinstance(dense, torch.Tensor)
    metrics = c.final_metrics_from_dense(dense, case, run_dir, bool(args.save_dense_dvf), args.log_base)
    field_metrics = field_mse_metrics(dense, target, mask)
    target_power_metrics = field_target_power_metrics(target, mask)
    history = result["history"]
    assert isinstance(history, list)
    c.write_csv(run_dir / ("stage_history.csv" if args.method == "direct_cp_4stage" else "history.csv"), history)
    if args.method == "direct_cp_4stage":
        c.write_csv(run_dir / "prolongation_metrics.csv", result.get("prolongation_diagnostics", []))  # type: ignore[arg-type]
    c.write_json(run_dir / "target_audit.json", target_audit)
    state: Dict[str, object] = {
        "args": vars(args),
        "run_name": run_name,
        "training_elapsed_sec": result["training_elapsed_sec"],
        "training_peak_mem_gb": result["training_peak_mem_gb"],
    }
    model = result.get("model")
    if isinstance(model, torch.nn.Module):
        state["model_state_dict"] = model.state_dict()
    if isinstance(result.get("control_disp"), torch.Tensor):
        state["control_disp"] = result["control_disp"].detach().cpu()  # type: ignore[union-attr]
        state["control_shape"] = result.get("control_shape")
    torch.save(state, run_dir / "model_state.pt")
    if isinstance(result.get("control_disp"), torch.Tensor):
        torch.save(
            {
                "control_disp": result["control_disp"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": result.get("control_shape"),
                "cps": result.get("cps"),
                "image_shape": case.shape_zyx,
                "objective": "ptvreg_field_fit_mse",
                "field_fit_target_units": "normalized_grid",
                "method_key": method_key(args.method),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )
    last = history[-1] if history else {}
    summary: Dict[str, object] = {
        "run_name": run_name,
        "status": "completed",
        "variant": variant_for_case(int(args.case_id)),
        "objective": "ptvreg_field_fit_mse",
        "method": args.method,
        "method_key": method_key(args.method),
        "method_label": method_label(args.method),
        "dataset": "4dct",
        "case_id": int(args.case_id),
        "fixed_phase": "T00",
        "moving_phase": "T50",
        "image_domain": "pTVReg landmark crop",
        "spacing": "isotropic_1mm",
        "data_root": str(args.data_root),
        "case_dir": str(case.case_dir),
        "out_dir": str(run_dir),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "eval_interval": int(args.eval_interval),
        "seed": int(args.seed),
        "activation": "direct_control_points" if args.method.startswith("direct_cp") else "Siren",
        "omega": float(args.idir_omega if args.method == "idir" else args.spline_omega),
        "regularizer": "none_mse_only",
        "image_loss": "none_training_posthoc_metrics_only",
        "sample_domain": "fixed_t00_lung_mask_reg1mm",
        "foreground_metric_mask": "fixed_t00_lung_mask_reg1mm",
        "mask_rule": "fixed_t00_lung_mask_field_fit_and_metric",
        "loss_domain": "fixed_t00_lung_mask_reg1mm",
        "protocol_tag": "ptvreg_field_fit+ptv_landmark_crop_1mm_fixed_t00_lung_mask_mse",
        "fixed_lung_mask_voxels": int(case.fixed_mask.sum()),
        "fixed_lung_mask_fraction": float(case.fixed_mask.mean()),
        "preprocess_audit": case.preprocess_audit,
        "training": {
            "optimizer": "Adam",
            "loss": "MSE to reconstructed pTVReg displacement",
            "regularizer": "none",
            "mse_domain": "fixed_t00_lung_mask_reg1mm",
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "stage_full_cps": result.get("stage_full_cps"),
            "stage_downsample": result.get("stage_downsample"),
            "stage_epochs": result.get("stage_epochs"),
            "stage_lr": result.get("stage_lr"),
            "coords_batch_size": result.get("coords_batch_size"),
            "train_coords": result.get("train_coords"),
            "param_count": result.get("param_count"),
            "finest_mse_diagnostic": bool(args.finest_mse_diagnostic),
            "finest_mse_every": int(args.finest_mse_every) if bool(args.finest_mse_diagnostic) else None,
            "finest_mse_domain": "full_1mm_crop_fixed_t00_lung_mask" if bool(args.finest_mse_diagnostic) else None,
            "finest_mse_affects_training": False if bool(args.finest_mse_diagnostic) else None,
        },
        "started_at": started_at,
        "completed_at": c.now_text(),
        "elapsed_sec": float(result["training_elapsed_sec"]),
        "peak_mem_gb": float(result["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": float(last.get("loss", float("nan"))),
        "final_data_loss": float(last.get("data_loss", float("nan"))),
        "final_reg_loss": float(last.get("reg_loss", float("nan"))),
        "history_tail": history[-10:],
        "notes": [
            "Field fitting optimizes MSE only over the fixed T00 lung mask in the 1mm crop.",
            "Full-crop MSE is diagnostic only.",
            "Target orientation is validated against saved pTVReg snap TRE before training.",
            "Foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm grid.",
        ],
    }
    summary.update(target_audit)
    summary.update(field_metrics)
    summary.update(target_power_metrics)
    target_power = float(target_power_metrics["field_target_power_masked_normalized_grid"])
    summary["field_rrmse_objective_percent"] = 100.0 * math.sqrt(
        float(field_metrics["field_mse_objective_normalized_grid"]) / (target_power + 1e-12)
    )
    summary.update(metrics)
    summary.update(
        {
            k: v
            for k, v in result.items()
            if k in ("stage_summaries", "prolongation_diagnostics")
        }
    )
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(
        json.dumps(
            {
                "summary_path": str(run_dir / "summary.json"),
                "method": args.method,
                "case_id": int(args.case_id),
                "field_mse_objective_normalized_grid": summary["field_mse_objective_normalized_grid"],
                "target_source_tre": summary["source_recomputed_snap_tre_mean_mm"],
                "elapsed_sec": summary["elapsed_sec"],
                "peak_mem_gb": summary["peak_mem_gb"],
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args)
    run_dir = Path(args.out_root) / method_key(args.method) / run_name
    if (run_dir / "DONE").exists() and not args.overwrite:
        print(json.dumps({"skip_existing_done": str(run_dir)}), flush=True)
        return

    total_start = time.time()
    started_at = c.now_text()
    case = c.load_case(args.data_root, int(args.case_id))
    target_np, target_audit = load_ptvreg_target(args, case)
    if args.validate_target_only:
        print(json.dumps({"case_id": int(args.case_id), "target_audit": target_audit}, indent=2), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    device = c.set_cuda_device(args.gpu)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "objective": "ptvreg_field_fit_mse",
                "method": args.method,
                "method_key": method_key(args.method),
                "gpu": args.gpu,
                "case_id": int(args.case_id),
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "stage_epochs": args.stage_epochs if args.method == "direct_cp_4stage" else None,
                "target_tre_validation": target_audit["source_recomputed_snap_tre_mean_mm"],
            },
            allow_nan=True,
        ),
        flush=True,
    )

    if args.method == "idir":
        result = train_idir_field(args, target, mask, device)
    elif args.method == "sinr_cps4":
        result = train_single_stage_field(args, target, mask, device, direct_cp=False)
    elif args.method == "direct_cp_cps4":
        result = train_single_stage_field(args, target, mask, device, direct_cp=True)
    elif args.method == "direct_cp_4stage":
        result = train_multistage_field(args, target, mask, device)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    write_outputs(args, case, target_audit, target, mask, result, run_dir, run_name, started_at, time.time() - total_start)


if __name__ == "__main__":
    main()
