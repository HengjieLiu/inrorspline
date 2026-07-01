#!/usr/bin/env python
"""Run one LungCT no-sweep reproduction job."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import common_dctfield as dctfield  # noqa: E402
import common as c


METHOD_IDIR = "idir_be_default_lungmask"
METHOD_IDIR_SAMPLED_PTV_TV = "idir_sampled_ptv_lcc_tv"
METHOD_SINR_CPS4_PTV = "sinr_cps4_ptv_lcc_isotv"
METHOD_DIRECT_CP_CPS4_PTV = "direct_cp_cps4_ptv_lcc_isotv"
METHOD_SINR_CPS4_GLOBAL_NCC_FD_BE = "sinr_cps4_global_ncc_fd_be"
METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE = "direct_cp_cps4_global_ncc_fd_be"
METHOD_DIRECT_CP_32_PTV = "direct_cp_32_16_8_4_ptv_lcc_isotv"
METHOD_DIRECT_CP_32_GLOBAL_NCC_BE = "direct_cp_32_16_8_4_global_ncc_bspline_be"
METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE = "direct_cp_32_16_8_4_global_ncc_fd_be"
METHOD_DIRECT_DVF_FD_BE = "direct_dvf_lungmask_sampled_ncc_fd_be"
METHOD_DCTFIELD_FD_BE = "dctfield_lungmask_sampled_ncc_fd_be"
METHODS = [
    METHOD_IDIR,
    METHOD_IDIR_SAMPLED_PTV_TV,
    METHOD_SINR_CPS4_PTV,
    METHOD_DIRECT_CP_CPS4_PTV,
    METHOD_SINR_CPS4_GLOBAL_NCC_FD_BE,
    METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE,
    METHOD_DIRECT_CP_32_PTV,
    METHOD_DIRECT_CP_32_GLOBAL_NCC_BE,
    METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE,
    METHOD_DIRECT_DVF_FD_BE,
    METHOD_DCTFIELD_FD_BE,
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
    parser.add_argument("--image-sim-eval-interval", type=int, default=0)
    parser.add_argument("--image-sim-eval-chunk-size", type=int, default=None)
    parser.add_argument("--global-ncc-diagnostic-interval", type=int, default=0)
    parser.add_argument("--global-ncc-diagnostic-chunk-size", type=int, default=None)
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
    parser.add_argument("--dct-size", type=int, default=32)

    parser.add_argument("--border-mask-voxels", type=int, default=5)
    parser.add_argument("--ptv-lcc-sigma-mm", type=float, default=2.1)
    parser.add_argument("--ptv-lcc-radius-sigma", type=float, default=2.0)
    parser.add_argument("--ptv-lcc-sigma-floor-pix", type=float, default=0.8)
    parser.add_argument("--ptv-isotv-weight", type=float, default=0.11)
    parser.add_argument("--ptv-csqrt", type=float, default=5e-3)
    parser.add_argument("--sampled-lcc-centers", type=int, default=512)
    parser.add_argument("--sampled-lcc-offsets", type=int, default=64)

    parser.add_argument("--stage-full-cps", type=str, default="32,16,8,4")
    parser.add_argument("--stage-downsample", type=str, default="8,4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="200,400,800,1000")
    parser.add_argument("--stage-lr", type=str, default="1e-3")
    parser.add_argument("--antialias-sigma-scale", type=float, default=0.5)
    parser.add_argument("--antialias-radius-sigma", type=float, default=3.0)
    parser.add_argument("--finest-loss-diagnostic", action="store_true")
    parser.add_argument("--finest-loss-every", type=int, default=1)
    parser.add_argument("--sample-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--checkpoint-epochs", type=str, default="0,1,10,25,50,100,250,500,1000,1500,2000,2500")
    args = parser.parse_args()
    if int(args.finest_loss_every) < 1:
        raise ValueError("--finest-loss-every must be >= 1")
    return args


class CheckpointSampler:
    def __init__(self, *, args: argparse.Namespace, case: c.LungCTCase, device: torch.device, out_dir: Path) -> None:
        self.enabled = bool(args.sample_checkpoints)
        self.out_dir = Path(out_dir)
        self.rows: List[Dict[str, object]] = []
        self.disp_norm_xyz: List[np.ndarray] = []
        self.disp_vox_xyz: List[np.ndarray] = []
        self.epochs = {
            int(token)
            for token in str(args.checkpoint_epochs).replace(" ", "").split(",")
            if token
        }
        self.shape = tuple(int(v) for v in case.shape_zyx)
        self.device = device
        self.flat_indices_np = np.asarray([], dtype=np.int64)
        self.flat_indices_t = torch.empty(0, dtype=torch.long, device=device)
        self.points_zyx = np.zeros((0, 3), dtype=np.int64)
        self.points_norm_xyz = np.zeros((0, 3), dtype=np.float32)
        if not self.enabled:
            return
        eligible = np.flatnonzero(case.fixed_mask.reshape(-1).astype(bool))
        if eligible.size == 0:
            raise RuntimeError("Cannot sample checkpoints from an empty LungCT fixed T00 mask.")
        count = min(int(args.checkpoint_sample_count), int(eligible.size))
        rng = np.random.default_rng(int(args.checkpoint_sample_seed))
        chosen = eligible if count >= eligible.size else rng.choice(eligible, size=count, replace=False)
        self.flat_indices_np = np.sort(chosen.astype(np.int64, copy=False))
        self.flat_indices_t = torch.from_numpy(self.flat_indices_np).to(device=device, dtype=torch.long)
        z, y, x = np.unravel_index(self.flat_indices_np, self.shape)
        self.points_zyx = np.stack((z, y, x), axis=1).astype(np.int64, copy=False)
        d, h, w = self.shape
        self.points_norm_xyz = np.stack(
            (
                2.0 * x.astype(np.float32) / max(float(w - 1), 1.0) - 1.0,
                2.0 * y.astype(np.float32) / max(float(h - 1), 1.0) - 1.0,
                2.0 * z.astype(np.float32) / max(float(d - 1), 1.0) - 1.0,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        self.coords_t = torch.from_numpy(self.points_norm_xyz).to(device=device, dtype=torch.float32)
        self.scale_xyz = torch.tensor(
            [(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0],
            dtype=torch.float32,
            device=device,
        )
        c.write_json(
            self.out_dir / "checkpoint_sample_metadata.json",
            {
                "enabled": True,
                "dataset": "lungct",
                "case_id": int(case.case_id),
                "sample_seed": int(args.checkpoint_sample_seed),
                "requested_sample_count": int(args.checkpoint_sample_count),
                "actual_sample_count": int(self.flat_indices_np.size),
                "sample_domain": "fixed_t00_lung_mask_reg1mm",
                "field_units_saved": ["normalized_grid_xyz", "reg1mm_voxel_xyz"],
                "field_direction": "fixed_T00_to_moving_T50",
                "component_order": "x,y,z",
                "grid_shape_zyx": list(self.shape),
                "checkpoint_epochs": sorted(self.epochs),
            },
        )

    def should_record(self, epoch: int) -> bool:
        return self.enabled and int(epoch) in self.epochs

    def record_samples(
        self,
        *,
        epoch: int,
        disp_norm_xyz: torch.Tensor,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        sample_norm = disp_norm_xyz.detach().to(dtype=torch.float32)
        sample_vox = sample_norm * self.scale_xyz.view(1, 3)
        self.disp_norm_xyz.append(sample_norm.cpu().numpy().astype(np.float32, copy=False))
        self.disp_vox_xyz.append(sample_vox.cpu().numpy().astype(np.float32, copy=False))
        mag = torch.linalg.norm(sample_vox, dim=1)
        row = {
            "checkpoint_index": len(self.rows),
            "epoch": int(epoch),
            "sample_count": int(sample_norm.shape[0]),
            "norm_abs_max": float(sample_norm.abs().max().detach().cpu()),
            "voxel_mag_mean": float(mag.mean().detach().cpu()),
            "voxel_mag_p95": float(torch.quantile(mag, 0.95).detach().cpu()),
        }
        if row_extra:
            row.update(row_extra)
        self.rows.append(row)

    def record_idir(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        model.eval()
        with torch.no_grad():
            disp = model(self.coords_t)
        self.record_samples(epoch=epoch, disp_norm_xyz=disp, row_extra=row_extra)

    def record_dense(
        self,
        *,
        epoch: int,
        dense_disp_norm_xyz: torch.Tensor,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        disp = dense_disp_norm_xyz.reshape(-1, 3)[self.flat_indices_t]
        self.record_samples(epoch=epoch, disp_norm_xyz=disp, row_extra=row_extra)

    def save(self) -> None:
        if not self.enabled:
            return
        c.write_csv(self.out_dir / "checkpoint_sample_rows.csv", self.rows)
        if self.rows:
            np.savez_compressed(
                self.out_dir / "checkpoint_samples.npz",
                epochs=np.asarray([int(row["epoch"]) for row in self.rows], dtype=np.int64),
                flat_indices=self.flat_indices_np,
                points_zyx=self.points_zyx,
                points_norm_xyz=self.points_norm_xyz,
                disp_norm_xyz=np.stack(self.disp_norm_xyz, axis=0),
                disp_vox_xyz=np.stack(self.disp_vox_xyz, axis=0),
            )


def default_epochs(method: str) -> int:
    if method in {METHOD_DIRECT_CP_32_PTV, METHOD_DIRECT_CP_32_GLOBAL_NCC_BE, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE}:
        raise ValueError("Multistage method uses --stage-epochs, not --epochs.")
    return 2500


def default_lr(method: str) -> float:
    if method in {
        METHOD_DIRECT_CP_CPS4_PTV,
        METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE,
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
    if method == METHOD_DCTFIELD_FD_BE:
        return (
            f"{method}_case{args.case_id:02d}_k{int(args.dct_size)}_e{epochs}"
            f"_lr{c.format_float(lr)}_bend{c.format_float(args.alpha_bending)}_seed{args.seed}"
        )
    if method in {METHOD_SINR_CPS4_GLOBAL_NCC_FD_BE, METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE}:
        return (
            f"{method}_case{args.case_id:02d}_e{epochs}"
            f"_lr{c.format_float(lr)}_a{c.format_float(args.alpha)}_seed{args.seed}"
        )
    extra = (
        f"bend{c.format_float(args.alpha_bending)}"
        if method in {METHOD_IDIR, METHOD_DIRECT_DVF_FD_BE}
        else f"w{c.format_float(args.ptv_isotv_weight)}"
    )
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


def should_image_sim_eval(args: argparse.Namespace, epoch: int, epochs: int) -> bool:
    interval = int(args.image_sim_eval_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(epochs)


def should_global_ncc_diagnostic(args: argparse.Namespace, epoch: int, epochs: int) -> bool:
    interval = int(args.global_ncc_diagnostic_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(epochs)


def global_ncc_diagnostic_row(
    *,
    fixed_t: torch.Tensor,
    moving_t: torch.Tensor,
    dense_disp: torch.Tensor,
    id_grid: torch.Tensor,
    epoch: int,
    global_epoch: int,
    eval_domain: str,
    device: torch.device,
) -> Dict[str, float | str | bool]:
    eval_start = time.time()
    with torch.no_grad():
        warped = c.warp_full(moving_t, id_grid + dense_disp, mode="bilinear")
        loss = c.ncc_loss(fixed_t[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    del warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(global_epoch),
        "global_ncc_loss": value,
        "global_ncc_value": -value,
        "diagnostic_name": "full_crop_global_ncc",
        "eval_domain": eval_domain,
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
    }


def evaluate_idir_ptv_lcc(
    args: argparse.Namespace,
    model: torch.nn.Module,
    fixed_t: torch.Tensor,
    moving_t: torch.Tensor,
    id_dense: torch.Tensor,
    border_mask: torch.Tensor,
    ptv_cache: Dict[str, torch.Tensor | List[float]],
    image_shape: Sequence[int],
    device: torch.device,
    chunk_size: int,
    epoch: int,
) -> Dict[str, float | str | bool]:
    eval_start = time.time()
    model.eval()
    with torch.no_grad():
        dense = evaluate_idir_dense(model, image_shape, device, chunk_size)
        warped = c.warp_full(moving_t, id_dense + dense, mode="bilinear")
        loss = c.ptv_local_cc_loss_with_cache(
            warped,
            border_mask,
            ptv_cache,
            (1.0, 1.0, 1.0),
            args.ptv_lcc_radius_sigma,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    del dense, warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(epoch),
        "similarity_loss": value,
        "similarity_value": -value,
        "similarity_name": "ptv_lcc_data_loss",
        "eval_domain": "border_masked_crop",
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
    }


def evaluate_idir_global_ncc(
    model: torch.nn.Module,
    fixed_t: torch.Tensor,
    moving_t: torch.Tensor,
    id_dense: torch.Tensor,
    image_shape: Sequence[int],
    device: torch.device,
    chunk_size: int,
    epoch: int,
) -> Dict[str, float | str | bool]:
    eval_start = time.time()
    model.eval()
    with torch.no_grad():
        dense = evaluate_idir_dense(model, image_shape, device, chunk_size)
        warped = c.warp_full(moving_t, id_dense + dense, mode="bilinear")
        loss = c.ncc_loss(fixed_t[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    del dense, warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(epoch),
        "global_ncc_loss": value,
        "global_ncc_value": -value,
        "diagnostic_name": "full_crop_global_ncc",
        "eval_domain": "full_1mm_crop",
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
    }


def train_idir(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    epochs = int(args.epochs if args.epochs is not None else 2500)
    lr = float(args.lr if args.lr is not None else 1e-4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
    image_shape = tuple(int(v) for v in case.shape_zyx)
    coords_all = c.make_coordinate_tensor(case.shape_zyx, device)
    coords_train = coords_all[mask_t.reshape(-1)].contiguous()
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty IDIR training coordinate set.")

    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if sampler is not None:
        sampler.record_idir(epoch=0, model=model, row_extra={"method": args.method, "checkpoint_source": "model_init"})
    history: List[Dict[str, float]] = []
    image_similarity_eval: List[Dict[str, float | str | bool]] = []
    image_sim_chunk_size = int(args.image_sim_eval_chunk_size or args.eval_chunk_size)
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    global_ncc_chunk_size = int(args.global_ncc_diagnostic_chunk_size or args.eval_chunk_size)
    id_dense = c.make_coordinate_grid(image_shape, device).detach()
    border_mask = c.make_border_mask(image_shape, int(args.border_mask_voxels), device, fixed_t.dtype)
    ptv_cache = c.make_ptv_lcc_fixed_cache(
        fixed_t,
        (1.0, 1.0, 1.0),
        args.ptv_lcc_sigma_mm,
        args.ptv_lcc_radius_sigma,
        args.ptv_lcc_sigma_floor_pix,
    )
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
        if should_image_sim_eval(args, epoch, epochs):
            sim_row = evaluate_idir_ptv_lcc(
                args,
                model,
                fixed_t,
                moving_t,
                id_dense,
                border_mask,
                ptv_cache,
                image_shape,
                device,
                image_sim_chunk_size,
                epoch,
            )
            image_similarity_eval.append(sim_row)
            print(json.dumps({"image_similarity_eval": sim_row}), flush=True)
        if should_global_ncc_diagnostic(args, epoch, epochs):
            diag_row = evaluate_idir_global_ncc(
                model,
                fixed_t,
                moving_t,
                id_dense,
                image_shape,
                device,
                global_ncc_chunk_size,
                epoch,
            )
            global_ncc_diagnostic.append(diag_row)
            print(json.dumps({"global_ncc_diagnostic": diag_row}), flush=True)
        if sampler is not None:
            sampler.record_idir(
                epoch=epoch,
                model=model,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    dense = evaluate_idir_dense(model, case.shape_zyx, device, args.eval_chunk_size)
    c.write_csv(out_dir / "loss_history.csv", history)
    if image_similarity_eval:
        c.write_csv(out_dir / "image_similarity_eval.csv", image_similarity_eval)
    if global_ncc_diagnostic:
        c.write_csv(out_dir / "global_ncc_diagnostic.csv", global_ncc_diagnostic)
    return {
        "dense_disp_norm_xyz": dense,
        "history_tail": history[-10:],
        "image_similarity_eval_tail": image_similarity_eval[-10:],
        "global_ncc_diagnostic_tail": global_ncc_diagnostic[-10:],
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
            "image_similarity_eval_interval": int(args.image_sim_eval_interval),
            "image_similarity_eval_name": "ptv_lcc_data_loss" if int(args.image_sim_eval_interval) > 0 else None,
            "image_similarity_eval_affects_training": False if int(args.image_sim_eval_interval) > 0 else None,
            "global_ncc_diagnostic_interval": int(args.global_ncc_diagnostic_interval),
            "global_ncc_diagnostic_name": (
                "full_crop_global_ncc"
                if int(args.global_ncc_diagnostic_interval) > 0
                else None
            ),
            "global_ncc_diagnostic_affects_training": (
                False if int(args.global_ncc_diagnostic_interval) > 0 else None
            ),
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def train_direct_dvf(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    epochs = int(args.epochs if args.epochs is not None else 2500)
    lr = float(args.lr if args.lr is not None else 1e-4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
    image_shape = tuple(int(v) for v in case.shape_zyx)
    coords_all = c.make_coordinate_tensor(image_shape, device)
    train_indices = torch.nonzero(mask_t.reshape(-1), as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty direct-DVF training coordinate set.")

    dense_disp = torch.nn.Parameter(torch.zeros((*image_shape, 3), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([dense_disp], lr=lr)
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=dense_disp.detach(),
            row_extra={"method": args.method, "checkpoint_source": "dense_dvf_init"},
        )
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(1, epochs + 1):
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving_t, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_t, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.dense_forward_difference_be(dense_disp, mask_t, component_reduction="mean")
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
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    dense = dense_disp.detach()
    c.write_csv(out_dir / "loss_history.csv", history)
    return {
        "dense_disp_norm_xyz": dense,
        "history_tail": history[-10:],
        "training": {
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "loss": "sampled NCC + dense forward-difference BE",
            "image_loss": "idir_global_ncc_sampled",
            "regularizer": "dense_forward_difference_be",
            "alpha_bending": float(args.alpha_bending),
            "batch_size": batch_size,
            "train_coords": n_train,
            "sample_domain": "fixed_t00_lung_mask_reg1mm_grid_points",
            "be_type": "dense_forward_difference_be",
            "native_analytic_be_formula": "not_available_dense_grid_dvf",
            "seed": int(args.seed),
            "network": {
                "type": "direct_dense_grid_displacement",
                "layers": None,
                "omega": None,
                "trainable_parameters": int(dense_disp.numel()),
            },
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def train_dctfield(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    epochs = int(args.epochs if args.epochs is not None else 2500)
    lr = float(args.lr if args.lr is not None else 1e-4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(case.fixed_mask.astype(np.bool_)).to(device=device)
    image_shape = tuple(int(v) for v in case.shape_zyx)
    dct_size = dctfield.normalize_dct_size(int(args.dct_size), image_shape)
    decoder = dctfield.DCTFieldDecoder(image_shape, dct_size, device=device, dtype=torch.float32)
    coeff = torch.nn.Parameter(torch.zeros((3, *dct_size), dtype=torch.float32, device=device))
    coords_all = c.make_coordinate_tensor(image_shape, device)
    train_indices = torch.nonzero(mask_t.reshape(-1), as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty DCTField training coordinate set.")

    optimizer = torch.optim.Adam([coeff], lr=lr)
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*image_shape, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "dct_coeff_init"},
        )
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(1, epochs + 1):
        dense_disp = decoder.decode(coeff)
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving_t, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_t, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.dense_forward_difference_be(dense_disp, mask_t, component_reduction="mean")
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
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    with torch.no_grad():
        dense = decoder.decode(coeff).detach()
    c.write_csv(out_dir / "loss_history.csv", history)
    torch.save(
        {
            "dct_coeff": coeff.detach().cpu(),
            "dct_size_zyx": list(dct_size),
            "dct_basis": decoder.basis_name,
            "image_shape_zyx": list(image_shape),
            "method": str(args.method),
            "alpha_bending": float(args.alpha_bending),
            "lr": float(lr),
        },
        out_dir / "dct_coefficients.pt",
    )
    return {
        "dense_disp_norm_xyz": dense,
        "history_tail": history[-10:],
        "training": {
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "loss": "sampled NCC + DCT-decoded dense forward-difference BE",
            "image_loss": "idir_global_ncc_sampled",
            "regularizer": "dct_decoded_dense_forward_difference_be",
            "alpha_bending": float(args.alpha_bending),
            "batch_size": batch_size,
            "train_coords": n_train,
            "sample_domain": "fixed_t00_lung_mask_reg1mm_grid_points",
            "be_type": "dct_decoded_dense_forward_difference_be",
            "native_analytic_be_formula": "not_used_dct_spectral_be_possible",
            "dct_size_zyx": list(dct_size),
            "dct_basis": decoder.basis_name,
            "seed": int(args.seed),
            "network": {
                "type": "dctfield_coefficients",
                "layers": None,
                "omega": None,
                "trainable_parameters": int(coeff.numel()),
            },
        },
        "elapsed_sec": elapsed,
        "peak_mem_gb": peak,
    }


def train_idir_sampled_ptv_tv(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, object]:
    epochs = int(args.epochs if args.epochs is not None else 2500)
    lr = float(args.lr if args.lr is not None else 1e-4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_t = torch.from_numpy(case.fixed).to(device=device, dtype=torch.float32)
    moving_t = torch.from_numpy(case.moving).to(device=device, dtype=torch.float32)
    image_shape = tuple(int(v) for v in case.shape_zyx)
    coords_all = c.make_coordinate_tensor(image_shape, device)
    border_mask = c.make_border_mask(image_shape, int(args.border_mask_voxels), device, fixed_t.dtype).bool()
    coords_train = coords_all[border_mask.reshape(-1)].contiguous()
    n_train = int(coords_train.shape[0])
    n_centers = min(int(args.sampled_lcc_centers), n_train)
    if n_centers <= 0:
        raise RuntimeError("Empty IDIR sampled pTV-LCC training coordinate set.")
    offsets_norm_xyz, offset_weights, offset_meta = c.make_sampled_lcc_offsets(
        image_shape,
        (1.0, 1.0, 1.0),
        args.ptv_lcc_sigma_mm,
        args.ptv_lcc_radius_sigma,
        args.ptv_lcc_sigma_floor_pix,
        int(args.sampled_lcc_offsets),
        device,
        fixed_t.dtype,
    )
    max_offset = max(float(v) for v in offset_meta["max_abs_offset_zyx"])
    if max_offset > float(args.border_mask_voxels):
        raise ValueError(
            f"sampled LCC max offset {max_offset:g} exceeds border mask {int(args.border_mask_voxels)}; "
            "increase --border-mask-voxels or reduce --sampled-lcc-offsets/radius."
        )

    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randint(n_train, (n_centers,), device=device)
        center_coords = coords_train[indices].detach()
        patch_coords = center_coords[:, None, :] + offsets_norm_xyz[None, :, :]
        patch_flat = patch_coords.reshape(-1, 3)
        patch_disp = model(patch_flat).reshape(n_centers, int(offsets_norm_xyz.shape[0]), 3)
        data_loss = c.sampled_ptv_local_cc_loss(
            fixed_t,
            moving_t,
            patch_coords,
            patch_disp,
            offset_weights,
            n_train,
            (1.0, 1.0, 1.0),
        )
        tv_coords = center_coords.detach().requires_grad_(True)
        tv_disp = model(tv_coords)
        reg_loss = c.idir_sampled_tv_physical(
            tv_coords,
            tv_disp,
            image_shape,
            (1.0, 1.0, 1.0),
            args.ptv_csqrt,
            domain_voxels=n_train,
        )
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
            "sampled_lcc_centers": float(n_centers),
            "sampled_lcc_offsets": float(offsets_norm_xyz.shape[0]),
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    dense = evaluate_idir_dense(model, image_shape, device, args.eval_chunk_size)
    c.write_csv(out_dir / "loss_history.csv", history)
    return {
        "dense_disp_norm_xyz": dense,
        "history_tail": history[-10:],
        "training": {
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "loss": "sampled pTV-style LCC + sampled analytic physical-unit TV",
            "image_loss": "sampled_ptv_lcc",
            "regularizer": "sampled_idir_physical_tv",
            "ptv_lcc_sigma_mm": float(args.ptv_lcc_sigma_mm),
            "ptv_lcc_radius_sigma": float(args.ptv_lcc_radius_sigma),
            "ptv_lcc_sigma_floor_pix": float(args.ptv_lcc_sigma_floor_pix),
            "ptv_isotv_weight": float(args.ptv_isotv_weight),
            "ptv_csqrt": float(args.ptv_csqrt),
            "border_mask_voxels": int(args.border_mask_voxels),
            "sampled_lcc_centers": int(n_centers),
            "sampled_lcc_offsets": int(offsets_norm_xyz.shape[0]),
            "sampled_lcc_offset_meta": offset_meta,
            "train_coords": n_train,
            "sample_domain": "ptv_1mm_border_masked_crop",
            "sample_domain_voxels": n_train,
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
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    method_epochs = int(args.epochs if args.epochs is not None else 2500)
    method_lr = float(args.lr if args.lr is not None else (1e-3 if direct_cp else 1e-4))
    is_global_fdbe = str(args.method) in {METHOD_SINR_CPS4_GLOBAL_NCC_FD_BE, METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE}
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
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    if sampler is not None:
        with torch.no_grad():
            if direct_cp:
                init_control = params
            else:
                assert model is not None
                init_control = model(control_coords)
            init_dense = c.expand_bspline_controls(init_control, control_shape, image_shape, args.cps)
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=init_dense,
            row_extra={"method": args.method, "checkpoint_source": "model_init"},
        )
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
        if is_global_fdbe:
            data_loss = c.ncc_loss(fixed_t[None, None], warped[None, None])
            reg_loss = c.dense_forward_difference_be(dense_disp, component_reduction="mean")
            reg_weight = float(args.alpha)
        else:
            data_loss = c.ptv_local_cc_loss_with_cache(
                warped,
                border_mask,
                ptv_cache,
                (1.0, 1.0, 1.0),
                args.ptv_lcc_radius_sigma,
            )
            reg_loss = c.isotv_control_loss(control, control_shape, image_shape, (1.0, 1.0, 1.0), args.cps, args.ptv_csqrt)
            reg_weight = float(args.ptv_isotv_weight)
        loss = data_loss + reg_weight * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "reg_weight": float(reg_weight),
            "alpha": float(args.alpha) if is_global_fdbe else float("nan"),
        }
        history.append(row)
        if is_global_fdbe and should_global_ncc_diagnostic(args, epoch, method_epochs):
            global_ncc_diagnostic.append(
                {
                    "epoch": float(epoch),
                    "global_epoch": float(epoch),
                    "global_ncc_loss": row["data_loss"],
                    "global_ncc_value": -row["data_loss"],
                    "diagnostic_name": "full_crop_global_ncc",
                    "eval_domain": "full_1mm_crop",
                    "affects_training": False,
                    "eval_elapsed_sec": 0.0,
                }
            )
            print(json.dumps({"global_ncc_diagnostic": global_ncc_diagnostic[-1]}), flush=True)
        if sampler is not None and sampler.should_record(epoch):
            with torch.no_grad():
                if direct_cp:
                    sample_control = params
                else:
                    assert model is not None
                    sample_control = model(control_coords)
                sample_dense = c.expand_bspline_controls(sample_control, control_shape, image_shape, args.cps)
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=sample_dense,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == method_epochs:
            print(json.dumps(row), flush=True)
    if dense_disp is None:
        dense_disp = torch.zeros((*image_shape, 3), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    c.write_csv(out_dir / "loss_history.csv", history)
    if global_ncc_diagnostic:
        c.write_csv(out_dir / "global_ncc_diagnostic.csv", global_ncc_diagnostic)
    return {
        "dense_disp_norm_xyz": dense_disp.detach(),
        "history_tail": history[-10:],
        "global_ncc_diagnostic_tail": global_ncc_diagnostic[-10:],
        "training": {
            "epochs": method_epochs,
            "lr": method_lr,
            "optimizer": "Adam",
            "loss": "full-crop global NCC + dense forward-difference BE" if is_global_fdbe else "pTV-style LCC + isoTV",
            "image_loss": "full_crop_global_ncc" if is_global_fdbe else "ptv_lcc",
            "regularizer": "dense_forward_difference_be" if is_global_fdbe else "ptv_isotv",
            "alpha": float(args.alpha) if is_global_fdbe else None,
            "cps": int(args.cps),
            "bspline_order": 3,
            "ptv_lcc_sigma_mm": None if is_global_fdbe else float(args.ptv_lcc_sigma_mm),
            "ptv_isotv_weight": None if is_global_fdbe else float(args.ptv_isotv_weight),
            "ptv_csqrt": None if is_global_fdbe else float(args.ptv_csqrt),
            "border_mask_voxels": 0 if is_global_fdbe else int(args.border_mask_voxels),
            "global_ncc_diagnostic_interval": int(args.global_ncc_diagnostic_interval),
            "global_ncc_diagnostic_name": (
                "full_crop_global_ncc"
                if int(args.global_ncc_diagnostic_interval) > 0
                else None
            ),
            "global_ncc_diagnostic_affects_training": (
                False if int(args.global_ncc_diagnostic_interval) > 0 else None
            ),
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


def finest_ptv_loss_diagnostic(
    args: argparse.Namespace,
    fixed_t: torch.Tensor,
    moving_t: torch.Tensor,
    full_border_mask: torch.Tensor,
    full_id_grid: torch.Tensor,
    full_ptv_cache: Dict[str, torch.Tensor | List[float]],
    control: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    reg_weight: float,
) -> Dict[str, object]:
    dense_full = c.expand_on_shape(control, control_shape, full_cps, fixed_t.shape, 1)
    warped_full = c.warp_full(moving_t, full_id_grid + dense_full, mode="bilinear")
    data_loss = c.ptv_local_cc_loss_with_cache(
        warped_full,
        full_border_mask,
        full_ptv_cache,
        (1.0, 1.0, 1.0),
        args.ptv_lcc_radius_sigma,
    )
    reg_loss = c.isotv_control_loss(control, control_shape, fixed_t.shape, (1.0, 1.0, 1.0), int(full_cps), args.ptv_csqrt)
    loss = data_loss + float(reg_weight) * reg_loss
    return {
        "finest_loss": float(loss.detach().cpu()),
        "finest_data_loss": float(data_loss.detach().cpu()),
        "finest_reg_loss": float(reg_loss.detach().cpu()),
        "finest_reg_weight": float(reg_weight),
        "finest_loss_domain": "full_1mm_crop",
    }


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
    global_epoch_offset: int,
    total_epochs: int,
    global_ncc_diagnostic: List[Dict[str, float | str | bool]],
    sampler: Optional[CheckpointSampler] = None,
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
    if bool(args.finest_loss_diagnostic) and is_be_loss:
        raise ValueError("--finest-loss-diagnostic is only implemented for pTV LCC + isoTV multistage runs.")
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
    full_id_grid = None
    full_ptv_cache = None
    if int(args.global_ncc_diagnostic_interval) > 0:
        full_id_grid = c.make_coordinate_grid(tuple(int(v) for v in fixed_t.shape), fixed_t.device).detach()
    if bool(args.finest_loss_diagnostic) and not is_be_loss and int(downsample) != 1:
        if full_id_grid is None:
            full_id_grid = c.make_coordinate_grid(tuple(int(v) for v in fixed_t.shape), fixed_t.device).detach()
        full_ptv_cache = c.make_ptv_lcc_fixed_cache(
            fixed_t,
            (1.0, 1.0, 1.0),
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
                "finest_loss_diagnostic": bool(args.finest_loss_diagnostic),
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
        finest_diag: Dict[str, object] = {}
        if bool(args.finest_loss_diagnostic) and not is_be_loss:
            should_eval_finest = epoch == 1 or epoch % int(args.finest_loss_every) == 0 or epoch == int(epochs)
            if should_eval_finest:
                if int(downsample) == 1:
                    finest_diag = {
                        "finest_loss": float(loss.detach().cpu()),
                        "finest_data_loss": float(data_loss.detach().cpu()),
                        "finest_reg_loss": float(reg_loss.detach().cpu()),
                        "finest_reg_weight": float(reg_weight),
                        "finest_loss_domain": "full_1mm_crop",
                    }
                else:
                    assert full_id_grid is not None and full_ptv_cache is not None
                    with torch.no_grad():
                        finest_diag = finest_ptv_loss_diagnostic(
                            args,
                            fixed_t,
                            moving_t,
                            full_border_mask,
                            full_id_grid,
                            full_ptv_cache,
                            control,
                            control_shape,
                            int(full_cps),
                            float(reg_weight),
                        )
            else:
                finest_diag = {
                    "finest_loss": float("nan"),
                    "finest_data_loss": float("nan"),
                    "finest_reg_loss": float("nan"),
                    "finest_reg_weight": float(reg_weight),
                    "finest_loss_domain": "full_1mm_crop",
                }
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "global_epoch": float(int(global_epoch_offset) + int(epoch)),
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
        row.update(finest_diag)
        rows.append(row)
        if should_global_ncc_diagnostic(args, int(row["global_epoch"]), int(total_epochs)):
            assert full_id_grid is not None
            with torch.no_grad():
                dense_full_diag = c.expand_on_shape(control, control_shape, int(full_cps), fixed_t.shape, 1)
            diag_row = global_ncc_diagnostic_row(
                fixed_t=fixed_t,
                moving_t=moving_t,
                dense_disp=dense_full_diag,
                id_grid=full_id_grid,
                epoch=int(epoch),
                global_epoch=int(row["global_epoch"]),
                eval_domain="full_1mm_crop",
                device=fixed_t.device,
            )
            diag_row["stage_idx"] = float(stage_idx)
            diag_row["stage_epoch"] = float(epoch)
            diag_row["full_cps"] = float(full_cps)
            diag_row["downsample"] = float(downsample)
            diag_row["domain_cps"] = float(domain_cps)
            global_ncc_diagnostic.append(diag_row)
            print(json.dumps({"global_ncc_diagnostic": diag_row}), flush=True)
        if sampler is not None and sampler.should_record(int(row["global_epoch"])):
            with torch.no_grad():
                sample_dense_full = c.expand_on_shape(control, control_shape, int(full_cps), fixed_t.shape, 1)
            sampler.record_dense(
                epoch=int(row["global_epoch"]),
                dense_disp_norm_xyz=sample_dense_full,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "stage_idx": int(stage_idx),
                    "stage_epoch": int(epoch),
                    "full_cps": int(full_cps),
                    "downsample": int(downsample),
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
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
        "finest_loss_diagnostic": bool(args.finest_loss_diagnostic),
        "finest_loss_every": int(args.finest_loss_every) if bool(args.finest_loss_diagnostic) else None,
        "finest_loss_domain": "full_1mm_crop" if bool(args.finest_loss_diagnostic) else None,
        "finest_loss_affects_training": False if bool(args.finest_loss_diagnostic) else None,
        "elapsed_sec": float(elapsed),
        "control_abs_max_norm": float(ctrl_abs.max().detach().cpu()),
        "control_abs_p95_norm": float(torch.quantile(ctrl_abs, 0.95).detach().cpu()),
    }
    return dense_full.detach(), summary, rows


def train_multistage(
    args: argparse.Namespace,
    case: c.LungCTCase,
    device: torch.device,
    out_dir: Path,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
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
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
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
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*image_shape, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "model_init"},
        )
    for stage_idx, (full_cps, downsample, epochs, lr) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        global_epoch_offset = int(sum(stage_epochs[:stage_idx]))
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
            global_epoch_offset,
            int(sum(stage_epochs)),
            global_ncc_diagnostic,
            sampler=sampler,
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
    if global_ncc_diagnostic:
        c.write_csv(out_dir / "global_ncc_diagnostic.csv", global_ncc_diagnostic)
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
        "global_ncc_diagnostic_tail": global_ncc_diagnostic[-10:],
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
            "finest_loss_diagnostic": bool(args.finest_loss_diagnostic),
            "finest_loss_every": int(args.finest_loss_every) if bool(args.finest_loss_diagnostic) else None,
            "finest_loss_domain": "full_1mm_crop" if bool(args.finest_loss_diagnostic) else None,
            "finest_loss_affects_training": False if bool(args.finest_loss_diagnostic) else None,
            "global_ncc_diagnostic_interval": int(args.global_ncc_diagnostic_interval),
            "global_ncc_diagnostic_name": (
                "full_crop_global_ncc"
                if int(args.global_ncc_diagnostic_interval) > 0
                else None
            ),
            "global_ncc_diagnostic_affects_training": (
                False if int(args.global_ncc_diagnostic_interval) > 0 else None
            ),
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
    sampler = CheckpointSampler(args=args, case=case, device=device, out_dir=out_dir)
    if method == METHOD_IDIR:
        result = train_idir(args, case, device, out_dir, sampler=sampler)
        protocol_tag = "ptv_landmark_crop_1mm_with_idir_fixed_t00_lungmask_sampling"
        mask_rule = "fixed_t00_lung_mask_sampling_and_metric"
        loss_domain = "fixed_t00_lung_mask_sampled"
        method_note = "IDIR uses the fixed T00 lung mask for sampled-coordinate training."
    elif method == METHOD_DCTFIELD_FD_BE:
        result = train_dctfield(args, case, device, out_dir, sampler=sampler)
        protocol_tag = (
            "ptv_landmark_crop_1mm_with_idir_fixed_t00_lungmask_sampling+"
            "dctfield_decoded_dense_fd_be"
        )
        mask_rule = "fixed_t00_lung_mask_sampling_and_metric"
        loss_domain = "fixed_t00_lung_mask_sampled_grid_points"
        method_note = (
            "DCTField optimizes low-frequency cosine coefficients and uses fixed T00 "
            "lung-mask sampled NCC like IDIR; BE is finite-difference on the decoded dense field."
        )
    elif method == METHOD_DIRECT_DVF_FD_BE:
        result = train_direct_dvf(args, case, device, out_dir, sampler=sampler)
        protocol_tag = (
            "ptv_landmark_crop_1mm_with_idir_fixed_t00_lungmask_sampling+"
            "direct_dense_dvf_fd_be"
        )
        mask_rule = "fixed_t00_lung_mask_sampling_and_metric"
        loss_domain = "fixed_t00_lung_mask_sampled_grid_points"
        method_note = (
            "Direct dense DVF uses fixed T00 lung-mask sampled NCC like IDIR; "
            "BE is dense finite-difference because no analytic continuous field exists."
        )
    elif method == METHOD_IDIR_SAMPLED_PTV_TV:
        result = train_idir_sampled_ptv_tv(args, case, device, out_dir)
        protocol_tag = "ptv_landmark_crop_1mm_idir_sampled_ptv_lcc_tv_border_crop"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "sampled_border_masked_crop"
        method_note = (
            "This IDIR variant trains on pTV border-crop samples with sampled pTV LCC "
            "and sampled analytic physical-unit TV; the lung mask is metric-only."
        )
    elif method == METHOD_SINR_CPS4_PTV:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=False, sampler=sampler)
        protocol_tag = "ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
        method_note = "pTV loss methods use border-mask loss weighting, not lung-mask loss weighting."
    elif method == METHOD_DIRECT_CP_CPS4_PTV:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=True, sampler=sampler)
        protocol_tag = "ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
        method_note = "pTV loss methods use border-mask loss weighting, not lung-mask loss weighting."
    elif method == METHOD_SINR_CPS4_GLOBAL_NCC_FD_BE:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=False, sampler=sampler)
        protocol_tag = (
            "ptv_landmark_crop_1mm_no_lungmask+single_stage_sinr_cps4+"
            "full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
        method_note = "This method uses full-crop global NCC with no border-mask or lung-mask loss weighting."
    elif method == METHOD_DIRECT_CP_CPS4_GLOBAL_NCC_FD_BE:
        result = train_single_stage_spline(args, case, device, out_dir, direct_cp=True, sampler=sampler)
        protocol_tag = (
            "ptv_landmark_crop_1mm_no_lungmask+single_stage_directcp_cps4+"
            "full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
        method_note = "This method uses full-crop global NCC with no border-mask or lung-mask loss weighting."
    elif method == METHOD_DIRECT_CP_32_PTV:
        result = train_multistage(args, case, device, out_dir, sampler=sampler)
        protocol_tag = "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+fixed_t00_lung_mask_metric"
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "border_masked_crop"
        method_note = "pTV loss methods use border-mask loss weighting, not lung-mask loss weighting."
    elif method == METHOD_DIRECT_CP_32_GLOBAL_NCC_BE:
        result = train_multistage(args, case, device, out_dir, sampler=sampler)
        protocol_tag = (
            "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+"
            "full_crop_global_ncc_bspline_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
        method_note = "This method uses full-crop global NCC with no border-mask or lung-mask loss weighting."
    elif method == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE:
        result = train_multistage(args, case, device, out_dir, sampler=sampler)
        protocol_tag = (
            "ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+"
            "full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric"
        )
        mask_rule = "fixed_t00_lung_mask_metric_only"
        loss_domain = "full_crop_global_ncc"
        method_note = "This method uses full-crop global NCC with no border-mask or lung-mask loss weighting."
    else:
        raise ValueError(f"Unknown method: {method}")

    sampler.save()
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
        "global_ncc_diagnostic_tail": result.get("global_ncc_diagnostic_tail", []),
        "notes": [
            "Inputs are canonical regenerated pTV-prep cropped 1mm arrays.",
            method_note,
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
