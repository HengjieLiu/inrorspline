#!/usr/bin/env python
"""Run one repo-local OASIS alpha-sweep job."""

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

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import common_dctfield as dctfield  # noqa: E402
import common as c  # noqa: E402


METHODS = [
    "idir_original",
    "direct_dvf_sampled_ncc_fd_be",
    "dctfield_sampled_ncc_fd_be",
    "sinr_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    "direct_cp_8_4_2_be_bspline_analytic_multistage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--dct-size", type=int, default=32)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--image-sim-eval-interval", type=int, default=0)
    parser.add_argument("--image-sim-eval-chunk-size", type=int, default=None)
    parser.add_argument("--global-ncc-diagnostic-interval", type=int, default=0)
    parser.add_argument("--global-ncc-diagnostic-chunk-size", type=int, default=None)
    parser.add_argument("--autograd-be-chunk-size", type=int, default=20000)
    parser.add_argument("--stage-full-cps", type=str, default="8,4,2")
    parser.add_argument("--stage-downsample", type=str, default="4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="400,800,1300")
    parser.add_argument("--stage-lr", type=str, default="")
    parser.add_argument("--antialias-sigma-scale", type=float, default=0.5)
    parser.add_argument("--antialias-radius-sigma", type=float, default=3.0)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-original-case-index", type=int, default=None)
    parser.add_argument("--expected-fixed-id", type=str, default=None)
    parser.add_argument("--expected-moving-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default="oasis_seed3_alpha_prerun_repo_local")
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--checkpoint-epochs", type=str, default="0,1,10,25,50,100,250,500,1000,1500,2000,2500")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-case-only", action="store_true")
    return parser.parse_args()


class CheckpointSampler:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.enabled = bool(args.sample_checkpoints)
        self.run_dir = Path(run_dir)
        self.rows: List[Dict[str, object]] = []
        self.disp_norm_xyz: List[np.ndarray] = []
        self.disp_vox_xyz: List[np.ndarray] = []
        self.epochs = {
            int(token)
            for token in str(args.checkpoint_epochs).replace(" ", "").split(",")
            if token
        }
        self.shape = tuple(int(v) for v in c.IMAGE_SHAPE)
        self.device = device
        self.flat_indices_np = np.asarray([], dtype=np.int64)
        self.flat_indices_t = torch.empty(0, dtype=torch.long, device=device)
        self.points_zyx = np.zeros((0, 3), dtype=np.int64)
        self.points_norm_xyz = np.zeros((0, 3), dtype=np.float32)
        if not self.enabled:
            return
        mask = batch["fixed_mask"]
        assert isinstance(mask, torch.Tensor)
        eligible = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1).astype(bool))
        if eligible.size == 0:
            raise RuntimeError("Cannot sample checkpoints from an empty OASIS foreground mask.")
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
            self.run_dir / "checkpoint_sample_metadata.json",
            {
                "enabled": True,
                "dataset": "oasis",
                "sample_seed": int(args.checkpoint_sample_seed),
                "requested_sample_count": int(args.checkpoint_sample_count),
                "actual_sample_count": int(self.flat_indices_np.size),
                "sample_domain": "fixed_image_positive",
                "field_units_saved": ["normalized_grid_xyz", "voxel_xyz"],
                "field_direction": "fixed_to_moving",
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
        row = {
            "checkpoint_index": len(self.rows),
            "epoch": int(epoch),
            "sample_count": int(sample_norm.shape[0]),
            "norm_abs_max": float(sample_norm.abs().max().detach().cpu()),
            "voxel_mag_mean": float(torch.linalg.norm(sample_vox, dim=1).mean().detach().cpu()),
            "voxel_mag_p95": float(torch.quantile(torch.linalg.norm(sample_vox, dim=1), 0.95).detach().cpu()),
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
        c.write_csv(self.run_dir / "checkpoint_sample_rows.csv", self.rows)
        if self.rows:
            np.savez_compressed(
                self.run_dir / "checkpoint_samples.npz",
                epochs=np.asarray([int(row["epoch"]) for row in self.rows], dtype=np.int64),
                flat_indices=self.flat_indices_np,
                points_zyx=self.points_zyx,
                points_norm_xyz=self.points_norm_xyz,
                disp_norm_xyz=np.stack(self.disp_norm_xyz, axis=0),
                disp_vox_xyz=np.stack(self.disp_vox_xyz, axis=0),
            )


def method_key(args: argparse.Namespace) -> str:
    if args.method == "idir_original":
        return "idir_original_dense_siren_be_autograd"
    return str(args.method)


def method_cps(method: str) -> Optional[int]:
    if method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return 2
    if method in {"sinr_cps2_be_bspline_analytic", "direct_cp_cps2_be_bspline_analytic"}:
        return 2
    if method in {"sinr_cps4_be_bspline_analytic", "direct_cp_cps4_be_bspline_analytic"}:
        return 4
    return None


def default_lr(method: str) -> float:
    if method.startswith("direct_cp_"):
        return 1e-3
    return 1e-4


def build_run_name(args: argparse.Namespace, lr: float) -> str:
    key = method_key(args)
    if args.run_name:
        return args.run_name
    if args.method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return (
            f"{key}_rank{args.case_rank:03d}_sched{args.stage_full_cps.replace(',', '-')}"
            f"_ds{args.stage_downsample.replace(',', '-')}_e{args.stage_epochs.replace(',', '-')}"
            f"_a{c.format_float(args.alpha)}_lr{c.format_float(lr)}"
        )
    if args.method == "dctfield_sampled_ncc_fd_be":
        return (
            f"{key}_rank{args.case_rank:03d}_k{int(args.dct_size)}"
            f"_a{c.format_float(args.alpha)}_e{args.epochs}_lr{c.format_float(lr)}"
        )
    return (
        f"{key}_rank{args.case_rank:03d}_a{c.format_float(args.alpha)}"
        f"_e{args.epochs}_lr{c.format_float(lr)}"
    )


def validate_expected_case(args: argparse.Namespace, *, original_index: int, fixed_id: str, moving_id: str) -> None:
    errors: List[str] = []
    if args.expected_original_case_index is not None and int(args.expected_original_case_index) != int(original_index):
        errors.append(
            f"original_case_index {original_index} != expected {int(args.expected_original_case_index)}"
        )
    if args.expected_fixed_id is not None and str(args.expected_fixed_id) != str(fixed_id):
        errors.append(f"fixed_id {fixed_id} != expected {args.expected_fixed_id}")
    if args.expected_moving_id is not None and str(args.expected_moving_id) != str(moving_id):
        errors.append(f"moving_id {moving_id} != expected {args.expected_moving_id}")
    if errors:
        raise RuntimeError("Case validation failed: " + "; ".join(errors))


def idir_gradient(input_coords: torch.Tensor, output: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        [input_coords],
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def idir_jacobian_matrix(input_coords: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    jac = torch.zeros(input_coords.shape[0], 3, 3, dtype=output.dtype, device=output.device)
    for i in range(3):
        jac[:, i, :] = idir_gradient(input_coords, output[:, i], create_graph=True)
    return jac


def idir_bending_energy(input_coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int) -> torch.Tensor:
    jac = idir_jacobian_matrix(input_coords, output_rel)
    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3, dtype=output_rel.dtype, device=output_rel.device)
    dy_xyz = torch.zeros_like(dx_xyz)
    dz_xyz = torch.zeros_like(dx_xyz)
    for i in range(3):
        dx_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 0], create_graph=True)
        dy_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 1], create_graph=True)
        dz_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 2], create_graph=True)

    dx_xyz = dx_xyz.square()
    dy_xyz = dy_xyz.square()
    dz_xyz = dz_xyz.square()
    loss = torch.mean(dx_xyz[:, :, 0]) + torch.mean(dy_xyz[:, :, 1]) + torch.mean(dz_xyz[:, :, 2])
    loss = loss + 2 * torch.mean(dx_xyz[:, :, 1]) + 2 * torch.mean(dx_xyz[:, :, 2]) + torch.mean(dy_xyz[:, :, 2])
    return loss / float(batch_size)


def make_idir_training_coords(
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    selected = coords_all[fixed_mask.detach().reshape(-1).bool()]
    if selected.numel() == 0:
        raise RuntimeError("OASIS fixed image-positive mask is empty.")
    return selected.contiguous()


def evaluate_dense_model(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
) -> Dict[str, object]:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
    dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
    return c.evaluate_from_dense_coords(dense_coords, batch)


def should_image_sim_eval(args: argparse.Namespace, epoch: int) -> bool:
    interval = int(args.image_sim_eval_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(args.epochs)


def should_global_ncc_diagnostic(args: argparse.Namespace, epoch: int, total_epochs: int) -> bool:
    interval = int(args.global_ncc_diagnostic_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(total_epochs)


def global_ncc_diagnostic_row(
    *,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    dense_coords: torch.Tensor,
    epoch: int,
    global_epoch: int,
    eval_domain: str,
    device: torch.device,
) -> Dict[str, float | str | bool]:
    eval_start = time.time()
    with torch.no_grad():
        warped = c.warp_full(moving, dense_coords, mode="bilinear")
        loss = c.ncc_loss(fixed[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    del warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(global_epoch),
        "global_ncc_loss": value,
        "global_ncc_value": -value,
        "diagnostic_name": "full_image_global_ncc",
        "eval_domain": eval_domain,
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
    }


def evaluate_idir_full_image_ncc(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
    epoch: int,
) -> Dict[str, float | str | bool]:
    fixed = batch["fixed"]
    moving = batch["moving"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving, torch.Tensor)
    eval_start = time.time()
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
        dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
        warped = c.warp_full(moving, dense_coords, mode="bilinear")
        loss = c.ncc_loss(fixed[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    del coords_all, chunks, dense_coords, warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(epoch),
        "similarity_loss": value,
        "similarity_value": -value,
        "similarity_name": "full_image_global_ncc",
        "eval_domain": "full_image",
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
    }


def train_idir(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if sampler is not None:
        sampler.record_idir(epoch=0, model=model, row_extra={"method": args.method, "checkpoint_source": "model_init"})
    coords_train = make_idir_training_coords(batch, device)
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    history: List[Dict[str, float]] = []
    image_similarity_eval: List[Dict[str, float | str | bool]] = []
    image_sim_chunk_size = int(args.image_sim_eval_chunk_size or args.eval_chunk_size)
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    global_ncc_chunk_size = int(args.global_ncc_diagnostic_chunk_size or args.eval_chunk_size)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach().requires_grad_(True)
        disp = model(coords)
        moved_coords = coords + disp
        warped = c.sample_volume(moving, moved_coords, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = idir_bending_energy(coords, disp, batch_size=batch_size)
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if should_image_sim_eval(args, epoch):
            sim_row = evaluate_idir_full_image_ncc(model, batch, device, image_sim_chunk_size, epoch)
            image_similarity_eval.append(sim_row)
            print(json.dumps({"image_similarity_eval": sim_row}), flush=True)
        if should_global_ncc_diagnostic(args, epoch, int(args.epochs)):
            diag_row = evaluate_idir_full_image_ncc(model, batch, device, global_ncc_chunk_size, epoch)
            global_ncc_diagnostic.append(
                {
                    "epoch": diag_row["epoch"],
                    "global_epoch": diag_row["global_epoch"],
                    "global_ncc_loss": diag_row["similarity_loss"],
                    "global_ncc_value": diag_row["similarity_value"],
                    "diagnostic_name": "full_image_global_ncc",
                    "eval_domain": "full_image",
                    "affects_training": False,
                    "eval_elapsed_sec": diag_row["eval_elapsed_sec"],
                }
            )
            print(json.dumps({"global_ncc_diagnostic": global_ncc_diagnostic[-1]}), flush=True)
        if sampler is not None:
            sampler.record_idir(
                epoch=epoch,
                model=model,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    metrics = evaluate_dense_model(model, batch, device, args.eval_chunk_size)
    metrics["history"] = history
    metrics["image_similarity_eval"] = image_similarity_eval
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["model"] = model
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = c.count_trainable_params(model)
    return metrics


def train_direct_dvf(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    fixed_mask_bool = fixed_mask.detach().bool()

    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    flat_mask = fixed_mask_bool.reshape(-1)
    train_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty direct-DVF training coordinate set.")

    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    dense_disp = torch.nn.Parameter(torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([dense_disp], lr=lr)
    history: List[Dict[str, float]] = []
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=dense_disp.detach(),
            row_extra={"method": args.method, "checkpoint_source": "dense_dvf_init"},
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.forward_valid_be(
            c.dhw3_to_ncdhw(dense_disp),
            fixed_mask_bool,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0

    final_dense_disp = dense_disp.detach()
    with torch.no_grad():
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = int(final_dense_disp.numel())
    return metrics


def train_dctfield(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    fixed_mask_bool = fixed_mask.detach().bool()

    dct_size = dctfield.normalize_dct_size(int(args.dct_size), c.IMAGE_SHAPE)
    decoder = dctfield.DCTFieldDecoder(c.IMAGE_SHAPE, dct_size, device=device, dtype=torch.float32)
    coeff = torch.nn.Parameter(torch.zeros((3, *dct_size), dtype=torch.float32, device=device))
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    flat_mask = fixed_mask_bool.reshape(-1)
    train_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty DCTField training coordinate set.")

    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    optimizer = torch.optim.Adam([coeff], lr=lr)
    history: List[Dict[str, float]] = []
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "dct_coeff_init"},
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        dense_disp = decoder.decode(coeff)
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.forward_valid_be(
            c.dhw3_to_ncdhw(dense_disp),
            fixed_mask_bool,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0

    with torch.no_grad():
        final_dense_disp = decoder.decode(coeff).detach()
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["dense_disp"] = final_dense_disp
    metrics["dct_coeff"] = coeff.detach()
    metrics["dct_size_zyx"] = list(dct_size)
    metrics["dct_basis"] = decoder.basis_name
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = int(coeff.numel())
    return metrics


def build_spline_generator(
    args: argparse.Namespace,
    control_shape: Tuple[int, int, int],
) -> torch.nn.Module:
    if args.method.startswith("direct_cp_"):
        return c.DirectCPGenerator(control_shape)
    return c.NetworkGenerator(c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega))


def train_spline(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    cps = method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, cps, device)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    generator = build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    history: List[Dict[str, float]] = []
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    if sampler is not None:
        generator.eval()
        with torch.no_grad():
            init_output = generator(control_coords, control_shape)
            init_dense_disp = c.expand_bspline_controls(init_output, control_shape, c.IMAGE_SHAPE, cps)
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=init_dense_disp,
            row_extra={"method": args.method, "checkpoint_source": "model_init"},
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        output = generator(control_coords, control_shape)
        control_disp = output.reshape(*control_shape, 3)
        dense_disp = c.expand_bspline_controls(output, control_shape, c.IMAGE_SHAPE, cps)
        warped = c.warp_full(moving, id_dense + dense_disp, mode="bilinear")
        data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
        reg_loss = c.analytic_bspline_sampled_be(
            control_disp,
            c.IMAGE_SHAPE,
            cps,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if should_global_ncc_diagnostic(args, epoch, int(args.epochs)):
            global_ncc_diagnostic.append(
                {
                    "epoch": float(epoch),
                    "global_epoch": float(epoch),
                    "global_ncc_loss": float(data_loss.detach().cpu()),
                    "global_ncc_value": float((-data_loss).detach().cpu()),
                    "diagnostic_name": "full_image_global_ncc",
                    "eval_domain": "full_image",
                    "affects_training": False,
                    "eval_elapsed_sec": 0.0,
                }
            )
            print(json.dumps({"global_ncc_diagnostic": global_ncc_diagnostic[-1]}), flush=True)
        if sampler is not None and sampler.should_record(epoch):
            generator.eval()
            with torch.no_grad():
                sample_output = generator(control_coords, control_shape)
                sample_dense = c.expand_bspline_controls(sample_output, control_shape, c.IMAGE_SHAPE, cps)
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=sample_dense,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(final_output, control_shape, c.IMAGE_SHAPE, cps).detach()
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["generator"] = generator
    metrics["control_shape"] = control_shape
    metrics["control_coords"] = control_coords.detach()
    metrics["control_disp"] = final_control_disp
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = c.count_trainable_params(generator)
    metrics["cps"] = cps
    return metrics


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
    rows: List[Dict[str, object]] = []
    dense = c.expand_on_shape(c.grid_to_flat(exact_grid), target_shape, target_cps, image_shape, 1)
    row: Dict[str, object] = {
        "stage_idx": int(stage_idx),
        "target_cps": int(target_cps),
        "method": "exact_knot",
        "source_shape_zyx": list(int(v) for v in prev_shape),
        "target_shape_zyx": list(int(v) for v in target_shape),
    }
    row.update(c.field_difference_metrics(dense - prev_dense_full, prefix="preserve_"))
    rows.append(row)
    return c.grid_to_flat(exact_grid).detach(), rows


def optimize_multistage_stage(
    args: argparse.Namespace,
    fixed_image: torch.Tensor,
    moving: torch.Tensor,
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
        fixed_image,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    moving_l, _sigma_m, _kernel_m = c.make_loss_volume(
        moving,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    eval_shape = tuple(int(v) for v in fixed_l.shape)
    domain_cps = c.domain_cps_from_full(int(full_cps), int(downsample))
    id_grid = c.make_coordinate_grid(eval_shape, fixed_image.device).detach()
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
                "eval_shape_zyx": list(eval_shape),
                "epochs": int(epochs),
                "lr": float(lr),
                "image_loss": "sinr_ncc",
                "regularizer": "analytic_cubic_bspline_be",
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        control_grid = c.flat_to_grid(control, control_shape)
        dense_disp = c.expand_on_shape(control, control_shape, int(full_cps), eval_shape, int(downsample))
        warped = c.warp_full(moving_l, id_grid + dense_disp, mode="bilinear")
        data_loss = c.ncc_loss(fixed_l[None, None], warped[None, None])
        reg_loss = c.analytic_bspline_sampled_be(
            control_grid,
            eval_shape,
            int(domain_cps),
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
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
            "alpha": float(args.alpha),
            "lr": float(lr),
        }
        rows.append(row)
        if should_global_ncc_diagnostic(args, int(row["global_epoch"]), int(total_epochs)):
            with torch.no_grad():
                dense_full_diag = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
                dense_coords_diag = c.make_coordinate_grid(c.IMAGE_SHAPE, fixed_image.device).detach() + dense_full_diag
            diag_row = global_ncc_diagnostic_row(
                fixed=fixed_image,
                moving=moving,
                dense_coords=dense_coords_diag,
                epoch=int(epoch),
                global_epoch=int(row["global_epoch"]),
                eval_domain="full_image",
                device=fixed_image.device,
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
                sample_dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
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
        dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
    ctrl_abs = control.detach().abs().reshape(-1)
    summary = {
        "stage_idx": int(stage_idx),
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(eval_shape),
        "n_control": int(np.prod(tuple(int(v) for v in control_shape))),
        "epochs": int(epochs),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": float(blur_sigma),
        "antialias_kernel_size": int(blur_kernel),
        "elapsed_sec": float(time.time() - stage_start),
        "control_abs_max_norm": float(ctrl_abs.max().detach().cpu()),
        "control_abs_p95_norm": float(torch.quantile(ctrl_abs, 0.95).detach().cpu()),
    }
    return dense_full.detach(), summary, rows


def train_multistage_direct_cp(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(lr))
    base_spacing = int(stage_cps[0])
    nested_shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, base_spacing=base_spacing, spacings=stage_cps)
    c.assert_nested_schedule(nested_shapes, stage_cps)

    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
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
    final_cps: Optional[int] = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "model_init"},
        )
    for stage_idx, (full_cps, downsample, epochs, stage_lr_value) in enumerate(
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
                c.IMAGE_SHAPE,
                int(stage_idx),
            )
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        dense_full, summary, rows = optimize_multistage_stage(
            args,
            fixed_image,
            moving,
            control,
            control_shape,
            int(full_cps),
            int(downsample),
            int(epochs),
            float(stage_lr_value),
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
        final_cps = int(full_cps)
        stage_summaries.append(summary)
        history.extend(rows)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if final_dense is None or final_control is None or final_control_shape is None or final_cps is None:
        raise RuntimeError("Multistage training produced no final control field.")
    metrics = c.evaluate_from_dense_coords(id_dense + final_dense.detach(), batch)
    final_control_grid = c.flat_to_grid(final_control, final_control_shape).detach()
    metrics["history"] = [
        {
            **row,
            "epoch": float(row["global_epoch"]),
            "stage_epoch": float(row["epoch"]),
        }
        for row in history
    ]
    metrics["stage_history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["stage_summaries"] = stage_summaries
    metrics["prolongation_diagnostics"] = prolong_rows
    metrics["control_shape"] = final_control_shape
    metrics["control_coords"] = torch.empty(0, 3, dtype=torch.float32)
    metrics["control_disp"] = final_control_grid
    metrics["dense_disp"] = final_dense.detach()
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = int(final_control.numel())
    metrics["cps"] = int(final_cps)
    metrics["stage_full_cps"] = stage_cps
    metrics["stage_downsample"] = stage_downsample
    metrics["stage_epochs"] = stage_epochs
    metrics["stage_lr"] = stage_lr
    metrics["nested_control_shapes_zyx"] = {str(k): list(v) for k, v in nested_shapes.items()}
    metrics["prolongation"] = "exact crop-aligned cubic B-spline dyadic knot insertion"
    return metrics


def write_outputs(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    lr: float,
    total_elapsed_sec: float,
) -> None:
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_coords = metrics["dense_coords"]
    assert isinstance(dense_coords, torch.Tensor)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_coords.device)
    dense_disp_norm = (dense_coords - id_grid).detach()
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    control_disp = metrics.get("control_disp")
    if isinstance(control_disp, torch.Tensor):
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(
            dense_disp_norm,
            fixed_mask.detach().bool(),
            control_disp_norm=control_disp,
            cps=int(metrics["cps"]),
        )
    else:
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp_norm, fixed_mask.detach().bool())
        model = metrics.get("model")
        if isinstance(model, torch.nn.Module):
            idir_be_rows = c.idir_autograd_be_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            idir_diffusion_rows = c.idir_autograd_diffusion_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            be_rows.extend(idir_be_rows)
            diffusion_rows.extend(idir_diffusion_rows)
            reg_metrics.update(c.idir_autograd_be_rows_to_columns(idir_be_rows))
            reg_metrics.update(c.idir_autograd_diffusion_rows_to_columns(idir_diffusion_rows))
            reg_metrics["idir_autograd_be_chunk_size"] = int(args.autograd_be_chunk_size)
            reg_metrics["idir_autograd_diffusion_chunk_size"] = int(args.autograd_be_chunk_size)
            c.add_native_analytic_be_columns(reg_metrics)
            c.add_native_analytic_diffusion_columns(reg_metrics)

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
    if args.method == "direct_dvf_sampled_ncc_fd_be" and "dense_disp" in metrics:
        state["dense_disp_norm"] = metrics["dense_disp"].detach().cpu()  # type: ignore[union-attr]
    if args.method == "dctfield_sampled_ncc_fd_be" and "dct_coeff" in metrics:
        state["dct_coeff"] = metrics["dct_coeff"].detach().cpu()  # type: ignore[union-attr]
        state["dct_size_zyx"] = metrics.get("dct_size_zyx")
        state["dct_basis"] = metrics.get("dct_basis")
    if "control_disp" in metrics and "generator" not in metrics:
        state["control_disp"] = metrics["control_disp"].detach().cpu()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    for key in ("stage_summaries", "prolongation_diagnostics", "stage_full_cps", "stage_downsample", "stage_epochs", "stage_lr"):
        if key in metrics:
            state[key] = metrics[key]
    torch.save(state, run_dir / "model_state.pt")
    if "control_disp" in metrics:
        torch.save(
            {
                "control_disp": metrics["control_disp"].detach().cpu(),  # type: ignore[union-attr]
                "control_coords": metrics["control_coords"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": metrics["control_shape"],
                "cps": metrics["cps"],
                "image_shape": c.IMAGE_SHAPE,
                "be_type": "be_bspline_analytic",
                "image_loss": "sinr_ncc",
                "method_key": method_key(args),
                "alpha": float(args.alpha),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )

    c.write_csv(run_dir / "history.csv", metrics["history"])  # type: ignore[arg-type]
    if "image_similarity_eval" in metrics and metrics["image_similarity_eval"]:
        c.write_csv(run_dir / "image_similarity_eval.csv", metrics["image_similarity_eval"])  # type: ignore[arg-type]
    if "global_ncc_diagnostic" in metrics and metrics["global_ncc_diagnostic"]:
        c.write_csv(run_dir / "global_ncc_diagnostic.csv", metrics["global_ncc_diagnostic"])  # type: ignore[arg-type]
    if "stage_history" in metrics:
        c.write_csv(run_dir / "stage_history.csv", metrics["stage_history"])  # type: ignore[arg-type]
    if "stage_summaries" in metrics:
        c.write_csv(run_dir / "stage_summaries.csv", metrics["stage_summaries"])  # type: ignore[arg-type]
    if "prolongation_diagnostics" in metrics:
        c.write_csv(run_dir / "prolongation_metrics.csv", metrics["prolongation_diagnostics"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "case_metrics.csv", [c.accuracy_row(metrics, fixed_id, moving_id)])
    c.write_csv(run_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(run_dir / "be_metrics.csv", be_rows)
    c.write_csv(run_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])

    last = metrics["history"][-1]  # type: ignore[index]
    is_idir = args.method == "idir_original"
    is_direct_dvf = args.method == "direct_dvf_sampled_ncc_fd_be"
    is_dctfield = args.method == "dctfield_sampled_ncc_fd_be"
    is_bspline = not is_idir and not is_direct_dvf and not is_dctfield
    activation = (
        "dctfield_coefficients"
        if is_dctfield
        else "direct_dense_grid_displacement"
        if is_direct_dvf
        else "direct_control_points"
        if args.method.startswith("direct_cp_")
        else "Siren"
    )
    omega = float(args.idir_omega) if is_idir else (None if (is_direct_dvf or is_dctfield) else float(args.spline_omega))
    be_type = (
        "be_native_analytic"
        if is_idir
        else "dct_decoded_dense_forward_difference_be"
        if is_dctfield
        else "dense_forward_difference_be"
        if is_direct_dvf
        else "be_bspline_analytic"
    )
    diffusion_type = (
        "diffusion_native_analytic"
        if is_idir
        else "dct_decoded_dense_forward_difference"
        if is_dctfield
        else "dense_forward_difference"
        if is_direct_dvf
        else "diffusion_bspline_analytic"
    )
    native_be_formula = (
        "idir_coordinate_autograd_hessian"
        if is_idir
        else "not_used_dct_spectral_be_possible"
        if is_dctfield
        else "not_available_dense_grid_dvf"
        if is_direct_dvf
        else "bspline_analytic"
    )
    native_diffusion_formula = (
        "idir_coordinate_autograd_jacobian"
        if is_idir
        else "not_used_dct_spectral_diffusion_possible"
        if is_dctfield
        else "not_available_dense_grid_dvf"
        if is_direct_dvf
        else "bspline_analytic"
    )
    be_training_reduction = (
        "original IDIR coordinate-autograd analytic BE; sampled spatial means then divided by batch size"
        if is_idir
        else "dense valid-stencil finite-difference BE over decoded DCT field and fixed image-positive mask; spatial mean plus component mean"
        if is_dctfield
        else "dense valid-stencil finite-difference BE over fixed image-positive mask; spatial mean plus component mean"
        if is_direct_dvf
        else "analytic cubic B-spline BE sampled at voxel centers; spatial mean plus component mean"
    )
    image_loss = "idir_global_ncc_sampled" if (is_idir or is_direct_dvf or is_dctfield) else "sinr_ncc"
    sample_domain = (
        "fixed_image_positive"
        if is_idir
        else "fixed_image_positive_grid_points"
        if is_dctfield
        else "fixed_image_positive_grid_points"
        if is_direct_dvf
        else "none_full_image_ncc"
    )
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": str(args.variant),
        "method": args.method,
        "method_key": method_key(args),
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": int(args.epochs),
        "eval_interval": int(args.eval_interval),
        "image_similarity_eval_interval": int(args.image_sim_eval_interval),
        "image_similarity_eval_name": (
            "full_image_global_ncc"
            if is_idir and int(args.image_sim_eval_interval) > 0
            else None
        ),
        "image_similarity_eval_affects_training": False if is_idir and int(args.image_sim_eval_interval) > 0 else None,
        "global_ncc_diagnostic_interval": int(args.global_ncc_diagnostic_interval),
        "global_ncc_diagnostic_name": (
            "full_image_global_ncc"
            if int(args.global_ncc_diagnostic_interval) > 0
            else None
        ),
        "global_ncc_diagnostic_affects_training": (
            False if int(args.global_ncc_diagnostic_interval) > 0 else None
        ),
        "global_ncc_diagnostic_tail": metrics.get("global_ncc_diagnostic", [])[-10:],
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "activation": activation,
        "omega": omega,
        "lr": float(lr),
        "alpha": float(args.alpha),
        "be_type": be_type,
        "diffusion_type": diffusion_type,
        "regularizer": be_type,
        "native_analytic_be_formula": native_be_formula,
        "native_analytic_diffusion_formula": native_diffusion_formula,
        "be_training_reduction": be_training_reduction,
        "image_loss": image_loss,
        "sample_domain": sample_domain,
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": metrics.get("cps"),
        "bspline_order": 3 if is_bspline else None,
        "bspline_mode": "disp" if is_bspline else None,
        "grid_order": "pytorch",
        "seed": int(args.seed),
        "coords_batch_size": metrics.get("coords_batch_size"),
        "train_coords": metrics.get("train_coords"),
        "param_count": metrics.get("param_count"),
        "dct_size_zyx": metrics.get("dct_size_zyx"),
        "dct_basis": metrics.get("dct_basis"),
        "stage_full_cps": metrics.get("stage_full_cps"),
        "stage_downsample": metrics.get("stage_downsample"),
        "stage_epochs": metrics.get("stage_epochs"),
        "stage_lr": metrics.get("stage_lr"),
        "stage_summaries": metrics.get("stage_summaries"),
        "prolongation_diagnostics": metrics.get("prolongation_diagnostics"),
        "nested_control_shapes_zyx": metrics.get("nested_control_shapes_zyx"),
        "prolongation": metrics.get("prolongation"),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": last["loss"],
        "final_data_loss": last["data_loss"],
        "final_reg_loss": last["reg_loss"],
        "dense_image_loss": metrics["image_loss"],
        "dice_mean": metrics["dice_mean"],
        "hd95_mean": metrics["hd95_mean"],
        "asd_backend": metrics["asd_backend"],
        "asd_warped_to_fixed_mean": metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": metrics["asd_symmetric_mean"],
    }
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(c.accuracy_row(metrics, fixed_id, moving_id))
    summary.update(reg_metrics)
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)


def main() -> None:
    args = parse_args()
    lr = float(args.lr if args.lr is not None else default_lr(args.method))
    if args.validate_case_only:
        case = c.resolve_case(rank=args.case_rank, pair_file=args.pair_file, oasis_root=args.oasis_root)
        print(json.dumps(case.__dict__, indent=2), flush=True)
        validate_expected_case(
            args,
            original_index=case.original_case_index,
            fixed_id=case.fixed_id,
            moving_id=case.moving_id,
        )
        return

    run_name = build_run_name(args, lr)
    key = method_key(args)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "method": args.method,
                "method_key": key,
                "gpu": args.gpu,
                "case_rank": batch["case_rank"],
                "original_case_index": batch["original_case_index"],
                "fixed_id": batch["fixed_id"],
                "moving_id": batch["moving_id"],
                "mask_audit": batch["mask_audit"],
                "alpha": args.alpha,
                "lr": lr,
                "epochs": args.epochs,
            },
            allow_nan=True,
        ),
        flush=True,
    )
    sampler = CheckpointSampler(args=args, batch=batch, device=device, run_dir=run_dir)

    if args.method == "idir_original":
        metrics = train_idir(args, batch, device, lr, sampler=sampler)
    elif args.method == "direct_dvf_sampled_ncc_fd_be":
        metrics = train_direct_dvf(args, batch, device, lr, sampler=sampler)
    elif args.method == "dctfield_sampled_ncc_fd_be":
        metrics = train_dctfield(args, batch, device, lr, sampler=sampler)
    elif args.method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        metrics = train_multistage_direct_cp(args, batch, device, lr, sampler=sampler)
    else:
        metrics = train_spline(args, batch, device, lr, sampler=sampler)
    sampler.save()
    metrics["started_at"] = started_at
    total_elapsed_sec = time.time() - total_start
    write_outputs(args, batch, metrics, run_dir, run_name, lr, total_elapsed_sec)


if __name__ == "__main__":
    main()
