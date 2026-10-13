"""Frozen VFA feature losses and dense-voxel optimization (no spline controls)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import run_one as prefix_runner

METHOD = "direct_cp_8_4_2_then_dense_vfa_feature_diffusion"
VARIANT = "oasis_seed3_first10_mrdbscp842_dense_vfa_features_v1"
ORDERS = ("warp_then_project", "project_then_warp")
LOSSES = ("mse", "global_ncc", "local_ncc", "unit_mse", "raw_ncc")
TOLERANCES = {"dice": 0.001, "hd95": 0.02, "asd": 0.005}
REFERENCE = {"dice": 0.79682034087069, "hd95": 2.0485631555197186,
             "asd": 0.4400145870843727}


@dataclass(frozen=True)
class FitConfig:
    order: str = "warp_then_project"
    loss: str = "mse"
    support: str = "full"
    alpha: float = 0.1
    lr: float = 0.1
    max_epochs: int = 2000
    min_epochs: int = 300
    check_interval: int = 25
    median_window: int = 25
    patience: int = 100
    rel_total: float = 1e-4
    rel_data: float = 5e-5
    max_lr_reductions: int = 2
    seed: int = 1

    def validate(self):
        if self.order not in ORDERS or self.loss not in LOSSES or self.support not in ("full", "fixed_positive"):
            raise ValueError("Unknown feature order, loss, or support")
        if not math.isfinite(self.alpha) or self.alpha < 0 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Require finite lambda >= 0 and LR > 0")
        if not 0 <= self.min_epochs <= self.max_epochs or self.max_epochs < 1:
            raise ValueError("Invalid epoch limits")
        if min(self.check_interval, self.median_window, self.patience) < 1:
            raise ValueError("Check/window/patience must be positive")
        if self.patience < self.median_window or self.max_lr_reductions < 0:
            raise ValueError("Invalid patience or LR reductions")
        if min(self.rel_total, self.rel_data) < 0:
            raise ValueError("Relative thresholds must be nonnegative")


def config_name(config: FitConfig, rank: int) -> str:
    config.validate()
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]
    return (f"rank{rank:03d}_{config.order}_{config.loss}_{config.support}"
            f"_a{c.format_float(config.alpha)}_lr{c.format_float(config.lr)}_{digest}")


def metric_deficit(metrics, reference=REFERENCE):
    return max((reference["dice"] - metrics["dice"]) / TOLERANCES["dice"],
               (metrics["hd95"] - reference["hd95"]) / TOLERANCES["hd95"],
               (metrics["asd"] - reference["asd"]) / TOLERANCES["asd"])


def selection_key(row):
    gap = metric_deficit(row)
    return (0, -row["dice"], row["asd"], row["hd95"]) if gap <= 1 else (
        1, gap, -row["dice"], row["asd"], row["hd95"])


def voxel_scale(field):
    d, h, w = field.shape[-4:-1]
    return field.new_tensor([(w - 1) / 2, (h - 1) / 2, (d - 1) / 2])


def warp_features(features, voxel_xyz, identity):
    return F.grid_sample(features, (identity + voxel_xyz / voxel_scale(voxel_xyz))[None],
                         mode="bilinear", padding_mode="border", align_corners=True)


def moving_project(project, features):
    # Native VFA pads before its padding=1 convolution. The central candidate
    # is the projected tensor cropped by one voxel on each side.
    return project(F.pad(features, (1, 1, 1, 1, 1, 1), mode="replicate"))[:, :, 1:-1, 1:-1, 1:-1]


def diffusion(voxel_xyz):
    return c.forward_valid_diffusion(voxel_xyz.permute(3, 0, 1, 2)[None], None)


def region_mean(values, mask=None):
    if mask is None:
        return values.mean()
    weights = mask.to(dtype=values.dtype)[None, None]
    return (values * weights).sum() / (weights.sum() * values.shape[1]).clamp_min(1)


def feature_loss(fixed, moving, loss, mask=None):
    if fixed.shape != moving.shape:
        raise ValueError("Feature shape mismatch")
    if loss in ("mse", "unit_mse"):
        if loss == "unit_mse":
            fixed, moving = F.normalize(fixed, dim=1, eps=1e-6), F.normalize(moving, dim=1, eps=1e-6)
        return region_mean((fixed - moving).square(), mask)
    if loss in ("global_ncc", "raw_ncc"):
        weights = torch.ones_like(fixed[:, :1]) if mask is None else mask[None, None].to(fixed)
        count = weights.sum().clamp_min(1)
        mean_f = (fixed * weights).sum((2, 3, 4), keepdim=True) / count
        mean_m = (moving * weights).sum((2, 3, 4), keepdim=True) / count
        f, m = fixed - mean_f, moving - mean_m
        cross = (f * m * weights).sum((2, 3, 4)) / count
        vf = (f.square() * weights).sum((2, 3, 4)) / count
        vm = (m.square() * weights).sum((2, 3, 4)) / count
        corr = cross / (vf * vm).clamp_min(1e-12).sqrt()
        return (1 - corr.clamp(-1, 1)).mean()
    if loss == "local_ncc":
        # Separable box means avoid a 9^3 convolution. Boundary windows use
        # their actual counts. Foreground support selects window centers only.
        def avg(x):
            for kernel, padding in (((9, 1, 1), (4, 0, 0)),
                                    ((1, 9, 1), (0, 4, 0)),
                                    ((1, 1, 9), (0, 0, 4))):
                x = F.avg_pool3d(x, kernel, stride=1, padding=padding, count_include_pad=False)
            return x
        f, m = avg(fixed), avg(moving)
        cov = avg(fixed * moving) - f * m
        vf, vm = (avg(fixed.square()) - f.square()).clamp_min(0), (avg(moving.square()) - m.square()).clamp_min(0)
        corr = cov / (vf * vm).clamp_min(1e-12).sqrt()
        return region_mean(1 - corr.clamp(-1, 1), mask)
    raise ValueError(loss)


class FeatureObjective:
    def __init__(self, cache, config: FitConfig, device):
        self.config = config
        self.initial = cache["prefix_voxel_xyz"].to(device).detach()
        self.identity = c.make_coordinate_grid(tuple(self.initial.shape[:-1]), device)
        self.mask = cache["fixed_mask"].to(device) if config.support == "fixed_positive" else None
        self.project = torch.nn.Conv3d(16, 8, 3, padding=1).to(device)
        self.project.load_state_dict(cache["projection_state"], strict=True)
        self.project.eval().requires_grad_(False)
        if config.loss == "raw_ncc":
            self.fixed = cache["fixed_image"][None, None].to(device)
            self.moving = cache["moving_image"][None, None].to(device)
        else:
            self.fixed = cache["fixed_projected"].to(device)
            self.moving = cache["moving_encoder" if config.order == "warp_then_project" else "moving_projected"].to(device)
        self.fixed, self.moving = self.fixed.detach(), self.moving.detach()
        with torch.no_grad():
            self.scale = max(float(self.raw(self.initial)), 1e-8)

    def raw(self, field):
        moving = warp_features(self.moving, field, self.identity)
        if self.config.loss != "raw_ncc" and self.config.order == "warp_then_project":
            moving = moving_project(self.project, moving)
        return feature_loss(self.fixed, moving, self.config.loss, self.mask)

    def __call__(self, field):
        raw = self.raw(field)
        regularizer = diffusion(field)
        return raw / self.scale + self.config.alpha * regularizer, raw, regularizer


def plateau(rows, config, last_action):
    epoch = rows[-1]["epoch"]
    if epoch < config.min_epochs or epoch % config.check_interval or epoch - last_action < config.patience:
        return False
    n, p, w = len(rows), config.patience, config.median_window
    if n < p + w:
        return False
    changes = []
    for key in ("loss", "data_loss"):
        old = float(np.median([r[key] for r in rows[n-p-w:n-p]]))
        new = float(np.median([r[key] for r in rows[n-w:n]]))
        changes.append((old - new) / max(abs(old), 1e-8))
    return changes[0] < config.rel_total and changes[1] < config.rel_data


def optimize(objective: FeatureObjective, config: FitConfig, output_dir: Path):
    config.validate()
    field = torch.nn.Parameter(objective.initial.detach().clone())
    if not torch.equal(field.detach(), objective.initial):
        raise RuntimeError("Dense initialization is not an exact clone")
    optimizer = torch.optim.Adam([field], lr=config.lr)
    best_field, best_loss, best_epoch = field.detach().clone(), math.inf, 0
    rows, reductions, last_action = [], 0, 0
    reason = "max_epochs"
    if field.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # Evaluate states 0..max_epochs, taking exactly max_epochs updates. A saved
    # checkpoint is always paired with the objective evaluated at THAT state.
    with (output_dir / "history.jsonl").open("w") as history:
        for epoch in range(config.max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, raw, regularizer = objective(field)
            values = torch.stack((loss, raw, regularizer)).detach().cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError("Nonfinite dense loss")
            if values[0] < best_loss:
                best_loss, best_epoch = values[0], epoch
                best_field.copy_(field.detach())
            row = {"epoch": epoch, "loss": values[0], "raw_feature_loss": values[1],
                   "data_loss": values[1] / objective.scale, "reg_loss": values[2],
                   "lr": optimizer.param_groups[0]["lr"], "elapsed_sec": time.monotonic() - started}
            rows.append(row)
            history.write(json.dumps(row) + "\n")
            if epoch % config.check_interval == 0:
                history.flush()
                print(json.dumps(row), flush=True)
                torch.save({"dense_voxel_xyz": best_field.detach().cpu(), "best_epoch": best_epoch,
                            "best_loss": best_loss, "config": asdict(config)}, output_dir / "best_checkpoint.pt")
            if plateau(rows, config, last_action):
                if reductions == config.max_lr_reductions:
                    reason = "adaptive_plateau_after_two_lr_reductions"
                    break
                optimizer.param_groups[0]["lr"] *= 0.5
                reductions += 1
                last_action = epoch
            if epoch == config.max_epochs:
                break
            loss.backward()
            if field.grad is None or not torch.isfinite(field.grad).all():
                raise FloatingPointError("Missing/nonfinite dense gradient")
            if any(p.grad is not None for p in objective.project.parameters()):
                raise RuntimeError("Frozen VFA projection acquired gradients")
            optimizer.step()
    if field.is_cuda:
        torch.cuda.synchronize()
    if not torch.isfinite(best_field).all():
        raise FloatingPointError("Nonfinite selected field")
    with torch.no_grad():
        checked = float(objective(best_field)[0])
    if not math.isclose(checked, best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Best checkpoint loss/state mismatch")
    state = {"dense_voxel_xyz": best_field.cpu(), "best_epoch": best_epoch, "best_loss": best_loss,
             "config": asdict(config), "representation": "dense_voxel", "control_shape": None,
             "cps": None, "bspline_order": None}
    torch.save(state, output_dir / "best_checkpoint.pt")
    torch.save({"dense_voxel_xyz": field.detach().cpu(), "optimizer": optimizer.state_dict(),
                "epoch": rows[-1]["epoch"], "config": asdict(config)}, output_dir / "last_checkpoint.pt")
    c.write_csv(output_dir / "history.csv", rows)
    return best_field, {"best_epoch": best_epoch, "best_loss": best_loss,
                        "optimizer_steps": rows[-1]["epoch"], "stopping_reason": reason,
                        "lr_reductions": reductions, "training_elapsed_sec": time.monotonic() - started,
                        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30 if field.is_cuda else 0,
                        "feature_loss_normalizer": objective.scale,
                        "equivalent_raw_alpha": config.alpha * objective.scale,
                        "dense_init_max_abs_voxel": 0.0}


def prefix_args():
    saved = sys.argv
    try:
        sys.argv = [saved[0], "--method", "direct_cp_8_4_2_be_bspline_analytic_multistage", "--alpha", "1000"]
        args = prefix_runner.parse_args()
    finally:
        sys.argv = saved
    for key, value in {"stage_full_cps": "8,4,2", "stage_downsample": "4,2,1",
                       "stage_alpha": "1000,1000,1000", "stage_epochs": "400,800,1300",
                       "stage_lr": "0.03,0.01,0.003", "stage_min_epochs": "40,80,130",
                       "stage_patience": "20,35,50", "stage_check_interval": "25,25,25",
                       "stage_rel_delta_total": "0.0001,0.0001,0.0001",
                       "stage_rel_delta_data": "0.00005,0.00005,0.00005",
                       "stage_early_stop_mode": "adaptive", "prolongation_preserve_check": True,
                       "cudnn_benchmark": False, "cudnn_deterministic": True, "seed": 1}.items():
        setattr(args, key, value)
    return args


def load_vfa(checkpoint: Path, vfa_root: Path, device):
    sys.path.insert(0, str(vfa_root))
    from vfa.models.vfa import Encoder, Decoder
    encoder = Encoder(3, 1, 4, 8, 64).to(device)
    decoder = Decoder(3, 4, 8, 8, 64, 0, 1.0, 0).to(device)
    decoder.R = decoder.R.to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    if any(not k.startswith(("unet_model.encoder.", "unet_model.decoder.")) for k in state):
        raise RuntimeError("Unexpected VFA checkpoint keys")
    for module, prefix in ((encoder, "unet_model.encoder."), (decoder, "unet_model.decoder.")):
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        module.eval().requires_grad_(False)
    if tuple(decoder.project[0].weight.shape) != (8, 16, 3, 3, 3) or decoder.skip:
        raise RuntimeError("Incorrect full-resolution projection")
    return encoder, decoder
