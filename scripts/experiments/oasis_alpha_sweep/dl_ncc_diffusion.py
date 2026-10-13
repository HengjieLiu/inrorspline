"""DL-matched NCC9/diffusion, normalized parameters and aligned cubic fields.

This module deliberately does not reuse the historical registration objectives.
Only geometry, knot insertion, data orientation and canonical metrics are shared.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c

VARIANT = "oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1"
LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
REPRESENTATIONS = ("842", "8421", "842_dense")
RESOLUTIONS = ("pyramid", "full")
METHODS = {
    "842": "direct_cp_8_4_2_ncc9_diffusion_multistage",
    "8421": "direct_cp_8_4_2_1_ncc9_diffusion_multistage",
    "842_dense": "direct_cp_8_4_2_then_dense_ncc9_diffusion",
}
PROTOCOL = {
    "version": 1, "parameter_units": "normalized_grid_xyz",
    "regularizer_units": "evaluation_grid_voxel_xyz", "window": 9,
    "image_preprocessing": "stored_float32_reorient_LIA_to_RAS_no_intensity_rescaling",
    "image_padding": "border", "align_corners": True,
    "geometry": "continuous_cubic_at_endpoint_aligned_native_coordinates",
    "regularizer": "unmasked_total_expanded_field_forward_valid_l2_component_mean_axes_mean",
    "loss": "1_mean_squared_local_correlation_plus_lambda_diffusion",
    "initial_loss_normalization": False, "best_state_every_stage": True,
    "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0., "amsgrad": False,
    "stage_lr": [.03, .01, .003, .001],
    "stage_max_epochs": [400, 800, 1300, 2000],
    "stage_min_epochs": [40, 80, 130, 300],
    "stage_requested_patience": [20, 35, 50, 100],
    "check_interval": 25, "median_window": 25,
    "relative_total": 1e-4, "relative_data": 5e-5,
    "prefix_lr_reductions": 0, "final_lr_reductions": 2,
    "antialias_sigma_scale": .5, "antialias_radius_sigma": 3.,
    "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False,
}


@dataclass(frozen=True)
class FitSpec:
    case_rank: int
    resolution: str
    alpha: float
    representation: str = "842"
    seed: int = 1
    smoke: bool = False

    def validate(self):
        if self.case_rank not in range(10) or self.resolution not in RESOLUTIONS:
            raise ValueError("Require first10 rank and pyramid/full resolution")
        if self.representation not in REPRESENTATIONS:
            raise ValueError("Invalid representation")
        if not math.isfinite(self.alpha) or self.alpha not in LAMBDAS:
            raise ValueError("Require a prespecified diffusion lambda")
        if self.seed != 1:
            raise ValueError("This experiment fixes seed1")

    def prefix(self):
        return FitSpec(self.case_rank, self.resolution, self.alpha, "842", self.seed, self.smoke)

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL},
                                           sort_keys=True).encode()).hexdigest()[:12]
        return (f"{'smoke_' if self.smoke else ''}rank{self.case_rank:03d}_{self.resolution}"
                f"_{self.representation}_lambda{c.format_float(self.alpha)}_{digest}")


@dataclass(frozen=True)
class StageSpec:
    index: int
    cps: int | None
    factor: int
    lr: float
    max_epochs: int
    min_epochs: int
    requested_patience: int
    max_lr_reductions: int
    check_interval: int = 25
    median_window: int = 25
    relative_total: float = 1e-4
    relative_data: float = 5e-5

    @property
    def patience(self):
        return max(self.requested_patience, self.median_window)


def stages(spec: FitSpec):
    spec.validate()
    cps_values = [8, 4, 2] + ([] if spec.representation == "842" else
                              [1 if spec.representation == "8421" else None])
    return [StageSpec(i, cps, (4, 2, 1, 1)[i] if spec.resolution == "pyramid" else 1,
                      PROTOCOL["stage_lr"][i], 1 if spec.smoke else PROTOCOL["stage_max_epochs"][i],
                      1 if spec.smoke else PROTOCOL["stage_min_epochs"][i],
                      PROTOCOL["stage_requested_patience"][i], 2 if i == 3 else 0)
            for i, cps in enumerate(cps_values)]


def voxel_scale(shape, *, device, dtype=torch.float32):
    d, h, w = shape
    return torch.tensor([(w-1)/2, (h-1)/2, (d-1)/2], device=device, dtype=dtype)


def cubic_basis(t):
    t = t.abs()
    return torch.where(t < 1, 2/3 + .5*t**3 - t**2,
                       torch.where(t < 2, (2-t)**3/6, torch.zeros_like(t)))


class AlignedBSpline:
    """Exact continuous basis on the endpoint-aligned image evaluation grid."""
    def __init__(self, control_shape, image_shape, eval_shape, cps, device):
        self.control_shape = tuple(control_shape)
        self.image_shape, self.eval_shape = tuple(image_shape), tuple(eval_shape)
        self.cps = int(cps)
        self.full = self.image_shape == self.eval_shape
        self.kernel = c.cubic_bspline1d(cps, dtype=torch.float32, device=device) if self.full else None
        self.stencils = []
        if not self.full:
            for ncp, native, nout in zip(control_shape, image_shape, eval_shape):
                # Full-grid crop starts at cps: coefficient i is centered at
                # native position (i-1)*cps. Never infer positions from factor.
                locations = torch.linspace(0., float(native-1), nout, device=device) / cps + 1
                left = locations.floor().long()-1
                indices = left[:, None] + torch.arange(4, device=device)[None]
                weights = cubic_basis(locations[:, None]-indices)
                valid = (indices >= 0) & (indices < ncp)
                self.stencils.append((indices.clamp(0, ncp-1), weights*valid))

    def __call__(self, controls):
        grid = controls.reshape(*self.control_shape, 3)
        if self.full:
            field = grid.permute(3, 0, 1, 2)[None]
            for axis in range(3):
                field = c.conv1d(field, kernel=self.kernel, dim=axis+2, stride=self.cps,
                                 padding=(self.kernel.numel()-1)//2, transpose=True)
            z, y, x = self.image_shape
            k = self.cps
            return field[0, :, k:k+z, k:k+y, k:k+x].permute(1, 2, 3, 0).contiguous()
        field = grid
        for axis, (indices, weights) in enumerate(self.stencils):
            moved = field.movedim(axis, 0)
            result = None
            for j in range(4):
                value = moved.index_select(0, indices[:, j]) * weights[:, j].reshape(-1, 1, 1, 1)
                result = value if result is None else result+value
            field = result.movedim(0, axis)
        return field.contiguous()


class NCC9:
    """Exact NCC_vfa arithmetic; cache only the frozen image's moments."""
    def __init__(self, fixed):
        if fixed.dtype != torch.float32 or fixed.ndim != 5 or fixed.shape[:2] != (1, 1):
            raise ValueError("NCC9 requires FP32 [1,1,D,H,W]")
        self.fixed = fixed.detach()
        self.kernel = torch.ones((1, 1, 9, 9, 9), device=fixed.device, dtype=fixed.dtype)/729
        self.mu = self.average(self.fixed)
        self.var = self.average(self.fixed*self.fixed)-self.mu.pow(2)

    def average(self, x):
        return F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=(4, 4, 4))

    def __call__(self, moving):
        if moving.shape != self.fixed.shape or moving.dtype != torch.float32:
            raise ValueError("NCC9 shape/dtype mismatch")
        mu = self.average(moving)
        var = self.average(moving*moving)-mu.pow(2)
        cov = self.average(self.fixed*moving)-self.mu*mu
        cc = (cov*cov)/torch.clamp(self.var*var, min=torch.finfo(cov.dtype).eps)
        return 1-torch.mean(cc)


class StageObjective:
    def __init__(self, fixed, moving, alpha, stage, control_shape=None):
        self.stage, self.alpha = stage, float(alpha)
        self.image_shape = tuple(fixed.shape)
        self.fixed, self.blur_sigma, self.blur_kernel_size = c.make_loss_volume(fixed, stage.factor, .5, 3.)
        self.moving, _, _ = c.make_loss_volume(moving, stage.factor, .5, 3.)
        self.eval_shape = tuple(self.fixed.shape)
        self.identity = c.make_coordinate_grid(self.eval_shape, fixed.device)
        self.scale = voxel_scale(self.eval_shape, device=fixed.device)
        self.ncc = NCC9(self.fixed[None, None])
        self.expand = (AlignedBSpline(control_shape, self.image_shape, self.eval_shape, stage.cps, fixed.device)
                       if stage.cps is not None else None)
        if stage.cps is None and stage.factor != 1:
            raise ValueError("Dense final field must be native resolution")

    def field(self, parameter):
        return self.expand(parameter) if self.expand is not None else parameter

    def __call__(self, parameter):
        q = self.field(parameter)
        warped = F.grid_sample(self.moving[None, None], (self.identity+q)[None],
                               mode="bilinear", padding_mode="border", align_corners=True)
        data = self.ncc(warped)
        vox = q*self.scale
        reg = c.forward_valid_diffusion(vox.permute(3, 0, 1, 2)[None], None)
        return data+self.alpha*reg, data, reg


def plateau(rows, stage: StageSpec, last_action):
    epoch, n, w, p = rows[-1]["epoch"], len(rows), stage.median_window, stage.patience
    if (epoch < stage.min_epochs or epoch % stage.check_interval or
            epoch-last_action < p or n < p+w):
        return False
    improvements = []
    for key in ("loss", "data_loss"):
        old = float(np.median([r[key] for r in rows[n-p-w:n-p]]))
        new = float(np.median([r[key] for r in rows[n-w:n]]))
        improvements.append((old-new)/max(abs(old), 1e-8))
    return improvements[0] < stage.relative_total and improvements[1] < stage.relative_data


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def optimize_stage(initial, objective, stage: StageSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=False)
    parameter = torch.nn.Parameter(initial.detach().clone())
    if not torch.equal(parameter.detach(), initial):
        raise RuntimeError("Initialization clone changed values")
    optimizer = torch.optim.Adam([parameter], lr=stage.lr, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0, amsgrad=False)
    best = parameter.detach().clone()
    best_loss, best_epoch, rows = math.inf, 0, []
    reductions, last_action, reason = 0, 0, "max_epochs"
    if parameter.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (out_dir/"history.jsonl").open("w") as history:
        for epoch in range(stage.max_epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss, data, reg = objective(parameter)
            values = torch.stack((loss, data, reg)).detach().cpu().tolist()
            if not all(math.isfinite(x) for x in values):
                raise FloatingPointError("Nonfinite stage objective")
            if values[0] < best_loss:
                best_loss, best_epoch = values[0], epoch
                best.copy_(parameter.detach())
            row = {"epoch": epoch, "optimizer_updates_completed": epoch,
                   "loss": values[0], "data_loss": values[1], "reg_loss": values[2],
                   "weighted_reg_loss": objective.alpha*values[2],
                   "lr": optimizer.param_groups[0]["lr"], "best_epoch": best_epoch,
                   "elapsed_sec": time.monotonic()-started}
            rows.append(row)
            history.write(json.dumps(row)+"\n")
            if epoch % stage.check_interval == 0:
                history.flush()
                print(json.dumps({"stage": stage.index, **row}), flush=True)
            if epoch == stage.max_epochs:
                break
            if plateau(rows, stage, last_action):
                if reductions >= stage.max_lr_reductions:
                    reason = "adaptive_plateau" if not reductions else "adaptive_plateau_after_two_lr_reductions"
                    break
                optimizer.param_groups[0]["lr"] *= .5
                reductions += 1
                last_action = epoch
            loss.backward()
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Missing/nonfinite parameter gradient")
            optimizer.step()
    if parameter.is_cuda:
        torch.cuda.synchronize()
    if not bool(torch.isfinite(best).all()) or not bool(torch.isfinite(parameter).all()):
        raise FloatingPointError("Nonfinite saved parameter")
    with torch.no_grad():
        selected = [float(x) for x in objective(best)]
    if not math.isclose(selected[0], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Saved best checkpoint loss does not match its state")
    metadata = {"stage": asdict(stage), "effective_patience": stage.patience,
                "representation": "dense_voxel" if stage.cps is None else "bspline_control",
                "parameter_units": "normalized_grid_xyz", "cps": stage.cps,
                "bspline_order": None if stage.cps is None else 3,
                "control_shape": None if stage.cps is None else list(initial.shape[:3]),
                "evaluation_shape": list(objective.eval_shape), "voxel_scale_xyz": objective.scale.tolist(),
                "objective_voxel_count": int(np.prod(objective.eval_shape)),
                "similarity_normalizer": 1.0, "regularizer": PROTOCOL["regularizer"],
                "alpha": objective.alpha, "actual_epochs": rows[-1]["epoch"],
                "best_epoch": best_epoch, "best_loss": selected[0],
                "best_data_loss": selected[1], "best_reg_loss": selected[2],
                "initial_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
                "stopping_reason": reason, "lr_reductions": reductions,
                "training_elapsed_sec": time.monotonic()-started,
                "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if parameter.is_cuda else 0.,
                "blur_sigma": objective.blur_sigma, "blur_kernel_size": objective.blur_kernel_size,
                "initial_clone_max_abs": 0.0}
    atomic_torch_save({"parameter_norm_xyz": best.cpu(), "summary": metadata}, out_dir/"best.pt")
    atomic_torch_save({"parameter_norm_xyz": parameter.detach().cpu(), "optimizer": optimizer.state_dict(),
                       "summary": metadata}, out_dir/"last.pt")
    c.write_csv(out_dir/"history.csv", rows)
    c.write_json(out_dir/"summary.json", metadata)
    return best.detach(), metadata
