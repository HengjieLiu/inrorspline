"""P77: two refined total SVFs, differentiable midpoint NCC9, Phase A only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import bspline_svf as s

c, d = s.c, s.d
VARIANT = "oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1"
METHOD = "direct_cp_842_two_total_svf_symmetric_midpoint"
PROTOCOL = {
    "version": 1, "phase": "a", "schedule": "842", "seed": 1,
    "native_shape": [160, 224, 192], "parameter_units": "normalized_grid_xyz",
    "independent_parameter": "twice_branch_velocity_controls",
    "branch_velocity_scale": .5, "integration_steps": 7, "integration_time": 1.,
    "integration_resolution": "native", "sampler": s.PROTOCOL["sampler"],
    "loss": "two_gradient_1-mean(local_NCC9_correlation_squared)",
    "regularizer": "sum_two_unmasked_forward_valid_component_axis_mean_velocity_diffusions",
    "regularizer_units": "current_loss_grid_voxel_xyz", "amplitude_weight": 0.,
    "balance_weight": 0., "image_preprocessing": d.PROTOCOL["image_preprocessing"],
    "forward_pull": "exp(v1) o exp(-v2)", "reverse_pull": "exp(v2) o exp(-v1)",
    "across_level_composition": False, "optimizer": d.PROTOCOL,
    "magnitude_units": "native_mm", "magnitude_reduction": "voxel_mean_component_sum",
    "magnitude_epsilon_mm2": 1e-8, "diagnostic_steps": [7, 8, 10],
    "diagnostic_points": 100000,
}


@dataclass(frozen=True)
class Spec:
    arm: str
    smoke: bool = False
    case_rank: int = 0

    def validate(self):
        if self.arm not in ("A1", "A2", "A3", "A1_swap") or self.case_rank != 0:
            raise ValueError("Phase A allows only A1/A2/A3/A1_swap on rank 0")

    @property
    def tied(self):
        return self.arm == "A3"

    @property
    def swapped(self):
        return self.arm == "A1_swap"

    @property
    def coefficient(self):
        return 1. if self.arm == "A2" else 2.

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL},
                                           sort_keys=True).encode()).hexdigest()[:10]
        return f"{'smoke_' if self.smoke else ''}rank000_842_{self.arm}_coef{self.coefficient:g}_{digest}"


def matrix(smoke=False):
    return [Spec(arm, smoke) for arm in ("A1", "A2", "A3", "A1_swap")]


def stages(spec):
    spec.validate()
    return d.stages(d.FitSpec(0, "pyramid", 1., "842", smoke=spec.smoke))


class NCC9Pair:
    """Historical mean-kernel/clamp arithmetic, with gradients to BOTH inputs."""
    def __init__(self, device, dtype=torch.float32):
        self.kernel = torch.ones((1, 1, 9, 9, 9), device=device, dtype=dtype)/729

    def average(self, x):
        return F.conv3d(x, self.kernel, padding=4)

    def __call__(self, a, b):
        if a.shape != b.shape or a.ndim != 5 or a.shape[:2] != (1, 1) or a.dtype != b.dtype:
            raise ValueError("NCC requires matched [1,1,D,H,W] arrays")
        ma, mb = self.average(a), self.average(b)
        va, vb = self.average(a*a)-ma.square(), self.average(b*b)-mb.square()
        cov = self.average(a*b)-ma*mb
        return 1-(cov.square()/torch.clamp(va*vb, min=torch.finfo(a.dtype).eps)).mean()


def branch_controls(parameter, tied=False):
    expected = 1 if tied else 2
    if parameter.ndim != 5 or parameter.shape[0] != expected or parameter.shape[-1] != 3:
        raise ValueError("Expected [branches,Dcp,Hcp,Wcp,3] twice-velocity controls")
    q1 = parameter[0]
    q2 = -q1 if tied else parameter[1]
    return q1*.5, q2*.5


def prolong(parameter, shape):
    return torch.stack([c.cubic_bspline_dyadic_knot_insert_3d(q, shape) for q in parameter])


def diffusion(q):
    return c.forward_valid_diffusion(q.permute(3, 0, 1, 2)[None], None)


def energies(v1_mm, v2_mm, mask=None):
    """No control-space norm; all reductions use matched spatial samples."""
    a, b = (v1_mm, v2_mm) if mask is None else (v1_mm[mask], v2_mm[mask])
    e1, e2 = a.square().sum(-1).mean(), b.square().sum(-1).mean()
    rms1, rms2 = (e1+1e-8).sqrt(), (e2+1e-8).sqrt()
    result = torch.stack((e1, e2, rms1, rms2,
                          2*(rms1-rms2).abs()/(rms1+rms2+1e-8),
                          ((a+b)*.5).square().sum(-1).mean().sqrt(),
                          e1/(e1+e2).clamp_min(1e-8),
                          (a*b).sum(-1).mean()/(e1*e2).sqrt().clamp_min(1e-8)))
    keys = ("E1_mm2", "E2_mm2", "a1_mm", "a2_mm", "relative_rms_imbalance",
            "common_mode_rms_mm", "energy_share1", "velocity_cosine")
    return dict(zip(keys, result.detach().cpu().tolist())) | {
        "near_zero_both": bool((e1+e2).detach() < 1e-8),
        "mean_v1_xyz_mm": a.reshape(-1, 3).mean(0).detach().cpu().tolist(),
        "mean_v2_xyz_mm": b.reshape(-1, 3).mean(0).detach().cpu().tolist()}


class Objective:
    def __init__(self, fixed, moving, stage, spec):
        self.stage, self.spec, self.alpha = stage, spec, spec.coefficient
        spec.validate()
        self.image_shape = tuple(fixed.shape)
        self.control_shape = c.build_nested_cp_shapes(self.image_shape, 8, (8, 4, 2))[stage.cps]
        self.fixed, self.blur_sigma, self.blur_kernel_size = c.make_loss_volume(fixed, stage.factor, .5, 3.)
        self.moving, _, _ = c.make_loss_volume(moving, stage.factor, .5, 3.)
        self.eval_shape = tuple(self.fixed.shape)
        self.identity = c.make_coordinate_grid(self.eval_shape, fixed.device)
        self.native_identity = c.make_coordinate_grid(self.image_shape, fixed.device)
        self.scale = d.voxel_scale(self.eval_shape, device=fixed.device)
        self.native_scale = d.voxel_scale(self.image_shape, device=fixed.device)
        self.expand = d.AlignedBSpline(self.control_shape, self.image_shape, self.image_shape, stage.cps, fixed.device)
        self.expand_eval = d.AlignedBSpline(self.control_shape, self.image_shape, self.eval_shape, stage.cps, fixed.device)
        self.ncc = NCC9Pair(fixed.device)

    def velocity(self, parameter, native=True):
        expand = self.expand if native else self.expand_eval
        return tuple(expand(q) for q in branch_controls(parameter, self.spec.tied))

    def positive_maps(self, parameter, steps=7):
        return tuple(s.exp_velocity(v, steps, self.native_identity) for v in self.velocity(parameter))

    def warp(self, image, field):
        u = field if self.eval_shape == self.image_shape else s.sample_displacement(field, self.identity)
        return F.grid_sample(image[None, None], (self.identity+u)[None], mode="bilinear",
                             padding_mode="border", align_corners=True)

    def __call__(self, parameter):
        v1, v2 = self.velocity(parameter)
        u1 = s.exp_velocity(v1, 7, self.native_identity)
        u2 = s.exp_velocity(v2, 7, self.native_identity)
        data = self.ncc(self.warp(self.moving, u1), self.warp(self.fixed, u2))
        ev1, ev2 = (v1, v2) if self.eval_shape == self.image_shape else self.velocity(parameter, False)
        r1, r2 = diffusion(ev1*self.scale), diffusion(ev2*self.scale)
        reg = r1+r2
        self.last_terms = dict(zip(("R1", "R2"), torch.stack((r1, r2)).detach().cpu().tolist()))
        return data+self.alpha*reg, data, reg


@torch.no_grad()
def reconstruct(parameter, cps, shape, tied=False, steps=7):
    identity = c.make_coordinate_grid(shape, parameter.device).to(parameter.dtype)
    expansion = d.AlignedBSpline(parameter.shape[1:4], shape, shape, cps, parameter.device)
    if parameter.dtype == torch.float64:
        expansion.kernel = c.cubic_bspline1d(cps, dtype=torch.float64, device=parameter.device)
    velocities = tuple(expansion(q) for q in branch_controls(parameter, tied))
    positive = tuple(s.exp_velocity(v, steps, identity) for v in velocities)
    negative = tuple(s.exp_velocity(-v, steps, identity) for v in velocities)
    forward_parts, reverse_parts = (positive[0], negative[1]), (positive[1], negative[0])
    forward = s.compose(forward_parts, identity)-identity
    reverse = s.compose(reverse_parts, identity)-identity
    return {"velocities": velocities, "positive": positive, "negative": negative,
            "forward_parts": forward_parts, "reverse_parts": reverse_parts,
            "forward": forward, "reverse": reverse}


def optimize(initial, objective, out):
    """Historical joint state-indexed Adam, with explicit pair metadata/history."""
    out.mkdir(parents=True, exist_ok=False)
    parameter = torch.nn.Parameter(initial.detach().clone())
    if not torch.equal(parameter.detach(), initial):
        raise RuntimeError("Clone changed initialization")
    stage = objective.stage
    optimizer = torch.optim.Adam([parameter], lr=stage.lr, betas=(.9, .999), eps=1e-8)
    best, best_loss, best_epoch = parameter.detach().clone(), math.inf, 0
    rows, reason = [], "max_epochs"
    if parameter.is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (out/"history.jsonl").open("x") as stream:
        for epoch in range(stage.max_epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss, data, reg = objective(parameter)
            values = torch.stack((loss, data, reg)).detach().cpu().tolist()
            if not all(math.isfinite(x) for x in values):
                raise FloatingPointError("Nonfinite midpoint objective")
            if values[0] < best_loss:
                best.copy_(parameter.detach()); best_loss, best_epoch = values[0], epoch
            row = {"epoch": epoch, "optimizer_updates_completed": epoch, "loss": values[0],
                   "data_loss": values[1], "reg_loss": values[2], "weighted_reg_loss": objective.alpha*values[2],
                   **objective.last_terms, "lr": stage.lr, "best_epoch": best_epoch,
                   "elapsed_sec": time.monotonic()-started}
            rows.append(row); stream.write(json.dumps(row)+"\n")
            if epoch % 25 == 0:
                stream.flush(); print(json.dumps({"cps": stage.cps, **row}), flush=True)
            if epoch == stage.max_epochs:
                break
            if d.plateau(rows, stage, 0):
                reason = "adaptive_plateau"; break
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Missing/nonfinite joint control gradient")
            optimizer.step()
    if parameter.is_cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        selected = [float(x) for x in objective(best)]
    if not math.isclose(selected[0], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Selected checkpoint objective mismatch")
    info = {"stage": asdict(stage), "effective_patience": stage.patience,
            "representation": "two_total_bspline_svfs" if not objective.spec.tied else "tied_single_total_bspline_svf",
            "parameter_shape": list(initial.shape), "control_shape": list(objective.control_shape),
            "parameter_units": PROTOCOL["parameter_units"], "branch_velocity_scale": .5,
            "evaluation_shape": list(objective.eval_shape), "coefficient": objective.alpha,
            "actual_epochs": rows[-1]["epoch"], "best_epoch": best_epoch,
            "best_loss": selected[0], "best_data_loss": selected[1], "best_reg_loss": selected[2],
            "initial_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
            "stopping_reason": reason, "lr_reductions": 0, "training_elapsed_sec": time.monotonic()-started,
            "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if parameter.is_cuda else 0.,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30 if parameter.is_cuda else 0.,
            "blur_sigma": objective.blur_sigma, "blur_kernel_size": objective.blur_kernel_size,
            "initial_clone_max_abs": 0., "checkpoint_selection": "joint_lowest_total_loss_including_initialization"}
    d.atomic_torch_save({"twice_branch_controls_norm_xyz": best.cpu(), "summary": info}, out/"best.pt")
    d.atomic_torch_save({"twice_branch_controls_norm_xyz": parameter.detach().cpu(),
                         "optimizer": optimizer.state_dict(), "summary": info}, out/"last.pt")
    c.write_csv(out/"history.csv", rows); c.write_json(out/"summary.json", info)
    return best.detach(), info
