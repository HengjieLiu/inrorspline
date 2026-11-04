"""P73: native-grid B-spline SVFs with explicit velocity/displacement penalties.

Historical geometry, NCC9 and Adam are imported without changing their sources.
All vectors here are DHW3 normalized-grid xyz, except explicitly named voxel
diagnostics. Integration always uses the full native grid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import dl_ncc_diffusion as d
from bounded_composed_bspline import sample_displacement, compose

VARIANT = "oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1"
METHODS = {"refine": "direct_cp_multistage_bspline_svf_refined",
           "compose": "direct_cp_multistage_bspline_svf_composed"}
PROTOCOL = {
    "version": 1, "native_shape": [160, 224, 192], "integration_steps": 7,
    "integration_time": 1., "integration_resolution": "native",
    "sampler": "trilinear_border_align_corners_true_unclamped_coordinates",
    "parameter_units": "normalized_grid_xyz", "bounded": False,
    "loss": "1-mean(local_NCC9_correlation_squared)", "stage_alpha": 1.,
    "regularizer": "forward_valid_component_mean_axis_mean_unmasked_diffusion",
    "regularizer_units": "evaluation_grid_voxel_xyz", "initial_loss_normalization": False,
    "composition_order": "previous o exp(active_velocity)",
    "composed_velocity_penalty": "active_velocity_only",
    "integration_diagnostics_steps": [7, 8, 10], "offgrid_points": 100000,
    "optimizer_protocol": d.PROTOCOL, "seed": 1,
}


@dataclass(frozen=True)
class Spec:
    schedule: str
    formulation: str
    regularize: str
    integration_steps: int = 7
    case_rank: int = 0
    smoke: bool = False

    def validate(self):
        if self.schedule not in ("842", "8421") or self.formulation not in METHODS:
            raise ValueError("Require schedule842/8421 and refine/compose formulation")
        if self.regularize not in ("velocity", "displacement"):
            raise ValueError("Require velocity/displacement diffusion")
        if self.integration_steps != 7 or self.case_rank != 0:
            raise ValueError("Pilot fixes seven integration steps and shuffled rank0")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL},
                                           sort_keys=True).encode()).hexdigest()[:10]
        return ("smoke_" if self.smoke else "") + f"rank000_cps{self.schedule}_{self.formulation}_{self.regularize}_ss7_a1_{digest}"


def matrix(smoke=False):
    return [Spec(schedule, form, reg, smoke=smoke) for form in METHODS
            for reg in ("velocity", "displacement") for schedule in ("842", "8421")]


def stages(spec):
    spec.validate()
    return d.stages(d.FitSpec(0, "pyramid", 1., spec.schedule, smoke=spec.smoke))


def exp_velocity(velocity, steps=7, identity=None):
    """Explicit 2**(-steps) scaling followed by displacement self-compositions."""
    if type(steps) is not int or steps < 0:
        raise ValueError("Nonnegative integer integration steps required")
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("Expected DHW3 normalized-grid xyz velocity")
    if identity is None:
        identity = c.make_coordinate_grid(velocity.shape[:3], velocity.device).to(velocity.dtype)
    displacement = velocity / (2 ** steps)
    for _ in range(steps):
        displacement = displacement + sample_displacement(displacement, identity + displacement)
    return displacement


class Objective:
    def __init__(self, fixed, moving, stage, spec, frozen=()):
        spec.validate()
        self.stage, self.spec, self.alpha = stage, spec, 1.
        self.image_shape = tuple(fixed.shape)
        self.control_shape = c.build_nested_cp_shapes(self.image_shape, 8, (8, 4, 2, 1))[stage.cps]
        self.fixed, self.blur_sigma, self.blur_kernel_size = c.make_loss_volume(fixed, stage.factor, .5, 3.)
        self.moving, _, _ = c.make_loss_volume(moving, stage.factor, .5, 3.)
        self.eval_shape = tuple(self.fixed.shape)
        self.identity = c.make_coordinate_grid(self.eval_shape, fixed.device)
        self.native_identity = c.make_coordinate_grid(self.image_shape, fixed.device)
        self.scale = d.voxel_scale(self.eval_shape, device=fixed.device)
        self.native_scale = d.voxel_scale(self.image_shape, device=fixed.device)
        self.expand = d.AlignedBSpline(self.control_shape, self.image_shape, self.image_shape, stage.cps, fixed.device)
        self.expand_eval = d.AlignedBSpline(self.control_shape, self.image_shape, self.eval_shape, stage.cps, fixed.device)
        self.frozen = tuple(x.detach() for x in frozen)
        if spec.formulation == "refine" and self.frozen:
            raise ValueError("Refined single-SVF must not have frozen stage maps")
        self.ncc = d.NCC9(self.fixed[None, None])

    def atomic(self, controls, steps=None):
        return exp_velocity(self.expand(controls), self.spec.integration_steps if steps is None else steps,
                            self.native_identity)

    def field(self, controls, native=False):
        identity = self.native_identity if native else self.identity
        atomic = self.atomic(controls)
        if self.spec.formulation == "refine":
            return atomic if native or self.eval_shape == self.image_shape else sample_displacement(atomic, identity)
        return compose((*self.frozen, atomic), identity) - identity

    def __call__(self, controls):
        u = self.field(controls)
        warped = F.grid_sample(self.moving[None, None], (self.identity + u)[None],
                               mode="bilinear", padding_mode="border", align_corners=True)
        data = self.ncc(warped)
        q = self.expand_eval(controls) if self.spec.regularize == "velocity" else u
        reg = c.forward_valid_diffusion((q*self.scale).permute(3, 0, 1, 2)[None], None)
        return data + self.alpha*reg, data, reg


def optimize(initial, objective, out):
    """Reuse the exact historical state-indexed Adam/plateau/checkpoint protocol."""
    best, info = d.optimize_stage(initial, objective, objective.stage, out)
    info.update(representation="bspline_velocity_controls", method=METHODS[objective.spec.formulation],
                regularization_target=objective.spec.regularize,
                regularizer=PROTOCOL["regularizer"], integration_steps=objective.spec.integration_steps,
                integration_resolution="native", bounded=False,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if best.is_cuda else 0.)
    d.atomic_torch_save({"velocity_controls_norm_xyz": best.cpu(), "summary": info}, out/"best.pt")
    last = torch.load(out/"last.pt", map_location="cpu", weights_only=False)
    last["velocity_controls_norm_xyz"] = last.pop("parameter_norm_xyz")
    last["summary"] = info
    d.atomic_torch_save(last, out/"last.pt")
    c.write_json(out/"summary.json", info)
    return best, info


def reconstruct(states, formulation, shape, device, steps=7, inverse=False):
    """Replay selected velocity controls; inverse reverses composition order."""
    active = states[-1:] if formulation == "refine" else states
    identity = c.make_coordinate_grid(shape, device)
    fields, velocities = [], []
    for cps, controls in active:
        controls = controls.to(device)
        v = d.AlignedBSpline(controls.shape[:3], shape, shape, cps, device)(controls)
        fields.append(exp_velocity(-v if inverse else v, steps, identity))
        velocities.append(v)
    if inverse:
        fields.reverse()
    u = fields[0] if formulation == "refine" else compose(fields, identity)-identity
    return u, fields, velocities


@torch.no_grad()
def inverse_consistency(fields, inverse_fields, mask, seed=1, count=100000):
    rng = np.random.default_rng(seed)
    fg = mask.cpu().numpy().astype(bool)
    index = rng.choice(np.flatnonzero(fg), count, replace=count > int(fg.sum()))
    xyz = np.stack(np.unravel_index(index, fg.shape), -1)[:, ::-1].copy()
    scale = d.voxel_scale(fg.shape, device=mask.device)
    points = torch.tensor(xyz, dtype=torch.float32, device=mask.device)/scale - 1
    result = {"count": count, "seed": seed, "support": "fixed_raw_positive_voxel_centers"}
    for name, first, second in (("inverse_after_forward", fields, inverse_fields),
                                 ("forward_after_inverse", inverse_fields, fields)):
        mapped = points
        valid = torch.ones(count, dtype=torch.bool, device=points.device)
        for parts in (first, second):
            for field in reversed(parts):
                mapped = mapped + sample_displacement(field, mapped)
                valid &= (mapped.abs() <= 1).all(-1)
        errors = torch.linalg.vector_norm((mapped-points)*scale, dim=-1)
        result[name] = {"all_mean_mm": float(errors.mean()), "all_p95_mm": float(errors.quantile(.95)),
                        "all_max_mm": float(errors.max()), "all_intermediates_inbounds_count": int(valid.sum()),
                        "inbounds_mean_mm": float(errors[valid].mean()) if valid.any() else None,
                        "inbounds_p95_mm": float(errors[valid].quantile(.95)) if valid.any() else None}
    return result
