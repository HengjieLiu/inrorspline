"""Native wavereg losses, full-resolution normalized-control/dense optimization.

P44 deliberately imports P43 geometry and Adam logic without changing them.
Only fixed-image descriptors are cached; moving descriptors follow image warping.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import dl_ncc_diffusion as base

VARIANT = "oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1"
LOSSES = ("mse", "mind", "rc", "ngf", "femse")
CENTERS = dict(mse=.005, mind=.04, rc=.1, ngf=.2, femse=.2)
SCALES = dict(mse=1., mind=25., rc=10., ngf=1., femse=1.)
MULTIPLIERS = (.01, .1, 1/3, 1., 3., 10., 100.)
METHOD = "direct_cp_8_4_2_then_dense_native_dl_loss_diffusion"
PROTOCOL = {**base.PROTOCOL, "version": 1, "window": None,
            "loss": "native_similarity_scale_times_similarity_plus_scale_times_lambda_times_diffusion",
            "resolution": "full", "stage_downsample": [1, 1, 1, 1],
            "loss_centers": CENTERS, "similarity_scales": SCALES,
            "multipliers": list(MULTIPLIERS), "feature_order": "encode_warped_image",
            "feature_encoder": "frozen_FEDD_A_projection_plus_block0",
            "training_mask": False, "prefix_reuse": False}
CHECKPOINT = Path(os.environ["INRORSPLINE_DL_FEATURE_CHECKPOINT"]) if os.environ.get(
    "INRORSPLINE_DL_FEATURE_CHECKPOINT"
) else None
CONFIG = Path(os.environ["INRORSPLINE_DL_FEATURE_CONFIG"]) if os.environ.get(
    "INRORSPLINE_DL_FEATURE_CONFIG"
) else None


@dataclass(frozen=True)
class FitSpec:
    case_rank: int
    loss: str
    multiplier: float
    seed: int = 1
    smoke: bool = False

    def validate(self):
        if self.case_rank not in range(10) or self.loss not in LOSSES:
            raise ValueError("Require first10 rank and a prespecified loss")
        if self.multiplier not in MULTIPLIERS or self.seed != 1:
            raise ValueError("Require prespecified multiplier and seed1")

    @property
    def alpha(self):
        return CENTERS[self.loss]*self.multiplier

    @property
    def scale(self):
        return SCALES[self.loss]

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL},
                                           sort_keys=True).encode()).hexdigest()[:12]
        return f"{'smoke_' if self.smoke else ''}rank{self.case_rank:03d}_{self.loss}_lambda{self.alpha:.12g}_{digest}"


def specs(smoke=False):
    # Costliest descriptor losses first; then center and ascending offsets.
    return [FitSpec(rank, loss, multiplier, smoke=smoke)
            for loss in ("mind", "femse", "ngf", "rc", "mse")
            for multiplier in ((1.,) if smoke else (1., .01, .1, 1/3, 3., 10., 100.))
            for rank in range(1 if smoke else 10)]


def stages(spec):
    spec.validate()
    return base.stages(base.FitSpec(spec.case_rank, "full", 1., "842_dense", spec.seed, spec.smoke))


def native_loss(key, root, device, checkpoint=CHECKPOINT, config=CONFIG):
    if key == "femse" and (checkpoint is None or config is None):
        raise ValueError(
            "Feature-MSE requires INRORSPLINE_DL_FEATURE_CHECKPOINT and "
            "INRORSPLINE_DL_FEATURE_CONFIG, or explicit CLI paths"
        )
    sys.path.insert(0, str(root))
    modules = dict(mse=("utils_losses_mse", "MSE_loss"), mind=("utils_losses_mind", "MIND_loss"),
                   rc=("utils_losses_rc", "RC_loss"), ngf=("utils_losses_ngf", "NormalizedGradientField3d"),
                   femse=("utils_losses_fedd", "FeatureMSELoss"))
    module, name = modules[key]
    cls = getattr(importlib.import_module(module), name)
    kwargs = ({"path_checkpoint": str(checkpoint), "dir_config": str(config), "device": device}
              if key == "femse" else {"alpha": .05, "reduction": "mean", "coeff_reduction": "mean"}
              if key == "rc" else {"mm_spacing": 1, "eps": 1e-5, "grad_method": "default"}
              if key == "ngf" else {})
    obj = cls(**kwargs).to(device).eval()
    for param in obj.parameters():
        param.requires_grad_(False)
    return obj


class Similarity:
    def __init__(self, key, native, fixed):
        self.key, self.native, self.fixed = key, native, fixed
        self.descriptor = (native.MINDSSC if key == "mind" else native.first_level_encoder
                           if key == "femse" else None)
        with torch.no_grad():
            self.fixed_features = self.descriptor(fixed).detach() if self.descriptor else None
        if self.fixed_features is not None and not bool(torch.isfinite(self.fixed_features).all()):
            raise FloatingPointError("Nonfinite native fixed-image descriptors")

    def __call__(self, warped):
        if self.descriptor:
            return F.mse_loss(self.descriptor(warped), self.fixed_features)
        return self.native(warped, self.fixed)


class Objective:
    def __init__(self, fixed, moving, similarity, spec, stage, control_shape=None):
        self.moving, self.similarity, self.spec = moving, similarity, spec
        self.eval_shape = tuple(fixed.shape)
        self.alpha = spec.scale*spec.alpha
        self.scale = base.voxel_scale(self.eval_shape, device=fixed.device)
        self.identity = c.make_coordinate_grid(self.eval_shape, fixed.device)
        self.expand = (base.AlignedBSpline(control_shape, self.eval_shape, self.eval_shape, stage.cps, fixed.device)
                       if stage.cps is not None else None)
        self.blur_sigma, self.blur_kernel_size = 0., 1

    def field(self, parameter):
        return self.expand(parameter) if self.expand is not None else parameter

    def __call__(self, parameter):
        q = self.field(parameter)
        warped = F.grid_sample(self.moving[None, None], (self.identity+q)[None],
                               mode="bilinear", padding_mode="border", align_corners=True)
        data = self.spec.scale*self.similarity(warped)
        reg = c.forward_valid_diffusion((q*self.scale).permute(3, 0, 1, 2)[None], None)
        return data+self.alpha*reg, data, reg


def optimize_stage(initial, objective, stage, out_dir):
    best, summary = base.optimize_stage(initial, objective, stage, out_dir)
    # P43 already records weighted data and raw diffusion at every evaluated state.
    # Supplement logs without altering its optimizer or checkpoint selection.
    rows = [json.loads(line) for line in (out_dir/"history.jsonl").read_text().splitlines()]
    for row in rows:
        row.update(raw_similarity=row["data_loss"]/objective.spec.scale,
                   weighted_similarity=row["data_loss"], raw_diffusion=row["reg_loss"],
                   effective_lambda=objective.spec.alpha, similarity_scale=objective.spec.scale,
                   actual_regularizer_coefficient=objective.alpha)
    (out_dir/"history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    c.write_csv(out_dir/"history.csv", rows)
    summary.update(loss_key=objective.spec.loss, effective_lambda=objective.spec.alpha,
                   similarity_scale=objective.spec.scale, actual_regularizer_coefficient=objective.alpha,
                   best_raw_similarity=summary["best_data_loss"]/objective.spec.scale,
                   final_stage_is_dense=stage.cps is None)
    for filename in ("best.pt", "last.pt"):
        saved = torch.load(out_dir/filename, map_location="cpu", weights_only=False)
        saved["summary"] = summary
        base.atomic_torch_save(saved, out_dir/filename)
    c.write_json(out_dir/"summary.json", summary)
    return best, summary


def parity(key, native, device):
    """Value/gradient equivalence of fixed-descriptor caching, with native loss."""
    generator = torch.Generator(device=device).manual_seed(17)
    fixed = torch.rand((1, 1, 16, 20, 24), generator=generator, device=device)
    moving = torch.rand(fixed.shape, generator=generator, device=device, requires_grad=True)
    adapter = Similarity(key, native, fixed)
    actual, expected = adapter(moving), native(moving, fixed)
    ga, = torch.autograd.grad(actual, moving)
    ge, = torch.autograd.grad(expected, moving)
    if not torch.allclose(actual, expected, atol=2e-6, rtol=1e-5) or not torch.allclose(ga, ge, atol=2e-6, rtol=1e-5):
        raise RuntimeError("Native similarity value/gradient parity failed")
    if not bool(torch.isfinite(ga).all()) or not float(ga.abs().max()) > 0:
        raise RuntimeError("Missing finite image gradient")
    if any(p.requires_grad or p.grad is not None for p in native.parameters()):
        raise RuntimeError("Loss network must be frozen, with no parameter gradients")
    constant_finite = None
    if key == "mind":
        with torch.no_grad():
            constant_finite = bool(torch.isfinite(native(torch.zeros_like(fixed), torch.zeros_like(fixed))))
    return {"passed": True, "value_abs": float((actual-expected).abs()),
            "gradient_max_abs": float((ga-ge).abs().max()), "image_gradient_max_abs": float(ga.abs().max()),
            "constant_image_loss_finite": constant_finite,
            "constant_image_policy": "native behavior preserved; nonfinite real objectives fail"}
