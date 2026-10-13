"""Original10k-coordinate IDIR with signed1-NCC and analytic voxel diffusion."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import common as c
from idir_ncc_diffusion import sha256, state_hash, write_json, cpu_state, verify_hashes, validate_label_records

VARIANT = "oasis_seed3_first10_idir_sampled_global_ncc_analytic_diffusion_staged_v1"
METHOD = "idir_siren_sampled10000_global_1ncc_analytic_voxel_diffusion"
OMEGAS = (8, 16, 32, 64, 128)
INITIAL_EXPONENTS = (-3, -2, -1, 0, 1, 2, 3)
SLOTS = (1, 1, 3, 3)
METRICS = ("dice_mean", "hd95_mean", "asd_symmetric_mean", "log_jdet_std_fg", "jdet_le0_pct_fg")
PROTOCOL = {
    "method": METHOD, "shape_dhw": [160, 224, 192], "layers": [3, 256, 256, 256, 3],
    "coordinates": "normalized_xyz", "output": "normalized_displacement_xyz",
    "initialization": "native_IDIR_omega_dependent_including_output_bias_unchanged",
    "seed": 1, "samples_per_update": 10000, "sampling": "torch_randperm_without_replacement_refreshed_each_update",
    "sample_domain": "fixed_raw_image_positive_voxel_centers",
    "preprocessing": "legacy_pair_minmax_after_LIA_to_RAS", "image_sampling": "trilinear_zero_padding_align_corners_true",
    "similarity": "1_plus_legacy_IDIR_signed_global_negative_NCC",
    "diffusion": "analytic_displacement_Jacobian_voxel_units_mean_over_samples_components_axes",
    "derivative_units": "du_voxel_component_over_dx_voxel_axis",
    "derivative_scale": "output_scale_component_divided_by_input_scale_axis",
    "diffusion_extra_batch_divisor": False, "diffusion_create_graph": True,
    "lambda_normalization": None, "local_ncc": False, "full_volume_training": False,
    "adam_lr": .0001, "adam_betas": [.9, .999], "adam_eps": 1e-8,
    "adam_weight_decay": 0., "adam_amsgrad": False, "updates": 2500,
    "scheduler": None, "early_stopping": False, "checkpoint_selection": "terminal2500",
    "precision": "float32", "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False,
    "cuda_caveat": "grid_sample backward can remain nondeterministic; historical effective backend incomplete",
    "direction": "moving_to_fixed", "field_sampling_map": "fixed_to_moving_pull_identity_plus_displacement",
    "foreground": "fixed_raw_image_positive", "jacobian_metrics": "canonical_LUMIR_crop2",
    "stdlogj": "natural_log_positive_J_population_std", "folding": "nonpositive_J_percentage",
    "asd": "pooled_surfel_area_weighted_symmetric_mm", "labels": list(range(1, 36)),
    "absent_labels": "exclude_if_absent_in_fixed_moving_or_warped",
}


@dataclass(frozen=True)
class FitSpec:
    rank: int
    exponent: int
    omega: int = 32
    smoke_steps: int = 0

    @property
    def alpha(self):
        return 10. ** self.exponent

    @property
    def updates(self):
        return self.smoke_steps or 2500

    def validate(self):
        if self.rank not in range(10) or self.omega not in OMEGAS or not isinstance(self.exponent, int):
            raise ValueError("Require first10 rank, approved frequency and integer power of10")
        if not -37 <= self.exponent <= 37 or self.smoke_steps not in (0, 1, 100):
            raise ValueError("Require representable FP32 lambda and production2500 or smoke1/100")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return f"{'smoke'+str(self.smoke_steps)+'_' if self.smoke_steps else ''}rank{self.rank:03d}_omega{self.omega}_lambda1e{self.exponent:+d}_seed1_{digest}"


def cohort(exponents, omega=32):
    return [FitSpec(rank, exponent, omega) for exponent in exponents for rank in range(10)]


def source_hashes():
    here = Path(__file__).resolve().parent
    names = ("idir_sampled_diffusion.py", "run_oasis_idir_sampled_diffusion_one.py",
             "launch_oasis_idir_sampled_diffusion.py", "calibrate_oasis_idir_sampled_diffusion.py",
             "common.py", "run_one.py", "idir_ncc_diffusion.py", "dl_ncc_diffusion.py",
             "run_oasis_sitreg_first100_eval.py", "sitreg_training.py", "smoke_dl_training_memory.py",
             "run_oasis_vfa_feature_dense_one.py", "vfa_feature_refinement.py",
             "run_oasis_saved_field_bspline_fit_one.py", "evaluate_saved_baselines_first10.py")
    files = [here / name for name in names] + [c.REPO_ROOT / "scripts/reports/oasis_dl_ncc_diffusion_first10_arc.py",
                                             here.parent / "common_dctfield.py"]
    return {str(path): sha256(path) for path in files}


def analytic_diffusion(coords, displacement, shape=c.IMAGE_SHAPE, create_graph=True):
    if coords.shape != displacement.shape or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Require matching Nx3 coordinates and displacement")
    scale = coords.new_tensor([(shape[2] - 1) / 2, (shape[1] - 1) / 2, (shape[0] - 1) / 2])
    jac = torch.stack([torch.autograd.grad(displacement[:, component], coords,
                      grad_outputs=torch.ones_like(displacement[:, component]), create_graph=create_graph,
                      retain_graph=True)[0] for component in range(3)], dim=1)
    return (jac * (scale[:, None] / scale[None, :])[None]).square().mean()


def objective(model, coords, fixed, moving, alpha):
    disp = model(coords)
    warped = c.sample_volume(moving, coords + disp)
    fixed_samples = c.sample_volume(fixed, coords)
    data = 1 + c.ncc_loss(fixed_samples, warped)
    diffusion = analytic_diffusion(coords, disp, tuple(fixed.shape))
    return data + alpha * diffusion, data, diffusion


def training_coords(batch, device):
    # Same ordering/filter as run_one.make_idir_training_coords.
    grid = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    selected = grid[batch["fixed_mask"].reshape(-1).bool()]
    if len(selected) < 10000:
        raise ValueError("The requested exact10000 samples exceed the training foreground")
    return selected.contiguous()


def sample_coords(pool):
    indices = torch.randperm(len(pool), device=pool.device)[:10000]
    coords = pool[indices].detach().requires_grad_(True)
    if coords.shape != (10000, 3):
        raise RuntimeError("Training sample count changed")
    return coords, indices


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.0001, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., amsgrad=False)


@torch.no_grad()
def dense_field(model, device, chunk=200000):
    grid = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    result = torch.empty_like(grid)
    for start in range(0, len(grid), chunk):
        result[start:start + chunk] = model(grid[start:start + chunk])
    return result.reshape(*c.IMAGE_SHAPE, 3)


def aggregate(rows):
    result = []
    groups = sorted({(r["omega"], r["exponent"]) for r in rows})
    for omega, exponent in groups:
        subset = [r for r in rows if (r["omega"], r["exponent"]) == (omega, exponent)]
        if len(subset) != 10 or sorted(r["rank"] for r in subset) != list(range(10)):
            continue
        row = {"omega": omega, "exponent": exponent, "alpha": 10. ** exponent, "n": 10}
        for key in METRICS:
            values = np.asarray([r[key] for r in subset], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite complete-cohort metric")
            row[key + "_mean"] = float(values.mean())
            row[key + "_sample_sd"] = float(values.std(ddof=1))
        result.append(row)
    return result


def select(means):
    if not means:
        raise ValueError("No complete cohorts to select")
    return min(means, key=lambda r: (-r["dice_mean_mean"], r["asd_symmetric_mean_mean"], r["hd95_mean_mean"],
                                    abs(r["exponent"]), r["exponent"], r["omega"]))


def boundary_decision(means, exponents):
    subset = [r for r in means if r["omega"] == 32 and r["exponent"] in exponents]
    if len(subset) != len(exponents) or {r["exponent"] for r in subset} != set(exponents):
        raise ValueError("Boundary decisions require all cases at every tested lambda")
    chosen = select(subset)
    lo, hi = min(exponents), max(exponents)
    extra = lo - 1 if chosen["exponent"] == lo else hi + 1 if chosen["exponent"] == hi else None
    if extra is not None and not -37 <= extra <= 37:
        raise RuntimeError("Boundary reached FP32 numerical limit; human review required")
    return {"selected": chosen, "extend_exponent": extra,
            "reason": "accuracy_winner_at_boundary_extend_one_decade" if extra is not None else "accuracy_winner_bracketed_freeze_lambda",
            "ranking": "highest_mean_Dice_then_lower_mean_ASD_then_lower_mean_HD95; exact_ties_prefer_power_nearest0"}
