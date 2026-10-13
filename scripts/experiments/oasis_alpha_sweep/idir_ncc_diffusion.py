"""Isolated full-volume IDIR SIREN frequency/diffusion experiment contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import dl_ncc_diffusion as d

VARIANT = "oasis_seed3_first10_idir_siren_omega_ncc9_diffusion_2500_v1"
METHOD = "idir_siren_fullres_ncc9_voxel_diffusion"
OMEGAS = (8, 16, 32, 64, 128)
LAMBDAS = {"0.1": .1, "1over3": 1 / 3, "1": 1., "3": 3., "10": 10.}
METRICS = ("dice_mean", "hd95_mean", "asd_symmetric_mean", "log_jdet_std_fg", "jdet_le0_pct_fg")
PROTOCOL = {
    "method": METHOD, "shape_dhw": [160, 224, 192], "coordinate_order": "xyz",
    "parameterization": "siren_weights_normalized_xyz_output", "layers": [3, 256, 256, 256, 3],
    "initialization": "native_IDIR_first_bound_1_over_fanin_later_sqrt6_over_fanin_over_omega_bias_unchanged",
    "fourier_encoding": False, "spline_initialization": False,
    "seed": 1, "adam_lr": 1e-4, "adam_betas": [.9, .999], "adam_eps": 1e-8,
    "adam_amsgrad": False, "adam_weight_decay": 0., "updates": 2500,
    "scheduler": None, "early_stopping": False, "primary_checkpoint": "terminal_2500",
    "diagnostic_checkpoint": "minimum_total_loss_including_initialization",
    "image_preprocessing": "stored_raw_float32_LIA_to_RAS_no_intensity_normalization",
    "similarity": "NCC_vfa9_1_minus_mean_squared_local_correlation",
    "ncc_padding": "zero4_divide729", "ncc_variance_clamp": "float32_epsilon",
    "image_sampling": "trilinear_border_align_corners_true",
    "training_support": "full_volume", "objective_voxel_count": 160 * 224 * 192,
    "regularization": "total_voxel_xyz_forward_valid_diffusion_component_and_axis_mean",
    "regularization_mask": None, "loss_normalization": None,
    "backprop": "full_field_loss_then_chunked_network_vector_jacobian_product_one_adam_step",
    "chunk_size": 262144, "precision": "float32", "cudnn_benchmark": False,
    "cudnn_deterministic": True, "tf32": False,
    "direction": "moving_to_fixed", "sampling_map": "fixed_to_moving_pull_identity_plus_displacement",
    "labels": list(range(1, 36)), "absent_labels": "exclude_if_absent_in_fixed_moving_or_warped",
    "asd": "pooled_surfel_area_weighted_symmetric_mm", "foreground": "fixed_raw_image_positive",
    "stdlogj": "LUMIR_crop2_population_std_natural_log_positive_jacobian_only",
    "folding": "LUMIR_crop2_nonpositive_jacobian_percentage",
}


@dataclass(frozen=True)
class FitSpec:
    rank: int
    omega: int
    lambda_key: str
    smoke_steps: int = 0

    def validate(self):
        if self.rank not in range(10) or self.omega not in OMEGAS or self.lambda_key not in LAMBDAS:
            raise ValueError("Require first10 rank, approved omega and exact lambda key")
        if self.smoke_steps not in (0, 1, 3, 10):
            raise ValueError("Only production2500 or explicitly separate smoke1/3/10 is allowed")

    @property
    def alpha(self):
        return LAMBDAS[self.lambda_key]

    @property
    def updates(self):
        return self.smoke_steps or PROTOCOL["updates"]

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return (f"{'smoke' + str(self.smoke_steps) + '_' if self.smoke_steps else ''}"
                f"rank{self.rank:03d}_omega{self.omega}_lambda{self.lambda_key}_seed1_{digest}")


def specs():
    # Rank-major gives an early complete frequency/weight comparison on rank0.
    return [FitSpec(rank, omega, key) for rank in range(10) for omega in OMEGAS for key in LAMBDAS]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    names = ("idir_ncc_diffusion.py", "run_oasis_idir_ncc_diffusion_one.py", "launch_oasis_idir_ncc_diffusion_sweep.py",
             "common.py", "dl_ncc_diffusion.py", "run_oasis_dl_ncc_diffusion_one.py",
             "run_oasis_saved_field_bspline_fit_one.py", "run_oasis_vfa_feature_dense_one.py",
             "vfa_feature_refinement.py", "run_one.py", "run_oasis_sitreg_first100_eval.py",
             "sitreg_training.py", "smoke_dl_training_memory.py", "evaluate_saved_baselines_first10.py")
    files = [Path(__file__).with_name(name) for name in names]
    files.append(c.REPO_ROOT / "scripts/reports/oasis_dl_ncc_diffusion_first10_arc.py")
    return {str(path): sha256(path) for path in files}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source/input changed: {path}")


class FullVolumeObjective:
    def __init__(self, fixed, moving, alpha):
        if fixed.shape != moving.shape or fixed.ndim != 3 or fixed.dtype != torch.float32:
            raise ValueError("Require matching full FP32 DHW images")
        self.shape = tuple(fixed.shape)
        self.identity = c.make_coordinate_grid(self.shape, fixed.device)
        self.coords = self.identity.reshape(-1, 3)
        self.scale = d.voxel_scale(self.shape, device=fixed.device)
        self.moving, self.alpha = moving, float(alpha)
        self.ncc = d.NCC9(fixed[None, None])

    def __call__(self, q):
        if q.shape != (*self.shape, 3):
            raise ValueError("Wrong normalized displacement shape")
        warped = F.grid_sample(self.moving[None, None], (self.identity + q)[None],
                               mode="bilinear", padding_mode="border", align_corners=True)
        similarity = self.ncc(warped)
        reg = c.forward_valid_diffusion((q * self.scale).permute(3, 0, 1, 2)[None], None)
        return similarity + self.alpha * reg, similarity, reg


@torch.no_grad()
def dense_field(model, objective, chunk_size=PROTOCOL["chunk_size"]):
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    field = torch.empty_like(objective.coords)
    for start in range(0, len(field), chunk_size):
        field[start:start + chunk_size] = model(objective.coords[start:start + chunk_size])
    return field.reshape(*objective.shape, 3)


def full_volume_gradients(model, objective, chunk_size=PROTOCOL["chunk_size"]):
    """Exact chain rule, bounded network activation storage, NO per-chunk loss."""
    q = dense_field(model, objective, chunk_size).requires_grad_(True)
    total, data, reg = objective(q)
    values = {"loss": float(total.detach()), "ncc9": float(data.detach()), "diffusion": float(reg.detach())}
    if not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError(f"Nonfinite full-volume objective: {values}")
    gradient, = torch.autograd.grad(total, q)
    del q, total, data, reg
    gradient = gradient.reshape(-1, 3)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite displacement gradient")
    model.zero_grad(set_to_none=True)
    for start in range(0, len(gradient), chunk_size):
        chunk = model(objective.coords[start:start + chunk_size])
        chunk.backward(gradient[start:start + chunk_size])
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("Missing/nonfinite SIREN parameter gradient")
    return values


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=PROTOCOL["adam_lr"], betas=tuple(PROTOCOL["adam_betas"]),
                            eps=PROTOCOL["adam_eps"], weight_decay=0., amsgrad=False)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def cuda_chunk_parity(omega, device):
    """Small-grid GPU parity gate; not a registration or hyperparameter fit."""
    import copy
    torch.manual_seed(11)
    shape = (5, 6, 7)
    objective = FullVolumeObjective(torch.rand(shape, device=device), torch.rand(shape, device=device), 1 / 3)
    a = c.build_siren(32, 3, omega).to(device)
    b = copy.deepcopy(a)
    opt_a, opt_b = optimizer(a), optimizer(b)
    values = objective(a(objective.coords).reshape(*shape, 3))
    values[0].backward()
    actual = full_volume_gradients(b, objective, chunk_size=47)
    difference = abs(actual["loss"] - float(values[0]))
    max_gradient = max(float((x.grad - y.grad).abs().max()) for x, y in zip(a.parameters(), b.parameters()))
    if difference > 2e-6 or any(not torch.allclose(x.grad, y.grad, atol=3e-6, rtol=5e-4) for x, y in zip(a.parameters(), b.parameters())):
        raise RuntimeError(f"GPU chunk gradient parity failed: loss{difference}, grad{max_gradient}")
    opt_a.step(); opt_b.step()
    max_update = max(float((x - y).abs().max()) for x, y in zip(a.parameters(), b.parameters()))
    if any(not torch.allclose(x, y, atol=2e-6, rtol=2e-4) for x, y in zip(a.parameters(), b.parameters())):
        raise RuntimeError(f"GPU chunk Adam parity failed: {max_update}")
    return {"passed": True, "loss_abs": difference, "gradient_max_abs": max_gradient, "parameter_max_abs_after_adam": max_update}


def validate_label_records(records, summary):
    if len(records) != 35 or {r["label_id"] for r in records} != set(range(1, 36)):
        raise ValueError("Require exactly35 segmentation records")
    for key, summary_key in (("dice", "dice_mean"), ("hd95_mm", "hd95_mean"), ("asd_symmetric_mm", "asd_symmetric_mean")):
        values = []
        for row in records:
            value = row[key]
            if row["included_in_case_mean"]:
                if value is None or not math.isfinite(value):
                    raise ValueError("Missing metric for present label")
                values.append(value)
            elif value is not None or not row["exclusion_reason"]:
                raise ValueError("Missing labels must be explicit nulls with reasons")
        if not values or not math.isclose(float(np.mean(values)), summary[summary_key], abs_tol=1e-12):
            raise ValueError("Per-label reconstruction disagrees with case mean")
