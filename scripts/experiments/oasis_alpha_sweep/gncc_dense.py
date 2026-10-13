"""Population GNCC objectives; unchanged P43/P44 geometry and stage optimizer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

import common as c
import dl_ncc_diffusion as base
from dl_loss_sweep import Objective, optimize_stage
from run_oasis_dl_loss_one import source_hashes as legacy_hashes
from run_oasis_saved_field_bspline_fit_one import sha256_file

VARIANT = "oasis_seed3_first10_mrdbscp842_dense_fullres_gncc_signed_squared_diffusion_v1"
LOSSES = ("gncc", "gncc2")
LAMBDAS = base.LAMBDAS
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg", "diffusion")
METHODS = {key: f"direct_cp_8_4_2_then_dense_{key}_diffusion" for key in (*LOSSES, "lncc")}
PROTOCOL = {**base.PROTOCOL, "version": 1, "loss": "population_global_1rho_or_1rho_squared_plus_lambda_diffusion",
            "resolution": "full", "stage_downsample": [1, 1, 1, 1], "training_mask": False,
            "prefix_reuse": False, "similarity_scale": 1., "window": None,
            "correlation": "centered_population_covariance_over_sqrt_clamped_population_variance_product",
            "variance_product_epsilon": float(torch.finfo(torch.float32).eps),
            "backward": "ordinary_autograd_no_StableStd", "calibration": "diagnostic_only_no_training_rescaling"}


@dataclass(frozen=True)
class FitSpec:
    case_rank: int
    loss: str
    alpha: float
    seed: int = 1
    smoke: bool = False

    def __post_init__(self):
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def scale(self):
        return 1.

    def validate(self):
        if self.case_rank not in range(10) or self.loss not in (*LOSSES, "lncc"):
            raise ValueError("Require first10 rank and GNCC/GNCC2/LNCC-control")
        if self.alpha not in LAMBDAS or self.seed != 1:
            raise ValueError("Require prescribed lambda and seed1")
        if self.loss == "lncc" and (self.case_rank != 0 or self.alpha != 1):
            raise ValueError("New LNCC is qualification-only rank0/lambda1")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return f"{'smoke_' if self.smoke else ''}rank{self.case_rank:03d}_{self.loss}_lambda{self.alpha:g}_{digest}"


def specs(smoke=False):
    if smoke:
        return [FitSpec(0, key, 1., smoke=True) for key in (*LOSSES, "lncc")]
    return [FitSpec(rank, key, alpha) for alpha in (1., .3, 3., .1, 10., .01, 100.)
            for rank in range(10) for key in LOSSES]


def stages(spec):
    spec.validate()
    return base.stages(base.FitSpec(spec.case_rank, "full", spec.alpha, "842_dense", spec.seed, spec.smoke))


class GNCC:
    def __init__(self, fixed, squared=False):
        self.fixed = fixed.detach()
        self.centered = self.fixed-self.fixed.mean()
        self.variance = self.centered.square().mean()
        self.squared = squared

    def correlation(self, warped):
        if warped.shape != self.fixed.shape or warped.dtype != self.fixed.dtype:
            raise ValueError("GNCC shape/dtype mismatch")
        centered = warped-warped.mean()
        covariance = (self.centered*centered).mean()
        denominator = (self.variance*centered.square().mean()).clamp_min(PROTOCOL["variance_product_epsilon"]).sqrt()
        return covariance/denominator

    def __call__(self, warped):
        rho = self.correlation(warped)
        return 1-rho.square() if self.squared else 1-rho


def similarity(key, fixed):
    if key == "lncc":
        return base.NCC9(fixed)
    if key not in LOSSES:
        raise ValueError(key)
    return GNCC(fixed, squared=key == "gncc2")


def source_hashes():
    result = legacy_hashes()
    here = Path(__file__).resolve().parent
    for name in ("gncc_dense.py", "run_oasis_gncc_dense_one.py", "launch_oasis_gncc_dense.py", "calibrate_oasis_gncc_dense.py",
                 "run_oasis_sitreg_first100_eval.py"):
        result[str(here/name)] = sha256_file(here/name)
    report = here.parents[1]/"reports/oasis_gncc_dense_report.py"
    result[str(report)] = sha256_file(report)
    return result


def verify_source_hashes(expected):
    if source_hashes() != expected:
        raise RuntimeError("Numerical source drift during experiment")


def validate_result(path, spec, expected_hashes):
    import numpy as np
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing completion marker")
    row = c.read_json(path/"summary.json")
    for key, value in (("spec", asdict(spec)), ("protocol", PROTOCOL), ("source_hashes", expected_hashes),
                       ("method", METHODS[spec.loss])):
        if row[key] != value:
            raise RuntimeError(f"Invariant failed: {key}")
    if (row["fixed_id"], row["moving_id"]) != c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec.case_rank] or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Case/direction mismatch")
    if row["representation"] != "dense_voxel" or any(row[k] is not None for k in ("cps", "control_shape", "bspline_order")):
        raise RuntimeError("Final field must be dense, not cps1 controls")
    if row["parameter_units"] != "normalized_grid_xyz" or row["objective_support"] != "full_native_volume":
        raise RuntimeError("Wrong units/support")
    if (row["uid"], row["gid"]) != (1009, 1020) or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("UID/precision changed")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"]:
        raise RuntimeError("Backend changed")
    if len(row["stages"]) != 4 or not all(math.isfinite(row[k]) for k in (*METRICS, "best_loss")):
        raise RuntimeError("Missing stage or nonfinite result")
    for actual, stage in zip(row["stages"], stages(spec)):
        if actual["stage"] != asdict(stage) or actual["objective_voxel_count"] != math.prod(c.IMAGE_SHAPE):
            raise RuntimeError("Stage schedule/support changed")
        if actual["alpha"] != spec.alpha or actual["similarity_scale"] != 1 or actual["loss_key"] != spec.loss:
            raise RuntimeError("Loss coefficients changed")
    if row["initialization_checks"][-1]["max_abs_voxel"] != 0:
        raise RuntimeError("Dense initialization changed")
    for filename, digest in row["outputs_sha256"].items():
        if sha256_file(path/filename) != digest:
            raise RuntimeError(f"Output changed: {filename}")
    field = np.load(row["dvf_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid field")
    labels = c.read_json(path/"per_label.json")
    if sorted(r["label_id"] for r in labels) != list(range(1, 36)):
        raise RuntimeError("Missing label records")
    for metric, key in (("dice", "dice"), ("hd95", "hd95_mm"), ("asd", "asd_symmetric_mm")):
        valid = [r[key] for r in labels if r["included_in_case_mean"]]
        if not valid or not math.isclose(float(np.mean(valid)), row[metric], abs_tol=1e-12):
            raise RuntimeError("Per-label reconstruction failed")
    return row
