"""P74 refined B-spline SVF sweep; P73 numerical sources remain immutable."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import bspline_svf as pilot

c, d = pilot.c, pilot.d
VARIANT = "oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1"
LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
METHOD = "direct_cp_multistage_bspline_svf_refined"
PROTOCOL = {**pilot.PROTOCOL, "version": 1, "formulation": "refine",
            "stage_alpha": "one swept lambda at every active stage", "lambdas": list(LAMBDAS),
            "ranks": list(range(10)), "independent_initialization": True,
            "composed_velocity_penalty": "not_applicable", "offgrid_points": 0,
            "integration_diagnostics_steps": [7, 8, 10],
            "case_placement": "case_affinity_on_available_gpu_with_one_new_process_per_gpu",
            "timing": "shared_load_optimizer_worker_and_independent_audit_separate",
            "memory": "process_torch_peak_allocated_and_reserved_GiB_separate_from_device_load"}


@dataclass(frozen=True)
class Spec:
    case_rank: int
    schedule: str
    regularize: str
    alpha: float
    smoke: bool = False
    formulation: str = "refine"
    integration_steps: int = 7

    def validate(self):
        if type(self.case_rank) is not int or self.case_rank not in range(10):
            raise ValueError("Require shuffled first10 rank0..9")
        if self.schedule not in ("842", "8421") or self.regularize not in ("velocity", "displacement"):
            raise ValueError("Require842/8421 and velocity/displacement regularization")
        if not math.isfinite(self.alpha) or self.alpha not in LAMBDAS:
            raise ValueError("Require one of the seven approved lambda values")
        if self.formulation != "refine" or self.integration_steps != 7:
            raise ValueError("Only a refined single SVF with seven integration steps is authorized")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return ("smoke_" if self.smoke else "") + f"rank{self.case_rank:03d}_cps{self.schedule}_{self.regularize}_lambda{c.format_float(self.alpha)}_ss7_{digest}"


def matrix(smoke=False):
    if smoke:
        return [Spec(0, "8421", reg, alpha, True) for reg in ("velocity", "displacement") for alpha in (.01, 100.)]
    return [Spec(rank, schedule, reg, alpha) for rank in range(10)
            for alpha in (1., .01, .1, .3, 3., 10., 100.)
            for reg in ("velocity", "displacement") for schedule in ("842", "8421")]


def stages(spec):
    spec.validate()
    return d.stages(d.FitSpec(spec.case_rank, "pyramid", spec.alpha, spec.schedule, smoke=spec.smoke))


class Objective(pilot.Objective):
    def __init__(self, fixed, moving, stage, spec):
        super().__init__(fixed, moving, stage, spec)
        self.alpha = spec.alpha


exp_velocity = pilot.exp_velocity
reconstruct = pilot.reconstruct
inverse_consistency = pilot.inverse_consistency
optimize = pilot.optimize
