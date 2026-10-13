"""Cohort and scheduling helpers for the OASIS first-100 NCC9 follow-up."""
from __future__ import annotations

import math

import dl_ncc_diffusion as d

VARIANT = "oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1"
LAMBDAS = (0.1, 0.3, 1.0, 3.0)
GPU_SLOTS = ("0", "2", "3", "3")
MEMORY_RESERVE_MIB_PER_WORKER = 4096


def install_first100_identity():
    """Widen only the accepted first10 worker's case-rank guard.

    The numerical protocol and all historical source files remain untouched;
    this adapter constrains the new launcher to seed-3 shuffled ranks 0–99.
    """
    d.VARIANT = VARIANT
    d.FitSpec.validate = validate_first100
    return d


def validate_first100(spec):
    if spec.case_rank not in range(100) or spec.resolution not in d.RESOLUTIONS:
        raise ValueError("Require seed-3 shuffled first100 rank and pyramid/full resolution")
    if spec.representation not in d.REPRESENTATIONS:
        raise ValueError("Invalid representation")
    if not math.isfinite(spec.alpha) or spec.alpha not in d.LAMBDAS:
        raise ValueError("Require a prespecified diffusion lambda")
    if spec.seed != 1:
        raise ValueError("This experiment fixes seed1")


class First100FitSpec(d.FitSpec):
    def validate(self):
        validate_first100(self)

    def prefix(self):
        return First100FitSpec(self.case_rank, self.resolution, self.alpha, "842", self.seed, self.smoke)


def specs(*, smoke=False):
    """Return pyramid prefixes and dense branches; production is 800 jobs."""
    ranks = range(1) if smoke else range(100)
    lambdas = (1.0,) if smoke else LAMBDAS
    result = []
    for rank in ranks:
        for alpha in lambdas:
            result.append(First100FitSpec(rank, "pyramid", alpha, "842", smoke=smoke))
            result.append(First100FitSpec(rank, "pyramid", alpha, "842_dense", smoke=smoke))
    return result


def validate_gpu_slots(value):
    slots = tuple(part.strip() for part in value.split(",") if part.strip())
    if slots != GPU_SLOTS:
        raise ValueError("This approved placement is exactly one worker on GPU0, one on GPU2, and two on GPU3")
    return slots


def required_memory_mib(smoke_peak_gib):
    measured = int(math.ceil(float(smoke_peak_gib) * 1024.0)) + 2048
    return max(MEMORY_RESERVE_MIB_PER_WORKER, measured)
