"""Approved rank4-only cps16 extension of the immutable P43 numerical protocol."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import common as c
import dl_ncc_diffusion as d

VARIANT = "oasis_seed3_rank004_cps16842_dense_ncc9_lambda1_v1"
METHOD = "direct_cp_16_8_4_2_then_dense_ncc9_diffusion"
PROTOCOL = {
    **d.PROTOCOL,
    "version": 1,
    "stage_cps": [16, 8, 4, 2, None],
    "stage_lr": [.03, .03, .01, .003, .001],
    "stage_max_epochs": [400, 400, 800, 1300, 2000],
    "stage_min_epochs": [40, 40, 80, 130, 300],
    "stage_requested_patience": [20, 20, 35, 50, 100],
    "pyramid_factors": [8, 4, 2, 1, 1],
    "full_factors": [1, 1, 1, 1, 1],
    "alpha": 1., "case_rank": 4, "fixed_id": "0084", "moving_id": "0108",
    "seed": 1, "physical_gpu": "3", "execution": "strictly_sequential",
    "initialization": "zero_cps16_then_dyadic_knot_insertion_then_exact_dense_clone",
    "stage_diagnostics": "native_field_and_canonical_metrics_after_best_loss_selection",
}
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg", "diffusion")


@dataclass(frozen=True)
class FitSpec:
    resolution: str
    smoke: bool = False
    case_rank: int = 4
    alpha: float = 1.
    seed: int = 1

    def validate(self):
        if self.resolution not in ("pyramid", "full"):
            raise ValueError("Require pyramid or full loss evaluation")
        if (self.case_rank, self.alpha, self.seed) != (4, 1., 1):
            raise ValueError("Only approved rank4, lambda1, seed1 are permitted")

    def name(self):
        self.validate()
        signature = {"method": METHOD, "spec": asdict(self), "protocol": PROTOCOL}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
        factors = "-".join(str(s.factor) for s in stages(self))
        return (f"{'smoke_' if self.smoke else ''}rank004_cps16-8-4-2-dense"
                f"_{self.resolution}_ds{factors}_lambda1_seed1_{digest}")


def stages(spec):
    spec.validate()
    factors = PROTOCOL[f"{spec.resolution}_factors"]
    return [d.StageSpec(
        index=i, cps=cps, factor=factors[i], lr=PROTOCOL["stage_lr"][i],
        max_epochs=1 if spec.smoke else PROTOCOL["stage_max_epochs"][i],
        min_epochs=1 if spec.smoke else PROTOCOL["stage_min_epochs"][i],
        requested_patience=PROTOCOL["stage_requested_patience"][i],
        max_lr_reductions=2 if cps is None else 0,
    ) for i, cps in enumerate(PROTOCOL["stage_cps"])]


def control_shapes():
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, 16, [16, 8, 4, 2])
    c.assert_nested_schedule(shapes, [16, 8, 4, 2])
    expected = {16: (14, 18, 16), 8: (24, 32, 28), 4: (44, 60, 52), 2: (84, 116, 100)}
    if shapes != expected:
        raise RuntimeError(f"Unexpected native control geometry: {shapes}")
    return shapes
