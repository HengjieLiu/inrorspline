"""P65: isolated cps16 extension; P64 numerical helpers remain immutable."""
from functools import lru_cache
from itertools import product
import math

import numpy as np

import bounded_composed_bspline as original

c, d = original.c, original.d
CPS = (16, 8, 4, 2, 1)
ALPHAS = (1., 1., 1., 5., 1.)
MULTIPLIER = .99
SCHEDULE = {
    "stage_cps": list(CPS), "stage_lr": [.03, .03, .01, .003, .001],
    "stage_max_epochs": [400, 400, 800, 1300, 2000],
    "stage_min_epochs": [40, 40, 80, 130, 300],
    "stage_requested_patience": [20, 20, 35, 50, 100],
    "pyramid_factors": [8, 4, 2, 1, 1],
    "stage_max_lr_reductions": [0, 0, 0, 0, 2],
}
capped_controls = original.capped_controls
compose = original.compose
sample_displacement = original.sample_displacement
optimize = original.optimize
corner_diagnostics = original.corner_diagnostics
offgrid_diagnostics = original.offgrid_diagnostics
derivative_stencil = original.derivative_stencil


def stages(smoke=False):
    return [d.StageSpec(
        index=i, cps=cps, factor=SCHEDULE["pyramid_factors"][i],
        lr=SCHEDULE["stage_lr"][i],
        max_epochs=1 if smoke else SCHEDULE["stage_max_epochs"][i],
        min_epochs=1 if smoke else SCHEDULE["stage_min_epochs"][i],
        requested_patience=SCHEDULE["stage_requested_patience"][i],
        max_lr_reductions=2 if cps == 1 else 0,
    ) for i, cps in enumerate(CPS)]


def control_shapes(image_shape=c.IMAGE_SHAPE):
    shapes = c.build_nested_cp_shapes(image_shape, 16, CPS)
    c.assert_nested_schedule(shapes, CPS)
    return shapes


@lru_cache(maxsize=None)
def compute_bound(cps, sampling="endpoint_aligned"):
    if cps not in CPS or sampling not in ("endpoint_aligned", "sitreg_half_voxel"):
        raise ValueError("Unsupported bound geometry")
    if cps != 16:
        return original.compute_bound(cps, sampling)
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    worst = -math.inf
    witness = None
    for phase in product(range(cps), repeat=3):
        for corner in product((0, 1), repeat=3):
            derivatives = derivative_stencil(cps, phase, corner, sampling)
            values = np.abs(signs @ derivatives.reshape(3, -1)).sum(axis=1)
            index = int(values.argmax())
            if values[index] > worst:
                worst = float(values[index])
                witness = {"phase_zyx": list(phase), "corner_zyx": list(corner),
                           "signs_zyx": signs[index].tolist()}
    return {"cps": cps, "sampling": sampling, "K_control_units": worst,
            "bound_control_units": 1 / worst, "multiplier": MULTIPLIER,
            "cap_native_voxel": MULTIPLIER * cps / worst,
            "calculation_dtype": "float64", "phase_count": cps**3,
            "corners": 8, "sign_patterns": 8, "witness": witness}


class Objective(original.Objective):
    """Identical P64 objective, with explicit base16 five-stage control geometry."""

    def __init__(self, fixed, moving, stage, alpha, frozen_fields, bound):
        if stage.cps not in CPS or bound["cps"] != stage.cps:
            raise ValueError("Stage/bound mismatch")
        self.stage, self.alpha, self.bound = stage, float(alpha), bound
        self.image_shape = tuple(fixed.shape)
        self.control_shape = control_shapes(self.image_shape)[stage.cps]
        self.expand = d.AlignedBSpline(self.control_shape, self.image_shape,
                                      self.image_shape, stage.cps, fixed.device)
        self.native_scale = d.voxel_scale(self.image_shape, device=fixed.device)
        self.cap_norm = bound["cap_native_voxel"] / self.native_scale
        self.frozen = tuple(field.detach() for field in frozen_fields)
        self.fixed, self.blur_sigma, self.blur_kernel_size = c.make_loss_volume(fixed, stage.factor, .5, 3.)
        self.moving, _, _ = c.make_loss_volume(moving, stage.factor, .5, 3.)
        self.eval_shape = tuple(self.fixed.shape)
        self.identity = c.make_coordinate_grid(self.eval_shape, fixed.device)
        self.scale = d.voxel_scale(self.eval_shape, device=fixed.device)
        self.ncc = d.NCC9(self.fixed[None, None])
