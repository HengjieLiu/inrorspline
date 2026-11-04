"""P79 approved proof-gated cps8/4/2 continuation; FP32 optimization unchanged."""
from __future__ import annotations

import math
import torch
import svf_transition_validation as old
from svf_transition_validation_v2 import require_controls

s = old.s
POLICY = {**old.POLICY, "version": 3, "scope": "P79 ranks29/44 velocity lambda0.01 continuations only",
          "fine_cps": [4, 2], "field_max_abs_voxel": .002, "field_rms_voxel": 1e-5,
          "rounded_controls_fp64_field_max_abs_voxel": 2e-4,
          "rounded_controls_fp64_field_rms_voxel": 1e-6,
          "exact_prescribed_fp32_controls_required": True, "optimization_precision_unchanged": True}


def finite_stats(stats):
    return all(math.isfinite(stats[k]) and stats[k] >= 0 for k in ("max_abs_voxel", "rms_voxel"))


def check_physical(stats):
    if (not finite_stats(stats) or stats["max_abs_voxel"] > POLICY["field_max_abs_voxel"]
            or stats["rms_voxel"] > POLICY["field_rms_voxel"]):
        raise RuntimeError(f"FP32 transition exceeds approved physical limits: {stats}")


def check_reference(proof):
    if not all(finite_stats(proof[k]) for k in ("velocity", "integrated_field", "rounded_controls_integrated_field")):
        raise RuntimeError("Invalid FP64 proof statistics")
    if (proof["velocity"]["max_abs_voxel"] > POLICY["fp64_velocity_max_abs_voxel"] or
            proof["integrated_field"]["max_abs_voxel"] > POLICY["fp64_field_max_abs_voxel"]):
        raise RuntimeError("Ideal FP64 knot/exponential preservation failed")
    rounded = proof["rounded_controls_integrated_field"]
    if (rounded["max_abs_voxel"] > POLICY["rounded_controls_fp64_field_max_abs_voxel"] or
            rounded["rms_voxel"] > POLICY["rounded_controls_fp64_field_rms_voxel"]):
        raise RuntimeError("Actual rounded controls fail FP64 preservation limits")


@torch.no_grad()
def fp64_reference(previous, controls, cps, shape, scale):
    if cps not in POLICY["fine_cps"]:
        raise ValueError("Only cps8->4 and cps4->2 are in scope")
    require_controls(previous, controls)
    coarse = previous.double()
    ideal = s.c.cubic_bspline_dyadic_knot_insert_3d(coarse, controls.shape[:3])
    identity = s.c.make_coordinate_grid(shape, controls.device).double()
    fields, velocities = [], []
    for spacing, cp in ((2*cps, coarse), (cps, ideal), (cps, controls.double())):
        expand = s.d.AlignedBSpline(cp.shape[:3], shape, shape, spacing, cp.device)
        expand.kernel = s.c.cubic_bspline1d(spacing, dtype=torch.float64, device=cp.device)
        velocity = expand(cp)
        velocities.append(velocity)
        fields.append(s.exp_velocity(velocity, 7, identity))
    proof = {"velocity": old.error_stats(velocities[0], velocities[1], scale),
             "integrated_field": old.error_stats(fields[0], fields[1], scale),
             "rounded_controls_velocity": old.error_stats(velocities[0], velocities[2], scale),
             "rounded_controls_integrated_field": old.error_stats(fields[0], fields[2], scale),
             "control_rounding": old.error_stats(controls.double(), ideal, scale)}
    check_reference(proof)
    return {**proof, "passed": True}


@torch.no_grad()
def verify_transition(previous_controls, controls, previous_velocity, velocity,
                      previous_field, field, cps, scale):
    if cps not in POLICY["fine_cps"]:
        raise ValueError("Transition outside approved continuation scope")
    require_controls(previous_controls, controls)
    vd = old.error_stats(velocity, previous_velocity, scale)
    ud = old.error_stats(field, previous_field, scale)
    if not torch.allclose(velocity, previous_velocity, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Knot insertion changed velocity")
    check_physical(ud)  # Enforced even when the older relative allclose passes.
    original_gate = bool(torch.allclose(field, previous_field, rtol=1e-4, atol=2e-6))
    proof = None
    if not original_gate:
        proof = fp64_reference(previous_controls, controls, cps, field.shape[:3], scale)
        check_reference(proof)
    return {"policy": POLICY, "velocity": vd, "integrated_field": ud,
            "original_field_gate_passed": original_gate, "fp64_reference": proof, "passed": True}
