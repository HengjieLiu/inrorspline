"""Approved cps2->cps1 roundoff proof; old numerical sources remain immutable."""
from __future__ import annotations

import math
import torch
import svf_transition_validation as old

s = old.s
POLICY = {**old.POLICY, 'version': 2, 'scope': 'cps2_to_cps1_only',
          'field_max_abs_voxel': .002, 'field_rms_voxel': 1e-5,
          'rounded_controls_fp64_field_max_abs_voxel': 1e-4,
          'rounded_controls_fp64_field_rms_voxel': 1e-6,
          'exact_prescribed_fp32_controls_required': True,
          'optimization_precision_unchanged': True}


def require_controls(previous, controls):
    if not torch.isfinite(previous).all() or not torch.isfinite(controls).all():
        raise RuntimeError('Nonfinite velocity controls')
    expected = s.c.cubic_bspline_dyadic_knot_insert_3d(previous, controls.shape[:3])
    if not torch.equal(expected, controls):
        raise RuntimeError('Transition controls differ from prescribed FP32 knot insertion')


def check_physical(stats):
    if (not all(math.isfinite(x) and x >= 0 for x in stats.values()) or
            stats['max_abs_voxel'] > POLICY['field_max_abs_voxel'] or
            stats['rms_voxel'] > POLICY['field_rms_voxel']):
        raise RuntimeError(f'FP32 field change exceeds approved physical limits: {stats}')


def check_reference(proof):
    for key in ('velocity','integrated_field','rounded_controls_integrated_field'):
        if not all(math.isfinite(x) and x >= 0 for x in proof[key].values()):
            raise RuntimeError('Invalid FP64 proof statistics')
    if (proof['velocity']['max_abs_voxel'] > POLICY['fp64_velocity_max_abs_voxel'] or
            proof['integrated_field']['max_abs_voxel'] > POLICY['fp64_field_max_abs_voxel']):
        raise RuntimeError('Ideal FP64 knot/exponential preservation failed')
    rounded = proof['rounded_controls_integrated_field']
    if (rounded['max_abs_voxel'] > POLICY['rounded_controls_fp64_field_max_abs_voxel'] or
            rounded['rms_voxel'] > POLICY['rounded_controls_fp64_field_rms_voxel']):
        raise RuntimeError('Actual rounded controls fail FP64 field-preservation limits')


@torch.no_grad()
def fp64_reference(previous, controls, cps, shape, scale):
    require_controls(previous, controls)
    coarse = previous.double()
    ideal = s.c.cubic_bspline_dyadic_knot_insert_3d(coarse, controls.shape[:3])
    fields, velocities = [], []
    identity = s.c.make_coordinate_grid(shape, controls.device).double()
    for spacing, cp in ((2*cps, coarse), (cps, ideal), (cps, controls.double())):
        expand = s.d.AlignedBSpline(cp.shape[:3], shape, shape, spacing, cp.device)
        expand.kernel = s.c.cubic_bspline1d(spacing, dtype=torch.float64, device=cp.device)
        velocity = expand(cp)
        velocities.append(velocity)
        fields.append(s.exp_velocity(velocity, 7, identity))
    proof = {'velocity': old.error_stats(velocities[0], velocities[1], scale),
             'integrated_field': old.error_stats(fields[0], fields[1], scale),
             'rounded_controls_velocity': old.error_stats(velocities[0], velocities[2], scale),
             'rounded_controls_integrated_field': old.error_stats(fields[0], fields[2], scale),
             'control_rounding': old.error_stats(controls.double(), ideal, scale)}
    check_reference(proof)
    return {**proof, 'passed': True}


@torch.no_grad()
def verify_transition(previous_controls, controls, previous_velocity, velocity,
                      previous_field, field, cps, scale):
    if cps != 1:
        return old.verify_transition(previous_controls, controls, previous_velocity, velocity,
                                     previous_field, field, cps, scale)
    require_controls(previous_controls, controls)
    vd = old.error_stats(velocity, previous_velocity, scale)
    ud = old.error_stats(field, previous_field, scale)
    if not torch.allclose(velocity, previous_velocity, rtol=1e-5, atol=1e-7):
        raise RuntimeError('Knot insertion changed velocity')
    original_gate = bool(torch.allclose(field, previous_field, rtol=1e-4, atol=2e-6))
    proof = None
    if not original_gate:
        # Proof is mandatory, including the actual rounded controls, not just an ideal refinement.
        proof = fp64_reference(previous_controls, controls, cps, field.shape[:3], scale)
        check_reference(proof)
    check_physical(ud)
    return {'policy': POLICY, 'velocity': vd, 'integrated_field': ud,
            'original_field_gate_passed': original_gate, 'fp64_reference': proof, 'passed': True}
