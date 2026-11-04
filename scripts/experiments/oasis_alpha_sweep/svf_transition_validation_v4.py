"""P80 explicitly approved cps2->1 proof gate; no numerical optimization changes."""
from __future__ import annotations
import torch
import svf_transition_validation as old
from svf_transition_validation_v2 import require_controls
import svf_transition_validation_v3 as v3

POLICY = {**v3.POLICY, "version":4, "scope":"P80 matched-prefix cps2_to_cps1_only", "fine_cps":[1]}


@torch.no_grad()
def fp64_reference(previous_controls,controls,shape,scale):
    require_controls(previous_controls,controls)
    s=old.s; coarse=previous_controls.double()
    ideal=s.c.cubic_bspline_dyadic_knot_insert_3d(coarse,controls.shape[:3])
    identity=s.c.make_coordinate_grid(shape,controls.device).double()
    fields,velocities=[],[]
    for spacing,cp in ((2,coarse),(1,ideal),(1,controls.double())):
        expand=s.d.AlignedBSpline(cp.shape[:3],shape,shape,spacing,cp.device)
        expand.kernel=s.c.cubic_bspline1d(spacing,dtype=torch.float64,device=cp.device)
        velocity64=expand(cp);velocities.append(velocity64)
        fields.append(s.exp_velocity(velocity64,7,identity))
    proof={"velocity":old.error_stats(velocities[0],velocities[1],scale),
           "integrated_field":old.error_stats(fields[0],fields[1],scale),
           "rounded_controls_velocity":old.error_stats(velocities[0],velocities[2],scale),
           "rounded_controls_integrated_field":old.error_stats(fields[0],fields[2],scale),
           "control_rounding":old.error_stats(controls.double(),ideal,scale)}
    v3.check_reference(proof)
    return {**proof,"passed":True}


@torch.no_grad()
def verify_transition(previous_controls, controls, previous_velocity, velocity, previous_field, field, cps, scale):
    if cps != 1:
        raise ValueError("P80 only introduces cps2->cps1; preserve earlier policy provenance")
    require_controls(previous_controls, controls)
    vd, ud = old.error_stats(velocity,previous_velocity,scale), old.error_stats(field,previous_field,scale)
    if not torch.allclose(velocity,previous_velocity,rtol=1e-5,atol=1e-7):
        raise RuntimeError("Knot insertion changed velocity")
    v3.check_physical(ud)
    original = bool(torch.allclose(field,previous_field,rtol=1e-4,atol=2e-6))
    proof = None
    if not original:
        proof=fp64_reference(previous_controls,controls,field.shape[:3],scale)
        v3.check_reference(proof)
    return {"policy":POLICY,"velocity":vd,"integrated_field":ud,
            "original_field_gate_passed":original,"fp64_reference":proof,"passed":True}
