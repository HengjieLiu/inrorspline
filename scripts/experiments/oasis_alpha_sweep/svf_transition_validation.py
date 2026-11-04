"""P74 validation-only recovery; never changes optimizer values or gradients."""
from __future__ import annotations

import torch
import bspline_svf as s

POLICY = {
    "version": 1,
    "field_max_abs_voxel": 1e-3,
    "field_rms_voxel": 1e-5,
    "fp64_velocity_max_abs_voxel": 1e-9,
    "fp64_field_max_abs_voxel": 1e-8,
    "original_velocity_gate_unchanged": True,
    "fallback_requires_fp64_knot_and_integration_replay": True,
}


def error_stats(a, b, scale):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Nonfinite or incompatible transition fields")
    delta = (a.double()-b.double())*scale.double()
    return {"max_abs_voxel": float(delta.abs().max()),
            "rms_voxel": float(delta.square().mean().sqrt())}


def within_physical_tolerance(stats):
    return (stats["max_abs_voxel"] <= POLICY["field_max_abs_voxel"]
            and stats["rms_voxel"] <= POLICY["field_rms_voxel"])


@torch.no_grad()
def fp64_reference(previous_controls, controls, cps, shape, scale):
    """Recompute exact knot insertion and the spline kernel in double precision.

    Also verify actual FP32 controls equal the prescribed knot-insertion output;
    high precision is diagnostic only and is never passed to the optimizer.
    """
    expected = s.c.cubic_bspline_dyadic_knot_insert_3d(previous_controls, controls.shape[:3])
    if not torch.equal(expected, controls):
        raise RuntimeError("Transition controls differ from prescribed knot insertion")
    coarse = previous_controls.double()
    fine = s.c.cubic_bspline_dyadic_knot_insert_3d(coarse, controls.shape[:3])
    velocities = []
    for spacing, cp in ((2*cps, coarse), (cps, fine)):
        expand = s.d.AlignedBSpline(cp.shape[:3], shape, shape, spacing, cp.device)
        expand.kernel = s.c.cubic_bspline1d(spacing, dtype=torch.float64, device=cp.device)
        velocities.append(expand(cp))
    vd = error_stats(*velocities, scale)
    grid = s.c.make_coordinate_grid(shape, controls.device).double()
    a = s.exp_velocity(velocities[0], 7, grid)
    b = s.exp_velocity(velocities[1], 7, grid)
    ud = error_stats(a, b, scale)
    if (vd["max_abs_voxel"] > POLICY["fp64_velocity_max_abs_voxel"]
            or ud["max_abs_voxel"] > POLICY["fp64_field_max_abs_voxel"]):
        raise RuntimeError("FP64 replay does not preserve spline velocity/exponential")
    return {"velocity": vd, "integrated_field": ud, "passed": True}


@torch.no_grad()
def verify_transition(previous_controls, controls, previous_velocity, velocity,
                      previous_field, field, cps, scale):
    vd = error_stats(velocity, previous_velocity, scale)
    ud = error_stats(field, previous_field, scale)
    if not torch.allclose(velocity, previous_velocity, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Knot insertion changed velocity")
    old_gate = torch.allclose(field, previous_field, rtol=1e-4, atol=2e-6)
    result = {"policy": POLICY, "velocity": vd, "integrated_field": ud,
              "original_field_gate_passed": old_gate, "fp64_reference": None}
    if not old_gate:
        if not within_physical_tolerance(ud):
            raise RuntimeError(f"Material integrated-field change in voxel units: {ud}")
        result["fp64_reference"] = fp64_reference(previous_controls, controls, cps,
                                                 field.shape[:3], scale)
    result["passed"] = True
    return result


def main():
    import argparse
    from pathlib import Path
    import numpy as np
    import run_oasis_bspline_svf_sweep as runner
    parser = argparse.ArgumentParser(description="Read-only replay of the preserved failed transition")
    parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    args = parser.parse_args()
    root = args.artifact_root/"runs"/runner.s.VARIANT
    out = root/"recovery_v1"
    if (out/"transition_fixture.json").exists():
        raise RuntimeError("Preserve existing fixture audit")
    manifest = s.c.read_json(root/"launch_manifest.json")
    spec = runner.s.Spec(3, "842", "velocity", .01)
    device, batch, _ = runner.setup(args, manifest, spec)
    previous_dir = root/spec.name()/"attempt0/stage1_cps4"
    path = previous_dir/"best.pt"
    cp = torch.load(path, map_location=device, weights_only=False)["velocity_controls_norm_xyz"]
    coarse_obj = runner.s.Objective(batch["fixed"], batch["moving"], runner.s.stages(spec)[1], spec)
    fine_obj = runner.s.Objective(batch["fixed"], batch["moving"], runner.s.stages(spec)[2], spec)
    with torch.no_grad():
        fine = s.c.cubic_bspline_dyadic_knot_insert_3d(cp, fine_obj.control_shape)
        a, b = coarse_obj.field(cp, native=True), fine_obj.field(fine, native=True)
        saved = np.load(previous_dir/"disp_norm_0117_0322.npy")
        if not np.array_equal(a.cpu().numpy(), saved):
            raise RuntimeError("Saved pre-transition field does not replay exactly")
        proof = verify_transition(cp, fine, coarse_obj.expand(cp), fine_obj.expand(fine),
                                  a, b, 2, fine_obj.native_scale)
    out.mkdir(exist_ok=True)
    old_metrics = runner.metric_row(a, batch, out/"fixture_coarse_metrics")
    new_metrics = runner.metric_row(b, batch, out/"fixture_prolonged_metrics")
    proof.update(checkpoint_path=str(path), checkpoint_sha256=runner.sha(path),
                 metrics_before=old_metrics, metrics_after=new_metrics,
                 metric_deltas={k: new_metrics[k]-old_metrics[k] for k in runner.METRICS},
                 created_at=s.c.now_text(), source_sha256=runner.sha(__file__),
                 saved_field_exact_replay=True, gpu=args.gpu)
    runner.atomic_json(out/"transition_fixture.json", proof)
    import json
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
