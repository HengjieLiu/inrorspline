"""Read-only physical-unit and sampling diagnostics for symmetric SVFs."""
import numpy as np
import torch
import torch.nn.functional as F

import symmetric_bspline_svf as m


def swapped_batch(batch):
    result = dict(batch)
    for suffix in ("", "_seg", "_id", "_image_mask"):
        result["fixed"+suffix], result["moving"+suffix] = batch["moving"+suffix], batch["fixed"+suffix]
    result["fixed_mask"] = result["fixed_image_mask"]
    result["fixed_seg_mask"] = result["fixed_seg"] > 0
    result["mask_audit"] = {"foreground_rule": "current fixed raw image > 0", "swapped": True}
    return result


def summary(errors):
    if not errors.numel():
        return {"mean_mm": None, "rms_mm": None, "p95_mm": None, "max_mm": None}
    return {"mean_mm": float(errors.mean()), "rms_mm": float(errors.square().mean().sqrt()),
            "p95_mm": float(errors.quantile(.95)), "max_mm": float(errors.max())}


def points_from_mask(mask, count=100000, seed=1):
    rng = np.random.default_rng(seed)
    fg = mask.cpu().numpy().astype(bool)
    pool = np.flatnonzero(fg)
    if not pool.size:
        raise ValueError("Empty foreground")
    chosen = rng.choice(pool, count, replace=pool.size<count)
    zyx = np.stack(np.unravel_index(chosen, fg.shape), -1).astype(np.float64)
    zyx += rng.uniform(-.5, .5, zyx.shape)
    zyx = np.clip(zyx, 0., np.array(fg.shape)-1)
    scale = m.d.voxel_scale(fg.shape, device=mask.device)
    return torch.tensor(zyx[:, ::-1].copy(), dtype=torch.float32, device=mask.device)/scale-1


def apply_chain(parts, points):
    mapped = points
    valid = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for field in reversed(parts):
        mapped = mapped+m.s.sample_displacement(field, mapped)
        valid &= (mapped.abs() <= 1).all(-1)
    return mapped, valid


@torch.no_grad()
def offgrid(reconstruction, batch, count=100000):
    scale = m.d.voxel_scale(batch["fixed"].shape, device=batch["fixed"].device)
    results = {"count_per_direction": count, "seed": 1,
               "support": "jittered_raw_positive_foreground_voxels_clipped_to_native_domain"}
    for direction, opposite, mask in (("forward", "reverse", batch["fixed"] > 0),
                                      ("reverse", "forward", batch["moving"] > 0)):
        points = points_from_mask(mask, count)
        mapped, valid1 = apply_chain(reconstruction[direction+"_parts"], points)
        returned, valid2 = apply_chain(reconstruction[opposite+"_parts"], mapped)
        errors = torch.linalg.vector_norm((returned-points)*scale, dim=-1)
        valid = valid1 & valid2
        exported = points+m.s.sample_displacement(reconstruction[direction], points)
        export_error = torch.linalg.vector_norm((exported-mapped)*scale, dim=-1)
        export_back = exported+m.s.sample_displacement(reconstruction[opposite], exported)
        export_cycle = torch.linalg.vector_norm((export_back-points)*scale, dim=-1)
        results[direction] = {"cycle_all": summary(errors), "cycle_inbounds": summary(errors[valid]),
                              "inbounds_count": int(valid.sum()), "inbounds_fraction": float(valid.float().mean()),
                              "export_vs_retained": summary(export_error), "export_cycle_all": summary(export_cycle)}
    return results


@torch.no_grad()
def branch_diagnostics(reconstruction, batch, objective=None, parameter=None):
    v1, v2 = reconstruction["velocities"]
    scale = m.d.voxel_scale(v1.shape[:3], device=v1.device)
    union = (batch["fixed"] > 0) | (batch["moving"] > 0)
    full = m.energies(v1*scale, v2*scale)
    full.update(R1_native=m.diffusion(v1*scale).item(), R2_native=m.diffusion(v2*scale).item(),
                common_diffusion_native=m.diffusion((v1+v2)*.5*scale).item(),
                relative_diffusion_native=m.diffusion((v1-v2)*scale).item())
    result = {"full_volume": full, "raw_foreground_union": m.energies(v1*scale, v2*scale, union)}
    if objective is not None:
        ev = objective.velocity(parameter, native=False)
        result["loss_grid_diffusion"] = {"R1": float(m.diffusion(ev[0]*objective.scale)),
                                          "R2": float(m.diffusion(ev[1]*objective.scale)),
                                          "coefficient": objective.alpha}
    identity = m.c.make_coordinate_grid(v1.shape[:3], v1.device)
    mid = []
    for i, (u, image) in enumerate(zip(reconstruction["positive"], (batch["moving"], batch["fixed"])), 1):
        coords = identity+u
        outside = (coords.abs() > 1).any(-1)
        warped = F.grid_sample(image[None, None], coords[None], mode="bilinear", padding_mode="border", align_corners=True)
        positive = warped[0, 0] > 0
        result[f"branch{i}_midpoint"] = {
            "oob_fraction": float(outside.float().mean()), "oob_union_fraction": float(outside[union].float().mean()),
            "image_mean": float(warped.mean()), "image_std": float(warped.std(unbiased=False)),
            "image_positive_fraction": float(positive.float().mean()),
            "source_positive_fraction": float((image > 0).float().mean())}
        mid.append(positive)
    result["midpoint_positive_overlap_fraction"] = float((mid[0] & mid[1]).float().mean())
    result["midpoint_positive_union_fraction"] = float((mid[0] | mid[1]).float().mean())
    return result
