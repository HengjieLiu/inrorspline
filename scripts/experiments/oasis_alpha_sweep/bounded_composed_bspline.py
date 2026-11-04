"""Bounded cubic increments, retained pull composition, and topology diagnostics.

No historical numerical module is modified. Each elementary map is defined by
native-grid B-spline samples and their border-extended trilinear interpolant.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import dl_ncc_diffusion as d

CPS = (8, 4, 2, 1)
ALPHAS = (1., 1., 5., 1.)
MULTIPLIER = .99


def basis_numpy(x):
    t = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(t < 1, 2 / 3 - t * t + t**3 / 2,
                    np.where(t < 2, (2-t)**3 / 6, 0.))


def derivative_stencil(cps, phase, corner, sampling="endpoint_aligned"):
    """Rows are derivatives w.r.t. native z/y/x, in control-grid units.

    Coefficient centers are integers -2..3 in each dimension. The derivative
    is for the *trilinear interpolant of samples*, not the continuous spline.
    """
    shift = .5 if sampling == "sitreg_half_voxel" and cps % 2 == 0 else 0.
    weights = [basis_numpy((p + shift + np.arange(2)[:, None]) / cps
                           - np.arange(-2, 4)[None]) for p in phase]
    derivatives = []
    for axis in range(3):
        selected = [w[1]-w[0] if j == axis else w[corner[j]] for j, w in enumerate(weights)]
        derivatives.append(cps * np.einsum("i,j,k->ijk", *selected))
    return np.stack(derivatives)


@lru_cache(maxsize=None)
def compute_bound(cps, sampling="endpoint_aligned"):
    """Exact FP64 stencil operator norm: max_{corners,phases,s} ||sum sD||1.

    max_{|coeff|<=1} sum_j |D_j coeff| = max_{s in {-1,+1}^3}
    sum_coeff |sum_j s_j D_j|. Border extension is nonexpansive in infinity
    norm; our control margin covers the entire native crop's support.
    """
    if cps not in CPS or sampling not in ("endpoint_aligned", "sitreg_half_voxel"):
        raise ValueError("Unsupported bound geometry")
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    worst = -math.inf; witness = None
    for phase in product(range(cps), repeat=3):
        for corner in product((0, 1), repeat=3):
            derivatives = derivative_stencil(cps, phase, corner, sampling)
            combined = signs @ derivatives.reshape(3, -1)
            values = np.abs(combined).sum(axis=1)
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


def capped_controls(latent, cap_norm_xyz):
    return cap_norm_xyz * torch.tanh(latent / cap_norm_xyz)


def sample_displacement(field, coordinates):
    """Interpolate displacement, not absolute coordinates; never clamp identity."""
    if field.ndim != 4 or field.shape[-1] != 3 or coordinates.shape[-1] != 3:
        raise ValueError("Expected DHW3 field and XYZ coordinates")
    grid = coordinates[None] if coordinates.ndim == 4 else coordinates.reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(field.permute(3, 0, 1, 2)[None], grid,
                            mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0].permute(1, 2, 3, 0).reshape(coordinates.shape)


def compose(fields, coordinates):
    """fields stored oldest→newest: Phi_old o delta_new, fixed→moving pull."""
    mapped = coordinates
    for field in reversed(fields):
        mapped = mapped + sample_displacement(field, mapped)
    return mapped


class Objective:
    def __init__(self, fixed, moving, stage, alpha, frozen_fields, bound):
        self.stage, self.alpha, self.bound = stage, float(alpha), bound
        self.image_shape = tuple(fixed.shape)
        shapes = c.build_nested_cp_shapes(self.image_shape, 8, CPS)
        self.control_shape = shapes[stage.cps]
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

    def atomic(self, latent):
        return self.expand(capped_controls(latent, self.cap_norm))

    def field(self, latent):
        return compose((*self.frozen, self.atomic(latent)), self.identity) - self.identity

    def __call__(self, latent):
        q = self.field(latent)
        warped = F.grid_sample(self.moving[None, None], (self.identity + q)[None],
                               mode="bilinear", padding_mode="border", align_corners=True)
        data = self.ncc(warped)
        reg = c.forward_valid_diffusion((q * self.scale).permute(3, 0, 1, 2)[None], None)
        return data + self.alpha * reg, data, reg


def optimize(objective, out):
    """Historical Adam/plateau semantics, with correctly named latent states."""
    stage = objective.stage
    out.mkdir(parents=True, exist_ok=False)
    parameter = torch.nn.Parameter(torch.zeros(*objective.control_shape, 3,
                                               device=objective.identity.device))
    optimizer = torch.optim.Adam([parameter], lr=stage.lr, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., amsgrad=False)
    best = parameter.detach().clone()
    rows = []; best_loss = math.inf; best_epoch = 0
    reductions = 0; last_action = 0; reason = "max_epochs"
    if parameter.is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (out / "history.jsonl").open("w") as history:
        for epoch in range(stage.max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, data, reg = objective(parameter)
            values = torch.stack((loss, data, reg)).detach().cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError("Nonfinite objective")
            if values[0] < best_loss:
                best_loss, best_epoch = values[0], epoch
                best.copy_(parameter.detach())
            row = {"epoch": epoch, "optimizer_updates_completed": epoch, "loss": values[0],
                   "data_loss": values[1], "reg_loss": values[2], "weighted_reg_loss": objective.alpha*values[2],
                   "lr": optimizer.param_groups[0]["lr"], "best_epoch": best_epoch,
                   "elapsed_sec": time.monotonic()-started}
            rows.append(row); history.write(json.dumps(row) + "\n")
            if epoch % stage.check_interval == 0:
                history.flush(); print(json.dumps({"stage": stage.index, **row}), flush=True)
            if epoch == stage.max_epochs:
                break
            if d.plateau(rows, stage, last_action):
                if reductions >= stage.max_lr_reductions:
                    reason = "adaptive_plateau_after_two_lr_reductions" if reductions else "adaptive_plateau"
                    break
                optimizer.param_groups[0]["lr"] *= .5
                reductions += 1; last_action = epoch
            loss.backward()
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Invalid latent gradient")
            optimizer.step()
    if parameter.is_cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        selected = [float(x) for x in objective(best)]
        controls = capped_controls(best, objective.cap_norm)
        saturation = (controls.abs() / objective.cap_norm > .95).float().mean().item()
    if not math.isclose(selected[0], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Best checkpoint loss mismatch")
    summary = {"stage": asdict(stage), "alpha": objective.alpha, "representation": "bounded_bspline_increment",
               "parameter_units": "normalized_grid_xyz_latent", "control_units": "normalized_grid_xyz",
               "cps": stage.cps, "control_shape": list(objective.control_shape), "bspline_order": 3,
               "evaluation_shape": list(objective.eval_shape), "voxel_scale_xyz": objective.scale.tolist(),
               "cap_norm_xyz": objective.cap_norm.tolist(), "bound": objective.bound,
               "saturated_coefficient_fraction_gt_95pct": saturation,
               "max_abs_coefficient_ratio": float((controls.abs()/objective.cap_norm).max()),
               "effective_patience": stage.patience, "actual_epochs": rows[-1]["epoch"],
               "best_epoch": best_epoch, "best_loss": selected[0], "best_data_loss": selected[1],
               "best_reg_loss": selected[2], "initial_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
               "stopping_reason": reason, "lr_reductions": reductions, "initial_latent_max_abs": 0.,
               "training_elapsed_sec": time.monotonic()-started,
               "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if parameter.is_cuda else 0.,
               "blur_sigma": objective.blur_sigma, "blur_kernel_size": objective.blur_kernel_size,
               "objective_voxel_count": int(np.prod(objective.eval_shape)), "similarity_normalizer": 1.,
               "regularizer": "unmasked_total_composed_field_forward_valid_l2_component_mean_axes_mean"}
    d.atomic_torch_save({"latent_norm_xyz": best.cpu(), "controls_norm_xyz": controls.cpu(), "summary": summary}, out/"best.pt")
    d.atomic_torch_save({"latent_norm_xyz": parameter.detach().cpu(), "optimizer": optimizer.state_dict(),
                         "summary": summary}, out/"last.pt")
    c.write_csv(out/"history.csv", rows); c.write_json(out/"summary.json", summary)
    return best.detach(), summary


@torch.no_grad()
def corner_diagnostics(field_voxel, chunk_depth=8):
    """Independent FP64 one-sided derivative audit of every trilinear cell corner."""
    nz, ny, nx, _ = field_voxel.shape
    maximum = 0.; minimum_det = math.inf; nonpositive = 0
    for start in range(0, nz-1, chunk_depth):
        end = min(nz-1, start+chunk_depth)
        slab = field_voxel[start:end+1].double()
        v = {(z,y,x): slab[z:z+end-start, y:y+ny-1, x:x+nx-1]
             for z,y,x in product((0,1), repeat=3)}
        for z,y,x in product((0,1), repeat=3):
            dx,dy,dz = v[z,y,1]-v[z,y,0], v[z,1,x]-v[z,0,x], v[1,y,x]-v[0,y,x]
            maximum = max(maximum, float((dx.abs()+dy.abs()+dz.abs()).max()))
            # Explicit 3x3 determinant avoids a large batched linalg workspace.
            a,b,cc = dx[...,0]+1, dy[...,0], dz[...,0]
            dd,e,f = dx[...,1], dy[...,1]+1, dz[...,1]
            g,h,i = dx[...,2], dy[...,2], dz[...,2]+1
            det = a*(e*i-f*h)-b*(dd*i-f*g)+cc*(dd*h-e*g)
            minimum_det = min(minimum_det, float(det.min()))
            nonpositive += int((det <= 0).sum())
    return {"max_displacement_jacobian_infinity_norm": maximum,
            "minimum_map_determinant_at_corners": minimum_det,
            "nonpositive_corner_count": nonpositive, "corner_count": 8*(nz-1)*(ny-1)*(nx-1),
            "dtype": "float64", "bound_below_one": maximum < 1.}


def offgrid_diagnostics(fields, exported, mask, count=100000, seed=1):
    """Same random foreground locations, actual chain vs reinterpolated export."""
    import hashlib
    foreground = mask.detach().cpu().numpy().astype(bool).copy()
    for axis in range(3):
        foreground.swapaxes(0,axis)[:2] = False
        foreground.swapaxes(0,axis)[-2:] = False
    rng = np.random.default_rng(seed)
    locations = np.flatnonzero(foreground)
    selected = rng.choice(locations, count, replace=count > len(locations))
    zyx = np.stack(np.unravel_index(selected, foreground.shape), axis=-1)
    xyz = zyx[:, ::-1].astype(np.float64) + rng.uniform(-.499, .499, (count,3))
    scale = np.asarray([(foreground.shape[2]-1)/2, (foreground.shape[1]-1)/2, (foreground.shape[0]-1)/2])
    point_values = xyz / scale - 1
    result = {"count": count, "seed": seed, "dtype": "float64",
              "support": "fixed_raw_positive_voxel_centers_plus_uniform_jitter_0.499_excluding_two_voxel_border",
              "points_sha256": hashlib.sha256(point_values.tobytes()).hexdigest()}
    for name, parts in (("retained_composition", fields), ("interpolated_export", [exported])):
        points = torch.tensor(point_values, device=exported.device, dtype=torch.float64, requires_grad=True)
        mapped = compose([field.detach().double() for field in parts], points)
        jac = torch.stack([torch.autograd.grad(mapped[:,j].sum(), points, retain_graph=j<2)[0]
                           for j in range(3)], dim=-2)
        determinant = torch.linalg.det(jac).detach()
        if not bool(torch.isfinite(determinant).all()):
            raise RuntimeError("Nonfinite off-grid determinant")
        result[name] = {"minimum_jacobian": float(determinant.min()),
                        "nonpositive_count": int((determinant<=0).sum()),
                        "nonpositive_percent": float((determinant<=0).double().mean()*100),
                        "positive_log_jacobian_std": float(determinant[determinant>0].log().std(correction=0))}
        del mapped, points, jac, determinant
    return result
