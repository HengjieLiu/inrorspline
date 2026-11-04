"""P66 selective bounds; preserve P64 composition, objective and optimizer schedule."""
from dataclasses import asdict
import json
import math
import time

import numpy as np
import torch

import bounded_composed_bspline as original

c, d = original.c, original.d
CPS, ALPHAS, MULTIPLIER = original.CPS, original.ALPHAS, original.MULTIPLIER
compose = original.compose
sample_displacement = original.sample_displacement
capped_controls = original.capped_controls
compute_bound = original.compute_bound
corner_diagnostics = original.corner_diagnostics
offgrid_diagnostics = original.offgrid_diagnostics


def stage_mask(bounded_last):
    if type(bounded_last) is not int or bounded_last not in (1, 2, 3):
        raise ValueError("Require one, two or three final bounded stages")
    return tuple(i >= 4 - bounded_last for i in range(4))


def stages(smoke=False):
    return d.stages(d.FitSpec(0, "pyramid", 1., "8421", smoke=smoke))


class Objective(original.Objective):
    def __init__(self, fixed, moving, stage, alpha, frozen_fields, bound, bounded):
        super().__init__(fixed, moving, stage, alpha, frozen_fields, bound)
        if type(bounded) is not bool:
            raise ValueError("Require explicit boolean bounded flag")
        self.bounded = bounded

    def controls(self, parameter):
        return capped_controls(parameter, self.cap_norm) if self.bounded else parameter

    def atomic(self, parameter):
        return self.expand(self.controls(parameter))


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
        controls = objective.controls(best)
        saturation = (controls.abs() / objective.cap_norm > .95).float().mean().item() if objective.bounded else None
    if not math.isclose(selected[0], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Best checkpoint loss mismatch")
    summary = {"stage": asdict(stage), "alpha": objective.alpha, "representation": "bounded_bspline_increment" if objective.bounded else "unbounded_bspline_increment",
               "bounded": objective.bounded, "parameterization": "cap*tanh(p/cap)" if objective.bounded else "identity_controls_equals_parameter",
               "parameter_units": "normalized_grid_xyz_latent", "control_units": "normalized_grid_xyz",
               "cps": stage.cps, "control_shape": list(objective.control_shape), "bspline_order": 3,
               "evaluation_shape": list(objective.eval_shape), "voxel_scale_xyz": objective.scale.tolist(),
               "cap_norm_xyz": objective.cap_norm.tolist() if objective.bounded else None,
               "bound": objective.bound if objective.bounded else None,
               "saturated_coefficient_fraction_gt_95pct": saturation,
               "max_abs_coefficient_ratio": float((controls.abs()/objective.cap_norm).max()) if objective.bounded else None,
               "max_abs_control_native_voxel": float((controls*objective.native_scale).abs().max()),
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

