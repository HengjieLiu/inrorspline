"""P81 bounded-cost mechanism diagnostics, separate from fitting objectives."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from numerics import conditioning_scales, compose_displacements, world_points, world_to_grid


def optimize_sampled_d0(basis, points, targets, initial, max_steps=1000,
                       lbfgs_evals=100, *, lr=.01, trace_every=25, callback=None,
                       conditioning_config=None):
    """Fixed-geometry Adam/L-BFGS diagnostic on the exact sampled D0 objective.

    Returns (final_coefficients, strict_JSON_report). Adam's scalar endpoint
    is retained; only the final coefficients are returned for native rendering.
    The line-search closure has a hard evaluation cap and restores its best
    feasible observed coefficients when that cap is reached. Column scales
    use the shared declared-box conditioner; the objective still uses only
    the supplied fit samples.
    """
    if max_steps < 0 or lbfgs_evals < 0 or lr <= 0 or trace_every < 1:
        raise ValueError("Invalid sampled optimizer budget/settings")
    if tuple(initial.shape) != (basis.feature_count, 3) or points.shape != targets.shape:
        raise ValueError("Sampled optimizer requires paired[N,3]points/targets and[M,3]initial coefficients")
    device = initial.device
    points = torch.as_tensor(points, device=device, dtype=initial.dtype).detach()
    targets = torch.as_tensor(targets, device=device, dtype=initial.dtype).detach()

    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)

    sync(); begun = time.monotonic()
    with torch.no_grad():
        features = basis.features(points).detach()
        scale, conditioning = conditioning_scales(basis, points, conditioning_config)
        inactive = features.square().mean(0) <= 1e-24
        design = features / scale
    sync(); setup_seconds = time.monotonic() - begun
    variable = torch.nn.Parameter(initial.detach().clone() * scale[:, None])

    def loss():
        return (design @ variable - targets).square().sum(-1).mean()

    with torch.no_grad():
        initial_loss = float(loss())
    if not np.isfinite(initial_loss):
        raise ValueError("Nonfinite sampled initial objective")
    best_loss, best = initial_loss, variable.detach().clone()
    trace = [dict(phase="initial", step=0, objective_evaluations=1, endpoint_mse_mm2=initial_loss, elapsed_seconds=0.)]
    adam = torch.optim.Adam([variable], lr=lr)
    sync(); adam_started = time.monotonic()
    adam_evals = 1
    for step in range(max_steps):
        adam.zero_grad(set_to_none=True)
        objective = loss(); adam_evals += 1
        value = float(objective.detach())
        if not np.isfinite(value):
            raise ValueError("Nonfinite sampled Adam objective")
        if value < best_loss:
            best_loss, best = value, variable.detach().clone()
        objective.backward(); adam.step()
        if (step + 1) % trace_every == 0 or step + 1 == max_steps:
            with torch.no_grad():
                value = float(loss()); adam_evals += 1
            if not np.isfinite(value): raise ValueError("Nonfinite sampled Adam checkpoint")
            if value < best_loss: best_loss, best = value, variable.detach().clone()
            sync()
            row = dict(phase="adam", step=step + 1, objective_evaluations=adam_evals,
                       endpoint_mse_mm2=value, best_endpoint_mse_mm2=best_loss,
                       elapsed_seconds=time.monotonic() - adam_started)
            trace.append(row)
            if callback is not None: callback(row)
    sync(); adam_seconds = time.monotonic() - adam_started
    with torch.no_grad(): variable.copy_(best)
    adam_loss = best_loss
    lbfgs_count = 0
    lbfgs_reason = "disabled"
    sync(); lbfgs_started = time.monotonic()

    class EvaluationCapReached(Exception):
        pass

    if lbfgs_evals:
        lbfgs = torch.optim.LBFGS([variable], lr=1., max_iter=lbfgs_evals,
                                 max_eval=lbfgs_evals, tolerance_grad=1e-9,
                                 tolerance_change=1e-12, history_size=20,
                                 line_search_fn="strong_wolfe")

        def closure():
            nonlocal lbfgs_count, best_loss, best
            if lbfgs_count >= lbfgs_evals: raise EvaluationCapReached()
            lbfgs.zero_grad(set_to_none=True)
            objective = loss(); lbfgs_count += 1
            value = float(objective.detach())
            if not np.isfinite(value): raise ValueError("Nonfinite sampled L-BFGS objective")
            if value < best_loss: best_loss, best = value, variable.detach().clone()
            objective.backward()
            if lbfgs_count % trace_every == 0 or lbfgs_count == lbfgs_evals:
                sync()
                row = dict(phase="lbfgs", objective_evaluations=lbfgs_count,
                           endpoint_mse_mm2=value, best_endpoint_mse_mm2=best_loss,
                           elapsed_seconds=time.monotonic() - lbfgs_started)
                trace.append(row)
                if callback is not None: callback(row)
            return objective

        try:
            lbfgs.step(closure)
            lbfgs_reason = "optimizer_stopped_or_evaluation_limit"
        except EvaluationCapReached:
            lbfgs_reason = "hard_closure_evaluation_cap"
    sync(); lbfgs_seconds = time.monotonic() - lbfgs_started
    result = best / scale[:, None]
    report = dict(objective="sample_mean_vector_endpoint_squared_error_mm2",
                  sample_count=len(points), scalar_feature_count=basis.feature_count,
                  initial_endpoint_mse_mm2=initial_loss, adam_endpoint_mse_mm2=adam_loss,
                  final_endpoint_mse_mm2=best_loss, adam_steps=max_steps,
                  adam_objective_evaluations=adam_evals, lbfgs_objective_evaluations=lbfgs_count,
                  lbfgs_evaluation_cap=lbfgs_evals, lbfgs_stop_reason=lbfgs_reason,
                  column_scaling="full_declared_box_RMS_floor; coefficient_transform_exact",
                  conditioning=conditioning,
                  inactive_columns=int(inactive.sum()), learning_rate=lr,
                  selection="lowest_observed_training_endpoint_MSE; no_labels",
                  native_adam_gap_evaluated=False, native_final_gap_evaluated_by_caller=True,
                  setup_seconds=setup_seconds, adam_seconds=adam_seconds, lbfgs_seconds=lbfgs_seconds,
                  total_seconds=time.monotonic() - begun, trace=trace)
    return result.detach(), report


def _array(value):
    return np.load(value, mmap_mode="r") if isinstance(value, (str, Path)) else np.asarray(value)


def _stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return dict(count=0, mean=None, rms=None, p95=None, maximum=None)
    return dict(count=int(values.size), mean=float(values.mean()), rms=float(np.sqrt(np.mean(values ** 2))),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def global_affine_residual_normalization(prediction_mm, target, *, chunk_depth=8):
    """Full-foreground best-affine residual denominator from a 4x4 LS system.

    Coordinates are centered/scaled before accumulation. This uses a tiny,
    explicitly condition-checked normal system, not a native-volume design.
    """
    pred, reference = _array(prediction_mm), _array(target["field"])
    mask, affine = _array(target["mask"]).astype(bool), np.asarray(target["affine"], np.float64)
    shape = reference.shape[:3]
    if pred.shape != reference.shape or pred.shape[-1] != 3 or mask.shape != shape:
        raise ValueError("Global-affine diagnostic input geometry mismatch")
    center_index = (np.asarray(shape, np.float64) - 1) / 2
    center_mm = affine[:3, :3] @ center_index + affine[:3, 3]
    # Axis-aligned world extent; valid also for rotated/oblique affine grids.
    scale_mm = np.abs(affine[:3, :3]) @ center_index
    if (scale_mm <= 0).any() or chunk_depth < 1:
        raise ValueError("Degenerate affine fit domain")
    gram, rhs = np.zeros((4, 4)), np.zeros((4, 3))

    def blocks():
        for start in range(0, shape[0], chunk_depth):
            stop = min(start + chunk_depth, shape[0])
            local = np.argwhere(mask[start:stop])
            if not len(local):
                continue
            index = local.copy(); index[:, 0] += start
            point = index @ affine[:3, :3].T + affine[:3, 3]
            x = np.concatenate((np.ones((len(point), 1)), (point - center_mm) / scale_mm), 1)
            select = tuple(index.T)
            yield x, np.asarray(reference[select], np.float64), np.asarray(pred[select], np.float64)

    count = 0
    for x, y, _ in blocks():
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite target in affine reference domain")
        gram += x.T @ x; rhs += x.T @ y; count += len(x)
    if count < 4:
        return dict(status="insufficient_foreground", count=count, relative_rmse_pct=None)
    eigenvalues = np.linalg.eigvalsh(gram)
    rank = int(np.count_nonzero(eigenvalues > max(eigenvalues[-1], 1) * 1e-12))
    condition = float(np.linalg.cond(gram))
    if rank != 4 or not np.isfinite(condition) or condition > 1e10:
        return dict(status="rank_or_condition_failure", count=count, rank=rank,
                    gram_condition=condition if np.isfinite(condition) else None, relative_rmse_pct=None)
    coefficients = np.linalg.solve(gram, rhs)
    residual_sq = fit_error_sq = 0.
    for x, y, prediction in blocks():
        affine_residual, fit_error = y - x @ coefficients, prediction - y
        residual_sq += float(np.sum(affine_residual ** 2))
        fit_error_sq += float(np.sum(fit_error ** 2))
    defined = residual_sq / count > 1e-12
    return dict(status="defined" if defined else "affine_residual_rms_below_1e-6_mm",
                count=count, fit_domain="full_fixed_raw_positive_foreground", rank=rank,
                gram_condition=condition, normalized_affine_coefficients=coefficients.tolist(),
                coordinate_center_mm=center_mm.tolist(), coordinate_scale_mm=scale_mm.tolist(),
                reference_affine_residual_rms_mm=float(np.sqrt(residual_sq / count)),
                prediction_endpoint_rmse_mm=float(np.sqrt(fit_error_sq / count)),
                relative_rmse_pct=100 * float(np.sqrt(fit_error_sq / residual_sq)) if defined else None,
                interpretation="endpoint_error_normalized_by_reference_residual_after_best_global_affine")


def blending_derivative_diagnostics(basis, coefficients, points_mm, *, max_points=512):
    """Decompose Du into weighted local slopes and blending derivatives.

    The tree result uses the explicit hierarchical-contrast decomposition,
    not a claim that individual leaves own independent translation parameters.
    """
    if basis.family == "bspline":
        return dict(status="not_applicable", reason="cubic_translation_basis_has_no_local_affine_blend_decomposition")
    if max_points < 1:
        raise ValueError("Positive diagnostic sample cap required")
    points = torch.as_tensor(points_mm, device=coefficients.device, dtype=coefficients.dtype)
    if points.ndim != 2 or points.shape[-1] != 3 or not len(points):
        raise ValueError("Require nonempty physical diagnostic points[N,3]")
    points = points[:max_points].detach().clone().requires_grad_(True)
    coeff = coefficients.detach()
    with torch.enable_grad():
        features = basis.features(points)
        values = features @ coeff
        jacobian = torch.stack([torch.autograd.grad(values[:, output].sum(), points, retain_graph=True)[0]
                                for output in range(3)], 1)
        root_scale = points.new_tensor([(b - a) / 2 for a, b in zip(basis.domain_min, basis.domain_max)])
        if basis.family == "gaussian_affine":
            blocks = coeff.reshape(-1, 4, 3)
            spacing = points.new_tensor([(b - a) / (n - 1)
                for a, b, n in zip(basis.domain_min, basis.domain_max, basis.lattice_shape)])
            slopes = blocks[:, 1:, :] / spacing[None, :, None]
            weights = features.reshape(len(points), -1, 4)[:, :, 0]
            local = torch.einsum("nk,kjo->noj", weights, slopes)
            representation = "normalized_gaussian_local_affine"
        else:
            local = (coeff[1:4] / root_scale[:, None]).T[None].expand(len(points), -1, -1)
            representation = "global_affine_plus_weighted_translation"
            if basis.is_tree and basis.is_affine_tree and basis.leaf_count > 1:
                internal = sorted((n for n in basis.nodes if "children" in n), key=lambda n: n["split_index"])
                scales = points.new_tensor([[(hi - lo) / 2 for lo, hi in zip(n["lo"], n["hi"])] for n in internal])
                contrast_coeff = coeff[4:].reshape(-1, 4, 3)
                slopes = contrast_coeff[:, 1:, :] / scales[:, :, None]
                q = features[:, 4:].reshape(len(points), -1, 4)[:, :, 0]
                local = local + torch.einsum("nk,kjo->noj", q, slopes)
                representation = "hierarchical_affine_contrast_tree"
            elif basis.is_tree:
                representation = "affine_root_translation_contrast_tree"
            elif basis.family == "global_affine":
                representation = "global_affine"
        blending = jacobian - local
    result = dict(status="evaluated", sample_count=len(points), sample_rule="first_max_points_of_supplied_fixed_sample",
                  representation=representation, derivatives="physical_Du; no_identity_added",
                  decomposition="Da=sum_weighted_local_slopes+sum_affine_value_times_weight_gradient",
                  continuous_basis=True, global_invertibility_certificate=False)
    for name, tensor in (("total_derivative", jacobian), ("local_slope_term", local), ("blending_term", blending)):
        result[name] = dict(frobenius=_stats(torch.linalg.matrix_norm(tensor, ord="fro").detach().cpu().numpy()),
                            induced_infinity=_stats(tensor.abs().sum(-1).amax(-1).detach().cpu().numpy()))
    denominator = torch.linalg.matrix_norm(jacobian, ord="fro").square().mean()
    result["blend_to_total_rms_ratio"] = (float(torch.sqrt(torch.linalg.matrix_norm(blending, ord="fro").square().mean() / denominator))
                                           if float(denominator) > 1e-20 else None)
    return result


def inverse_consistency(forward_mm, inverse_mm, affine, *, mask=None):
    """Two dense composition residuals with explicit intermediate-domain coverage.

    This diagnostic measures exported interpolants. Border-padded results on
    invalid intermediate points are reported separately from valid residuals.
    """
    if forward_mm.shape != inverse_mm.shape or forward_mm.ndim != 4 or forward_mm.shape[-1] != 3:
        raise ValueError("Forward/inverse must be equal DHW3 displacement tensors")
    shape = forward_mm.shape[:3]
    if mask is None:
        regions = {"full": torch.ones(shape, device=forward_mm.device, dtype=torch.bool)}
    else:
        fg = torch.as_tensor(mask, device=forward_mm.device, dtype=torch.bool)
        if tuple(fg.shape) != tuple(shape):
            raise ValueError("Inverse diagnostic mask shape mismatch")
        regions = {"full": torch.ones_like(fg), "foreground": fg}
    with torch.no_grad():
        points = world_points(shape, affine, device=forward_mm.device, dtype=forward_mm.dtype).reshape(*shape, 3)
        result = dict(status="evaluated", construction="caller_supplied_inverse",
                      field_interpolation="trilinear_border_align_corners_true",
                      scope="exported_interpolant_inverse_consistency_not_input_swap_symmetry",
                      boundary_caveat="invalid_intermediate_points_use_border_extension; finite_residual_does_not_validate_them")
        for name, outer, inner in (("inverse_after_forward", inverse_mm, forward_mm),
                                   ("forward_after_inverse", forward_mm, inverse_mm)):
            sample_grid = world_to_grid(points + inner, shape, affine)
            valid = (sample_grid.abs() <= 1).all(-1)
            residual = compose_displacements(outer, inner, affine)
            error = torch.linalg.vector_norm(residual, dim=-1)
            row = {}
            for region_name, region in regions.items():
                count = int(region.sum())
                both = valid & region
                row[region_name] = dict(count=count, valid_count=int(both.sum()),
                    valid_intermediate_fraction=float(both.sum() / count) if count else None,
                    valid_residual_mm=_stats(error[both].cpu().numpy()),
                    all_border_extended_residual_mm=_stats(error[region].cpu().numpy()))
            result[name] = row
    return result
