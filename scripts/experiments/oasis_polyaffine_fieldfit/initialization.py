"""A shared global-affine least-squares starting velocity for all SVF families."""
from __future__ import annotations

import time

import torch

from basis import create_basis
from numerics import solve_d0


@torch.no_grad()
def fit_global_affine_initialization(basis, points, values, config=None):
    """Fit a foreground displacement affine and embed it exactly as velocity.

    Only the initialization is affine. Subsequent optimization uses the complete
    requested family and original loss. This is not the logarithm of an affine
    transformation and does not reuse unconstrained local-D0 extrapolation.
    """
    started = time.monotonic()
    config = dict(config or {})
    ls_config = dict(config.get("lsqr", config))
    ls_config["iter_lim"] = ls_config.get("final_iter_lim", ls_config.get("iter_lim", 1500))
    affine_basis = create_basis("global_affine", 12, basis.domain_min, basis.domain_max)
    affine_coeff, solver = solve_d0(affine_basis, points, values, config=ls_config)
    if not solver["converged"] or not torch.isfinite(affine_coeff).all():
        raise RuntimeError("Global-affine initialization least-squares solve did not converge")
    center = affine_coeff.new_tensor([(lo + hi)/2 for lo, hi in zip(basis.domain_min, basis.domain_max)])
    scale = affine_coeff.new_tensor([(hi - lo)/2 for lo, hi in zip(basis.domain_min, basis.domain_max)])
    matrix = (affine_coeff[1:4] / scale[:, None]).T
    translation = affine_coeff[0] - matrix @ center
    coefficients = basis.coefficients_for_affine(matrix, translation)
    probe = points[:min(512, len(points))].to(coefficients)
    error = float((basis.evaluate(probe, coefficients) - affine_basis.evaluate(probe, affine_coeff)).abs().max())
    if error > 1e-8:
        raise RuntimeError(f"Global-affine initialization embedding error {error:g} mm")
    return coefficients, dict(
        kind="global_affine_ls_displacement_used_as_velocity",
        solver=solver,
        motion_parameters_fitted=12,
        embedded_family=basis.family,
        world_velocity_matrix=matrix.cpu().tolist(),
        world_velocity_translation_mm=translation.cpu().tolist(),
        max_embedding_sample_error_mm=error,
        changes_initialization_only=True,
        fitting_loss_and_function_space_unchanged=True,
        affine_transform_logarithm=False,
        elapsed_sec=time.monotonic() - started,
    )
