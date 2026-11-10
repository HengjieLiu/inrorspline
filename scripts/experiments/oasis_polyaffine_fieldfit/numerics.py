"""P81 physical-coordinate fitting and native-grid rendering.

Fields are [D,H,W,3] world-mm vectors.  Affines map array indices (D,H,W)
to world coordinates, whereas grid_sample vectors are ordered (W,H,D).
Basis geometry is fixed while coefficients are optimized.  Dense rendering
recomputes features during its backward pass and never saves a volume-sized
design matrix.  Historical registration code and output fields are untouched.
"""
from __future__ import annotations

import itertools
import hashlib
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, lsqr
from torch.utils.checkpoint import checkpoint


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _affine(value, *, device, dtype):
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.shape != (4, 4) or not torch.isfinite(result).all():
        raise ValueError("Require a finite 4x4 array-index-to-world affine")
    if not torch.allclose(result[3], result.new_tensor([0, 0, 0, 1])):
        raise ValueError("Require an affine, not a projective transform")
    if abs(float(torch.linalg.det(result[:3, :3]))) < 1e-12:
        raise ValueError("Singular grid affine")
    return result


def world_points(shape, affine, start=0, stop=None, *, device=None, dtype=None,
                 _validated=False):
    """World-mm voxel centers for a flat, C-order range (without meshgrid)."""
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("Require three grid dimensions >=2")
    if device is None:
        device = affine.device if torch.is_tensor(affine) else "cpu"
    if dtype is None:
        dtype = affine.dtype if torch.is_tensor(affine) else torch.float32
    h = affine if _validated else _affine(affine, device=device, dtype=dtype)
    total = math.prod(shape)
    stop = total if stop is None else int(stop)
    if not 0 <= start <= stop <= total:
        raise ValueError("Invalid flat grid range")
    flat = torch.arange(start, stop, device=device)
    d = torch.div(flat, shape[1] * shape[2], rounding_mode="floor")
    remainder = flat % (shape[1] * shape[2])
    coords = torch.stack((d, torch.div(remainder, shape[2], rounding_mode="floor"),
                          remainder % shape[2]), dim=-1).to(dtype)
    return coords @ h[:3, :3].T + h[:3, 3]


def normalized_identity(shape, *, device, dtype):
    d, h, w = torch.meshgrid(*(torch.linspace(-1, 1, int(n), device=device, dtype=dtype)
                              for n in shape), indexing="ij")
    return torch.stack((w, h, d), dim=-1)


def world_to_grid(points_mm, shape, affine):
    h = _affine(affine, device=points_mm.device, dtype=points_mm.dtype)
    index = (points_mm - h[:3, 3]) @ torch.linalg.inv(h[:3, :3]).T
    span = points_mm.new_tensor([int(n)-1 for n in shape])
    return (2 * index / span - 1).flip(-1)


def sample_field_points(field, points_mm, affine, *, padding_mode="border"):
    """Sample a world-vector field at arbitrary world-mm points."""
    if field.ndim != 4 or field.shape[-1] != 3 or points_mm.shape[-1] != 3:
        raise ValueError("Require DHW3 field and ...3 world points")
    points_mm = points_mm.to(device=field.device, dtype=field.dtype)
    coords = world_to_grid(points_mm, field.shape[:3], affine)
    grid = coords.reshape(1, -1, 1, 1, 3)
    value = F.grid_sample(field.permute(3, 0, 1, 2)[None], grid,
                          mode="bilinear", padding_mode=padding_mode,
                          align_corners=True)
    return value[0, :, :, 0, 0].T.reshape(points_mm.shape)


def compose_displacements(outer, inner, affine, *, identity=None):
    """(Id+outer) o (Id+inner), evaluated at native grid centers."""
    if outer.shape != inner.shape or outer.ndim != 4 or outer.shape[-1] != 3:
        raise ValueError("Composition requires equal DHW3 world-vector grids")
    h = _affine(affine, device=inner.device, dtype=inner.dtype)
    identity = normalized_identity(inner.shape[:3], device=inner.device,
                                   dtype=inner.dtype) if identity is None else identity
    delta_index = inner @ torch.linalg.inv(h[:3, :3]).T
    span = inner.new_tensor([n-1 for n in inner.shape[:3]])
    grid = identity + (2 * delta_index / span).flip(-1)
    sampled = F.grid_sample(outer.permute(3, 0, 1, 2)[None], grid[None],
                            mode="bilinear", padding_mode="border",
                            align_corners=True)[0].permute(1, 2, 3, 0)
    return inner + sampled


def exp_velocity(velocity, affine, steps=7, *, checkpoint_integration=True):
    """Stationary velocity exponential by differentiable scaling/squaring."""
    if type(steps) is not int or steps < 0:
        raise ValueError("Nonnegative integer squaring steps required")
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("Require DHW3 world-mm velocity")
    identity = normalized_identity(velocity.shape[:3], device=velocity.device,
                                   dtype=velocity.dtype)
    field = velocity / (2 ** steps)
    for _ in range(steps):
        def square(current):
            return compose_displacements(current, current, affine, identity=identity)
        if checkpoint_integration and torch.is_grad_enabled() and field.requires_grad:
            field = checkpoint(square, field, use_reentrant=False)
        else:
            field = square(field)
    return field


class _DenseBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, basis, shape, affine, chunk):
        shape = tuple(int(x) for x in shape)
        h = _affine(affine, device=coefficients.device, dtype=coefficients.dtype)
        ctx.basis, ctx.shape, ctx.chunk = basis, shape, int(chunk)
        ctx.save_for_backward(h)
        ctx.coefficient_shape = coefficients.shape
        fast = getattr(basis, "dense_forward", None)
        if fast is not None:
            result = fast(coefficients, shape, h)
            if result is not None:
                ctx.used_separable = True
                return result
        ctx.used_separable = False
        total = math.prod(shape)
        output = coefficients.new_empty((total, 3))
        for start in range(0, total, chunk):
            stop = min(total, start + chunk)
            points = world_points(shape, h, start, stop, _validated=True)
            output[start:stop] = basis.features(points) @ coefficients
        return output.reshape(*shape, 3)

    @staticmethod
    def backward(ctx, gradient):
        (h,) = ctx.saved_tensors
        if ctx.used_separable:
            result = ctx.basis.dense_adjoint(gradient, ctx.shape, h)
            if result is None:
                raise RuntimeError("Separable dense forward/adjoint eligibility changed")
            return result, None, None, None, None
        total = math.prod(ctx.shape)
        flat_gradient = gradient.reshape(-1, 3)
        grad_coeff = gradient.new_zeros(ctx.coefficient_shape)
        # No feature/design tensor survives an iteration. Fixed basis geometry
        # intentionally has no gradient in this experiment.
        for start in range(0, total, ctx.chunk):
            stop = min(total, start + ctx.chunk)
            points = world_points(ctx.shape, h, start, stop, _validated=True)
            features = ctx.basis.features(points)
            grad_coeff.add_(features.T @ flat_gradient[start:stop])
        return grad_coeff, None, None, None, None


def dense_basis_field(basis, coefficients, shape, affine, chunk=16384):
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError("Coefficients must have shape [scalar_features,3]")
    if int(chunk) < 1:
        raise ValueError("Positive render chunk required")
    return _DenseBasis.apply(coefficients, basis, tuple(shape), affine, int(chunk))


def _cell_jacobians(vertices, inv_linear, corner):
    """Return output-row / world-input-column displacement derivatives."""
    a, b, c = corner
    # vertices is [...,2,2,2,3]; spatial corner axes precede vector channel.
    j_index = torch.stack((vertices[..., 1, b, c, :] - vertices[..., 0, b, c, :],
                           vertices[..., a, 1, c, :] - vertices[..., a, 0, c, :],
                           vertices[..., a, b, 1, :] - vertices[..., a, b, 0, :]), dim=-1)
    return j_index @ inv_linear


def _grid_bound_value(field, affine, chunk_slices=4):
    if field.ndim != 4 or field.shape[-1] != 3 or min(field.shape[:3]) < 2:
        raise ValueError("Require DHW3 displacement with each grid side >=2")
    if not torch.isfinite(field).all():
        raise ValueError("Nonfinite displacement cannot be certified")
    h = _affine(affine, device=field.device, dtype=torch.float64)
    inv_linear = torch.linalg.inv(h[:3, :3])
    d, height, width = field.shape[:3]
    best = -1.0
    witness = None
    # Slice slabs keep all eight neighboring vertices, but never materialize
    # every cell's eight Jacobians at once.
    for start in range(0, d-1, int(chunk_slices)):
        stop = min(d-1, start + int(chunk_slices))
        slab = field[start:stop+1].double()
        for corner in itertools.product((0, 1), repeat=3):
            a, b, c = corner
            jd = slab[1:, b:height-1+b, c:width-1+c] - slab[:-1, b:height-1+b, c:width-1+c]
            jh = slab[a:stop-start+a, 1:, c:width-1+c] - slab[a:stop-start+a, :-1, c:width-1+c]
            jw = slab[a:stop-start+a, b:height-1+b, 1:] - slab[a:stop-start+a, b:height-1+b, :-1]
            jacobian = torch.stack((jd, jh, jw), dim=-1) @ inv_linear
            row_norm = jacobian.abs().sum(-1)
            value, position = row_norm.reshape(-1).max(0)
            scalar = float(value)
            if scalar > best:
                flat = int(position)
                row = flat % 3
                cell = flat // 3
                local_d = cell // ((height-1) * (width-1))
                remainder = cell % ((height-1) * (width-1))
                witness = {"cell": [start+local_d, remainder//(width-1), remainder%(width-1)],
                           "corner": list(corner), "row": row}
                best = scalar
    return best, witness


class _GridBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, affine, chunk_slices):
        value, witness = _grid_bound_value(field, affine, chunk_slices)
        d, h, w = witness["cell"]
        ctx.witness, ctx.field_shape = witness, field.shape
        vertices = field[d:d+2, h:h+2, w:w+2].detach().double()
        matrix = _affine(affine, device=field.device, dtype=torch.float64)
        ctx.save_for_backward(vertices, torch.linalg.inv(matrix[:3, :3]))
        # Float64 calculation; conservative safety is applied separately by
        # bounded rendering/certification, not claimed from rounded FP32 alone.
        return field.new_tensor(value)

    @staticmethod
    def backward(ctx, output_gradient):
        vertices, inv_linear = ctx.saved_tensors
        with torch.enable_grad():
            vertices = vertices.detach().requires_grad_(True)
            jacobian = _cell_jacobians(vertices, inv_linear, ctx.witness["corner"])
            row_norm = jacobian[ctx.witness["row"]].abs().sum()
            derivative, = torch.autograd.grad(row_norm, vertices)
        result = output_gradient.new_zeros(ctx.field_shape)
        d, h, w = ctx.witness["cell"]
        result[d:d+2, h:h+2, w:w+2] = derivative.to(result) * output_gradient
        return result, None, None


def grid_lipschitz_bound(field, affine, *, chunk_slices=4, return_metadata=False):
    """Full-cell induced-infinity bound on physical displacement Du, not Dphi.

    Differentiable with a valid subgradient at ties. A float64 recheck and
    explicit safety allowance are required before certifying a saved field.
    """
    if int(chunk_slices) < 1:
        raise ValueError("Positive cell slab size required")
    if return_metadata:
        value, witness = _grid_bound_value(field, affine, chunk_slices)
        return value, {"norm": "induced_infinity", "jacobian": "physical_displacement_Du",
                       "interpolant": "trilinear", "cell_count": math.prod(n-1 for n in field.shape[:3]),
                       "corner_count_per_cell": 8, "calculation_dtype": "float64", "witness": witness}
    return _GridBound.apply(field, affine, int(chunk_slices))


def bounded_displacement(field, affine, rho=.9, *, safety_margin=1e-5,
                         chunk_slices=4, return_metadata=False):
    if not 0 < rho < 1 or not 0 < safety_margin < rho:
        raise ValueError("Require 0<safety_margin<rho<1")
    bound = grid_lipschitz_bound(field, affine, chunk_slices=chunk_slices)
    translation = field.mean(dim=(0, 1, 2), keepdim=True)
    alpha = torch.minimum(bound.new_ones(()), (rho-safety_margin)/(bound+1e-12))
    output = translation + alpha * (field-translation)
    if return_metadata:
        return output, {"bound_before": float(bound.detach()), "alpha": float(alpha.detach()),
                        "rho": float(rho), "safety_margin": float(safety_margin),
                        "constant_translation_preserved": True,
                        "training_constraint": "radial_reparameterization"}
    return output


def render_field(basis, coefficients, shape, affine, mode="direct_dvf", *, chunk=16384,
                 steps=7, rho=.9, return_metadata=False, checkpoint_integration=True):
    aliases = {"D0": "direct_dvf", "D1": "bounded_dvf", "V": "svf", "dvf": "direct_dvf"}
    mode = aliases.get(mode, mode)
    field = dense_basis_field(basis, coefficients, shape, affine, chunk)
    metadata = {"mode": mode, "shape": list(shape), "units": "world_mm",
                "render_chunk": int(chunk), "output_interpolant": "trilinear"}
    if mode == "svf":
        field = exp_velocity(field, affine, steps, checkpoint_integration=checkpoint_integration)
        metadata.update(integration_steps=int(steps), integration_grid=list(shape),
                        padding_mode="border", align_corners=True)
    elif mode == "bounded_dvf":
        field, bound_meta = bounded_displacement(field, affine, rho, return_metadata=True)
        metadata.update(bound_meta)
    elif mode != "direct_dvf":
        raise ValueError(f"Unknown renderer: {mode}")
    return (field, metadata) if return_metadata else field


def conditioning_scales(basis, points, config=None):
    """Invertible coefficient scaling from a deterministic *full-box* design.

    Foreground-only column RMS can be arbitrarily small for boundary controls
    while their outside-foreground influence is order one. Dividing by that
    number amplifies optimizer updates and LS null-space components. Full-box
    conditioning with a fixed positive floor changes only parameter coordinates,
    never the fitting samples, loss, basis span or regularization.
    """
    config = {} if config is None else dict(config)
    side = int(config.get("conditioning_samples_per_axis", 32))
    floor = float(config.get("conditioning_min_scale", .01))
    chunk = int(config.get("chunk", 16384))
    if side < 2 or not math.isfinite(floor) or floor < .01 or chunk < 1:
        raise ValueError("Require full-box conditioning side>=2, scale floor>=.01 and positive chunk")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Conditioning requires a nonempty N3 device/dtype reference")
    if hasattr(basis, "domain_min") and hasattr(basis, "domain_max"):
        lower = points.new_tensor(basis.domain_min)
        upper = points.new_tensor(basis.domain_max)
        domain_source = "declared_basis_box"
    else:
        # Small standalone mathematical fixtures without a geometric Basis.
        lower, upper = points.amin(0), points.amax(0)
        domain_source = "fixture_point_bounding_box"
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not (upper > lower).all():
        raise ValueError("Conditioning box must be finite and nondegenerate")
    count = side**3
    sums = None
    with torch.no_grad():
        for start in range(0, count, chunk):
            ids = torch.arange(start, min(count, start+chunk), device=points.device)
            coords = torch.stack((torch.div(ids, side*side, rounding_mode="floor"),
                                  torch.div(ids % (side*side), side, rounding_mode="floor"),
                                  ids % side), -1).to(points.dtype)
            grid = lower + coords/(side-1)*(upper-lower)
            features = basis.features(grid)
            if not torch.isfinite(features).all():
                raise FloatingPointError("Nonfinite full-box conditioning features")
            contribution = features.square().sum(0)
            sums = contribution if sums is None else sums+contribution
        rms = (sums/count).sqrt()
        scales = rms.clamp_min(floor)
    metadata = {"kind": "full_declared_box_rms_floor", "domain_source": domain_source,
                "domain_min": lower.detach().cpu().tolist(), "domain_max": upper.detach().cpu().tolist(),
                "sample_shape": [side]*3, "sample_count": count, "endpoint_aligned": True,
                "minimum_scale_floor": floor, "minimum_raw_rms": float(rms.min()),
                "maximum_raw_rms": float(rms.max()), "minimum_scale": float(scales.min()),
                "maximum_scale": float(scales.max()), "floored_column_count": int((rms < floor).sum()),
                "scale_sha256": hashlib.sha256(scales.detach().double().cpu().numpy().tobytes()).hexdigest(),
                "coefficient_transform": "theta=coefficient*scale; coefficient=theta/scale",
                "objective_and_function_space_unchanged": True}
    return scales, metadata


class SampledLinearOperator:
    """Scaled, optionally cached fitting operator; no native-volume matrix."""
    def __init__(self, basis, points, *, chunk=16384, cache_bytes=512*1024**2,
                 dtype=torch.float64, conditioning_config=None):
        self.basis, self.points = basis, points.to(dtype=dtype)
        self.device, self.dtype = self.points.device, dtype
        self.chunk = int(chunk)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.chunk < 1:
            raise ValueError("Require N3 fitting points and positive chunk")
        probe = basis.features(self.points[:1])
        self.n, self.m = len(points), probe.shape[1]
        self.cached = None
        if self.n * self.m * probe.element_size() <= int(cache_bytes):
            self.cached = torch.empty((self.n, self.m), dtype=dtype, device=self.device)
        with torch.no_grad():
            if self.cached is not None:
                for start in range(0, self.n, self.chunk):
                    stop = min(self.n, start+self.chunk)
                    features = basis.features(self.points[start:stop])
                    self.cached[start:stop] = features
        self.scales, self.conditioning = conditioning_scales(basis, self.points,
            {"chunk": self.chunk, **({} if conditioning_config is None else conditioning_config)})

    def features(self, start, stop):
        return self.cached[start:stop] if self.cached is not None else self.basis.features(self.points[start:stop])

    def forward(self, vector):
        vector = torch.as_tensor(vector, device=self.device, dtype=self.dtype)
        result = torch.empty(self.n, dtype=self.dtype, device=self.device)
        for start in range(0, self.n, self.chunk):
            stop = min(self.n, start+self.chunk)
            result[start:stop] = self.features(start, stop) @ (vector/self.scales)
        return result

    def adjoint(self, vector):
        vector = torch.as_tensor(vector, device=self.device, dtype=self.dtype)
        result = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        for start in range(0, self.n, self.chunk):
            stop = min(self.n, start+self.chunk)
            result += self.features(start, stop).T @ vector[start:stop]
        return result/self.scales

    def scipy_operator(self):
        return LinearOperator((self.n, self.m),
                              matvec=lambda v: self.forward(v).detach().cpu().numpy(),
                              rmatvec=lambda v: self.adjoint(v).detach().cpu().numpy(),
                              dtype=np.float64)


def solve_d0(basis, points, targets_mm, initial=None, config=None, callback=None):
    """Column-scaled LSQR, independently solving the three world components."""
    config = {} if config is None else dict(config)
    if len(points) != len(targets_mm) or targets_mm.shape != (len(points), 3):
        raise ValueError("Require matching N3 points and target displacement")
    if not torch.isfinite(targets_mm).all() or not torch.isfinite(points).all():
        raise ValueError("Nonfinite fitting input")
    _sync(points.device)
    started = time.monotonic()
    operator = SampledLinearOperator(basis, points, chunk=int(config.get("chunk", 16384)),
                                     cache_bytes=int(config.get("cache_bytes", 512*1024**2)),
                                     conditioning_config=config)
    setup_time = time.monotonic()-started
    matrix = operator.scipy_operator()
    target = targets_mm.detach().double().cpu().numpy()
    scales_np = operator.scales.detach().cpu().numpy()
    init = None if initial is None else initial.detach().double().cpu().numpy() * scales_np[:, None]
    coefficients = np.empty((operator.m, 3), np.float64)
    components = []
    for axis in range(3):
        result = lsqr(matrix, target[:, axis], damp=0,
                      atol=float(config.get("atol", 1e-6)), btol=float(config.get("btol", 1e-6)),
                      conlim=float(config.get("conlim", 1e10)),
                      iter_lim=int(config.get("iter_lim", max(1000, 4*operator.m))),
                      x0=None if init is None else init[:, axis])
        coefficients[:, axis] = result[0]/scales_np
        row = {"component": axis, "istop": int(result[1]), "iterations": int(result[2]),
               "residual_norm": float(result[3]), "operator_norm": float(result[5]),
               "condition_estimate": float(result[6]), "normal_residual_norm": float(result[7]),
               "solution_scaled_norm": float(result[8]), "converged": result[1] in (0, 1, 2, 4, 5)}
        components.append(row)
        if callback is not None:
            callback({"event": "lsqr_component", **row, "elapsed_sec": time.monotonic()-started})
    _sync(points.device)
    elapsed = time.monotonic()-started
    coeff = torch.as_tensor(coefficients, device=points.device, dtype=targets_mm.dtype)
    if not torch.isfinite(coeff).all():
        raise FloatingPointError("Nonfinite LSQR coefficients")
    return coeff, {"solver": "scipy_lsqr_matrix_free_column_scaled", "dtype": "float64",
                   "components": components, "converged": all(c["converged"] for c in components),
                   "basis_setup_time": setup_time, "fit_solver_time": elapsed-setup_time,
                   "total_fit_time": elapsed, "sample_count": operator.n,
                   "scalar_feature_count": operator.m, "cached_sample_design": operator.cached is not None,
                   "cache_bytes": 0 if operator.cached is None else operator.cached.numel()*operator.cached.element_size(),
                   "damping": 0., "atol": float(config.get("atol", 1e-6)),
                   "btol": float(config.get("btol", 1e-6)), "conditioning": operator.conditioning,
                   "minimum_norm_caveat": "LSQR scaled-space/minimum-correction choice may depend on nullspace and initialization; no unique coefficient solution claimed"}


def _select_values(field, affine, indices, points):
    if points is not None:
        return sample_field_points(field, points, affine)
    if indices is None:
        raise ValueError("Either flat voxel indices or physical points are required")
    return field.reshape(-1, 3)[indices.to(device=field.device, dtype=torch.long)]


def optimize_nonlinear(basis, initial, shape, affine, fit_indices=None, fit_targets_mm=None,
                       mode="svf", validation_indices=None, validation_targets_mm=None,
                       config=None, callback=None, *, fit_points_mm=None,
                       validation_points_mm=None):
    """Fit renderer endpoints with normalized Adam and recorded stopping.

    Physical-point arguments permit endpoint-aligned coarse integration grids;
    targets and points always remain in the same native world-mm convention.
    Caller owns path-wide step budgets and final native-grid refinement.
    """
    config = {} if config is None else dict(config)
    if fit_targets_mm is None or initial.ndim != 2 or initial.shape[-1] != 3:
        raise ValueError("Require initial M3 coefficients and N3 endpoint targets")
    device, dtype = initial.device, initial.dtype
    target = fit_targets_mm.to(device=device, dtype=dtype)
    h = _affine(affine, device=device, dtype=dtype)
    if fit_points_mm is None:
        # Recover only fit-point coordinates; no full-world grid allocation.
        indices = fit_indices.to(device=device, dtype=torch.long)
        d = torch.div(indices, int(shape[1])*int(shape[2]), rounding_mode="floor")
        remainder = indices % (int(shape[1])*int(shape[2]))
        xyz = torch.stack((d, torch.div(remainder, int(shape[2]), rounding_mode="floor"),
                           remainder % int(shape[2])), -1).to(dtype)
        sample_points = xyz @ h[:3, :3].T + h[:3, 3]
    else:
        sample_points = fit_points_mm.to(device=device, dtype=dtype)
    if len(target) != len(sample_points) or target.shape[1:] != (3,):
        raise ValueError("Fitting sample/target shape mismatch")
    if not torch.isfinite(initial).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite fitting initialization/targets")
    chunk = int(config.get("chunk", 16384))
    # Full declared box, shared with LSQR and independent of fitting ROI/grid.
    scales, conditioning = conditioning_scales(basis, sample_points, config)
    parameter = torch.nn.Parameter(initial.detach().clone()*scales[:, None])
    optimizer = torch.optim.Adam([parameter], lr=float(config.get("lr", .03)))
    maximum = int(config.get("max_steps", 1000))
    minimum = int(config.get("min_steps", 150))
    check_interval = int(config.get("check_interval", 25))
    patience = int(config.get("patience", 5))
    relative_tol = float(config.get("relative_tol", 1e-4))
    if maximum < 0 or minimum < 0 or check_interval < 1 or patience < 1:
        raise ValueError("Invalid optimizer budget/stopping configuration")
    trace, best_loss, best_coeff, plateau = [], math.inf, initial.detach().clone(), 0
    previous_validation = None
    _sync(device)
    started = time.monotonic()
    stopping_reason = "iteration_limit"
    for step in range(maximum+1):
        optimizer.zero_grad(set_to_none=True)
        coeff = parameter / scales[:, None]
        field = render_field(basis, coeff, shape, h, mode, chunk=chunk,
                              steps=int(config.get("steps", 7)), rho=float(config.get("rho", .9)))
        prediction = _select_values(field, h, fit_indices, fit_points_mm)
        loss = (prediction-target).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {mode} loss at step {step}")
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_coeff = value, coeff.detach().clone()
        checked = step % check_interval == 0 or step == maximum
        if checked:
            with torch.no_grad():
                if validation_targets_mm is not None:
                    val = _select_values(field, h, validation_indices, validation_points_mm)
                    val_target = validation_targets_mm.to(device=device, dtype=dtype)
                    validation = float((val-val_target).square().sum(-1).mean())
                else:
                    validation = value
            if not math.isfinite(validation):
                raise FloatingPointError("Nonfinite validation loss")
            if previous_validation is not None and step >= minimum:
                improvement = (previous_validation-validation)/max(abs(previous_validation), 1e-12)
                plateau = plateau+1 if improvement < relative_tol else 0
            previous_validation = validation
            row = {"event": "optimizer_check", "step": step, "loss_mm2": value,
                   "validation_mm2": validation, "best_training_loss_mm2": best_loss,
                   "elapsed_sec": time.monotonic()-started}
            if device.type == "cuda":
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            trace.append(row)
            if callback is not None:
                callback(row)
            if step >= minimum and plateau >= patience:
                stopping_reason = "validation_plateau"
                break
        if step == maximum:
            break
        loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite/missing {mode} gradient at step {step}")
        optimizer.step()
        del loss, field, prediction, coeff
    _sync(device)
    elapsed = time.monotonic()-started
    return best_coeff, {"solver": "column_rms_normalized_adam", "mode": mode,
                        "actual_steps": step, "function_evaluations": step+1,
                        "gradient_evaluations": step, "stopping_reason": stopping_reason,
                        "converged": stopping_reason == "validation_plateau",
                        "best_training_loss_mm2": best_loss, "fit_solver_time": elapsed,
                        "trace": trace, "integration_shape": list(shape),
                        "integration_steps": int(config.get("steps", 7)),
                        "lr_normalized": float(config.get("lr", .03)),
                        "sampling": "physical_points" if fit_points_mm is not None else "voxel_centers",
                        "lbfgs_evaluations": 0, "conditioning": conditioning}
