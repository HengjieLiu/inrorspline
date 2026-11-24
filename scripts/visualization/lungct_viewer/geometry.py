"""Physical geometry and pull inversion; independent of registration optimizers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def to_lps(points_xyz, spacing_xyz, depth):
    p = np.asarray(points_xyz, dtype=np.float64) * spacing_xyz
    p[..., 2] = (depth - 1) * spacing_xyz[2] - p[..., 2]
    return p


def display_geometry(audit):
    shape = np.asarray(audit['registration_shape_zyx'])
    crop = np.asarray(audit['crop_shape_zyx'])
    bounds = np.asarray(audit['crop_bounds_zyx_inclusive'])
    sp = np.asarray(audit['native_spacing_zyx_mm'])
    effective = (crop - 1) * sp / (shape - 1)
    origin = to_lps(bounds[:, 0][::-1], sp[::-1], audit['raw_shape_zyx'][0])
    step = effective[::-1] * [1, 1, -1]
    end = origin + step * (shape[::-1] - 1)
    return dict(shape_zyx=shape.tolist(), origin_lps_mm=origin.tolist(),
                step_lps_mm=step.tolist(), min_lps_mm=np.minimum(origin, end).tolist(),
                max_lps_mm=np.maximum(origin, end).tolist(),
                frame='pair-local LPS; raw X=L, Y=P, Z=I; not scanner DICOM')


class PhysicalField:
    """Sample original field at original native XYZ, returning raw XYZ mm."""

    def __init__(self, data, spacing_xyz, bounds_xyz, reg_geometry=None, ptv=False):
        self.data = data
        self.spacing = np.asarray(spacing_xyz, dtype=np.float64)
        self.bounds = np.asarray(bounds_xyz, dtype=np.float64)
        self.ptv = ptv
        if reg_geometry:
            self.origin = np.asarray(reg_geometry['crop_bounds_zyx_inclusive'])[:, 0][::-1]
            self.scale = ((np.asarray(reg_geometry['crop_shape_zyx']) - 1) /
                          (np.asarray(reg_geometry['registration_shape_zyx']) - 1))[::-1]
        else:
            self.origin = np.zeros(3)
            self.scale = np.ones(3)

    def valid(self, points):
        p = np.asarray(points)
        return np.all((p >= self.bounds[:, 0] - 1e-6) & (p <= self.bounds[:, 1] + 1e-6), axis=-1)

    def sample(self, points):
        p = np.asarray(points, dtype=np.float64)
        flat = p.reshape(-1, 3)
        coords = ((flat - self.origin) / self.scale)[:, ::-1].T
        shape = np.asarray(self.data.shape[1:] if self.ptv else self.data.shape[:3])
        coords = np.clip(coords, 0, shape[:, None] - 1)
        values = np.stack([map_coordinates(self.data[c] if self.ptv else self.data[..., 2-c],
                                          coords, order=1, mode='nearest', prefilter=False)
                           for c in range(3)], axis=-1).astype(np.float64)
        values[~self.valid(flat)] = np.nan
        return values.reshape(p.shape)


def invert_points(field, moving_xyz, tolerance=0.05):
    """Multistart damped Newton; no fixed/ground-truth point used as a seed."""
    sp = field.spacing
    low, high = field.bounds[:, 0]*sp, field.bounds[:, 1]*sp
    axes = [np.linspace(l, h, max(2, int(np.ceil((h-l)/4))+1)) for l, h in zip(low, high)]
    coarse = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    mapped = coarse + field.sample(coarse/sp)
    keep = np.isfinite(mapped).all(axis=1)
    tree = cKDTree(mapped[keep]); coarse = coarse[keep]
    target = np.asarray(moving_xyz, dtype=np.float64)*sp
    _, ids = tree.query(target, k=min(3, len(coarse)))
    ids = np.asarray(ids).reshape(len(target), -1)
    seeds = np.concatenate([coarse[ids], np.clip(target[:, None], low, high)], axis=1)
    q = seeds.reshape(-1, 3).copy()
    t = np.repeat(target, seeds.shape[1], axis=0)

    def residual(x):
        return x + field.sample(x/sp) - t

    for _ in range(45):
        r = residual(q)
        norms = np.linalg.norm(r, axis=1)
        active = np.isfinite(norms) & (norms > 0.005)
        if not active.any():
            break
        jac = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
        for axis in range(3):
            a, b = q.copy(), q.copy()
            a[:, axis] = np.maximum(low[axis], q[:, axis]-0.2)
            b[:, axis] = np.minimum(high[axis], q[:, axis]+0.2)
            jac[:, :, axis] += (field.sample(b/sp)-field.sample(a/sp)) / (b[:, axis]-a[:, axis])[:, None]
        jac[~active] = np.eye(3); r[~active] = 0
        step = np.einsum('nij,nj->ni', np.linalg.pinv(jac, rcond=1e-7), r)
        step *= np.minimum(1, 10/np.maximum(np.linalg.norm(step, axis=1), 1e-12))[:, None]
        improved = np.zeros(len(q), dtype=bool)
        for factor in (1, .5, .25, .125, .0625):
            trial = q-factor*step
            ok = np.all((trial >= low) & (trial <= high), axis=1)
            err = np.linalg.norm(residual(trial), axis=1)
            take = active & ~improved & ok & np.isfinite(err) & (err < norms)
            q[take] = trial[take]; improved |= take
        if not improved.any():
            break
    errors = np.linalg.norm(residual(q), axis=1).reshape(seeds.shape[:2])
    q = q.reshape(seeds.shape)
    results = []
    for i in range(len(target)):
        good = np.flatnonzero(np.isfinite(errors[i]) & (errors[i] <= tolerance))
        if not len(good):
            results.append(dict(status='unresolved', point_native_xyz=None, inverse_residual_mm=None, roots=0))
            continue
        roots = []
        for j in sorted(good, key=lambda k: (errors[i, k], k)):
            if all(np.linalg.norm(q[i, j]-q[i, k]) > .25 for k in roots):
                roots.append(j)
        j = roots[0]
        results.append(dict(status='ambiguous' if len(roots)>1 else 'resolved',
                            point_native_xyz=(q[i, j]/sp).tolist(),
                            inverse_residual_mm=float(errors[i, j]), roots=len(roots)))
    return results


def compact(array, tolerance):
    a = np.asarray(array)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite display array')
    half = a.astype('<f2')
    error = float(np.max(np.abs(half.astype(np.float32)-a)))
    return (half, error) if np.isfinite(half).all() and error <= tolerance else (a.astype('<f4'), 0.)
