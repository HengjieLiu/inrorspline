"""Opt-in experiment backends; production defaults are never replaced on disk.

The CP-space BE uses finite differences followed by QR factors of sampled
lower-degree spline bases. It is a sum of squares, not a cancellation-prone
quadratic form. Factors encode the exact finite output crop.
"""
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from scripts.experiments.oasis_alpha_sweep import run_one

# run_one uses its historical top-level import. Patch that exact module, not a
# second package-qualified copy of common.py.
c = run_one.c

TERMS = (((2, 0, 0), 1), ((0, 2, 0), 1), ((0, 0, 2), 1),
         ((1, 1, 0), 2), ((1, 0, 1), 2), ((0, 1, 1), 2))
VARIANTS = {"reference": (), "cache": ("cache",), "batch": ("batch",),
            "ncc": ("ncc",), "buffer": ("buffer",), "cpbe": ("cpbe",),
            "combined": ("cache", "batch", "ncc", "buffer", "cpbe")}


def batched_conv(data, *, kernel, dim, stride=1, dilation=1, padding=0, transpose=False):
    value = data.to(dtype=kernel.dtype).transpose(dim, -1)
    shape = value.shape
    value = value.reshape(-1, 1, shape[-1])
    fn = F.conv_transpose1d if transpose else F.conv1d
    value = fn(value, kernel.reshape(1, 1, -1), stride=stride,
               dilation=dilation, padding=padding)
    return value.reshape(*shape[:-1], value.shape[-1]).transpose(dim, -1)


def axis_basis(n, m, spacing, order):
    """L such that the order-q derivative equals L @ diff(control,q)."""
    t = np.arange(n, dtype=np.float64)[:, None] / spacing
    t = np.abs(t + 1 - order / 2 - np.arange(m - order)[None, :])
    if order == 0:
        value = np.where(t < 1, 2 / 3 - t*t + t*t*t / 2,
                         np.where(t < 2, (2-t)**3 / 6, 0))
    elif order == 1:
        value = np.where(t < .5, .75 - t*t,
                         np.where(t < 1.5, .5*(1.5-t)**2, 0))
    elif order == 2:
        value = np.maximum(1-t, 0)
    else:
        raise ValueError(order)
    return value / spacing**order


class Backend:
    def __init__(self, features):
        self.features = tuple(features)
        self.kernels = {}
        self.factors = {}
        self.fixed_stats = {}
        self.original = {}

    def clear_case(self):
        self.fixed_stats.clear()

    def kernel(self, stride, derivative=0, dtype=None, device=None):
        key = (int(stride), int(derivative), dtype, str(device))
        if key not in self.kernels:
            self.kernels[key] = self.original["cubic_bspline1d"](
                stride, derivative, dtype=dtype, device=device)
        return self.kernels[key]

    def ncc(self, fixed, moving, eps=1e-10):
        if fixed.shape != moving.shape:
            raise ValueError("NCC shape mismatch")
        key = (fixed.data_ptr(), tuple(fixed.shape), fixed.dtype, str(fixed.device))
        if key not in self.fixed_stats:
            self.fixed_stats[key] = (fixed, fixed - fixed.mean(), c.StableStd.apply(fixed))
        _, centered, std = self.fixed_stats[key]
        return -torch.mean(centered * (moving - moving.mean())) / (
            std * c.StableStd.apply(moving) + eps)

    def factor(self, n, m, spacing, order, value):
        key = (n, m, spacing, order, value.dtype, str(value.device))
        if key not in self.factors:
            r = np.linalg.qr(axis_basis(n, m, spacing, order), mode="reduced")[1]
            self.factors[key] = torch.as_tensor(r.copy(), dtype=value.dtype, device=value.device)
        return self.factors[key]

    def be(self, controls, image_shape, cps, mask=None, *, component_reduction="mean"):
        if mask is not None or component_reduction != "mean":
            raise ValueError("CP BE experiment supports only unmasked component mean")
        total = controls.new_zeros(())
        for orders, weight in TERMS:
            value = controls
            for axis, order in enumerate(orders):
                if order:
                    value = torch.diff(value, n=order, dim=axis)
            for axis, order in enumerate(orders):
                r = self.factor(int(image_shape[axis]), controls.shape[axis], cps, order, value)
                value = (value.movedim(axis, -1) @ r.T).movedim(-1, axis)
            total = total + weight * value.square().sum()
        return total / (3 * int(np.prod(image_shape)))

    @contextmanager
    def activate(self):
        replacements = {"cache": ("cubic_bspline1d", self.kernel),
                        "batch": ("conv1d", batched_conv),
                        "ncc": ("ncc_loss", self.ncc),
                        "cpbe": ("analytic_bspline_sampled_be", self.be)}
        try:
            for feature in self.features:
                if feature in replacements:
                    name, fn = replacements[feature]
                    self.original[name] = getattr(c, name)
                    setattr(c, name, fn)
            yield self
        finally:
            for name, fn in self.original.items():
                setattr(c, name, fn)


class LossBuffer:
    """Detached device history, flushed before every CPU consumer/check."""
    def __init__(self, epochs, device):
        self.values = torch.empty((epochs, 3), dtype=torch.float32, device=device)
        self.flushed = 0

    def append(self, index, loss, data, reg):
        self.values[index].copy_(torch.stack((loss.detach(), data.detach(), reg.detach())))

    def flush(self, rows):
        end = len(rows)
        if end > self.flushed:
            values = self.values[self.flushed:end].cpu().tolist()
            for row, value in zip(rows[self.flushed:end], values):
                row.update(zip(("loss", "data_loss", "reg_loss"), value))
            self.flushed = end
