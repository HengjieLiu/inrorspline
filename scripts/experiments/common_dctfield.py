#!/usr/bin/env python
"""Shared DCTField utilities for experiment runners."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


def normalize_dct_size(dct_size: int | Sequence[int], image_shape: Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(dct_size, int):
        size = (int(dct_size), int(dct_size), int(dct_size))
    else:
        if len(dct_size) != 3:
            raise ValueError(f"Expected 3 DCT sizes, got {tuple(dct_size)}")
        size = tuple(int(v) for v in dct_size)
    shape = tuple(int(v) for v in image_shape)
    if any(v <= 0 for v in size):
        raise ValueError(f"DCT sizes must be positive, got {size}")
    too_large = [f"{k}>{n}" for k, n in zip(size, shape) if k > n]
    if too_large:
        raise ValueError(f"DCT size exceeds image shape: size_zyx={size}, shape_zyx={shape}, bad={too_large}")
    return size


def dct_orthonormal_basis(
    length: int,
    n_coefficients: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    length_i = int(length)
    n_coeff_i = int(n_coefficients)
    if n_coeff_i <= 0 or n_coeff_i > length_i:
        raise ValueError(f"Invalid DCT coefficient count {n_coeff_i} for length {length_i}")
    n = torch.arange(length_i, dtype=dtype, device=device).unsqueeze(1)
    k = torch.arange(n_coeff_i, dtype=dtype, device=device).unsqueeze(0)
    basis = torch.cos((math.pi / float(length_i)) * (n + 0.5) * k)
    scale = torch.full((n_coeff_i,), math.sqrt(2.0 / float(length_i)), dtype=dtype, device=device)
    scale[0] = math.sqrt(1.0 / float(length_i))
    return basis * scale.view(1, n_coeff_i)


class DCTFieldDecoder:
    """Decode low-frequency DCT coefficients into a dense displacement field."""

    basis_name = "orthonormal_dct_ii_separable_zyx"

    def __init__(
        self,
        image_shape: Sequence[int],
        dct_size: int | Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.image_shape = tuple(int(v) for v in image_shape)
        self.dct_size = normalize_dct_size(dct_size, self.image_shape)
        kz, ky, kx = self.dct_size
        d, h, w = self.image_shape
        self.basis_z = dct_orthonormal_basis(d, kz, device=device, dtype=dtype)
        self.basis_y = dct_orthonormal_basis(h, ky, device=device, dtype=dtype)
        self.basis_x = dct_orthonormal_basis(w, kx, device=device, dtype=dtype)

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        expected = (3, *self.dct_size)
        if tuple(int(v) for v in coefficients.shape) != expected:
            raise ValueError(f"Expected coefficients {expected}, got {tuple(coefficients.shape)}")
        tmp_z = torch.einsum("zk,cklm->czlm", self.basis_z, coefficients)
        tmp_zy = torch.einsum("yl,czlm->czym", self.basis_y, tmp_z)
        dense_czyx = torch.einsum("xm,czym->czyx", self.basis_x, tmp_zy)
        return dense_czyx.permute(1, 2, 3, 0).contiguous()


def dct_param_count(dct_size: int | Sequence[int]) -> int:
    if isinstance(dct_size, int):
        return 3 * int(dct_size) ** 3
    kz, ky, kx = (int(v) for v in dct_size)
    return 3 * kz * ky * kx
