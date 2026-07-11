#!/usr/bin/env python
"""Sanity checks for the shared DCTField decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import common_dctfield as dctfield  # noqa: E402


def check_shape_and_grad(shape: tuple[int, int, int], k: int, device: torch.device) -> dict[str, object]:
    decoder = dctfield.DCTFieldDecoder(shape, k, device=device, dtype=torch.float32)
    coeff = torch.zeros((3, k, k, k), dtype=torch.float32, device=device, requires_grad=True)
    dense_zero = decoder.decode(coeff)
    if tuple(dense_zero.shape) != (*shape, 3):
        raise RuntimeError(f"Bad decoded shape for shape={shape}, k={k}: {tuple(dense_zero.shape)}")
    if float(dense_zero.abs().max().detach().cpu()) != 0.0:
        raise RuntimeError(f"Zero coefficients did not decode to zero for shape={shape}, k={k}")
    coeff2 = torch.zeros((3, k, k, k), dtype=torch.float32, device=device, requires_grad=True)
    coeff2.data[:, 0, 0, 0] = 1.0
    dense = decoder.decode(coeff2)
    loss = dense.square().mean()
    loss.backward()
    grad_norm = float(coeff2.grad.detach().norm().cpu()) if coeff2.grad is not None else 0.0
    if grad_norm <= 0.0:
        raise RuntimeError(f"No coefficient gradient for shape={shape}, k={k}")
    return {
        "shape_zyx": list(shape),
        "dct_size": int(k),
        "decoded_shape": list(dense.shape),
        "zero_abs_max": float(dense_zero.abs().max().detach().cpu()),
        "grad_norm": grad_norm,
        "param_count": dctfield.dct_param_count(k),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        check_shape_and_grad((16, 18, 20), 8, device),
        check_shape_and_grad((32, 34, 36), 16, device),
    ]
    try:
        dctfield.DCTFieldDecoder((8, 8, 8), 9, device=device, dtype=torch.float32)
    except ValueError as exc:
        rows.append({"oversized_check": "passed", "message": str(exc)})
    else:
        raise RuntimeError("Expected oversized DCT size to raise ValueError")
    print(json.dumps({"status": "passed", "device": str(device), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
