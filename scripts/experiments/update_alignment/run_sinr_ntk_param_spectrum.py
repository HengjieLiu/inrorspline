#!/usr/bin/env python
"""Randomized parameter-space spectrum probe for SINR CP generator.

The nonzero eigenvalues of the CP-space kernel J J^T equal those of the
parameter-space Gram J^T J. This script estimates a finite effective spectrum
inside a randomized parameter subspace. It is not an exact smallest-nonzero
eigenvalue computation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_sinr_ntk_conditioning as ntk  # noqa: E402


def parse_csv_list(text: str) -> List[str]:
    return [token for token in str(text).replace(" ", "").split(",") if token]


def flatten_param_dict(action: ntk.KernelAction, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1).float() for name in action.names], dim=0)


def unflatten_param_vector(action: ntk.KernelAction, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for name in action.names:
        ref = action.params[name]
        count = int(ref.numel())
        out[name] = flat[offset : offset + count].reshape_as(ref).to(device=ref.device, dtype=ref.dtype)
        offset += count
    if offset != int(flat.numel()):
        raise ValueError(f"Unused flat parameter entries: consumed {offset}, total {flat.numel()}")
    return out


def h_param_matvec(action: ntk.KernelAction, vector: torch.Tensor) -> torch.Tensor:
    tangent = unflatten_param_vector(action, vector)
    jv = action.jvp(tangent)
    jt_j_v = action.vjp(jv)
    return flatten_param_dict(action, jt_j_v).detach()


def randomized_ritz(
    action: ntk.KernelAction,
    *,
    dim: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    param_count = sum(int(action.params[name].numel()) for name in action.names)
    gen = torch.Generator(device=action.control_coords.device)
    gen.manual_seed(int(seed))
    omega = torch.randn((param_count, int(dim)), generator=gen, device=action.control_coords.device, dtype=torch.float32)
    # Standard randomized range finder for the PSD operator H = J^T J. Using
    # Q = orth(H Omega) makes the Ritz values track the leading supported
    # spectrum better than projecting onto raw random directions.
    y_cols: List[torch.Tensor] = []
    for idx in range(int(dim)):
        y_cols.append(h_param_matvec(action, omega[:, idx]))
    y = torch.stack(y_cols, dim=1)
    q, _r = torch.linalg.qr(y, mode="reduced")
    hq_cols: List[torch.Tensor] = []
    for idx in range(int(dim)):
        hq_cols.append(h_param_matvec(action, q[:, idx]))
    hq = torch.stack(hq_cols, dim=1)
    small = q.T @ hq
    small = 0.5 * (small + small.T)
    eigvals = torch.linalg.eigvalsh(small).detach().cpu()
    return eigvals, small.detach().cpu()


def summarize_eigs(eigs: torch.Tensor, *, state: str, dim: int, matrix_summary: Mapping[str, object]) -> Dict[str, object]:
    vals = eigs.detach().cpu().numpy().astype(np.float64)
    positive = vals[vals > 1e-12]
    row: Dict[str, object] = {
        "state": state,
        "ritz_dim": int(dim),
        "cp_coefficients": int(matrix_summary["cp_coefficients"]),
        "sinr_param_count": int(matrix_summary["sinr_param_count"]),
        "rank_upper_bound": int(matrix_summary["rank_upper_bound"]),
        "nullity_lower_bound": int(matrix_summary["nullity_lower_bound"]),
        "ritz_min": float(np.min(vals)),
        "ritz_p05": float(np.percentile(vals, 5)),
        "ritz_p50": float(np.percentile(vals, 50)),
        "ritz_p95": float(np.percentile(vals, 95)),
        "ritz_max": float(np.max(vals)),
        "positive_count_tol1e-12": int(positive.size),
    }
    denom = float(np.min(positive)) if positive.size else float("nan")
    row["ritz_condition_proxy_max_over_min_positive"] = float(np.max(vals) / denom) if denom > 0 else float("nan")
    row["ritz_condition_proxy_p95_over_p05"] = (
        float(np.percentile(vals, 95) / np.percentile(vals, 5)) if np.percentile(vals, 5) > 0 else float("nan")
    )
    row["note"] = "Randomized range-finder Ritz proxy for leading parameter-space spectrum; not exact smallest nonzero eigenvalue."
    return row


def run(args: argparse.Namespace) -> Dict[str, object]:
    review_root = Path(args.review_root)
    out_root = Path(args.out_root)
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "tables").mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = {
            "status": "dry_run",
            "states": parse_csv_list(args.states),
            "ritz_dim": int(args.ritz_dim),
            "review_root": str(review_root),
            "out_root": str(out_root),
        }
        ntk.write_json(review_root / "param_spectrum_dry_run_plan.json", plan)
        ntk.write_json(out_root / "param_spectrum_dry_run_plan.json", plan)
        return plan

    device = ntk.c.set_cuda_device(str(args.gpu))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    control_coords, control_shape = ntk.c.bspline_control_coords(ntk.c.IMAGE_SHAPE, int(args.cps), device)
    rows: List[Dict[str, object]] = []
    eig_rows: List[Dict[str, object]] = []
    start = time.time()
    for state in parse_csv_list(args.states):
        generator = ntk.build_sinr_generator(args, control_shape, device)
        if state == "final":
            ntk.load_final_state(generator, Path(args.sinr_final_run))
        elif state != "init":
            raise ValueError(f"Unsupported state {state!r}")
        action = ntk.KernelAction(generator, control_coords, control_shape)
        matrix_summary = ntk.matrix_size_summary(generator, control_shape)
        eigs, _small = randomized_ritz(action, dim=int(args.ritz_dim), seed=int(args.seed) + 3000 + len(state))
        row = summarize_eigs(eigs, state=state, dim=int(args.ritz_dim), matrix_summary=matrix_summary)
        rows.append(row)
        for idx, val in enumerate(eigs.tolist()):
            eig_rows.append({"state": state, "ritz_dim": int(args.ritz_dim), "ritz_index": int(idx), "eigenvalue": float(val)})
    summary = {
        "status": "completed",
        "states": parse_csv_list(args.states),
        "ritz_dim": int(args.ritz_dim),
        "rows": len(rows),
        "elapsed_sec": float(time.time() - start),
        "review_root": str(review_root),
        "out_root": str(out_root),
        "note": "Randomized range-finder Ritz proxy for leading parameter-space spectrum; exact full-space condition number remains infinite from CP nullity.",
    }
    ntk.write_csv(review_root / "tables/ntk_param_ritz_summary.csv", rows)
    ntk.write_csv(review_root / "tables/ntk_param_ritz_eigenvalues.csv", eig_rows)
    ntk.write_json(review_root / "param_spectrum_summary.json", summary)
    ntk.write_json(out_root / "param_spectrum_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=str, default="init,final")
    parser.add_argument("--ritz-dim", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--cps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--sinr-final-run", type=Path, default=ntk.DEFAULT_SINR_FINAL_RUN)
    parser.add_argument("--out-root", type=Path, default=ntk.DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=ntk.DEFAULT_REVIEW_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({"sinr_ntk_param_spectrum": result}, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
