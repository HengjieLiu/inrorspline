#!/usr/bin/env python
"""Backfill native analytic regularity columns for completed OASIS pre-runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


IDIR_METHOD_KEY = "idir_original_dense_siren_be_autograd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--autograd-chunk-size", type=int, default=20000)
    return parser.parse_args()


def load_torch(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def prune_formula(rows: List[Dict[str, object]], formula: str) -> List[Dict[str, object]]:
    return [row for row in rows if str(row.get("formula")) != formula]


def update_native_columns(
    summary: Dict[str, object],
    regularity_rows: List[Dict[str, object]],
    be_rows: List[Dict[str, object]],
    diffusion_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    if len(regularity_rows) > 1:
        raise RuntimeError(f"Expected at most one regularity row, got {len(regularity_rows)}")
    reg: Dict[str, object] = dict(regularity_rows[0]) if regularity_rows else {}
    reg.update(c.be_rows_to_columns(be_rows))
    reg.update(c.idir_autograd_be_rows_to_columns(be_rows))
    c.add_native_analytic_be_columns(reg)
    reg.update(c.diffusion_rows_to_columns(diffusion_rows))
    reg.update(c.idir_autograd_diffusion_rows_to_columns(diffusion_rows))
    c.add_native_analytic_diffusion_columns(reg)
    summary.update(reg)
    return reg


def backfill_idir(
    run_dir: Path,
    summary: Dict[str, object],
    be_rows: List[Dict[str, object]],
    diffusion_rows: List[Dict[str, object]],
    device: torch.device,
    chunk_size: int,
) -> None:
    state_path = run_dir / "model_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = load_torch(state_path, device)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise RuntimeError(f"IDIR model state missing model_state_dict: {state_path}")

    model = c.build_siren(
        int(summary["hidden_dim"]),
        int(summary["hidden_layers"]),
        float(summary["omega"]),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    batch = c.load_oasis_case(
        rank=int(float(summary["case_rank"])),
        device=device,
        oasis_root=Path(str(summary["sorted_pair_file"])).parent,
        pair_file=Path(str(summary["pair_file"])),
    )
    expected_ids = (str(summary["fixed_id"]), str(summary["moving_id"]))
    actual_ids = (str(batch["fixed_id"]), str(batch["moving_id"]))
    if actual_ids != expected_ids:
        raise RuntimeError(f"Case ID mismatch for {run_dir}: actual {actual_ids}, expected {expected_ids}")
    fixed_mask = batch["fixed_mask"]
    if not isinstance(fixed_mask, torch.Tensor):
        raise RuntimeError("Loaded OASIS batch has no tensor fixed_mask")

    be_rows[:] = prune_formula(be_rows, "idir_autograd")
    diffusion_rows[:] = prune_formula(diffusion_rows, "idir_autograd")
    be_rows.extend(
        c.idir_autograd_be_metric_rows(
            model,
            fixed_mask.detach().bool(),
            device,
            chunk_size,
        )
    )
    diffusion_rows.extend(
        c.idir_autograd_diffusion_metric_rows(
            model,
            fixed_mask.detach().bool(),
            device,
            chunk_size,
        )
    )
    summary["be_type"] = "be_native_analytic"
    summary["diffusion_type"] = "diffusion_native_analytic"
    summary["regularizer"] = "be_native_analytic"
    summary["native_analytic_be_formula"] = "idir_coordinate_autograd_hessian"
    summary["native_analytic_diffusion_formula"] = "idir_coordinate_autograd_jacobian"
    summary["idir_autograd_be_chunk_size"] = int(chunk_size)
    summary["idir_autograd_diffusion_chunk_size"] = int(chunk_size)


def backfill_run(run_dir: Path, device: torch.device, chunk_size: int) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    regularity_path = run_dir / "regularity_metrics.csv"
    be_path = run_dir / "be_metrics.csv"
    diffusion_path = run_dir / "diffusion_metrics.csv"
    for path in (summary_path, regularity_path, be_path, diffusion_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text())
    be_rows: List[Dict[str, object]] = list(c.read_csv(be_path))
    diffusion_rows: List[Dict[str, object]] = list(c.read_csv(diffusion_path))
    regularity_rows: List[Dict[str, object]] = list(c.read_csv(regularity_path))
    method_key = str(summary.get("method_key"))

    if method_key == IDIR_METHOD_KEY:
        backfill_idir(run_dir, summary, be_rows, diffusion_rows, device, chunk_size)
    else:
        summary["be_type"] = "be_bspline_analytic"
        summary["diffusion_type"] = "diffusion_bspline_analytic"
        summary["regularizer"] = "be_bspline_analytic"
        summary["native_analytic_be_formula"] = "bspline_analytic"
        summary["native_analytic_diffusion_formula"] = "bspline_analytic"

    reg = update_native_columns(summary, regularity_rows, be_rows, diffusion_rows)
    c.write_csv(regularity_path, [reg])
    c.write_csv(be_path, be_rows)
    c.write_csv(diffusion_path, diffusion_rows)
    c.write_json(summary_path, summary)
    return {
        "run_dir": str(run_dir),
        "method_key": method_key,
        "native_be": summary.get("be_native_analytic_fixed_fg_voxel"),
        "native_diffusion": summary.get("diffusion_native_analytic_fixed_fg_voxel"),
    }


def main() -> None:
    args = parse_args()
    device = c.set_cuda_device(str(args.gpu))
    run_dirs = sorted(path.parent for path in Path(args.out_root).glob("*/*/summary.json"))
    if not run_dirs:
        raise RuntimeError(f"No run summaries found under {args.out_root}")
    rows = [backfill_run(run_dir, device, int(args.autograd_chunk_size)) for run_dir in run_dirs]
    print(json.dumps({"updated_runs": len(rows), "rows": rows}, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
