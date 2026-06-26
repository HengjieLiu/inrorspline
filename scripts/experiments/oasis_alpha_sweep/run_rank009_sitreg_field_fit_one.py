#!/usr/bin/env python
"""Fit one OASIS model directly to the saved SITReg dense displacement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_one as image_runner  # noqa: E402
import run_rank009_greedy_field_fit_one as field_runner  # noqa: E402
from evaluate_saved_baselines_first10 import (  # noqa: E402
    DEFAULT_DL_ROOT,
    load_saved_disp_zyx,
    voxel_zyx_to_norm_dhw3,
)


METHODS = field_runner.METHODS
VARIANT = "oasis_rank009_sitreg_fit_modes_lr_sweep_v1"
OBJECTIVE = "sitreg_field_fit_mse"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1")
DEFAULT_SITREG_ROOT = (
    DEFAULT_DL_ROOT
    / "gpu8_OASIS_v1_bs4_sitreg_v0_run1_20250407165940"
    / "inference"
    / "inference_bestdice"
)
EXPECTED_RAW_SHAPE = "3,160,224,192"
EXPECTED_EVAL_SHAPE = "1,3,160,224,192"
EXPECTED_TARGET_SHAPE = "160,224,192,3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Recorded operating-point alpha; not used in MSE training.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--case-rank", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--field-chunk-size", type=int, default=200000)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--sitreg-root", type=Path, default=DEFAULT_SITREG_ROOT)
    parser.add_argument("--expected-original-case-index", type=int, default=88)
    parser.add_argument("--expected-fixed-id", type=str, default="0173")
    parser.add_argument("--expected-moving-id", type=str, default="0136")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-target-only", action="store_true")
    return parser.parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"sitreg_field_fit_{field_runner.field_method_key(args.method)}_rank{args.case_rank:03d}"
        f"_a{c.format_float(args.alpha)}_e{args.epochs}_lr{c.format_float(args.lr)}"
    )


def sitreg_disp_path(args: argparse.Namespace, fixed_id: str, moving_id: str) -> Path:
    return Path(args.sitreg_root) / f"disp_{fixed_id}_{moving_id}.nii.gz"


def shape_text(shape: Tuple[int, ...]) -> str:
    return ",".join(str(int(v)) for v in shape)


def load_sitreg_target(
    args: argparse.Namespace,
    *,
    fixed_id: str,
    moving_id: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    path = sitreg_disp_path(args, fixed_id, moving_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing saved SITReg displacement: {path}")
    loaded = load_saved_disp_zyx(path)
    if loaded.source_disp_raw_shape != EXPECTED_RAW_SHAPE:
        raise ValueError(f"SITReg raw shape {loaded.source_disp_raw_shape} != {EXPECTED_RAW_SHAPE}")
    if loaded.source_disp_eval_shape != EXPECTED_EVAL_SHAPE:
        raise ValueError(f"SITReg eval shape {loaded.source_disp_eval_shape} != {EXPECTED_EVAL_SHAPE}")
    if loaded.source_disp_storage_layout != "channel_first_zyx_repo_ras":
        raise ValueError(f"Unexpected SITReg storage layout: {loaded.source_disp_storage_layout}")
    if loaded.source_disp_reorientation != "none":
        raise ValueError(f"Unexpected SITReg reorientation: {loaded.source_disp_reorientation}")

    target_np = voxel_zyx_to_norm_dhw3(loaded.disp_zyx)
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    target_shape = shape_text(tuple(target.shape))
    if target_shape != EXPECTED_TARGET_SHAPE:
        raise ValueError(f"SITReg normalized target shape {target_shape} != {EXPECTED_TARGET_SHAPE}")

    audit: Dict[str, object] = {
        "source_kind": "dl_pretrained",
        "source_family": "bs",
        "source_method_key": "bs4_sitreg_v0",
        "source_method_label": "BS4 SITReg",
        "source_selector": "bestdice",
        "source_disp_path": str(path),
        "source_disp_component_order": loaded.source_disp_component_order,
        "source_disp_raw_shape": loaded.source_disp_raw_shape,
        "source_disp_shape": loaded.source_disp_eval_shape,
        "source_disp_storage_layout": loaded.source_disp_storage_layout,
        "source_disp_reorientation": loaded.source_disp_reorientation,
        "source_displacement_units": "voxel",
        "field_fit_target_units": "normalized_grid",
        "field_fit_target_shape": target_shape,
        "field_fit_target_extent": "full_dense_field_available",
    }
    return target, audit


def main() -> None:
    args = parse_args()
    args.field_fit_objective = OBJECTIVE
    run_name = build_run_name(args)
    key = field_runner.field_method_key(args.method)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite and not args.validate_target_only:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return

    total_start = time.time()
    started_at = c.now_text()
    device = c.set_cuda_device(args.gpu)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    batch = c.load_oasis_case(
        rank=int(args.case_rank),
        device=device,
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
    )
    image_runner.validate_expected_case(
        args,
        original_index=int(batch["original_case_index"]),
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
    )
    target_disp, target_audit = load_sitreg_target(
        args,
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
        device=device,
    )
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    target_audit.update(field_runner.field_domain_audit(fixed_mask))
    if args.validate_target_only:
        print(
            json.dumps(
                {
                    "case": {k: str(batch[k]) for k in ["case_rank", "original_case_index", "fixed_id", "moving_id"]},
                    "target_audit": target_audit,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "objective": OBJECTIVE,
                "method": args.method,
                "method_key": key,
                "gpu": args.gpu,
                "case_rank": batch["case_rank"],
                "original_case_index": batch["original_case_index"],
                "fixed_id": batch["fixed_id"],
                "moving_id": batch["moving_id"],
                "target_audit": target_audit,
                "alpha_recorded_not_used": args.alpha,
                "lr": args.lr,
                "epochs": args.epochs,
            },
            allow_nan=True,
        ),
        flush=True,
    )

    if args.method == "idir_original":
        metrics = field_runner.train_idir_field(args, target_disp, fixed_mask, device)
    else:
        metrics = field_runner.train_spline_field(args, target_disp, fixed_mask, device)
    metrics["started_at"] = started_at
    total_elapsed_sec = time.time() - total_start
    field_runner.write_outputs(args, batch, target_audit, target_disp, metrics, run_dir, run_name, total_elapsed_sec)


if __name__ == "__main__":
    main()
