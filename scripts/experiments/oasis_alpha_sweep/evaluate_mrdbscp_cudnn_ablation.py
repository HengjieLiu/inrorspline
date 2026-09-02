#!/usr/bin/env python
"""Evaluate saved MR-D-BSCP cuDNN-ablation DVFs after timing is complete."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402
from run_mrdbscp_cudnn_ablation import (  # noqa: E402
    CANDIDATE_NAME,
    CONFIGS,
    config_out_root,
    config_review_root,
    parse_case_ranks,
    run_dir,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--case-ranks", type=str, default="0-4")
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--variant", type=str, default="oasis_seed3_first5_mrdbscp_cudnn_ablation_v1")
    return parser.parse_args()


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def saved_dvf_path(run_path: Path, summary: Mapping[str, object]) -> Path:
    recorded = summary.get("dvf_file_path")
    if recorded:
        candidate = Path(str(recorded))
        if candidate.exists():
            return candidate
        fallback = run_path / candidate.name
        if fallback.exists():
            return fallback
    candidates = sorted(run_path.glob("disp_norm_*.npy"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one dense DVF in {run_path}, found {len(candidates)}")
    return candidates[0]


def evaluate_config(args: argparse.Namespace, config: object, device: torch.device) -> List[Dict[str, object]]:
    ranks = parse_case_ranks(str(args.case_ranks))
    rows: List[Dict[str, object]] = []
    for rank in ranks:
        started = time.perf_counter()
        target = run_dir(args, config, rank)  # type: ignore[arg-type]
        summary_path = target / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing timing-run summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"Expected JSON object in {summary_path}")
        if summary.get("dense_output_units") != "normalized_grid_xyz_displacement":
            raise ValueError(f"Unexpected DVF coordinate units in {summary_path}: {summary.get('dense_output_units')!r}")
        if summary.get("grid_order") != "pytorch":
            raise ValueError(f"Unexpected DVF grid order in {summary_path}: {summary.get('grid_order')!r}")
        dvf_path = saved_dvf_path(target, summary)
        disp_norm = np.load(dvf_path, allow_pickle=False)
        expected_shape = tuple(int(value) for value in c.IMAGE_SHAPE) + (3,)
        if disp_norm.shape != expected_shape:
            raise ValueError(f"{dvf_path} has shape {disp_norm.shape}, expected {expected_shape}")
        if disp_norm.dtype != np.float32:
            raise ValueError(f"{dvf_path} has dtype {disp_norm.dtype}, expected float32")
        if not np.isfinite(disp_norm).all():
            raise ValueError(f"{dvf_path} contains non-finite values")

        batch = c.load_oasis_case(
            rank=int(rank),
            device=device,
            oasis_root=Path(args.oasis_root),
            pair_file=Path(args.pair_file),
        )
        if str(batch["fixed_id"]) != str(summary["fixed_id"]) or str(batch["moving_id"]) != str(summary["moving_id"]):
            raise ValueError(f"Case ID mismatch for {target}")
        dense_disp = torch.from_numpy(disp_norm).to(device=device, dtype=torch.float32)
        id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
        metrics = c.evaluate_from_dense_coords(id_grid + dense_disp, batch)
        accuracy = c.accuracy_row(metrics, str(batch["fixed_id"]), str(batch["moving_id"]))
        row: Dict[str, object] = {
            "config": config.slug,  # type: ignore[attr-defined]
            "config_label": config.label,  # type: ignore[attr-defined]
            "cudnn_benchmark": config.benchmark,  # type: ignore[attr-defined]
            "cudnn_deterministic": config.deterministic,  # type: ignore[attr-defined]
            "gpu": str(args.gpu),
            "case_rank": int(rank),
            "fixed_id": batch["fixed_id"],
            "moving_id": batch["moving_id"],
            "dvf_path": str(dvf_path),
            "dice_mean": accuracy["dice_mean"],
            "evaluation_elapsed_sec": time.perf_counter() - started,
        }
        for key, value in accuracy.items():
            if str(key).startswith("dice_label_"):
                row[str(key)] = value
        rows.append(row)
        del dense_disp, id_grid, metrics, batch
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def aggregate(rows: Sequence[Mapping[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for config in CONFIGS:
        selected = [row for row in rows if row.get("config") == config.slug]
        values = [finite(row.get("dice_mean")) for row in selected]
        values = [value for value in values if math.isfinite(value)]
        summaries.append(
            {
                "config": config.slug,
                "config_label": config.label,
                "cudnn_benchmark": config.benchmark,
                "cudnn_deterministic": config.deterministic,
                "gpu": str(args.gpu),
                "case_count": len(selected),
                "dice_mean": mean(values) if values else float("nan"),
                "dice_sd": stdev(values) if len(values) > 1 else float("nan"),
            }
        )
    write_csv(Path(args.review_root) / "dice_summary.csv", summaries)
    write_json(Path(args.review_root) / "dice_summary.json", summaries)
    return summaries


def main() -> int:
    args = parse_args()
    if str(args.gpu) != "2":
        raise ValueError("This focused evaluator is fixed to physical GPU 2.")
    if parse_case_ranks(str(args.case_ranks)) != [0, 1, 2, 3, 4]:
        raise ValueError("This focused evaluator requires case ranks 0-4.")
    device = c.set_cuda_device("2", cudnn_benchmark=False, cudnn_deterministic=True)
    rows: List[Dict[str, object]] = []
    for config in CONFIGS:
        rows.extend(evaluate_config(args, config, device))
    write_csv(Path(args.review_root) / "dice_case_metrics.csv", rows)
    aggregate(rows, args)
    print(json.dumps({"completed": True, "case_count": len(rows), "candidate": CANDIDATE_NAME}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
