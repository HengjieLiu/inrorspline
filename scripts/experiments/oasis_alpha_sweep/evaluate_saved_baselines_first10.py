#!/usr/bin/env python
"""Evaluate saved OASIS first10 DL and Greedy baseline displacement fields."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import nibabel as nib
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from aggregate_first10_population import parse_case_ranks  # noqa: E402


DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_saved_baselines"
DEFAULT_DL_ROOT = Path("/database/wavereg/wipmidl_0407/OASIS_v1/round1_MIDL")
DEFAULT_GREEDY_ROOT = Path("/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy")
VARIANT = "oasis_seed3_first10_saved_baselines_v1"


@dataclass(frozen=True)
class BaselineSpec:
    method_key: str
    method_label: str
    family: str
    source_kind: str
    source_root: str
    selector: str


@dataclass(frozen=True)
class Job:
    job_index: int
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    method_key: str
    method_label: str
    family: str
    source_kind: str
    selector: str
    disp_path: str


@dataclass(frozen=True)
class LoadedDisplacement:
    disp_zyx: np.ndarray
    source_disp_raw_shape: str
    source_disp_eval_shape: str
    source_disp_storage_layout: str
    source_disp_component_order: str
    source_disp_reorientation: str


BASELINE_SPECS: List[BaselineSpec] = [
    BaselineSpec(
        "bs1_vxm_v0_2",
        "BS1 VoxelMorph",
        "bs",
        "dl_pretrained",
        "gpu8_OASIS_v1_bs1_vxm_v0_2_run1_20250407144836/inference/inference_bestdice",
        "bestdice",
    ),
    BaselineSpec(
        "bs2_tm_v0",
        "BS2 TransMorph",
        "bs",
        "dl_pretrained",
        "gpu8_OASIS_v1_bs2_tm_v0_run1_20250407145111/inference/inference_bestdice",
        "bestdice",
    ),
    BaselineSpec(
        "bs3_vfa_v0",
        "BS3 VFA",
        "bs",
        "dl_pretrained",
        "gpu8_OASIS_v1_bs3_vfa_v0_run1_20250407145115/inference/inference_bestdice",
        "bestdice",
    ),
    BaselineSpec(
        "bs4_sitreg_v0",
        "BS4 SITReg",
        "bs",
        "dl_pretrained",
        "gpu8_OASIS_v1_bs4_sitreg_v0_run1_20250407165940/inference/inference_bestdice",
        "bestdice",
    ),
    BaselineSpec("greedy", "Greedy", "greedy", "greedy", ".", "canonical"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(spec.method_key for spec in BASELINE_SPECS),
        help="Comma-separated baseline method keys.",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--dl-root", type=Path, default=DEFAULT_DL_ROOT)
    parser.add_argument("--greedy-root", type=Path, default=DEFAULT_GREEDY_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def selected_specs(methods_text: str) -> List[BaselineSpec]:
    requested = [token.strip() for token in str(methods_text).split(",") if token.strip()]
    by_key = {spec.method_key: spec for spec in BASELINE_SPECS}
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"Unknown saved baseline method key(s): {missing}")
    return [by_key[key] for key in requested]


def set_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def disp_root(args: argparse.Namespace, spec: BaselineSpec) -> Path:
    if spec.source_kind == "dl_pretrained":
        return Path(args.dl_root) / spec.source_root
    if spec.source_kind == "greedy":
        return Path(args.greedy_root)
    raise ValueError(f"Unknown source kind: {spec.source_kind}")


def disp_path(args: argparse.Namespace, spec: BaselineSpec, fixed_id: str, moving_id: str) -> Path:
    return disp_root(args, spec) / f"disp_{fixed_id}_{moving_id}.nii.gz"


def build_jobs(args: argparse.Namespace) -> List[Job]:
    specs = selected_specs(str(args.methods))
    jobs: List[Job] = []
    for rank in parse_case_ranks(str(args.case_ranks)):
        case = c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for spec in specs:
            jobs.append(
                Job(
                    job_index=len(jobs),
                    case_rank=int(case.rank),
                    original_case_index=int(case.original_case_index),
                    fixed_id=str(case.fixed_id),
                    moving_id=str(case.moving_id),
                    method_key=spec.method_key,
                    method_label=spec.method_label,
                    family=spec.family,
                    source_kind=spec.source_kind,
                    selector=spec.selector,
                    disp_path=str(disp_path(args, spec, case.fixed_id, case.moving_id)),
                )
            )
    return jobs


def run_name(job: Job) -> str:
    return f"{job.method_key}_rank{job.case_rank:03d}_{job.selector}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / job.method_key / run_name(job)


def shape_text(shape: Sequence[int]) -> str:
    return ",".join(str(int(v)) for v in shape)


def reorient_greedy_component(component: np.ndarray) -> np.ndarray:
    """Apply the scalar OASIS LIA-to-RAS reorientation to one DVF component."""

    return c.reorient_lia_to_ras(np.asarray(component, dtype=np.float32))


def load_saved_disp_zyx(path: Path) -> LoadedDisplacement:
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 3:
        if tuple(arr.shape[1:]) != c.IMAGE_SHAPE:
            raise ValueError(f"Expected displacement spatial shape {c.IMAGE_SHAPE}, got {arr.shape[1:]} from {path}")
        return LoadedDisplacement(
            disp_zyx=arr[None],
            source_disp_raw_shape=shape_text(arr.shape),
            source_disp_eval_shape=shape_text((1, *arr.shape)),
            source_disp_storage_layout="channel_first_zyx_repo_ras",
            source_disp_component_order="zyx_voxel",
            source_disp_reorientation="none",
        )

    if arr.ndim == 5 and arr.shape[3] == 1 and arr.shape[4] == 3:
        arr4 = np.squeeze(arr, axis=3)
        components = [reorient_greedy_component(arr4[..., index]) for index in range(3)]
        disp = np.stack([-components[0], -components[1], components[2]], axis=0).astype(np.float32, copy=False)
        if tuple(disp.shape[1:]) != c.IMAGE_SHAPE:
            raise ValueError(
                f"Expected reoriented Greedy displacement spatial shape {c.IMAGE_SHAPE}, got {disp.shape[1:]} from {path}"
            )
        return LoadedDisplacement(
            disp_zyx=disp[None],
            source_disp_raw_shape=shape_text(arr.shape),
            source_disp_eval_shape=shape_text((1, *disp.shape)),
            source_disp_storage_layout="greedy_5d_channel_last_reoriented_to_repo_ras",
            source_disp_component_order="zyx_voxel_after_component_sign_fix",
            source_disp_reorientation="scalar_oasis_lia_to_ras_each_component_plus_signs_-,-,+",
        )

    raise ValueError(f"Expected saved displacement shape [3,D,H,W] or Greedy [D,H,W,1,3], got {arr.shape} from {path}")


def voxel_zyx_to_norm_dhw3(disp_zyx: np.ndarray) -> np.ndarray:
    if disp_zyx.shape != (1, 3, *c.IMAGE_SHAPE):
        raise ValueError(f"Expected [1,3,{c.IMAGE_SHAPE[0]},{c.IMAGE_SHAPE[1]},{c.IMAGE_SHAPE[2]}], got {disp_zyx.shape}")
    d, h, w = c.IMAGE_SHAPE
    disp = disp_zyx[0]
    out = np.empty((*c.IMAGE_SHAPE, 3), dtype=np.float32)
    out[..., 0] = disp[2] / ((w - 1) / 2.0)
    out[..., 1] = disp[1] / ((h - 1) / 2.0)
    out[..., 2] = disp[0] / ((d - 1) / 2.0)
    return out


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def evaluate_job(args: argparse.Namespace, job: Job, device: torch.device) -> None:
    out_dir = run_dir(args, job)
    done_path = out_dir / "DONE"
    if done_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(out_dir), "job": asdict(job)}), flush=True)
        return
    if not Path(job.disp_path).exists():
        raise FileNotFoundError(f"Missing saved displacement for {job.method_key}: {job.disp_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = c.now_text()
    total_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    batch = c.load_oasis_case(
        rank=int(job.case_rank),
        device=device,
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
    )
    if int(batch["original_case_index"]) != int(job.original_case_index):
        raise RuntimeError(f"{run_name(job)} original_case_index mismatch")
    if str(batch["fixed_id"]) != job.fixed_id or str(batch["moving_id"]) != job.moving_id:
        raise RuntimeError(f"{run_name(job)} fixed/moving mismatch")

    loaded_disp = load_saved_disp_zyx(Path(job.disp_path))
    dense_disp_norm = torch.from_numpy(voxel_zyx_to_norm_dhw3(loaded_disp.disp_zyx)).to(device=device, dtype=torch.float32)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    metrics = c.evaluate_from_dense_coords(id_grid + dense_disp_norm, batch)

    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp_norm, fixed_mask.detach().bool())
    total_elapsed_sec = time.time() - total_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0

    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    case_metrics = c.accuracy_row(metrics, fixed_id, moving_id)
    summary: Dict[str, object] = {
        "run_name": run_name(job),
        "run_dir": str(out_dir),
        "variant": VARIANT,
        "method": job.method_key,
        "method_key": job.method_key,
        "method_label": job.method_label,
        "case_rank": int(job.case_rank),
        "original_case_index": int(job.original_case_index),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": None,
        "eval_interval": None,
        "hidden_dim": None,
        "hidden_layers": None,
        "activation": "saved_dense_displacement",
        "omega": None,
        "lr": None,
        "alpha": None,
        "be_type": "dense_dvf_finite_difference_eval",
        "diffusion_type": "dense_dvf_finite_difference_eval",
        "regularizer": "not_applicable_saved_field",
        "native_analytic_be_formula": "not_available_saved_dense_field",
        "native_analytic_diffusion_formula": "not_available_saved_dense_field",
        "be_training_reduction": "n/a; saved dense displacement evaluated posthoc",
        "image_loss": "posthoc_sinr_ncc",
        "sample_domain": "none_posthoc_full_image",
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": None,
        "bspline_order": None,
        "bspline_mode": None,
        "grid_order": "voxel_zyx_saved_dvf_to_pytorch",
        "seed": None,
        "coords_batch_size": None,
        "train_coords": None,
        "param_count": None,
        "started_at": started_at,
        "completed_at": c.now_text(),
        "training_elapsed_sec": 0.0,
        "training_peak_mem_gb": peak_mem_gb,
        "evaluation_peak_mem_gb": peak_mem_gb,
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": float("nan"),
        "final_data_loss": finite_or_nan(metrics["image_loss"]),
        "final_reg_loss": float("nan"),
        "dense_image_loss": finite_or_nan(metrics["image_loss"]),
        "source_kind": job.source_kind,
        "source_family": job.family,
        "source_selector": job.selector,
        "source_disp_path": job.disp_path,
        "source_disp_component_order": loaded_disp.source_disp_component_order,
        "source_disp_raw_shape": loaded_disp.source_disp_raw_shape,
        "source_disp_shape": loaded_disp.source_disp_eval_shape,
        "source_disp_storage_layout": loaded_disp.source_disp_storage_layout,
        "source_disp_reorientation": loaded_disp.source_disp_reorientation,
        "source_displacement_units": "voxel",
    }
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(case_metrics)
    summary.update(reg_metrics)

    c.write_csv(out_dir / "case_metrics.csv", [case_metrics])
    c.write_csv(out_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(out_dir / "be_metrics.csv", be_rows)
    c.write_csv(out_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_csv(out_dir / "history.csv", [])
    c.write_json(out_dir / "mask_audit.json", batch["mask_audit"])
    c.write_json(out_dir / "summary.json", summary)
    done_path.write_text("done\n")
    print(
        json.dumps(
            {
                "done": str(out_dir),
                "method_key": job.method_key,
                "case_rank": job.case_rank,
                "dice_mean": summary["dice_mean"],
                "hd95_mean": summary["hd95_mean"],
                "total_elapsed_sec": total_elapsed_sec,
            },
            allow_nan=True,
        ),
        flush=True,
    )


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    rows = [asdict(job) for job in jobs]
    args.out_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_json(
        args.out_root / "job_manifest.json",
        {
            "created_at": c.now_text(),
            "variant": VARIANT,
            "job_count": len(rows),
            "case_ranks": parse_case_ranks(str(args.case_ranks)),
            "methods": [spec.method_key for spec in selected_specs(str(args.methods))],
            "out_root": str(args.out_root),
            "review_root": str(args.review_root),
            "dl_root": str(args.dl_root),
            "greedy_root": str(args.greedy_root),
            "oasis_root": str(args.oasis_root),
            "pair_file": str(args.pair_file),
            "jobs": rows,
        },
    )


def preflight(jobs: Sequence[Job]) -> None:
    missing = [job.disp_path for job in jobs if not Path(job.disp_path).exists()]
    if missing:
        raise FileNotFoundError("Missing saved displacement files:\n" + "\n".join(missing[:50]))


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_saved_baseline_overlay.py"),
        "--baseline-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--case-ranks",
        str(args.case_ranks),
    ]
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    import subprocess

    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    preflight(jobs)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "evaluate",
                "job_count": len(jobs),
                "case_ranks": parse_case_ranks(str(args.case_ranks)),
                "methods": [spec.method_key for spec in selected_specs(str(args.methods))],
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"job": asdict(job), "run_dir": str(run_dir(args, job))}), flush=True)
        return

    device = set_device(args)
    failures: List[Dict[str, object]] = []
    started = time.time()
    for job in jobs:
        try:
            evaluate_job(args, job, device)
        except Exception as exc:
            failures.append({"job": asdict(job), "error": repr(exc)})
            raise
    c.write_json(
        args.out_root / "evaluation_summary.json",
        {
            "completed_at": c.now_text(),
            "job_count": len(jobs),
            "failures": failures,
            "elapsed_sec": time.time() - started,
        },
    )
    if not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
