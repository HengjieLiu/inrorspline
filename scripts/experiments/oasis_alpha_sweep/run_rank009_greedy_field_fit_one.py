#!/usr/bin/env python
"""Fit one OASIS model directly to the canonical Greedy dense displacement."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_one as image_runner  # noqa: E402
from evaluate_saved_baselines_first10 import (  # noqa: E402
    DEFAULT_GREEDY_ROOT,
    load_saved_disp_zyx,
    voxel_zyx_to_norm_dhw3,
)


METHODS = [
    "idir_original",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
]

VARIANT = "oasis_rank009_fit_modes_lr_sweep_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Recorded operating-point alpha; not used in MSE training.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="0")
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
    parser.add_argument("--greedy-root", type=Path, default=DEFAULT_GREEDY_ROOT)
    parser.add_argument("--expected-original-case-index", type=int, default=88)
    parser.add_argument("--expected-fixed-id", type=str, default="0173")
    parser.add_argument("--expected-moving-id", type=str, default="0136")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-target-only", action="store_true")
    return parser.parse_args()


def field_method_key(method: str) -> str:
    if method == "idir_original":
        return "idir_original_dense_siren_field_mse"
    if method == "sinr_cps4_be_bspline_analytic":
        return "sinr_cps4_field_mse"
    if method == "direct_cp_cps4_be_bspline_analytic":
        return "direct_cp_cps4_field_mse"
    raise ValueError(f"Unsupported field-fit method: {method}")


def field_method_label(method: str) -> str:
    if method == "idir_original":
        return "IDIR field MSE"
    if method == "sinr_cps4_be_bspline_analytic":
        return "SINR cps4 field MSE"
    if method == "direct_cp_cps4_be_bspline_analytic":
        return "DirectCP cps4 field MSE"
    raise ValueError(f"Unsupported field-fit method: {method}")


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"{field_method_key(args.method)}_rank{args.case_rank:03d}"
        f"_a{c.format_float(args.alpha)}_e{args.epochs}_lr{c.format_float(args.lr)}"
    )


def validate_expected_case(args: argparse.Namespace, *, original_index: int, fixed_id: str, moving_id: str) -> None:
    image_runner.validate_expected_case(
        args,
        original_index=original_index,
        fixed_id=fixed_id,
        moving_id=moving_id,
    )


def greedy_disp_path(args: argparse.Namespace, fixed_id: str, moving_id: str) -> Path:
    return Path(args.greedy_root) / f"disp_{fixed_id}_{moving_id}.nii.gz"


def shape_text(shape: Tuple[int, ...]) -> str:
    return ",".join(str(int(v)) for v in shape)


def load_greedy_target(
    args: argparse.Namespace,
    *,
    fixed_id: str,
    moving_id: str,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    path = greedy_disp_path(args, fixed_id, moving_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical Greedy displacement: {path}")
    loaded = load_saved_disp_zyx(path)
    target_np = voxel_zyx_to_norm_dhw3(loaded.disp_zyx)
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    audit: Dict[str, object] = {
        "source_kind": "greedy",
        "source_family": "greedy",
        "source_selector": "canonical",
        "source_disp_path": str(path),
        "source_disp_component_order": loaded.source_disp_component_order,
        "source_disp_raw_shape": loaded.source_disp_raw_shape,
        "source_disp_shape": loaded.source_disp_eval_shape,
        "source_disp_storage_layout": loaded.source_disp_storage_layout,
        "source_disp_reorientation": loaded.source_disp_reorientation,
        "source_displacement_units": "voxel",
        "field_fit_target_units": "normalized_grid",
        "field_fit_target_shape": shape_text(tuple(target.shape)),
        "field_fit_target_extent": "full_dense_field_available",
    }
    return target, audit


def field_domain_audit(fixed_mask: torch.Tensor) -> Dict[str, object]:
    mask = fixed_mask.detach().bool()
    voxels = int(mask.sum().detach().cpu())
    return {
        "field_fit_domain": "fixed_image_positive",
        "field_fit_mask_definition": "fixed raw image > 0 after standard OASIS reorientation",
        "field_fit_domain_voxels": voxels,
        "field_fit_domain_fraction": float(voxels / np.prod(c.IMAGE_SHAPE)),
    }


def field_mse_metrics(
    pred_disp: torch.Tensor,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> Dict[str, float]:
    diff = pred_disp.detach() - target_disp.detach()
    mse_norm = float(diff.square().mean().detach().cpu())
    rmse_norm = math.sqrt(mse_norm)
    diff_vox = c.norm_grid_dhw3_to_voxel_dhw3(diff, c.IMAGE_SHAPE)
    mse_vox = float(diff_vox.square().mean().detach().cpu())
    rmse_vox = math.sqrt(mse_vox)
    mask = fixed_mask.detach().reshape(-1).bool()
    diff_masked = diff.reshape(-1, 3)[mask]
    diff_vox_masked = diff_vox.reshape(-1, 3)[mask]
    mse_masked_norm = float(diff_masked.square().mean().detach().cpu())
    rmse_masked_norm = math.sqrt(mse_masked_norm)
    mse_masked_vox = float(diff_vox_masked.square().mean().detach().cpu())
    rmse_masked_vox = math.sqrt(mse_masked_vox)
    return {
        "field_mse_full_normalized_grid": mse_norm,
        "field_rmse_full_normalized_grid": rmse_norm,
        "field_mse_full_voxel": mse_vox,
        "field_rmse_full_voxel": rmse_vox,
        "field_mse_masked_normalized_grid": mse_masked_norm,
        "field_rmse_masked_normalized_grid": rmse_masked_norm,
        "field_mse_masked_voxel": mse_masked_vox,
        "field_rmse_masked_voxel": rmse_masked_vox,
        "field_mse_objective_normalized_grid": mse_masked_norm,
        "field_rmse_objective_normalized_grid": rmse_masked_norm,
        "field_mse_objective_voxel": mse_masked_vox,
        "field_rmse_objective_voxel": rmse_masked_vox,
    }


def evaluate_idir_dense_disp(
    model: torch.nn.Module,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(model(coords))
    return torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3).contiguous()


def train_idir_field(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device).detach()
    mask_flat = fixed_mask.detach().reshape(-1).bool()
    coords_train = coords_all[mask_flat]
    target_train = target_disp.reshape(-1, 3).detach()[mask_flat]
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach()
        target = target_train[indices]
        pred = model(coords)
        loss = F.mse_loss(pred, target, reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        entry = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_sampled_normalized_grid": value,
            "coords_batch_size": float(batch_size),
            "train_coords": float(n_train),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    dense_disp = evaluate_idir_dense_disp(model, device, int(args.eval_chunk_size))
    return {
        "history": history,
        "model": model,
        "dense_disp": dense_disp,
        "training_elapsed_sec": time.time() - train_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "param_count": c.count_trainable_params(model),
        "coords_batch_size": batch_size,
        "train_coords": n_train,
        "field_chunk_size": None,
        "field_chunks": None,
    }


def train_spline_field(
    args: argparse.Namespace,
    target_disp: torch.Tensor,
    fixed_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    cps = image_runner.method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, cps, device)
    generator = image_runner.build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(args.lr))
    mask_flat = fixed_mask.detach().reshape(-1).bool()
    target_flat = target_disp.reshape(-1, 3).detach()
    n_train = int(mask_flat.sum().detach().cpu())
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        output = generator(control_coords, control_shape)
        dense_disp = c.expand_bspline_controls(output, control_shape, c.IMAGE_SHAPE, cps)
        loss = F.mse_loss(dense_disp.reshape(-1, 3)[mask_flat], target_flat[mask_flat], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        entry = {
            "epoch": float(epoch),
            "loss": value,
            "data_loss": value,
            "reg_loss": 0.0,
            "field_mse_objective_normalized_grid": value,
            "field_mse_masked_normalized_grid": value,
            "train_coords": float(n_train),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()

    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(final_output, control_shape, c.IMAGE_SHAPE, cps).detach()
    return {
        "history": history,
        "generator": generator,
        "control_shape": control_shape,
        "control_coords": control_coords.detach(),
        "control_disp": final_control_disp,
        "dense_disp": final_dense_disp,
        "training_elapsed_sec": time.time() - train_start,
        "training_peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "param_count": c.count_trainable_params(generator),
        "cps": cps,
        "coords_batch_size": None,
        "train_coords": n_train,
    }


def write_outputs(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    target_audit: Dict[str, object],
    target_disp: torch.Tensor,
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    total_elapsed_sec: float,
) -> None:
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_disp = metrics["dense_disp"]
    assert isinstance(dense_disp, torch.Tensor)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_disp.device)
    dense_coords = id_grid + dense_disp
    eval_metrics = c.evaluate_from_dense_coords(dense_coords, batch)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    field_metrics = field_mse_metrics(dense_disp, target_disp, fixed_mask)

    control_disp = metrics.get("control_disp")
    if isinstance(control_disp, torch.Tensor):
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(
            dense_disp,
            fixed_mask.detach().bool(),
            control_disp_norm=control_disp,
            cps=int(metrics["cps"]),
        )
    else:
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp, fixed_mask.detach().bool())

    if args.save_final_dense:
        c.save_dense_disp(run_dir / f"disp_norm_{fixed_id}_{moving_id}.npy", dense_coords)

    state: Dict[str, object] = {
        "args": vars(args),
        "run_name": run_name,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "training_elapsed_sec": metrics["training_elapsed_sec"],
        "training_peak_mem_gb": metrics["training_peak_mem_gb"],
    }
    if "model" in metrics:
        state["model_state_dict"] = metrics["model"].state_dict()  # type: ignore[union-attr]
    if "generator" in metrics:
        state["generator_state_dict"] = metrics["generator"].state_dict()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    torch.save(state, run_dir / "model_state.pt")

    objective_name = str(getattr(args, "field_fit_objective", "greedy_field_fit_mse"))

    if isinstance(control_disp, torch.Tensor):
        torch.save(
            {
                "control_disp": control_disp.detach().cpu(),
                "control_coords": metrics["control_coords"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": metrics["control_shape"],
                "cps": metrics["cps"],
                "image_shape": c.IMAGE_SHAPE,
                "objective": objective_name,
                "field_fit_target_units": "normalized_grid",
                "method_key": field_method_key(args.method),
                "alpha_recorded_not_used": float(args.alpha),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )

    c.write_csv(run_dir / "history.csv", metrics["history"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "case_metrics.csv", [c.accuracy_row(eval_metrics, fixed_id, moving_id)])
    c.write_csv(run_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(run_dir / "be_metrics.csv", be_rows)
    c.write_csv(run_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])
    c.write_json(run_dir / "target_audit.json", target_audit)

    last = metrics["history"][-1]  # type: ignore[index]
    is_idir = args.method == "idir_original"
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": str(args.variant),
        "objective": objective_name,
        "method": args.method,
        "method_key": field_method_key(args.method),
        "method_label": field_method_label(args.method),
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": int(args.epochs),
        "eval_interval": int(args.eval_interval),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "activation": "direct_control_points" if args.method.startswith("direct_cp_") else "Siren",
        "omega": float(args.idir_omega if is_idir else args.spline_omega),
        "lr": float(args.lr),
        "alpha": float(args.alpha),
        "alpha_role": "recorded_operating_point_not_used_by_mse_objective",
        "be_type": "dense_dvf_finite_difference_eval" if is_idir else "be_bspline_analytic_posthoc",
        "diffusion_type": "dense_dvf_finite_difference_eval" if is_idir else "diffusion_bspline_analytic_posthoc",
        "regularizer": "none_mse_only",
        "native_analytic_be_formula": "not_computed_for_field_fit_idir" if is_idir else "bspline_analytic_posthoc",
        "native_analytic_diffusion_formula": "not_computed_for_field_fit_idir" if is_idir else "bspline_analytic_posthoc",
        "be_training_reduction": "not_applicable_mse_only",
        "image_loss": "none_training_posthoc_sinr_ncc",
        "sample_domain": "fixed_image_positive",
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": metrics.get("cps"),
        "bspline_order": 3 if not is_idir else None,
        "bspline_mode": "disp" if not is_idir else None,
        "grid_order": "pytorch",
        "seed": int(args.seed),
        "coords_batch_size": metrics.get("coords_batch_size"),
        "train_coords": metrics.get("train_coords"),
        "field_chunk_size": metrics.get("field_chunk_size"),
        "field_chunks": metrics.get("field_chunks"),
        "param_count": metrics.get("param_count"),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": last["loss"],
        "final_data_loss": last["data_loss"],
        "final_reg_loss": last["reg_loss"],
        "dense_image_loss": eval_metrics["image_loss"],
        "dice_mean": eval_metrics["dice_mean"],
        "hd95_mean": eval_metrics["hd95_mean"],
        "asd_backend": eval_metrics["asd_backend"],
        "asd_warped_to_fixed_mean": eval_metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": eval_metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": eval_metrics["asd_symmetric_mean"],
    }
    summary.update(target_audit)
    summary.update(field_metrics)
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(c.accuracy_row(eval_metrics, fixed_id, moving_id))
    summary.update(reg_metrics)
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args)
    key = field_method_key(args.method)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite:
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
    validate_expected_case(
        args,
        original_index=int(batch["original_case_index"]),
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
    )
    target_disp, target_audit = load_greedy_target(
        args,
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
        device=device,
    )
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    target_audit.update(field_domain_audit(fixed_mask))
    if args.validate_target_only:
        print(json.dumps({"case": {k: str(batch[k]) for k in ["case_rank", "original_case_index", "fixed_id", "moving_id"]}, "target_audit": target_audit}, indent=2), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "objective": "greedy_field_fit_mse",
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
        metrics = train_idir_field(args, target_disp, fixed_mask, device)
    else:
        metrics = train_spline_field(args, target_disp, fixed_mask, device)
    metrics["started_at"] = started_at
    total_elapsed_sec = time.time() - total_start
    write_outputs(args, batch, target_audit, target_disp, metrics, run_dir, run_name, total_elapsed_sec)


if __name__ == "__main__":
    main()
