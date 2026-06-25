#!/usr/bin/env python
"""Run one repo-local OASIS alpha-sweep job."""

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


METHODS = [
    "idir_original",
    "sinr_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--autograd-be-chunk-size", type=int, default=20000)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-original-case-index", type=int, default=None)
    parser.add_argument("--expected-fixed-id", type=str, default=None)
    parser.add_argument("--expected-moving-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-case-only", action="store_true")
    return parser.parse_args()


def method_key(args: argparse.Namespace) -> str:
    if args.method == "idir_original":
        return "idir_original_dense_siren_be_autograd"
    return str(args.method)


def method_cps(method: str) -> Optional[int]:
    if method == "sinr_cps2_be_bspline_analytic":
        return 2
    if method in {"sinr_cps4_be_bspline_analytic", "direct_cp_cps4_be_bspline_analytic"}:
        return 4
    return None


def default_lr(method: str) -> float:
    if method == "direct_cp_cps4_be_bspline_analytic":
        return 1e-3
    return 1e-4


def build_run_name(args: argparse.Namespace, lr: float) -> str:
    key = method_key(args)
    if args.run_name:
        return args.run_name
    return (
        f"{key}_rank{args.case_rank:03d}_a{c.format_float(args.alpha)}"
        f"_e{args.epochs}_lr{c.format_float(lr)}"
    )


def validate_expected_case(args: argparse.Namespace, *, original_index: int, fixed_id: str, moving_id: str) -> None:
    errors: List[str] = []
    if args.expected_original_case_index is not None and int(args.expected_original_case_index) != int(original_index):
        errors.append(
            f"original_case_index {original_index} != expected {int(args.expected_original_case_index)}"
        )
    if args.expected_fixed_id is not None and str(args.expected_fixed_id) != str(fixed_id):
        errors.append(f"fixed_id {fixed_id} != expected {args.expected_fixed_id}")
    if args.expected_moving_id is not None and str(args.expected_moving_id) != str(moving_id):
        errors.append(f"moving_id {moving_id} != expected {args.expected_moving_id}")
    if errors:
        raise RuntimeError("Case validation failed: " + "; ".join(errors))


def idir_gradient(input_coords: torch.Tensor, output: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        [input_coords],
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def idir_jacobian_matrix(input_coords: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    jac = torch.zeros(input_coords.shape[0], 3, 3, dtype=output.dtype, device=output.device)
    for i in range(3):
        jac[:, i, :] = idir_gradient(input_coords, output[:, i], create_graph=True)
    return jac


def idir_bending_energy(input_coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int) -> torch.Tensor:
    jac = idir_jacobian_matrix(input_coords, output_rel)
    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3, dtype=output_rel.dtype, device=output_rel.device)
    dy_xyz = torch.zeros_like(dx_xyz)
    dz_xyz = torch.zeros_like(dx_xyz)
    for i in range(3):
        dx_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 0], create_graph=True)
        dy_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 1], create_graph=True)
        dz_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 2], create_graph=True)

    dx_xyz = dx_xyz.square()
    dy_xyz = dy_xyz.square()
    dz_xyz = dz_xyz.square()
    loss = torch.mean(dx_xyz[:, :, 0]) + torch.mean(dy_xyz[:, :, 1]) + torch.mean(dz_xyz[:, :, 2])
    loss = loss + 2 * torch.mean(dx_xyz[:, :, 1]) + 2 * torch.mean(dx_xyz[:, :, 2]) + torch.mean(dy_xyz[:, :, 2])
    return loss / float(batch_size)


def make_idir_training_coords(
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    selected = coords_all[fixed_mask.detach().reshape(-1).bool()]
    if selected.numel() == 0:
        raise RuntimeError("OASIS fixed image-positive mask is empty.")
    return selected.contiguous()


def evaluate_dense_model(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
) -> Dict[str, object]:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
    dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
    return c.evaluate_from_dense_coords(dense_coords, batch)


def train_idir(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    coords_train = make_idir_training_coords(batch, device)
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach().requires_grad_(True)
        disp = model(coords)
        moved_coords = coords + disp
        warped = c.sample_volume(moving, moved_coords, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = idir_bending_energy(coords, disp, batch_size=batch_size)
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    metrics = evaluate_dense_model(model, batch, device, args.eval_chunk_size)
    metrics["history"] = history
    metrics["model"] = model
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = c.count_trainable_params(model)
    return metrics


def build_spline_generator(
    args: argparse.Namespace,
    control_shape: Tuple[int, int, int],
) -> torch.nn.Module:
    if args.method == "direct_cp_cps4_be_bspline_analytic":
        return c.DirectCPGenerator(control_shape)
    return c.NetworkGenerator(c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega))


def train_spline(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
) -> Dict[str, object]:
    cps = method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, cps, device)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    generator = build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    history: List[Dict[str, float]] = []

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        output = generator(control_coords, control_shape)
        control_disp = output.reshape(*control_shape, 3)
        dense_disp = c.expand_bspline_controls(output, control_shape, c.IMAGE_SHAPE, cps)
        warped = c.warp_full(moving, id_dense + dense_disp, mode="bilinear")
        data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
        reg_loss = c.analytic_bspline_sampled_be(
            control_disp,
            c.IMAGE_SHAPE,
            cps,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(final_output, control_shape, c.IMAGE_SHAPE, cps).detach()
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["generator"] = generator
    metrics["control_shape"] = control_shape
    metrics["control_coords"] = control_coords.detach()
    metrics["control_disp"] = final_control_disp
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = c.count_trainable_params(generator)
    metrics["cps"] = cps
    return metrics


def write_outputs(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    lr: float,
    total_elapsed_sec: float,
) -> None:
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_coords = metrics["dense_coords"]
    assert isinstance(dense_coords, torch.Tensor)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_coords.device)
    dense_disp_norm = (dense_coords - id_grid).detach()
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    control_disp = metrics.get("control_disp")
    if isinstance(control_disp, torch.Tensor):
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(
            dense_disp_norm,
            fixed_mask.detach().bool(),
            control_disp_norm=control_disp,
            cps=int(metrics["cps"]),
        )
    else:
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp_norm, fixed_mask.detach().bool())
        model = metrics.get("model")
        if isinstance(model, torch.nn.Module):
            idir_be_rows = c.idir_autograd_be_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            idir_diffusion_rows = c.idir_autograd_diffusion_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            be_rows.extend(idir_be_rows)
            diffusion_rows.extend(idir_diffusion_rows)
            reg_metrics.update(c.idir_autograd_be_rows_to_columns(idir_be_rows))
            reg_metrics.update(c.idir_autograd_diffusion_rows_to_columns(idir_diffusion_rows))
            reg_metrics["idir_autograd_be_chunk_size"] = int(args.autograd_be_chunk_size)
            reg_metrics["idir_autograd_diffusion_chunk_size"] = int(args.autograd_be_chunk_size)
            c.add_native_analytic_be_columns(reg_metrics)
            c.add_native_analytic_diffusion_columns(reg_metrics)

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
    if "control_disp" in metrics:
        torch.save(
            {
                "control_disp": metrics["control_disp"].detach().cpu(),  # type: ignore[union-attr]
                "control_coords": metrics["control_coords"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": metrics["control_shape"],
                "cps": metrics["cps"],
                "image_shape": c.IMAGE_SHAPE,
                "be_type": "be_bspline_analytic",
                "image_loss": "sinr_ncc",
                "method_key": method_key(args),
                "alpha": float(args.alpha),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )

    c.write_csv(run_dir / "history.csv", metrics["history"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "case_metrics.csv", [c.accuracy_row(metrics, fixed_id, moving_id)])
    c.write_csv(run_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(run_dir / "be_metrics.csv", be_rows)
    c.write_csv(run_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])

    last = metrics["history"][-1]  # type: ignore[index]
    is_idir = args.method == "idir_original"
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": "oasis_seed3_alpha_prerun_repo_local",
        "method": args.method,
        "method_key": method_key(args),
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
        "activation": "Siren" if args.method != "direct_cp_cps4_be_bspline_analytic" else "direct_control_points",
        "omega": float(args.idir_omega if is_idir else args.spline_omega),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "be_type": "be_native_analytic" if is_idir else "be_bspline_analytic",
        "diffusion_type": "diffusion_native_analytic" if is_idir else "diffusion_bspline_analytic",
        "regularizer": "be_native_analytic" if is_idir else "be_bspline_analytic",
        "native_analytic_be_formula": "idir_coordinate_autograd_hessian" if is_idir else "bspline_analytic",
        "native_analytic_diffusion_formula": "idir_coordinate_autograd_jacobian" if is_idir else "bspline_analytic",
        "be_training_reduction": (
            "original IDIR coordinate-autograd analytic BE; sampled spatial means then divided by batch size"
            if is_idir
            else "analytic cubic B-spline BE sampled at voxel centers; spatial mean plus component mean"
        ),
        "image_loss": "idir_global_ncc_sampled" if is_idir else "sinr_ncc",
        "sample_domain": "fixed_image_positive" if is_idir else "none_full_image_ncc",
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": metrics.get("cps"),
        "bspline_order": 3 if not is_idir else None,
        "bspline_mode": "disp" if not is_idir else None,
        "grid_order": "pytorch",
        "seed": int(args.seed),
        "coords_batch_size": metrics.get("coords_batch_size"),
        "train_coords": metrics.get("train_coords"),
        "param_count": metrics.get("param_count"),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": last["loss"],
        "final_data_loss": last["data_loss"],
        "final_reg_loss": last["reg_loss"],
        "dice_mean": metrics["dice_mean"],
        "hd95_mean": metrics["hd95_mean"],
        "asd_backend": metrics["asd_backend"],
        "asd_warped_to_fixed_mean": metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": metrics["asd_symmetric_mean"],
    }
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(c.accuracy_row(metrics, fixed_id, moving_id))
    summary.update(reg_metrics)
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)


def main() -> None:
    args = parse_args()
    lr = float(args.lr if args.lr is not None else default_lr(args.method))
    if args.validate_case_only:
        case = c.resolve_case(rank=args.case_rank, pair_file=args.pair_file, oasis_root=args.oasis_root)
        print(json.dumps(case.__dict__, indent=2), flush=True)
        validate_expected_case(
            args,
            original_index=case.original_case_index,
            fixed_id=case.fixed_id,
            moving_id=case.moving_id,
        )
        return

    run_name = build_run_name(args, lr)
    key = method_key(args)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "method": args.method,
                "method_key": key,
                "gpu": args.gpu,
                "case_rank": batch["case_rank"],
                "original_case_index": batch["original_case_index"],
                "fixed_id": batch["fixed_id"],
                "moving_id": batch["moving_id"],
                "mask_audit": batch["mask_audit"],
                "alpha": args.alpha,
                "lr": lr,
                "epochs": args.epochs,
            },
            allow_nan=True,
        ),
        flush=True,
    )

    if args.method == "idir_original":
        metrics = train_idir(args, batch, device, lr)
    else:
        metrics = train_spline(args, batch, device, lr)
    metrics["started_at"] = started_at
    total_elapsed_sec = time.time() - total_start
    write_outputs(args, batch, metrics, run_dir, run_name, lr, total_elapsed_sec)


if __name__ == "__main__":
    main()
