#!/usr/bin/env python3
"""Run one N0/N1/P0/P1 controlled DUAL-INR DIR-Lab record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from scripts.experiments.lungct_dual_inr_author.external_api import (
    git_directory,
    load_external_api,
    repository_root,
)
from scripts.experiments.lungct_dual_inr_controlled.adapter import (
    LOCKED_STAGE_LENGTHS,
    VARIANTS,
    build_controlled_config,
    resolve_device,
    seed_everything,
    train_controlled,
    variant_loss_branch,
    variant_protocol,
    variant_schedule,
)
from scripts.experiments.lungct_dual_inr_controlled.data import (
    ControlledCase,
    evaluate_controlled_metrics,
    load_controlled_case,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (dict, list, tuple)):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            f"--git-dir={git_directory(path)}",
            f"--work-tree={path}",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def git_head(path: Path) -> str | None:
    result = _run_git(path.resolve(), "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else None


def git_status(path: Path) -> str:
    result = _run_git(path.resolve(), "status", "--porcelain")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git status failed below {path}")
    return result.stdout.strip()


def source_bundle() -> dict[str, Any]:
    root = repository_root().resolve()
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("adapter.py").resolve(),
        Path(__file__).with_name("aggregate_lungmask_revision.py").resolve(),
        Path(__file__).with_name("data.py").resolve(),
        Path(__file__).with_name("launch_all.py").resolve(),
        Path(__file__).with_name("launch_lungmask_revision.py").resolve(),
    ]
    bundle = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.name):
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(content).hexdigest()
        bundle.update(relative.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(str(len(content)).encode("ascii"))
        bundle.update(b"\0")
        bundle.update(content)
        bundle.update(b"\0")
        records.append(
            {
                "path": str(path),
                "repository_relative_path": relative,
                "size_bytes": len(content),
                "sha256": digest,
            }
        )
    return {"algorithm": "sha256", "bundle_sha256": bundle.hexdigest(), "sources": records}


def validate_case_id(value: str) -> int:
    case_id = int(value)
    if case_id not in range(1, 11):
        raise argparse.ArgumentTypeError("case id must be in 1..10")
    return case_id


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_output_root(output_root: Path, data_roots: Sequence[Path]) -> Path:
    if not output_root.is_absolute():
        raise ValueError("--output-root must be absolute")
    output_root = output_root.resolve()
    if _is_below(output_root, repository_root()):
        raise ValueError("large case outputs must remain outside the Git repository")
    for data_root in data_roots:
        if _is_below(output_root, data_root):
            raise ValueError(f"output root must remain outside data root {data_root}")
    return output_root


def new_output_dir(
    output_root: Path,
    variant: str,
    case_id: int,
    *,
    run_tag: str | None = None,
) -> Path:
    output_dir = (
        output_root / str(run_tag) / variant.lower() / f"case_{case_id:02d}"
        if run_tag
        else output_root / variant.lower() / f"case_{case_id:02d}"
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to reuse or overwrite case output {output_dir}"
        ) from exc
    return output_dir


def decode_dense_field(
    *,
    model: torch.nn.Module,
    shape_xyz: Sequence[int],
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode on the full XYZ grid without materializing the full GPU grid."""

    shape = tuple(int(value) for value in shape_xyz)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(f"invalid decode shape {shape}")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    n_voxels = int(np.prod(shape, dtype=np.int64))
    norm_flat = np.empty((n_voxels, 3), dtype=np.float32)
    yz = int(shape[1] * shape[2])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n_voxels, int(chunk_size)):
            stop = min(start + int(chunk_size), n_voxels)
            index = torch.arange(start, stop, device=device, dtype=torch.int64)
            xx = torch.div(index, yz, rounding_mode="floor")
            remainder = index - xx * yz
            yy = torch.div(remainder, shape[2], rounding_mode="floor")
            zz = remainder - yy * shape[2]
            coords = torch.stack(
                (
                    xx.to(torch.float32) / float(shape[0] - 1) * 2.0 - 1.0,
                    yy.to(torch.float32) / float(shape[1] - 1) * 2.0 - 1.0,
                    zz.to(torch.float32) / float(shape[2] - 1) * 2.0 - 1.0,
                ),
                dim=1,
            )
            prediction = model(coords)
            if isinstance(prediction, tuple):
                prediction = prediction[0] + prediction[1]
            norm_flat[start:stop] = prediction.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
    if was_training:
        model.train()
    dense_norm_xyz = norm_flat.reshape(shape + (3,))
    scale = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    dense_vox_xyz = dense_norm_xyz * scale.reshape((1, 1, 1, 3))
    if not np.isfinite(dense_vox_xyz).all():
        raise FloatingPointError("decoded dense displacement contains non-finite values")
    return dense_norm_xyz, dense_vox_xyz.astype(np.float32, copy=False)


def _case_metadata(case: ControlledCase) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": case.protocol,
        "case_id": int(case.case_id),
        "shape_xyz": list(case.shape_xyz),
        "spacing_xyz_mm": [float(value) for value in case.spacing_xyz],
        "sampling_mask_voxels": int(np.count_nonzero(case.sampling_mask_xyz)),
        "metric_mask_voxels": int(np.count_nonzero(case.metric_mask_xyz)),
        "source_paths": case.source_paths,
        "data_provenance": case.provenance,
    }
    return payload


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    case_started = time.perf_counter()
    variant = str(args.variant).upper()
    protocol = variant_protocol(variant)
    loss_branch = variant_loss_branch(variant)
    native_raw_root = args.native_raw_root.resolve()
    native_mask_root = args.native_mask_root.resolve()
    ptv_root = args.ptv_root.resolve()
    for label, path in (
        ("native raw", native_raw_root),
        ("native mask", native_mask_root),
        ("pTV", ptv_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} root does not exist: {path}")
    output_root = validate_output_root(
        args.output_root,
        (native_raw_root, native_mask_root, ptv_root),
    )
    api = load_external_api()
    external_status = git_status(api.repository)
    if external_status:
        raise RuntimeError(
            "pinned external DUAL-INR-DIR checkout must be clean:\n" + external_status
        )
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        cuda_device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        cuda_device_index = None
    seed_everything(int(args.seed))
    run_mode = "short_smoke" if args.smoke else "full_length"
    lengths = (1, 1, 1, 1) if args.smoke else LOCKED_STAGE_LENGTHS
    config = build_controlled_config(
        variant,
        stage_lengths=lengths,
        regularization_multiplier=float(args.regularization_multiplier),
        ptv_isotv_weight=args.ptv_isotv_weight,
    )
    config["controlled_adapter"]["run_mode"] = run_mode
    config["controlled_adapter"]["seed"] = int(args.seed)
    config["controlled_adapter"]["run_tag"] = args.run_tag or None

    case = load_controlled_case(
        protocol=protocol,
        case_id=int(args.case_id),
        native_raw_root=native_raw_root,
        native_mask_root=native_mask_root,
        ptv_root=ptv_root,
    )
    output_dir = new_output_dir(
        output_root,
        variant,
        int(args.case_id),
        run_tag=args.run_tag or None,
    )
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stderr_path.touch()
    stdout_path.write_text(
        f"START {utc_now()} variant={variant} case={args.case_id:02d} "
        f"protocol={protocol} loss_branch={loss_branch} "
        f"seed={args.seed} device={device}\n",
        encoding="utf-8",
    )
    partial_policy = {
        "on_failure": "preserve partial directory, write failed RUN_STATE, omit DONE",
        "retry": "launcher archives partial directory before a fresh attempt",
        "overwrite_complete": False,
    }
    write_json(
        output_dir / "RUN_STATE.json",
        {
            "status": "running",
            "variant": variant,
            "protocol": protocol,
            "loss_branch_identifier": loss_branch,
            "run_tag": args.run_tag or None,
            "case_id": int(args.case_id),
            "seed": int(args.seed),
            "run_mode": run_mode,
            "started_utc": utc_now(),
            "partial_output_policy": partial_policy,
        },
    )

    try:
        bundle = source_bundle()
        config["paths"] = {
            "native_raw_root": native_raw_root,
            "native_mask_root": native_mask_root,
            "ptv_root": ptv_root,
            "output_dir": output_dir,
        }
        config["data_contract"] = _case_metadata(case)
        config["source_bundle_sha256"] = bundle["bundle_sha256"]
        write_json(output_dir / "resolved_config.json", config)
        api.utils.save_config(output_dir / "resolved_config.yaml", config)
        write_json(
            output_dir / "provenance.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "main_repository_commit": git_head(repository_root()),
                "main_repository_dirty": bool(git_status(repository_root())),
                "external_repository": str(api.repository),
                "external_commit": api.commit,
                "external_git_status_porcelain": external_status,
                "source_bundle": bundle,
                "container_image_env": os.environ.get("CODEX_CONTAINER_IMAGE"),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_version": torch.version.cuda,
                "device_requested": str(args.device),
                "device_resolved": str(device),
                "variant": variant,
                "protocol": protocol,
                "loss_branch_identifier": loss_branch,
                "run_tag": args.run_tag or None,
                "case_id": int(args.case_id),
                "seed": int(args.seed),
                "run_mode": run_mode,
                "data": _case_metadata(case),
                "scientific_stop_rule": None,
                "partial_output_policy": partial_policy,
            },
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
            torch.cuda.synchronize(cuda_device_index)
        training_started = time.perf_counter()
        result = train_controlled(
            fixed_image_xyz=case.fixed,
            moving_image_xyz=case.moving,
            sampling_mask_xyz=case.sampling_mask_xyz,
            config=config,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(cuda_device_index)
            training_peak_allocated = int(
                torch.cuda.max_memory_allocated(cuda_device_index)
            )
            training_peak_reserved = int(
                torch.cuda.max_memory_reserved(cuda_device_index)
            )
        else:
            training_peak_allocated = None
            training_peak_reserved = None
        training_wall_seconds = time.perf_counter() - training_started
        history = result["history"]
        write_csv(output_dir / "history.csv", history)
        write_csv(output_dir / "stage_trace.csv", result["stages"])
        write_json(
            output_dir / "stage_trace.json",
            {
                "stages": result["stages"],
                "fine_activation_step": result["fine_activation_step"],
            },
        )
        model = result["model"]
        torch.save(model.state_dict(), output_dir / "model_state.pt")

        dense_norm_xyz, dense_vox_xyz = decode_dense_field(
            model=model,
            shape_xyz=case.shape_xyz,
            device=device,
            chunk_size=int(args.eval_chunk_size),
        )
        np.save(output_dir / "dense_dvf_xyz_voxel.npy", dense_vox_xyz)
        np.save(
            output_dir / "fixed_metric_mask_xyz.npy",
            case.metric_mask_xyz.astype(np.uint8, copy=False),
        )
        metrics = evaluate_controlled_metrics(
            case,
            dense_vox_xyz,
            model=model,
            device=device,
        )
        headline_key = str(metrics["headline_tre_metric"])
        headline_payload = metrics.get(headline_key)
        if not isinstance(headline_payload, dict):
            raise RuntimeError(
                f"headline metric {headline_key!r} is missing from evaluation payload"
            )
        metrics["headline_tre"] = headline_payload
        write_json(output_dir / "tre_scores.json", metrics)

        if device.type == "cuda":
            torch.cuda.synchronize(cuda_device_index)
            total_peak_allocated = int(
                torch.cuda.max_memory_allocated(cuda_device_index)
            )
            total_peak_reserved = int(
                torch.cuda.max_memory_reserved(cuda_device_index)
            )
        else:
            total_peak_allocated = None
            total_peak_reserved = None
        total_wall_seconds = time.perf_counter() - case_started
        stage_starts = [
            int(stage["start_step"])
            for stage in result["stages"]
            if "start_step" in stage
        ]
        stage_lengths = [
            int(stage.get("length", int(stage.get("end_step_exclusive", 0)) - int(stage["start_step"])))
            for stage in result["stages"]
            if "start_step" in stage
        ]
        image_factors = [
            int(stage.get("image_factor", 1))
            for stage in result["stages"]
            if "image_factor" in stage
        ]
        blur_sigmas = [
            float(stage["blur_sigma"])
            for stage in result["stages"]
            if "blur_sigma" in stage
        ]
        summary = {
            "schema_version": 1,
            "status": "complete",
            "variant": variant,
            "protocol": protocol,
            "loss_branch_identifier": loss_branch,
            "run_tag": args.run_tag or None,
            "case_id": int(args.case_id),
            "seed": int(args.seed),
            "run_mode": run_mode,
            "shape_xyz": list(case.shape_xyz),
            "spacing_xyz_mm": [float(value) for value in case.spacing_xyz],
            "sampling_rule": case.provenance.get("optimization_mask_rule"),
            "metric_mask_rule": case.provenance.get("metric_mask_rule"),
            "loss_branch_identifier": loss_branch,
            "protocol_tags": case.provenance.get("protocol_tags", {}),
            "schedule_name": variant_schedule(variant),
            "stage_starts": stage_starts or config["training"]["stage_starts"],
            "stage_lengths": stage_lengths or config["training"]["stage_lengths"],
            "image_factors": image_factors or config["training"]["image_factors"],
            "blur_sigmas": blur_sigmas,
            "stages": result["stages"],
            "fine_activation_step": result["fine_activation_step"],
            "final_training_record": history[-1],
            "metrics": metrics,
            "runtime": {
                "training_wall_seconds": training_wall_seconds,
                "total_wall_seconds": total_wall_seconds,
            },
            "cuda_peak_memory_bytes": {
                "training_allocated": training_peak_allocated,
                "training_reserved": training_peak_reserved,
                "total_allocated": total_peak_allocated,
                "total_reserved": total_peak_reserved,
            },
            "optimizer_resolved": result["optimizer_settings"],
            "scheduler_resolved": result["scheduler_settings"],
            "regularization_multiplier": float(args.regularization_multiplier),
            "resolved_regularization_weights": config["controlled_adapter"].get(
                "resolved_regularization_weights", {}
            ),
            "data": _case_metadata(case),
            "artifacts": {
                "history": "history.csv",
                "stages": "stage_trace.csv",
                "config": "resolved_config.yaml",
                "provenance": "provenance.json",
                "model": "model_state.pt",
                "dense_dvf_xyz_voxel": "dense_dvf_xyz_voxel.npy",
                "fixed_metric_mask_xyz": "fixed_metric_mask_xyz.npy",
                "tre": "tre_scores.json",
                "regularity": "common_regularity.json (posthoc)",
            },
            "claim_limits": {
                "schedule_isolation": (
                    "compare N0/N1 or P0/P1 only; N/P/PL changes preprocessing or mask protocol"
                ),
                "multiresolution": (
                    "MR-D-BSCP-shaped image pyramid, not a B-spline control schedule"
                ),
                "scientific_performance_gate": None,
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "RUN_STATE.json",
            {
                "status": "complete",
                "variant": variant,
                "protocol": protocol,
                "loss_branch_identifier": loss_branch,
                "run_tag": args.run_tag or None,
                "case_id": int(args.case_id),
                "seed": int(args.seed),
                "run_mode": run_mode,
                "finished_utc": utc_now(),
                "training_wall_seconds": training_wall_seconds,
                "total_wall_seconds": total_wall_seconds,
                "partial_output_policy": partial_policy,
            },
        )
        with stdout_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"COMPLETE {utc_now()} variant={variant} case={args.case_id:02d} "
                f"headline_tre_mm={metrics['headline_tre']['tre_mean_mm']:.6f} "
                f"training_seconds={training_wall_seconds:.3f}\n"
            )
        (output_dir / "DONE").touch()
        print(
            f"{variant} case{args.case_id:02d} complete | "
            f"headline TRE={metrics['headline_tre']['tre_mean_mm']:.4f} mm | {output_dir}",
            flush=True,
        )
        return summary
    except BaseException as exc:
        failure = {
            "status": "failed",
            "variant": variant,
            "protocol": protocol,
            "loss_branch_identifier": loss_branch,
            "run_tag": args.run_tag or None,
            "case_id": int(args.case_id),
            "seed": int(args.seed),
            "run_mode": run_mode,
            "failed_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "partial_output_policy": partial_policy,
        }
        write_json(output_dir / "RUN_STATE.json", failure)
        stderr_path.write_text(failure["traceback"], encoding="utf-8")
        with stdout_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"FAILED {failure['failed_utc']} {failure['exception_type']}: "
                f"{failure['exception']}\n"
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=str.upper, choices=VARIANTS, required=True)
    parser.add_argument("--case-id", type=validate_case_id, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-raw-root", type=Path, required=True)
    parser.add_argument("--native-mask-root", type=Path, required=True)
    parser.add_argument("--ptv-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--regularization-multiplier", type=float, default=1.0)
    parser.add_argument("--ptv-isotv-weight", type=float, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=262144)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one step at each of the four stages; never use for scientific output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_chunk_size <= 0:
        raise ValueError("--eval-chunk-size must be positive")
    if args.regularization_multiplier <= 0.0:
        raise ValueError("--regularization-multiplier must be positive")
    if args.ptv_isotv_weight is not None and args.ptv_isotv_weight <= 0.0:
        raise ValueError("--ptv-isotv-weight must be positive when provided")
    run_case(args)


if __name__ == "__main__":
    main()
