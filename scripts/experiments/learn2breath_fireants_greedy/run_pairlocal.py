#!/usr/bin/env python3
"""CT-only FireANTs Greedy NCC on Learn2Breath-style 1.5-mm pairs.

This adapter deliberately shares the P2e pair loader, fixed-grid DVF
contract, physical TRE evaluator, and Learn2Breath challenge evaluator.  The
only optimisation input is a pair of in-memory, window-normalised CT images.
R231 masks, lobe labels, and COPD landmarks are loaded strictly after/beside
that path for preflight, evaluation, and qualitative QC.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
P2E_RUNNER = THIS_DIR.parent / "learn2breath_mrdbscp_p2e" / "run_pairlocal.py"
EXPECTED_SHAPE_XYZ = (224, 192, 224)
EXPECTED_SPACING_XYZ = (1.5, 1.5, 1.5)
METHOD_ID = "fireants_greedy_ct_only_ncc"
RUNTIME_ENVIRONMENT = os.environ.get("FIREANTS_RUNTIME_ENV", "fireants_runtime_unspecified")
SCALES = (16, 8, 4, 2, 1)
ITERATIONS = (200, 200, 200, 200, 200)
INTENSITY_PROFILES: dict[str, dict[str, Any]] = {
    "p2e_equiv_hu": {
        "low_hu": -920.0,
        "high_hu": -100.0,
        "description": "Historical P2e stored [80,900] expressed in physical HU.",
        "protocol_tag": "historical_p2e_stored80_900_equivalent_hu_minus920_minus100",
    },
    "l2b_lung_hu": {
        "low_hu": -1000.0,
        "high_hu": 200.0,
        "description": "Learn2Breath baseline lung window in physical HU.",
        "protocol_tag": "learn2breath_baseline_lung_window_hu_minus1000_200",
    },
}


class FireantsAuditError(RuntimeError):
    """Raised when the CT-only registration or field contract is violated."""


def p2e_module() -> ModuleType:
    """Load the existing common pair/evaluation contract without name clashes."""

    module_name = "learn2breath_p2e_pairlocal_contract"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    if not P2E_RUNNER.is_file():
        raise FileNotFoundError(f"Required P2e contract runner not found: {P2E_RUNNER}")
    spec = importlib.util.spec_from_file_location(module_name, P2E_RUNNER)
    if spec is None or spec.loader is None:
        raise FireantsAuditError(f"Cannot load P2e contract runner: {P2E_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def import_fireants_api() -> dict[str, Any]:
    """Import in the order required by the CUDA 12.1 FireANTs image."""

    import torch  # noqa: PLC0415

    # The fused extension must see libtorch already resident in this image.
    import fireants_fused_ops  # noqa: F401, PLC0415
    import fireants  # noqa: PLC0415
    import SimpleITK as sitk  # noqa: PLC0415
    from fireants.interpolator import fireants_interpolator  # noqa: PLC0415
    from fireants.io.image import BatchedImages, Image  # noqa: PLC0415
    from fireants.registration.greedy import GreedyRegistration  # noqa: PLC0415

    return {
        "fireants": fireants,
        "sitk": sitk,
        "torch": torch,
        "Image": Image,
        "BatchedImages": BatchedImages,
        "GreedyRegistration": GreedyRegistration,
        "fireants_interpolator": fireants_interpolator,
    }


def normalize_ct_hu(array_hu: np.ndarray, intensity_profile: str) -> np.ndarray:
    """Apply a locked physical-HU profile, never the legacy raw [80,900] rule."""

    profile = intensity_profile_config(intensity_profile)
    low = np.float32(profile["low_hu"])
    high = np.float32(profile["high_hu"])
    if not high > low:
        raise FireantsAuditError(f"Invalid intensity bounds for {intensity_profile}: {low}, {high}")
    values = np.asarray(array_hu, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise FireantsAuditError("CT must be a finite 3-D physical-HU array")
    return np.ascontiguousarray((np.clip(values, low, high) - low) / (high - low), dtype=np.float32)


def profile_occupancy(array_hu: np.ndarray, intensity_profile: str) -> dict[str, float]:
    profile = intensity_profile_config(intensity_profile)
    values = np.asarray(array_hu, dtype=np.float32)
    low = float(profile["low_hu"])
    high = float(profile["high_hu"])
    return {
        "low_hu": low,
        "high_hu": high,
        "pct_le_low": float((values <= low).mean() * 100.0),
        "pct_inside_open_window": float(np.logical_and(values > low, values < high).mean() * 100.0),
        "pct_ge_high": float((values >= high).mean() * 100.0),
    }


def ct_only_config(intensity_profile: str) -> dict[str, Any]:
    return {
        "method": METHOD_ID,
        "ct_only": True,
        "registration_inputs": "in_memory_scalar_ct_only",
        "excluded_from_registration_inputs_and_loss": ["copd_landmarks", "r231_masks", "validation_lobe_labels"],
        "intensity": {
            "input_units": "physical_HU",
            "selected_profile": intensity_profile_config(intensity_profile),
            "rule": "clip_physical_HU_to_profile_then_scale_to_[0,1]",
            "legacy_direct_stored_80_900_hu_normalizer_used": False,
        },
        "registration": {
            "algorithm": "FireANTs_Greedy",
            "loss_type": "local_normalized_cross_correlation",
            "cc_kernel_size": 5,
            "cc_kernel_type": "rectangular",
            "scales": list(SCALES),
            "iterations": list(ITERATIONS),
            "optimizer": "Adam",
            "optimizer_lr": 0.2,
            "deformation_type": "compositive",
            "smooth_warp_sigma": 0.25,
            "smooth_grad_sigma": 0.5,
            "tolerance": 1.0e-4,
            "max_tolerance_iters": 10,
            "init_affine": "identity",
        },
        "field_contract": "float32_[3,X,Y,Z]_XYZ_voxels_fixed_grid_moving(fixed+u)",
    }


def intensity_profile_config(name: str) -> dict[str, Any]:
    """Return the same audited physical-HU profile schema as the P2e audit."""

    try:
        profile = INTENSITY_PROFILES[str(name)]
    except KeyError as exc:
        raise FireantsAuditError(f"Unknown intensity profile: {name!r}") from exc
    return {"name": str(name), **profile}


def image_from_normalized_hu(
    source_path: Path,
    normalized_xyz: np.ndarray,
    *,
    device: str,
    dtype: Any,
) -> Any:
    """Build a FireANTs image in memory while retaining CT geometry exactly.

    The active FireANTs CUDA image hangs in ``torch.inverse`` while constructing
    the *constant* geometry cache on GPU.  Constructing the cache on CPU and
    transferring it is mathematically the same float32 image contract; it
    avoids that environment defect without changing the registration loss,
    optimiser, deformation, or sampling configuration.
    """

    api = import_fireants_api()
    sitk = api["sitk"]
    Image = api["Image"]
    torch = api["torch"]
    source = sitk.ReadImage(str(source_path))
    # SimpleITK arrays are Z,Y,X while nibabel/P2e arrays are X,Y,Z.
    array_zyx = np.ascontiguousarray(np.asarray(normalized_xyz, dtype=np.float32).transpose(2, 1, 0))
    normalized = sitk.GetImageFromArray(array_zyx, isVector=False)
    normalized.CopyInformation(source)
    image = Image(normalized, device="cpu", dtype=dtype, winsorize=False)
    # Keep the audited normalized float32 buffer rather than the Image helper's
    # default float64 SimpleITK roundtrip.  This has identical voxel values and
    # avoids a CUDA transfer stall observed for that temporary float64 buffer.
    image.array = torch.from_numpy(array_zyx).to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    image.channels = 1
    for attribute in ("array", "torch2phy", "phy2torch", "torch2px", "px2torch", "px2phy", "phy2px"):
        value = getattr(image, attribute, None)
        if value is not None:
            setattr(image, attribute, value.to(device=device))
    image.device = device
    return image


def load_ct_only_batches(case: Any, *, device: str) -> tuple[Any, Any]:
    """Return scalar CT batches; no mask, lobe, or landmark enters this function."""

    api = import_fireants_api()
    torch = api["torch"]
    BatchedImages = api["BatchedImages"]
    fixed = image_from_normalized_hu(
        case.fixed_path,
        normalize_ct_hu(case.fixed_hu_xyz, case.intensity_profile),
        device=device,
        dtype=torch.float32,
    )
    moving = image_from_normalized_hu(
        case.moving_path,
        normalize_ct_hu(case.moving_hu_xyz, case.intensity_profile),
        device=device,
        dtype=torch.float32,
    )
    if tuple(fixed.array.shape) != (1, 1, 224, 192, 224) or tuple(moving.array.shape) != (1, 1, 224, 192, 224):
        raise FireantsAuditError(f"Expected scalar [1,1,Z,Y,X] batches, got {fixed.array.shape}/{moving.array.shape}")
    return BatchedImages([fixed]), BatchedImages([moving])


def fireants_tensor_to_xyz(array_zyx: np.ndarray) -> np.ndarray:
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise FireantsAuditError(f"Expected [Z,Y,X] array, got {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 1, 0), dtype=np.float32)


def fireants_warped_ct_xyz(registration: Any, moved_params: dict[str, Any]) -> np.ndarray:
    api = import_fireants_api()
    torch = api["torch"]
    interpolator = api["fireants_interpolator"]
    with torch.no_grad():
        warped = interpolator(registration.moving_images()[:, :1], **moved_params, mode="bilinear", align_corners=True)
    return fireants_tensor_to_xyz(warped[0, 0].detach().cpu().numpy())


def materialize_fireants_dvf_zyx_xyz(registration: Any, moved_params: dict[str, Any]) -> np.ndarray:
    """Materialise FireANTs' native field before resolving public field axes/sign."""

    api = import_fireants_api()
    torch = api["torch"]
    interpolator = api["fireants_interpolator"]
    if int(getattr(registration, "dims")) != 3:
        raise FireantsAuditError("Only 3-D FireANTs registrations are supported")
    with torch.no_grad():
        grid = moved_params["grid"].detach()
        affine = moved_params["affine"].detach().clone()
        affine[:, :3, :3] -= torch.eye(3, device=affine.device, dtype=affine.dtype)
        displacement = interpolator.affine_warp(affine=affine.contiguous(), grid=grid.contiguous())
        for component, size in enumerate(list(displacement.shape[1:-1])[::-1]):
            displacement[..., component] *= (int(size) - 1) / 2.0
        native = displacement[0].detach().cpu().numpy()
    if not np.isfinite(native).all():
        raise FireantsAuditError("Materialized FireANTs field contains nonfinite values")
    return np.ascontiguousarray(native, dtype=np.float32)


def channel_last_candidates(native_zyx_xyz: np.ndarray, expected_shape_xyz: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Enumerate explicit spatial/component/sign candidates, selected by CT only."""

    native = np.asarray(native_zyx_xyz, dtype=np.float32)
    if native.ndim != 4 or native.shape[-1] != 3:
        raise FireantsAuditError(f"Expected FireANTs [Z,Y,X,3] field, got {native.shape}")
    spatial = {
        "transpose_zyx_to_xyz": np.ascontiguousarray(native.transpose(2, 1, 0, 3)),
        "native_zyx": np.ascontiguousarray(native),
    }
    candidates: dict[str, np.ndarray] = {}
    for spatial_name, field in spatial.items():
        if tuple(field.shape[:3]) != tuple(expected_shape_xyz):
            continue
        for component_name, mapped in (("xyz", field), ("zyx_to_xyz", field[..., [2, 1, 0]])):
            candidates[f"{spatial_name}_{component_name}"] = np.ascontiguousarray(mapped, dtype=np.float32)
            candidates[f"neg_{spatial_name}_{component_name}"] = np.ascontiguousarray(-mapped, dtype=np.float32)
    if not candidates:
        raise FireantsAuditError("No FireANTs field candidate matches the fixed XYZ grid")
    return candidates


def select_challenge_field(
    native_zyx_xyz: np.ndarray,
    moving_normalized_xyz: np.ndarray,
    fireants_warped_normalized_xyz: np.ndarray,
    *,
    evaluator: ModuleType,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose only by native FireANTs warped CT versus challenge-resampled CT."""

    candidates = channel_last_candidates(native_zyx_xyz, tuple(moving_normalized_xyz.shape))
    target = np.asarray(fireants_warped_normalized_xyz, dtype=np.float32)
    robust_range = float(np.percentile(target, 99) - np.percentile(target, 1))
    robust_range = max(robust_range, 1.0e-6)
    rows: list[dict[str, Any]] = []
    selected: np.ndarray | None = None
    selected_name = ""
    selected_rmse = math.inf
    for name, candidate in candidates.items():
        warped = evaluator.spatial_transform(moving_normalized_xyz, candidate, mode="bilinear", device=device)
        diff = warped - target
        rmse = float(np.sqrt(np.mean(diff * diff)))
        row = {
            "candidate": name,
            "finite": bool(np.isfinite(candidate).all() and np.isfinite(warped).all()),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": rmse,
            "normalized_rmse": rmse / robust_range,
            "max_abs": float(np.max(np.abs(diff))),
        }
        rows.append(row)
        if row["finite"] and rmse < selected_rmse:
            selected, selected_name, selected_rmse = candidate, name, rmse
    if selected is None:
        raise FireantsAuditError("No finite candidate field was compatible with challenge sampling")
    rows.sort(key=lambda row: float(row["rmse"]))
    return selected, {
        "selection_input": "normalized_CT_only_no_labels_masks_or_landmarks",
        "selected_candidate": selected_name,
        "robust_intensity_range": robust_range,
        "best_rmse": selected_rmse,
        "best_normalized_rmse": selected_rmse / robust_range,
        "candidates": rows,
    }


def field_cxyz_from_xyzc(field_xyzc: np.ndarray) -> np.ndarray:
    field = np.asarray(field_xyzc, dtype=np.float32)
    if field.shape != (*EXPECTED_SHAPE_XYZ, 3):
        raise FireantsAuditError(f"Expected challenge field {(*EXPECTED_SHAPE_XYZ, 3)}, got {field.shape}")
    return np.ascontiguousarray(np.moveaxis(field, -1, 0), dtype=np.float32)


def dependency_preflight(*, device: str) -> dict[str, Any]:
    api = import_fireants_api()
    torch = api["torch"]
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise FireantsAuditError("FireANTs production audit requires a visible CUDA device")
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return {
        "fireants_file": str(Path(api["fireants"].__file__).resolve()),
        "fireants_version": str(getattr(api["fireants"], "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "cuda_device_name": str(torch.cuda.get_device_name(device_index)),
        "cuda_device_count": int(torch.cuda.device_count()),
        "fused_ops_imported": True,
        "passed": True,
    }


def preflight_job(
    *,
    dataset: str,
    case_id: str,
    intensity_profile: str,
    data_root: Path,
    r231_root: Path | None,
    learn2breath_repo: Path,
    device: str,
) -> dict[str, Any]:
    """Run all geometry/evaluation gates before creating a FireANTs registration."""

    p2e = p2e_module()
    case = p2e.load_pair(dataset, data_root, case_id, intensity_profile)
    r231 = r231_root if dataset == "copd" else None
    base = p2e.preflight(
        case,
        data_root=data_root,
        r231_root=r231,
        case_id=case_id,
        learn2breath_repo=learn2breath_repo,
    )
    return {
        "passed": bool(base["passed"]),
        "method": METHOD_ID,
        "dataset": dataset,
        "case_id": str(case_id),
        "case_label": case.case_label,
        "intensity_profile": intensity_profile,
        "pair_preflight": base,
        "fixed_window_occupancy": profile_occupancy(case.fixed_hu_xyz, intensity_profile),
        "moving_window_occupancy": profile_occupancy(case.moving_hu_xyz, intensity_profile),
        "ct_only_config": ct_only_config(intensity_profile),
        "fireants_dependencies": dependency_preflight(device=device),
    }


def run_registration(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one field through export/reload and cohort-specific evaluation."""

    p2e = p2e_module()
    full_start = time.perf_counter()
    if args.out_root.exists() and (args.out_root / "summary.json").is_file() and not args.force:
        return json.loads((args.out_root / "summary.json").read_text(encoding="utf-8"))
    if args.out_root.exists() and any(args.out_root.iterdir()) and not args.force:
        raise FireantsAuditError(f"Refusing to overwrite incomplete output directory: {args.out_root}")
    case = p2e.load_pair(args.dataset, args.data_root, args.case_id, args.intensity_profile)
    evaluator = p2e.load_learn2breath_evaluator(args.learn2breath_repo)
    gate = preflight_job(
        dataset=args.dataset,
        case_id=args.case_id,
        intensity_profile=args.intensity_profile,
        data_root=args.data_root,
        r231_root=args.r231_root,
        learn2breath_repo=args.learn2breath_repo,
        device=args.device,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    p2e.write_json(args.out_root / "preflight.json", gate)
    config = ct_only_config(args.intensity_profile)
    p2e.write_json(args.out_root / "config.json", config)

    api = import_fireants_api()
    torch = api["torch"]
    if not torch.cuda.is_available():
        raise FireantsAuditError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    fixed_batch, moving_batch = load_ct_only_batches(case, device=args.device)
    registration = api["GreedyRegistration"](
        scales=list(SCALES),
        iterations=list(ITERATIONS),
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="cc",
        cc_kernel_size=5,
        cc_kernel_type="rectangular",
        optimizer="Adam",
        optimizer_lr=0.2,
        deformation_type="compositive",
        smooth_warp_sigma=0.25,
        smooth_grad_sigma=0.5,
        tolerance=1.0e-4,
        max_tolerance_iters=10,
        dtype=torch.float32,
        progress_bar=bool(args.progress),
        reduction="mean",
    )
    torch.cuda.synchronize(device)
    optimisation_start = time.perf_counter()
    registration.optimize()
    torch.cuda.synchronize(device)
    optimizer_elapsed = time.perf_counter() - optimisation_start

    moved = registration.get_warp_parameters(registration.fixed_images, registration.moving_images)
    fireants_warped = fireants_warped_ct_xyz(registration, moved)
    native = materialize_fireants_dvf_zyx_xyz(registration, moved)
    moving_normalized = normalize_ct_hu(case.moving_hu_xyz, case.intensity_profile)
    field_xyzc, compatibility = select_challenge_field(
        native,
        moving_normalized,
        fireants_warped,
        evaluator=evaluator,
        device=args.device,
    )
    field_cxyz = field_cxyz_from_xyzc(field_xyzc)
    field_path = args.out_root / "fields" / f"{case.case_label}_DVF.nii.gz"
    p2e.write_field(case, field_cxyz, field_path)
    np.save(args.out_root / "fields" / f"{case.case_label}_field_cxyz_vox.npy", field_cxyz, allow_pickle=False)
    p2e.write_json(args.out_root / "compatibility.json", compatibility)

    lobe_mask: np.ndarray | None = None
    if args.dataset == "validation":
        nib = p2e.import_nibabel()
        lobe_path = Path(evaluator.validation_paths(args.data_root, int(args.case_id))["insp_lobe"])
        lobe_mask = np.asarray(nib.load(str(lobe_path)).get_fdata(dtype=np.float32)) > 0
    quality = p2e.field_quality(field_cxyz, challenge_evaluator=evaluator, fixed_lobe_mask_xyz=lobe_mask)
    tre = p2e.tre_metrics(case, field_cxyz, args.out_root / "results" / "per_landmark_tre.csv")
    validation_metrics = (
        p2e.validation_evaluate(field_path, args.data_root, args.case_id, evaluator, args.device)
        if args.dataset == "validation"
        else None
    )
    full_elapsed = time.perf_counter() - full_start
    summary: dict[str, Any] = {
        "method": METHOD_ID,
        "runtime_environment": RUNTIME_ENVIRONMENT,
        "dataset": args.dataset,
        "case_id": str(args.case_id),
        "case_label": case.case_label,
        "intensity_profile": args.intensity_profile,
        "optimizer_elapsed_sec": float(optimizer_elapsed),
        "full_case_elapsed_sec": float(full_elapsed),
        "peak_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "field_path": field_path,
        "field_contract": config["field_contract"],
        "field_quality": quality,
        "compatibility": compatibility,
        "preflight_path": args.out_root / "preflight.json",
        "config_path": args.out_root / "config.json",
        "tre": tre,
        "validation": validation_metrics,
        "ct_only": True,
        "completed": True,
    }
    p2e.write_json(args.out_root / "summary.json", summary)
    del registration, fixed_batch, moving_batch
    gc.collect()
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("copd", "validation"), required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--intensity-profile", choices=("p2e_equiv_hu", "l2b_lung_hu"), required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--r231-root", type=Path)
    result.add_argument("--learn2breath-repo", type=Path, required=True)
    result.add_argument("--out-root", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--force", action="store_true")
    result.add_argument("--progress", action="store_true")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = run_registration(args)
    print(json.dumps(p2e_module().json_ready(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
