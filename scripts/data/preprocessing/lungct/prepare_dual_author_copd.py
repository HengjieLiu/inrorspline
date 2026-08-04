#!/usr/bin/env python3
"""Prepare a versioned COPD artifact for the released DUAL-INR-DIR runner.

The external COPD converter requires the original ZIP distribution.  This
adapter instead consumes the canonical extracted DIR-Lab tree while preserving
its author-facing convention: phase_00 is fixed iBH, phase_05 is moving eBH,
both raw images and landmark z coordinates are flipped exactly as in the
released converter, and a two-voxel padded TotalSegmentator five-lobe union
defines the DUAL crop.  The companion R231 mask is metric-only and is never
passed to the author optimizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file():
            return parent
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.common.lungct import (  # noqa: E402
    COPDCaseFiles,
    load_landmarks_xyz,
    resolve_copd_case,
)
from scripts.data.common.volume import parse_case_ids  # noqa: E402
from scripts.data.preprocessing.lungct.prepare_dual_author_4dct import (  # noqa: E402
    AUTHOR_CROP_PADDING,
    AUTHOR_FIXED_PHASE,
    AUTHOR_LUNG_CLASSES,
    AUTHOR_LUNG_CLASS_NAMES,
    AUTHOR_MOVING_PHASE,
    SCHEMA_VERSION,
    audit_author_case,
    author_image_from_raw,
    author_landmarks_xyz,
    copy_segmentation_as_nii,
    file_record,
    find_reusable_segmentation,
    permission_audit,
    require_sitk,
    run_totalsegmentator,
    sha256_file,
    validate_output_location,
    validate_totalsegmentator_installation,
    verify_checksums,
    write_author_image,
    write_checksums,
    write_json,
    write_landmarks,
)


ADAPTER_ID = "dual_author_copd_ibh_ebh"
FIXED_PHASE_NAME = "iBH"
MOVING_PHASE_NAME = "eBH"
METRIC_MASK_FILENAME = "fixed_iBH_r231_metricmask_author_full_xyz.npy"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_paths(case: COPDCaseFiles) -> dict[str, Path]:
    values = {
        "fixed_image_iBH": case.image_ibh,
        "moving_image_eBH": case.image_ebh,
        "fixed_landmarks_iBH": case.landmarks_ibh,
        "moving_landmarks_eBH": case.landmarks_ebh,
    }
    if any(value is None for value in values.values()):
        raise ValueError(f"Case {case.case_id:02d} has incomplete raw COPD paths")
    return {key: Path(value) for key, value in values.items() if value is not None}


def _resolve_metric_mask(native_mask_root: Path, case_id: int) -> Path:
    root = native_mask_root.expanduser().resolve()
    name = f"case{case_id:02d}_iBH_R231_binary_zyx.npy"
    candidates = (
        root / "COPD" / f"case{case_id:02d}" / name,
        root / "copd" / f"case{case_id:02d}" / name,
        root / f"case{case_id:02d}" / name,
        root / "processed" / "lungmask_R231_native" / "COPD" / f"case{case_id:02d}" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate fixed iBH R231 mask {name} below {root}")


def parse_case_ids_str(text: str) -> list[int]:
    values = parse_case_ids(text)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Case IDs must be a nonempty unique subset of 1..10: {values}")
    invalid = [value for value in values if value not in range(1, 11)]
    if invalid:
        raise ValueError(f"COPD case IDs must be in 1..10, got {invalid}")
    return values


def validate_source_case(
    case_id: int, *, data_root: Path, native_mask_root: Path
) -> tuple[COPDCaseFiles, dict[str, Any]]:
    case = resolve_copd_case(case_id, data_root)
    if case.status != "complete":
        raise ValueError(f"COPD case {case_id:02d} source is {case.status}: {case.notes}")
    paths = _source_paths(case)
    expected_bytes = int(np.prod(case.shape_zyx)) * np.dtype(np.int16).itemsize
    for key in ("fixed_image_iBH", "moving_image_eBH"):
        if paths[key].stat().st_size != expected_bytes:
            raise ValueError(f"{paths[key]} does not have expected raw-image byte count")
    shape_xyz = np.asarray(case.shape_zyx[::-1], dtype=np.float64)
    landmark_shapes: dict[str, list[int]] = {}
    for key in ("fixed_landmarks_iBH", "moving_landmarks_eBH"):
        points = load_landmarks_xyz(paths[key])
        if points.shape != (300, 3) or not np.isfinite(points).all():
            raise ValueError(f"{paths[key]} must contain 300 finite xyz landmarks")
        if not np.all((points >= 1.0) & (points <= shape_xyz[None, :])):
            raise ValueError(f"{paths[key]} contains landmarks outside 1-based native bounds")
        landmark_shapes[key] = [300, 3]
    metric_mask = _resolve_metric_mask(native_mask_root, case_id)
    mask = np.load(metric_mask, allow_pickle=False)
    if mask.shape != tuple(case.shape_zyx) or not np.any(mask):
        raise ValueError(f"Invalid fixed iBH R231 mask {metric_mask}: {mask.shape}")
    return case, {
        "case_id": int(case_id),
        "status": "pass",
        "shape_zyx": [int(value) for value in case.shape_zyx],
        "spacing_zyx_mm": [float(value) for value in case.spacing_zyx],
        "source_paths": {key: str(value) for key, value in paths.items()},
        "landmark_shapes": landmark_shapes,
        "metric_mask": str(metric_mask),
    }


def _required_paths(case_dir: Path) -> dict[str, Path]:
    return {
        "fixed_image": case_dir / "images" / "phase_00.nii",
        "moving_image": case_dir / "images" / "phase_05.nii",
        "fixed_landmarks": case_dir / "landmarks_raw" / "extreme_landmarks_0.csv",
        "moving_landmarks": case_dir / "landmarks_raw" / "extreme_landmarks_5.csv",
        "fixed_segmentation": case_dir / "segmentations_00.nii",
        "moving_segmentation": case_dir / "segmentations_05.nii",
        "metadata": case_dir / "metadata.yaml",
        "fixed_r231_metric_mask": case_dir / "metric_masks" / METRIC_MASK_FILENAME,
    }


def _write_metadata(path: Path, case: COPDCaseFiles) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required in the Docker environment") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_shape": [int(value) for value in case.shape_zyx[::-1]],
        "image_spacing": [float(value) for value in case.spacing_zyx[::-1]],
        "fixed_phase": "iBH / author phase_00",
        "moving_phase": "eBH / author phase_05",
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _metric_mask_author_xyz(mask_zyx: np.ndarray) -> np.ndarray:
    # The raw R231 mask is in canonical raw zyx order; author images flip z
    # before SimpleITK serialization, then DUAL operates on xyz NumPy arrays.
    return np.ascontiguousarray(np.transpose(np.flip(mask_zyx > 0, axis=0), (2, 1, 0)), dtype=np.uint8)


def _identity_tre(
    fixed_xyz: np.ndarray, moving_xyz: np.ndarray, spacing_xyz: Sequence[float]
) -> dict[str, float | int]:
    distances = np.linalg.norm(
        (np.asarray(fixed_xyz, dtype=np.float64) - np.asarray(moving_xyz, dtype=np.float64))
        * np.asarray(spacing_xyz, dtype=np.float64)[None, :],
        axis=1,
    )
    return {
        "n": int(distances.size),
        "mean_mm": float(np.mean(distances)),
        "std_mm_population": float(np.std(distances)),
    }


def _build_case(
    case: COPDCaseFiles,
    *,
    dataset_root: Path,
    native_mask_root: Path,
    segmentation_mode: str,
    reuse_segmentations_root: Path | None,
    totalsegmentator_device: str,
) -> dict[str, Any]:
    case_dir = dataset_root / f"case_{case.case_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    paths = _required_paths(case_dir)
    sources = _source_paths(case)
    shape_xyz = tuple(int(value) for value in case.shape_zyx[::-1])

    fixed = author_image_from_raw(sources["fixed_image_iBH"], case.shape_zyx)
    moving = author_image_from_raw(sources["moving_image_eBH"], case.shape_zyx)
    write_author_image(paths["fixed_image"], fixed, case.spacing_zyx)
    write_author_image(paths["moving_image"], moving, case.spacing_zyx)
    fixed_landmarks = author_landmarks_xyz(
        load_landmarks_xyz(sources["fixed_landmarks_iBH"]), shape_xyz
    )
    moving_landmarks = author_landmarks_xyz(
        load_landmarks_xyz(sources["moving_landmarks_eBH"]), shape_xyz
    )
    write_landmarks(paths["fixed_landmarks"], fixed_landmarks)
    write_landmarks(paths["moving_landmarks"], moving_landmarks)
    _write_metadata(paths["metadata"], case)

    raw_metric_path = _resolve_metric_mask(native_mask_root, case.case_id)
    metric_mask_raw = np.load(raw_metric_path, allow_pickle=False)
    metric_mask_author = _metric_mask_author_xyz(metric_mask_raw)
    paths["fixed_r231_metric_mask"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["fixed_r231_metric_mask"], metric_mask_author)

    if segmentation_mode == "totalsegmentator":
        run_totalsegmentator(paths["fixed_image"], paths["fixed_segmentation"], totalsegmentator_device)
        run_totalsegmentator(paths["moving_image"], paths["moving_segmentation"], totalsegmentator_device)
        segmentation_sources = {
            "fixed_iBH": "generated_from_author_formatted_iBH_image",
            "moving_eBH": "generated_from_author_formatted_eBH_image",
        }
    elif segmentation_mode == "reuse":
        if reuse_segmentations_root is None:
            raise ValueError("--reuse-segmentations-root is required for reuse mode")
        copy_segmentation_as_nii(
            find_reusable_segmentation(reuse_segmentations_root, case.case_id, AUTHOR_FIXED_PHASE),
            paths["fixed_segmentation"],
        )
        copy_segmentation_as_nii(
            find_reusable_segmentation(reuse_segmentations_root, case.case_id, AUTHOR_MOVING_PHASE),
            paths["moving_segmentation"],
        )
        segmentation_sources = {"fixed_iBH": "reused", "moving_eBH": "reused"}
    else:
        raise ValueError(f"Unknown segmentation mode: {segmentation_mode}")

    audit = audit_author_case(case_dir, case.case_id)
    if audit["status"] != "pass":
        failed = [key for key, value in audit["checks"].items() if not value]
        raise RuntimeError(f"COPD case {case.case_id:02d} author audit failed: {failed}")
    if metric_mask_author.shape != tuple(audit["image_geometry"]["size_xyz"]):
        raise RuntimeError("Metric-mask geometry does not match author images")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "case_id": int(case.case_id),
        "generated_at_utc": utc_now(),
        "fixed_phase": "iBH / author phase_00",
        "moving_phase": "eBH / author phase_05",
        "registration_direction": "eBH_to_iBH",
        "segmentation_mode": segmentation_mode,
        "segmentation_sources": segmentation_sources,
        "identity_tre_mm": _identity_tre(fixed_landmarks, moving_landmarks, case.spacing_zyx[::-1]),
        "source_files": {key: file_record(value) for key, value in sources.items()},
        "fixed_r231_metric_mask_source": file_record(raw_metric_path),
        "fixed_r231_metric_mask_contract": {
            "role": "posthoc metric only; never passed to DUAL optimization",
            "array_order": "xyz after author z-flip",
            "voxels": int(metric_mask_author.sum()),
        },
        "audit": audit,
    }
    write_json(case_dir / "adapter_case_manifest.json", manifest)
    return {
        "case_id": int(case.case_id),
        "status": "pass",
        "case_manifest": f"case_{case.case_id:02d}/adapter_case_manifest.json",
        "case_manifest_sha256": sha256_file(case_dir / "adapter_case_manifest.json"),
        "dual_union_crop": audit["dual_union_crop"],
    }


def _copy_executed_script(dataset_root: Path) -> dict[str, Any]:
    target = dataset_root / "provenance" / "prepare_dual_author_copd_executed.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), target)
    target.chmod(0o444)
    return file_record(target, relative_to=dataset_root)


def _contract(case_ids: Sequence[int], version: str, segmentation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "artifact_version": version,
        "case_ids": [int(value) for value in case_ids],
        "dataset": "copd",
        "registration_direction": "eBH_to_iBH",
        "author_phase_mapping": {
            "phase_00": "fixed_iBH",
            "phase_05": "moving_eBH",
        },
        "image_and_landmark_transform": "released DUAL COPD converter raw z-flip",
        "crop": {
            "rule": "fixed_moving_totalsegmentator_five_lobe_union",
            "labels": {str(key): value for key, value in AUTHOR_LUNG_CLASS_NAMES.items()},
            "padding_voxels": AUTHOR_CROP_PADDING,
        },
        "metric_mask": "fixed iBH R231, mapped to author geometry; metric-only",
        "segmentation": dict(segmentation),
        "reference": {
            "external_path": "external/DUAL-INR-DIR",
            "adapter_script": Path(__file__).relative_to(REPO_ROOT).as_posix(),
            "adapter_sha256": sha256_file(Path(__file__)),
        },
    }


def dry_validate(args: argparse.Namespace) -> dict[str, Any]:
    target = validate_output_location(args.output_parent, args.version)
    if target.exists():
        raise FileExistsError(f"Target version exists: {target}")
    case_ids = parse_case_ids_str(args.case_ids)
    sources = [
        validate_source_case(case_id, data_root=args.data_root, native_mask_root=args.native_mask_root)[1]
        for case_id in case_ids
    ]
    segmenter = validate_totalsegmentator_installation(require_weights=True, hash_weights=False)
    return {
        "mode": "prepare_dry_run",
        "status": "pass",
        "writes_performed": False,
        "prospective_target": str(target),
        "case_ids": case_ids,
        "source_validation": sources,
        "segmentation": segmenter,
    }


def prepare(args: argparse.Namespace) -> Path:
    target = validate_output_location(args.output_parent, args.version)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")
    case_ids = parse_case_ids_str(args.case_ids)
    cases = [
        validate_source_case(case_id, data_root=args.data_root, native_mask_root=args.native_mask_root)[0]
        for case_id in case_ids
    ]
    segmenter = validate_totalsegmentator_installation(require_weights=True, hash_weights=False)
    args.output_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.version}.staging-", dir=args.output_parent) as staging_name:
        staging = Path(staging_name)
        rows = [
            _build_case(
                case,
                dataset_root=staging,
                native_mask_root=args.native_mask_root,
                segmentation_mode=args.segmentation_mode,
                reuse_segmentations_root=args.reuse_segmentations_root,
                totalsegmentator_device=args.totalsegmentator_device,
            )
            for case in cases
        ]
        write_json(staging / "preprocess_contract.json", _contract(case_ids, args.version, segmenter))
        write_json(
            staging / "dataset_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "adapter_id": ADAPTER_ID,
                "artifact_version": args.version,
                "generated_at_utc": utc_now(),
                "status": "pass",
                "case_ids": case_ids,
                "case_count": len(case_ids),
                "rows": rows,
            },
        )
        _copy_executed_script(staging)
        staging.chmod(0o755)
        write_checksums(staging)
        checksum = verify_checksums(staging)
        if checksum["status"] != "pass":
            raise RuntimeError(f"Checksum self-audit failed: {checksum['failures']}")
        staging.replace(target)
    permission = permission_audit(target)
    if permission["status"] != "pass":
        raise RuntimeError(f"Published artifact permissions invalid: {permission}")
    return target


def validate_existing(dataset_root: Path, case_ids_text: str | None = None) -> dict[str, Any]:
    root = dataset_root.resolve()
    manifest_path = root / "dataset_manifest.json"
    contract_path = root / "preprocess_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Missing dataset_manifest.json or preprocess_contract.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if manifest.get("adapter_id") != ADAPTER_ID or contract.get("adapter_id") != ADAPTER_ID:
        raise ValueError("Artifact was not produced by the COPD author adapter")
    case_ids = parse_case_ids_str(case_ids_text) if case_ids_text else [int(value) for value in manifest["case_ids"]]
    audits = []
    for case_id in case_ids:
        case_dir = root / f"case_{case_id:02d}"
        audit = audit_author_case(case_dir, case_id)
        metric_path = case_dir / "metric_masks" / METRIC_MASK_FILENAME
        if not metric_path.is_file():
            audit["status"] = "fail"
            audit.setdefault("checks", {})["fixed_r231_metric_mask_present"] = False
        else:
            metric = np.load(metric_path, allow_pickle=False)
            expected = tuple(audit["image_geometry"]["size_xyz"])
            audit.setdefault("checks", {})["fixed_r231_metric_mask_geometry_match"] = metric.shape == expected and bool(metric.sum())
            audit["status"] = "pass" if audit["status"] == "pass" and all(audit["checks"].values()) else "fail"
        audits.append(audit)
    checksums = verify_checksums(root)
    permissions = permission_audit(root)
    executed = root / "provenance" / "prepare_dual_author_copd_executed.py"
    executed_ok = executed.is_file() and (executed.stat().st_mode & 0o777) == 0o444
    passed = checksums["status"] == "pass" and permissions["status"] == "pass" and executed_ok and all(row["status"] == "pass" for row in audits)
    return {
        "mode": "validate_existing",
        "status": "pass" if passed else "fail",
        "writes_performed": False,
        "dataset_root": str(root),
        "case_ids": case_ids,
        "case_audits": audits,
        "checksum_audit": checksums,
        "permission_audit": permissions,
        "executed_adapter": {"path": str(executed), "status": "pass" if executed_ok else "fail"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--native-mask-root", type=Path, required=True)
    prepare_parser.add_argument("--output-parent", type=Path, required=True)
    prepare_parser.add_argument("--version", required=True)
    prepare_parser.add_argument("--case-ids", default="1-10")
    prepare_parser.add_argument("--segmentation-mode", choices=("totalsegmentator", "reuse"), default="totalsegmentator")
    prepare_parser.add_argument("--reuse-segmentations-root", type=Path)
    prepare_parser.add_argument("--totalsegmentator-device", default="gpu")
    prepare_parser.add_argument("--dry-run", action="store_true")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--dataset-root", type=Path, required=True)
    validate_parser.add_argument("--case-ids")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare" and args.dry_run:
            result = dry_validate(args)
        elif args.command == "prepare":
            target = prepare(args)
            result = {"mode": "prepare", "status": "pass", "dataset_root": str(target)}
        else:
            result = validate_existing(args.dataset_root, args.case_ids)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "pass":
            raise SystemExit(1)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
