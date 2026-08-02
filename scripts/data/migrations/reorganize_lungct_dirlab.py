#!/usr/bin/env python3
"""Reorganize the local LungCT dataset into the canonical DIR-Lab layout.

The migration is intentionally staged. ``inventory`` records and verifies the
source trees, ``move`` performs same-filesystem renames and rewrites textual
metadata, ``verify`` audits the destination, and ``cleanup`` removes only
source trees whose duplicate checks passed during ``inventory``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASETS_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_DATASETS_ROOT",
        f"/data/{os.environ.get('USER', '$USER')}/datasets",
    )
)
LUNGCT_ROOT = DATASETS_ROOT / "lungCT"
DIRLAB_ROOT = LUNGCT_ROOT / "DIR-Lab"
MANIFEST_ROOT = DATASETS_ROOT / "manifests" / "lungCT_DIR-Lab_reorganization_2026-07-11"

RAW_SOURCES = {
    "4DCT": LUNGCT_ROOT / "raw" / "4DCT",
    "COPD": LUNGCT_ROOT / "raw" / "COPD",
}
RAW_DUPLICATES = {
    "4DCT": LUNGCT_ROOT / "4DCT",
    "COPD": LUNGCT_ROOT / "COPD",
}
MASK_SOURCE = LUNGCT_ROOT / "processed" / "masks"
MASK_STAGE = (
    LUNGCT_ROOT
    / "processed"
    / "preprocessing"
    / "lung_masks"
    / "r231_t00_t50_v1"
)
PTV_SOURCE = (
    LUNGCT_ROOT
    / "processed"
    / "preprocessing"
    / "ptvprep"
    / "ptvreg_landmark_crop_1mm_trilinear_v1"
)

TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".tsv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["relative_path", "bytes", "sha256"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def inventory_one(name: str, root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise RuntimeError(f"Missing inventory source: {root}")
    rows = tree_rows(root)
    write_tsv(MANIFEST_ROOT / "pre" / f"{name}.sha256.tsv", rows)
    return rows


def require_identical(
    label: str, left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> None:
    if left != right:
        raise RuntimeError(f"Duplicate verification failed for {label}")


def inventory() -> None:
    if DIRLAB_ROOT.exists():
        raise RuntimeError(f"Destination already exists: {DIRLAB_ROOT}")
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    mapping_rows = [
        {
            "source": str(RAW_SOURCES["4DCT"]),
            "destination": str(DIRLAB_ROOT / "raw" / "4DCT"),
            "action": "move",
        },
        {
            "source": str(RAW_SOURCES["COPD"]),
            "destination": str(DIRLAB_ROOT / "raw" / "COPD"),
            "action": "move",
        },
        {
            "source": str(MASK_SOURCE),
            "destination": str(DIRLAB_ROOT / "processed" / "lungmask_R231_native"),
            "action": "restructure_move",
        },
        {
            "source": str(PTV_SOURCE),
            "destination": str(DIRLAB_ROOT / "processed" / "pTVReg_iso1mm"),
            "action": "restructure_move",
        },
        {
            "source": str(LUNGCT_ROOT / "manifests"),
            "destination": str(MANIFEST_ROOT),
            "action": "replace_after_verify",
        },
    ]
    write_tsv(MANIFEST_ROOT / "path_mapping.tsv", mapping_rows)

    raw_4dct = inventory_one("canonical_raw_4DCT", RAW_SOURCES["4DCT"])
    raw_copd = inventory_one("canonical_raw_COPD", RAW_SOURCES["COPD"])
    duplicate_4dct = inventory_one("duplicate_top_4DCT", RAW_DUPLICATES["4DCT"])
    duplicate_copd = inventory_one("duplicate_top_COPD", RAW_DUPLICATES["COPD"])
    require_identical("top-level 4DCT", raw_4dct, duplicate_4dct)
    require_identical("top-level COPD", raw_copd, duplicate_copd)

    masks_4dct = inventory_one("canonical_masks_4DCT", MASK_SOURCE / "4dct")
    masks_copd = inventory_one("canonical_masks_COPD", MASK_SOURCE / "copd")
    staged_4dct = inventory_one("duplicate_staged_masks_4DCT", MASK_STAGE / "4dct")
    require_identical("staged 4DCT masks", masks_4dct, staged_4dct)
    ptv = inventory_one("canonical_pTVReg_iso1mm", PTV_SOURCE)

    summary = {
        "status": "inventory_complete",
        "duplicate_verification": {
            "top_4DCT": "sha256_match",
            "top_COPD": "sha256_match",
            "staged_masks_4DCT": "sha256_match",
        },
        "source_totals": {
            "raw_4DCT": {"files": len(raw_4dct), "bytes": sum(r["bytes"] for r in raw_4dct)},
            "raw_COPD": {"files": len(raw_copd), "bytes": sum(r["bytes"] for r in raw_copd)},
            "masks_4DCT": {"files": len(masks_4dct), "bytes": sum(r["bytes"] for r in masks_4dct)},
            "masks_COPD": {"files": len(masks_copd), "bytes": sum(r["bytes"] for r in masks_copd)},
            "pTVReg_iso1mm": {"files": len(ptv), "bytes": sum(r["bytes"] for r in ptv)},
        },
    }
    (MANIFEST_ROOT / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def rename(source: Path, destination: Path) -> None:
    if destination.exists() and not source.exists():
        return
    if not source.exists():
        raise RuntimeError(f"Missing move source: {source}")
    if destination.exists():
        raise RuntimeError(f"Move destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)


def move_optional(source: Path, destination: Path) -> None:
    if source.exists():
        rename(source, destination)


def path_replacements() -> list[tuple[str, str]]:
    host = str(LUNGCT_ROOT)
    docker = os.environ.get("INRORSPLINE_DOCKER_LUNGCT_ROOT", "/database/datasets/lungCT")
    pairs: list[tuple[str, str]] = []
    for prefix in (host, docker):
        pairs.extend(
            [
                (
                    f"{prefix}/DIR-Lab/processed/pTVReg_iso1mm/4dct",
                    f"{prefix}/DIR-Lab/processed/pTVReg_iso1mm/4DCT",
                ),
                (
                    f"{prefix}/DIR-Lab/processed/pTVReg_iso1mm/copd",
                    f"{prefix}/DIR-Lab/processed/pTVReg_iso1mm/COPD",
                ),
                (
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native/4dct",
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native/4DCT",
                ),
                (
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native/copd",
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native/COPD",
                ),
                (
                    f"{prefix}/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1",
                    f"{prefix}/DIR-Lab/processed/pTVReg_iso1mm",
                ),
                (
                    f"{prefix}/processed/preprocessing/lung_masks/r231_t00_t50_v1",
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native",
                ),
                (
                    f"{prefix}/processed/masks",
                    f"{prefix}/DIR-Lab/processed/lungmask_R231_native",
                ),
                (f"{prefix}/raw/4DCT", f"{prefix}/DIR-Lab/raw/4DCT"),
                (f"{prefix}/raw/COPD", f"{prefix}/DIR-Lab/raw/COPD"),
                (f"{prefix}/4DCT", f"{prefix}/DIR-Lab/raw/4DCT"),
                (f"{prefix}/COPD", f"{prefix}/DIR-Lab/raw/COPD"),
                (prefix, f"{prefix}/DIR-Lab"),
            ]
        )
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def rewrite_string(value: str) -> str:
    prefixes = (
        str(LUNGCT_ROOT),
        os.environ.get("INRORSPLINE_DOCKER_LUNGCT_ROOT", "/database/datasets/lungCT"),
    )
    for prefix in prefixes:
        repeated = f"{prefix}/DIR-Lab/DIR-Lab"
        while repeated in value:
            value = value.replace(repeated, f"{prefix}/DIR-Lab")
        canonical_prefix = f"{prefix}/DIR-Lab"
        if canonical_prefix in value:
            value = value.replace("/pTVReg_iso1mm/4dct/", "/pTVReg_iso1mm/4DCT/")
            value = value.replace("/pTVReg_iso1mm/copd/", "/pTVReg_iso1mm/COPD/")
            value = value.replace("/lungmask_R231_native/4dct/", "/lungmask_R231_native/4DCT/")
            value = value.replace("/lungmask_R231_native/copd/", "/lungmask_R231_native/COPD/")
            return value
    for old, new in path_replacements():
        if old in value:
            return value.replace(old, new)
    if value == "ptvreg_landmark_crop_1mm_trilinear_v1":
        return "pTVReg_iso1mm"
    return value


def rewrite_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return rewrite_string(value)
    if isinstance(value, list):
        return [rewrite_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_json_value(item) for key, item in value.items()}
    return value


def rewrite_metadata(root: Path) -> list[str]:
    changed: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() == ".json":
            original = json.loads(path.read_text())
            rewritten = rewrite_json_value(original)
            if rewritten != original:
                path.write_text(json.dumps(rewritten, indent=2, allow_nan=True) + "\n")
                changed.append(path.relative_to(root).as_posix())
        elif path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle, delimiter=delimiter))
            rewritten_rows = [[rewrite_string(cell) for cell in row] for row in rows]
            if rewritten_rows != rows:
                with path.open("w", newline="") as handle:
                    csv.writer(handle, delimiter=delimiter).writerows(rewritten_rows)
                changed.append(path.relative_to(root).as_posix())
    return changed


def move() -> None:
    if not (MANIFEST_ROOT / "inventory_summary.json").is_file():
        raise RuntimeError("Run inventory successfully before move")
    rename(RAW_SOURCES["4DCT"], DIRLAB_ROOT / "raw" / "4DCT")
    rename(RAW_SOURCES["COPD"], DIRLAB_ROOT / "raw" / "COPD")

    mask_target = DIRLAB_ROOT / "processed" / "lungmask_R231_native"
    rename(MASK_SOURCE / "4dct", mask_target / "4DCT")
    rename(MASK_SOURCE / "copd", mask_target / "COPD")
    move_optional(MASK_SOURCE / "copd_generation_summary.json", mask_target / "COPD" / "generation_summary.json")
    move_optional(MASK_STAGE / "4dct_generation_summary.json", mask_target / "4DCT" / "generation_summary.json")
    for report in sorted(MASK_STAGE.glob("4dct_*report.json")):
        rename(report, mask_target / "4DCT" / report.name.removeprefix("4dct_"))

    ptv_target = DIRLAB_ROOT / "processed" / "pTVReg_iso1mm"
    rename(PTV_SOURCE / "4dct", ptv_target / "4DCT")
    rename(PTV_SOURCE / "copd", ptv_target / "COPD")
    move_optional(PTV_SOURCE / "4dct_ptvprep_summary.json", ptv_target / "4DCT" / "summary.json")
    move_optional(PTV_SOURCE / "4dct_ptvprep_summary.csv", ptv_target / "4DCT" / "summary.csv")
    move_optional(PTV_SOURCE / "copd_ptvprep_summary.json", ptv_target / "COPD" / "summary.json")
    move_optional(PTV_SOURCE / "copd_ptvprep_summary.csv", ptv_target / "COPD" / "summary.csv")
    move_optional(PTV_SOURCE / "figures", ptv_target / "4DCT" / "figures")

    changed = rewrite_metadata(DIRLAB_ROOT)
    (MANIFEST_ROOT / "metadata_rewrite.json").write_text(
        json.dumps({"changed_files": changed, "count": len(changed)}, indent=2) + "\n"
    )


def expected_rows(name: str) -> list[dict[str, str]]:
    return read_tsv(MANIFEST_ROOT / "pre" / f"{name}.sha256.tsv")


def verify_immutable_tree(name: str, root: Path) -> None:
    expected = {
        row["relative_path"]: row
        for row in expected_rows(name)
        if Path(row["relative_path"]).suffix.lower() not in TEXT_SUFFIXES
    }
    actual = {
        row["relative_path"]: {"bytes": str(row["bytes"]), "sha256": row["sha256"]}
        for row in tree_rows(root)
        if Path(row["relative_path"]).suffix.lower() not in TEXT_SUFFIXES
    }
    if set(expected) != set(actual):
        raise RuntimeError(f"Immutable file set mismatch for {name}")
    for relative_path, row in expected.items():
        if row["bytes"] != actual[relative_path]["bytes"] or row["sha256"] != actual[relative_path]["sha256"]:
            raise RuntimeError(f"Immutable checksum mismatch: {name}/{relative_path}")


def verify() -> None:
    targets = {
        "canonical_raw_4DCT": DIRLAB_ROOT / "raw" / "4DCT",
        "canonical_raw_COPD": DIRLAB_ROOT / "raw" / "COPD",
        "canonical_masks_4DCT": DIRLAB_ROOT / "processed" / "lungmask_R231_native" / "4DCT",
        "canonical_masks_COPD": DIRLAB_ROOT / "processed" / "lungmask_R231_native" / "COPD",
    }
    for name, root in targets.items():
        verify_immutable_tree(name, root)

    ptv_root = DIRLAB_ROOT / "processed" / "pTVReg_iso1mm"
    expected_ptv = expected_rows("canonical_pTVReg_iso1mm")
    expected_binary = {
        row["relative_path"]: row
        for row in expected_ptv
        if Path(row["relative_path"]).suffix.lower() not in TEXT_SUFFIXES
    }
    actual_binary: dict[str, dict[str, str]] = {}
    for dataset in ("4DCT", "COPD"):
        source_dataset = dataset.lower()
        for row in tree_rows(ptv_root / dataset):
            relative = row["relative_path"]
            if relative.startswith("figures/"):
                original_relative = relative
            else:
                original_relative = f"{source_dataset}/{relative}"
            if Path(relative).suffix.lower() not in TEXT_SUFFIXES:
                actual_binary[original_relative] = {
                    "bytes": str(row["bytes"]),
                    "sha256": row["sha256"],
                }
    if set(expected_binary) != set(actual_binary):
        missing = sorted(set(expected_binary) - set(actual_binary))
        extra = sorted(set(actual_binary) - set(expected_binary))
        raise RuntimeError(f"pTV immutable file set mismatch; missing={missing}; extra={extra}")
    for relative_path, row in expected_binary.items():
        actual = actual_binary[relative_path]
        if row["bytes"] != actual["bytes"] or row["sha256"] != actual["sha256"]:
            raise RuntimeError(f"pTV immutable checksum mismatch: {relative_path}")

    stale_files: list[str] = []
    for path in DIRLAB_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            if "/datasets/lungCT/processed/" in text or "/datasets/lungCT/4DCT" in text or "/datasets/lungCT/COPD" in text:
                stale_files.append(str(path))
            if "/pTVReg_iso1mm/4dct/" in text or "/pTVReg_iso1mm/copd/" in text:
                stale_files.append(str(path))
            if "/lungmask_R231_native/4dct/" in text or "/lungmask_R231_native/copd/" in text:
                stale_files.append(str(path))
            if path.suffix.lower() == ".json":
                json.loads(text)
    if stale_files:
        raise RuntimeError(f"Stale embedded paths remain: {stale_files}")

    final_rows = tree_rows(DIRLAB_ROOT)
    write_tsv(MANIFEST_ROOT / "final_DIR-Lab.sha256.tsv", final_rows)
    report = {
        "status": "verified",
        "root": str(DIRLAB_ROOT),
        "files": len(final_rows),
        "bytes": sum(row["bytes"] for row in final_rows),
        "top_level_entries": sorted(path.name for path in DIRLAB_ROOT.iterdir()),
        "metadata_references": "no stale pre-migration paths",
        "immutable_content": "matches pre-migration SHA-256 manifests",
    }
    (MANIFEST_ROOT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")


def open_handles(path: Path) -> list[str]:
    if shutil.which("lsof") is None:
        return []
    result = subprocess.run(
        ["lsof", "+D", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[1:] if result.stdout else []


def remove_tree(path: Path, removed: list[str]) -> None:
    if path.exists():
        shutil.rmtree(path)
        removed.append(str(path))


def verify_cleanup_source(name: str, root: Path) -> None:
    expected = expected_rows(name)
    actual = tree_rows(root)
    normalized_expected = [
        {
            "relative_path": row["relative_path"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
        for row in expected
    ]
    if actual != normalized_expected:
        raise RuntimeError(f"Cleanup source changed since inventory: {root}")


def cleanup() -> None:
    report_path = MANIFEST_ROOT / "verification_report.json"
    if not report_path.is_file() or json.loads(report_path.read_text()).get("status") != "verified":
        raise RuntimeError("A successful destination verification is required before cleanup")
    cleanup_targets = [
        RAW_DUPLICATES["4DCT"],
        RAW_DUPLICATES["COPD"],
        MASK_STAGE / "4dct",
        LUNGCT_ROOT / "manifests",
    ]
    handles = [line for path in cleanup_targets if path.exists() for line in open_handles(path)]
    if handles:
        raise RuntimeError("Open handles under the old LungCT tree block cleanup:\n" + "\n".join(handles))

    verify_cleanup_source("duplicate_top_4DCT", RAW_DUPLICATES["4DCT"])
    verify_cleanup_source("duplicate_top_COPD", RAW_DUPLICATES["COPD"])
    verify_cleanup_source("duplicate_staged_masks_4DCT", MASK_STAGE / "4dct")

    removed: list[str] = []
    remove_tree(RAW_DUPLICATES["4DCT"], removed)
    remove_tree(RAW_DUPLICATES["COPD"], removed)
    remove_tree(MASK_STAGE / "4dct", removed)
    remove_tree(LUNGCT_ROOT / "manifests", removed)

    for path in sorted(
        [item for item in LUNGCT_ROOT.rglob("*") if item.is_dir() and DIRLAB_ROOT not in item.parents],
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
            removed.append(str(path))
        except OSError:
            pass

    unexpected = [path.name for path in LUNGCT_ROOT.iterdir() if path != DIRLAB_ROOT]
    if unexpected:
        raise RuntimeError(f"Unexpected legacy entries remain: {unexpected}")
    (MANIFEST_ROOT / "cleanup_report.json").write_text(
        json.dumps({"status": "complete", "removed": removed}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["inventory", "move", "verify", "cleanup"])
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == "__main__":
    main()
