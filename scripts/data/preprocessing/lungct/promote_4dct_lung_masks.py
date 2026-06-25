#!/usr/bin/env python
"""Verify and optionally promote staged paired 4DCT lung masks."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common.volume import parse_case_ids, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--compat-root", type=Path, required=True)
    parser.add_argument("--case-ids", type=str, default="1-10")
    parser.add_argument("--modelname", type=str, default="R231")
    parser.add_argument("--require-fixed-match", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually promote after verification. Default is verify-only.")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def mask_paths(root: Path, case_id: int, phase: str, modelname: str) -> Dict[str, Path]:
    case_dir = root / "4dct" / f"case{case_id:02d}"
    prefix = f"case{case_id:02d}_T{phase}_{modelname}"
    return {
        "binary": case_dir / f"{prefix}_binary_zyx.npy",
        "labels": case_dir / f"{prefix}_labels_zyx.npy",
        "label_nii": case_dir / f"{prefix}_labels.nii.gz",
        "summary": case_dir / f"{prefix}_summary.json",
    }


def compare_arrays(path_a: Path, path_b: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"path_a": str(path_a), "path_b": str(path_b)}
    if not path_a.exists() or not path_b.exists():
        out["match"] = False
        out["reason"] = "missing"
        out["path_a_exists"] = path_a.exists()
        out["path_b_exists"] = path_b.exists()
        return out
    arr_a = np.load(path_a)
    arr_b = np.load(path_b)
    out["shape_a"] = list(arr_a.shape)
    out["shape_b"] = list(arr_b.shape)
    if arr_a.shape != arr_b.shape:
        out["match"] = False
        out["reason"] = "shape_mismatch"
        return out
    diff = arr_a != arr_b
    out["different_voxels"] = int(diff.sum())
    out["match"] = bool(not np.any(diff))
    if np.any(diff):
        out["reason"] = "value_mismatch"
        out["max_abs_diff"] = float(np.max(np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16))))
    else:
        out["reason"] = "exact_match"
    return out


def verify_case(case_id: int, args: argparse.Namespace) -> Dict[str, object]:
    staged_t00 = mask_paths(args.stage_root, case_id, "00", args.modelname)
    staged_t50 = mask_paths(args.stage_root, case_id, "50", args.modelname)
    processed_t00 = mask_paths(args.processed_root, case_id, "00", args.modelname)
    compat_t00 = mask_paths(args.compat_root, case_id, "00", args.modelname)
    required = list(staged_t00.values()) + list(staged_t50.values())
    missing_staged = [str(p) for p in required if not p.exists()]
    row: Dict[str, object] = {
        "case_id": case_id,
        "missing_staged": missing_staged,
        "processed_vs_compat_binary": compare_arrays(processed_t00["binary"], compat_t00["binary"]),
        "processed_vs_compat_labels": compare_arrays(processed_t00["labels"], compat_t00["labels"]),
        "stage_vs_processed_binary": compare_arrays(staged_t00["binary"], processed_t00["binary"]),
        "stage_vs_processed_labels": compare_arrays(staged_t00["labels"], processed_t00["labels"]),
        "stage_vs_compat_binary": compare_arrays(staged_t00["binary"], compat_t00["binary"]),
        "stage_vs_compat_labels": compare_arrays(staged_t00["labels"], compat_t00["labels"]),
    }
    checks = [
        row["processed_vs_compat_binary"],
        row["processed_vs_compat_labels"],
        row["stage_vs_processed_binary"],
        row["stage_vs_processed_labels"],
        row["stage_vs_compat_binary"],
        row["stage_vs_compat_labels"],
    ]
    row["fixed_exact_match"] = (not missing_staged) and all(bool(check["match"]) for check in checks)  # type: ignore[index]
    return row


def promote_tree(stage_root: Path, target_root: Path, timestamp: str) -> Dict[str, str]:
    source = stage_root / "4dct"
    target = target_root / "4dct"
    backup = target_root / f"4dct_backup_{timestamp}"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if backup.exists():
            raise FileExistsError(backup)
        shutil.move(str(target), str(backup))
    shutil.copytree(source, target)
    return {"source": str(source), "target": str(target), "backup": str(backup)}


def main() -> None:
    args = parse_args()
    case_ids = parse_case_ids(args.case_ids)
    rows = [verify_case(case_id, args) for case_id in case_ids]
    all_match = all(bool(row["fixed_exact_match"]) for row in rows)
    report: Dict[str, object] = {
        "stage_root": str(args.stage_root),
        "processed_root": str(args.processed_root),
        "compat_root": str(args.compat_root),
        "case_ids": case_ids,
        "require_fixed_match": args.require_fixed_match,
        "execute": args.execute,
        "all_fixed_exact_match": all_match,
        "rows": rows,
    }
    if args.require_fixed_match and not all_match:
        report["promotion_status"] = "blocked_fixed_mismatch"
    elif args.execute:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report["promoted"] = [
            promote_tree(args.stage_root, args.processed_root, timestamp),
            promote_tree(args.stage_root, args.compat_root, timestamp),
        ]
        report["promotion_status"] = "promoted"
    else:
        report["promotion_status"] = "verified_only"

    report_path = args.report or (args.stage_root / "4dct_promotion_verification_report.json")
    write_json(report_path, report)
    print(f"Fixed masks exact match: {all_match}")
    print(f"Promotion status: {report['promotion_status']}")
    print(f"Report: {report_path}")
    if args.require_fixed_match and not all_match:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
