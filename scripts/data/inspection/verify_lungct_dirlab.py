#!/usr/bin/env python3
"""Audit the canonical DIR-Lab raw and pTVReg_iso1mm datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common.lungct import (  # noqa: E402
    load_raw_4dct_image,
    load_raw_copd_image,
    resolve_4dct_case,
    resolve_copd_case,
)


def load_experiment_common():
    path = REPO_ROOT / "scripts" / "experiments" / "lungct_no_sweep_repro" / "common.py"
    spec = importlib.util.spec_from_file_location("dirlab_verification_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_finite(label: str, value: np.ndarray) -> None:
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(f"Non-finite values found in {label}")


def processed_cases_pass(rows: list[dict[str, object]]) -> bool:
    return all(bool(row.get("verification_pass")) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    raw_rows = []
    for dataset, resolver in (("4dct", resolve_4dct_case), ("copd", resolve_copd_case)):
        for case_id in range(1, 11):
            case = resolver(case_id, args.data_root)
            if case.status != "complete":
                raise RuntimeError(f"Raw {dataset} case {case_id} is {case.status}: {case.notes}")
            raw_rows.append({"dataset": dataset, "case_id": case_id, "root": case.root, "status": case.status})

    fourdct_case = resolve_4dct_case(1, args.data_root)
    copd_case = resolve_copd_case(1, args.data_root)
    raw_4dct = load_raw_4dct_image(fourdct_case.image_t00, fourdct_case.shape_zyx)
    raw_copd = load_raw_copd_image(copd_case.image_ibh, copd_case.shape_zyx)
    require_finite("representative raw 4DCT", raw_4dct)
    require_finite("representative raw COPD", raw_copd)

    exp_common = load_experiment_common()
    processed_root = args.data_root / "processed" / "pTVReg_iso1mm"
    processed_rows = []
    for dataset in ("4dct", "copd"):
        for case_id in range(1, 11):
            case = exp_common.load_case(processed_root, case_id, dataset=dataset)
            for name in (
                "fixed",
                "moving",
                "fixed_mask",
                "moving_mask",
                "landmarks_fixed",
                "landmarks_moving",
                "landmarks_fixed_crop",
                "landmarks_moving_crop",
            ):
                require_finite(f"{dataset} case {case_id} {name}", getattr(case, name))
            processed_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "case_dir": str(case.case_dir),
                    "shape_zyx": list(case.shape_zyx),
                    "n_landmarks": int(case.landmarks_fixed.shape[0]),
                    "fixed_mask_voxels": int(case.fixed_mask.sum()),
                    "moving_mask_voxels": int(case.moving_mask.sum()),
                    "verification_pass": exp_common._audit_verification_pass(case.preprocess_audit),
                }
            )

    all_processed_verification_pass = processed_cases_pass(processed_rows)
    report = {
        "status": "passed" if all_processed_verification_pass else "failed",
        "data_root": str(args.data_root),
        "raw_cases": raw_rows,
        "processed_cases": processed_rows,
        "raw_case_count": len(raw_rows),
        "processed_case_count": len(processed_rows),
        "all_arrays_finite": True,
        "all_processed_verification_pass": all_processed_verification_pass,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "raw_case_count",
                    "processed_case_count",
                    "all_arrays_finite",
                    "all_processed_verification_pass",
                )
            },
            indent=2,
        )
    )
    if not all_processed_verification_pass:
        raise RuntimeError("One or more processed LungCT cases failed preprocessing verification")


if __name__ == "__main__":
    main()
