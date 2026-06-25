#!/usr/bin/env python
"""Visual audit for paired 4DCT LungCT fixed/moving lung masks."""

from __future__ import annotations

import argparse
import csv
import sys
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

from common.lungct import load_raw_4dct_image, mask_root_case_dir, resolve_4dct_case  # noqa: E402
from common.volume import mask_stats, overlap_stats, parse_case_ids, save_lungct_mask_figure, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--previous-mask-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-ids", type=str, default="1-10")
    parser.add_argument("--modelname", type=str, default="R231")
    return parser.parse_args()


def load_mask(mask_root: Path, case_id: int, phase: str, modelname: str) -> np.ndarray:
    path = mask_root_case_dir(mask_root, case_id) / f"case{case_id:02d}_T{phase}_{modelname}_binary_zyx.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(bool)


def audit_case(case_id: int, args: argparse.Namespace, fig_dir: Path) -> Dict[str, object]:
    case = resolve_4dct_case(case_id, args.data_root)
    if case.status != "complete":
        raise ValueError(f"Case {case_id} is {case.status}; notes={case.notes}")
    fixed = load_raw_4dct_image(case.image_t00, case.shape_zyx)
    moving = load_raw_4dct_image(case.image_t50, case.shape_zyx)
    fixed_mask = load_mask(args.mask_root, case_id, "00", args.modelname)
    moving_mask = load_mask(args.mask_root, case_id, "50", args.modelname)
    previous_fixed = None
    previous_match = None
    previous_overlap = None
    if args.previous_mask_root is not None:
        previous_path = mask_root_case_dir(args.previous_mask_root, case_id) / (
            f"case{case_id:02d}_T00_{args.modelname}_binary_zyx.npy"
        )
        if previous_path.exists():
            previous_fixed = np.load(previous_path).astype(bool)
            previous_match = bool(np.array_equal(previous_fixed, fixed_mask))
            previous_overlap = overlap_stats(previous_fixed, fixed_mask, "previous_t00", "generated_t00")

    fig_path = fig_dir / f"4dct_case{case_id:02d}_t00_t50_lungmask_coronal.png"
    save_lungct_mask_figure(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        case_id,
        fig_path,
        previous_fixed_mask=previous_fixed,
        spacing_zyx=case.spacing_zyx,
    )
    return {
        "case_id": case_id,
        "shape_zyx": list(case.shape_zyx),
        "spacing_zyx": list(case.spacing_zyx),
        "fixed_image": case.image_t00,
        "moving_image": case.image_t50,
        "fixed_mask": mask_stats(fixed_mask),
        "moving_mask": mask_stats(moving_mask),
        "previous_fixed_exact_match": previous_match,
        "previous_overlap": previous_overlap,
        "figure": str(fig_path),
    }


def flatten_row(row: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "case_id": row["case_id"],
        "shape_zyx": row["shape_zyx"],
        "spacing_zyx": row["spacing_zyx"],
        "previous_fixed_exact_match": row["previous_fixed_exact_match"],
        "figure": row["figure"],
    }
    for prefix in ("fixed_mask", "moving_mask"):
        stats = row[prefix]
        assert isinstance(stats, dict)
        out[f"{prefix}_voxels"] = stats["voxels"]
        out[f"{prefix}_fraction"] = stats["fraction"]
        out[f"{prefix}_bbox_min_zyx"] = stats["min_zyx"]
        out[f"{prefix}_bbox_max_zyx"] = stats["max_zyx"]
    overlap = row["previous_overlap"]
    if isinstance(overlap, dict):
        out.update({f"previous_{k}": v for k, v in overlap.items() if isinstance(v, (int, float, bool))})
    return out


def main() -> None:
    args = parse_args()
    case_ids = parse_case_ids(args.case_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    rows = [audit_case(case_id, args, fig_dir) for case_id in case_ids]
    flat_rows = [flatten_row(row) for row in rows]
    write_json(args.out_dir / "lungct_4dct_mask_visual_audit_summary.json", {"rows": rows})
    csv_path = args.out_dir / "lungct_4dct_mask_visual_audit_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {len(rows)} LungCT mask audits to {args.out_dir}")


if __name__ == "__main__":
    main()
