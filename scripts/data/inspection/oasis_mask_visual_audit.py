#!/usr/bin/env python
"""Visual audit for the canonical OASIS image-positive foreground mask."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common.volume import (  # noqa: E402
    load_sitk_zyx_with_metadata,
    mask_stats,
    overlap_stats,
    save_oasis_mask_figure,
    write_json,
)


PAIR_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_pairs(path: Path, num_pairs: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PAIR_RE.match(line)
        if not match:
            raise ValueError(f"Could not parse pair line in {path}: {line}")
        pairs.append((match.group(1), match.group(2)))
        if len(pairs) >= num_pairs:
            break
    if len(pairs) != num_pairs:
        raise ValueError(f"Expected {num_pairs} pairs from {path}, found {len(pairs)}")
    return pairs


def unique_ids_in_pair_order(pairs: List[Tuple[str, str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for fixed, moving in pairs:
        for subject_id in (fixed, moving):
            if subject_id not in seen:
                out.append(subject_id)
                seen.add(subject_id)
    return out


def audit_subject(subject_id: str, oasis_root: Path, fig_dir: Path) -> Dict[str, object]:
    image_path = oasis_root / "test" / f"img{subject_id}.nii.gz"
    seg_path = oasis_root / "test" / f"seg{subject_id}.nii.gz"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not seg_path.exists():
        raise FileNotFoundError(seg_path)

    image, image_meta = load_sitk_zyx_with_metadata(image_path)
    seg, seg_meta = load_sitk_zyx_with_metadata(seg_path)
    image = image.astype(np.float32, copy=False)
    if image.shape != seg.shape:
        raise ValueError(f"Image/seg shape mismatch for {subject_id}: {image.shape} vs {seg.shape}")
    for key in ("spacing_xyz", "direction"):
        if not np.allclose(image_meta[key], seg_meta[key]):
            raise ValueError(f"Image/seg {key} mismatch for {subject_id}: {image_meta[key]} vs {seg_meta[key]}")

    canonical_fg = image > 0
    seg_label_fg = seg > 0
    fig_path = fig_dir / f"oasis_{subject_id}_canonical_imagefg_vs_segfg_coronal.png"
    coronal_view = save_oasis_mask_figure(
        image,
        seg_label_fg,
        canonical_fg,
        subject_id,
        fig_path,
        direction=image_meta["direction"],
        spacing_xyz=image_meta["spacing_xyz"],
    )

    row: Dict[str, object] = {
        "subject_id": subject_id,
        "image_path": str(image_path),
        "seg_path": str(seg_path),
        "shape_zyx": list(image.shape),
        "size_xyz": image_meta["size_xyz"],
        "spacing_xyz": image_meta["spacing_xyz"],
        "origin_xyz": image_meta["origin_xyz"],
        "direction": image_meta["direction"],
        "orientation": image_meta["orientation"],
        "coronal_view": coronal_view,
        "foreground_rule": "raw image > 0",
        "seg_label_role": "label/comparison mask, not default foreground",
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "image_mean": float(np.mean(image)),
        "seg_unique_labels": [int(v) for v in np.unique(seg)],
        "figure": str(fig_path),
    }
    row["canonical_fg"] = mask_stats(canonical_fg)
    row["seg_label_fg"] = mask_stats(seg_label_fg)
    row["overlap"] = overlap_stats(seg_label_fg, canonical_fg, "seg_label_fg", "canonical_fg")
    return row


def flatten_row(row: Dict[str, object]) -> Dict[str, object]:
    flat: Dict[str, object] = {
        "subject_id": row["subject_id"],
        "shape_zyx": row["shape_zyx"],
        "orientation": row["orientation"],
        "spacing_xyz": row["spacing_xyz"],
        "foreground_rule": row["foreground_rule"],
        "image_min": row["image_min"],
        "image_max": row["image_max"],
        "image_mean": row["image_mean"],
        "figure": row["figure"],
    }
    coronal_view = row["coronal_view"]
    assert isinstance(coronal_view, dict)
    for key in (
        "fixed_array_axis",
        "fixed_index_axis_xyz",
        "fixed_index",
        "row_array_axis",
        "row_index_axis_xyz",
        "col_array_axis",
        "col_index_axis_xyz",
        "flip_rows",
        "flip_cols",
        "row_spacing",
        "col_spacing",
        "aspect",
    ):
        flat[f"coronal_{key}"] = coronal_view[key]
    for prefix in ("canonical_fg", "seg_label_fg"):
        stats = row[prefix]
        assert isinstance(stats, dict)
        flat[f"{prefix}_voxels"] = stats["voxels"]
        flat[f"{prefix}_fraction"] = stats["fraction"]
        flat[f"{prefix}_bbox_min_zyx"] = stats["min_zyx"]
        flat[f"{prefix}_bbox_max_zyx"] = stats["max_zyx"]
    overlap = row["overlap"]
    assert isinstance(overlap, dict)
    flat.update(overlap)
    return flat


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pair_file, args.num_pairs)
    subject_ids = unique_ids_in_pair_order(pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    rows = [audit_subject(subject_id, args.oasis_root, fig_dir) for subject_id in subject_ids]
    flat_rows = [flatten_row(row) for row in rows]

    write_json(
        args.out_dir / "oasis_mask_visual_audit_summary.json",
        {
            "pair_file": str(args.pair_file),
            "oasis_root": str(args.oasis_root),
            "num_pairs": args.num_pairs,
            "pairs": pairs,
            "subject_ids": subject_ids,
            "n_subjects": len(subject_ids),
            "canonical_foreground_rule": "raw image > 0",
            "segmentation_mask_role": "label/comparison mask only unless explicitly overridden",
            "rows": rows,
        },
    )
    csv_path = args.out_dir / "oasis_mask_visual_audit_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {len(rows)} subject audits to {args.out_dir}")


if __name__ == "__main__":
    main()
