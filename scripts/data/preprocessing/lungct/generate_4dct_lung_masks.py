#!/usr/bin/env python
"""Generate paired fixed/moving 4DCT lung masks with lungmask R231."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repo root.")


REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common.lungct import (  # noqa: E402
    landmark_mask_fraction,
    landmarks_xyz_to_zyx,
    load_landmarks_xyz,
    load_raw_4dct_image,
    phase_image_path,
    phase_landmark_path,
    resolve_4dct_case,
)
from common.volume import make_sitk_image, parse_case_ids, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", type=str, default="1-10")
    parser.add_argument("--phases", nargs="+", choices=["00", "50"], default=["00", "50"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--modelname", type=str, default="R231")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--hu-offset", type=float, default=-1024.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_mask_outputs(
    case_id: int,
    phase: str,
    modelname: str,
    stage_root: Path,
    mask_labels: np.ndarray,
    spacing_zyx,
    overwrite: bool,
) -> Dict[str, str]:
    out_dir = stage_root / "4dct" / f"case{case_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_npy = out_dir / f"case{case_id:02d}_T{phase}_{modelname}_labels_zyx.npy"
    binary_npy = out_dir / f"case{case_id:02d}_T{phase}_{modelname}_binary_zyx.npy"
    label_nii = out_dir / f"case{case_id:02d}_T{phase}_{modelname}_labels.nii.gz"
    summary_path = out_dir / f"case{case_id:02d}_T{phase}_{modelname}_summary.json"
    targets = [label_npy, binary_npy, label_nii, summary_path]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing mask outputs: " + ", ".join(existing))

    binary = (mask_labels > 0).astype(np.uint8)
    np.save(label_npy, mask_labels.astype(np.uint8, copy=False))
    np.save(binary_npy, binary)
    sitk.WriteImage(make_sitk_image(mask_labels.astype(np.uint8, copy=False), spacing_zyx), str(label_nii))
    return {
        "label_npy": str(label_npy),
        "binary_npy": str(binary_npy),
        "label_nii": str(label_nii),
        "summary_path": str(summary_path),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        from lungmask import LMInferer
    except Exception as exc:
        raise SystemExit("lungmask is not installed in this environment.") from exc

    inferer = LMInferer(modelname=args.modelname, force_cpu=args.device == "cpu")
    case_ids = parse_case_ids(args.case_ids)
    summaries: List[Dict[str, object]] = []

    for case_id in case_ids:
        case = resolve_4dct_case(case_id, args.data_root)
        if case.status != "complete":
            raise SystemExit(f"Case {case_id} is {case.status}; cannot generate mask. Notes: {case.notes}")
        for phase in args.phases:
            image_path = phase_image_path(case, phase)
            landmark_path = phase_landmark_path(case, phase)
            image = load_raw_4dct_image(image_path, case.shape_zyx)
            image_for_mask = image + args.hu_offset
            mask_labels = inferer.apply(make_sitk_image(image_for_mask.astype(np.int16), case.spacing_zyx)).astype(np.uint8)
            binary = mask_labels > 0
            paths = write_mask_outputs(
                case_id,
                phase,
                args.modelname,
                args.stage_root,
                mask_labels,
                case.spacing_zyx,
                args.overwrite,
            )
            landmarks = landmarks_xyz_to_zyx(load_landmarks_xyz(landmark_path))
            summary: Dict[str, object] = {
                "dataset": "4dct",
                "case_id": case_id,
                "phase": f"T{phase}",
                "modelname": args.modelname,
                "hu_offset": args.hu_offset,
                "data_root": str(args.data_root),
                "stage_root": str(args.stage_root),
                "image_path": str(image_path),
                "landmark_path": str(landmark_path),
                "shape_zyx": list(case.shape_zyx),
                "spacing_zyx": list(case.spacing_zyx),
                "foreground_voxels": int(binary.sum()),
                "foreground_fraction": float(binary.mean()),
                "unique_labels": [int(v) for v in np.unique(mask_labels)],
                "landmark_inside_fraction": landmark_mask_fraction(binary, landmarks),
            }
            summary.update(paths)
            write_json(Path(paths["summary_path"]), summary)
            summaries.append(summary)
            print(
                f"case{case_id:02d} T{phase}: voxels={summary['foreground_voxels']} "
                f"fraction={summary['foreground_fraction']:.6f}"
            )

    write_json(args.stage_root / "4dct_generation_summary.json", {"rows": summaries})
    print(f"Wrote {len(summaries)} masks to {args.stage_root}")


if __name__ == "__main__":
    main()
