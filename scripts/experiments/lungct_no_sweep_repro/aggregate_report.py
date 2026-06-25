#!/usr/bin/env python
"""Aggregate LungCT no-sweep reproduction summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import common as c
from run_one import METHODS


REFERENCE_ROWS = {
    "IDIR iso-1mm + lung-mask sampled NCC": {
        "snap_tre": 1.132,
        "sub_tre": 1.212,
        "fg_logj_std": 0.139823,
        "fg_jle0_pct": 0.0,
        "be_fg": 0.004565,
        "diff_fg": 0.013311,
        "note": "legacy task4 pTV-prep IDIR row",
    },
    "DirectCP 32-16-8-4 + pTV LCC + isoTV full": {
        "snap_tre": 1.040,
        "sub_tre": 1.134,
        "fg_logj_std": 0.147097,
        "fg_jle0_pct": 0.0057,
        "be_fg": 0.011850,
        "diff_fg": 0.012041,
        "note": "legacy task4 300,500,800,1200 schedule",
    },
    "DirectCP 32-16-8-4 + pTV LCC + isoTV reduced lr1e-3": {
        "snap_tre": 1.045,
        "sub_tre": 1.139,
        "fg_logj_std": 0.1468,
        "fg_jle0_pct": 0.0079,
        "be_fg": 0.0119,
        "diff_fg": 0.0121,
        "note": "legacy task4 200,200,200,400 schedule",
    },
    "DirectCP 32-16-8-4 + pTV LCC + isoTV reduced lr1e-2": {
        "snap_tre": 1.035,
        "sub_tre": 1.135,
        "fg_logj_std": 0.1498,
        "fg_jle0_pct": 0.0075,
        "be_fg": 0.0127,
        "diff_fg": 0.0125,
        "note": "legacy task4 200,200,200,400 schedule",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=c.DEFAULT_REVIEW_ROOT)
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    return parser.parse_args()


def flatten_summary(path: Path, payload: Dict[str, object]) -> Dict[str, object]:
    reg = payload.get("regularity_fg_registration_crop", {})
    snap_1mm = payload.get("tre_snap_to_voxel", {})
    sub_1mm = payload.get("tre_subvoxel", {})
    identity_1mm = payload.get("identity_tre", {})
    training = payload.get("training", {})
    native_tre = native_crop_tre_from_summary(path, payload)
    snap_native = native_tre.get("snap", {})
    sub_native = native_tre.get("subvoxel", {})
    identity_native = native_tre.get("identity", {})
    if not isinstance(reg, dict):
        reg = {}
    if not isinstance(snap_1mm, dict):
        snap_1mm = {}
    if not isinstance(sub_1mm, dict):
        sub_1mm = {}
    if not isinstance(identity_1mm, dict):
        identity_1mm = {}
    if not isinstance(training, dict):
        training = {}
    return {
        "method": payload.get("method"),
        "case_id": payload.get("case_id"),
        "status": payload.get("status"),
        "summary_path": str(path),
        "native_crop_identity_tre_mean_mm": identity_native.get("tre_mean_mm"),
        "native_crop_snap_tre_mean_mm": snap_native.get("tre_mean_mm"),
        "native_crop_snap_tre_std_mm": snap_native.get("tre_std_mm"),
        "native_crop_snap_tre_median_mm": snap_native.get("tre_median_mm"),
        "native_crop_sub_tre_mean_mm": sub_native.get("tre_mean_mm"),
        "native_crop_sub_tre_std_mm": sub_native.get("tre_std_mm"),
        "native_crop_sub_tre_median_mm": sub_native.get("tre_median_mm"),
        "reg1mm_identity_tre_mean_mm": identity_1mm.get("tre_mean_mm"),
        "reg1mm_snap_tre_mean_mm": snap_1mm.get("tre_mean_mm"),
        "reg1mm_snap_tre_std_mm": snap_1mm.get("tre_std_mm"),
        "reg1mm_snap_tre_median_mm": snap_1mm.get("tre_median_mm"),
        "reg1mm_sub_tre_mean_mm": sub_1mm.get("tre_mean_mm"),
        "reg1mm_sub_tre_std_mm": sub_1mm.get("tre_std_mm"),
        "reg1mm_sub_tre_median_mm": sub_1mm.get("tre_median_mm"),
        "fg_log_jdet_std": reg.get("log_jdet_std_fg"),
        "fg_jdet_le0_pct": reg.get("jdet_le0_pct_fg"),
        "fg_bending_energy": reg.get("bending_energy_fg"),
        "fg_diffusion_l2": reg.get("diffusion_l2_fg"),
        "fg_disp_mag_mean_vox": reg.get("disp_mag_mean_fg_vox"),
        "fg_disp_mag_p95_vox": reg.get("disp_mag_p95_fg_vox"),
        "fg_disp_mag_max_vox": reg.get("disp_mag_max_fg_vox"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "peak_mem_gb": payload.get("peak_mem_gb"),
        "protocol_tag": payload.get("protocol_tag"),
        "seed": payload.get("seed"),
        "epochs": training.get("epochs"),
        "stage_epochs": ",".join(str(v) for v in training.get("stage_epochs", []))
        if isinstance(training.get("stage_epochs"), list)
        else "",
        "lr": training.get("lr"),
        "stage_lr": ",".join(str(v) for v in training.get("stage_lr", []))
        if isinstance(training.get("stage_lr"), list)
        else "",
        "ptv_lcc_sigma_mm": training.get("ptv_lcc_sigma_mm"),
        "ptv_isotv_weight": training.get("ptv_isotv_weight"),
        "ptv_csqrt": training.get("ptv_csqrt"),
    }


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def mean_std(values: Sequence[object]) -> tuple[float, float]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def existing_path(value: object) -> Path | None:
    if not value:
        return None
    text = str(value)
    candidates = [Path(text)]
    if text.startswith("/database/"):
        candidates.append(Path("/data/hengjie") / text.removeprefix("/database/"))
    if text.startswith("/data/hengjie/"):
        candidates.append(Path("/database") / text.removeprefix("/data/hengjie/"))
    if text.startswith("/homebase/"):
        candidates.append(Path("/home/hengjie") / text.removeprefix("/homebase/"))
    if text.startswith("/home/hengjie/"):
        candidates.append(Path("/homebase") / text.removeprefix("/home/hengjie/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resize_disp_phys_to_shape(disp_phys_zyx: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(disp_phys_zyx, dtype=np.float32)).permute(3, 0, 1, 2)[None]
    resized = F.interpolate(tensor, size=tuple(int(v) for v in shape_zyx), mode="trilinear", align_corners=True)
    return resized[0].permute(1, 2, 3, 0).numpy()


def tre_from_dense_disp(
    disp_vox_zyx: np.ndarray,
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    sample_mode: str,
    snap_output: bool,
) -> Dict[str, object]:
    if sample_mode == "nearest":
        disp_at_points = c.sample_dense_disp_nearest(disp_vox_zyx, fixed_zyx)
    elif sample_mode == "trilinear":
        disp_at_points = c.sample_dense_disp_trilinear(disp_vox_zyx, fixed_zyx)
    else:
        raise ValueError(f"Unknown TRE sample mode: {sample_mode}")
    predicted = np.asarray(fixed_zyx, dtype=np.float64) + disp_at_points
    if snap_output:
        predicted = np.rint(predicted)
    out = c.compute_tre_mm(predicted, np.asarray(moving_zyx, dtype=np.float64), spacing_zyx)
    out["sample_mode"] = sample_mode
    out["snap_output"] = bool(snap_output)
    out["dense_disp_units"] = "native_crop_voxel"
    return out


def native_crop_tre_from_summary(summary_path: Path, payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    disp_path = summary_path.parent / "disp_vox_reg1mm_zyx.npy"
    if not disp_path.exists():
        return {}

    audit = payload.get("preprocess_audit")
    if not isinstance(audit, dict):
        return {}
    array_paths = audit.get("array_paths")
    if not isinstance(array_paths, dict):
        return {}

    fixed_path = existing_path(array_paths.get("landmarks_fixed_T00_crop_zyx"))
    moving_path = existing_path(array_paths.get("landmarks_moving_T50_crop_zyx"))
    if fixed_path is None or moving_path is None:
        return {}

    crop_shape = audit.get("crop_shape_zyx")
    spacing = audit.get("native_spacing_zyx_mm") or c.FOURDCT_SPACING_ZYX.get(int(payload.get("case_id") or 0))
    if crop_shape is None or spacing is None:
        return {}

    disp_phys_reg = np.load(disp_path).astype(np.float32, copy=False)
    disp_phys_crop = resize_disp_phys_to_shape(disp_phys_reg, crop_shape)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    disp_vox_crop = disp_phys_crop / spacing_arr.reshape(1, 1, 1, 3)
    fixed_crop = np.load(fixed_path).astype(np.float64, copy=False)
    moving_crop = np.load(moving_path).astype(np.float64, copy=False)
    return {
        "snap": tre_from_dense_disp(disp_vox_crop, fixed_crop, moving_crop, spacing_arr, "nearest", True),
        "subvoxel": tre_from_dense_disp(disp_vox_crop, fixed_crop, moving_crop, spacing_arr, "trilinear", False),
        "identity": c.compute_tre_mm(fixed_crop, moving_crop, spacing_arr),
    }


def fmt(value: object, digits: int = 3) -> str:
    f = safe_float(value)
    if math.isnan(f):
        return "n/a"
    return f"{f:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    methods = [m for m in args.methods.replace(" ", "").split(",") if m]
    rows: List[Dict[str, object]] = []
    for method in methods:
        for summary_path in sorted((args.out_root / method).glob("*/summary.json")):
            payload = json.loads(summary_path.read_text())
            if payload.get("status") == "completed":
                rows.append(flatten_summary(summary_path, payload))

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs.csv", rows)
    c.write_json(args.review_root / "summary_runs.json", rows)

    aggregate_rows: List[List[object]] = []
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        native_snap_mean, native_snap_std = mean_std([row.get("native_crop_snap_tre_mean_mm") for row in method_rows])
        native_sub_mean, native_sub_std = mean_std([row.get("native_crop_sub_tre_mean_mm") for row in method_rows])
        reg1mm_snap_mean, reg1mm_snap_std = mean_std([row.get("reg1mm_snap_tre_mean_mm") for row in method_rows])
        reg1mm_sub_mean, reg1mm_sub_std = mean_std([row.get("reg1mm_sub_tre_mean_mm") for row in method_rows])
        logj_mean, _ = mean_std([row.get("fg_log_jdet_std") for row in method_rows])
        jle0_mean, _ = mean_std([row.get("fg_jdet_le0_pct") for row in method_rows])
        be_mean, _ = mean_std([row.get("fg_bending_energy") for row in method_rows])
        diff_mean, _ = mean_std([row.get("fg_diffusion_l2") for row in method_rows])
        elapsed_mean, _ = mean_std([row.get("elapsed_sec") for row in method_rows])
        aggregate_rows.append(
            [
                method,
                len(method_rows),
                f"{fmt(native_snap_mean)} +/- {fmt(native_snap_std)}",
                f"{fmt(native_sub_mean)} +/- {fmt(native_sub_std)}",
                f"{fmt(reg1mm_snap_mean)} +/- {fmt(reg1mm_snap_std)}",
                f"{fmt(reg1mm_sub_mean)} +/- {fmt(reg1mm_sub_std)}",
                fmt(logj_mean, 6),
                fmt(jle0_mean, 4),
                fmt(be_mean, 6),
                fmt(diff_mean, 6),
                fmt(elapsed_mean, 1),
            ]
        )

    per_case_rows = [
        [
            row.get("case_id"),
            row.get("method"),
            fmt(row.get("native_crop_identity_tre_mean_mm")),
            fmt(row.get("native_crop_snap_tre_mean_mm")),
            fmt(row.get("native_crop_sub_tre_mean_mm")),
            fmt(row.get("reg1mm_identity_tre_mean_mm")),
            fmt(row.get("reg1mm_snap_tre_mean_mm")),
            fmt(row.get("reg1mm_sub_tre_mean_mm")),
            fmt(row.get("fg_log_jdet_std"), 6),
            fmt(row.get("fg_jdet_le0_pct"), 4),
            fmt(row.get("fg_bending_energy"), 6),
            fmt(row.get("fg_diffusion_l2"), 6),
        ]
        for row in sorted(rows, key=lambda r: (str(r.get("method")), int(r.get("case_id") or 0)))
    ]

    ref_rows = [
        [
            name,
            fmt(ref["snap_tre"]),
            fmt(ref["sub_tre"]),
            fmt(ref["fg_logj_std"], 6),
            fmt(ref["fg_jle0_pct"], 4),
            fmt(ref["be_fg"], 6),
            fmt(ref["diff_fg"], 6),
            ref["note"],
        ]
        for name, ref in REFERENCE_ROWS.items()
    ]

    report = "\n".join(
        [
            "# LungCT pTV-Prep No-Sweep Reproduction Report",
            "",
            "Generated from completed `summary.json` files under:",
            "",
            f"`{args.out_root}`",
            "",
            "## Mean Results",
            "",
            markdown_table(
                [
                    "Method",
                    "Cases",
                    "Original Snap TRE",
                    "Original Sub TRE",
                    "1mm Snap TRE",
                    "1mm Sub TRE",
                    "FG logJ std",
                    "FG J<=0%",
                    "BE fg",
                    "Diff fg",
                    "Time sec",
                ],
                aggregate_rows,
            ),
            "",
            "Original TRE means native/crop-space TRE: the saved 1mm displacement field is resized back to the corrected pTVReg native landmark crop, converted to native voxel units, and evaluated with native spacing. This is the pTVReg-comparable headline convention. 1mm TRE is evaluated directly in the cropped 1mm registration grid.",
            "",
            "## Legacy Reference Rows",
            "",
            "Reference rows are comparison anchors only. Their TRE columns are old task4 native/crop snap/sub values. The new 4-stage run uses stage epochs `200,400,800,1000`, so it is not byte-identical to the legacy full or reduced schedules, and the canonical regenerated crop uses the corrected pTVReg crop-origin convention.",
            "",
            markdown_table(
                ["Reference", "Original Snap TRE", "Original Sub TRE", "FG logJ std", "FG J<=0%", "BE fg", "Diff fg", "Note"],
                ref_rows,
            ),
            "",
            "## Per-Case Rows",
            "",
            markdown_table(
                [
                    "Case",
                    "Method",
                    "Original Identity TRE",
                    "Original Snap TRE",
                    "Original Sub TRE",
                    "1mm Identity TRE",
                    "1mm Snap TRE",
                    "1mm Sub TRE",
                    "FG logJ std",
                    "FG J<=0%",
                    "BE fg",
                    "Diff fg",
                ],
                per_case_rows,
            ),
            "",
            "## Protocol Notes",
            "",
            "- Inputs are canonical regenerated pTVReg-style landmark crops resampled to 1mm.",
            "- pTVReg's own DIR examples report snap-to-voxel TRE after resizing/carrying the displacement back to the original crop/full native domain; see `external/pTVreg/examples_ptv/DIR_test_all.m`, `external/pTVreg/mutils/My/DIR_helpers/DIR_movepoints.m`, and `external/pTVreg/DIRandCOPD.md`.",
            "- In the 1mm registration grid, landmarks are fractional after crop/resampling; snap TRE can therefore be larger than subvoxel TRE. In original/native-crop space, integer landmark targets usually make snap TRE smaller than subvoxel TRE.",
            "- IDIR uses the fixed T00 lung mask for sampled-coordinate training.",
            "- SINR/DirectCP pTV methods use border-masked pTV LCC plus isoTV, not lung-mask-weighted loss.",
            "- All foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm grid.",
        ]
    )
    (args.review_root / "review_report.md").write_text(report + "\n")
    print(
        json.dumps(
            {
                "completed_runs": len(rows),
                "review_root": str(args.review_root),
                "summary_csv": str(args.review_root / "summary_runs.csv"),
                "report": str(args.review_root / "review_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
