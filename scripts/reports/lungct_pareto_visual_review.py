#!/usr/bin/env python
"""Utilities for the LungCT all-10 Pareto visual review notebook."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import torch


if "INRORSPLINE_REPO_ROOT" in os.environ:
    REPO_ROOT = Path(os.environ["INRORSPLINE_REPO_ROOT"]).expanduser()
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]

LUNGCT_SCRIPT_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
if str(LUNGCT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LUNGCT_SCRIPT_DIR))

import common as c  # noqa: E402
from run_case01_ptvreg_field_fit_one import (  # noqa: E402
    crop_v_xyz1_to_bounds_zyx0,
    matlab_snap_tre_from_h5,
    normalize_crop_v,
    resize_phys_crop_to_reg1mm,
)


METHOD_IDIR = "idir_be_default_lungmask"
METHOD_PTV_DIRECTCP = "direct_cp_32_16_8_4_ptv_lcc_isotv"
METHOD_FDBE = "direct_cp_32_16_8_4_global_ncc_fd_be"
METHOD_PTVREG = "ptvreg_reference"

METHOD_ORDER = [METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE, METHOD_PTVREG]
METHOD_LABELS = {
    METHOD_IDIR: "IDIR BE",
    METHOD_PTV_DIRECTCP: "DirectCP 4stage pTV-LCC+isoTV",
    METHOD_FDBE: "DirectCP 4stage global-NCC+FD-BE",
    METHOD_PTVREG: "pTVReg",
}
METHOD_COLORS = {
    METHOD_IDIR: "#0072B2",
    METHOD_PTV_DIRECTCP: "#D55E00",
    METHOD_FDBE: "#009E73",
    METHOD_PTVREG: "#111111",
}
METHOD_MARKERS = {
    METHOD_IDIR: "o",
    METHOD_PTV_DIRECTCP: "s",
    METHOD_FDBE: "^",
    METHOD_PTVREG: "*",
}

REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep"
DEFAULT_CASE_ROWS = REVIEW_ROOT / "comparison_case_rows.csv"
DEFAULT_OUTPUT_ROOT = REVIEW_ROOT / "pareto_visual_review"
DEFAULT_PTVREG_ROOT = Path(os.environ.get("PTVREG_4DCT_ROOT", "/database/inrorspline/ptvreg/4DCT"))
DEFAULT_PTVPREP_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT",
        os.environ.get(
            "INRORSPLINE_LUNGCT_PTVPREP_ROOT",
            "/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm",
        ),
    )
)
EXPECTED_CASES = list(range(1, 11))

METRIC_KEYS = [
    "original_snap_tre_mean_mm",
    "original_subvoxel_tre_mean_mm",
    "reg1mm_snap_tre_mean_mm",
    "reg1mm_subvoxel_tre_mean_mm",
    "fg_logj_std",
    "fg_jdet_le0_pct",
    "fg_bending_energy",
    "fg_diffusion_l2",
    "fg_disp_mag_mean_vox",
    "fg_disp_mag_p95_vox",
    "fg_disp_mag_max_vox",
]
PLOT_METRICS = [
    ("fg_logj_std", "FG stdLogJ", "fg_logj_std"),
    ("fg_jdet_le0_pct", "FG J <= 0 (%)", "fg_jdet_le0_pct"),
    ("fg_bending_energy", "FG bending energy", "fg_bending_energy"),
    ("fg_diffusion_l2", "FG diffusion energy", "fg_diffusion_l2"),
]


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def mean_std(values: Iterable[object]) -> Tuple[float, float]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt(value: object, digits: int = 3) -> str:
    f = safe_float(value)
    if not math.isfinite(f):
        return "n/a"
    if abs(f) >= 10000:
        return f"{f:.3g}"
    if 0 < abs(f) < 10 ** (-digits):
        return f"{f:.3g}"
    return f"{f:.{digits}f}"


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def trilinear_sample_xyz(field_xyz: np.ndarray, pts_xyz1: np.ndarray) -> np.ndarray:
    field = np.asarray(field_xyz, dtype=np.float64)
    pts0 = np.asarray(pts_xyz1, dtype=np.float64) - 1.0
    shape = np.asarray(field.shape[:3], dtype=np.float64)
    pts0 = np.clip(pts0, 0.0, shape[None, :] - 1.0)
    lo = np.floor(pts0).astype(np.int64)
    hi = np.minimum(lo + 1, np.asarray(field.shape[:3], dtype=np.int64) - 1)
    frac = pts0 - lo
    x0, y0, z0 = lo[:, 0], lo[:, 1], lo[:, 2]
    x1, y1, z1 = hi[:, 0], hi[:, 1], hi[:, 2]
    wx, wy, wz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c001 = field[x0, y0, z1]
    c110 = field[x1, y1, z0]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    return (
        c000 * (1 - wx) * (1 - wy) * (1 - wz)
        + c100 * wx * (1 - wy) * (1 - wz)
        + c010 * (1 - wx) * wy * (1 - wz)
        + c001 * (1 - wx) * (1 - wy) * wz
        + c110 * wx * wy * (1 - wz)
        + c101 * wx * (1 - wy) * wz
        + c011 * (1 - wx) * wy * wz
        + c111 * wx * wy * wz
    )


def tre_xyz_mm(pred_xyz1: np.ndarray, target_xyz1: np.ndarray, spacing_xyz: np.ndarray) -> Dict[str, object]:
    err = np.sqrt(np.sum(((np.asarray(pred_xyz1) - np.asarray(target_xyz1)) * spacing_xyz.reshape(1, 3)) ** 2, axis=1))
    return {
        "tre_mean_mm": float(err.mean()),
        "tre_std_mm": float(err.std(ddof=1)),
        "tre_median_mm": float(np.median(err)),
        "per_landmark_tre_mm": [float(v) for v in err],
    }


def load_method_case_rows(case_rows_path: Path = DEFAULT_CASE_ROWS) -> pd.DataFrame:
    rows = read_csv_dicts(case_rows_path)
    df = pd.DataFrame(rows)
    keep = df["method"].isin([METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE])
    df = df.loc[keep].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    for col in ["case_id", "reg_value", "reg_sort", *METRIC_KEYS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_one_ptvreg_case(case_id: int, ptvreg_root: Path, ptvprep_root: Path) -> Dict[str, object]:
    mat_path = Path(ptvreg_root) / f"case{case_id:02d}_T50_to_T00_deformation.mat"
    case = c.load_case(Path(ptvprep_root), int(case_id))
    required = {"Tptv_rsz", "pts_fix", "pts_mov", "spc_orig_saved", "crop_v_saved", "TRE_phys", "TREstd_case"}
    with h5py.File(mat_path, "r") as f:
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(f"{mat_path}: missing keys {missing}")
        t_h5 = f["Tptv_rsz"][:].astype(np.float32, copy=False)
        pts_fix_xyz1 = f["pts_fix"][:].T.astype(np.float64, copy=False)
        pts_mov_xyz1 = f["pts_mov"][:].T.astype(np.float64, copy=False)
        spc_xyz = f["spc_orig_saved"][:].reshape(-1).astype(np.float64, copy=False)
        crop_v_xyz1 = normalize_crop_v(f["crop_v_saved"][:])
        saved_tre = float(np.asarray(f["TRE_phys"][()]).reshape(-1)[0])
        saved_std = float(np.asarray(f["TREstd_case"][()]).reshape(-1)[0])

    if t_h5.ndim != 4 or t_h5.shape[0] != 3:
        raise ValueError(f"{mat_path}: expected Tptv_rsz shape (3,z,y,x), got {t_h5.shape}")

    snap_native = matlab_snap_tre_from_h5(t_h5, pts_fix_xyz1, pts_mov_xyz1, spc_xyz)
    snap_err = abs(float(snap_native["tre_mean_mm"]) - saved_tre)
    std_err = abs(float(snap_native["tre_std_mm"]) - saved_std)
    if snap_err > 1e-8 or std_err > 1e-8:
        raise RuntimeError(f"{mat_path}: TRE validation failed mean_err={snap_err}, std_err={std_err}")
    snap_native_median = float(np.median(np.asarray(snap_native["per_landmark_tre_mm"], dtype=np.float64)))

    crop_bounds_zyx = crop_v_xyz1_to_bounds_zyx0(crop_v_xyz1)
    audit_bounds = np.asarray(case.preprocess_audit["crop_bounds_zyx_inclusive"], dtype=np.int64)
    if not np.array_equal(crop_bounds_zyx, audit_bounds):
        raise RuntimeError(f"{mat_path}: crop_v_saved {crop_bounds_zyx.tolist()} != audit {audit_bounds.tolist()}")

    field_xyz = np.asarray(t_h5).transpose(3, 2, 1, 0)
    disp_sub_phys_xyz = trilinear_sample_xyz(field_xyz, pts_fix_xyz1)
    sub_native = tre_xyz_mm(pts_fix_xyz1 + disp_sub_phys_xyz / spc_xyz.reshape(1, 3), pts_mov_xyz1, spc_xyz)

    disp_phys_native_zyx_xyz = t_h5.transpose(1, 2, 3, 0)
    z0, z1 = crop_bounds_zyx[0]
    y0, y1 = crop_bounds_zyx[1]
    x0, x1 = crop_bounds_zyx[2]
    disp_phys_crop_zyx_xyz = disp_phys_native_zyx_xyz[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1, :]
    disp_phys_reg_zyx_xyz = resize_phys_crop_to_reg1mm(disp_phys_crop_zyx_xyz, case.shape_zyx)
    disp_vox_reg_zyx = np.stack(
        (disp_phys_reg_zyx_xyz[..., 2], disp_phys_reg_zyx_xyz[..., 1], disp_phys_reg_zyx_xyz[..., 0]),
        axis=-1,
    ).astype(np.float32)

    reg1mm_snap = c.landmark_tre_from_dense_disp(
        disp_vox_reg_zyx, case.landmarks_fixed, case.landmarks_moving, sample_mode="nearest", snap_output=True
    )
    reg1mm_sub = c.landmark_tre_from_dense_disp(
        disp_vox_reg_zyx, case.landmarks_fixed, case.landmarks_moving, sample_mode="trilinear", snap_output=False
    )
    reg = c.displacement_regularities(disp_vox_reg_zyx, foreground_mask=case.fixed_mask.astype(bool), log_base="e")

    return {
        "method": METHOD_PTVREG,
        "method_label": METHOD_LABELS[METHOD_PTVREG],
        "case_id": int(case_id),
        "source": "ptvreg_uploaded_mat",
        "reg_sort": float("nan"),
        "reg_value": float("nan"),
        "reg_label": "pTVReg",
        "ratio_to_best_tre_weight": float("nan"),
        "summary_path": str(mat_path),
        "source_saved_tre_phys_mean_mm": saved_tre,
        "source_saved_trestd_case_mm": saved_std,
        "source_recomputed_snap_tre_mean_mm": float(snap_native["tre_mean_mm"]),
        "source_recomputed_snap_tre_std_mm": float(snap_native["tre_std_mm"]),
        "source_tre_validation_abs_err": snap_err,
        "source_trestd_validation_abs_err": std_err,
        "source_disp_h5py_shape": str(tuple(int(v) for v in t_h5.shape)),
        "source_disp_h5py_layout": "component,z,y,x",
        "source_disp_component_order": "x,y,z",
        "source_displacement_units": "physical_mm",
        "source_transform_direction": "fixed_T00_to_moving_T50",
        "crop_bounds_zyx_0based_inclusive": str(crop_bounds_zyx.astype(int).tolist()),
        "original_snap_tre_mean_mm": float(snap_native["tre_mean_mm"]),
        "original_snap_tre_std_mm": float(snap_native["tre_std_mm"]),
        "original_snap_tre_median_mm": snap_native_median,
        "original_subvoxel_tre_mean_mm": float(sub_native["tre_mean_mm"]),
        "original_subvoxel_tre_std_mm": float(sub_native["tre_std_mm"]),
        "original_subvoxel_tre_median_mm": float(sub_native["tre_median_mm"]),
        "reg1mm_snap_tre_mean_mm": float(reg1mm_snap["tre_mean_mm"]),
        "reg1mm_snap_tre_std_mm": float(reg1mm_snap["tre_std_mm"]),
        "reg1mm_snap_tre_median_mm": float(reg1mm_snap["tre_median_mm"]),
        "reg1mm_subvoxel_tre_mean_mm": float(reg1mm_sub["tre_mean_mm"]),
        "reg1mm_subvoxel_tre_std_mm": float(reg1mm_sub["tre_std_mm"]),
        "reg1mm_subvoxel_tre_median_mm": float(reg1mm_sub["tre_median_mm"]),
        "fg_logj_std": float(reg["log_jdet_std_fg"]),
        "fg_jdet_le0_pct": float(reg["jdet_le0_pct_fg"]),
        "fg_bending_energy": float(reg["bending_energy_fg"]),
        "fg_diffusion_l2": float(reg["diffusion_l2_fg"]),
        "fg_disp_mag_mean_vox": float(reg["disp_mag_mean_fg_vox"]),
        "fg_disp_mag_p95_vox": float(reg["disp_mag_p95_fg_vox"]),
        "fg_disp_mag_max_vox": float(reg["disp_mag_max_fg_vox"]),
    }


def load_ptvreg_case_rows(
    ptvreg_root: Path = DEFAULT_PTVREG_ROOT,
    ptvprep_root: Path = DEFAULT_PTVPREP_ROOT,
    case_ids: Sequence[int] = EXPECTED_CASES,
) -> pd.DataFrame:
    rows = [load_one_ptvreg_case(case_id, ptvreg_root, ptvprep_root) for case_id in case_ids]
    return pd.DataFrame(rows)


def add_ratios(case_df: pd.DataFrame) -> pd.DataFrame:
    df = case_df.copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df.get("method_label", df["method"]))
    for col in ["case_id", "reg_value", "reg_sort", *METRIC_KEYS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ratio_to_best_tre_weight"] = np.nan
    for method, group in df[df["method"] != METHOD_PTVREG].groupby("method"):
        point = (
            group.groupby("reg_value", dropna=False)["original_snap_tre_mean_mm"]
            .mean()
            .reset_index()
            .sort_values("original_snap_tre_mean_mm")
            .iloc[0]
        )
        best_weight = float(point["reg_value"])
        mask = df["method"].eq(method)
        df.loc[mask, "ratio_to_best_tre_weight"] = df.loc[mask, "reg_value"] / best_weight
    return df


def aggregate_case_rows(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (method, reg_value), group in case_df.groupby(["method", "reg_value"], dropna=False, sort=False):
        method = str(method)
        row: Dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, str(group["method_label"].iloc[0])),
            "reg_value": float(reg_value) if pd.notna(reg_value) else np.nan,
            "reg_label": str(group["reg_label"].iloc[0]) if "reg_label" in group else "",
            "ratio_to_best_tre_weight": float(group["ratio_to_best_tre_weight"].iloc[0])
            if "ratio_to_best_tre_weight" in group and pd.notna(group["ratio_to_best_tre_weight"].iloc[0])
            else np.nan,
            "cases": int(group["case_id"].nunique()),
        }
        for key in METRIC_KEYS:
            mean, std = mean_std(group[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    out["sort_ratio"] = out["ratio_to_best_tre_weight"].fillna(1.0)
    return out.sort_values(["method_order", "sort_ratio", "reg_value"], na_position="last").drop(columns=["method_order", "sort_ratio"])


def screening_table(mean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mean_df.iterrows():
        rows.append(
            {
                "Method": row["method_label"],
                "Weight": "pTVReg" if row["method"] == METHOD_PTVREG else fmt(row["reg_value"], 4),
                "Ratio": "ref" if row["method"] == METHOD_PTVREG else fmt(row["ratio_to_best_tre_weight"], 3),
                "Cases": int(row["cases"]),
                "Original Snap TRE": f"{fmt(row['original_snap_tre_mean_mm_mean'])} +/- {fmt(row['original_snap_tre_mean_mm_std'])}",
                "Original Sub TRE": f"{fmt(row['original_subvoxel_tre_mean_mm_mean'])} +/- {fmt(row['original_subvoxel_tre_mean_mm_std'])}",
                "FG stdLogJ": f"{fmt(row['fg_logj_std_mean'], 4)} +/- {fmt(row['fg_logj_std_std'], 4)}",
                "FG J<=0 %": f"{fmt(row['fg_jdet_le0_pct_mean'], 4)} +/- {fmt(row['fg_jdet_le0_pct_std'], 4)}",
                "FG BE": f"{fmt(row['fg_bending_energy_mean'], 6)} +/- {fmt(row['fg_bending_energy_std'], 6)}",
                "FG diffusion": f"{fmt(row['fg_diffusion_l2_mean'], 6)} +/- {fmt(row['fg_diffusion_l2_std'], 6)}",
            }
        )
    return pd.DataFrame(rows)


def should_exclude(row: Mapping[str, object], exclude_by_ratio: Mapping[str, Sequence[float]], exclude_by_weight: Mapping[str, Sequence[float]]) -> bool:
    method = str(row["method"])
    if method == METHOD_PTVREG:
        return False
    ratio = safe_float(row.get("ratio_to_best_tre_weight"))
    weight = safe_float(row.get("reg_value"))
    for value in exclude_by_ratio.get(method, []):
        if math.isfinite(ratio) and abs(ratio - float(value)) <= 1e-9 * max(1.0, abs(float(value))):
            return True
    for value in exclude_by_weight.get(method, []):
        if math.isfinite(weight) and abs(weight - float(value)) <= 1e-9 * max(1.0, abs(float(value))):
            return True
    return False


def subset_mean_rows(
    mean_df: pd.DataFrame,
    subset_name: str,
    exclude_by_ratio: Mapping[str, Sequence[float]],
    exclude_by_weight: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    df = mean_df.copy()
    if subset_name in {"default_exclusions", "default_exclusions_no_ptvlcc"}:
        keep = [not should_exclude(row, exclude_by_ratio, exclude_by_weight) for _, row in df.iterrows()]
        df = df.loc[keep].copy()
    if subset_name == "default_exclusions_no_ptvlcc":
        df = df[df["method"] != METHOD_PTV_DIRECTCP].copy()
    return df


def plot_one_metric(mean_df: pd.DataFrame, metric_key: str, metric_label: str, title: str, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for method in METHOD_ORDER:
        method_rows = mean_df[mean_df["method"] == method].copy()
        if method_rows.empty:
            continue
        method_rows = method_rows.sort_values("ratio_to_best_tre_weight", na_position="last")
        color = METHOD_COLORS.get(method, "#444444")
        marker = METHOD_MARKERS.get(method, "o")
        xs = method_rows[f"{metric_key}_mean"].to_numpy(dtype=float)
        ys = method_rows["original_snap_tre_mean_mm_mean"].to_numpy(dtype=float)
        xerr = method_rows[f"{metric_key}_std"].to_numpy(dtype=float)
        yerr = method_rows["original_snap_tre_mean_mm_std"].to_numpy(dtype=float)
        label = METHOD_LABELS.get(method, method)
        if method != METHOD_PTVREG and len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75, zorder=2)
        for x, y, xe, ye in zip(xs, ys, xerr, yerr):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if math.isfinite(xe) and math.isfinite(ye) and (xe > 0 or ye > 0):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=max(2.0 * xe, 0.0),
                        height=max(2.0 * ye, 0.0),
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.12 if method != METHOD_PTVREG else 0.18,
                        linewidth=1.0,
                        zorder=1,
                    )
                )
        size = 120 if method == METHOD_PTVREG else 50
        ax.scatter(xs, ys, s=size, color=color, marker=marker, edgecolor="white", linewidth=0.7, label=label, zorder=3)
        for _, row in method_rows.iterrows():
            x = safe_float(row[f"{metric_key}_mean"])
            y = safe_float(row["original_snap_tre_mean_mm_mean"])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            text = "pTVReg" if method == METHOD_PTVREG else f"{row['ratio_to_best_tre_weight']:g}x"
            ax.annotate(text, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Original snap TRE (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    return fig


def make_all_plots(
    mean_df: pd.DataFrame,
    output_root: Path,
    exclude_by_ratio: Mapping[str, Sequence[float]],
    exclude_by_weight: Mapping[str, Sequence[float]],
) -> Dict[str, List[Path]]:
    saved: Dict[str, List[Path]] = {}
    for subset_name in ["all_data", "default_exclusions", "default_exclusions_no_ptvlcc"]:
        subset = subset_mean_rows(mean_df, subset_name, exclude_by_ratio, exclude_by_weight)
        saved[subset_name] = []
        for metric_key, metric_label, slug in PLOT_METRICS:
            path = Path(output_root) / "plots" / subset_name / f"original_snap_tre_vs_{slug}.png"
            title = f"{subset_name.replace('_', ' ')}: TRE vs {metric_label}"
            fig = plot_one_metric(subset, metric_key, metric_label, title, path)
            plt.show()
            plt.close(fig)
            saved[subset_name].append(path)
    return saved


def save_review_artifacts(
    case_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    table_df: pd.DataFrame,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    case_path = output_root / "combined_case_rows_with_ptvreg.csv"
    mean_path = output_root / "combined_mean_rows_with_ptvreg.csv"
    table_csv = output_root / "screening_table.csv"
    table_md = output_root / "screening_table.md"
    case_df.to_csv(case_path, index=False)
    mean_df.to_csv(mean_path, index=False)
    table_df.to_csv(table_csv, index=False)
    lines = [
        "| " + " | ".join(str(col) for col in table_df.columns) + " |",
        "| " + " | ".join("---" for _ in table_df.columns) + " |",
    ]
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in table_df.columns) + " |")
    table_md.write_text("\n".join(lines) + "\n")
    return {
        "case_rows": case_path,
        "mean_rows": mean_path,
        "screening_csv": table_csv,
        "screening_md": table_md,
    }


def validate_outputs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Dict[str, object]:
    output_root = Path(output_root)
    expected_files = [
        output_root / "combined_case_rows_with_ptvreg.csv",
        output_root / "combined_mean_rows_with_ptvreg.csv",
        output_root / "screening_table.csv",
        output_root / "screening_table.md",
    ]
    for subset_name in ["all_data", "default_exclusions", "default_exclusions_no_ptvlcc"]:
        for _, _, slug in PLOT_METRICS:
            expected_files.append(output_root / "plots" / subset_name / f"original_snap_tre_vs_{slug}.png")
    missing = [str(path) for path in expected_files if not path.exists() or path.stat().st_size <= 0]
    return {"ok": not missing, "missing_or_empty": missing, "expected_count": len(expected_files)}
