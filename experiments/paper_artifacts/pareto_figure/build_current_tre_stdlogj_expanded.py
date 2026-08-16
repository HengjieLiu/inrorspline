#!/usr/bin/env python3
"""Build the current 2x2 Pareto preview with an expanded LungCT TRE/stdLogJ panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ARTIFACT_DIR = REPO_ROOT / "experiments/paper_artifacts"
if str(PAPER_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_ARTIFACT_DIR))

import build_standardized_pareto as pareto_base  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension"
)
DEFAULT_OASIS_SOURCE = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100/"
    "oasis100_pareto_source_rows.csv"
)
DEFAULT_LUNG_SCHEDULE_SOURCE = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/"
    "v009_lungct_fig2_schedule_refresh_comparison/source_rows.csv"
)
DEFAULT_LEGACY_W0022_SOURCE = (
    REPO_ROOT
    / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/"
    "directcp_refined_mean_rows.csv"
)
DEFAULT_DUAL_STAGE2_SOURCE = (
    REPO_ROOT
    / "experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/"
    "stage2_curve_points.csv"
)
DEFAULT_DUAL_STAGE3_SOURCE = (
    REPO_ROOT
    / "experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/"
    "stage3_pl1a_refinement_curve_points.csv"
)
DEFAULT_FDBE_JOB_MANIFEST = (
    REPO_ROOT / "experiments/2026-08-02_fit_modes_population_n10/job_manifest.csv"
)
DEFAULT_FDBE_CASE01_CASE08_BRIDGE = (
    REPO_ROOT
    / "experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick/"
    "masked_vs_unmasked_long.csv"
)
DEFAULT_FDBE_ALPHA_EXTENSION_CASE_ROWS = (
    REPO_ROOT
    / "experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/"
    "case_rows.csv"
)

X_COL = "fg_logj_std_mean"
Y_COL = "original_snap_tre_mean_mm_mean"
X_STD_COL = "fg_logj_std_std"
Y_STD_COL = "original_snap_tre_mean_mm_std"

PANEL3_METHODS = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]
PANEL4_METHODS = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "dual_inr_pl1a_author_blur_native_loss",
    "sinr_cps4_global_ncc_fd_be",
    "direct_cp_cps4_global_ncc_fd_be",
    "ptvreg_reference",
]

METHOD_SPECS = {
    "idir_be_default_lungmask": {
        "label": "IDIR (IDIR loss)",
        "color": "#0072B2",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_global_ncc_fd_be": {
        "label": "MR-D-BSCP (IDIR loss)",
        "color": "#009E73",
        "marker": "^",
        "line": True,
    },
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {
        "label": "MR-D-BSCP (pTVReg loss)",
        "color": "#D55E00",
        "marker": "s",
        "line": True,
    },
    "dual_inr_pl1a_author_blur_native_loss": {
        "label": "Dual-INR PL1A",
        "color": "#7E57C2",
        "marker": "X",
        "line": True,
    },
    "sinr_cps4_global_ncc_fd_be": {
        "label": "INR-BSCP cps4 (FD-BE)",
        "color": "#CC79A7",
        "marker": "D",
        "line": True,
    },
    "direct_cp_cps4_global_ncc_fd_be": {
        "label": "D-BSCP cps4 (FD-BE)",
        "color": "#B23A48",
        "marker": "P",
        "line": True,
    },
    "ptvreg_reference": {
        "label": "pTVReg",
        "color": "#111111",
        "marker": "*",
        "line": False,
    },
}

FDBE_SINGLE_POINT_SPECS = {
    "inr_bscp": {
        "method_id": "sinr_cps4_global_ncc_fd_be",
        "display_method": "INR-BSCP cps4 (FD-BE)",
        "method_family": "sinr",
        "lr": 1e-4,
    },
    "d_bscp": {
        "method_id": "direct_cp_cps4_global_ncc_fd_be",
        "display_method": "D-BSCP cps4 (FD-BE)",
        "method_family": "directcp",
        "lr": 1e-3,
    },
}
FDBE_BASE_ALPHA = 10000.0

COMPOSITE_STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 18,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 7.4,
    "marker_size": 48,
    "ptvreg_marker_size": 120,
    "line_width": 1.55,
    "ellipse_alpha": 0.12,
    "grid_alpha": 0.25,
    "wspace": 0.20,
    "hspace": 0.30,
}


def default_data_root() -> Path:
    explicit = os.environ.get("INROSPLINE_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if explicit:
        return Path(explicit)
    user = os.environ.get("USER", "")
    return Path("/data") / user / "inrorspline" if user else Path("/data/inrorspline")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def close(value: float, target: float) -> bool:
    return math.isfinite(value) and abs(value - target) <= 1e-9 * max(1.0, abs(target))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return float("nan"), float("nan")
    return float(series.mean()), float(series.std(ddof=0))


def metric_mean(metric: str) -> str:
    return f"{metric}_mean"


def metric_std(metric: str) -> str:
    return f"{metric}_std"


def format_weight(prefix: str, value: float) -> str:
    if not math.isfinite(value):
        return prefix
    return f"{prefix}={value:g}"


def resolve_placeholders(path_text: str, *, data_root: Path, run_root: Path) -> Path:
    return Path(
        path_text.replace("${DATA_ROOT}", str(data_root)).replace("${RUN_ROOT}", str(run_root))
    )


def placeholder_summary_path(row: Mapping[str, Any]) -> str:
    if str(row.get("status")) == "reused":
        source_path = str(row.get("source_path", ""))
        return str(Path(source_path).with_name("summary.json"))
    return str(row.get("summary_path", ""))


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def headline_tre_mean(payload: Mapping[str, Any]) -> tuple[float, str]:
    preferred = payload.get("headline_tre_metric")
    for key in [preferred, "native_crop_tre_snap_to_voxel", "tre_snap_to_voxel"]:
        if isinstance(key, str) and isinstance(payload.get(key), Mapping):
            value = finite_float(nested(payload, key, "tre_mean_mm"))
            if math.isfinite(value):
                return value, key
    return float("nan"), "missing"


def load_oasis_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path).copy()
    if "dataset" not in rows:
        rows["dataset"] = "oasis100"
    if "method_id" not in rows and "method_key" in rows:
        rows["method_id"] = rows["method_key"].astype(str)
    return rows


def load_schedule_refreshed_lung_rows(path: Path, legacy_w0022_source: Path) -> pd.DataFrame:
    rows = pd.read_csv(path).copy()
    rows = rows[rows["method_id"].astype(str).isin(PANEL3_METHODS)].copy()
    direct_mask = rows["method_id"].astype(str).str.startswith("direct_cp_32_16_8_4")
    rows = rows[(~direct_mask) | (rows["schedule"].astype(str) == "2500")].copy()
    rows["source_group"] = "v009_schedule_refresh"
    rows["panel3_include"] = True
    rows["panel4_include"] = True
    rows["masked_loss_used"] = False
    rows["legacy_context"] = False

    legacy = pd.read_csv(legacy_w0022_source).copy()
    legacy = legacy[
        legacy["method"].astype(str).eq("direct_cp_32_16_8_4_ptv_lcc_isotv")
        & legacy["reg_value"].map(finite_float).map(lambda value: close(value, 0.022))
    ].copy()
    if legacy.empty:
        raise RuntimeError(f"missing legacy w=0.022 row in {legacy_w0022_source}")
    source_row = legacy.iloc[0].to_dict()
    legacy_row = {column: float("nan") for column in rows.columns}
    for column, value in source_row.items():
        if column in legacy_row:
            legacy_row[column] = value
    legacy_row.update(
        {
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "method_id": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "method_label": "MR-D-BSCP pTV-LCC+isoTV legacy w=0.022",
            "display_method": "MR-D-BSCP (pTVReg loss)",
            "method_family": "directcp_multistage",
            "reg_value": 0.022,
            "reg_label": "w=0.022*",
            "ratio_to_best_tre_weight": 0.4,
            "dataset": "lungct10",
            "selected_lr": 1e-3,
            "selected_lr_label": "1e-3",
            "source_path": repo_relative(legacy_w0022_source),
            "source_group": "legacy_w0022_refined_context",
            "schedule": "2400_legacy_context",
            "comparison_panel": "both",
            "schedule_role": "legacy_context",
            "panel3_include": True,
            "panel4_include": True,
            "masked_loss_used": False,
            "legacy_context": True,
        }
    )
    return pd.concat([rows, pd.DataFrame([legacy_row])], ignore_index=True, sort=False)


def load_dual_pl1a_rows(stage2_path: Path, stage3_path: Path) -> pd.DataFrame:
    frames = []
    for source_path in [stage2_path, stage3_path]:
        source = pd.read_csv(source_path).copy()
        source = source[source["variant"].astype(str).eq("PL1A")].copy()
        source = source[source["technical_complete"].astype(str).str.lower().eq("true")].copy()
        source = source[~source["weight_label"].astype(str).eq("m100")].copy()
        source["source_path"] = repo_relative(source_path)
        frames.append(source)
    source_rows = pd.concat(frames, ignore_index=True, sort=False)
    if source_rows.empty:
        raise RuntimeError("no PL1A rows available")

    tre = pd.to_numeric(source_rows["headline_native_snap_tre_mm_mean"], errors="coerce")
    best_weight = finite_float(source_rows.loc[tre.idxmin(), "weight_value"])
    out_rows: list[dict[str, Any]] = []
    for _, row in source_rows.iterrows():
        weight = finite_float(row.get("weight_value"))
        out_rows.append(
            {
                "method": "dual_inr_pl1a_author_blur_native_loss",
                "method_id": "dual_inr_pl1a_author_blur_native_loss",
                "method_label": "Dual-INR PL1A original DUAL loss",
                "display_method": "Dual-INR PL1A",
                "method_family": "dual_inr",
                "reg_value": weight,
                "reg_label": format_weight("m", weight),
                "ratio_to_best_tre_weight": weight / best_weight if best_weight else float("nan"),
                "cases": int(row.get("n", 10)),
                "original_snap_tre_mean_mm_mean": finite_float(
                    row.get("headline_native_snap_tre_mm_mean")
                ),
                "original_snap_tre_mean_mm_std": finite_float(
                    row.get("headline_native_snap_tre_mm_sd")
                ),
                "original_subvoxel_tre_mean_mm_mean": finite_float(
                    row.get("headline_native_subvoxel_tre_mm_mean")
                ),
                "original_subvoxel_tre_mean_mm_std": finite_float(
                    row.get("headline_native_subvoxel_tre_mm_sd")
                ),
                "fg_logj_std_mean": finite_float(row.get("regularity_log_jdet_std_fg_mean")),
                "fg_logj_std_std": finite_float(row.get("regularity_log_jdet_std_fg_sd")),
                "fg_jdet_le0_pct_mean": finite_float(
                    row.get("regularity_jdet_le0_pct_fg_mean")
                ),
                "fg_jdet_le0_pct_std": finite_float(row.get("regularity_jdet_le0_pct_fg_sd")),
                "elapsed_sec_mean": finite_float(row.get("total_wall_seconds_mean")),
                "elapsed_sec_std": finite_float(row.get("total_wall_seconds_sd")),
                "dataset": "lungct10",
                "source_path": str(row.get("source_path")),
                "source_group": "dual_inr_pl1a_stage2_stage3",
                "schedule": "dual_author",
                "panel3_include": False,
                "panel4_include": True,
                "masked_loss_used": False,
                "legacy_context": False,
                "weight_label": str(row.get("weight_label")),
            }
        )
    return pd.DataFrame(out_rows)


def load_case01_case08_native_tre_bridge(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    bridge = pd.read_csv(path)
    bridge = bridge[
        bridge["condition"].astype(str).eq("unmasked_full_crop_fdbe")
        & bridge["method_slug"].astype(str).isin(FDBE_SINGLE_POINT_SPECS)
    ].copy()
    values: dict[tuple[str, str], float] = {}
    for _, row in bridge.iterrows():
        values[(str(row["method_slug"]), str(row["case_key"]))] = finite_float(
            row.get("headline_tre_mean_mm")
        )
    return values


def load_fdbe_single_point_case_rows(
    job_manifest_path: Path,
    bridge_path: Path,
    *,
    data_root: Path,
    run_root: Path,
) -> pd.DataFrame:
    bridge = load_case01_case08_native_tre_bridge(bridge_path)
    manifest = pd.read_csv(job_manifest_path)
    manifest = manifest[
        manifest["dataset"].astype(str).eq("dirlab")
        & manifest["panel"].astype(str).eq("A")
        & manifest["method_slug"].astype(str).isin(FDBE_SINGLE_POINT_SPECS)
    ].copy()
    case_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        source_summary = placeholder_summary_path(row)
        summary_path = resolve_placeholders(source_summary, data_root=data_root, run_root=run_root)
        payload = load_json(summary_path)
        tre_value, tre_source = headline_tre_mean(payload)
        bridge_key = (str(row["method_slug"]), str(row["case_key"]))
        if bridge_key in bridge and math.isfinite(bridge[bridge_key]):
            tre_value = bridge[bridge_key]
            tre_source = "accepted_unmasked_native_tre_bridge"
        regularity = payload.get("regularity_fg_registration_crop", {})
        case_rows.append(
            {
                "dataset": "lungct10",
                "case_key": str(row["case_key"]),
                "case_id": int(row["case_id"]),
                "method_slug": str(row["method_slug"]),
                "display_method": FDBE_SINGLE_POINT_SPECS[str(row["method_slug"])]["display_method"],
                "method_id": FDBE_SINGLE_POINT_SPECS[str(row["method_slug"])]["method_id"],
                "source_summary_path": source_summary,
                "summary_exists": summary_path.is_file(),
                "tre_source": tre_source,
                "original_snap_tre_mean_mm": tre_value,
                "fg_logj_std": finite_float(regularity.get("log_jdet_std_fg")),
                "fg_jdet_le0_pct": finite_float(regularity.get("jdet_le0_pct_fg")),
                "fg_bending_energy": finite_float(regularity.get("bending_energy_fg")),
                "fg_diffusion_l2": finite_float(regularity.get("diffusion_l2_fg")),
                "elapsed_sec": finite_float(payload.get("elapsed_sec")),
                "peak_mem_gb": finite_float(payload.get("peak_mem_gb")),
                "lr": finite_float(row.get("lr")),
                "alpha": finite_float(row.get("alpha")),
                "cps": str(row.get("cps")),
                "epochs": int(row.get("epochs")),
                "seed": int(row.get("seed")),
                "masked_loss_used": False,
            }
        )
    return pd.DataFrame(case_rows)


def load_fdbe_alpha_extension_case_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing FD-BE alpha-extension case rows: {path}. "
            "Run aggregate_fdbe_cps4_alpha_extension_all10.py after the 40 jobs finish."
        )
    rows = pd.read_csv(path).copy()
    rows = rows[rows["method_slug"].astype(str).isin(FDBE_SINGLE_POINT_SPECS)].copy()
    rows = rows[rows["method_id"].astype(str).isin(
        {spec["method_id"] for spec in FDBE_SINGLE_POINT_SPECS.values()}
    )].copy()
    rows = rows[rows["status"].astype(str).eq("completed")].copy()
    if rows.empty:
        raise RuntimeError(f"No completed FD-BE alpha-extension rows found in {path}")
    rows["source_case_rows"] = repo_relative(path)
    rows["masked_loss_used"] = False
    return rows


def aggregate_fdbe_points(case_rows: pd.DataFrame, source_path: str) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    grouped = case_rows.copy()
    grouped["alpha_numeric"] = pd.to_numeric(grouped["alpha"], errors="coerce")
    for (method_slug, alpha), group in grouped.groupby(["method_slug", "alpha_numeric"], sort=True):
        spec = FDBE_SINGLE_POINT_SPECS[str(method_slug)]
        alpha_value = finite_float(alpha)
        tre_mean, tre_std = mean_std(group["original_snap_tre_mean_mm"].tolist())
        sub_mean, sub_std = mean_std(group.get("original_subvoxel_tre_mean_mm", pd.Series(dtype=float)).tolist())
        reg1_snap_mean, reg1_snap_std = mean_std(group.get("reg1mm_snap_tre_mean_mm", pd.Series(dtype=float)).tolist())
        reg1_sub_mean, reg1_sub_std = mean_std(group.get("reg1mm_subvoxel_tre_mean_mm", pd.Series(dtype=float)).tolist())
        logj_mean, logj_std = mean_std(group["fg_logj_std"].tolist())
        jdet_mean, jdet_std = mean_std(group["fg_jdet_le0_pct"].tolist())
        be_mean, be_std = mean_std(group["fg_bending_energy"].tolist())
        diffusion_mean, diffusion_std = mean_std(group["fg_diffusion_l2"].tolist())
        full_ncc_mean, full_ncc_std = mean_std(group.get("full_crop_one_minus_ncc", pd.Series(dtype=float)).tolist())
        mask_ncc_mean, mask_ncc_std = mean_std(group.get("fixed_t00_lung_mask_one_minus_ncc", pd.Series(dtype=float)).tolist())
        final_data_mean, final_data_std = mean_std(group.get("final_data_loss", pd.Series(dtype=float)).tolist())
        final_reg_mean, final_reg_std = mean_std(group.get("final_reg_loss", pd.Series(dtype=float)).tolist())
        elapsed_mean, elapsed_std = mean_std(group["elapsed_sec"].tolist())
        peak_mean, peak_std = mean_std(group["peak_mem_gb"].tolist())
        out_rows.append(
            {
                "method": spec["method_id"],
                "method_id": spec["method_id"],
                "method_label": spec["display_method"],
                "display_method": spec["display_method"],
                "method_family": spec["method_family"],
                "reg_value": alpha_value,
                "reg_label": format_weight("a", alpha_value),
                "ratio_to_best_tre_weight": alpha_value / FDBE_BASE_ALPHA if math.isfinite(alpha_value) else float("nan"),
                "cases": int(group["case_key"].nunique()),
                "original_snap_tre_mean_mm_mean": tre_mean,
                "original_snap_tre_mean_mm_std": tre_std,
                "original_subvoxel_tre_mean_mm_mean": sub_mean,
                "original_subvoxel_tre_mean_mm_std": sub_std,
                "reg1mm_snap_tre_mean_mm_mean": reg1_snap_mean,
                "reg1mm_snap_tre_mean_mm_std": reg1_snap_std,
                "reg1mm_subvoxel_tre_mean_mm_mean": reg1_sub_mean,
                "reg1mm_subvoxel_tre_mean_mm_std": reg1_sub_std,
                "fg_logj_std_mean": logj_mean,
                "fg_logj_std_std": logj_std,
                "fg_jdet_le0_pct_mean": jdet_mean,
                "fg_jdet_le0_pct_std": jdet_std,
                "fg_bending_energy_mean": be_mean,
                "fg_bending_energy_std": be_std,
                "fg_diffusion_l2_mean": diffusion_mean,
                "fg_diffusion_l2_std": diffusion_std,
                "full_crop_one_minus_ncc_mean": full_ncc_mean,
                "full_crop_one_minus_ncc_std": full_ncc_std,
                "fixed_t00_lung_mask_one_minus_ncc_mean": mask_ncc_mean,
                "fixed_t00_lung_mask_one_minus_ncc_std": mask_ncc_std,
                "final_data_loss_mean": final_data_mean,
                "final_data_loss_std": final_data_std,
                "final_reg_loss_mean": final_reg_mean,
                "final_reg_loss_std": final_reg_std,
                "elapsed_sec_mean": elapsed_mean,
                "elapsed_sec_std": elapsed_std,
                "peak_mem_gb_mean": peak_mean,
                "peak_mem_gb_std": peak_std,
                "dataset": "lungct10",
                "selected_lr": spec["lr"],
                "selected_lr_label": pareto_base.format_lr(spec["lr"]),
                "source_path": source_path,
                "source_group": "fit_modes_population_n10_plus_fdbe_cps4_alpha_extension",
                "schedule": "2500",
                "loss_domain": "full_crop_global_ncc",
                "regularizer": "dense_forward_difference_be",
                "panel3_include": False,
                "panel4_include": True,
                "masked_loss_used": False,
                "legacy_context": False,
            }
        )
    return pd.DataFrame(out_rows)


def finite_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[X_COL] = pd.to_numeric(out[X_COL], errors="coerce")
    out[Y_COL] = pd.to_numeric(out[Y_COL], errors="coerce")
    out = out[out[X_COL].notna() & out[Y_COL].notna()].copy()
    if out.empty:
        return out
    out["sort_ratio"] = pd.to_numeric(out.get("ratio_to_best_tre_weight"), errors="coerce")
    out["sort_reg"] = pd.to_numeric(out.get("reg_value"), errors="coerce")
    return out.sort_values(["sort_ratio", "sort_reg"], na_position="last")


def annotation_text(row: Mapping[str, Any]) -> str:
    method = str(row.get("method_id"))
    if method == "ptvreg_reference":
        return "pTVReg"
    label = str(row.get("reg_label") or "")
    if label:
        return label
    value = finite_float(row.get("reg_value"))
    return f"{value:g}" if math.isfinite(value) else method


def draw_lung_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    methods: Sequence[str],
    *,
    title: str,
    annotate: bool,
    ellipses: bool,
    zoom_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    annotate_offscale: bool = False,
) -> None:
    panel = rows[rows["method_id"].astype(str).isin(methods)].copy()
    for method in methods:
        group = panel[panel["method_id"].astype(str).eq(method)].copy()
        if group.empty:
            continue
        group = finite_pair_rows(group)
        if group.empty:
            continue
        spec = METHOD_SPECS[method]
        color = str(spec["color"])
        marker = str(spec["marker"])
        xs = group[X_COL].to_numpy(dtype=float)
        ys = group[Y_COL].to_numpy(dtype=float)
        if bool(spec["line"]) and len(group) > 1:
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=COMPOSITE_STYLE["line_width"],
                alpha=0.78,
                zorder=2,
            )
        if ellipses:
            for _, row in group.iterrows():
                xerr = max(finite_float(row.get(X_STD_COL), 0.0), 0.0)
                yerr = max(finite_float(row.get(Y_STD_COL), 0.0), 0.0)
                if xerr > 0 or yerr > 0:
                    ax.add_patch(
                        Ellipse(
                            (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                            width=2.0 * xerr,
                            height=2.0 * yerr,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=0.9,
                            alpha=COMPOSITE_STYLE["ellipse_alpha"],
                            zorder=1,
                        )
                    )
        size = (
            COMPOSITE_STYLE["ptvreg_marker_size"]
            if method == "ptvreg_reference"
            else COMPOSITE_STYLE["marker_size"]
        )
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=str(spec["label"]),
            zorder=3,
        )
        if annotate:
            for _, row in group.iterrows():
                ax.annotate(
                    annotation_text(row),
                    (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                )

    ax.set_title(title, fontsize=COMPOSITE_STYLE["title_fontsize"])
    ax.set_xlabel("Foreground SDlogJ")
    ax.set_ylabel("TRE (mm)")
    ax.tick_params(axis="both", labelsize=COMPOSITE_STYLE["tick_label_fontsize"])
    ax.xaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.yaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.grid(True, alpha=COMPOSITE_STYLE["grid_alpha"])
    if zoom_limits is not None:
        ax.set_xlim(*zoom_limits[0])
        ax.set_ylim(*zoom_limits[1])
        if annotate_offscale:
            add_offscale_notes(ax, panel, methods, zoom_limits)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="best",
        fontsize=COMPOSITE_STYLE["legend_fontsize"],
        frameon=True,
    )


def add_offscale_notes(
    ax: plt.Axes,
    rows: pd.DataFrame,
    methods: Sequence[str],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    (_xmin, _xmax), (_ymin, ymax) = limits
    for method in methods:
        group = finite_pair_rows(rows[rows["method_id"].astype(str).eq(method)])
        if group.empty:
            continue
        spec = METHOD_SPECS[method]
        color = str(spec["color"])
        for _, row in group.iterrows():
            y = finite_float(row[Y_COL])
            x = finite_float(row[X_COL])
            if y > ymax and math.isfinite(x):
                ax.annotate(
                    f"{spec['label']}: {y:.2f}",
                    (x, ymax),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=6.2,
                    color=color,
                    arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": color},
                )


def write_composite(
    output_dir: Path,
    oasis: pd.DataFrame,
    lung: pd.DataFrame,
    *,
    annotate: bool,
    filename: str,
    panel4_zoom: tuple[tuple[float, float], tuple[float, float]] | None = None,
    panel4_match_panel3_y: bool = False,
) -> Path:
    nicknames = pareto_base.load_nicknames()
    fig, axes = plt.subplots(2, 2, figsize=COMPOSITE_STYLE["figsize"], squeeze=False)
    oasis_plot = pareto_base._configure_oasis_plot_module(nicknames, annotate=annotate)
    pareto_base._draw_composite_oasis_panel(
        axes[0][0],
        oasis_plot,
        oasis,
        title="Dice vs SDlogJ (OASIS N=100)",
        acc_key="dice_mean",
        annotate=annotate,
    )
    pareto_base._draw_composite_oasis_panel(
        axes[0][1],
        oasis_plot,
        oasis,
        title="HD95 vs SDlogJ (OASIS N=100)",
        acc_key="hd95_mean",
        annotate=annotate,
    )
    draw_lung_panel(
        axes[1][0],
        lung[lung["panel3_include"].astype(bool)].copy(),
        PANEL3_METHODS,
        title="TRE vs SDlogJ (4DCT N=10)",
        annotate=annotate,
        ellipses=True,
    )
    panel3_ylim = axes[1][0].get_ylim()
    panel4_title = "Expanded TRE vs SDlogJ (4DCT N=10)"
    if panel4_zoom is not None:
        panel4_title = "Expanded TRE vs SDlogJ (zoomed)"
    elif panel4_match_panel3_y:
        panel4_title = "Expanded TRE vs SDlogJ (matched y)"
    draw_lung_panel(
        axes[1][1],
        lung[lung["panel4_include"].astype(bool)].copy(),
        PANEL4_METHODS,
        title=panel4_title,
        annotate=annotate,
        ellipses=False,
        zoom_limits=panel4_zoom,
        annotate_offscale=panel4_zoom is not None,
    )
    if panel4_match_panel3_y:
        axes[1][1].set_ylim(*panel3_ylim)
    fig.subplots_adjust(wspace=COMPOSITE_STYLE["wspace"], hspace=COMPOSITE_STYLE["hspace"])
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=COMPOSITE_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_review_report(
    output_dir: Path,
    manifest: Mapping[str, Any],
    lung: pd.DataFrame,
    fdbe_case_rows: pd.DataFrame,
) -> None:
    panel4 = lung[lung["panel4_include"].astype(bool)].copy()
    lines = [
        "# Current TRE/stdLogJ Expanded Pareto Preview with FD-BE cps4 Alpha Extension",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- OASIS panels are copied from the OASIS100 paper-ready source rows.",
        "- Bottom-left keeps the original LungCT TRE/stdLogJ structure with ellipses.",
        "- Bottom-right is expanded LungCT TRE/stdLogJ without ellipses and includes FD-BE cps4 alpha sweeps.",
        "- Masked-loss method IDs are not included in `source_rows.csv`.",
        "- The `w=0.022*` MR-D-BSCP point is tagged as legacy 2400-schedule context.",
        "",
        "## Outputs",
        "",
    ]
    for output in manifest["outputs"]:
        lines.append(f"- `{output}`")
    lines.extend(
        [
            "",
            "## Panel 4 Rows",
            "",
            "| Method | Points | TRE range | SDlogJ range |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in PANEL4_METHODS:
        group = panel4[panel4["method_id"].astype(str).eq(method)]
        if group.empty:
            continue
        label = METHOD_SPECS[method]["label"]
        lines.append(
            f"| {label} | {len(group)} | "
            f"{group[Y_COL].min():.3f}-{group[Y_COL].max():.3f} | "
            f"{group[X_COL].min():.4f}-{group[X_COL].max():.4f} |"
        )
    lines.extend(
        [
            "",
            "## FD-BE cps4 Alpha-Sweep Case Rows",
            "",
            "| Method | Alpha | Cases | TRE mean | SDlogJ mean | TRE source notes |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    fdbe_case_rows = fdbe_case_rows.copy()
    fdbe_case_rows["alpha_numeric"] = pd.to_numeric(fdbe_case_rows["alpha"], errors="coerce")
    for (method_slug, alpha), group in fdbe_case_rows.groupby(["method_slug", "alpha_numeric"], sort=True):
        notes = ", ".join(sorted(group["tre_source"].astype(str).unique()))
        lines.append(
            f"| {group['display_method'].iloc[0]} | {alpha:g} | {len(group)} | "
            f"{group['original_snap_tre_mean_mm'].mean():.3f} | "
            f"{group['fg_logj_std'].mean():.4f} | {notes} |"
        )
    (output_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def validate_outputs(oasis_source: Path, output_dir: Path, lung: pd.DataFrame) -> dict[str, Any]:
    output_oasis = output_dir / "oasis100_pareto_source_rows.csv"
    original = pd.read_csv(oasis_source)
    copied = pd.read_csv(output_oasis)
    masked_methods = [
        method for method in lung["method_id"].astype(str).unique() if "masked" in method.lower()
    ]
    source_paths = lung.get("source_path", pd.Series(dtype=str)).astype(str)
    masked_source_paths = [path for path in source_paths if "masked_ncc" in path.lower()]
    panel3 = lung[lung["panel3_include"].astype(bool)]
    panel4 = lung[lung["panel4_include"].astype(bool)]
    fdbe_alpha_values: dict[str, list[float]] = {}
    for method in ["sinr_cps4_global_ncc_fd_be", "direct_cp_cps4_global_ncc_fd_be"]:
        group = panel4[panel4["method_id"].astype(str).eq(method)]
        values = sorted(
            value
            for value in pd.to_numeric(group.get("reg_value", pd.Series(dtype=float)), errors="coerce").dropna().unique()
            if math.isfinite(float(value))
        )
        fdbe_alpha_values[method] = [float(value) for value in values]
    return {
        "oasis_rows_match_v003": bool(oasis_source.read_bytes() == output_oasis.read_bytes()),
        "oasis_row_count": int(len(copied)),
        "masked_method_ids_in_source_rows": masked_methods,
        "masked_ncc_source_paths_in_source_rows": masked_source_paths,
        "panel3_methods": sorted(panel3["method_id"].astype(str).unique()),
        "panel4_methods": sorted(panel4["method_id"].astype(str).unique()),
        "panel3_row_count": int(len(panel3)),
        "panel4_row_count": int(len(panel4)),
        "panel4_ellipse_policy": "none",
        "fdbe_cps4_alpha_values": fdbe_alpha_values,
    }


def build(
    *,
    output_dir: Path,
    oasis_source: Path,
    lung_schedule_source: Path,
    legacy_w0022_source: Path,
    dual_stage2_source: Path,
    dual_stage3_source: Path,
    fdbe_job_manifest: Path,
    fdbe_case01_case08_bridge: Path,
    fdbe_alpha_extension_case_rows: Path,
    data_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = data_root / "runs/fit_modes_population_n10_v1"

    oasis = load_oasis_rows(oasis_source)
    lung_base = load_schedule_refreshed_lung_rows(lung_schedule_source, legacy_w0022_source)
    dual_pl1a = load_dual_pl1a_rows(dual_stage2_source, dual_stage3_source)
    fdbe_case_rows = load_fdbe_single_point_case_rows(
        fdbe_job_manifest,
        fdbe_case01_case08_bridge,
        data_root=data_root,
        run_root=run_root,
    )
    fdbe_alpha_extension = load_fdbe_alpha_extension_case_rows(fdbe_alpha_extension_case_rows)
    fdbe_case_rows = pd.concat([fdbe_case_rows, fdbe_alpha_extension], ignore_index=True, sort=False)
    fdbe_case_rows_path = output_dir / "fdbe_cps4_alpha_sweep_case_rows.csv"
    fdbe_case_rows.to_csv(fdbe_case_rows_path, index=False)
    fdbe_points = aggregate_fdbe_points(
        fdbe_case_rows,
        source_path=repo_relative(fdbe_case_rows_path),
    )

    lung = pd.concat([lung_base, dual_pl1a, fdbe_points], ignore_index=True, sort=False)
    lung["panel3_include"] = lung["panel3_include"].fillna(False).astype(bool)
    lung["panel4_include"] = lung["panel4_include"].fillna(False).astype(bool)
    lung["masked_loss_used"] = lung["masked_loss_used"].fillna(False).astype(bool)
    lung["legacy_context"] = lung["legacy_context"].fillna(False).astype(bool)

    oasis_output = output_dir / "oasis100_pareto_source_rows.csv"
    lung_output = output_dir / "source_rows.csv"
    shutil.copyfile(oasis_source, oasis_output)
    lung.to_csv(lung_output, index=False)

    outputs = [
        write_composite(
            output_dir,
            oasis,
            lung,
            annotate=False,
            filename="pareto_four_panel_current_fdbe_alpha_extension_no_annotations.png",
        ),
        write_composite(
            output_dir,
            oasis,
            lung,
            annotate=True,
            filename="pareto_four_panel_current_fdbe_alpha_extension_annotated_review.png",
        ),
        write_composite(
            output_dir,
            oasis,
            lung,
            annotate=True,
            filename="pareto_four_panel_current_fdbe_alpha_extension_zoomed_review.png",
            panel4_zoom=((0.07, 0.50), (0.86, 2.15)),
        ),
    ]

    validation = validate_outputs(oasis_source, output_dir, lung)
    manifest: dict[str, Any] = {
        "artifact": "pareto_current_tre_stdlogj_fdbe_cps4_alpha_extension",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_inputs": {
            "oasis100_base": repo_relative(oasis_source),
            "lungct_schedule_refresh": repo_relative(lung_schedule_source),
            "legacy_w0022_source": repo_relative(legacy_w0022_source),
            "dual_pl1a_stage2": repo_relative(dual_stage2_source),
            "dual_pl1a_stage3": repo_relative(dual_stage3_source),
            "fdbe_population_job_manifest": repo_relative(fdbe_job_manifest),
            "fdbe_alpha_extension_case_rows": repo_relative(fdbe_alpha_extension_case_rows),
        },
        "source_rows": repo_relative(lung_output),
        "oasis_source_rows": repo_relative(oasis_output),
        "fdbe_cps4_alpha_sweep_case_rows": repo_relative(fdbe_case_rows_path),
        "outputs": [repo_relative(path) for path in outputs],
        "panel_layout": {
            "A": "Dice vs SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "B": "HD95 vs SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "C": "TRE vs SDlogJ, LungCT N=10, original structure plus 2500 MR-D-BSCP refresh",
            "D": "Expanded TRE vs SDlogJ, LungCT N=10, no ellipses",
        },
        "new_fdbe_alpha_extension_runs_included": True,
        "masked_loss_rows_included": False,
        "legacy_w0022_policy": "Included as pTVReg-loss MR-D-BSCP legacy 2400-schedule context row.",
        "fdbe_cps4_alpha_sweep_policy": (
            "Use accepted unmasked full-crop global-NCC + dense FD-BE cps4 alpha=10000 population rows "
            "plus the new alpha=100000/1000000 all-10 extension rows; case01/case08 alpha=10000 "
            "native TRE values use the accepted unmasked native-TRE bridge when available."
        ),
        "validation": validation,
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_review_report(output_dir, manifest, lung, fdbe_case_rows)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oasis-source", type=Path, default=DEFAULT_OASIS_SOURCE)
    parser.add_argument("--lung-schedule-source", type=Path, default=DEFAULT_LUNG_SCHEDULE_SOURCE)
    parser.add_argument("--legacy-w0022-source", type=Path, default=DEFAULT_LEGACY_W0022_SOURCE)
    parser.add_argument("--dual-stage2-source", type=Path, default=DEFAULT_DUAL_STAGE2_SOURCE)
    parser.add_argument("--dual-stage3-source", type=Path, default=DEFAULT_DUAL_STAGE3_SOURCE)
    parser.add_argument("--fdbe-job-manifest", type=Path, default=DEFAULT_FDBE_JOB_MANIFEST)
    parser.add_argument("--fdbe-case01-case08-bridge", type=Path, default=DEFAULT_FDBE_CASE01_CASE08_BRIDGE)
    parser.add_argument("--fdbe-alpha-extension-case-rows", type=Path, default=DEFAULT_FDBE_ALPHA_EXTENSION_CASE_ROWS)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build(
        output_dir=args.output_dir.resolve(),
        oasis_source=args.oasis_source.resolve(),
        lung_schedule_source=args.lung_schedule_source.resolve(),
        legacy_w0022_source=args.legacy_w0022_source.resolve(),
        dual_stage2_source=args.dual_stage2_source.resolve(),
        dual_stage3_source=args.dual_stage3_source.resolve(),
        fdbe_job_manifest=args.fdbe_job_manifest.resolve(),
        fdbe_case01_case08_bridge=args.fdbe_case01_case08_bridge.resolve(),
        fdbe_alpha_extension_case_rows=args.fdbe_alpha_extension_case_rows.resolve(),
        data_root=args.data_root.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
