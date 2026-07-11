#!/usr/bin/env python
"""Build the LungCT DirectCP 4-stage mid-ratio Pareto extension review."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


if "INRORSPLINE_REPO_ROOT" in os.environ:
    REPO_ROOT = Path(os.environ["INRORSPLINE_REPO_ROOT"]).expanduser()
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LUNGCT_SCRIPT_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
if str(LUNGCT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LUNGCT_SCRIPT_DIR))

import aggregate_report as flat  # noqa: E402
from scripts.reports import lungct_pareto_visual_review as visual  # noqa: E402


METHOD_IDIR = visual.METHOD_IDIR
METHOD_PTV_DIRECTCP = visual.METHOD_PTV_DIRECTCP
METHOD_FDBE = visual.METHOD_FDBE
METHOD_PTVREG = visual.METHOD_PTVREG
METHOD_LABELS = visual.METHOD_LABELS

EXPECTED_CASES = list(range(1, 11))
BASE_PTV_ISOTV = 0.11
BASE_FDBE_ALPHA = 10000.0
BASE_IDIR_ALPHA = 10.0
EXPECTED_IDIR_ALPHAS = [BASE_IDIR_ALPHA * 0.3, BASE_IDIR_ALPHA * 3.0]
EXPECTED_PTV_WEIGHTS = [BASE_PTV_ISOTV * 0.3, BASE_PTV_ISOTV * 3.0]
EXPECTED_FDBE_ALPHAS = [BASE_FDBE_ALPHA * 0.3, BASE_FDBE_ALPHA * 3.0]
RATIO_ANCHOR_WEIGHTS = {
    METHOD_IDIR: BASE_IDIR_ALPHA,
    METHOD_PTV_DIRECTCP: BASE_PTV_ISOTV,
    METHOD_FDBE: BASE_FDBE_ALPHA,
}

SOURCE_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep"
DEFAULT_EXISTING_CASE_ROWS = SOURCE_REVIEW_ROOT / "comparison_case_rows.csv"
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension"
DEFAULT_OUTPUT_ROOT = DEFAULT_REVIEW_ROOT / "pareto_visual_review_midratio_extension"
DEFAULT_IDIR_ALPHA3_ROOT = Path("/database/inrorspline/runs/lungct_idir_midratio_alpha3_v1")
DEFAULT_IDIR_ALPHA30_ROOT = Path("/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1")
DEFAULT_PTV_EXTENSION_ROOT = Path("/database/inrorspline/runs/lungct_directcp4stage_midratio_ptvlcc_v1")
DEFAULT_FDBE_EXTENSION_ROOT = Path("/database/inrorspline/runs/lungct_directcp4stage_midratio_fdbe_v1")

METRIC_MAP = {
    "original_snap_tre_mean_mm": "native_crop_snap_tre_mean_mm",
    "original_snap_tre_std_mm": "native_crop_snap_tre_std_mm",
    "original_snap_tre_median_mm": "native_crop_snap_tre_median_mm",
    "original_subvoxel_tre_mean_mm": "native_crop_sub_tre_mean_mm",
    "original_subvoxel_tre_std_mm": "native_crop_sub_tre_std_mm",
    "original_subvoxel_tre_median_mm": "native_crop_sub_tre_median_mm",
    "reg1mm_snap_tre_mean_mm": "reg1mm_snap_tre_mean_mm",
    "reg1mm_snap_tre_std_mm": "reg1mm_snap_tre_std_mm",
    "reg1mm_snap_tre_median_mm": "reg1mm_snap_tre_median_mm",
    "reg1mm_subvoxel_tre_mean_mm": "reg1mm_sub_tre_mean_mm",
    "reg1mm_subvoxel_tre_std_mm": "reg1mm_sub_tre_std_mm",
    "reg1mm_subvoxel_tre_median_mm": "reg1mm_sub_tre_median_mm",
    "fg_logj_std": "fg_log_jdet_std",
    "fg_jdet_le0_pct": "fg_jdet_le0_pct",
    "fg_bending_energy": "fg_bending_energy",
    "fg_diffusion_l2": "fg_diffusion_l2",
    "fg_disp_mag_mean_vox": "fg_disp_mag_mean_vox",
    "fg_disp_mag_p95_vox": "fg_disp_mag_p95_vox",
    "fg_disp_mag_max_vox": "fg_disp_mag_max_vox",
    "elapsed_sec": "elapsed_sec",
    "peak_mem_gb": "peak_mem_gb",
}


def fmt_reg(value: object) -> str:
    return f"{float(value):g}"


def close_to_any(value: float, expected: Sequence[float]) -> bool:
    return any(abs(float(value) - float(item)) <= 1e-9 * max(1.0, abs(float(item))) for item in expected)


def value_from_run_name(summary_path: Path, prefix: str) -> float:
    match = re.search(rf"_{re.escape(prefix)}([0-9p]+)_seed", summary_path.parent.name)
    if not match:
        return float("nan")
    return float(match.group(1).replace("p", "."))


def summary_metric_values(summary_path: Path, payload: Mapping[str, object]) -> Dict[str, object]:
    flat_row = flat.flatten_summary(summary_path, dict(payload))
    return {dst: flat_row.get(src, "") for dst, src in METRIC_MAP.items()}


def row_from_idir_summary(summary_path: Path, payload: Mapping[str, object], source: str) -> Dict[str, object]:
    flat_row = flat.flatten_summary(summary_path, dict(payload))
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    alpha = float(training.get("alpha_bending", float("nan")))
    if not math.isfinite(alpha):
        alpha = value_from_run_name(summary_path, "bend")
    row: Dict[str, object] = {
        "method": METHOD_IDIR,
        "method_label": METHOD_LABELS[METHOD_IDIR],
        "case_id": int(flat_row.get("case_id") or payload.get("case_id")),
        "source": source,
        "reg_sort": alpha / BASE_IDIR_ALPHA,
        "reg_factor": alpha / BASE_IDIR_ALPHA,
        "reg_value": alpha,
        "reg_label": f"a={fmt_reg(alpha)}",
        "alpha": alpha,
        "summary_path": str(summary_path),
    }
    row.update(summary_metric_values(summary_path, payload))
    return row


def row_from_ptv_summary(summary_path: Path, payload: Mapping[str, object], source: str) -> Dict[str, object]:
    flat_row = flat.flatten_summary(summary_path, dict(payload))
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    weight = float(training.get("ptv_isotv_weight", float("nan")))
    if not math.isfinite(weight):
        weight = value_from_run_name(summary_path, "w")
    row: Dict[str, object] = {
        "method": METHOD_PTV_DIRECTCP,
        "method_label": METHOD_LABELS[METHOD_PTV_DIRECTCP],
        "case_id": int(flat_row.get("case_id") or payload.get("case_id")),
        "source": source,
        "reg_sort": weight / BASE_PTV_ISOTV,
        "reg_factor": weight / BASE_PTV_ISOTV,
        "reg_value": weight,
        "reg_label": f"w={fmt_reg(weight)}",
        "summary_path": str(summary_path),
    }
    row.update(summary_metric_values(summary_path, payload))
    return row


def row_from_fdbe_summary(summary_path: Path, payload: Mapping[str, object], source: str) -> Dict[str, object]:
    flat_row = flat.flatten_summary(summary_path, dict(payload))
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    alpha = float(training.get("alpha", float("nan")))
    if not math.isfinite(alpha):
        alpha = value_from_run_name(summary_path, "a")
    row: Dict[str, object] = {
        "method": METHOD_FDBE,
        "method_label": METHOD_LABELS[METHOD_FDBE],
        "case_id": int(flat_row.get("case_id") or payload.get("case_id")),
        "source": source,
        "reg_sort": alpha,
        "reg_value": alpha,
        "reg_label": f"a={fmt_reg(alpha)}",
        "alpha": alpha,
        "summary_path": str(summary_path),
    }
    row.update(summary_metric_values(summary_path, payload))
    return row


def collect_completed_summaries(root: Path, method: str) -> List[tuple[Path, Dict[str, object]]]:
    rows: List[tuple[Path, Dict[str, object]]] = []
    for summary_path in sorted((Path(root) / method).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") == "completed":
            rows.append((summary_path, payload))
    return rows


def dedupe_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[tuple[str, int, float], Dict[str, object]] = {}
    duplicates: List[str] = []
    for row in rows:
        key = (str(row["method"]), int(row["case_id"]), round(float(row["reg_value"]), 12))
        if key in by_key:
            duplicates.append(f"{key[0]} case{key[1]:02d} weight={key[2]:g}")
            continue
        by_key[key] = dict(row)
    if duplicates:
        raise RuntimeError("Duplicate extension rows: " + "; ".join(duplicates))
    return [by_key[key] for key in sorted(by_key)]


def load_extension_case_rows(
    idir_alpha3_root: Path = DEFAULT_IDIR_ALPHA3_ROOT,
    idir_alpha30_root: Path = DEFAULT_IDIR_ALPHA30_ROOT,
    ptv_root: Path = DEFAULT_PTV_EXTENSION_ROOT,
    fdbe_root: Path = DEFAULT_FDBE_EXTENSION_ROOT,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path, payload in collect_completed_summaries(idir_alpha3_root, METHOD_IDIR):
        row = row_from_idir_summary(path, payload, "midratio_idir_alpha3_extension")
        if int(row["case_id"]) in EXPECTED_CASES and close_to_any(float(row["reg_value"]), [BASE_IDIR_ALPHA * 0.3]):
            rows.append(row)
    for path, payload in collect_completed_summaries(idir_alpha30_root, METHOD_IDIR):
        row = row_from_idir_summary(path, payload, "midratio_idir_alpha30_reused")
        if int(row["case_id"]) in EXPECTED_CASES and close_to_any(float(row["reg_value"]), [BASE_IDIR_ALPHA * 3.0]):
            rows.append(row)
    for path, payload in collect_completed_summaries(ptv_root, METHOD_PTV_DIRECTCP):
        row = row_from_ptv_summary(path, payload, "midratio_ptvlcc_extension")
        if int(row["case_id"]) in EXPECTED_CASES and close_to_any(float(row["reg_value"]), EXPECTED_PTV_WEIGHTS):
            rows.append(row)
    for path, payload in collect_completed_summaries(fdbe_root, METHOD_FDBE):
        row = row_from_fdbe_summary(path, payload, "midratio_fdbe_extension")
        if int(row["case_id"]) in EXPECTED_CASES and close_to_any(float(row["reg_value"]), EXPECTED_FDBE_ALPHAS):
            rows.append(row)
    rows = dedupe_rows(rows)
    return pd.DataFrame(rows)


def extension_completion_errors(extension_df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if extension_df.empty:
        return ["No extension rows found."]
    for method, expected_values in (
        (METHOD_IDIR, EXPECTED_IDIR_ALPHAS),
        (METHOD_PTV_DIRECTCP, EXPECTED_PTV_WEIGHTS),
        (METHOD_FDBE, EXPECTED_FDBE_ALPHAS),
    ):
        method_df = extension_df[extension_df["method"].eq(method)]
        for value in expected_values:
            got_cases = sorted(
                int(v)
                for v in method_df.loc[
                    method_df["reg_value"].map(lambda x: abs(float(x) - float(value)) <= 1e-9 * max(1.0, abs(float(value)))),
                    "case_id",
                ].tolist()
            )
            if got_cases != EXPECTED_CASES:
                errors.append(
                    f"{METHOD_LABELS[method]} weight {value:g}: got cases {got_cases}, expected {EXPECTED_CASES}"
                )
    return errors


def load_existing_case_rows(path: Path = DEFAULT_EXISTING_CASE_ROWS) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin([METHOD_IDIR, METHOD_PTV_DIRECTCP, METHOD_FDBE])].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method_label"] if "method_label" in df else df["method"])
    for col in ["case_id", "reg_sort", "reg_value", *visual.METRIC_KEYS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_fixed_anchor_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ratio_to_best_tre_weight"] = np.nan
    for method, anchor in RATIO_ANCHOR_WEIGHTS.items():
        mask = out["method"].eq(method)
        out.loc[mask, "ratio_to_best_tre_weight"] = pd.to_numeric(out.loc[mask, "reg_value"], errors="coerce") / anchor
    return out


def combine_case_rows(
    existing_df: pd.DataFrame,
    extension_df: pd.DataFrame,
    ptvreg_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([existing_df, extension_df, ptvreg_df], ignore_index=True, sort=False)
    combined["method_label"] = combined["method"].map(METHOD_LABELS).fillna(combined.get("method_label", combined["method"]))
    for col in ["case_id", "reg_sort", "reg_value", *visual.METRIC_KEYS]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return apply_fixed_anchor_ratios(combined)


def build_review_tables(
    existing_case_rows: Path = DEFAULT_EXISTING_CASE_ROWS,
    idir_alpha3_root: Path = DEFAULT_IDIR_ALPHA3_ROOT,
    idir_alpha30_root: Path = DEFAULT_IDIR_ALPHA30_ROOT,
    ptv_root: Path = DEFAULT_PTV_EXTENSION_ROOT,
    fdbe_root: Path = DEFAULT_FDBE_EXTENSION_ROOT,
    ptvreg_root: Path = visual.DEFAULT_PTVREG_ROOT,
    ptvprep_root: Path = visual.DEFAULT_PTVPREP_ROOT,
) -> Dict[str, pd.DataFrame | List[str]]:
    existing_df = load_existing_case_rows(existing_case_rows)
    extension_df = load_extension_case_rows(idir_alpha3_root, idir_alpha30_root, ptv_root, fdbe_root)
    errors = extension_completion_errors(extension_df)
    ptvreg_df = visual.load_ptvreg_case_rows(ptvreg_root=ptvreg_root, ptvprep_root=ptvprep_root)
    case_df = combine_case_rows(existing_df, extension_df, ptvreg_df)
    mean_df = visual.aggregate_case_rows(case_df)
    table_df = visual.screening_table(mean_df)
    return {
        "existing_df": existing_df,
        "extension_df": extension_df,
        "ptvreg_df": ptvreg_df,
        "case_df": case_df,
        "mean_df": mean_df,
        "table_df": table_df,
        "completion_errors": errors,
    }


def save_and_plot_review(
    case_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    table_df: pd.DataFrame,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exclude_by_ratio: Mapping[str, Sequence[float]] | None = None,
    exclude_by_weight: Mapping[str, Sequence[float]] | None = None,
) -> Dict[str, object]:
    exclude_by_ratio = exclude_by_ratio or {
        METHOD_IDIR: [],
        METHOD_PTV_DIRECTCP: [100.0],
        METHOD_FDBE: [0.0, 10.0],
    }
    exclude_by_weight = exclude_by_weight or {METHOD_IDIR: [], METHOD_PTV_DIRECTCP: [], METHOD_FDBE: []}
    artifacts = visual.save_review_artifacts(case_df, mean_df, table_df, output_root=output_root)
    plots = visual.make_all_plots(mean_df, output_root, exclude_by_ratio, exclude_by_weight)
    validation = visual.validate_outputs(output_root)
    return {"artifacts": artifacts, "plots": plots, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-case-rows", type=Path, default=DEFAULT_EXISTING_CASE_ROWS)
    parser.add_argument("--idir-alpha3-root", type=Path, default=DEFAULT_IDIR_ALPHA3_ROOT)
    parser.add_argument("--idir-alpha30-root", type=Path, default=DEFAULT_IDIR_ALPHA30_ROOT)
    parser.add_argument("--ptv-root", type=Path, default=DEFAULT_PTV_EXTENSION_ROOT)
    parser.add_argument("--fdbe-root", type=Path, default=DEFAULT_FDBE_EXTENSION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = build_review_tables(
        args.existing_case_rows,
        args.idir_alpha3_root,
        args.idir_alpha30_root,
        args.ptv_root,
        args.fdbe_root,
    )
    errors = list(tables["completion_errors"])
    result = save_and_plot_review(
        tables["case_df"],  # type: ignore[arg-type]
        tables["mean_df"],  # type: ignore[arg-type]
        tables["table_df"],  # type: ignore[arg-type]
        output_root=args.output_root,
    )
    validation = result["validation"]
    payload = {
        "existing_rows": int(len(tables["existing_df"])),  # type: ignore[arg-type]
        "extension_rows": int(len(tables["extension_df"])),  # type: ignore[arg-type]
        "ptvreg_rows": int(len(tables["ptvreg_df"])),  # type: ignore[arg-type]
        "case_rows": int(len(tables["case_df"])),  # type: ignore[arg-type]
        "mean_rows": int(len(tables["mean_df"])),  # type: ignore[arg-type]
        "completion_errors": errors,
        "output_validation": validation,
    }
    print(json.dumps(payload, indent=2))
    if args.strict and (errors or not validation.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
