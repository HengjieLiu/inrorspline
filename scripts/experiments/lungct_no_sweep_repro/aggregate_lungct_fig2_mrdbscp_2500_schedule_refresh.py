#!/usr/bin/env python
"""Aggregate LungCT Figure 2 MR-D-BSCP 2500-step schedule refresh runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import aggregate_report as base
import common as c
from launch_lungct_fig2_mrdbscp_2500_schedule_refresh import (
    DEFAULT_ALPHAS,
    DEFAULT_IDIR_OUT_ROOT,
    DEFAULT_PTV_ANCHOR_ROOT,
    DEFAULT_PTV_OUT_ROOT,
    DEFAULT_PTV_WEIGHTS,
    DEFAULT_STAGE_EPOCHS,
    FAMILY_IDIR_FDBE,
    FAMILY_PTV,
)
from run_one import METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE, METHOD_DIRECT_CP_32_PTV


EXPECTED_CASES = list(range(1, 11))
EXPECTED_STAGE_EPOCHS = [400, 400, 800, 900]
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh"
DEFAULT_OLD_MIDRATIO_MEAN = (
    c.REPO_ROOT
    / "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/"
    / "pareto_visual_review_midratio_extension/combined_mean_rows_with_ptvreg.csv"
)
DEFAULT_OLD_MIDRATIO_CASE = (
    c.REPO_ROOT
    / "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/"
    / "pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv"
)
DEFAULT_OLD_PTV_REFINED_MEAN = (
    c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_mean_rows.csv"
)
DEFAULT_OLD_PTV_REFINED_CASE = (
    c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_case_rows.csv"
)

METRIC_KEYS = [
    "original_snap_tre_mean_mm",
    "original_subvoxel_tre_mean_mm",
    "reg1mm_snap_tre_mean_mm",
    "reg1mm_subvoxel_tre_mean_mm",
    "fg_logj_std",
    "fg_jdet_le0_pct",
    "fg_bending_energy",
    "fg_diffusion_l2",
    "elapsed_sec",
    "peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idir-run-root", type=Path, default=DEFAULT_IDIR_OUT_ROOT)
    parser.add_argument("--ptv-run-root", type=Path, default=DEFAULT_PTV_OUT_ROOT)
    parser.add_argument("--ptv-anchor-root", type=Path, default=DEFAULT_PTV_ANCHOR_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--old-midratio-mean", type=Path, default=DEFAULT_OLD_MIDRATIO_MEAN)
    parser.add_argument("--old-midratio-case", type=Path, default=DEFAULT_OLD_MIDRATIO_CASE)
    parser.add_argument("--old-ptv-refined-mean", type=Path, default=DEFAULT_OLD_PTV_REFINED_MEAN)
    parser.add_argument("--old-ptv-refined-case", type=Path, default=DEFAULT_OLD_PTV_REFINED_CASE)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--ptv-weights", type=str, default=DEFAULT_PTV_WEIGHTS)
    parser.add_argument("--stage-epochs", type=str, default=DEFAULT_STAGE_EPOCHS)
    return parser.parse_args()


def parse_float_list(value: str) -> List[float]:
    return [float(v) for v in value.replace(" ", "").split(",") if v]


def parse_stage_epochs(value: object) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v) for v in value.replace(" ", "").split(",") if v]
    return []


def close(a: object, b: object) -> bool:
    av = base.safe_float(a)
    bv = base.safe_float(b)
    return math.isfinite(av) and math.isfinite(bv) and abs(av - bv) <= 1e-8 * max(1.0, abs(bv))


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return base.markdown_table(list(headers), [[str(v) for v in row] for row in rows])


def read_csv_rows(path: Path, source_kind: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["source_csv"] = str(path)
            row["source_kind"] = source_kind
            rows.append(dict(row))
    return rows


def row_from_summary(path: Path, payload: Dict[str, object], family: str, reg_value: float) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    out: Dict[str, object] = {
        "schedule": "2500",
        "family": family,
        "method": payload.get("method"),
        "case_id": int(payload.get("case_id") or row.get("case_id")),
        "status": payload.get("status"),
        "reg_value": float(reg_value),
        "reg_label": f"a={reg_value:g}" if family == FAMILY_IDIR_FDBE else f"w={reg_value:g}",
        "summary_path": str(path),
        "source_csv": "",
        "source_kind": "new_summary",
        "stage_epochs": ",".join(str(v) for v in parse_stage_epochs(training.get("stage_epochs"))),
        "stage_epochs_total": sum(parse_stage_epochs(training.get("stage_epochs"))),
        "lr": ",".join(str(v) for v in training.get("stage_lr", []))
        if isinstance(training.get("stage_lr"), list)
        else training.get("lr", ""),
        "alpha": training.get("alpha"),
        "ptv_isotv_weight": training.get("ptv_isotv_weight"),
        "loss": training.get("loss", ""),
        "image_loss": training.get("image_loss", ""),
        "regularizer": training.get("regularizer", ""),
    }
    out.update(
        {
            "original_snap_tre_mean_mm": row.get("native_crop_snap_tre_mean_mm"),
            "original_snap_tre_std_mm": row.get("native_crop_snap_tre_std_mm"),
            "original_snap_tre_median_mm": row.get("native_crop_snap_tre_median_mm"),
            "original_subvoxel_tre_mean_mm": row.get("native_crop_sub_tre_mean_mm"),
            "original_subvoxel_tre_std_mm": row.get("native_crop_sub_tre_std_mm"),
            "reg1mm_snap_tre_mean_mm": row.get("reg1mm_snap_tre_mean_mm"),
            "reg1mm_snap_tre_std_mm": row.get("reg1mm_snap_tre_std_mm"),
            "reg1mm_subvoxel_tre_mean_mm": row.get("reg1mm_sub_tre_mean_mm"),
            "reg1mm_subvoxel_tre_std_mm": row.get("reg1mm_sub_tre_std_mm"),
            "fg_logj_std": row.get("fg_log_jdet_std"),
            "fg_jdet_le0_pct": row.get("fg_jdet_le0_pct"),
            "fg_bending_energy": row.get("fg_bending_energy"),
            "fg_diffusion_l2": row.get("fg_diffusion_l2"),
            "elapsed_sec": row.get("elapsed_sec"),
            "peak_mem_gb": row.get("peak_mem_gb"),
        }
    )
    return out


def expected_name(method: str, case_id: int, reg_value: float, stage_epochs: str) -> str:
    import argparse as _argparse

    fake = _argparse.Namespace(
        run_name="",
        method=method,
        dataset="4dct",
        case_id=int(case_id),
        seed=1,
        lr=0.001,
        epochs=None,
        alpha=float(reg_value) if method == METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE else 10.0,
        alpha_bending=10.0,
        ptv_isotv_weight=float(reg_value) if method == METHOD_DIRECT_CP_32_PTV else 0.11,
        stage_full_cps="32,16,8,4",
        stage_downsample="8,4,2,1",
        stage_epochs=stage_epochs,
    )
    from run_one import run_name

    return run_name(fake, method, 0.001)


def collect_new_rows(args: argparse.Namespace, method: str, family: str, reg_values: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for reg_value in reg_values:
        for case_id in EXPECTED_CASES:
            name = expected_name(method, case_id, reg_value, args.stage_epochs)
            roots = [args.idir_run_root] if family == FAMILY_IDIR_FDBE else [args.ptv_run_root, args.ptv_anchor_root]
            summary_path = None
            for root in roots:
                candidate = root / method / name / "summary.json"
                if candidate.is_file():
                    summary_path = candidate
                    break
            if summary_path is None:
                rows.append(
                    {
                        "schedule": "2500",
                        "family": family,
                        "method": method,
                        "case_id": case_id,
                        "status": "missing",
                        "reg_value": float(reg_value),
                        "reg_label": f"a={reg_value:g}" if family == FAMILY_IDIR_FDBE else f"w={reg_value:g}",
                        "summary_path": "",
                        "stage_epochs": "",
                    }
                )
                continue
            payload = json.loads(summary_path.read_text())
            rows.append(row_from_summary(summary_path, payload, family, reg_value))
    return sorted(rows, key=lambda row: (str(row["family"]), float(row["reg_value"]), int(row["case_id"])))


def old_row_from_case(row: Dict[str, object], family: str, reg_value: float) -> Dict[str, object]:
    out: Dict[str, object] = {
        "schedule": "2400",
        "family": family,
        "method": row.get("method"),
        "case_id": int(float(str(row.get("case_id")))),
        "status": "completed",
        "reg_value": float(reg_value),
        "reg_label": f"a={reg_value:g}" if family == FAMILY_IDIR_FDBE else f"w={reg_value:g}",
        "summary_path": row.get("summary_path", ""),
        "source_csv": row.get("source_csv", ""),
        "source_kind": row.get("source_kind", ""),
        "stage_epochs": "200,400,800,1000",
        "stage_epochs_total": 2400,
    }
    key_map = {
        "original_snap_tre_mean_mm": ["original_snap_tre_mean_mm", "native_crop_snap_tre_mean_mm"],
        "original_snap_tre_std_mm": ["original_snap_tre_std_mm", "native_crop_snap_tre_std_mm"],
        "original_subvoxel_tre_mean_mm": ["original_subvoxel_tre_mean_mm", "native_crop_sub_tre_mean_mm"],
        "original_subvoxel_tre_std_mm": ["original_subvoxel_tre_std_mm", "native_crop_sub_tre_std_mm"],
        "reg1mm_snap_tre_mean_mm": ["reg1mm_snap_tre_mean_mm"],
        "reg1mm_snap_tre_std_mm": ["reg1mm_snap_tre_std_mm"],
        "reg1mm_subvoxel_tre_mean_mm": ["reg1mm_subvoxel_tre_mean_mm", "reg1mm_sub_tre_mean_mm"],
        "reg1mm_subvoxel_tre_std_mm": ["reg1mm_subvoxel_tre_std_mm", "reg1mm_sub_tre_std_mm"],
        "fg_logj_std": ["fg_logj_std", "fg_log_jdet_std"],
        "fg_jdet_le0_pct": ["fg_jdet_le0_pct"],
        "fg_bending_energy": ["fg_bending_energy"],
        "fg_diffusion_l2": ["fg_diffusion_l2"],
        "elapsed_sec": ["elapsed_sec"],
        "peak_mem_gb": ["peak_mem_gb"],
    }
    for out_key, candidates in key_map.items():
        out[out_key] = next((row.get(k) for k in candidates if row.get(k) not in (None, "")), "")
    return out


def select_old_case_rows(args: argparse.Namespace, family: str, method: str, reg_values: Sequence[float]) -> List[Dict[str, object]]:
    sources = []
    if family == FAMILY_PTV:
        sources.extend(
            [
                read_csv_rows(args.old_midratio_case, "old_midratio_case"),
                read_csv_rows(args.old_ptv_refined_case, "old_ptv_refined_case"),
            ]
        )
    else:
        sources.append(read_csv_rows(args.old_midratio_case, "old_midratio_case"))
    rows: List[Dict[str, object]] = []
    for reg_value in reg_values:
        for case_id in EXPECTED_CASES:
            match = None
            for source_rows in sources:
                for row in source_rows:
                    if row.get("method") == method and close(row.get("reg_value"), reg_value) and int(float(row.get("case_id"))) == case_id:
                        match = row
                        break
                if match is not None:
                    break
            if match is None:
                rows.append(
                    {
                        "schedule": "2400",
                        "family": family,
                        "method": method,
                        "case_id": case_id,
                        "status": "missing",
                        "reg_value": float(reg_value),
                        "reg_label": f"a={reg_value:g}" if family == FAMILY_IDIR_FDBE else f"w={reg_value:g}",
                    }
                )
            else:
                rows.append(old_row_from_case(match, family, reg_value))
    return sorted(rows, key=lambda row: (str(row["family"]), float(row["reg_value"]), int(row["case_id"])))


def aggregate_rows(rows: Sequence[Dict[str, object]], label: str, *, schedule: str, family: str, reg_value: float) -> Dict[str, object]:
    out: Dict[str, object] = {
        "label": label,
        "schedule": schedule,
        "family": family,
        "method": rows[0].get("method") if rows else "",
        "reg_value": float(reg_value),
        "cases": len([row for row in rows if row.get("status") == "completed"]),
        "source_paths": ";".join(sorted({str(row.get("summary_path") or row.get("source_csv") or "") for row in rows if row.get("summary_path") or row.get("source_csv")})),
    }
    for key in METRIC_KEYS:
        mean, std = base.mean_std([row.get(key) for row in rows if row.get("status") == "completed"])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def validate_new_rows(rows: Sequence[Dict[str, object]], method: str, reg_key: str, reg_values: Sequence[float]) -> List[str]:
    errors: List[str] = []
    for reg_value in reg_values:
        got_cases = sorted(
            int(row.get("case_id", -1))
            for row in rows
            if close(row.get("reg_value"), reg_value) and row.get("status") == "completed"
        )
        if got_cases != EXPECTED_CASES:
            errors.append(f"{method} {reg_key}={reg_value:g}: completed cases {got_cases} != {EXPECTED_CASES}")
    for row in rows:
        case = int(row.get("case_id", -1))
        reg_value = row.get("reg_value")
        if row.get("status") != "completed":
            errors.append(f"{method} case {case:02d} {reg_key}={reg_value}: status {row.get('status')} != completed")
            continue
        if row.get("method") != method:
            errors.append(f"case {case:02d}: method {row.get('method')} != {method}")
        if parse_stage_epochs(row.get("stage_epochs")) != EXPECTED_STAGE_EPOCHS:
            errors.append(f"case {case:02d} {reg_key}={reg_value}: stage_epochs {row.get('stage_epochs')} != 400,400,800,900")
        got_reg = base.safe_float(row.get(reg_key))
        if not close(got_reg, reg_value):
            errors.append(f"case {case:02d}: {reg_key} {got_reg} != {reg_value}")
        if "e200-400-800-1000" in str(row.get("summary_path")):
            errors.append(f"case {case:02d}: new summary path points to old schedule")
    return errors


def comparison_rows(mean_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, float, str], Dict[str, object]] = {}
    for row in mean_rows:
        by_key[(str(row["family"]), float(row["reg_value"]), str(row["schedule"]))] = row
    out: List[Dict[str, object]] = []
    for family in [FAMILY_IDIR_FDBE, FAMILY_PTV]:
        regs = sorted({float(row["reg_value"]) for row in mean_rows if row["family"] == family})
        for reg in regs:
            old = by_key.get((family, reg, "2400"), {})
            new = by_key.get((family, reg, "2500"), {})
            out.append(
                {
                    "family": family,
                    "method": new.get("method") or old.get("method"),
                    "reg_value": reg,
                    "old_schedule": "200,400,800,1000",
                    "new_schedule": "400,400,800,900",
                    "old_cases": old.get("cases", 0),
                    "new_cases": new.get("cases", 0),
                    "old_original_snap_tre_mean_mm": old.get("original_snap_tre_mean_mm_mean"),
                    "old_original_snap_tre_std_mm": old.get("original_snap_tre_mean_mm_std"),
                    "new_original_snap_tre_mean_mm": new.get("original_snap_tre_mean_mm_mean"),
                    "new_original_snap_tre_std_mm": new.get("original_snap_tre_mean_mm_std"),
                    "delta_original_snap_tre_mean_mm": base.safe_float(new.get("original_snap_tre_mean_mm_mean"))
                    - base.safe_float(old.get("original_snap_tre_mean_mm_mean")),
                    "old_fg_logj_std_mean": old.get("fg_logj_std_mean"),
                    "old_fg_logj_std_std": old.get("fg_logj_std_std"),
                    "new_fg_logj_std_mean": new.get("fg_logj_std_mean"),
                    "new_fg_logj_std_std": new.get("fg_logj_std_std"),
                    "old_fg_jdet_le0_pct_mean": old.get("fg_jdet_le0_pct_mean"),
                    "new_fg_jdet_le0_pct_mean": new.get("fg_jdet_le0_pct_mean"),
                    "old_elapsed_sec_mean": old.get("elapsed_sec_mean"),
                    "new_elapsed_sec_mean": new.get("elapsed_sec_mean"),
                    "old_peak_mem_gb_mean": old.get("peak_mem_gb_mean"),
                    "new_peak_mem_gb_mean": new.get("peak_mem_gb_mean"),
                    "old_source_paths": old.get("source_paths"),
                    "new_source_paths": new.get("source_paths"),
                }
            )
    return out


def write_report(review_root: Path, mean_rows: Sequence[Dict[str, object]], comp_rows: Sequence[Dict[str, object]], errors: Sequence[str]) -> None:
    table = []
    for row in comp_rows:
        prefix = "IDIR" if row["family"] == FAMILY_IDIR_FDBE else "pTV"
        reg_label = f"a={row['reg_value']:g}" if row["family"] == FAMILY_IDIR_FDBE else f"w={row['reg_value']:g}"
        table.append(
            [
                prefix,
                reg_label,
                f"{fmt(row['old_original_snap_tre_mean_mm'])} +/- {fmt(row['old_original_snap_tre_std_mm'])}",
                f"{fmt(row['new_original_snap_tre_mean_mm'])} +/- {fmt(row['new_original_snap_tre_std_mm'])}",
                fmt(row["delta_original_snap_tre_mean_mm"]),
                f"{fmt(row['old_fg_logj_std_mean'], 6)} -> {fmt(row['new_fg_logj_std_mean'], 6)}",
                f"{fmt(row['old_fg_jdet_le0_pct_mean'], 4)} -> {fmt(row['new_fg_jdet_le0_pct_mean'], 4)}",
                f"{fmt(row['old_elapsed_sec_mean'], 1)} -> {fmt(row['new_elapsed_sec_mean'], 1)}",
                f"{fmt(row['old_peak_mem_gb_mean'], 2)} -> {fmt(row['new_peak_mem_gb_mean'], 2)}",
            ]
        )
    lines = [
        "# LungCT Figure 2 MR-D-BSCP 2500-Step Schedule Refresh",
        "",
        "## Validation",
        "",
        f"- Errors: `{list(errors)}`.",
        "- New schedule: `400,400,800,900`.",
        "- Old comparison schedule: `200,400,800,1000`.",
        "- No Figure 2 plotting source was updated by this report.",
        "",
        "## Schedule Comparison",
        "",
        markdown_table(
            [
                "Family",
                "Reg",
                "Old TRE",
                "New TRE",
                "Delta",
                "FG stdLogJ old -> new",
                "J<=0% old -> new",
                "Elapsed old -> new",
                "Peak GB old -> new",
            ],
            table,
        ),
        "",
        "## Artifacts",
        "",
        "- `case_rows.csv/json`: old and new case rows.",
        "- `mean_rows.csv/json`: old and new aggregate rows.",
        "- `schedule_comparison_rows.csv/json`: old-vs-new rows with source paths.",
    ]
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expected_stage_epochs = parse_stage_epochs(args.stage_epochs)
    if expected_stage_epochs != EXPECTED_STAGE_EPOCHS:
        raise ValueError(f"This refresh expects {EXPECTED_STAGE_EPOCHS}, got {expected_stage_epochs}")
    args.review_root.mkdir(parents=True, exist_ok=True)

    alphas = parse_float_list(args.alphas)
    ptv_weights = parse_float_list(args.ptv_weights)
    new_idir = collect_new_rows(args, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE, FAMILY_IDIR_FDBE, alphas)
    new_ptv = collect_new_rows(args, METHOD_DIRECT_CP_32_PTV, FAMILY_PTV, ptv_weights)
    old_idir = select_old_case_rows(args, FAMILY_IDIR_FDBE, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE, alphas)
    old_ptv = select_old_case_rows(args, FAMILY_PTV, METHOD_DIRECT_CP_32_PTV, ptv_weights)

    case_rows = old_idir + old_ptv + new_idir + new_ptv
    mean_rows: List[Dict[str, object]] = []
    for family, regs, old_rows, new_rows in [
        (FAMILY_IDIR_FDBE, alphas, old_idir, new_idir),
        (FAMILY_PTV, ptv_weights, old_ptv, new_ptv),
    ]:
        for reg in regs:
            old_group = [row for row in old_rows if close(row.get("reg_value"), reg)]
            new_group = [row for row in new_rows if close(row.get("reg_value"), reg)]
            mean_rows.append(aggregate_rows(old_group, f"2400 {family} {reg:g}", schedule="2400", family=family, reg_value=reg))
            mean_rows.append(aggregate_rows(new_group, f"2500 {family} {reg:g}", schedule="2500", family=family, reg_value=reg))

    comp_rows = comparison_rows(mean_rows)
    errors = []
    errors.extend(validate_new_rows(new_idir, METHOD_DIRECT_CP_32_GLOBAL_NCC_FD_BE, "alpha", alphas))
    errors.extend(validate_new_rows(new_ptv, METHOD_DIRECT_CP_32_PTV, "ptv_isotv_weight", ptv_weights))
    for row in mean_rows:
        if int(row.get("cases") or 0) != 10:
            errors.append(f"{row.get('label')}: cases {row.get('cases')} != 10")

    c.write_csv(args.review_root / "case_rows.csv", case_rows)
    c.write_json(args.review_root / "case_rows.json", case_rows)
    c.write_csv(args.review_root / "mean_rows.csv", mean_rows)
    c.write_json(args.review_root / "mean_rows.json", mean_rows)
    c.write_csv(args.review_root / "schedule_comparison_rows.csv", comp_rows)
    c.write_json(args.review_root / "schedule_comparison_rows.json", comp_rows)
    c.write_json(
        args.review_root / "manifest.json",
        {
            "errors": errors,
            "idir_alphas": alphas,
            "ptv_weights": ptv_weights,
            "stage_epochs": DEFAULT_STAGE_EPOCHS,
            "idir_run_root": str(args.idir_run_root),
            "ptv_run_root": str(args.ptv_run_root),
            "ptv_anchor_root": str(args.ptv_anchor_root),
        },
    )
    write_report(args.review_root, mean_rows, comp_rows, errors)
    print(
        json.dumps(
            {
                "case_rows": len(case_rows),
                "mean_rows": len(mean_rows),
                "comparison_rows": len(comp_rows),
                "errors": errors,
                "report": str(args.review_root / "review_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
