#!/usr/bin/env python
"""Aggregate LungCT 400-400-800-900 schedule-standardization pilots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import aggregate_report as base
import common as c
from launch_lungct_2500_schedule_pilots import (
    DEFAULT_ALPHA,
    DEFAULT_MASKED_RUN_ROOT,
    DEFAULT_PTV_RUN_ROOT,
    DEFAULT_PTV_WEIGHT,
    DEFAULT_STAGE_EPOCHS,
    PILOT_MASKED,
    PILOT_PTV,
    parse_pilots,
)
from run_one import METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE, METHOD_DIRECT_CP_32_PTV


EXPECTED_CASES = list(range(1, 11))
EXPECTED_STAGE_EPOCHS = [400, 400, 800, 900]
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

DEFAULT_PTV_REVIEW_ROOT = (
    c.REPO_ROOT / "experiments/2026-08-12_lungct_directcp32_ptvlcc_w0p11_schedule_standardization"
)
DEFAULT_MASKED_REVIEW_ROOT = (
    c.REPO_ROOT
    / "experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10"
)
DEFAULT_DECISION_ROOT = c.REPO_ROOT / "experiments/2026-08-12_lungct_2500_schedule_standardization_pilot"
DEFAULT_PTV_BASELINE = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.csv"
DEFAULT_MASKED_BASELINE = (
    c.REPO_ROOT / "experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep/mean_rows.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptv-run-root", type=Path, default=DEFAULT_PTV_RUN_ROOT)
    parser.add_argument("--masked-run-root", type=Path, default=DEFAULT_MASKED_RUN_ROOT)
    parser.add_argument("--ptv-review-root", type=Path, default=DEFAULT_PTV_REVIEW_ROOT)
    parser.add_argument("--masked-review-root", type=Path, default=DEFAULT_MASKED_REVIEW_ROOT)
    parser.add_argument("--decision-root", type=Path, default=DEFAULT_DECISION_ROOT)
    parser.add_argument("--ptv-baseline", type=Path, default=DEFAULT_PTV_BASELINE)
    parser.add_argument("--masked-baseline", type=Path, default=DEFAULT_MASKED_BASELINE)
    parser.add_argument("--stage-epochs", type=str, default=DEFAULT_STAGE_EPOCHS)
    parser.add_argument("--ptv-isotv-weight", type=float, default=DEFAULT_PTV_WEIGHT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--pilots", type=str, default=f"{PILOT_PTV},{PILOT_MASKED}")
    return parser.parse_args()


def fmt(value: object, digits: int = 3) -> str:
    return base.fmt(value, digits)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return base.markdown_table(list(headers), [[str(v) for v in row] for row in rows])


def as_stage_epochs(value: object) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v) for v in value.replace(" ", "").split(",") if v]
    return []


def row_from_summary(path: Path, payload: Dict[str, object], pilot: str) -> Dict[str, object]:
    row = base.flatten_summary(path, payload)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    out: Dict[str, object] = {
        "pilot": pilot,
        "method": payload.get("method"),
        "case_id": int(payload.get("case_id") or row.get("case_id")),
        "status": payload.get("status"),
        "summary_path": str(path),
        "stage_epochs": ",".join(str(v) for v in as_stage_epochs(training.get("stage_epochs"))),
        "stage_epochs_total": sum(as_stage_epochs(training.get("stage_epochs"))),
        "lr": ",".join(str(v) for v in training.get("stage_lr", []))
        if isinstance(training.get("stage_lr"), list)
        else training.get("lr", ""),
        "ptv_isotv_weight": training.get("ptv_isotv_weight"),
        "alpha": training.get("alpha"),
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
            "original_subvoxel_tre_median_mm": row.get("native_crop_sub_tre_median_mm"),
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


def collect_new_rows(root: Path, method: str, pilot: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((root / method).glob("*/summary.json")):
        payload = json.loads(path.read_text())
        rows.append(row_from_summary(path, payload, pilot))
    return sorted(rows, key=lambda row: int(row["case_id"]))


def aggregate_rows(rows: Sequence[Dict[str, object]], label: str) -> Dict[str, object]:
    out: Dict[str, object] = {"label": label, "cases": len(rows)}
    for key in METRIC_KEYS:
        mean, std = base.mean_std([row.get(key) for row in rows])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def validate_rows(
    rows: Sequence[Dict[str, object]],
    *,
    method: str,
    stage_epochs: Sequence[int],
    reg_key: str,
    reg_value: float,
) -> List[str]:
    errors: List[str] = []
    got_cases = sorted(int(row.get("case_id", -1)) for row in rows if row.get("status") == "completed")
    if got_cases != EXPECTED_CASES:
        errors.append(f"completed cases {got_cases} != {EXPECTED_CASES}")
    for row in rows:
        case = int(row.get("case_id", -1))
        if row.get("status") != "completed":
            errors.append(f"case {case:02d}: status {row.get('status')} != completed")
        if row.get("method") != method:
            errors.append(f"case {case:02d}: method {row.get('method')} != {method}")
        if as_stage_epochs(row.get("stage_epochs")) != list(stage_epochs):
            errors.append(f"case {case:02d}: stage_epochs {row.get('stage_epochs')} != {','.join(map(str, stage_epochs))}")
        got_reg = base.safe_float(row.get(reg_key))
        if abs(got_reg - float(reg_value)) > 1e-8 * max(1.0, abs(float(reg_value))):
            errors.append(f"case {case:02d}: {reg_key} {got_reg} != {reg_value}")
        if "e200-400-800-1000" in str(row.get("summary_path")):
            errors.append(f"case {case:02d}: summary_path still points to old 2400 schedule")
    return errors


def read_ptv_baseline(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") != METHOD_DIRECT_CP_32_PTV:
                continue
            rows.append(
                {
                    "case_id": int(row["case_id"]),
                    "stage_epochs": row.get("stage_epochs", ""),
                    "original_snap_tre_mean_mm": row.get("native_crop_snap_tre_mean_mm"),
                    "original_subvoxel_tre_mean_mm": row.get("native_crop_sub_tre_mean_mm"),
                    "reg1mm_snap_tre_mean_mm": row.get("reg1mm_snap_tre_mean_mm"),
                    "reg1mm_subvoxel_tre_mean_mm": row.get("reg1mm_sub_tre_mean_mm"),
                    "fg_logj_std": row.get("fg_log_jdet_std"),
                    "fg_jdet_le0_pct": row.get("fg_jdet_le0_pct"),
                    "fg_bending_energy": row.get("fg_bending_energy"),
                    "fg_diffusion_l2": row.get("fg_diffusion_l2"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "peak_mem_gb": row.get("peak_mem_gb"),
                    "summary_path": row.get("summary_path", ""),
                }
            )
    return sorted(rows, key=lambda row: int(row["case_id"]))


def read_masked_baseline_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_pilot_report(
    *,
    review_root: Path,
    title: str,
    new_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    comparison_rows: Sequence[Dict[str, object]],
    validation_errors: Sequence[str],
    baseline_label: str,
    new_label: str,
) -> None:
    review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(review_root / "case_rows.csv", list(new_rows))
    c.write_json(review_root / "case_rows.json", list(new_rows))
    c.write_csv(review_root / "baseline_case_rows.csv", list(baseline_rows))
    c.write_json(review_root / "baseline_case_rows.json", list(baseline_rows))
    c.write_csv(review_root / "comparison_rows.csv", list(comparison_rows))
    c.write_json(review_root / "comparison_rows.json", list(comparison_rows))

    case_table = []
    baseline_by_case = {int(row["case_id"]): row for row in baseline_rows if "case_id" in row}
    for row in new_rows:
        case_id = int(row["case_id"])
        old = baseline_by_case.get(case_id, {})
        new_tre = base.safe_float(row.get("original_snap_tre_mean_mm"))
        old_tre = base.safe_float(old.get("original_snap_tre_mean_mm"))
        case_table.append(
            [
                f"{case_id:02d}",
                fmt(old_tre),
                fmt(new_tre),
                fmt(new_tre - old_tre),
                fmt(row.get("fg_logj_std"), 6),
                fmt(row.get("fg_jdet_le0_pct"), 4),
                fmt(row.get("elapsed_sec"), 1),
                fmt(row.get("peak_mem_gb"), 2),
            ]
        )

    mean_table = []
    for row in comparison_rows:
        mean_table.append(
            [
                row["label"],
                f"{fmt(row.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(row.get('original_snap_tre_mean_mm_std'))}",
                f"{fmt(row.get('fg_logj_std_mean'), 6)} +/- {fmt(row.get('fg_logj_std_std'), 6)}",
                f"{fmt(row.get('fg_jdet_le0_pct_mean'), 4)} +/- {fmt(row.get('fg_jdet_le0_pct_std'), 4)}",
                f"{fmt(row.get('elapsed_sec_mean'), 1)} +/- {fmt(row.get('elapsed_sec_std'), 1)}",
                f"{fmt(row.get('peak_mem_gb_mean'), 2)} +/- {fmt(row.get('peak_mem_gb_std'), 2)}",
            ]
        )

    lines = [
        f"# {title}",
        "",
        "## Validation",
        "",
        f"- Errors: `{list(validation_errors)}`.",
        f"- New rows: `{len(new_rows)}`.",
        f"- New stage epochs: `{DEFAULT_STAGE_EPOCHS}`.",
        "",
        "## Aggregate Comparison",
        "",
        markdown_table(
            ["Row", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0% mean +/- SD", "Elapsed sec", "Peak GB"],
            mean_table,
        ),
        "",
        "## Case TRE Comparison",
        "",
        markdown_table(
            ["Case", f"{baseline_label} TRE", f"{new_label} TRE", "Delta", "New FG stdLogJ", "New FG J<=0%", "Elapsed sec", "Peak GB"],
            case_table,
        ),
        "",
        "## Artifacts",
        "",
        "- `case_rows.csv`, `case_rows.json`",
        "- `baseline_case_rows.csv`, `baseline_case_rows.json`",
        "- `comparison_rows.csv`, `comparison_rows.json`",
    ]
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")
    c.write_json(
        review_root / "manifest.json",
        {
            "title": title,
            "validation_errors": list(validation_errors),
            "new_rows": len(new_rows),
            "baseline_rows": len(baseline_rows),
            "stage_epochs": DEFAULT_STAGE_EPOCHS,
        },
    )


def write_decision_report(
    decision_root: Path,
    *,
    pilots: Sequence[str],
    ptv_comparison: Sequence[Dict[str, object]],
    masked_comparison: Sequence[Dict[str, object]],
    ptv_errors: Sequence[str],
    masked_errors: Sequence[str],
) -> None:
    decision_root.mkdir(parents=True, exist_ok=True)
    c.write_json(
        decision_root / "manifest.json",
        {
            "pilots": pilots,
            "ptv_validation_errors": list(ptv_errors),
            "masked_validation_errors": list(masked_errors),
            "stage_epochs": DEFAULT_STAGE_EPOCHS,
        },
    )
    c.write_csv(decision_root / "pilot_comparison_rows.csv", list(ptv_comparison) + list(masked_comparison))
    lines = [
        "# LungCT 2500-Step Schedule Standardization Pilot",
        "",
        "## Status",
        "",
        f"- pTVReg-loss validation errors: `{list(ptv_errors)}`.",
        f"- Corrected IDIR-loss validation errors: `{list(masked_errors)}`.",
        "- No full sweep has been launched by this pilot.",
        "",
        "## pTVReg-Loss Pilot",
        "",
        markdown_table(
            ["Row", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0%"],
            [
                [
                    row["label"],
                    f"{fmt(row.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(row.get('original_snap_tre_mean_mm_std'))}",
                    f"{fmt(row.get('fg_logj_std_mean'), 6)} +/- {fmt(row.get('fg_logj_std_std'), 6)}",
                    f"{fmt(row.get('fg_jdet_le0_pct_mean'), 4)} +/- {fmt(row.get('fg_jdet_le0_pct_std'), 4)}",
                ]
                for row in ptv_comparison
            ],
        ),
        "",
        "## Corrected IDIR-Loss Pilot",
        "",
        markdown_table(
            ["Row", "TRE mean +/- SD", "FG stdLogJ mean +/- SD", "FG J<=0%"],
            [
                [
                    row["label"],
                    f"{fmt(row.get('original_snap_tre_mean_mm_mean'))} +/- {fmt(row.get('original_snap_tre_mean_mm_std'))}",
                    f"{fmt(row.get('fg_logj_std_mean'), 6)} +/- {fmt(row.get('fg_logj_std_std'), 6)}",
                    f"{fmt(row.get('fg_jdet_le0_pct_mean'), 4)} +/- {fmt(row.get('fg_jdet_le0_pct_std'), 4)}",
                ]
                for row in masked_comparison
            ],
        ),
        "",
        "## Decision",
        "",
        "- pTVReg-loss pilot: schedule-stable but not better than the existing",
        "  2400-step `w=0.11` row; useful as a standardized replacement only if",
        "  all LungCT rows must use exactly 2500 steps.",
        "- Corrected IDIR-loss pilot: `alpha=33333.333333` is smoother than",
        "  `alpha=10000` but has worse TRE and less smoothness than the existing",
        "  2400-step `alpha=100000` row; it is not the best corrected-IDIR point",
        "  from the current evidence.",
        "- Stop here as requested. Do not launch full 2500-step sweeps without a",
        "  new decision.",
    ]
    (decision_root / "decision_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    pilots = parse_pilots(args.pilots)
    expected_stage_epochs = as_stage_epochs(args.stage_epochs)
    if expected_stage_epochs != EXPECTED_STAGE_EPOCHS:
        raise ValueError(f"This pilot expects {EXPECTED_STAGE_EPOCHS}, got {expected_stage_epochs}")

    ptv_new: List[Dict[str, object]] = []
    ptv_errors: List[str] = []
    ptv_comparison: List[Dict[str, object]] = []
    if PILOT_PTV in pilots:
        ptv_new = collect_new_rows(args.ptv_run_root, METHOD_DIRECT_CP_32_PTV, PILOT_PTV)
        ptv_old = read_ptv_baseline(args.ptv_baseline)
        ptv_errors = validate_rows(
            ptv_new,
            method=METHOD_DIRECT_CP_32_PTV,
            stage_epochs=EXPECTED_STAGE_EPOCHS,
            reg_key="ptv_isotv_weight",
            reg_value=float(args.ptv_isotv_weight),
        )
        ptv_comparison = [
            aggregate_rows(ptv_old, "2400-step pTVReg-loss baseline w=0.11"),
            aggregate_rows(ptv_new, "2500-step pTVReg-loss pilot w=0.11"),
        ]
        write_pilot_report(
            review_root=args.ptv_review_root,
            title="LungCT MR-D-BSCP pTVReg-Loss Schedule Standardization",
            new_rows=ptv_new,
            baseline_rows=ptv_old,
            comparison_rows=ptv_comparison,
            validation_errors=ptv_errors,
            baseline_label="2400",
            new_label="2500",
        )

    masked_new: List[Dict[str, object]] = []
    masked_errors: List[str] = []
    masked_comparison: List[Dict[str, object]] = []
    if PILOT_MASKED in pilots:
        masked_new = collect_new_rows(args.masked_run_root, METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE, PILOT_MASKED)
        masked_errors = validate_rows(
            masked_new,
            method=METHOD_DIRECT_CP_32_MASKED_NCC_BSPLINE_BE,
            stage_epochs=EXPECTED_STAGE_EPOCHS,
            reg_key="alpha",
            reg_value=float(args.alpha),
        )
        masked_baselines = read_masked_baseline_rows(args.masked_baseline)
        masked_baseline_by_alpha = {float(row["alpha"]): row for row in masked_baselines}
        masked_comparison = [
            {**masked_baseline_by_alpha[10000.0], "label": "2400-step corrected IDIR-loss alpha=10000"},
            {**masked_baseline_by_alpha[100000.0], "label": "2400-step corrected IDIR-loss alpha=100000"},
            aggregate_rows(masked_new, "2500-step corrected IDIR-loss alpha=33333.333333"),
        ]
        write_pilot_report(
            review_root=args.masked_review_root,
            title="LungCT Corrected MR-D-BSCP IDIR-Loss Schedule Standardization",
            new_rows=masked_new,
            baseline_rows=[],
            comparison_rows=masked_comparison,
            validation_errors=masked_errors,
            baseline_label="2400",
            new_label="2500",
        )

    write_decision_report(
        args.decision_root,
        pilots=pilots,
        ptv_comparison=ptv_comparison,
        masked_comparison=masked_comparison,
        ptv_errors=ptv_errors,
        masked_errors=masked_errors,
    )
    errors = list(ptv_errors) + list(masked_errors)
    print(
        json.dumps(
            {
                "ptv_rows": len(ptv_new),
                "masked_rows": len(masked_new),
                "errors": errors,
                "decision_report": str(args.decision_root / "decision_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
