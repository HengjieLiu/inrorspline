#!/usr/bin/env python
"""Aggregate the OASIS MR-SINR 8/4/2 one-case pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_mr_sinr842_revision_pilot_v1")
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-08-07_oasis_mr_sinr842_revision"
EXPECTED_METHODS = {"mr_sinr842_composed", "mr_sinr842_requery"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    return c.read_csv(path) if path.exists() and path.read_text().strip() else []


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def first_present(row: Mapping[str, object], keys: Sequence[str]) -> float:
    for key in keys:
        if key in row and finite(row[key]):
            return float(row[key])
    return float("nan")


def collect_metric_file(summary_rows: Sequence[Mapping[str, object]], filename: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summary_rows:
        run_dir = Path(str(summary["run_dir"]))
        for metric_row in read_rows(run_dir / filename):
            rows.append(
                {
                    "method": summary.get("method"),
                    "method_key": summary.get("method_key"),
                    "run_name": summary.get("run_name"),
                    **metric_row,
                }
            )
    return rows


def load_summary_rows(out_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_root.glob("*/*/summary.json")):
        payload = c.read_json(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("method")) not in EXPECTED_METHODS:
            continue
        rows.append({**payload, "summary_path": str(path), "done_exists": (path.parent / "DONE").exists()})
    return rows


def diagnostic_status(summary: Mapping[str, object]) -> Dict[str, object]:
    run_dir = Path(str(summary["run_dir"]))
    method = str(summary.get("method"))
    if method == "mr_sinr842_composed":
        rows = read_rows(run_dir / "composition_diagnostics.csv")
        final = rows[-1] if rows else {}
        return {
            "diagnostic_file": "composition_diagnostics.csv",
            "diagnostic_rows": len(rows),
            "diagnostic_required_rows": 3,
            "diagnostic_pass": len(rows) >= 3 and finite(final.get("image_max_abs")),
            "diagnostic_primary": as_float(final.get("image_max_abs")),
            "diagnostic_primary_name": "final_composed_vs_sequential_image_max_abs",
        }
    rows = read_rows(run_dir / "stage_transition_diagnostics.csv")
    max_jump = max((as_float(row.get("jump_max_mag")) for row in rows), default=float("nan"))
    return {
        "diagnostic_file": "stage_transition_diagnostics.csv",
        "diagnostic_rows": len(rows),
        "diagnostic_required_rows": 2,
        "diagnostic_pass": len(rows) >= 2 and finite(max_jump),
        "diagnostic_primary": max_jump,
        "diagnostic_primary_name": "max_requery_transition_jump_mag",
    }


def comparison_rows(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summary_rows:
        diag = diagnostic_status(summary)
        dense_image_loss = as_float(summary.get("dense_image_loss"))
        row: Dict[str, object] = {
            "method": summary.get("method"),
            "method_key": summary.get("method_key"),
            "run_name": summary.get("run_name"),
            "run_dir": summary.get("run_dir"),
            "case_rank": summary.get("case_rank"),
            "original_case_index": summary.get("original_case_index"),
            "fixed_id": summary.get("fixed_id"),
            "moving_id": summary.get("moving_id"),
            "alpha": summary.get("alpha"),
            "lr": summary.get("lr"),
            "epochs": summary.get("epochs"),
            "stage_full_cps": summary.get("stage_full_cps"),
            "stage_downsample": summary.get("stage_downsample"),
            "stage_epochs": summary.get("stage_epochs"),
            "stage_actual_epochs": summary.get("stage_actual_epochs"),
            "dice_mean": summary.get("dice_mean"),
            "hd95_mean": summary.get("hd95_mean"),
            "asd_symmetric_mean": summary.get("asd_symmetric_mean"),
            "final_image_ncc": -dense_image_loss if finite(dense_image_loss) else float("nan"),
            "dense_image_loss": summary.get("dense_image_loss"),
            "bending_energy": first_present(
                summary,
                [
                    "be_native_analytic_fixed_fg_voxel",
                    "be_bspline_analytic_fixed_fg_voxel",
                    "be_fd_forward_fixed_fg_voxel",
                    "be_native_analytic_full_voxel",
                    "be_fd_forward_full_voxel",
                ],
            ),
            "diffusion": first_present(
                summary,
                [
                    "diffusion_native_analytic_fixed_fg_voxel",
                    "diffusion_bspline_analytic_fixed_fg_voxel",
                    "diffusion_fd_forward_fixed_fg_voxel",
                    "diffusion_l2_fg",
                ],
            ),
            "log_jdet_std_fg": summary.get("log_jdet_std_fg"),
            "jdet_le0_pct_fg": summary.get("jdet_le0_pct_fg"),
            "training_elapsed_sec": summary.get("training_elapsed_sec"),
            "training_peak_mem_gb": summary.get("training_peak_mem_gb"),
            "total_elapsed_sec": summary.get("total_elapsed_sec"),
            "final_loss": summary.get("final_loss"),
            "final_data_loss": summary.get("final_data_loss"),
            "final_reg_loss": summary.get("final_reg_loss"),
            "multistage_strategy": summary.get("multistage_strategy"),
            "field_composition": summary.get("field_composition"),
            "final_field_is_single_bspline": summary.get("final_field_is_single_bspline"),
            "done_exists": summary.get("done_exists"),
            "finite_core_metrics": all(
                finite(summary.get(key))
                for key in ["dice_mean", "hd95_mean", "asd_symmetric_mean", "dense_image_loss"]
            ),
            **diag,
        }
        row["technical_pass"] = bool(row["done_exists"] and row["finite_core_metrics"] and row["diagnostic_pass"])
        rows.append(row)
    return rows


def decision_row(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    by_method = {str(row.get("method")): row for row in rows}
    composed = by_method.get("mr_sinr842_composed")
    requery = by_method.get("mr_sinr842_requery")
    complete = composed is not None and requery is not None
    technical_pass = bool(complete and composed.get("technical_pass") and requery.get("technical_pass")) if complete else False
    dice_delta = (
        as_float(composed.get("dice_mean")) - as_float(requery.get("dice_mean"))
        if complete
        else float("nan")
    )
    hd95_delta = (
        as_float(composed.get("hd95_mean")) - as_float(requery.get("hd95_mean"))
        if complete
        else float("nan")
    )
    logj_delta = (
        as_float(composed.get("log_jdet_std_fg")) - as_float(requery.get("log_jdet_std_fg"))
        if complete
        else float("nan")
    )
    fold_delta = (
        as_float(composed.get("jdet_le0_pct_fg")) - as_float(requery.get("jdet_le0_pct_fg"))
        if complete
        else float("nan")
    )
    not_clearly_worse = bool(
        technical_pass
        and finite(dice_delta)
        and finite(hd95_delta)
        and dice_delta >= -0.005
        and hd95_delta <= 0.10
        and (not finite(logj_delta) or logj_delta <= 0.05)
        and (not finite(fold_delta) or fold_delta <= 0.5)
    )
    return {
        "completed_methods": sorted(by_method),
        "expected_methods": sorted(EXPECTED_METHODS),
        "complete": complete,
        "technical_pass": technical_pass,
        "composed_minus_requery_dice": dice_delta,
        "composed_minus_requery_hd95": hd95_delta,
        "composed_minus_requery_log_jdet_std_fg": logj_delta,
        "composed_minus_requery_jdet_le0_pct_fg": fold_delta,
        "promote_composed_candidate": not_clearly_worse,
        "decision_rule": (
            "Promote only if both runs pass diagnostics and composed is not clearly worse: "
            "Dice >= requery-0.005, HD95 <= requery+0.10, logJ std <= requery+0.05, "
            "and J<=0 pct <= requery+0.5."
        ),
    }


def fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "NA"


def write_report(review_root: Path, rows: Sequence[Mapping[str, object]], decision: Mapping[str, object]) -> None:
    lines = [
        "# OASIS MR-SINR 8/4/2 Pilot Summary",
        "",
        f"- Complete: `{decision.get('complete')}`.",
        f"- Technical pass: `{decision.get('technical_pass')}`.",
        f"- Promote composed candidate: `{decision.get('promote_composed_candidate')}`.",
        f"- Decision rule: {decision.get('decision_rule')}",
        "",
        "## Method Rows",
        "",
        "| Method | Dice | HD95 | ASD | NCC | BE | Diffusion | logJ std fg | J<=0 pct fg | Runtime s | Diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method")),
                    fmt(row.get("dice_mean")),
                    fmt(row.get("hd95_mean")),
                    fmt(row.get("asd_symmetric_mean")),
                    fmt(row.get("final_image_ncc")),
                    fmt(row.get("bending_energy"), 6),
                    fmt(row.get("diffusion"), 6),
                    fmt(row.get("log_jdet_std_fg")),
                    fmt(row.get("jdet_le0_pct_fg")),
                    fmt(row.get("training_elapsed_sec"), 1),
                    f"{row.get('diagnostic_primary_name')}={fmt(row.get('diagnostic_primary'), 6)}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- Composed - requery Dice: `{fmt(decision.get('composed_minus_requery_dice'))}`.",
            f"- Composed - requery HD95: `{fmt(decision.get('composed_minus_requery_hd95'))}`.",
            f"- Composed - requery foreground log-J std: `{fmt(decision.get('composed_minus_requery_log_jdet_std_fg'))}`.",
            f"- Composed - requery foreground J<=0 pct: `{fmt(decision.get('composed_minus_requery_jdet_le0_pct_fg'))}`.",
        ]
    )
    (review_root / "pilot_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.review_root.mkdir(parents=True, exist_ok=True)
    summaries = load_summary_rows(args.out_root)
    rows = comparison_rows(summaries)
    decision = decision_row(rows)
    c.write_csv(args.review_root / "pilot_comparison_summary.csv", rows)
    c.write_json(args.review_root / "pilot_comparison_summary.json", rows)
    c.write_json(args.review_root / "pilot_decision.json", decision)
    c.write_csv(args.review_root / "case_metrics_combined.csv", collect_metric_file(summaries, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_combined.csv", collect_metric_file(summaries, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_combined.csv", collect_metric_file(summaries, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_combined.csv", collect_metric_file(summaries, "diffusion_metrics.csv"))
    c.write_csv(args.review_root / "stage_summaries_combined.csv", collect_metric_file(summaries, "stage_summaries.csv"))
    c.write_csv(args.review_root / "composition_diagnostics_combined.csv", collect_metric_file(summaries, "composition_diagnostics.csv"))
    c.write_csv(args.review_root / "stage_transition_diagnostics_combined.csv", collect_metric_file(summaries, "stage_transition_diagnostics.csv"))
    write_report(args.review_root, rows, decision)
    print(json.dumps({"rows": len(rows), "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
