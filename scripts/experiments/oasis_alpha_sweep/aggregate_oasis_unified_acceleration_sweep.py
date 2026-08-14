#!/usr/bin/env python
"""Aggregate the unified OASIS acceleration sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from launch_first10_be_2500e import parse_case_ranks  # noqa: E402
from launch_oasis_unified_acceleration_sweep import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    VARIANT,
    candidate_map,
    phase_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--phase", choices=["primary", "cps4", "all", "smoke"], default="primary")
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--candidates", type=str, default="phase")
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(rows: Iterable[Mapping[str, object]], key: str) -> List[float]:
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def mean(rows: Iterable[Mapping[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return float(np.mean(values)) if values else float("nan")


def std(rows: Iterable[Mapping[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else float("nan")


def max_value(rows: Iterable[Mapping[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return float(np.max(values)) if values else float("nan")


def selected_candidate_names(args: argparse.Namespace) -> List[str]:
    if str(args.candidates).strip() not in {"", "phase", "all"}:
        return [token.strip() for token in str(args.candidates).split(",") if token.strip()]
    if str(args.phase) == "all":
        return [candidate.name for candidate in candidate_map().values()]
    if str(args.phase) == "smoke":
        return [candidate.name for candidate in candidate_map().values()]
    return [candidate.name for candidate in phase_candidates(str(args.phase))]


def collect_rows(args: argparse.Namespace, names: Sequence[str], ranks: Sequence[int]) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    known = candidate_map()
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    rank_set = {int(rank) for rank in ranks}
    name_set = set(names)
    phase = str(args.phase)
    for path in sorted(Path(args.out_root).glob("*/*/summary.json")):
        run_name = path.parent.name
        if phase == "all":
            if "unified_primary_" not in run_name and "unified_cps4_" not in run_name:
                continue
        elif f"unified_{phase}_" not in run_name:
            continue
        summary = json.loads(path.read_text())
        candidate_name = str(summary.get("runtime_candidate") or summary.get("candidate") or "")
        if candidate_name not in name_set:
            continue
        rank = int(as_float(summary.get("case_rank")))
        if rank not in rank_set:
            continue
        candidate = known[candidate_name]
        timing_path = path.parent / "timing.json"
        timing = {}
        if timing_path.exists():
            try:
                timing = json.loads(timing_path.read_text())
            except Exception:
                timing = {}
        row = {
            "candidate": candidate.name,
            "phase": candidate.phase,
            "method": candidate.method,
            "method_label": candidate.method_label,
            "family": candidate.family,
            "baseline_name": candidate.baseline_name,
            "template": candidate.template,
            "description": candidate.description,
            "lr_policy": candidate.lr_policy,
            "is_baseline": bool(candidate.is_baseline),
            "lr": float(summary.get("lr", candidate.lr)),
            "epochs_configured": int(as_float(summary.get("epochs", candidate.epochs))),
            "early_stop_mode": str(summary.get("stage_early_stop_mode", candidate.early_stop_mode)),
            "case_rank": rank,
            "warm_status": "cold_rank0" if rank == min(rank_set) else "warm_persistent",
            "fixed_id": summary.get("fixed_id"),
            "moving_id": summary.get("moving_id"),
            "original_case_index": summary.get("original_case_index"),
            "dice_mean": as_float(summary.get("dice_mean")),
            "hd95_mean": as_float(summary.get("hd95_mean")),
            "asd_symmetric_mean": as_float(summary.get("asd_symmetric_mean")),
            "asd_warped_to_fixed_mean": as_float(summary.get("asd_warped_to_fixed_mean")),
            "total_elapsed_sec": as_float(summary.get("total_elapsed_sec")),
            "full_registration_time_sec": as_float(summary.get("full_registration_time_sec", summary.get("total_elapsed_sec"))),
            "training_elapsed_sec": as_float(summary.get("training_elapsed_sec")),
            "loaded_images_to_dvf_file_sec": as_float(summary.get("loaded_images_to_dvf_file_sec")),
            "full_script_to_dvf_file_sec": as_float(summary.get("full_script_to_dvf_file_sec")),
            "post_dvf_evaluation_sec": as_float(summary.get("post_dvf_evaluation_sec")),
            "training_peak_mem_gb": as_float(summary.get("training_peak_mem_gb")),
            "actual_epochs": as_float(summary.get("actual_epochs")),
            "stage_actual_epochs": summary.get("stage_actual_epochs"),
            "summary_path": str(path),
            "timing_path": str(timing_path) if timing_path.exists() else "",
            "timing_event_count": len(timing.get("events", [])) if isinstance(timing, dict) else "",
            "run_dir": str(path.parent),
        }
        rows.append(row)
    observed = {(row["candidate"], int(row["case_rank"])) for row in rows}
    for name in names:
        for rank in ranks:
            if (name, int(rank)) not in observed:
                missing.append({"candidate": name, "case_rank": int(rank)})
    rows.sort(key=lambda row: (str(row["family"]), str(row["method_label"]), str(row["candidate"]), int(row["case_rank"])))
    return rows, missing


def dice_grade(dice_drop: float) -> str:
    if not math.isfinite(dice_drop):
        return "missing"
    if dice_drop <= 0.0:
        return "no_loss_or_gain"
    if dice_drop <= 0.02:
        return "small_0_to_0p02"
    if dice_drop <= 0.05:
        return "moderate_0p02_to_0p05"
    return "large_gt_0p05"


def summarize_candidates(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["candidate"])].append(row)
    known = candidate_map()
    summaries: List[Dict[str, object]] = []
    for name, group in groups.items():
        candidate = known[name]
        cold = [row for row in group if str(row.get("warm_status")) == "cold_rank0"]
        warm = [row for row in group if str(row.get("warm_status")) == "warm_persistent"]
        summaries.append(
            {
                **asdict(candidate),
                "case_count": len(group),
                "dice_mean": mean(group, "dice_mean"),
                "dice_std": std(group, "dice_mean"),
                "hd95_mean": mean(group, "hd95_mean"),
                "hd95_std": std(group, "hd95_mean"),
                "asd_symmetric_mean": mean(group, "asd_symmetric_mean"),
                "asd_symmetric_std": std(group, "asd_symmetric_mean"),
                "full_registration_time_sec_mean": mean(group, "full_registration_time_sec"),
                "full_registration_time_sec_std": std(group, "full_registration_time_sec"),
                "training_elapsed_sec_mean": mean(group, "training_elapsed_sec"),
                "loaded_images_to_dvf_file_sec_mean": mean(group, "loaded_images_to_dvf_file_sec"),
                "loaded_images_to_dvf_file_sec_std": std(group, "loaded_images_to_dvf_file_sec"),
                "actual_epochs_mean": mean(group, "actual_epochs"),
                "actual_epochs_std": std(group, "actual_epochs"),
                "peak_mem_gb_mean": mean(group, "training_peak_mem_gb"),
                "peak_mem_gb_max": max_value(group, "training_peak_mem_gb"),
                "cold_rank0_full_registration_sec": mean(cold, "full_registration_time_sec"),
                "warm_ranks1_9_full_registration_sec_mean": mean(warm, "full_registration_time_sec"),
                "warm_ranks1_9_full_registration_sec_std": std(warm, "full_registration_time_sec"),
                "cold_rank0_loaded_images_to_dvf_file_sec": mean(cold, "loaded_images_to_dvf_file_sec"),
                "warm_ranks1_9_loaded_images_to_dvf_file_sec_mean": mean(warm, "loaded_images_to_dvf_file_sec"),
                "warm_ranks1_9_loaded_images_to_dvf_file_sec_std": std(warm, "loaded_images_to_dvf_file_sec"),
                "cold_rank0_peak_mem_gb": mean(cold, "training_peak_mem_gb"),
                "warm_ranks1_9_peak_mem_gb_mean": mean(warm, "training_peak_mem_gb"),
                "warm_ranks1_9_peak_mem_gb_max": max_value(warm, "training_peak_mem_gb"),
            }
        )
    by_name = {row["name"]: row for row in summaries}
    for row in summaries:
        baseline = by_name.get(str(row["baseline_name"]))
        if baseline is None or row["name"] == row["baseline_name"]:
            row.update(
                {
                    "dice_drop_vs_baseline": 0.0 if row["name"] == row["baseline_name"] else float("nan"),
                    "hd95_increase_vs_baseline_mm": 0.0 if row["name"] == row["baseline_name"] else float("nan"),
                    "asd_increase_vs_baseline_mm": 0.0 if row["name"] == row["baseline_name"] else float("nan"),
                    "full_registration_saving_pct_vs_baseline": 0.0 if row["name"] == row["baseline_name"] else float("nan"),
                    "warm_dvf_file_saving_pct_vs_baseline": 0.0 if row["name"] == row["baseline_name"] else float("nan"),
                    "dice_degradation_grade": "baseline" if row["name"] == row["baseline_name"] else "missing_baseline",
                }
            )
            continue
        dice_drop = as_float(baseline["dice_mean"]) - as_float(row["dice_mean"])
        full_base = as_float(baseline["full_registration_time_sec_mean"])
        full_cur = as_float(row["full_registration_time_sec_mean"])
        warm_base = as_float(baseline["warm_ranks1_9_loaded_images_to_dvf_file_sec_mean"])
        warm_cur = as_float(row["warm_ranks1_9_loaded_images_to_dvf_file_sec_mean"])
        row.update(
            {
                "dice_drop_vs_baseline": dice_drop,
                "hd95_increase_vs_baseline_mm": as_float(row["hd95_mean"]) - as_float(baseline["hd95_mean"]),
                "asd_increase_vs_baseline_mm": as_float(row["asd_symmetric_mean"]) - as_float(baseline["asd_symmetric_mean"]),
                "full_registration_saving_pct_vs_baseline": 100.0 * (full_base - full_cur) / full_base if full_base > 0 else float("nan"),
                "warm_dvf_file_saving_pct_vs_baseline": 100.0 * (warm_base - warm_cur) / warm_base if warm_base > 0 else float("nan"),
                "dice_degradation_grade": dice_grade(dice_drop),
            }
        )
    summaries.sort(
        key=lambda row: (
            str(row["phase"]),
            str(row["family"]),
            str(row["method_label"]),
            0 if row["dice_degradation_grade"] in {"no_loss_or_gain", "small_0_to_0p02"} else 1,
            -as_float(row["full_registration_saving_pct_vs_baseline"]),
            -as_float(row["dice_mean"]),
        )
    )
    return summaries


def format_num(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_report(args: argparse.Namespace, rows: Sequence[Mapping[str, object]], summaries: Sequence[Mapping[str, object]], missing: Sequence[Mapping[str, object]]) -> None:
    review_root = Path(args.review_root)
    phase_title = {"primary": "Primary cps2 + MR-D-BSCP", "cps4": "cps4 Context", "all": "All", "smoke": "Smoke"}.get(str(args.phase), str(args.phase))
    lines: List[str] = []
    lines.append(f"# OASIS Unified Acceleration Sweep: {phase_title}")
    lines.append("")
    lines.append(f"- Variant: `{args.variant}`")
    lines.append(f"- Output root: `{args.out_root}`")
    lines.append(f"- Case ranks: `{args.case_ranks}`")
    lines.append("- Process model: one persistent worker per candidate, with the 10 registrations run sequentially on one assigned GPU.")
    lines.append("- Timing convention: rank 0 is the cold case for that candidate process; ranks 1-9 are warm persistent-process cases.")
    lines.append("- Primary accuracy readout: Dice degradation grade versus the same-method baseline.")
    lines.append("- HD95 and ASD degradation are reported as continuous deltas, not hard cutoffs.")
    lines.append("- Full registration time includes optimization, DVF save, evaluation, regularity metrics, and output packaging. Loaded-images-to-DVF-file time stops immediately after the dense DVF is saved.")
    lines.append("")
    lines.append("## Dice Degradation Grades")
    lines.append("")
    lines.append("| Grade | Meaning |")
    lines.append("|---|---|")
    lines.append("| `no_loss_or_gain` | Mean Dice is equal to or better than baseline. |")
    lines.append("| `small_0_to_0p02` | Mean Dice drop is greater than 0 and no more than 0.02. |")
    lines.append("| `moderate_0p02_to_0p05` | Mean Dice drop is greater than 0.02 and no more than 0.05. |")
    lines.append("| `large_gt_0p05` | Mean Dice drop is greater than 0.05. |")
    lines.append("")
    if missing:
        lines.append("## Missing Outputs")
        lines.append("")
        lines.append(f"- Missing summaries: `{len(missing)}`")
        lines.append("")
    lines.append("## Candidate Summary")
    lines.append("")
    lines.append("| Method | Candidate | Template | Dice | Dice Drop | Grade | HD95 | HD95 Delta | ASD Delta | Full Time Mean | Full Saving % | Cold Full | Warm Full | Cold DVF File | Warm DVF File | Epochs | Peak Mem GB |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method_label"]),
                    str(row["name"]),
                    str(row["template"]),
                    format_num(row["dice_mean"], 4),
                    format_num(row["dice_drop_vs_baseline"], 4),
                    str(row["dice_degradation_grade"]),
                    format_num(row["hd95_mean"], 3),
                    format_num(row["hd95_increase_vs_baseline_mm"], 3),
                    format_num(row["asd_increase_vs_baseline_mm"], 3),
                    format_num(row["full_registration_time_sec_mean"], 2),
                    format_num(row["full_registration_saving_pct_vs_baseline"], 1),
                    format_num(row["cold_rank0_full_registration_sec"], 2),
                    format_num(row["warm_ranks1_9_full_registration_sec_mean"], 2),
                    format_num(row["cold_rank0_loaded_images_to_dvf_file_sec"], 2),
                    format_num(row["warm_ranks1_9_loaded_images_to_dvf_file_sec_mean"], 2),
                    format_num(row["actual_epochs_mean"], 1),
                    format_num(row["peak_mem_gb_max"], 2),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Selection Guidance")
    lines.append("")
    lines.append("- Start from candidates in `no_loss_or_gain` or `small_0_to_0p02` Dice grades.")
    lines.append("- Use HD95/ASD deltas to reject candidates whose boundary behavior worsens materially, even if Dice remains acceptable.")
    lines.append("- Prefer warm loaded-images-to-DVF-file time for deployment speed discussion, and report cold rank-0 separately.")
    lines.append("- Peak memory is recorded from `training_peak_mem_gb`; use max over first10 as the conservative value.")
    lines.append("")
    report_name = "method_ordered_report.md" if str(args.phase) in {"primary", "all"} else "method_ordered_report_cps4.md"
    (review_root / report_name).write_text("\n".join(lines) + "\n")
    phase_report = {
        "primary": "phase1_primary_report.md",
        "cps4": "phase2_cps4_context_report.md",
        "smoke": "smoke_report.md",
        "all": "all_phase_report.md",
    }[str(args.phase)]
    (review_root / phase_report).write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    names = selected_candidate_names(args)
    unknown = [name for name in names if name not in candidate_map()]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    ranks = parse_case_ranks(str(args.case_ranks))
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    rows, missing = collect_rows(args, names, ranks)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Missing {len(missing)} candidate/case summaries; rerun with --allow-incomplete to aggregate partial results")
    summaries = summarize_candidates(rows)
    prefix = "" if str(args.phase) == "primary" else f"{args.phase}_"
    c.write_csv(Path(args.review_root) / f"{prefix}case_results.csv", rows)
    c.write_json(Path(args.review_root) / f"{prefix}case_results.json", rows)
    c.write_csv(Path(args.review_root) / f"{prefix}candidate_summary.csv", summaries)
    c.write_json(Path(args.review_root) / f"{prefix}candidate_summary.json", summaries)
    c.write_csv(Path(args.review_root) / f"{prefix}accuracy_runtime_tradeoff.csv", summaries)
    c.write_csv(Path(args.review_root) / f"{prefix}dice_grade_selection.csv", summaries)
    c.write_csv(
        Path(args.review_root) / f"{prefix}cold_warm_timing_summary.csv",
        [
            {
                key: row.get(key)
                for key in [
                    "name",
                    "method_label",
                    "template",
                    "dice_degradation_grade",
                    "cold_rank0_full_registration_sec",
                    "warm_ranks1_9_full_registration_sec_mean",
                    "cold_rank0_loaded_images_to_dvf_file_sec",
                    "warm_ranks1_9_loaded_images_to_dvf_file_sec_mean",
                    "peak_mem_gb_max",
                ]
            }
            for row in summaries
        ],
    )
    c.write_json(
        Path(args.review_root) / f"{prefix}aggregation_status.json",
        {
            "created_at": c.now_text(),
            "phase": str(args.phase),
            "candidate_count_expected": len(names),
            "rank_count_expected": len(ranks),
            "summary_count": len(rows),
            "missing_count": len(missing),
            "missing": missing,
        },
    )
    write_report(args, rows, summaries, missing)
    print(json.dumps({"summary_count": len(rows), "missing_count": len(missing), "review_root": str(args.review_root)}, indent=2))


if __name__ == "__main__":
    main()
