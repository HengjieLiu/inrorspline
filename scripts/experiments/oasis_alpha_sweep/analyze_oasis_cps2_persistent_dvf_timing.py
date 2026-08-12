#!/usr/bin/env python
"""Summarize persistent-process DVF-only timing for OASIS cps2 candidates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import launch_oasis_cps2_runtime_sweep as sweep  # noqa: E402


DEFAULT_REVIEW_ROOT = sweep.DEFAULT_REVIEW_ROOT
METRICS = [
    "full_script_to_dvf_in_memory_sec",
    "loaded_images_to_dvf_in_memory_sec",
    "full_script_to_dvf_file_sec",
    "loaded_images_to_dvf_file_sec",
    "external_wall_sec",
    "training_sec",
    "total_optimizer_loop_sec",
    "total_non_loop_training_sec",
    "dvf_file_write_sec",
    "actual_epochs",
]
BASELINE_BY_FAMILY = {
    "D-BSCP": "D2_baseline_lr1e-3_e2500",
    "SINR": "S2_baseline_lr1e-4_e2500",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--screen-review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--phase", type=str, default="first10")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_sec(value: object) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    if number != 0.0 and abs(number) < 1.0:
        return f"{number * 1000.0:.1f}ms"
    return f"{number:.2f}s"


def format_float(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def format_pct(value: object) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{100.0 * number:.1f}%"


def mean_std(values: Iterable[float]) -> Dict[str, float | int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
    }


def candidate_order() -> Dict[str, int]:
    return {candidate.name: idx for idx, candidate in enumerate(sweep.CANDIDATES)}


def candidate_meta(name: str, screen_rows: Dict[str, Dict[str, str]]) -> Dict[str, object]:
    screen = screen_rows.get(str(name), {})
    candidate = sweep.candidate_map().get(str(name))
    return {
        "method": screen.get("method") or (candidate.method if candidate else ""),
        "method_label": screen.get("method_label") or (candidate.method_label if candidate else ""),
        "family": screen.get("family") or (candidate.family if candidate else ""),
        "description": screen.get("description") or (candidate.description if candidate else ""),
        "is_baseline": screen.get("is_baseline") if screen else (candidate.is_baseline if candidate else ""),
    }


def rows_for(rows: Sequence[Dict[str, str]], candidate: str, group: str) -> List[Dict[str, str]]:
    selected = [row for row in rows if str(row.get("candidate")) == str(candidate)]
    if group == "cold_rank0":
        return [row for row in selected if as_int(row.get("case_rank")) == 0 or str(row.get("warm_status")) == "cold_rank0"]
    if group == "warm_ranks1_9":
        return [row for row in selected if as_int(row.get("case_rank")) != 0]
    if group == "amortized_ranks0_9":
        return selected
    raise ValueError(group)


def build_summary_rows(rows: Sequence[Dict[str, str]], screen_rows: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    order = candidate_order()
    candidates = sorted({str(row.get("candidate")) for row in rows if row.get("candidate")}, key=lambda name: order.get(name, 999))
    groups = ["cold_rank0", "warm_ranks1_9", "amortized_ranks0_9"]
    out: List[Dict[str, object]] = []
    for candidate in candidates:
        meta = candidate_meta(candidate, screen_rows)
        screen = screen_rows.get(candidate, {})
        for group in groups:
            group_rows = rows_for(rows, candidate, group)
            base: Dict[str, object] = {
                "candidate": candidate,
                "method": meta["method"],
                "method_label": meta["method_label"],
                "family": meta["family"],
                "description": meta["description"],
                "is_baseline": meta["is_baseline"],
                "group": group,
                "case_count": len(group_rows),
                "case_ranks": ",".join(str(as_int(row.get("case_rank"))) for row in group_rows),
                "screen_promising_candidate": screen.get("promising_candidate"),
                "screen_accuracy_gate_pass": screen.get("accuracy_gate_pass"),
                "screen_runtime_gate_pass": screen.get("runtime_gate_pass"),
                "screen_dice_mean": screen.get("dice_mean_mean"),
                "screen_hd95_mean": screen.get("hd95_mean_mean"),
                "screen_dice_delta_vs_baseline": screen.get("dice_delta_vs_baseline"),
                "screen_hd95_delta_vs_baseline_mm": screen.get("hd95_delta_vs_baseline_mm"),
                "screen_total_elapsed_saving_frac_vs_baseline": screen.get("total_elapsed_saving_frac_vs_baseline"),
                "screen_beats_same_family_cps4": screen.get("beats_same_family_cps4"),
            }
            for metric in METRICS:
                stats = mean_std(as_float(row.get(metric)) for row in group_rows)
                base[f"{metric}_mean"] = stats["mean"]
                base[f"{metric}_std"] = stats["std"]
            out.append(base)
    return out


def build_case_rows(rows: Sequence[Dict[str, str]], screen_rows: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    order = candidate_order()
    out: List[Dict[str, object]] = []
    for row in rows:
        meta = candidate_meta(str(row.get("candidate")), screen_rows)
        out.append(
            {
                "candidate": row.get("candidate"),
                "method_label": meta["method_label"],
                "family": meta["family"],
                "case_rank": as_int(row.get("case_rank")),
                "original_case_index": row.get("original_case_index"),
                "fixed_id": row.get("fixed_id"),
                "moving_id": row.get("moving_id"),
                "warm_status": row.get("warm_status") or ("cold_rank0" if as_int(row.get("case_rank")) == 0 else "warm_persistent"),
                "full_script_to_dvf_in_memory_sec": row.get("full_script_to_dvf_in_memory_sec"),
                "loaded_images_to_dvf_in_memory_sec": row.get("loaded_images_to_dvf_in_memory_sec"),
                "full_script_to_dvf_file_sec": row.get("full_script_to_dvf_file_sec"),
                "loaded_images_to_dvf_file_sec": row.get("loaded_images_to_dvf_file_sec"),
                "external_wall_sec": row.get("external_wall_sec"),
                "training_sec": row.get("training_sec"),
                "total_optimizer_loop_sec": row.get("total_optimizer_loop_sec"),
                "total_non_loop_training_sec": row.get("total_non_loop_training_sec"),
                "dvf_file_write_sec": row.get("dvf_file_write_sec"),
                "actual_epochs": row.get("actual_epochs"),
                "evaluation_event_count": row.get("evaluation_event_count"),
                "dense_dvf_exists": row.get("dense_dvf_exists"),
                "run_dir": row.get("run_dir"),
            }
        )
    return sorted(out, key=lambda item: (order.get(str(item["candidate"]), 999), as_int(item["case_rank"])))


def summary_by_candidate_group(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        out.setdefault(str(row.get("group")), {})[str(row.get("candidate"))] = dict(row)
    return out


def build_vs_baseline_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = summary_by_candidate_group(summary_rows)
    out: List[Dict[str, object]] = []
    for group, candidates in grouped.items():
        for candidate, row in candidates.items():
            family = str(row.get("family"))
            baseline_name = BASELINE_BY_FAMILY.get(family, "")
            baseline = candidates.get(baseline_name)
            if not baseline:
                continue
            for metric in METRICS:
                key = f"{metric}_mean"
                baseline_value = as_float(baseline.get(key))
                candidate_value = as_float(row.get(key))
                saving = baseline_value - candidate_value
                out.append(
                    {
                        "group": group,
                        "family": family,
                        "baseline_candidate": baseline_name,
                        "candidate": candidate,
                        "metric": metric,
                        "baseline_mean": baseline_value,
                        "candidate_mean": candidate_value,
                        "baseline_minus_candidate": saving,
                        "saving_frac_vs_baseline": saving / baseline_value
                        if math.isfinite(saving) and math.isfinite(baseline_value) and baseline_value != 0
                        else math.nan,
                    }
                )
    return out


def build_joined_rows(summary_rows: Sequence[Dict[str, object]], vs_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    savings = {
        (str(row.get("candidate")), str(row.get("group")), str(row.get("metric"))): row
        for row in vs_rows
        if str(row.get("metric")) == "loaded_images_to_dvf_file_sec"
    }
    out: List[Dict[str, object]] = []
    for row in summary_rows:
        key = (str(row.get("candidate")), str(row.get("group")), "loaded_images_to_dvf_file_sec")
        saving = savings.get(key, {})
        out.append(
            {
                "candidate": row.get("candidate"),
                "method_label": row.get("method_label"),
                "family": row.get("family"),
                "group": row.get("group"),
                "screen_promising_candidate": row.get("screen_promising_candidate"),
                "screen_dice_mean": row.get("screen_dice_mean"),
                "screen_hd95_mean": row.get("screen_hd95_mean"),
                "screen_dice_delta_vs_baseline": row.get("screen_dice_delta_vs_baseline"),
                "screen_hd95_delta_vs_baseline_mm": row.get("screen_hd95_delta_vs_baseline_mm"),
                "screen_total_elapsed_saving_frac_vs_baseline": row.get("screen_total_elapsed_saving_frac_vs_baseline"),
                "screen_beats_same_family_cps4": row.get("screen_beats_same_family_cps4"),
                "persistent_loaded_images_to_dvf_file_mean_sec": row.get("loaded_images_to_dvf_file_sec_mean"),
                "persistent_loaded_images_to_dvf_file_std_sec": row.get("loaded_images_to_dvf_file_sec_std"),
                "persistent_full_script_to_dvf_file_mean_sec": row.get("full_script_to_dvf_file_sec_mean"),
                "persistent_external_wall_mean_sec": row.get("external_wall_sec_mean"),
                "persistent_optimizer_loop_mean_sec": row.get("total_optimizer_loop_sec_mean"),
                "persistent_actual_epochs_mean": row.get("actual_epochs_mean"),
                "persistent_loaded_images_to_dvf_file_saving_frac_vs_baseline": saving.get("saving_frac_vs_baseline"),
            }
        )
    return out


def read_screen_rows(root: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(Path(root) / "candidate_summary.csv")
    return {str(row.get("runtime_candidate")): row for row in rows if row.get("runtime_candidate")}


def reconciliation_status(review_root: Path) -> Dict[str, int]:
    rows = read_csv(Path(review_root) / "dvf_only_reconciliation.csv")
    if not rows:
        return {"row_count": 0, "residual_failures": 0, "large_self_unaccounted": 0}
    residual_failures = sum(1 for row in rows if str(row.get("within_tolerance")).strip().lower() == "false")
    large_self = sum(
        1
        for row in rows
        if row.get("row_type") == "script_parent"
        and as_int(row.get("child_count")) > 0
        and str(row.get("self_unaccounted_within_tolerance")).strip().lower() == "false"
    )
    return {"row_count": len(rows), "residual_failures": residual_failures, "large_self_unaccounted": large_self}


def report_lines(
    *,
    args: argparse.Namespace,
    method_rows: Sequence[Dict[str, str]],
    summary_rows: Sequence[Dict[str, object]],
    case_rows: Sequence[Dict[str, object]],
    vs_rows: Sequence[Dict[str, object]],
    recon: Dict[str, int],
) -> List[str]:
    candidates = [row.get("candidate") for row in summary_rows if row.get("group") == "amortized_ranks0_9"]
    eval_events = sum(as_int(row.get("evaluation_event_count")) + as_int(row.get("full_write_outputs_event_count")) for row in method_rows)
    non_target = sum(
        as_int(row.get("non_target_gpu_apps_pre_count")) + as_int(row.get("non_target_gpu_apps_post_count"))
        for row in method_rows
    )
    lines: List[str] = []
    lines.append("# OASIS cps2 Persistent-Process DVF Timing")
    lines.append("")
    lines.append(f"- Phase: `{args.phase}`")
    lines.append(f"- Candidate rows: {len(method_rows)}")
    lines.append(f"- Candidates: {', '.join(str(candidate) for candidate in candidates)}")
    lines.append("- Scope: DVF-only timing after the full-pipeline accuracy/runtime screening sweep.")
    lines.append("- Process model: one long-lived Python worker subprocess per candidate; ranks are sequential inside that worker on GPU `0` unless changed at launch.")
    lines.append("- Cold case: rank `0` in a fresh worker process for that candidate.")
    lines.append("- Warm cases: ranks `1-9` in the same worker process. Warm means process/GPU warm, not registration warm-start.")
    lines.append("- Each case still gets a fresh image pair, model/control state, optimizer state, and seed reset inside `run_one.py`.")
    lines.append("- DVF-only excludes image/segmentation evaluation, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, and full packaging.")
    lines.append("- D-BSCP cps2 and SINR cps2 are single-stage runs here; there is no multistage knot insertion/prolongation in these timing rows.")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"- Evaluation/full-output events counted in DVF-only rows: {eval_events}")
    lines.append(f"- Non-target GPU compute-app observations in pre/post snapshots: {non_target}")
    lines.append(f"- Reconciliation rows: {recon['row_count']}")
    lines.append(f"- Reconciliation residual failures: {recon['residual_failures']}")
    lines.append(f"- Parent intervals with large self-unaccounted time: {recon['large_self_unaccounted']}")
    lines.append("")
    lines.append("## Timing Definitions")
    lines.append("")
    lines.append("- `full_script_to_dvf_in_memory`: `run_one.py` script timer start to final dense DVF tensor in memory.")
    lines.append("- `loaded_images_to_dvf_in_memory`: after OASIS case/image load to final dense DVF tensor in memory.")
    lines.append("- `full_script_to_dvf_file`: script timer start to dense DVF `.npy` written.")
    lines.append("- `loaded_images_to_dvf_file`: after OASIS case/image load to dense DVF `.npy` written.")
    lines.append("- `external_wall`: persistent worker measured wall time for one in-process `run_one.main()` call.")
    lines.append("- `optimizer_loop`: epoch work only; it is a strict subset of `training`.")
    lines.append("- `non_loop_training`: training minus optimizer-loop time, including loss-volume/model/grid setup and final dense expansion.")
    lines.append("- Screening Dice/HD95/runtime values come from full evaluation mode in `candidate_summary.csv`; persistent timing values are DVF-only and should not be mixed as if they include evaluation.")
    lines.append("")
    lines.append("## Screening Gate Context")
    lines.append("")
    lines.append("| candidate | method | promising | Dice | HD95 | Dice delta | HD95 delta | full-pipeline saving | beats cps4 ctx |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        if row.get("group") != "amortized_ranks0_9":
            continue
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('method_label')} | {row.get('screen_promising_candidate')} | "
            + f"{format_float(row.get('screen_dice_mean'), 4)} | "
            + f"{format_float(row.get('screen_hd95_mean'), 3)} | "
            + f"{format_float(row.get('screen_dice_delta_vs_baseline'), 4)} | "
            + f"{format_float(row.get('screen_hd95_delta_vs_baseline_mm'), 3)} | "
            + f"{format_pct(row.get('screen_total_elapsed_saving_frac_vs_baseline'))} | "
            + f"{row.get('screen_beats_same_family_cps4')} |"
        )
    lines.append("")
    lines.append("## Persistent Timing Summary")
    lines.append("")
    lines.append("| candidate | group | cases | loaded images -> DVF file | full script -> DVF file | wall/case | optimizer | non-loop training | epochs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('group')} | {row.get('case_count')} | "
            + f"{format_sec(row.get('loaded_images_to_dvf_file_sec_mean'))} +/- {format_sec(row.get('loaded_images_to_dvf_file_sec_std'))} | "
            + f"{format_sec(row.get('full_script_to_dvf_file_sec_mean'))} +/- {format_sec(row.get('full_script_to_dvf_file_sec_std'))} | "
            + f"{format_sec(row.get('external_wall_sec_mean'))} +/- {format_sec(row.get('external_wall_sec_std'))} | "
            + f"{format_sec(row.get('total_optimizer_loop_sec_mean'))} +/- {format_sec(row.get('total_optimizer_loop_sec_std'))} | "
            + f"{format_sec(row.get('total_non_loop_training_sec_mean'))} +/- {format_sec(row.get('total_non_loop_training_sec_std'))} | "
            + f"{format_float(row.get('actual_epochs_mean'), 1)} +/- {format_float(row.get('actual_epochs_std'), 1)} |"
        )
    lines.append("")
    lines.append("## Persistent Timing Versus Same-Method Baseline")
    lines.append("")
    lines.append("Positive saving means the candidate is faster/lower than the same-method cps2 baseline.")
    lines.append("")
    lines.append("| group | candidate | baseline | metric | baseline | candidate | saving | saving frac |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    wanted = {
        "loaded_images_to_dvf_file_sec",
        "full_script_to_dvf_file_sec",
        "external_wall_sec",
        "total_optimizer_loop_sec",
        "total_non_loop_training_sec",
        "actual_epochs",
    }
    for row in vs_rows:
        metric = str(row.get("metric"))
        if metric not in wanted:
            continue
        formatter = format_float if metric == "actual_epochs" else format_sec
        lines.append(
            "| "
            + f"{row.get('group')} | {row.get('candidate')} | {row.get('baseline_candidate')} | {metric} | "
            + f"{formatter(row.get('baseline_mean'))} | "
            + f"{formatter(row.get('candidate_mean'))} | "
            + f"{formatter(row.get('baseline_minus_candidate'))} | "
            + f"{format_pct(row.get('saving_frac_vs_baseline'))} |"
        )
    lines.append("")
    lines.append("## Per-Rank Rows")
    lines.append("")
    lines.append("| candidate | rank | fixed | moving | status | loaded images -> DVF file | wall | optimizer | epochs |")
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
    for row in case_rows:
        lines.append(
            "| "
            + f"{row.get('candidate')} | {row.get('case_rank')} | {row.get('fixed_id')} | {row.get('moving_id')} | "
            + f"{row.get('warm_status')} | "
            + f"{format_sec(row.get('loaded_images_to_dvf_file_sec'))} | "
            + f"{format_sec(row.get('external_wall_sec'))} | "
            + f"{format_sec(row.get('total_optimizer_loop_sec'))} | "
            + f"{format_float(row.get('actual_epochs'), 0)} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `persistent_cps2_dvf_timing_report.md`")
    lines.append("- `persistent_cps2_case_timing.csv`")
    lines.append("- `persistent_cps2_summary.csv`")
    lines.append("- `persistent_cps2_runtime_vs_baseline.csv`")
    lines.append("- `persistent_cps2_screen_and_timing.csv`")
    lines.append("- `dvf_only_method_comparison.csv`")
    lines.append("- `dvf_only_stage_breakdown.csv`")
    lines.append("- `dvf_only_reconciliation.csv`")
    return lines


def main() -> None:
    args = parse_args()
    method_rows = read_csv(Path(args.review_root) / "dvf_only_method_comparison.csv")
    method_rows = [row for row in method_rows if not row.get("phase") or str(row.get("phase")) == str(args.phase)]
    if not method_rows:
        raise RuntimeError(f"No DVF-only method rows found under {args.review_root}")
    screen_rows = read_screen_rows(Path(args.screen_review_root))
    summary_rows = build_summary_rows(method_rows, screen_rows)
    case_rows = build_case_rows(method_rows, screen_rows)
    vs_rows = build_vs_baseline_rows(summary_rows)
    joined_rows = build_joined_rows(summary_rows, vs_rows)
    recon = reconciliation_status(Path(args.review_root))
    root = Path(args.review_root)
    c.write_csv(root / "persistent_cps2_case_timing.csv", case_rows)
    c.write_csv(root / "persistent_cps2_summary.csv", summary_rows)
    c.write_csv(root / "persistent_cps2_runtime_vs_baseline.csv", vs_rows)
    c.write_csv(root / "persistent_cps2_screen_and_timing.csv", joined_rows)
    report = "\n".join(
        report_lines(
            args=args,
            method_rows=method_rows,
            summary_rows=summary_rows,
            case_rows=case_rows,
            vs_rows=vs_rows,
            recon=recon,
        )
    )
    (root / "persistent_cps2_dvf_timing_report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
