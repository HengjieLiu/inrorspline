#!/usr/bin/env python
"""Aggregate OASIS cps4 runtime candidates for D-BSCP and SINR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402
from launch_oasis_cps4_runtime_sweep import (  # noqa: E402
    ALPHA,
    DEFAULT_OUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    DIRECT_CP_CPS4,
    SINR_CPS4,
    VARIANT,
    Candidate,
    candidate_map,
    effective_epochs,
    effective_min_epochs,
    parse_candidate_names,
)


DIRECTCP_CPS2_DICE = 0.8023349453
DIRECTCP_CPS2_HD95 = 2.2787526502
SINR_CPS2_DICE = 0.7992483155
SINR_CPS2_HD95 = 2.2374157696
DICE_DROP_LIMIT = 0.02
HD95_WORSE_LIMIT_MM = 0.2
RUNTIME_IMPROVEMENT_TARGET = 0.25
EPOCH_REDUCTION_TARGET = 0.25
D_BASELINE = "D_baseline_lr1e-3_e2500"
S_BASELINE = "S_baseline_lr1e-4_e2500"


SUMMARY_METRICS = [
    "dense_image_loss",
    "final_loss",
    "final_data_loss",
    "final_reg_loss",
    "dice_mean",
    "hd95_mean",
    "asd_symmetric_mean",
    "be_native_analytic_fixed_fg_voxel",
    "diffusion_native_analytic_fixed_fg_voxel",
    "log_jdet_std_fg",
    "jdet_le0_pct_fg",
    "ndv_frac_fg",
    "training_elapsed_sec",
    "total_elapsed_sec",
    "training_peak_mem_gb",
    "actual_epochs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--epoch-override", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    return base.as_float(value)


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def float_close(value: object, target: float) -> bool:
    return math.isclose(as_float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)


def stats(values: Iterable[object]) -> Dict[str, float | int]:
    return base.stats(values)


def fmt(value: object, digits: int = 4) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.3g}"
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def row_list(value: object) -> List[float]:
    if isinstance(value, list):
        return [as_float(item) for item in value]
    return []


def candidate_expected(args: argparse.Namespace, candidate: Candidate) -> Tuple[int, int]:
    return (
        int(effective_epochs(args, candidate)),
        int(effective_min_epochs(args, candidate)),
    )


def collect_summaries(root: Path, candidates: Sequence[str]) -> List[Dict[str, object]]:
    allowed = set(str(name) for name in candidates)
    rows = [
        row
        for row in base.collect_summaries(Path(root))
        if str(row.get("runtime_candidate")) in allowed
        and str(row.get("method")) == candidate_map()[str(row.get("runtime_candidate"))].method
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("runtime_candidate")),
            int(as_float(row.get("case_rank"))),
        )
    )
    return rows


def expected_cases(args: argparse.Namespace) -> Dict[int, c.CaseInfo]:
    return {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in base.parse_case_ranks(str(args.case_ranks))
    }


def combo_key(row: Mapping[str, object]) -> Tuple[str, int]:
    return str(row.get("runtime_candidate")), int(as_float(row.get("case_rank")))


def row_errors(row: Mapping[str, object], args: argparse.Namespace, cases: Mapping[int, c.CaseInfo]) -> List[str]:
    errors: List[str] = []
    run_name = str(row.get("run_name"))
    candidate_name = str(row.get("runtime_candidate"))
    candidate = candidate_map().get(candidate_name)
    if candidate is None:
        return [f"{run_name}: unknown runtime_candidate {candidate_name}"]
    expected_epochs, expected_min_epochs = candidate_expected(args, candidate)
    rank = int(as_float(row.get("case_rank")))
    case = cases.get(rank)
    if case is None:
        return [f"{run_name}: unexpected case_rank {rank}"]
    actual_stage_epochs = [int(value) for value in row_list(row.get("stage_actual_epochs"))]
    actual_epochs = int(as_float(row.get("actual_epochs")))

    if str(row.get("variant")) != str(args.variant):
        errors.append(f"{run_name}: variant {row.get('variant')} != {args.variant}")
    if str(row.get("method")) != candidate.method or str(row.get("method_key")) != candidate.method:
        errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {candidate.method}")
    if not float_close(row.get("alpha"), float(args.alpha)):
        errors.append(f"{run_name}: alpha {row.get('alpha')} != {float(args.alpha):g}")
    if not float_close(row.get("lr"), float(candidate.lr)):
        errors.append(f"{run_name}: lr {row.get('lr')} != {float(candidate.lr):g}")
    if int(as_float(row.get("eval_interval"))) != int(args.expected_eval_interval):
        errors.append(f"{run_name}: eval_interval {row.get('eval_interval')} != {args.expected_eval_interval}")
    if int(as_float(row.get("seed"))) != int(args.expected_seed):
        errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
    if int(as_float(row.get("original_case_index"))) != int(case.original_case_index):
        errors.append(f"{run_name}: original_case_index {row.get('original_case_index')} != {case.original_case_index}")
    if str(row.get("fixed_id")) != case.fixed_id or str(row.get("moving_id")) != case.moving_id:
        errors.append(f"{run_name}: fixed/moving {row.get('fixed_id')}/{row.get('moving_id')} != {case.fixed_id}/{case.moving_id}")
    if str(row.get("sample_domain")) != "none_full_image_ncc":
        errors.append(f"{run_name}: sample_domain {row.get('sample_domain')} != none_full_image_ncc")
    if str(row.get("native_analytic_be_formula")) != "bspline_analytic":
        errors.append(f"{run_name}: native BE formula {row.get('native_analytic_be_formula')} != bspline_analytic")
    if str(row.get("native_analytic_diffusion_formula")) != "bspline_analytic":
        errors.append(f"{run_name}: native diffusion formula {row.get('native_analytic_diffusion_formula')} != bspline_analytic")
    if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
        errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
    if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
        errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
    if int(as_float(row.get("requested_epochs"))) != int(expected_epochs):
        errors.append(f"{run_name}: requested_epochs {row.get('requested_epochs')} != {expected_epochs}")
    if row.get("stage_requested_epochs") != [int(expected_epochs)]:
        errors.append(f"{run_name}: stage_requested_epochs {row.get('stage_requested_epochs')} != [{expected_epochs}]")
    if actual_stage_epochs and actual_stage_epochs != [actual_epochs]:
        errors.append(f"{run_name}: stage_actual_epochs {actual_stage_epochs} != [{actual_epochs}]")
    if str(candidate.early_stop_mode) == "none":
        if actual_epochs != int(expected_epochs):
            errors.append(f"{run_name}: fixed actual_epochs {actual_epochs} != {expected_epochs}")
    else:
        if actual_epochs > int(expected_epochs):
            errors.append(f"{run_name}: adaptive actual_epochs {actual_epochs} > {expected_epochs}")
        if actual_epochs < int(expected_min_epochs):
            errors.append(f"{run_name}: adaptive actual_epochs {actual_epochs} < min {expected_min_epochs}")
        if str(row.get("stage_early_stop_mode")) != "adaptive":
            errors.append(f"{run_name}: stage_early_stop_mode {row.get('stage_early_stop_mode')} != adaptive")
    for metric in SUMMARY_METRICS:
        if not is_finite(row.get(metric)):
            errors.append(f"{run_name}: missing/non-finite required metric {metric}")
    errors.extend(f"{run_name}: {error}" for error in base.required_artifact_errors(row))
    run_dir = Path(str(row["run_dir_actual"]))
    for name in ["stage_history.csv", "stage_summaries.csv", "prolongation_metrics.csv"]:
        if not (run_dir / name).exists():
            errors.append(f"{run_name}: missing {run_dir / name}")
    return errors


def validate_rows(rows: Sequence[Dict[str, object]], args: argparse.Namespace, candidates: Sequence[str]) -> List[str]:
    errors: List[str] = []
    ranks = base.parse_case_ranks(str(args.case_ranks))
    cases = expected_cases(args)
    expected = {(candidate, rank) for candidate in candidates for rank in ranks}
    counts = Counter(combo_key(row) for row in rows)
    observed = set(counts)
    for key in sorted(expected - observed):
        errors.append(f"missing candidate={key[0]} case_rank={key[1]}")
    for key in sorted(observed - expected):
        errors.append(f"unexpected candidate={key[0]} case_rank={key[1]}")
    for key, count in sorted(counts.items()):
        if count != 1:
            errors.append(f"duplicate candidate={key[0]} case_rank={key[1]} count={count}")
    if len(rows) != len(expected):
        errors.append(f"expected {len(expected)} summaries, found {len(rows)}")
    for row in rows:
        errors.extend(row_errors(row, args, cases))
    return errors


def metric_stats_into(row: Dict[str, object], prefix: str, values: Iterable[object]) -> None:
    for stat_name, value in stats(values).items():
        row[f"{prefix}_{stat_name}"] = value


def population_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("runtime_candidate"))].append(row)
    out: List[Dict[str, object]] = []
    candidates = candidate_map()
    for candidate_name in [candidate for candidate in candidates if candidate in grouped]:
        candidate = candidates[candidate_name]
        group = grouped[candidate_name]
        case_ranks = sorted(int(as_float(row.get("case_rank"))) for row in group)
        row: Dict[str, object] = {
            "runtime_candidate": candidate_name,
            "method": candidate.method,
            "method_label": candidate.method_label,
            "family": candidate.family,
            "description": candidate.description,
            "is_baseline": bool(candidate.is_baseline),
            "lr": float(candidate.lr),
            "requested_epochs": int(candidate.epochs),
            "early_stop_mode": candidate.early_stop_mode,
            "min_epochs": int(candidate.min_epochs),
            "patience": int(candidate.patience),
            "min_delta": float(candidate.min_delta),
            "n_cases": len(group),
            "case_ranks_text": ",".join(str(rank) for rank in case_ranks),
        }
        for metric in SUMMARY_METRICS:
            metric_stats_into(row, metric, (group_row.get(metric) for group_row in group))
        stage_actuals = [row_list(group_row.get("stage_actual_epochs")) for group_row in group]
        metric_stats_into(row, "stage0_actual_epochs", (values[0] for values in stage_actuals if values))
        out.append(row)
    return out


def baseline_name_for_candidate(candidate_name: str) -> str:
    candidate = candidate_map()[candidate_name]
    if candidate.method == DIRECT_CP_CPS4:
        return D_BASELINE
    if candidate.method == SINR_CPS4:
        return S_BASELINE
    raise ValueError(f"No baseline for method {candidate.method}")


def cps2_reference(candidate_name: str) -> Tuple[str, float, float]:
    candidate = candidate_map()[candidate_name]
    if candidate.method == DIRECT_CP_CPS4:
        return "DirectCP cps2", DIRECTCP_CPS2_DICE, DIRECTCP_CPS2_HD95
    return "SINR cps2", SINR_CPS2_DICE, SINR_CPS2_HD95


def enrich_candidate_summary(pop_rows: Sequence[Dict[str, object]], expected_n: int) -> List[Dict[str, object]]:
    by_name = {str(row["runtime_candidate"]): row for row in pop_rows}
    enriched: List[Dict[str, object]] = []
    for row in pop_rows:
        out = dict(row)
        candidate_name = str(row["runtime_candidate"])
        baseline_name = baseline_name_for_candidate(candidate_name)
        baseline = by_name.get(baseline_name, {})
        dice = as_float(row.get("dice_mean_mean"))
        hd95 = as_float(row.get("hd95_mean_mean"))
        total = as_float(row.get("total_elapsed_sec_mean"))
        train = as_float(row.get("training_elapsed_sec_mean"))
        epochs = as_float(row.get("actual_epochs_mean"))
        baseline_dice = as_float(baseline.get("dice_mean_mean"))
        baseline_hd95 = as_float(baseline.get("hd95_mean_mean"))
        baseline_total = as_float(baseline.get("total_elapsed_sec_mean"))
        baseline_train = as_float(baseline.get("training_elapsed_sec_mean"))
        baseline_epochs = as_float(baseline.get("actual_epochs_mean"))
        dice_delta = dice - baseline_dice
        hd95_delta = hd95 - baseline_hd95
        total_saving_sec = baseline_total - total
        total_saving_frac = total_saving_sec / baseline_total if math.isfinite(baseline_total) and baseline_total > 0 else float("nan")
        train_saving_sec = baseline_train - train
        train_saving_frac = train_saving_sec / baseline_train if math.isfinite(baseline_train) and baseline_train > 0 else float("nan")
        epoch_reduction = baseline_epochs - epochs
        epoch_reduction_frac = epoch_reduction / baseline_epochs if math.isfinite(baseline_epochs) and baseline_epochs > 0 else float("nan")
        ref_label, ref_dice, ref_hd95 = cps2_reference(candidate_name)
        directcp_cps2_beats = math.isfinite(dice) and dice >= DIRECTCP_CPS2_DICE and hd95 <= DIRECTCP_CPS2_HD95
        sinr_cps2_beats = math.isfinite(dice) and dice >= SINR_CPS2_DICE and hd95 <= SINR_CPS2_HD95
        same_cps2_beats = math.isfinite(dice) and dice >= ref_dice and hd95 <= ref_hd95
        complete = int(as_float(row.get("n_cases"))) == int(expected_n)
        accuracy_pass = (
            complete
            and math.isfinite(dice_delta)
            and math.isfinite(hd95_delta)
            and dice_delta >= -DICE_DROP_LIMIT
            and hd95_delta <= HD95_WORSE_LIMIT_MM
        )
        runtime_pass = math.isfinite(total_saving_frac) and total_saving_frac >= RUNTIME_IMPROVEMENT_TARGET
        no_accuracy_loss = math.isfinite(dice_delta) and math.isfinite(hd95_delta) and dice_delta >= 0.0 and hd95_delta <= 0.0
        clear_epoch_reduction_no_accuracy_loss = (
            math.isfinite(epoch_reduction_frac)
            and epoch_reduction_frac >= EPOCH_REDUCTION_TARGET
            and no_accuracy_loss
        )
        promising = (
            not bool(row.get("is_baseline"))
            and accuracy_pass
            and (runtime_pass or clear_epoch_reduction_no_accuracy_loss)
        )
        reject_reasons: List[str] = []
        if not complete:
            reject_reasons.append(f"incomplete_n{int(as_float(row.get('n_cases')))}_of_{expected_n}")
        if not accuracy_pass:
            if math.isfinite(dice_delta) and dice_delta < -DICE_DROP_LIMIT:
                reject_reasons.append("dice_drop_gt_0p02")
            if math.isfinite(hd95_delta) and hd95_delta > HD95_WORSE_LIMIT_MM:
                reject_reasons.append("hd95_worse_gt_0p2mm")
            if not math.isfinite(dice_delta) or not math.isfinite(hd95_delta):
                reject_reasons.append("missing_baseline_or_metric")
        if not bool(row.get("is_baseline")) and accuracy_pass and not (runtime_pass or clear_epoch_reduction_no_accuracy_loss):
            reject_reasons.append("runtime_saving_lt_25pct_and_no_clear_epoch_reduction_without_accuracy_loss")

        out.update(
            {
                "baseline_candidate": baseline_name,
                "baseline_dice_mean": baseline_dice,
                "baseline_hd95_mean": baseline_hd95,
                "baseline_total_elapsed_sec_mean": baseline_total,
                "baseline_training_elapsed_sec_mean": baseline_train,
                "baseline_actual_epochs_mean": baseline_epochs,
                "dice_delta_vs_baseline": dice_delta,
                "hd95_delta_vs_baseline_mm": hd95_delta,
                "total_elapsed_saving_sec_vs_baseline": total_saving_sec,
                "total_elapsed_saving_frac_vs_baseline": total_saving_frac,
                "training_saving_sec_vs_baseline": train_saving_sec,
                "training_saving_frac_vs_baseline": train_saving_frac,
                "epoch_reduction_vs_baseline": epoch_reduction,
                "epoch_reduction_frac_vs_baseline": epoch_reduction_frac,
                "same_family_cps2_reference": ref_label,
                "same_family_cps2_dice": ref_dice,
                "same_family_cps2_hd95": ref_hd95,
                "beats_same_family_cps2": same_cps2_beats,
                "beats_directcp_cps2": directcp_cps2_beats,
                "beats_sinr_cps2": sinr_cps2_beats,
                "accuracy_gate_pass": accuracy_pass,
                "runtime_gate_pass": runtime_pass,
                "clear_epoch_reduction_no_accuracy_loss": clear_epoch_reduction_no_accuracy_loss,
                "promising_candidate": promising,
                "reject_reasons": ",".join(reject_reasons),
            }
        )
        enriched.append(out)
    enriched.sort(
        key=lambda item: (
            0 if item.get("promising_candidate") else 1,
            str(item.get("family")),
            0 if item.get("is_baseline") else 1,
            -as_float(item.get("total_elapsed_saving_frac_vs_baseline")),
            -as_float(item.get("dice_mean_mean")),
            as_float(item.get("hd95_mean_mean")),
        )
    )
    return enriched


def context_from_summary(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "run_name": row.get("run_name"),
        "runtime_candidate": row.get("runtime_candidate"),
        "method": row.get("method"),
        "method_key": row.get("method_key"),
        "case_rank": row.get("case_rank"),
        "original_case_index": row.get("original_case_index"),
        "fixed_id": row.get("fixed_id"),
        "moving_id": row.get("moving_id"),
        "alpha": row.get("alpha"),
        "lr": row.get("lr"),
        "requested_epochs": row.get("requested_epochs"),
        "actual_epochs": row.get("actual_epochs"),
        "training_elapsed_sec": row.get("training_elapsed_sec"),
        "total_elapsed_sec": row.get("total_elapsed_sec"),
        "final_loss": row.get("final_loss"),
        "final_data_loss": row.get("final_data_loss"),
        "final_reg_loss": row.get("final_reg_loss"),
        "dice_mean": row.get("dice_mean"),
        "hd95_mean": row.get("hd95_mean"),
        "asd_symmetric_mean": row.get("asd_symmetric_mean"),
        "early_stop_epoch": row.get("early_stop_epoch"),
        "early_stop_reason": row.get("early_stop_reason"),
        "best_loss": row.get("best_loss"),
        "best_loss_epoch": row.get("best_loss_epoch"),
        "summary_path": row.get("summary_path"),
        "run_dir_actual": row.get("run_dir_actual"),
    }


def collect_metric_file(rows: Sequence[Dict[str, object]], filename: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        path = Path(str(row["run_dir_actual"])) / filename
        if not path.exists():
            continue
        context = context_from_summary(row)
        for metric_row in c.read_csv(path):
            merged = dict(context)
            merged.update(metric_row)
            merged[f"{filename}_path"] = str(path)
            out.append(merged)
    return out


def method_order(row: Mapping[str, object]) -> Tuple[int, int, float, str]:
    family_rank = 0 if str(row.get("family")) == "D-BSCP" else 1
    baseline_rank = 0 if bool(row.get("is_baseline")) else 1
    return (
        family_rank,
        baseline_rank,
        -as_float(row.get("total_elapsed_saving_frac_vs_baseline")),
        str(row.get("runtime_candidate")),
    )


def markdown_table(rows: Sequence[Mapping[str, object]]) -> List[str]:
    headers = [
        "candidate",
        "family",
        "n",
        "Dice",
        "dDice",
        "HD95",
        "dHD95",
        "total s",
        "save %",
        "epochs",
        "cps2",
        "promising",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [
            str(row.get("runtime_candidate")),
            str(row.get("family")),
            fmt(row.get("n_cases"), 0),
            fmt(row.get("dice_mean_mean")),
            fmt(row.get("dice_delta_vs_baseline")),
            fmt(row.get("hd95_mean_mean")),
            fmt(row.get("hd95_delta_vs_baseline_mm")),
            fmt(row.get("total_elapsed_sec_mean"), 1),
            fmt(100.0 * as_float(row.get("total_elapsed_saving_frac_vs_baseline")), 1),
            fmt(row.get("actual_epochs_mean"), 1),
            "yes" if row.get("beats_same_family_cps2") else "no",
            "yes" if row.get("promising_candidate") else "no",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    summary_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    promising = [row for row in summary_rows if row.get("promising_candidate")]
    ordered = sorted(summary_rows, key=method_order)
    lines = [
        "# OASIS cps4 Runtime Sweep: D-BSCP And SINR",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Out root: `{args.out_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Summaries: `{len(rows)}`",
        f"- Case ranks: `{args.case_ranks}`",
        f"- Candidates: `{args.candidates}`",
        f"- Alpha: `{float(args.alpha):g}`",
        f"- Variant: `{args.variant}`",
        f"- Gate: Dice drop `<= {DICE_DROP_LIMIT}` and HD95 increase `<= {HD95_WORSE_LIMIT_MM} mm` versus same-method cps4 baseline.",
        f"- Runtime target: full-pipeline total elapsed improvement `>= {100.0 * RUNTIME_IMPROVEMENT_TARGET:.0f}%`, or `>= {100.0 * EPOCH_REDUCTION_TARGET:.0f}%` epoch reduction with no Dice/HD95 loss.",
        f"- cps2 anchors: DirectCP cps2 Dice/HD95 `{DIRECTCP_CPS2_DICE:.4f}/{DIRECTCP_CPS2_HD95:.4f}`; SINR cps2 `{SINR_CPS2_DICE:.4f}/{SINR_CPS2_HD95:.4f}`.",
        "",
        "## Validation",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:100])
        if len(errors) > 100:
            lines.append(f"- FAIL: ... {len(errors) - 100} additional errors")
    else:
        lines.append("- PASS: all expected rows and artifacts are present.")
    lines.extend(
        [
            "",
            "## Promising Candidates",
            "",
        ]
    )
    if promising:
        for row in promising:
            lines.append(
                f"- `{row['runtime_candidate']}`: Dice `{fmt(row.get('dice_mean_mean'))}` "
                f"(delta `{fmt(row.get('dice_delta_vs_baseline'))}`), HD95 `{fmt(row.get('hd95_mean_mean'))}` "
                f"(delta `{fmt(row.get('hd95_delta_vs_baseline_mm'))}`), total `{fmt(row.get('total_elapsed_sec_mean'), 1)}s`, "
                f"saving `{fmt(100.0 * as_float(row.get('total_elapsed_saving_frac_vs_baseline')), 1)}%`."
            )
    else:
        lines.append("- None under the current gate.")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
        ]
    )
    lines.extend(markdown_table(ordered))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `candidate_summary.csv/json`: population means, deltas, gates, and cps2 reference comparisons.",
            "- `case_results.csv`: per-case full-pipeline results.",
            "- `accuracy_runtime_tradeoff.csv`: candidates sorted by gate status and runtime saving.",
            "- `promising_candidates.csv`: candidates selected for sequential timing follow-up.",
            "- `stage_summaries_combined.csv`: single-stage epoch and timing metadata.",
            "- `job_manifest.csv/json`: launcher manifest.",
            "- `launch_summary.json`: launcher completion summary.",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "method_ordered_report.md").write_text("\n".join(lines) + "\n")


def write_outputs(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    summary_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    case_rows = [context_from_summary(row) for row in rows]
    tradeoff_rows = sorted(
        summary_rows,
        key=lambda row: (
            0 if row.get("promising_candidate") else 1,
            str(row.get("family")),
            -as_float(row.get("total_elapsed_saving_frac_vs_baseline")),
            -as_float(row.get("dice_mean_mean")),
            as_float(row.get("hd95_mean_mean")),
        ),
    )
    promising_rows = [row for row in tradeoff_rows if row.get("promising_candidate")]
    c.write_csv(args.review_root / "case_results.csv", case_rows)
    c.write_json(args.review_root / "case_results.json", case_rows)
    c.write_csv(args.review_root / "candidate_summary.csv", summary_rows)
    c.write_json(args.review_root / "candidate_summary.json", summary_rows)
    c.write_csv(args.review_root / "accuracy_runtime_tradeoff.csv", tradeoff_rows)
    c.write_json(args.review_root / "accuracy_runtime_tradeoff.json", tradeoff_rows)
    c.write_csv(args.review_root / "promising_candidates.csv", promising_rows)
    c.write_json(args.review_root / "promising_candidates.json", promising_rows)
    (args.review_root / "promising_candidates.txt").write_text(
        ",".join(str(row["runtime_candidate"]) for row in promising_rows) + "\n"
    )
    c.write_csv(args.review_root / "case_metrics_combined.csv", collect_metric_file(rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_combined.csv", collect_metric_file(rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "stage_summaries_combined.csv", collect_metric_file(rows, "stage_summaries.csv"))
    c.write_csv(args.review_root / "prolongation_metrics_combined.csv", collect_metric_file(rows, "prolongation_metrics.csv"))
    if errors:
        c.write_json(args.review_root / "aggregation_errors.json", errors)
    write_report(args, rows, summary_rows, errors)


def main() -> None:
    args = parse_args()
    candidates = parse_candidate_names(str(args.candidates))
    ranks = base.parse_case_ranks(str(args.case_ranks))
    rows = collect_summaries(Path(args.out_root), candidates)
    errors = validate_rows(rows, args, candidates)
    pop_rows = population_summary(rows)
    summary_rows = enrich_candidate_summary(pop_rows, expected_n=len(ranks))
    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("Cps4 runtime sweep aggregation failed: " + "; ".join(errors[:20]))
    write_outputs(args, rows, summary_rows, errors)
    print(
        json.dumps(
            {
                "combined_rows": len(rows),
                "candidate_count": len(summary_rows),
                "promising_candidates": [row["runtime_candidate"] for row in summary_rows if row.get("promising_candidate")],
                "errors": errors,
                "review_root": str(args.review_root),
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
