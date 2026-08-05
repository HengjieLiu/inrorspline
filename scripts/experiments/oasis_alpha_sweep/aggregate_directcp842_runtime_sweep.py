#!/usr/bin/env python
"""Aggregate OASIS DirectCP 8-4-2 runtime optimization candidates."""

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
from launch_directcp842_runtime_sweep import (  # noqa: E402
    ALPHA,
    DEFAULT_OUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    METHOD,
    VARIANT,
    candidate_map,
    parse_candidate_names,
)


BASELINE_DICE = 0.8069694157
BASELINE_HD95 = 2.2020018341
DIRECTCP_CPS2_DICE = 0.8023349453
DIRECTCP_CPS2_HD95 = 2.2787526502
SINR_CPS2_DICE = 0.7992483155
SINR_CPS2_HD95 = 2.2374157696
RUNTIME_TOTAL_TARGET_SEC = 180.0
RUNTIME_TRAIN_STRONG_SEC = 125.0
RUNTIME_TOTAL_STRONG_SEC = 160.0
SMOKE_DICE_DROP_LIMIT = 0.05
SMOKE_HD95_WORSE_LIMIT_MM = 0.5
DEFAULT_BASELINE_SUMMARY = (
    c.REPO_ROOT
    / "experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/summary_runs_combined.csv"
)


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
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
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


def collect_summaries(root: Path, candidates: Sequence[str]) -> List[Dict[str, object]]:
    allowed = set(str(name) for name in candidates)
    rows = [
        row
        for row in base.collect_summaries(Path(root))
        if str(row.get("method_key")) == METHOD and str(row.get("runtime_candidate")) in allowed
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


def candidate_expected_stage_min(candidate_name: str) -> List[int]:
    candidate = candidate_map()[candidate_name]
    requested = c.parse_int_list(candidate.stage_epochs)
    if str(candidate.stage_min_epochs).strip():
        return c.parse_stage_ints(candidate.stage_min_epochs, len(requested))
    return requested


def row_errors(row: Mapping[str, object], args: argparse.Namespace, cases: Mapping[int, c.CaseInfo]) -> List[str]:
    errors: List[str] = []
    run_name = str(row.get("run_name"))
    candidate_name = str(row.get("runtime_candidate"))
    candidates = candidate_map()
    candidate = candidates.get(candidate_name)
    if candidate is None:
        return [f"{run_name}: unknown runtime_candidate {candidate_name}"]
    rank = int(as_float(row.get("case_rank")))
    case = cases.get(rank)
    if case is None:
        return [f"{run_name}: unexpected case_rank {rank}"]
    expected_stage_epochs = c.parse_int_list(candidate.stage_epochs)
    actual_stage_epochs = [int(value) for value in row_list(row.get("stage_actual_epochs"))]
    min_stage_epochs = candidate_expected_stage_min(candidate_name)
    if str(row.get("variant")) != str(args.variant):
        errors.append(f"{run_name}: variant {row.get('variant')} != {args.variant}")
    if str(row.get("method")) != METHOD or str(row.get("method_key")) != METHOD:
        errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {METHOD}")
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
    if row.get("stage_full_cps") != [8, 4, 2]:
        errors.append(f"{run_name}: stage_full_cps {row.get('stage_full_cps')} != [8, 4, 2]")
    if row.get("stage_downsample") != [4, 2, 1]:
        errors.append(f"{run_name}: stage_downsample {row.get('stage_downsample')} != [4, 2, 1]")
    if row.get("stage_epochs") != expected_stage_epochs:
        errors.append(f"{run_name}: stage_epochs {row.get('stage_epochs')} != {expected_stage_epochs}")
    if str(row.get("stage_early_stop_mode")) != str(candidate.early_stop_mode):
        errors.append(f"{run_name}: stage_early_stop_mode {row.get('stage_early_stop_mode')} != {candidate.early_stop_mode}")
    if actual_stage_epochs:
        for idx, (actual, requested, min_epochs) in enumerate(zip(actual_stage_epochs, expected_stage_epochs, min_stage_epochs)):
            if actual > requested:
                errors.append(f"{run_name}: stage {idx} actual_epochs {actual} > requested {requested}")
            if str(candidate.early_stop_mode) == "adaptive" and actual < min_epochs:
                errors.append(f"{run_name}: stage {idx} actual_epochs {actual} < min {min_epochs}")
            if str(candidate.early_stop_mode) == "none" and actual != requested:
                errors.append(f"{run_name}: fixed stage {idx} actual_epochs {actual} != requested {requested}")
    for metric in [
        "dense_image_loss",
        "dice_mean",
        "hd95_mean",
        "asd_symmetric_mean",
        "be_native_analytic_fixed_fg_voxel",
        "diffusion_native_analytic_fixed_fg_voxel",
        "log_jdet_std_fg",
        "jdet_le0_pct_fg",
        "training_elapsed_sec",
        "total_elapsed_sec",
    ]:
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


def baseline_for_ranks(path: Path, ranks: Sequence[int]) -> Dict[str, float | int | str]:
    if not path.exists():
        return {
            "baseline_source": "hardcoded_first10",
            "baseline_n_cases": 10,
            "baseline_dice_mean": BASELINE_DICE,
            "baseline_hd95_mean": BASELINE_HD95,
        }
    rows = [
        row
        for row in c.read_csv(path)
        if str(row.get("method_key")) == METHOD
        and int(as_float(row.get("case_rank"))) in {int(rank) for rank in ranks}
        and float_close(row.get("alpha"), ALPHA)
    ]
    if not rows:
        return {
            "baseline_source": "hardcoded_first10",
            "baseline_n_cases": 10,
            "baseline_dice_mean": BASELINE_DICE,
            "baseline_hd95_mean": BASELINE_HD95,
        }
    return {
        "baseline_source": str(path),
        "baseline_n_cases": len(rows),
        "baseline_dice_mean": float(np.mean([as_float(row.get("dice_mean")) for row in rows])),
        "baseline_hd95_mean": float(np.mean([as_float(row.get("hd95_mean")) for row in rows])),
    }


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


def population_summary(rows: Sequence[Dict[str, object]], baseline: Mapping[str, object]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("runtime_candidate"))].append(row)
    out: List[Dict[str, object]] = []
    for candidate_name, group in sorted(grouped.items()):
        candidate = candidate_map()[candidate_name]
        case_ranks = sorted(int(as_float(row.get("case_rank"))) for row in group)
        row: Dict[str, object] = {
            "runtime_candidate": candidate_name,
            "description": candidate.description,
            "lr": float(candidate.lr),
            "stage_epochs": candidate.stage_epochs,
            "stage_lr": candidate.stage_lr,
            "stage_early_stop_mode": candidate.early_stop_mode,
            "stage_min_epochs": candidate.stage_min_epochs,
            "stage_patience": candidate.stage_patience,
            "stage_check_interval": candidate.stage_check_interval,
            "stage_rel_delta_total": candidate.stage_rel_delta_total,
            "stage_rel_delta_data": candidate.stage_rel_delta_data,
            "n_cases": len(group),
            "case_ranks_text": ",".join(str(rank) for rank in case_ranks),
            **baseline,
        }
        for metric in SUMMARY_METRICS:
            metric_stats = stats(group_row.get(metric) for group_row in group)
            for stat_name, value in metric_stats.items():
                row[f"{metric}_{stat_name}"] = value
        stage_actuals = [row_list(group_row.get("stage_actual_epochs")) for group_row in group]
        for stage_idx in range(3):
            metric_stats = stats(values[stage_idx] for values in stage_actuals if len(values) > stage_idx)
            for stat_name, value in metric_stats.items():
                row[f"stage{stage_idx}_actual_epochs_{stat_name}"] = value
        out.append(row)
    return out


def acceptance_rows(pop_rows: Sequence[Mapping[str, object]], expected_n: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in pop_rows:
        dice = as_float(row.get("dice_mean_mean"))
        hd95 = as_float(row.get("hd95_mean_mean"))
        train = as_float(row.get("training_elapsed_sec_mean"))
        total = as_float(row.get("total_elapsed_sec_mean"))
        baseline_dice = as_float(row.get("baseline_dice_mean"))
        baseline_hd95 = as_float(row.get("baseline_hd95_mean"))
        n_cases = int(as_float(row.get("n_cases")))
        finite = all(math.isfinite(value) for value in [dice, hd95, train, total])
        smoke_pass = (
            finite
            and n_cases == int(expected_n)
            and dice >= baseline_dice - SMOKE_DICE_DROP_LIMIT
            and hd95 <= baseline_hd95 + SMOKE_HD95_WORSE_LIMIT_MM
        )
        beats_directcp_cps2 = finite and dice >= DIRECTCP_CPS2_DICE and hd95 <= DIRECTCP_CPS2_HD95
        beats_sinr_cps2 = finite and dice >= SINR_CPS2_DICE and hd95 <= SINR_CPS2_HD95
        within_tolerance = finite and dice >= BASELINE_DICE - 0.02 and hd95 <= BASELINE_HD95 + 0.2
        runtime_target = finite and total <= RUNTIME_TOTAL_TARGET_SEC
        strong_runtime = finite and train <= RUNTIME_TRAIN_STRONG_SEC and total <= RUNTIME_TOTAL_STRONG_SEC
        if beats_directcp_cps2:
            tier = "A"
        elif beats_sinr_cps2:
            tier = "B"
        elif within_tolerance:
            tier = "C"
        else:
            tier = "drop"
        drop_reasons: List[str] = []
        if not finite:
            drop_reasons.append("nonfinite_metric")
        if n_cases != int(expected_n):
            drop_reasons.append(f"incomplete_n{n_cases}_of_{expected_n}")
        if dice < baseline_dice - SMOKE_DICE_DROP_LIMIT:
            drop_reasons.append("smoke_dice_drop_gt_0p05")
        if hd95 > baseline_hd95 + SMOKE_HD95_WORSE_LIMIT_MM:
            drop_reasons.append("smoke_hd95_worse_gt_0p5mm")
        rows.append(
            {
                "runtime_candidate": row.get("runtime_candidate"),
                "n_cases": n_cases,
                "dice_mean": dice,
                "hd95_mean": hd95,
                "asd_symmetric_mean": as_float(row.get("asd_symmetric_mean_mean")),
                "train_sec_mean": train,
                "total_sec_mean": total,
                "actual_epochs_mean": as_float(row.get("actual_epochs_mean")),
                "stage0_actual_epochs_mean": as_float(row.get("stage0_actual_epochs_mean")),
                "stage1_actual_epochs_mean": as_float(row.get("stage1_actual_epochs_mean")),
                "stage2_actual_epochs_mean": as_float(row.get("stage2_actual_epochs_mean")),
                "smoke_pass": smoke_pass,
                "beats_directcp_cps2": beats_directcp_cps2,
                "beats_sinr_cps2": beats_sinr_cps2,
                "within_user_tolerance": within_tolerance,
                "runtime_target": runtime_target,
                "strong_runtime_target": strong_runtime,
                "acceptance_tier": tier,
                "drop_reasons": ",".join(drop_reasons),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["smoke_pass"] else 1,
            {"A": 0, "B": 1, "C": 2, "drop": 3}.get(str(row["acceptance_tier"]), 4),
            as_float(row["total_sec_mean"]),
        )
    )
    return rows


def context_from_summary(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "run_name": row.get("run_name"),
        "runtime_candidate": row.get("runtime_candidate"),
        "case_rank": row.get("case_rank"),
        "original_case_index": row.get("original_case_index"),
        "fixed_id": row.get("fixed_id"),
        "moving_id": row.get("moving_id"),
        "alpha": row.get("alpha"),
        "lr": row.get("lr"),
        "summary_path": row.get("summary_path"),
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


def markdown_table(rows: Sequence[Mapping[str, object]]) -> List[str]:
    headers = [
        "candidate",
        "tier",
        "n",
        "Dice",
        "HD95",
        "train s",
        "total s",
        "epochs",
        "smoke",
        "runtime",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [
            str(row.get("runtime_candidate")),
            str(row.get("acceptance_tier")),
            fmt(row.get("n_cases"), 0),
            fmt(row.get("dice_mean"), 4),
            fmt(row.get("hd95_mean"), 4),
            fmt(row.get("train_sec_mean"), 1),
            fmt(row.get("total_sec_mean"), 1),
            fmt(row.get("actual_epochs_mean"), 1),
            "yes" if row.get("smoke_pass") else "no",
            "yes" if row.get("runtime_target") else "no",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    accept_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    baseline: Mapping[str, object],
) -> None:
    promoted = [str(row["runtime_candidate"]) for row in accept_rows if row.get("smoke_pass")]
    best_runtime = min(
        [row for row in accept_rows if row.get("smoke_pass")],
        key=lambda row: as_float(row.get("total_sec_mean")),
        default=None,
    )
    lines = [
        "# OASIS DirectCP 8-4-2 Runtime Sweep",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Out root: `{args.out_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Summaries: `{len(rows)}`",
        f"- Case ranks: `{args.case_ranks}`",
        f"- Candidates: `{args.candidates}`",
        f"- Alpha: `{float(args.alpha):g}`",
        f"- Variant: `{args.variant}`",
        f"- Baseline source: `{baseline.get('baseline_source')}`",
        f"- Matched baseline Dice/HD95: `{fmt(baseline.get('baseline_dice_mean'))}` / `{fmt(baseline.get('baseline_hd95_mean'))}`",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all expected runtime-sweep rows and artifacts are present.")
    if best_runtime:
        lines.append(
            "- Fastest smoke-passing candidate: "
            f"`{best_runtime['runtime_candidate']}` with total `{fmt(best_runtime['total_sec_mean'], 1)}s`, "
            f"Dice `{fmt(best_runtime['dice_mean'])}`, HD95 `{fmt(best_runtime['hd95_mean'])}`."
        )
    lines.extend(
        [
            f"- Smoke-promoted candidates: `{','.join(promoted)}`" if promoted else "- Smoke-promoted candidates: none.",
            "",
            "## Candidate Table",
            "",
        ]
    )
    lines.extend(markdown_table(accept_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs.csv/json`: per-case candidate summaries.",
            "- `candidate_population_summary.csv/json`: candidate population means and spreads.",
            "- `candidate_acceptance.csv/json`: smoke, tier, and runtime target decisions.",
            "- `promoted_candidates.txt`: comma-separated smoke-passing candidates.",
            "- `stage_summaries_combined.csv`: compact per-stage timing and stop reasons.",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def write_outputs(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    accept_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    baseline: Mapping[str, object],
) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs.csv", rows)
    c.write_json(args.review_root / "summary_runs.json", rows)
    c.write_csv(args.review_root / "candidate_population_summary.csv", pop_rows)
    c.write_json(args.review_root / "candidate_population_summary.json", pop_rows)
    c.write_csv(args.review_root / "candidate_acceptance.csv", accept_rows)
    c.write_json(args.review_root / "candidate_acceptance.json", accept_rows)
    promoted = [str(row["runtime_candidate"]) for row in accept_rows if row.get("smoke_pass")]
    (args.review_root / "promoted_candidates.txt").write_text(",".join(promoted) + "\n")
    c.write_csv(args.review_root / "case_metrics_combined.csv", collect_metric_file(rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_combined.csv", collect_metric_file(rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "stage_summaries_combined.csv", collect_metric_file(rows, "stage_summaries.csv"))
    c.write_csv(args.review_root / "prolongation_metrics_combined.csv", collect_metric_file(rows, "prolongation_metrics.csv"))
    if errors:
        c.write_json(args.review_root / "aggregation_errors.json", errors)
    write_report(args, rows, pop_rows, accept_rows, errors, baseline)


def main() -> None:
    args = parse_args()
    candidates = parse_candidate_names(str(args.candidates))
    ranks = base.parse_case_ranks(str(args.case_ranks))
    rows = collect_summaries(Path(args.out_root), candidates)
    errors = validate_rows(rows, args, candidates)
    baseline = baseline_for_ranks(Path(args.baseline_summary), ranks)
    pop_rows = population_summary(rows, baseline)
    accept_rows = acceptance_rows(pop_rows, expected_n=len(ranks))
    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("Runtime sweep aggregation failed: " + "; ".join(errors[:20]))
    write_outputs(args, rows, pop_rows, accept_rows, errors, baseline)
    print(
        json.dumps(
            {
                "combined_rows": len(rows),
                "candidate_count": len(pop_rows),
                "promoted_candidates": [row["runtime_candidate"] for row in accept_rows if row.get("smoke_pass")],
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
