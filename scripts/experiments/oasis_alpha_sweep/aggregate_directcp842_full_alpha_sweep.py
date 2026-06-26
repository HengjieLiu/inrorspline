#!/usr/bin/env python
"""Aggregate the OASIS DirectCP 8-4-2 full six-alpha sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
METHOD_LABEL = "DirectCP 8-4-2 lr1e-2"
REMAINING_VARIANT = "oasis_seed3_first20_directcp842_alpha_sweep_remaining5_v1"
ALPHA1000_VARIANT = "oasis_seed3_first10_directcp842_alpha1000_prelim_v1"
DEFAULT_REMAINING_ROOT = Path(f"/database/inrorspline/runs/{REMAINING_VARIANT}")
DEFAULT_ALPHA1000_ROOT = Path(f"/database/inrorspline/runs/{ALPHA1000_VARIANT}")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_seed3_first20_directcp842_alpha_sweep"
REMAINING_ALPHAS = [0.0, 10.0, 100.0, 10000.0, 100000.0]
ALL_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
LR = 1e-2
STAGE_FULL_CPS = [8, 4, 2]
STAGE_DOWNSAMPLE = [4, 2, 1]
STAGE_EPOCHS = [400, 800, 1300]

MAIN_METRICS = [
    "dice_mean",
    "hd95_mean",
    "asd_warped_to_fixed_mean",
    "asd_fixed_to_warped_mean",
    "asd_symmetric_mean",
    "be_fd_forward_fixed_fg_voxel",
    "be_native_analytic_fixed_fg_voxel",
    "diffusion_l2_fg",
    "diffusion_fd_forward_fixed_fg_voxel",
    "diffusion_native_analytic_fixed_fg_voxel",
    "log_jdet_std_fg",
    "jdet_le0_pct_fg",
    "ndv_frac_fg",
    "training_elapsed_sec",
    "training_peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remaining-root", type=Path, default=DEFAULT_REMAINING_ROOT)
    parser.add_argument("--alpha1000-root", type=Path, default=DEFAULT_ALPHA1000_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-19")
    parser.add_argument("--expected-epochs", type=int, default=sum(STAGE_EPOCHS))
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    return base.as_float(value)


def is_finite(value: object) -> bool:
    return base.is_finite(value)


def alpha_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def float_close(value: object, target: float) -> bool:
    return math.isclose(as_float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)


def combo_key(row: Mapping[str, object]) -> Tuple[int, str]:
    return int(as_float(row.get("case_rank"))), alpha_id(row.get("alpha"))


def collect_method_summaries(root: Path, source: str) -> List[Dict[str, object]]:
    rows = [
        row
        for row in base.collect_summaries(Path(root))
        if str(row.get("method_key")) == METHOD
    ]
    for row in rows:
        row["source_root_role"] = source
        row["source_root"] = str(root)
    return rows


def configure_method_spec() -> None:
    base.METHOD_SPECS = {
        METHOD: {
            "label": METHOD_LABEL,
            "method": METHOD,
            "lr": LR,
            "alphas": ALL_ALPHAS,
            "color": "#332288",
        }
    }


def expected_cases(args: argparse.Namespace) -> Dict[int, c.CaseInfo]:
    return {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in base.parse_case_ranks(str(args.case_ranks))
    }


def stage_list(value: object) -> object:
    return value


def row_errors(row: Mapping[str, object], args: argparse.Namespace, cases: Mapping[int, c.CaseInfo]) -> List[str]:
    errors: List[str] = []
    run_name = str(row.get("run_name"))
    rank = int(as_float(row.get("case_rank")))
    alpha = as_float(row.get("alpha"))
    case = cases.get(rank)
    if case is None:
        return [f"{run_name}: unexpected case_rank {rank}"]
    expected_variant = ALPHA1000_VARIANT if float_close(alpha, 1000.0) else REMAINING_VARIANT
    if str(row.get("variant")) != expected_variant:
        errors.append(f"{run_name}: variant {row.get('variant')} != {expected_variant}")
    if str(row.get("method")) != METHOD or str(row.get("method_key")) != METHOD:
        errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {METHOD}")
    if not any(float_close(alpha, expected_alpha) for expected_alpha in ALL_ALPHAS):
        errors.append(f"{run_name}: unexpected alpha {row.get('alpha')}")
    if not float_close(row.get("lr"), LR):
        errors.append(f"{run_name}: lr {row.get('lr')} != {LR:g}")
    if int(as_float(row.get("epochs"))) != int(args.expected_epochs):
        errors.append(f"{run_name}: epochs {row.get('epochs')} != {args.expected_epochs}")
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
    if str(row.get("foreground_rule")) != "fixed raw image > 0":
        errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
    if stage_list(row.get("stage_full_cps")) != STAGE_FULL_CPS:
        errors.append(f"{run_name}: stage_full_cps {row.get('stage_full_cps')} != {STAGE_FULL_CPS}")
    if stage_list(row.get("stage_downsample")) != STAGE_DOWNSAMPLE:
        errors.append(f"{run_name}: stage_downsample {row.get('stage_downsample')} != {STAGE_DOWNSAMPLE}")
    if stage_list(row.get("stage_epochs")) != STAGE_EPOCHS:
        errors.append(f"{run_name}: stage_epochs {row.get('stage_epochs')} != {STAGE_EPOCHS}")
    for metric in [
        "dense_image_loss",
        "dice_mean",
        "hd95_mean",
        "asd_symmetric_mean",
        "be_native_analytic_fixed_fg_voxel",
        "diffusion_native_analytic_fixed_fg_voxel",
        "log_jdet_std_fg",
        "jdet_le0_pct_fg",
    ]:
        if not is_finite(row.get(metric)):
            errors.append(f"{run_name}: missing/non-finite required metric {metric}")
    errors.extend(f"{run_name}: {error}" for error in base.required_artifact_errors(row))
    run_dir = Path(str(row["run_dir_actual"]))
    for name in ["stage_history.csv", "stage_summaries.csv", "prolongation_metrics.csv"]:
        if not (run_dir / name).exists():
            errors.append(f"{run_name}: Missing {run_dir / name}")
    return errors


def validate_rows(
    remaining_rows: Sequence[Dict[str, object]],
    alpha1000_rows: Sequence[Dict[str, object]],
    combined_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> List[str]:
    errors: List[str] = []
    ranks = base.parse_case_ranks(str(args.case_ranks))
    cases = expected_cases(args)

    remaining_expected = {(rank, alpha_id(alpha)) for rank in ranks for alpha in REMAINING_ALPHAS}
    alpha1000_expected = {(rank, alpha_id(1000.0)) for rank in ranks}
    combined_expected = {(rank, alpha_id(alpha)) for rank in ranks for alpha in ALL_ALPHAS}

    for label, rows, expected in [
        ("remaining", remaining_rows, remaining_expected),
        ("alpha1000", alpha1000_rows, alpha1000_expected),
        ("combined", combined_rows, combined_expected),
    ]:
        counts = Counter(combo_key(row) for row in rows)
        observed = set(counts)
        for key in sorted(expected - observed, key=lambda item: (item[0], as_float(item[1]))):
            errors.append(f"{label}: missing case_rank={key[0]} alpha={key[1]}")
        for key in sorted(observed - expected, key=lambda item: (item[0], as_float(item[1]))):
            errors.append(f"{label}: unexpected case_rank={key[0]} alpha={key[1]}")
        for key, count in sorted(counts.items(), key=lambda item: (item[0][0], as_float(item[0][1]))):
            if count != 1:
                errors.append(f"{label}: duplicate case_rank={key[0]} alpha={key[1]} count={count}")

    if len(remaining_rows) != len(remaining_expected):
        errors.append(f"remaining: expected {len(remaining_expected)} summaries, found {len(remaining_rows)}")
    if len(alpha1000_rows) != len(alpha1000_expected):
        errors.append(f"alpha1000: expected {len(alpha1000_expected)} summaries, found {len(alpha1000_rows)}")
    if len(combined_rows) != len(combined_expected):
        errors.append(f"combined: expected {len(combined_expected)} summaries, found {len(combined_rows)}")
    for row in combined_rows:
        errors.extend(row_errors(row, args, cases))
    return errors


def metric_mean(row: Mapping[str, object], metric: str) -> float:
    return as_float(row.get(f"{metric}_mean"))


def fmt(value: object, digits: int = 4) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.3g}"
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def markdown_table(rows: Sequence[Mapping[str, object]], metrics: Sequence[str]) -> List[str]:
    headers = ["alpha", "n"] + [metric.replace("_", " ") for metric in metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in sorted(rows, key=lambda item: as_float(item.get("alpha"))):
        values = [fmt(row.get("alpha"), digits=0), fmt(row.get("n_cases"), digits=0)]
        values.extend(fmt(metric_mean(row, metric)) for metric in metrics)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def population_for_ranks(rows: Sequence[Dict[str, object]], ranks: Sequence[int]) -> List[Dict[str, object]]:
    configure_method_spec()
    selected = base.filter_summaries_by_case_rank(rows, ranks)
    return base.population_summary(selected)


def collect_metric_file_safe(rows: Sequence[Dict[str, object]], name: str) -> List[Dict[str, object]]:
    return base.collect_metric_file(rows, name)


def write_scope_outputs(args: argparse.Namespace, scope: str, pop_rows: Sequence[Dict[str, object]]) -> None:
    c.write_csv(args.review_root / f"population_summary_directcp842_{scope}.csv", pop_rows)
    c.write_json(args.review_root / f"population_summary_directcp842_{scope}.json", pop_rows)


def best_line(scope: str, pop_rows: Sequence[Dict[str, object]]) -> str:
    best_dice = max(pop_rows, key=lambda row: metric_mean(row, "dice_mean"))
    best_hd95 = min(pop_rows, key=lambda row: metric_mean(row, "hd95_mean"))
    best_asd = min(pop_rows, key=lambda row: metric_mean(row, "asd_symmetric_mean"))
    return (
        f"- {scope}: best Dice alpha `{fmt(best_dice.get('alpha'), digits=0)}` "
        f"(Dice `{fmt(metric_mean(best_dice, 'dice_mean'))}`), best HD95 alpha "
        f"`{fmt(best_hd95.get('alpha'), digits=0)}`, best symmetric ASD alpha "
        f"`{fmt(best_asd.get('alpha'), digits=0)}`."
    )


def write_report(
    args: argparse.Namespace,
    remaining_rows: Sequence[Dict[str, object]],
    alpha1000_rows: Sequence[Dict[str, object]],
    combined_rows: Sequence[Dict[str, object]],
    scoped_population: Mapping[str, Sequence[Dict[str, object]]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# OASIS DirectCP 8-4-2 Full Alpha Sweep",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Remaining-root: `{args.remaining_root}`",
        f"- Reused alpha1000 root: `{args.alpha1000_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Remaining summaries: `{len(remaining_rows)}`",
        f"- Reused alpha1000 summaries: `{len(alpha1000_rows)}`",
        f"- Combined summaries: `{len(combined_rows)}`",
        "- Alpha grid: `0,10,100,1000,10000,100000`; alpha `1000` is reused, not rerun.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all expected combined DirectCP 8-4-2 rows and artifacts are present.")
    lines.extend(["", "## Best Alpha Summary", ""])
    for scope, pop_rows in scoped_population.items():
        lines.append(best_line(scope, pop_rows))
    for scope, pop_rows in scoped_population.items():
        lines.extend(["", f"## {scope}", ""])
        lines.extend(
            markdown_table(
                pop_rows,
                [
                    "dice_mean",
                    "hd95_mean",
                    "asd_symmetric_mean",
                    "be_native_analytic_fixed_fg_voxel",
                    "diffusion_native_analytic_fixed_fg_voxel",
                    "log_jdet_std_fg",
                    "jdet_le0_pct_fg",
                    "ndv_frac_fg",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs_combined.csv/json`: all 120 DirectCP 8-4-2 summaries.",
            "- `summary_runs_remaining5.csv/json`: the 100 newly launched remaining-alpha summaries.",
            "- `summary_runs_alpha1000_reused.csv/json`: the 20 reused alpha1000 summaries.",
            "- `population_summary_directcp842_first10/first20.csv/json`: first10 and first20 population curves.",
            "- `stage_history_combined.csv`, `stage_summaries_combined.csv`, and `prolongation_metrics_combined.csv`: multistage diagnostics.",
        ]
    )
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def write_outputs(
    args: argparse.Namespace,
    remaining_rows: Sequence[Dict[str, object]],
    alpha1000_rows: Sequence[Dict[str, object]],
    combined_rows: Sequence[Dict[str, object]],
    scoped_population: Mapping[str, Sequence[Dict[str, object]]],
    errors: Sequence[str],
) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs_remaining5.csv", remaining_rows)
    c.write_json(args.review_root / "summary_runs_remaining5.json", remaining_rows)
    c.write_csv(args.review_root / "summary_runs_alpha1000_reused.csv", alpha1000_rows)
    c.write_json(args.review_root / "summary_runs_alpha1000_reused.json", alpha1000_rows)
    c.write_csv(args.review_root / "summary_runs_combined.csv", combined_rows)
    c.write_json(args.review_root / "summary_runs_combined.json", combined_rows)
    c.write_csv(args.review_root / "case_metrics_combined.csv", collect_metric_file_safe(combined_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_combined.csv", collect_metric_file_safe(combined_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_combined.csv", collect_metric_file_safe(combined_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_combined.csv", collect_metric_file_safe(combined_rows, "diffusion_metrics.csv"))
    c.write_csv(args.review_root / "history_combined.csv", base.collect_history(combined_rows))
    c.write_csv(args.review_root / "stage_history_combined.csv", collect_metric_file_safe(combined_rows, "stage_history.csv"))
    c.write_csv(args.review_root / "stage_summaries_combined.csv", collect_metric_file_safe(combined_rows, "stage_summaries.csv"))
    c.write_csv(args.review_root / "prolongation_metrics_combined.csv", collect_metric_file_safe(combined_rows, "prolongation_metrics.csv"))
    for scope, pop_rows in scoped_population.items():
        write_scope_outputs(args, scope, pop_rows)
    write_report(args, remaining_rows, alpha1000_rows, combined_rows, scoped_population, errors)


def main() -> None:
    args = parse_args()
    ranks = base.parse_case_ranks(str(args.case_ranks))
    remaining_rows = collect_method_summaries(Path(args.remaining_root), "remaining5")
    alpha1000_rows = collect_method_summaries(Path(args.alpha1000_root), "alpha1000_reused")
    combined_rows = sorted(
        list(remaining_rows) + list(alpha1000_rows),
        key=lambda row: (int(as_float(row.get("case_rank"))), as_float(row.get("alpha"))),
    )
    errors = validate_rows(remaining_rows, alpha1000_rows, combined_rows, args)
    scoped_population: Dict[str, List[Dict[str, object]]] = {}
    if set(range(10)).issubset(set(ranks)):
        scoped_population["first10"] = population_for_ranks(combined_rows, list(range(10)))
    if set(range(20)).issubset(set(ranks)):
        scoped_population["first20"] = population_for_ranks(combined_rows, list(range(20)))
    if not scoped_population:
        scoped_population["requested"] = population_for_ranks(combined_rows, ranks)

    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("DirectCP 8-4-2 full alpha aggregation failed: " + "; ".join(errors[:20]))

    write_outputs(args, remaining_rows, alpha1000_rows, combined_rows, scoped_population, errors)
    print(
        json.dumps(
            {
                "remaining_rows": len(remaining_rows),
                "alpha1000_reused_rows": len(alpha1000_rows),
                "combined_rows": len(combined_rows),
                "population_scopes": {scope: len(rows) for scope, rows in scoped_population.items()},
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
