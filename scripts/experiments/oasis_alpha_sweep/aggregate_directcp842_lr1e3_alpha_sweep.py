#!/usr/bin/env python
"""Aggregate the OASIS DirectCP 8-4-2 canonical lr=1e-3 alpha sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
METHOD_LABEL = "DirectCP 8-4-2"
VARIANT = "oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1"
DEFAULT_OUT_ROOT = Path(f"/database/inrorspline/runs/{VARIANT}")
DEFAULT_REVIEW_ROOT = (
    c.REPO_ROOT / "experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization"
)
LR = 1e-3
ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
STAGE_FULL_CPS = [8, 4, 2]
STAGE_DOWNSAMPLE = [4, 2, 1]
STAGE_EPOCHS = [400, 800, 1300]


def parse_float_list(text: str) -> List[float]:
    values = [float(token.strip()) for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError(f"No floats parsed from {text!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-19")
    parser.add_argument("--alphas", type=str, default=",".join(f"{alpha:g}" for alpha in ALPHAS))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--stage-full-cps", type=str, default=",".join(str(value) for value in STAGE_FULL_CPS))
    parser.add_argument("--stage-downsample", type=str, default=",".join(str(value) for value in STAGE_DOWNSAMPLE))
    parser.add_argument("--stage-epochs", type=str, default=",".join(str(value) for value in STAGE_EPOCHS))
    parser.add_argument("--expected-epochs", type=int, default=sum(STAGE_EPOCHS))
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--include-history-csvs",
        action="store_true",
        help="Also write high-volume per-epoch history_combined.csv and stage_history_combined.csv.",
    )
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


def collect_method_summaries(root: Path) -> List[Dict[str, object]]:
    rows = [
        row
        for row in base.collect_summaries(Path(root))
        if str(row.get("method_key")) == METHOD
    ]
    for row in rows:
        row["source_root_role"] = "canonical_lr1e3"
        row["source_root"] = str(root)
    return rows


def configure_method_spec(args: argparse.Namespace) -> None:
    base.METHOD_SPECS = {
        METHOD: {
            "label": METHOD_LABEL,
            "method": METHOD,
            "lr": float(args.lr),
            "alphas": parse_float_list(str(args.alphas)),
            "color": "#332288",
        }
    }


def expected_cases(args: argparse.Namespace) -> Dict[int, c.CaseInfo]:
    return {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in base.parse_case_ranks(str(args.case_ranks))
    }


def row_errors(row: Mapping[str, object], args: argparse.Namespace, cases: Mapping[int, c.CaseInfo]) -> List[str]:
    errors: List[str] = []
    run_name = str(row.get("run_name"))
    rank = int(as_float(row.get("case_rank")))
    alpha = as_float(row.get("alpha"))
    expected_alphas = parse_float_list(str(args.alphas))
    expected_stage_full_cps = c.parse_int_list(str(args.stage_full_cps))
    expected_stage_downsample = c.parse_int_list(str(args.stage_downsample))
    expected_stage_epochs = c.parse_int_list(str(args.stage_epochs))
    case = cases.get(rank)
    if case is None:
        return [f"{run_name}: unexpected case_rank {rank}"]
    if str(row.get("variant")) != str(args.variant):
        errors.append(f"{run_name}: variant {row.get('variant')} != {args.variant}")
    if str(row.get("method")) != METHOD or str(row.get("method_key")) != METHOD:
        errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {METHOD}")
    if not any(float_close(alpha, expected_alpha) for expected_alpha in expected_alphas):
        errors.append(f"{run_name}: unexpected alpha {row.get('alpha')}")
    if not float_close(row.get("lr"), float(args.lr)):
        errors.append(f"{run_name}: lr {row.get('lr')} != {float(args.lr):g}")
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
    if row.get("stage_full_cps") != expected_stage_full_cps:
        errors.append(f"{run_name}: stage_full_cps {row.get('stage_full_cps')} != {expected_stage_full_cps}")
    if row.get("stage_downsample") != expected_stage_downsample:
        errors.append(f"{run_name}: stage_downsample {row.get('stage_downsample')} != {expected_stage_downsample}")
    if row.get("stage_epochs") != expected_stage_epochs:
        errors.append(f"{run_name}: stage_epochs {row.get('stage_epochs')} != {expected_stage_epochs}")
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


def validate_rows(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    ranks = base.parse_case_ranks(str(args.case_ranks))
    alphas = parse_float_list(str(args.alphas))
    cases = expected_cases(args)
    expected = {(rank, alpha_id(alpha)) for rank in ranks for alpha in alphas}
    counts = Counter(combo_key(row) for row in rows)
    observed = set(counts)
    for key in sorted(expected - observed, key=lambda item: (item[0], as_float(item[1]))):
        errors.append(f"missing case_rank={key[0]} alpha={key[1]}")
    for key in sorted(observed - expected, key=lambda item: (item[0], as_float(item[1]))):
        errors.append(f"unexpected case_rank={key[0]} alpha={key[1]}")
    for key, count in sorted(counts.items(), key=lambda item: (item[0][0], as_float(item[0][1]))):
        if count != 1:
            errors.append(f"duplicate case_rank={key[0]} alpha={key[1]} count={count}")
    if len(rows) != len(expected):
        errors.append(f"expected {len(expected)} summaries, found {len(rows)}")
    for row in rows:
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


def population_for_ranks(args: argparse.Namespace, rows: Sequence[Dict[str, object]], ranks: Sequence[int]) -> List[Dict[str, object]]:
    configure_method_spec(args)
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
    rows: Sequence[Dict[str, object]],
    scoped_population: Mapping[str, Sequence[Dict[str, object]]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# OASIS DirectCP 8-4-2 Canonical lr=1e-3 Alpha Sweep",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Out root: `{args.out_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Summaries: `{len(rows)}`",
        f"- Alpha grid: `{args.alphas}`",
        f"- LR: `{float(args.lr):g}`",
        f"- Variant: `{args.variant}`",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all expected DirectCP 8-4-2 lr=1e-3 rows and artifacts are present.")
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
            "- `summary_runs_combined.csv/json`: all DirectCP 8-4-2 lr=1e-3 summaries.",
            "- `population_summary_directcp842_first10/first20.csv/json`: first10 and first20 population curves.",
            "- `stage_summaries_combined.csv` and `prolongation_metrics_combined.csv`: compact multistage diagnostics.",
            "- `history_combined.csv` and `stage_history_combined.csv` are opt-in high-volume diagnostics; use `--include-history-csvs` when needed.",
        ]
    )
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def write_outputs(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    scoped_population: Mapping[str, Sequence[Dict[str, object]]],
    errors: Sequence[str],
) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs_combined.csv", rows)
    c.write_json(args.review_root / "summary_runs_combined.json", rows)
    c.write_csv(args.review_root / "case_metrics_combined.csv", collect_metric_file_safe(rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_combined.csv", collect_metric_file_safe(rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_combined.csv", collect_metric_file_safe(rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_combined.csv", collect_metric_file_safe(rows, "diffusion_metrics.csv"))
    if args.include_history_csvs:
        c.write_csv(args.review_root / "history_combined.csv", base.collect_history(rows))
        c.write_csv(args.review_root / "stage_history_combined.csv", collect_metric_file_safe(rows, "stage_history.csv"))
    c.write_csv(args.review_root / "stage_summaries_combined.csv", collect_metric_file_safe(rows, "stage_summaries.csv"))
    c.write_csv(args.review_root / "prolongation_metrics_combined.csv", collect_metric_file_safe(rows, "prolongation_metrics.csv"))
    for scope, pop_rows in scoped_population.items():
        write_scope_outputs(args, scope, pop_rows)
    write_report(args, rows, scoped_population, errors)


def main() -> None:
    args = parse_args()
    ranks = base.parse_case_ranks(str(args.case_ranks))
    rows = sorted(
        collect_method_summaries(Path(args.out_root)),
        key=lambda row: (int(as_float(row.get("case_rank"))), as_float(row.get("alpha"))),
    )
    errors = validate_rows(rows, args)
    scoped_population: Dict[str, List[Dict[str, object]]] = {}
    if set(range(10)).issubset(set(ranks)):
        scoped_population["first10"] = population_for_ranks(args, rows, list(range(10)))
    if set(range(20)).issubset(set(ranks)):
        scoped_population["first20"] = population_for_ranks(args, rows, list(range(20)))
    if not scoped_population:
        scoped_population["requested"] = population_for_ranks(args, rows, ranks)

    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("DirectCP 8-4-2 lr=1e-3 aggregation failed: " + "; ".join(errors[:20]))

    write_outputs(args, rows, scoped_population, errors)
    print(
        json.dumps(
            {
                "combined_rows": len(rows),
                "population_scopes": {scope: len(pop_rows) for scope, pop_rows in scoped_population.items()},
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
