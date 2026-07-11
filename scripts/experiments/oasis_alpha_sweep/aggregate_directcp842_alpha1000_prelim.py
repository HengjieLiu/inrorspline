#!/usr/bin/env python
"""Aggregate OASIS DirectCP 8-4-2 alpha=1000 preliminary results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import aggregate_first10_population as base  # noqa: E402


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
METHOD_LABEL = "DirectCP 8-4-2 lr1e-2"
VARIANT = "oasis_seed3_first10_directcp842_alpha1000_prelim_v1"
DEFAULT_OUT_ROOT = Path(f"/database/inrorspline/runs/{VARIANT}")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim"
DEFAULT_REFERENCE_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay"
DEFAULT_NEXT10_REFERENCE_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_seed3_next10_alpha_sweep"
REFERENCE_FIRST10_SUMMARY_CSV = "summary_runs_with_direct_cp_cps2_full.csv"
REFERENCE_NEXT10_SUMMARY_CSV = "summary_runs.csv"
ALPHA = 1000.0
LR = 1e-2
STAGE_FULL_CPS = [8, 4, 2]
STAGE_DOWNSAMPLE = [4, 2, 1]
STAGE_EPOCHS = [400, 800, 1300]

REFERENCE_ROWS: Sequence[Tuple[str, str, float, float]] = [
    ("IDIR best Dice", "idir_original_dense_siren_be_autograd", 1.0, 1e-4),
    ("SINR cps4 alpha1000", "sinr_cps4_be_bspline_analytic", 1000.0, 1e-4),
    ("SINR cps2 alpha1000", "sinr_cps2_be_bspline_analytic", 1000.0, 1e-4),
    ("DirectCP cps4 alpha1000", "direct_cp_cps4_be_bspline_analytic", 1000.0, 1e-3),
    ("DirectCP cps2 alpha1000", "direct_cp_cps2_be_bspline_analytic", 1000.0, 1e-3),
]

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
    "jdet_le0_pct_fg",
    "log_jdet_std_fg",
    "ndv_frac_fg",
    "training_elapsed_sec",
    "training_peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--next10-reference-root", type=Path, default=DEFAULT_NEXT10_REFERENCE_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--expected-epochs", type=int, default=sum(STAGE_EPOCHS))
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    return base.as_float(value)


def float_close(value: object, target: float) -> bool:
    return math.isclose(as_float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def method_specs_for_table() -> Dict[str, Dict[str, object]]:
    colors = {
        "idir_original_dense_siren_be_autograd": "#1f77b4",
        "sinr_cps4_be_bspline_analytic": "#2ca02c",
        "sinr_cps2_be_bspline_analytic": "#9467bd",
        "direct_cp_cps4_be_bspline_analytic": "#d62728",
        "direct_cp_cps2_be_bspline_analytic": "#ff7f0e",
        METHOD: "#8c564b",
    }
    specs: Dict[str, Dict[str, object]] = {}
    for label, method_key, alpha, lr in REFERENCE_ROWS:
        specs[method_key] = {
            "label": label,
            "method": method_key,
            "lr": lr,
            "alphas": [alpha],
            "color": colors[method_key],
        }
    specs[METHOD] = {
        "label": METHOD_LABEL,
        "method": METHOD,
        "lr": LR,
        "alphas": [ALPHA],
        "color": colors[METHOD],
    }
    return specs


def configure_table_method_specs() -> None:
    base.METHOD_SPECS = method_specs_for_table()


def comparison_label(method_key: str, alpha: object, lr: object) -> str:
    for label, ref_method, ref_alpha, ref_lr in REFERENCE_ROWS:
        if method_key == ref_method and float_close(alpha, ref_alpha) and float_close(lr, ref_lr):
            return label
    if method_key == METHOD and float_close(alpha, ALPHA) and float_close(lr, LR):
        return "DirectCP 8-4-2 alpha1000"
    return method_key


def select_reference_case_rows(rows: Sequence[Dict[str, object]], ranks: Sequence[int], source: str) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    rank_set = {int(rank) for rank in ranks}
    for label, method_key, alpha, lr in REFERENCE_ROWS:
        by_rank: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            if (
                int(as_float(row.get("case_rank"))) in rank_set
                and str(row.get("method_key")) == method_key
                and float_close(row.get("alpha"), alpha)
                and float_close(row.get("lr"), lr)
            ):
                by_rank[int(as_float(row.get("case_rank")))].append(row)
        for rank in ranks:
            group = by_rank.get(int(rank), [])
            if len(group) != 1:
                raise RuntimeError(f"Expected one {source} reference row for {label} rank {rank}, found {len(group)}")
            row = dict(group[0])
            row["comparison_label"] = label
            row["source"] = source
            selected.append(row)
    return selected


def load_reference_case_rows(args: argparse.Namespace, ranks: Sequence[int]) -> List[Dict[str, object]]:
    ranks_first10 = [rank for rank in ranks if 0 <= int(rank) <= 9]
    ranks_next10 = [rank for rank in ranks if 10 <= int(rank) <= 19]
    unsupported = sorted(set(int(rank) for rank in ranks) - set(ranks_first10) - set(ranks_next10))
    if unsupported:
        raise RuntimeError(f"No reference rows configured for ranks outside 0-19: {unsupported}")

    selected: List[Dict[str, object]] = []
    if ranks_first10:
        first10_path = Path(args.reference_root) / REFERENCE_FIRST10_SUMMARY_CSV
        selected.extend(select_reference_case_rows(read_csv_rows(first10_path), ranks_first10, "existing_first10_reference"))
    if ranks_next10:
        next10_path = Path(args.next10_reference_root) / REFERENCE_NEXT10_SUMMARY_CSV
        selected.extend(select_reference_case_rows(read_csv_rows(next10_path), ranks_next10, "existing_next10_reference"))
    return selected


def expected_cases(args: argparse.Namespace) -> Dict[int, c.CaseInfo]:
    return {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in base.parse_case_ranks(str(args.case_ranks))
    }


def check_new_rows(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    cases = expected_cases(args)
    by_rank: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_rank.setdefault(int(as_float(row.get("case_rank"))), []).append(row)
    for rank, case in cases.items():
        group = by_rank.get(int(rank), [])
        if len(group) != 1:
            errors.append(f"Expected one new DirectCP 8-4-2 row for rank {rank}, found {len(group)}")
            continue
        row = group[0]
        run_name = str(row.get("run_name"))
        if str(row.get("variant")) != VARIANT:
            errors.append(f"{run_name}: variant {row.get('variant')} != {VARIANT}")
        if str(row.get("method")) != METHOD or str(row.get("method_key")) != METHOD:
            errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {METHOD}")
        if not float_close(row.get("alpha"), ALPHA):
            errors.append(f"{run_name}: alpha {row.get('alpha')} != {ALPHA:g}")
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
            errors.append(f"{run_name}: native BE formula is {row.get('native_analytic_be_formula')}")
        if str(row.get("native_analytic_diffusion_formula")) != "bspline_analytic":
            errors.append(f"{run_name}: native diffusion formula is {row.get('native_analytic_diffusion_formula')}")
        if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
            errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
        if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
            errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
        if str(row.get("foreground_rule")) != "fixed raw image > 0":
            errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
        if row.get("stage_full_cps") != STAGE_FULL_CPS:
            errors.append(f"{run_name}: stage_full_cps {row.get('stage_full_cps')} != {STAGE_FULL_CPS}")
        if row.get("stage_downsample") != STAGE_DOWNSAMPLE:
            errors.append(f"{run_name}: stage_downsample {row.get('stage_downsample')} != {STAGE_DOWNSAMPLE}")
        if row.get("stage_epochs") != STAGE_EPOCHS:
            errors.append(f"{run_name}: stage_epochs {row.get('stage_epochs')} != {STAGE_EPOCHS}")
        for metric in [
            "dense_image_loss",
            "dice_mean",
            "hd95_mean",
            "asd_symmetric_mean",
            "be_native_analytic_fixed_fg_voxel",
            "diffusion_native_analytic_fixed_fg_voxel",
        ]:
            if not base.is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite required metric {metric}")
        errors.extend(f"{run_name}: {error}" for error in base.required_artifact_errors(row))
        run_dir = Path(str(row["run_dir_actual"]))
        for name in ["stage_history.csv", "stage_summaries.csv", "prolongation_metrics.csv"]:
            if not (run_dir / name).exists():
                errors.append(f"{run_name}: Missing {run_dir / name}")
    extra_ranks = sorted(set(by_rank) - set(cases))
    for rank in extra_ranks:
        errors.append(f"Unexpected new DirectCP 8-4-2 case rank {rank}")
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
    headers = ["Method", "n", "alpha", "lr"] + [metric.replace("_", " ") for metric in metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [
            str(row.get("comparison_label") or row.get("method_label") or row.get("method_key")),
            fmt(row.get("n_cases"), digits=0),
            fmt(row.get("alpha"), digits=0),
            fmt(row.get("lr"), digits=5),
        ]
        values.extend(fmt(metric_mean(row, metric)) for metric in metrics)
        lines.append("| " + " | ".join(values) + " |")
    return lines


COMPREHENSIVE_TABLE_METRICS = [
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
]


def annotate_population_rows(rows: Sequence[Dict[str, object]], scope: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        annotated = dict(row)
        annotated["comparison_label"] = comparison_label(str(row.get("method_key")), row.get("alpha"), row.get("lr"))
        annotated["scope"] = scope
        annotated["source"] = "population_from_case_summaries"
        out.append(annotated)
    return out


def population_for_scope(
    args: argparse.Namespace,
    scope: str,
    ranks: Sequence[int],
    new_rows_all: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    reference_case_rows = load_reference_case_rows(args, ranks)
    rank_set = {int(rank) for rank in ranks}
    new_scope_rows = [
        row
        for row in base.filter_summaries_by_case_rank(new_rows_all, ranks)
        if str(row.get("method_key")) == METHOD and int(as_float(row.get("case_rank"))) in rank_set
    ]
    configure_table_method_specs()
    pop_rows = base.population_summary(list(reference_case_rows) + list(new_scope_rows))
    return annotate_population_rows(pop_rows, scope)


def compact_population_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    compact_rows: List[Dict[str, object]] = []
    for row in rows:
        compact = {
            "scope": row.get("scope"),
            "comparison_label": row.get("comparison_label"),
            "method_key": row.get("method_key"),
            "alpha": row.get("alpha"),
            "lr": row.get("lr"),
            "n_cases": row.get("n_cases"),
            "case_ranks_text": row.get("case_ranks_text"),
        }
        for metric in MAIN_METRICS:
            compact[f"{metric}_mean"] = row.get(f"{metric}_mean")
            compact[f"{metric}_std"] = row.get(f"{metric}_std")
        compact_rows.append(compact)
    return compact_rows


def write_population_outputs(args: argparse.Namespace, scope: str, rows: Sequence[Dict[str, object]]) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / f"comparison_population_{scope}_comprehensive.csv", rows)
    c.write_json(args.review_root / f"comparison_population_{scope}_comprehensive.json", rows)
    c.write_csv(args.review_root / f"comparison_population_{scope}_main.csv", compact_population_rows(rows))


def write_report(
    args: argparse.Namespace,
    scoped_population: Mapping[str, Sequence[Dict[str, object]]],
    new_summary_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# OASIS DirectCP 8-4-2 Alpha1000 Preliminary Comparison",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- New output root: `{args.out_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Reference root: `{args.reference_root}`",
        f"- Next10 reference root: `{args.next10_reference_root}`",
        f"- New summaries: `{len(new_summary_rows)}` / `{len(base.parse_case_ranks(str(args.case_ranks)))}`",
        f"- New case ranks: `{','.join(str(rank) for rank in base.parse_case_ranks(str(args.case_ranks)))}`",
        "- New method: DirectCP multistage `8-4-2`, downsample `4-2-1`, stage epochs `400-800-1300`, lr `1e-2`, alpha `1000`.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all new rows and required artifacts are present with expected metadata.")

    lines.extend(["", "## Comparison Tables", ""])
    for scope, rows in scoped_population.items():
        best_dice = max(rows, key=lambda row: metric_mean(row, "dice_mean"))
        new_rows = [row for row in rows if str(row.get("method_key")) == METHOD]
        if new_rows:
            new_row = new_rows[0]
            lines.extend(
                [
                    f"### {scope}",
                    "",
                    f"- Best mean Dice: `{best_dice.get('comparison_label') or best_dice.get('method_label')}` with Dice `{fmt(metric_mean(best_dice, 'dice_mean'))}`.",
                    f"- New mean Dice/HD95/symmetric ASD: `{fmt(metric_mean(new_row, 'dice_mean'))}` / `{fmt(metric_mean(new_row, 'hd95_mean'))}` / `{fmt(metric_mean(new_row, 'asd_symmetric_mean'))}`.",
                    f"- New foreground native BE/logJ std/fold%: `{fmt(metric_mean(new_row, 'be_native_analytic_fixed_fg_voxel'))}` / `{fmt(metric_mean(new_row, 'log_jdet_std_fg'))}` / `{fmt(metric_mean(new_row, 'jdet_le0_pct_fg'))}`.",
                    "",
                ]
            )
        else:
            lines.extend([f"### {scope}", "", "- New DirectCP 8-4-2 row is not complete for this scope.", ""])
        lines.extend(markdown_table(rows, COMPREHENSIVE_TABLE_METRICS))
        lines.append("")

    if scoped_population:
        final_scope, final_rows = list(scoped_population.items())[-1]
        lines.extend(["## Runtime", ""])
        lines.append(f"Runtime table for `{final_scope}`.")
        lines.append("")
        lines.extend(markdown_table(final_rows, ["training_elapsed_sec", "total_elapsed_sec", "training_peak_mem_gb"]))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison_population_first10_comprehensive.csv/json` and `comparison_population_first10_main.csv`: first10 comparison rows.",
            "- `comparison_population_first20_comprehensive.csv/json` and `comparison_population_first20_main.csv`: first20 comparison rows, when ranks `0-19` are available.",
            "- `comparison_population_comprehensive.csv/json`: compatibility copy of the largest completed/requested scope.",
            "- `summary_runs_new.csv/json`: new per-case summaries for the requested ranks.",
            "- `stage_history_new.csv`, `stage_summaries_new.csv`, and `prolongation_metrics_new.csv`: multistage diagnostics.",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "comparison_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    new_rows_all = base.collect_summaries(Path(args.out_root))
    new_rows = [
        row
        for row in base.filter_summaries_by_case_rank(new_rows_all, case_ranks)
        if str(row.get("method_key")) == METHOD
    ]
    errors = check_new_rows(new_rows, args)

    requested = set(case_ranks)
    scopes: List[Tuple[str, List[int]]] = []
    if set(range(10)).issubset(requested):
        scopes.append(("first10", list(range(10))))
    if set(range(20)).issubset(requested):
        scopes.append(("first20", list(range(20))))
    if not scopes:
        scopes.append(("requested", case_ranks))

    scoped_population: Dict[str, List[Dict[str, object]]] = {}
    for scope, ranks in scopes:
        rows = population_for_scope(args, scope, ranks, new_rows_all)
        new_scope_rows = [row for row in rows if str(row.get("method_key")) == METHOD]
        if len(new_scope_rows) != 1:
            errors.append(f"Expected one new population row for {scope}, found {len(new_scope_rows)}")
        elif int(as_float(new_scope_rows[0].get("n_cases"))) != len(ranks):
            errors.append(f"{scope}: new DirectCP 8-4-2 n_cases {new_scope_rows[0].get('n_cases')} != {len(ranks)}")
        scoped_population[scope] = rows

    args.review_root.mkdir(parents=True, exist_ok=True)
    if errors and not args.allow_incomplete:
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("DirectCP 8-4-2 alpha1000 aggregation failed: " + "; ".join(errors[:20]))

    c.write_csv(args.review_root / "summary_runs_new.csv", new_rows)
    c.write_json(args.review_root / "summary_runs_new.json", new_rows)
    c.write_csv(args.review_root / "case_metrics_new.csv", base.collect_metric_file(new_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_new.csv", base.collect_metric_file(new_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_new.csv", base.collect_metric_file(new_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_new.csv", base.collect_metric_file(new_rows, "diffusion_metrics.csv"))
    c.write_csv(args.review_root / "history_new.csv", base.collect_history(new_rows))
    c.write_csv(args.review_root / "stage_history_new.csv", base.collect_metric_file(new_rows, "stage_history.csv"))
    c.write_csv(args.review_root / "stage_summaries_new.csv", base.collect_metric_file(new_rows, "stage_summaries.csv"))
    c.write_csv(args.review_root / "prolongation_metrics_new.csv", base.collect_metric_file(new_rows, "prolongation_metrics.csv"))
    for scope, rows in scoped_population.items():
        write_population_outputs(args, scope, rows)
    if scoped_population:
        _, largest_rows = list(scoped_population.items())[-1]
        c.write_csv(args.review_root / "comparison_population_comprehensive.csv", largest_rows)
        c.write_json(args.review_root / "comparison_population_comprehensive.json", largest_rows)
        c.write_csv(args.review_root / "comparison_population_main.csv", compact_population_rows(largest_rows))
    write_report(args, scoped_population, new_rows, errors)
    print(
        json.dumps(
            {
                "new_summary_rows": len(new_rows),
                "comparison_scopes": {scope: len(rows) for scope, rows in scoped_population.items()},
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
