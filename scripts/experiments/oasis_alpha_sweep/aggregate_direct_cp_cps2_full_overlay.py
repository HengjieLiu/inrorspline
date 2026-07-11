#!/usr/bin/env python
"""Aggregate original first10 results plus the full DirectCP cps2 alpha overlay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


DEFAULT_ORIGINAL_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
DEFAULT_PRELIM_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1")
DEFAULT_COMPLETION_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay"

ADDON_METHOD = "direct_cp_cps2_be_bspline_analytic"
ADDON_LABEL = "DirectCP cps2 lr1e-3"
ADDON_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
ADDON_LR = 1e-3
PRELIM_VARIANT = "oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1"
COMPLETION_VARIANT = "oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1"
PRELIM_ALPHAS = {base.alpha_id(alpha) for alpha in [100.0, 1000.0, 10000.0]}
COMPLETION_ALPHAS = {base.alpha_id(alpha) for alpha in [0.0, 10.0, 100000.0]}
SUFFIX = "with_direct_cp_cps2_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--prelim-root", type=Path, default=DEFAULT_PRELIM_ROOT)
    parser.add_argument("--completion-root", type=Path, default=DEFAULT_COMPLETION_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-history", action="store_true")
    return parser.parse_args()


def configure_method_specs() -> None:
    specs = dict(base.METHOD_SPECS)
    specs[ADDON_METHOD] = {
        "label": ADDON_LABEL,
        "method": ADDON_METHOD,
        "lr": ADDON_LR,
        "alphas": ADDON_ALPHAS,
        "color": "#ff7f0e",
    }
    base.METHOD_SPECS = specs


def isclose_alpha(value: object, target: float) -> bool:
    return math.isclose(base.as_float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)


def expected_cps2_combos(case_ranks: Sequence[int]) -> Dict[Tuple[int, str, str, str], Dict[str, object]]:
    expected: Dict[Tuple[int, str, str, str], Dict[str, object]] = {}
    for rank in case_ranks:
        for alpha in ADDON_ALPHAS:
            key = (int(rank), ADDON_METHOD, base.alpha_id(alpha), base.alpha_id(ADDON_LR))
            expected[key] = {
                "case_rank": int(rank),
                "method_key": ADDON_METHOD,
                "alpha": float(alpha),
                "lr": ADDON_LR,
            }
    return expected


def expected_variant(alpha: object) -> str | None:
    alpha_text = base.alpha_id(alpha)
    if alpha_text in PRELIM_ALPHAS:
        return PRELIM_VARIANT
    if alpha_text in COMPLETION_ALPHAS:
        return COMPLETION_VARIANT
    return None


def collect_cps2_rows(args: argparse.Namespace, case_ranks: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for root_label, root in [("prelim", args.prelim_root), ("completion", args.completion_root)]:
        for row in base.filter_summaries_by_case_rank(base.collect_summaries(Path(root)), case_ranks):
            if str(row.get("method_key")) == ADDON_METHOD:
                row = dict(row)
                row["cps2_source_root_label"] = root_label
                row["cps2_source_root"] = str(root)
                rows.append(row)
    rows.sort(key=lambda row: (int(base.as_float(row.get("case_rank"))), base.as_float(row.get("alpha"))))
    return rows


def check_cps2_expected(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    cases = {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in case_ranks
    }
    expected = expected_cps2_combos(case_ranks)
    observed: Dict[Tuple[int, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        observed[base.combo_key(row)].append(row)

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    for key in missing:
        errors.append(f"Missing DirectCP cps2 combo: case_rank={key[0]}, alpha={key[2]}, lr={key[3]}")
    for key in extra:
        errors.append(f"Unexpected DirectCP cps2 combo: case_rank={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate DirectCP cps2 combo {key}: {len(group)} summaries")

    for row in rows:
        run_name = str(row.get("run_name"))
        rank = int(base.as_float(row.get("case_rank")))
        case = cases.get(rank)
        if case is None:
            continue
        if int(base.as_float(row.get("original_case_index"))) != int(case.original_case_index):
            errors.append(f"{run_name}: original_case_index {row.get('original_case_index')} != {case.original_case_index}")
        if str(row.get("fixed_id")) != case.fixed_id or str(row.get("moving_id")) != case.moving_id:
            errors.append(f"{run_name}: fixed/moving {row.get('fixed_id')}/{row.get('moving_id')} != {case.fixed_id}/{case.moving_id}")
        variant = expected_variant(row.get("alpha"))
        if variant is None:
            errors.append(f"{run_name}: alpha {row.get('alpha')} is outside DirectCP cps2 full-alpha grid")
        elif str(row.get("variant")) != variant:
            errors.append(f"{run_name}: variant {row.get('variant')} != {variant}")
        if str(row.get("method")) != ADDON_METHOD or str(row.get("method_key")) != ADDON_METHOD:
            errors.append(f"{run_name}: method/method_key {row.get('method')}/{row.get('method_key')} != {ADDON_METHOD}")
        if int(base.as_float(row.get("epochs"))) != int(args.expected_epochs):
            errors.append(f"{run_name}: epochs {row.get('epochs')} != {args.expected_epochs}")
        if int(base.as_float(row.get("eval_interval"))) != int(args.expected_eval_interval):
            errors.append(f"{run_name}: eval_interval {row.get('eval_interval')} != {args.expected_eval_interval}")
        if int(base.as_float(row.get("seed"))) != int(args.expected_seed):
            errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
        if not isclose_alpha(row.get("lr"), ADDON_LR):
            errors.append(f"{run_name}: lr {row.get('lr')} != {ADDON_LR}")
        if not any(isclose_alpha(row.get("alpha"), alpha) for alpha in ADDON_ALPHAS):
            errors.append(f"{run_name}: alpha {row.get('alpha')} not in {ADDON_ALPHAS}")
        if int(base.as_float(row.get("cps"))) != 2:
            errors.append(f"{run_name}: cps {row.get('cps')} != 2")
        if str(row.get("activation")) != "direct_control_points":
            errors.append(f"{run_name}: activation {row.get('activation')} != direct_control_points")
        if str(row.get("sample_domain")) != "none_full_image_ncc":
            errors.append(f"{run_name}: sample_domain {row.get('sample_domain')} != none_full_image_ncc")
        if str(row.get("native_analytic_be_formula")) != "bspline_analytic":
            errors.append(f"{run_name}: native BE formula is {row.get('native_analytic_be_formula')}")
        if str(row.get("native_analytic_diffusion_formula")) != "bspline_analytic":
            errors.append(f"{run_name}: native diffusion formula is {row.get('native_analytic_diffusion_formula')}")
        if str(row.get("bspline_mode")) != "disp":
            errors.append(f"{run_name}: bspline_mode {row.get('bspline_mode')} != disp")
        if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
            errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
        if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
            errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
        if str(row.get("foreground_rule")) != "fixed raw image > 0":
            errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
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
    return errors


def best_alpha_lines(pop_rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method_key, spec in base.METHOD_SPECS.items():
        group = [row for row in pop_rows if str(row.get("method_key")) == method_key]
        if not group:
            continue
        best_dice = max(group, key=lambda row: base.as_float(row.get("dice_mean_mean")))
        best_hd95 = min(group, key=lambda row: base.as_float(row.get("hd95_mean_mean")))
        best_asd = min(group, key=lambda row: base.as_float(row.get("asd_symmetric_mean_mean")))
        lines.append(
            f"- {spec['label']}: best mean Dice alpha `{base.as_float(best_dice.get('alpha')):g}`, "
            f"best mean HD95 alpha `{base.as_float(best_hd95.get('alpha')):g}`, "
            f"best mean symmetric ASD alpha `{base.as_float(best_asd.get('alpha')):g}`."
        )
    return lines


def format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{key}:{counter[key]}" for key in sorted(counter, key=lambda item: base.as_float(item)))


def write_report(
    args: argparse.Namespace,
    original_rows: Sequence[Dict[str, object]],
    cps2_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    *,
    skip_history: bool,
) -> None:
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    counts_by_alpha = Counter(base.alpha_id(row.get("alpha")) for row in cps2_rows)
    counts_by_rank = Counter(f"{int(base.as_float(row.get('case_rank'))):03d}" for row in cps2_rows)
    lines = [
        "# OASIS Seed-3 First10 DirectCP cps2 Full Overlay",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Original root: `{args.original_root}`",
        f"- DirectCP cps2 preliminary root: `{args.prelim_root}`",
        f"- DirectCP cps2 completion root: `{args.completion_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Original completed summaries: `{len(original_rows)}` / `240`",
        f"- DirectCP cps2 summaries: `{len(cps2_rows)}` / `{len(case_ranks) * len(ADDON_ALPHAS)}`",
        f"- Case ranks: `{','.join(str(rank) for rank in case_ranks)}`",
        f"- DirectCP cps2 alphas: `{','.join(f'{alpha:g}' for alpha in ADDON_ALPHAS)}`",
        f"- DirectCP cps2 counts by alpha: `{format_counter(counts_by_alpha)}`",
        f"- DirectCP cps2 counts by rank: `{', '.join(f'{rank}:{counts_by_rank[rank]}' for rank in sorted(counts_by_rank))}`",
        "- Original first10 report, figures, and preliminary cps2 overlay are not overwritten.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: original first10 and all six DirectCP cps2 alpha rows are present with expected metadata and artifacts.")
    lines.extend(["", "## Best Alpha By Population Mean", ""])
    lines.extend(best_alpha_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `summary_runs_{SUFFIX}.csv/json`: original plus full DirectCP cps2 run summaries.",
            f"- `population_summary_{SUFFIX}.csv/json`: method/alpha population means and spreads.",
            f"- `population_label_metrics_{SUFFIX}.csv`: per-label population aggregates.",
            f"- `case_metrics_all_{SUFFIX}.csv`: per-run Dice/HD95/ASD with per-label columns.",
            f"- `regularity_metrics_all_{SUFFIX}.csv`: per-run full-field and foreground regularity metrics.",
            f"- `be_metrics_all_{SUFFIX}.csv` and `diffusion_metrics_all_{SUFFIX}.csv`: long-form regularity audit rows.",
            f"- `population_arc_primary_{SUFFIX}.pdf`: focused overlay ARC-style Dice-vs-regularity curves.",
            f"- `population_pareto_contact_sheet_{SUFFIX}.pdf`: all overlay accuracy-regularity population curves.",
            f"- `loss_curves_population_{SUFFIX}.pdf`: overlay loss curves." if not skip_history else "- Loss history aggregation skipped by request.",
            "",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "direct_cp_cps2_full_overlay_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    case_ranks = base.parse_case_ranks(str(args.case_ranks))

    original_rows = base.filter_summaries_by_case_rank(base.collect_summaries(Path(args.original_root)), case_ranks)
    original_args = argparse.Namespace(
        oasis_root=args.oasis_root,
        pair_file=args.pair_file,
        case_ranks=args.case_ranks,
        expected_epochs=args.expected_epochs,
        expected_eval_interval=args.expected_eval_interval,
        expected_seed=args.expected_seed,
    )
    original_errors = base.check_expected(original_rows, original_args)

    cps2_rows = collect_cps2_rows(args, case_ranks)
    cps2_errors = check_cps2_expected(cps2_rows, args)
    errors = original_errors + cps2_errors

    configure_method_specs()
    combined_rows = list(original_rows) + list(cps2_rows)
    combined_rows.sort(
        key=lambda row: (
            list(base.METHOD_SPECS).index(str(row.get("method_key"))),
            int(base.as_float(row.get("case_rank"))),
            base.as_float(row.get("alpha")),
        )
    )

    args.review_root.mkdir(parents=True, exist_ok=True)
    if errors and not args.allow_incomplete:
        c.write_json(args.review_root / f"aggregation_errors_{SUFFIX}.json", errors)
        raise RuntimeError("DirectCP cps2 full overlay aggregation failed: " + "; ".join(errors[:20]))

    c.write_csv(args.review_root / f"summary_runs_{SUFFIX}.csv", combined_rows)
    c.write_json(args.review_root / f"summary_runs_{SUFFIX}.json", combined_rows)
    c.write_csv(args.review_root / f"case_metrics_all_{SUFFIX}.csv", base.collect_metric_file(combined_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / f"regularity_metrics_all_{SUFFIX}.csv", base.collect_metric_file(combined_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / f"be_metrics_all_{SUFFIX}.csv", base.collect_metric_file(combined_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / f"diffusion_metrics_all_{SUFFIX}.csv", base.collect_metric_file(combined_rows, "diffusion_metrics.csv"))

    pop_rows = base.population_summary(combined_rows)
    label_rows = base.population_label_metrics(combined_rows)
    c.write_csv(args.review_root / f"population_summary_{SUFFIX}.csv", pop_rows)
    c.write_json(args.review_root / f"population_summary_{SUFFIX}.json", pop_rows)
    c.write_csv(args.review_root / f"population_label_metrics_{SUFFIX}.csv", label_rows)
    base.plot_population_grid(
        pop_rows,
        args.review_root / f"population_pareto_contact_sheet_{SUFFIX}.pdf",
        "OASIS seed-3 first10: population accuracy vs foreground regularity with full DirectCP cps2 overlay",
    )
    base.plot_primary_arc(
        pop_rows,
        args.review_root / f"population_arc_primary_{SUFFIX}.pdf",
        "OASIS seed-3 first10: primary population ARC curves with full DirectCP cps2 overlay",
    )

    history_rows: List[Dict[str, object]] = []
    if not args.skip_history:
        history_rows = base.collect_history(combined_rows)
        loss_rows = base.aggregate_loss_history(history_rows)
        base.write_csv_stream(args.review_root / f"loss_history_all_{SUFFIX}.csv", history_rows)
        c.write_csv(args.review_root / f"loss_history_population_{SUFFIX}.csv", loss_rows)
        base.plot_loss_curves(loss_rows, args.review_root / f"loss_curves_population_{SUFFIX}.pdf")

    write_report(args, original_rows, cps2_rows, pop_rows, errors, skip_history=bool(args.skip_history))
    print(
        json.dumps(
            {
                "original_summary_rows": len(original_rows),
                "direct_cp_cps2_summary_rows": len(cps2_rows),
                "combined_summary_rows": len(combined_rows),
                "population_rows": len(pop_rows),
                "history_rows": len(history_rows),
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
