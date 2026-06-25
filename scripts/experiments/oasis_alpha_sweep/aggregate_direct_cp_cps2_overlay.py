#!/usr/bin/env python
"""Aggregate original first10 results plus the DirectCP cps2 preliminary add-on."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import aggregate_first10_population as base  # noqa: E402


DEFAULT_ORIGINAL_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
DEFAULT_ADDON_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay"
ADDON_METHOD = "direct_cp_cps2_be_bspline_analytic"
ADDON_LABEL = "DirectCP cps2 lr1e-3"
ADDON_ALPHAS = [100.0, 1000.0, 10000.0]
ADDON_LR = 1e-3
ADDON_VARIANT = "oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--addon-root", type=Path, default=DEFAULT_ADDON_ROOT)
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


def expected_addon_combos(case_ranks: Sequence[int]) -> Dict[Tuple[int, str, str, str], Dict[str, object]]:
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


def isclose_alpha(value: object, target: float) -> bool:
    return math.isclose(base.as_float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)


def check_addon_expected(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    cases = {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in case_ranks
    }
    expected = expected_addon_combos(case_ranks)
    observed: Dict[Tuple[int, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        observed.setdefault(base.combo_key(row), []).append(row)

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    for key in missing:
        errors.append(f"Missing add-on run combo: case_rank={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    for key in extra:
        errors.append(f"Unexpected add-on run combo: case_rank={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate add-on run combo {key}: {len(group)} summaries")

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
        if str(row.get("variant")) != ADDON_VARIANT:
            errors.append(f"{run_name}: variant {row.get('variant')} != {ADDON_VARIANT}")
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


def write_csv_stream(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def write_report(
    args: argparse.Namespace,
    original_rows: Sequence[Dict[str, object]],
    addon_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    *,
    skip_history: bool,
) -> None:
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    lines = [
        "# OASIS Seed-3 First10 DirectCP cps2 Preliminary Overlay",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Original root: `{args.original_root}`",
        f"- Add-on root: `{args.addon_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Original completed summaries: `{len(original_rows)}` / `240`",
        f"- DirectCP cps2 preliminary summaries: `{len(addon_rows)}` / `{len(case_ranks) * len(ADDON_ALPHAS)}`",
        f"- Case ranks: `{','.join(str(rank) for rank in case_ranks)}`",
        f"- DirectCP cps2 alphas: `{','.join(f'{alpha:g}' for alpha in ADDON_ALPHAS)}`",
        "- This is a preliminary 3-alpha DirectCP cps2 overlay, not a complete six-alpha DirectCP cps2 sweep.",
        "- Original first10 report and figures are not overwritten.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: original first10 and DirectCP cps2 add-on rows are present with expected metadata and artifacts.")
    lines.extend(["", "## Best Alpha By Population Mean", ""])
    lines.extend(best_alpha_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs_with_direct_cp_cps2.csv/json`: original plus add-on run summaries.",
            "- `population_summary_with_direct_cp_cps2.csv/json`: method/alpha population means and spreads.",
            "- `population_label_metrics_with_direct_cp_cps2.csv`: per-label population aggregates.",
            "- `case_metrics_all_with_direct_cp_cps2.csv`: per-run Dice/HD95/ASD with per-label columns.",
            "- `regularity_metrics_all_with_direct_cp_cps2.csv`: per-run full-field and foreground regularity metrics.",
            "- `be_metrics_all_with_direct_cp_cps2.csv` and `diffusion_metrics_all_with_direct_cp_cps2.csv`: long-form regularity audit rows.",
            "- `population_arc_primary_with_direct_cp_cps2.pdf`: focused overlay ARC-style Dice-vs-regularity curves.",
            "- `population_pareto_contact_sheet_with_direct_cp_cps2.pdf`: all overlay accuracy-regularity population curves.",
            "- `loss_curves_population_with_direct_cp_cps2.pdf`: overlay loss curves." if not skip_history else "- Loss history aggregation skipped by request.",
            "",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "direct_cp_cps2_overlay_report.md").write_text("\n".join(lines) + "\n")


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

    addon_rows_all = base.collect_summaries(Path(args.addon_root))
    addon_rows = [
        row
        for row in base.filter_summaries_by_case_rank(addon_rows_all, case_ranks)
        if str(row.get("method_key")) == ADDON_METHOD
    ]
    addon_errors = check_addon_expected(addon_rows, args)
    errors = original_errors + addon_errors

    configure_method_specs()
    combined_rows = list(original_rows) + list(addon_rows)
    combined_rows.sort(
        key=lambda row: (
            list(base.METHOD_SPECS).index(str(row.get("method_key"))),
            int(base.as_float(row.get("case_rank"))),
            base.as_float(row.get("alpha")),
        )
    )

    args.review_root.mkdir(parents=True, exist_ok=True)
    if errors and not args.allow_incomplete:
        c.write_json(args.review_root / "aggregation_errors_with_direct_cp_cps2.json", errors)
        raise RuntimeError("DirectCP cps2 overlay aggregation failed: " + "; ".join(errors[:20]))

    c.write_csv(args.review_root / "summary_runs_with_direct_cp_cps2.csv", combined_rows)
    c.write_json(args.review_root / "summary_runs_with_direct_cp_cps2.json", combined_rows)
    c.write_csv(args.review_root / "case_metrics_all_with_direct_cp_cps2.csv", base.collect_metric_file(combined_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_all_with_direct_cp_cps2.csv", base.collect_metric_file(combined_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_all_with_direct_cp_cps2.csv", base.collect_metric_file(combined_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_all_with_direct_cp_cps2.csv", base.collect_metric_file(combined_rows, "diffusion_metrics.csv"))

    pop_rows = base.population_summary(combined_rows)
    label_rows = base.population_label_metrics(combined_rows)
    c.write_csv(args.review_root / "population_summary_with_direct_cp_cps2.csv", pop_rows)
    c.write_json(args.review_root / "population_summary_with_direct_cp_cps2.json", pop_rows)
    c.write_csv(args.review_root / "population_label_metrics_with_direct_cp_cps2.csv", label_rows)
    base.plot_population_grid(
        pop_rows,
        args.review_root / "population_pareto_contact_sheet_with_direct_cp_cps2.pdf",
        "OASIS seed-3 first10: population accuracy vs foreground regularity with DirectCP cps2 preliminary overlay",
    )
    base.plot_primary_arc(
        pop_rows,
        args.review_root / "population_arc_primary_with_direct_cp_cps2.pdf",
        "OASIS seed-3 first10: primary population ARC curves with DirectCP cps2 preliminary overlay",
    )

    history_rows: List[Dict[str, object]] = []
    if not args.skip_history:
        history_rows = base.collect_history(combined_rows)
        loss_rows = base.aggregate_loss_history(history_rows)
        write_csv_stream(args.review_root / "loss_history_all_with_direct_cp_cps2.csv", history_rows)
        c.write_csv(args.review_root / "loss_history_population_with_direct_cp_cps2.csv", loss_rows)
        base.plot_loss_curves(loss_rows, args.review_root / "loss_curves_population_with_direct_cp_cps2.pdf")

    write_report(args, original_rows, addon_rows, pop_rows, errors, skip_history=bool(args.skip_history))
    print(
        json.dumps(
            {
                "original_summary_rows": len(original_rows),
                "addon_summary_rows": len(addon_rows),
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
