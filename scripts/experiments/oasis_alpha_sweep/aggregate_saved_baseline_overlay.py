#!/usr/bin/env python
"""Aggregate saved OASIS first10 baselines and overlay them on first10 curves."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


DEFAULT_BASELINE_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1")
DEFAULT_FIRST10_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep"
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_saved_baselines"
VARIANT = "oasis_seed3_first10_saved_baselines_v1"

BASELINE_SPECS = {
    "bs1_vxm_v0_2": {"label": "BS1 VoxelMorph", "color": "#8c564b"},
    "bs2_tm_v0": {"label": "BS2 TransMorph", "color": "#e377c2"},
    "bs3_vfa_v0": {"label": "BS3 VFA", "color": "#7f7f7f"},
    "bs4_sitreg_v0": {"label": "BS4 SITReg", "color": "#bcbd22"},
    "greedy": {"label": "Greedy", "color": "#17becf"},
}

FD_PRIMARY_PANELS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "jdet_le0_pct_fg", True),
    ("dice_mean", "log_jdet_std_fg", True),
    ("dice_mean", "be_fd_forward_fixed_fg_voxel", True),
    ("dice_mean", "diffusion_fd_forward_fixed_fg_voxel", True),
]

FD_REGULARITY_METRICS: List[Tuple[str, str]] = [
    ("be_fd_forward_fixed_fg_voxel", "FG BE forward FD"),
    ("diffusion_fd_forward_fixed_fg_voxel", "FG diffusion forward FD"),
    ("diffusion_l2_fg", "FG diffusion L2"),
    ("jdet_le0_pct_fg", "FG Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "FG std(log Jdet)"),
    ("ndv_frac_fg", "FG NDV fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--first10-review-root", type=Path, default=DEFAULT_FIRST10_REVIEW_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_stream(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def configure_method_specs() -> None:
    specs = dict(base.METHOD_SPECS)
    for method_key, spec in BASELINE_SPECS.items():
        specs[method_key] = {
            "label": spec["label"],
            "method": method_key,
            "lr": float("nan"),
            "alphas": [float("nan")],
            "color": spec["color"],
        }
    base.METHOD_SPECS = specs


def expected_baseline_keys(case_ranks: Sequence[int]) -> set[Tuple[int, str]]:
    return {(int(rank), method_key) for rank in case_ranks for method_key in BASELINE_SPECS}


def check_baseline_rows(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    cases = {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in case_ranks
    }
    expected = expected_baseline_keys(case_ranks)
    observed: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        observed[(int(base.as_float(row.get("case_rank"))), str(row.get("method_key")))].append(row)

    for rank, method_key in sorted(expected - set(observed)):
        errors.append(f"Missing saved baseline row: case_rank={rank}, method={method_key}")
    for rank, method_key in sorted(set(observed) - expected):
        errors.append(f"Unexpected saved baseline row: case_rank={rank}, method={method_key}")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate saved baseline row {key}: {len(group)} summaries")

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
        if str(row.get("variant")) != VARIANT:
            errors.append(f"{run_name}: variant {row.get('variant')} != {VARIANT}")
        if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
            errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
        if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
            errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
        if str(row.get("foreground_rule")) != "fixed raw image > 0":
            errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
        if str(row.get("source_displacement_units")) != "voxel":
            errors.append(f"{run_name}: source_displacement_units {row.get('source_displacement_units')} != voxel")
        if not Path(str(row.get("source_disp_path"))).exists():
            errors.append(f"{run_name}: source displacement is missing: {row.get('source_disp_path')}")
        for filename in ["DONE", "summary.json", "case_metrics.csv", "regularity_metrics.csv", "be_metrics.csv", "diffusion_metrics.csv", "mask_audit.json"]:
            run_dir = Path(str(row.get("run_dir_actual", row.get("run_dir"))))
            if not (run_dir / filename).exists():
                errors.append(f"{run_name}: missing {run_dir / filename}")
        for metric in [
            "dense_image_loss",
            "dice_mean",
            "hd95_mean",
            "asd_symmetric_mean",
            "be_fd_forward_fixed_fg_voxel",
            "diffusion_fd_forward_fixed_fg_voxel",
            "jdet_le0_pct_fg",
            "log_jdet_std_fg",
            "ndv_frac_fg",
        ]:
            if not base.is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite required metric {metric}")
    return errors


def method_sort_key(row: Mapping[str, object]) -> Tuple[int, int, float]:
    method_key = str(row.get("method_key"))
    methods = list(base.METHOD_SPECS)
    method_index = methods.index(method_key) if method_key in methods else len(methods)
    return method_index, int(base.as_float(row.get("case_rank"))), base.as_float(row.get("alpha"))


def stats(values: Iterable[object]) -> Dict[str, float | int]:
    return base.stats(values)


def subset_label(case_ranks: Sequence[int]) -> str:
    if list(case_ranks) == list(range(10)):
        return "first10"
    if list(case_ranks) == list(range(20)):
        return "first20"
    return "ranks " + ",".join(str(rank) for rank in case_ranks)


def baseline_reference_lines(pop_rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method_key in BASELINE_SPECS:
        rows = [row for row in pop_rows if str(row.get("method_key")) == method_key]
        if not rows:
            continue
        row = rows[0]
        lines.append(
            f"- {BASELINE_SPECS[method_key]['label']}: mean Dice `{base.as_float(row.get('dice_mean_mean')):.4f}`, "
            f"mean HD95 `{base.as_float(row.get('hd95_mean_mean')):.4f}`, "
            f"mean symmetric ASD `{base.as_float(row.get('asd_symmetric_mean_mean')):.4f}`."
        )
    return lines


def plot_with_saved_baselines(
    pop_rows: Sequence[Dict[str, object]],
    path: Path,
    title: str,
    panels: Sequence[Tuple[str, str, bool]],
    *,
    grid: bool,
) -> None:
    if grid:
        panel_specs = [
            (acc_key, acc_label, higher, reg_key, reg_label)
            for acc_key, acc_label, higher in base.ACCURACY_METRICS
            for reg_key, reg_label in FD_REGULARITY_METRICS
        ]
        n_rows = len(base.ACCURACY_METRICS)
        n_cols = len(FD_REGULARITY_METRICS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.3 * n_rows), squeeze=False)
        flat_axes = axes.ravel()
    else:
        label_by_acc = {key: label for key, label, _higher in base.ACCURACY_METRICS}
        label_by_reg = dict(FD_REGULARITY_METRICS)
        panel_specs = [(acc, label_by_acc[acc], higher, reg, label_by_reg[reg]) for acc, reg, higher in panels]
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
        flat_axes = axes.ravel()

    baseline_keys = set(BASELINE_SPECS)
    for ax, (acc_key, acc_label, higher, reg_key, reg_label) in zip(flat_axes, panel_specs):
        acc_mean = f"{acc_key}_mean"
        reg_mean = f"{reg_key}_mean"
        for method_key, spec in base.METHOD_SPECS.items():
            group = [
                row
                for row in pop_rows
                if str(row.get("method_key")) == method_key
                and base.is_finite(row.get(acc_mean))
                and base.is_finite(row.get(reg_mean))
            ]
            group.sort(key=lambda row: base.as_float(row.get("alpha")))
            if not group:
                continue
            xs = [base.as_float(row.get(reg_mean)) for row in group]
            ys = [base.as_float(row.get(acc_mean)) for row in group]
            if method_key in baseline_keys:
                ax.scatter(xs, ys, marker="D", s=42, color=str(spec["color"]), label=str(spec["label"]), zorder=4)
                for x, y in zip(xs, ys):
                    ax.annotate(str(spec["label"]), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
            else:
                ax.errorbar(
                    xs,
                    ys,
                    xerr=[base.err_value(row, reg_key) for row in group],
                    yerr=[base.err_value(row, acc_key) for row in group],
                    marker="o",
                    linewidth=1.2,
                    markersize=4.5,
                    capsize=2,
                    color=str(spec["color"]),
                    label=str(spec["label"]),
                )
                for row, x, y in zip(group, xs, ys):
                    ax.annotate(f"{base.as_float(row.get('alpha')):g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
        front = base.pareto_front(list(pop_rows), acc_mean, reg_mean, higher)
        if len(front) >= 2:
            front = sorted(front, key=lambda row: base.as_float(row.get(reg_mean)))
            ax.plot(
                [base.as_float(row.get(reg_mean)) for row in front],
                [base.as_float(row.get(acc_mean)) for row in front],
                color="black",
                linewidth=1.0,
                alpha=0.5,
                linestyle="--",
                label="population Pareto",
            )
        ax.set_xlabel(reg_label)
        ax.set_ylabel(acc_label)
        ax.grid(True, alpha=0.25)
        if "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key:
            ax.set_xscale("symlog", linthresh=1e-8)
    flat_axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if not grid else 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    baseline_rows: Sequence[Dict[str, object]],
    original_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    subset = subset_label(case_ranks)
    lines = [
        f"# OASIS Seed-3 {subset} Saved Baseline Evaluation",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Baseline output root: `{args.baseline_root}`",
        f"- Reference review root: `{args.first10_review_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Saved baseline summaries: `{len(baseline_rows)}` / `{len(case_ranks) * len(BASELINE_SPECS)}`",
        f"- Reference summaries loaded from review CSV: `{len(original_rows)}`",
        f"- Case ranks: `{','.join(str(rank) for rank in case_ranks)}`",
        "- Baselines are saved dense displacement fields; training-time fields are not applicable.",
        "- Baseline regularity uses dense finite-difference metrics. Native analytic B-spline/IDIR columns are intentionally unavailable for these rows.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all requested saved baseline rows are present with expected cases, metadata, and artifacts.")
    lines.extend(["", "## Saved Baseline Population Means", ""])
    lines.extend(baseline_reference_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs_saved_baselines.csv/json`: saved baseline run summaries only.",
            "- `summary_runs_with_saved_baselines.csv/json`: reference summaries, if provided, plus saved baseline rows.",
            "- `population_summary_with_saved_baselines.csv/json`: method population aggregates.",
            "- `population_label_metrics_with_saved_baselines.csv`: per-label population aggregates.",
            "- `population_arc_fd_with_saved_baselines.pdf`: primary Dice-vs-FD/Jacobian overlay.",
            "- `population_pareto_contact_sheet_with_saved_baselines.pdf`: all accuracy-vs-FD/Jacobian overlay panels.",
            "",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_method_specs()
    case_ranks = base.parse_case_ranks(str(args.case_ranks))
    subset = subset_label(case_ranks)

    baseline_rows_all = base.collect_summaries(Path(args.baseline_root))
    baseline_rows = [
        row
        for row in base.filter_summaries_by_case_rank(baseline_rows_all, case_ranks)
        if str(row.get("method_key")) in BASELINE_SPECS
    ]
    baseline_errors = check_baseline_rows(baseline_rows, args)

    original_rows = read_csv(Path(args.first10_review_root) / "summary_runs.csv")
    original_rows = base.filter_summaries_by_case_rank(original_rows, case_ranks)
    combined_rows = list(original_rows) + list(baseline_rows)
    combined_rows.sort(key=method_sort_key)

    errors = list(baseline_errors)
    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors_saved_baselines.json", errors)
        raise RuntimeError("Saved baseline aggregation failed: " + "; ".join(errors[:20]))

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs_saved_baselines.csv", baseline_rows)
    c.write_json(args.review_root / "summary_runs_saved_baselines.json", baseline_rows)
    c.write_csv(args.review_root / "case_metrics_saved_baselines.csv", base.collect_metric_file(baseline_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_saved_baselines.csv", base.collect_metric_file(baseline_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_saved_baselines.csv", base.collect_metric_file(baseline_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_saved_baselines.csv", base.collect_metric_file(baseline_rows, "diffusion_metrics.csv"))

    c.write_csv(args.review_root / "summary_runs_with_saved_baselines.csv", combined_rows)
    c.write_json(args.review_root / "summary_runs_with_saved_baselines.json", combined_rows)
    for source_name, baseline_name, out_name in [
        ("case_metrics_all.csv", "case_metrics.csv", "case_metrics_all_with_saved_baselines.csv"),
        ("regularity_metrics_all.csv", "regularity_metrics.csv", "regularity_metrics_all_with_saved_baselines.csv"),
        ("be_metrics_all.csv", "be_metrics.csv", "be_metrics_all_with_saved_baselines.csv"),
        ("diffusion_metrics_all.csv", "diffusion_metrics.csv", "diffusion_metrics_all_with_saved_baselines.csv"),
    ]:
        original_metric_rows = read_csv(Path(args.first10_review_root) / source_name)
        baseline_metric_rows = base.collect_metric_file(baseline_rows, baseline_name)
        write_csv_stream(args.review_root / out_name, [*original_metric_rows, *baseline_metric_rows])

    baseline_pop_rows = base.population_summary(baseline_rows)
    baseline_label_rows = base.population_label_metrics(baseline_rows)
    c.write_csv(args.review_root / "population_summary_saved_baselines.csv", baseline_pop_rows)
    c.write_json(args.review_root / "population_summary_saved_baselines.json", baseline_pop_rows)
    c.write_csv(args.review_root / "population_label_metrics_saved_baselines.csv", baseline_label_rows)

    pop_rows = base.population_summary(combined_rows)
    label_rows = base.population_label_metrics(combined_rows)
    c.write_csv(args.review_root / "population_summary_with_saved_baselines.csv", pop_rows)
    c.write_json(args.review_root / "population_summary_with_saved_baselines.json", pop_rows)
    c.write_csv(args.review_root / "population_label_metrics_with_saved_baselines.csv", label_rows)
    plot_with_saved_baselines(
        pop_rows,
        args.review_root / "population_arc_fd_with_saved_baselines.pdf",
        f"OASIS seed-3 {subset}: primary FD/Jacobian ARC curves with saved baselines",
        FD_PRIMARY_PANELS,
        grid=False,
    )
    plot_with_saved_baselines(
        pop_rows,
        args.review_root / "population_pareto_contact_sheet_with_saved_baselines.pdf",
        f"OASIS seed-3 {subset}: population accuracy vs FD/Jacobian regularity with saved baselines",
        FD_PRIMARY_PANELS,
        grid=True,
    )
    write_report(args, baseline_rows, original_rows, pop_rows, errors)
    print(
        json.dumps(
            {
                "baseline_summary_rows": len(baseline_rows),
                "original_summary_rows": len(original_rows),
                "combined_summary_rows": len(combined_rows),
                "population_rows": len(pop_rows),
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
