#!/usr/bin/env python
"""Generate OASIS Pareto source plots with saved baselines plus DirectCP 8-4-2 lr=1e-3."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt

import common as c
import plot_first10_first20_alpha5_saved_baselines as p


DCP842_METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
DEFAULT_REVIEW_ROOT = (
    c.REPO_ROOT
    / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/"
    "pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3_paper_v1"
)
DEFAULT_DCP842_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1")
LR = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--first10-main-root", type=Path, default=p.DEFAULT_FIRST10_MAIN_ROOT)
    parser.add_argument("--first10-cps2-prelim-root", type=Path, default=p.DEFAULT_FIRST10_CPS2_PRELIM_ROOT)
    parser.add_argument("--first10-cps2-completion-root", type=Path, default=p.DEFAULT_FIRST10_CPS2_COMPLETION_ROOT)
    parser.add_argument("--next10-root", type=Path, default=p.DEFAULT_NEXT10_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=p.DEFAULT_BASELINE_ROOT)
    parser.add_argument("--directcp842-root", type=Path, default=DEFAULT_DCP842_ROOT)
    parser.add_argument("--directcp842-lr", type=float, default=LR)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def configure_plot_module(lr: float) -> None:
    if DCP842_METHOD not in p.SWEEP_METHODS:
        p.SWEEP_METHODS.append(DCP842_METHOD)
    p.METHOD_SPECS[DCP842_METHOD] = {
        "label": "DirectCP 8-4-2",
        "dot_label": "DCP842",
        "lr": float(lr),
        "alphas": p.SPLINE_ALPHAS,
        "color": "#332288",
    }
    p.validate_population_rows = validate_population_rows_plus  # type: ignore[assignment]
    p.plot_contact_sheet = plot_contact_sheet_plus  # type: ignore[assignment]
    p.write_cohort_report = write_cohort_report_plus  # type: ignore[assignment]


def collect_directcp842_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    return [
        row
        for row in p.collect_summary_rows(Path(args.directcp842_root))
        if str(row.get("method_key")) == DCP842_METHOD
    ]


def build_input_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    inputs = p.build_input_rows(args)
    directcp842_rows = collect_directcp842_rows(args)
    first10_ranks = list(range(10))
    first20_ranks = list(range(20))
    first10_directcp842 = p.filter_ranks(directcp842_rows, first10_ranks)
    first20_directcp842 = p.filter_ranks(directcp842_rows, first20_ranks)
    inputs["first10_sweeps"] = inputs["first10_sweeps"] + first10_directcp842
    inputs["first20_sweeps"] = inputs["first20_sweeps"] + first20_directcp842
    inputs["directcp842_rows"] = directcp842_rows
    return inputs


def validate_population_rows_plus(rows: Sequence[Dict[str, object]], cohort: str, n_cases: int) -> List[str]:
    errors: List[str] = []
    expected_counts = {method: 6 for method in p.SWEEP_METHODS}
    expected_counts.update({method: 1 for method in p.BASELINE_METHODS})
    observed = Counter(str(row.get("method_key")) for row in rows)
    expected_total = sum(expected_counts.values())
    if len(rows) != expected_total:
        errors.append(f"{cohort}: expected {expected_total} population rows, found {len(rows)}")
    for method, expected in expected_counts.items():
        if observed.get(method, 0) != expected:
            errors.append(f"{cohort}: method {method} expected {expected} rows, found {observed.get(method, 0)}")
    for row in rows:
        method = str(row.get("method_key"))
        if int(p.as_float(row.get("n_cases"))) != int(n_cases):
            errors.append(f"{cohort}: {method} n_cases {row.get('n_cases')} != {n_cases}")
    for acc_key, _acc_label, _higher in p.ACCURACY_METRICS:
        for reg_key, _reg_label in p.REGULARITY_METRICS:
            for key in [p.metric_mean(acc_key), p.metric_std(acc_key), p.metric_mean(reg_key), p.metric_std(reg_key)]:
                bad = [str(row.get("method_key")) for row in rows if not p.is_finite(row.get(key))]
                if bad:
                    errors.append(f"{cohort}: column {key} has non-finite values for {', '.join(bad[:8])}")
    return errors


def plot_contact_sheet_plus(
    rows: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    cohort: str,
    style: str,
    dpi: int,
) -> List[Path]:
    n_rows = len(p.ACCURACY_METRICS)
    n_cols = len(p.REGULARITY_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.55 * n_rows), squeeze=False)
    for row_idx, (acc_key, acc_label, _higher) in enumerate(p.ACCURACY_METRICS):
        for col_idx, (reg_key, reg_label) in enumerate(p.REGULARITY_METRICS):
            ax = axes[row_idx][col_idx]
            p.draw_panel(
                ax,
                rows,
                acc_key=acc_key,
                reg_key=reg_key,
                style=style,
                show_legend=(row_idx == 0 and col_idx == 0),
                annotate=True,
            )
            ax.set_title(f"{acc_label} vs {reg_label}", fontsize=8.5)
    fig.suptitle(
        f"OASIS seed-3 {cohort}: six alpha sweeps with DL/Greedy references ({style})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return p.save_figure(fig, out_dir / f"contact_sheet_{style}", dpi)


def write_cohort_report_plus(
    cohort_dir: Path,
    *,
    cohort: str,
    sweep_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    output_paths: Sequence[Path],
) -> None:
    lines = [
        f"# OASIS Seed-3 {cohort} Six-Sweep Pareto Plot Package",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Sweep rows: `{len(sweep_rows)}`",
        f"- Baseline rows: `{len(baseline_rows)}`",
        f"- Population rows: `{len(pop_rows)}`",
        "- Styles: line segments, line segments with x/y 1-SD bars, and axis-aligned 1-SD ellipses.",
        "- Added DirectCP `8-4-2` as a sixth full six-alpha curve at canonical `lr=1e-3`.",
        "- Foreground log-J metric: `log_jdet_std_fg`, the foreground standard deviation of log Jacobian determinant.",
        "",
        "## Best/Reference Points",
        "",
    ]
    lines.extend(p.best_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `population_summary_{cohort}_with_saved_baselines.csv/json`",
            f"- `population_label_metrics_{cohort}_with_saved_baselines.csv`",
            "- `small_plots/`: individual accuracy-regularity panels.",
            "- `big_plots/`: high-DPI contact sheets and spotlight plots.",
            f"- Generated figure files: `{len(output_paths)}`",
            "",
        ]
    )
    (cohort_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def write_root_report(review_root: Path, results: Sequence[Mapping[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# OASIS First10/First20 Six-Sweep Pareto Plot Package",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Review root: `{review_root}`",
        f"- First10 main root: `{args.first10_main_root}`",
        f"- First10 DirectCP cps2 prelim root: `{args.first10_cps2_prelim_root}`",
        f"- First10 DirectCP cps2 completion root: `{args.first10_cps2_completion_root}`",
        f"- Next10 five-model root: `{args.next10_root}`",
        f"- Saved-baseline root: `{args.baseline_root}`",
        f"- DirectCP `8-4-2` canonical lr=1e-3 root: `{args.directcp842_root}`",
        "- Existing five-sweep and lr=1e-2 plus-DirectCP842 figure directories were not overwritten.",
        "",
        "## Cohorts",
        "",
    ]
    for result in results:
        lines.append(
            f"- {result['cohort']}: sweep rows `{result['sweep_rows']}`, baseline rows `{result['baseline_rows']}`, "
            f"population rows `{result['population_rows']}`, figure files `{result['output_files']}`."
        )
    lines.extend(
        [
            "",
            "## Expected Shape",
            "",
            "- Population rows per cohort: `41` = six sweep methods x six alphas + five saved baselines.",
            "- Sweep rows: `360` for first10 and `720` for first20.",
            "",
            "## Files",
            "",
            "- `first10/` and `first20/`: cohort-specific summaries, manifests, reports, small plots, and big plots.",
            "",
        ]
    )
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def assert_expected_shape(results: Sequence[Mapping[str, object]]) -> None:
    expected = {
        "first10": {"sweep_rows": 360, "baseline_rows": 50, "population_rows": 41},
        "first20": {"sweep_rows": 720, "baseline_rows": 100, "population_rows": 41},
    }
    errors: List[str] = []
    for result in results:
        cohort = str(result.get("cohort"))
        for key, target in expected.get(cohort, {}).items():
            value = int(p.as_float(result.get(key)))
            if value != target:
                errors.append(f"{cohort}: {key} {value} != {target}")
    if errors:
        raise RuntimeError("Unexpected plus-DirectCP842 lr=1e-3 plot package shape: " + "; ".join(errors))


def main() -> None:
    args = parse_args()
    configure_plot_module(float(args.directcp842_lr))
    inputs = build_input_rows(args)
    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)

    first10_result = p.process_cohort(
        args,
        cohort="first10",
        ranks=list(range(10)),
        sweep_rows=inputs["first10_sweeps"],
        baseline_rows=inputs["first10_baselines"],
    )
    first20_result = p.process_cohort(
        args,
        cohort="first20",
        ranks=list(range(20)),
        sweep_rows=inputs["first20_sweeps"],
        baseline_rows=inputs["first20_baselines"],
    )
    results = [first10_result, first20_result]
    assert_expected_shape(results)
    root_manifest = {
        "generated_at": c.now_text(),
        "review_root": str(review_root),
        "results": results,
        "input_roots": {
            "first10_main_root": str(args.first10_main_root),
            "first10_cps2_prelim_root": str(args.first10_cps2_prelim_root),
            "first10_cps2_completion_root": str(args.first10_cps2_completion_root),
            "next10_root": str(args.next10_root),
            "baseline_root": str(args.baseline_root),
            "directcp842_root": str(args.directcp842_root),
        },
        "directcp842_lr": float(args.directcp842_lr),
        "directcp842_rows": len(inputs["directcp842_rows"]),
    }
    p.write_json(review_root / "plot_manifest.json", root_manifest)
    write_root_report(review_root, results, args)
    print(json.dumps(root_manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
