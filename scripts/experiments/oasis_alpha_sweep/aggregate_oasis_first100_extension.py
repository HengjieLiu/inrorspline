#!/usr/bin/env python
"""Aggregate OASIS seed-3 first100 extension rows and make publication plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines as p  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as plus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=f100.FIRST100_ARTIFACT_ROOT)
    parser.add_argument("--first10-main-root", type=Path, default=f100.FIRST10_MAIN_ROOT)
    parser.add_argument("--first10-cps2-prelim-root", type=Path, default=f100.FIRST10_CPS2_PRELIM_ROOT)
    parser.add_argument("--first10-cps2-completion-root", type=Path, default=f100.FIRST10_CPS2_COMPLETION_ROOT)
    parser.add_argument("--next10-root", type=Path, default=f100.NEXT10_ROOT)
    parser.add_argument("--single-stage-extension-root", type=Path, default=f100.SINGLE_STAGE_EXTENSION_ROOT)
    parser.add_argument("--baseline-first20-root", type=Path, default=f100.FIRST20_BASELINE_ROOT)
    parser.add_argument("--baseline-extension-root", type=Path, default=f100.BASELINE_EXTENSION_ROOT)
    parser.add_argument("--directcp842-first20-root", type=Path, default=f100.FIRST20_DCP842_ROOT)
    parser.add_argument("--directcp842-extension-root", type=Path, default=f100.DCP842_EXTENSION_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def collect_existing_first20_sweeps(args: argparse.Namespace) -> List[Dict[str, object]]:
    first10_ranks = list(range(10))
    next10_ranks = list(range(10, 20))
    rows: List[Dict[str, object]] = []
    rows.extend(p.filter_ranks(p.collect_summary_rows(Path(args.first10_main_root)), first10_ranks))
    rows.extend(p.filter_ranks(p.collect_summary_rows(Path(args.first10_cps2_prelim_root)), first10_ranks))
    rows.extend(p.filter_ranks(p.collect_summary_rows(Path(args.first10_cps2_completion_root)), first10_ranks))
    rows.extend(p.filter_ranks(p.collect_summary_rows(Path(args.next10_root)), next10_ranks))
    return rows


def collect_first100_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    first20_sweep_rows = collect_existing_first20_sweeps(args)
    extension_sweep_rows = p.filter_ranks(
        p.collect_summary_rows(Path(args.single_stage_extension_root)),
        f100.RANKS_EXTENSION,
    )
    first20_dcp842_rows = [
        row
        for row in p.filter_ranks(p.collect_summary_rows(Path(args.directcp842_first20_root)), f100.RANKS_FIRST20)
        if str(row.get("method_key")) == f100.DCP842_METHOD
    ]
    extension_dcp842_rows = [
        row
        for row in p.filter_ranks(p.collect_summary_rows(Path(args.directcp842_extension_root)), f100.RANKS_EXTENSION)
        if str(row.get("method_key")) == f100.DCP842_METHOD
    ]
    first20_baseline_rows = p.filter_ranks(
        p.collect_summary_rows(Path(args.baseline_first20_root)),
        f100.RANKS_FIRST20,
    )
    extension_baseline_rows = p.filter_ranks(
        p.collect_summary_rows(Path(args.baseline_extension_root)),
        f100.RANKS_EXTENSION,
    )
    return {
        "sweep_rows": first20_sweep_rows + extension_sweep_rows + first20_dcp842_rows + extension_dcp842_rows,
        "baseline_rows": first20_baseline_rows + extension_baseline_rows,
        "extension_sweep_rows": extension_sweep_rows,
        "extension_dcp842_rows": extension_dcp842_rows,
        "extension_baseline_rows": extension_baseline_rows,
    }


def write_root_report(review_root: Path, manifest: Dict[str, object]) -> None:
    lines = [
        "# OASIS Seed-3 First100 Six-Sweep Pareto Plot Package",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Review root: `{review_root}`",
        "- Cohort: seed-3 shuffled ranks `0-99`; ranks `0-19` reused and ranks `20-99` newly evaluated.",
        "- Final figures include only complete first100 population rows.",
        "",
        "## Counts",
        "",
        f"- Sweep rows: `{manifest['sweep_rows']}`",
        f"- Baseline rows: `{manifest['baseline_rows']}`",
        f"- Population rows: `{manifest['population_rows']}`",
        f"- Figure files: `{manifest['output_files']}`",
        "",
        "## Files",
        "",
        "- `oasis100/population_summary_oasis100_with_saved_baselines.csv`",
        "- `oasis100/big_plots/spotlight_line_segments_errorbars__dice_mean_vs_log_jdet_std_fg.png`",
        "- `oasis100/big_plots/spotlight_line_segments_errorbars__hd95_mean_vs_log_jdet_std_fg.png`",
        "- `oasis100/big_plots/contact_sheet_line_segments_errorbars.png`",
        "",
    ]
    (review_root / "review_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    plus.configure_plot_module(float(f100.DCP842_LR))
    rows = collect_first100_rows(args)
    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)
    result = p.process_cohort(
        args,
        cohort="oasis100",
        ranks=f100.RANKS_FIRST100,
        sweep_rows=rows["sweep_rows"],
        baseline_rows=rows["baseline_rows"],
    )
    manifest: Dict[str, object] = {
        "generated_at": c.now_text(),
        "review_root": str(review_root),
        "result": result,
        "sweep_rows": len(rows["sweep_rows"]),
        "baseline_rows": len(rows["baseline_rows"]),
        "population_rows": int(result["population_rows"]),
        "output_files": int(result["output_files"]),
        "input_roots": {
            "first10_main_root": str(args.first10_main_root),
            "first10_cps2_prelim_root": str(args.first10_cps2_prelim_root),
            "first10_cps2_completion_root": str(args.first10_cps2_completion_root),
            "next10_root": str(args.next10_root),
            "single_stage_extension_root": str(args.single_stage_extension_root),
            "baseline_first20_root": str(args.baseline_first20_root),
            "baseline_extension_root": str(args.baseline_extension_root),
            "directcp842_first20_root": str(args.directcp842_first20_root),
            "directcp842_extension_root": str(args.directcp842_extension_root),
        },
        "extension_counts": {
            "single_stage_summaries": len(rows["extension_sweep_rows"]),
            "directcp842_summaries": len(rows["extension_dcp842_rows"]),
            "baseline_summaries": len(rows["extension_baseline_rows"]),
        },
    }
    c.write_json(review_root / "plot_manifest.json", manifest)
    write_root_report(review_root, manifest)
    print(json.dumps(manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

