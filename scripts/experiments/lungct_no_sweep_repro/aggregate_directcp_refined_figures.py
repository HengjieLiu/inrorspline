#!/usr/bin/env python
"""Build non-overwriting refined DirectCP comparison figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import aggregate_reg_sweep_idir_directcp32 as sweep
import aggregate_report as base
import common as c
from launch_reg_sweep_idir_directcp32 import (
    DEFAULT_SWEEP_ROOT,
    METHOD_DIRECTCP32,
    METHOD_IDIR,
)


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"
DEFAULT_IDIR_FACTORS = "0.01,0.1,1,10,100"
DEFAULT_DIRECTCP_FACTORS = "0.01,0.1,0.2,0.5,1,2,5,10,100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--idir-factors", type=str, default=DEFAULT_IDIR_FACTORS)
    parser.add_argument("--directcp-factors", type=str, default=DEFAULT_DIRECTCP_FACTORS)
    parser.add_argument("--plot-dir-name", type=str, default="plots_refined_directcp")
    parser.add_argument("--artifact-prefix", type=str, default="directcp_refined")
    parser.add_argument("--report-name", type=str, default="directcp_refined_review.md")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def parse_factors(value: str) -> List[float]:
    return [float(v) for v in value.replace(" ", "").split(",") if v]


def factor_in(factor: float, allowed: Sequence[float]) -> bool:
    return any(abs(float(factor) - float(item)) < 1e-10 for item in allowed)


def filter_rows(rows: Sequence[Dict[str, object]], idir_factors: Sequence[float], directcp_factors: Sequence[float]) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for row in rows:
        method = str(row["method"])
        factor = float(row["reg_factor"])
        if method == METHOD_IDIR and factor_in(factor, idir_factors):
            selected.append(dict(row))
        elif method == METHOD_DIRECTCP32 and factor_in(factor, directcp_factors):
            selected.append(dict(row))
    return selected


def factor_count_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = sorted(
        {(str(row["method"]), float(row["reg_factor"])) for row in rows},
        key=lambda item: (item[0], item[1]),
    )
    out: List[Dict[str, object]] = []
    for method, factor in groups:
        count = sum(1 for row in rows if str(row["method"]) == method and abs(float(row["reg_factor"]) - factor) < 1e-10)
        out.append({"method": method, "method_label": sweep.method_label(method), "reg_factor": factor, "cases": count})
    return out


def assert_complete(count_rows: Sequence[Dict[str, object]], idir_factors: Sequence[float], directcp_factors: Sequence[float]) -> None:
    expected = {
        (METHOD_IDIR, float(factor)): 10 for factor in idir_factors
    }
    expected.update({(METHOD_DIRECTCP32, float(factor)): 10 for factor in directcp_factors})
    observed = {(str(row["method"]), float(row["reg_factor"])): int(row["cases"]) for row in count_rows}
    missing = []
    for key, count in sorted(expected.items(), key=lambda item: (item[0][0], item[0][1])):
        if observed.get(key) != count:
            missing.append({"method": key[0], "factor": key[1], "expected": count, "observed": observed.get(key, 0)})
    if missing:
        raise RuntimeError(f"Missing/incomplete refined factor rows: {missing}")


def markdown_counts(count_rows: Sequence[Dict[str, object]]) -> str:
    return base.markdown_table(
        ["Method", "Factor", "Cases"],
        [[row["method_label"], sweep.fmt(row["reg_factor"], 2), row["cases"]] for row in count_rows],
    )


def factor_text(factors: Sequence[float]) -> str:
    return ", ".join(f"`{factor:g}x`" for factor in factors)


def main() -> None:
    args = parse_args()
    idir_factors = parse_factors(args.idir_factors)
    directcp_factors = parse_factors(args.directcp_factors)
    methods = [METHOD_IDIR, METHOD_DIRECTCP32]
    rows = sweep.collect_rows(args.baseline_root, methods, "baseline_1e0")
    rows.extend(sweep.collect_rows(args.sweep_root, methods, "sweep"))
    rows = filter_rows(rows, idir_factors, directcp_factors)
    rows = sorted(rows, key=lambda row: (str(row["method"]), float(row["reg_factor"]), int(row["case_id"])))
    agg_rows = sweep.aggregate_rows(rows)
    count_rows = factor_count_rows(rows)
    if args.require_complete:
        assert_complete(count_rows, idir_factors, directcp_factors)

    args.review_root.mkdir(parents=True, exist_ok=True)
    plot_dir = args.review_root / args.plot_dir_name
    sweep.write_plots(agg_rows, plot_dir)
    c.write_csv(args.review_root / f"{args.artifact_prefix}_case_rows.csv", rows)
    c.write_json(args.review_root / f"{args.artifact_prefix}_case_rows.json", rows)
    c.write_csv(args.review_root / f"{args.artifact_prefix}_mean_rows.csv", agg_rows)
    c.write_json(args.review_root / f"{args.artifact_prefix}_mean_rows.json", agg_rows)
    c.write_csv(args.review_root / f"{args.artifact_prefix}_factor_counts.csv", count_rows)

    report = "\n".join(
        [
            "# LungCT DirectCP Refined Regularization Figures",
            "",
            "This supplemental report builds non-overwriting ellipse-style ARC/Pareto figures from selected IDIR and DirectCP regularization factors. Original figures under `plots/` are not overwritten; regenerated refined figures are under `" + args.plot_dir_name + "/`.",
            "",
            f"IDIR is shown at factors {factor_text(idir_factors)}. DirectCP is shown at factors {factor_text(directcp_factors)}.",
            "",
            "Combined plot ellipses are axis-aligned 1-SD ellipses: horizontal radius is one standard deviation of the x-axis regularity metric across cases and vertical radius is one standard deviation of original-snap TRE across cases.",
            "",
            "## Mean Rows",
            "",
            sweep.markdown_mean_table(agg_rows),
            "",
            "## Case Counts",
            "",
            markdown_counts(count_rows),
            "",
            "## Artifacts",
            "",
            f"- Refined plots: `{args.plot_dir_name}/`.",
            f"- Tables: `{args.artifact_prefix}_case_rows.csv/json`, `{args.artifact_prefix}_mean_rows.csv/json`, and `{args.artifact_prefix}_factor_counts.csv`.",
        ]
    )
    report_path = args.review_root / args.report_name
    report_path.write_text(report + "\n")
    print(
        json.dumps(
            {
                "case_rows": len(rows),
                "mean_rows": len(agg_rows),
                "factor_counts": len(count_rows),
                "plot_dir": str(plot_dir),
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
