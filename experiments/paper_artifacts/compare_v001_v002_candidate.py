#!/usr/bin/env python
"""Compare paper artifact v001 against v002_candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from paper_artifact_common import repo_path, repo_relative


DEFAULT_OUT_DIR = repo_path("experiments/paper_artifacts/comparison/v001_vs_v002_candidate")
DEFAULT_TABLE_V1 = repo_path("experiments/paper_artifacts/best_result_table/v001")
DEFAULT_TABLE_V2 = repo_path("experiments/paper_artifacts/best_result_table/v002_candidate")
DEFAULT_PARETO_V1 = repo_path("experiments/paper_artifacts/pareto_figure/v001")
DEFAULT_PARETO_V2 = repo_path("experiments/paper_artifacts/pareto_figure/v002_candidate")
DCP842 = "direct_cp_8_4_2_be_bspline_analytic_multistage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--table-v1", type=Path, default=DEFAULT_TABLE_V1)
    parser.add_argument("--table-v2", type=Path, default=DEFAULT_TABLE_V2)
    parser.add_argument("--pareto-v1", type=Path, default=DEFAULT_PARETO_V1)
    parser.add_argument("--pareto-v2", type=Path, default=DEFAULT_PARETO_V2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return float("nan")
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def fmt_delta(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) < 1e-5 and value != 0:
        return f"{value:.3e}"
    return f"{value:.6g}"


def validation_counts(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    return dict(Counter(row.get("status", "") for row in rows))


def table_delta(args: argparse.Namespace) -> list[dict[str, Any]]:
    v1_rows = {row.get("Method", ""): row for row in read_csv(args.table_v1 / "oasis20_summary.csv")}
    v2_rows = {row.get("Method", ""): row for row in read_csv(args.table_v2 / "oasis20_summary.csv")}
    methods = sorted(set(v1_rows) | set(v2_rows))
    fields = ["Selected", "Cases", "Dice", "HD95", "ASD sym", "FG stdLogJ", "FG bending", "FG Jdet<0 %"]
    out: list[dict[str, Any]] = []
    for method in methods:
        row = {"Method": method}
        for field in fields:
            row[f"v001_{field}"] = v1_rows.get(method, {}).get(field, "")
            row[f"v002_candidate_{field}"] = v2_rows.get(method, {}).get(field, "")
        out.append(row)
    return out


def by_alpha(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("method_id") or row.get("method_key")) != DCP842:
            continue
        alpha = safe_float(row.get("alpha"))
        if math.isfinite(alpha):
            out[f"{alpha:.12g}"] = row
    return out


def alpha_curve_delta(args: argparse.Namespace) -> list[dict[str, Any]]:
    v1 = by_alpha(read_csv(args.pareto_v1 / "oasis20_pareto_source_rows.csv"))
    v2 = by_alpha(read_csv(args.pareto_v2 / "oasis20_pareto_source_rows.csv"))
    metrics = [
        "dice_mean_mean",
        "hd95_mean_mean",
        "asd_symmetric_mean_mean",
        "log_jdet_std_fg_mean",
        "jdet_le0_pct_fg_mean",
        "ndv_frac_fg_mean",
    ]
    rows: list[dict[str, Any]] = []
    for alpha in sorted(set(v1) | set(v2), key=safe_float):
        row: dict[str, Any] = {
            "alpha": alpha,
            "v001_lr": v1.get(alpha, {}).get("lr", ""),
            "v002_candidate_lr": v2.get(alpha, {}).get("lr", ""),
        }
        for metric in metrics:
            a = safe_float(v1.get(alpha, {}).get(metric))
            b = safe_float(v2.get(alpha, {}).get(metric))
            row[f"v001_{metric}"] = v1.get(alpha, {}).get(metric, "")
            row[f"v002_candidate_{metric}"] = v2.get(alpha, {}).get(metric, "")
            row[f"delta_{metric}"] = fmt_delta(b - a)
        rows.append(row)
    return rows


def side_by_side(v1_path: Path, v2_path: Path, out_path: Path, title: str) -> str:
    if not v1_path.exists() or not v2_path.exists():
        return ""
    img1 = plt.imread(v1_path)
    img2 = plt.imread(v2_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, img, label in zip(axes, [img1, img2], ["v001", "v002 candidate"]):
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return repo_relative(out_path)


def write_report(
    args: argparse.Namespace,
    table_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    side_by_side_paths: Sequence[str],
) -> None:
    table_v1_counts = validation_counts(args.table_v1 / "standard_validation.csv")
    table_v2_counts = validation_counts(args.table_v2 / "standard_validation.csv")
    pareto_v1_counts = validation_counts(args.pareto_v1 / "standard_validation.csv")
    pareto_v2_counts = validation_counts(args.pareto_v2 / "standard_validation.csv")
    dcp_table = [row for row in table_rows if row.get("Method") == "MR-D-BSCP (8-4-2)"]
    dcp_row = dcp_table[0] if dcp_table else {}
    recommendation = "inspect_further"
    if (
        table_v2_counts.get("temporary_exception", 0) == 0
        and pareto_v2_counts.get("temporary_exception", 0) == 0
        and table_v2_counts.get("mismatch", 0) == 0
        and pareto_v2_counts.get("mismatch", 0) == 0
    ):
        recommendation = "candidate_is_standardized_review_metrics_before_promotion"
    lines = [
        "# v001 vs v002 Candidate Paper Artifact Comparison",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- v001 table: `{repo_relative(args.table_v1)}`",
        f"- v002 candidate table: `{repo_relative(args.table_v2)}`",
        f"- v001 Pareto: `{repo_relative(args.pareto_v1)}`",
        f"- v002 candidate Pareto: `{repo_relative(args.pareto_v2)}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Validation Status Counts",
        "",
        f"- Table v001: `{table_v1_counts}`",
        f"- Table v002 candidate: `{table_v2_counts}`",
        f"- Pareto v001: `{pareto_v1_counts}`",
        f"- Pareto v002 candidate: `{pareto_v2_counts}`",
        "",
        "## OASIS DirectCP 8-4-2 Table Row",
        "",
        f"- v001 selected: `{dcp_row.get('v001_Selected', '')}`",
        f"- v002 candidate selected: `{dcp_row.get('v002_candidate_Selected', '')}`",
        f"- v001 Dice: `{dcp_row.get('v001_Dice', '')}`",
        f"- v002 candidate Dice: `{dcp_row.get('v002_candidate_Dice', '')}`",
        "",
        "## Files",
        "",
        "- `best_result_table_delta.csv`",
        "- `oasis_directcp842_alpha_curve_delta.csv`",
        "- `validation_status_comparison.csv`",
    ]
    if side_by_side_paths:
        lines.extend(["", "## Side-by-Side Plots", ""])
        lines.extend(f"- `{path}`" for path in side_by_side_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = table_delta(args)
    curve_rows = alpha_curve_delta(args)
    table_fields = list(table_rows[0].keys()) if table_rows else ["Method"]
    curve_fields = list(curve_rows[0].keys()) if curve_rows else ["alpha"]
    write_csv(args.out_dir / "best_result_table_delta.csv", table_rows, table_fields)
    write_csv(args.out_dir / "oasis_directcp842_alpha_curve_delta.csv", curve_rows, curve_fields)
    validation_rows = [
        {"artifact": "best_result_table", "version": "v001", **validation_counts(args.table_v1 / "standard_validation.csv")},
        {"artifact": "best_result_table", "version": "v002_candidate", **validation_counts(args.table_v2 / "standard_validation.csv")},
        {"artifact": "pareto_figure", "version": "v001", **validation_counts(args.pareto_v1 / "standard_validation.csv")},
        {"artifact": "pareto_figure", "version": "v002_candidate", **validation_counts(args.pareto_v2 / "standard_validation.csv")},
    ]
    validation_fields = sorted({key for row in validation_rows for key in row})
    write_csv(args.out_dir / "validation_status_comparison.csv", validation_rows, validation_fields)
    plot_specs = [
        (
            "oasis20_paper_no_annotations/big_plots/spotlight_line_segments__dice_mean_vs_log_jdet_std_fg.png",
            "side_by_side_dice_vs_foreground_logj.png",
            "Mean Dice vs Foreground std(log Jdet)",
        ),
        (
            "oasis20_paper_no_annotations/big_plots/spotlight_line_segments__hd95_mean_vs_log_jdet_std_fg.png",
            "side_by_side_hd95_vs_foreground_logj.png",
            "Mean HD95 vs Foreground std(log Jdet)",
        ),
        (
            "oasis20_paper_no_annotations/big_plots/contact_sheet_line_segments.png",
            "side_by_side_contact_sheet_line_segments.png",
            "OASIS line-segment contact sheet",
        ),
    ]
    side_paths = [
        side_by_side(args.pareto_v1 / rel, args.pareto_v2 / rel, args.out_dir / out_name, title)
        for rel, out_name, title in plot_specs
    ]
    side_paths = [path for path in side_paths if path]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table_v1": repo_relative(args.table_v1),
        "table_v2": repo_relative(args.table_v2),
        "pareto_v1": repo_relative(args.pareto_v1),
        "pareto_v2": repo_relative(args.pareto_v2),
        "outputs": {
            "best_result_table_delta": repo_relative(args.out_dir / "best_result_table_delta.csv"),
            "oasis_directcp842_alpha_curve_delta": repo_relative(args.out_dir / "oasis_directcp842_alpha_curve_delta.csv"),
            "validation_status_comparison": repo_relative(args.out_dir / "validation_status_comparison.csv"),
            "side_by_side": side_paths,
        },
    }
    (args.out_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(args, table_rows, curve_rows, side_paths)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
