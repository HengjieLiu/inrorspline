#!/usr/bin/env python
"""Aggregate selected fast OASIS100 schedules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from launch_oasis_selected_fast_first100 import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    FIRST10_SOURCE_ROOT,
    SELECTED_CANDIDATES,
    VARIANT,
)
from launch_oasis_unified_acceleration_sweep import candidate_map  # noqa: E402


METRICS = [
    "dice_mean",
    "hd95_mean",
    "asd_symmetric_mean",
    "asd_warped_to_fixed_mean",
    "asd_fixed_to_warped_mean",
    "jdet_le0_pct_fg",
    "log_jdet_std_fg",
    "be_native_analytic_fixed_fg_voxel",
    "diffusion_native_analytic_fixed_fg_voxel",
    "training_elapsed_sec",
    "total_elapsed_sec",
    "full_registration_time_sec",
    "loaded_images_to_dvf_file_sec",
    "post_dvf_evaluation_sec",
    "actual_epochs",
    "training_peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--first10-source-root", type=Path, default=FIRST10_SOURCE_ROOT)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(rows: Iterable[Mapping[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def std(rows: Iterable[Mapping[str, object]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else float("nan")


def collect_command_rows(out_root: Path) -> Dict[str, Dict[str, object]]:
    by_run_dir: Dict[str, Dict[str, object]] = {}
    for path in sorted(out_root.glob("command_timing_selected_first100_*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                by_run_dir[str(row.get("run_dir"))] = dict(row)
    return by_run_dir


def row_from_summary(path: Path, *, source: str, command_by_run_dir: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    summary = json.loads(path.read_text())
    candidate_name = str(summary.get("runtime_candidate") or "")
    candidate = candidate_map()[candidate_name]
    command = command_by_run_dir.get(str(path.parent), {})
    row: Dict[str, object] = {
        "source": source,
        "candidate": candidate_name,
        "method": candidate.method,
        "method_label": candidate.method_label,
        "family": candidate.family,
        "template": candidate.template,
        "case_rank": int(as_float(summary.get("case_rank"))),
        "original_case_index": summary.get("original_case_index"),
        "fixed_id": summary.get("fixed_id"),
        "moving_id": summary.get("moving_id"),
        "summary_path": str(path),
        "run_dir": str(path.parent),
        "worker_name": command.get("worker_name", ""),
        "gpu": command.get("gpu", ""),
        "case_sequence_index": command.get("case_sequence_index", ""),
        "warm_status": command.get("warm_status", "first10_reused"),
        "external_case_wall_sec": as_float(command.get("external_case_wall_sec")),
    }
    for key in METRICS:
        row[key] = as_float(summary.get(key))
    return row


def collect_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    command_by_run_dir = collect_command_rows(Path(args.out_root))
    rows: List[Dict[str, object]] = []
    for path in sorted(Path(args.first10_source_root).glob("*/*/summary.json")):
        run_name = path.parent.name
        if "unified_smoke" in run_name:
            continue
        if "unified_primary_" not in run_name and "unified_cps4_" not in run_name:
            continue
        payload = json.loads(path.read_text())
        if str(payload.get("runtime_candidate")) not in SELECTED_CANDIDATES:
            continue
        if int(as_float(payload.get("case_rank"))) not in range(10):
            continue
        rows.append(row_from_summary(path, source="first10_reused", command_by_run_dir={}))
    for path in sorted(Path(args.out_root).glob("*/*/summary.json")):
        run_name = path.parent.name
        if "selected_first100_smoke" in run_name:
            continue
        if "selected_first100" not in run_name:
            continue
        payload = json.loads(path.read_text())
        if str(payload.get("runtime_candidate")) not in SELECTED_CANDIDATES:
            continue
        if int(as_float(payload.get("case_rank"))) not in range(10, 100):
            continue
        rows.append(row_from_summary(path, source="ranks10_99_new", command_by_run_dir=command_by_run_dir))
    rows.sort(key=lambda row: (str(row["candidate"]), int(row["case_rank"])))
    return rows


def missing_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    observed = {(str(row["candidate"]), int(as_float(row["case_rank"]))) for row in rows}
    missing: List[Dict[str, object]] = []
    for candidate in SELECTED_CANDIDATES:
        for rank in range(100):
            if (candidate, rank) not in observed:
                missing.append({"candidate": candidate, "case_rank": rank})
    return missing


def summary_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate"])].append(row)
    out: List[Dict[str, object]] = []
    for candidate_name in SELECTED_CANDIDATES:
        group = grouped.get(candidate_name, [])
        candidate = candidate_map()[candidate_name]
        peak_rows = sorted(
            [row for row in group if math.isfinite(as_float(row.get("training_peak_mem_gb")))],
            key=lambda row: as_float(row.get("training_peak_mem_gb")),
            reverse=True,
        )
        row: Dict[str, object] = {
            "candidate": candidate_name,
            "method": candidate.method,
            "method_label": candidate.method_label,
            "family": candidate.family,
            "template": candidate.template,
            "case_count": len(group),
            "first10_reused_count": sum(1 for item in group if item.get("source") == "first10_reused"),
            "new_rank10_99_count": sum(1 for item in group if item.get("source") == "ranks10_99_new"),
        }
        for key in METRICS:
            row[f"{key}_mean"] = mean(group, key)
            row[f"{key}_std"] = std(group, key)
        row["training_peak_mem_gb_max"] = as_float(peak_rows[0].get("training_peak_mem_gb")) if peak_rows else float("nan")
        row["training_peak_mem_worst_case_rank"] = int(as_float(peak_rows[0].get("case_rank"))) if peak_rows else ""
        row["training_peak_mem_worst_fixed_id"] = peak_rows[0].get("fixed_id") if peak_rows else ""
        row["training_peak_mem_worst_moving_id"] = peak_rows[0].get("moving_id") if peak_rows else ""
        out.append(row)
    return out


def runtime_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    metrics = ["full_registration_time_sec", "loaded_images_to_dvf_file_sec", "external_case_wall_sec"]
    for candidate in SELECTED_CANDIDATES:
        candidate_rows = [row for row in rows if row.get("candidate") == candidate and row.get("source") == "ranks10_99_new"]
        groups = {
            "all_new_ranks10_99": candidate_rows,
            "worker_cold_cases": [row for row in candidate_rows if row.get("warm_status") == "cold_worker_first_case"],
            "worker_warm_cases": [row for row in candidate_rows if row.get("warm_status") == "warm_worker_later_case"],
        }
        for group_name, group in groups.items():
            row: Dict[str, object] = {"candidate": candidate, "group": group_name, "case_count": len(group)}
            for metric in metrics:
                row[f"{metric}_mean"] = mean(group, metric)
                row[f"{metric}_std"] = std(group, metric)
            out.append(row)
    return out


def fmt(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_report(args: argparse.Namespace, rows: Sequence[Mapping[str, object]], summaries: Sequence[Mapping[str, object]], runtimes: Sequence[Mapping[str, object]], missing: Sequence[Mapping[str, object]]) -> None:
    lines: List[str] = [
        "# OASIS100 Selected Fast-Schedule Evaluation",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Variant: `{args.variant}`",
        f"- New output root: `{args.out_root}`",
        f"- Reused first10 root: `{args.first10_source_root}`",
        f"- Combined rows: `{len(rows)}/500`",
        f"- Missing rows: `{len(missing)}`",
        "- Runtime in this report is reference-only because GPUs were shared/occupied during this run.",
        "- Final paper benchmark timing should be rerun under isolated persistent-process conditions.",
        "",
        "## Accuracy And Regularity Summary",
        "",
        "| Candidate | Method | Cases | Dice | HD95 | ASD | FG stdLogJ | FG J<=0% | Epochs | Peak Mem Max GB | Worst Mem Rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["method_label"]),
                    str(row["case_count"]),
                    fmt(row["dice_mean_mean"], 4),
                    fmt(row["hd95_mean_mean"], 3),
                    fmt(row["asd_symmetric_mean_mean"], 3),
                    fmt(row["log_jdet_std_fg_mean"], 4),
                    fmt(row["jdet_le0_pct_fg_mean"], 4),
                    fmt(row["actual_epochs_mean"], 1),
                    fmt(row["training_peak_mem_gb_max"], 2),
                    str(row["training_peak_mem_worst_case_rank"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reference Runtime",
            "",
            "These values are not the final benchmark. They are retained to check practical scale while other jobs may have shared the GPUs.",
            "",
            "| Candidate | Group | Cases | Full Reg Mean | Loaded Images To DVF File Mean | External Wall Mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in runtimes:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["group"]),
                    str(row["case_count"]),
                    fmt(row["full_registration_time_sec_mean"], 2),
                    fmt(row["loaded_images_to_dvf_file_sec_mean"], 2),
                    fmt(row["external_case_wall_sec_mean"], 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reuse Policy",
            "",
            "- Ranks `0-9` are reused from the completed unified first10 acceleration sweep.",
            "- Ranks `10-99` are newly evaluated in this selected-first100 output root.",
            "- The combined report therefore contains 100 rows per selected candidate.",
            "",
        ]
    )
    if missing:
        lines.extend(["## Missing Rows", "", f"- Missing expected rows: `{len(missing)}`", ""])
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    (Path(args.review_root) / "selected_fast_oasis100_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args)
    missing = missing_rows(rows)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Missing {len(missing)} rows; rerun with --allow-incomplete to aggregate partial results")
    summaries = summary_rows(rows)
    runtimes = runtime_rows(rows)
    c.write_csv(Path(args.review_root) / "selected_fast_oasis100_case_results.csv", rows)
    c.write_json(Path(args.review_root) / "selected_fast_oasis100_case_results.json", rows)
    c.write_csv(Path(args.review_root) / "selected_fast_oasis100_summary.csv", summaries)
    c.write_json(Path(args.review_root) / "selected_fast_oasis100_summary.json", summaries)
    c.write_csv(Path(args.review_root) / "selected_fast_oasis100_reference_runtime.csv", runtimes)
    c.write_json(Path(args.review_root) / "missing_or_failed_jobs.json", missing)
    c.write_json(
        Path(args.review_root) / "selected_fast_oasis100_aggregation_status.json",
        {
            "created_at": c.now_text(),
            "combined_rows": len(rows),
            "missing_count": len(missing),
            "first10_source_root": str(args.first10_source_root),
            "out_root": str(args.out_root),
        },
    )
    write_report(args, rows, summaries, runtimes, missing)
    print(json.dumps({"combined_rows": len(rows), "missing_count": len(missing), "review_root": str(args.review_root)}, indent=2))


if __name__ == "__main__":
    main()
