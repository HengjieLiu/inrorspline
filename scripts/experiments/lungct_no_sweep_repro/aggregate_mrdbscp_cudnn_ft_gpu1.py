#!/usr/bin/env python3
"""Validate the cuDNN F/T rerun and compare it with P34 and P35 timings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import common as c
import launch_mrdbscp_cudnn_ft_gpu1 as study

study.configure_base()

import aggregate_mrdbscp_adaptive_fixed50_gpu1 as base_aggregate  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    return parser.parse_args(argv)


def reference_review_root(review_root: Path, experiment_id: str, run_id: str) -> Path:
    review_root = review_root.resolve()
    if len(review_root.parents) < 3 or review_root.parents[1].name != "repo_experiments":
        raise ValueError(f"Unexpected review-root layout: {review_root}")
    artifact_root = review_root.parents[2]
    return artifact_root / "repo_experiments" / experiment_id / run_id


def read_reference_json(path: Path, errors: list[str]) -> Any:
    if not path.is_file():
        errors.append(f"missing comparison reference: {path}")
        return []
    try:
        return c.read_json(path)
    except Exception as exc:
        errors.append(f"invalid comparison reference {path}: {exc!r}")
        return []


def backend_errors(rows: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        summary_path = Path(str(row["summary_path"]))
        summary = c.read_json(summary_path)
        backend = summary.get("backend", {}) if isinstance(summary, dict) else {}
        expected = {
            "requested_cudnn_benchmark": False,
            "requested_cudnn_deterministic": True,
            "effective_cudnn_benchmark": False,
            "effective_cudnn_deterministic": True,
        }
        for key, value in expected.items():
            if not isinstance(backend, dict) or backend.get(key) is not value:
                errors.append(
                    f"{row['dataset']}/{row['variant']}/case{int(row['case_id']):02d}/"
                    f"{row['timing_condition']}: backend {key}={backend.get(key) if isinstance(backend, dict) else None!r}"
                )
        for key in ("torch_version", "cuda_runtime_version", "cudnn_version", "cuda_device_name"):
            if not isinstance(backend, dict) or backend.get(key) in (None, "", 0):
                errors.append(f"{summary_path}: missing backend provenance {key}")
    return errors


def build_reference_comparison(
    rows: Sequence[dict[str, Any]],
    p34_rows: Sequence[dict[str, Any]],
    p35_pairs: Sequence[dict[str, Any]],
    errors: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    p34_index = {
        (str(row["dataset"]), str(row["variant"]), int(row["case_id"]), str(row["timing_condition"])): row
        for row in p34_rows
    }
    p35_index = {
        (str(row["dataset"]), str(row["variant"]), int(row["case_id"])): row
        for row in p35_pairs
    }
    warm_rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        case_id = int(row["case_id"])
        condition = str(row["timing_condition"])
        if condition == "warm":
            old = p34_index.get((dataset, variant, case_id, "warm"))
            pair = p35_index.get((dataset, variant, case_id))
            if old is None or pair is None:
                errors.append(f"missing warm comparison row for {dataset}/{variant}/case{case_id:02d}")
                continue
            new_time = base_aggregate.finite(row["loaded_images_to_dvf_file_sec"])
            old_time = base_aggregate.finite(old["loaded_images_to_dvf_file_sec"])
            repeat_time = base_aggregate.finite(pair["repeat_loaded_images_to_dvf_file_sec"])
            new_tre = base_aggregate.finite(row["native_crop_tre_snap_mean_mm"])
            old_tre = base_aggregate.finite(old["native_crop_tre_snap_mean_mm"])
            repeat_tre = base_aggregate.finite(pair["repeat_native_crop_tre_snap_to_voxel_mean_mm"])
            warm_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "case_id": case_id,
                    "ft_warm_sec": new_time,
                    "prior_tf_warm_sec": old_time,
                    "interleaved_tf_repeat_sec": repeat_time,
                    "speedup_prior_tf_over_ft": old_time / new_time,
                    "ft_over_interleaved_repeat_ratio": new_time / repeat_time,
                    "ft_actual_optimizer_steps": int(row["actual_optimizer_steps"]),
                    "prior_tf_actual_optimizer_steps": int(old["actual_optimizer_steps"]),
                    "interleaved_tf_repeat_optimizer_steps": int(pair["repeat_actual_optimizer_steps"]),
                    "ft_native_crop_tre_snap_mean_mm": new_tre,
                    "prior_tf_native_crop_tre_snap_mean_mm": old_tre,
                    "interleaved_tf_repeat_native_crop_tre_snap_mean_mm": repeat_tre,
                    "ft_minus_prior_tf_tre_mm": new_tre - old_tre,
                    "ft_minus_interleaved_repeat_tre_mm": new_tre - repeat_tre,
                }
            )
        elif condition == "cold" and case_id == 1:
            old = p34_index.get((dataset, variant, 1, "cold"))
            pair = p35_index.get((dataset, variant, 1))
            if old is None or pair is None:
                errors.append(f"missing cold comparison row for {dataset}/{variant}")
                continue
            new_time = base_aggregate.finite(row["loaded_images_to_dvf_file_sec"])
            old_time = base_aggregate.finite(old["loaded_images_to_dvf_file_sec"])
            first_time = base_aggregate.finite(pair["first_loaded_images_to_dvf_file_sec"])
            cold_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "ft_cold_case01_sec": new_time,
                    "prior_tf_cold_case01_sec": old_time,
                    "interleaved_tf_shape_first_case01_sec": first_time,
                    "speedup_prior_tf_cold_over_ft": old_time / new_time,
                    "speedup_interleaved_tf_first_over_ft": first_time / new_time,
                    "ft_actual_optimizer_steps": int(row["actual_optimizer_steps"]),
                    "prior_tf_actual_optimizer_steps": int(old["actual_optimizer_steps"]),
                    "interleaved_tf_shape_first_optimizer_steps": int(pair["first_actual_optimizer_steps"]),
                }
            )

    if len(warm_rows) != 40:
        errors.append(f"backend warm comparison rows={len(warm_rows)}, expected=40")
    if len(cold_rows) != 4:
        errors.append(f"backend cold comparison rows={len(cold_rows)}, expected=4")

    aggregate_rows: list[dict[str, Any]] = []
    for dataset in study.base.DATASETS:
        for variant in study.base.VARIANTS:
            selected = [row for row in warm_rows if row["dataset"] == dataset and row["variant"] == variant]
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "case_count": len(selected),
                    "ft_warm_mean_sec": base_aggregate.mean(row["ft_warm_sec"] for row in selected),
                    "ft_warm_sd_sec": base_aggregate.sample_sd(row["ft_warm_sec"] for row in selected),
                    "prior_tf_warm_mean_sec": base_aggregate.mean(row["prior_tf_warm_sec"] for row in selected),
                    "prior_tf_warm_sd_sec": base_aggregate.sample_sd(row["prior_tf_warm_sec"] for row in selected),
                    "interleaved_tf_repeat_mean_sec": base_aggregate.mean(row["interleaved_tf_repeat_sec"] for row in selected),
                    "interleaved_tf_repeat_sd_sec": base_aggregate.sample_sd(row["interleaved_tf_repeat_sec"] for row in selected),
                    "speedup_prior_tf_over_ft_ratio_of_means": (
                        base_aggregate.mean(row["prior_tf_warm_sec"] for row in selected)
                        / base_aggregate.mean(row["ft_warm_sec"] for row in selected)
                    ),
                    "ft_over_interleaved_repeat_ratio_of_means": (
                        base_aggregate.mean(row["ft_warm_sec"] for row in selected)
                        / base_aggregate.mean(row["interleaved_tf_repeat_sec"] for row in selected)
                    ),
                    "step_mismatch_vs_prior_tf_count": sum(
                        int(row["ft_actual_optimizer_steps"]) != int(row["prior_tf_actual_optimizer_steps"])
                        for row in selected
                    ),
                    "step_mismatch_vs_interleaved_repeat_count": sum(
                        int(row["ft_actual_optimizer_steps"]) != int(row["interleaved_tf_repeat_optimizer_steps"])
                        for row in selected
                    ),
                    "ft_minus_prior_tf_tre_mean_mm": base_aggregate.mean(
                        row["ft_minus_prior_tf_tre_mm"] for row in selected
                    ),
                    "ft_minus_interleaved_repeat_tre_mean_mm": base_aggregate.mean(
                        row["ft_minus_interleaved_repeat_tre_mm"] for row in selected
                    ),
                }
            )
    return warm_rows, cold_rows, aggregate_rows


def append_backend_report(
    report_path: Path,
    warm_rows: Sequence[dict[str, Any]],
    cold_rows: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
    p34_root: Path,
    p35_root: Path,
) -> None:
    fmt = base_aggregate.fmt
    lines = [
        "",
        "## cuDNN backend and timing references",
        "",
        "- This run requests and records `benchmark=False, deterministic=True` for all 44 measurements.",
        f"- Prior original-layout reference: `{p34_root}` (`benchmark=True, deterministic=False`).",
        f"- Prior adjacent-pair reference: `{p35_root}` (`benchmark=True, deterministic=False`).",
        "- Fixed-50 is the primary backend comparison because all references execute identical optimizer steps.",
        "",
        "| Dataset | Variant | New F/T warm s | Prior T/F warm s | Prior/interleaved T/F repeat s | Prior T/F / new F/T | New F/T / interleaved floor | Step mismatches vs prior / repeat |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {fmt(row['ft_warm_mean_sec'])} +/- {fmt(row['ft_warm_sd_sec'])} | "
            f"{fmt(row['prior_tf_warm_mean_sec'])} +/- {fmt(row['prior_tf_warm_sd_sec'])} | "
            f"{fmt(row['interleaved_tf_repeat_mean_sec'])} +/- {fmt(row['interleaved_tf_repeat_sd_sec'])} | "
            f"{fmt(row['speedup_prior_tf_over_ft_ratio_of_means'], 2)}x | "
            f"{fmt(row['ft_over_interleaved_repeat_ratio_of_means'], 2)}x | "
            f"{row['step_mismatch_vs_prior_tf_count']} / {row['step_mismatch_vs_interleaved_repeat_count']} |"
        )
    lines.extend(
        [
            "",
            "### Case-01 process-cold comparison",
            "",
            "| Dataset | Variant | New F/T cold s | Prior T/F cold s | Interleaved T/F shape-first s | Prior/new speedup | Interleaved/new speedup | Steps new/prior/interleaved |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cold_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {fmt(row['ft_cold_case01_sec'])} | "
            f"{fmt(row['prior_tf_cold_case01_sec'])} | {fmt(row['interleaved_tf_shape_first_case01_sec'])} | "
            f"{fmt(row['speedup_prior_tf_cold_over_ft'], 2)}x | {fmt(row['speedup_interleaved_tf_first_over_ft'], 2)}x | "
            f"{row['ft_actual_optimizer_steps']}/{row['prior_tf_actual_optimizer_steps']}/{row['interleaved_tf_shape_first_optimizer_steps']} |"
        )
    for dataset in study.base.DATASETS:
        lines.extend(
            [
                "",
                f"### {dataset.upper()} casewise warm timing",
                "",
                "| Case | Variant | New F/T warm s | Prior T/F warm s | Interleaved T/F repeat s | Steps new/prior/repeat |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in warm_rows:
            if row["dataset"] != dataset:
                continue
            lines.append(
                f"| {int(row['case_id']):02d} | {row['variant']} | {fmt(row['ft_warm_sec'])} | "
                f"{fmt(row['prior_tf_warm_sec'])} | {fmt(row['interleaved_tf_repeat_sec'])} | "
                f"{row['ft_actual_optimizer_steps']}/{row['prior_tf_actual_optimizer_steps']}/"
                f"{row['interleaved_tf_repeat_optimizer_steps']} |"
            )
    with report_path.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows, errors = base_aggregate.read_case_rows(args.run_root)
    errors.extend(base_aggregate.cohort_errors(rows))
    errors.extend(backend_errors(rows))

    p34_root = reference_review_root(args.review_root, study.P34_EXPERIMENT_ID, study.P34_RUN_ID)
    p35_root = reference_review_root(args.review_root, study.P35_EXPERIMENT_ID, study.P35_RUN_ID)
    p34_acceptance = read_reference_json(p34_root / "acceptance.json", errors)
    p35_acceptance = read_reference_json(p35_root / "acceptance.json", errors)
    if not isinstance(p34_acceptance, dict) or not p34_acceptance.get("passed"):
        errors.append("P34 comparison reference is not accepted")
    if not isinstance(p35_acceptance, dict) or not p35_acceptance.get("passed"):
        errors.append("P35 comparison reference is not accepted")
    p34_rows = read_reference_json(p34_root / "case_rows.json", errors)
    p35_pairs = read_reference_json(p35_root / "paired_case_rows.json", errors)
    if not isinstance(p34_rows, list):
        errors.append("P34 case_rows.json is not a list")
        p34_rows = []
    if not isinstance(p35_pairs, list):
        errors.append("P35 paired_case_rows.json is not a list")
        p35_pairs = []

    warm_comparison, cold_comparison, backend_aggregates = build_reference_comparison(
        rows, p34_rows, p35_pairs, errors
    )
    timing_rows, quality_rows, comparison_rows, stage_rows = base_aggregate.aggregate_rows(rows)
    c.write_csv(args.review_root / "case_rows.csv", rows)
    c.write_json(args.review_root / "case_rows.json", rows)
    c.write_csv(args.review_root / "timing_aggregate.csv", timing_rows)
    c.write_json(args.review_root / "timing_aggregate.json", timing_rows)
    c.write_csv(args.review_root / "quality_aggregate.csv", quality_rows)
    c.write_json(args.review_root / "quality_aggregate.json", quality_rows)
    c.write_csv(args.review_root / "fixed50_vs_adaptive.csv", comparison_rows)
    c.write_json(args.review_root / "fixed50_vs_adaptive.json", comparison_rows)
    c.write_csv(args.review_root / "stage_rows.csv", stage_rows)
    c.write_json(args.review_root / "stage_rows.json", stage_rows)
    c.write_csv(args.review_root / "backend_warm_case_comparison.csv", warm_comparison)
    c.write_json(args.review_root / "backend_warm_case_comparison.json", warm_comparison)
    c.write_csv(args.review_root / "backend_cold_case01_comparison.csv", cold_comparison)
    c.write_json(args.review_root / "backend_cold_case01_comparison.json", cold_comparison)
    c.write_csv(args.review_root / "backend_timing_aggregate.csv", backend_aggregates)
    c.write_json(args.review_root / "backend_timing_aggregate.json", backend_aggregates)

    base_aggregate.write_report(
        args.review_root, rows, timing_rows, quality_rows, comparison_rows, errors
    )
    append_backend_report(
        args.review_root / "review_report.md",
        warm_comparison,
        cold_comparison,
        backend_aggregates,
        p34_root,
        p35_root,
    )
    acceptance = {
        "experiment_id": study.EXPERIMENT_ID,
        "passed": not errors,
        "errors": errors,
        "primary_row_count": len(rows),
        "warm_row_count": len([row for row in rows if row["timing_condition"] == "warm"]),
        "cold_row_count": len([row for row in rows if row["timing_condition"] == "cold"]),
        "backend": {"cudnn_benchmark": False, "cudnn_deterministic": True},
        "physical_gpu": study.base.PHYSICAL_GPU,
        "timing_endpoint": "case_load_complete_to_dvf_file_ready",
        "references": {"p34": str(p34_root), "p35": str(p35_root)},
    }
    c.write_json(args.review_root / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
