#!/usr/bin/env python3
"""Run adjacent first/repeat pairs for every LungCT case on physical GPU 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import launch_mrdbscp_adaptive_fixed50_gpu1 as base


EXPERIMENT_ID = "2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1"
RUN_PREFIX = "lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_"
EXPECTED_PRIMARY_ROWS = 80
EXPECTED_REPEAT_ROWS = 40


def paired_case_sequence(smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "case_id": 1,
                "sequence_index": 0,
                "timing_condition": "smoke",
                "case_occurrence": "smoke",
            }
        ]
    rows: list[dict[str, Any]] = []
    for case_id in base.CASES:
        for repeat_index, condition in enumerate(("shape_first", "shape_repeat")):
            rows.append(
                {
                    "case_id": int(case_id),
                    "sequence_index": 2 * (int(case_id) - 1) + repeat_index,
                    "timing_condition": condition,
                    "case_occurrence": condition,
                }
            )
    return rows


def paired_run_name(dataset: str, variant: str, row: dict[str, Any]) -> str:
    return (
        f"{EXPERIMENT_ID}_{dataset}_{variant}_case{int(row['case_id']):02d}_"
        f"{row['case_occurrence']}"
    )


_original_manifest = base.immutable_manifest


def diagnostic_manifest(args: Any) -> dict[str, Any]:
    payload = _original_manifest(args)
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
            "expected_shape_repeat_rows": EXPECTED_REPEAT_ROWS,
            "case_sequence": "for each case01-10: shape_first then adjacent shape_repeat",
            "diagnostic_question": (
                "separate process-warm/shape-first latency from adjacent same-shape repeat latency"
            ),
            "interpretation": (
                "fixed50 isolates shape/backend state because steps are identical; adaptive also records step drift"
            ),
            "prior_result_policy": "P34 fields and timings remain immutable; this is a separate diagnostic",
        }
    )
    payload.pop("cold_warm_sequence", None)
    payload.pop("historical_anchor_policy", None)
    return payload


def configure_base() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.RUN_PREFIX = RUN_PREFIX
    base.EXPECTED_PRIMARY_ROWS = EXPECTED_PRIMARY_ROWS
    base.EXPECTED_WARM_ROWS = EXPECTED_REPEAT_ROWS
    base.PRIMARY_ROWS_PER_ARM = 20
    base.WORKER_SCRIPT_NAME = Path(__file__).name
    base.AGGREGATOR_SCRIPT_NAME = "aggregate_mrdbscp_shape_pair_cache_diagnostic_gpu1.py"
    base.case_sequence = paired_case_sequence
    base.run_name = paired_run_name
    base.immutable_manifest = diagnostic_manifest
    base.SOURCE_PATHS = (
        Path("AGENTS.md"),
        Path("docs/standards/lungct_protocol_tags.md"),
        Path("docs/standards/lungct_tre_metric_conventions.md"),
        Path("scripts/experiments/artifact_paths.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/common.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/run_one.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_shape_pair_cache_diagnostic_gpu1.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/aggregate_mrdbscp_shape_pair_cache_diagnostic_gpu1.py"),
        Path("tests/test_lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1.py"),
        Path("experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/codex_execution_spec.md"),
        Path("experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/technical_log.md"),
    )


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
