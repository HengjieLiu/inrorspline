#!/usr/bin/env python3
"""Rerun the LungCT cold/warm study with cuDNN benchmark off/deterministic on."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import launch_mrdbscp_adaptive_fixed50_gpu1 as base


EXPERIMENT_ID = "2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1"
RUN_PREFIX = "lungct_mrdbscp_cudnn_ft_gpu1_"
P34_EXPERIMENT_ID = "2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1"
P34_RUN_ID = "lungct_mrdbscp_adaptive_fixed50_gpu1_20260929T170843Z"
P35_EXPERIMENT_ID = "2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1"
P35_RUN_ID = "lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_20260929T211725Z"

_original_run_one_argv = base.run_one_argv
_original_manifest = base.immutable_manifest


def backend_run_one_argv(
    args: Any,
    dataset: str,
    variant: str,
    row: dict[str, Any],
    *,
    smoke: bool,
) -> list[str]:
    argv = _original_run_one_argv(args, dataset, variant, row, smoke=smoke)
    argv.extend(["--no-cudnn-benchmark", "--cudnn-deterministic"])
    return argv


def backend_manifest(args: Any) -> dict[str, Any]:
    payload = _original_manifest(args)
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "backend": {
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "comparison_references": {
                "original_cold_warm": {
                    "experiment_id": P34_EXPERIMENT_ID,
                    "run_id": P34_RUN_ID,
                    "backend": "benchmark_true_deterministic_false",
                },
                "adjacent_shape_pairs": {
                    "experiment_id": P35_EXPERIMENT_ID,
                    "run_id": P35_RUN_ID,
                    "backend": "benchmark_true_deterministic_false",
                },
            },
            "comparison_policy": (
                "new ten-case warm cohort versus P34 warm cohort and P35 immediate repeats; "
                "fixed50 is primary because optimizer steps match"
            ),
        }
    )
    return payload


def configure_base() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.RUN_PREFIX = RUN_PREFIX
    base.WORKER_SCRIPT_NAME = Path(__file__).name
    base.AGGREGATOR_SCRIPT_NAME = "aggregate_mrdbscp_cudnn_ft_gpu1.py"
    base.run_one_argv = backend_run_one_argv
    base.immutable_manifest = backend_manifest
    base.SOURCE_PATHS = (
        Path("AGENTS.md"),
        Path("docs/standards/lungct_protocol_tags.md"),
        Path("docs/standards/lungct_tre_metric_conventions.md"),
        Path("scripts/experiments/artifact_paths.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/common.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/run_one.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/aggregate_mrdbscp_adaptive_fixed50_gpu1.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_cudnn_ft_gpu1.py"),
        Path("scripts/experiments/lungct_no_sweep_repro/aggregate_mrdbscp_cudnn_ft_gpu1.py"),
        Path("tests/test_lungct_mrdbscp_cudnn_ft_gpu1.py"),
        Path("experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/codex_execution_spec.md"),
        Path("experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/technical_log.md"),
    )


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
