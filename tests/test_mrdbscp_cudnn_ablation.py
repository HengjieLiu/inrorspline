from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
OASIS_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "oasis_alpha_sweep"
for path in (EXPERIMENTS_DIR, OASIS_EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402
import run_mrdbscp_cudnn_ablation as ablation  # noqa: E402


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "configs": "",
        "case_ranks": "0-4",
        "repeat_cold_rank": False,
        "smoke": False,
        "alpha": 1000.0,
        "gpu_monitor_interval": 1.0,
        "max_gpu_retries": 5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_historical_defaults_preserve_p2y_case_sequence_and_names() -> None:
    args = make_args()

    assert [config.slug for config in ablation.selected_configs(args)] == [config.slug for config in ablation.CONFIGS]
    steps = ablation.timing_steps(args)
    assert [(step.rank, step.timing_condition, step.warm_sequence_index) for step in steps] == [
        (0, "cold", None),
        (1, "warm", 1),
        (2, "warm", 2),
        (3, "warm", 3),
        (4, "warm", 4),
    ]

    candidate = unified.candidate_map()[ablation.CANDIDATE_NAME]
    assert ablation.timing_run_name(
        args,
        candidate,
        rank=0,
        timing_condition="cold",
        warm_sequence_index=None,
    ) == unified.run_name(candidate, 0, 1000.0, "primary")


def test_table1_mode_selects_three_configs_and_repeats_rank_zero() -> None:
    args = make_args(
        configs=(
            "benchmark_false_deterministic_true,"
            "benchmark_false_deterministic_false,"
            "benchmark_true_deterministic_false"
        ),
        case_ranks="0-9",
        repeat_cold_rank=True,
    )

    assert [config.slug for config in ablation.selected_configs(args)] == [
        "benchmark_false_deterministic_true",
        "benchmark_false_deterministic_false",
        "benchmark_true_deterministic_false",
    ]
    steps = ablation.timing_steps(args)
    assert len(steps) == 11
    assert [(step.rank, step.timing_condition, step.warm_sequence_index) for step in steps[:3]] == [
        (0, "cold", None),
        (0, "warm", 1),
        (1, "warm", 2),
    ]
    assert steps[-1].rank == 9
    assert steps[-1].warm_sequence_index == 10
    assert ablation.unique_case_ranks(args) == list(range(10))
    assert len(ablation.canonical_case_steps(args)) == 10

    candidate = unified.candidate_map()[ablation.CANDIDATE_NAME]
    cold_name = ablation.timing_run_name(
        args,
        candidate,
        rank=0,
        timing_condition="cold",
        warm_sequence_index=None,
    )
    warm_name = ablation.timing_run_name(
        args,
        candidate,
        rank=0,
        timing_condition="warm",
        warm_sequence_index=1,
    )
    assert cold_name.endswith("_cold")
    assert warm_name.endswith("_warm01")
    assert cold_name != warm_name


def test_table1_candidate_mode_preserves_paper_order_and_separates_outputs() -> None:
    args = make_args(
        candidates=",".join(ablation.TABLE1_FAST_CANDIDATES),
        configs="benchmark_false_deterministic_true,benchmark_false_deterministic_false,benchmark_true_deterministic_false",
        case_ranks="0-9",
        repeat_cold_rank=True,
        out_root=Path("/tmp/table1-cudnn-runs"),
        review_root=Path("/tmp/table1-cudnn-review"),
    )

    assert [candidate.name for candidate in ablation.selected_candidates(args)] == list(ablation.TABLE1_FAST_CANDIDATES)
    assert len(ablation.selected_candidates(args)) * len(ablation.selected_configs(args)) * len(ablation.timing_steps(args)) == 165
    candidate = unified.candidate_map()["D2_A_bal_lr10x"]
    config = ablation.CONFIG_BY_SLUG["benchmark_false_deterministic_true"]
    assert ablation.candidate_out_root(args, config, candidate) == Path("/tmp/table1-cudnn-runs") / config.slug / candidate.name
    assert ablation.run_dir(
        args,
        config,
        0,
        candidate=candidate,
        timing_condition="warm",
        warm_sequence_index=1,
    ).parent == ablation.candidate_out_root(args, config, candidate) / candidate.method


def test_table1_base_fast_mode_has_ten_candidates_and_110_timing_rows() -> None:
    args = make_args(
        candidates=",".join(ablation.TABLE1_BASE_FAST_CANDIDATES),
        configs="benchmark_false_deterministic_true",
        case_ranks="0-9",
        repeat_cold_rank=True,
    )

    assert [candidate.name for candidate in ablation.selected_candidates(args)] == list(ablation.TABLE1_BASE_FAST_CANDIDATES)
    assert len(ablation.selected_candidates(args)) == 10
    assert len(ablation.timing_steps(args)) == 11
    assert len(ablation.selected_candidates(args)) * len(ablation.timing_steps(args)) == 110
    assert ablation.timing_run_name(
        args,
        unified.candidate_map()["MR_A_fast_decay"],
        rank=0,
        timing_condition="cold",
        warm_sequence_index=None,
        attempt=2,
    ).endswith("_cold_retry02")


def test_compute_app_parser_and_step_key() -> None:
    apps = ablation.parse_compute_apps("123, python, 512\nNo running processes found")
    assert apps == [{"pid": 123, "process_name": "python", "used_memory_mb": "512"}]
    assert ablation.timing_step_key(ablation.TimingStep(0, "cold", None)) == "cold:0:"
    assert ablation.timing_step_key(ablation.TimingStep(4, "warm", 5)) == "warm:4:5"


def test_config_selection_rejects_duplicates_and_unknown_values() -> None:
    duplicate_args = make_args(configs="benchmark_false_deterministic_true,benchmark_false_deterministic_true")
    try:
        ablation.selected_configs(duplicate_args)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate configuration selection was accepted")

    unknown_args = make_args(configs="not_a_real_configuration")
    try:
        ablation.selected_configs(unknown_args)
    except ValueError as exc:
        assert "Unknown configurations" in str(exc)
    else:
        raise AssertionError("unknown configuration selection was accepted")
