"""Guard external-only destinations for the active P2 launch/aggregate scripts."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "output_flags", "resolver_markers", "main_marker"),
    [
        (
            "scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py",
            ("--out-root", "--review-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/oasis_alpha_sweep/aggregate_oasis_unified_acceleration_sweep.py",
            ("--out-root", "--review-root"),
            ("resolve_artifact_paths,",),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/oasis_alpha_sweep/launch_oasis_selected_fast_first100.py",
            ("--out-root", "--review-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/oasis_alpha_sweep/aggregate_oasis_selected_fast_first100.py",
            ("--out-root", "--review-root"),
            ("resolve_artifact_paths,",),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/oasis_alpha_sweep/run_oasis_paper_cold_warm_timing.py",
            ("--out-root", "--review-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/oasis_alpha_sweep/analyze_oasis_paper_cold_warm_timing.py",
            ("--out-root", "--review-root"),
            ("runner.resolve_artifact_paths(parse_args())",),
            "args = runner.resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py",
            ("--idir-out-root", "--ptv-out-root", "--log-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_fig2_mrdbscp_2500_schedule_refresh.py",
            ("--idir-run-root", "--ptv-run-root", "--review-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py",
            ("--ptv-out-root", "--masked-out-root", "--log-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_2500_schedule_pilots.py",
            ("--ptv-run-root", "--masked-run-root", "--ptv-review-root", "--masked-review-root", "--decision-root"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py",
            ("--run-root", "--review-dir"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "return resolve_artifact_paths(args)",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_extension_all10.py",
            ("--run-root", "--review-dir"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "args = resolve_artifact_paths(parse_args())",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_landscape_all10.py",
            ("--run-root", "--review-dir"),
            ("artifact_paths.resolve_run_root(", "artifact_paths.resolve_review_root("),
            "return resolve_artifact_paths(args)",
        ),
        (
            "scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_landscape_all10.py",
            (),
            ("base.artifact_paths.resolve_run_root(", "base.artifact_paths.resolve_review_root("),
            "base.resolve_artifact_paths = resolve_artifact_paths",
        ),
    ],
)
def test_current_p2_producers_resolve_all_generated_destinations_externally(
    relative_path: str,
    output_flags: tuple[str, ...],
    resolver_markers: tuple[str, ...],
    main_marker: str,
) -> None:
    source = (REPO_ROOT / relative_path).read_text()

    for flag in output_flags:
        assert f'parser.add_argument("{flag}", type=Path, default=None)' in source
    for marker in resolver_markers:
        assert marker in source
    assert main_marker in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py",
        "scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py",
        "scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py",
        "scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_landscape_all10.py",
    ],
)
def test_current_lungct_launchers_do_not_keep_repo_local_generated_root_defaults(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text()

    assert "DEFAULT_LOG_ROOT = c.REPO_ROOT" not in source
    assert "DEFAULT_REVIEW_DIR = REPO_ROOT / \"experiments" not in source
    assert "DEFAULT_REVIEW_DIR = base.REPO_ROOT / \"experiments" not in source
