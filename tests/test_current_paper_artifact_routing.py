from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "required_inputs", "input_resolver"),
    [
        (
            "experiments/paper_artifacts/pareto_figure/build_current_tre_stdlogj_fdbe_landscape.py",
            (
                "--oasis-source",
                "--lung-schedule-source",
                "--legacy-w0022-source",
                "--dual-stage2-source",
                "--dual-stage3-source",
                "--landscape-case-rows",
                "--accepted-extension-case-rows",
                "--legacy-v011-case-rows",
            ),
            "resolve_external_input_file",
        ),
        (
            "experiments/paper_artifacts/pareto_figure/build_final_dirlab4dct_axes_labels.py",
            ("--oasis-source", "--source-rows"),
            "resolve_external_input_file",
        ),
        (
            "experiments/paper_artifacts/motion_pattern_figure/build_motion_pattern_figure.py",
            ("--sitreg-root", "--ptvreg-root"),
            "resolve_external_input_directory",
        ),
    ],
)
def test_current_paper_artifact_producers_require_external_routing(
    relative_path: str,
    required_inputs: tuple[str, ...],
    input_resolver: str,
) -> None:
    source = (REPO_ROOT / relative_path).read_text()

    assert 'parser.add_argument("--review-root", type=Path, default=None)' in source
    assert "artifact_paths.resolve_review_root(" in source
    assert f"artifact_paths.{input_resolver}(" in source
    assert "resolve_artifact_paths(parse_args(" in source
    assert 'parser.add_argument("--output-dir"' not in source
    for option in required_inputs:
        assert f'parser.add_argument("{option}", type=Path, required=True)' in source


def test_v012_v013_manifests_use_anonymous_safe_public_paths() -> None:
    base_source = (
        REPO_ROOT / "experiments" / "paper_artifacts" / "pareto_figure" / "build_current_tre_stdlogj_expanded.py"
    ).read_text()
    final_source = (
        REPO_ROOT / "experiments" / "paper_artifacts" / "pareto_figure" / "build_final_dirlab4dct_axes_labels.py"
    ).read_text()

    assert "return pareto_base.repo_relative(path)" in base_source
    assert "base.repo_relative(oasis_source)" in final_source
    assert "base.repo_relative(source_rows)" in final_source
