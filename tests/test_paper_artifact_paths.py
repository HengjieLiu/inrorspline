from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ARTIFACTS_DIR = REPO_ROOT / "experiments" / "paper_artifacts"
if str(PAPER_ARTIFACTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_ARTIFACTS_DIR))

import paper_artifact_common as common  # noqa: E402


def standard() -> dict[str, object]:
    return {
        "artifact_versions": {
            "pareto_figure": {
                "current": "v003_candidate_oasis100",
                # Historical identity metadata must not become a write path.
                "output_dir": "experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100",
            }
        }
    }


def configured_store(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    store_root = tmp_path / "artifact-store"
    store_root.mkdir()
    return store_root, {"ARTIFACT_ROOT": str(store_root)}


def test_paper_artifact_default_output_is_external(tmp_path: Path) -> None:
    store_root, environ = configured_store(tmp_path)

    resolved = common.artifact_output_dir("pareto_figure", standard(), environ=environ)

    assert resolved == (
        store_root
        / "repo_experiments"
        / "paper_artifacts"
        / "pareto_figure"
        / "v003_candidate_oasis100"
    )


def test_paper_artifact_output_rejects_missing_or_repo_local_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ARTIFACT_ROOT is required"):
        common.artifact_output_dir("pareto_figure", standard(), environ={})

    _store_root, environ = configured_store(tmp_path)
    with pytest.raises(ValueError, match="inside the external artifact root"):
        common.artifact_output_dir(
            "pareto_figure",
            standard(),
            REPO_ROOT / "experiments" / "paper_artifacts" / "manual",
            environ=environ,
        )


def test_paper_artifact_explicit_external_output_is_allowed(tmp_path: Path) -> None:
    store_root, environ = configured_store(tmp_path)
    explicit = store_root / "repo_experiments" / "paper_artifacts" / "manual"

    assert common.artifact_output_dir(
        "pareto_figure",
        standard(),
        explicit,
        environ=environ,
    ) == explicit


def test_pareto_builder_exposes_external_review_root() -> None:
    source = (REPO_ROOT / "experiments" / "paper_artifacts" / "build_standardized_pareto.py").read_text()

    assert 'parser.add_argument("--review-root", type=Path, default=None)' in source
    for option in ("--oasis-source", "--lung-mean-source", "--lung-no-sweep-source"):
        assert f'parser.add_argument("{option}", type=Path, required=True)' in source
    assert source.count("artifact_paths.resolve_external_input_file(") >= 3
    assert 'artifact_output_dir(\n        "pareto_figure",\n        standard,\n        args.review_root,' in source


def test_public_manifest_paths_symbolize_the_artifact_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root, _environ = configured_store(tmp_path)
    artifact_file = store_root / "repo_experiments" / "paper_artifacts" / "rows.csv"
    external_file = tmp_path / "private-host" / "rows.csv"
    monkeypatch.setenv("ARTIFACT_ROOT", str(store_root))

    assert common.repo_relative(artifact_file) == "${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/rows.csv"
    assert common.repo_relative(external_file) == "${EXTERNAL_INPUT}/rows.csv"
