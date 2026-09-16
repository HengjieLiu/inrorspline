from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import artifact_paths  # noqa: E402


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RESULT_SELECTOR = load_module(
    "inrorspline_result_table_builder",
    "experiments/2026-06-26_auto_result_selection_tables/build_result_tables.py",
)
PUBLICATION_TABLES = load_module(
    "inrorspline_publication_table_builder",
    "experiments/paper_artifacts/best_result_table/build_publication_tables.py",
)


def configured_store(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    store_root = tmp_path / "artifact-store"
    review_root = store_root / "repo_experiments"
    review_root.mkdir(parents=True)
    return store_root, review_root, {"ARTIFACT_ROOT": str(store_root)}


def test_result_selector_requires_external_root_and_explicit_external_inputs(tmp_path: Path) -> None:
    store_root, review_root, environ = configured_store(tmp_path)
    oasis_source = review_root / "migration_2026-09-02" / "oasis.csv"
    lung_mean_source = review_root / "migration_2026-09-02" / "lung_mean.csv"
    lung_no_sweep_source = review_root / "migration_2026-09-02" / "lung_case_rows.csv"
    for path in (oasis_source, lung_mean_source, lung_no_sweep_source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n")

    assert RESULT_SELECTOR.resolve_output_dir(None, environ=environ) == (
        review_root / "2026-06-26_auto_result_selection_tables"
    )
    assert RESULT_SELECTOR.resolve_external_review_file(
        oasis_source,
        label="--oasis-source",
        environ=environ,
    ) == oasis_source
    assert RESULT_SELECTOR.public_path(oasis_source, artifact_root=store_root) == (
        "${ARTIFACT_ROOT}/repo_experiments/migration_2026-09-02/oasis.csv"
    )

    with pytest.raises(artifact_paths.ArtifactPathError, match="ARTIFACT_ROOT is required"):
        RESULT_SELECTOR.resolve_output_dir(None, environ={})
    with pytest.raises(artifact_paths.ArtifactPathError):
        RESULT_SELECTOR.resolve_output_dir(REPO_ROOT / "experiments" / "local-output", environ=environ)
    with pytest.raises(artifact_paths.ArtifactPathError):
        RESULT_SELECTOR.resolve_external_review_file(
            REPO_ROOT / "experiments" / "2026-06-26_auto_result_selection_tables" / "table_specs.json",
            label="--oasis-source",
            environ=environ,
        )

    with pytest.raises(SystemExit):
        RESULT_SELECTOR.parse_args([])
    parsed = RESULT_SELECTOR.parse_args(
        [
            "--oasis-source",
            str(oasis_source),
            "--lung-mean-source",
            str(lung_mean_source),
            "--lung-no-sweep-source",
            str(lung_no_sweep_source),
        ]
    )
    assert parsed.output_dir is None


def test_publication_table_builder_requires_external_source_and_output(tmp_path: Path) -> None:
    store_root, review_root, environ = configured_store(tmp_path)
    source_dir = review_root / "migration_2026-09-02" / "selected_rows"
    source_dir.mkdir(parents=True)

    assert PUBLICATION_TABLES.resolve_output_dir(None, environ=environ) == (
        review_root / "paper_artifacts" / "best_result_table" / "v002_publication"
    )
    assert PUBLICATION_TABLES.resolve_external_review_dir(
        source_dir,
        label="--source-dir",
        environ=environ,
    ) == source_dir
    assert PUBLICATION_TABLES.public_path(source_dir, artifact_root=store_root) == (
        "${ARTIFACT_ROOT}/repo_experiments/migration_2026-09-02/selected_rows"
    )

    with pytest.raises(artifact_paths.ArtifactPathError, match="ARTIFACT_ROOT is required"):
        PUBLICATION_TABLES.resolve_output_dir(None, environ={})
    with pytest.raises(artifact_paths.ArtifactPathError):
        PUBLICATION_TABLES.resolve_output_dir(REPO_ROOT / "experiments" / "local-output", environ=environ)
    with pytest.raises(artifact_paths.ArtifactPathError):
        PUBLICATION_TABLES.resolve_external_review_dir(
            REPO_ROOT / "experiments" / "paper_artifacts" / "best_result_table",
            label="--source-dir",
            environ=environ,
        )

    with pytest.raises(SystemExit):
        PUBLICATION_TABLES.parse_args([])
    parsed = PUBLICATION_TABLES.parse_args(["--source-dir", str(source_dir)])
    assert parsed.output_dir is None
