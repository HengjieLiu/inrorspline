from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


def configured_store(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store_root = tmp_path / "artifact-store"
    store_root.mkdir()
    return repo_root, store_root, {"ARTIFACT_ROOT": str(store_root)}


def test_default_review_root_uses_external_repo_experiments_store(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)

    resolved = artifact_paths.resolve_review_root(
        "2026-09-01_example",
        environ=environ,
        repo_root=repo_root,
    )

    assert resolved == store_root / "repo_experiments" / "2026-09-01_example"
    assert not resolved.exists()


def test_default_run_root_uses_external_runs_store(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)

    resolved = artifact_paths.resolve_run_root(
        "example_run_v1",
        environ=environ,
        repo_root=repo_root,
    )

    assert resolved == store_root / "runs" / "example_run_v1"
    assert not resolved.exists()


def test_explicit_run_root_must_stay_within_external_runs_store(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)
    requested = store_root / "runs" / "manual_recheck"

    assert artifact_paths.resolve_run_root(
        "example_run_v1",
        requested,
        environ=environ,
        repo_root=repo_root,
    ) == requested


def test_explicit_review_root_must_stay_within_external_review_store(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)
    requested = store_root / "repo_experiments" / "manual_recheck" / "review"

    resolved = artifact_paths.resolve_review_root(
        "2026-09-01_example",
        requested,
        environ=environ,
        repo_root=repo_root,
    )

    assert resolved == requested


def test_artifact_root_is_required_and_must_be_an_existing_external_directory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(artifact_paths.ArtifactPathError, match="ARTIFACT_ROOT is required"):
        artifact_paths.resolve_review_root("example", environ={}, repo_root=repo_root)

    with pytest.raises(artifact_paths.ArtifactPathError, match="absolute path"):
        artifact_paths.resolve_review_root("example", environ={"ARTIFACT_ROOT": "relative"}, repo_root=repo_root)

    with pytest.raises(artifact_paths.ArtifactPathError, match="does not exist"):
        artifact_paths.resolve_review_root(
            "example",
            environ={"ARTIFACT_ROOT": str(tmp_path / "missing-store")},
            repo_root=repo_root,
        )

    repo_local_store = repo_root / "artifacts"
    repo_local_store.mkdir()
    with pytest.raises(artifact_paths.ArtifactPathError, match="disjoint from the Git worktree"):
        artifact_paths.resolve_review_root(
            "example",
            environ={"ARTIFACT_ROOT": str(repo_local_store)},
            repo_root=repo_root,
        )

    with pytest.raises(artifact_paths.ArtifactPathError, match="disjoint from the Git worktree"):
        artifact_paths.resolve_review_root(
            "example",
            environ={"ARTIFACT_ROOT": str(tmp_path)},
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("experiment_key", "requested_review_root", "message"),
    [
        ("../escape", None, "experiment_key"),
        ("", None, "experiment_key"),
        ("example", "relative/review", "absolute path"),
        ("example", "repo-local", "absolute path"),
    ],
)
def test_resolver_rejects_relative_and_traversal_inputs(
    tmp_path: Path,
    experiment_key: str,
    requested_review_root: str | None,
    message: str,
) -> None:
    repo_root, _store_root, environ = configured_store(tmp_path)

    with pytest.raises(artifact_paths.ArtifactPathError, match=message):
        artifact_paths.resolve_review_root(
            experiment_key,
            requested_review_root,
            environ=environ,
            repo_root=repo_root,
        )


def test_resolver_rejects_repo_local_wrong_root_and_parent_traversal(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)
    repo_local = repo_root / "experiments" / "example"
    wrong_external_root = tmp_path / "another-store" / "review"
    traversing_external_root = store_root / "repo_experiments" / "example" / ".." / "other"

    for requested in (repo_local, wrong_external_root, traversing_external_root):
        with pytest.raises(artifact_paths.ArtifactPathError):
            artifact_paths.resolve_review_root(
                "example",
                requested,
                environ=environ,
                repo_root=repo_root,
            )


def test_resolver_rejects_review_store_symlink_that_escapes_artifact_root(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)
    repo_local_target = repo_root / "experiments"
    repo_local_target.mkdir()
    (store_root / "repo_experiments").symlink_to(repo_local_target, target_is_directory=True)

    with pytest.raises(artifact_paths.ArtifactPathError, match="must resolve inside ARTIFACT_ROOT"):
        artifact_paths.resolve_review_root("example", environ=environ, repo_root=repo_root)


def test_resolver_rejects_experiment_key_symlink_that_escapes_review_store(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)
    review_store = store_root / "repo_experiments"
    review_store.mkdir()
    repo_local_target = repo_root / "experiments"
    repo_local_target.mkdir()
    (review_store / "example").symlink_to(repo_local_target, target_is_directory=True)

    with pytest.raises(artifact_paths.ArtifactPathError, match="default review root"):
        artifact_paths.resolve_review_root("example", environ=environ, repo_root=repo_root)


def test_run_resolver_rejects_wrong_root_and_symlink_escape(tmp_path: Path) -> None:
    repo_root, store_root, environ = configured_store(tmp_path)

    with pytest.raises(artifact_paths.ArtifactPathError):
        artifact_paths.resolve_run_root(
            "example",
            repo_root / "runs" / "example",
            environ=environ,
            repo_root=repo_root,
        )

    run_store = store_root / "runs"
    run_store.mkdir()
    escaped_target = tmp_path / "escaped"
    escaped_target.mkdir()
    (run_store / "example").symlink_to(escaped_target, target_is_directory=True)
    with pytest.raises(artifact_paths.ArtifactPathError, match="default run root"):
        artifact_paths.resolve_run_root("example", environ=environ, repo_root=repo_root)


def test_log_or_other_managed_paths_cannot_escape_review_root(tmp_path: Path) -> None:
    repo_root, _store_root, environ = configured_store(tmp_path)
    review_root = artifact_paths.resolve_review_root("example", environ=environ, repo_root=repo_root)

    assert artifact_paths.resolve_managed_path(
        review_root / "logs",
        allowed_root=review_root,
        label="--log-root",
    ) == review_root / "logs"

    with pytest.raises(artifact_paths.ArtifactPathError, match="inside the external artifact root"):
        artifact_paths.resolve_managed_path(
            repo_root / "logs",
            allowed_root=review_root,
            label="--log-root",
        )


def test_external_input_file_requires_an_existing_absolute_file_outside_the_repo(tmp_path: Path) -> None:
    repo_root, store_root, _environ = configured_store(tmp_path)
    external_file = store_root / "repo_experiments" / "prior" / "rows.csv"
    external_file.parent.mkdir(parents=True)
    external_file.write_text("value\n1\n")

    assert artifact_paths.resolve_external_input_file(
        external_file,
        label="--source-rows",
        repo_root=repo_root,
    ) == external_file

    with pytest.raises(artifact_paths.ArtifactPathError, match="absolute path"):
        artifact_paths.resolve_external_input_file(
            "relative.csv",
            label="--source-rows",
            repo_root=repo_root,
        )

    repo_file = repo_root / "experiments" / "rows.csv"
    repo_file.parent.mkdir(parents=True)
    repo_file.write_text("value\n1\n")
    with pytest.raises(artifact_paths.ArtifactPathError, match="outside the Git worktree"):
        artifact_paths.resolve_external_input_file(
            repo_file,
            label="--source-rows",
            repo_root=repo_root,
        )

    with pytest.raises(artifact_paths.ArtifactPathError, match="existing file"):
        artifact_paths.resolve_external_input_file(
            store_root / "missing.csv",
            label="--source-rows",
            repo_root=repo_root,
        )


def test_external_input_directory_requires_a_directory_outside_the_repo(tmp_path: Path) -> None:
    repo_root, store_root, _environ = configured_store(tmp_path)
    external_directory = store_root / "ptvreg"
    external_directory.mkdir()

    assert artifact_paths.resolve_external_input_directory(
        external_directory,
        label="--ptvreg-root",
        repo_root=repo_root,
    ) == external_directory

    with pytest.raises(artifact_paths.ArtifactPathError, match="existing directory"):
        artifact_paths.resolve_external_input_directory(
            store_root / "missing",
            label="--ptvreg-root",
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py",
        "scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_acceleration_timing.py",
        "scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py",
        "scripts/experiments/ptvreg_python_repro/aggregate_copd_fixed_all10.py",
    ],
)
def test_current_review_producers_keep_the_flag_and_resolve_it_externally(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text()

    assert 'parser.add_argument("--review-root", type=Path, default=None)' in source
    assert 'parser.add_argument("--out-root", type=Path, default=None)' in source
    assert "artifact_paths.resolve_review_root(" in source
    assert "resolve_run_root(" in source
    assert "resolve_artifact_paths(parse_args())" in source


def test_ptvreg_worker_fallback_requires_the_external_run_resolver() -> None:
    source = (REPO_ROOT / "scripts/experiments/ptvreg_python_repro/run_copd_case.py").read_text()

    assert "def resolve_run_root(" in source
    assert "artifact_paths.resolve_run_root(" in source
    assert "args.out_root = resolve_run_root(args.out_root)" in source
    assert "args.paper_hu_root = resolve_paper_hu_root" in source
    assert "--audit-output" in source
    assert "--synthetic-smoke-output" in source
    assert "--prepare-output" in source
