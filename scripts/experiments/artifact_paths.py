"""Safe locations for generated experiment-review artifacts.

Generated review material and run output belong in the external artifact store,
never in the Git worktree.  The default run layout remains ``runs/<run-key>``,
including when the store is mounted at the legacy container path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


ARTIFACT_ROOT_ENV = "ARTIFACT_ROOT"
REPO_EXPERIMENTS_DIRNAME = "repo_experiments"
RUNS_DIRNAME = "runs"


class ArtifactPathError(ValueError):
    """Raised when an artifact destination violates the external-store policy."""


def repository_root() -> Path:
    """Return the repository containing this resolver."""

    return Path(__file__).resolve().parents[2]


def _contains_path(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _has_parent_component(path: Path) -> bool:
    return ".." in path.parts


def _absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ArtifactPathError(f"{label} must be an absolute path, got {value!r}.")
    if _has_parent_component(path):
        raise ArtifactPathError(f"{label} must not contain '..', got {value!r}.")
    return path.resolve(strict=False)


def _experiment_key_path(experiment_key: str | Path) -> Path:
    key_text = str(experiment_key).strip()
    key = Path(key_text)
    if not key_text or key.is_absolute() or key == Path(".") or _has_parent_component(key):
        raise ArtifactPathError(
            "experiment_key must be a non-empty relative path without traversal, "
            f"got {experiment_key!r}."
        )
    return key


def artifact_root(
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return the configured external artifact root after safety checks.

    The root must already exist.  This prevents a typo in ``ARTIFACT_ROOT``
    from silently creating a new local output tree during a launch.
    """

    environment = os.environ if environ is None else environ
    configured = str(environment.get(ARTIFACT_ROOT_ENV, "")).strip()
    if not configured:
        raise ArtifactPathError(
            f"{ARTIFACT_ROOT_ENV} is required for generated review artifacts. "
            "Set it to the mounted external artifact-store root."
        )

    root = _absolute_path(configured, label=ARTIFACT_ROOT_ENV)
    if not root.exists():
        raise ArtifactPathError(
            f"{ARTIFACT_ROOT_ENV} does not exist: {root}. Mount or create the external artifact store first."
        )
    if not root.is_dir():
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} must name a directory, got {root}.")

    source_root = (repository_root() if repo_root is None else Path(repo_root)).resolve(strict=False)
    if _contains_path(source_root, root) or _contains_path(root, source_root):
        raise ArtifactPathError(
            f"{ARTIFACT_ROOT_ENV} must be disjoint from the Git worktree; got {root}."
        )
    return root


def review_store_root(
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return the external root reserved for repository experiment artifacts."""

    root = artifact_root(environ=environ, repo_root=repo_root)
    store_root = (root / REPO_EXPERIMENTS_DIRNAME).resolve(strict=False)
    if not _contains_path(root, store_root):
        raise ArtifactPathError(
            f"{REPO_EXPERIMENTS_DIRNAME} must resolve inside {ARTIFACT_ROOT_ENV}; "
            f"got {store_root}."
        )
    return store_root


def run_store_root(
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return the external root reserved for run/checkpoint output."""

    root = artifact_root(environ=environ, repo_root=repo_root)
    store_root = (root / RUNS_DIRNAME).resolve(strict=False)
    if not _contains_path(root, store_root):
        raise ArtifactPathError(
            f"{RUNS_DIRNAME} must resolve inside {ARTIFACT_ROOT_ENV}; got {store_root}."
        )
    return store_root


def resolve_managed_path(value: str | Path, *, allowed_root: Path, label: str) -> Path:
    """Resolve an explicit artifact path that must remain within ``allowed_root``."""

    root = Path(allowed_root).resolve(strict=False)
    candidate = _absolute_path(value, label=label)
    if not _contains_path(root, candidate):
        raise ArtifactPathError(
            f"{label} must be inside the external artifact root {root}, got {candidate}."
        )
    return candidate


def _resolve_external_input(
    value: str | Path,
    *,
    label: str,
    repo_root: Path | None = None,
    kind: str,
) -> Path:
    """Resolve an existing generated input outside the Git worktree."""

    candidate = _absolute_path(value, label=label)
    source_root = (repository_root() if repo_root is None else Path(repo_root)).resolve(strict=False)
    if _contains_path(source_root, candidate):
        raise ArtifactPathError(f"{label} must be outside the Git worktree, got {candidate}.")
    if kind == "file" and not candidate.is_file():
        raise ArtifactPathError(f"{label} must name an existing file, got {candidate}.")
    if kind == "directory" and not candidate.is_dir():
        raise ArtifactPathError(f"{label} must name an existing directory, got {candidate}.")
    return candidate


def resolve_external_input_file(
    value: str | Path,
    *,
    label: str,
    repo_root: Path | None = None,
) -> Path:
    """Resolve an existing generated-input file outside the Git worktree.

    Paper builders often consume earlier generated tables or manifests.  Those
    inputs may live under ``ARTIFACT_ROOT`` or another external data store, but
    must never silently fall back to a repo-local generated file.  Requiring an
    absolute, existing file also makes reruns explicit after source-only cleanup.
    """

    return _resolve_external_input(value, label=label, repo_root=repo_root, kind="file")


def resolve_external_input_directory(
    value: str | Path,
    *,
    label: str,
    repo_root: Path | None = None,
) -> Path:
    """Resolve an existing generated-input directory outside the Git worktree."""

    return _resolve_external_input(value, label=label, repo_root=repo_root, kind="directory")


def resolve_review_root(
    experiment_key: str | Path,
    requested_review_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a review destination under ``$ARTIFACT_ROOT/repo_experiments``.

    With no explicit ``--review-root``, the output is placed at
    ``$ARTIFACT_ROOT/repo_experiments/<experiment_key>``.  Explicit paths stay
    supported for compatibility, but must be absolute, traversal-free, and
    contained by that same review store.
    """

    key = _experiment_key_path(experiment_key)
    store_root = review_store_root(environ=environ, repo_root=repo_root)
    if requested_review_root is None:
        candidate = (store_root / key).resolve(strict=False)
        if not _contains_path(store_root, candidate) or candidate == store_root:
            raise ArtifactPathError(
                "The default review root must resolve inside the external review store; "
                f"got {candidate}."
            )
        return candidate

    candidate = resolve_managed_path(
        requested_review_root,
        allowed_root=store_root,
        label="--review-root",
    )
    if candidate == store_root:
        raise ArtifactPathError(
            f"--review-root must name an experiment directory below {store_root}, not the shared store root."
        )
    return candidate


def resolve_run_root(
    run_key: str | Path,
    requested_run_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a run destination under ``$ARTIFACT_ROOT/runs``.

    A runtime with ``ARTIFACT_ROOT=/database/inrorspline`` therefore retains
    the historic ``/database/inrorspline/runs/<run-key>`` layout.  Other
    mounted locations use the same layout beneath their configured root.
    """

    key = _experiment_key_path(run_key)
    store_root = run_store_root(environ=environ, repo_root=repo_root)
    if requested_run_root is None:
        candidate = (store_root / key).resolve(strict=False)
        if not _contains_path(store_root, candidate) or candidate == store_root:
            raise ArtifactPathError(
                "The default run root must resolve inside the external run store; "
                f"got {candidate}."
            )
        return candidate

    candidate = resolve_managed_path(
        requested_run_root,
        allowed_root=store_root,
        label="--out-root",
    )
    if candidate == store_root:
        raise ArtifactPathError(
            f"--out-root must name a run directory below {store_root}, not the shared run root."
        )
    return candidate
