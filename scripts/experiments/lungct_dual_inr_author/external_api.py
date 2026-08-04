"""Load the pinned DUAL-INR-DIR implementation without editable-install leakage."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


AUTHOR_EXTERNAL_COMMIT = "f7c8e833435ec9c6b7ebec6c9a8098b566709146"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def external_repository() -> Path:
    return repository_root() / "external" / "DUAL-INR-DIR"


def git_directory(repository: Path) -> Path:
    """Resolve a normal or submodule ``.git`` location without invoking Git."""

    repository = repository.resolve()
    dot_git = repository / ".git"
    if dot_git.is_dir():
        return dot_git.resolve()
    if dot_git.is_file():
        content = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not content.startswith(prefix):
            raise RuntimeError(f"Malformed submodule gitdir pointer: {dot_git}")
        resolved = (repository / content[len(prefix) :]).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Submodule gitdir does not exist: {resolved}")
        return resolved
    raise FileNotFoundError(f"Git metadata is missing below {repository}")


def _module_is_below(module: ModuleType, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def external_commit(repo: Path | None = None) -> str:
    repo = (repo or external_repository()).resolve()
    result = subprocess.run(
        [
            "git",
            f"--git-dir={git_directory(repo)}",
            f"--work-tree={repo}",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_external_api(*, require_pinned_commit: bool = True) -> SimpleNamespace:
    """Import ``inrdir`` from the audited external checkout and verify its origin.

    A pre-imported package is accepted only when it came from the same checkout.
    The external ``src`` directory is removed from ``sys.path`` immediately after
    importing; loaded modules retain their normal references in ``sys.modules``.
    """

    repo = external_repository().resolve()
    src = (repo / "src").resolve()
    package_dir = src / "inrdir"
    if not (package_dir / "trainer.py").is_file():
        raise FileNotFoundError(
            f"Expected DUAL-INR-DIR package is missing at {package_dir}"
        )

    commit = external_commit(repo)
    if require_pinned_commit and commit != AUTHOR_EXTERNAL_COMMIT:
        raise RuntimeError(
            "DUAL-INR-DIR checkout changed since the author baseline audit: "
            f"expected {AUTHOR_EXTERNAL_COMMIT}, found {commit}"
        )

    existing = sys.modules.get("inrdir")
    if existing is not None and not _module_is_below(existing, package_dir):
        raise RuntimeError(
            "Refusing to reuse an inrdir package imported outside the audited "
            f"checkout: {getattr(existing, '__file__', '<unknown>')}"
        )

    inserted = False
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
        inserted = True
    try:
        modules = {
            name: importlib.import_module(f"inrdir.{name}")
            for name in (
                "configs",
                "model",
                "ncc",
                "regularizers",
                "scheduler",
                "trainer",
                "utils",
            )
        }
        package = importlib.import_module("inrdir")
    finally:
        if inserted:
            try:
                sys.path.remove(src_text)
            except ValueError:
                pass

    for name, module in {"package": package, **modules}.items():
        if not _module_is_below(module, package_dir):
            raise RuntimeError(
                f"Imported inrdir {name} from an unexpected location: "
                f"{getattr(module, '__file__', '<unknown>')}"
            )

    return SimpleNamespace(
        commit=commit,
        repository=repo,
        package_dir=package_dir,
        **modules,
    )
