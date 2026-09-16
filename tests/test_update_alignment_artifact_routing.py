from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
UPDATE_ALIGNMENT_DIR = EXPERIMENTS_DIR / "update_alignment"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(UPDATE_ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(UPDATE_ALIGNMENT_DIR))

import artifact_paths  # noqa: E402


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KERNEL_FOLLOWUP = load_module(
    "inrorspline_kernel_followup_routing",
    "scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py",
)
MECHANISM_EXPLORE = load_module(
    "inrorspline_mechanism_explore_routing",
    "scripts/experiments/update_alignment/explore_mechanism_followup.py",
)
STANDARD_PYRAMID = load_module(
    "inrorspline_standard_pyramid_routing",
    "scripts/experiments/update_alignment/run_standard_pyramid_probe.py",
)
FIREANTS_AUDIT = load_module(
    "inrorspline_fireants_two_window_routing",
    "scripts/experiments/learn2breath_fireants_greedy/run_all_two_window_audit.py",
)


def configured_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store_root = tmp_path / "artifact-store"
    store_root.mkdir()
    monkeypatch.setenv("ARTIFACT_ROOT", str(store_root))
    return store_root


@pytest.mark.parametrize(
    ("module", "run_key"),
    [
        (KERNEL_FOLLOWUP, "sinr_cp_kernel_followup_20260629_v1"),
        (MECHANISM_EXPLORE, "sinr_cp_mechanism_explore_20260629_v1"),
        (STANDARD_PYRAMID, "sinr_cp_mechanism_explore_20260629_v1"),
    ],
)
def test_p6_writers_default_to_external_run_and_review_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    run_key: str,
) -> None:
    store_root = configured_store(tmp_path, monkeypatch)
    args = SimpleNamespace(out_root=None, review_root=None)

    resolved = module.resolve_artifact_paths(args)

    assert resolved.out_root == store_root / "runs" / run_key
    assert resolved.review_root == store_root / "repo_experiments" / "2026-06-29_sinr_cp_kernel_followup"


@pytest.mark.parametrize(
    "module",
    [KERNEL_FOLLOWUP, MECHANISM_EXPLORE, STANDARD_PYRAMID],
)
def test_p6_writers_reject_repo_local_or_wrong_root_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    store_root = configured_store(tmp_path, monkeypatch)

    with pytest.raises(artifact_paths.ArtifactPathError):
        module.resolve_artifact_paths(
            SimpleNamespace(out_root=REPO_ROOT / "runs" / "local", review_root=None)
        )
    with pytest.raises(artifact_paths.ArtifactPathError):
        module.resolve_artifact_paths(
            SimpleNamespace(
                out_root=store_root / "runs" / "p6",
                review_root=store_root / "runs" / "not-a-review-root",
            )
        )
    with pytest.raises(artifact_paths.ArtifactPathError):
        module.resolve_artifact_paths(
            SimpleNamespace(
                out_root=store_root / "runs" / "p6" / ".." / "escape",
                review_root=None,
            )
        )


def test_p6_monitor_delegates_output_path_validation_to_the_controller() -> None:
    source = (
        REPO_ROOT / "scripts/experiments/update_alignment/monitor_mechanism_explore.py"
    ).read_text()

    assert 'parser.add_argument("--out-root", type=Path, default=None)' in source
    assert 'parser.add_argument("--review-root", type=Path, default=None)' in source
    assert "args = explore.resolve_artifact_paths(parser.parse_args())" in source


def test_fireants_audit_requires_an_explicit_external_output_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = configured_store(tmp_path, monkeypatch)
    requested = store_root / "runs" / "learn2breath_fireants_audit"

    resolved = FIREANTS_AUDIT.resolve_artifact_paths(SimpleNamespace(out_root=requested))

    assert resolved.out_root == requested
    for unsafe in (REPO_ROOT / "experiments" / "local-output", Path("relative-output"), store_root):
        with pytest.raises(artifact_paths.ArtifactPathError):
            FIREANTS_AUDIT.resolve_artifact_paths(SimpleNamespace(out_root=unsafe))
