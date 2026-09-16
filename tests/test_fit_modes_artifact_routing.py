from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "2026-06-26_fit_modes_paper_figure"
    / "fit_modes_method_comparison.py"
)
NOTEBOOK_PATH = MODULE_PATH.with_suffix(".ipynb")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_module():
    spec = importlib.util.spec_from_file_location("inrorspline_fit_modes_routing", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT_MODES = load_module()


class RecordingFigure:
    """A minimal figure stand-in that records only the requested save paths."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def savefig(self, path: str | Path, **_kwargs: object) -> None:
        path = Path(path)
        path.write_text("test figure\n")
        self.saved_paths.append(path)


def configured_store(tmp_path: Path) -> tuple[Path, Path]:
    store_root = tmp_path / "artifact-store"
    review_root = store_root / "repo_experiments"
    review_root.mkdir(parents=True)
    return store_root, review_root


def test_save_helpers_default_to_the_external_fitting_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root, review_root = configured_store(tmp_path)
    monkeypatch.setenv("ARTIFACT_ROOT", str(store_root))
    expected_dir = review_root / "paper_artifacts" / "fitting_figure" / "v002_candidate"

    figure = RecordingFigure()
    outputs = FIT_MODES.save_main_figure(figure, stem="routing-smoke")
    relative_outputs = FIT_MODES.save_relative_field_fit_figure(figure, stem="relative-routing-smoke")
    trajectory_outputs = FIT_MODES.save_rrmse_lr_trajectory_figure(figure, stem="trajectory-routing-smoke")

    assert FIT_MODES.resolve_fitting_output_dir() == expected_dir
    for path in [*outputs.values(), *relative_outputs.values(), *trajectory_outputs.values()]:
        assert path.is_file()
        assert path.is_relative_to(expected_dir)
        assert not path.is_relative_to(REPO_ROOT)
    assert set(figure.saved_paths) == {
        *outputs.values(),
        *relative_outputs.values(),
        *trajectory_outputs.values(),
    }


def test_fitting_output_rejects_missing_and_worktree_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    with pytest.raises(ValueError, match="ARTIFACT_ROOT is required"):
        FIT_MODES.resolve_fitting_output_dir()

    store_root, _review_root = configured_store(tmp_path)
    monkeypatch.setenv("ARTIFACT_ROOT", str(store_root))
    with pytest.raises(ValueError, match="inside the external artifact root"):
        FIT_MODES.save_main_figure(
            RecordingFigure(),
            review_root=REPO_ROOT / "experiments" / "local-fitting-output",
        )
    with pytest.raises(ValueError, match="inside the external artifact root"):
        FIT_MODES._save_figure_to_dir(  # noqa: SLF001 - assert the shared internal writer is defensive.
            RecordingFigure(),
            REPO_ROOT / "experiments" / "local-fitting-output",
            "routing-smoke",
        )


def test_public_fit_mode_builders_expose_only_compliant_review_roots() -> None:
    source = MODULE_PATH.read_text()
    for function in (
        FIT_MODES.save_main_figure,
        FIT_MODES.save_relative_field_fit_figure,
        FIT_MODES.save_rrmse_lr_trajectory_figure,
        FIT_MODES.build_outputs,
        FIT_MODES.build_relative_field_fit_outputs,
        FIT_MODES.build_rrmse_lr_trajectory_outputs,
        FIT_MODES.build_standardized_global_ncc_panelA_outputs,
        FIT_MODES.build_standardized_rrmse_field_fit_outputs,
        FIT_MODES.build_standardized_panelAB_outputs,
    ):
        assert "review_root" in inspect.signature(function).parameters

    assert "HISTORIC_OASIS_INPUT_ROOT" in source
    assert "REVIEW_ROOT" not in source
    assert "resolve_fitting_output_dir(" in source


def test_notebook_source_routes_writes_through_external_artifact_directory() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)

    assert all(not cell.get("outputs") for cell in code_cells)
    assert "ARTIFACT_OUTPUT_DIR = resolve_fitting_output_dir()" in source
    assert "to_csv(ARTIFACT_OUTPUT_DIR" in source
    assert "review_root=ARTIFACT_OUTPUT_DIR" in source
    assert "to_csv(REVIEW_ROOT" not in source
    assert "save_main_figure(" not in source
