from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "experiments" / "check_tracked_notebook_outputs.py"
SPEC = importlib.util.spec_from_file_location("tracked_notebook_output_check", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def write_notebook(path: Path, outputs: list[object]) -> None:
    path.write_text(
        """{
  "cells": [
    {"cell_type": "markdown", "source": ["# Title"]},
    {"cell_type": "code", "execution_count": 1, "outputs": %s, "source": ["print(1)"]}
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 5
}
"""
        % repr(outputs).replace("'", '"')
    )


def test_output_cell_indices_distinguishes_clean_and_executed_notebooks(tmp_path: Path) -> None:
    clean = tmp_path / "clean.ipynb"
    executed = tmp_path / "executed.ipynb"
    write_notebook(clean, [])
    write_notebook(executed, [{"output_type": "stream", "text": "done"}])

    assert CHECKER.output_cell_indices(clean) == []
    assert CHECKER.output_cell_indices(executed) == [1]
    assert CHECKER.notebooks_with_outputs([clean, executed]) == {executed: [1]}


def test_main_returns_nonzero_for_tracked_notebook_outputs(tmp_path: Path, monkeypatch, capsys) -> None:
    notebook = tmp_path / "executed.ipynb"
    write_notebook(notebook, [{"output_type": "display_data"}])
    monkeypatch.setattr(CHECKER, "tracked_notebook_paths", lambda _root: [notebook])

    assert CHECKER.main(["--repo-root", str(tmp_path)]) == 1
    assert "executed.ipynb" in capsys.readouterr().out
