#!/usr/bin/env python
"""Fail when Git-tracked notebooks retain embedded cell outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracked_notebook_paths(repo_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.ipynb"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [repo_root / Path(item.decode()) for item in completed.stdout.split(b"\0") if item]


def output_cell_indices(notebook_path: Path) -> list[int]:
    payload = json.loads(notebook_path.read_text())
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook has no cell list: {notebook_path}")
    return [
        index
        for index, cell in enumerate(cells)
        if isinstance(cell, dict) and cell.get("cell_type") == "code" and cell.get("outputs")
    ]


def notebooks_with_outputs(paths: Sequence[Path]) -> dict[Path, list[int]]:
    violations: dict[Path, list[int]] = {}
    for path in paths:
        indices = output_cell_indices(path)
        if indices:
            violations[path] = indices
    return violations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    violations = notebooks_with_outputs(tracked_notebook_paths(root))
    if not violations:
        print("No Git-tracked notebooks contain cell outputs.")
        return 0

    print("Git-tracked notebooks contain cell outputs:")
    for path, indices in sorted(violations.items()):
        print(f"- {path.relative_to(root)}: code cells {','.join(str(index) for index in indices)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
