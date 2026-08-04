#!/usr/bin/env python3
"""Run one controlled COPD DUAL-INR N0/N1/P0/P1 record."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.experiments.lungct_dual_inr_controlled import run_one as core
from scripts.experiments.lungct_dual_inr_controlled_copd import data


def source_bundle() -> dict[str, object]:
    root = core.repository_root().resolve()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("data.py").resolve(),
        root / "scripts/experiments/lungct_dual_inr_controlled/run_one.py",
        root / "scripts/experiments/lungct_dual_inr_controlled/adapter.py",
        root / "scripts/experiments/lungct_dual_inr_author/external_api.py",
    )
    digest = hashlib.sha256()
    records = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\\0")
        digest.update(content)
        digest.update(b"\\0")
        records.append({"repository_relative_path": relative, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
    return {"algorithm": "sha256", "bundle_sha256": digest.hexdigest(), "sources": records}


def run_case(args):
    core.load_controlled_case = data.load_controlled_case
    core.evaluate_controlled_metrics = data.evaluate_controlled_metrics
    core.source_bundle = source_bundle
    return core.run_case(args)


def main() -> None:
    args = core.build_parser().parse_args()
    if args.eval_chunk_size <= 0:
        raise ValueError("--eval-chunk-size must be positive")
    run_case(args)


if __name__ == "__main__":
    main()
