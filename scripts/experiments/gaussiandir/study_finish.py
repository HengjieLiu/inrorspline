"""Wait for the frozen timing queue, then score, report and execute external review."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import atomic_json, external, sha256
from .model import REPO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = external(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/"finish_progress.json", dict(status="waiting_for_timed_queue"))
    began = time.monotonic()
    while not (args.run_root/"completed.json").exists():
        failures = list(args.run_root.glob("**/failure.json"))
        if failures or (args.run_root/"blocked.json").exists() or time.monotonic()-began > 7200:
            raise RuntimeError(f"Timing queue failed/blocked or exceeded wait allowance: {failures}")
        time.sleep(10)
    commands = [
        [sys.executable, "-B", "-m", "scripts.experiments.gaussiandir.study_evaluate",
         "--config", str(args.config), "--run-root", str(args.run_root), "--out", str(out/"scoring"), "--workers", "4"],
        [sys.executable, "-B", "-m", "scripts.experiments.gaussiandir.study_report",
         "--review", str(out/"scoring"), "--references", str(args.references), "--out", str(out/"report_v1")]
    ]
    for index, command in enumerate(commands):
        atomic_json(out/"finish_progress.json", dict(status="scoring" if index == 0 else "reporting"))
        with (out/f"stage_{index}.log").open("w") as log:
            try:
                subprocess.run(command, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                atomic_json(out/"finish_progress.json", dict(status="failed", stage=index,
                    error=repr(exc), log=str(out/f"stage_{index}.log")))
                raise
    atomic_json(out/"finish_progress.json", dict(status="executing_external_notebook"))
    import nbformat
    from nbclient import NotebookClient
    source = REPO/"experiments/2026-11-06_oasis_rank000_gaussiandir/convergence_review.ipynb"
    notebook = nbformat.read(source, as_version=4)
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.cells):
        raise RuntimeError("Notebook source contains outputs")
    report = out/"report_v1"
    os.environ["GAUSSIANDIR_CONVERGENCE_REVIEW"] = str(report)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata":{"path":str(report)}}).execute()
    except Exception as exc:
        atomic_json(out/"finish_progress.json", dict(status="failed", stage="external_notebook",
            error=repr(exc)))
        raise
    nbformat.write(notebook, report/"review_executed.ipynb")
    errors = [o for c in notebook.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    images = sum("image/png" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", []))
    if errors or images != 4:
        raise RuntimeError("Notebook review lacks the required inline figures or contains errors")
    atomic_json(report/"notebook_receipt.json", dict(source_sha256=sha256(source),
        executed_sha256=sha256(report/"review_executed.ipynb"), errors=len(errors), inline_pngs=images))
    atomic_json(out/"finish_progress.json", dict(status="complete", report=str(report/"report.md")))
    print(report/"report.md", flush=True)


if __name__ == "__main__":
    main()
