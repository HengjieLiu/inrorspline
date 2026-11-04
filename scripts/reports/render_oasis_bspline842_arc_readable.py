#!/usr/bin/env python
"""Presentation-only companion: wrap ARC titles, preserve accepted P79 outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import report_oasis_bspline842_first100 as original


def run(report: Path):
    accepted_path = report / "ACCEPTED.json"
    accepted = json.loads(accepted_path.read_text())
    if not accepted["complete"] or accepted["candidate_count"] != 2100:
        raise RuntimeError("Require the accepted complete P79 report")
    for name, digest in accepted["output_sha256"].items():
        if original.e.sha(report / name) != digest:
            raise RuntimeError(f"Accepted report changed: {name}")
    with (report / "summary.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["cohort"] == "first100"]
    if len(rows) != 23 or any(int(r["n"]) != 100 for r in rows):
        raise RuntimeError("Require 21 sweep settings and two 100-case references")
    for row in rows:
        row["alpha"] = float(row["alpha"]) if row["alpha"] else None
        for key in original.e.METRICS:
            for suffix in ("mean", "sd"):
                row[f"{key}_{suffix}"] = float(row[f"{key}_{suffix}"])
    out = report / "readable_arc_v2"
    out.mkdir(exist_ok=False)
    set_title = Axes.set_title

    def wrapped_title(self, label, *args, **kwargs):
        return set_title(self, label.replace(" · ", "\n"), *args, **kwargs)

    Axes.set_title = wrapped_title
    try:
        original.arc(rows, out)
    finally:
        Axes.set_title = set_title
    original.e.atomic_json(out / "VALIDATED.json", {
        "passed": True, "change": "Wrap subplot titles only; identical data, axes, styles and SD ellipses",
        "source_report": str(report), "source_acceptance_sha256": original.e.sha(accepted_path),
        "source_summary_sha256": original.e.sha(report / "summary.csv"),
        "renderer_sha256": original.e.sha(Path(__file__)),
        "original_renderer_sha256": original.e.sha(Path(original.__file__)),
        "outputs": {p.name: original.e.sha(p) for p in out.iterdir() if p.is_file()},
        "created_at": original.e.c.now_text(),
    })
    for name, digest in accepted["output_sha256"].items():
        if original.e.sha(report / name) != digest:
            raise RuntimeError("Presentation export changed an accepted output")
    print(json.dumps({"readable_arc": str(out), "original_outputs_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve().is_relative_to(original.e.c.REPO_ROOT):
        parser.error("External artifacts only")
    run(args.report)
