#!/usr/bin/env python
"""Report-only P73 recovery: normalize historical n/a timing, keep fit sources frozen."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import run_oasis_bspline_svf as r


def optional_number(value):
    if value is None or value in ("", "n/a", "N/A"):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Historical timing must be finite or explicitly unavailable")
    return number


def normalized_references(references):
    result = copy.deepcopy(references)
    for row in result:
        row["optimizer_sec"] = optional_number(row.get("optimizer_sec", row.get("pipeline_training_elapsed_sec")))
        row["worker_sec"] = optional_number(row.get("worker_sec"))
        row["actual_adam_updates"] = optional_number(row.get("actual_adam_updates"))
        if row["actual_adam_updates"] is not None:
            row["actual_adam_updates"] = int(row["actual_adam_updates"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    args = parser.parse_args(argv)
    if args.artifact_root.resolve().is_relative_to(r.c.REPO_ROOT):
        parser.error("External artifact root required")
    run = args.artifact_root/"runs"/r.s.VARIANT
    out = args.artifact_root/"repo_experiments"/r.s.VARIANT
    if (out/"ACCEPTED.json").exists():
        raise RuntimeError("Preserve accepted report; no overwrite")
    original = r.c.read_json(run/"launch_manifest.json")
    r.assert_manifest(original)
    rows = []
    for spec in r.s.matrix():
        matches = [p for p in (run/spec.name()).glob("attempt*") if p.is_dir() and (p/"independent_audit.json").is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"Require one independently accepted run for {spec.name()}")
        rows.append(r.validate(matches[0],original,spec))
    failed_files = ("comparison.csv","paired_differences.csv","stage_diagnostics.csv","integration_sensitivity.csv","results.json")
    preserved = out/"failed_aggregation_v1"
    if not preserved.exists():
        preserved.mkdir(parents=True)
        for name in failed_files:
            if (out/name).exists(): (out/name).rename(preserved/name)
        r.atomic_json(preserved/"failure.json",{"error":"ValueError: Unknown format code f for historical string runtime n/a",
                      "fit_count":8,"independent_replay_count":8,"numerical_failure":False})
    manifest = copy.deepcopy(original)
    manifest["references"] = normalized_references(original["references"])
    manifest["report_recovery"] = {
        "aggregator_sha256":r.sha(Path(__file__)),
        "original_launch_manifest_sha256":r.sha(run/"launch_manifest.json"),
        "change":"Historical unavailable timing strings converted to JSON null; available P43 pipeline time retained",
        "numerical_sources_changed":False,"registration_reruns":0,
    }
    r.report(rows,manifest,out)
    r.atomic_json(out/"AGGREGATION_RECOVERY.json",manifest["report_recovery"])
    r.atomic_json(run/"COMPLETE.json",{"count":8,"completed_at":r.c.now_text(),"report_only_recovery":True})
    r.atomic_json(run/"progress.json",{"phase":"accepted","completed":8,"total":8,"updated_at":r.c.now_text()})
    print(str(out/"report.md"))


if __name__=="__main__": main()
