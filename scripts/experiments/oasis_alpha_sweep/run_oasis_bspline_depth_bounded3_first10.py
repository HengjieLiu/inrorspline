#!/usr/bin/env python
"""P71: add bounded three-map composition to the accepted P70 depth tables.

P70's numerical implementation is imported without modification. This adapter
freezes it to a single new bounded-three-map configuration, runs forty fresh
cells, independently replays them, then writes a separate combined report that
joins those cells with the immutable accepted P70 cells and references.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_oasis_bspline_depth_factorial_first10 as base  # noqa: E402

P70_VARIANT = base.VARIANT
BASE_MAKE_MANIFEST = base.make_manifest
VARIANT = "oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1"
CONFIGS = {
    "bounded_3_compose": {
        "bounded": True,
        "maps_per_level": 3,
        "method": "direct_cp_multistage_bspline_tanh_bounded_three_map_composition",
    }
}
CONFIG_ORDER = tuple(CONFIGS)
REPORT_METHODS = (
    "unbounded_1", "unbounded_2_compose", "bounded_1", "bounded_2_compose",
    "bounded_3_compose",
)
METHOD_LABELS = {
    **{k: base.CONFIGS[k]["method"] for k in base.CONFIGS},
    "bounded_3_compose": CONFIGS["bounded_3_compose"]["method"],
}
METRICS = base.METRICS_TO_REPORT
NEW_SOURCE_KEY = "bounded3_extension_runner"


def configure_base():
    """Install this extension's isolated factorial identity in the P70 engine."""
    base.CONFIGS = CONFIGS
    base.CONFIG_ORDER = CONFIG_ORDER
    base.VARIANT = VARIANT
    base.PROTOCOL = {
        **base.PROTOCOL,
        "version": 1,
        "extension": "bounded_three_map_composition",
        "configs": CONFIGS,
        "method": CONFIGS["bounded_3_compose"]["method"],
        "maps_per_level": 3,
        "composition": "three independently optimized bounded identity increments per active CPS level",
        "budget": "each map gets the complete existing per-stage Adam schedule with fresh state",
    }


def extension_sources() -> dict:
    return {
        "runner": base.sha(Path(__file__)),
        "p70_runner": base.sha(HERE / "run_oasis_bspline_depth_factorial_first10.py"),
        "p70_auditor": base.sha(HERE / "audit_oasis_bspline_depth_factorial_first10.py"),
        "aggregator": base.sha(Path(__file__)),
    }


def assert_extension_manifest(manifest: dict):
    base.assert_manifest(manifest)
    if extension_sources() != manifest.get("extension_source_hashes"):
        raise RuntimeError("P71 extension source drift")


def verify_p70_parent(artifact_root: Path) -> tuple[dict, dict, Path]:
    run_root = artifact_root / "runs" / P70_VARIANT
    report_root = artifact_root / "repo_experiments" / P70_VARIANT
    manifest_path = run_root / "launch_manifest.json"
    manifest = base.c.read_json(manifest_path)
    base.assert_manifest(manifest)
    accepted_path = report_root / "ACCEPTED.json"
    accepted = base.c.read_json(accepted_path)
    for name, digest in accepted["output_sha256"].items():
        if base.sha(report_root / name) != digest:
            raise RuntimeError(f"Read-only accepted P70 report changed: {name}")
    results_path = report_root / "results.json"
    results = base.c.read_json(results_path)
    if not results.get("complete") or len(results.get("candidates", [])) != 160:
        raise RuntimeError("Accepted P70 parent must contain 160 complete candidates")
    if len(results.get("summary", [])) != 24:
        raise RuntimeError("Accepted P70 parent summary is incomplete")
    for row in results["candidates"]:
        run_dir = Path(row["run_dir"])
        if not (run_dir / "DONE").is_file() or base.c.read_json(run_dir / "summary.json") != row:
            raise RuntimeError("P70 parent candidate changed or is incomplete")
        base.audit_ok(row)
        if base.sha(Path(row["dvf_path"])) != row["dvf_sha256"]:
            raise RuntimeError("P70 parent DVF changed")
    return accepted, results, report_root


def make_manifest(artifact_root: Path, gpus: list[str]) -> dict:
    manifest = BASE_MAKE_MANIFEST(artifact_root, gpus)
    accepted, results, report_root = verify_p70_parent(artifact_root)
    manifest.update({
        "variant": VARIANT,
        "protocol": base.PROTOCOL,
        "extension_source_hashes": extension_sources(),
        "parent_p70_variant": P70_VARIANT,
        "parent_p70_report_root": str(report_root),
        "parent_p70_accepted_sha256": base.sha(report_root / "ACCEPTED.json"),
        "parent_p70_results_sha256": base.sha(report_root / "results.json"),
        "parent_p70_candidates_for_comparison": results["candidates"],
        "parent_p70_results_provenance": results["provenance"],
        "production_fit_count": 40,
        "smoke_fit_count": 4,
        "prior_p70_runs_reused_as_new_cells": 0,
        "map_budget_note": "three bounded maps per active level; each receives a full fresh-Adam stage schedule",
    })
    return manifest


def start_process(run_root, rank, depth, key, gpu, required_mib, smoke=False, audit=False):
    state = None
    snapshot = None
    for attempt in range(3):
        snapshot = base.legacy.gpu_snapshot()
        state = next((r for r in snapshot.get("rows", []) if str(r.get("physical_gpu")) == str(gpu)), None)
        if state is not None:
            break
        if attempt < 2:
            time.sleep(1)
    if state is None:
        raise RuntimeError(f"GPU{gpu} missing from live snapshot: {snapshot}")
    if state["memory_free_mib"] < required_mib:
        return None
    path = run_root / base.run_name(rank, depth, key, smoke)
    if audit:
        command = [sys.executable, "-u", "-B", str(Path(__file__)), "audit",
                   "--manifest", str(run_root / "launch_manifest.json"),
                   "--run-dir", str(path), "--gpu", str(gpu)]
    else:
        if path.exists():
            raise RuntimeError(f"Preserve incomplete attempt for audit: {path}")
        command = [sys.executable, "-u", "-B", str(Path(__file__)), "worker",
                   "--manifest", str(run_root / "launch_manifest.json"),
                   "--out-dir", str(path), "--case-rank", str(rank),
                   "--depth", str(depth), "--config", key, "--gpu", str(gpu)]
        if smoke:
            command.append("--smoke")
    log_path = run_root / (path.name + ("_audit" if audit else "") + ".log")
    log = log_path.open("x")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    except BaseException:
        log.close()
        raise
    print(json.dumps({"launched": path.name, "audit": audit, "gpu": gpu,
                      "pid": process.pid, "gpu_before": state}), flush=True)
    return {"process": process, "log": log, "run_dir": path, "rank": rank,
            "depth": depth, "config": key, "gpu": str(gpu)}


def write_extension_metadata(run_dir: Path, manifest: dict):
    for name in ("manifest.json", "summary.json"):
        path = run_dir / name
        record = base.c.read_json(path)
        record["extension_variant"] = VARIANT
        record["extension_source_hashes"] = manifest["extension_source_hashes"]
        record["parent_p70_results_sha256"] = manifest["parent_p70_results_sha256"]
        base.atomic_json(path, record)


def worker(args):
    manifest = base.c.read_json(args.manifest)
    assert_extension_manifest(manifest)
    base.worker(args)
    write_extension_metadata(args.out_dir, manifest)


def audit(args):
    manifest = base.c.read_json(args.manifest)
    assert_extension_manifest(manifest)
    # Reuse P70's independent reconstruction auditor under this process's
    # isolated three-map configuration.
    import audit_oasis_bspline_depth_factorial_first10 as replay
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *old_argv[2:]]
        replay.main()
    finally:
        sys.argv = old_argv
    path = args.run_dir / "independent_audit.json"
    record = base.c.read_json(path)
    record["extension_source_hashes"] = manifest["extension_source_hashes"]
    record["extension_variant"] = VARIANT
    base.atomic_json(path, record)


def launch_with_extra_smokes(args):
    """Use P70's queue/admission logic, adding native one-update smokes per depth."""
    configure_base()
    base.make_manifest = make_manifest
    base.start_process = start_process
    base.write_report = write_combined_report
    smoke_args = copy.copy(args)
    smoke_args.stop_after = "smoke"
    base.launch(smoke_args)
    run_root = args.artifact_root / "runs" / VARIANT
    manifest = base.c.read_json(run_root / "launch_manifest.json")
    assert_extension_manifest(manifest)
    initial_smoke_marker = base.c.read_json(run_root / "smokes_passed.json")
    required_mib = max(8192, int(initial_smoke_marker.get("required_free_mib", 8192)))
    smoke_runs = []
    # P70's launcher already ran/validated the deepest schedule. Add the other
    # three depths, each on a separate currently available GPU where possible.
    for depth in range(1, 4):
        rank, key = 1, "bounded_3_compose"
        path = run_root / base.run_name(rank, depth, key, True)
        if path.exists():
            raise RuntimeError(f"Preserve existing noncanonical smoke attempt: {path}")
        gpu = args.gpus[(depth - 1) % len(args.gpus)]
        item = start_process(run_root, rank, depth, key, gpu, required_mib, smoke=True)
        if item is None:
            raise RuntimeError(f"GPU{gpu} lacks memory for depth-{depth} full-size smoke")
        code = item["process"].wait()
        item["log"].close()
        if code:
            raise RuntimeError(f"Depth-{depth} smoke failed; preserved at {path}")
        row = base.validate(path, manifest, rank, depth, key, smoke=True)
        write_extension_metadata(path, manifest)
        row = base.c.read_json(path / "summary.json")
        required_mib = max(required_mib, math.ceil(row["peak_cuda_gib"] * 1024) + 2048)
        smoke_runs.append(str(path))
    # Include the full-depth smoke already produced by P70's launcher.
    full_path = run_root / base.run_name(1, 4, "bounded_3_compose", True)
    full_row = base.validate(full_path, manifest, 1, 4, "bounded_3_compose", smoke=True)
    write_extension_metadata(full_path, manifest)
    smoke_runs.append(str(full_path))
    required_mib = max(required_mib, math.ceil(full_row["peak_cuda_gib"] * 1024) + 2048)
    base.atomic_json(run_root / "smokes_passed.json", {
        "passed": True, "smoke_count": 4, "required_free_mib": required_mib,
        "runs": smoke_runs, "completed_at": base.c.now_text(),
    })
    if args.stop_after == "smoke":
        print("All four bounded-three-map native-grid smokes passed; production not launched.", flush=True)
        return
    base.launch(args)
    # P70's launch marker lists just its single configuration smoke. Replace it
    # with the complete extension-specific set after successful acceptance.
    accepted = base.c.read_json(args.artifact_root / "repo_experiments" / VARIANT / "ACCEPTED.json")
    base.atomic_json(run_root / "smokes_passed.json", {
        "passed": True, "smoke_count": 4, "required_free_mib": required_mib,
        "runs": smoke_runs, "completed_at": base.c.now_text(),
        "combined_report_sha256": accepted["output_sha256"]["report.md"],
    })
    progress_path = run_root / "progress.json"
    progress = base.c.read_json(progress_path)
    progress.update({"completed": 40, "audited": 40, "total": 40,
                     "new_production_registrations": 40,
                     "updated_at": base.c.now_text()})
    base.atomic_json(progress_path, progress)


def verify_parent_results(manifest):
    root = Path(manifest["parent_p70_report_root"])
    if base.sha(root / "ACCEPTED.json") != manifest["parent_p70_accepted_sha256"]:
        raise RuntimeError("Accepted P70 parent report changed after launch")
    if base.sha(root / "results.json") != manifest["parent_p70_results_sha256"]:
        raise RuntimeError("Accepted P70 parent results changed after launch")
    accepted = base.c.read_json(root / "ACCEPTED.json")
    for name, digest in accepted["output_sha256"].items():
        if base.sha(root / name) != digest:
            raise RuntimeError(f"Accepted P70 report hash mismatch: {name}")
    return base.c.read_json(root / "results.json")


def grouped_combined_summary(rows, refs):
    summary, by_depth = [], {}
    groups = REPORT_METHODS + ("VFA (original)", "SITReg λ=1 (original)")
    for depth in range(1, 5):
        by_depth[depth] = []
        for key in groups:
            if key in REPORT_METHODS:
                method_rows = [r for r in rows if r["depth"] == depth and r["config_key"] == key]
                method = METHOD_LABELS[key]
                time_rows = method_rows
            else:
                method_rows = [{**r, "case_rank": r["rank"]} for r in refs
                               if r["method"] == key]
                method = key
                time_rows = []
            if len(method_rows) != 10 or {r["case_rank"] for r in method_rows} != set(range(10)):
                raise RuntimeError(f"Depth {depth}/{method}: incomplete 10-pair cohort")
            record = {"depth": depth, "schedule": "→".join(map(str, base.SCHEDULES[depth])),
                      "config_key": key if key in REPORT_METHODS else "reference",
                      "method": method, "n": 10}
            for metric in METRICS:
                values = [float(r[metric]) for r in method_rows]
                record[metric + "_mean"] = statistics.mean(values)
                record[metric + "_sd"] = statistics.stdev(values)
            for name in ("training_elapsed_sec", "total_elapsed_sec", "actual_adam_updates"):
                if time_rows and all(name in r for r in time_rows):
                    values = [float(r[name]) for r in time_rows]
                    record[name + "_mean"] = statistics.mean(values)
                    record[name + "_sd"] = statistics.stdev(values)
            summary.append(record)
            by_depth[depth].append(record)
    return summary, by_depth


def make_extension_prefix_diagnostics(rows):
    lookup = {(r["case_rank"], r["depth"]): r for r in rows}
    scale = np.asarray([95.5, 111.5, 79.5], dtype=np.float32)
    records = []
    for rank in range(10):
        for depth in range(1, 4):
            shallow, deeper = lookup[(rank, depth)], lookup[(rank, depth + 1)]
            for step in range(depth * 3):
                a, z = shallow["blocks"][step], deeper["blocks"][step]
                af = np.load(a["dvf_path"], mmap_mode="r")
                zf = np.load(z["dvf_path"], mmap_mode="r")
                delta = (af - zf) * scale
                maximum = float(np.abs(delta).max())
                rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
                info_a, info_z = a["optimizer"], z["optimizer"]
                loss_delta = max(abs(info_a[k] - info_z[k]) for k in
                                 ("best_loss", "best_data_loss", "best_reg_loss"))
                exact = (a["dvf_sha256"] == z["dvf_sha256"] and
                         info_a["best_epoch"] == info_z["best_epoch"] and
                         info_a["actual_epochs"] == info_z["actual_epochs"] and loss_delta == 0)
                records.append({"case_rank": rank, "shallow_depth": depth,
                    "deeper_depth": depth + 1, "step": step, "cps": a["cps"],
                    "map_index": a["map_index"], "field_hash_identical": exact,
                    "max_abs_voxel": maximum, "rms_voxel": rms,
                    "max_objective_component_delta": loss_delta,
                    "best_epoch_delta": abs(info_a["best_epoch"] - info_z["best_epoch"]),
                    "actual_epoch_delta": abs(info_a["actual_epochs"] - info_z["actual_epochs"]),
                    "exact_replay": exact})
    if len(records) != 180:
        raise RuntimeError(f"Expected180 P71 adjacent-depth map diagnostics, got{len(records)}")
    return records


def write_combined_report(new_rows, manifest, out):
    assert_extension_manifest(manifest)
    parent = verify_parent_results(manifest)
    old_rows = parent["candidates"]
    if len(old_rows) != 160 or len(new_rows) != 40:
        raise RuntimeError("Require all160 accepted P70 plus40 fresh P71 cells")
    combined = old_rows + new_rows
    refs = manifest["references"]
    summaries, table_rows = grouped_combined_summary(combined, refs)
    per_case, paired, stage_rows = [], [], []
    for row in combined:
        case = manifest["cases"][str(row["case_rank"])]
        per_case.append({"case_rank": row["case_rank"], "fixed_id": case["fixed_id"],
            "moving_id": case["moving_id"], "depth": row["depth"],
            "schedule": "→".join(map(str, row["cps_schedule"])),
            "config_key": row["config_key"], "method": row["method"],
            **{k: row[k] for k in METRICS}, "optimizer_sec": row["training_elapsed_sec"],
            "worker_sec": row["total_elapsed_sec"], "actual_adam_updates": row["actual_adam_updates"],
            "gpu": row["gpu"], "run_dir": row["run_dir"]})
        for ref_method in ("VFA (original)", "SITReg λ=1 (original)"):
            ref = next(r for r in refs if r["rank"] == row["case_rank"] and r["method"] == ref_method)
            record = {"case_rank": row["case_rank"], "depth": row["depth"],
                      "config_key": row["config_key"], "method": row["method"], "reference": ref_method}
            for metric in METRICS:
                record["delta_" + metric] = row[metric] - ref[metric]
            paired.append(record)
        for block in row["blocks"]:
            info = block["optimizer"]
            stage_rows.append({"case_rank": row["case_rank"], "depth": row["depth"],
                "config_key": row["config_key"], "step": block["step_index"], "cps": block["cps"],
                "map_index": block["map_index"], "bounded": row["bounded"], "alpha": 1.0,
                "best_epoch": info["best_epoch"], "actual_adam_updates": info["actual_epochs"],
                "stopping_reason": info["stopping_reason"], "best_loss": info["best_loss"],
                "best_data_loss": info["best_data_loss"], "best_reg_loss": info["best_reg_loss"],
                "optimizer_sec": info["training_elapsed_sec"], "gpu": row["gpu"]})
    for depth in range(1, 5):
        for rank in range(10):
            case = manifest["cases"][str(rank)]
            for method in ("VFA (original)", "SITReg λ=1 (original)"):
                ref = next(r for r in refs if r["rank"] == rank and r["method"] == method)
                per_case.append({"case_rank": rank, "fixed_id": case["fixed_id"], "moving_id": case["moving_id"],
                    "depth": depth, "schedule": "→".join(map(str, base.SCHEDULES[depth])),
                    "config_key": "reference", "method": method,
                    **{k: ref[k] for k in METRICS}, "optimizer_sec": "", "worker_sec": "",
                    "actual_adam_updates": "", "gpu": "n/a", "run_dir": ref.get("dvf_path", "")})
    if len(summaries) != 28 or len(per_case) != 280 or len(paired) != 400 or len(stage_rows) != 900:
        raise RuntimeError("Combined report row-count validation failed")
    prefix_checks = make_extension_prefix_diagnostics(new_rows)
    if out.exists():
        raise RuntimeError(f"Refuse to replace existing report directory: {out}")
    out.mkdir(parents=True)
    base.c.write_csv(out / "summary.csv", summaries)
    base.c.write_csv(out / "per_case.csv", per_case)
    base.c.write_csv(out / "paired_reference_differences.csv", paired)
    base.c.write_csv(out / "stage_diagnostics.csv", stage_rows)
    base.c.write_csv(out / "bounded3_shared_prefix_checks.csv", prefix_checks)
    metadata = {"complete": True, "new_p71_registrations": 40,
        "included_accepted_p70_registrations": 160, "combined_registration_cells": 200,
        "independent_p71_audits": 40, "p71_smokes": 4,
        "p70_accepted_sha256": manifest["parent_p70_accepted_sha256"],
        "p70_results_sha256": manifest["parent_p70_results_sha256"],
        "extension_source_hashes": manifest["extension_source_hashes"],
        "bounded3_shared_prefix_diagnostics": len(prefix_checks), "completed_at": base.c.now_text()}
    base.atomic_json(out / "aggregation_metadata.json", metadata)
    base.atomic_json(out / "results.json", {"complete": True, "candidates": combined,
        "new_bounded3_candidates": new_rows, "summary": summaries, "per_case": per_case,
        "paired_reference_differences": paired, "stage_diagnostics": stage_rows,
        "bounded3_shared_prefix_checks": prefix_checks, "provenance": manifest,
        "aggregation_metadata": metadata})
    columns = [("dice", "Dice ↑"), ("hd95", "HD95 mm ↓"), ("asd", "ASD mm ↓"),
        ("log_jdet_std_fg", "FG SD(log J)"), ("jdet_le0_pct_fg", "FG folding %"),
        ("training_elapsed_sec", "Optimizer sec"), ("total_elapsed_sec", "Worker sec"),
        ("actual_adam_updates", "Adam updates")]
    fmt = lambda rec, k: f"{rec[k+'_mean']:.4f} ± {rec[k+'_sd']:.4f}" if k+"_mean" in rec else "n/a"
    lines = ["# First-ten B-spline depth × constraint comparison, extended with bounded three-map composition", "",
        "Each of the four schedules is shown with the original four P70 settings plus a fresh bounded-three-map composition arm; original VFA and SITReg are references.",
        "P70's160 cells and accepted outputs remain unchanged. P71 adds40 fresh cells (10 pairs × 4 schedules) and does not reuse old registrations as new cells.",
        "At every active level, the new arm composes three independent bounded identity-initialized B-spline maps. Each map receives the full stage schedule with fresh Adam. This increases optimizer updates/capacity; comparisons are not equal-compute.",
        "All arms use λ=1 per active stage, raw squared local NCC9, unmasked total-field voxel-unit diffusion, existing preprocessing/border sampling, and canonical metrics.",
        "Values are mean ± across-case sample SD (n=10, ddof=1). This first-ten cohort is exploratory because it informed prior tuning.", ""]
    for depth in range(1, 5):
        lines += [f"## CPS schedule {'→'.join(map(str, base.SCHEDULES[depth]))}", "",
            "| Setting | " + " | ".join(label for _, label in columns) + " |",
            "|---|" + "---:|" * len(columns)]
        for rec in table_rows[depth]:
            lines.append("| " + rec["method"] + " | " + " | ".join(fmt(rec, key) for key, _ in columns) + " |")
        lines.append("")
    lines += ["## Diagnostics and artifacts", "",
        "`per_case.csv` contains200 method rows and80 repeated per-schedule VFA/SITReg references. `paired_reference_differences.csv` contains400 candidate/reference comparisons.",
        "`stage_diagnostics.csv` contains900 map-level rows, including stopping/best epochs and runtime. `bounded3_shared_prefix_checks.csv` records cross-depth repeatability diagnostics for the new arm; cps2 CUDA variation is reported rather than used to drop independent cells.",
        "The original P70 accepted report is read-only and hash-verified. P71 outputs, histories, checkpoints, fields, and audits are in the separate extension run directory.",
        "Optimizer time sums each map's training time; worker time also includes loading, metrics, and serialization. GPU contention remains a timing limitation."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    accepted = {"complete": True, "new_p71_registrations": 40,
        "included_p70_registrations": 160, "combined_registration_cells": 200,
        "independent_p71_audits": 40, "p71_smokes": 4,
        "bounded3_shared_prefix_checks": len(prefix_checks), "completed_at": base.c.now_text(),
        "output_sha256": {p.name: base.sha(p) for p in out.iterdir() if p.is_file()}}
    base.atomic_json(out / "ACCEPTED.json", accepted)


def main():
    configure_base()
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=("audit",))
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
        args = parser.parse_args()
        audit(args)
        return
    args = base.parse_args()
    if args.command == "worker":
        worker(args)
    elif args.command == "launch":
        launch_with_extra_smokes(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
