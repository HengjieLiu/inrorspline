#!/usr/bin/env python
"""First10 raw-NCC/diffusion lambda extension; prior P40 artifacts are read-only."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import common as c
import vfa_feature_refinement as f
import launch_oasis_vfa_feature_dense as prior
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file

VARIANT = "oasis_seed3_first10_mrdbscp842_dense_raw_ncc_diffusion_lambda_extension_v1"
LAMBDAS = (100., 300., 1000.)
METRICS = ("dice", "hd95", "asd")


def parse_args(argv=None):
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=root / "runs" / f.VARIANT)
    parser.add_argument("--source-review-root", type=Path, default=root / "repo_experiments" / f.VARIANT)
    parser.add_argument("--out-root", type=Path, default=root / "runs" / VARIANT)
    parser.add_argument("--review-root", type=Path, default=root / "repo_experiments" / VARIANT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.gpu_list = prior.parse_gpus(args.gpus)
    args.retry_preflight = False
    if not 1 <= args.max_concurrent <= min(4, len(args.gpu_list)) or args.cpu_threads < 1:
        parser.error("Require positive CPU threads and <=1 new worker per GPU")
    for path in (args.out_root, args.review_root):
        resolved = path.resolve()
        if resolved.is_relative_to(c.REPO_ROOT.resolve()):
            parser.error("Generated artifacts must be outside the repository")
        for source in (args.source_root, args.source_review_root):
            if resolved.is_relative_to(source.resolve()) or source.resolve().is_relative_to(resolved):
                parser.error("Output roots must not overlap immutable source roots")
    if args.out_root.resolve() == args.review_root.resolve():
        parser.error("Use separate run and report roots")
    return args


def control_config(frozen):
    config = f.FitConfig(**frozen["control_config"])
    expected = f.FitConfig(order="project_then_warp", loss="raw_ncc", support="full", alpha=30., lr=.3)
    if config != expected:
        raise RuntimeError("Source control differs from the approved lambda30 configuration")
    return config


def specs_for(config):
    return [prior.job(replace(config, alpha=alpha), rank, "raw_ncc_lambda_extension")
            for rank in range(10) for alpha in LAMBDAS]


class ExtensionScheduler(prior.Scheduler):
    def command(self, spec, gpu, path):
        return super().command(spec, gpu, path) + ["--experiment-variant", VARIANT]


def validate_candidate(row, prepared, expected_variant=None):
    if expected_variant is not None and row["variant"] != expected_variant:
        raise RuntimeError("Wrong experiment identity")
    for key, reference in (("prefix_sha256", "prefix_norm_sha256"), ("cache_sha256", "cache_sha256")):
        if row[key] != prepared[reference]:
            raise RuntimeError("Candidate changed its cached initialization")
    if row["config"]["loss"] != "raw_ncc" or not row["image_loss_in_optimization"]:
        raise RuntimeError("Not a raw-image NCC control")
    if row["regularizer"] != "unmasked_total_voxel_field_forward_valid_l2_diffusion_component_mean":
        raise RuntimeError("Regularizer changed")
    if row["parameter_units"] != "xyz_voxel_1mm" or row["dense_init_max_abs_voxel"] != 0:
        raise RuntimeError("Initialization or parameter units changed")
    if row["objective_support_voxels"] != int(np.prod(c.IMAGE_SHAPE)):
        raise RuntimeError("Loss is not full-volume")
    if any(abs(row["vfa_reference"][k] - prepared["vfa_metrics"][k]) > 1e-12 for k in METRICS):
        raise RuntimeError("VFA reference changed")


def load_sources(args, scheduler, config):
    state = c.read_json(args.source_root / "launcher_state.json")
    accepted = c.read_json(args.source_review_root / "ACCEPTED.json")
    if state["phase"] != "complete" or not accepted["technical_completion"]:
        raise RuntimeError("Prior first10 experiment is not complete/accepted")
    cohort = c.read_json(args.source_review_root / "first10_report.json")["case_metrics"]
    snapshot_paths = [args.source_root / "launcher_state.json", args.source_root / "launch_manifest.json",
                      args.source_review_root / "ACCEPTED.json", args.source_review_root / "frozen_selection.json",
                      args.source_review_root / "first10_report.json"]
    prepared, controls = {}, []
    for rank in range(10):
        prep_spec = prior.job(rank=rank)
        cache_dir = Path(state["results"][prep_spec["key"]])
        prepared[rank] = prior.validate_result(cache_dir, prep_spec)
        if sha256_file(cache_dir / "prefix_norm.npy") != prepared[rank]["prefix_norm_sha256"]:
            raise RuntimeError("Prefix field checksum changed")
        # Preparation predates P40's final scheduler/report preflight fixes.
        # Check numerical code against preparation and scheduling against the
        # actual accepted lambda30 runs below, not an earlier prepare attempt.
        for name in ("vfa_feature_refinement.py", "common.py", "run_one.py"):
            path = THIS_DIR / name
            if prepared[rank]["source_hashes"][str(path)] != sha256_file(path):
                raise RuntimeError(f"Numerical implementation drift: {name}")
        scheduler.state["results"][prep_spec["key"]] = str(cache_dir)
        control_spec = prior.job(config, rank, "reference")
        control_dir = Path(state["results"][control_spec["key"]])
        control = prior.validate_result(control_dir, control_spec)
        validate_candidate(control, prepared[rank], f.VARIANT)
        for name in ("vfa_feature_refinement.py", "common.py", "run_one.py", "launch_oasis_vfa_feature_dense.py"):
            path = THIS_DIR / name
            if control["source_hashes"][str(path)] != sha256_file(path):
                raise RuntimeError(f"Implementation differs from accepted lambda30 run: {name}")
        controls.append(control)
        snapshot_paths += [cache_dir / "summary.json", control_dir / "summary.json"]
        for method in ("VFA", "MR-D-BSCP 8-4-2", "Dense VFA features", "Dense raw NCC + diffusion"):
            matches = [r for r in cohort if r["method"] == method and r["case_rank"] == rank]
            if len(matches) != 1:
                raise RuntimeError("Incomplete or duplicate prior cohort")
            row = matches[0]
            if (row["fixed_id"], row["moving_id"]) != (control["fixed_id"], control["moving_id"]):
                raise RuntimeError("Prior cohort case mismatch")
            if method == "Dense raw NCC + diffusion" and any(row[k] != control[k] for k in METRICS):
                raise RuntimeError("Prior lambda30 report mismatch")
    snapshot = {str(path): sha256_file(path) for path in snapshot_paths}
    snapshot_path = args.out_root / "source_artifact_hashes.json"
    if snapshot_path.exists() and c.read_json(snapshot_path) != snapshot:
        raise RuntimeError("Immutable source metadata changed since initial launch")
    c.write_json(snapshot_path, snapshot)
    scheduler.save()
    return prepared, controls, cohort, snapshot


def cohort_means(rows):
    result = []
    for first in (0, 1):
        references = [r for r in rows if r["method"] == "VFA" and r["case_rank"] >= first]
        reference = {k: float(np.mean([r[k] for r in references])) for k in METRICS}
        for method in dict.fromkeys(r["method"] for r in rows):
            subset = [r for r in rows if r["method"] == method and r["case_rank"] >= first]
            if sorted(r["case_rank"] for r in subset) != list(range(first, 10)):
                raise RuntimeError("Incomplete or duplicate cohort; cannot aggregate")
            values = {k: float(np.mean([r[k] for r in subset])) for k in METRICS}
            row = {"subset": f"ranks_{first}_9", "method": method, "alpha": subset[0].get("alpha"),
                   "n": len(subset), **values,
                   **{"delta_vfa_" + k: values[k] - reference[k] for k in METRICS},
                   "cases_matching_all": sum(bool(r["matches_vfa"]) for r in subset),
                   "mean_matches_all": f.metric_deficit(values, reference) <= 1,
                   "metric_deficit": f.metric_deficit(values, reference)}
            for key in ("training_elapsed_sec", "optimizer_steps", "best_epoch", "diffusion",
                        "jdet_le0_pct_fg", "jdet_le0_pct_full", "change_from_prefix_mean_mm"):
                if all(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]) for r in subset):
                    row["mean_" + key] = float(np.mean([r[key] for r in subset]))
            result.append(row)
    return result


def write_report(args, new_rows, old_controls, source_cohort):
    expected = {(rank, alpha) for rank in range(10) for alpha in LAMBDAS}
    if len(new_rows) != 30 or {(r["case_rank"], r["config"]["alpha"]) for r in new_rows} != expected:
        raise RuntimeError("Require all 30 successful production fits before aggregation")
    rows = [dict(r) for r in source_cohort if r["method"] != "Dense raw NCC + diffusion"]
    references = {r["case_rank"]: r for r in rows if r["method"] == "VFA"}
    old = {r["case_rank"]: r for r in old_controls}
    for value in old_controls + new_rows:
        rank, alpha = value["case_rank"], value["config"]["alpha"]
        row = {**prior.flat_row(value), "method": f"Raw NCC + diffusion lambda{alpha:g}",
               "source_kind": "immutable_prior" if alpha == 30 else "new_fit",
               **{"delta_vfa_" + k: value[k] - references[rank][k] for k in METRICS},
               **{"delta_lambda30_" + k: value[k] - old[rank][k] for k in METRICS},
               "feature_loss_normalizer": value["feature_loss_normalizer"],
               "equivalent_raw_alpha": value["equivalent_raw_alpha"],
               "prefix_sha256": value["prefix_sha256"], "cache_sha256": value["cache_sha256"]}
        rows.append(row)
    rows.sort(key=lambda r: (r["case_rank"], r["method"]))
    means = cohort_means(rows)
    eligible = [r for r in means if r["subset"] == "ranks_0_9" and r["method"].startswith("Raw NCC")]
    selected = min(eligible, key=lambda r: (-r["dice"], r["asd"], r["hd95"]))
    winners = {k: min(eligible, key=lambda r: -r[k] if k == "dice" else r[k])["alpha"] for k in METRICS}
    report = {"status": "complete", "variant": VARIANT, "new_fit_count": len(new_rows),
              "lambdas_compared": [30., *LAMBDAS], "case_metrics": rows, "means": means,
              "selected_by_first10_mean_dice": selected, "best_lambda_per_mean_metric": winners,
              "best_joint_tolerance_mean": min(eligible, key=lambda r: (r["metric_deficit"], -r["dice"])),
              "all_first10_are_exploratory_tuning": True, "ranks1_9_are_not_held_out_here": True,
              "prior_P40_outputs_unchanged": True, "new_candidates": new_rows}
    c.write_csv(args.review_root / "first10_case_metrics.csv", rows)
    c.write_csv(args.review_root / "first10_means.csv", means)
    c.write_json(args.review_root / "report.json", report)
    lines = ["# First10 raw-NCC/diffusion lambda extension", "",
             "All ten cases are exploratory tuning in this follow-up, including ranks1–9.",
             "Only lambda changes from the prior raw-NCC control. Prior lambda30 and VFA-feature outputs are immutable.", "",
             "## Means", "", "| Subset | Method | Dice | HD95 | ASD | Matching cases | Mean matches |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in means:
        lines.append(f"| {r['subset']} | {r['method']} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} | {r['cases_matching_all']}/{r['n']} | {r['mean_matches_all']} |")
    lines += ["", f"Selected by first10 mean Dice, then ASD/HD95: lambda {selected['alpha']:g}.",
              f"Per-metric best lambda: {winners}.", "", "## Per case", "",
              "| Rank | Fixed / Moving | Method | Dice | HD95 | ASD | Best epoch | Steps | GPU |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case_rank']} | {r['fixed_id']} / {r['moving_id']} | {r['method']} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} | {r.get('best_epoch', '')} | {r.get('optimizer_steps', '')} | {r.get('gpu', '')} |")
    lines += ["", "CSV/JSON include paired differences versus VFA/lambda30, folding, diffusion and runtime diagnostics.",
              "Runtime is not a controlled speed comparison: unrelated GPU workloads remained active."]
    (args.review_root / "report.md").write_text("\n".join(lines) + "\n")
    c.write_json(args.review_root / "ACCEPTED.json", {"technical_completion": True, "new_fit_count": 30,
                 "reference_lambda30_count": 10, "selected_mean_matches_vfa": selected["mean_matches_all"],
                 "prior_artifacts_unchanged": True, "exploratory_all_first10": True})


def execute(args, config):
    scheduler = ExtensionScheduler(args)
    prepared, old_controls, cohort, snapshot = load_sources(args, scheduler, config)
    smoke = prior.job(replace(config, alpha=1000., max_epochs=1, min_epochs=0), 0, "smoke")
    scheduler.state["phase"] = "smoke"
    scheduler.save()
    smoke_row = scheduler.run([smoke])[0]
    validate_candidate(smoke_row, prepared[0], VARIANT)
    if args.stop_after == "smoke":
        scheduler.state["phase"] = "smoke_passed"
        scheduler.save()
        return
    scheduler.state["phase"] = "first10_lambda_extension"
    scheduler.save()
    completed = scheduler.run(specs_for(config))
    for row in completed:
        validate_candidate(row, prepared[row["case_rank"]], VARIANT)
    if any(sha256_file(Path(path)) != digest for path, digest in snapshot.items()):
        raise RuntimeError("Prior artifacts changed during the extension")
    write_report(args, completed, old_controls, cohort)
    scheduler.state["phase"] = "complete"
    scheduler.save()
    scheduler.event("experiment_complete", new_fits=30, review_root=str(args.review_root))


def main(argv=None):
    args = parse_args(argv)
    old_args = c.read_json(args.source_root / "launch_manifest.json")["args"]
    for key in ("oasis_root", "dl_root", "vfa_root", "checkpoint", "pair_file"):
        setattr(args, key, Path(old_args[key]))
    config = control_config(c.read_json(args.source_review_root / "frozen_selection.json"))
    if args.dry_run:
        print(json.dumps({"variant": VARIANT, "production_fits": specs_for(config),
                          "source_control": asdict(config), "args": vars(args)}, default=str, indent=2))
        return
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Run as container UID:GID 1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {"args": vars(args), "variant": VARIANT, "source": prior.source_metadata(),
                    "source_hashes": {str(p): sha256_file(p) for p in
                                      (Path(__file__), THIS_DIR / "run_oasis_vfa_feature_dense_one.py",
                                       THIS_DIR / "vfa_feature_refinement.py", THIS_DIR / "common.py")},
                    "initial_gpu_state": gpu_snapshot(), "uid": os.getuid(), "gid": os.getgid(),
                    "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
                    "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded"),
                    "source_control": asdict(config), "new_lambdas": LAMBDAS,
                    "time": c.now_text(), "pid": os.getpid()}
        path = args.out_root / "launch_manifest.json"
        if not path.exists():
            c.write_json(path, manifest)
        c.write_json(args.out_root / "invocations" / f"{time.time_ns()}_{os.getpid()}.json", manifest)
        with (args.out_root / "launcher.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                c.write_json(args.out_root / "current_status.json", {"status": "running", "pid": os.getpid(), "time": c.now_text()})
                execute(args, config)
                c.write_json(args.out_root / "current_status.json", {"status": "complete" if args.stop_after == "full" else "smoke_passed", "time": c.now_text()})
            except Exception:
                failure = {"time": c.now_text(), "traceback": traceback.format_exc()}
                c.write_json(args.out_root / "launcher_failures" / f"{time.time_ns()}.json", failure)
                c.write_json(args.out_root / "current_status.json", {"status": "failed", **failure})
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
