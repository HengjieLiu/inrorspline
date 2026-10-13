#!/usr/bin/env python
"""Resumable, bounded four-GPU VFA-feature refinement experiment."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import itertools
import json
import math
import os
import subprocess
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
from launch_oasis_mrdbscp842_dense_lambda_sweep import parse_gpus, source_metadata
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    for name in ("oasis-root", "dl-root", "vfa-root", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser.add_argument("--out-root", type=Path, default=root / "runs" / f.VARIANT)
    parser.add_argument("--review-root", type=Path, default=root / "repo_experiments" / f.VARIANT)
    parser.add_argument("--stop-after", choices=("prepare", "smoke", "search", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-preflight", action="store_true",
                        help="Explicitly retry a reviewed failed prepare/smoke in a new preserved attempt")
    args = parser.parse_args(argv)
    args.gpu_list = parse_gpus(args.gpus)
    if not 1 <= args.max_concurrent <= min(4, len(args.gpu_list)):
        parser.error("At most one worker per GPU and four concurrent workers")
    for path in (args.out_root, args.review_root):
        if path.resolve().is_relative_to(c.REPO_ROOT.resolve()):
            parser.error("Generated artifacts must be outside the repository")
    return args


def phase_a():
    # Interleave objective families early, with expensive warp/project work first.
    return [f.FitConfig(order=o, loss=l, alpha=a, lr=lr)
            for a, lr, o, l in itertools.product((0, .001, .01, .1, 1., 10.), (.03, .1), f.ORDERS, ("mse", "global_ncc"))]


def family(row):
    cfg = row["config"]
    return cfg["order"], cfg["loss"], cfg["support"]


def positive_center(rows, winner):
    eligible = [r for r in rows if family(r) == family(winner) and r["config"]["alpha"] > 0]
    if not eligible:
        raise RuntimeError("Selected family has no positive-lambda candidate")
    return min(eligible, key=f.selection_key)["config"]["alpha"]


def phase_b(rows):
    representatives = {}
    for row in rows:
        key = family(row)
        if key not in representatives or f.selection_key(row) < f.selection_key(representatives[key]):
            representatives[key] = row
    winners = sorted(representatives.values(), key=f.selection_key)[:2]
    return [replace(f.FitConfig(**winner["config"]), alpha=positive_center(rows, winner) * factor, lr=lr)
            for winner in winners for factor, lr in itertools.product((1/3, 1., 3.), (.01, .05, .2, .3))]


def phase_c(rows):
    winner = min(rows, key=f.selection_key)
    center = positive_center(rows, winner)
    base = f.FitConfig(**winner["config"])
    return [replace(base, loss=loss, support=support, alpha=center * factor)
            for loss, support, factor in itertools.product(("local_ncc", "unit_mse"),
                                                           ("full", "fixed_positive"), (.1, .3, 1., 3., 10.))]


def controls(rows, winner):
    center = positive_center(rows, winner)
    return [replace(f.FitConfig(**winner["config"]), loss="raw_ncc", support="full", alpha=center * factor)
            for factor in (.1, .3, 1., 3.)]


def job(config=None, rank=0, phase="prepare"):
    key = f"prepare_rank{rank:03d}" if config is None else f.config_name(config, rank)
    return {"key": key, "mode": "prepare" if config is None else "fit", "rank": rank,
            "phase": phase, "config": None if config is None else asdict(config)}


def validate_result(path, spec):
    if not (path / "DONE").is_file():
        raise RuntimeError(f"Incomplete result {path}")
    row = c.read_json(path / "summary.json")
    if row["case_rank"] != spec["rank"] or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Wrong rank or direction")
    expected = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec["rank"]]
    if (row["fixed_id"], row["moving_id"]) != expected:
        raise RuntimeError("Wrong subject IDs")
    backend = row["cuda_backend"]
    if backend["cudnn_benchmark"] or not backend["cudnn_deterministic"]:
        raise RuntimeError("Wrong CUDA backend flags")
    if spec["mode"] == "prepare":
        if not row["vfa_inference_parity"]["passed"]:
            raise RuntimeError("Failed VFA checkpoint parity")
        if sha256_file(path / "cache.pt") != row["cache_sha256"]:
            raise RuntimeError("Prepared cache changed")
    else:
        if row["config"] != spec["config"] or row["method"] != f.METHOD:
            raise RuntimeError("Candidate configuration mismatch")
        if row["representation"] != "dense_voxel" or any(row[k] is not None for k in ("cps", "control_shape", "bspline_order")):
            raise RuntimeError("Final output is not a direct dense field")
        if row["uses_cps1"] or row["reference_dvf_supervision"] or row["segmentation_loss"]:
            raise RuntimeError("Forbidden optimizer supervision/representation")
        if any(not math.isfinite(row[k]) for k in ("dice", "hd95", "asd", "best_loss")):
            raise RuntimeError("Nonfinite candidate metric")
        field = np.load(row["dvf_path"], mmap_mode="r")
        if field.shape != (*c.IMAGE_SHAPE, 3) or not np.isfinite(field).all():
            raise RuntimeError("Invalid serialized field")
        if sha256_file(Path(row["dvf_path"])) != row["dvf_sha256"]:
            raise RuntimeError("DVF checksum mismatch")
    return row


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.path = args.out_root / "launcher_state.json"
        self.state = c.read_json(self.path) if self.path.exists() else {"results": {}, "attempts": [], "phase": "preflight"}
        self.revision = source_metadata()["source_revision"]

    def save(self):
        temp = self.path.with_suffix(".tmp")
        c.write_json(temp, self.state)
        temp.replace(self.path)

    def event(self, event, **details):
        row = {"time": c.now_text(), "event": event, **details}
        print(json.dumps(row), flush=True)
        with (self.args.out_root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def cached(self, rank):
        return Path(self.state["results"][f"prepare_rank{rank:03d}"])

    def command(self, spec, gpu, path):
        args = self.args
        cmd = [sys.executable, "-B", str(THIS_DIR / "run_oasis_vfa_feature_dense_one.py"),
               "--mode", spec["mode"], "--gpu", gpu, "--case-rank", str(spec["rank"]),
               "--out-dir", str(path), "--cache-dir", str(path if spec["mode"] == "prepare" else self.cached(spec["rank"])),
               "--phase", spec["phase"], "--source-revision", self.revision,
               "--cpu-threads", str(args.cpu_threads)]
        for name in ("oasis_root", "dl_root", "vfa_root", "checkpoint", "pair_file"):
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if spec["config"] is not None:
            for key, value in spec["config"].items():
                flag = {"order": "feature-order", "loss": "feature-loss", "support": "feature-support"}.get(key, key.replace("_", "-"))
                cmd += ["--" + flag, str(value)]
        return cmd

    def run(self, specs):
        if len({s["key"] for s in specs}) != len(specs):
            raise RuntimeError("Duplicate jobs in phase")
        pending = []
        for spec in specs:
            if spec["key"] in self.state["results"]:
                validate_result(Path(self.state["results"][spec["key"]]), spec)
            else:
                prior = [a for a in self.state["attempts"] if a["key"] == spec["key"]]
                recovered = False
                for attempt in prior:
                    path = Path(attempt["path"])
                    if (path / "DONE").exists():
                        validate_result(path, spec)
                        self.state["results"][spec["key"]] = str(path)
                        recovered = True
                        break
                    if attempt.get("status") == "running":
                        raise RuntimeError(f"Unresolved previous worker; inspect PID {attempt['pid']} before resuming")
                if not recovered:
                    if prior and not (self.args.retry_preflight and spec["phase"] in ("prepare", "smoke")):
                        raise RuntimeError(f"Previous unsuccessful attempts require review: {spec['key']}")
                    pending.append((spec, max((a["attempt"] for a in prior), default=-1) + 1, 0))
        active, failures, finished = {}, [], 0
        self.save()
        while pending or active:
            for gpu, info in list(active.items()):
                status = info["process"].poll()
                if status is None:
                    continue
                info["log"].close()
                del active[gpu]
                finished += 1
                spec, record = info["spec"], info["record"]
                try:
                    if status != 0:
                        raise RuntimeError(f"Worker exit {status}")
                    validate_result(Path(record["path"]), spec)
                    self.state["results"][spec["key"]] = record["path"]
                    record["status"] = "complete"
                    self.event("complete", key=spec["key"], gpu=gpu)
                except Exception as exc:
                    record.update(status="failed", error=str(exc), exit_code=status)
                    log_text = Path(record["log_path"]).read_text(errors="replace")
                    oom = "out of memory" in log_text.lower() and "cuda" in log_text.lower()
                    if oom and record["attempt"] == 0:
                        pending.append((spec, 1, finished + (1 if active else 0)))
                        self.event("oom_retry_queued", key=spec["key"])
                    else:
                        failures.append({"key": spec["key"], "error": str(exc), "log": record["log_path"]})
                        self.event("failure", **failures[-1])
                self.save()
            if failures:
                pending.clear()  # Let already-running workers finish; never kill them.
            free = [g for g in self.args.gpu_list if g not in active]
            while pending and free and len(active) < self.args.max_concurrent:
                eligible = [i for i, (_, _, ready) in enumerate(pending) if finished >= ready]
                if not eligible:
                    break
                spec, attempt, _ = pending.pop(eligible[0])
                snapshot = gpu_snapshot()
                if attempt:
                    memory = {r["physical_gpu"]: r["memory_free_mib"] for r in snapshot["rows"]}
                    gpu = max(free, key=lambda g: memory.get(g, -1))
                else:
                    gpu = free[0]
                free.remove(gpu)
                path = self.args.out_root / "jobs" / spec["key"] / f"attempt{attempt}"
                log_path = self.args.out_root / "logs" / f"{spec['key']}_attempt{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or log_path.exists():
                    raise FileExistsError(f"Refusing to overwrite attempt/log: {path}")
                cmd = self.command(spec, gpu, path)
                log = log_path.open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(self.args.cpu_threads),
                           MKL_NUM_THREADS=str(self.args.cpu_threads), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                record = {"key": spec["key"], "path": str(path), "log_path": str(log_path), "gpu": gpu,
                          "pid": process.pid, "attempt": attempt, "status": "running", "command": cmd,
                          "initial_gpu_state": snapshot, "started_at": c.now_text()}
                self.state["attempts"].append(record)
                active[gpu] = {"process": process, "log": log, "record": record, "spec": spec}
                self.event("launched", key=spec["key"], gpu=gpu, pid=process.pid, attempt=attempt)
                self.save()
            if active:
                time.sleep(2)
            elif pending and not any(finished >= ready for _, _, ready in pending):
                raise RuntimeError("OOM retry waiting condition cannot be satisfied")
        if failures:
            raise RuntimeError(f"Phase has failed candidates; no aggregation: {failures}")
        return [validate_result(Path(self.state["results"][s["key"]]), s) for s in specs]


def flat_row(row):
    cfg = row.get("config", {})
    return {k: row.get(k) for k in ("run_name", "phase", "case_rank", "fixed_id", "moving_id", "dice", "hd95", "asd",
                                   "matches_vfa", "metric_deficit", "best_epoch", "optimizer_steps", "stopping_reason",
                                   "training_elapsed_sec", "total_elapsed_sec", "gpu", "run_dir")} | cfg | {
        "delta_" + k: v for k, v in row.get("deltas_vfa", {}).items()} | row.get("regularity", {}) | {
        "diffusion": row.get("field_objectives", {}).get("selected", {}).get("diffusion"),
        "change_from_prefix_mean_mm": row.get("change_from_prefix_mean_mm")}


def pareto(rows):
    return [r["run_name"] for r in rows if not any(
        s["dice"] >= r["dice"] and s["hd95"] <= r["hd95"] and s["asd"] <= r["asd"] and
        (s["dice"] > r["dice"] or s["hd95"] < r["hd95"] or s["asd"] < r["asd"]) for s in rows)]


def report_search(args, rows, winner=None, control=None):
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "rank0_candidates.csv", [flat_row(r) for r in rows])
    features = [r for r in rows if r["config"]["loss"] != "raw_ncc"]
    payload = {"status": "search_complete" if winner else "phase_complete_search_ongoing",
               "candidate_count": len(rows), "rank0_fit_ceiling": 96, "tolerances": f.TOLERANCES,
               "reference": f.REFERENCE, "candidates": rows,
               "selected_feature": winner, "selected_control": control,
               "pareto_run_names": pareto(features), "rank0_is_exploratory_test_case_tuning": True}
    lines = ["# OASIS rank-0 VFA-feature dense refinement", "",
             "Rank 0 is exploratory test-case tuning. Timings are diagnostic under shared GPU workloads.", "",
             f"Status: {payload['status']}; completed candidates: {len(rows)}/96 maximum.", "",
             "| Phase | Order | Loss | Support | Lambda | LR | Dice | HD95 | ASD | Match | Epoch | GPU |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        cfg = r["config"]
        lines.append(f"| {r['phase']} | {cfg['order']} | {cfg['loss']} | {cfg['support']} | {cfg['alpha']:.6g} | {cfg['lr']:.4g} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} | {r['matches_vfa']} | {r['best_epoch']} | {r['gpu']} |")
    if winner:
        lines += ["", f"Selected feature: `{winner['run_name']}`. Meets all tolerances: **{winner['matches_vfa']}**.",
                  f"Worst tolerance-normalized deficit: {winner['metric_deficit']:.6f} (match requires <=1)."]
        prepared = c.read_json(Path(winner["prefix_cache"]) / "summary.json")
        comparisons = [("Original VFA", prepared["vfa_metrics"]),
                       ("Reproduced camera-ready 8-4-2 prefix", prepared["prefix_metrics"]),
                       ("Historical MR-D-BSCP 8-4-2 reference", {"dice": .7607446419879308,
                        "hd95": 2.65234193162559, "asd": .5925002179651616}), ("Selected feature", winner),
                       ("Selected raw NCC + diffusion", control)]
        prior = historical_be_row()
        if prior:
            comparisons.append(("Previous raw NCC + BE (lambda100)", prior))
        payload["selected_comparison"] = [
            {"method": method, **{k: r[k] for k in ("dice", "hd95", "asd")},
             **{"delta_" + k: r[k] - f.REFERENCE[k] for k in f.REFERENCE}} for method, r in comparisons]
        c.write_csv(args.review_root / "rank0_selected_comparison.csv", [
            {"method": method, **{k: r[k] for k in ("dice", "hd95", "asd")},
             **{"delta_" + k: r[k] - f.REFERENCE[k] for k in f.REFERENCE}}
            for method, r in comparisons])
        lines += ["", "## Selected comparison", "", "| Method | Dice | HD95 | ASD |",
                  "|---|---:|---:|---:|"]
        for method, r in comparisons:
            lines.append(f"| {method} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} |")
    c.write_json(args.review_root / "rank0_search.json", payload)
    (args.review_root / "rank0_search.md").write_text("\n".join(lines) + "\n")


def historical_be_row():
    path = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline")) / "repo_experiments" / "oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1/comparison.csv"
    if not path.exists():
        return None
    matches = [r for r in c.read_csv(path) if r.get("kind") == "dense_candidate" and float(r.get("dense_alpha") or -1) == 100]
    if len(matches) != 1:
        raise RuntimeError("Historical raw-NCC/BE lambda100 reference is ambiguous")
    return {"source": str(path), **matches[0], **{k: float(matches[0][k]) for k in ("dice", "hd95", "asd")}}


def report_cohort(args, scheduler, feature_rows, control_rows):
    comparison = []
    for rank in range(10):
        prepared = c.read_json(scheduler.cached(rank) / "summary.json")
        reference = prepared["vfa_metrics"]
        for method, values in (("VFA", reference), ("MR-D-BSCP 8-4-2", prepared["prefix_metrics"]),
                               ("Dense VFA features", feature_rows[rank]), ("Dense raw NCC + diffusion", control_rows[rank])):
            row = {"method": method, "case_rank": rank, "fixed_id": prepared["fixed_id"], "moving_id": prepared["moving_id"],
                   **{k: values[k] for k in ("dice", "hd95", "asd")},
                   **{"delta_" + k: values[k] - reference[k] for k in reference},
                   "matches_vfa": f.metric_deficit(values, reference) <= 1}
            if "config" in values:
                row.update({k: v for k, v in flat_row(values).items() if k not in row})
            else:
                row["training_elapsed_sec"] = prepared["prefix_elapsed_sec"] if method.startswith("MR-") else None
            comparison.append(row)
    means = []
    for first in (1, 0):
        for method in dict.fromkeys(r["method"] for r in comparison):
            subset = [r for r in comparison if r["method"] == method and r["case_rank"] >= first]
            values = {k: float(np.mean([r[k] for r in subset])) for k in ("dice", "hd95", "asd")}
            ref_rows = [r for r in comparison if r["method"] == "VFA" and r["case_rank"] >= first]
            ref = {k: float(np.mean([r[k] for r in ref_rows])) for k in values}
            means.append({"subset": f"ranks_{first}_9", "method": method, "n": len(subset), **values,
                          **{"delta_" + k: values[k] - ref[k] for k in values},
                          "cases_matching_all": sum(r["matches_vfa"] for r in subset),
                          "mean_matches_all": f.metric_deficit(values, ref) <= 1})
    c.write_csv(args.review_root / "first10_case_metrics.csv", comparison)
    c.write_csv(args.review_root / "first10_means.csv", means)
    prior = historical_be_row()
    report = {"status": "complete", "case_metrics": comparison, "means": means,
              "historical_rank0_raw_ncc_be_lambda100": prior,
              "rank0_selection_frozen_before_ranks1_9": True, "no_claim_of_unbiased_test_tuning": True}
    c.write_json(args.review_root / "first10_report.json", report)
    lines = ["# OASIS first10: frozen VFA-feature dense refinement", "",
             "Rank 0 was used for exploratory selection. Ranks 1–9 were not used for further tuning.", "",
             "## Means", "", "| Subset | Method | Dice | HD95 | ASD | Matching cases | Mean matches |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in means:
        lines.append(f"| {r['subset']} | {r['method']} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} | {r['cases_matching_all']}/{r['n']} | {r['mean_matches_all']} |")
    lines += ["", "## Per case", "", "| Rank | Method | Dice | HD95 | ASD | ΔDice | ΔHD95 | ΔASD | Match |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in comparison:
        lines.append(f"| {r['case_rank']} | {r['method']} | {r['dice']:.6f} | {r['hd95']:.6f} | {r['asd']:.6f} | {r['delta_dice']:+.6f} | {r['delta_hd95']:+.6f} | {r['delta_asd']:+.6f} | {r['matches_vfa']} |")
    lines += ["", "Runtime is diagnostic because existing GPU workloads remained active.",
              "Folding, diffusion, stopping diagnostics and GPU assignments are in the companion CSV/JSON."]
    (args.review_root / "first10_report.md").write_text("\n".join(lines) + "\n")
    c.write_json(args.review_root / "ACCEPTED.json", {"technical_completion": True,
        "rank0_match": feature_rows[0]["matches_vfa"], "selected_feature_count": 10, "selected_control_count": 10,
        "ranks1_9_mean_match": next(r["mean_matches_all"] for r in means if r["subset"] == "ranks_1_9" and r["method"] == "Dense VFA features")})


def execute(args):
    scheduler = Scheduler(args)
    scheduler.run([job()])
    if args.stop_after == "prepare":
        scheduler.state["phase"] = "prepare_passed"
        scheduler.save()
        return
    smoke_configs = [f.FitConfig(order=o, loss=l, max_epochs=1, min_epochs=0)
                     for o, l in itertools.product(f.ORDERS, ("mse", "global_ncc"))]
    scheduler.run([job(cfg, phase="smoke") for cfg in smoke_configs])
    scheduler.state["smoke_passed"] = True
    scheduler.save()
    if args.stop_after == "smoke":
        scheduler.state["phase"] = "smoke_passed"
        scheduler.save()
        return
    rows = []
    for phase, builder in (("A", lambda _: phase_a()), ("B", phase_b), ("C", phase_c)):
        if rows and any(r["matches_vfa"] for r in rows):
            break
        scheduler.state["phase"] = phase
        scheduler.save()
        specs = [job(cfg, phase=phase) for cfg in builder(rows)]
        rows.extend(scheduler.run(specs))
        if len(rows) > 92:
            raise RuntimeError("Feature search ceiling exceeded")
        report_search(args, rows)
    winner = min(rows, key=f.selection_key)
    control_rows = scheduler.run([job(cfg, phase="controls") for cfg in controls(rows, winner)])
    control = min(control_rows, key=f.selection_key)
    if len(rows) + len(control_rows) > 96:
        raise RuntimeError("Rank-0 search ceiling exceeded")
    report_search(args, rows + control_rows, winner, control)
    frozen = {"feature_config": winner["config"], "control_config": control["config"],
              "feature_rank0_run": winner["run_dir"], "control_rank0_run": control["run_dir"],
              "rank0_matches": winner["matches_vfa"], "rank0_fit_count": len(rows) + len(control_rows),
              "selection_rule": "meet all tolerances then Dice/ASD/HD95; otherwise worst normalized deficit"}
    freeze_path = args.review_root / "frozen_selection.json"
    if freeze_path.exists() and c.read_json(freeze_path) != frozen:
        raise RuntimeError("Refusing to change an already frozen selection")
    c.write_json(freeze_path, frozen)
    if args.stop_after == "search":
        scheduler.state["phase"] = "search_complete"
        scheduler.save()
        return
    scheduler.state["phase"] = "confirmation"
    scheduler.save()
    scheduler.run([job(rank=rank) for rank in range(1, 10)])
    specs = [job(f.FitConfig(**cfg), rank, "confirmation")
             for rank in range(1, 10) for cfg in (winner["config"], control["config"])]
    completed = scheduler.run(specs)
    features = [winner] + [r for r in completed if r["config"]["loss"] != "raw_ncc"]
    control_cohort = [control] + [r for r in completed if r["config"]["loss"] == "raw_ncc"]
    report_cohort(args, scheduler, features, control_cohort)
    scheduler.state["phase"] = "complete"
    scheduler.save()
    scheduler.event("experiment_complete", rank0_match=winner["matches_vfa"], review_root=str(args.review_root))


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps({"variant": f.VARIANT, "phase_a": [asdict(x) for x in phase_a()],
                          "max_rank0_fits": 96, "gpus": args.gpu_list, "args": vars(args)}, default=str, indent=2))
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("This approved experiment must run as container UID:GID 1009:1020")
    with (args.out_root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out_root / "launch_manifest.json"
        if not manifest_path.exists():
            c.write_json(manifest_path, {"args": vars(args), "source": source_metadata(),
                         "initial_gpu_state": gpu_snapshot(), "uid": os.getuid(), "gid": os.getgid(),
                         "checkpoint_sha256": sha256_file(args.checkpoint), "started_at": c.now_text()})
        c.write_json(args.out_root / "invocations" / f"{time.time_ns()}_{os.getpid()}.json",
                     {"args": vars(args), "source": source_metadata(),
                      "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
                      "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded"),
                      "time": c.now_text(), "pid": os.getpid()})
        with (args.out_root / "launcher.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                c.write_json(args.out_root / "current_status.json", {"status": "running", "pid": os.getpid(),
                             "time": c.now_text(), "stop_after": args.stop_after})
                execute(args)
                c.write_json(args.out_root / "current_status.json", {"status": "complete" if args.stop_after == "full" else "preflight_or_search_complete",
                             "time": c.now_text(), "stop_after": args.stop_after})
            except Exception:
                c.write_json(args.out_root / "LAUNCHER_FAILED.json", {"time": c.now_text(), "traceback": traceback.format_exc()})
                c.write_json(args.out_root / "launcher_failures" / f"{time.time_ns()}_{os.getpid()}.json",
                             {"time": c.now_text(), "traceback": traceback.format_exc()})
                c.write_json(args.out_root / "current_status.json", {"status": "failed", "time": c.now_text()})
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
