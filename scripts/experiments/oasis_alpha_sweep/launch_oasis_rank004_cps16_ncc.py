#!/usr/bin/env python
"""GPU3-only sequential smoke/production execution and complete-only P47 report."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
import cps16_ncc as n
import dl_ncc_diffusion as d
from run_oasis_rank004_cps16_ncc_one import source_hashes
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file


def read_labels(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or (rows[0]["fixed_id"], rows[0]["moving_id"]) != ("0084", "0108"):
        raise RuntimeError(f"Unexpected label metric identity: {path}")
    return rows[0]


def verify_label_means(row, path):
    labels = read_labels(path)
    for metric, prefix in (("dice", "dice"), ("hd95", "hd95"), ("asd", "asd_symmetric")):
        values = [float(labels[f"{prefix}_label_{i:02d}"]) for i in c.OASIS_LABELS]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("Expected all35 present rank4 labels; do not silently drop any")
        if not math.isclose(float(np.mean(values)), row[metric], abs_tol=1e-12, rel_tol=1e-12):
            raise RuntimeError("Canonical label mean differs from summary")
    return labels


def references(args):
    source = args.reference_root/"results.json"
    accepted = args.reference_root/"ACCEPTED.json"
    payload = c.read_json(source)
    if not payload["complete"] or not c.read_json(accepted)["technical_completion"]:
        raise RuntimeError("P43 references are incomplete")
    hashes = {str(p): sha256_file(p) for p in (source, accepted)}
    selected = []
    current = source_hashes()
    for row in payload["case_metrics"]:
        if row["case_rank"] != 4:
            continue
        is_old = row["representation"] == "842_dense" and row["alpha"] == 1.
        is_dl = row["representation"] in ("VFA", "SITReg")
        if not (is_old or is_dl):
            continue
        record = dict(row)
        if is_old:
            run = Path(row["run_dir"])
            summary_path = run/"summary.json"
            summary = c.read_json(summary_path)
            if not (run/"DONE").is_file():
                raise RuntimeError("Incomplete saved optimization reference")
            for key, value in summary["source_hashes"].items():
                if current.get(key) != value:
                    raise RuntimeError(f"Historical numerical source drift: {key}")
            for key in ("dvf_path", "model_path"):
                path = Path(summary[key])
                digest = sha256_file(path)
                if digest != summary[key.replace("_path", "_sha256")]:
                    raise RuntimeError("Historical field/checkpoint checksum mismatch")
                hashes[str(path)] = digest
            hashes[str(summary_path)] = sha256_file(summary_path)
            record.update(name=f"842_dense_{row['resolution']}", source_kind="saved_P43",
                          training_elapsed_sec=summary["pipeline_training_elapsed_sec"],
                          input_hashes=summary["input_hashes"],
                          pair_file_sha256=summary["pair_file_sha256"],
                          native_source_hashes=summary["native_source_hashes"])
            label_path = run/"case_metrics.csv"
        else:
            record.update(name=row["representation"], source_kind="original_DL")
            label_path = args.baseline_root/f"{row['representation'].lower()}_rank004_labels.csv"
            audit_path = args.baseline_root/f"{row['representation'].lower()}_rank004.json"
            audit = c.read_json(audit_path)
            field_path = Path(audit["row"]["source_path"])
            digest = sha256_file(field_path)
            if digest != audit["row"]["source_sha256"]:
                raise RuntimeError("Original DL reference field changed")
            hashes[str(field_path)] = digest
            hashes[str(audit_path)] = sha256_file(audit_path)
        record["case_metrics_path"] = str(label_path)
        hashes[str(label_path)] = sha256_file(label_path)
        verify_label_means(record, label_path)
        selected.append(record)
    if {r["name"] for r in selected} != {"842_dense_full", "842_dense_pyramid", "VFA", "SITReg"}:
        raise RuntimeError("Require exactly four rank4 references")
    if len(selected) != 4:
        raise RuntimeError("Duplicate rank4 reference")
    return selected, hashes


def validate_result(path, spec, hashes, refs):
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing worker completion marker")
    row = c.read_json(path/"summary.json")
    expected = {"spec": asdict(spec), "protocol": n.PROTOCOL, "source_hashes": hashes,
                "method": n.METHOD, "variant": n.VARIANT, "gpu": "3", "case_rank": 4,
                "fixed_id": "0084", "moving_id": "0108", "direction": "moving_to_fixed",
                "representation": "dense_voxel", "parameter_units": "normalized_grid_xyz",
                "cps": None, "control_shape": None, "bspline_order": None,
                "uid": 1009, "gid": 1020, "tf32_matmul": False, "tf32_cudnn": False,
                "segmentation_loss": False, "feature_loss": False, "reference_dvf_supervision": False,
                "valid_label_count": 35, "excluded_labels": []}
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"Invalid worker metadata: {key}")
    backend = row["cuda_backend"]
    if (backend["cuda_visible_devices"] != "3" or backend["cuda_device_index"] != 0 or
            backend["cudnn_benchmark"] or not backend["cudnn_deterministic"]):
        raise RuntimeError("Wrong physical/logical device or CUDA backend")
    ref = next(r for r in refs if r["name"] == f"842_dense_{spec.resolution}")
    for key in ("input_hashes", "pair_file_sha256", "native_source_hashes"):
        if row[key] != ref[key]:
            raise RuntimeError(f"New/saved reference mismatch: {key}")
    if len(row["stages"]) != 5 or len(row["stage_diagnostics"]) != 5 or len(row["initialization_checks"]) != 4:
        raise RuntimeError("Incomplete five-stage record")
    for info, stage, metric in zip(row["stages"], n.stages(spec), row["stage_diagnostics"]):
        if info["stage"] != asdict(stage) or info["alpha"] != 1.:
            raise RuntimeError("Wrong schedule or diffusion lambda")
        shape = c.downsample_shape(c.IMAGE_SHAPE, stage.factor)
        if tuple(info["evaluation_shape"]) != shape or info["objective_voxel_count"] != math.prod(shape):
            raise RuntimeError("Incorrect loss grid")
        if not 0 <= info["best_epoch"] <= info["actual_epochs"] <= stage.max_epochs:
            raise RuntimeError("Invalid checkpoint epoch")
        if info["best_loss"] > min(info["initial_loss"], info["last_loss"])+1e-6:
            raise RuntimeError("Best checkpoint invariant failed")
        if info["lr_reductions"] > stage.max_lr_reductions:
            raise RuntimeError("LR reductions applied to wrong stage")
        field = np.load(metric["dvf_path"], mmap_mode="r")
        if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
            raise RuntimeError("Invalid stage field")
        if sha256_file(Path(metric["dvf_path"])) != metric["dvf_sha256"]:
            raise RuntimeError("Stage field checksum mismatch")
        if not all(math.isfinite(metric[k]) for k in n.METRICS):
            raise RuntimeError("Nonfinite stage metrics")
        verify_label_means(metric, Path(metric["case_metrics_path"]))
    for check in row["initialization_checks"]:
        if check["max_abs_voxel"] > 1e-4 or check["rms_voxel"] > 1e-5:
            raise RuntimeError("Knot insertion changed field")
    last = row["initialization_checks"][-1]
    if last["kind"] != "exact_cps2_dense_clone" or last["max_abs_voxel"] != 0.:
        raise RuntimeError("Dense clone not exact")
    if sha256_file(Path(row["model_path"])) != row["model_sha256"]:
        raise RuntimeError("Final checkpoint checksum mismatch")
    if spec.smoke and not row["native_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    verify_label_means(row, Path(row["case_metrics_path"]))
    return row


def write_report(results, refs, reference_hashes, root):
    if len(results) != 2 or {r["spec"]["resolution"] for r in results} != {"pyramid", "full"}:
        raise RuntimeError("Require both successful production runs before aggregation")
    rows = []
    for r in refs+results:
        name = r.get("name", f"16842_dense_{r['spec']['resolution']}" if "spec" in r else "")
        rows.append({"name": name, "case_rank": 4, "fixed_id": "0084", "moving_id": "0108",
                     **{k: r[k] for k in n.METRICS}, "gpu": r.get("gpu"),
                     "training_elapsed_sec": r.get("training_elapsed_sec"),
                     "stopping_epochs": "/".join(str(s["actual_epochs"]) for s in r["stages"]) if "stages" in r else r.get("stage_actual_epochs", ""),
                     "best_epochs": "/".join(str(s["best_epoch"]) for s in r["stages"]) if "stages" in r else r.get("stage_best_epochs", ""),
                     "case_metrics_path": r["case_metrics_path"]})
    lookup = {r["name"]: r for r in rows}
    deltas, label_rows, label_deltas, stage_rows = [], [], [], []
    for row in rows:
        labels = read_labels(Path(row["case_metrics_path"]))
        for label in c.OASIS_LABELS:
            label_rows.append({"name": row["name"], "label": label,
                               **{m: float(labels[f"{p}_label_{label:02d}"]) for m, p in
                                  (("dice", "dice"), ("hd95", "hd95"), ("asd", "asd_symmetric"))}})
    for result in results:
        mode = result["spec"]["resolution"]
        name = f"16842_dense_{mode}"
        for baseline in ("842_dense_full", "842_dense_pyramid", "VFA", "SITReg"):
            deltas.append({"name": name, "baseline": baseline,
                           **{k: result[k]-lookup[baseline][k] for k in n.METRICS}})
            for label in c.OASIS_LABELS:
                a = next(r for r in label_rows if r["name"] == name and r["label"] == label)
                b = next(r for r in label_rows if r["name"] == baseline and r["label"] == label)
                label_deltas.append({"name": name, "baseline": baseline, "label": label,
                                     **{k: a[k]-b[k] for k in ("dice", "hd95", "asd")}})
        for info, metric in zip(result["stages"], result["stage_diagnostics"]):
            stage_rows.append({"name": name, "stage": info["stage"]["index"], "cps": info["cps"],
                               "actual_epochs": info["actual_epochs"], "best_epoch": info["best_epoch"],
                               "stopping_reason": info["stopping_reason"],
                               "training_elapsed_sec": info["training_elapsed_sec"],
                               "best_loss": info["best_loss"], "best_data_loss": info["best_data_loss"],
                               "best_reg_loss": info["best_reg_loss"],
                               **{k: metric[k] for k in n.METRICS}, "dvf_path": metric["dvf_path"],
                               "case_metrics_path": metric["case_metrics_path"]})
    root.mkdir(parents=True, exist_ok=True)
    for filename, data in (("comparison.csv", rows), ("paired_deltas.csv", deltas),
                           ("per_label.csv", label_rows), ("per_label_deltas.csv", label_deltas),
                           ("stages.csv", stage_rows)):
        c.write_csv(root/filename, data)
    payload = {"complete": True, "variant": n.VARIANT, "protocol": n.PROTOCOL,
               "case_metrics": rows, "paired_deltas": deltas, "per_label": label_rows,
               "per_label_deltas": label_deltas, "stages": stage_rows, "candidates": results,
               "references": refs, "reference_hashes": reference_hashes,
               "limitations": ["Exploratory single-case rescue, not a first10 improvement claim",
                               "Extra stage adds compute; not an equal-budget comparison",
                               "Pyramid changes NCC physical support and regularizer sampling",
                               "GPU3 shared workloads retained; runtime diagnostic only"]}
    c.write_json(root/"results.json", payload)
    lines = ["# Rank4 cps16 NCC9/diffusion experiment", "",
             "Fixed0084, moving0108. GPU3 only, sequential pyramid then full. Lambda1 throughout.", "",
             "Exploratory single case; extra optimization work and pyramid objectives differ. No across-case SD.", "",
             "| Method | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| "+row["name"]+" | "+" | ".join(f"{row[k]:.6f}" for k in n.METRICS[:5])+" |")
    lines += ["", "## New minus saved reference", "", "| New | Reference | Dice | HD95 | ASD | Folding percentage points |",
              "|---|---|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append(f"| {row['name']} | {row['baseline']} | "+" | ".join(
            f"{row[k]:+.6f}" for k in ("dice", "hd95", "asd", "jdet_le0_pct_fg"))+" |")
    lines += ["", "## Stage diagnostics", "", "| Method | Stage/CPS | Best/last epochs | Dice | HD95 | ASD | FG folding % |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in stage_rows:
        lines.append(f"| {row['name']} | {row['stage']}/{row['cps'] or 'dense'} | {row['best_epoch']}/{row['actual_epochs']} | "+
                     " | ".join(f"{row[k]:.6f}" for k in ("dice", "hd95", "asd", "jdet_le0_pct_fg"))+" |")
    lines += ["", "All35 label metrics and paired differences are in per_label.csv/per_label_deltas.csv.",
              "Native stage fields and label metrics are linked in stages.csv. Selection uses loss only.",
              "Foreground is fixed raw image>0 with the canonical Jacobian crop; log spread uses positive J only.", ""]
    (root/"report.md").write_text("\n".join(lines))
    c.write_json(root/"ACCEPTED.json", {"technical_completion": True, "production_count": 2,
                                       "reference_count": 4, "smoke_count": 2, "completed_at": c.now_text()})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    artifact = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    p.add_argument("--out-root", type=Path, default=artifact/"runs"/n.VARIANT)
    p.add_argument("--review-root", type=Path, default=artifact/"repo_experiments"/n.VARIANT)
    p.add_argument("--reference-root", type=Path, default=artifact/"repo_experiments"/d.VARIANT)
    p.add_argument("--baseline-root", type=Path, default=artifact/"repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1/baseline_cases")
    for name in ("oasis-root", "dl-code-root", "vfa-root"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--gpu", choices=("3",), default="3")
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    args = p.parse_args(argv)
    outputs = [args.out_root.resolve(), args.review_root.resolve()]
    inputs = [c.REPO_ROOT, args.oasis_root, args.dl_code_root, args.vfa_root, args.reference_root, args.baseline_root]
    if args.cpu_threads < 1 or any(o.is_relative_to(s.resolve()) or s.resolve().is_relative_to(o)
                                 for o in outputs for s in inputs):
        p.error("Output roots must be external and disjoint from sources/references")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        p.error("Run/review directories must be disjoint")
    return args


def execute(args):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = source_hashes()
        refs, reference_hashes = references(args)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=c.REPO_ROOT, text=True).strip():
            revision += "-dirty"
        state_path = args.out_root/"launcher_state.json"
        state = c.read_json(state_path) if state_path.exists() else {
            "phase": "preflight", "source_hashes": hashes, "reference_hashes": reference_hashes,
            "results": {}, "attempts": [], "gpu": "3", "max_concurrent": 1,
            "source_revision": revision, "started_at": c.now_text(), "args": vars(args)}
        if state["source_hashes"] != hashes or state["reference_hashes"] != reference_hashes:
            raise RuntimeError("Source/reference drift on resume; preserve this experiment")
        def save():
            temporary = state_path.with_suffix(".tmp")
            c.write_json(temporary, state); temporary.replace(state_path)
        save()
        (args.out_root/"launcher.pid").write_text(str(os.getpid())+"\n")
        smoke_results, production = [], []
        for smoke in (True, False):
            state["phase"] = "smoke" if smoke else "production"; save()
            for mode in ("pyramid", "full"):
                spec = n.FitSpec(mode, smoke)
                path = args.out_root/"jobs"/spec.name()/"attempt0"
                if not (path/"DONE").is_file():
                    if path.exists():
                        raise RuntimeError(f"Preserved incomplete attempt requires review: {path}")
                    required = max(8192., max((max(s["peak_cuda_gib"] for s in r["stages"])
                                              for r in smoke_results), default=0.)*1024+4096)
                    while True:
                        snapshot = gpu_snapshot()
                        if snapshot["error"]:
                            raise RuntimeError(snapshot["error"])
                        gpu = next(r for r in snapshot["rows"] if r["physical_gpu"] == "3")
                        if gpu["memory_free_mib"] >= required:
                            break
                        print(json.dumps({"waiting_for_gpu3_memory_mib": required, "current": gpu}), flush=True)
                        time.sleep(10)
                    if source_hashes() != hashes:
                        raise RuntimeError("Source drift before launch")
                    cmd = [sys.executable, "-B", str(HERE/"run_oasis_rank004_cps16_ncc_one.py"),
                           "--resolution", mode, "--gpu", "3", "--out-dir", str(path),
                           "--source-revision", revision, "--cpu-threads", str(args.cpu_threads)]
                    for key in ("oasis_root", "dl_code_root", "vfa_root", "pair_file"):
                        cmd += ["--"+key.replace("_", "-"), str(getattr(args, key))]
                    if smoke:
                        cmd.append("--smoke")
                    log_path = args.out_root/"logs"/(spec.name()+".log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    record = {"name": spec.name(), "command": cmd, "path": str(path), "log_path": str(log_path),
                              "initial_gpu_state": snapshot, "started_at": c.now_text(), "status": "running"}
                    state["attempts"].append(record); save()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
                    print(json.dumps({"launch": spec.name(), "gpu": "3", "required_free_mib": required}), flush=True)
                    with log_path.open("x") as log:
                        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                        record["pid"] = process.pid; save()
                        code = process.wait()  # Strictly one worker; never launches the next while this is alive.
                    record.update(exit_code=code, completed_at=c.now_text(), status="complete" if code == 0 else "failed")
                    save()
                    if code:
                        state["phase"] = "failed"; save()
                        raise RuntimeError(f"Worker failed; logs preserved and no incomplete aggregation: {log_path}")
                row = validate_result(path, spec, hashes, refs)
                state["results"][spec.name()] = str(path); save()
                (smoke_results if smoke else production).append(row)
                print(json.dumps({"complete": spec.name(), **{k: row[k] for k in n.METRICS}}), flush=True)
            if smoke:
                args.review_root.mkdir(parents=True, exist_ok=True)
                c.write_json(args.review_root/"smoke_results.json", {"passed": True, "results": smoke_results})
                state["phase"] = "smoke_passed"; save()
                if args.stop_after == "smoke":
                    return
        if source_hashes() != hashes or references(args)[1] != reference_hashes:
            raise RuntimeError("Source/reference changed during execution")
        write_report(production, refs, reference_hashes, args.review_root)
        state["phase"] = "complete"; state["completed_at"] = c.now_text(); save()
        print(json.dumps({"experiment_complete": n.VARIANT, "report": str(args.review_root/"report.md")}), flush=True)


if __name__ == "__main__":
    execute(parse_args())
