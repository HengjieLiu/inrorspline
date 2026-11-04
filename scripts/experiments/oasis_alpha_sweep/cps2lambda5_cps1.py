#!/usr/bin/env python
"""Rank-0 cps1 versus saved dense refinement after the same [1,1,5] prefix."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import cps2lambda5_dense as previous

c, d, legacy = previous.c, previous.d, previous.legacy
sha, atomic_json = previous.sha, previous.atomic_json
METRICS = previous.METRICS
VARIANT = "oasis_seed3_rank000_mrdbscp8421_lambda1151_vs_dense_v1"
METHOD = "direct_cp_8_4_2_1_lambda1151_ncc9_fd_diffusion"


def stage_for(smoke=False):
    return d.stages(d.FitSpec(0, "pyramid", 1., "8421", smoke=smoke))[-1]


def run_name(smoke=False):
    return ("smoke_" if smoke else "") + "rank000_bspline8421_lambda1_1_5_1"


def sources():
    return {"previous": previous.sources(), "cps2lambda5_cps1.py": sha(Path(__file__))}


def assert_sources(manifest):
    if manifest["source_hashes"] != sources():
        raise RuntimeError("Numerical sources changed; preserve run and audit")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--artifact-root", type=Path, required=True)
    launch.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--out-dir", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    for parser in (launch, worker):
        parser.add_argument("--gpu", choices=("0", "1", "3"), default="3")
    args = p.parse_args(argv)
    output = args.artifact_root if args.command == "launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Outputs must be outside the source repository")
    return args


def preflight(args):
    report_root = args.artifact_root / "repo_experiments" / previous.VARIANT
    data = previous.accepted_data(report_root)
    old_manifest_path = args.artifact_root / "runs" / previous.VARIANT / "launch_manifest.json"
    old = c.read_json(old_manifest_path)
    previous.assert_sources(old)
    candidates = [r for r in data["candidates"] if r["case_rank"] == 0 and r["spec"]["dense_alpha"] == 1.]
    if len(candidates) != 1:
        raise RuntimeError("Missing or ambiguous rank0 dense reference")
    dense = previous.validate(Path(candidates[0]["run_dir"]), previous.Spec(0, 1.), old)
    if dense != candidates[0]:
        raise RuntimeError("Dense report differs from its validated run")
    prefix = old["prefixes"]["0"]
    p61_manifest = c.read_json(args.artifact_root / "runs" / previous.p61.VARIANT / "launch_manifest.json")
    previous.p61.assert_sources(p61_manifest)
    checked = previous.p61.validate(Path(prefix["run_dir"]), previous.p61.Spec(0, 5.), p61_manifest)
    if checked != prefix or prefix["stage_alpha"] != [1., 1., 5.]:
        raise RuntimeError("Prefix protocol mismatch")
    if (prefix["fixed_id"], prefix["moving_id"], prefix["direction"]) != ("0333", "0115", "moving_to_fixed"):
        raise RuntimeError("Wrong case or direction")
    a, b = asdict(stage_for()), dense["stages"][-1]["stage"].copy()
    a.pop("cps"); b.pop("cps")
    if a != b:
        raise RuntimeError("Final schedules differ beyond representation")
    refs = [r for r in old["references"] if r["rank"] == 0]
    if len(refs) != 3 or any((r["fixed_id"], r["moving_id"]) != ("0333", "0115") for r in refs):
        raise RuntimeError("Reference case mismatch")
    return {"variant": VARIANT, "method": METHOD, "source_hashes": sources(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
            "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=c.REPO_ROOT, text=True).strip(),
            "settings": old["settings"], "prefix": prefix, "dense_reference": dense, "references": refs,
            "source_report_sha256": sha(report_root / "results.json"),
            "source_manifest_sha256": sha(old_manifest_path), "source_manifest_path": str(old_manifest_path),
            "source_report_path": str(report_root / "results.json"),
            "gpu": args.gpu, "initial_gpu_state": legacy.gpu_snapshot(), "created_at": c.now_text(),
            "container_image": os.environ["INRORSPLINE_CONTAINER_IMAGE"],
            "container_image_id": os.environ["INRORSPLINE_CONTAINER_IMAGE_ID"]}


def worker(args):
    manifest = c.read_json(args.manifest); assert_sources(manifest)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID 1009:1020")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); initial_gpu = legacy.gpu_snapshot()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    args.case_rank = 0
    for key in ("oasis_root", "pair_file", "dl_code_root", "vfa_root"):
        setattr(args, key, Path(manifest["settings"][key]))
    batch, input_hashes = legacy.load_batch(args, device)
    prefix = manifest["prefix"]
    if input_hashes != prefix["input_hashes"] or sha(args.pair_file) != prefix["pair_file_sha256"]:
        raise RuntimeError("Input/split changed")
    if not (Path(prefix["run_dir"]) / "DONE").is_file():
        raise RuntimeError("Incomplete prefix")
    for key in ("model", "dvf"):
        if sha(Path(prefix[key + "_path"])) != prefix[key + "_sha256"]:
            raise RuntimeError("Prefix artifact changed")
    saved = torch.load(prefix["model_path"], map_location="cpu", weights_only=False)
    if saved["summary"] != prefix["stages"][-1]:
        raise RuntimeError("Prefix model summary mismatch")
    controls = saved["parameter_norm_xyz"].to(device)
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, 8, [8, 4, 2, 1])
    if tuple(controls.shape) != (*shapes[2], 3) or not bool(torch.isfinite(controls).all()):
        raise RuntimeError("Invalid cps2 controls")
    initial_field = torch.from_numpy(np.load(prefix["dvf_path"])).to(device)
    stage = stage_for(args.smoke)
    objective = d.StageObjective(batch["fixed"], batch["moving"], 1., stage, shapes[1])
    with torch.no_grad():
        coarse_field = d.AlignedBSpline(shapes[2], c.IMAGE_SHAPE, c.IMAGE_SHAPE, 2, device)(controls)
        reconstruction = legacy.verify_init(initial_field, coarse_field)
        initial = c.cubic_bspline_dyadic_knot_insert_3d(controls, shapes[1]).detach().clone()
        fine_field = objective.field(initial)
        prolongation = legacy.verify_init(initial_field, fine_field)
        dense_objective = previous.objective_for(batch["fixed"], batch["moving"], previous.Spec(0, 1., smoke=args.smoke))
        spline_loss = [float(x) for x in objective(initial)]
        dense_loss = [float(x) for x in dense_objective(initial_field)]
        same_field_loss = [float(x) for x in dense_objective(fine_field)]
        if spline_loss != same_field_loss or not np.allclose(spline_loss, dense_loss, rtol=1e-6, atol=1e-7):
            raise RuntimeError("Cps1/dense initialization loss mismatch")
        if not math.isclose(dense_loss[0], manifest["dense_reference"]["stages"][-1]["initial_loss"], rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("Historical dense initial objective mismatch")
    del saved, controls, coarse_field, fine_field, dense_objective, initial_field
    row = {"variant": VARIANT, "method": METHOD, "run_name": run_name(args.smoke), "smoke": args.smoke,
           "stage_alpha": [1., 1., 5., 1.], "protocol": d.PROTOCOL, "source_hashes": manifest["source_hashes"],
           "source_revision": manifest["source_revision"], "source_branch": manifest["source_branch"],
           "case_rank": 0, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
           "direction": "moving_to_fixed", "sampling": "fixed_to_moving_pull_identity_plus_displacement",
           "input_hashes": input_hashes, "pair_file_sha256": sha(args.pair_file), "gpu": args.gpu,
           "visible_cuda_device": "cuda:0", "initial_gpu_state": initial_gpu,
           "cuda_backend": c.cuda_backend_metadata(device), "tf32_matmul": False, "tf32_cudnn": False,
           "uid": os.getuid(), "gid": os.getgid(), "seed": 1,
           "container_image": manifest["container_image"], "container_image_id": manifest["container_image_id"],
           "cuda_nondeterminism_caveat": "grid_sample backward may remain nondeterministic",
           "representation": "bspline_control", "parameter_units": "normalized_grid_xyz",
           "cps": 1, "bspline_order": 3, "control_shape": list(shapes[1]),
           "prefix_path": prefix["run_dir"], "prefix_dvf_sha256": prefix["dvf_sha256"],
           "prefix_model_sha256": prefix["model_sha256"],
           "initialization_checks": {"cps2_reconstruction": reconstruction, "cps2_to_cps1": prolongation,
                                     "spline_loss_data_reg": spline_loss, "dense_loss_data_reg": dense_loss,
                                     "same_field_objective_exact": True},
           "regularizer_backend": "finite_difference_on_expanded_total_voxel_field",
           "segmentation_loss": False, "analytic_bspline_regularizer": False,
           "started_at": c.now_text(), "argv": sys.argv}
    if args.smoke:
        row["native_parity"] = legacy.native_parity(device, args.dl_code_root, args.vfa_root)
    atomic_json(args.out_dir / "manifest.json", row)
    best, info = d.optimize_stage(initial, objective, stage, args.out_dir / "stage3")
    with torch.no_grad():
        field = objective.field(best)
    del objective, initial, best
    atomic_json(args.out_dir / "progress.json", {"phase": "canonical_metrics", "optimization_seconds": info["training_elapsed_sec"]})
    field_path = args.out_dir / "disp_norm_0333_0115.npy"
    np.save(field_path, field.detach().cpu().numpy())
    restored = torch.from_numpy(np.load(field_path)).to(device)
    if restored.shape != (*c.IMAGE_SHAPE, 3) or not bool(torch.isfinite(restored).all()) or not torch.equal(field, restored):
        raise RuntimeError("Invalid field serialization")
    vox = restored * d.voxel_scale(c.IMAGE_SHAPE, device=device)
    metrics, labels = legacy.canonical_metrics(vox, batch)
    labels["canonical_legacy_global_ncc_diagnostic"] = labels.pop("dense_image_loss")
    c.write_csv(args.out_dir / "case_metrics.csv", [labels])
    zyx = vox.detach().cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = c.jacobian_metrics_voxel(zyx, batch["fixed_mask"].cpu().numpy().astype(bool))
    diffusion = float(c.forward_valid_diffusion(vox.permute(3, 0, 1, 2)[None], None))
    if not math.isclose(diffusion, info["best_reg_loss"], rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Saved field regularizer differs from saved controls")
    excluded = [label for label in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    row.update(**metrics, **regularity, diffusion=diffusion, stages=prefix["stages"] + [info],
               best_loss=info["best_loss"], excluded_labels=excluded, valid_label_count=len(c.OASIS_LABELS)-len(excluded),
               dvf_path=str(field_path), dvf_sha256=sha(field_path), model_path=str(args.out_dir / "stage3/best.pt"),
               model_sha256=sha(args.out_dir / "stage3/best.pt"), training_elapsed_sec=info["training_elapsed_sec"],
               pipeline_training_elapsed_sec=sum(s["training_elapsed_sec"] for s in prefix["stages"])+info["training_elapsed_sec"],
               total_elapsed_sec=time.monotonic()-started, run_dir=str(args.out_dir), completed_at=c.now_text())
    assert_sources(manifest)
    atomic_json(args.out_dir / "summary.json", row)
    (args.out_dir / "DONE").write_text(c.now_text() + "\n")


def validate(path, manifest, smoke=False):
    if not (path / "DONE").is_file():
        raise RuntimeError("Missing completion marker")
    row = c.read_json(path / "summary.json"); prefix = manifest["prefix"]
    expected = {"method": METHOD, "variant": VARIANT, "smoke": smoke, "case_rank": 0, "seed": 1,
                "fixed_id": "0333", "moving_id": "0115", "direction": "moving_to_fixed",
                "stage_alpha": [1., 1., 5., 1.], "representation": "bspline_control", "cps": 1,
                "bspline_order": 3, "control_shape": [164, 228, 196], "protocol": d.PROTOCOL,
                "source_hashes": manifest["source_hashes"], "parameter_units": "normalized_grid_xyz",
                "uid": 1009, "gid": 1020, "gpu": manifest["gpu"], "segmentation_loss": False,
                "analytic_bspline_regularizer": False, "input_hashes": prefix["input_hashes"],
                "pair_file_sha256": prefix["pair_file_sha256"]}
    for key, value in expected.items():
        if row[key] != value:
            raise RuntimeError(f"Invalid {key}")
    if row["stages"][:3] != prefix["stages"] or any(row["prefix_" + k + "_sha256"] != prefix[k + "_sha256"] for k in ("model", "dvf")):
        raise RuntimeError("Prefix mismatch")
    last = row["stages"][-1]
    if last["stage"] != asdict(stage_for(smoke)) or last["alpha"] != 1. or last["representation"] != "bspline_control":
        raise RuntimeError("Final stage mismatch")
    if not 0 <= last["best_epoch"] <= last["actual_epochs"] <= stage_for(smoke).max_epochs:
        raise RuntimeError("Invalid epochs")
    if last["best_loss"] > min(last["initial_loss"], last["last_loss"]) + 1e-6:
        raise RuntimeError("Not lowest-loss state")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Backend mismatch")
    for key in ("cps2_reconstruction", "cps2_to_cps1"):
        check = row["initialization_checks"][key]
        if check["max_abs_voxel"] > 1e-4 or check["rms_voxel"] > 1e-5:
            raise RuntimeError("Initialization mismatch")
    if smoke and not row["native_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    if not all(math.isfinite(row[k]) for k in (*METRICS, "diffusion", "best_loss")):
        raise RuntimeError("Nonfinite result")
    field = np.load(row["dvf_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid final field")
    for key in ("model", "dvf"):
        if sha(Path(row[key + "_path"])) != row[key + "_sha256"]:
            raise RuntimeError("Output hash mismatch")
    return row


def report(result, manifest, out):
    out.mkdir(parents=True, exist_ok=False)
    dense = manifest["dense_reference"]
    rows = []
    for name, item in [("8→4→2→cps1 [1,1,5,1]", result),
                       ("8→4→2→dense [1,1,5,1]", dense),
                       ("Prefix 8→4→2 [1,1,5]", manifest["prefix"])] + [(r["method"], r) for r in manifest["references"]]:
        row = {"method": name, "rank": 0, "fixed_id": "0333", "moving_id": "0115", **{k: item[k] for k in METRICS}}
        row.update({"delta_dense_" + k: item[k] - dense[k] for k in METRICS})
        if "stages" in item:
            last = item["stages"][-1]
            row.update(best_epoch=last["best_epoch"], stopping_epoch=last["actual_epochs"],
                       stopping_reason=last["stopping_reason"], training_seconds=last["training_elapsed_sec"], gpu=item["gpu"])
        rows.append(row)
    c.write_csv(out / "comparison.csv", rows)
    atomic_json(out / "results.json", {"complete": True, "comparison": rows, "candidate": result, "provenance": manifest})
    lines = ["# Rank-0 cps1 B-spline versus dense refinement", "",
             "Fixed0333, moving0115, seed3 shuffled rank0. Same saved cps8/4/2 prefix [1,1,5]; final lambda1.",
             "Same full-resolution local squared NCC9, expanded-total-field voxel finite-difference diffusion, normalized LR0.001 and adaptive schedule.",
             "Only the final representation changes: cubic B-spline controls at cps1 versus direct dense parameters. No analytic B-spline regularizer.", "",
             "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + row["method"] + " | " + " | ".join(f"{row[k]:.6f}" for k in METRICS) + " |")
    lines += ["", "Cps1 minus dense: " + ", ".join(f"{k}={rows[0]['delta_dense_'+k]:+.9f}" for k in METRICS) + ".", "",
              "Foreground is fixed raw image>0. SD(log J) uses positive Jacobians; folding uses J≤0 percent, same canonical cropped-interior pipeline.",
              "Canonical label metrics retain the historical absent-label rules; all labels, histories and checkpoints are saved.",
              "One exploratory case: no across-case SD or generalization claim. B-splines and diffusion do not enforce a positive Jacobian.",
              "Matching nominal Adam settings across representations does not imply identical field-space updates. Runtime is diagnostic under shared GPU load.", "",
              "| Variant | Stopping epoch | Best epoch | Final-stage optimization seconds | GPU |",
              "|---|---:|---:|---:|---:|"]
    for row in rows[:2]:
        lines.append(f"| {row['method']} | {row['stopping_epoch']} | {row['best_epoch']} | {row['training_seconds']:.3f} | {row['gpu']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with (out / "comparison.csv").open() as stream:
        restored = list(csv.DictReader(stream))
    if len(restored) != 6 or any(not math.isclose(float(a[k]), b[k], abs_tol=1e-12) for a, b in zip(restored, rows) for k in METRICS):
        raise RuntimeError("Report CSV/JSON mismatch")
    assert_sources(manifest)
    for kind in ("report", "manifest"):
        if sha(Path(manifest[f"source_{kind}_path"])) != manifest[f"source_{kind}_sha256"]:
            raise RuntimeError("Historical inputs changed")
    atomic_json(out / "ACCEPTED.json", {"complete": True, "production_fits": 1, "case_rank": 0, "completed_at": c.now_text(),
                                       "output_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})


def launch(args):
    root = args.artifact_root / "runs" / VARIANT
    out = args.artifact_root / "repo_experiments" / VARIANT
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "ACCEPTED.json").exists():
            print("Already accepted", flush=True); return
        manifest_path = root / "launch_manifest.json"
        if manifest_path.exists():
            manifest = c.read_json(manifest_path); assert_sources(manifest)
            if manifest["gpu"] != args.gpu:
                raise RuntimeError("GPU changed")
        else:
            manifest = preflight(args); atomic_json(manifest_path, manifest)
        required_mib = 4096
        for smoke in (True, False):
            path = root / run_name(smoke)
            if (path / "DONE").exists():
                row = validate(path, manifest, smoke)
            else:
                if path.exists():
                    raise RuntimeError("Incomplete attempt preserved; audit before retry")
                gpu = next(r for r in legacy.gpu_snapshot()["rows"] if r["physical_gpu"] == args.gpu)
                if gpu["memory_free_mib"] < required_mib:
                    raise RuntimeError(f"Insufficient GPU memory: need {required_mib} MiB")
                command = [sys.executable, "-u", "-B", str(Path(__file__)), "worker", "--manifest", str(manifest_path),
                           "--out-dir", str(path), "--gpu", args.gpu] + (["--smoke"] if smoke else [])
                atomic_json(root / "progress.json", {"phase": "smoke" if smoke else "production", "command": command,
                                                     "gpu_before": gpu, "updated_at": c.now_text()})
                print(f"Launching {run_name(smoke)} on GPU{args.gpu}", flush=True)
                with (root / (run_name(smoke) + ".log")).open("x") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = validate(path, manifest, smoke)
            if smoke:
                required_mib = max(4096, int(math.ceil(row["stages"][-1]["peak_cuda_gib"] * 1024)) + 2048)
                atomic_json(root / "smoke_results.json", {"passed": True, "required_free_mib": required_mib, "result": row})
                if args.stop_after == "smoke":
                    return
            else:
                report(row, manifest, out)
        atomic_json(root / "progress.json", {"phase": "complete", "completed": 1, "total": 1, "updated_at": c.now_text()})
        print(f"Accepted report: {out}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        worker(args) if args.command == "worker" else launch(args)
    except BaseException as exc:
        if args.command == "worker" and args.out_dir.exists():
            atomic_json(args.out_dir / "FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
