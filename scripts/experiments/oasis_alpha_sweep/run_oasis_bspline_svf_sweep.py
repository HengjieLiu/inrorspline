#!/usr/bin/env python
"""P74: isolated first10 SVF sweep, independent replay, resource-aware queue."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import bspline_svf_sweep as s
import run_oasis_bspline_svf as p

c, d, legacy, topology = p.c, p.d, p.legacy, p.topology
sha, atomic_json, metric_row, save_field, close = p.sha, p.atomic_json, p.metric_row, p.save_field, p.close
HERE = Path(__file__).resolve().parent
METRICS = p.METRICS


def source_hashes():
    return {**p.source_hashes(), **{name: sha(HERE/name) for name in
            ("bspline_svf_sweep.py", "run_oasis_bspline_svf_sweep.py", "report_oasis_bspline_svf_sweep.py")},
            "test_oasis_bspline_svf_sweep.py": sha(c.REPO_ROOT/"tests/test_oasis_bspline_svf_sweep.py")}


def assert_manifest(m):
    if m["protocol"] != s.PROTOCOL or m["source_hashes"] != source_hashes():
        raise RuntimeError("P74 numerical/report source or protocol drift")
    for path, digest in m["reference_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Read-only reference changed: {path}")
    if sha(m["settings"]["pair_file"]) != m["pair_file_sha256"]:
        raise RuntimeError("Pair order changed")


def prepare(root, args):
    parent_path = root/"runs"/p.P70/"launch_manifest.json"
    parent = c.read_json(parent_path)
    previous, olddir = p.accepted_report(root, s.pilot.VARIANT)
    refs, refdir = p.accepted_report(root, p.P71)
    settings = {k: parent["settings"][k] for k in ("oasis_root", "pair_file", "dl_code_root", "vfa_root")}
    cases = {str(i): parent["cases"][str(i)] for i in range(10)}
    pairs = c.parse_pair_file(Path(settings["pair_file"]))
    for rank, case in cases.items():
        if tuple(pairs[int(rank)]) != (case["fixed_id"], case["moving_id"]):
            raise RuntimeError("Historical case order differs from canonical split")
        for name, digest in case["input_hashes"].items():
            if sha(Path(settings["oasis_root"])/"test"/name) != digest:
                raise RuntimeError("Input hash mismatch")
    references = [x for x in refs["per_case"] if x["depth"] == 3 and x["config_key"] == "reference"]
    if len(references) != 20 or {x["case_rank"] for x in references} != set(range(10)):
        raise RuntimeError("Require original VFA/SITReg first10 references")
    for row in references:
        case = cases[str(row["case_rank"])]
        if any(row[k] != case[k] for k in ("fixed_id", "moving_id")):
            raise RuntimeError("Baseline case mismatch")
    hashes = {str(parent_path): sha(parent_path)}
    for directory in (olddir, refdir):
        for name in ("results.json", "ACCEPTED.json"):
            hashes[str(directory/name)] = sha(directory/name)
    free = shutil.disk_usage(root).free/2**30
    if free < 400:
        raise RuntimeError(f"Require400GiB free artifact capacity, got{free:.1f}")
    return {"variant": s.VARIANT, "protocol": s.PROTOCOL, "cases": cases, "settings": settings,
            "pair_file_sha256": sha(settings["pair_file"]), "reference_hashes": hashes,
            "references": references, "pilot_rank0": [x for x in previous["candidates"] if x["spec"]["formulation"] == "refine"],
            "source_hashes": source_hashes(), "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
            "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=c.REPO_ROOT, text=True)),
            "container_image": args.container_image, "container_image_id": args.container_image_id,
            "gpus": args.gpus.split(","), "max_concurrent": 4, "uid": os.getuid(), "gid": os.getgid(),
            "initial_gpu_state": legacy.gpu_snapshot(), "initial_storage_free_gib": free,
            "production_fit_count": 280, "smoke_fit_count": 4, "created_at": c.now_text()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("launch", "report"):
        q = sub.add_parser(command)
        q.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
        if command == "launch":
            q.add_argument("--gpus", choices=("0,1,2,3",), default="0,1,2,3")
            q.add_argument("--max-concurrent", type=int, choices=(4,), default=4)
            q.add_argument("--stop-after", choices=("smoke", "full"), default="full")
            q.add_argument("--container-image", required=True)
            q.add_argument("--container-image-id", required=True)
    for command in ("worker", "audit"):
        q = sub.add_parser(command)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--out-dir", type=Path, required=True)
        q.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
        if command == "worker":
            q.add_argument("--case-rank", type=int, choices=range(10), required=True)
            q.add_argument("--schedule", choices=("842", "8421"), required=True)
            q.add_argument("--regularize", choices=("velocity", "displacement"), required=True)
            q.add_argument("--alpha", type=float, choices=s.LAMBDAS, required=True)
            q.add_argument("--integration-steps", type=int, choices=(7,), default=7)
            q.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    output = args.artifact_root if args.command in ("launch", "report") else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        parser.error("External artifacts only")
    return args


def setup(args, manifest, spec):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    assert_manifest(manifest)
    spec.validate()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    args.case_rank = spec.case_rank
    for key, value in manifest["settings"].items():
        setattr(args, key, Path(value))
    batch, hashes = legacy.load_batch(args, device)
    case = manifest["cases"][str(spec.case_rank)]
    if hashes != case["input_hashes"] or any(batch[k] != case[k] for k in ("fixed_id", "moving_id")):
        raise RuntimeError("Wrong case or changed inputs")
    if not torch.equal(batch["fixed_mask"].bool(), batch["fixed"] > 0):
        raise RuntimeError("Wrong fixed raw foreground")
    return device, batch, case


def worker(args):
    started = time.monotonic()
    manifest = c.read_json(args.manifest)
    spec = s.Spec(args.case_rank, args.schedule, args.regularize, args.alpha, args.smoke,
                  integration_steps=args.integration_steps)
    spec.validate()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    gpu_before = legacy.gpu_snapshot()
    device, batch, case = setup(args, manifest, spec)
    row = {"variant": s.VARIANT, "spec": asdict(spec), "method": s.METHOD,
           "run_name": spec.name(), "run_dir": str(args.out_dir), "protocol": s.PROTOCOL,
           **case, "direction": "moving_to_fixed", "sampling": "fixed_to_moving_pull",
           "shape": [160, 224, 192, 3], "stage_alpha": [spec.alpha]*len(spec.schedule),
           "gpu": args.gpu, "visible_cuda_device": "cuda:0", "initial_gpu_state": gpu_before,
           "source_hashes": manifest["source_hashes"], "source_revision": manifest["source_revision"],
           "source_dirty": manifest["source_dirty"], "pair_file_sha256": manifest["pair_file_sha256"],
           "cuda_backend": c.cuda_backend_metadata(device), "seed": 1, "tf32_matmul": False,
           "tf32_cudnn": False, "uid": os.getuid(), "gid": os.getgid(), "bounded": False,
           "container_image": manifest["container_image"], "container_image_id": manifest["container_image_id"],
           "representation": "exponential_of_bspline_velocity", "segmentation_loss": False,
           "analytic_regularizer": False, "reference_field_supervision": False,
           "cuda_nondeterminism_caveat": "grid_sample backward may remain nondeterministic",
           "started_at": c.now_text(), "argv": sys.argv}
    if spec.smoke:
        row["native_loss_parity"] = legacy.native_parity(device, args.dl_code_root, args.vfa_root)
    atomic_json(args.out_dir/"manifest.json", row)
    states, blocks = [], []
    previous_controls = previous_velocity = previous_field = None
    allocated = reserved = 0.
    for stage in s.stages(spec):
        objective = s.Objective(batch["fixed"], batch["moving"], stage, spec)
        initial = (c.cubic_bspline_dyadic_knot_insert_3d(previous_controls, objective.control_shape)
                   if previous_controls is not None else torch.zeros(*objective.control_shape, 3, device=device))
        with torch.no_grad():
            values = [float(x) for x in objective(initial)]
            init_field = objective.field(initial, native=True)
            expected = previous_field if previous_field is not None else torch.zeros_like(init_field)
            init_diff = c.field_difference_metrics((init_field-expected)*objective.native_scale)
            if previous_velocity is not None:
                vdiff = (objective.expand(initial)-previous_velocity)*objective.native_scale
                init_diff["velocity_max_abs_voxel"] = float(vdiff.abs().max())
                if not torch.allclose(objective.expand(initial), previous_velocity, rtol=1e-5, atol=1e-7):
                    raise RuntimeError("Knot insertion changed velocity")
                if not torch.allclose(init_field, expected, rtol=1e-4, atol=2e-6):
                    raise RuntimeError("Knot insertion changed integrated field materially")
            elif torch.count_nonzero(init_field):
                raise RuntimeError("Registration must begin at identity")
        del init_field
        stage_dir = args.out_dir/f"stage{stage.index}_cps{stage.cps}"
        atomic_json(args.out_dir/"progress.json", {"phase": "optimization", "cps": stage.cps})
        best, info = s.optimize(initial, objective, stage_dir)
        with torch.no_grad():
            state = torch.load(stage_dir/"best.pt", map_location=device, weights_only=False)
            if not torch.equal(best, state["velocity_controls_norm_xyz"]) or info["alpha"] != spec.alpha:
                raise RuntimeError("Wrong saved controls or alpha")
            v = objective.expand(best).detach()
            field = objective.field(best, native=True).detach()
            field_path = stage_dir/f"disp_norm_{case['fixed_id']}_{case['moving_id']}.npy"
            block = {"cps": stage.cps, "stage_index": stage.index, "alpha": spec.alpha,
                     "optimizer": info, "initial_loss_data_reg": values, "initialization": init_diff,
                     "model_path": str(stage_dir/"best.pt"), "model_sha256": sha(stage_dir/"best.pt"),
                     "last_model_sha256": sha(stage_dir/"last.pt"), "history_sha256": sha(stage_dir/"history.csv"),
                     "dvf_path": str(field_path), "dvf_sha256": save_field(field_path, field),
                     "scaled_initial_map_corners": topology.corner_diagnostics(v*objective.native_scale/128),
                     "velocity_diffusion_native": float(c.forward_valid_diffusion((v*objective.native_scale).permute(3,0,1,2)[None])),
                     "displacement_diffusion_native": float(c.forward_valid_diffusion((field*objective.native_scale).permute(3,0,1,2)[None]))}
            atomic_json(stage_dir/"validated.json", block)
        blocks.append(block); states.append((stage.cps, best.cpu()))
        previous_controls, previous_velocity, previous_field = best.detach(), v, field
        allocated = max(allocated, torch.cuda.max_memory_allocated()/2**30)
        reserved = max(reserved, torch.cuda.max_memory_reserved()/2**30)
        print(json.dumps({"run": spec.name(), "stage_complete": stage.cps, "epochs": info["actual_epochs"],
                          "best_epoch": info["best_epoch"], "optimizer_sec": info["training_elapsed_sec"]}), flush=True)
        del objective, initial, best, state
    metric_started = time.monotonic()
    atomic_json(args.out_dir/"progress.json", {"phase": "metrics_and_diagnostics"})
    metrics = metric_row(previous_field, batch, args.out_dir/"final_metrics")
    diagnostics = []
    scale = d.voxel_scale(c.IMAGE_SHAPE, device=device)
    for steps in ((7,) if spec.smoke else (7, 8, 10)):
        with torch.no_grad():
            u, parts, velocities = s.reconstruct(states, "refine", c.IMAGE_SHAPE, device, steps)
            _, inverses, _ = s.reconstruct(states, "refine", c.IMAGE_SHAPE, device, steps, inverse=True)
            record = {"integration_steps": steps,
                      **c.field_difference_metrics((u-previous_field)*scale, prefix="delta_from_ss7_voxel_")}
            record["inverse_consistency"] = s.inverse_consistency(parts, inverses, batch["fixed_mask"])
            if steps == 7:
                if not torch.equal(u, previous_field):
                    raise RuntimeError("Saved velocity does not reconstruct final field exactly")
                record["metrics"] = metrics
            else:
                folder = args.out_dir/f"integration_ss{steps}"
                folder.mkdir()
                path = folder/f"disp_norm_{case['fixed_id']}_{case['moving_id']}.npy"
                record.update(dvf_path=str(path), dvf_sha256=save_field(path, u), metrics=metric_row(u, batch, folder))
            record["delta_primary_metrics"] = {k: record["metrics"][k]-metrics[k] for k in METRICS}
        diagnostics.append(record)
        del u, parts, velocities, inverses
    row.update(**metrics, blocks=blocks, integration_diagnostics=diagnostics,
               dvf_path=blocks[-1]["dvf_path"], dvf_sha256=blocks[-1]["dvf_sha256"],
               actual_adam_updates=sum(x["optimizer"]["actual_epochs"] for x in blocks),
               optimizer_sec=sum(x["optimizer"]["training_elapsed_sec"] for x in blocks),
               metrics_diagnostics_sec=time.monotonic()-metric_started, worker_sec=time.monotonic()-started,
               peak_cuda_gib=max(allocated, torch.cuda.max_memory_allocated()/2**30),
               peak_reserved_gib=max(reserved, torch.cuda.max_memory_reserved()/2**30),
               final_gpu_state=legacy.gpu_snapshot(), completed_at=c.now_text())
    assert_manifest(manifest)
    atomic_json(args.out_dir/"summary.json", row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")
    atomic_json(args.out_dir/"progress.json", {"phase": "worker_complete_pending_independent_audit"})


def audit(args):
    started = time.monotonic()
    manifest = c.read_json(args.manifest)
    row = c.read_json(args.out_dir/"summary.json")
    spec = s.Spec(**row["spec"])
    if not (args.out_dir/"DONE").is_file() or row["protocol"] != s.PROTOCOL:
        raise RuntimeError("Incomplete or wrong replay target")
    if row["source_hashes"] != manifest["source_hashes"] or row["stage_alpha"] != [spec.alpha]*len(spec.schedule):
        raise RuntimeError("Provenance or stage weights mismatch")
    device, batch, case = setup(args, manifest, spec)
    if row["input_hashes"] != case["input_hashes"] or any(row[k] != case[k] for k in ("fixed_id", "moving_id", "case_rank")):
        raise RuntimeError("Replay case metadata mismatch")
    if row["direction"] != "moving_to_fixed" or row["shape"] != [160,224,192,3]:
        raise RuntimeError("Replay direction/geometry mismatch")
    if len(row["blocks"]) != len(spec.schedule):
        raise RuntimeError("Missing stage states")
    states, checks = [], []
    previous = None
    for stage, block in zip(s.stages(spec), row["blocks"]):
        path = Path(block["model_path"])
        for filename, digest in ((path, block["model_sha256"]), (path.parent/"last.pt", block["last_model_sha256"]),
                                  (path.parent/"history.csv", block["history_sha256"]), (Path(block["dvf_path"]), block["dvf_sha256"])):
            if sha(filename) != digest: raise RuntimeError("Checkpoint/history/field changed")
        state = torch.load(path, map_location=device, weights_only=False)
        controls = state["velocity_controls_norm_xyz"]
        obj = s.Objective(batch["fixed"], batch["moving"], stage, spec)
        initial = c.cubic_bspline_dyadic_knot_insert_3d(previous, obj.control_shape) if previous is not None else torch.zeros_like(controls)
        with torch.no_grad():
            values, initial_values = [float(x) for x in obj(controls)], [float(x) for x in obj(initial)]
            info = block["optimizer"]
            if (state["summary"] != info or info["alpha"] != spec.alpha or block["alpha"] != spec.alpha
                    or info["stage"] != asdict(stage)):
                raise RuntimeError("Checkpoint metadata or stage schedule mismatch")
            if any(not close(a,b) for a,b in zip(initial_values, block["initial_loss_data_reg"])):
                raise RuntimeError("Initial loss mismatch")
            if any(not close(v, info[k]) for v,k in zip(values, ("best_loss", "best_data_loss", "best_reg_loss"))):
                raise RuntimeError("Selected loss mismatch")
            current = obj.field(controls, native=True)
            if not np.array_equal(current.cpu().numpy(), np.load(block["dvf_path"])):
                raise RuntimeError("Independent stage replay mismatch")
            v = obj.expand(controls)
            if topology.corner_diagnostics(v*obj.native_scale/128) != block["scaled_initial_map_corners"]:
                raise RuntimeError("Derivative diagnostics mismatch")
        with (path.parent/"history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda x: float(x["loss"]))
        if (int(best["epoch"]) != info["best_epoch"] or not close(best["loss"], info["best_loss"])
                or len(history) != info["actual_epochs"]+1 or info["best_loss"] > info["initial_loss"]+1e-7):
            raise RuntimeError("Best checkpoint/history mismatch")
        if not all(int(h["epoch"]) == i and close(float(h["loss"]), float(h["data_loss"])+spec.alpha*float(h["reg_loss"])) for i,h in enumerate(history)):
            raise RuntimeError("History does not contain the requested weighted objective")
        checks.append({"cps": stage.cps, "exact_field_replay": True, "loss_history_verified": True})
        states.append((stage.cps, controls.cpu())); previous = controls
        del obj, initial, state, v
    replay = metric_row(current, batch, args.out_dir/"independent_metrics")
    for key in (*METRICS, "valid_label_count", "excluded_labels"):
        if replay[key] != row[key]: raise RuntimeError(f"Canonical replay mismatch:{key}")
    with (args.out_dir/"final_metrics/case_metrics.csv").open() as stream:
        original_labels = list(csv.DictReader(stream))
    with (args.out_dir/"independent_metrics/case_metrics.csv").open() as stream:
        if list(csv.DictReader(stream)) != original_labels: raise RuntimeError("Label replay mismatch")
    expected_steps = [7] if spec.smoke else [7,8,10]
    if [x["integration_steps"] for x in row["integration_diagnostics"]] != expected_steps:
        raise RuntimeError("Missing integration sensitivity evaluations")
    for diag in row["integration_diagnostics"]:
        steps = diag["integration_steps"]
        with torch.no_grad():
            u, parts, _ = s.reconstruct(states, "refine", c.IMAGE_SHAPE, device, steps)
            _, inverse, _ = s.reconstruct(states, "refine", c.IMAGE_SHAPE, device, steps, inverse=True)
            if s.inverse_consistency(parts, inverse, batch["fixed_mask"]) != diag["inverse_consistency"]:
                raise RuntimeError("Inverse residual mismatch")
            if steps != 7:
                if sha(diag["dvf_path"]) != diag["dvf_sha256"] or not np.array_equal(u.cpu().numpy(), np.load(diag["dvf_path"])):
                    raise RuntimeError("Sensitivity export mismatch")
                other = metric_row(u, batch, args.out_dir/f"independent_ss{steps}_metrics")
                if any(other[k] != diag["metrics"][k] for k in (*METRICS, "valid_label_count", "excluded_labels")):
                    raise RuntimeError("Sensitivity metrics mismatch")
        del u, parts, inverse
    assert_manifest(manifest)
    atomic_json(args.out_dir/"independent_audit.json", {"passed": True, "checks": checks,
                "integration_steps": expected_steps, "all_label_metrics_reproduced": True,
                "summary_sha256": sha(args.out_dir/"summary.json"), "source_hashes": source_hashes(),
                "gpu": args.gpu, "audit_sec": time.monotonic()-started,
                "audit_peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30,
                "audit_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "completed_at": c.now_text()})


def validate(path, manifest, spec):
    row = c.read_json(path/"summary.json")
    check = c.read_json(path/"independent_audit.json")
    if (not (path/"DONE").is_file() or row["spec"] != asdict(spec) or not check["passed"]
            or row["source_hashes"] != manifest["source_hashes"] or check["source_hashes"] != manifest["source_hashes"]
            or check["summary_sha256"] != sha(path/"summary.json") or sha(row["dvf_path"]) != row["dvf_sha256"]):
        raise RuntimeError("Incomplete or inconsistent audited run")
    return {**row, **{k: check[k] for k in ("audit_sec", "audit_peak_cuda_gib", "audit_peak_reserved_gib")}}


def phase(root, manifest, smoke, required_mib):
    jobs = [{"spec": spec, "status": "pending", "attempt": 0, "gpu": None} for spec in s.matrix(smoke)]
    active, completed, failures, affinity = {}, [], [], {}
    phase_name = "smoke" if smoke else "production"
    started, last_sample = time.monotonic(), 0.
    for job in jobs:
        for attempt in (1, 0):
            path = root/job["spec"].name()/f"attempt{attempt}"
            if (path/"independent_audit.json").is_file():
                completed.append(validate(path, manifest, job["spec"]))
                job.update(status="complete", attempt=attempt, out=path, gpu=completed[-1]["gpu"])
                break
            if path.exists():
                if (path/"DONE").is_file():
                    job.update(status="audit_pending", attempt=attempt, out=path, gpu=c.read_json(path/"summary.json")["gpu"])
                    break
                raise RuntimeError(f"Preserve unfinished attempt for audit:{path}")
        if job["gpu"] is not None and not smoke: affinity[job["spec"].case_rank] = job["gpu"]
    completed_at_resume = len(completed)
    while any(x["status"] not in ("complete", "failed") for x in jobs) or active:
        for gpu, item in list(active.items()):
            proc, job = item["process"], item["job"]
            if proc.poll() is None: continue
            item["log"].close()
            if proc.returncode:
                content = item["log_path"].read_text(errors="replace")
                failure = {"spec": asdict(job["spec"]), "gpu": gpu, "attempt": job["attempt"],
                           "action": item["action"], "log": str(item["log_path"]), "returncode": proc.returncode}
                atomic_json(job["out"].parent/f"failure_{item['action']}_{time.time_ns()}.json", failure)
                oom = "out of memory" in content.lower() and "cuda" in content.lower()
                if oom and not job.get("oom_retried", False):
                    job.update(status="retry" if item["action"] == "worker" else "audit_pending",
                               oom_retried=True, retry_after=len(completed), retry_wait=True)
                    if item["action"] == "worker": job["attempt"] += 1
                else:
                    job["status"] = "failed"; failures.append(failure)
            elif item["action"] == "worker":
                job["status"] = "audit_pending"
            else:
                completed.append(validate(job["out"], manifest, job["spec"]))
                job["status"] = "complete"
            del active[gpu]
        snapshot = legacy.gpu_snapshot()
        available = {str(x["physical_gpu"]): x for x in snapshot.get("rows", [])
                     if str(x["physical_gpu"]) in manifest["gpus"] and str(x["physical_gpu"]) not in active
                     and x["memory_free_mib"] >= required_mib}
        if time.monotonic()-last_sample >= 30:
            with (root/"gpu_load.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": c.now_text(), "phase": phase_name, "snapshot": snapshot})+"\n")
            last_sample = time.monotonic()
        # Audits keep priority and share the same per-GPU slot as optimizers.
        for job in sorted(jobs, key=lambda j: j["status"] != "audit_pending"):
            if job["status"] in ("running", "auditing", "complete", "failed"): continue
            if failures and job["status"] != "audit_pending":
                job["status"] = "failed"; continue
            spec = job["spec"]
            if job.get("retry_wait") and active and len(completed) <= job["retry_after"]:
                continue
            rank = spec.case_rank
            # A new case may take a GPU only after its previous case is finished.
            busy_affinity = {gpu for case, gpu in affinity.items()
                             if any(j["spec"].case_rank == case and j["status"] not in ("complete", "failed") for j in jobs)}
            if job.get("retry_wait"):
                choices = list(available)
            elif job["status"] == "audit_pending":
                choices = [job["gpu"]] if job["gpu"] in available else []
            elif not smoke and rank in affinity:
                choices = [affinity[rank]] if affinity[rank] in available else []
            else:
                choices = [g for g in available if smoke or g not in busy_affinity]
            if not choices: continue
            gpu = max(choices, key=lambda g: available[g]["memory_free_mib"])
            action = "audit" if job["status"] == "audit_pending" else "worker"
            path = job.get("out") if action == "audit" else root/spec.name()/f"attempt{job['attempt']}"
            if action == "worker" and path.exists(): raise RuntimeError(f"Refuse overwrite:{path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-B", str(Path(__file__)), action, "--manifest", str(root/"launch_manifest.json"),
                       "--out-dir", str(path), "--gpu", gpu]
            if action == "worker":
                command += ["--case-rank", str(rank), "--schedule", spec.schedule, "--regularize", spec.regularize,
                            "--alpha", str(spec.alpha)]
                if smoke: command.append("--smoke")
            log_path = path.parent/f"attempt{job['attempt']}_{action}_{time.time_ns()}.log"
            log = log_path.open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            job.update(status="auditing" if action == "audit" else "running", gpu=gpu, out=path, retry_wait=False)
            if not smoke and rank not in affinity: affinity[rank] = gpu
            active[gpu] = {"process": proc, "job": job, "action": action, "log": log, "log_path": log_path}
            event = {"launched": spec.name(), "action": action, "gpu": gpu, "pid": proc.pid,
                     "log_path": str(log_path), "gpu_before": available[gpu], "time": c.now_text()}
            with (root/"launch_events.jsonl").open("a") as stream: stream.write(json.dumps(event)+"\n")
            print(json.dumps(event), flush=True)
            del available[gpu]
        elapsed = time.monotonic()-started
        new_done = [x for x in completed if x.get("worker_sec")]
        # This is observed audited-fit wall throughput, not an idle-GPU speed claim.
        observed_count = len(completed)-completed_at_resume
        eta = elapsed/observed_count*(len(jobs)-len(completed)) if observed_count else None
        active_phases = []
        for gpu,item in active.items():
            path = item["job"]["out"]/"progress.json"
            detail = c.read_json(path) if path.exists() else {"phase":"loading"}
            active_phases.append({"gpu":gpu,"action":item["action"],"run":item["job"]["spec"].name(),**detail})
        status = {"phase": phase_name, "completed": len(completed), "total": len(jobs), "elapsed_sec": elapsed,
                  "eta_sec_observed_audited_throughput": eta, "required_free_mib": required_mib,
                  "failed": failures, "updated_at": c.now_text(), "case_gpu_affinity": affinity,
                  "optimizer_sec_completed": sum(x["optimizer_sec"] for x in new_done),
                  "worker_sec_completed": sum(x["worker_sec"] for x in new_done),
                  "audit_sec_completed": sum(x["audit_sec"] for x in new_done),
                  "active_phases": active_phases,
                  "workers_finished": sum(j["status"] in ("complete","audit_pending","auditing") for j in jobs),
                  "jobs": [{"spec": asdict(x["spec"]), "status": x["status"], "gpu": x["gpu"],
                            "attempt": x["attempt"], "out": str(x.get("out", ""))} for x in jobs]}
        atomic_json(root/"progress.json", status)
        if active or any(x["status"] not in ("complete", "failed") for x in jobs): time.sleep(5)
    if failures:
        atomic_json(root/f"{phase_name}_FAILED.json", {"failures": failures, "completed": len(completed)})
        raise RuntimeError("Numerical/worker failure preserved; stopped new dispatch for audit")
    return completed


def report(root):
    from report_oasis_bspline_svf_sweep import generate
    run = root/"runs"/s.VARIANT
    manifest = c.read_json(run/"launch_manifest.json")
    assert_manifest(manifest)
    rows = []
    for spec in s.matrix():
        matches = [x for x in (run/spec.name()).glob("attempt*") if x.is_dir() and (x/"independent_audit.json").is_file()]
        if len(matches) != 1: raise RuntimeError(f"Require exactly one accepted fit:{spec.name()}")
        rows.append(validate(matches[0], manifest, spec))
    out = root/"repo_experiments"/s.VARIANT
    generate(rows, manifest, out)
    assert_manifest(manifest)
    atomic_json(run/"COMPLETE.json", {"count": 280, "completed_at": c.now_text()})
    atomic_json(run/"progress.json", {"phase": "accepted", "completed": 280, "total": 280, "updated_at": c.now_text()})


def launch(args):
    if (os.getuid(), os.getgid()) != (1009, 1020): raise RuntimeError("Require1009:1020")
    root = args.artifact_root/"runs"/s.VARIANT
    root.mkdir(parents=True, exist_ok=True)
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = root/"launch_manifest.json"
        manifest = c.read_json(path) if path.exists() else prepare(args.artifact_root, args)
        assert_manifest(manifest)
        if not path.exists(): atomic_json(path, manifest)
        if (root/"COMPLETE.json").exists(): return
        smokes = phase(root, manifest, True, 8192)
        required = max(8192, math.ceil((max(max(x["peak_reserved_gib"], x["audit_peak_reserved_gib"]) for x in smokes)+2)*1024))
        atomic_json(root/"smokes_passed.json", {"count": 4, "required_free_mib": required,
                    "runs": [x["run_dir"] for x in smokes], "source_hashes": source_hashes()})
        if args.stop_after == "smoke": return
        phase(root, manifest, False, required)
        report(args.artifact_root)


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.command == "report": report(args.artifact_root)
        else: {"launch": launch, "worker": worker, "audit": audit}[args.command](args)
    except BaseException:
        traceback.print_exc()
        if getattr(args, "out_dir", None) and args.out_dir.is_dir():
            atomic_json(args.out_dir/f"FAILED_{args.command}.json", {"traceback": traceback.format_exc(), "created_at": c.now_text()})
        raise
