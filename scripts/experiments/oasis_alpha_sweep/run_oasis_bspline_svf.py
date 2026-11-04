#!/usr/bin/env python
"""P73 isolated rank0 B-spline SVF pilot, replay, queue and comparison report."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
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

import bspline_svf as s
import bounded_composed_bspline as topology
import run_oasis_dl_ncc_diffusion_one as legacy
from run_oasis_bounded_composed import metric_row

c, d = s.c, s.d
HERE = Path(__file__).resolve().parent
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
P70 = "oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3"
P71 = "oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1"
P43 = d.VARIANT


def sha(path):
    return legacy.sha256_file(Path(path))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    names = ("bspline_svf.py", "run_oasis_bspline_svf.py", "bounded_composed_bspline.py",
             "run_oasis_bounded_composed.py")
    return {**legacy.source_hashes(), **{name: sha(HERE/name) for name in names},
            "test_oasis_bspline_svf.py": sha(c.REPO_ROOT/"tests/test_oasis_bspline_svf.py")}


def assert_manifest(manifest):
    if manifest["protocol"] != s.PROTOCOL or manifest["source_hashes"] != source_hashes():
        raise RuntimeError("SVF source/protocol drift")
    for path, digest in manifest["reference_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Read-only reference changed: {path}")
    if sha(manifest["settings"]["pair_file"]) != manifest["pair_file_sha256"]:
        raise RuntimeError("Pair split changed")


def accepted_report(root, key):
    directory = root/"repo_experiments"/key
    accepted = c.read_json(directory/"ACCEPTED.json")
    for name, digest in accepted.get("output_sha256", {}).items():
        if sha(directory/name) != digest:
            raise RuntimeError(f"Accepted {key} report changed: {name}")
    return c.read_json(directory/"results.json"), directory


def prepare(root, gpus):
    parent_path = root/"runs"/P70/"launch_manifest.json"
    parent = c.read_json(parent_path)
    p71, p71dir = accepted_report(root, P71)
    p43, p43dir = accepted_report(root, P43)
    _, p70dir = accepted_report(root, P70)
    references = []
    hashes = {str(parent_path): sha(parent_path)}
    for directory in (p71dir, p43dir, p70dir):
        for name in ("results.json", "ACCEPTED.json"):
            hashes[str(directory/name)] = sha(directory/name)
    for row in p71["per_case"]:
        if row["case_rank"] == 0 and row["depth"] in (3, 4):
            references.append({**row, "schedule": "842" if row["depth"] == 3 else "8421",
                               "source_kind": "historical_composition_or_DL", "source_report": str(p71dir)})
    for row in p71["candidates"]:
        if row["case_rank"] == 0 and row["depth"] in (3, 4):
            if sha(row["dvf_path"]) != row["dvf_sha256"] or not (Path(row["run_dir"])/"DONE").is_file():
                raise RuntimeError("Historical composed field incomplete or changed")
            hashes[row["dvf_path"]] = row["dvf_sha256"]
    for row in p43["case_metrics"]:
        if (row["case_rank"] == 0 and row.get("resolution") == "pyramid" and
                row.get("alpha") == 1. and row.get("representation") in ("842", "8421")):
            references.append({**row, "schedule": row["representation"],
                               "method": "Historical prolonged displacement B-spline",
                               "source_kind": "historical_prolonged_displacement", "source_report": str(p43dir)})
            summary = c.read_json(Path(row["run_dir"])/"summary.json")
            if sha(summary["dvf_path"]) != row["dvf_sha256"]:
                raise RuntimeError("Historical prolonged field changed")
            hashes[summary["dvf_path"]] = row["dvf_sha256"]
    if len(references) != 16:
        raise RuntimeError(f"Require eight historical rows per schedule, got {len(references)}")
    families = [(form, reg) for form in s.METHODS for reg in ("velocity", "displacement")]
    return {"variant": s.VARIANT, "protocol": s.PROTOCOL, "case": parent["cases"]["0"],
            "settings": parent["settings"], "pair_file_sha256": parent["pair_file_sha256"],
            "references": references, "reference_hashes": hashes, "source_hashes": source_hashes(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
            "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=c.REPO_ROOT, text=True)),
            "gpus": gpus, "family_gpu": {f"{a}_{b}": gpu for (a,b),gpu in zip(families,gpus)},
            "container_image": parent["container_image"], "container_image_id": parent["container_image_id"],
            "initial_gpu_state": legacy.gpu_snapshot(), "uid": os.getuid(), "gid": os.getgid(),
            "created_at": c.now_text(), "production_fit_count": 8, "smoke_fit_count": 8}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    launch.add_argument("--gpus", default="0,1,2,3", choices=("0,1,2,3",))
    launch.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    for command in ("worker", "audit"):
        q = sub.add_parser(command)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--out-dir", type=Path, required=True)
        q.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
        if command == "worker":
            q.add_argument("--schedule", choices=("842", "8421"), required=True)
            q.add_argument("--formulation", choices=tuple(s.METHODS), required=True)
            q.add_argument("--regularize", choices=("velocity", "displacement"), required=True)
            q.add_argument("--integration-steps", type=int, choices=(7,), default=7)
            q.add_argument("--case-rank", type=int, choices=(0,), default=0)
            q.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    output = args.artifact_root if args.command == "launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Artifacts must be outside the source repository")
    return args


def setup(args, manifest):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    assert_manifest(manifest)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    args.case_rank = 0
    for key in ("oasis_root", "pair_file", "dl_code_root", "vfa_root"):
        setattr(args, key, Path(manifest["settings"][key]))
    batch, hashes = legacy.load_batch(args, device)
    if hashes != manifest["case"]["input_hashes"] or (batch["fixed_id"], batch["moving_id"]) != ("0333", "0115"):
        raise RuntimeError("Wrong case or changed inputs")
    if not torch.equal(batch["fixed_mask"].bool(), batch["fixed"] > 0):
        raise RuntimeError("Wrong fixed foreground convention")
    return device, batch


def save_field(path, field):
    if field.shape != (*c.IMAGE_SHAPE, 3) or not bool(torch.isfinite(field).all()):
        raise RuntimeError("Nonfinite or incorrect field geometry")
    np.save(path, field.detach().cpu().numpy())
    if not np.array_equal(np.load(path), field.detach().cpu().numpy()):
        raise RuntimeError("DVF serialization mismatch")
    return sha(path)


def close(actual, expected, rel=1e-6, abs_=1e-7):
    return math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_)


def worker(args):
    started = time.monotonic()
    manifest = c.read_json(args.manifest)
    spec = s.Spec(args.schedule, args.formulation, args.regularize, args.integration_steps, 0, args.smoke)
    spec.validate()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    initial_gpu = legacy.gpu_snapshot()
    device, batch = setup(args, manifest)
    row = {"variant": s.VARIANT, "spec": asdict(spec), "method": s.METHODS[spec.formulation],
           "protocol": s.PROTOCOL, "case_rank": 0, "fixed_id": "0333", "moving_id": "0115",
           "direction": "moving_to_fixed", "sampling": "fixed_to_moving_pull", "shape": [160,224,192,3],
           "stage_alpha": [1.]*len(spec.schedule), "gpu": args.gpu, "visible_cuda_device": "cuda:0",
           "source_hashes": manifest["source_hashes"], "source_revision": manifest["source_revision"],
           "source_dirty": manifest["source_dirty"], "input_hashes": manifest["case"]["input_hashes"],
           "pair_file_sha256": manifest["pair_file_sha256"], "initial_gpu_state": initial_gpu,
           "cuda_backend": c.cuda_backend_metadata(device), "seed": 1, "tf32_matmul": False,
           "tf32_cudnn": False, "uid": os.getuid(), "gid": os.getgid(), "bounded": False,
           "container_image": manifest["container_image"], "container_image_id": manifest["container_image_id"],
           "segmentation_loss": False, "reference_field_supervision": False, "analytic_regularizer": False,
           "run_dir": str(args.out_dir), "run_name": spec.name(), "started_at": c.now_text(),
           "representation": "exponential_of_bspline_velocity" if spec.formulation == "refine" else "retained_composition_of_bspline_velocity_exponentials",
           "cuda_nondeterminism_caveat": "grid_sample backward can remain nondeterministic", "argv": sys.argv}
    if spec.smoke:
        row["native_loss_parity"] = legacy.native_parity(device, args.dl_code_root, args.vfa_root)
    atomic_json(args.out_dir/"manifest.json", row)
    states, fields, blocks = [], [], []
    previous_velocity = previous_field = previous_controls = None
    peak_allocated = peak_reserved = 0.
    for stage in s.stages(spec):
        objective = s.Objective(batch["fixed"], batch["moving"], stage, spec,
                                fields if spec.formulation == "compose" else ())
        initial = (c.cubic_bspline_dyadic_knot_insert_3d(previous_controls, objective.control_shape)
                   if spec.formulation == "refine" and previous_controls is not None else
                   torch.zeros(*objective.control_shape, 3, device=device))
        with torch.no_grad():
            initial_values = [float(x) for x in objective(initial)]
            init_field = objective.field(initial, native=True)
            expected = previous_field if previous_field is not None else torch.zeros_like(init_field)
            initialization = c.field_difference_metrics((init_field-expected)*objective.native_scale)
            if spec.formulation == "refine" and previous_velocity is not None:
                velocity_diff = (objective.expand(initial)-previous_velocity)*objective.native_scale
                initialization["velocity_max_abs_voxel"] = float(velocity_diff.abs().max())
                if not torch.allclose(objective.expand(initial), previous_velocity, rtol=1e-5, atol=1e-7):
                    raise RuntimeError("Velocity knot insertion failed")
                if not torch.allclose(init_field, expected, rtol=1e-4, atol=2e-6):
                    raise RuntimeError("Integrated field changed materially under knot insertion")
            elif float((init_field-expected).abs().max()) > 2e-6:
                raise RuntimeError("Identity increment failed")
        del init_field
        stage_dir = args.out_dir/f"stage{stage.index}_cps{stage.cps}"
        atomic_json(args.out_dir/"progress.json", {"phase":"optimization", "stage":stage.index, "cps":stage.cps})
        best, info = s.optimize(initial, objective, stage_dir)
        with torch.no_grad():
            checkpoint = torch.load(stage_dir/"best.pt", map_location=device, weights_only=False)
            if not torch.equal(best, checkpoint["velocity_controls_norm_xyz"]):
                raise RuntimeError("Velocity checkpoint mismatch")
            v = objective.expand(best).detach()
            atomic = s.exp_velocity(v, 7, objective.native_identity).detach()
            current = objective.field(best, native=True).detach()
            corners = topology.corner_diagnostics(v*objective.native_scale/128)
            path = stage_dir/"disp_norm_0333_0115.npy"
            digest = save_field(path, current)
            block = {"cps": stage.cps, "stage_index": stage.index, "alpha": 1., "optimizer": info,
                     "initial_loss_data_reg": initial_values, "initialization": initialization,
                     "model_path": str(stage_dir/"best.pt"), "model_sha256": sha(stage_dir/"best.pt"),
                     "last_model_sha256": sha(stage_dir/"last.pt"), "history_sha256": sha(stage_dir/"history.csv"),
                     "dvf_path": str(path), "dvf_sha256": digest, "scaled_initial_map_corners": corners,
                     "velocity_diffusion_native": float(c.forward_valid_diffusion((v*objective.native_scale).permute(3,0,1,2)[None])),
                     "displacement_diffusion_native": float(c.forward_valid_diffusion((current*objective.native_scale).permute(3,0,1,2)[None]))}
            atomic_json(stage_dir/"validated.json", block)
        blocks.append(block); states.append((stage.cps, best.detach().cpu()))
        if spec.formulation == "compose":
            fields.append(atomic)
        previous_velocity, previous_field, previous_controls = v, current, best.detach()
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated()/2**30)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved()/2**30)
        print(json.dumps({"stage_complete":stage.cps, "epochs":info["actual_epochs"], "best_epoch":info["best_epoch"],
                          "optimizer_sec":info["training_elapsed_sec"], "run":spec.name()}), flush=True)
        del objective, initial, checkpoint, best
    fields = fields if spec.formulation == "compose" else [previous_field]
    atomic_json(args.out_dir/"progress.json", {"phase":"metrics_and_diagnostics"})
    metrics = metric_row(previous_field, batch, args.out_dir/"final_metrics")
    diagnostics = []
    scale = d.voxel_scale(c.IMAGE_SHAPE, device=device)
    for steps in ((7,) if spec.smoke else (7,8,10)):
        with torch.no_grad():
            u, parts, velocities = s.reconstruct(states, spec.formulation, c.IMAGE_SHAPE, device, steps)
            _, inverses, _ = s.reconstruct(states, spec.formulation, c.IMAGE_SHAPE, device, steps, inverse=True)
            record = {"integration_steps":steps,
                      **c.field_difference_metrics((u-previous_field)*scale, prefix="delta_from_ss7_voxel_")}
            record["inverse_consistency"] = s.inverse_consistency(parts, inverses, batch["fixed_mask"])
            if steps == 7:
                if not torch.equal(u, previous_field):
                    raise RuntimeError("Final checkpoint reconstruction mismatch")
                record["metrics"] = metrics
            else:
                folder = args.out_dir/f"integration_ss{steps}"
                folder.mkdir()
                record["dvf_path"] = str(folder/"disp_norm_0333_0115.npy")
                record["dvf_sha256"] = save_field(Path(record["dvf_path"]), u)
                record["metrics"] = metric_row(u, batch, folder)
            record["delta_primary_metrics"] = {key:record["metrics"][key]-metrics[key] for key in METRICS}
        if spec.formulation == "compose":
            record["offgrid"] = topology.offgrid_diagnostics(parts, u, batch["fixed_mask"], count=100000, seed=1)
        diagnostics.append(record)
        del u, parts, velocities, inverses
    row.update(**metrics, blocks=blocks, integration_diagnostics=diagnostics,
               final_map_count=len(fields), dvf_path=blocks[-1]["dvf_path"], dvf_sha256=blocks[-1]["dvf_sha256"],
               actual_adam_updates=sum(x["optimizer"]["actual_epochs"] for x in blocks),
               optimizer_sec=sum(x["optimizer"]["training_elapsed_sec"] for x in blocks),
               worker_sec=time.monotonic()-started, peak_cuda_gib=max(peak_allocated, torch.cuda.max_memory_allocated()/2**30),
               peak_reserved_gib=max(peak_reserved, torch.cuda.max_memory_reserved()/2**30), completed_at=c.now_text())
    if not all(math.isfinite(row[key]) for key in METRICS):
        raise RuntimeError("Nonfinite canonical metric")
    assert_manifest(manifest)
    atomic_json(args.out_dir/"summary.json", row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")
    atomic_json(args.out_dir/"progress.json", {"phase":"worker_complete_pending_independent_audit"})


def audit(args):
    manifest = c.read_json(args.manifest)
    row = c.read_json(args.out_dir/"summary.json")
    spec = s.Spec(**row["spec"])
    if not (args.out_dir/"DONE").is_file() or row["gpu"] != args.gpu or row["protocol"] != s.PROTOCOL:
        raise RuntimeError("Incomplete or incompatible replay target")
    if row["source_hashes"] != manifest["source_hashes"] or row["input_hashes"] != manifest["case"]["input_hashes"]:
        raise RuntimeError("Run provenance mismatch")
    device, batch = setup(args, manifest)
    states, fields, checks = [], [], []
    previous = None
    for stage, block in zip(s.stages(spec), row["blocks"]):
        path = Path(block["model_path"])
        for filename, digest in ((path,block["model_sha256"]), (path.parent/"last.pt",block["last_model_sha256"]),
                                  (path.parent/"history.csv",block["history_sha256"]), (Path(block["dvf_path"]),block["dvf_sha256"])):
            if sha(filename) != digest:
                raise RuntimeError("Saved checkpoint/history/field changed")
        state = torch.load(path, map_location=device, weights_only=False)
        controls = state["velocity_controls_norm_xyz"]
        obj = s.Objective(batch["fixed"], batch["moving"], stage, spec, fields if spec.formulation == "compose" else ())
        initial = (c.cubic_bspline_dyadic_knot_insert_3d(previous, obj.control_shape)
                   if spec.formulation == "refine" and previous is not None else torch.zeros_like(controls))
        with torch.no_grad():
            values = [float(x) for x in obj(controls)]
            initial_values = [float(x) for x in obj(initial)]
            info = block["optimizer"]
            if state["summary"] != info or any(not close(a,b) for a,b in zip(initial_values,block["initial_loss_data_reg"])):
                raise RuntimeError("Checkpoint metadata or initial loss mismatch")
            if any(not close(value,info[key]) for value,key in zip(values,("best_loss","best_data_loss","best_reg_loss"))):
                raise RuntimeError("Checkpoint loss mismatch")
            current = obj.field(controls, native=True)
            if not np.array_equal(current.cpu().numpy(), np.load(block["dvf_path"])):
                raise RuntimeError("Independent stage field replay mismatch")
            v = obj.expand(controls)
            corners = topology.corner_diagnostics(v*obj.native_scale/128)
            if corners != block["scaled_initial_map_corners"]:
                raise RuntimeError("Scaled elementary-map derivative replay mismatch")
            atomic = s.exp_velocity(v, 7, obj.native_identity)
        with (path.parent/"history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda x:float(x["loss"]))
        if (int(best["epoch"]) != info["best_epoch"] or not close(best["loss"],info["best_loss"]) or
                len(history) != info["actual_epochs"]+1 or info["best_loss"] > info["initial_loss"]+1e-7):
            raise RuntimeError("State-indexed best/history mismatch")
        if not all(int(h["epoch"]) == i and close(h["loss"], float(h["data_loss"])+float(h["reg_loss"])) for i,h in enumerate(history)):
            raise RuntimeError("History epoch/objective mismatch")
        checks.append({"cps":stage.cps,"checkpoint_loss_verified":True,"exact_field_replay":True,"history_verified":True})
        states.append((stage.cps, controls.cpu()))
        if spec.formulation == "compose": fields.append(atomic)
        previous = controls
        del obj, initial, state, v
    if len(checks) != len(spec.schedule):
        raise RuntimeError("Missing stage checkpoints")
    replay = metric_row(current, batch, args.out_dir/"independent_metrics")
    for key in (*METRICS, "valid_label_count", "excluded_labels"):
        if replay[key] != row[key]:
            raise RuntimeError(f"Canonical replay mismatch: {key}")
    with (args.out_dir/"final_metrics/case_metrics.csv").open() as f:
        original_labels = next(csv.DictReader(f))
    with (args.out_dir/"independent_metrics/case_metrics.csv").open() as f:
        replay_labels = next(csv.DictReader(f))
    if original_labels != replay_labels:
        raise RuntimeError("Per-label canonical metrics differ")
    diagnostic_checks = []
    for diagnostic in row["integration_diagnostics"]:
        steps = diagnostic["integration_steps"]
        with torch.no_grad():
            u, parts, _ = s.reconstruct(states, spec.formulation, c.IMAGE_SHAPE, device, steps)
            _, inverse, _ = s.reconstruct(states, spec.formulation, c.IMAGE_SHAPE, device, steps, inverse=True)
            if s.inverse_consistency(parts, inverse, batch["fixed_mask"]) != diagnostic["inverse_consistency"]:
                raise RuntimeError("Inverse diagnostic replay mismatch")
            if steps != 7:
                if sha(diagnostic["dvf_path"]) != diagnostic["dvf_sha256"] or not np.array_equal(u.cpu().numpy(),np.load(diagnostic["dvf_path"])):
                    raise RuntimeError("Integration sensitivity field mismatch")
                other = metric_row(u, batch, args.out_dir/f"independent_ss{steps}_metrics")
                if any(other[key] != diagnostic["metrics"][key] for key in METRICS):
                    raise RuntimeError("Sensitivity canonical metric mismatch")
        if spec.formulation == "compose":
            probe = topology.offgrid_diagnostics(parts,u,batch["fixed_mask"],count=100000,seed=1)
            if probe != diagnostic["offgrid"]: raise RuntimeError("Offgrid diagnostic mismatch")
        diagnostic_checks.append({"steps":steps,"passed":True})
        del u,parts,inverse
    assert_manifest(manifest)
    atomic_json(args.out_dir/"independent_audit.json", {"passed":True,"checks":checks,
                "integration_checks":diagnostic_checks,"all_label_metrics_reproduced":True,
                "summary_sha256":sha(args.out_dir/"summary.json"),"source_hashes":source_hashes(),"completed_at":c.now_text()})


def validate(path, manifest, spec):
    row = c.read_json(path/"summary.json")
    audit_row = c.read_json(path/"independent_audit.json")
    if (not (path/"DONE").is_file() or row["spec"] != asdict(spec) or
            row["source_hashes"] != manifest["source_hashes"] or not audit_row["passed"] or
            audit_row["summary_sha256"] != sha(path/"summary.json") or sha(row["dvf_path"]) != row["dvf_sha256"]):
        raise RuntimeError("Missing or inconsistent accepted run")
    return row


def report(rows, manifest, out):
    if len(rows) != 8 or {json.dumps(r["spec"],sort_keys=True) for r in rows} != {json.dumps(asdict(x),sort_keys=True) for x in s.matrix()}:
        raise RuntimeError("Require exactly eight completed production cells")
    assert_manifest(manifest)
    out.mkdir(parents=True,exist_ok=True)
    combined, stage_rows, sensitivity, paired = [], [], [], []
    for row in rows:
        spec = s.Spec(**row["spec"])
        flat = {"schedule":spec.schedule,"method":f"SVF {spec.formulation}; diffusion on {spec.regularize}",
                "source_kind":"new_svf",**{k:row[k] for k in (*METRICS,"gpu","optimizer_sec","worker_sec","actual_adam_updates","peak_reserved_gib","run_dir")}}
        combined.append(flat)
        for block in row["blocks"]:
            stage_rows.append({"run_name":row["run_name"],"schedule":spec.schedule,"method":flat["method"],
                "cps":block["cps"],**block["optimizer"],"initial_map_derivative_bound":block["scaled_initial_map_corners"]["max_displacement_jacobian_infinity_norm"]})
        for diag in row["integration_diagnostics"]:
            sensitivity.append({"run_name":row["run_name"],"schedule":spec.schedule,"method":flat["method"],
                "integration_steps":diag["integration_steps"],**{k:diag["metrics"][k] for k in METRICS},
                **{k:v for k,v in diag.items() if k.startswith("delta_from")},
                **{f"delta_ss7_{k}":v for k,v in diag["delta_primary_metrics"].items()},
                "inverse_after_forward_mean_mm":diag["inverse_consistency"]["inverse_after_forward"]["all_mean_mm"],
                "inverse_after_forward_p95_mm":diag["inverse_consistency"]["inverse_after_forward"]["all_p95_mm"]})
        for ref in manifest["references"]:
            if ref["schedule"] == spec.schedule:
                paired.append({"schedule":spec.schedule,"method":flat["method"],"reference":ref["method"],
                               **{f"delta_{k}":row[k]-ref[k] for k in METRICS}})
    for ref in manifest["references"]:
        combined.append({k:ref.get(k) for k in ("schedule","method","source_kind",*METRICS,"gpu","optimizer_sec","worker_sec","actual_adam_updates","run_dir")})
    c.write_csv(out/"comparison.csv",combined)
    c.write_csv(out/"paired_differences.csv",paired)
    c.write_csv(out/"stage_diagnostics.csv",stage_rows)
    c.write_csv(out/"integration_sensitivity.csv",sensitivity)
    atomic_json(out/"results.json", {"complete":True,"variant":s.VARIANT,"candidates":rows,
                "comparison":combined,"paired_differences":paired,"stage_diagnostics":stage_rows,
                "integration_sensitivity":sensitivity,"provenance":manifest})
    text = ["# Rank-0 B-spline SVF pilot", "", "Exploratory shuffled rank0: fixed0333, moving0115. All active stage weights1; squared local NCC9. No across-case SD.",
            "", "Seven-step native-grid integration is primary. Velocity versus displacement lambda1 does not imply equal constraint strength. Historical prolonged-displacement coarse-grid evaluation is directly cubic; SVF deformations are integrated natively and interpolated onto that grid. Shared-workload timing is diagnostic.", ""]
    for schedule in ("842","8421"):
        text += [f"## {' → '.join(schedule)}", "", "| Method | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % | Optimizer s | Updates |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in (x for x in combined if x["schedule"]==schedule):
            values = " | ".join(f"{row[k]:.6f}" for k in METRICS)
            runtime = "n/a" if row.get("optimizer_sec") is None else f"{row['optimizer_sec']:.2f}"
            text.append(f"| {row['method']} | {values} | {runtime} | {row.get('actual_adam_updates') or 'n/a'} |")
        text.append("")
    text += ["## Integration diagnostics", "", "See integration_sensitivity.csv for 7/8/10-step accuracy, regularity, displacement changes and inverse residuals. Complete corner/off-grid diagnostics and forward/reverse inverse residuals are retained per run in results.json. Higher-step exports never replace primary seven-step fields.", ""]
    (out/"report.md").write_text("\n".join(text))
    if len(combined)!=24 or len(sensitivity)!=24 or len(stage_rows)!=28 or len(paired)!=64:
        raise RuntimeError("Report row-count check failed")
    with (out/"comparison.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    if len(saved)!=24 or any(not close(float(a[k]),b[k],rel=0,abs_=1e-12) for a,b in zip(saved,combined) for k in METRICS):
        raise RuntimeError("CSV/JSON comparison consistency failed")
    assert_manifest(manifest)
    outputs = ("comparison.csv","paired_differences.csv","stage_diagnostics.csv","integration_sensitivity.csv","results.json","report.md")
    atomic_json(out/"ACCEPTED.json", {"passed":True,"candidate_count":8,"independent_replay_count":8,
                "output_sha256":{name:sha(out/name) for name in outputs},"completed_at":c.now_text()})


def phase(root, manifest, smoke, required_mib):
    specs = s.matrix(smoke)
    jobs = [{"spec":spec,"status":"pending","attempt":0,"gpu":manifest["family_gpu"][f"{spec.formulation}_{spec.regularize}"]} for spec in specs]
    active, completed, failed = {}, [], []
    phase_name = "smoke" if smoke else "production"
    started = time.monotonic()
    # Idempotent recovery requires a complete independent audit, never just DONE.
    for job in jobs:
        for attempt in (0,1):
            path = root/job["spec"].name()/f"attempt{attempt}"
            if (path/"independent_audit.json").is_file():
                completed.append(validate(path,manifest,job["spec"]))
                job.update(status="complete",attempt=attempt,out=path)
                break
    while any(x["status"] not in ("complete","failed") for x in jobs) or active:
        for gpu, item in list(active.items()):
            proc, job = item["process"], item["job"]
            if proc.poll() is None: continue
            item["log"].close()
            if proc.returncode:
                content = item["log_path"].read_text(errors="replace")
                failure = {"spec":asdict(job["spec"]),"gpu":gpu,"attempt":job["attempt"],"phase":item["action"],"returncode":proc.returncode,"log":str(item["log_path"])}
                atomic_json(job["out"].parent/f"failed_attempt{job['attempt']}_{item['action']}.json",failure)
                oom = "out of memory" in content.lower() and "cuda" in content.lower()
                if oom and job["attempt"]==0 and item["action"]=="worker":
                    job.update(status="retry",attempt=1)
                else:
                    job["status"]="failed"; failed.append(failure)
                del active[gpu]
                continue
            if item["action"]=="worker":
                job["status"]="audit_pending"
            else:
                completed.append(validate(job["out"],manifest,job["spec"]))
                job["status"]="complete"
            del active[gpu]
        snapshot = legacy.gpu_snapshot()
        available = {str(x["physical_gpu"]):x for x in snapshot.get("rows",[]) if str(x["physical_gpu"]) not in active and x["memory_free_mib"] >= required_mib}
        for job in jobs:
            if job["status"] in ("running","auditing","complete","failed"): continue
            if failed and job["status"] not in ("audit_pending",):
                job["status"]="failed"; continue
            spec = job["spec"]
            if job["status"]=="pending" and spec.schedule=="8421":
                predecessor=next(x for x in jobs if x["spec"].formulation==spec.formulation and x["spec"].regularize==spec.regularize and x["spec"].schedule=="842")
                if predecessor["status"]!="complete": continue
            choices = list(available) if job["status"]=="retry" else [job["gpu"]]
            choices = [gpu for gpu in choices if gpu in available]
            if not choices: continue
            gpu=max(choices,key=lambda x:available[x]["memory_free_mib"])
            action="audit" if job["status"]=="audit_pending" else "worker"
            path=job.get("out") if action=="audit" else root/spec.name()/f"attempt{job['attempt']}"
            if action=="worker" and path.exists():
                # A stopped controller can resume a finished worker's audit.
                if (path/"DONE").is_file():
                    job.update(status="audit_pending",out=path,gpu=c.read_json(path/"summary.json")["gpu"])
                    continue
                raise RuntimeError(f"Preserve incomplete run for audit: {path}")
            path.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,"-u","-B",str(Path(__file__)),action,"--manifest",str(root/"launch_manifest.json"),"--out-dir",str(path),"--gpu",gpu]
            if action=="worker":
                command += ["--schedule",spec.schedule,"--formulation",spec.formulation,"--regularize",spec.regularize,"--integration-steps","7","--case-rank","0"]
                if smoke: command.append("--smoke")
            log_path=path.parent/f"attempt{job['attempt']}_{action}.log"
            log=log_path.open("x")
            env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            job.update(status="auditing" if action=="audit" else "running",gpu=gpu,out=path)
            active[gpu]={"process":proc,"job":job,"action":action,"log":log,"log_path":log_path}
            print(json.dumps({"launched":spec.name(),"action":action,"gpu":gpu,"pid":proc.pid,"gpu_before":available[gpu]}),flush=True)
            del available[gpu]
        status={"phase":phase_name,"completed":len(completed),"total":8,"elapsed_sec":time.monotonic()-started,
                "required_free_mib":required_mib,"failed":failed,"updated_at":c.now_text(),
                "jobs":[{"spec":asdict(x["spec"]),"status":x["status"],"gpu":x["gpu"],"attempt":x["attempt"]} for x in jobs]}
        atomic_json(root/"progress.json",status)
        if active or any(x["status"] not in ("complete","failed") for x in jobs): time.sleep(5)
    if failed:
        atomic_json(root/f"{phase_name}_FAILED.json",{"failures":failed,"completed":len(completed)})
        raise RuntimeError(f"{phase_name} failure preserved; stop for numerical audit")
    return completed


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    root=args.artifact_root/"runs"/s.VARIANT
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=root/"launch_manifest.json"
        manifest=c.read_json(manifest_path) if manifest_path.exists() else prepare(args.artifact_root,args.gpus.split(","))
        assert_manifest(manifest)
        if not manifest_path.exists(): atomic_json(manifest_path,manifest)
        smokes=phase(root,manifest,True,8192)
        required=max(8192,math.ceil((max(r["peak_reserved_gib"] for r in smokes)+2)*1024))
        atomic_json(root/"smokes_passed.json",{"count":8,"required_free_mib":required,"runs":[r["run_dir"] for r in smokes],"source_hashes":source_hashes()})
        if args.stop_after=="smoke": return
        rows=phase(root,manifest,False,required)
        report(rows,manifest,args.artifact_root/"repo_experiments"/s.VARIANT)
        atomic_json(root/"COMPLETE.json",{"count":8,"completed_at":c.now_text()})
        atomic_json(root/"progress.json",{"phase":"accepted","completed":8,"total":8,"updated_at":c.now_text()})


if __name__=="__main__":
    args=parse_args()
    try:
        {"launch":launch,"worker":worker,"audit":audit}[args.command](args)
    except BaseException:
        traceback.print_exc()
        if getattr(args,"out_dir",None) and args.out_dir.is_dir():
            atomic_json(args.out_dir/f"FAILED_{args.command}.json",{"traceback":traceback.format_exc(),"created_at":c.now_text()})
        raise
