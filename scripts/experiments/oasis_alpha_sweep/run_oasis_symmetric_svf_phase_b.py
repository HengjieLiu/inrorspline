#!/usr/bin/env python
"""Isolated P78 Phase-B preparation, GPU worker and independent replay."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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

import symmetric_bspline_svf_phase_b as m
import run_oasis_symmetric_svf as phase_a
import symmetric_svf_diagnostics as diag
import run_oasis_bspline_svf as p
import svf_transition_validation as transition

c, d, sha, atomic_json = m.c, m.d, p.sha, p.atomic_json
HERE = Path(__file__).resolve().parent
METRICS = p.METRICS
P76 = "oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1"


def source_hashes():
    return {**phase_a.source_hashes(), **{name: sha(HERE/name) for name in (
        "symmetric_bspline_svf_phase_b.py", "run_oasis_symmetric_svf_phase_b.py",
        "queue_oasis_symmetric_svf_phase_b.py")},
        "test_oasis_symmetric_svf_phase_b.py": sha(c.REPO_ROOT/"tests/test_oasis_symmetric_svf_phase_b.py")}


def assert_manifest(manifest):
    if manifest["source_hashes"] != source_hashes() or manifest["protocol"] != m.PROTOCOL:
        raise RuntimeError("Numerical source/protocol drift")
    for path, digest in manifest["reference_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Reference changed: {path}")
    if sha(manifest["settings"]["pair_file"]) != manifest["pair_file_sha256"]:
        raise RuntimeError("Split changed")


def gradient_parity(device):
    rng = torch.Generator(device=device).manual_seed(19)
    a = torch.rand((1,1,10,12,14), generator=rng, device=device, requires_grad=True)
    b = torch.rand(a.shape, generator=rng, device=device, requires_grad=True)
    pair = m.NCC9Pair(device)
    value = pair(a,b)
    ga,gb = torch.autograd.grad(value,(a,b))
    va,vb = d.NCC9(b)(a),d.NCC9(a)(b)
    ra, = torch.autograd.grad(va,a)
    rb, = torch.autograd.grad(vb,b)
    for actual,expected in ((value,va),(value,vb),(ga,ra),(gb,rb),(value,pair(b,a))):
        if not torch.allclose(actual,expected,atol=2e-6,rtol=1e-5):
            raise RuntimeError("Two-gradient NCC parity failed")
    return {"passed": True, "a_gradient_max_abs_error": float((ga-ra).abs().max()),
            "b_gradient_max_abs_error": float((gb-rb).abs().max()),
            "value_abs_error": float((value-vb).abs())}


def prepare(args):
    if (os.getuid(), os.getgid()) != (1009,1020): raise RuntimeError("Require 1009:1020")
    root = args.artifact_root/"runs"/m.VARIANT/m.PHASE
    oldroot = args.artifact_root/"runs"/m.VARIANT/"phase_a"
    report = args.artifact_root/"repo_experiments"/m.VARIANT/"phase_a"
    parent = c.read_json(oldroot/"launch_manifest.json")
    phase_a.assert_manifest(parent)
    accepted = c.read_json(report/"ACCEPTED.json")
    if not accepted["complete"] or accepted["fits"] != 4: raise RuntimeError("Phase A incomplete")
    hashes = {str(oldroot/"launch_manifest.json"): sha(oldroot/"launch_manifest.json"),
              str(report/"ACCEPTED.json"): sha(report/"ACCEPTED.json"),
              **parent["reference_hashes"]}
    for name, digest in accepted["output_sha256"].items():
        if sha(report/name) != digest: raise RuntimeError("Accepted Phase-A report drift")
        hashes[str(report/name)] = digest
    refs = list(parent["references"])
    oldresult = c.read_json(report/"results.json")
    for row in oldresult["comparison"]:
        if row["arm"] in ("A1", "A2", "A3"):
            refs.append({**row, "series": row["arm"]})
    free = shutil.disk_usage(args.artifact_root).free/2**30
    if free < 100: raise RuntimeError("Require at least 100 GiB artifact capacity")
    manifest = {**parent, "phase": m.PHASE, "protocol": m.PROTOCOL,
                "reference_hashes": hashes, "references": refs, "source_hashes": source_hashes(),
                "source_revision": subprocess.check_output(["git","rev-parse","HEAD"],cwd=c.REPO_ROOT,text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git","status","--porcelain"],cwd=c.REPO_ROOT,text=True)),
                "container_image": args.container_image, "container_image_id": args.container_image_id,
                "uid": os.getuid(), "gid": os.getgid(), "initial_gpu_state": p.legacy.gpu_snapshot(),
                "storage_free_gib": free, "production_fits": 28, "smoke_fits": 4,
                "selection": "original_forward_dice_then_asd_then_hd95_then_lower_weights",
                "amplitude_grid": list(m.ALPHAS), "balance_grid": list(m.BETAS),
                "created_at": c.now_text()}
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root/"launch_manifest.json", manifest)
    print(json.dumps({"manifest": str(root/"launch_manifest.json"), "fits":28}),flush=True)


def setup(args, manifest, spec):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID 1009:1020")
    assert_manifest(manifest); spec.validate()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    args.case_rank = 0
    for key, value in manifest["settings"].items(): setattr(args, key, Path(value))
    batch, hashes = p.legacy.load_batch(args, device)
    if hashes != manifest["case"]["input_hashes"] or (batch["fixed_id"], batch["moving_id"]) != ("0333", "0115"):
        raise RuntimeError("Wrong case/input hashes")
    if spec.swapped: batch = diag.swapped_batch(batch)
    for role in ("fixed", "moving"):
        if not torch.equal(batch[role+"_image_mask"].bool(), batch[role] > 0):
            raise RuntimeError("Wrong raw-positive mask")
    return device, batch


@torch.no_grad()
def validate_initial(previous, initial, objective):
    if previous is None:
        if torch.count_nonzero(initial): raise RuntimeError("Nonidentity start")
        return {"identity": True}
    previous_controls = m.branch_controls(previous, objective.spec.tied)
    controls = m.branch_controls(initial, objective.spec.tied)
    results = []
    for a, b in zip(previous_controls, controls):
        old_expand = d.AlignedBSpline(a.shape[:3], objective.image_shape, objective.image_shape,
                                      2*objective.stage.cps, a.device)
        v0, v1 = old_expand(a), objective.expand(b)
        u0 = m.s.exp_velocity(v0, 7, objective.native_identity)
        u1 = m.s.exp_velocity(v1, 7, objective.native_identity)
        results.append(transition.verify_transition(a, b, v0, v1, u0, u1,
                                                    objective.stage.cps, objective.native_scale))
    return {"branches": results}


def check_history(path, info, coefficient):
    with path.open() as f: rows = list(csv.DictReader(f))
    if len(rows) != info["actual_epochs"]+1: raise RuntimeError("Incomplete history")
    best = min(rows, key=lambda x: float(x["loss"]))
    if int(best["epoch"]) != info["best_epoch"] or not p.close(best["loss"], info["best_loss"]):
        raise RuntimeError("Incorrect joint best checkpoint")
    for i, row in enumerate(rows):
        if (int(row["epoch"]) != i or int(row["optimizer_updates_completed"]) != i
                or float(row["lr"]) != info["stage"]["lr"]):
            raise RuntimeError("Update/LR schedule drift")
        if not all(math.isfinite(float(row[k])) for k in ("loss", "data_loss", "reg_loss", "R1", "R2", "amplitude_mm2", "balance_mm2",
                         "weighted_amplitude", "weighted_balance", "E1_mm2", "E2_mm2", "a1_mm", "a2_mm")):
            raise RuntimeError("Nonfinite history")
        if (not p.close(row["reg_loss"], float(row["R1"])+float(row["R2"]))
                or not p.close(row["loss"], float(row["data_loss"])+coefficient*float(row["reg_loss"])
                              +float(row["weighted_amplitude"])+float(row["weighted_balance"]))):
            raise RuntimeError("Incorrect weighted objective history")
        if (not p.close(row["amplitude_mm2"], float(row["E1_mm2"])+float(row["E2_mm2"]))
                or not p.close(row["balance_mm2"], (float(row["a1_mm"])-float(row["a2_mm"]))**2)
                or not p.close(row["weighted_amplitude"], info["amplitude_weight"]*float(row["amplitude_mm2"]))
                or not p.close(row["weighted_balance"], info["balance_weight"]*float(row["balance_mm2"]))):
            raise RuntimeError("Incorrect amplitude/balance history")
    for key, value in info["selected_terms"].items():
        if not p.close(best[key], value): raise RuntimeError("Selected penalty terms mismatch")
    return len(rows)


def same_tree(a, b, path="", tolerance=1e-10):
    if isinstance(a, dict):
        if set(a) != set(b): raise RuntimeError(f"Replay keys differ:{path}")
        for k in a: same_tree(a[k], b[k], f"{path}/{k}", tolerance)
    elif isinstance(a, list):
        if len(a) != len(b): raise RuntimeError(f"Replay length differs:{path}")
        for i, (x, y) in enumerate(zip(a, b)): same_tree(x, y, f"{path}/{i}", tolerance)
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
            raise RuntimeError(f"Replay numeric mismatch:{path}:{a}!={b}")
    elif a != b: raise RuntimeError(f"Replay mismatch:{path}")


@torch.no_grad()
def evaluations(parameter, spec, batch, out):
    result, primary = [], None
    for steps in ([7] if spec.smoke else [7, 8, 10]):
        state = m.reconstruct(parameter, 2, c.IMAGE_SHAPE, spec.tied, steps)
        folder = out/f"ss{steps}"
        folder.mkdir(parents=True, exist_ok=False)
        record = {"integration_steps": steps}
        for direction, target in (("forward", batch), ("reverse", diag.swapped_batch(batch))):
            path = folder/f"{direction}_disp_norm_xyz.npy"
            record[direction] = {"dvf_path": str(path), "dvf_sha256": p.save_field(path, state[direction]),
                                 "metrics": p.metric_row(state[direction], target, folder/direction)}
            if primary is not None:
                record[direction]["difference_from_ss7"] = c.field_difference_metrics(
                    (state[direction]-primary[direction])*d.voxel_scale(c.IMAGE_SHAPE, device=parameter.device))
        record["offgrid"] = diag.offgrid(state, batch)
        if steps == 7:
            primary = {k: state[k] for k in ("forward", "reverse")}
            record["branch_diagnostics"] = branch_diagnostics(state, batch)
        result.append(record)
        del state
    return result


@torch.no_grad()
def branch_diagnostics(state, batch, obj=None, parameter=None):
    result = diag.branch_diagnostics(state, batch, obj, parameter)
    scale = d.voxel_scale(state["velocities"][0].shape[:3], device=batch["fixed"].device)
    opposing = (state["velocities"][0]-state["velocities"][1])*.5*scale
    union = (batch["fixed"]>0) | (batch["moving"]>0)
    for key, field in (("full_volume",opposing),("raw_foreground_union",opposing[union])):
        result[key]["opposing_mode_rms_mm"] = float(field.square().sum(-1).mean().sqrt())
    return result


def worker(args):
    started = time.monotonic()
    manifest, spec = c.read_json(args.manifest), m.Spec(args.arm, args.alpha, args.beta, args.smoke, args.swapped, args.repeat)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    gpu_before = p.legacy.gpu_snapshot()
    device, batch = setup(args, manifest, spec)
    row = {"variant": m.VARIANT, "phase": m.PHASE, "method": m.METHOD, "spec": asdict(spec),
           "protocol": m.PROTOCOL, "run_name": spec.name(), "run_dir": str(args.out_dir),
           "case_rank": 0, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
           "canonical_pair": ["0333", "0115"], "direction": "moving_to_fixed",
           "sampling": "exp(v1) o exp(-v2)_fixed_to_moving_pull", "shape": [160,224,192,3],
           "input_hashes": manifest["case"]["input_hashes"], "source_hashes": manifest["source_hashes"],
           "source_revision": manifest["source_revision"], "source_dirty": manifest["source_dirty"],
           "container_image": manifest["container_image"], "container_image_id": manifest["container_image_id"],
           "pair_file_sha256": manifest["pair_file_sha256"], "gpu": args.gpu, "visible_device": "cuda:0",
           "uid": os.getuid(), "gid": os.getgid(), "backend": c.cuda_backend_metadata(device),
           "tf32": False, "seed": 1, "initial_gpu_state": gpu_before,
           "diffusion_coefficient": spec.coefficient, "equal_split_effective_lambda": spec.coefficient/2,
           "amplitude_weight": spec.alpha, "balance_weight": spec.beta, "segmentation_supervision": False,
           "cuda_nondeterminism_caveat": "grid_sample backward may remain nondeterministic",
           "started_at": c.now_text(), "argv": sys.argv}
    if spec.smoke:
        row["native_legacy_parity"] = p.legacy.native_parity(device, args.dl_code_root, args.vfa_root)
        row["two_gradient_ncc_parity"] = gradient_parity(device)
        row["phase_a_and_exchange_parity"] = m.zero_parity(device)
    atomic_json(args.out_dir/"manifest.json", row)
    previous, blocks, peak_allocated, peak_reserved = None, [], 0., 0.
    for stage in m.stages(spec):
        obj = m.Objective(batch["fixed"], batch["moving"], stage, spec)
        initial = (m.prolong(previous, obj.control_shape) if previous is not None else
                   torch.zeros(1 if spec.tied else 2, *obj.control_shape, 3, device=device))
        init_check = validate_initial(previous, initial, obj)
        with torch.no_grad(): initial_losses = [float(v) for v in obj(initial)]
        if spec.smoke:
            probe = initial.detach().clone().requires_grad_()
            gradient, = torch.autograd.grad(obj(probe)[0], probe)
            norms = gradient.flatten(1).abs().amax(1)
            if not torch.isfinite(gradient).all() or not (norms > 0).all():
                raise RuntimeError("Full-size midpoint gradient missing in a branch")
            init_check["full_size_gradient_max_abs_per_parameter_branch"] = norms.cpu().tolist()
            del probe, gradient, norms
        stage_dir = args.out_dir/f"stage{stage.index}_cps{stage.cps}"
        atomic_json(args.out_dir/"progress.json", {"phase": "optimization", "cps": stage.cps})
        best, info = m.optimize(initial, obj, stage_dir)
        with torch.no_grad():
            state = m.reconstruct(best, stage.cps, c.IMAGE_SHAPE, spec.tied)
            block = {"cps": stage.cps, "stage_index": stage.index, "optimizer": info,
                     "initialization": init_check, "initial_loss_data_reg": initial_losses,
                     "model_path": str(stage_dir/"best.pt"), "model_sha256": sha(stage_dir/"best.pt"),
                     "last_model_sha256": sha(stage_dir/"last.pt"), "history_sha256": sha(stage_dir/"history.csv"),
                     "branches": branch_diagnostics(state, batch, obj, best),
                     "scaled_initial_map_corners": [p.topology.corner_diagnostics(v*obj.native_scale/128)
                                                    for v in state["velocities"]]}
            for direction in ("forward", "reverse"):
                path = stage_dir/f"{direction}_disp_norm_xyz.npy"
                block[direction] = {"dvf_path": str(path), "dvf_sha256": p.save_field(path, state[direction])}
            check_history(stage_dir/"history.csv", info, spec.coefficient)
            atomic_json(stage_dir/"validated.json", block)
        blocks.append(block); previous = best
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated()/2**30)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved()/2**30)
        print(json.dumps({"arm": spec.arm, "completed_cps": stage.cps, "updates": info["actual_epochs"],
                          "best_epoch": info["best_epoch"], "branches": block["branches"]["full_volume"]}), flush=True)
        del obj, initial, state
    metric_started = time.monotonic()
    atomic_json(args.out_dir/"progress.json", {"phase": "metrics_and_diagnostics"})
    ev = evaluations(previous, spec, batch, args.out_dir/"evaluations")
    row.update(blocks=blocks, evaluations=ev, **ev[0]["forward"]["metrics"],
               reverse_metrics=ev[0]["reverse"]["metrics"],
               dvf_path=ev[0]["forward"]["dvf_path"], dvf_sha256=ev[0]["forward"]["dvf_sha256"],
               actual_adam_updates=sum(b["optimizer"]["actual_epochs"] for b in blocks),
               optimizer_sec=sum(b["optimizer"]["training_elapsed_sec"] for b in blocks),
               metrics_diagnostics_sec=time.monotonic()-metric_started, worker_sec=time.monotonic()-started,
               peak_cuda_gib=max(peak_allocated, torch.cuda.max_memory_allocated()/2**30),
               peak_reserved_gib=max(peak_reserved, torch.cuda.max_memory_reserved()/2**30),
               final_gpu_state=p.legacy.gpu_snapshot(), completed_at=c.now_text())
    assert_manifest(manifest)
    atomic_json(args.out_dir/"summary.json", row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def audit(args):
    started = time.monotonic()
    manifest, row = c.read_json(args.manifest), c.read_json(args.out_dir/"summary.json")
    spec = m.Spec(**row["spec"])
    if not (args.out_dir/"DONE").is_file(): raise RuntimeError("Incomplete worker")
    device, batch = setup(args, manifest, spec)
    if (row["source_hashes"] != manifest["source_hashes"] or row["protocol"] != m.PROTOCOL
            or row["shape"] != [160,224,192,3] or row["direction"] != "moving_to_fixed"
            or row["fixed_id"] != batch["fixed_id"] or row["moving_id"] != batch["moving_id"]
            or row["input_hashes"] != manifest["case"]["input_hashes"] or len(row["blocks"]) != 3):
        raise RuntimeError("Replay metadata mismatch")
    previous, checked_rows = None, 0
    for stage, block in zip(m.stages(spec), row["blocks"]):
        path = Path(block["model_path"])
        for name, digest in ((path, block["model_sha256"]), (path.parent/"last.pt", block["last_model_sha256"]),
                             (path.parent/"history.csv", block["history_sha256"])):
            if sha(name) != digest: raise RuntimeError("Checkpoint/history drift")
        saved = torch.load(path, map_location=device, weights_only=False)
        parameter, info = saved["twice_branch_controls_norm_xyz"], block["optimizer"]
        if saved["summary"] != info or info["stage"] != asdict(stage):
            raise RuntimeError("Saved schedule/state mismatch")
        obj = m.Objective(batch["fixed"], batch["moving"], stage, spec)
        initial = m.prolong(previous, obj.control_shape) if previous is not None else torch.zeros_like(parameter)
        actual_initial = validate_initial(previous, initial, obj)
        if spec.smoke:
            probe=initial.detach().clone().requires_grad_()
            gradient,=torch.autograd.grad(obj(probe)[0],probe)
            actual_initial["full_size_gradient_max_abs_per_parameter_branch"] = gradient.flatten(1).abs().amax(1).detach().cpu().tolist()
            del probe, gradient
        same_tree(actual_initial, block["initialization"], tolerance=1e-6)
        with torch.no_grad():
            for actual, expected in zip(obj(initial), block["initial_loss_data_reg"]):
                if not p.close(actual, expected): raise RuntimeError("Initial loss replay mismatch")
            for actual, key in zip(obj(parameter), ("best_loss", "best_data_loss", "best_reg_loss")):
                if not p.close(actual, info[key]): raise RuntimeError("Selected loss replay mismatch")
            for key, value in info["selected_terms"].items():
                if not p.close(obj.last_terms[key], value): raise RuntimeError("Selected term replay mismatch")
            state = m.reconstruct(parameter, stage.cps, c.IMAGE_SHAPE, spec.tied)
            for direction in ("forward", "reverse"):
                entry = block[direction]
                if sha(entry["dvf_path"]) != entry["dvf_sha256"] or not np.array_equal(np.load(entry["dvf_path"]), state[direction].cpu().numpy()):
                    raise RuntimeError("Stage field does not reconstruct exactly")
            same_tree(branch_diagnostics(state, batch, obj, parameter), block["branches"])
            same_tree([p.topology.corner_diagnostics(v*obj.native_scale/128) for v in state["velocities"]], block["scaled_initial_map_corners"])
        checked_rows += check_history(path.parent/"history.csv", info, spec.coefficient)
        previous = parameter
        del obj, initial, state, saved
    replay = evaluations(previous, spec, batch, args.out_dir/f"independent_replay_{time.time_ns()}")
    if len(replay) != len(row["evaluations"]): raise RuntimeError("Missing sensitivity evaluations")
    for actual, expected in zip(replay, row["evaluations"]):
        for direction in ("forward", "reverse"):
            a, b = actual[direction], expected[direction]
            if a["dvf_sha256"] != b["dvf_sha256"] or sha(b["dvf_path"]) != b["dvf_sha256"]:
                raise RuntimeError("Final field/sensitivity replay mismatch")
            same_tree(a["metrics"], b["metrics"], tolerance=1e-12)
            pa, pb = Path(a["dvf_path"]).parent/direction/"case_metrics.csv", Path(b["dvf_path"]).parent/direction/"case_metrics.csv"
            if sha(pa) != sha(pb): raise RuntimeError("Per-label metrics replay mismatch")
            if "difference_from_ss7" in a: same_tree(a["difference_from_ss7"], b["difference_from_ss7"])
        same_tree(actual["offgrid"], expected["offgrid"])
        if "branch_diagnostics" in actual: same_tree(actual["branch_diagnostics"], expected["branch_diagnostics"])
    assert_manifest(manifest)
    atomic_json(args.out_dir/"independent_audit.json", {"passed": True, "history_rows_checked": checked_rows,
                "exact_field_replay": True, "all_label_metrics_reproduced": True,
                "source_hashes": source_hashes(), "summary_sha256": sha(args.out_dir/"summary.json"),
                "gpu": args.gpu, "audit_sec": time.monotonic()-started,
                "audit_peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30,
                "audit_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "completed_at": c.now_text()})


def validate(path, manifest, spec):
    row, check = c.read_json(path/"summary.json"), c.read_json(path/"independent_audit.json")
    if (row["spec"] != asdict(spec) or not check["passed"] or not (path/"DONE").is_file()
            or check["summary_sha256"] != sha(path/"summary.json") or row["source_hashes"] != manifest["source_hashes"]
            or check["source_hashes"] != manifest["source_hashes"] or sha(row["dvf_path"]) != row["dvf_sha256"]):
        raise RuntimeError("Incomplete accepted fit")
    return {**row, "audit": check}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    prepare_parser.add_argument("--container-image", required=True)
    prepare_parser.add_argument("--container-image-id", required=True)
    for command in ("worker", "audit"):
        q = sub.add_parser(command)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--out-dir", type=Path, required=True)
        q.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
        if command == "worker":
            q.add_argument("--arm", choices=("base", "amplitude", "balance", "combined"), required=True)
            q.add_argument("--alpha", type=float, default=0.)
            q.add_argument("--beta", type=float, default=0.)
            q.add_argument("--swapped", action="store_true")
            q.add_argument("--repeat", action="store_true")
            q.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "worker":
        try: m.Spec(args.arm,args.alpha,args.beta,args.smoke,args.swapped,args.repeat).validate()
        except ValueError as error: parser.error(str(error))
    path = args.artifact_root if args.command == "prepare" else args.out_dir
    if path.resolve().is_relative_to(c.REPO_ROOT): parser.error("External artifacts only")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        {"prepare": prepare, "worker": worker, "audit": audit}[args.command](args)
    except Exception as error:
        if hasattr(args, "out_dir"):
            args.out_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.out_dir/f"FAILED_{args.command}_{time.time_ns()}.json",
                        {"error": repr(error), "traceback": traceback.format_exc(), "time": c.now_text()})
        raise

