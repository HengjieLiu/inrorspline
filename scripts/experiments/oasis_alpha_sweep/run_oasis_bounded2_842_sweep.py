#!/usr/bin/env python
"""P76: isolated explicit-lambda bounded-two-map cps842 worker and validation."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
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
import run_oasis_bspline_depth_factorial_first10 as parent

b, c, d, legacy = parent.b, parent.c, parent.d, parent.legacy
sha, atomic_json, metric_row = parent.sha, parent.atomic_json, parent.metric_row
VARIANT = "oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1"
LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
CONFIG = "bounded_2_compose"
CONFIGS = {CONFIG: parent.CONFIGS[CONFIG]}
SCHEDULES = {3: (8, 4, 2)}
METRICS_TO_REPORT = parent.METRICS_TO_REPORT
PROTOCOL = {**parent.PROTOCOL, "version": 1, "stage_alpha": "same selected lambda at every level and map",
            "stage_schedule": [{**s, "lambda": "selected"} for s in parent.STAGE_TABLE[:3]],
            "configs": CONFIGS, "depths": {"3": [8, 4, 2]}, "lambda_options": list(LAMBDAS),
            "source_protocol": "P70 depth3 bounded_2_compose; only lambda varies",
            "allowed_adam_updates": 5000, "final_map_count": 6}


def alpha_value(value):
    value = float(value)
    if not math.isfinite(value) or value not in LAMBDAS:
        raise ValueError("Require one of 0.01/0.1/0.3/1/3/10/100")
    return value


def cell_protocol(alpha):
    alpha = alpha_value(alpha)
    return {**PROTOCOL, "depth": 3, "cps_schedule": [8, 4, 2], "stage_alpha": [alpha] * 3,
            "stage_schedule": [{**s, "lambda": alpha} for s in parent.STAGE_TABLE[:3]],
            "config_key": CONFIG, "method": CONFIGS[CONFIG]["method"]}


def run_name(rank, alpha, smoke=False):
    if type(rank) is not int or rank not in range(10):
        raise ValueError("Require shuffled rank0..9")
    spec = cell_protocol(alpha)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    tag = format(alpha, "g").replace(".", "p")
    return ("smoke_" if smoke else "") + f"rank{rank:03d}_cps842_bounded2_lambda{tag}_{digest}"


def sources(sitreg_root):
    paths = [Path(__file__), Path(__file__).with_name("audit_oasis_bounded2_842_sweep.py"),
             Path(__file__).with_name("queue_oasis_bounded2_842_sweep.py"),
             c.REPO_ROOT / "tests/test_oasis_bounded2_842_sweep.py"]
    return {"p70": parent.sources(Path(sitreg_root)),
            "p76": {str(p.relative_to(c.REPO_ROOT)): sha(p) for p in paths}}


def assert_manifest(manifest):
    if manifest["variant"] != VARIANT or manifest["protocol"] != PROTOCOL:
        raise RuntimeError("Wrong P76 manifest")
    if sources(manifest["sitreg_root"]) != manifest["source_hashes"]:
        raise RuntimeError("P76 source drift")
    for path, digest in manifest["reference_hashes"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Read-only reference drift: {path}")


def prepare(root):
    run_root = root / "runs" / VARIANT
    path = run_root / "launch_manifest.json"
    if path.exists():
        manifest = c.read_json(path); assert_manifest(manifest)
        return manifest
    source_root = root / "repo_experiments" / parent.VARIANT
    old_path = root / "runs" / parent.VARIANT / "launch_manifest.json"
    old = c.read_json(old_path)
    parent.assert_manifest(old)
    accepted = c.read_json(source_root / "ACCEPTED.json")
    if not accepted.get("complete"):
        raise RuntimeError("Unaccepted historical P70")
    hashes = {str(source_root / name): digest for name, digest in accepted["output_sha256"].items()}
    hashes.update({str(old_path): sha(old_path), str(source_root / "ACCEPTED.json"): sha(source_root / "ACCEPTED.json")})
    for p, digest in hashes.items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"Historical report changed: {p}")
    data = c.read_json(source_root / "results.json")
    prior = [r for r in data["candidates"] if r["depth"] == 3 and r["config_key"] == CONFIG]
    if sorted(r["case_rank"] for r in prior) != list(range(10)):
        raise RuntimeError("Require all ten accepted lambda1 references")
    for row in prior:
        parent.audit_ok(row)
        if sha(Path(row["dvf_path"])) != row["dvf_sha256"]:
            raise RuntimeError("Historical lambda1 DVF changed")
    manifest = {k: old[k] for k in ("cases", "settings", "bounds", "pair_file_sha256", "references",
                "sitreg_root", "sitreg_revision", "container_image", "container_image_id")}
    manifest.update(variant=VARIANT, protocol=PROTOCOL, source_hashes=sources(old["sitreg_root"]),
                    reference_hashes=hashes, historical_lambda1=prior, gpus=["0", "1", "2", "3"],
                    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=c.REPO_ROOT, text=True),
                    created_at=c.now_text(), uid=os.getuid(), gid=os.getgid(), seed=1,
                    production_fit_count=70, smoke_fit_count=3, initial_gpu_state=legacy.gpu_snapshot(),
                    memory_admission="max(8 GiB, smoke max allocated/reserved + 2 GiB)",
                    fresh_registrations=True, historical_reuse_count=0)
    # Revalidate endpoint-aware bounds and canonical pair ordering before any worker.
    for cps in (8, 4, 2):
        if b.compute_bound(cps) != manifest["bounds"][str(cps)]:
            raise RuntimeError("Bound geometry drift")
    pair_path = c.REPO_ROOT / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    import re
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", pair_path.read_text())[:10]
    if len(pairs) != 10 or any(tuple(manifest["cases"][str(i)][k] for k in ("fixed_id", "moving_id")) != pair for i, pair in enumerate(pairs)):
        raise RuntimeError("Canonical seed3 rank mapping differs")
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(path, manifest)
    return manifest


def verify_objective_parity(objective):
    """Check value and control-gradient parity with unchanged evaluated-field loss."""
    reference = d.StageObjective(objective.fixed, objective.moving, objective.alpha,
                                 replace(objective.stage, cps=None, factor=1))
    numel = math.prod(objective.control_shape) * 3
    p = torch.linspace(-.001, .001, numel, device=objective.identity.device).reshape(*objective.control_shape, 3).requires_grad_()
    actual = objective(p)
    grad_a, = torch.autograd.grad(actual[0], p)
    expected = reference(objective.field(p))
    grad_b, = torch.autograd.grad(expected[0], p)
    if not all(torch.allclose(a, z, rtol=1e-6, atol=1e-7) for a, z in zip(actual, expected)):
        raise RuntimeError("Identical-field objective value parity failed")
    if not torch.allclose(grad_a, grad_b, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Identical-field objective gradient parity failed")
    return {"passed": True, "alpha": objective.alpha, "max_gradient_difference": float((grad_a-grad_b).abs().max()),
            "total_data_reg": [float(x) for x in actual]}



def worker(args):
    manifest = c.read_json(args.manifest)
    assert_manifest(manifest)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID 1009:1020")
    config = CONFIGS[args.config]
    protocol = cell_protocol(args.alpha)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial_gpu = legacy.gpu_snapshot()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    for key in ("oasis_root", "pair_file", "dl_code_root", "vfa_root"):
        setattr(args, key, Path(manifest["settings"][key]))
    batch, input_hashes = legacy.load_batch(args, device)
    case = manifest["cases"][str(args.case_rank)]
    if input_hashes != case["input_hashes"] or sha(args.pair_file) != manifest["pair_file_sha256"]:
        raise RuntimeError("OASIS inputs or split changed")
    if (batch["fixed_id"], batch["moving_id"]) != (case["fixed_id"], case["moving_id"]):
        raise RuntimeError("Wrong shuffled pair identity")
    if not torch.equal(batch["fixed_mask"].bool(), batch["fixed"] > 0):
        raise RuntimeError("Fixed foreground mask mismatch")
    if args.smoke:
        native_parity = legacy.native_parity(device, args.dl_code_root, args.vfa_root)
    else:
        native_parity = None

    row = {
        "variant": VARIANT, "method": config["method"], "config_key": args.config,
        "protocol": protocol, "run_name": run_name(args.case_rank, args.alpha, args.smoke),
        "smoke": args.smoke, "depth": args.depth, "cps_schedule": list(SCHEDULES[args.depth]),
        "alpha": args.alpha, "stage_alpha": [args.alpha] * args.depth, "bounded": config["bounded"],
        "maps_per_level": config["maps_per_level"], "case_rank": args.case_rank,
        "fixed_id": case["fixed_id"], "moving_id": case["moving_id"],
        "direction": "moving_to_fixed", "sampling": "fixed_to_moving_pull",
        "gpu": args.gpu, "visible_cuda_device": "cuda:0", "source_hashes": manifest["source_hashes"],
        "source_revision": manifest["source_revision"], "sitreg_revision": manifest["sitreg_revision"],
        "input_hashes": input_hashes, "pair_file_sha256": manifest["pair_file_sha256"],
        "initial_gpu_state": initial_gpu, "cuda_backend": c.cuda_backend_metadata(device),
        "tf32_matmul": False, "tf32_cudnn": False, "uid": os.getuid(), "gid": os.getgid(),
        "seed": 1, "container_image": manifest["container_image"], "container_image_id": manifest["container_image_id"],
        "representation": "retained_composition_of_independent_cubic_bspline_generated_maps",
        "bound_multiplier": 0.99 if config["bounded"] else None,
        "initialization": "fresh_identity_per_case_and_each_increment", "segmentation_loss": False,
        "reference_field_supervision": False, "analytic_regularizer": False,
        "cuda_nondeterminism_caveat": "grid_sample backward may remain nondeterministic",
        "native_loss_parity": native_parity, "started_at": c.now_text(), "argv": sys.argv,
    }
    atomic_json(args.out_dir / "manifest.json", row)

    identity = c.make_coordinate_grid(c.IMAGE_SHAPE, device)
    scale = d.voxel_scale(c.IMAGE_SHAPE, device=device)
    previous_field = torch.zeros_like(identity)
    atomics, blocks = [], []
    schedule = b.stages(args.smoke)[:args.depth]
    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    global_step = 0
    for level, stage in enumerate(schedule):
        for map_index in range(config["maps_per_level"]):
            step = global_step
            frozen_ids = list(range(step))
            frozen = [atomics[i] for i in frozen_ids]
            objective = b.Objective(batch["fixed"], batch["moving"], stage, args.alpha,
                                    frozen, manifest["bounds"][str(stage.cps)], config["bounded"])
            parity = verify_objective_parity(objective) if args.smoke else None
            latent0 = torch.zeros(*objective.control_shape, 3, device=device)
            with torch.no_grad():
                init_atomic = objective.atomic(latent0)
                init_field = b.compose([*frozen, init_atomic], identity) - identity
                init_check = legacy.verify_init(previous_field, init_field)
                initial_values = [float(x) for x in objective(latent0)]
                if bool(latent0.count_nonzero()) or bool(init_atomic.count_nonzero()):
                    raise RuntimeError("Identity increment did not initialize to zero")
                if args.smoke:
                    from dataclasses import replace
                    q = objective.field(latent0)
                    reference = d.StageObjective(objective.fixed, objective.moving, args.alpha,
                                                 replace(stage, cps=None, factor=1))
                    if initial_values != [float(x) for x in reference(q)]:
                        raise RuntimeError("Identical-field NCC9/diffusion parity failed")
                    del q, reference
            del latent0, init_atomic, init_field
            block_dir = args.out_dir / f"level{level}_cps{stage.cps}" / f"map{map_index}"
            atomic_json(args.out_dir / "progress.json", {"phase": "optimization", "level": level,
                        "cps": stage.cps, "map": map_index, "step": step, "updated_at": c.now_text()})
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**30)
            best, info = b.optimize(objective, block_dir)
            checkpoint = torch.load(block_dir / "best.pt", map_location=device, weights_only=False)
            with torch.no_grad():
                if not torch.equal(best, checkpoint["latent_norm_xyz"]) or checkpoint["summary"] != info:
                    raise RuntimeError("Selected checkpoint mismatch")
                controls = objective.controls(checkpoint["latent_norm_xyz"])
                if not torch.equal(controls, checkpoint["controls_norm_xyz"]):
                    raise RuntimeError("Control checkpoint reconstruction mismatch")
                atomic = objective.expand(controls).detach()
                current = b.compose([*frozen, atomic], identity) - identity
                values = [float(x) for x in objective(checkpoint["latent_norm_xyz"])]
                for value, key in zip(values, ("best_loss", "best_data_loss", "best_reg_loss")):
                    if not math.isclose(value, info[key], rel_tol=1e-6, abs_tol=1e-7):
                        raise RuntimeError("Best objective does not reconstruct")
            if not math.isclose(info["initial_loss"], initial_values[0], rel_tol=1e-6, abs_tol=1e-7) or info["best_loss"] > initial_values[0] + 1e-6:
                raise RuntimeError("Selected checkpoint is worse than initialization")
            corner = None
            if config["bounded"]:
                corner = b.corner_diagnostics(atomic * scale)
                if not corner["bound_below_one"] or corner["nonpositive_corner_count"]:
                    raise RuntimeError("Bounded elementary-map contraction check failed")
                if info["max_abs_coefficient_ratio"] > 1.0 + 1e-6:
                    raise RuntimeError("Bounded coefficient exceeded its cap")
            field_path = block_dir / f"disp_norm_{case['fixed_id']}_{case['moving_id']}.npy"
            np.save(field_path, current.cpu().numpy())
            restored = torch.from_numpy(np.load(field_path)).to(device)
            if not torch.equal(restored, current) or restored.shape != (*c.IMAGE_SHAPE, 3) or not bool(torch.isfinite(restored).all()):
                raise RuntimeError("Map field serialization/geometry failure")
            block = {
                "step_index": step, "level_index": level, "map_index": map_index,
                "cps": stage.cps, "alpha": args.alpha, "new_map": True,
                "bounded": config["bounded"], "frozen_block_indices": frozen_ids,
                "retained_block_indices": list(range(step + 1)), "optimizer": info,
                "objective_parity": parity, "initialization_check": init_check, "initial_loss_data_reg": initial_values,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "objective_improvement": initial_values[0] - info["best_loss"],
                "atomic_corner_audit": corner, "model_path": str(block_dir / "best.pt"),
                "model_sha256": sha(block_dir / "best.pt"), "dvf_path": str(field_path),
                "dvf_sha256": sha(field_path),
            }
            atomic_json(block_dir / "validated.json", block)
            blocks.append(block)
            atomics.append(atomic)
            previous_field = current.detach()
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**30)
            global_step += 1
            print(json.dumps({"block_complete": step, "rank": args.case_rank, "depth": args.depth,
                              "config": args.config, "cps": stage.cps, "map": map_index,
                              "actual_epochs": info["actual_epochs"], "elapsed_sec": info["training_elapsed_sec"]}), flush=True)
            del restored, checkpoint, controls, objective, best

    atomic_json(args.out_dir / "progress.json", {"phase": "metrics_and_topology", "updated_at": c.now_text()})
    final_dir = args.out_dir / "final_metrics"
    final_dir.mkdir()
    metrics = metric_row(previous_field, batch, final_dir)
    offgrid = b.offgrid_diagnostics(atomics, previous_field, batch["fixed_mask"], count=100000, seed=1)
    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**30)
    updates = sum(int(x["optimizer"]["actual_epochs"]) for x in blocks)
    row.update(**metrics, blocks=blocks, final_map_count=len(atomics), final_retained_block_indices=list(range(len(atomics))),
               offgrid_audit=offgrid, peak_cuda_gib=peak, peak_reserved_gib=peak_reserved, dvf_path=blocks[-1]["dvf_path"],
               dvf_sha256=blocks[-1]["dvf_sha256"], actual_adam_updates=updates,
               training_elapsed_sec=sum(x["optimizer"]["training_elapsed_sec"] for x in blocks),
               total_elapsed_sec=time.monotonic() - started, run_dir=str(args.out_dir),
               completed_at=c.now_text())
    if not all(math.isfinite(float(row[k])) for k in METRICS_TO_REPORT):
        raise FloatingPointError("Nonfinite final accuracy/regularity metric")
    assert_manifest(manifest)
    atomic_json(args.out_dir / "summary.json", row)
    (args.out_dir / "DONE").write_text(c.now_text() + "\n")


def validate(run_dir: Path, manifest: dict, rank: int, alpha: float, smoke=False):
    depth, key = 3, CONFIG
    assert_manifest(manifest)
    if not (run_dir / "DONE").is_file():
        raise RuntimeError("Missing DONE marker")
    row = c.read_json(run_dir / "summary.json")
    protocol = cell_protocol(alpha)
    case = manifest["cases"][str(rank)]
    config = CONFIGS[key]
    expected = {"variant": VARIANT, "method": config["method"], "config_key": key,
        "protocol": protocol, "run_name": run_name(rank, alpha, smoke), "smoke": smoke,
        "depth": depth, "cps_schedule": list(SCHEDULES[depth]), "alpha": alpha, "stage_alpha": [alpha] * depth,
        "bounded": config["bounded"], "maps_per_level": config["maps_per_level"],
        "case_rank": rank, "fixed_id": case["fixed_id"], "moving_id": case["moving_id"],
        "direction": "moving_to_fixed", "input_hashes": case["input_hashes"],
        "source_hashes": manifest["source_hashes"], "pair_file_sha256": manifest["pair_file_sha256"],
        "uid": 1009, "gid": 1020, "seed": 1,
        "final_map_count": depth * config["maps_per_level"],
        "final_retained_block_indices": list(range(depth * config["maps_per_level"]))}
    if any(row.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Run identity/protocol mismatch")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("CUDA determinism flags mismatch")
    stages = b.stages(smoke)[:depth]
    expected_blocks = depth * config["maps_per_level"]
    if len(row["blocks"]) != expected_blocks:
        raise RuntimeError("Wrong number of map checkpoints")
    for step, block in enumerate(row["blocks"]):
        level, map_index = divmod(step, config["maps_per_level"])
        info = block["optimizer"]
        stage = stages[level]
        if (block["level_index"], block["map_index"], block["step_index"], block["cps"]) != (level, map_index, step, stage.cps):
            raise RuntimeError("Map/stage ordering mismatch")
        if info["stage"] != __import__("dataclasses").asdict(stage) or info["alpha"] != alpha or block["alpha"] != alpha:
            raise RuntimeError("Stage schedule or lambda mismatch")
        if info["bounded"] != config["bounded"] or block["bounded"] != config["bounded"]:
            raise RuntimeError("Bounded/unbounded representation mismatch")
        if block["frozen_block_indices"] != list(range(step)) or block["retained_block_indices"] != list(range(step + 1)):
            raise RuntimeError("Composition chain mismatch")
        if not info["initial_latent_max_abs"] == 0.0:
            raise RuntimeError("Map did not start from identity")
        if not 0 <= info["best_epoch"] <= info["actual_epochs"] <= stage.max_epochs or info["best_loss"] > info["initial_loss"] + 1e-6:
            raise RuntimeError("Invalid stopping/checkpoint epoch")
        if config["bounded"]:
            corner = block["atomic_corner_audit"]
            if not corner or not corner["bound_below_one"] or corner["nonpositive_corner_count"]:
                raise RuntimeError("Bounded elementary map failed contraction audit")
            if info["max_abs_coefficient_ratio"] > 1.0 + 1e-6:
                raise RuntimeError("Bounded control exceeded cap")
        elif block["atomic_corner_audit"] is not None:
            raise RuntimeError("Unbounded map was misreported as bounded-audited")
        for kind in ("model", "dvf"):
            if sha(Path(block[kind + "_path"])) != block[kind + "_sha256"]:
                raise RuntimeError("Checkpoint/field hash mismatch")
        field = np.load(block["dvf_path"], mmap_mode="r")
        if field.dtype != np.float32 or field.shape != tuple(PROTOCOL["field_shape"]) or not np.isfinite(field).all():
            raise RuntimeError("Invalid saved deformation geometry/values")
        with (Path(block["model_path"]).parent / "history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda x: float(x["loss"]))
        if int(best["epoch"]) != info["best_epoch"] or not math.isclose(float(best["loss"]), info["best_loss"], rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("History/best checkpoint mismatch")
    if not all(math.isfinite(float(row[k])) for k in METRICS_TO_REPORT):
        raise RuntimeError("Nonfinite final metric")
    if row["offgrid_audit"]["count"] != 100000:
        raise RuntimeError("Missing off-grid audit")
    if smoke and not row["native_loss_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    return row


def audit_ok(row):
    audit = c.read_json(Path(row["run_dir"]) / "independent_audit.json")
    if not audit["passed"] or audit["summary_sha256"] != sha(Path(row["run_dir"]) / "summary.json"):
        raise RuntimeError("Independent audit missing or stale")
    if len(audit["checks"]) != len(row["blocks"]) or not audit["offgrid_reproduced"]:
        raise RuntimeError("Incomplete independent map replay")
    if (audit["case_rank"], audit["depth"], audit["config_key"], audit["gpu"]) != (
            row["case_rank"], row["depth"], row["config_key"], row["gpu"]):
        raise RuntimeError("Independent audit identity mismatch")
    if audit["alpha"] != row["alpha"]:
        raise RuntimeError("Audit lambda mismatch")
    return audit



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    work = sub.add_parser("worker")
    work.add_argument("--manifest", type=Path, required=True)
    work.add_argument("--out-dir", type=Path, required=True)
    work.add_argument("--case-rank", type=int, choices=range(10), required=True)
    work.add_argument("--alpha", type=alpha_value, required=True)
    work.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    work.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "worker":
        args.depth, args.config = 3, CONFIG
    output = args.artifact_root if args.command == "prepare" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        parser.error("Generated artifacts must be external")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        if (os.getuid(), os.getgid()) != (1009, 1020):
            raise RuntimeError("Require UID:GID1009:1020")
        if args.command == "prepare":
            print(json.dumps({"prepared": prepare(args.artifact_root)["variant"]}), flush=True)
        else:
            worker(args)
    except Exception:
        if args.command == "worker" and args.out_dir.is_dir():
            atomic_json(args.out_dir / f"FAILED_worker_{time.time_ns()}.json",
                        {"error": traceback.format_exc(), "at": c.now_text()})
        raise

