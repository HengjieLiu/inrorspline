#!/usr/bin/env python
"""P70: first-ten factorial comparison of B-spline pyramid depth and bounds."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import bounded_two_blocks as b
import run_oasis_two_increments_first10 as p69

c, d, legacy = p69.c, b.d, p69.legacy
sha, atomic_json, METRICS = p69.sha, p69.atomic_json, p69.METRICS
metric_row = p69.pilot.previous.metric_row
VARIANT = "oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3"
PARENT_VARIANT = p69.VARIANT
SCHEDULES = {
    1: (8,), 2: (8, 4), 3: (8, 4, 2), 4: (8, 4, 2, 1),
}
CONFIGS = {
    "unbounded_1": {"bounded": False, "maps_per_level": 1,
                    "method": "direct_cp_multistage_bspline_unbounded_one_map"},
    "unbounded_2_compose": {"bounded": False, "maps_per_level": 2,
                             "method": "direct_cp_multistage_bspline_unbounded_two_map_composition"},
    "bounded_1": {"bounded": True, "maps_per_level": 1,
                  "method": "direct_cp_multistage_bspline_tanh_bounded_one_map"},
    "bounded_2_compose": {"bounded": True, "maps_per_level": 2,
                           "method": "direct_cp_multistage_bspline_tanh_bounded_two_map_composition"},
}
CONFIG_ORDER = tuple(CONFIGS)
ACCURACY = ("dice", "hd95", "asd")
REGULARITY = ("log_jdet_std_fg", "jdet_le0_pct_fg")
METRICS_TO_REPORT = ACCURACY + REGULARITY
STAGE_TABLE = [
    {"cps": 8, "downsample": 4, "lambda": 1.0, "normalized_lr": 0.03, "max_epochs": 400, "min_epochs": 40, "patience": 20},
    {"cps": 4, "downsample": 2, "lambda": 1.0, "normalized_lr": 0.01, "max_epochs": 800, "min_epochs": 80, "patience": 35},
    {"cps": 2, "downsample": 1, "lambda": 1.0, "normalized_lr": 0.003, "max_epochs": 1300, "min_epochs": 130, "patience": 50},
    {"cps": 1, "downsample": 1, "lambda": 1.0, "normalized_lr": 0.001, "max_epochs": 2000, "min_epochs": 300, "patience": 100,
     "max_lr_reductions": 2, "min_epochs_between_reductions": 100},
]
PROTOCOL = {
    "version": 3, "stage_alpha": "1 at every active cps level", "stage_schedule": STAGE_TABLE,
    "configs": CONFIGS, "depths": {str(k): list(v) for k, v in SCHEDULES.items()},
    "bound_multiplier": 0.99, "bounded_parameterization": "cap*tanh(latent/cap), unit derivative at zero",
    "unbounded_parameterization": "controls equal latent; no clipping or tanh",
    "two_map_initialization": "each independent B-spline increment starts at identity and composes with frozen earlier maps",
    "two_map_budget": "full stage optimizer schedule for each map; fresh Adam state per map",
    "loss": "raw squared local NCC9 (1-mean(correlation^2)) plus lambda-weighted unmasked total-field voxel-unit diffusion",
    "sampling": "existing raw-intensity preprocessing and border sampling; pull mapping Phi_previous o delta_current",
    "regularizer": "forward-valid component-mean, axis-mean L2 diffusion on total composed displacement in current stage voxel units",
    "checkpoint": "lowest total loss including initialization; never selected by segmentation metrics",
    "stopping": {"check_interval": 25, "median_window": 25, "relative_total_threshold": 1e-4,
                 "relative_data_threshold": 5e-5},
    "seed": 1, "precision": "FP32", "cudnn_benchmark": False,
    "cudnn_deterministic": True, "tf32": False,
    "field_shape": [160, 224, 192, 3], "direction": "moving_to_fixed",
    "independent_fresh_fit_per_case_depth_config": True,
}


def cell_protocol(depth: int, config_key: str) -> dict:
    if depth not in SCHEDULES or config_key not in CONFIGS:
        raise ValueError("Invalid factorial cell")
    return {**PROTOCOL, "depth": depth, "cps_schedule": list(SCHEDULES[depth]),
            "stage_alpha": [1.0] * depth, "config_key": config_key,
            "method": CONFIGS[config_key]["method"]}


def run_name(rank: int, depth: int, config_key: str, smoke=False) -> str:
    if type(rank) is not int or rank not in range(10):
        raise ValueError("Require shuffled rank0..9")
    spec = cell_protocol(depth, config_key)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "") + f"rank{rank:03d}_depth{depth}_{config_key}_{digest}"


def sources(sitreg_root: Path) -> dict:
    return {
        "runner": sha(Path(__file__)),
        "audit": sha(Path(__file__).with_name("audit_oasis_bspline_depth_factorial_first10.py")),
        "numerical_core": sha(Path(b.__file__)),
        "bounded_composition_core": sha(Path(b.original.__file__)),
        "p69_adapter": sha(Path(p69.__file__)),
        "p69_dependency_sources": p69.sources(Path(sitreg_root)),
    }


def parent_paths(root: Path):
    run_root = root / "runs" / PARENT_VARIANT
    report_root = root / "repo_experiments" / PARENT_VARIANT
    return run_root, report_root


def parent_source_check(manifest: dict):
    parent_manifest_path = Path(manifest["parent_manifest_path"])
    parent = c.read_json(parent_manifest_path)
    if p69.sources(Path(parent["sitreg_root"])) != parent["source_hashes"]:
        raise RuntimeError("P69 source drift")
    for path, digest in manifest["parent_reference_hashes"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Read-only P69 reference drift: {path}")


def assert_manifest(manifest: dict):
    if sources(Path(manifest["sitreg_root"])) != manifest["source_hashes"]:
        raise RuntimeError("P70 source drift")
    parent_source_check(manifest)


def verify_parent(root: Path):
    run_root, report_root = parent_paths(root)
    manifest_path = run_root / "launch_manifest.json"
    parent = c.read_json(manifest_path)
    p69.assert_manifest(parent)
    accepted = c.read_json(report_root / "ACCEPTED.json")
    for name, digest in accepted["output_sha256"].items():
        if sha(report_root / name) != digest:
            raise RuntimeError(f"P69 accepted report hash mismatch: {name}")
    results = c.read_json(report_root / "results.json")
    p69.require_cohort(results["candidates"])
    if len(parent["references"]) != 20:
        raise RuntimeError("Expected VFA/SITReg saved references for all ten pairs")
    for row in results["candidates"]:
        run_dir = Path(row["run_dir"])
        if c.read_json(run_dir / "summary.json") != row or not (run_dir / "DONE").is_file():
            raise RuntimeError("P69 candidate summary or completion marker changed")
        p69.check_audit(row)
        if sha(Path(row["dvf_path"])) != row["dvf_sha256"]:
            raise RuntimeError("P69 comparison DVF changed")
    hashes = {
        str(manifest_path): sha(manifest_path),
        str(report_root / "ACCEPTED.json"): sha(report_root / "ACCEPTED.json"),
        str(report_root / "results.json"): sha(report_root / "results.json"),
    }
    for row in results["candidates"]:
        for name in ("summary.json", "independent_audit.json", "DONE"):
            path = Path(row["run_dir"]) / name
            hashes[str(path)] = sha(path)
    return parent, results, manifest_path, hashes


def make_manifest(root: Path, gpus: list[str]):
    parent, results, parent_manifest_path, hashes = verify_parent(root)
    refs = parent["references"]
    for method in ("VFA (original)", "SITReg λ=1 (original)"):
        if len([r for r in refs if r["method"] == method]) != 10:
            raise RuntimeError(f"Missing ten saved {method} fields")
    # Case grouping keeps all 16 settings for a pair on one physical GPU.
    placement = {str(rank): gpus[rank % len(gpus)] for rank in range(10)}
    return {
        "variant": VARIANT, "protocol": PROTOCOL, "parent_variant": PARENT_VARIANT,
        "parent_manifest_path": str(parent_manifest_path), "parent_reference_hashes": hashes,
        "parent_accepted_report_sha256": sha(parent_paths(root)[1] / "ACCEPTED.json"),
        "parent_candidates_for_replication_check_only": results["candidates"],
        "cases": parent["cases"], "settings": parent["settings"], "bounds": parent["bounds"],
        "pair_file_sha256": parent["pair_file_sha256"],
        "references": refs, "gpus": gpus, "case_to_gpu": placement,
        "initial_gpu_state": legacy.gpu_snapshot(), "created_at": c.now_text(),
        "source_hashes": sources(Path(parent["sitreg_root"])),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
        "sitreg_root": parent["sitreg_root"], "sitreg_revision": parent["sitreg_revision"],
        "container_image": parent["container_image"],
        "container_image_id": parent["container_image_id"],
        "uid": os.getuid(), "gid": os.getgid(), "seed": 1,
        "production_fit_count": 160, "smoke_fit_count": 4,
        "prior_runs_reused_as_factorial_cells": 0,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    launch.add_argument("--gpus", default="0,1,2,3", choices=("0,1,2,3",))
    launch.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--out-dir", type=Path, required=True)
    worker.add_argument("--case-rank", type=int, choices=range(10), required=True)
    worker.add_argument("--depth", type=int, choices=(1, 2, 3, 4), required=True)
    worker.add_argument("--config", choices=CONFIG_ORDER, required=True)
    worker.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    worker.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "launch":
        args.gpus = args.gpus.split(",")
        if args.gpus != ["0", "1", "2", "3"]:
            parser.error("This approved run requires physical GPUs0,1,2,3 exactly once")
    output = args.artifact_root if args.command == "launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        parser.error("Experiment artifacts must be outside the repository")
    return args


def metric_record(row):
    return {key: row[key] for key in METRICS_TO_REPORT}


def worker(args):
    manifest = c.read_json(args.manifest)
    assert_manifest(manifest)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID 1009:1020")
    config = CONFIGS[args.config]
    protocol = cell_protocol(args.depth, args.config)
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
        "protocol": protocol, "run_name": run_name(args.case_rank, args.depth, args.config, args.smoke),
        "smoke": args.smoke, "depth": args.depth, "cps_schedule": list(SCHEDULES[args.depth]),
        "stage_alpha": [1.0] * args.depth, "bounded": config["bounded"],
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
    peak = 0.0
    global_step = 0
    for level, stage in enumerate(schedule):
        for map_index in range(config["maps_per_level"]):
            step = global_step
            frozen_ids = list(range(step))
            frozen = [atomics[i] for i in frozen_ids]
            objective = b.Objective(batch["fixed"], batch["moving"], stage, 1.0,
                                    frozen, manifest["bounds"][str(stage.cps)], config["bounded"])
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
                    reference = d.StageObjective(objective.fixed, objective.moving, 1.0,
                                                 replace(stage, cps=None, factor=1))
                    if initial_values != [float(x) for x in reference(q)]:
                        raise RuntimeError("Identical-field NCC9/diffusion parity failed")
                    del q, reference
            del latent0, init_atomic, init_field
            block_dir = args.out_dir / f"level{level}_cps{stage.cps}" / f"map{map_index}"
            atomic_json(args.out_dir / "progress.json", {"phase": "optimization", "level": level,
                        "cps": stage.cps, "map": map_index, "step": step, "updated_at": c.now_text()})
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
                "cps": stage.cps, "alpha": 1.0, "new_map": True,
                "bounded": config["bounded"], "frozen_block_indices": frozen_ids,
                "retained_block_indices": list(range(step + 1)), "optimizer": info,
                "initialization_check": init_check, "initial_loss_data_reg": initial_values,
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
            global_step += 1
            print(json.dumps({"block_complete": step, "rank": args.case_rank, "depth": args.depth,
                              "config": args.config, "cps": stage.cps, "map": map_index,
                              "actual_epochs": info["actual_epochs"], "elapsed_sec": info["training_elapsed_sec"]}), flush=True)
            del restored, checkpoint, controls, objective, best

    final_dir = args.out_dir / "final_metrics"
    final_dir.mkdir()
    metrics = metric_row(previous_field, batch, final_dir)
    offgrid = b.offgrid_diagnostics(atomics, previous_field, batch["fixed_mask"], count=100000, seed=1)
    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    updates = sum(int(x["optimizer"]["actual_epochs"]) for x in blocks)
    row.update(**metrics, blocks=blocks, final_map_count=len(atomics), final_retained_block_indices=list(range(len(atomics))),
               offgrid_audit=offgrid, peak_cuda_gib=peak, dvf_path=blocks[-1]["dvf_path"],
               dvf_sha256=blocks[-1]["dvf_sha256"], actual_adam_updates=updates,
               training_elapsed_sec=sum(x["optimizer"]["training_elapsed_sec"] for x in blocks),
               total_elapsed_sec=time.monotonic() - started, run_dir=str(args.out_dir),
               completed_at=c.now_text())
    if not all(math.isfinite(float(row[k])) for k in METRICS_TO_REPORT):
        raise FloatingPointError("Nonfinite final accuracy/regularity metric")
    assert_manifest(manifest)
    atomic_json(args.out_dir / "summary.json", row)
    (args.out_dir / "DONE").write_text(c.now_text() + "\n")


def validate(run_dir: Path, manifest: dict, rank: int, depth: int, key: str, smoke=False):
    assert_manifest(manifest)
    if not (run_dir / "DONE").is_file():
        raise RuntimeError("Missing DONE marker")
    row = c.read_json(run_dir / "summary.json")
    protocol = cell_protocol(depth, key)
    case = manifest["cases"][str(rank)]
    config = CONFIGS[key]
    expected = {"variant": VARIANT, "method": config["method"], "config_key": key,
        "protocol": protocol, "run_name": run_name(rank, depth, key, smoke), "smoke": smoke,
        "depth": depth, "cps_schedule": list(SCHEDULES[depth]), "stage_alpha": [1.0] * depth,
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
        if info["stage"] != __import__("dataclasses").asdict(stage) or info["alpha"] != 1.0 or block["alpha"] != 1.0:
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
    return audit


def start_process(run_root, rank, depth, key, gpu, required_mib, smoke=False, audit=False):
    state = None
    last_snapshot = None
    # nvidia-smi can briefly return an incomplete row set while a newly
    # admitted CUDA process is initializing. Recheck live state before fail.
    for attempt in range(3):
        last_snapshot = legacy.gpu_snapshot()
        state = next((r for r in last_snapshot.get("rows", [])
                      if str(r.get("physical_gpu")) == str(gpu)), None)
        if state is not None:
            break
        if attempt < 2:
            time.sleep(1)
    if state is None:
        raise RuntimeError(f"GPU{gpu} missing from current GPU snapshot: {last_snapshot}")
    if state["memory_free_mib"] < required_mib:
        return None
    path = run_root / run_name(rank, depth, key, smoke)
    if audit:
        command = [sys.executable, "-u", "-B", str(Path(__file__).with_name("audit_oasis_bspline_depth_factorial_first10.py")),
                   "--manifest", str(run_root / "launch_manifest.json"), "--run-dir", str(path), "--gpu", gpu]
    else:
        if path.exists():
            raise RuntimeError(f"Preserve incomplete attempt for audit: {path}")
        command = [sys.executable, "-u", "-B", str(Path(__file__)), "worker", "--manifest", str(run_root / "launch_manifest.json"),
                   "--out-dir", str(path), "--case-rank", str(rank), "--depth", str(depth), "--config", key, "--gpu", gpu]
        if smoke:
            command.append("--smoke")
    log_path = run_root / (path.name + ("_audit" if audit else "") + ".log")
    log = log_path.open("x")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    except BaseException:
        log.close()
        raise
    print(json.dumps({"launched": path.name, "audit": audit, "gpu": gpu, "pid": process.pid,
                      "gpu_before": state}), flush=True)
    return {"process": process, "log": log, "run_dir": path, "rank": rank,
            "depth": depth, "config": key, "gpu": gpu}


def config_rows():
    return [(depth, key) for depth in range(1, 5) for key in CONFIG_ORDER]


def grouped_summary(rows, refs):
    summaries, table_rows = [], {}
    for depth in range(1, 5):
        rows_depth = [r for r in rows if r["depth"] == depth]
        if len(rows_depth) != 40:
            raise RuntimeError(f"Depth {depth} requires 40 registrations")
        groups = [(key, CONFIGS[key]["method"], [r for r in rows_depth if r["config_key"] == key]) for key in CONFIG_ORDER]
        refs_depth = [{**r, "case_rank": r["rank"]} for r in refs]
        groups += [(None, "VFA (original)", [r for r in refs_depth if r["method"] == "VFA (original)"]),
                   (None, "SITReg λ=1 (original)", [r for r in refs_depth if r["method"] == "SITReg λ=1 (original)"])]
        table_rows[depth] = []
        for key, method, group in groups:
            if len(group) != 10 or {r["case_rank"] for r in group} != set(range(10)):
                raise RuntimeError(f"Depth{depth}/{method} missing cases")
            rec = {"depth": depth, "schedule": "→".join(map(str, SCHEDULES[depth])),
                   "config_key": key or "reference", "method": method, "n": 10}
            for metric in METRICS_TO_REPORT:
                vals = [float(r[metric]) for r in group]
                rec[metric + "_mean"] = statistics.mean(vals)
                rec[metric + "_sd"] = statistics.stdev(vals)
            for key_time in ("training_elapsed_sec", "total_elapsed_sec", "actual_adam_updates"):
                if all(key_time in r for r in group):
                    vals = [float(r[key_time]) for r in group]
                    rec[key_time + "_mean"] = statistics.mean(vals)
                    rec[key_time + "_sd"] = statistics.stdev(vals)
            summaries.append(rec); table_rows[depth].append(rec)
    return summaries, table_rows


def write_report(rows, manifest, out):
    if len(rows) != 160 or len({(r["case_rank"], r["depth"], r["config_key"]) for r in rows}) != 160:
        raise RuntimeError("Require all 160 independent factorial registrations")
    if any(not (Path(r["run_dir"]) / "DONE").is_file() for r in rows):
        raise RuntimeError("Missing production completion marker")
    for row in rows:
        audit_ok(row)
    refs = manifest["references"]
    summaries, table_rows = grouped_summary(rows, refs)
    per_case, paired, stage_rows = [], [], []
    for row in rows:
        case = manifest["cases"][str(row["case_rank"])]
        per_case.append({"case_rank": row["case_rank"], "fixed_id": case["fixed_id"], "moving_id": case["moving_id"],
                         "depth": row["depth"], "schedule": "→".join(map(str, row["cps_schedule"])),
                         "config_key": row["config_key"], "method": row["method"],
                         **metric_record(row), "optimizer_sec": row["training_elapsed_sec"],
                         "worker_sec": row["total_elapsed_sec"], "actual_adam_updates": row["actual_adam_updates"],
                         "gpu": row["gpu"], "run_dir": row["run_dir"]})
        for ref_method in ("VFA (original)", "SITReg λ=1 (original)"):
            ref = next(r for r in refs if r["rank"] == row["case_rank"] and r["method"] == ref_method)
            # Keep a self-contained per-case table: references appear once for
            # each depth table, with optimizer diagnostics explicitly n/a.
            per_case.append({"case_rank": row["case_rank"], "fixed_id": case["fixed_id"],
                "moving_id": case["moving_id"], "depth": row["depth"],
                "schedule": "→".join(map(str, SCHEDULES[row["depth"]])),
                "config_key": "reference", "method": ref_method,
                **metric_record(ref), "optimizer_sec": "", "worker_sec": "",
                "actual_adam_updates": "", "gpu": "n/a", "run_dir": ref.get("dvf_path", "")})
            rec = {"case_rank": row["case_rank"], "depth": row["depth"], "config_key": row["config_key"],
                   "method": row["method"], "reference": ref_method}
            for metric in METRICS_TO_REPORT:
                rec["delta_" + metric] = row[metric] - ref[metric]
            paired.append(rec)
        for block in row["blocks"]:
            info = block["optimizer"]
            stage_rows.append({"case_rank": row["case_rank"], "depth": row["depth"],
                "config_key": row["config_key"], "step": block["step_index"], "cps": block["cps"],
                "map_index": block["map_index"], "bounded": row["bounded"], "alpha": 1.0,
                "best_epoch": info["best_epoch"], "actual_adam_updates": info["actual_epochs"],
                "stopping_reason": info["stopping_reason"], "best_loss": info["best_loss"],
                "best_data_loss": info["best_data_loss"], "best_reg_loss": info["best_reg_loss"],
                "optimizer_sec": info["training_elapsed_sec"], "gpu": row["gpu"]})
    # Compare all shared prefixes across the four independently restarted depths
    # for each configuration and case.
    prefix_checks = []
    for rank in range(10):
        for config_key in CONFIG_ORDER:
            group = {r["depth"]: r for r in rows if r["case_rank"] == rank and r["config_key"] == config_key}
            max_depth = 4
            for depth in range(1, max_depth):
                shallow = group[depth]
                deeper = group[depth + 1]
                blocks_per_level = CONFIGS[config_key]["maps_per_level"]
                for step in range(depth * blocks_per_level):
                    a, z = shallow["blocks"][step], deeper["blocks"][step]
                    delta = (np.load(a["dvf_path"], mmap_mode="r") - np.load(z["dvf_path"], mmap_mode="r")) * np.asarray([95.5,111.5,79.5],dtype=np.float32)
                    maximum = float(np.abs(delta).max())
                    rms = float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
                    same_epoch = (a["optimizer"]["best_epoch"],a["optimizer"]["actual_epochs"]) == (z["optimizer"]["best_epoch"],z["optimizer"]["actual_epochs"])
                    same_loss = all(abs(a["optimizer"][k]-z["optimizer"][k]) <= 1e-6 for k in ("best_loss","best_data_loss","best_reg_loss"))
                    if maximum > 1e-4 or rms > 1e-5 or not same_epoch or not same_loss:
                        raise RuntimeError(f"Shared prefix mismatch: rank{rank}/{config_key}/depth{depth}/step{step}")
                    prefix_checks.append({"case_rank":rank,"config_key":config_key,"shallow_depth":depth,
                        "deeper_depth":depth+1,"step":step,"max_abs_voxel":maximum,"rms_voxel":rms,
                        "same_epoch":same_epoch,"loss_within_1e-6":same_loss})

    # Fresh reproduction check: this matching full-depth cell was rerun, never reused.
    historical = [r for r in manifest["parent_candidates_for_replication_check_only"]
                  if tuple(r["stage_alpha"]) == (1.,1.,1.,1.)]
    reproduction = []
    fresh_full = [r for r in rows if r["depth"] == 4 and r["config_key"] == "bounded_2_compose"]
    for row in fresh_full:
        old = next(x for x in historical if x["case_rank"] == row["case_rank"])
        diffs = {k: row[k] - old[k] for k in METRICS_TO_REPORT}
        reproduction.append({"case_rank": row["case_rank"], **diffs,
            "new_optimizer_sec": row["training_elapsed_sec"], "prior_optimizer_sec": old["training_elapsed_sec"],
            "new_worker_sec": row["total_elapsed_sec"], "prior_worker_sec": old["total_elapsed_sec"]})

    out.mkdir(parents=True, exist_ok=False)
    c.write_csv(out / "summary.csv", summaries)
    c.write_csv(out / "per_case.csv", per_case)
    c.write_csv(out / "paired_reference_differences.csv", paired)
    c.write_csv(out / "stage_diagnostics.csv", stage_rows)
    c.write_csv(out / "shared_prefix_checks.csv", prefix_checks)
    c.write_csv(out / "fresh_vs_prior_p69.csv", reproduction)
    atomic_json(out / "results.json", {"complete": True, "candidates": rows, "summary": summaries,
        "per_case": per_case, "paired_reference_differences": paired, "stage_diagnostics": stage_rows,
        "shared_prefix_checks": prefix_checks, "fresh_vs_prior_p69": reproduction, "provenance": manifest})

    lines = ["# First-ten B-spline depth × constraint comparison", "",
        "Fresh 4×4 factorial: four schedule depths × unbounded/bounded one-map and two-map composition.",
        "Each active stage uses diffusion λ=1. Each two-map increment receives the full stage schedule and fresh Adam; those arms use more optimization steps than one-map arms.",
        "Raw squared local NCC9 and unmasked total-field voxel-unit diffusion; all fits start from identity.",
        "Mean ± across-case sample SD (n=10, ddof=1). First-ten results are exploratory; ranks were part of earlier tuning.", ""]
    def fmt(rec, key):
        return f"{rec[key + '_mean']:.4f} ± {rec[key + '_sd']:.4f}" if key + "_mean" in rec else "n/a"
    metric_columns = [("dice", "Dice ↑"), ("hd95", "HD95 mm ↓"), ("asd", "Symmetric ASD mm ↓"),
                      ("log_jdet_std_fg", "FG SD(log J)"), ("jdet_le0_pct_fg", "FG folding %"),
                      ("training_elapsed_sec", "Optimizer sec"), ("total_elapsed_sec", "Worker sec"),
                      ("actual_adam_updates", "Adam updates")]
    for depth in range(1,5):
        lines += [f"## CPS schedule { '→'.join(map(str,SCHEDULES[depth])) }", "",
            "| Setting | " + " | ".join(label for _,label in metric_columns) + " |",
            "|---|" + "---:|"*len(metric_columns)]
        for rec in table_rows[depth]:
            lines.append("| " + rec["method"] + " | " + " | ".join(fmt(rec,k) for k,_ in metric_columns) + " |")
        lines.append("")
    lines += ["## Per-case and paired data", "",
        "`per_case.csv` contains all 160 new case/settings rows plus 20 VFA/SITReg reference rows per schedule. `paired_reference_differences.csv` gives per-case differences from VFA and SITReg.",
        "`stage_diagnostics.csv` records each map's stop/best epoch, actual updates, objective components and optimizer time.", "",
        "## Verification and interpretation", "",
        f"Independent replay passed for all {len(rows)} registrations. Shared-prefix consistency checks: {len(prefix_checks)}.",
        "Bounded elementary maps must pass the validated contraction check. Unbounded configurations have no contraction guarantee; report folding rather than rejecting it.",
        "The full-depth bounded two-map λ=1 cell was rerun fresh and is compared with accepted P69 only as a reproduction diagnostic.",
        "Optimizer time sums per-map Adam durations; worker time also includes input loading, metrics, and serialization but excludes independent audit. GPU contention can affect both."]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if len(summaries) != 24 or len(per_case) != 240 or len(paired) != 160 or len(stage_rows) != 600 or len(reproduction) != 10:
        raise RuntimeError("Final table row-count check failed")
    assert_manifest(manifest)
    atomic_json(out / "ACCEPTED.json", {"complete": True, "production_registrations": 160,
        "independent_audits": 160, "smoke_registrations": 4, "paired_prefix_checks": len(prefix_checks),
        "completed_at": c.now_text(), "output_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})


def write_progress(path, phase, completed, total, started, active, pending, failures, extra=None):
    elapsed = time.monotonic() - started
    value = {"phase": phase, "completed": completed, "total": total, "elapsed_sec": elapsed,
             "eta_sec": None if not completed else elapsed / completed * max(0, total-completed),
             "active": [{"gpu": g, "rank": x["rank"], "depth": x["depth"], "config": x["config"], "pid": x["process"].pid} for g,x in active.items()],
             "pending_by_gpu": {g: len(q) for g,q in pending.items()}, "failures": failures,
             "updated_at": c.now_text()}
    value.update(extra or {})
    atomic_json(path, value)


def run_queue(run_root, manifest, tasks_by_gpu, required_mib, total, audit=False):
    active, failures = {}, []
    completed = 0
    started = time.monotonic()
    task_count = sum(len(q) for q in tasks_by_gpu.values())
    while any(tasks_by_gpu.values()) or active:
        if not failures:
            for gpu, queue in tasks_by_gpu.items():
                if queue and gpu not in active:
                    rank, depth, config, smoke = queue[0]
                    item = start_process(run_root, rank, depth, config, gpu, required_mib, smoke, audit)
                    if item is not None:
                        active[gpu] = item
        for gpu,item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close(); del active[gpu]
            task = tasks_by_gpu[gpu][0]
            rank, depth, config, smoke = task
            try:
                if code:
                    raise RuntimeError(f"Worker exit {code}")
                run_dir = run_root / run_name(rank, depth, config, smoke)
                if audit:
                    check_row = c.read_json(run_dir / "summary.json")
                    audit_ok(check_row)
                else:
                    validate(run_dir, manifest, rank, depth, config, smoke)
                tasks_by_gpu[gpu].pop(0)
                completed += 1
                print(json.dumps({"phase": "audit" if audit else "production", "completed": completed,
                    "total": task_count, "rank": rank, "depth": depth, "config": config, "gpu": gpu}), flush=True)
            except BaseException as exc:
                failures.append({"rank": rank, "depth": depth, "config": config, "gpu": gpu,
                                 "run_dir": str(run_root / run_name(rank, depth, config, smoke)), "error": str(exc)})
        write_progress(run_root / "progress.json", "audit_failed_draining" if audit and failures else
            ("production_failed_draining" if failures else ("independent_audits" if audit else "production")),
            completed, task_count, started, active, tasks_by_gpu, failures)
        if failures and not active:
            raise RuntimeError(f"{('Audit' if audit else 'Production')} failed; preserve outputs and stop queue: {failures}")
        if any(tasks_by_gpu.values()) or active:
            time.sleep(3)
    if failures:
        raise RuntimeError(f"Failures: {failures}")
    return completed


def launch(args):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID 1009:1020")
    run_root = args.artifact_root / "runs" / VARIANT
    report_root = args.artifact_root / "repo_experiments" / VARIANT
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (report_root / "ACCEPTED.json").exists():
            accepted = c.read_json(report_root / "ACCEPTED.json")
            for name,digest in accepted["output_sha256"].items():
                if sha(report_root / name) != digest: raise RuntimeError(f"Accepted report changed: {name}")
            print(f"Already accepted: {report_root}", flush=True); return
        manifest_path = run_root / "launch_manifest.json"
        if manifest_path.exists():
            manifest = c.read_json(manifest_path); assert_manifest(manifest)
            if manifest["gpus"] != args.gpus: raise RuntimeError("GPU set differs from frozen manifest")
        else:
            manifest = make_manifest(args.artifact_root, args.gpus)
            atomic_json(manifest_path, manifest)
        required_mib = 8192

        smoke_tasks = {gpu: [] for gpu in args.gpus}
        for i,key in enumerate(CONFIG_ORDER):
            smoke_tasks[args.gpus[i % len(args.gpus)]].append((1,4,key,True))
        smoke_tasks = {g:q for g,q in smoke_tasks.items()}
        # Four full-native-grid smoke cases, one setting per physical GPU.
        smoke_active, smoke_failures = {}, []
        for gpu, queue in smoke_tasks.items():
            if not queue: continue
            rank,depth,key,smoke = queue[0]
            path = run_root / run_name(rank,depth,key,True)
            if (path / "DONE").is_file():
                result = validate(path,manifest,rank,depth,key,True)
                required_mib = max(required_mib, math.ceil(result["peak_cuda_gib"]*1024)+2048)
                smoke_tasks[gpu].pop(0)
            elif path.exists():
                raise RuntimeError(f"Preserve incomplete smoke attempt: {path}")
            else:
                item = start_process(run_root,rank,depth,key,gpu,required_mib,True)
                if item is None: raise RuntimeError(f"GPU{gpu} has insufficient smoke memory")
                smoke_active[gpu] = item
        while smoke_active:
            for gpu,item in list(smoke_active.items()):
                code=item["process"].poll()
                if code is None: continue
                item["log"].close(); del smoke_active[gpu]
                if code:
                    smoke_failures.append(str(item["run_dir"])); continue
                rank,depth,key,_=smoke_tasks[gpu][0]
                result=validate(item["run_dir"],manifest,rank,depth,key,True)
                required_mib=max(required_mib,math.ceil(result["peak_cuda_gib"]*1024)+2048)
                smoke_tasks[gpu].pop(0)
            if smoke_failures and not smoke_active:
                raise RuntimeError(f"Smoke failed and is preserved: {smoke_failures}")
            if smoke_active: time.sleep(2)
        atomic_json(run_root/"smokes_passed.json",{"passed":True,"smoke_count":4,"required_free_mib":required_mib,
                     "runs":[str(run_root/run_name(1,4,key,True)) for key in CONFIG_ORDER],"completed_at":c.now_text()})
        if args.stop_after == "smoke":
            print("All four full-size smoke settings passed; production not launched.",flush=True); return

        # Case-centric queues: all16 settings for each case run sequentially on
        # that case's assigned GPU; no GPU receives more than one new worker.
        production={gpu:[] for gpu in args.gpus}
        for rank in range(10):
            gpu=manifest["case_to_gpu"][str(rank)]
            for depth,key in config_rows(): production[gpu].append((rank,depth,key,False))
        total=160
        for gpu,queue in production.items():
            for task in list(queue):
                rank,depth,key,_=task; path=run_root/run_name(rank,depth,key)
                if (path/"DONE").is_file():
                    validate(path,manifest,rank,depth,key,False); queue.remove(task)
                elif path.exists():
                    raise RuntimeError(f"Incomplete production attempt preserved: {path}")
        run_queue(run_root,manifest,production,required_mib,total,audit=False)
        all_rows=[]
        for rank in range(10):
            for depth,key in config_rows():
                all_rows.append(validate(run_root/run_name(rank,depth,key),manifest,rank,depth,key,False))

        audits={gpu:[] for gpu in args.gpus}
        for row in all_rows:
            path=Path(row["run_dir"])/"independent_audit.json"
            if path.exists(): audit_ok(row)
            else: audits[row["gpu"]].append((row["case_rank"],row["depth"],row["config_key"],False))
        run_queue(run_root,manifest,audits,required_mib,160,audit=True)
        for row in all_rows: audit_ok(row)
        write_report(all_rows,manifest,report_root)
        atomic_json(run_root/"progress.json",{"phase":"complete","completed":160,"audited":160,
                                              "updated_at":c.now_text()})
        print(f"Accepted report: {report_root}",flush=True)


if __name__ == "__main__":
    args=parse_args()
    try:
        if args.command=="worker": worker(args)
        else: launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise
