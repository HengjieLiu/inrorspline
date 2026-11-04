#!/usr/bin/env python
"""P76: independent explicit-lambda replay of all six maps and canonical metrics."""
from __future__ import annotations

import argparse
import csv
import math
import time
import traceback
import os
from pathlib import Path

import numpy as np
import torch

import run_oasis_bounded2_842_sweep as r


def close(a, b, *, rel=1e-6, abs_=1e-7):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    args = parser.parse_args()
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID 1009:1020")
    audit_path = args.run_dir / "independent_audit.json"
    if audit_path.exists():
        raise RuntimeError(f"Preserve prior independent audit: {audit_path}")

    manifest = r.c.read_json(args.manifest)
    row = r.c.read_json(args.run_dir / "summary.json")
    r.validate(args.run_dir, manifest, row["case_rank"], row["alpha"], row["smoke"])
    if row["gpu"] != args.gpu:
        raise RuntimeError("Replay must use the original physical GPU")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = r.c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    for key in ("oasis_root", "pair_file", "dl_code_root", "vfa_root"):
        setattr(args, key, Path(manifest["settings"][key]))
    args.case_rank = row["case_rank"]
    batch, input_hashes = r.legacy.load_batch(args, device)
    if input_hashes != row["input_hashes"]:
        raise RuntimeError("Input hashes changed before independent replay")
    case = manifest["cases"][str(row["case_rank"])]
    if (batch["fixed_id"], batch["moving_id"]) != (case["fixed_id"], case["moving_id"]):
        raise RuntimeError("Independent replay loaded the wrong pair")

    config = r.CONFIGS[row["config_key"]]
    identity = r.c.make_coordinate_grid(r.c.IMAGE_SHAPE, device)
    scale = r.d.voxel_scale(r.c.IMAGE_SHAPE, device=device)
    atomics, checks = [], []
    previous_field = torch.zeros_like(identity)
    stages = r.b.stages(row["smoke"])[:row["depth"]]
    for step, record in enumerate(row["blocks"]):
        level, map_index = divmod(step, config["maps_per_level"])
        stage = stages[level]
        frozen = atomics[:step]
        objective = r.b.Objective(batch["fixed"], batch["moving"], stage, row["alpha"],
            frozen, manifest["bounds"][str(stage.cps)], config["bounded"])
        checkpoint = torch.load(record["model_path"], map_location=device, weights_only=False)
        latent = checkpoint["latent_norm_xyz"]
        zero = torch.zeros(*objective.control_shape, 3, device=device)
        with torch.no_grad():
            init_atomic = objective.atomic(zero)
            initialized = r.b.compose([*frozen, init_atomic], identity) - identity
            init_check = r.legacy.verify_init(previous_field, initialized)
            if init_check != record["initialization_check"]:
                raise RuntimeError(f"Step {step}: initialization replay differs")
            initial_values = [float(x) for x in objective(zero)]
            if any(not close(a, b) for a, b in zip(initial_values, record["initial_loss_data_reg"])):
                raise RuntimeError(f"Step {step}: initial objective differs")
            controls = objective.controls(latent)
            if not torch.equal(controls, checkpoint["controls_norm_xyz"]):
                raise RuntimeError(f"Step {step}: control checkpoint reconstruction differs")
            atomic = objective.expand(controls).detach()
            field = r.b.compose([*frozen, atomic], identity) - identity
            saved = torch.from_numpy(np.load(record["dvf_path"])).to(device)
            if not torch.equal(field, saved):
                raise RuntimeError(f"Step {step}: saved map field does not reconstruct")
            values = [float(x) for x in objective(latent)]
            info = record["optimizer"]
            for value, key in zip(values, ("best_loss", "best_data_loss", "best_reg_loss")):
                if not close(value, info[key]):
                    raise RuntimeError(f"Step {step}: best objective {key} differs")
            if config["bounded"]:
                corner = r.b.corner_diagnostics(atomic * scale)
                if corner != record["atomic_corner_audit"]:
                    raise RuntimeError(f"Step {step}: bounded-map derivative replay differs")
                if not corner["bound_below_one"] or corner["nonpositive_corner_count"]:
                    raise RuntimeError(f"Step {step}: bounded-map contraction check failed")
            elif record["atomic_corner_audit"] is not None:
                raise RuntimeError(f"Step {step}: unbounded map has a bounded audit")

            history_path = Path(record["model_path"]).parent / "history.csv"
            with history_path.open() as stream:
                history = list(csv.DictReader(stream))
            best_history = min(history, key=lambda x: float(x["loss"]))
            if int(best_history["epoch"]) != info["best_epoch"] or not close(best_history["loss"], info["best_loss"]):
                raise RuntimeError(f"Step {step}: best checkpoint/history consistency failed")

        checks.append({"step": step, "level": level, "map_index": map_index,
            "exact_field_reconstruction": True, "initial_objective": initial_values,
            "selected_loss_data_reg": values,
            "corner_audit_reproduced": bool(config["bounded"])})
        r.atomic_json(args.run_dir / "audit_progress.json", {"step": step, "updated_at": r.c.now_text()})
        atomics.append(atomic)
        previous_field = field.detach()
        del objective, checkpoint, latent, zero, init_atomic, initialized, controls, saved
        print(f"Audited rank{row['case_rank']} depth{row['depth']} {row['config_key']} step{step}", flush=True)

    probes = r.b.offgrid_diagnostics(atomics, previous_field, batch["fixed_mask"], count=100000, seed=1)
    if probes != row["offgrid_audit"]:
        raise RuntimeError("Independent 100,000-point off-grid replay differs")

    # Recompute the five headline metrics and all per-label canonical metrics
    # from the exported final DVF, without modifying the run's saved CSV.
    vox = previous_field * scale
    accuracy, labels = r.legacy.canonical_metrics(vox, batch)
    vox_zyx = vox.cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = r.c.jacobian_metrics_voxel(vox_zyx, batch["fixed_mask"].cpu().numpy().astype(bool))
    metrics = {**accuracy, **regularity}
    for key in r.METRICS_TO_REPORT:
        if not close(metrics[key], row[key], rel=0, abs_=1e-12):
            raise RuntimeError(f"Final canonical metric {key} does not reproduce")
    absent = [label for label in r.c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    if absent != row["excluded_labels"] or len(r.c.OASIS_LABELS) - len(absent) != row["valid_label_count"]:
        raise RuntimeError("Canonical absent-label handling differs")
    saved_csv = args.run_dir / "final_metrics" / "case_metrics.csv"
    with saved_csv.open() as stream:
        saved_labels = next(csv.DictReader(stream))
    label_fields = ("dice", "hd95", "asd_warped_to_fixed", "asd_fixed_to_warped", "asd_symmetric")
    for label in r.c.OASIS_LABELS:
        for metric in label_fields:
            key = f"{metric}_label_{label:02d}"
            actual, expected = labels[key], float(saved_labels[key])
            if math.isnan(actual) and math.isnan(expected):
                continue
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"Canonical label result mismatch: {key}")

    r.assert_manifest(manifest)
    r.atomic_json(audit_path, {"passed": True, "case_rank": row["case_rank"],
        "depth": row["depth"], "config_key": row["config_key"], "gpu": args.gpu,
        "alpha": row["alpha"], "smoke": row["smoke"],
        "audit_elapsed_sec": time.monotonic() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checks": checks, "offgrid_reproduced": True,
        "all_five_canonical_metrics_reproduced": True, "all_label_metrics_reproduced": True,
        "audit_source_sha256": r.sha(Path(__file__)),
        "summary_sha256": r.sha(args.run_dir / "summary.json"), "completed_at": r.c.now_text()})


if __name__ == "__main__":
    main()

