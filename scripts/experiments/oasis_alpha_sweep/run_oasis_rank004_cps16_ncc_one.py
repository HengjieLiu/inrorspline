#!/usr/bin/env python
"""One GPU3-only cps16/8/4/2/dense registration, with native stage diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
import cps16_ncc as n
import dl_ncc_diffusion as d
import run_oasis_dl_ncc_diffusion_one as previous
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file
from run_oasis_vfa_feature_dense_one import canonical_metrics


def source_hashes():
    result = previous.source_hashes()
    for name in ("cps16_ncc.py", "run_oasis_rank004_cps16_ncc_one.py",
                 "launch_oasis_rank004_cps16_ncc.py"):
        result[name] = sha256_file(HERE/name)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", choices=("pyramid", "full"), required=True)
    p.add_argument("--gpu", choices=("3",), default="3")
    p.add_argument("--case-rank", type=int, choices=(4,), default=4)
    p.add_argument("--alpha", type=float, choices=(1.,), default=1.)
    p.add_argument("--seed", type=int, choices=(1,), default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=6)
    for name in ("oasis-root", "dl-code-root", "vfa-root", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--source-revision", required=True)
    args = p.parse_args(argv)
    n.FitSpec(args.resolution, args.smoke, args.case_rank, args.alpha, args.seed).validate()
    sources = (c.REPO_ROOT, args.oasis_root, args.dl_code_root, args.vfa_root)
    if args.cpu_threads < 1 or any(args.out_dir.resolve().is_relative_to(s.resolve()) or
                                 s.resolve().is_relative_to(args.out_dir.resolve()) for s in sources):
        p.error("Require positive threads and outputs disjoint from sources")
    return args


def measure(field, batch, destination):
    """Posthoc only: label metrics cannot influence optimization or selection."""
    field = field.detach()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(path, field.cpu().numpy())
    reloaded = torch.from_numpy(np.load(path)).to(field.device)
    if (reloaded.shape != (*c.IMAGE_SHAPE, 3) or reloaded.dtype != torch.float32 or
            not bool(torch.isfinite(reloaded).all()) or not torch.equal(field, reloaded)):
        raise RuntimeError("Finite native-field serialization failed")
    vox = reloaded*d.voxel_scale(c.IMAGE_SHAPE, device=field.device)
    metrics, labels = canonical_metrics(vox, batch)
    labels["legacy_global_ncc_diagnostic_only"] = labels.pop("dense_image_loss")
    c.write_csv(destination/"case_metrics.csv", [labels])
    excluded = [i for i in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{i:02d}"])]
    zyx = vox.cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    metrics.update(c.jacobian_metrics_voxel(zyx, batch["fixed_mask"].cpu().numpy().astype(bool)))
    metrics["diffusion"] = float(c.forward_valid_diffusion(vox.permute(3, 0, 1, 2)[None], None))
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise FloatingPointError("Nonfinite posthoc metrics")
    result = {**metrics, "dvf_path": str(path), "dvf_sha256": sha256_file(path),
              "case_metrics_path": str(destination/"case_metrics.csv"),
              "excluded_labels": excluded, "valid_label_count": len(c.OASIS_LABELS)-len(excluded)}
    c.write_json(destination/"metrics.json", result)
    return result


def run(args):
    started = time.monotonic()
    spec = n.FitSpec(args.resolution, args.smoke, args.case_rank, args.alpha, args.seed)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    if args.gpu != "3" or os.environ.get("CUDA_VISIBLE_DEVICES", "3") != "3":
        raise RuntimeError("Only physical GPU3 is authorized")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    initial_gpu_state = gpu_snapshot()
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device("3", cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    batch, input_hashes = previous.load_batch(args, device)
    if (batch["fixed_id"], batch["moving_id"], batch["original_case_index"]) != ("0084", "0108", 47):
        raise RuntimeError("Unexpected shuffled rank4 case identity")
    manifest = {
        "variant": n.VARIANT, "method": n.METHOD, "spec": asdict(spec), "protocol": n.PROTOCOL,
        "run_name": spec.name(), "source_revision": args.source_revision, "source_hashes": source_hashes(),
        "input_hashes": input_hashes, "pair_file": str(args.pair_file),
        "pair_file_sha256": sha256_file(args.pair_file), "case_rank": 4,
        "fixed_id": "0084", "moving_id": "0108", "original_case_index": 47,
        "direction": "moving_to_fixed", "sampling": "fixed_to_moving_pull_identity_plus_displacement",
        "gpu": "3", "cuda_backend": c.cuda_backend_metadata(device),
        "initial_gpu_state": initial_gpu_state, "tf32_matmul": False, "tf32_cudnn": False,
        "cuda_nondeterminism_caveat": "CUDA grid_sample backward is not guaranteed bitwise deterministic",
        "native_source_hashes": {str(p): sha256_file(p) for p in
            (args.dl_code_root/"utils_losses_ncc.py", args.vfa_root/"vfa/losses/grad_loss.py")},
        "uid": os.getuid(), "gid": os.getgid(), "argv": sys.argv, "started_at": c.now_text(),
        "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
        "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded"),
    }
    c.write_json(args.out_dir/"manifest.json", manifest)
    if spec.smoke:
        manifest["native_parity"] = previous.native_parity(device, args.dl_code_root, args.vfa_root)
        c.write_json(args.out_dir/"manifest.json", manifest)
    shapes = n.control_shapes()
    parameter = torch.zeros((*shapes[16], 3), device=device)
    summaries, diagnostics, checks = [], [], []
    field = None
    for stage in n.stages(spec):
        if stage.index:
            if stage.cps is None:
                parameter = field.detach().clone()
                if not torch.equal(parameter, field):
                    raise RuntimeError("Dense initialization is not an exact cps2 clone")
                checks.append({"stage": stage.index, "kind": "exact_cps2_dense_clone",
                               "max_abs_voxel": 0., "rms_voxel": 0.})
            else:
                parameter = c.cubic_bspline_dyadic_knot_insert_3d(parameter, shapes[stage.cps]).detach()
                with torch.no_grad():
                    reproduced = d.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE,
                                                 c.IMAGE_SHAPE, stage.cps, device)(parameter)
                    checks.append({"stage": stage.index, "kind": "dyadic_knot_insertion",
                                   **previous.verify_init(field, reproduced)})
                del reproduced
            del field
        objective = d.StageObjective(batch["fixed"], batch["moving"], 1., stage,
                                     shapes[stage.cps] if stage.cps is not None else None)
        stage_dir = args.out_dir/f"stage{stage.index}"
        parameter, info = d.optimize_stage(parameter, objective, stage, stage_dir)
        summaries.append(info)
        with torch.no_grad():
            field = (parameter.detach() if stage.cps is None else
                     d.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE,
                                      c.IMAGE_SHAPE, stage.cps, device)(parameter).detach())
        del objective
        diagnostic = measure(field, batch, stage_dir/"native_diagnostics")
        diagnostics.append({"index": stage.index, "cps": stage.cps, **diagnostic})
        print(json.dumps({"stage_diagnostic": stage.index,
                          **{k: diagnostic[k] for k in n.METRICS}}), flush=True)
    model = args.out_dir/"stage4/best.pt"
    summary = {
        **manifest, **diagnostics[-1], "stages": summaries, "stage_diagnostics": diagnostics,
        "initialization_checks": checks, "run_dir": str(args.out_dir),
        "model_path": str(model), "model_sha256": sha256_file(model),
        "parameter_units": "normalized_grid_xyz", "dvf_units": "normalized_grid_xyz",
        "representation": "dense_voxel", "cps": None, "control_shape": None, "bspline_order": None,
        "parameter_count": parameter.numel(), "best_epoch": summaries[-1]["best_epoch"],
        "best_loss": summaries[-1]["best_loss"], "alpha": 1.,
        "training_elapsed_sec": sum(s["training_elapsed_sec"] for s in summaries),
        "total_elapsed_sec": time.monotonic()-started,
        "segmentation_loss": False, "feature_loss": False, "reference_dvf_supervision": False,
        "objective_support": "full_stage_grid", "completed_at": c.now_text(),
    }
    c.write_json(args.out_dir/"summary.json", summary)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")
    print(json.dumps({"complete": spec.name(), **{k: summary[k] for k in n.METRICS}}), flush=True)


if __name__ == "__main__":
    args = parse_args()
    # Do not annotate an already-existing attempt if mkdir fails.
    existed = args.out_dir.exists()
    try:
        run(args)
    except Exception as exc:
        if not existed and args.out_dir.is_dir():
            c.write_json(args.out_dir/"FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
