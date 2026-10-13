#!/usr/bin/env python
"""Independent full-resolution cps8/4/2-to-dense native DL-loss worker."""
from __future__ import annotations

import argparse
import importlib.metadata
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
import dl_ncc_diffusion as geometry
import dl_loss_sweep as d
import run_oasis_dl_ncc_diffusion_one as previous
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file
from run_oasis_vfa_feature_dense_one import canonical_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-rank", type=int, required=True)
    p.add_argument("--loss", choices=d.LOSSES, required=True)
    p.add_argument("--multiplier", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--oasis-root", type=Path, required=True)
    p.add_argument("--wavereg-code-root", type=Path, required=True)
    p.add_argument("--vfa-root", type=Path, required=True)
    p.add_argument("--feature-checkpoint", type=Path, default=d.CHECKPOINT)
    p.add_argument("--feature-config", type=Path, default=d.CONFIG)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    args = p.parse_args(argv)
    if args.feature_checkpoint is None or args.feature_config is None:
        p.error("--feature-checkpoint and --feature-config are required for native Feature-MSE")
    spec_from_args(args).validate()
    if args.cpu_threads < 1 or args.out_dir.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Require positive threads and external output directory")
    return args


def spec_from_args(args):
    return d.FitSpec(args.case_rank, args.loss, args.multiplier, args.seed, args.smoke)


def source_hashes():
    result = previous.source_hashes()
    for name in ("dl_loss_sweep.py", "run_oasis_dl_loss_one.py", "launch_oasis_dl_loss_sweep.py"):
        result[name] = sha256_file(HERE/name)
    report = c.REPO_ROOT/"scripts/reports/oasis_dl_loss_first10_arc.py"
    result[report.name] = sha256_file(report)
    return result


def native_hashes(args):
    paths = sorted(args.wavereg_code_root.rglob("*.py"))
    paths += sorted(args.feature_config.glob("*.yaml"))
    paths += [args.feature_checkpoint, args.vfa_root/"vfa/losses/grad_loss.py"]
    paths += sorted((args.wavereg_code_root.parent/"SITReg/src").rglob("*.py"))
    if not paths or not args.feature_checkpoint.is_file():
        raise FileNotFoundError("Native losses or frozen feature checkpoint missing")
    return {str(path): sha256_file(path) for path in paths}


def run(args):
    started = time.monotonic()
    spec = spec_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    batch, input_hashes = previous.load_batch(args, device)
    manifest = {"variant": d.VARIANT, "method": d.METHOD, "spec": asdict(spec), "protocol": d.PROTOCOL,
                "run_name": spec.name(), "source_revision": args.source_revision,
                "source_hashes": source_hashes(), "native_source_hashes": native_hashes(args),
                "input_hashes": input_hashes, "pair_file": str(args.pair_file),
                "pair_file_sha256": sha256_file(args.pair_file), "case_rank": spec.case_rank,
                "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
                "original_case_index": batch["original_case_index"], "direction": "moving_to_fixed",
                "sampling": "fixed_to_moving_pull_identity_plus_displacement", "gpu": args.gpu,
                "cuda_backend": c.cuda_backend_metadata(device), "tf32_matmul": False, "tf32_cudnn": False,
                "cuda_nondeterminism_caveat": "cuDNN deterministic does not guarantee grid_sample backward bitwise determinism",
                "uid": os.getuid(), "gid": os.getgid(), "argv": sys.argv,
                "python_version": sys.version, "pythonpath": os.environ.get("PYTHONPATH", ""),
                "package_versions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                "initial_gpu_state": gpu_snapshot(), "started_at": c.now_text(),
                "effective_lambda": spec.alpha, "similarity_scale": spec.scale,
                "actual_regularizer_coefficient": spec.scale*spec.alpha,
                "feature_checkpoint": str(args.feature_checkpoint) if spec.loss == "femse" else None,
                "feature_checkpoint_sha256": sha256_file(args.feature_checkpoint) if spec.loss == "femse" else None,
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded")}
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    c.write_json(args.out_dir/"manifest.json", manifest)
    native = d.native_loss(spec.loss, args.wavereg_code_root, device, args.feature_checkpoint, args.feature_config)
    if spec.smoke:
        manifest["native_parity"] = d.parity(spec.loss, native, device)
        grad = previous.import_file("native_diffusion_parity", args.vfa_root/"vfa/losses/grad_loss.py").Grad()
        field = torch.randn(1, 3, 12, 14, 16, device=device, requires_grad=True)
        a, b = grad(field), c.forward_valid_diffusion(field, None)
        ga, = torch.autograd.grad(a, field); gb, = torch.autograd.grad(b, field)
        if not torch.allclose(a, b, atol=2e-6, rtol=1e-5) or not torch.allclose(ga, gb, atol=2e-6, rtol=1e-5):
            raise RuntimeError("Native diffusion parity failed")
        manifest["diffusion_parity_passed"] = True
        del field, a, b, ga, gb
    similarity = d.Similarity(spec.loss, native, batch["fixed"][None, None])
    manifest["fixed_feature_shape"] = list(similarity.fixed_features.shape) if similarity.fixed_features is not None else None
    manifest["frozen_loss_parameter_count"] = sum(p.numel() for p in native.parameters())
    c.write_json(args.out_dir/"manifest.json", manifest)
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, 8, [8, 4, 2])
    parameter = torch.zeros((*shapes[8], 3), device=device)
    summaries, checks, previous_field = [], [], None
    for stage in d.stages(spec):
        if stage.index:
            if stage.cps is None:
                parameter = previous_field.detach().clone()
                if not torch.equal(parameter, previous_field):
                    raise RuntimeError("Dense initialization differs from cps2 expansion")
                checks.append({"stage": 3, "kind": "exact_cps2_expansion_clone", "max_abs_voxel": 0., "rms_voxel": 0.})
            else:
                parameter = c.cubic_bspline_dyadic_knot_insert_3d(parameter, shapes[stage.cps]).detach()
                candidate = geometry.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE, c.IMAGE_SHAPE, stage.cps, device)(parameter)
                checks.append({"stage": stage.index, **previous.verify_init(previous_field, candidate)})
                del candidate
            del previous_field
        objective = d.Objective(batch["fixed"], batch["moving"], similarity, spec, stage,
                                shapes[stage.cps] if stage.cps is not None else None)
        parameter, info = d.optimize_stage(parameter, objective, stage, args.out_dir/f"stage{stage.index}")
        if any(p.requires_grad or p.grad is not None for p in native.parameters()):
            raise RuntimeError("Frozen similarity parameters acquired gradients")
        summaries.append(info)
        with torch.no_grad():
            previous_field = objective.field(parameter).detach()
        del objective
    field = previous_field
    field_path = args.out_dir/f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.cpu().numpy())
    reloaded = torch.from_numpy(np.load(field_path)).to(device)
    if reloaded.shape != (*c.IMAGE_SHAPE, 3) or not torch.equal(field, reloaded) or not bool(torch.isfinite(field).all()):
        raise RuntimeError("Finite full-resolution DVF serialization failed")
    vox = reloaded*geometry.voxel_scale(c.IMAGE_SHAPE, device=device)
    metrics, labels = canonical_metrics(vox, batch)
    labels["legacy_global_ncc_diagnostic_only"] = labels.pop("dense_image_loss")
    c.write_csv(args.out_dir/"case_metrics.csv", [labels])
    excluded = [label for label in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    zyx = vox.cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = c.jacobian_metrics_voxel(zyx, batch["fixed_mask"].cpu().numpy().astype(bool))
    diffusion = float(c.forward_valid_diffusion(vox.permute(3, 0, 1, 2)[None], None))
    if not all(math.isfinite(float(x)) for x in (*metrics.values(), *regularity.values(), diffusion)):
        raise FloatingPointError("Nonfinite final metrics")
    model = args.out_dir/"stage3/best.pt"
    summary = {**manifest, **metrics, **regularity, "diffusion": diffusion,
               "stages": summaries, "initialization_checks": checks, "run_dir": str(args.out_dir),
               "dvf_path": str(field_path), "dvf_sha256": sha256_file(field_path),
               "model_path": str(model), "model_sha256": sha256_file(model),
               "parameter_units": "normalized_grid_xyz", "dvf_units": "normalized_grid_xyz",
               "representation": "dense_voxel", "cps": None, "control_shape": None, "bspline_order": None,
               "parameter_count": parameter.numel(), "best_epoch": summaries[-1]["best_epoch"],
               "best_loss": summaries[-1]["best_loss"], "valid_label_count": len(c.OASIS_LABELS)-len(excluded),
               "excluded_labels": excluded, "training_elapsed_sec": sum(s["training_elapsed_sec"] for s in summaries),
               "total_elapsed_sec": time.monotonic()-started, "segmentation_loss": False,
               "feature_loss": spec.loss == "femse", "reference_dvf_supervision": False,
               "objective_support": "full_native_volume", "completed_at": c.now_text()}
    c.write_json(args.out_dir/"summary.json", summary)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")
    print(json.dumps({"complete": spec.name(), **metrics}), flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.out_dir.exists():
            c.write_json(args.out_dir/"FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
