#!/usr/bin/env python
"""One isolated DL-NCC9/diffusion prefix or final refinement worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
import dl_ncc_diffusion as d
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file
from run_oasis_vfa_feature_dense_one import canonical_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-rank", type=int, required=True)
    p.add_argument("--representation", choices=d.REPRESENTATIONS, required=True)
    p.add_argument("--resolution", choices=d.RESOLUTIONS, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--oasis-root", type=Path, required=True)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--dl-code-root", type=Path, required=True)
    p.add_argument("--vfa-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix-dir", type=Path)
    p.add_argument("--source-revision", required=True)
    args = p.parse_args(argv)
    spec_from_args(args).validate()
    if (args.representation != "842") != (args.prefix_dir is not None):
        p.error("Only refinement workers require a prefix directory")
    if args.cpu_threads < 1 or args.out_dir.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Require positive threads and an external output directory")
    return args


def spec_from_args(args):
    return d.FitSpec(args.case_rank, args.resolution, args.alpha, args.representation, args.seed, args.smoke)


def source_hashes():
    names = ("dl_ncc_diffusion.py", "run_oasis_dl_ncc_diffusion_one.py",
             "launch_oasis_dl_ncc_diffusion_sweep.py", "common.py", "run_one.py",
             "vfa_feature_refinement.py", "run_oasis_vfa_feature_dense_one.py",
             "run_oasis_saved_field_bspline_fit_one.py", "evaluate_saved_baselines_first10.py")
    result = {name: sha256_file(HERE/name) for name in names}
    report = c.REPO_ROOT/"scripts/reports/oasis_dl_ncc_diffusion_first10_arc.py"
    result[report.name] = sha256_file(report)
    return result


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_parity(device, dl_code_root, vfa_root):
    native = import_file("dl_ncc9_native_audit", dl_code_root/"utils_losses_ncc.py").NCC_vfa(9)
    grad = import_file("vfa_grad_native_audit", vfa_root/"vfa/losses/grad_loss.py").Grad()
    rng = torch.Generator(device=device).manual_seed(19)
    a = torch.rand((1, 1, 10, 12, 14), generator=rng, device=device)
    b = torch.rand(a.shape, generator=rng, device=device, requires_grad=True)
    x, y = native(b, a), d.NCC9(a)(b)
    gx, = torch.autograd.grad(x, b)
    gy, = torch.autograd.grad(y, b)
    field = torch.randn((1, 3, 10, 12, 14), generator=rng, device=device, requires_grad=True)
    rx, ry = grad(field), c.forward_valid_diffusion(field, None)
    grx, = torch.autograd.grad(rx, field)
    gry, = torch.autograd.grad(ry, field)
    checks = {"ncc_value_abs": float((x-y).abs()), "ncc_gradient_max_abs": float((gx-gy).abs().max()),
              "diffusion_value_abs": float((rx-ry).abs()),
              "diffusion_gradient_max_abs": float((grx-gry).abs().max())}
    for actual, expected in ((y, x), (gy, gx), (ry, rx), (gry, grx)):
        if not torch.allclose(actual, expected, atol=2e-6, rtol=1e-5):
            raise RuntimeError(f"Native DL loss/gradient parity failed: {checks}")
    checks["passed"] = True
    return checks


def load_batch(args, device):
    batch = c.load_oasis_case(rank=args.case_rank, device=device, oasis_root=args.oasis_root, pair_file=args.pair_file)
    hashes = {}
    for role in ("fixed", "moving"):
        for kind in ("img", "seg"):
            path = args.oasis_root/"test"/f"{kind}{batch[role+'_id']}.nii.gz"
            ni = nib.load(str(path))
            if ni.shape != (160, 192, 224) or tuple(ni.header.get_zooms()[:3]) != (1., 1., 1.):
                raise RuntimeError("Unexpected image geometry/spacing")
            hashes[path.name] = sha256_file(path)
        raw = c.load_nifti_reoriented(args.oasis_root/"test"/f"img{batch[role+'_id']}.nii.gz", dtype=np.float32)
        if not np.isfinite(raw).all() or raw.shape != c.IMAGE_SHAPE:
            raise RuntimeError("Invalid raw input")
        batch[role] = torch.from_numpy(raw).to(device)
    return batch, hashes


def verify_init(before, after):
    scale = d.voxel_scale(before.shape[:3], device=before.device)
    diff = (after-before)*scale
    values = {"max_abs_voxel": float(diff.abs().max()), "rms_voxel": float(diff.square().mean().sqrt())}
    if values["max_abs_voxel"] > 1e-4 or values["rms_voxel"] > 1e-5:
        raise RuntimeError(f"Prolongation changed the initial field: {values}")
    return values


def load_prefix(args, spec, manifest, device):
    root = args.prefix_dir
    if not (root/"DONE").is_file():
        raise RuntimeError("Incomplete prefix")
    summary = c.read_json(root/"summary.json")
    for key, expected in (("spec", asdict(spec.prefix())), ("protocol", d.PROTOCOL),
                          ("source_hashes", manifest["source_hashes"]),
                          ("input_hashes", manifest["input_hashes"]),
                          ("pair_file_sha256", manifest["pair_file_sha256"])):
        if summary[key] != expected:
            raise RuntimeError(f"Incompatible prefix: {key}")
    for key in ("model_path", "dvf_path"):
        if sha256_file(Path(summary[key])) != summary[key.replace("_path", "_sha256")]:
            raise RuntimeError("Prefix checksum changed")
    saved = torch.load(summary["model_path"], map_location="cpu", weights_only=False)
    controls = saved["parameter_norm_xyz"].to(device)
    field = torch.from_numpy(np.load(summary["dvf_path"])).to(device)
    reproduced = d.AlignedBSpline(controls.shape[:3], c.IMAGE_SHAPE, c.IMAGE_SHAPE, 2, device)(controls)
    delta = verify_init(field, reproduced)
    return controls, field, summary, delta


def run(args):
    started = time.monotonic()
    spec = spec_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed)
    batch, input_hashes = load_batch(args, device)
    manifest = {"variant": d.VARIANT, "spec": asdict(spec), "protocol": d.PROTOCOL,
                "run_name": spec.name(), "method": d.METHODS[spec.representation],
                "source_revision": args.source_revision, "source_hashes": source_hashes(),
                "input_hashes": input_hashes, "pair_file": str(args.pair_file),
                "pair_file_sha256": sha256_file(args.pair_file),
                "native_source_hashes": {str(p): sha256_file(p) for p in
                    (args.dl_code_root/"utils_losses_ncc.py", args.vfa_root/"vfa/losses/grad_loss.py")},
                "case_rank": spec.case_rank, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
                "original_case_index": batch["original_case_index"], "direction": "moving_to_fixed",
                "sampling": "fixed_to_moving_pull_identity_plus_displacement",
                "gpu": args.gpu, "cuda_backend": c.cuda_backend_metadata(device),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cuda_nondeterminism_caveat": "cuDNN deterministic does not guarantee grid_sample backward bitwise determinism",
                "uid": os.getuid(), "gid": os.getgid(), "argv": sys.argv,
                "initial_gpu_state": gpu_snapshot(), "started_at": c.now_text(),
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded")}
    if (manifest["uid"], manifest["gid"]) != (1009, 1020):
        raise RuntimeError("Production/smoke requires container UID:GID1009:1020")
    if spec.smoke:
        manifest["native_parity"] = native_parity(device, args.dl_code_root, args.vfa_root)
    c.write_json(args.out_dir/"manifest.json", manifest)
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, 8, [8, 4, 2, 1])
    stage_summaries, init_checks, prefix = [], [], None
    if spec.representation == "842":
        parameter = torch.zeros((*shapes[8], 3), device=device)
        previous_field = None
        for stage in d.stages(spec):
            if stage.index:
                parameter = c.cubic_bspline_dyadic_knot_insert_3d(parameter, shapes[stage.cps]).detach()
                candidate = d.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE, c.IMAGE_SHAPE, stage.cps, device)(parameter)
                init_checks.append({"stage": stage.index, **verify_init(previous_field, candidate)})
                del candidate, previous_field
            objective = d.StageObjective(batch["fixed"], batch["moving"], spec.alpha, stage, shapes[stage.cps])
            parameter, info = d.optimize_stage(parameter, objective, stage, args.out_dir/f"stage{stage.index}")
            stage_summaries.append(info)
            with torch.no_grad():
                previous_field = d.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE, c.IMAGE_SHAPE,
                                                  stage.cps, device)(parameter)
            del objective
        field = previous_field
    else:
        controls, initial, prefix, diagnostic = load_prefix(args, spec, manifest, device)
        init_checks.append({"kind": "prefix_reconstruction", **diagnostic})
        stage = d.stages(spec)[-1]
        if stage.cps is None:
            parameter = initial.detach().clone()
            if not torch.equal(parameter, initial):
                raise RuntimeError("Dense normalized initialization is not exact")
            init_checks.append({"kind": "dense_exact_clone", "max_abs_voxel": 0., "rms_voxel": 0.})
        else:
            parameter = c.cubic_bspline_dyadic_knot_insert_3d(controls, shapes[1]).detach()
            expanded = d.AlignedBSpline(shapes[1], c.IMAGE_SHAPE, c.IMAGE_SHAPE, 1, device)(parameter)
            init_checks.append({"kind": "cps2_to_cps1", **verify_init(initial, expanded)})
            del expanded
        del controls, initial
        objective = d.StageObjective(batch["fixed"], batch["moving"], spec.alpha, stage,
                                     shapes[1] if stage.cps is not None else None)
        parameter, info = d.optimize_stage(parameter, objective, stage, args.out_dir/"stage3")
        stage_summaries = prefix["stages"]+[info]
        with torch.no_grad():
            field = objective.field(parameter).detach()
        del objective
    field_path = args.out_dir/f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.detach().cpu().numpy())
    reload = torch.from_numpy(np.load(field_path)).to(device)
    if reload.shape != (*c.IMAGE_SHAPE, 3) or not bool(torch.isfinite(reload).all()) or not torch.equal(reload, field):
        raise RuntimeError("DVF serialization mismatch")
    scale = d.voxel_scale(c.IMAGE_SHAPE, device=device)
    vox = reload*scale
    metrics, labels = canonical_metrics(vox, batch)
    labels["canonical_legacy_global_ncc_diagnostic"] = labels.pop("dense_image_loss")
    c.write_csv(args.out_dir/"case_metrics.csv", [labels])
    excluded = [label for label in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    zyx = vox.detach().cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = c.jacobian_metrics_voxel(zyx, batch["fixed_mask"].cpu().numpy().astype(bool))
    with torch.no_grad():
        diffusion = float(c.forward_valid_diffusion(vox.permute(3, 0, 1, 2)[None], None))
    if not all(math.isfinite(float(v)) for v in (*metrics.values(), *regularity.values(), diffusion)):
        raise FloatingPointError("Nonfinite final metric")
    last_stage = d.stages(spec)[-1]
    model_path = args.out_dir/f"stage{last_stage.index}"/"best.pt"
    current_stages = stage_summaries if prefix is None else stage_summaries[-1:]
    training_elapsed = sum(s["training_elapsed_sec"] for s in current_stages)
    summary = {**manifest, **metrics, **regularity, "diffusion": diffusion,
               "run_dir": str(args.out_dir), "dvf_path": str(field_path), "dvf_sha256": sha256_file(field_path),
               "model_path": str(model_path), "model_sha256": sha256_file(model_path),
               "parameter_units": "normalized_grid_xyz", "dvf_units": "normalized_grid_xyz",
               "representation": "dense_voxel" if last_stage.cps is None else "bspline_control",
               "cps": last_stage.cps, "bspline_order": None if last_stage.cps is None else 3,
               "control_shape": None if last_stage.cps is None else list(shapes[last_stage.cps]),
               "parameter_count": parameter.numel(), "stages": stage_summaries,
               "initialization_checks": init_checks, "best_epoch": stage_summaries[-1]["best_epoch"],
               "best_loss": stage_summaries[-1]["best_loss"], "excluded_labels": excluded,
               "valid_label_count": len(c.OASIS_LABELS)-len(excluded),
               "training_elapsed_sec": training_elapsed,
               "pipeline_training_elapsed_sec": sum(s["training_elapsed_sec"] for s in stage_summaries),
               "total_elapsed_sec": time.monotonic()-started,
               "prefix_path": None if prefix is None else str(args.prefix_dir),
               "prefix_dvf_sha256": None if prefix is None else prefix["dvf_sha256"],
               "prefix_model_sha256": None if prefix is None else prefix["model_sha256"],
               "prefix_gpu": args.gpu if prefix is None else prefix["gpu"],
               "segmentation_loss": False, "feature_loss": False, "reference_dvf_supervision": False,
               "objective_support": "full_stage_grid", "completed_at": c.now_text()}
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
