"""One independent GNCC/GNCC2 cps8->4->2->dense registration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import nibabel as nib
import numpy as np
import torch

import common as c
import gncc_dense as d
import dl_ncc_diffusion as geometry
import run_oasis_dl_ncc_diffusion_one as previous
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file
from run_oasis_sitreg_first100_eval import label_rows, json_safe


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-rank", type=int, required=True)
    p.add_argument("--loss", choices=(*d.LOSSES, "lncc"), required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--oasis-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--cpu-threads", type=int, default=4)
    args = p.parse_args(argv)
    d.FitSpec(args.case_rank, args.loss, args.alpha, smoke=args.smoke).validate()
    if args.pair_file.resolve() != c.DEFAULT_PAIR_FILE.resolve() or args.cpu_threads < 1:
        p.error("Require canonical split and positive threads")
    for source in (c.REPO_ROOT, args.oasis_root):
        if args.out_dir.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.out_dir.resolve()):
            p.error("Output overlaps protected sources")
    return args


def run(args):
    begin = time.monotonic()
    spec = d.FitSpec(args.case_rank, args.loss, args.alpha, smoke=args.smoke)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    gpu_before = gpu_snapshot()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    batch, inputs = previous.load_batch(args, device)
    inputs["dataset.json"] = sha256_file(args.oasis_root/"dataset.json")
    manifest = {"variant": d.VARIANT, "method": d.METHODS[spec.loss], "spec": asdict(spec), "protocol": d.PROTOCOL,
                "run_name": spec.name(), "source_hashes": d.source_hashes(), "input_hashes": inputs,
                "source_revision": subprocess.check_output(["git", "-C", str(c.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "case_rank": spec.case_rank, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
                "original_case_index": batch["original_case_index"], "direction": "moving_to_fixed",
                "sampling": "fixed_to_moving_pull_identity_plus_displacement", "gpu": args.gpu,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "visible_cuda_device": "cuda:0",
                "initial_gpu_state": gpu_before, "existing_gpu_processes": subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True),
                "cuda_backend": c.cuda_backend_metadata(device), "tf32_matmul": False, "tf32_cudnn": False,
                "uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid(), "argv": sys.argv,
                "torch": torch.__version__, "pair_file_sha256": sha256_file(args.pair_file),
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID"),
                "effective_lambda": spec.alpha, "similarity_scale": 1., "actual_regularizer_coefficient": spec.alpha,
                "started_unix": time.time()}
    c.write_json(args.out_dir/"manifest.json", manifest)
    sim = d.similarity(spec.loss, batch["fixed"][None, None])
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, 8, [8, 4, 2])
    parameter = torch.zeros((*shapes[8], 3), device=device)
    summaries, checks = [], []
    field = None
    for stage in d.stages(spec):
        if stage.index:
            if stage.cps is None:
                parameter = field.detach().clone()
                if not torch.equal(parameter, field):
                    raise RuntimeError("Dense initialization changed")
                checks.append({"stage": 3, "kind": "exact_cps2_expansion_clone", "max_abs_voxel": 0., "rms_voxel": 0.})
            else:
                parameter = c.cubic_bspline_dyadic_knot_insert_3d(parameter, shapes[stage.cps]).detach()
                expanded = geometry.AlignedBSpline(shapes[stage.cps], c.IMAGE_SHAPE, c.IMAGE_SHAPE, stage.cps, device)(parameter)
                checks.append({"stage": stage.index, **previous.verify_init(field, expanded)})
                del expanded
            del field
        objective = d.Objective(batch["fixed"], batch["moving"], sim, spec, stage,
                                shapes[stage.cps] if stage.cps is not None else None)
        parameter, info = d.optimize_stage(parameter, objective, stage, args.out_dir/f"stage{stage.index}")
        with torch.no_grad():
            field = objective.field(parameter).detach()
        np.save(args.out_dir/f"stage{stage.index}/disp_norm.npy", field.cpu().numpy())
        summaries.append(info)
        del objective
    field_path = args.out_dir/f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.cpu().numpy())
    restored = np.load(field_path)
    if restored.shape != (*c.IMAGE_SHAPE, 3) or not np.isfinite(restored).all() or not np.array_equal(restored, field.cpu().numpy()):
        raise RuntimeError("DVF roundtrip failed")
    peak_reserved = torch.cuda.max_memory_reserved()/2**30
    del field, parameter, sim
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    torch.cuda.empty_cache()
    norm = torch.from_numpy(np.ascontiguousarray(restored))
    # Match the canonical field-voxel->normalized evaluation roundtrip.
    scale = geometry.voxel_scale(c.IMAGE_SHAPE, device="cpu")
    voxel = norm*scale
    metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(c.IMAGE_SHAPE, torch.device("cpu"))+voxel/scale, batch)
    metadata = {k: manifest[k] for k in ("case_rank", "fixed_id", "moving_id", "direction")}
    metadata.update(loss=spec.loss, alpha=spec.alpha)
    labels = json_safe(label_rows(metrics, batch["fixed_seg"], batch["moving_seg"], metadata))
    zyx = voxel.numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = c.jacobian_metrics_voxel(zyx, batch["fixed_mask"].bool().numpy())
    short = {key: float(metrics[value]) for key, value in
             (("dice", "dice_mean"), ("hd95", "hd95_mean"), ("asd", "asd_symmetric_mean"))}
    diffusion = float(c.forward_valid_diffusion(voxel.permute(3, 0, 1, 2)[None], None))
    if not all(math.isfinite(v) for v in (*short.values(), diffusion)):
        raise RuntimeError("Nonfinite metrics")
    c.write_json(args.out_dir/"per_label.json", labels); c.write_csv(args.out_dir/"per_label.csv", labels)
    nib.save(nib.Nifti1Image(metrics["warped_seg"].numpy().astype(np.int16), np.eye(4)), str(args.out_dir/"warped_seg.nii.gz"))
    model = args.out_dir/"stage3/best.pt"
    row = {**manifest, **short, **regularity, "diffusion": diffusion, "stages": summaries,
           "initialization_checks": checks, "run_dir": str(args.out_dir),
           "dvf_path": str(field_path), "dvf_sha256": sha256_file(field_path),
           "model_path": str(model), "model_sha256": sha256_file(model),
           "parameter_units": "normalized_grid_xyz", "representation": "dense_voxel",
           "cps": None, "control_shape": None, "bspline_order": None,
           "objective_support": "full_native_volume", "valid_label_count": sum(r["included_in_case_mean"] for r in labels),
           "excluded_labels": [r["label_id"] for r in labels if not r["included_in_case_mean"]],
           "best_epoch": summaries[-1]["best_epoch"], "best_loss": summaries[-1]["best_loss"],
           "training_elapsed_sec": sum(s["training_elapsed_sec"] for s in summaries),
           "total_elapsed_sec": time.monotonic()-begin, "peak_reserved_gib": peak_reserved,
           "segmentation_loss": False, "feature_loss": False, "reference_dvf_supervision": False,
           "completed_unix": time.time()}
    row["outputs_sha256"] = {str(p.relative_to(args.out_dir)): sha256_file(p) for p in args.out_dir.rglob("*") if p.is_file()}
    d.verify_source_hashes(manifest["source_hashes"])
    c.write_json(args.out_dir/"summary.json", row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")
    print(json.dumps({"complete": spec.name(), **short}), flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception:
        if args.out_dir.exists():
            c.write_json(args.out_dir/"FAILED.json", {"traceback": traceback.format_exc()})
        raise
