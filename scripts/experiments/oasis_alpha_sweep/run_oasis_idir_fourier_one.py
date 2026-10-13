"""One Fourier/control fit, preserving sampled-IDIR optimization and metrics."""
from __future__ import annotations

import argparse
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

import nibabel as nib
import numpy as np
import torch

import common as c
import idir_sampled_diffusion as old
import idir_fourier_diffusion as p
from run_oasis_idir_sampled_diffusion_one import configure, load_batch
from run_oasis_sitreg_first100_eval import label_rows, json_safe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--gpu", type=int, choices=p.GPUS, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.spec = p.FitSpec(**json.loads(args.spec_json)); args.spec.validate()
    args.rank = args.spec.rank
    if args.cpu_threads < 1:
        parser.error("Positive CPU thread count required")
    for source in (c.REPO_ROOT, args.oasis_root):
        if args.out_dir.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.out_dir.resolve()):
            parser.error("Outputs must be outside code/data trees")
    return args


def checkpoint(path, model, optimizer, spec, updates, source):
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": old.cpu_state(model), "optimizer": optimizer.state_dict(), "spec": asdict(spec),
                "updates": updates, "protocol": p.PROTOCOL, "source_sha256": source}, temporary)
    temporary.replace(path)


def run(args):
    begin = time.monotonic(); spec = args.spec
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    device, gpu_state, fraction = configure(args.gpu, args.cpu_threads)
    random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed); torch.cuda.manual_seed_all(spec.seed)
    batch, inputs = load_batch(args, device)
    model = p.build_model(spec).to(device)
    pool = old.training_coords(batch, device)
    # Local stream prevents feature/model initialization from changing training samples.
    sampler = torch.Generator(device=device).manual_seed(1)
    initial_hash = old.state_hash(model)
    initial_model_state = old.cpu_state(model)
    opt = old.optimizer(model)
    meta = {"rank": spec.rank, "family": spec.family, "sigma": spec.sigma, "features": spec.features if spec.sigma else 0,
            "width": spec.width, "omega": spec.omega, "alpha": spec.alpha, "seed": spec.seed,
            "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"], "direction": "moving_to_fixed"}
    manifest = {**meta, "variant": p.VARIANT, "method": p.METHOD, "spec": asdict(spec), "protocol": p.PROTOCOL,
                "source_sha256": p.source_hashes(), "input_sha256": inputs, "pair_file_sha256": old.sha256(c.DEFAULT_PAIR_FILE),
                "original_case_index": batch["original_case_index"], "initial_state_sha256": initial_hash,
                "B0_sha256": p.tensor_hash(model.B0) if spec.sigma else None, "B0_frozen": not spec.sigma or not model.B0.requires_grad,
                "train_coords": len(pool), "param_count": c.count_trainable_params(model), "initial_diagnostics": None,
                "derivative_check": None, "physical_gpu": args.gpu, "visible_cuda_device": "cuda:0",
                "gpu_uuid": next(g["uuid"] for g in gpu_state if g["index"] == args.gpu), "gpu_before": gpu_state,
                "allocator_fraction": fraction, "backend": c.cuda_backend_metadata(device),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "existing_gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True),
                "uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid(), "argv": sys.argv,
                "repo_revision": subprocess.check_output(["git", "-C", str(c.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE"), "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID"),
                "started_unix": time.time()}
    if manifest["param_count"] != spec.param_count:
        raise RuntimeError("Unexpected parameter count")
    old.write_json(args.out_dir / "manifest.json", manifest)
    points = 0
    def count_points(_module, data):
        nonlocal points
        if data[0].shape != (10000, 3):
            raise RuntimeError("Require exactly10000 original coordinates per update")
        points += 10000
    hook = model.register_forward_pre_hook(count_points)
    sequence = hashlib.sha256(); history = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    train_begin = time.monotonic()
    with (args.out_dir / "history.jsonl").open("w") as stream:
        for epoch in range(spec.updates):
            tick = time.monotonic()
            coords, indices = p.sample_coords(pool, sampler)
            sequence.update(indices.cpu().numpy().tobytes())
            if epoch == 0:
                first_indices = indices.detach().clone()  # Preserve the native training allocation path.
                first_hash = p.tensor_hash(indices)
            opt.zero_grad(set_to_none=True)
            total, data, reg = old.objective(model, coords, batch["fixed"], batch["moving"], spec.alpha)
            values = {"loss": float(total.detach()), "data_loss": float(data.detach()), "diffusion": float(reg.detach())}
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError("Nonfinite objective")
            total.backward()
            if any(x.grad is None for x in model.parameters()) or not bool(torch.stack([torch.isfinite(x.grad).all() for x in model.parameters()]).all()):
                raise FloatingPointError("Nonfinite/missing gradient")
            opt.step(); del total, data, reg, coords, indices
            torch.cuda.synchronize()
            row = {"loss_state_index": epoch, "updates_completed": epoch+1, **values,
                   "weighted_diffusion": spec.alpha * values["diffusion"], "lr": .0001, "sample_count": 10000,
                   "seconds": time.monotonic()-tick, "elapsed_seconds": time.monotonic()-train_begin}
            history.append(row); stream.write(json.dumps(row)+"\n")
            if epoch < 3 or (epoch+1) % 25 == 0 or epoch+1 == spec.updates:
                stream.flush(); old.write_json(args.out_dir / "progress.json", {**row, "target_updates": spec.updates, "gpu": args.gpu})
                print(json.dumps(row), flush=True)
            if (epoch+1) % 250 == 0 and epoch+1 != spec.updates:
                checkpoint(args.out_dir / "latest.pt", model, opt, spec, epoch+1, manifest["source_sha256"])
    hook.remove()
    training_seconds = time.monotonic()-train_begin
    if points != spec.updates*10000 or any(int(s["step"]) != spec.updates for s in opt.state.values()):
        raise RuntimeError("Changed training-coordinate/Adam budget")
    allocated, reserved = torch.cuda.max_memory_allocated()/1024**3, torch.cuda.max_memory_reserved()/1024**3
    if spec.sigma and p.tensor_hash(model.B0) != manifest["B0_sha256"]:
        raise RuntimeError("Frozen Fourier matrix changed")
    # Audit the saved initial state only AFTER optimization: even extra autograd
    # executions before native training can alter its numerical execution path.
    with torch.random.fork_rng(devices=[]):
        cpu_model = p.build_model(spec)
    cpu_model.load_state_dict(initial_model_state, strict=True)
    if old.state_hash(cpu_model) != initial_hash:
        raise RuntimeError("Initial diagnostic copy changed")
    cpu_pool = pool.cpu()
    coords, _ = p.sample_coords(cpu_pool, torch.Generator(device="cpu").manual_seed(1))
    manifest["initial_diagnostics"] = p.initial_diagnostics(cpu_model, coords, batch["fixed"].cpu(), batch["moving"].cpu(), spec.alpha)
    manifest["derivative_check"] = p.derivative_check(cpu_model, torch.device("cpu")) if spec.smoke_steps else None
    old.write_json(args.out_dir / "manifest.json", manifest)
    del coords, cpu_pool, cpu_model, initial_model_state
    checkpoint(args.out_dir / "terminal.pt", model, opt, spec, spec.updates, manifest["source_sha256"])
    old.verify_hashes(manifest["source_sha256"]); old.verify_hashes(inputs)
    field = old.dense_field(model, device, chunk=100000)
    field_path = args.out_dir / f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.cpu().numpy())
    restored = np.load(field_path)
    if restored.shape != (*c.IMAGE_SHAPE, 3) or not np.isfinite(restored).all() or not np.array_equal(restored, field.cpu().numpy()):
        raise RuntimeError("DVF roundtrip failed")
    saved = torch.load(args.out_dir / "terminal.pt", map_location="cpu", weights_only=False)
    other = p.build_model(p.FitSpec(**saved["spec"])).to(device)
    other.load_state_dict(saved["state_dict"], strict=True)
    if saved["updates"] != spec.updates or not torch.equal(old.dense_field(other, device, chunk=100000), field):
        raise RuntimeError("Checkpoint reconstruction failed")
    del model, other, opt, saved, field, pool, first_indices
    batch = {k:v.cpu() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    torch.cuda.empty_cache()
    dense = torch.from_numpy(np.ascontiguousarray(restored))
    metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(c.IMAGE_SHAPE, torch.device("cpu"))+dense, batch)
    labels = json_safe(label_rows(metrics, batch["fixed_seg"], batch["moving_seg"], meta))
    voxel = c.dhw3_norm_to_zyx_voxel_ncdhw(dense)
    summary = {**manifest, **{k:metrics[k] for k in p.METRICS[:3]},
               **c.jacobian_metrics_voxel(voxel, batch["fixed_mask"].bool().numpy()),
               "diffusion_l2_full": c.diffusion_loss_voxel(voxel), "actual_updates": spec.updates,
               "network_training_coordinates": points, "sample_sequence_sha256": sequence.hexdigest(),
               "first_sample_sha256": first_hash, "training_seconds": training_seconds,
               "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
               "valid_label_count": sum(r["included_in_case_mean"] for r in labels),
               "field_path": str(field_path), "field_sha256": old.sha256(field_path),
               "checkpoint_sha256": old.sha256(args.out_dir / "terminal.pt"), "run_dir": str(args.out_dir),
               "case_seconds": time.monotonic()-begin, "completed_unix": time.time()}
    if any(not math.isfinite(summary[k]) for k in p.METRICS):
        raise FloatingPointError("Nonfinite metric")
    old.validate_label_records(labels, summary)
    nib.save(nib.Nifti1Image(metrics["warped_seg"].numpy().astype(np.int16), np.eye(4)), str(args.out_dir / "warped_seg.nii.gz"))
    old.write_json(args.out_dir / "per_label.json", labels); c.write_csv(args.out_dir / "per_label.csv", labels)
    old.write_json(args.out_dir / "history.json", history)
    summary["outputs_sha256"] = {n:old.sha256(args.out_dir / n) for n in ("per_label.json", "per_label.csv", "history.json", "warped_seg.nii.gz")}
    old.write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").write_text("Fourier/control terminal field and all35label metrics validated\n")
    print(json.dumps({"complete": spec.name(), "seconds": summary["case_seconds"], "metrics": {k:summary[k] for k in p.METRICS}}), flush=True)


def main():
    args = parse_args(); existed = args.out_dir.exists()
    try:
        run(args)
    except Exception:
        error = traceback.format_exc()
        if not existed and args.out_dir.is_dir():
            (args.out_dir / "FAILED.txt").write_text(error)
        print(error, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
