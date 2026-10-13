"""One original-sampling IDIR fit; dense evaluation only after training."""
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
import idir_sampled_diffusion as p
from run_oasis_sitreg_first100_eval import label_rows, json_safe
from sitreg_training import gpu_rows, DATA_SHA


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--exponent", type=int, required=True)
    parser.add_argument("--omega", type=int, choices=p.OMEGAS, default=32)
    parser.add_argument("--gpu", type=int, choices=(1, 3), required=True)
    parser.add_argument("--slot", type=int, choices=range(4), required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_from_args(args).validate()
    if p.SLOTS[args.slot] != args.gpu or args.cpu_threads < 1:
        parser.error("Slot/GPU mismatch or invalid thread count")
    for source in (c.REPO_ROOT, args.oasis_root):
        if args.out_dir.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.out_dir.resolve()):
            parser.error("Outputs must not overlap sources/data")
    return args


def spec_from_args(args):
    return p.FitSpec(args.rank, args.exponent, args.omega, args.smoke_steps)


def configure(gpu_index, cpu_threads):
    rows = gpu_rows()
    gpu = next(r for r in rows if r["index"] == gpu_index)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(0)
    # Bound each of two co-resident workers to12GiB; do not reserve the whole GPU.
    fraction = min(12 * 1024, gpu["free_mib"] - 2048) / gpu["total_mib"]
    if fraction < .08:
        raise RuntimeError("Insufficient free memory for sampled IDIR worker")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda:0"), rows, fraction


def load_batch(args, device):
    if p.sha256(args.oasis_root / "dataset.json") != DATA_SHA:
        raise RuntimeError("Unexpected dataset manifest")
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    if len(pairs) != 200 or pairs[0] != ("0333", "0115"):
        raise RuntimeError("Wrong seed3 pair list")
    batch = c.load_oasis_case(rank=args.rank, device=device, oasis_root=args.oasis_root)
    if (batch["fixed_id"], batch["moving_id"]) != pairs[args.rank]:
        raise RuntimeError("Wrong case identity")
    hashes = {}
    for role in ("fixed", "moving"):
        for kind in ("img", "seg"):
            path = args.oasis_root / "test" / f"{kind}{batch[role+'_id']}.nii.gz"
            ni = nib.load(str(path))
            if ni.shape != (160, 192, 224) or tuple(ni.header.get_zooms()[:3]) != (1., 1., 1.):
                raise RuntimeError("Unexpected input shape/spacing")
            hashes[str(path)] = p.sha256(path)
        if not bool(torch.isfinite(batch[role]).all()):
            raise RuntimeError("Nonfinite image")
    return batch, hashes


def checkpoint(path, model, optimizer, spec, updates, manifest):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": p.cpu_state(model), "optimizer": optimizer.state_dict(),
                "omega": spec.omega, "layers": p.PROTOCOL["layers"], "updates": updates,
                "spec": asdict(spec), "protocol": p.PROTOCOL, "source_sha256": manifest["source_sha256"]}, temporary)
    temporary.replace(path)


def run(args):
    begin = time.monotonic()
    spec = spec_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    device, gpu_state, fraction = configure(args.gpu, args.cpu_threads)
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    batch, inputs = load_batch(args, device)
    model = c.build_siren(256, 3, spec.omega).to(device)
    pool = p.training_coords(batch, device)
    opt = p.optimizer(model)
    manifest = {"variant": p.VARIANT, "method": p.METHOD, "spec": asdict(spec), "protocol": p.PROTOCOL,
                "rank": spec.rank, "omega": spec.omega, "exponent": spec.exponent, "alpha": spec.alpha,
                "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"], "direction": "moving_to_fixed",
                "original_case_index": batch["original_case_index"], "source_sha256": p.source_hashes(),
                "input_sha256": inputs, "pair_file_sha256": p.sha256(c.DEFAULT_PAIR_FILE),
                "dataset_manifest_sha256": p.sha256(args.oasis_root / "dataset.json"),
                "initial_state_sha256": p.state_hash(model), "train_coords": len(pool), "param_count": c.count_trainable_params(model),
                "physical_gpu": args.gpu, "slot": args.slot, "gpu_before": gpu_state, "visible_cuda_device": "cuda:0",
                "gpu_uuid": gpu_state[args.gpu]["uuid"], "allocator_fraction": fraction,
                "existing_gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True),
                "backend": c.cuda_backend_metadata(device), "tf32_matmul": False, "tf32_cudnn": False,
                "historical_backend": "not_recorded_in_old_checkpoint; current_repo_False_True_default_explicit",
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid(), "argv": sys.argv,
                "repo_revision": subprocess.check_output(["git", "-C", str(c.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID"), "started_unix": time.time()}
    if manifest["param_count"] != 133379:
        raise RuntimeError("Wrong architecture")
    p.write_json(args.out_dir / "manifest.json", manifest)
    evaluated_points = 0

    def count_points(_module, data):
        nonlocal evaluated_points
        if data[0].shape != (10000, 3):
            raise RuntimeError("Non10000-coordinate forward during training")
        evaluated_points += len(data[0])

    hook = model.register_forward_pre_hook(count_points)
    sequence_hash = hashlib.sha256()
    history = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    train_begin = time.monotonic()
    with (args.out_dir / "history.jsonl").open("w") as stream:
        for epoch in range(spec.updates):
            tick = time.monotonic()
            coords, indices = p.sample_coords(pool)
            sequence_hash.update(indices.detach().cpu().numpy().tobytes())
            if epoch == 0:
                first_indices = indices.detach().clone()
                first_sample_hash = hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest()
            opt.zero_grad(set_to_none=True)
            total, data, reg = p.objective(model, coords, batch["fixed"], batch["moving"], spec.alpha)
            values = {"loss": float(total.detach()), "data_loss": float(data.detach()), "diffusion": float(reg.detach())}
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError("Nonfinite sampled objective")
            total.backward()
            if any(param.grad is None for param in model.parameters()) or not bool(torch.stack([torch.isfinite(param.grad).all() for param in model.parameters()]).all()):
                raise FloatingPointError("Missing/nonfinite parameter gradient")
            opt.step()
            del total, data, reg, coords, indices
            torch.cuda.synchronize()
            row = {"loss_state_index": epoch, "updates_completed": epoch + 1, **values,
                   "weighted_diffusion": spec.alpha * values["diffusion"], "lr": .0001,
                   "sample_count": 10000, "seconds": time.monotonic() - tick,
                   "elapsed_seconds": time.monotonic() - train_begin}
            history.append(row); stream.write(json.dumps(row) + "\n")
            if epoch < 3 or (epoch + 1) % 25 == 0 or epoch + 1 == spec.updates:
                stream.flush()
                p.write_json(args.out_dir / "progress.json", {**row, "target_updates": spec.updates, "physical_gpu": args.gpu, "slot": args.slot})
                print(json.dumps(row), flush=True)
            if (epoch + 1) % 250 == 0 and epoch + 1 != spec.updates:
                checkpoint(args.out_dir / "latest.pt", model, opt, spec, epoch + 1, manifest)
    hook.remove()
    training_seconds = time.monotonic() - train_begin
    if evaluated_points != spec.updates * 10000 or any(int(state["step"]) != spec.updates for state in opt.state.values()):
        raise RuntimeError("Network evaluation or Adam step count mismatch")
    allocated, reserved = torch.cuda.max_memory_allocated()/1024**3, torch.cuda.max_memory_reserved()/1024**3
    # This diagnostic is a single extra sampled evaluation, not a training update.
    diagnostic_coords = pool[first_indices].detach().requires_grad_(True)
    values = p.objective(model, diagnostic_coords, batch["fixed"], batch["moving"], spec.alpha)
    terminal_diag = dict(zip(("loss", "data_loss", "diffusion"), (float(x.detach()) for x in values)))
    del values, diagnostic_coords, pool, first_indices
    checkpoint(args.out_dir / "terminal.pt", model, opt, spec, spec.updates, manifest)
    p.verify_hashes(manifest["source_sha256"]); p.verify_hashes(inputs)
    field = p.dense_field(model, device)
    field_path = args.out_dir / f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.cpu().numpy())
    restored = np.load(field_path)
    if restored.shape != (*c.IMAGE_SHAPE, 3) or not np.isfinite(restored).all() or not np.array_equal(restored, field.cpu().numpy()):
        raise RuntimeError("DVF roundtrip failed")
    saved = torch.load(args.out_dir / "terminal.pt", map_location="cpu", weights_only=False)
    other = c.build_siren(256, 3, saved["omega"]).to(device)
    other.load_state_dict(saved["state_dict"], strict=True)
    if saved["updates"] != spec.updates or not torch.equal(p.dense_field(other, device), field):
        raise RuntimeError("Checkpoint reconstruction failed")
    del model, opt, other, saved, field
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    torch.cuda.empty_cache()
    dense = torch.from_numpy(np.ascontiguousarray(restored))
    coords = c.make_coordinate_grid(c.IMAGE_SHAPE, torch.device("cpu")) + dense
    metrics = c.evaluate_from_dense_coords(coords, batch)
    meta = {k: manifest[k] for k in ("rank", "omega", "exponent", "alpha", "fixed_id", "moving_id", "direction")}
    details = json_safe(label_rows(metrics, batch["fixed_seg"], batch["moving_seg"], meta))
    voxel_zyx = c.dhw3_norm_to_zyx_voxel_ncdhw(dense)
    regularity = c.jacobian_metrics_voxel(voxel_zyx, batch["fixed_mask"].bool().numpy())
    summary = {**manifest, **{k: metrics[k] for k in p.METRICS[:3]}, **regularity,
               "diffusion_l2_full": c.diffusion_loss_voxel(voxel_zyx), "terminal_diagnostic_first_sample": terminal_diag,
               "actual_updates": spec.updates, "network_training_coordinates": evaluated_points,
               "first_sample_sha256": first_sample_hash, "sample_sequence_sha256": sequence_hash.hexdigest(),
               "training_seconds": training_seconds, "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
               "valid_label_count": sum(r["included_in_case_mean"] for r in details),
               "field_path": str(field_path), "field_sha256": p.sha256(field_path),
               "checkpoint_sha256": p.sha256(args.out_dir / "terminal.pt"), "run_dir": str(args.out_dir),
               "case_seconds": time.monotonic() - begin, "completed_unix": time.time()}
    if any(not math.isfinite(float(summary[k])) for k in p.METRICS):
        raise FloatingPointError("Nonfinite final metrics")
    p.validate_label_records(details, summary)
    nib.save(nib.Nifti1Image(metrics["warped_seg"].numpy().astype(np.int16), np.eye(4)), str(args.out_dir / "warped_seg.nii.gz"))
    p.write_json(args.out_dir / "per_label.json", details); c.write_csv(args.out_dir / "per_label.csv", details)
    c.write_csv(args.out_dir / "case_metrics.csv", [summary]); p.write_json(args.out_dir / "history.json", history)
    summary["outputs_sha256"] = {name: p.sha256(args.out_dir / name) for name in
                                 ("per_label.json", "per_label.csv", "history.json", "warped_seg.nii.gz", "case_metrics.csv")}
    p.write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").write_text("Terminal sampled-IDIR field and35label records validated\n")
    print(json.dumps({"complete": spec.name(), "metrics": {k: summary[k] for k in p.METRICS}, "seconds": summary["case_seconds"]}), flush=True)


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
