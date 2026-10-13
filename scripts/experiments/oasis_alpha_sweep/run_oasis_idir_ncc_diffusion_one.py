"""One full-resolution IDIR SIREN NCC9/voxel-diffusion registration."""
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

import numpy as np
import nibabel as nib
import torch

import common as c
import idir_ncc_diffusion as p
from run_oasis_dl_ncc_diffusion_one import load_batch, native_parity
from run_oasis_sitreg_first100_eval import label_rows, json_safe
from sitreg_training import gpu_rows, DATA_SHA


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-rank", type=int, required=True)
    parser.add_argument("--omega", type=int, choices=p.OMEGAS, required=True)
    parser.add_argument("--lambda-key", choices=p.LAMBDAS, required=True)
    parser.add_argument("--gpu", type=int, choices=(1, 3), required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    for name in ("out-dir", "oasis-root", "dl-code-root", "vfa-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    spec_from_args(args).validate()
    output = args.out_dir.resolve()
    for source in (c.REPO_ROOT, args.oasis_root, args.dl_code_root, args.vfa_root):
        if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
            parser.error("Output must not overlap sources/data")
    if args.cpu_threads < 1:
        parser.error("Require positive thread count")
    return args


def spec_from_args(args):
    return p.FitSpec(args.case_rank, args.omega, args.lambda_key, args.smoke_steps)


def save_checkpoint(path, model, optimizer, spec, state_index, losses, manifest):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": p.cpu_state(model), "optimizer": optimizer.state_dict(),
                "omega": spec.omega, "layers": p.PROTOCOL["layers"], "spec": asdict(spec),
                "state_index": state_index, "losses": losses, "protocol": p.PROTOCOL,
                "source_sha256": manifest["source_sha256"], "initial_state_sha256": manifest["initial_state_sha256"]}, temporary)
    temporary.replace(path)


def run(args):
    begin = time.monotonic()
    spec = spec_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    state = gpu_rows()
    gpu = next(row for row in state if row["index"] == args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(0)
    fraction = min(.95, (gpu["free_mib"] - 2048) / gpu["total_mib"])
    if fraction < .15:
        raise RuntimeError("Insufficient current free memory")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    pairs = c.parse_pair_file(args.pair_file)
    if len(pairs) != 200 or pairs[0] != ("0333", "0115") or p.sha256(args.pair_file) != p.sha256(c.DEFAULT_PAIR_FILE):
        raise RuntimeError("Wrong canonical seed3 case list")
    parity = native_parity(device, args.dl_code_root, args.vfa_root) if spec.smoke_steps else None
    chunk_parity = p.cuda_chunk_parity(spec.omega, device) if spec.smoke_steps else None
    if p.sha256(args.oasis_root / "dataset.json") != DATA_SHA:
        raise RuntimeError("Dataset manifest changed")
    # Reseed AFTER any parity diagnostic to preserve identical initial states.
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    batch, input_hashes = load_batch(args, device)
    if (batch["fixed_id"], batch["moving_id"]) != pairs[spec.rank]:
        raise RuntimeError("Wrong fixed/moving identity")
    model = c.build_siren(256, 3, spec.omega).to(device)
    if c.count_trainable_params(model) != 133379:
        raise RuntimeError("Unexpected SIREN architecture")
    objective = p.FullVolumeObjective(batch["fixed"], batch["moving"], spec.alpha)
    opt = p.optimizer(model)
    manifest = {"variant": p.VARIANT, "method": p.METHOD, "spec": asdict(spec), "protocol": p.PROTOCOL,
                "rank": spec.rank, "omega": spec.omega, "lambda_key": spec.lambda_key, "alpha": spec.alpha,
                "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"], "direction": "moving_to_fixed",
                "original_case_index": batch["original_case_index"], "input_sha256": input_hashes,
                "pair_file": str(args.pair_file), "pair_file_sha256": p.sha256(args.pair_file),
                "dataset_manifest_sha256": p.sha256(args.oasis_root / "dataset.json"),
                "source_sha256": p.source_hashes(),
                "native_source_sha256": {str(path): p.sha256(path) for path in
                    (args.dl_code_root / "utils_losses_ncc.py", args.vfa_root / "vfa/losses/grad_loss.py")},
                "repo_revision": subprocess.check_output(["git", "-C", str(c.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "repo_dirty_status": subprocess.check_output(["git", "-C", str(c.REPO_ROOT), "status", "--short"], text=True),
                "physical_gpu": args.gpu, "gpu_uuid": gpu["uuid"], "visible_cuda_device": "cuda:0",
                "gpu_before": state, "allocator_fraction": fraction,
                "existing_gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True),
                "backend": c.cuda_backend_metadata(device), "tf32_matmul": False, "tf32_cudnn": False,
                "cuda_nondeterminism": "grid_sample CUDA backward may remain nondeterministic",
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid(), "argv": sys.argv,
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID"),
                "initial_state_sha256": p.state_hash(model), "param_count": 133379,
                "native_parity": parity, "chunk_parity": chunk_parity, "started_unix": time.time()}
    p.write_json(args.out_dir / "manifest.json", manifest)
    history, best_loss, best_epoch, best_state = [], math.inf, None, None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    train_begin = time.monotonic()
    with (args.out_dir / "history.jsonl").open("w") as stream:
        for state_index in range(spec.updates + 1):
            tick = time.monotonic()
            if state_index < spec.updates:
                losses = p.full_volume_gradients(model, objective)
            else:
                with torch.no_grad():
                    q = p.dense_field(model, objective)
                    values = objective(q)
                    losses = dict(zip(("loss", "ncc9", "diffusion"), (float(v) for v in values)))
                    del q, values
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError("Nonfinite checkpoint loss")
            if losses["loss"] < best_loss:
                best_loss, best_epoch, best_state = losses["loss"], state_index, p.cpu_state(model)
            # losses belong to state_index completed Adam updates, NOT its successor.
            if state_index < spec.updates:
                opt.step()
            torch.cuda.synchronize()
            entry = {"state_index": state_index, **losses, "lr": p.PROTOCOL["adam_lr"],
                     "updates_completed": min(state_index + 1, spec.updates),
                     "seconds": time.monotonic() - tick, "elapsed_seconds": time.monotonic() - train_begin}
            history.append(entry)
            stream.write(json.dumps(entry) + "\n"); stream.flush()
            if state_index < 3 or (state_index + 1) % 25 == 0 or state_index == spec.updates:
                p.write_json(args.out_dir / "progress.json", {**entry, "best_epoch": best_epoch, "best_loss": best_loss,
                             "target_updates": spec.updates, "physical_gpu": args.gpu})
                print(json.dumps(entry), flush=True)
            if (state_index + 1) % 250 == 0 and state_index < spec.updates:
                # Latest optimizer/model are after the just-completed update; loss evaluated next iteration.
                save_checkpoint(args.out_dir / "latest.pt", model, opt, spec, state_index + 1, None, manifest)
        training_seconds = time.monotonic() - train_begin
    allocated, reserved = torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.max_memory_reserved() / 1024**3
    p.verify_hashes(manifest["source_sha256"]); p.verify_hashes(manifest["native_source_sha256"])
    p.verify_hashes({str(args.oasis_root / "test" / key): value for key, value in input_hashes.items()})
    save_checkpoint(args.out_dir / "terminal.pt", model, opt, spec, spec.updates, losses, manifest)
    torch.save({"state_dict": best_state, "omega": spec.omega, "layers": p.PROTOCOL["layers"],
                "state_index": best_epoch, "loss": best_loss, "selection": "diagnostic_only_minimum_total_loss",
                "spec": asdict(spec), "protocol": p.PROTOCOL}, args.out_dir / "best_loss_diagnostic.pt")
    with torch.no_grad():
        field = p.dense_field(model, objective)
    field_path = args.out_dir / f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, field.cpu().numpy())
    restored = np.load(field_path)
    if restored.shape != (*c.IMAGE_SHAPE, 3) or not np.isfinite(restored).all() or not np.array_equal(restored, field.cpu().numpy()):
        raise RuntimeError("Invalid DVF serialization")
    restored_model = c.build_siren(256, 3, spec.omega).to(device)
    saved = torch.load(args.out_dir / "terminal.pt", map_location="cpu", weights_only=False)
    restored_model.load_state_dict(saved["state_dict"], strict=True)
    with torch.no_grad():
        roundtrip = p.dense_field(restored_model, objective)
    if not torch.equal(roundtrip, field) or saved["omega"] != spec.omega or saved["state_index"] != spec.updates:
        raise RuntimeError("Model/checkpoint roundtrip mismatch")
    # Metrics use the serialized terminal field, never best-loss/test-Dice epoch selection.
    del restored_model, roundtrip, field, model, opt, objective, saved, best_state
    batch = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    torch.cuda.empty_cache()
    dense = torch.from_numpy(np.ascontiguousarray(restored))
    coordinates = c.make_coordinate_grid(c.IMAGE_SHAPE, torch.device("cpu")) + dense
    metrics = c.evaluate_from_dense_coords(coordinates, batch)
    meta = {k: manifest[k] for k in ("rank", "omega", "lambda_key", "alpha", "fixed_id", "moving_id", "direction")}
    details = json_safe(label_rows(metrics, batch["fixed_seg"], batch["moving_seg"], meta))
    voxel_zyx = c.dhw3_norm_to_zyx_voxel_ncdhw(dense)
    regularity = c.jacobian_metrics_voxel(voxel_zyx, batch["fixed_mask"].bool().numpy())
    row = {**meta, **{key: float(metrics[key]) for key in p.METRICS[:3]}, **regularity,
           "diffusion_l2_full": c.diffusion_loss_voxel(voxel_zyx), "valid_label_count": sum(r["included_in_case_mean"] for r in details),
           "training_seconds": training_seconds, "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
           "physical_gpu": args.gpu, "actual_updates": spec.updates, "checkpoint_selection": "terminal",
           "best_loss_epoch_diagnostic": best_epoch, "best_loss_diagnostic": best_loss,
           "terminal_loss": losses["loss"], "terminal_ncc9": losses["ncc9"], "terminal_diffusion": losses["diffusion"],
           "field_path": str(field_path), "field_sha256": p.sha256(field_path),
           "checkpoint_path": str(args.out_dir / "terminal.pt"), "checkpoint_sha256": p.sha256(args.out_dir / "terminal.pt"),
           "run_dir": str(args.out_dir), "case_seconds": time.monotonic() - begin}
    if not all(math.isfinite(row[key]) for key in (*p.METRICS, "diffusion_l2_full")):
        raise FloatingPointError("Nonfinite final metric")
    p.validate_label_records(details, row)
    nib.save(nib.Nifti1Image(metrics["warped_seg"].numpy().astype(np.int16), np.eye(4)), str(args.out_dir / "warped_seg.nii.gz"))
    p.write_json(args.out_dir / "per_label.json", details)
    c.write_csv(args.out_dir / "per_label.csv", details)
    c.write_csv(args.out_dir / "case_metrics.csv", [row])
    p.write_json(args.out_dir / "history.json", history)
    p.write_json(args.out_dir / "summary.json", {**manifest, **row, "completed_unix": time.time(),
                 "per_label_sha256": p.sha256(args.out_dir / "per_label.json"),
                 "warped_seg_sha256": p.sha256(args.out_dir / "warped_seg.nii.gz"),
                 "history_sha256": p.sha256(args.out_dir / "history.json"), "completion": "success"})
    (args.out_dir / "DONE").write_text("terminal field and all35 label records validated\n")
    print(json.dumps({"completed": row}), flush=True)


def main():
    args = parse_args()
    existed = args.out_dir.exists()
    try:
        run(args)
    except Exception:
        error = traceback.format_exc()
        # Never mutate an existing completed run, including accidental duplicate invocation.
        if not existed and args.out_dir.is_dir() and not (args.out_dir / "DONE").exists() and not (args.out_dir / "FAILED.txt").exists():
            (args.out_dir / "FAILED.txt").write_text(error)
        print(error, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
