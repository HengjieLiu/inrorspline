#!/usr/bin/env python
"""Prepare an audited frozen-feature cache or run one dense refinement."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import common as c
import run_one as prefix_runner
import vfa_feature_refinement as f
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, load_target, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "fit"), default="fit")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--dl-root", type=Path, required=True)
    parser.add_argument("--vfa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--phase", default="manual")
    parser.add_argument("--experiment-variant", default=f.VARIANT,
                        help="Artifact identity for explicitly approved follow-up sweeps")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--feature-order", dest="order", choices=f.ORDERS, default="warp_then_project")
    parser.add_argument("--feature-loss", dest="loss", choices=f.LOSSES, default="mse")
    parser.add_argument("--feature-support", dest="support", choices=("full", "fixed_positive"), default="full")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--check-interval", type=int, default=25)
    parser.add_argument("--median-window", type=int, default=25)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rel-total", type=float, default=1e-4)
    parser.add_argument("--rel-data", type=float, default=5e-5)
    parser.add_argument("--max-lr-reductions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    config_from_args(args).validate()
    if not 0 <= args.case_rank < 10 or args.cpu_threads < 1:
        parser.error("Require first10 rank and positive CPU thread count")
    return args


def config_from_args(args):
    return f.FitConfig(**{key: getattr(args, key) for key in f.FitConfig.__dataclass_fields__})


def provenance(args):
    paths = [Path(__file__), THIS_DIR / "vfa_feature_refinement.py", THIS_DIR / "common.py",
             THIS_DIR / "run_one.py", THIS_DIR / "launch_oasis_vfa_feature_dense.py",
             args.vfa_root / "vfa/models/vfa.py", args.vfa_root / "vfa/utils/utils.py"]
    return {"source_revision": args.source_revision,
            "source_hashes": {str(p): sha256_file(p) for p in paths if p.is_file()},
            "checkpoint_path": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
            "pair_file": str(args.pair_file), "pair_file_sha256": sha256_file(args.pair_file),
            "uid": os.getuid(), "gid": os.getgid(), "argv": sys.argv,
            "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
            "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded"),
            "requested_cudnn_benchmark": False, "requested_cudnn_deterministic": True,
            "cuda_nondeterminism_caveat": "cuDNN deterministic does not guarantee bitwise CUDA grid_sample backward",
            "initial_gpu_state": gpu_snapshot()}


def canonical_metrics(field_voxel, batch):
    scale = f.voxel_scale(field_voxel)
    identity = c.make_coordinate_grid(c.IMAGE_SHAPE, field_voxel.device)
    with torch.no_grad():
        metrics = c.evaluate_from_dense_coords(identity + field_voxel / scale, batch)
    row = c.accuracy_row(metrics, str(batch["fixed_id"]), str(batch["moving_id"]))
    short = {"dice": float(row["dice_mean"]), "hd95": float(row["hd95_mean"]),
             "asd": float(row["asd_symmetric_mean"])}
    if not all(math.isfinite(v) for v in short.values()):
        raise FloatingPointError("Nonfinite canonical metrics")
    # Preserve and verify canonical exclusion of any label absent in fixed,
    # original moving, or warped moving segmentation.
    for label in c.OASIS_LABELS:
        absent = any(not bool(torch.any(seg == label)) for seg in
                     (batch["fixed_seg"], batch["moving_seg"], metrics["warped_seg"]))
        if absent:
            for key in ("dice", "hd95", "asd_symmetric"):
                if not math.isnan(float(row[f"{key}_label_{label:02d}"])):
                    raise RuntimeError(f"Incorrect absent-label handling: {key}/{label}")
        else:
            for key in ("dice", "hd95", "asd_symmetric"):
                if not math.isfinite(float(row[f"{key}_label_{label:02d}"])):
                    raise RuntimeError(f"Nonfinite metric for present label: {key}/{label}")
    return short, row


def prepare(args, batch, device, manifest):
    started = time.monotonic()
    output = args.out_dir
    prefix_args = f.prefix_args()
    prefix_args.case_rank = args.case_rank
    prefix_args.source_revision = args.source_revision
    prefix = prefix_runner.train_multistage_direct_cp(prefix_args, batch, device, lr=0.01)
    initial_norm = prefix["dense_disp"].detach()
    initial_vox = initial_norm * f.voxel_scale(initial_norm)
    # Compare in the parameter's declared FP32 voxel units. A redundant
    # voxel->normalized->voxel round trip has its own quantization, and is not
    # the initialization equality that this experiment requires.
    cloned_initial = initial_vox.detach().clone()
    init_difference = (cloned_initial - initial_vox).abs().max().item()
    roundtrip_difference = ((initial_vox / f.voxel_scale(initial_vox) - initial_norm)
                           * f.voxel_scale(initial_vox)).abs().max().item()
    if init_difference > 1e-6:
        raise RuntimeError(f"Prefix-to-dense initialization exceeds tolerance: {init_difference}")
    if args.case_rank == 0 and prefix["stage_actual_epochs"] != [125, 150, 175]:
        raise RuntimeError("Rank-0 frozen prefix stopping epochs changed")
    prefix_metrics, prefix_row = canonical_metrics(initial_vox, batch)
    c.write_csv(output / "prefix_case_metrics.csv", [prefix_row])
    c.write_csv(output / "prefix_history.csv", prefix["stage_history"])
    c.write_json(output / "prefix_stages.json", prefix["stage_summaries"])
    torch.save({"control_disp": prefix["control_disp"].cpu(), "control_shape": prefix["control_shape"],
                "cps": 2, "stage_summaries": prefix["stage_summaries"]}, output / "prefix_model_state.pt")
    np.save(output / "prefix_norm.npy", initial_norm.cpu().numpy())
    prefix_elapsed = prefix["training_elapsed_sec"]
    del prefix
    torch.cuda.empty_cache()

    target_args = argparse.Namespace(target="vfa", dl_root=args.dl_root)
    target_norm, target_audit = load_target(target_args, fixed_id=str(batch["fixed_id"]),
                                           moving_id=str(batch["moving_id"]), device=device)
    target_vox = target_norm * f.voxel_scale(target_norm)
    reference, target_row = canonical_metrics(target_vox, batch)
    c.write_csv(output / "vfa_case_metrics.csv", [target_row])
    if args.case_rank == 0 and any(abs(reference[k] - f.REFERENCE[k]) > 1e-6 for k in reference):
        raise RuntimeError(f"Saved rank-0 reference metric mismatch: {reference}")
    encoder, decoder = f.load_vfa(args.checkpoint, args.vfa_root, device)
    raw = {}
    for role in ("fixed", "moving"):
        array = c.load_nifti_reoriented(args.oasis_root / "test" / f"img{batch[role + '_id']}.nii.gz", dtype=np.float32)
        raw[role] = torch.from_numpy(array).to(device)[None, None]
    # The historical VFA loader applies orientation + float32 only, not the
    # pair-minmax scaling used by the spline prefix.
    with torch.no_grad():
        fixed_features = encoder(raw["fixed"])
        moving_features = encoder(raw["moving"])
        if tuple(fixed_features[0].shape) != (1, 16, *c.IMAGE_SHAPE):
            raise RuntimeError("Incorrect native full-resolution feature shape")
        from vfa.utils.utils import identity_grid_like
        native_grid = decoder(fixed_features, moving_features)[-1]
        native_zyx = native_grid - identity_grid_like(native_grid, normalize=False)
        native_xyz = native_zyx[0].flip(0).permute(1, 2, 3, 0).contiguous()
        error = native_xyz - target_vox
        native_metrics, native_row = canonical_metrics(native_xyz, batch)
        parity = {"voxel_rmse": error.square().mean().sqrt().item(),
                  "voxel_max_abs": error.abs().max().item(),
                  "metric_deltas": {k: native_metrics[k] - reference[k] for k in reference},
                  "limits": {"voxel_rmse": 0.001, "dice": 0.0001, "hd95": 0.01, "asd": 0.001}}
        parity["passed"] = parity["voxel_rmse"] <= 0.001 and all(
            abs(parity["metric_deltas"][k]) <= parity["limits"][k] for k in reference)
        c.write_json(output / "vfa_inference_parity.json", parity)
        c.write_csv(output / "fresh_vfa_case_metrics.csv", [native_row])
        if not parity["passed"]:
            raise RuntimeError(f"VFA inference parity failed; no fitting authorized: {parity}")
        cache = {"prefix_voxel_xyz": initial_vox.cpu(), "vfa_voxel_xyz": target_vox.cpu(),
                 "fixed_projected": decoder.project[0](fixed_features[0]).cpu(),
                 "moving_projected": f.moving_project(decoder.project[0], moving_features[0]).cpu(),
                 "moving_encoder": moving_features[0].cpu(),
                 "projection_state": {k: v.cpu() for k, v in decoder.project[0].state_dict().items()},
                 "fixed_image": batch["fixed"].cpu(), "moving_image": batch["moving"].cpu(),
                 "fixed_mask": batch["fixed_mask"].cpu(),
                 "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
                 "case_rank": args.case_rank, "checkpoint_sha256": manifest["checkpoint_sha256"],
                 "prefix_conversion_max_abs_voxel": init_difference,
                 "normalized_roundtrip_max_abs_voxel": roundtrip_difference}
    torch.save(cache, output / "cache.pt")
    summary = {**manifest, "variant": args.experiment_variant, "mode": "prepare", "case_rank": args.case_rank,
               "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
               "original_case_index": batch["original_case_index"], "direction": "moving_to_fixed",
               "prefix_metrics": prefix_metrics, "vfa_metrics": reference,
               "prefix_elapsed_sec": prefix_elapsed, "prefix_conversion_max_abs_voxel": init_difference,
               "normalized_roundtrip_max_abs_voxel": roundtrip_difference,
               "prefix_norm_sha256": sha256_file(output / "prefix_norm.npy"),
               "cache_sha256": sha256_file(output / "cache.pt"), "target_audit": target_audit,
               "vfa_inference_parity": parity, "preprocessing": "native RAS raw float32 for VFA; pair-minmax for prefix",
               "cuda_backend": c.cuda_backend_metadata(device), "elapsed_sec": time.monotonic() - started}
    c.write_json(output / "summary.json", summary)
    (output / "DONE").write_text(c.now_text() + "\n")
    print(json.dumps({"prepared": str(output), "parity": parity}), flush=True)


def load_cache(args, batch, manifest):
    root = args.cache_dir
    if not (root / "DONE").exists():
        raise RuntimeError("Prefix/features cache is incomplete")
    summary = c.read_json(root / "summary.json")
    if not summary["vfa_inference_parity"]["passed"]:
        raise RuntimeError("Cache has not passed inference parity")
    if summary["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed since preparation")
    if summary["pair_file_sha256"] != manifest["pair_file_sha256"]:
        raise RuntimeError("Pair file changed since preparation")
    if sha256_file(root / "cache.pt") != summary["cache_sha256"]:
        raise RuntimeError("Feature cache hash mismatch")
    cache = torch.load(root / "cache.pt", map_location="cpu", weights_only=False)
    for key in ("fixed_id", "moving_id", "case_rank"):
        if cache[key] != batch[key]:
            raise RuntimeError(f"Cache case mismatch: {key}")
    for key in ("prefix_voxel_xyz", "vfa_voxel_xyz"):
        if tuple(cache[key].shape) != (*c.IMAGE_SHAPE, 3) or not torch.isfinite(cache[key]).all():
            raise RuntimeError(f"Invalid cache field: {key}")
    return cache, summary


def fit(args, batch, device, manifest):
    started = time.monotonic()
    config = config_from_args(args)
    cache, prepared = load_cache(args, batch, manifest)
    objective = f.FeatureObjective(cache, config, device)
    field, diagnostics = f.optimize(objective, config, args.out_dir)
    # Canonical metrics consume the serialized normalized DVF, not a separate
    # in-memory result, so the report describes exactly the deliverable.
    field_path = args.out_dir / f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path, (field / f.voxel_scale(field)).detach().cpu().numpy())
    reloaded = torch.from_numpy(np.load(field_path)).to(device)
    if tuple(reloaded.shape) != (*c.IMAGE_SHAPE, 3) or not torch.isfinite(reloaded).all():
        raise RuntimeError("Serialized DVF validation failed")
    reloaded_vox = reloaded * f.voxel_scale(reloaded)
    metrics, row = canonical_metrics(reloaded_vox, batch)
    c.write_csv(args.out_dir / "case_metrics.csv", [row])
    voxel_zyx = reloaded_vox.detach().cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    regularity = c.jacobian_metrics_voxel(voxel_zyx, batch["fixed_mask"].cpu().numpy().astype(bool))
    diagnostic_fields = {}
    with torch.no_grad():
        for name, value in (("prefix", objective.initial), ("vfa", cache["vfa_voxel_xyz"].to(device)),
                            ("selected", reloaded_vox)):
            total, raw, reg = objective(value)
            diagnostic_fields[name] = {"loss": float(total), "raw_feature_loss": float(raw),
                                      "data_loss": float(raw) / objective.scale, "diffusion": float(reg)}
    change = (reloaded_vox - objective.initial).norm(dim=-1)
    summary = {**manifest, **metrics, **diagnostics, "config": asdict(config),
               "run_name": f.config_name(config, args.case_rank), "run_dir": str(args.out_dir),
               "phase": args.phase, "variant": args.experiment_variant, "method": f.METHOD,
               "case_rank": args.case_rank, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"],
               "original_case_index": batch["original_case_index"], "direction": "moving_to_fixed",
               "representation": "dense_voxel", "parameter_units": "xyz_voxel_1mm",
               "dense_shape": list(field.shape), "parameter_count": field.numel(), "cps": None,
               "control_shape": None, "bspline_order": None, "uses_cps1": False,
               "regularizer": "unmasked_total_voxel_field_forward_valid_l2_diffusion_component_mean",
               "image_loss_in_optimization": config.loss == "raw_ncc", "segmentation_loss": False,
               "reference_dvf_supervision": False, "objective_support_voxels": int(
                   cache["fixed_mask"].sum()) if config.support == "fixed_positive" else int(np.prod(c.IMAGE_SHAPE)),
               "prefix_cache": str(args.cache_dir), "prefix_sha256": prepared["prefix_norm_sha256"],
               "cache_sha256": prepared["cache_sha256"], "vfa_reference": prepared["vfa_metrics"],
               "deltas_vfa": {k: metrics[k] - prepared["vfa_metrics"][k] for k in metrics},
               "metric_deficit": f.metric_deficit(metrics, prepared["vfa_metrics"]),
               "matches_vfa": f.metric_deficit(metrics, prepared["vfa_metrics"]) <= 1,
               "regularity": regularity, "field_objectives": diagnostic_fields,
               "change_from_prefix_mean_mm": float(change.mean()),
               "change_from_prefix_max_mm": float(change.max()),
               "dvf_path": str(field_path), "dvf_sha256": sha256_file(field_path),
               "cuda_backend": c.cuda_backend_metadata(device), "gpu": args.gpu,
               "total_elapsed_sec": time.monotonic() - started, "completed_at": c.now_text()}
    c.write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").write_text(c.now_text() + "\n")
    print(json.dumps({"complete": str(args.out_dir), **metrics, "matches_vfa": summary["matches_vfa"]}), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing attempt: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    manifest = provenance(args)
    c.write_json(args.out_dir / "launch_manifest.json", {**manifest, "args": vars(args), "started_at": c.now_text()})
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch = c.load_oasis_case(rank=args.case_rank, device=device, oasis_root=args.oasis_root, pair_file=args.pair_file)
    if args.case_rank == 0 and (batch["fixed_id"], batch["moving_id"], batch["original_case_index"]) != ("0333", "0115", 158):
        raise RuntimeError("Rank-0 identity changed")
    try:
        if args.mode == "prepare":
            prepare(args, batch, device, manifest)
        else:
            fit(args, batch, device, manifest)
    except Exception as exc:
        c.write_json(args.out_dir / "FAILED.json", {"error": f"{type(exc).__name__}: {exc}", "time": c.now_text()})
        raise


if __name__ == "__main__":
    main()
