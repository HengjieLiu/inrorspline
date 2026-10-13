"""Read-only common-field similarity-gradient calibration; never rescales training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import common as c
import gncc_dense as d
import dl_ncc_diffusion as geometry
import run_oasis_dl_ncc_diffusion_one as previous
from run_oasis_saved_field_bspline_fit_one import sha256_file, gpu_snapshot


def run(args):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sources = d.source_hashes()
    candidates = c.read_json(args.ncc_root/"results.json")["candidates"]
    identity = c.make_coordinate_grid(c.IMAGE_SHAPE, device)
    scale = geometry.voxel_scale(c.IMAGE_SHAPE, device=device)
    rows, hashes = [], {}
    for rank in range(10):
        ref = next(r for r in candidates if r["case_rank"] == rank and r["spec"]["resolution"] == "full" and r["spec"]["representation"] == "842_dense" and r["spec"]["alpha"] == 1.)
        prefix = c.read_json(Path(ref["prefix_path"])/"summary.json")
        batch, _ = previous.load_batch(SimpleNamespace(case_rank=rank, oasis_root=args.oasis_root, pair_file=c.DEFAULT_PAIR_FILE), device)
        similarities = {k: d.similarity(k, batch["fixed"][None, None]) for k in (*d.LOSSES, "lncc")}
        for state, path, digest in (("identity", None, None), ("lncc_cps2", prefix["dvf_path"], ref["prefix_dvf_sha256"]),
                                    ("lncc_dense", ref["dvf_path"], ref["dvf_sha256"])):
            if path:
                if sha256_file(Path(path)) != digest:
                    raise RuntimeError("Reference field changed")
                hashes[path] = digest
            norm = torch.from_numpy(np.load(path)).to(device) if path else torch.zeros_like(identity)
            voxel = (norm*scale).detach().requires_grad_(True)
            gradients, values = {}, {}
            for key, sim in similarities.items():
                warped = F.grid_sample(batch["moving"][None, None], (identity+voxel/scale)[None], mode="bilinear", padding_mode="border", align_corners=True)
                loss = sim(warped)
                gradient, = torch.autograd.grad(loss, voxel)
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("Nonfinite calibration gradient")
                gradients[key] = gradient.detach()
                values[key] = float(loss)
                del warped, gradient, loss
            row = {"case_rank": rank, "state": state, "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"]}
            rms = {k: float(v.square().mean().sqrt()) for k, v in gradients.items()}
            for key in (*d.LOSSES, "lncc"):
                row[key+"_loss"] = values[key]
                row[key+"_gradient_rms"] = rms[key]
            for key in d.LOSSES:
                if min(rms[key], rms["lncc"]) <= 1e-15:
                    raise RuntimeError("Degenerate calibration gradient; do not invent a multiplier")
                row[key+"_local_over_global_gradient_ratio"] = rms["lncc"]/rms[key]
                row[key+"_cosine_with_local"] = float((gradients[key]*gradients["lncc"]).mean())/(rms[key]*rms["lncc"])
            rows.append(row)
            print(f"calibration rank={rank} state={state}", flush=True)
            del voxel, norm, gradients
        del similarities, batch
    scales = {}
    for key in d.LOSSES:
        ratios = np.array([r[key+"_local_over_global_gradient_ratio"] for r in rows])
        scales[key] = {"median": float(np.median(ratios)), "p10": float(np.quantile(ratios, .1)), "p90": float(np.quantile(ratios, .9)),
                       "min": float(ratios.min()), "max": float(ratios.max())}
    d.verify_source_hashes(sources)
    c.write_json(args.review_root/"calibration.json", {"complete": True, "rows": rows, "scales": scales,
                 "source_hashes": sources, "reference_hashes": hashes, "gpu": args.gpu, "gpu_state": gpu_snapshot(),
                 "interpretation": "exploratory_gradient_scale_only; no training normalization or claim of equivalent optimal lambda"})
    c.write_csv(args.review_root/"calibration.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("ncc-root", "oasis-root", "review-root"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    run(p.parse_args())
