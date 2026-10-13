"""Read-only historical BE audit and sampled loss/gradient-scale diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

import common as c
import idir_sampled_diffusion as p
from run_one import idir_bending_energy
from run_oasis_idir_sampled_diffusion_one import configure, load_batch


def grad_norm(value, model):
    grads = torch.autograd.grad(value, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    return float(torch.stack([g.detach().double().square().sum() for g in grads if g is not None]).sum().sqrt())


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device, state, fraction = configure(1, 4)
    source = p.source_hashes()
    historical, hashes = [], {}
    for file in sorted(args.be_root.glob("*/summary.json")):
        row = json.loads(file.read_text())
        if not (file.parent / "DONE").exists() or row["omega"] != 32 or row["epochs"] != 2500 or row["coords_batch_size"] != 10000:
            raise RuntimeError("Unexpected historical BE protocol")
        historical.append({"rank": row["case_rank"], "alpha": row["alpha"], **{k: row[k] for k in p.METRICS[:3]},
                           "source": str(file), "final_data_loss": row["final_data_loss"], "final_reg_loss": row["final_reg_loss"]})
        hashes[str(file)] = p.sha256(file)
    if len(historical) != 60:
        raise RuntimeError("Historical BE reference expected60runs")
    means = []
    for alpha in sorted({r["alpha"] for r in historical}):
        rows = [r for r in historical if r["alpha"] == alpha]
        if sorted(r["rank"] for r in rows) != list(range(10)):
            raise RuntimeError("Historical case pairing mismatch")
        means.append({"alpha": alpha, **{k: float(np.mean([r[k] for r in rows])) for k in p.METRICS[:3]}})
    best = min(means, key=lambda r: (-r["dice_mean"], r["asd_symmetric_mean"], r["hd95_mean"]))
    diagnostics = []
    for rank in (0, 4, 9):
        args.rank = rank
        torch.manual_seed(1); torch.cuda.manual_seed_all(1)
        batch, inputs = load_batch(args, device)
        pool = p.training_coords(batch, device)
        model = c.build_siren(256, 3, 32).to(device)
        selected = next(r for r in historical if r["rank"] == rank and r["alpha"] == best["alpha"])
        checkpoint = Path(selected["source"]).parent / "model_state.pt"
        hashes[str(checkpoint)] = p.sha256(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        old_args = saved["args"] if isinstance(saved["args"], dict) else vars(saved["args"])
        for kind in ("fresh", "historical_best_BE"):
            if kind != "fresh":
                model.load_state_dict(saved["model_state_dict"], strict=True)
            torch.cuda.manual_seed_all(1)
            coords, _ = p.sample_coords(pool)
            disp = model(coords)
            data = 1 + c.ncc_loss(c.sample_volume(batch["fixed"], coords), c.sample_volume(batch["moving"], coords + disp))
            diff = p.analytic_diffusion(coords, disp)
            be = idir_bending_energy(coords, disp, batch_size=10000)
            data_grad, diff_grad, be_grad = (grad_norm(v, model) for v in (data, diff, be))
            row = {"rank": rank, "state": kind, "sample_count": 10000, "data_loss": float(data),
                   "analytic_diffusion": float(diff), "legacy_BE_divided_by10000": float(be),
                   "data_parameter_gradient_norm": data_grad, "diffusion_parameter_gradient_norm": diff_grad,
                   "BE_parameter_gradient_norm": be_grad,
                   "diffusion_lambda_equal_BE_weighted_gradient_norm": best["alpha"] * be_grad / max(diff_grad, 1e-30),
                   "unit_loss_balance_lambda": float(data.detach()) / max(float(diff.detach()), 1e-30),
                   "historical_backend_arguments": {k: old_args.get(k, "not_recorded") for k in ("cudnn_benchmark", "cudnn_deterministic")},
                   "input_sha256": inputs, "checkpoint_sha256": hashes[str(checkpoint)] if kind != "fresh" else None}
            row["grid_scales"] = [{"lambda": 10. ** e, "weighted_diffusion": (10. ** e) * float(diff),
                                    "regularizer_to_data_gradient_norm_ratio": (10. ** e) * diff_grad / max(data_grad, 1e-30)} for e in p.INITIAL_EXPONENTS]
            diagnostics.append(row)
            del data, diff, be, coords, disp
        del model, pool, batch, saved
    p.verify_hashes(source); p.verify_hashes(hashes)
    payload = {"complete": True, "source_sha256": source, "historical_sha256": hashes, "historical_means": means,
               "historical_cases": historical, "best_historical_BE": best, "diagnostics": diagnostics,
               "gpu_before": state, "allocator_fraction": fraction, "completed_unix": time.time(),
               "interpretation": "Gradient norms calibrate scale, not equivalent losses or optimal lambda; retain full requested grid"}
    p.write_json(args.out / "calibration.json", payload)
    c.write_csv(args.out / "historical_BE_means.csv", means)
    c.write_csv(args.out / "loss_scale_diagnostics.csv", diagnostics)
    (args.out / "DONE").write_text("Read-only BE reference and sampled scale audit complete\n")
    print(json.dumps({"best_historical_BE": best, "scale_rows": [{k: r[k] for k in ("rank", "state", "data_loss", "analytic_diffusion", "diffusion_lambda_equal_BE_weighted_gradient_norm")} for r in diagnostics]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--be-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
