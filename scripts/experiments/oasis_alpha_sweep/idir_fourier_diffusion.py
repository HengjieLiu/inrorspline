"""Audited Fourier-input SIREN ablation; reuse the immutable sampled objective."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import common as c
import idir_sampled_diffusion as old

VARIANT = "oasis_seed3_first10_idir_fourier_sampled_diffusion_v2"
METHOD = "idir_fourier_siren_sampled10000_global_1ncc_analytic_voxel_diffusion"
SIGMAS = (.5, 1., 2., 4., 8., 16.)
LAMBDAS = (.3, 1., 3.)
GPUS = (0, 1, 2, 3)
METRICS = old.METRICS
PROTOCOL = {**old.PROTOCOL, "method": METHOD, "layers": "spec.input_dim,width,width,width,3",
            "encoding": "concat(q,sin(2*pi*sigma*B0*((q+1)/2)),cos(same))",
            "B0": "frozen64x3_standard_gaussian_CPU_local_generator_seed1_initial_sweep",
            "amplitude": 1., "initialization": "unchanged_native_rule_recomputed_for_input_fan_in",
            "sampling_rng": "dedicated_CUDA_generator_seed1_matches_P53_default_seed1_sequence",
            "jacobian_wrt": "original_normalized_xyz_including_encoding_chain_rule",
            "checkpoint_selection": "terminal2500", "init_rescaling": None,
            "diagnostics": "after_training_CPU_saved_initial_model_copy_dedicated_CPU_seed1_sample"}


@dataclass(frozen=True)
class FitSpec:
    rank: int
    alpha: float
    sigma: float = 0.
    width: int = 256
    features: int = 64
    omega: int = 32
    seed: int = 1
    smoke_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "sigma", float(self.sigma))

    @property
    def family(self):
        return "fourier" if self.sigma else "plain"

    @property
    def input_dim(self):
        return 3 + 2 * self.features if self.sigma else 3

    @property
    def updates(self):
        return self.smoke_steps or 2500

    @property
    def param_count(self):
        return 2 * self.width ** 2 + (self.input_dim + 6) * self.width + 3

    def validate(self):
        if self.rank not in range(10) or self.width not in (256, 286) or self.features not in (32, 64, 128):
            raise ValueError("Unapproved case/architecture")
        if self.omega not in (32, 64) or self.seed not in (1, 2, 3) or self.smoke_steps not in (0, 1, 100):
            raise ValueError("Unapproved frequency/seed/update budget")
        if not math.isfinite(self.alpha) or not 1e-10 <= self.alpha <= 1e10:
            raise ValueError("Diffusion coefficient requires numerical review")
        if self.sigma != 0 and not 2 ** -8 <= self.sigma <= 64:
            raise ValueError("Fourier bandwidth outside audited range; review before extension")
        if self.sigma and self.width != 256:
            raise ValueError("Fourier arm uses width256")

    def key(self):
        return (self.family, self.sigma, self.width, self.features if self.sigma else 0, self.omega, self.alpha, self.seed)

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return f"{'smoke'+str(self.smoke_steps)+'_' if self.smoke_steps else ''}{self.family}_rank{self.rank:03d}_s{self.sigma:g}_m{self.features if self.sigma else 0}_w{self.width}_o{self.omega}_a{self.alpha:g}_seed{self.seed}_{digest}"


class FourierSiren(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.sigma = float(spec.sigma)
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        self.register_buffer("B0", torch.randn(128, 3, generator=generator)[:spec.features].clone())
        self.network = c.build_siren(spec.width, 3, spec.omega, input_dim=spec.input_dim)

    def encode(self, q):
        phase = 2 * math.pi * self.sigma * (((q + 1) / 2) @ self.B0.T)
        return torch.cat((q, phase.sin(), phase.cos()), dim=-1)

    def forward(self, q):
        return self.network(self.encode(q))


def build_model(spec):
    spec.validate()
    return FourierSiren(spec) if spec.sigma else c.build_siren(spec.width, 3, spec.omega)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_hashes():
    result = old.source_hashes()
    for name in ("idir_fourier_diffusion.py", "run_oasis_idir_fourier_one.py", "launch_oasis_idir_fourier.py"):
        path = Path(__file__).with_name(name)
        result[str(path)] = old.sha256(path)
    return result


def grid(sigmas=SIGMAS, lambdas=LAMBDAS):
    specs = [FitSpec(r, a, s) for s in sigmas for a in lambdas for r in range(10)]
    specs += [FitSpec(r, a, width=w) for w in (256, 286) for a in lambdas for r in range(10)]
    return specs  # 240 identities; 10 plain256/alpha1 identities are read-only P53 reuse.


def sample_coords(pool, generator):
    indices = torch.randperm(len(pool), device=pool.device, generator=generator)[:10000]
    return pool[indices].detach().requires_grad_(True), indices


def initial_diagnostics(model, coords, fixed, moving, alpha):
    """Read-only, no RNG consumption; no parameter update or extra training points."""
    captured = {}
    network = model.network if isinstance(model, FourierSiren) else model
    hook = network.layers[0].register_forward_hook(lambda _m, _x, y: captured.update(pre=y.detach()))
    total, data, reg = old.objective(model, coords, fixed, moving, alpha)
    hook.remove()
    params = list(model.parameters())
    norms = []
    for loss in (data, reg):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        norms.append(float(torch.stack([g.detach().square().sum() for g in grads if g is not None]).sum().sqrt()))
    with torch.no_grad():
        displacement = model(coords)
        scale = coords.new_tensor([95.5, 111.5, 79.5])
        epe = (displacement * scale).norm(dim=-1)
    row = {"data_loss": float(data.detach()), "diffusion": float(reg.detach()),
           "data_grad_norm": norms[0], "diffusion_grad_norm": norms[1],
           "weighted_reg_to_data_grad_ratio": alpha * norms[1] / max(norms[0], 1e-30),
           "first_pre_sine_std": float((captured["pre"] * network.omega).std()),
           "initial_displacement_mean_voxel": float(epe.mean()), "initial_displacement_max_voxel": float(epe.max())}
    if not all(math.isfinite(v) for v in row.values()) or norms[0] <= 0 or norms[1] <= 0:
        raise RuntimeError("Invalid initialization diagnostics")
    return row


def derivative_check(model, device):
    """FP64 numerical audit of original-coordinate chain rule without changing model."""
    import copy
    clone = copy.deepcopy(model).to(device=device, dtype=torch.float64)
    q = torch.tensor([[-.31, .17, .42], [.13, -.26, .09], [.21, .33, -.41]], device=device, dtype=torch.float64, requires_grad=True)
    u = clone(q)
    jac = torch.stack([torch.autograd.grad(u[:, k].sum(), q, retain_graph=True)[0] for k in range(3)], dim=1)
    numeric = torch.empty_like(jac)
    with torch.no_grad():
        for axis in range(3):
            delta = torch.zeros_like(q); delta[:, axis] = 1e-6
            numeric[:, :, axis] = (clone(q + delta) - clone(q - delta)) / 2e-6
    max_error = float((jac - numeric).abs().max())
    torch.testing.assert_close(jac, numeric, atol=1e-7, rtol=2e-4)
    return {"passed": True, "max_absolute_error": max_error, "coordinate": "original_q_xyz", "precision": "float64", "epsilon": 1e-6}


def aggregate(rows):
    result = []
    for key in sorted({FitSpec(**r["spec"]).key() for r in rows}):
        group = [r for r in rows if FitSpec(**r["spec"]).key() == key]
        if len(group) != 10 or sorted(r["rank"] for r in group) != list(range(10)):
            continue
        spec = FitSpec(**group[0]["spec"])
        row = {"family": spec.family, "sigma": spec.sigma, "width": spec.width, "features": spec.features if spec.sigma else 0,
               "omega": spec.omega, "alpha": spec.alpha, "seed": spec.seed, "n": 10, "param_count": spec.param_count}
        for metric in METRICS:
            values = np.array([r[metric] for r in group])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite cohort")
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = float(values.std(ddof=1))
        result.append(row)
    return result


def select(rows):
    if not rows:
        raise ValueError("No complete cohorts")
    return min(rows, key=lambda r: (-r["dice_mean_mean"], r["asd_symmetric_mean_mean"], r["hd95_mean_mean"],
                                    abs(math.log10(r["alpha"])), abs(math.log2(r["sigma"]/2)) if r["sigma"] else 0,
                                    r["sigma"], r["width"]))


def boundary_decision(means, sigmas, lambdas):
    main = [r for r in means if r["omega"] == 32 and r["seed"] == 1 and (r["family"] == "plain" or r["features"] == 64)]
    expected = {(s, 256, a) for s in sigmas for a in lambdas} | {(0., w, a) for w in (256, 286) for a in lambdas}
    seen = {(r["sigma"], r["width"], r["alpha"]) for r in main}
    if seen != expected or any(r["n"] != 10 for r in main):
        raise ValueError("Require every full10 cohort before a boundary decision")
    best = select([r for r in main if r["family"] == "fourier"])
    extra_sigma = min(sigmas)/2 if best["sigma"] == min(sigmas) else max(sigmas)*2 if best["sigma"] == max(sigmas) else None
    winners = [best] + [select([r for r in main if r["family"] == "plain" and r["width"] == w]) for w in (256, 286)]
    extra_lambdas = []
    if any(r["alpha"] == min(lambdas) for r in winners): extra_lambdas.append(min(lambdas)/10)
    if any(r["alpha"] == max(lambdas) for r in winners): extra_lambdas.append(max(lambdas)*10)
    return {"selected_fourier": best, "selected_controls": winners[1:], "extra_sigma": extra_sigma,
            "extra_lambdas": extra_lambdas, "tested_sigmas": list(sigmas), "tested_lambdas": list(lambdas)}


def promising(decision):
    a = decision["selected_fourier"]
    return all(a["dice_mean_mean"] > b["dice_mean_mean"] and all(a[k+"_mean"] <= b[k+"_mean"] for k in METRICS[1:])
               for b in decision["selected_controls"])


def followups(decision):
    a = decision["selected_fourier"]
    baseline = select(decision["selected_controls"])
    jobs = [FitSpec(r, a["alpha"], a["sigma"], omega=64) for r in range(10)]
    jobs += [FitSpec(r, a["alpha"], a["sigma"], features=m) for m in (32, 128) for r in range(10)]
    for seed in (2, 3):
        jobs += [FitSpec(r, a["alpha"], a["sigma"], seed=seed) for r in range(10)]
        jobs += [FitSpec(r, baseline["alpha"], width=baseline["width"], seed=seed) for r in range(10)]
    return jobs
