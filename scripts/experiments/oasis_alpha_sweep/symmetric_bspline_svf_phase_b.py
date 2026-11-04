"""P78: approved amplitude/balance ablation; Phase-A numerical sources immutable."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
import torch

import symmetric_bspline_svf as a

c, d, s = a.c, a.d, a.s
VARIANT = a.VARIANT
PHASE = "phase_b_v1"
METHOD = "direct_cp_842_two_total_svf_midpoint_amplitude_balance"
ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
BETAS = (.01, .03, .1, .3, 1., 3., 10.)
PROTOCOL = {**a.PROTOCOL, "version": 1, "phase": PHASE,
            "amplitude_weight": "per_spec", "balance_weight": "per_spec",
            "amplitude": "native_full_voxel_mean_sum_xyz_actual_branch_velocity_mm_squared",
            "balance": "squared_difference_of_sqrt_E_plus_1e-8_mm2",
            "amplitude_epsilon_constant_removed": True,
            "diffusion_coefficient": 2., "amplitude_grid": list(ALPHAS),
            "balance_grid": list(BETAS), "maximum_production_fits": 28}
NCC9Pair, branch_controls, prolong, reconstruct = a.NCC9Pair, a.branch_controls, a.prolong, a.reconstruct


@dataclass(frozen=True)
class Spec:
    arm: str
    alpha: float = 0.
    beta: float = 0.
    smoke: bool = False
    swapped: bool = False
    repeat: bool = False
    case_rank: int = 0

    def validate(self):
        if self.case_rank != 0 or self.arm not in ("base", "amplitude", "balance", "combined"):
            raise ValueError("Only approved rank-0 Phase-B arms")
        if not all(math.isfinite(x) for x in (self.alpha, self.beta)):
            raise ValueError("Finite weights required")
        allowed = {"base": self.alpha == self.beta == 0,
                   "amplitude": self.alpha in ALPHAS and self.beta == 0,
                   "balance": self.alpha == 0 and self.beta in BETAS,
                   "combined": self.alpha in ALPHAS and self.beta in BETAS}
        if not allowed[self.arm] or (self.swapped and self.repeat) or (self.smoke and (self.swapped or self.repeat)):
            raise ValueError("Invalid approved weight/diagnostic combination")

    @property
    def tied(self): return False

    @property
    def coefficient(self): return 2.

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL},
                                           sort_keys=True).encode()).hexdigest()[:10]
        suffix = "_swap" if self.swapped else "_repeat" if self.repeat else ""
        return f"{'smoke_' if self.smoke else ''}rank000_842_{self.arm}_a{self.alpha:g}_b{self.beta:g}{suffix}_{digest}"


def initial_matrix():
    return [Spec("base")] + [Spec("amplitude", x) for x in ALPHAS] + [Spec("balance", 0., x) for x in BETAS]


def smoke_matrix():
    return [Spec("base", smoke=True), Spec("amplitude", ALPHAS[-1], smoke=True),
            Spec("balance", beta=BETAS[-1], smoke=True),
            Spec("combined", ALPHAS[-1], BETAS[-1], smoke=True)]


def rank_key(row):
    return (-row["dice"], row["asd"], row["hd95"],
            row["spec"]["alpha"], row["spec"]["beta"], row["run_name"])


def neighbors(grid, center):
    index = grid.index(center)
    start = max(0, min(index-1, len(grid)-3))
    return tuple(grid[start:start+3])


def combined_matrix(rows):
    if {Spec(**row["spec"]).name() for row in rows} != {x.name() for x in initial_matrix()} or len(rows) != 15:
        raise ValueError("Require all 15 unique original-direction initial candidates")
    amp = min((x for x in rows if x["spec"]["arm"] == "amplitude"), key=rank_key)
    bal = min((x for x in rows if x["spec"]["arm"] == "balance"), key=rank_key)
    av, bv = neighbors(ALPHAS, amp["spec"]["alpha"]), neighbors(BETAS, bal["spec"]["beta"])
    return [Spec("combined", x, y) for x in av for y in bv]


def diagnostic_matrix(rows):
    if len(rows) != 24 or len({x["run_name"] for x in rows}) != 24:
        raise ValueError("Require all 24 unique primary fits")
    output = []
    for arm in ("amplitude", "balance", "combined"):
        best = min((x for x in rows if x["spec"]["arm"] == arm), key=rank_key)
        output.append((Spec(**{**best["spec"], "swapped": True}), str(best["gpu"]), best["run_name"]))
    best = min(rows, key=rank_key)
    output.append((Spec(**{**best["spec"], "repeat": True}), str(best["gpu"]), best["run_name"]))
    return output


def stages(spec):
    return a.stages(spec)


def penalties(v1_norm, v2_norm, native_scale):
    v1, v2 = v1_norm*native_scale, v2_norm*native_scale
    e1, e2 = v1.square().sum(-1).mean(), v2.square().sum(-1).mean()
    rms1, rms2 = (e1+1e-8).sqrt(), (e2+1e-8).sqrt()
    return e1+e2, (rms1-rms2).square(), (e1, e2, rms1, rms2)


class Objective(a.Objective):
    def __call__(self, parameter):
        v1, v2 = self.velocity(parameter)
        u1, u2 = (s.exp_velocity(v, 7, self.native_identity) for v in (v1, v2))
        data = self.ncc(self.warp(self.moving, u1), self.warp(self.fixed, u2))
        ev1, ev2 = (v1, v2) if self.eval_shape == self.image_shape else self.velocity(parameter, False)
        r1, r2 = a.diffusion(ev1*self.scale), a.diffusion(ev2*self.scale)
        reg = r1+r2
        amplitude, balance, energies = penalties(v1, v2, self.native_scale)
        wa, wb = self.spec.alpha*amplitude, self.spec.beta*balance
        loss = data + self.alpha*reg
        # Preserve the exact historical base computation when weights are zero.
        if self.spec.alpha: loss = loss + wa
        if self.spec.beta: loss = loss + wb
        names = ("R1", "R2", "amplitude_mm2", "balance_mm2", "E1_mm2", "E2_mm2",
                 "a1_mm", "a2_mm", "weighted_amplitude", "weighted_balance")
        self.last_terms = dict(zip(names, torch.stack((r1, r2, amplitude, balance, *energies, wa, wb)).detach().cpu().tolist()))
        return loss, data, reg


def zero_parity(device):
    """Nonzero-field full objective and BOTH branch gradients; no optimization."""
    torch.manual_seed(23)
    shape = (16, 20, 24)
    fixed, moving = torch.rand(shape, device=device), torch.rand(shape, device=device)
    spec = Spec("base", smoke=True)
    stage = stages(spec)[0]
    old = a.Objective(fixed, moving, stage, a.Spec("A1", True))
    new = Objective(fixed, moving, stage, spec)
    q = (torch.randn((2, *new.control_shape, 3), device=device)*.002).requires_grad_()
    x, y = old(q)[0], new(q)[0]
    gx, = torch.autograd.grad(x, q)
    gy, = torch.autograd.grad(y, q)
    if not torch.allclose(x, y, atol=1e-7, rtol=1e-6) or not torch.allclose(gx, gy, atol=2e-6, rtol=1e-5):
        raise RuntimeError("Phase-A zero-penalty objective/gradient parity failed")
    swap_spec = Spec("combined", ALPHAS[-1], BETAS[-1], smoke=True)
    one, two = Objective(fixed, moving, stage, swap_spec), Objective(moving, fixed, stage, swap_spec)
    x, y = one(q)[0], two(q.flip(0))[0]
    gx, = torch.autograd.grad(x, q)
    gy, = torch.autograd.grad(y, q)
    if not torch.allclose(x, y, atol=2e-6, rtol=1e-5) or not torch.allclose(gx, gy, atol=2e-6, rtol=1e-5):
        raise RuntimeError("Nonzero-field input-exchange objective/gradient test failed")
    return {"passed": True, "exchange_value_error": float((x-y).abs()),
            "exchange_gradient_max_abs": float((gx-gy).abs().max())}


def optimize(initial, objective, out):
    """Historical joint state-indexed Adam, with explicit pair metadata/history."""
    out.mkdir(parents=True, exist_ok=False)
    parameter = torch.nn.Parameter(initial.detach().clone())
    if not torch.equal(parameter.detach(), initial):
        raise RuntimeError("Clone changed initialization")
    stage = objective.stage
    optimizer = torch.optim.Adam([parameter], lr=stage.lr, betas=(.9, .999), eps=1e-8)
    best, best_loss, best_epoch = parameter.detach().clone(), math.inf, 0
    rows, reason = [], "max_epochs"
    if parameter.is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (out/"history.jsonl").open("x") as stream:
        for epoch in range(stage.max_epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss, data, reg = objective(parameter)
            values = torch.stack((loss, data, reg)).detach().cpu().tolist()
            if not all(math.isfinite(x) for x in values):
                raise FloatingPointError("Nonfinite midpoint objective")
            if values[0] < best_loss:
                best.copy_(parameter.detach()); best_loss, best_epoch = values[0], epoch
            row = {"epoch": epoch, "optimizer_updates_completed": epoch, "loss": values[0],
                   "data_loss": values[1], "reg_loss": values[2], "weighted_reg_loss": objective.alpha*values[2],
                   **objective.last_terms, "lr": stage.lr, "best_epoch": best_epoch,
                   "elapsed_sec": time.monotonic()-started}
            rows.append(row); stream.write(json.dumps(row)+"\n")
            if epoch % 25 == 0:
                stream.flush(); print(json.dumps({"cps": stage.cps, **row}), flush=True)
            if epoch == stage.max_epochs:
                break
            if d.plateau(rows, stage, 0):
                reason = "adaptive_plateau"; break
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Missing/nonfinite joint control gradient")
            optimizer.step()
    if parameter.is_cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        selected = [float(x) for x in objective(best)]
    if not math.isclose(selected[0], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Selected checkpoint objective mismatch")
    info = {"stage": asdict(stage), "effective_patience": stage.patience,
            "representation": "two_total_bspline_svfs" if not objective.spec.tied else "tied_single_total_bspline_svf",
            "parameter_shape": list(initial.shape), "control_shape": list(objective.control_shape),
            "parameter_units": PROTOCOL["parameter_units"], "branch_velocity_scale": .5,
            "evaluation_shape": list(objective.eval_shape), "coefficient": objective.alpha,
            "amplitude_weight": objective.spec.alpha, "balance_weight": objective.spec.beta,
            "selected_terms": objective.last_terms,
            "actual_epochs": rows[-1]["epoch"], "best_epoch": best_epoch,
            "best_loss": selected[0], "best_data_loss": selected[1], "best_reg_loss": selected[2],
            "initial_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
            "stopping_reason": reason, "lr_reductions": 0, "training_elapsed_sec": time.monotonic()-started,
            "peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30 if parameter.is_cuda else 0.,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30 if parameter.is_cuda else 0.,
            "blur_sigma": objective.blur_sigma, "blur_kernel_size": objective.blur_kernel_size,
            "initial_clone_max_abs": 0., "checkpoint_selection": "joint_lowest_total_loss_including_initialization"}
    d.atomic_torch_save({"twice_branch_controls_norm_xyz": best.cpu(), "summary": info}, out/"best.pt")
    d.atomic_torch_save({"twice_branch_controls_norm_xyz": parameter.detach().cpu(),
                         "optimizer": optimizer.state_dict(), "summary": info}, out/"last.pt")
    c.write_csv(out/"history.csv", rows); c.write_json(out/"summary.json", info)
    return best.detach(), info

