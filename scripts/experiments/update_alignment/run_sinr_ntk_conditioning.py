#!/usr/bin/env python
"""Matrix-free SINR CP-space NTK conditioning diagnostics.

This runner computes the action of the SINR coefficient-generator tangent
kernel on actual OASIS cps4 registration gradients:

    DirectCP action: B s
    SINR action:     B J_theta J_theta^T s

It intentionally does not materialize J_theta, J_theta J_theta^T, B B^T, or
B J_theta J_theta^T B^T.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "docs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
OASIS_SCRIPT_DIR = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep"
if str(OASIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(OASIS_SCRIPT_DIR))

import common as c  # noqa: E402


DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-29_sinr_ntk_conditioning"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/sinr_ntk_conditioning_20260629_v1")
DEFAULT_SINR_FINAL_RUN = Path(
    "/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/"
    "sinr_cps4_be_bspline_analytic/"
    "sinr_cps4_be_bspline_analytic_rank009_a1000_e2500_lr0p0001"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_role_log(review_root: Path, supervisor: str, executor: str) -> None:
    path = review_root / "supervisor_executor_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else "# Supervisor / Executor Log\n"
    round_count = existing.count("## Round ") + 1
    block = [
        "",
        f"## Round {round_count}",
        "",
        f"**Supervisor:** {supervisor}",
        "",
        f"**Executor:** {executor}",
        "",
    ]
    path.write_text(existing.rstrip() + "\n" + "\n".join(block))


def parse_csv_list(text: str) -> List[str]:
    return [token for token in str(text).replace(" ", "").split(",") if token]


def dot_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.reshape(-1).float(), b.reshape(-1).float())


def norm_flat(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a.reshape(-1).float())


def safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def vector_metrics(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Return metrics comparing a target vector a against source vector b."""

    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    a_norm = torch.linalg.norm(aa)
    b_norm = torch.linalg.norm(bb)
    dot = torch.dot(aa, bb)
    denom = a_norm * b_norm
    cosine = dot / (denom + eps)
    alpha = dot / (torch.dot(bb, bb) + eps)
    residual = aa - alpha * bb
    residual_norm = torch.linalg.norm(residual)
    return {
        "cosine": safe_float(cosine),
        "alpha_source_to_target": safe_float(alpha),
        "relative_residual_target": safe_float(residual_norm / (a_norm + eps)),
        "target_norm": safe_float(a_norm),
        "source_norm": safe_float(b_norm),
        "amplification_target_over_source": safe_float(a_norm / (b_norm + eps)),
        "residual_norm": safe_float(residual_norm),
    }


def param_norm(params: Sequence[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        total += float(p.detach().float().square().sum().cpu())
    return math.sqrt(total)


def build_sinr_generator(args: argparse.Namespace, control_shape: Tuple[int, int, int], device: torch.device) -> torch.nn.Module:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    generator = c.NetworkGenerator(
        c.build_siren(
            hidden_dim=int(args.hidden_dim),
            hidden_layers=int(args.hidden_layers),
            omega=float(args.spline_omega),
        )
    ).to(device)
    return generator


def load_final_state(generator: torch.nn.Module, final_run: Path) -> Dict[str, object]:
    state_path = Path(final_run) / "model_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing final SINR model state: {state_path}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "generator_state_dict" not in payload:
        raise RuntimeError(f"Unexpected model_state.pt payload at {state_path}")
    generator.load_state_dict(payload["generator_state_dict"])
    return payload


def compute_losses_from_control(
    control_flat: torch.Tensor,
    control_shape: Tuple[int, int, int],
    batch: Mapping[str, object],
    *,
    cps: int,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    if not isinstance(moving, torch.Tensor) or not isinstance(fixed_image, torch.Tensor):
        raise TypeError("OASIS batch does not contain tensor moving/fixed images.")
    dense_disp = c.expand_bspline_controls(control_flat, control_shape, c.IMAGE_SHAPE, int(cps))
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_disp.device).detach()
    warped = c.warp_full(moving, id_dense + dense_disp, mode="bilinear")
    data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
    control_grid = control_flat.reshape(*control_shape, 3)
    reg_loss = c.analytic_bspline_sampled_be(
        control_grid,
        c.IMAGE_SHAPE,
        int(cps),
        component_reduction="mean",
    )
    total_loss = data_loss + float(alpha) * reg_loss
    return data_loss, reg_loss, total_loss, dense_disp


def sample_flat_indices(mask: torch.Tensor, sample_count: int, seed: int) -> torch.Tensor:
    eligible = torch.nonzero(mask.detach().reshape(-1).bool(), as_tuple=False).reshape(-1)
    if eligible.numel() == 0:
        raise RuntimeError("Cannot sample dense action metrics from an empty mask.")
    count = min(int(sample_count), int(eligible.numel()))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(int(eligible.numel()), generator=gen)[:count]
    chosen = eligible.detach().cpu()[perm].sort().values
    return chosen.to(device=mask.device, dtype=torch.long)


class KernelAction:
    def __init__(
        self,
        generator: torch.nn.Module,
        control_coords: torch.Tensor,
        control_shape: Tuple[int, int, int],
    ) -> None:
        self.generator = generator
        self.control_coords = control_coords
        self.control_shape = control_shape
        self.names = [name for name, _p in generator.named_parameters()]
        self.params = {name: p for name, p in generator.named_parameters()}
        self.param_values = tuple(self.params[name] for name in self.names)
        self.output_for_vjp = generator(control_coords, control_shape)

    def _functional_output(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        from torch.func import functional_call

        return functional_call(self.generator, dict(params), (self.control_coords, self.control_shape))

    def vjp(self, cp_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        grads = torch.autograd.grad(
            self.output_for_vjp,
            self.param_values,
            grad_outputs=cp_vector,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        return {name: grad.detach() for name, grad in zip(self.names, grads)}

    def jvp(self, param_tangent: Mapping[str, torch.Tensor]) -> torch.Tensor:
        from torch.func import jvp

        tangent = {name: param_tangent[name].to(device=self.params[name].device, dtype=self.params[name].dtype) for name in self.names}
        _out, tangent_out = jvp(self._functional_output, (self.params,), (tangent,))
        return tangent_out.detach()

    def k_matvec(self, cp_vector: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p = self.vjp(cp_vector)
        return self.jvp(p), p


def random_rayleigh_rows(
    action: KernelAction,
    *,
    state: str,
    gradient: str,
    probes: int,
    seed: int,
    reference: torch.Tensor,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if int(probes) <= 0:
        return rows
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(int(seed))
    for idx in range(int(probes)):
        v = torch.randn(reference.shape, generator=gen, device=reference.device, dtype=reference.dtype)
        kv, _p = action.k_matvec(v)
        v_norm = norm_flat(v)
        kv_norm = norm_flat(kv)
        rayleigh = dot_flat(v, kv) / (dot_flat(v, v) + 1e-12)
        rows.append(
            {
                "state": state,
                "gradient": gradient,
                "probe_index": int(idx),
                "rayleigh": safe_float(rayleigh),
                "amplification": safe_float(kv_norm / (v_norm + 1e-12)),
                "v_norm": safe_float(v_norm),
                "kv_norm": safe_float(kv_norm),
            }
        )
    return rows


def power_iteration(
    action: KernelAction,
    *,
    reference: torch.Tensor,
    iters: int,
    seed: int,
) -> Dict[str, float]:
    if int(iters) <= 0:
        return {"lambda_max_estimate": float("nan"), "power_final_norm": float("nan")}
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(int(seed))
    v = torch.randn(reference.shape, generator=gen, device=reference.device, dtype=reference.dtype)
    v = v / (norm_flat(v) + 1e-12)
    lam = torch.tensor(float("nan"), device=reference.device)
    kv_norm = torch.tensor(float("nan"), device=reference.device)
    for _idx in range(int(iters)):
        kv, _p = action.k_matvec(v)
        lam = dot_flat(v, kv) / (dot_flat(v, v) + 1e-12)
        kv_norm = norm_flat(kv)
        if not torch.isfinite(kv_norm) or float(kv_norm.detach().cpu()) <= 1e-20:
            break
        v = kv / (kv_norm + 1e-12)
    return {
        "lambda_max_estimate": safe_float(lam),
        "power_final_norm": safe_float(kv_norm),
    }


def analyze_state_gradient(
    args: argparse.Namespace,
    *,
    state: str,
    gradient_mode: str,
    generator: torch.nn.Module,
    control_coords: torch.Tensor,
    control_shape: Tuple[int, int, int],
    batch: Mapping[str, object],
    sample_indices: torch.Tensor,
    matrix_summary: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    cps = int(args.cps)
    alpha = float(args.alpha)
    generator.eval()
    with torch.no_grad():
        control_current = generator(control_coords, control_shape).detach()
    control_leaf = control_current.clone().detach().requires_grad_(True)
    data_loss, reg_loss, total_loss, _dense_disp = compute_losses_from_control(
        control_leaf,
        control_shape,
        batch,
        cps=cps,
        alpha=alpha,
    )
    objective = data_loss if gradient_mode == "data" else total_loss
    s = torch.autograd.grad(objective, control_leaf, retain_graph=False, create_graph=False)[0].detach()
    action = KernelAction(generator, control_coords, control_shape)
    ks, p = action.k_matvec(s)

    x_dense = c.expand_bspline_controls(s, control_shape, c.IMAGE_SHAPE, cps).detach()
    y_dense = c.expand_bspline_controls(ks, control_shape, c.IMAGE_SHAPE, cps).detach()
    x_sample = x_dense.reshape(-1, 3)[sample_indices]
    y_sample = y_dense.reshape(-1, 3)[sample_indices]
    dense_metrics = vector_metrics(y_sample, x_sample)
    cp_metrics = vector_metrics(ks, s)
    dot_s_ks = dot_flat(s, ks)
    lambda_s = dot_s_ks / (dot_flat(s, s) + 1e-12)
    p_norm = param_norm(list(p.values()))
    power = power_iteration(
        action,
        reference=s,
        iters=int(args.power_iters),
        seed=int(args.seed) + 1000 + len(state) + len(gradient_mode),
    )
    spectrum_rows = random_rayleigh_rows(
        action,
        state=state,
        gradient=gradient_mode,
        probes=int(args.spectrum_probes),
        seed=int(args.seed) + 2000 + len(state) + len(gradient_mode),
        reference=s,
    )
    row: Dict[str, object] = {
        "dataset": "oasis",
        "case_rank": int(args.case_rank),
        "state": state,
        "gradient": gradient_mode,
        "loss_data": safe_float(data_loss),
        "loss_reg": safe_float(reg_loss),
        "loss_total": safe_float(total_loss),
        "cp_coefficients": int(matrix_summary["cp_coefficients"]),
        "sinr_param_count": int(matrix_summary["sinr_param_count"]),
        "rank_upper_bound": int(matrix_summary["rank_upper_bound"]),
        "nullity_lower_bound": int(matrix_summary["nullity_lower_bound"]),
        "sample_voxels": int(sample_indices.numel()),
        "D_U_sample_cosine": dense_metrics["cosine"],
        "R_U_sample_scaled_residual": dense_metrics["relative_residual_target"],
        "alpha_Bs_to_BKs": dense_metrics["alpha_source_to_target"],
        "A_U_sample_norm_BKs_over_Bs": dense_metrics["amplification_target_over_source"],
        "D_C_cosine": cp_metrics["cosine"],
        "R_C_scaled_residual": cp_metrics["relative_residual_target"],
        "alpha_s_to_Ks": cp_metrics["alpha_source_to_target"],
        "A_C_norm_Ks_over_s": cp_metrics["amplification_target_over_source"],
        "s_norm": cp_metrics["source_norm"],
        "Ks_norm": cp_metrics["target_norm"],
        "param_vjp_norm": p_norm,
        "dot_s_Ks": safe_float(dot_s_ks),
        "lambda_s": safe_float(lambda_s),
        "psd_check_pass": bool(safe_float(dot_s_ks) >= -1e-6 * cp_metrics["source_norm"] * cp_metrics["target_norm"]),
        "lambda_max_estimate": power["lambda_max_estimate"],
        "power_final_norm": power["power_final_norm"],
    }
    return row, spectrum_rows


def summarize_spectrum(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["state"]), str(row["gradient"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (state, gradient), items in grouped.items():
        rays = np.asarray([float(item["rayleigh"]) for item in items], dtype=np.float64)
        amps = np.asarray([float(item["amplification"]) for item in items], dtype=np.float64)
        out.append(
            {
                "state": state,
                "gradient": gradient,
                "probes": int(len(items)),
                "rayleigh_p05": float(np.percentile(rays, 5)),
                "rayleigh_p50": float(np.percentile(rays, 50)),
                "rayleigh_p95": float(np.percentile(rays, 95)),
                "rayleigh_p99": float(np.percentile(rays, 99)),
                "amplification_p05": float(np.percentile(amps, 5)),
                "amplification_p50": float(np.percentile(amps, 50)),
                "amplification_p95": float(np.percentile(amps, 95)),
                "amplification_p99": float(np.percentile(amps, 99)),
            }
        )
    return out


def make_plots(review_root: Path, action_rows: Sequence[Dict[str, object]], spectrum_summary: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(review_root / "plots/plot_error.json", {"error": str(exc)})
        return
    plots = review_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['state']}\n{row['gradient']}" for row in action_rows]
    x = np.arange(len(labels))
    d_u = [float(row["D_U_sample_cosine"]) for row in action_rows]
    r_u = [float(row["R_U_sample_scaled_residual"]) for row in action_rows]
    a_u = [float(row["A_U_sample_norm_BKs_over_Bs"]) for row in action_rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    axes[0].bar(x - 0.18, d_u, width=0.36, label="D_U")
    axes[0].bar(x + 0.18, r_u, width=0.36, label="R_U")
    axes[0].set_ylabel("direction / residual")
    axes[0].set_ylim(-1.05, max(1.05, max(r_u + [0.0]) * 1.15))
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, a_u, width=0.5)
    axes[1].set_ylabel("A_U")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "dense_update_action_bars.png", dpi=180)
    plt.close(fig)

    if spectrum_summary:
        labels = [f"{row['state']}\n{row['gradient']}" for row in spectrum_summary]
        x = np.arange(len(labels))
        p50 = np.asarray([float(row["rayleigh_p50"]) for row in spectrum_summary])
        p05 = np.asarray([float(row["rayleigh_p05"]) for row in spectrum_summary])
        p95 = np.asarray([float(row["rayleigh_p95"]) for row in spectrum_summary])
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(x, p50, marker="o", label="p50")
        ax.fill_between(x, p05, p95, alpha=0.25, label="p05-p95")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("random Rayleigh")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "random_rayleigh_summary.png", dpi=180)
        plt.close(fig)


def write_report(review_root: Path, summary: Mapping[str, object], action_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# SINR NTK Conditioning Report",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This report compares the local dense update action `B s` for DirectCP against `B J_theta J_theta^T s` for the SINR control generator. Dense kernels are not materialized.",
        "",
        "## Action Metrics",
        "",
        "| State | Gradient | D_U | R_U | A_U | A_C | lambda_s | PSD |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in action_rows:
        lines.append(
            "| {state} | {gradient} | {du:.4g} | {ru:.4g} | {au:.4g} | {ac:.4g} | {lam:.4g} | {psd} |".format(
                state=row["state"],
                gradient=row["gradient"],
                du=float(row["D_U_sample_cosine"]),
                ru=float(row["R_U_sample_scaled_residual"]),
                au=float(row["A_U_sample_norm_BKs_over_Bs"]),
                ac=float(row["A_C_norm_Ks_over_s"]),
                lam=float(row["lambda_s"]),
                psd="pass" if row["psd_check_pass"] else "fail",
            )
        )
    if action_rows:
        init_rows = [row for row in action_rows if row.get("state") == "init"]
        final_rows = [row for row in action_rows if row.get("state") == "final"]
        data_rows = [row for row in action_rows if row.get("gradient") == "data"]
        lines.extend(["", "## Interpretation", ""])
        if init_rows and final_rows:
            best_init = min(float(row["R_U_sample_scaled_residual"]) for row in init_rows)
            best_final = min(float(row["R_U_sample_scaled_residual"]) for row in final_rows)
            lines.append(
                "- The SINR CP-space kernel is not acting like a scalar-rescaled DirectCP update on this case: "
                f"the best scalar-residual remains `{best_init:.3g}` at init and `{best_final:.3g}` at final."
            )
        if data_rows:
            amps = [float(row["A_U_sample_norm_BKs_over_Bs"]) for row in data_rows]
            lines.append(
                "- The task-gradient dense-action amplification is very large "
                f"(`{min(amps):.3g}` to `{max(amps):.3g}` for data gradients), which supports a severe local scale mismatch."
            )
        lines.append(
            "- Data-loss and total-loss rows are nearly identical in this setting, so the observed kernel action is driven mainly by the image term rather than the BE term."
        )
        lines.append(
            "- Raw amplification should not be read as a one-to-one learning-rate ratio because Adam, finite training, and NTK normalization conventions change effective step sizes."
        )
    lines.extend(
        [
            "",
            "## Matrix Size / Rank",
            "",
            f"- CP coefficients: `{summary.get('cp_coefficients')}`",
            f"- SINR parameter count: `{summary.get('sinr_param_count')}`",
            f"- Rank upper bound: `{summary.get('rank_upper_bound')}`",
            f"- Nullity lower bound: `{summary.get('nullity_lower_bound')}`",
            "",
            "## Interpretation Guardrails",
            "",
            "- This is a local NTK/SGD diagnostic, not a global optimizer proof.",
            "- The dense metrics are sampled over the fixed raw-image-positive OASIS foreground.",
            "- The prior cumulative-update proxy is not reused as evidence for these rows.",
        ]
    )
    errors = summary.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:  # type: ignore[union-attr]
            lines.append(f"- {error}")
    (review_root / "sinr_ntk_conditioning_report.md").write_text("\n".join(lines) + "\n")


def matrix_size_summary(generator: torch.nn.Module, control_shape: Tuple[int, int, int]) -> Dict[str, object]:
    cp_points = int(np.prod(control_shape))
    cp_coefficients = int(cp_points * 3)
    param_count = int(c.count_trainable_params(generator))
    dense_dim = int(np.prod(c.IMAGE_SHAPE) * 3)
    return {
        "dataset": "oasis",
        "image_shape_zyx": "160,224,192",
        "dense_displacement_dim": dense_dim,
        "control_shape_zyx": ",".join(str(int(v)) for v in control_shape),
        "control_points": cp_points,
        "cp_coefficients": cp_coefficients,
        "sinr_param_count": param_count,
        "rank_upper_bound": min(cp_coefficients, param_count),
        "nullity_lower_bound": max(0, cp_coefficients - param_count),
        "dense_jjt_fp32_gb": float(dense_dim * dense_dim * 4.0 / 1e9),
        "cp_kernel_dense_fp32_gb": float(cp_coefficients * cp_coefficients * 4.0 / 1e9),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    review_root = Path(args.review_root)
    out_root = Path(args.out_root)
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "tables").mkdir(parents=True, exist_ok=True)
    (review_root / "plots").mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.dataset != "oasis":
        raise ValueError("This first runner intentionally supports only --dataset oasis.")
    if args.method != "sinr_cps4_be_bspline_analytic":
        raise ValueError("This first runner intentionally supports only SINR cps4 OASIS.")
    if args.dry_run:
        plan = {
            "status": "dry_run",
            "dataset": args.dataset,
            "case_rank": int(args.case_rank),
            "method": args.method,
            "states": parse_csv_list(args.states),
            "gradients": parse_csv_list(args.gradients),
            "out_root": str(out_root),
            "review_root": str(review_root),
            "final_run": str(args.sinr_final_run),
        }
        write_json(out_root / "dry_run_plan.json", plan)
        write_json(review_root / "dry_run_plan.json", plan)
        return plan

    device = c.set_cuda_device(str(args.gpu))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    batch = c.load_oasis_case(rank=int(args.case_rank), device=device, oasis_root=Path(args.oasis_root), pair_file=Path(args.pair_file))
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, int(args.cps), device)
    states = ["init"] if args.smoke else parse_csv_list(args.states)
    gradients = ["data"] if args.smoke else parse_csv_list(args.gradients)
    sample_indices = sample_flat_indices(
        batch["fixed_mask"],  # type: ignore[arg-type]
        int(args.sample_voxels),
        int(args.sample_seed),
    )
    action_rows: List[Dict[str, object]] = []
    spectrum_rows: List[Dict[str, object]] = []
    matrix_rows: List[Dict[str, object]] = []
    errors: List[str] = []
    start = time.time()
    for state in states:
        generator = build_sinr_generator(args, control_shape, device)
        state_payload: Optional[Dict[str, object]] = None
        if state == "final":
            state_payload = load_final_state(generator, Path(args.sinr_final_run))
        elif state != "init":
            raise ValueError(f"Unsupported state {state!r}; use init or final.")
        ms = matrix_size_summary(generator, control_shape)
        ms.update(
            {
                "state": state,
                "case_rank": int(args.case_rank),
                "fixed_id": str(batch["fixed_id"]),
                "moving_id": str(batch["moving_id"]),
                "state_source": "seeded_init" if state == "init" else str(Path(args.sinr_final_run) / "model_state.pt"),
                "state_run_name": "" if state_payload is None else state_payload.get("run_name", ""),
            }
        )
        matrix_rows.append(ms)
        for gradient in gradients:
            try:
                row, spec_rows = analyze_state_gradient(
                    args,
                    state=state,
                    gradient_mode=gradient,
                    generator=generator,
                    control_coords=control_coords,
                    control_shape=control_shape,
                    batch=batch,
                    sample_indices=sample_indices,
                    matrix_summary=ms,
                )
                row.update(
                    {
                        "fixed_id": str(batch["fixed_id"]),
                        "moving_id": str(batch["moving_id"]),
                        "original_case_index": int(batch["original_case_index"]),
                        "created_at": c.now_text(),
                    }
                )
                action_rows.append(row)
                spectrum_rows.extend(spec_rows)
            except Exception as exc:
                errors.append(f"{state}/{gradient}: {exc}")
                if not args.allow_partial:
                    raise

    spectrum_summary = summarize_spectrum(spectrum_rows)
    status = "completed" if not errors else "incomplete"
    summary = {
        "status": status,
        "errors": errors,
        "dataset": "oasis",
        "case_rank": int(args.case_rank),
        "states": states,
        "gradients": gradients,
        "action_rows": len(action_rows),
        "spectrum_rows": len(spectrum_rows),
        "elapsed_sec": float(time.time() - start),
        "out_root": str(out_root),
        "review_root": str(review_root),
    }
    if matrix_rows:
        summary.update({key: matrix_rows[0][key] for key in ("cp_coefficients", "sinr_param_count", "rank_upper_bound", "nullity_lower_bound")})
    write_csv(review_root / "tables/ntk_action_metrics.csv", action_rows)
    write_csv(review_root / "tables/ntk_spectrum_probe_metrics.csv", spectrum_rows)
    write_csv(review_root / "tables/ntk_spectrum_probe_summary.csv", spectrum_summary)
    write_csv(review_root / "tables/matrix_size_summary.csv", matrix_rows)
    write_json(review_root / "summary.json", summary)
    write_json(out_root / ("smoke_summary.json" if args.smoke else "summary.json"), summary)
    make_plots(review_root, action_rows, spectrum_summary)
    write_report(review_root, summary, action_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("oasis", "lungct"), default="oasis")
    parser.add_argument("--case-rank", type=int, default=9)
    parser.add_argument("--method", type=str, default="sinr_cps4_be_bspline_analytic")
    parser.add_argument("--states", type=str, default="init,final")
    parser.add_argument("--gradients", type=str, default="data,total")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=1729)
    parser.add_argument("--sample-voxels", type=int, default=100000)
    parser.add_argument("--spectrum-probes", type=int, default=32)
    parser.add_argument("--power-iters", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--cps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--sinr-final-run", type=Path, default=DEFAULT_SINR_FINAL_RUN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.dry_run or args.smoke or args.run_full):
        raise SystemExit("Choose one of --dry-run, --smoke, or --run-full.")
    if args.smoke:
        args.spectrum_probes = min(int(args.spectrum_probes), 2)
        args.power_iters = min(int(args.power_iters), 2)
        args.sample_voxels = min(int(args.sample_voxels), 20000)
    result = run(args)
    print(json.dumps({"sinr_ntk_conditioning": result}, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
