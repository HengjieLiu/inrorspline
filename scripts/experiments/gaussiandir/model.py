"""Pinned released control and explicit paper reconstruction for P83.

Coordinates and displacements are grid XYZ with align_corners=False. The paper
model uses spatial deltas for local rigid motion and covariance-frame deltas
only for weights. The upstream tree is never edited.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "external/GaussianDIR"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def upstream():
    logger_module = _module("gaussiandir_source_logger", UPSTREAM / "logger.py")
    previous = sys.modules.get("logger")
    sys.modules["logger"] = logger_module
    try:
        network = _module("gaussiandir_source_network", UPSTREAM / "networks.py")
    finally:
        if previous is None:
            sys.modules.pop("logger", None)
        else:
            sys.modules["logger"] = previous
    loss = _module("gaussiandir_source_loss", UPSTREAM / "losses.py")
    return network, loss


def knn_indices(x, centers, k, backend="pytorch3d"):
    if centers.shape[0] < k or k < 1:
        raise ValueError(f"Require at least K={k} nodes, got {centers.shape[0]}")
    with torch.no_grad():
        if backend == "brute_test":
            # Only for small independent numerical tests, never production.
            return torch.cdist(x, centers).topk(k, largest=False).indices
        if backend != "pytorch3d":
            raise ValueError(backend)
        from pytorch3d.ops import knn_points
        return knn_points(x.detach()[None], centers.detach()[None], K=k).idx[0]


def quaternion_matrix(q):
    """Scalar-first quaternion, same rotation convention as the paper/release."""
    r, i, j, k = q.unbind(-1)
    two_s = 2 / q.square().sum(-1)
    return torch.stack((1-two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r),
                        two_s*(i*j+k*r), 1-two_s*(i*i+k*k), two_s*(j*k-i*r),
                        two_s*(i*k-j*r), two_s*(j*k+i*r), 1-two_s*(i*i+j*j)), -1).reshape(*q.shape[:-1], 3, 3)


def grid_initialization(mask, sparsification, dtype=torch.float32):
    shape = [int(n / sparsification - 1) for n in mask.shape]
    if min(shape) < 2:
        raise ValueError("Initialization needs at least two nodes along every axis")
    # Match the release arithmetic and initial width exactly.
    axes = [torch.linspace(-1, 1, n+1, dtype=dtype)[:-1] + 1/n for n in shape]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], -1).reshape(-1, 3)
    diagonal = shape[1]*shape[2] + shape[2] + 1
    radius = torch.linalg.vector_norm(coordinates[diagonal]-coordinates[0])*.5
    keep = F.grid_sample(mask.cpu().to(dtype)[None, None], coordinates[None, None, None],
                         mode="nearest", align_corners=False).reshape(-1).bool()
    return coordinates[keep], radius, shape


def stable_release_weights(self, x, nodes=None, K=None):
    """Algebraically equivalent isotropic release weights, avoiding tail 0/0.

The release's determinant epsilon is retained. No change to its motion,
gradient statistic, cloning, pruning or optimizer ordering is hidden here.
"""
    nodes = self.node_position if nodes is None else nodes
    indices = knn_indices(x, nodes, self.K if K is None else K)
    delta = x[:, None] - nodes[indices]
    log_radius = self._node_radius[indices].squeeze(-1)
    exponent = (delta / log_radius.exp()[..., None]).square().sum(-1)
    log_volume = torch.logaddexp(3*log_radius, log_radius.new_tensor(math.log(1e-7)))
    weights = torch.softmax(-.5*exponent-log_volume, -1)
    return weights, indices, delta


def released_layer(mask, sparsification, k, device):
    source, _ = upstream()
    layer = source.GaussianWarp(K=k, node_shape=[10]*3, max_densify_num=5000,
        sparsification=sparsification, max_contribution=.002, min_contribution=1e-7,
        with_quaternion=False, anisotropy=dict(scaling=False, quaternion=False))
    layer.reinitialize(mask)
    layer.to(device)
    layer.cal_nn_weight = MethodType(stable_release_weights, layer)
    layer.variant = "released_control"
    layer.backend = "pytorch3d"
    validate_population(layer)
    return layer


class PaperLayer(nn.Module):
    def __init__(self, centers, radius, k, cap, *, backend="pytorch3d"):
        super().__init__()
        n = len(centers)
        self.K, self.max_densify_num, self.backend = k, int(cap), backend
        self.variant = "paper_reconstruction"
        self.node_initial_num = n
        self.node_position = nn.Parameter(centers.clone())
        self.translation = nn.Parameter(torch.zeros_like(centers))
        self._node_scaling = nn.Parameter(torch.full_like(centers, math.log(float(radius))))
        identity = centers.new_zeros((n, 4))
        identity[:, 0] = 1
        self.node_quaternion = nn.Parameter(identity.clone())
        self.quaternion = nn.Parameter(identity.clone())
        self.register_buffer("contribution", centers.new_zeros(n))
        self.register_buffer("counter", centers.new_zeros(n))
        self.max_contribution, self.min_contribution = .002, 1e-7
        validate_population(self)

    def forward(self, x):
        indices = knn_indices(x, self.node_position, self.K, self.backend)
        delta = x[:, None] - self.node_position[indices]
        covariance_rotation = quaternion_matrix(self.node_quaternion)[indices]
        covariance_delta = (delta[..., None, :] @ covariance_rotation).squeeze(-2)
        logs = self._node_scaling[indices]
        exponent = (covariance_delta / logs.exp()).square().sum(-1)
        weights = torch.softmax(-.5*exponent-logs.sum(-1), -1)
        motion_rotation = quaternion_matrix(self.quaternion)[indices]
        local = (motion_rotation @ delta[..., None]).squeeze(-1) - delta + self.translation[indices]
        return (weights[..., None]*local).sum(1)

    def trained_parameters(self, lr):
        return [dict(params=[value], name=name, lr=getattr(lr, name.removeprefix("_")
                    if name.startswith("_") else name)) for name, value in self.named_parameters()]

    @torch.no_grad()
    def add_densification_stats(self):
        if self.node_position.grad is None:
            raise RuntimeError("Missing center gradients for paper adaptation")
        self.contribution.add_(self.node_position.grad.norm(dim=-1))
        self.counter.add_(1)

    @torch.no_grad()
    def densify_and_prune(self, optimizer):
        score = self.contribution / self.counter.clamp_min(1)
        if not torch.isfinite(score).all():
            raise FloatingPointError("Nonfinite position-gradient statistics")
        keep = score > self.min_contribution
        keep_indices = keep.nonzero().flatten()
        if len(keep_indices) < self.K:
            raise RuntimeError(f"Paper pruning would leave {len(keep_indices)} < K={self.K}; refusing silent protection")
        clone = ((score >= self.max_contribution) & keep).nonzero().flatten()
        slots = self.max_densify_num - len(keep_indices)
        if slots < 0:
            raise RuntimeError("Population exceeds configured cap")
        if len(clone) > slots:
            clone = clone[score[clone].argsort(descending=True)[:slots]]
        old_n = len(score)
        if len(keep_indices) != old_n or len(clone):
            for group in optimizer.param_groups:
                name = group["name"]
                old = group["params"][0]
                extension = old[clone].clone()
                if name == "node_position" and len(clone):
                    # Supplement specifies axis-aligned N(0,s), as the release.
                    extension += torch.normal(torch.zeros_like(extension), self._node_scaling[clone].exp())
                new = nn.Parameter(torch.cat((old[keep_indices], extension), 0))
                state = optimizer.state.pop(old, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key] = torch.cat((state[key][keep_indices], torch.zeros_like(extension)), 0)
                group["params"][0] = new
                setattr(self, name, new)
                if state:
                    optimizer.state[new] = state
        n = len(self.node_position)
        self.contribution = self.node_position.new_zeros(n)
        self.counter = self.node_position.new_zeros(n)
        return dict(before=old_n, pruned=old_n-len(keep_indices), cloned=len(clone), after=n)


def validate_population(layer):
    n = len(layer.node_position)
    if n < layer.K or layer.max_densify_num < n:
        raise ValueError(f"Invalid node count/cap: N={n}, K={layer.K}, cap={layer.max_densify_num}")


def check_finite(layer, *, gradients=False):
    for name, param in layer.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Nonfinite parameter {name}")
        if "quaternion" in name and torch.any(param.square().sum(-1) <= 1e-20):
            raise FloatingPointError(f"Degenerate {name}")
        if "scaling" in name or "radius" in name:
            if not torch.isfinite(param.exp()).all() or torch.any(param.exp() <= 0):
                raise FloatingPointError(f"Nonpositive/nonfinite scale {name}")
        if gradients and (param.grad is None or not torch.isfinite(param.grad).all()):
            raise FloatingPointError(f"Missing/nonfinite gradient {name}")


class GaussianModel(nn.Module):
    def __init__(self, variant, config, mask, device, *, backend="pytorch3d"):
        super().__init__()
        self.variant, self.config, self.backend = variant, config, backend
        self.layers = nn.ModuleList()
        self.initial_counts = []
        self.domain_voxels = int(mask.sum())
        self.add_stage(mask, device)

    def add_stage(self, mask, device):
        index = len(self.layers)
        settings = self.config[self.variant]
        sparsification = settings["sparsifications"][index]
        for layer in self.layers:
            layer.requires_grad_(False)
        if self.variant == "released_control":
            if index:
                raise ValueError("Released control has one stage")
            layer = released_layer(mask, sparsification, self.config["knn_k"], device)
        else:
            centers, radius, _ = grid_initialization(mask, sparsification)
            total_cap = self.domain_voxels // settings["total_population_cap_divisors"][index]
            cap = total_cap - sum(len(l.node_position) for l in self.layers)
            layer = PaperLayer(centers.to(device), radius, self.config["knn_k"], cap, backend=self.backend)
        layer.max_contribution = self.config["adaptive_thresholds"]["clone"]
        layer.min_contribution = self.config["adaptive_thresholds"]["prune"]
        self.layers.append(layer)
        self.initial_counts.append(len(layer.node_position))

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)

    @property
    def active(self):
        return self.layers[-1]

    def optimizer(self, *, joint=False):
        lr = {key.removeprefix("_"): value for key, value in self.config["learning_rates"].items()}
        if joint:
            self.requires_grad_(True)
        layers = self.layers if joint else [self.active]
        groups = []
        for index, layer in enumerate(layers):
            for group in layer.trained_parameters(SimpleNamespace(**lr)):
                group["layer_index"] = index if joint else len(self.layers)-1
                groups.append(group)
        return torch.optim.Adam(groups)

    def counts(self):
        return dict(nodes=sum(len(l.node_position) for l in self.layers),
                    nodes_by_stage=[len(l.node_position) for l in self.layers],
                    caps_by_stage=[int(l.max_densify_num) for l in self.layers],
                    parameters=sum(p.numel() for p in self.parameters()),
                    active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def payload(self):
        return dict(variant=self.variant, config=self.config, domain_voxels=self.domain_voxels,
                    initial_counts=self.initial_counts, backend=self.backend,
                    layers=[dict(parameters={k:v.detach().cpu().clone() for k,v in l.named_parameters()},
                                 requires_grad={k:v.requires_grad for k,v in l.named_parameters()},
                                 contribution=l.contribution.detach().cpu(), counter=l.counter.detach().cpu(),
                                 cap=l.max_densify_num, initial=int(l.node_initial_num)) for l in self.layers])

    @classmethod
    def restore(cls, payload, device):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        for key in ("variant", "config", "domain_voxels", "initial_counts", "backend"):
            setattr(model, key, payload[key])
        model.layers = nn.ModuleList()
        for index, data in enumerate(payload["layers"]):
            centers = data["parameters"]["node_position"].to(device)
            if model.variant == "released_control":
                source, _ = upstream()
                layer = source.GaussianWarp.__new__(source.GaussianWarp)
                nn.Module.__init__(layer)
                layer.K = model.config["knn_k"]
                layer.anisotropy = SimpleNamespace(scaling=False, quaternion=False)
                layer.with_quaternion = False
                layer.cal_nn_weight = MethodType(stable_release_weights, layer)
            else:
                layer = PaperLayer(centers, 1., model.config["knn_k"], data["cap"], backend=model.backend)
            for name, value in data["parameters"].items():
                trainable = data.get("requires_grad", {}).get(name, index==len(payload["layers"])-1)
                setattr(layer, name, nn.Parameter(value.to(device), requires_grad=trainable))
            layer.contribution, layer.counter = data["contribution"].to(device), data["counter"].to(device)
            layer.max_densify_num, layer.node_initial_num = data["cap"], data["initial"]
            layer.max_contribution = model.config["adaptive_thresholds"]["clone"]
            layer.min_contribution = model.config["adaptive_thresholds"]["prune"]
            model.layers.append(layer)
        return model


def paper_ncc(a, b):
    a, b = a-a.mean(), b-b.mean()
    return -(a*b).mean() / (torch.sqrt(a.square().mean()*b.square().mean())+1e-10)


def tv_loss(flow, variant):
    if len(flow) % 4:
        raise ValueError("TV queries must contain centers followed by three neighbors per center")
    n = len(flow)//4
    norms = (flow[n:].reshape(n, 3, 3)-flow[:n, None]).norm(dim=-1)
    return norms.mean() if variant == "released_control" else norms.sum(-1).mean()


def lr_multiplier(step, maximum, warmup):
    if step <= warmup:
        return step / warmup
    return .5*(1+math.cos(math.pi*(step-warmup)/(maximum+1-warmup)))


def adaptation_due(variant, step, maximum, config):
    settings = config[variant]
    if variant == "released_control":
        return (settings["adaptation_start_step"] <= step <= settings["adaptation_end_step"]
                and step % settings["adaptation_interval"] == 0)
    return (settings["adaptation_start_local_step"] <= step < maximum
            and step % settings["adaptation_interval"] == 0)
