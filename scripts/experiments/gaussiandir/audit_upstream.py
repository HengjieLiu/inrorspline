"""CPU-only synthetic audit of the pinned GaussianDIR source; no patient data.

This executes selected upstream AST definitions without importing its CUDA-only
constructor or unavailable dependencies. Tiny exact torch.cdist KNN replaces
PyTorch3D ONLY in this audit. It is not a registration backend or CUDA test.
Outputs must be written outside the repository under ARTIFACT_ROOT.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "external/GaussianDIR"


def definitions(filename, names):
    tree = ast.parse((UPSTREAM / filename).read_text())
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in names]
    if {node.name for node in body} != set(names):
        raise ValueError(f"Missing expected definitions in {filename}")
    return compile(ast.Module(body=body, type_ignores=[]), str(UPSTREAM / filename), "exec")


def tiny_knn(x, nodes, *_args, K):
    distances, indices = torch.cdist(x, nodes).square().topk(K, largest=False)
    return distances, indices, None


def run():
    torch.set_num_threads(1)
    namespace = dict(torch=torch, nn=nn, np=np, SimpleNamespace=SimpleNamespace,
                     ops=SimpleNamespace(knn_points=tiny_knn),
                     logger=SimpleNamespace(info=lambda *_args: None))
    exec(definitions("networks.py", ["GaussianWarp"]), namespace)
    Warp = namespace["GaussianWarp"]

    def fixture(*, anisotropy=False, rigid=False, k=2):
        # Bypass only device-specific allocation; methods below are upstream AST.
        model = Warp.__new__(Warp)
        nn.Module.__init__(model)
        model.K = k
        model.anisotropy = SimpleNamespace(scaling=anisotropy, quaternion=anisotropy)
        model.with_quaternion = rigid
        model.node_position = nn.Parameter(torch.tensor([[-.4, 0., 0.], [.4, 0., 0.]], dtype=torch.float64))
        model.translation = nn.Parameter(torch.tensor([[.1, .2, 0.], [-.1, .1, 0.]], dtype=torch.float64))
        identity = torch.tensor([[1., 0., 0., 0.]] * 2, dtype=torch.float64)
        model.quaternion = nn.Parameter(identity.clone())
        model.node_quaternion = nn.Parameter(identity.clone())
        model._node_radius = nn.Parameter(torch.zeros(2, 1, dtype=torch.float64))
        model._node_scaling = nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
        return model

    checks = {}
    x = torch.tensor([[.1, .2, .3], [.2, -.3, .1]], dtype=torch.float64)
    model = fixture()
    with torch.no_grad():
        model.translation[:] = torch.tensor([.2, -.1, .3])
    error = float((model(x) - model.translation[0]).abs().max())
    assert error < 1e-12
    checks["constant_translation_partition_of_unity"] = dict(max_error=error)

    model = fixture(anisotropy=True, rigid=True)
    q = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)], dtype=torch.float64)
    with torch.no_grad():
        model.node_quaternion[:] = q
        model.quaternion[:] = q
    weights, indices, _ = model.cal_nn_weight(x, model.node_position)
    original_delta = x[:, None] - model.node_position[indices]
    rotation = model.quaternion_to_matrix(model.quaternion)[indices]
    paper_local = (rotation @ original_delta[..., None]).squeeze(-1) - original_delta + model.translation[indices]
    paper = (weights[..., None] * paper_local).sum(1)
    error = float((model(x) - paper).abs().max())
    assert error > .1
    checks["anisotropic_rigid_forward_disagrees_with_paper"] = dict(max_error=error)

    model = fixture(rigid=True)
    with torch.no_grad():
        model.quaternion[:] = q
    model(x)[:, 0].sum().backward()
    autograd = float(model.node_position.grad[0, 0])
    eps = 1e-6
    with torch.no_grad():
        model.node_position[0, 0] += eps
        plus = float(model(x)[:, 0].sum())
        model.node_position[0, 0] -= 2 * eps
        minus = float(model(x)[:, 0].sum())
        model.node_position[0, 0] += eps
    fd = (plus - minus) / (2 * eps)
    assert abs(autograd - fd) > .1
    checks["rigid_center_detach_gradient_mismatch"] = dict(autograd=autograd, finite_difference=fd)

    model = fixture()
    model.contribution = torch.zeros(2, dtype=torch.float64)
    model.counter = torch.zeros(2, dtype=torch.float64)
    model.node_position.grad = torch.full_like(model.node_position, 10.)
    model.translation.grad = torch.zeros_like(model.translation)
    model.add_densification_stats()
    assert torch.count_nonzero(model.contribution) == 0
    checks["density_uses_translation_not_position_gradient"] = dict(accumulated=model.contribution.tolist())

    model = fixture()
    with torch.no_grad():
        model._node_radius[:] = -100
    weights = model.cal_nn_weight(x, model.node_position)[0]
    assert not torch.isfinite(weights).all()
    checks["weight_sum_underflow_not_guarded"] = dict(nonfinite_weights=int((~torch.isfinite(weights)).sum()))
    nan_nodes = torch.tensor([True, False])
    assert not nan_nodes.isnan().any()
    checks["boolean_isnan_does_not_prune_nan_nodes"] = dict(result=nan_nodes.isnan().tolist())

    model = fixture()
    model.max_contribution, model.min_contribution, model.max_densify_num = 100., 1e-7, 10
    model.contribution = torch.tensor([1., 0.])
    model.counter = torch.ones(2)
    # Replacing a CUDA-only buffer reset leaves the optimizer surgery untouched.
    model.densification_postfix = lambda: None
    optimizer = torch.optim.Adam([
        dict(params=[model.node_position], name="node_position"),
        dict(params=[model.translation], name="translation"),
        dict(params=[model._node_radius], name="_node_radius"),
    ], lr=.01)
    for group in optimizer.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
    optimizer.step()  # Create Adam state as in a real adaptation event after warmup.
    optimizer.zero_grad()
    model(x).sum().backward()
    before = model.translation[0].detach().clone()
    model.densify_and_prune(optimizer)
    missing_gradients = all(group["params"][0].grad is None for group in optimizer.param_groups)
    optimizer.step()
    change = float((model.translation[0] - before).abs().max())
    assert missing_gradients and change == 0.
    checks["prune_before_adam_discards_current_update"] = dict(all_gradients_missing=missing_gradients, survivor_change=change)

    namespace.update(Tensor=torch.Tensor, _Loss=nn.modules.loss._Loss)
    exec(definitions("losses.py", ["StableStd", "NCC"]), namespace)
    values = torch.arange(8, dtype=torch.float64)
    ncc = float(namespace["NCC"]()(values, values))
    assert abs(ncc + 7 / 8) < 1e-9
    checks["ncc_population_covariance_sample_std"] = dict(perfect_correlation_loss=ncc, pearson_loss=-1.)

    tree = ast.parse((UPSTREAM / "registrators.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "tv_regulizer")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "tv_regulizer", "exec"), namespace)
    fixture_self = SimpleNamespace(dataset=SimpleNamespace(neighs=3))
    # One center, three neighbors; each displacement difference has norm one.
    flow = torch.cat([torch.zeros(1, 3), torch.eye(3)])
    tv = float(namespace["tv_regulizer"](fixture_self, flow))
    assert tv == 1.
    checks["tv_means_over_directions"] = dict(code=tv, paper_sum_per_center=3.)

    model = fixture(k=1)
    points = torch.tensor([[-1e-8, 0., 0.], [1e-8, 0., 0.]], dtype=torch.float64)
    jump = float(torch.linalg.vector_norm(model(points)[0] - model(points)[1]))
    assert jump > .1
    checks["hard_knn_can_be_discontinuous"] = dict(k=1, query_separation=2e-8, displacement_jump=jump,
        scope="constructive counterexample, not a measured K=20 patient-field discontinuity")

    sizes = torch.tensor([192., 224., 160.], dtype=torch.float64)
    index = torch.tensor([11., 23., 37.], dtype=torch.float64)
    voxel_displacement = torch.tensor([.3, -1.2, 2.1], dtype=torch.float64)
    true_coordinates = 2 * index / (sizes - 1) - 1
    false_coordinates = (true_coordinates * (sizes - 1)) / sizes
    expected_false = 2 * (index + .5) / sizes - 1
    error = float((false_coordinates - expected_false).abs().max())
    false_displacement = 2 * voxel_displacement / sizes
    recovered_true_displacement = false_displacement * sizes / (sizes - 1)
    assert error < 1e-12
    assert torch.allclose(recovered_true_displacement, 2 * voxel_displacement / (sizes - 1))
    checks["align_corners_conversion"] = dict(coordinate_max_error=error)

    hashes = {f: hashlib.sha256((UPSTREAM / f).read_bytes()).hexdigest()
              for f in ("networks.py", "registrators.py", "losses.py", "datasets.py", "utils.py",
                        "configs/gaussian.yaml", "configs/network/gaussian.yaml")}
    return dict(scope="CPU synthetic characterization; no native imports, CUDA, images, or registration",
                source_sha256=hashes, torch_version=torch.__version__, checks=checks,
                checks_passed=len(checks), interpretation="Assertions confirm observed behavior, including defects; not an upstream correctness certificate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(REPO):
        raise ValueError("Output must be external and beneath ARTIFACT_ROOT")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
