#!/usr/bin/env python
"""Serial GPU0 native-volume resource gates before admitting P81 workers.

Every profile has its own process/log/receipt. A failed profile is recorded and
the remaining profiles continue. This command never starts the production queue.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from basis import create_basis
from data import load_target, indices_to_mm
from numerics import (solve_d0, SampledLinearOperator, render_field, optimize_nonlinear,
                      sample_field_points, grid_lipschitz_bound)
from runtime import (FAMILIES, HERE, atomic_json, read_json, sha, now, code_hashes,
                     assert_external)

MODES = {"d0": "direct_dvf", "svf": "svf", "d1": "bounded_dvf"}
D1_FAMILIES = ("bspline", "tree_affine_uniform", "tree_affine_adaptive")


def scientific_source_hashes():
    """Files exercised by these profiles; queue/report edits cannot change math."""
    names = ("basis.py", "data.py", "numerics.py", "preflight.py", "runtime.py")
    return {name: sha(HERE/name) for name in names}


def profiles():
    return [dict(mode=mode, family=family, budget_cap=3072,
                 key=f"{mode}__{family}") for mode in ("d0", "svf") for family in FAMILIES] + [
        dict(mode="d1", family=family, budget_cap=1536, key=f"d1__{family}") for family in D1_FAMILIES]


def required_memory_mib(peak_reserved_mib, *, allowance_mib=2048, rounding_mib=256):
    """Per-job peak plus allocator/interpolation/evaluation allowance."""
    if not math.isfinite(peak_reserved_mib) or peak_reserved_mib < 0:
        raise ValueError("Invalid peak memory measurement")
    return int(math.ceil(max(2048, peak_reserved_mib+allowance_mib)/rounding_mib)*rounding_mib)


def coarse_geometry(shape, affine, factor=4):
    coarse = tuple(max(2, int(round((n-1)/factor))+1) for n in shape)
    h = np.array(affine, dtype=np.float64, copy=True)
    h[:3, :3] = h[:3, :3] @ np.diag([(n-1)/(m-1) for n, m in zip(shape, coarse)])
    return coarse, h


def _setup_cuda():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Preflight requires CUDA_VISIBLE_DEVICES=0 (physical GPU0)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _make_profile_basis(spec, record, points, targets):
    adaptive = spec["family"].endswith("_adaptive")
    basis = create_basis(spec["family"], int(spec["budget_cap"]),
                         record["domain_min"], record["domain_max"], start_root=adaptive)
    if adaptive:
        coeff = points.new_zeros((basis.feature_count, 3))
        while basis.leaf_count < basis.target_leaf_count:
            before = basis.leaf_count
            basis, coeff, event = basis.split(coeff, points[:4096], -targets[:4096],
                                              max_leaves=basis.target_leaf_count, growth=.25)
            if basis.leaf_count == before:
                raise RuntimeError("Profile tree cannot reach its declared maximum budget")
    return basis


def profile_one(root, spec, *, render_chunk=16384):
    root = assert_external(root)
    folder = root/"preflight"/spec["key"]
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/"profile.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite profile receipt")
    _setup_cuda()
    started = time.monotonic()
    stages = []
    snapshot = scientific_source_hashes()

    def progress(phase, **extra):
        payload = dict(phase=phase, at=now(), elapsed_sec=time.monotonic()-started,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        payload.update(extra)
        atomic_json(folder/"progress.json", payload)
        print(f"{spec['key']}: {phase}", flush=True)

    try:
        inputs = read_json(root/"input_manifest.json")
        matches = [t for t in inputs["targets"] if t.get("pair_rank") == 0
                   and t.get("source_method") == "sitreg" and t.get("status") == "passed"]
        if len(matches) != 1:
            raise RuntimeError("Need one passed rank0 SITReg target")
        record = matches[0]
        if sha(record["canonical_path"]) != record["canonical_sha256"]:
            raise RuntimeError("Canonical target changed before resource profile")
        data = load_target(record)
        device = torch.device("cuda:0")
        points = torch.tensor(data["points_mm"], device=device, dtype=torch.float64)
        targets = torch.tensor(data["targets_mm"], device=device, dtype=torch.float64)
        shape, affine = tuple(data["shape"]), data["affine"]
        basis = _make_profile_basis(spec, record, points, targets)
        progress("basis_ready", feature_count=basis.feature_count,
                 actual_motion_params=basis.actual_motion_params, leaf_count=basis.leaf_count)

        # Native fitting-sample operator memory, including possible LS warm starts.
        then = time.monotonic()
        solved, ls_meta = solve_d0(basis, points, targets,
            config={"iter_lim": 1, "chunk": render_chunk, "cache_bytes": 512*1024**2},
            callback=lambda row: progress("lsqr_one_iteration", **row))
        torch.cuda.synchronize()
        stages.append(dict(stage="sample_lsqr_one_iteration", elapsed_sec=time.monotonic()-then,
                           solver=ls_meta, intentionally_unconverged_profile=True))
        del solved
        # Forward/adjoint test is a separate tiny fixture, not a convergence claim.
        operator = SampledLinearOperator(basis, points[:257], chunk=64, cache_bytes=0)
        x = torch.randn(operator.m, device=device, dtype=torch.float64)
        y = torch.randn(operator.n, device=device, dtype=torch.float64)
        lhs, rhs = torch.dot(operator.forward(x), y), torch.dot(x, operator.adjoint(y))
        adjoint_error = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1))
        if adjoint_error > 1e-10:
            raise RuntimeError(f"GPU operator adjoint failure {adjoint_error}")
        del operator, x, y, lhs, rhs
        gc.collect(); torch.cuda.empty_cache()

        # Nonzero coefficients exercise coordinate gradients of the integrator.
        initial = torch.randn((basis.feature_count, 3), device=device, dtype=torch.float32)*.002
        with torch.no_grad():
            sample_prediction = basis.evaluate(points, initial.double())
            if not torch.isfinite(sample_prediction).all():
                raise FloatingPointError("Nonfinite full-sample analytic prediction")
            del sample_prediction
        gc.collect(); torch.cuda.empty_cache()
        progress("native_adam_step_start")
        then = time.monotonic()
        coefficients, optimizer = optimize_nonlinear(basis, initial, shape, affine,
            fit_targets_mm=targets.float(), fit_points_mm=points.float(), mode=MODES[spec["mode"]],
            config={"max_steps": 1, "min_steps": 0, "check_interval": 1,
                    "chunk": render_chunk, "steps": 7, "lr": .03},
            callback=lambda row: progress("native_adam_check", **row))
        torch.cuda.synchronize()
        native_step_elapsed = time.monotonic()-then
        stages.append(dict(stage="native_adam_one_update_plus_initial_final_render",
                           elapsed_sec=native_step_elapsed, optimizer=optimizer))

        with torch.no_grad():
            native = render_field(basis, initial, shape, affine, MODES[spec["mode"]],
                                  chunk=render_chunk, steps=7)
            if not torch.isfinite(native).all():
                raise FloatingPointError("Nonfinite native rendered field")
            index = torch.as_tensor(np.asarray(data["fit_indices"])[:257], device=device)
            point_mm = points[:257].float()
            sampled = sample_field_points(native, point_mm, affine)
            grid_values = native.reshape(-1, 3)[index]
            sample_parity = float((sampled-grid_values).abs().max())
            if sample_parity > 2e-4:
                raise RuntimeError(f"Native physical-point/voxel-center mismatch: {sample_parity}")
            if spec["mode"] == "svf":
                progress("svf_7_8_convergence")
                extra = render_field(basis, initial, shape, affine, "svf", chunk=render_chunk, steps=8)
                diff = extra-native
                integration = dict(rmse_mm=float(diff.double().square().sum(-1).mean().sqrt()),
                                   max_component_mm=float(diff.abs().max()),
                                   test="fixed_small_nonzero_coefficients_native7_vs8")
                if integration["rmse_mm"] > 1e-4 or integration["max_component_mm"] > .005:
                    raise RuntimeError(f"Small-field native7/8 convergence gate failed: {integration}")
                del extra, diff
            else:
                integration = None
            if spec["mode"] == "d1":
                value, certificate = grid_lipschitz_bound(native, affine, return_metadata=True)
                certificate["bound_value"] = value
                if value > .9:
                    raise RuntimeError("Native D1 physical all-cell certificate failed")
            else:
                certificate = None
            del native

        # Explicit affine representer parity on endpoint-aligned quarter grid.
        # Non-affine fields are not expected to agree after coarse interpolation.
        affine_basis = create_basis("global_affine", 12, record["domain_min"], record["domain_max"])
        affine_coeff = torch.tensor([[.02, -.01, .03], [.01, .002, -.001],
                                    [.003, .01, .004], [.002, -.003, .01]], device=device)
        coarse_shape, coarse_h = coarse_geometry(shape, affine, 4)
        with torch.no_grad():
            coarse = render_field(affine_basis, affine_coeff, coarse_shape, coarse_h,
                                  "direct_dvf", chunk=render_chunk)
            actual = sample_field_points(coarse, points[:1024].float(), coarse_h)
            exact = affine_basis.evaluate(points[:1024].float(), affine_coeff)
            coarse_error = float((actual-exact).abs().max())
            if coarse_error > 1e-5:
                raise RuntimeError("Endpoint-aligned coarse physical affine parity failed")
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()/2**20
        if scientific_source_hashes() != snapshot:
            raise RuntimeError("Scientific source changed during native profile")
        result = dict(status="passed", profile=spec, created_at=now(), code_hashes=snapshot,
                      input_manifest_sha256=sha(root/"input_manifest.json"), target_id=record["id"],
                      target_sha256=record["canonical_sha256"], shape=list(shape),
                      actual_motion_params=basis.actual_motion_params, leaf_count=basis.leaf_count,
                      profile_geometry="reference_residual_without_refits" if basis.is_adaptive else "fixed_uniform",
                      sample_count=len(points), render_chunk=render_chunk, stages=stages,
                      adjoint_relative_error=adjoint_error, native_sample_parity_max_mm=sample_parity,
                      coarse_affine_parity_max_mm=coarse_error, integration=integration,
                      certificate=certificate, peak_reserved_mib=peak,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      required_mib=required_memory_mib(peak), elapsed_sec=time.monotonic()-started,
                      native_one_update_profile_sec=native_step_elapsed,
                      estimated_1000_native_updates_sec=native_step_elapsed*1000,
                      estimator_note="Conservative repeated one-update profile includes two renders; not measured production runtime.",
                      requested_backend=dict(cudnn_benchmark=False,cudnn_deterministic=True,tf32=False),
                      effective_backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                          cudnn_deterministic=torch.backends.cudnn.deterministic,
                          tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),
                      torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0),
                      limitations=["One update profiles resources; it does not certify optimizer convergence.",
                                   "Integration convergence fixture uses small coefficients; fitted fields need separate checks.",
                                   "Adaptive profile geometry is target-residual selected without optimization/refits."])
        atomic_json(output, result)
        progress("passed")
        return result
    except BaseException as exc:
        failure = dict(status="failed", profile=spec, created_at=now(), code_hashes=snapshot,
                       error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc(),
                       stages=stages, elapsed_sec=time.monotonic()-started)
        atomic_json(output, failure)
        raise


def summarize(receipts):
    expected = {s["key"] for s in profiles()}
    found = {r["profile"]["key"] for r in receipts}
    failed = [r["profile"]["key"] for r in receipts if r.get("status") != "passed"]
    memory = {r["profile"]["key"]: r["required_mib"] for r in receipts if r.get("status") == "passed"}
    # The small affine anchor shares the largest D0 reservation conservatively.
    if memory:
        memory["d0__global_affine"] = max(v for k,v in memory.items() if k.startswith("d0__")) if any(k.startswith("d0__") for k in memory) else max(memory.values())
    duplicates = len(receipts) != len(found)
    return dict(passed=found == expected and not failed and not duplicates, expected_profiles=len(expected),
                completed_profiles=len(found), missing_profiles=sorted(expected-found),
                failed_profiles=failed, duplicate_receipts=duplicates, memory_profiles_mib=memory,
                estimated_sum_1000_native_update_sec=sum(r.get("estimated_1000_native_updates_sec",0) for r in receipts if r.get("status")=="passed"))


def run(root, *, profile_timeout_sec=1800, render_chunk=16384):
    root = assert_external(root)
    folder = root/"preflight"
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    start_hashes = scientific_source_hashes()
    for spec in profiles():
        child = folder/spec["key"]
        child.mkdir(exist_ok=True)
        receipt = child/"profile.json"
        if receipt.exists():
            prior = read_json(receipt)
            if prior.get("code_hashes") != start_hashes:
                raise RuntimeError("Existing preflight receipt belongs to different code; use a new preflight run")
            receipts.append(prior)
            continue
        print(f"Starting serial GPU0 profile {spec['key']}", flush=True)
        try:
            with (child/"profile.log").open("xb") as log:
                subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--root",str(root),
                                "--profile-key",spec["key"],"--render-chunk",str(render_chunk)],
                               stdout=log,stderr=subprocess.STDOUT,check=False,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONDONTWRITEBYTECODE="1"),
                               timeout=profile_timeout_sec)
            if not receipt.exists():
                raise RuntimeError("Profile child exited without a receipt")
        except Exception as exc:
            if not receipt.exists():
                atomic_json(receipt,dict(status="failed",profile=spec,created_at=now(),
                            code_hashes=start_hashes,error=type(exc).__name__+": "+str(exc)))
        receipts.append(read_json(receipt))
        atomic_json(folder/"progress.json",dict(updated_at=now(),**summarize(receipts)))
    summary = dict(created_at=now(),code_hashes=start_hashes,
                   input_manifest_sha256=sha(root/"input_manifest.json"),
                   profiles=[dict(profile=r["profile"],status=r["status"],receipt=str(folder/r["profile"]["key"]/"profile.json")) for r in receipts],
                   **summarize(receipts))
    svf_cost = []
    for result in receipts:
        spec = result["profile"]
        if result.get("status") == "passed" and spec["mode"] == "svf":
            jobs = 80 if spec["family"].startswith("tree_") else 240
            svf_cost.append(dict(family=spec["family"], jobs_including_two_initializations=jobs,
                path_update_budget=1000, native_one_update_profile_sec=result["native_one_update_profile_sec"],
                native_only_serial_work_proxy_sec=jobs*1000*result["native_one_update_profile_sec"]))
    summary["runtime_estimator"] = dict(svf_main=svf_cost,
        native_only_serial_work_proxy_sec=sum(r["native_only_serial_work_proxy_sec"] for r in svf_cost),
        caveat="Largest-budget native one-update fixture includes two renders; coarse stages are cheaper. This is a conservative work proxy, not a completion-time promise or an eight-fold parallel speedup estimate.",
        d0="LSQR iteration count/convergence must be measured; setup-dominated one-iteration profiles do not support linear full-study extrapolation.")
    atomic_json(folder/"summary.json",summary)
    # Root orchestration must combine this resource gate with the data/unit/
    # worker/report gates before writing PREFLIGHT_PASSED.json.
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--profile-key")
    parser.add_argument("--render-chunk",type=int,default=16384)
    parser.add_argument("--profile-timeout-sec",type=int,default=1800)
    args=parser.parse_args()
    if args.profile_key:
        matches=[s for s in profiles() if s["key"]==args.profile_key]
        if len(matches)!=1:raise ValueError("Unknown native profile")
        profile_one(args.root,matches[0],render_chunk=args.render_chunk)
    else:
        result=run(args.root,profile_timeout_sec=args.profile_timeout_sec,render_chunk=args.render_chunk)
        print(result,flush=True)
        if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":
    main()
