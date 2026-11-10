"""Paper-loss convergence/resource worker. Registration never transfers labels/targets."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .convergence import MonitoringBanks, Plateau, out_of_time, schedule
from .data import Sampler, atomic_json, external, flat_coordinates, load_inputs, sample_volume, sha256
from .model import GaussianModel, REPO, PaperLayer, check_finite, lr_multiplier, paper_ncc, upstream
from .resources import NvmlSampler, Resources
from .run import environment, render, save_checkpoint, source_receipts, training_step


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_configuration(path):
    settings = json.loads(path.read_text())
    base = json.loads((REPO/settings["base_protocol"]).read_text())
    if base["unsupervised"].get("lambda_tv") != 8. or base["unsupervised"].get("loss"):
        raise ValueError("This study requires the original sampled signed NCC/TV lambda8")
    return base, settings


def warm_kernels():
    """Synthetic generic kernels only; no registration update or reusable fit."""
    began = time.monotonic()
    centers = torch.linspace(-.8, .8, 120, device="cuda").reshape(40, 3)
    layer = PaperLayer(centers, .3, 20, 40).cuda()
    x = torch.linspace(-.9, .9, 240, device="cuda").reshape(80, 3)
    flow = layer(x)
    volume = torch.arange(8**3, device="cuda", dtype=torch.float32).reshape(8, 8, 8)
    loss = upstream()[1].NCC()(sample_volume(volume, x+flow), sample_volume(volume, x))
    loss.backward()
    torch.cuda.synchronize()
    del layer, x, volume, flow, loss
    gc.collect()
    torch.cuda.empty_cache()
    return time.monotonic()-began


@torch.no_grad()
def usable_output(model, inputs, config):
    shape = inputs["shape"]
    field = render(model, shape, config["render_chunk"])
    image = np.empty(shape, np.float32)
    flat = image.reshape(-1)
    displacement = torch.from_numpy(field.reshape(-1, 3))
    for start in range(0, len(flat), config["render_chunk"]):
        end = min(start+config["render_chunk"], len(flat))
        ids = torch.arange(start, end, device="cuda")
        query = flat_coordinates(ids, shape)+displacement[start:end].cuda()
        flat[start:end] = sample_volume(inputs["moving"], query).cpu().numpy()
    if not np.isfinite(image).all():
        raise FloatingPointError("Nonfinite usable warped image")
    return field, image


def run_one(args, repeat, warmup_seconds):
    base, settings = load_configuration(args.config)
    config = schedule(base, args.budget, smoke=args.smoke)
    if args.smoke:
        settings = copy.deepcopy(settings)
        settings["refinement"].update(monitor_interval=2, window_steps=4, checkpoint_interval=4,
            monitor_batches=2, probe_points=256, warmup_steps=2)
        config["log_interval"] = 1
    out = external(args.out if args.repeats == 1 else args.out/f"repeat{repeat+1}")
    out.mkdir(parents=True, exist_ok=False)
    sampler_nvml = NvmlSampler(args.gpu_uuid, settings["memory_sampling_seconds"])
    try:
        return _run(args, config, settings, out, repeat, warmup_seconds, sampler_nvml)
    except Exception as exc:
        atomic_json(out/"failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        sampler_nvml.close()


def _run(args, config, settings, out, repeat, warmup_seconds, nvml):
    # Immutable inputs are checked on CPU; target and labels never enter inputs_gpu.
    setup_began = time.monotonic()
    inputs = load_inputs(config)
    shape = inputs["mask"].shape
    mask = torch.as_tensor(inputs["mask"], device="cuda").bool()
    inputs_gpu = dict(shape=shape, affine=inputs["affine"],
        fixed=torch.as_tensor(inputs["fixed_image"], device="cuda"),
        moving=torch.as_tensor(inputs["moving_image"], device="cuda"))
    torch.cuda.synchronize()
    setup_seconds = time.monotonic()-setup_began
    resources = Resources(nvml)
    nvml.sample()
    baseline_nvml = max((p["mib"] for p in nvml.rows[-1]["processes"] if p["pid"] == nvml.pid), default=0.)
    sources = source_receipts()
    manifest = dict(config=config, settings=settings, variant=args.variant, budget=args.budget,
        kind=args.kind, repeat=repeat+1, smoke=args.smoke, command=sys.argv,
        sources=sources, input_receipts=inputs["receipts"], environment=environment(),
        config_sha256=sha256(args.config), gpu_uuid=args.gpu_uuid,
        repo_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        registration_inputs_gpu=["fixed_image", "moving_image", "fixed_image_positive_mask"],
        generic_kernel_warmup_seconds=warmup_seconds,
        launch_monotonic=args.launch_monotonic,
        timing="synchronized GPU-ready inputs to usable full DVF/image; analysis/storage separately excluded")
    atomic_json(out/"manifest.json", manifest)
    for relative in sources:
        path = out/"source_snapshot"/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO/relative).read_bytes())
    (out/"config_source.json").write_bytes(args.config.read_bytes())
    # Start latency AFTER provenance I/O, but before every registration-specific allocation.
    started = time.monotonic()
    history, monitors, transitions, snapshots = [], [], [], []
    stage_index = local_step = total_steps = refinement_step = 0
    phase = "published"
    controller = Plateau(settings["refinement"])
    banks = None
    best_payload = None
    best_total_step = None
    peak_nodes = peak_parameters = 0
    with resources.phase("initialization"):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = GaussianModel(args.variant, config, mask, "cuda")
        optimizer = model.optimizer()
        sampler = Sampler(mask, config["query_batch"]//4)
        ncc = upstream()[1].NCC() if args.variant == "released_control" else paper_ncc
        unit = torch.eye(3, device="cuda")[None]*(2/torch.tensor(shape[::-1], device="cuda"))
        probe_generator = torch.Generator(device="cuda").manual_seed(73454)
        transition_probe = torch.rand(1024, 3, generator=probe_generator, device="cuda")*2-1
    initial_counts = model.counts()
    peak_nodes, peak_parameters = initial_counts["nodes"], initial_counts["parameters"]
    frozen_counts = None
    stage_steps = config[args.variant]["stage_steps"]
    reserve = args.export_reserve*settings["export_reserve_multiplier"]
    operation_estimate = .1
    monitor_estimate = .5
    outcome = "fixed_budget_completed"
    time_limit = settings["registration_seconds_limit"]

    def charged():
        return time.monotonic()-started-resources.seconds("analysis_storage")

    def state():
        return dict(model=model.payload(), optimizer=copy.deepcopy(optimizer.state_dict()),
            sampler=sampler.state(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
            phase=phase, stage_index=stage_index, local_step=local_step, total_steps=total_steps,
            refinement_step=refinement_step, controller=controller.state(),
            optimization_seconds=resources.seconds("optimization"), charged_seconds=charged())

    def cpu_state():
        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k:cpu(v) for k,v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value
        return cpu(state())

    def snapshot(tag):
        with resources.phase("analysis_storage"):
            payload = cpu_state()
            path = out/"checkpoints"/f"{tag}.pt"
            path.parent.mkdir(exist_ok=True)
            save_checkpoint(path, payload)
            snapshots.append(dict(tag=tag, step=total_steps, phase=phase,
                path=str(path.relative_to(out)), optimization_seconds=resources.seconds("optimization"),
                charged_seconds=payload["charged_seconds"], counts=model.counts()))

    if args.kind == "convergence" or args.smoke:
        snapshot("step000000")
    while True:
        if out_of_time(charged(), reserve, operation_estimate, time_limit):
            outcome = "convergence_unresolved_at_time_limit" if args.kind == "convergence" else "fixed_budget_time_limit"
            break
        before = time.monotonic()
        with resources.phase("optimization"):
            if phase == "published" and local_step == stage_steps[stage_index]:
                with torch.no_grad():
                    previous = model(transition_probe).clone()
                model.add_stage(mask, "cuda")
                with torch.no_grad():
                    error = float((model(transition_probe)-previous).abs().max())
                if error != 0.:
                    raise RuntimeError("Fine residual changed the existing field")
                transitions.append(dict(kind="coarse_to_fine", step=total_steps, max_error=error,
                    coarse_frozen=all(not p.requires_grad for p in model.layers[0].parameters())))
                optimizer = model.optimizer()
                stage_index += 1
                local_step = 0
            total_steps += 1
            local_step += 1
            if phase == "refinement":
                refinement_step += 1
                multiplier = controller.multiplier(refinement_step)
            else:
                multiplier = lr_multiplier(local_step, stage_steps[stage_index], config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rates"][group["name"]]*multiplier
            indices, centers = sampler.next()
            coordinates = torch.cat((centers, (centers[:, None]+unit).reshape(-1, 3)))
            values = training_step(model, optimizer, coordinates, inputs_gpu, "unsupervised", indices,
                local_step, stage_steps[stage_index], config, ncc, adaptation_enabled=phase == "published")
            if phase == "refinement" and refinement_step % config["log_interval"] == 0:
                for layer in model.layers:
                    check_finite(layer, gradients=True)
            counts = model.counts()
            if frozen_counts is not None and counts["nodes_by_stage"] != frozen_counts:
                raise RuntimeError("Refinement changed frozen populations")
            peak_nodes = max(peak_nodes, counts["nodes"])
            peak_parameters = max(peak_parameters, counts["parameters"])
        operation_estimate = max(time.monotonic()-before, operation_estimate*.99)
        if total_steps % config["log_interval"] == 0 or values["adaptation"] or local_step == 1:
            row = dict(total_steps=total_steps, phase=phase, stage_index=stage_index,
                local_step=local_step, refinement_step=refinement_step, lr_multiplier=multiplier,
                optimization_seconds=resources.seconds("optimization"), charged_seconds=charged(),
                allocated_mib=torch.cuda.memory_allocated()/2**20,
                reserved_mib=torch.cuda.memory_reserved()/2**20, **counts, **values)
            history.append(row)
            with resources.phase("analysis_storage"):
                atomic_json(out/"progress.json", row)
            if total_steps % 500 == 0:
                print(json.dumps(dict(run=out.name, variant=args.variant, step=total_steps,
                                      seconds=charged(), loss=values["loss"], nodes=counts["nodes"])), flush=True)
        if args.kind == "convergence" or args.smoke:
            if ((phase == "published" and total_steps in settings["checkpoint_steps"]) or
                    (phase == "refinement" and refinement_step % settings["refinement"]["checkpoint_interval"] == 0)):
                snapshot(f"step{total_steps:06d}")
        if phase == "published" and total_steps == sum(stage_steps):
            if args.kind != "convergence":
                break
            if not any(s["step"] == total_steps for s in snapshots):
                snapshot(f"step{total_steps:06d}")
            with resources.phase("optimization"):
                before = model(transition_probe).detach()
                optimizer = model.optimizer(joint=True)
                after = model(transition_probe).detach()
                error = float((after-before).abs().max())
                if error != 0.:
                    raise RuntimeError("Joint unfreezing changed the field")
                transitions.append(dict(kind="joint_refinement", step=total_steps, max_error=error,
                                        all_scales_trainable=all(p.requires_grad for p in model.parameters())))
                frozen_counts = list(model.counts()["nodes_by_stage"])
            phase = "refinement"
            with resources.phase("monitoring"):
                banks = MonitoringBanks(mask, config, settings["refinement"])
        should_monitor = phase == "refinement" and refinement_step % settings["refinement"]["monitor_interval"] == 0
        if should_monitor:
            if out_of_time(charged(), reserve, monitor_estimate, time_limit):
                outcome = "convergence_unresolved_at_time_limit"
                break
            with resources.phase("monitoring"):
                t = time.monotonic()
                scores, probe, components = banks.evaluate(model, inputs_gpu, ncc)
                decision = controller.observe(refinement_step, scores, probe)
                monitor_estimate = max(monitor_estimate, (time.monotonic()-t)*1.1)
                monitors.append(dict(step=total_steps, refinement_step=refinement_step,
                    optimization_seconds=resources.seconds("optimization"), scores=scores,
                    components=components, **decision))
            if decision["selected"]:
                with resources.phase("analysis_storage"):
                    best_payload = cpu_state()
                    best_total_step = total_steps
            if decision["reduced"] or decision["plateau"]:
                snapshot(f"event{total_steps:06d}")
            if decision["plateau"]:
                outcome = "operational_plateau"
                break
            if args.smoke and refinement_step >= 8:
                outcome = "smoke_refinement_limit"
                break
    terminal_counts = model.counts()
    snapshot("terminal")
    with resources.phase("analysis_storage"):
        if best_payload is None:
            best_payload = cpu_state()
            best_total_step = total_steps
        save_checkpoint(out/"selected.pt", best_payload)
    # All work until termination is charged. Selection is by monitoring loss alone.
    with resources.phase("output_generation"):
        if best_total_step != total_steps:
            del optimizer, model
            model = GaussianModel.restore(best_payload["model"], "cuda")
        field, warped = usable_output(model, inputs_gpu, config)
    torch.cuda.synchronize()
    registration_seconds = charged()
    actual_elapsed = time.monotonic()-started
    fresh_latency = (time.monotonic()-args.launch_monotonic if args.kind == "startup" else None)
    selected_counts = model.counts()
    with resources.phase("analysis_storage"):
        np.save(out/"field_false_xyz.npy", field)
        np.save(out/"warped_image.npy", warped)
        # Independent serialized-model replay uses off-grid queries; never charged as usable output.
        restored = GaussianModel.restore(torch.load(out/"selected.pt", map_location="cpu", weights_only=False)["model"], "cuda")
        with torch.no_grad():
            replay_error = float((model(transition_probe)-restored(transition_probe)).abs().max())
        del restored
        if replay_error > 1e-7:
            raise RuntimeError(f"Checkpoint replay discrepancy {replay_error}")
        atomic_json(out/"history.json", history)
        atomic_json(out/"monitoring.json", monitors)
        atomic_json(out/"snapshots.json", snapshots)
    nvml.sample()
    summary = dict(variant=args.variant, kind=args.kind, budget=args.budget, repeat=repeat+1, smoke=args.smoke,
        outcome=outcome, total_steps=total_steps, refinement_steps=refinement_step,
        selected_step=best_total_step, selected_primary_loss=None if not monitors else controller.best,
        setup_seconds=setup_seconds, registration_seconds=registration_seconds,
        elapsed_including_preoutput_storage_seconds=actual_elapsed,
        fresh_process_seconds=fresh_latency, generic_kernel_warmup_seconds=warmup_seconds,
        export_reserve_seconds=reserve, time_limit_seconds=time_limit,
        operation_overshoot_seconds=max(0., registration_seconds-time_limit),
        initial_counts=initial_counts, terminal_counts=terminal_counts, selected_counts=selected_counts,
        peak_nodes=peak_nodes, peak_parameters=peak_parameters, transitions=transitions,
        checkpoint_replay_max_error=replay_error, resources=resources.summary(),
        process_memory=nvml.summary(baseline_nvml), field_sha256=sha256(out/"field_false_xyz.npy"),
        warped_image_sha256=sha256(out/"warped_image.npy"), monitoring_events=len(monitors),
        lr_reductions=controller.reductions, population_frozen_in_refinement=frozen_counts,
        nonphase_registration_seconds=registration_seconds-sum(resources.seconds(k) for k in
            ("initialization", "optimization", "monitoring", "output_generation")))
    atomic_json(out/"process_memory.json", nvml.rows)
    atomic_json(out/"summary.json", summary)
    print(json.dumps(dict(completed=str(out), outcome=outcome, steps=total_steps,
                         selected_step=best_total_step, registration_seconds=registration_seconds)), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variant", choices=("released_control", "paper_reconstruction"), required=True)
    p.add_argument("--kind", choices=("fixed", "convergence", "startup"), required=True)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--export-reserve", type=float, default=10.)
    p.add_argument("--launch-monotonic", type=float)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.kind == "startup" and args.launch_monotonic is None:
        p.error("startup requires the parent process launch clock")
    configure()
    warmup = 0. if args.kind == "startup" else warm_kernels()
    for repeat in range(args.repeats):
        run_one(args, repeat, warmup if repeat == 0 else 0.)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
