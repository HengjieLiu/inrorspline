"""Independent P83 single-pair fit or image registration; all outputs external."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .data import (Sampler, atomic_json, external, false_to_mm, flat_coordinates,
                   load_inputs, sample_volume, sha256)
from .model import (REPO, GaussianModel, adaptation_due, check_finite,
                    lr_multiplier, paper_ncc, upstream, validate_population)


def source_receipts():
    paths = list((REPO/"scripts/experiments/gaussiandir").glob("*.py"))
    paths += [REPO/"scripts/experiments/gaussiandir/python_env.sh",
              REPO/"scripts/experiments/oasis_alpha_sweep/common.py",
              REPO/"scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py",
              REPO/"scripts/experiments/oasis_polyaffine_fieldfit/data.py",
              REPO/"scripts/experiments/oasis_polyaffine_fieldfit/metrics.py"]
    paths += [REPO/"external/GaussianDIR"/name for name in ("networks.py","losses.py","datasets.py","registrators.py")]
    return {str(path.relative_to(REPO)):sha256(path) for path in paths}


def environment():
    import pytorch3d
    import scipy
    import nibabel
    return dict(torch=torch.__version__,pytorch3d=pytorch3d.__version__,cuda_build=torch.version.cuda,
                python=sys.version,numpy=np.__version__,scipy=scipy.__version__,nibabel=nibabel.__version__,
                docker_image_receipt=json.loads((Path(sys.prefix)/"docker_receipt.json").read_text()),
                cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                requested_backend=dict(cudnn_benchmark=False,cudnn_deterministic=True,tf32=False),
                effective_backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                                       cudnn_deterministic=torch.backends.cudnn.deterministic,
                                       matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                       cudnn_tf32=torch.backends.cudnn.allow_tf32))


def synchronize():
    torch.cuda.synchronize()


def save_checkpoint(path, data):
    path = external(path)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    torch.save(data, temporary)
    temporary.replace(path)


@torch.no_grad()
def render(model, shape, chunk, device="cuda"):
    result = np.empty((*shape,3),np.float32)
    flat = result.reshape(-1,3)
    for start in range(0,len(flat),chunk):
        ids = torch.arange(start,min(start+chunk,len(flat)),device=device)
        values = model(flat_coordinates(ids,shape))
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite whole-box field at flat index {start}")
        flat[start:start+len(ids)] = values.cpu().numpy()
    return result


def training_step(model, optimizer, coordinates, inputs_gpu, mode, indices, local_step, maximum, config, ncc,
                  *, adaptation_enabled=True):
    from .model import tv_loss
    optimizer.zero_grad(set_to_none=True)
    local_objective = mode == "unsupervised" and config["unsupervised"].get("loss") == "lncc9_squared_diffusion"
    if local_objective:
        local_batch = ncc.prepare(indices)
        coordinates = local_batch["coordinates"]
    flow = model(coordinates)
    if mode == "supervised":
        prediction = false_to_mm(flow,inputs_gpu["shape"],inputs_gpu["affine"])
        loss = (prediction-inputs_gpu["field"].reshape(-1,3)[indices]).square().sum(-1).mean()
        similarity, regularization = loss, loss.new_zeros(())
    elif local_objective:
        similarity, regularization = ncc(flow, local_batch)
        loss = similarity+config["unsupervised"]["lambda_diffusion"]*regularization
    else:
        with torch.no_grad():
            fixed = sample_volume(inputs_gpu["fixed"],coordinates)
        moving = sample_volume(inputs_gpu["moving"],coordinates+flow)
        similarity = ncc(moving,fixed)
        regularization = tv_loss(flow,model.variant)
        loss = similarity+config["unsupervised"]["lambda_tv"]*regularization
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite objective")
    loss.backward()
    active = model.active
    event = None
    diagnostics = None
    if local_step % config["log_interval"] == 0 or local_step == 1:
        check_finite(active,gradients=True)
        diagnostics = dict(gradient_l2={name:float(p.grad.norm()) for name,p in active.named_parameters()},
                           scale_range={name:[float(p.exp().min()),float(p.exp().max())]
                                        for name,p in active.named_parameters() if "radius" in name or "scaling" in name})
    if mode == "unsupervised" and adaptation_enabled:
        accumulate = model.variant == "paper_reconstruction" or local_step <= config[model.variant]["adaptation_end_step"]
        if accumulate:
            active.add_densification_stats()
        due = adaptation_due(model.variant,local_step,maximum,config)
        if due and model.variant == "released_control":
            score = torch.nan_to_num(active.contribution/active.counter,nan=0.)
            if int((score>active.min_contribution).sum()) < active.K:
                raise RuntimeError("Released pruning would leave fewer than K surviving nodes")
            before = len(active.node_position)
            active.densify_and_prune(optimizer)
            event = dict(before=before,after=len(active.node_position),order="before_adam_release")
            validate_population(active)
        optimizer.step()
        if due and model.variant == "paper_reconstruction":
            event = active.densify_and_prune(optimizer)
            event["order"] = "after_adam_corrected"
            validate_population(active)
    else:
        optimizer.step()
    if event:
        check_finite(active)
    values = dict(loss=float(loss.detach()),similarity=float(similarity.detach()),
                  adaptation=event,diagnostics=diagnostics)
    values["diffusion" if local_objective else "tv"] = float(regularization.detach())
    if local_objective:
        values["queries"] = len(coordinates)
    return values


def run(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = copy.deepcopy(config)
        for variant in ("released_control","paper_reconstruction"):
            config[variant]["stage_steps"] = [12]*len(config[variant]["stage_steps"])
            config[variant]["adaptation_interval"] = 3
        config["released_control"].update(adaptation_start_step=3,adaptation_end_step=9)
        config["paper_reconstruction"]["adaptation_start_local_step"] = 3
        config.update(warmup_steps=2,log_interval=1,checkpoint_interval=6,status="native_resource_smoke")
    out = external(args.out)
    if args.resume:
        if not (out/"checkpoint.pt").is_file() or (out/"summary.json").exists():
            raise ValueError("Resume requires an incomplete run with a checkpoint")
    else:
        out.mkdir(parents=True,exist_ok=False)
    try:
        return _run(args,config,out)
    except Exception as exc:
        atomic_json(out/"failure.json",dict(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc(),
                                           mode=args.mode,variant=args.variant,smoke=args.smoke))
        raise


def _run(args,config,out):
    inputs = load_inputs(config)
    sources = source_receipts()
    if args.resume:
        receipt = json.loads((out/"manifest.json").read_text())
        if receipt["sources"] != sources or receipt["config"] != config:
            raise ValueError("Source/config drift since original run")
        if (receipt["mode"],receipt["variant"]) != (args.mode,args.variant):
            raise ValueError("Run identity mismatch")
    else:
        atomic_json(out/"manifest.json",dict(mode=args.mode,variant=args.variant,smoke=args.smoke,command=sys.argv,
                    config=config,config_file_sha256=sha256(args.config),sources=sources,
                    input_receipts=inputs["receipts"],environment=environment(),
                    repo_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    timing="active includes model construction, steps, adaptation, checkpoint/telemetry; excludes loading and final render/evaluation"))
        # Frozen source bytes accompany hashes even though the main repo is dirty.
        source_dir=out/"source_snapshot"
        for relative in sources:
            destination=source_dir/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((REPO/relative).read_bytes())
    shape = inputs["mask"].shape
    mask = torch.as_tensor(inputs["mask"],device="cuda").bool()
    training_mask = torch.ones_like(mask) if args.mode == "supervised" else mask
    local_objective = args.mode == "unsupervised" and config["unsupervised"].get("loss") == "lncc9_squared_diffusion"
    sampling_mask = torch.ones_like(mask) if local_objective else training_mask
    sample_count = (config["unsupervised"]["window_centers"] if local_objective else
                    config["query_batch"]//(4 if args.mode=="unsupervised" else 1))
    inputs_gpu = dict(shape=shape,affine=inputs["affine"],
                     fixed=torch.as_tensor(inputs["fixed_image"],device="cuda"),
                     moving=torch.as_tensor(inputs["moving_image"],device="cuda"),
                     field=torch.as_tensor(inputs["field_mm"],device="cuda"))
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    began = time.monotonic()
    spent = 0.
    stage_index = 0
    local_step = 0
    total_steps = 0
    peak_nodes = peak_parameters = 0
    history = []
    transitions = []
    if args.resume:
        checkpoint = torch.load(out/"checkpoint.pt",map_location="cpu",weights_only=False)
        model = GaussianModel.restore(checkpoint["model"],"cuda")
        optimizer = model.optimizer()
        optimizer.load_state_dict(checkpoint["optimizer"])
        stage_index,local_step,total_steps = (checkpoint[k] for k in ("stage_index","local_step","total_steps"))
        spent = checkpoint["active_seconds"]
        history,transitions = checkpoint["history"],checkpoint["transitions"]
        peak_nodes,peak_parameters = checkpoint["peak_nodes"],checkpoint["peak_parameters"]
        sampler = Sampler(sampling_mask,sample_count,state=checkpoint["sampler"])
        torch.set_rng_state(checkpoint["rng_cpu"])
        torch.cuda.set_rng_state(checkpoint["rng_cuda"])
    else:
        model = GaussianModel(args.variant,config,training_mask,"cuda")
        optimizer = model.optimizer()
        sampler = Sampler(sampling_mask,sample_count)
    if local_objective:
        from .lncc_diffusion import LocalObjective
        ncc = LocalObjective(inputs_gpu["fixed"], inputs_gpu["moving"], config["unsupervised"])
    elif args.variant == "released_control":
        ncc = upstream()[1].NCC()
    else:
        ncc = paper_ncc
    stage_steps = config[args.variant]["stage_steps"]
    unit = torch.eye(3,device="cuda")[None]*(2/torch.tensor(shape[::-1],device="cuda"))
    limit = config[args.mode]["active_seconds_limit"]
    outcome = "completed"

    def checkpoint_now():
        synchronize()
        save_checkpoint(out/"checkpoint.pt",dict(model=model.payload(),optimizer=optimizer.state_dict(),
              stage_index=stage_index,local_step=local_step,total_steps=total_steps,
              active_seconds=spent+time.monotonic()-began,history=history,transitions=transitions,
              sampler=sampler.state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
              peak_nodes=peak_nodes,peak_parameters=peak_parameters))

    while stage_index < len(stage_steps):
        maximum = stage_steps[stage_index]
        while local_step < maximum:
            if spent+time.monotonic()-began >= limit:
                outcome="time_limit"
                break
            local_step += 1
            total_steps += 1
            multiplier = lr_multiplier(local_step,maximum,config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rates"][group["name"]]*multiplier
            indices,coordinates = sampler.next()
            if args.mode=="unsupervised" and not local_objective:
                coordinates = torch.cat((coordinates,(coordinates[:,None]+unit).reshape(-1,3)))
            values = training_step(model,optimizer,coordinates,inputs_gpu,args.mode,indices,local_step,maximum,config,ncc)
            counts=model.counts()
            peak_nodes=max(peak_nodes,counts["nodes"])
            peak_parameters=max(peak_parameters,counts["parameters"])
            if total_steps % config["log_interval"] == 0 or values["adaptation"] or local_step==maximum:
                synchronize()
                row=dict(stage_index=stage_index,local_step=local_step,total_steps=total_steps,
                         active_seconds=spent+time.monotonic()-began,lr_multiplier=multiplier,**counts,**values)
                history.append(row)
                atomic_json(out/"progress.json",row)
                print(json.dumps(row),flush=True)
            if total_steps % config["checkpoint_interval"] == 0:
                checkpoint_now()
        checkpoint_now()
        if outcome=="time_limit" or stage_index==len(stage_steps)-1:
            break
        probe=flat_coordinates(torch.arange(1024,device="cuda"),shape)
        with torch.no_grad():
            before=model(probe).clone()
        model.add_stage(training_mask,"cuda")
        with torch.no_grad():
            transition_error=float((model(probe)-before).abs().max())
        if transition_error!=0.:
            raise RuntimeError("Zero-motion fine initialization changed the coarse map")
        transitions.append(dict(at_total_step=total_steps,identity_residual_max_error=transition_error,
                                coarse_frozen=all(not p.requires_grad for p in model.layers[0].parameters())))
        optimizer=model.optimizer()
        stage_index+=1
        local_step=0
    synchronize()
    active_seconds=spent+time.monotonic()-began
    training_memory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    render_started=time.monotonic()
    prediction=render(model,shape,config["render_chunk"])
    np.save(out/"field_false_xyz.npy",prediction)
    # Independently reload the serialized model and compare native/off-grid queries.
    restored=GaussianModel.restore(torch.load(out/"checkpoint.pt",map_location="cpu",weights_only=False)["model"],"cuda")
    with torch.no_grad():
        queries=torch.rand(8192,3,device="cuda")*2-1
        replay_error=float((model(queries)-restored(queries)).abs().max())
    if replay_error > 1e-7:
        raise RuntimeError(f"Checkpoint replay error {replay_error}")
    del restored
    synchronize()
    render_seconds=time.monotonic()-render_started
    summary=dict(mode=args.mode,variant=args.variant,smoke=args.smoke,outcome=outcome,
                 total_steps=total_steps,expected_steps=sum(stage_steps),completed_stages=stage_index+(local_step==stage_steps[stage_index]),
                 active_seconds=active_seconds,render_export_replay_seconds=render_seconds,
                 time_limit_seconds=limit,time_limit_overshoot_seconds=max(0.,active_seconds-limit),
                 initial_counts=model.initial_counts,final_counts=model.counts(),peak_nodes=peak_nodes,
                 peak_parameters=peak_parameters,training_memory=training_memory,transitions=transitions,
                 field_sha256=sha256(out/"field_false_xyz.npy"),checkpoint_replay_max_error=replay_error)
    atomic_json(out/"training_result.json",summary)
    if local_objective:
        from .lncc_diffusion import dense_objective
        exact_began = time.monotonic()
        exact = dense_objective(torch.as_tensor(prediction, device="cuda"),inputs_gpu["fixed"],inputs_gpu["moving"],
                                config["unsupervised"]["lambda_diffusion"])
        synchronize()
        exact["evaluation_seconds"] = time.monotonic()-exact_began
        atomic_json(out/"objective_evaluation.json",exact)
    if not args.skip_metrics:
        from .evaluate import evaluate_path
        evaluate_path(str(out/"field_false_xyz.npy"),config,out/"evaluation")
    atomic_json(out/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--variant",choices=("released_control","paper_reconstruction"),required=True)
    parser.add_argument("--mode",choices=("supervised","unsupervised"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--skip-metrics",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args),indent=2),flush=True)


if __name__=="__main__":
    main()
