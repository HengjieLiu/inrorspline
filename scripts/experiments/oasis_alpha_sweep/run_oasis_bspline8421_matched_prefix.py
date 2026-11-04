#!/usr/bin/env python
"""Extend one immutable accepted cps842 prefix; independently audit new cps1 maps."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch

import bspline8421_matched_prefix as e
import dl_ncc_diffusion as d
import bounded_two_blocks as b
import bspline_svf_sweep as s
import run_oasis_dl_ncc_diffusion_one as legacy
import run_oasis_bspline_svf as helpers
import svf_transition_validation_v4 as transition

c, sha, atomic_json = e.c, e.sha, e.atomic_json


class SVFSpec(s.Spec):
    def validate(self):
        e.Spec(self.case_rank,"svf_velocity",self.alpha,self.smoke).validate()
        if (self.schedule,self.regularize,self.formulation,self.integration_steps) != ("8421","velocity","refine",7):
            raise ValueError("Only total-velocity refined cps8421 SVF")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("worker","audit","fixture"))
    p.add_argument("--manifest", "--prefix-manifest",dest="manifest",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--audit-dir",type=Path)
    p.add_argument("--family",choices=e.FAMILIES,required=True)
    p.add_argument("--alpha",type=float,choices=e.LAMBDAS,required=True)
    p.add_argument("--case-rank",type=int,required=True)
    p.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    p.add_argument("--smoke",action="store_true")
    a = p.parse_args(argv)
    e.Spec(a.case_rank,a.family,a.alpha,a.smoke).validate()
    if a.out_dir.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Artifacts must be external")
    return a


def stage_for(spec):
    # Historical schedule generation is rank independent; do not monkeypatch its first10 guard.
    return d.stages(d.FitSpec(0,"pyramid",spec.alpha,"8421",smoke=spec.smoke))[-1]


def setup(args,m,spec):
    if (os.getuid(),os.getgid()) != (1009,1020):
        raise RuntimeError("Require UID:GID1009:1020")
    e.assert_manifest(m)
    if args.gpu != m["prefixes"][spec.key()]["gpu"]:
        raise RuntimeError("Extension and audit must use original prefix GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.set_num_threads(4)
    device = c.set_cuda_device(args.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    for key in e.old.PATH_SETTINGS:
        setattr(args,key,Path(m["settings"][key]))
    batch,hashes = legacy.load_batch(args,device)
    case = m["cases"][str(spec.case_rank)]
    if hashes != case["input_hashes"] or any(batch[k] != case[k] for k in ("fixed_id","moving_id")):
        raise RuntimeError("Raw input/case mismatch")
    if not torch.equal(batch["fixed_mask"].bool(),batch["fixed"]>0):
        raise RuntimeError("Foreground differs from raw-positive fixed image")
    return device,batch,e.load_prefix(m,spec)


@torch.no_grad()
def reconstruct_prefix(prefix,family,m,device):
    identity = c.make_coordinate_grid(c.IMAGE_SHAPE,device)
    controls = velocity = None
    atomics, corners = [], []
    if family == "bounded2":
        for block in prefix["blocks"]:
            state = torch.load(block["model_path"],map_location=device,weights_only=False)
            cap = torch.tensor(block["optimizer"]["cap_norm_xyz"],device=device)
            cp = b.capped_controls(state["latent_norm_xyz"],cap)
            if not torch.equal(cp,state["controls_norm_xyz"]):
                raise RuntimeError("Bounded saved controls do not reconstruct")
            atomic = d.AlignedBSpline(cp.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,block["cps"],device)(cp)
            atomics.append(atomic.detach())
            corner = b.corner_diagnostics(atomic*d.voxel_scale(c.IMAGE_SHAPE,device=device))
            if not corner["bound_below_one"] or corner["nonpositive_corner_count"]:
                raise RuntimeError("Saved elementary map fails contraction audit")
            corners.append(corner)
        if len(atomics) != 6:
            raise RuntimeError("Need exactly six retained prefix maps")
        field = b.compose(atomics,identity)-identity
    else:
        key = "parameter_norm_xyz" if family == "displacement" else "velocity_controls_norm_xyz"
        path = prefix["model_path"] if family == "displacement" else prefix["blocks"][-1]["model_path"]
        controls = torch.load(path,map_location=device,weights_only=False)[key]
        expanded = d.AlignedBSpline(controls.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,2,device)(controls)
        velocity = expanded if family == "svf_velocity" else None
        field = s.exp_velocity(expanded,7,identity) if velocity is not None else expanded
    saved = torch.from_numpy(np.load(prefix["dvf_path"])).to(device)
    if not torch.equal(saved,field):
        raise RuntimeError(f"Matched prefix failed exact same-GPU reconstruction: {legacy.verify_init(saved,field)}")
    return controls,velocity,field,atomics,corners


def objective_for(spec,stage,batch,m,atomics):
    if spec.family == "bounded2":
        return b.Objective(batch["fixed"],batch["moving"],stage,spec.alpha,atomics,m["bounds"]["1"],True)
    if spec.family == "svf_velocity":
        return s.Objective(batch["fixed"],batch["moving"],stage,
                           SVFSpec(spec.case_rank,"8421","velocity",spec.alpha,spec.smoke))
    shape = c.build_nested_cp_shapes(c.IMAGE_SHAPE,8,[8,4,2,1])[1]
    obj=d.StageObjective(batch["fixed"],batch["moving"],spec.alpha,stage,shape)
    obj.control_shape=shape
    return obj


@torch.no_grad()
def initialize(spec,obj,controls,velocity,previous,atomics):
    if spec.family == "bounded2":
        initial = torch.zeros(*obj.control_shape,3,device=previous.device)
        atomic = obj.atomic(initial)
        if bool(atomic.count_nonzero()):
            raise RuntimeError("New bounded increment not identity")
        field = b.compose([*atomics,atomic],obj.identity)-obj.identity
        check = legacy.verify_init(previous,field)
    else:
        initial = c.cubic_bspline_dyadic_knot_insert_3d(controls,obj.control_shape).detach()
        if spec.family == "svf_velocity":
            field = obj.field(initial,native=True)
            check = transition.verify_transition(controls,initial,velocity,obj.expand(initial),previous,field,1,obj.native_scale)
        else:
            field = obj.field(initial)
            check = legacy.verify_init(previous,field)
    return initial,check,[float(v) for v in obj(initial)]


def check_losses(obj,parameter,info,initial_values):
    with torch.no_grad():
        values = [float(v) for v in obj(parameter)]
    for x,k in zip(values,("best_loss","best_data_loss","best_reg_loss")):
        if not helpers.close(x,info[k]):
            raise RuntimeError(f"Selected checkpoint loss mismatch: {k}")
    if not helpers.close(initial_values[0],info["initial_loss"]) or info["best_loss"]>initial_values[0]+1e-6:
        raise RuntimeError("Initialization/best selection mismatch")
    return values


@torch.no_grad()
def diagnostics(spec,field,controls,atomics,batch,out):
    result = {}
    scale = d.voxel_scale(c.IMAGE_SHAPE,device=field.device)
    if spec.family == "bounded2":
        result["offgrid_audit"] = b.offgrid_diagnostics(atomics,field,batch["fixed_mask"],count=100000,seed=1)
    if spec.family == "svf_velocity":
        records=[]
        for steps in (7,8,10):
            u,parts,velocities = s.reconstruct([(1,controls)],"refine",c.IMAGE_SHAPE,field.device,steps)
            _,inverses,_ = s.reconstruct([(1,controls)],"refine",c.IMAGE_SHAPE,field.device,steps,inverse=True)
            folder=out/f"integration_ss{steps}"; folder.mkdir(parents=True)
            path=folder/"disp_norm.npy"
            records.append({"integration_steps":steps,"dvf_path":str(path),"dvf_sha256":helpers.save_field(path,u),
                "metrics":helpers.metric_row(u,batch,folder),
                "difference":c.field_difference_metrics((u-field)*scale),
                "inverse_consistency":s.inverse_consistency(parts,inverses,batch["fixed_mask"])})
            if steps==7 and not torch.equal(u,field):
                raise RuntimeError("Saved velocity does not reconstruct seven-step field")
        result["integration_diagnostics"]=records
        result["scaled_initial_map_corners"]=b.corner_diagnostics(velocities[0]*scale/128)
        result["velocity_diffusion_native"]=float(c.forward_valid_diffusion((velocities[0]*scale).permute(3,0,1,2)[None]))
    return result


def worker(args,m,spec):
    started=time.monotonic(); args.out_dir.mkdir(parents=True,exist_ok=False)
    device,batch,prefix=setup(args,m,spec)
    row={"variant":e.VARIANT,"spec":asdict(spec),"protocol":e.PROTOCOL,"run_name":spec.name(),
         "run_dir":str(args.out_dir),**m["cases"][str(spec.case_rank)],"direction":"moving_to_fixed",
         "sampling":"fixed_to_moving_pull_border_align_corners_true", "shape":[160,224,192,3],
         "parameter_units":"normalized_grid_xyz","dvf_units":"normalized_grid_xyz","stage_alpha":[spec.alpha]*4,
         "source_hashes":m["source_hashes"],"source_revision":m["source_revision"],
         "prefix_summary_path":m["prefixes"][spec.key()]["summary_path"],
         "prefix_summary_sha256":m["prefixes"][spec.key()]["summary_sha256"],
         "prefix_model_hashes":m["prefixes"][spec.key()]["model_hashes"],"prefix_dvf_sha256":prefix["dvf_sha256"],
         "prefix_stage_records":prefix.get("blocks",prefix.get("stages")),
         "pair_file_sha256":m["pair_file_sha256"],"gpu":args.gpu,"visible_cuda_device":"cuda:0",
         "initial_gpu_state":legacy.gpu_snapshot(),"cuda_backend":c.cuda_backend_metadata(device),
         "tf32_matmul":False,"tf32_cudnn":False,"seed":1,"uid":os.getuid(),"gid":os.getgid(),
         "container_image":m["container_image"],"container_image_id":m["container_image_id"],
         "representation":{"displacement":"prolonged_cubic_bspline_displacement","bounded2":"retained_eight_bounded_map_composition",
                           "svf_velocity":"exponential_of_single_total_cubic_bspline_velocity"}[spec.family],
         "cuda_nondeterminism_caveat":"grid_sample backward may remain nondeterministic",
         "started_at":c.now_text(),"argv":sys.argv}
    atomic_json(args.out_dir/"manifest.json",row)
    if spec.smoke:
        row["native_loss_parity"]=legacy.native_parity(device,args.dl_code_root,args.vfa_root)
    controls,velocity,field,atomics,corners=reconstruct_prefix(prefix,spec.family,m,device)
    row["prefix_reconstructed_exactly"]=True; row["prefix_corner_checks"]=corners
    stage=stage_for(spec); blocks=[]; peak_alloc=peak_res=0.
    for index in range(2 if spec.family=="bounded2" else 1):
        obj=objective_for(spec,stage,batch,m,atomics)
        initial,check,initial_values=initialize(spec,obj,controls,velocity,field,atomics)
        folder=args.out_dir/f"level3_cps1/map{index}"
        atomic_json(args.out_dir/"progress.json",{"phase":"optimization","map":index,"cps":1,"at":c.now_text()})
        atomic_json(args.out_dir/f"initialization_map{index}.json",check)
        peak_alloc=max(peak_alloc,torch.cuda.max_memory_allocated()/2**30)
        peak_res=max(peak_res,torch.cuda.max_memory_reserved()/2**30)
        if spec.family=="bounded2":
            best,info=b.optimize(obj,folder,initial)
            key="latent_norm_xyz"
        elif spec.family=="svf_velocity":
            best,info=s.optimize(initial,obj,folder); key="velocity_controls_norm_xyz"
        else:
            best,info=d.optimize_stage(initial,obj,stage,folder); key="parameter_norm_xyz"
        state=torch.load(folder/"best.pt",map_location=device,weights_only=False)
        if not torch.equal(best,state[key]) or state["summary"]!=info:
            raise RuntimeError("Best checkpoint mismatch")
        check_losses(obj,best,info,initial_values)
        with torch.no_grad():
            if spec.family=="bounded2":
                atomic=obj.atomic(best).detach(); atomics.append(atomic)
                field=b.compose(atomics,obj.identity)-obj.identity
                corner=b.corner_diagnostics(atomic*obj.native_scale)
                if not corner["bound_below_one"] or corner["nonpositive_corner_count"] or info["max_abs_coefficient_ratio"]>1+1e-6:
                    raise RuntimeError("New bounded increment violates contraction/control bound")
            else:
                controls=best.detach()
                field=obj.field(best,native=True).detach() if spec.family=="svf_velocity" else obj.field(best).detach()
                corner=None
        path=folder/"disp_norm.npy"
        block={"cps":1,"level_index":3,"map_index":index,"alpha":spec.alpha,"optimizer":info,
               "initialization":check,"initial_loss_data_reg":initial_values,"atomic_corner_audit":corner,
               "model_path":str(folder/"best.pt"),"model_sha256":sha(folder/"best.pt"),
               "dvf_path":str(path),"dvf_sha256":helpers.save_field(path,field),
               "last_model_sha256":sha(folder/"last.pt"),"history_sha256":sha(folder/"history.csv")}
        blocks.append(block); atomic_json(folder/"validated.json",block)
        peak_alloc=max(peak_alloc,torch.cuda.max_memory_allocated()/2**30)
        peak_res=max(peak_res,torch.cuda.max_memory_reserved()/2**30)
        del obj,initial,best,state
    atomic_json(args.out_dir/"progress.json",{"phase":"metrics_and_diagnostics","at":c.now_text()})
    t=time.monotonic(); folder=args.out_dir/"final_metrics"; folder.mkdir()
    metrics=helpers.metric_row(field,batch,folder)
    diag=diagnostics(spec,field,controls,atomics,batch,args.out_dir)
    entry=m["prefixes"][spec.key()]["accepted_row"]
    newtime=sum(x["optimizer"]["training_elapsed_sec"] for x in blocks)
    row.update(**metrics,**diag,new_blocks=blocks,dvf_path=blocks[-1]["dvf_path"],dvf_sha256=blocks[-1]["dvf_sha256"],
               final_map_count=8 if spec.family=="bounded2" else 1,
               new_optimizer_sec=newtime,retained_prefix_optimizer_sec=entry["optimizer_sec"],
               combined_optimizer_sec=newtime+entry["optimizer_sec"],retained_prefix_worker_sec=entry["worker_sec"],
               actual_new_adam_updates=sum(x["optimizer"]["actual_epochs"] for x in blocks),
               retained_prefix_adam_updates=entry["actual_adam_updates"],metrics_diagnostics_sec=time.monotonic()-t,
               worker_sec=time.monotonic()-started,peak_allocated_gib=max(peak_alloc,torch.cuda.max_memory_allocated()/2**30),
               peak_reserved_gib=max(peak_res,torch.cuda.max_memory_reserved()/2**30),completed_at=c.now_text())
    if any(not math.isfinite(row[k]) for k in e.METRICS):
        raise RuntimeError("Nonfinite final metric")
    e.assert_manifest(m); atomic_json(args.out_dir/"summary.json",row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def compare_metrics(a,z):
    if any(not math.isclose(a[k],z[k],rel_tol=0,abs_tol=1e-12) for k in e.METRICS):
        raise RuntimeError("Canonical metrics did not independently reproduce")
    if a["excluded_labels"]!=z["excluded_labels"] or a["valid_label_count"]!=z["valid_label_count"]:
        raise RuntimeError("Absent-label handling differs")


def compare_labels(a,z):
    with Path(a).open() as f: ra=next(csv.DictReader(f))
    with Path(z).open() as f: rz=next(csv.DictReader(f))
    for key in ra:
        if "_label_" in key:
            x,y=float(ra[key]),float(rz[key])
            if not ((math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=0,abs_tol=1e-12)):
                raise RuntimeError(f"Per-label mismatch: {key}")


def fixture(args,m,spec):
    """Read-only replay of the two historical roundoff witnesses; zero Adam calls."""
    import svf_transition_validation_v3 as v3
    if spec.family!="svf_velocity" or spec.alpha!=.01 or spec.case_rank not in (29,44) or spec.smoke:
        raise ValueError("Only the two approved historical transition fixtures")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    device,_,prefix=setup(args,m,spec)
    coarse_index=0 if spec.case_rank==29 else 1
    block=prefix["blocks"][coarse_index]; fine_cps=block["cps"]//2
    coarse=torch.load(block["model_path"],map_location=device,weights_only=False)["velocity_controls_norm_xyz"]
    shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,8,(8,4,2,1))
    with torch.no_grad():
        fine=c.cubic_bspline_dyadic_knot_insert_3d(coarse,shapes[fine_cps])
        a=d.AlignedBSpline(coarse.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,block["cps"],device)(coarse)
        z=d.AlignedBSpline(fine.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,fine_cps,device)(fine)
        proof=v3.verify_transition(coarse,fine,a,z,s.exp_velocity(a),s.exp_velocity(z),fine_cps,
                                   d.voxel_scale(c.IMAGE_SHAPE,device=device))
        # The newly approved v4 numerical thresholds also accept this old evidence;
        # its transition dispatch remains restricted to cps2->1.
        v3.check_physical(proof["integrated_field"])
        if proof["fp64_reference"] is not None: v3.check_reference(proof["fp64_reference"])
    atomic_json(args.out_dir/"ACCEPTED.json",{"passed":True,"adam_updates":0,"spec":asdict(spec),
                "proof":proof,"source_summary_sha256":m["prefixes"][spec.key()]["summary_sha256"],
                "source_hashes":m["source_hashes"],"gpu":args.gpu,"at":c.now_text()})


def audit(args,m,spec):
    started=time.monotonic(); out=args.audit_dir or args.out_dir/f"audit_attempt_{time.time_ns()}"
    out.mkdir(parents=True,exist_ok=False)
    if not (args.out_dir/"DONE").is_file():
        raise RuntimeError("Cannot audit incomplete worker")
    row=c.read_json(args.out_dir/"summary.json")
    if row["spec"]!=asdict(spec) or row["protocol"]!=e.PROTOCOL or row["gpu"]!=args.gpu:
        raise RuntimeError("Wrong replay target")
    device,batch,prefix=setup(args,m,spec)
    if row["prefix_stage_records"]!=prefix.get("blocks",prefix.get("stages")):
        raise RuntimeError("Retained stages changed")
    controls,velocity,field,atomics,corners=reconstruct_prefix(prefix,spec.family,m,device)
    stage=stage_for(spec); checks=[]
    if len(row["new_blocks"])!=(2 if spec.family=="bounded2" else 1):
        raise RuntimeError("Wrong new-map count")
    for block in row["new_blocks"]:
        folder=Path(block["model_path"]).parent
        for path,digest in ((folder/"best.pt",block["model_sha256"]),(folder/"last.pt",block["last_model_sha256"]),
                            (folder/"history.csv",block["history_sha256"]),(Path(block["dvf_path"]),block["dvf_sha256"])):
            if sha(path)!=digest: raise RuntimeError("Saved artifact changed")
        obj=objective_for(spec,stage,batch,m,atomics)
        initial,check,values=initialize(spec,obj,controls,velocity,field,atomics)
        if check!=block["initialization"] or values!=block["initial_loss_data_reg"]:
            raise RuntimeError("Initialization replay mismatch")
        state=torch.load(folder/"best.pt",map_location=device,weights_only=False)
        key={"bounded2":"latent_norm_xyz","displacement":"parameter_norm_xyz","svf_velocity":"velocity_controls_norm_xyz"}[spec.family]
        parameter=state[key]; info=block["optimizer"]
        if info!=state["summary"] or info["stage"]!=asdict(stage) or info["alpha"]!=spec.alpha:
            raise RuntimeError("Checkpoint schedule mismatch")
        e.old.check_history(folder/"history.csv",info,spec.alpha)
        check_losses(obj,parameter,info,values)
        with torch.no_grad():
            if spec.family=="bounded2":
                if not torch.equal(obj.controls(parameter),state["controls_norm_xyz"]):
                    raise RuntimeError("Saved tanh controls mismatch")
                atomic=obj.atomic(parameter).detach(); atomics.append(atomic)
                corner=b.corner_diagnostics(atomic*obj.native_scale)
                if corner!=block["atomic_corner_audit"] or not corner["bound_below_one"] or corner["nonpositive_corner_count"]:
                    raise RuntimeError("New contraction replay failed")
                field=b.compose(atomics,obj.identity)-obj.identity
            else:
                controls=parameter.detach()
                field=obj.field(parameter,native=True).detach() if spec.family=="svf_velocity" else obj.field(parameter).detach()
        if not np.array_equal(field.cpu().numpy(),np.load(block["dvf_path"])):
            raise RuntimeError("Independent field replay differs")
        checks.append({"map":block["map_index"],"history_losses_field":True})
        del initial,obj,state,parameter
    metrics=helpers.metric_row(field,batch,out)
    compare_metrics(metrics,row); compare_labels(out/"case_metrics.csv",args.out_dir/"final_metrics/case_metrics.csv")
    diag=diagnostics(spec,field,controls,atomics,batch,out)
    if spec.family=="bounded2" and diag["offgrid_audit"]!=row["offgrid_audit"]:
        raise RuntimeError("Offgrid replay differs")
    if spec.family=="svf_velocity":
        for x,z in zip(diag["integration_diagnostics"],row["integration_diagnostics"]):
            if x["dvf_sha256"]!=z["dvf_sha256"] or x["inverse_consistency"]!=z["inverse_consistency"]:
                raise RuntimeError("Integration sensitivity/inverse replay differs")
            compare_metrics(x["metrics"],z["metrics"])
    files={str(p):sha(p) for p in args.out_dir.rglob("*") if p.is_file() and "audit_attempt_" not in str(p.relative_to(args.out_dir))
           and p.name not in ("ACCEPTED.json","progress.json") and not p.name.startswith("FAILED")}
    receipt={"passed":True,"files":files,"checks":checks,"summary_sha256":sha(args.out_dir/"summary.json"),
             "audit_dir":str(out),"audit_sec":time.monotonic()-started,"gpu":args.gpu,
             "audit_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
             "audit_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"at":c.now_text()}
    atomic_json(out/"audit.json",receipt)
    e.assert_manifest(m); atomic_json(args.out_dir/"ACCEPTED.json",receipt)
    e.validate(args.out_dir,m,spec)


if __name__=="__main__":
    args=parse_args(); m=c.read_json(args.manifest); spec=e.Spec(args.case_rank,args.family,args.alpha,args.smoke)
    try:
        {"worker":worker,"audit":audit,"fixture":fixture}[args.action](args,m,spec)
    except Exception:
        if args.out_dir.exists():
            atomic_json(args.out_dir/f"FAILED_{args.action}_{time.time_ns()}.json",{"traceback":traceback.format_exc(),"at":c.now_text()})
        raise
