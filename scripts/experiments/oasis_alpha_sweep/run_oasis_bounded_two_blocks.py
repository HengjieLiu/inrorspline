#!/usr/bin/env python
"""P68: ranks0/4, two bounded maps per level versus two blocks on one map."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import bounded_two_blocks as b
import run_oasis_selective_bounded_first10 as previous

c,d,legacy=previous.c,previous.d,previous.legacy
sha,atomic_json,METRICS=previous.sha,previous.atomic_json,previous.METRICS
VARIANT="oasis_seed3_rank000_004_mrdbscp8421_tanh_two_blocks_lambda1151_v1"
METHODS={"two_increments":"direct_cp_8_4_2_1_tanh_two_increments_composition",
         "one_map_two_blocks":"direct_cp_8_4_2_1_tanh_one_increment_two_optimizer_blocks"}
PROTOCOL={**previous.PROTOCOL,"version":1,"stage_bounded_by_last":None,
          "all_maps_bounded":True,"optimization_blocks_per_level":2,
          "increments_per_stage":{"two_increments":2,"one_map_two_blocks":1},
          "previous_maps_frozen":True,"fresh_adam_each_block":True,
          "fresh_stopping_history_each_block":True,"lr_schedule_restarted_each_block":True,
          "cps1_lr_reductions_per_block":2,"lambda_split_between_blocks":False,
          "initialization":"fresh_identity; continuation block restores selected same-map latent"}
RANKS=(0,4)
PLACEMENT=((0,"two_increments","0"),(0,"one_map_two_blocks","1"),
           (4,"two_increments","2"),(4,"one_map_two_blocks","3"))


def run_name(rank,mode,smoke=False):
    if type(rank) is not int or rank not in RANKS or mode not in b.MODES:
        raise ValueError("Require rank0/4 and approved two-block mode")
    digest=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "")+f"rank{rank:03d}_cps8421_lambda1_1_5_1_{mode}_{digest}"


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch")
    launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--gpus",default="0,1,2,3",choices=("0,1,2,3",))
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    worker=sub.add_parser("worker")
    worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--case-rank",type=int,choices=RANKS,required=True)
    worker.add_argument("--mode",choices=b.MODES,required=True)
    worker.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    worker.add_argument("--smoke",action="store_true")
    for parser in (launch,worker):
        parser.add_argument("--stage-alpha",type=float,nargs=4,default=list(b.ALPHAS))
        parser.add_argument("--blocks-per-level",type=int,choices=(2,),default=2)
        parser.add_argument("--bound-multiplier",type=float,choices=(.99,),default=.99)
    args=p.parse_args(argv)
    if tuple(args.stage_alpha)!=b.ALPHAS: p.error("Frozen weights are1,1,5,1")
    output=args.artifact_root if args.command=="launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT): p.error("External artifacts required")
    return args


def sources(sitreg_root):
    return {"historical":previous.sources(sitreg_root),"runner":sha(Path(__file__)),"core":sha(Path(b.__file__)),
            "audit":sha(Path(__file__).with_name("audit_oasis_bounded_two_blocks.py"))}


def assert_sources(manifest):
    if sources(Path(manifest["sitreg_root"]))!=manifest["source_hashes"]:
        raise RuntimeError("Numerical source drift; stop for audit")


def assert_references(manifest):
    previous.assert_references(manifest)
    for row in manifest["baseline_rows"]:
        if not (Path(row["run_dir"])/"DONE").exists(): raise RuntimeError("Reference incomplete")
        if sha(Path(row["dvf_path"]))!=row["dvf_sha256"]: raise RuntimeError("Reference field drift")


def preflight(args):
    root=args.artifact_root
    prior_root=root/"runs"/previous.VARIANT
    old=c.read_json(prior_root/"launch_manifest.json")
    previous.assert_sources(old); previous.assert_references(old)
    old_report=root/"repo_experiments"/previous.VARIANT
    data=previous.reference.accepted_data(old_report)
    audit=c.read_json(prior_root/"independent_audit.json")
    if not audit["passed"] or audit["cases"]!=40: raise RuntimeError("Require independently audited references")
    hashes={str(old_report/name):sha(old_report/name) for name in ("ACCEPTED.json","results.json")}
    hashes[str(prior_root/"independent_audit.json")]=sha(prior_root/"independent_audit.json")
    cases={str(rank):old["cases"][str(rank)] for rank in RANKS}
    for case in cases.values():
        resolved=c.resolve_case(rank=case["case_rank"],pair_file=Path(old["settings"]["pair_file"]),
                                oasis_root=Path(old["settings"]["oasis_root"]))
        if (resolved.fixed_id,resolved.moving_id)!=(case["fixed_id"],case["moving_id"]):
            raise RuntimeError("Case/direction mismatch")
        for name,digest in case["input_hashes"].items():
            if sha(Path(old["settings"]["oasis_root"])/"test"/name)!=digest: raise RuntimeError("Input drift")
    if sha(Path(old["settings"]["pair_file"]))!=old["pair_file_sha256"]: raise RuntimeError("Split drift")
    result={"variant":VARIANT,"protocol":PROTOCOL,"methods":METHODS,"cases":cases,
            "settings":old["settings"],"bounds":old["bounds"],"pair_file_sha256":old["pair_file_sha256"],
            "baseline_rows":[r for r in data["candidates"] if r["case_rank"] in RANKS],
            "references":[r for r in old["references"] if r["rank"] in RANKS],
            "reference_hashes":hashes,"source_hashes":sources(Path(old["sitreg_root"])),
            "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=c.REPO_ROOT,text=True).strip(),
            "sitreg_root":old["sitreg_root"],"sitreg_revision":old["sitreg_revision"],
            "native_bound_audit":old["native_bound_audit"],
            "gpu_placement":[{"case_rank":r,"mode":m,"gpu":g} for r,m,g in PLACEMENT],
            "initial_gpu_state":legacy.gpu_snapshot(),"created_at":c.now_text(),
            "container_image":os.environ["INRORSPLINE_CONTAINER_IMAGE"],
            "container_image_id":os.environ["INRORSPLINE_CONTAINER_IMAGE_ID"]}
    assert_references(result)
    return result


def baseline(manifest,rank):
    return next(r for r in manifest["baseline_rows"] if r["case_rank"]==rank and r["bounded_last"]==4)


def compare_fields(path_a,path_b):
    a=np.load(path_a,mmap_mode="r"); bb=np.load(path_b,mmap_mode="r")
    diff=(a-bb)*np.asarray([95.5,111.5,79.5],dtype=np.float32)
    maximum=float(np.abs(diff).max()); rms=float(np.sqrt(np.mean(diff.astype(np.float64)**2)))
    if maximum>1e-4 or rms>1e-5: raise RuntimeError("Shared first-block field drift")
    return {"max_abs_voxel":maximum,"rms_voxel":rms,"identical_sha256":sha(Path(path_a))==sha(Path(path_b))}


def worker(args):
    manifest=c.read_json(args.manifest); assert_sources(manifest)
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); initial_gpu=legacy.gpu_snapshot()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.set_num_threads(4)
    device=c.set_cuda_device(args.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    for key in ("oasis_root","pair_file","dl_code_root","vfa_root"):
        setattr(args,key,Path(manifest["settings"][key]))
    batch,input_hashes=legacy.load_batch(args,device)
    case=manifest["cases"][str(args.case_rank)]
    if input_hashes!=case["input_hashes"] or sha(args.pair_file)!=manifest["pair_file_sha256"]:
        raise RuntimeError("Inputs/split changed")
    if (batch["fixed_id"],batch["moving_id"])!=(case["fixed_id"],case["moving_id"]) or not torch.equal(batch["fixed_mask"].bool(),batch["fixed"]>0):
        raise RuntimeError("Wrong case/foreground")
    row={"variant":VARIANT,"method":METHODS[args.mode],"mode":args.mode,"protocol":PROTOCOL,
         "run_name":run_name(args.case_rank,args.mode,args.smoke),"smoke":args.smoke,
         "stage_alpha":list(b.ALPHAS),"case_rank":args.case_rank,"fixed_id":case["fixed_id"],"moving_id":case["moving_id"],
         "direction":"moving_to_fixed","sampling":"fixed_to_moving_pull","gpu":args.gpu,"visible_cuda_device":"cuda:0",
         "source_hashes":manifest["source_hashes"],"source_revision":manifest["source_revision"],
         "sitreg_revision":manifest["sitreg_revision"],"input_hashes":input_hashes,
         "pair_file_sha256":manifest["pair_file_sha256"],"initial_gpu_state":initial_gpu,
         "cuda_backend":c.cuda_backend_metadata(device),"tf32_matmul":False,"tf32_cudnn":False,
         "uid":os.getuid(),"gid":os.getgid(),"seed":1,
         "container_image":manifest["container_image"],"container_image_id":manifest["container_image_id"],
         "representation":"retained_bounded_bspline_generated_trilinear_maps",
         "bounds":manifest["bounds"],"segmentation_loss":False,"reference_supervision":False,
         "analytic_regularizer":False,"initialization":"identity_no_reused_prefix",
         "increments_per_level":2 if args.mode=="two_increments" else 1,"optimizer_blocks_per_level":2,
         "cuda_nondeterminism_caveat":"grid_sample backward may remain nondeterministic",
         "started_at":c.now_text(),"argv":sys.argv}
    if args.smoke: row["native_loss_parity"]=legacy.native_parity(device,args.dl_code_root,args.vfa_root)
    atomic_json(args.out_dir/"manifest.json",row)
    identity=c.make_coordinate_grid(c.IMAGE_SHAPE,device)
    scale=d.voxel_scale(c.IMAGE_SHAPE,device=device)
    previous_field=torch.zeros_like(identity); previous_best=None
    atomics=[]; records=[]; peak=0.
    for level,stage in enumerate(b.stages(args.smoke)):
        for block in range(2):
            step=2*level+block; alpha=b.ALPHAS[level]
            frozen_ids=b.frozen_indices(args.mode,step); frozen=[atomics[i] for i in frozen_ids]
            initial=b.initial_parameter(args.mode,step,previous_best)
            obj=b.Objective(batch["fixed"],batch["moving"],stage,alpha,frozen,manifest["bounds"][str(stage.cps)],True)
            seed=torch.zeros(*obj.control_shape,3,device=device) if initial is None else initial
            with torch.no_grad():
                initial_atomic=obj.atomic(seed)
                init_field=b.compose([*frozen,initial_atomic],identity)-identity
                init_check=legacy.verify_init(previous_field,init_field)
                initial_values=[float(x) for x in obj(seed)]
                if block and not math.isclose(initial_values[0],records[-1]["optimizer"]["best_loss"],rel_tol=1e-6,abs_tol=1e-7):
                    raise RuntimeError("Second block changed initial objective")
                if initial is None and (bool(seed.count_nonzero()) or bool(initial_atomic.count_nonzero())):
                    raise RuntimeError("New increment is not identity")
                if args.smoke:
                    q=obj.field(seed)
                    ref=d.StageObjective(obj.fixed,obj.moving,alpha,replace(stage,cps=None,factor=1))
                    if initial_values!=[float(x) for x in ref(q)]: raise RuntimeError("Same-field loss parity")
                    del q,ref
            del seed,initial_atomic,init_field
            folder=args.out_dir/f"level{level}_cps{stage.cps}"/f"block{block}"
            atomic_json(args.out_dir/"progress.json",{"phase":"optimization","level":level,"block":block,"step":step,"cps":stage.cps,"updated_at":c.now_text()})
            best,info=b.optimize(obj,folder,initial)
            saved=torch.load(folder/"best.pt",map_location=device,weights_only=False)
            with torch.no_grad():
                if not torch.equal(best,saved["latent_norm_xyz"]) or saved["summary"]!=info:
                    raise RuntimeError("Checkpoint latent/summary mismatch")
                controls=obj.controls(saved["latent_norm_xyz"])
                if not torch.equal(controls,saved["controls_norm_xyz"]): raise RuntimeError("Control reconstruction mismatch")
                atomic=obj.expand(controls).detach()
                current=b.compose([*frozen,atomic],identity)-identity
                values=[float(x) for x in obj(saved["latent_norm_xyz"])]
                for value,key in zip(values,("best_loss","best_data_loss","best_reg_loss")):
                    if not math.isclose(value,info[key],rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Selected loss mismatch")
            if not math.isclose(info["initial_loss"],initial_values[0],rel_tol=1e-6,abs_tol=1e-7) or info["best_loss"]>initial_values[0]+1e-6:
                raise RuntimeError("Best selection worsened objective")
            previous_best=best.detach().clone()
            del saved,controls,obj,best,initial
            atomics.append(atomic)
            atomic_json(args.out_dir/"progress.json",{"phase":"block_validation","level":level,"block":block,"step":step,"updated_at":c.now_text()})
            topology=b.corner_diagnostics(atomic*scale)
            if not topology["bound_below_one"] or topology["nonpositive_corner_count"]:
                raise RuntimeError("Bounded elementary-map audit failed")
            field_path=folder/f"disp_norm_{case['fixed_id']}_{case['moving_id']}.npy"
            np.save(field_path,current.cpu().numpy())
            restored=torch.from_numpy(np.load(field_path)).to(device)
            if not torch.equal(restored,current) or restored.shape!=(*c.IMAGE_SHAPE,3) or not bool(torch.isfinite(restored).all()):
                raise RuntimeError("Field serialization mismatch")
            metrics=previous.metric_row(restored,batch,folder)
            record={"level_index":level,"block_index":block,"step_index":step,"cps":stage.cps,"alpha":alpha,
                    "new_map":args.mode=="two_increments" or block==0,"frozen_block_indices":frozen_ids,
                    "retained_block_indices":b.retained_indices(args.mode,step),
                    "optimizer":info,"initialization_check":init_check,"initial_loss_data_reg":initial_values,
                    "objective_improvement":initial_values[0]-info["best_loss"],"metrics":metrics,
                    "atomic_corner_audit":topology,"model_path":str(folder/"best.pt"),"model_sha256":sha(folder/"best.pt"),
                    "dvf_path":str(field_path),"dvf_sha256":sha(field_path)}
            if not args.smoke and step==0:
                old=baseline(manifest,args.case_rank)["stages"][0]
                record["historical_first_block_parity"]=compare_fields(field_path,old["dvf_path"])
                if any(info[k]!=old["optimizer"][k] for k in ("best_epoch","actual_epochs")) or abs(info["best_loss"]-old["optimizer"]["best_loss"])>1e-6:
                    raise RuntimeError("Historical first-block schedule/loss drift")
            records.append(record); atomic_json(folder/"validated.json",record)
            previous_field=current.detach(); peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
            del restored
            print(json.dumps({"block_complete":step,"level":level,"block":block,"maps":len(record["retained_block_indices"]),
                              "objective_improvement":record["objective_improvement"],**{k:metrics[k] for k in METRICS}}),flush=True)
    final_indices=b.retained_indices(args.mode,7); final_fields=[atomics[i] for i in final_indices]
    atomic_json(args.out_dir/"progress.json",{"phase":"offgrid_validation","updated_at":c.now_text()})
    probes=b.offgrid_diagnostics(final_fields,current,batch["fixed_mask"],count=100000,seed=1)
    if probes["points_sha256"]!=baseline(manifest,args.case_rank)["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Offgrid point mismatch")
    peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
    row.update(**records[-1]["metrics"],blocks=records,final_retained_block_indices=final_indices,
               final_map_count=len(final_indices),offgrid_audit=probes,peak_cuda_gib=peak,
               dvf_path=records[-1]["dvf_path"],dvf_sha256=records[-1]["dvf_sha256"],
               training_elapsed_sec=sum(x["optimizer"]["training_elapsed_sec"] for x in records),
               total_elapsed_sec=time.monotonic()-started,run_dir=str(args.out_dir),completed_at=c.now_text())
    assert_sources(manifest)
    atomic_json(args.out_dir/"summary.json",row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def validate(path,manifest,rank,mode,smoke=False):
    assert_sources(manifest)
    if not (path/"DONE").is_file(): raise RuntimeError("Missing completion marker")
    row=c.read_json(path/"summary.json"); case=manifest["cases"][str(rank)]
    expected={"variant":VARIANT,"method":METHODS[mode],"mode":mode,"case_rank":rank,"fixed_id":case["fixed_id"],
              "moving_id":case["moving_id"],"direction":"moving_to_fixed","protocol":PROTOCOL,"smoke":smoke,
              "stage_alpha":list(b.ALPHAS),"input_hashes":case["input_hashes"],"source_hashes":manifest["source_hashes"],
              "pair_file_sha256":manifest["pair_file_sha256"],"uid":1009,"gid":1020,"seed":1,
              "segmentation_loss":False,"reference_supervision":False,"analytic_regularizer":False,
              "final_map_count":8 if mode=="two_increments" else 4,
              "final_retained_block_indices":b.retained_indices(mode,7)}
    if any(row[k]!=v for k,v in expected.items()): raise RuntimeError("Wrong output metadata")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong backend")
    if len(row["blocks"])!=8: raise RuntimeError("Wrong block count")
    for step,record in enumerate(row["blocks"]):
        stage=b.stages(smoke)[step//2]; info=record["optimizer"]
        if record["level_index"]!=step//2 or record["block_index"]!=step%2 or record["step_index"]!=step or info["stage"]!=asdict(stage):
            raise RuntimeError("Wrong schedule/index")
        if record["alpha"]!=b.ALPHAS[step//2] or info["alpha"]!=record["alpha"] or info["bound"]!=manifest["bounds"][str(stage.cps)]:
            raise RuntimeError("Wrong weight or bound")
        if not info["bounded"] or record["frozen_block_indices"]!=b.frozen_indices(mode,step) or record["retained_block_indices"]!=b.retained_indices(mode,step):
            raise RuntimeError("Wrong composition structure")
        if record["new_map"]!=(mode=="two_increments" or step%2==0): raise RuntimeError("Wrong initialization mode")
        if not 0<=info["best_epoch"]<=info["actual_epochs"]<=stage.max_epochs or info["best_loss"]>info["initial_loss"]+1e-6:
            raise RuntimeError("Invalid selected epoch/loss")
        if record["new_map"] and info["initial_latent_max_abs"]!=0.: raise RuntimeError("Nonidentity new map")
        if step%2 and not math.isclose(info["initial_loss"],row["blocks"][step-1]["optimizer"]["best_loss"],rel_tol=1e-6,abs_tol=1e-7):
            raise RuntimeError("Second-block objective discontinuity")
        topology=record["atomic_corner_audit"]
        if not topology["bound_below_one"] or topology["nonpositive_corner_count"] or info["max_abs_coefficient_ratio"]>1+1e-6:
            raise RuntimeError("Invalid atomic bound")
        for kind in ("model","dvf"):
            if sha(Path(record[kind+"_path"]))!=record[kind+"_sha256"]: raise RuntimeError("Artifact hash drift")
        field=np.load(record["dvf_path"],mmap_mode="r")
        if field.dtype!=np.float32 or field.shape!=(*c.IMAGE_SHAPE,3) or not np.isfinite(field).all(): raise RuntimeError("Invalid field")
        with (Path(record["model_path"]).parent/"history.csv").open() as stream: history=list(csv.DictReader(stream))
        best=min(history,key=lambda x:float(x["loss"]))
        if int(best["epoch"])!=info["best_epoch"] or not math.isclose(float(best["loss"]),info["best_loss"],rel_tol=1e-6,abs_tol=1e-7):
            raise RuntimeError("History mismatch")
    if not all(math.isfinite(row[k]) for k in METRICS) or row["offgrid_audit"]["count"]!=100000:
        raise RuntimeError("Invalid metrics/probes")
    if row["offgrid_audit"]["points_sha256"]!=baseline(manifest,rank)["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Different offgrid points")
    if smoke and not row["native_loss_parity"]["passed"]: raise RuntimeError("Native loss parity failed")
    return row


def report(rows,manifest,out,audits):
    if len(rows)!=4 or {(r["case_rank"],r["mode"]) for r in rows}!={(rank,mode) for rank in RANKS for mode in b.MODES}:
        raise RuntimeError("Incomplete approved pilot")
    if len(audits)!=4 or not all(a["passed"] for a in audits): raise RuntimeError("Missing independent replay")
    rows=sorted(rows,key=lambda x:(x["case_rank"],b.MODES.index(x["mode"])))
    comparison=[]; blocks=[]; changes=[]; shared=[]
    for rank in RANKS:
        group=[r for r in rows if r["case_rank"]==rank]
        shared.append({"case_rank":rank,**compare_fields(group[0]["blocks"][0]["dvf_path"],group[1]["blocks"][0]["dvf_path"])})
        old=baseline(manifest,rank)
        items=[("Two bounded increments/level",group[0]),("One map/level, two optimizer blocks",group[1])]
        items += [(f"Historical bound last{x['bounded_last']}" if x["bounded_last"]<4 else "Historical all4, one block",
                   x) for x in sorted(manifest["baseline_rows"],key=lambda x:x["bounded_last"]) if x["case_rank"]==rank]
        items += [(x["method"],x) for x in manifest["references"] if x["rank"]==rank]
        for name,item in items:
            rec={"case_rank":rank,"fixed_id":old["fixed_id"],"moving_id":old["moving_id"],"method":name,
                 **{k:item[k] for k in METRICS}}
            for key in METRICS: rec["delta_historical_all4_"+key]=item[key]-old[key]
            comparison.append(rec)
        for row in group:
            for rec in row["blocks"]:
                info=rec["optimizer"]
                blocks.append({"case_rank":rank,"mode":row["mode"],"level":rec["level_index"],"block":rec["block_index"],
                    "cps":rec["cps"],"retained_maps":len(rec["retained_block_indices"]),
                    "objective_improvement":rec["objective_improvement"],**rec["metrics"],
                    **{k:info[k] for k in ("initial_loss","best_loss","best_data_loss","best_reg_loss","best_epoch","actual_epochs",
                    "lr_reductions","training_elapsed_sec","stopping_reason","saturated_coefficient_fraction_gt_95pct","max_abs_control_native_voxel")},
                    **rec["atomic_corner_audit"]})
                if rec["block_index"]:
                    first=row["blocks"][rec["step_index"]-1]
                    changes.append({"case_rank":rank,"mode":row["mode"],"cps":rec["cps"],
                                    "total_loss_change":info["best_loss"]-first["optimizer"]["best_loss"],
                                    **{"delta_"+k:rec["metrics"][k]-first["metrics"][k] for k in METRICS}})
    out.mkdir(parents=True,exist_ok=False)
    c.write_csv(out/"comparison.csv",comparison); c.write_csv(out/"blocks.csv",blocks); c.write_csv(out/"second_block_changes.csv",changes)
    pairs=[]
    for rank in RANKS:
        a=next(r for r in rows if r["case_rank"]==rank and r["mode"]=="two_increments")
        ctrl=next(r for r in rows if r["case_rank"]==rank and r["mode"]=="one_map_two_blocks")
        pairs.append({"case_rank":rank,**{k:a[k]-ctrl[k] for k in METRICS}})
    c.write_csv(out/"two_minus_control.csv",pairs)
    atomic_json(out/"results.json",{"complete":True,"candidates":rows,"comparison":comparison,"second_block_changes":changes,
               "two_minus_control":pairs,"shared_first_block_checks":shared,"independent_audits":audits,"provenance":manifest})
    lines=["# Two bounded increments per level: ranks0 and4","",
           "Fresh identity registrations; cps8→4→2→1, lambda[1,1,5,1], loss downsampling4/2/1/1.",
           "Same raw NCC9 and unmasked total-composed-field evaluation-grid voxel diffusion; no lambda splitting.",
           "Two-increment arm retains8 maps. Control retains4 maps, continuing each selected latent with a fresh Adam/LR/stopping block.",
           "Both use two full schedule blocks per level. Budgets are matched, not actual updates, FLOPs or runtime. No tuning.",""]
    for rank in RANKS:
        case=manifest["cases"][str(rank)]
        lines += [f"## Rank{rank}: fixed{case['fixed_id']} ← moving{case['moving_id']}","",
                  "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
                  "|---|---:|---:|---:|---:|---:|"]
        for rec in comparison:
            if rec["case_rank"]==rank: lines.append("| "+rec["method"]+" | "+" | ".join(f"{rec[k]:.6f}" for k in METRICS)+" |")
        lines.append("")
    lines += ["## Effect of the second optimization block","",
              "| Rank | Arm | CPS | Total loss change | Dice change | HD95 change | ASD change | FG SDlogJ change | FG folding pp change |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rec in changes:
        lines.append(f"| {rec['case_rank']} | {rec['mode']} | {rec['cps']} | {rec['total_loss_change']:.8f} | "+
                     " | ".join(f"{rec['delta_'+k]:+.6f}" for k in METRICS)+" |")
    lines += ["","## Topology and runtime","",
              "| Rank | Arm | Maps | Chain nonpositive % | Export nonpositive % | Chain minimum J | Export minimum J | Optimization s | Worker s | Peak GiB | GPU |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        chain=row["offgrid_audit"]["retained_composition"]; export=row["offgrid_audit"]["interpolated_export"]
        lines.append(f"| {row['case_rank']} | {row['mode']} | {row['final_map_count']} | {chain['nonpositive_percent']:.6f} | {export['nonpositive_percent']:.6f} | {chain['minimum_jacobian']:.6f} | {export['minimum_jacobian']:.6f} | {row['training_elapsed_sec']:.3f} | {row['total_elapsed_sec']:.3f} | {row['peak_cuda_gib']:.3f} | {row['gpu']} |")
    lines += ["","All bounded elementary maps passed full native-cell-corner contraction checks. No gate forces final exported-field folding to zero.",
              "Same100k seed1 off-grid fixed-raw-positive probes per case; retained maps and trilinearly interpolated final export are evaluated separately.",
              "Canonical accuracy uses unchanged absent-label handling; foreground regularity uses fixed raw image>0 and cropped-interior positive-J SDlogJ / J≤0 percent.",
              "All32 block checkpoints, selected objective values, canonical per-label metrics, corner audits and final off-grid probes were independently replayed.",
              "Two deliberately selected exploratory cases, not a first-ten benchmark or a generalization claim. Existing jobs remained active; runtime is diagnostic."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for file,count in (("comparison.csv",16),("blocks.csv",32),("second_block_changes.csv",16),("two_minus_control.csv",2)):
        with (out/file).open() as stream: values=list(csv.DictReader(stream))
        if len(values)!=count: raise RuntimeError("Report row count mismatch")
    assert_sources(manifest); assert_references(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"new_fits":4,"ranks":list(RANKS),"independent_audit":True,
                "completed_at":c.now_text(),"output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    root=args.artifact_root/"runs"/VARIANT; out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): previous.reference.accepted_data(out); print("Already accepted"); return
        mp=root/"launch_manifest.json"
        if mp.exists(): manifest=c.read_json(mp); assert_sources(manifest); assert_references(manifest)
        else: manifest=preflight(args); atomic_json(mp,manifest)
        required=8192
        def start(rank,mode,gpu,smoke=False,audit=False):
            state=next(x for x in legacy.gpu_snapshot()["rows"] if x["physical_gpu"]==gpu)
            if state["memory_free_mib"]<required: raise RuntimeError("Insufficient GPU memory; no worker launched")
            path=root/run_name(rank,mode,smoke)
            if not audit and path.exists(): raise RuntimeError(f"Incomplete attempt preserved: {path}")
            if audit:
                cmd=[sys.executable,"-u","-B",str(Path(__file__).with_name("audit_oasis_bounded_two_blocks.py")),
                     "--manifest",str(mp),"--run-dir",str(path),"--gpu",gpu]
                logpath=root/(path.name+"_audit.log")
            else:
                cmd=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(mp),"--out-dir",str(path),
                     "--case-rank",str(rank),"--mode",mode,"--gpu",gpu]
                if smoke: cmd.append("--smoke")
                logpath=root/(path.name+".log")
            log=logpath.open("x")
            env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
            try: proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            except BaseException: log.close(); raise
            print(json.dumps({"launched":path.name,"audit":audit,"gpu":gpu,"pid":proc.pid,"gpu_before":state}),flush=True)
            return {"process":proc,"log":log,"path":path,"case_rank":rank,"mode":mode,"gpu":gpu}
        for mode in b.MODES:
            path=root/run_name(0,mode,True)
            if not (path/"DONE").exists():
                item=start(0,mode,"3",True); code=item["process"].wait(); item["log"].close()
                if code: raise RuntimeError("Smoke failed; preserved for audit")
            row=validate(path,manifest,0,mode,True)
            required=max(required,math.ceil(row["peak_cuda_gib"]*1024)+2048)
            atomic_json(root/f"smoke_{mode}.json",{"passed":True,"required_free_mib":required,"result":row})
        if args.stop_after=="smoke": return
        rows=[]; running=[]; failures=[]; started=time.monotonic()
        try:
            for rank,mode,gpu in PLACEMENT:
                path=root/run_name(rank,mode)
                if (path/"DONE").exists(): rows.append(validate(path,manifest,rank,mode))
                else: running.append(start(rank,mode,gpu))
            while running:
                for item in running[:]:
                    code=item["process"].poll()
                    if code is None: continue
                    item["log"].close(); running.remove(item)
                    if code: failures.append({"path":str(item["path"]),"exit_code":code}); continue
                    try: rows.append(validate(item["path"],manifest,item["case_rank"],item["mode"]))
                    except BaseException as exc: failures.append({"path":str(item["path"]),"error":str(exc)})
                    print(json.dumps({"completed":len(rows),"total":4,"case_rank":item["case_rank"],"mode":item["mode"]}),flush=True)
                atomic_json(root/"progress.json",{"phase":"production","completed":len(rows),"total":4,
                    "elapsed_sec":time.monotonic()-started,"active":[{k:x[k] for k in ("case_rank","mode","gpu")} for x in running],
                    "failures":failures,"updated_at":c.now_text()})
                if running: time.sleep(3)
        finally:
            # Never terminate existing jobs or our other already-started workers.
            for item in running:
                item["process"].wait(); item["log"].close()
        if failures: raise RuntimeError(f"Failed attempts preserved: {failures}")
        audits=[]; running=[]
        atomic_json(root/"progress.json",{"phase":"independent_audit","completed":4,"total":4,"updated_at":c.now_text()})
        try:
            for rank,mode,gpu in PLACEMENT:
                path=root/run_name(rank,mode)
                if (path/"independent_audit.json").exists(): audits.append(c.read_json(path/"independent_audit.json"))
                else: running.append(start(rank,mode,gpu,audit=True))
            for item in running:
                code=item["process"].wait(); item["log"].close()
                if code: failures.append({"audit_path":str(item["path"]),"exit_code":code})
                else: audits.append(c.read_json(item["path"]/"independent_audit.json"))
        finally:
            for item in running:
                if item["process"].poll() is None: item["process"].wait()
                if not item["log"].closed: item["log"].close()
        if failures: raise RuntimeError(f"Independent audit failed; preserve: {failures}")
        report(rows,manifest,out,audits)
        atomic_json(root/"progress.json",{"phase":"complete","completed":4,"total":4,"updated_at":c.now_text()})
        print(f"Accepted: {out}",flush=True)


if __name__=="__main__":
    args=parse_args()
    try:
        worker(args) if args.command=="worker" else launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise

