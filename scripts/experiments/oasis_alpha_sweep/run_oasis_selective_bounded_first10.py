#!/usr/bin/env python
"""P67: frozen last1/last2/last3/all4 bounded-composition first-ten extension."""
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
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import run_oasis_selective_bounded as p66
import run_oasis_bounded_composed as p64
import cps2lambda5_dense as reference
b, c, d, legacy = p66.b, p66.c, p66.d, p66.legacy
sha, atomic_json, METRICS = p66.sha, p66.atomic_json, p66.METRICS
VARIANT = "oasis_seed3_first10_mrdbscp8421_selective_tanh_lambda1151_v1"
METHOD = p66.METHOD
PROTOCOL = {**p66.PROTOCOL, "stage_bounded_by_last":
            {str(n): [i >= 4-n for i in range(4)] for n in range(1,5)}}


def stage_mask(n):
    if type(n) is not int or n not in range(1,5):
        raise ValueError("Require bounded-last 1,2,3,4")
    return (True,)*4 if n == 4 else b.stage_mask(n)


def run_name(rank, bounded_last, smoke=False):
    if type(rank) is not int or rank not in range(10):
        raise ValueError("Require seed3 shuffled rank0..9")
    stage_mask(bounded_last)
    digest=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "")+f"rank{rank:03d}_cps8421_lambda1_1_5_1_boundlast{bounded_last}_comp1_{digest}"


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch")
    launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--sitreg-root",type=Path,required=True)
    launch.add_argument("--gpus",default="0,1,2,3")
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    worker=sub.add_parser("worker")
    worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--smoke",action="store_true")
    worker.add_argument("--bounded-last",type=int,choices=range(1,5),required=True)
    worker.add_argument("--case-rank",type=int,choices=range(10),required=True)
    worker.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    args=p.parse_args(argv)
    output=args.artifact_root if args.command=="launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT): p.error("External outputs required")
    if args.command=="launch":
        args.gpus=args.gpus.split(",")
        if not args.gpus or len(set(args.gpus))!=len(args.gpus) or any(g not in ("0","1","2","3") for g in args.gpus):
            p.error("Unique physical GPUs from 0,1,2,3 required")
    return args


def sources(sitreg_root):
    return {"historical":p66.sources(sitreg_root),"first10_runner":sha(Path(__file__))}


def assert_sources(manifest):
    if sources(Path(manifest["sitreg_root"]))!=manifest["source_hashes"]:
        raise RuntimeError("Source drift; preserve and audit")


def assert_references(manifest):
    for path,digest in manifest["reference_hashes"].items():
        if sha(Path(path))!=digest: raise RuntimeError("Read-only reference drift")


def preflight(args):
    roots=args.artifact_root
    old_reports={}
    hashes={}
    for key in (p64.VARIANT,p66.VARIANT,reference.VARIANT):
        folder=roots/"repo_experiments"/key
        old_reports[key]=reference.accepted_data(folder)
        for name in ("ACCEPTED.json","results.json"):
            hashes[str(folder/name)]=sha(folder/name)
    old_manifest=c.read_json(roots/"runs"/p66.VARIANT/"launch_manifest.json")
    old_all_manifest=c.read_json(roots/"runs"/p64.VARIANT/"launch_manifest.json")
    reused=[]
    for row in old_reports[p66.VARIANT]["candidates"]:
        checked=p66.validate(Path(row["run_dir"]),old_manifest,row["bounded_last"])
        if checked!=row: raise RuntimeError("Historical selective report mismatch")
        reused.append({"bounded_last":row["bounded_last"],"row":row,
                       "summary_sha256":sha(Path(row["run_dir"])/"summary.json")})
    row=old_reports[p64.VARIANT]["candidate"]
    if p64.validate(Path(row["run_dir"]),old_all_manifest)!=row:
        raise RuntimeError("Historical all-bounded report mismatch")
    reused.append({"bounded_last":4,"row":row,"summary_sha256":sha(Path(row["run_dir"])/"summary.json")})
    if sorted(x["bounded_last"] for x in reused)!=[1,2,3,4]:
        raise RuntimeError("Require four accepted rank0 results")
    data=old_reports[reference.VARIANT]["provenance"]
    settings=data["settings"]; cases={}
    pair_file=Path(settings["pair_file"]); split_sha=sha(pair_file)
    for rank in range(10):
        prior=data["prefixes"][str(rank)]
        case=c.resolve_case(rank=rank,pair_file=pair_file,oasis_root=Path(settings["oasis_root"]))
        if (case.fixed_id,case.moving_id)!=(prior["fixed_id"],prior["moving_id"]) or prior["pair_file_sha256"]!=split_sha:
            raise RuntimeError("Wrong first-ten pair/split")
        input_hashes={}
        for role in ("fixed","moving"):
            for kind in ("img","seg"):
                path=Path(settings["oasis_root"])/"test"/f"{kind}{getattr(case,role+'_id')}.nii.gz"
                input_hashes[path.name]=sha(path)
        if input_hashes!=prior["input_hashes"]: raise RuntimeError("Raw input drift")
        cases[str(rank)]={"case_rank":rank,"fixed_id":case.fixed_id,"moving_id":case.moving_id,
                          "original_case_index":case.original_case_index,"input_hashes":input_hashes}
    refs=[x for x in data["references"] if x["method"] in ("VFA (original)","SITReg λ=1 (original)")]
    if len(refs)!=20 or {(r["method"],r["rank"]) for r in refs}!={(m,r) for m in ("VFA (original)","SITReg λ=1 (original)") for r in range(10)}:
        raise RuntimeError("Require matched first-ten DL references")
    for r in refs:
        case=cases[str(r["rank"])]
        if any(r[k]!=case[k] for k in ("fixed_id","moving_id")) or not all(math.isfinite(r[k]) for k in METRICS):
            raise RuntimeError("Invalid reference")
    for item in reused:
        r=item["row"]
        if r["input_hashes"]!=cases["0"]["input_hashes"] or r["pair_file_sha256"]!=split_sha:
            raise RuntimeError("Rank0 input mismatch")
    return {"variant":VARIANT,"method":METHOD,"protocol":PROTOCOL,"cases":cases,
            "settings":settings,"pair_file_sha256":split_sha,"reused_rank0":reused,"references":refs,
            "reference_hashes":hashes,"source_hashes":sources(args.sitreg_root),
            "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=c.REPO_ROOT,text=True).strip(),
            "source_branch":subprocess.check_output(["git","branch","--show-current"],cwd=c.REPO_ROOT,text=True).strip(),
            "sitreg_root":str(args.sitreg_root),"sitreg_revision":old_manifest["sitreg_revision"],
            "native_bound_audit":p66.native_bound_audit(args.sitreg_root),
            "bounds":old_manifest["bounds"],"gpus":args.gpus,"initial_gpu_state":legacy.gpu_snapshot(),
            "created_at":c.now_text(),"container_image":os.environ["INRORSPLINE_CONTAINER_IMAGE"],
            "container_image_id":os.environ["INRORSPLINE_CONTAINER_IMAGE_ID"],
            "new_production_fits":36,"reused_production_fits":4,
            "initialization":"fresh_identity_not_historical_prefix","sample_sd_ddof":1}


metric_row=p66.metric_row

def worker(args):
    manifest=c.read_json(args.manifest); assert_sources(manifest)
    case=manifest["cases"][str(args.case_rank)]
    if (os.getuid(),os.getgid())!=(1009,1020):
        raise RuntimeError("Require UID:GID1009:1020")
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
    if input_hashes!=case["input_hashes"] or sha(args.pair_file)!=manifest["pair_file_sha256"]:
        raise RuntimeError("Input or split changed")
    if (batch["fixed_id"],batch["moving_id"])!=(case["fixed_id"],case["moving_id"]):
        raise RuntimeError("Wrong case pair")
    if not torch.equal(batch["fixed_mask"].bool(),batch["fixed"]>0):
        raise RuntimeError("Wrong foreground mask")
    row={"variant":VARIANT,"method":METHOD,"run_name":run_name(args.case_rank,args.bounded_last,args.smoke),"smoke":args.smoke,
         "bounded_last":args.bounded_last,"stage_bounded":list(stage_mask(args.bounded_last)),
         "protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),"case_rank":args.case_rank,"fixed_id":case["fixed_id"],"moving_id":case["moving_id"],
         "direction":"moving_to_fixed","sampling":"fixed_to_moving_pull","gpu":args.gpu,"visible_cuda_device":"cuda:0",
         "source_hashes":manifest["source_hashes"],"source_revision":manifest["source_revision"],
         "sitreg_revision":manifest["sitreg_revision"],"input_hashes":input_hashes,
         "pair_file_sha256":manifest["pair_file_sha256"],"initial_gpu_state":initial_gpu,
         "cuda_backend":c.cuda_backend_metadata(device),"tf32_matmul":False,"tf32_cudnn":False,
         "uid":os.getuid(),"gid":os.getgid(),"seed":1,
         "container_image":manifest["container_image"],"container_image_id":manifest["container_image_id"],
         "representation":"retained_composition_of_four_selectively_bounded_bspline_generated_trilinear_maps",
         "bounds":manifest["bounds"],"bound_table_semantics":"reference_lookup_only; applied bounds are recorded per stage",
         "segmentation_loss":False,"reference_supervision":False,
         "analytic_regularizer":False,"initialization":"identity_no_reused_prefix",
         "cuda_nondeterminism_caveat":"grid_sample backward may remain nondeterministic",
         "started_at":c.now_text(),"argv":sys.argv}
    if args.smoke:
        row["native_loss_parity"]=legacy.native_parity(device,args.dl_code_root,args.vfa_root)
    atomic_json(args.out_dir/"manifest.json",row)
    fields=[]; records=[]; peak=0.
    identity=c.make_coordinate_grid(c.IMAGE_SHAPE,device)
    previous_field=torch.zeros_like(identity)
    schedule=b.stages(args.smoke)
    for stage,alpha,bounded in zip(schedule,b.ALPHAS,stage_mask(args.bounded_last)):
        progress={"phase":"optimization","stage":stage.index,"cps":stage.cps,"updated_at":c.now_text()}
        atomic_json(args.out_dir/"progress.json",progress)
        obj=b.Objective(batch["fixed"],batch["moving"],stage,alpha,fields,manifest["bounds"][str(stage.cps)],bounded)
        with torch.no_grad():
            zeros=torch.zeros(*obj.control_shape,3,device=device)
            initialized=b.compose([*fields,obj.atomic(zeros)],identity)-identity
            init_check=legacy.verify_init(previous_field,initialized)
            if args.smoke:
                q=obj.field(zeros)
                ref=d.StageObjective(obj.fixed,obj.moving,alpha,replace(stage,cps=None,factor=1))
                values=[float(v) for v in obj(zeros)]; expected=[float(v) for v in ref(q)]
                if values!=expected:
                    raise RuntimeError("Same-field objective parity failed")
                del q,ref
            del zeros,initialized
        stage_dir=args.out_dir/f"stage{stage.index}"
        best,info=b.optimize(obj,stage_dir)
        checkpoint=torch.load(stage_dir/"best.pt",map_location="cpu",weights_only=False)
        if checkpoint["summary"]!=info or not torch.equal(checkpoint["latent_norm_xyz"].to(device),best):
            raise RuntimeError("Checkpoint latent mismatch")
        with torch.no_grad():
            rebuilt=obj.controls(checkpoint["latent_norm_xyz"].to(device))
            if not torch.equal(rebuilt,checkpoint["controls_norm_xyz"].to(device)):
                raise RuntimeError("Bounded controls mismatch")
            atomic=obj.expand(rebuilt).detach()
            current=b.compose([*fields,atomic],identity)-identity
            values=[float(v) for v in obj(checkpoint["latent_norm_xyz"].to(device))]
            if not math.isclose(values[0],info["best_loss"],rel_tol=1e-6,abs_tol=1e-7):
                raise RuntimeError("Checkpoint loss mismatch")
        del best,checkpoint,rebuilt,obj
        fields.append(atomic)
        atomic_json(args.out_dir/"progress.json",{**progress,"phase":"stage_validation"})
        topology=b.corner_diagnostics(atomic*d.voxel_scale(c.IMAGE_SHAPE,device=device))
        if bounded and (not topology["bound_below_one"] or topology["nonpositive_corner_count"]):
            raise RuntimeError(f"Elementary-map bound failed: {topology}")
        field_path=stage_dir/f"disp_norm_{case['fixed_id']}_{case['moving_id']}.npy"
        np.save(field_path,current.cpu().numpy())
        restored=torch.from_numpy(np.load(field_path)).to(device)
        if not torch.equal(restored,current) or restored.shape!=(*c.IMAGE_SHAPE,3) or not bool(torch.isfinite(restored).all()):
            raise RuntimeError("Invalid field serialization")
        metrics=metric_row(restored,batch,stage_dir)
        record={"index":stage.index,"cps":stage.cps,"alpha":alpha,"bounded":bounded,"optimizer":info,
                "identity_increment_check":init_check,"atomic_corner_audit":topology,"metrics":metrics,
                "model_path":str(stage_dir/"best.pt"),"model_sha256":sha(stage_dir/"best.pt"),
                "dvf_path":str(field_path),"dvf_sha256":sha(field_path)}
        records.append(record); atomic_json(stage_dir/"validated.json",record)
        previous_field=current.detach()
        peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
        del restored
        print(json.dumps({"stage_complete":stage.index,"metrics":{k:metrics[k] for k in METRICS},
                          "atomic_max_derivative":topology["max_displacement_jacobian_infinity_norm"]}),flush=True)
    atomic_json(args.out_dir/"progress.json",{"phase":"offgrid_validation","updated_at":c.now_text()})
    probes=b.offgrid_diagnostics(fields,current,batch["fixed_mask"],count=100000,seed=1)
    # Earlier unbounded maps may fold: measure without rejecting expected topology.
    peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
    row.update(**records[-1]["metrics"],stages=records,offgrid_audit=probes,peak_cuda_gib=peak,
               dvf_path=records[-1]["dvf_path"],dvf_sha256=records[-1]["dvf_sha256"],
               training_elapsed_sec=sum(r["optimizer"]["training_elapsed_sec"] for r in records),
               total_elapsed_sec=time.monotonic()-started,run_dir=str(args.out_dir),completed_at=c.now_text())
    assert_sources(manifest)
    atomic_json(args.out_dir/"summary.json",row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def validate(path,manifest,rank,bounded_last,smoke=False):
    case=manifest["cases"][str(rank)]
    assert_sources(manifest)
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing completion")
    row=c.read_json(path/"summary.json")
    expected={"variant":VARIANT,"method":METHOD,"protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),
              "case_rank":rank,"fixed_id":case["fixed_id"],"moving_id":case["moving_id"],"direction":"moving_to_fixed",
              "uid":1009,"gid":1020,"seed":1,"smoke":smoke,
              "bounded_last":bounded_last,"stage_bounded":list(stage_mask(bounded_last)),
              "source_hashes":manifest["source_hashes"],"input_hashes":case["input_hashes"],
              "pair_file_sha256":manifest["pair_file_sha256"],"initialization":"identity_no_reused_prefix",
              "segmentation_loss":False,"reference_supervision":False,"analytic_regularizer":False}
    for key,value in expected.items():
        if row[key]!=value:
            raise RuntimeError(f"Invalid {key}")
    if row["gpu"] not in manifest["gpus"]: raise RuntimeError("Unapproved GPU")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong backend")
    if len(row["stages"])!=4 or len(row["bounds"])!=4:
        raise RuntimeError("Wrong map count")
    for record,stage,alpha,bounded in zip(row["stages"],b.stages(smoke),b.ALPHAS,stage_mask(bounded_last)):
        info=record["optimizer"]
        if info["stage"]!=asdict(stage) or info["alpha"]!=alpha or info["bound"]!=(manifest["bounds"][str(stage.cps)] if bounded else None) or info["bounded"]!=bounded or record["bounded"]!=bounded:
            raise RuntimeError("Stage schedule or bound mismatch")
        if info["representation"]!=("bounded_bspline_increment" if bounded else "unbounded_bspline_increment"):
            raise RuntimeError("Wrong stage representation")
        if not bounded and any(info[k] is not None for k in ("cap_norm_xyz","saturated_coefficient_fraction_gt_95pct","max_abs_coefficient_ratio")):
            raise RuntimeError("Unbounded stage has misleading applied-cap metadata")
        if not 0<=info["best_epoch"]<=info["actual_epochs"]<=stage.max_epochs:
            raise RuntimeError("Invalid epochs")
        if info["best_loss"]>min(info["initial_loss"],info["last_loss"])+1e-6 or (bounded and info["max_abs_coefficient_ratio"]>1.+1e-6):
            raise RuntimeError("Invalid state selection or cap")
        check=record["identity_increment_check"]
        if check["max_abs_voxel"]>1e-4 or check["rms_voxel"]>1e-5:
            raise RuntimeError("Nonidentity new increment")
        if bounded and (not record["atomic_corner_audit"]["bound_below_one"] or record["atomic_corner_audit"]["nonpositive_corner_count"]):
            raise RuntimeError("Invalid atomic topology")
        for kind in ("model","dvf"):
            if sha(Path(record[kind+"_path"]))!=record[kind+"_sha256"]:
                raise RuntimeError("Output hash mismatch")
        field=np.load(record["dvf_path"],mmap_mode="r")
        if field.dtype!=np.float32 or field.shape!=(*c.IMAGE_SHAPE,3) or not np.isfinite(field).all():
            raise RuntimeError("Invalid field")
        with (Path(record["model_path"]).parent/"history.csv").open() as stream:
            history=list(csv.DictReader(stream))
        best=min(history,key=lambda x:float(x["loss"]))
        if int(best["epoch"])!=info["best_epoch"] or not math.isclose(float(best["loss"]),info["best_loss"],rel_tol=1e-6,abs_tol=1e-7):
            raise RuntimeError("History/best mismatch")
    if not all(math.isfinite(row[k]) for k in METRICS):
        raise RuntimeError("Nonfinite metrics")
    if row["offgrid_audit"]["count"]!=100000:
        raise RuntimeError("Off-grid audit failure")
    if smoke and not row["native_loss_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    return row



def cohort_rows(manifest):
    rows=[]
    for entry in manifest["reused_rank0"]:
        row=entry["row"]
        if sha(Path(row["run_dir"])/"summary.json")!=entry["summary_sha256"]:
            raise RuntimeError("Historical rank0 summary changed")
        # Never modify historical outputs; the copy adds cohort-only labels.
        rows.append({**row,"bounded_last":entry["bounded_last"],"reused":True,
                     "cohort_original_variant":row["variant"]})
    return rows


def aggregate(rows):
    expected={(r,n) for r in range(10) for n in range(1,5)}
    if len(rows)!=40 or {(r["case_rank"],r["bounded_last"]) for r in rows}!=expected:
        raise RuntimeError("Require all forty complete case/variant combinations")
    output=[]
    for last in range(1,5):
        for subset,ranks in (("ranks0-9",range(10)),("ranks1-9",range(1,10))):
            group=[r for r in rows if r["bounded_last"]==last and r["case_rank"] in ranks]
            result={"method":f"Bound last{last}" if last<4 else "All4 bounded",
                    "bounded_last":last,"subset":subset,"n":len(group)}
            for k in METRICS:
                values=[float(r[k]) for r in group]
                if not all(math.isfinite(x) for x in values): raise RuntimeError("Nonfinite cohort metric")
                result[k+"_mean"]=statistics.mean(values)
                result[k+"_sd"]=statistics.stdev(values)
            output.append(result)
    return output


def report(rows,manifest,out):
    summary=aggregate(rows)
    rows=sorted(rows,key=lambda r:(r["case_rank"],r["bounded_last"]))
    common=[]; per_case=[]; stages=[]
    for rank in range(10):
        group=[r for r in rows if r["case_rank"]==rank]
        points={r["offgrid_audit"]["points_sha256"] for r in group}
        if len(points)!=1: raise RuntimeError("Different off-grid points within a case")
        checks=p66.common_prefix_audit([r for r in group if r["bounded_last"]<4])
        common.append({"case_rank":rank,"checks":checks,"offgrid_points_sha256":next(iter(points))})
        all4=next(r for r in group if r["bounded_last"]==4)
        for r in group:
            rec={"method":f"Bound last{r['bounded_last']}" if r["bounded_last"]<4 else "All4 bounded",
                 "case_rank":rank,"bounded_last":r["bounded_last"],"fixed_id":r["fixed_id"],"moving_id":r["moving_id"],
                 "reused_rank0":bool(r.get("reused",False)),"gpu":r["gpu"],
                 "training_elapsed_sec":r["training_elapsed_sec"],"total_elapsed_sec":r["total_elapsed_sec"],
                 "peak_cuda_gib":r["peak_cuda_gib"],"run_dir":r["run_dir"],**{k:r[k] for k in METRICS}}
            for k in METRICS: rec["delta_all4_"+k]=r[k]-all4[k]
            for ref in manifest["references"]:
                if ref["rank"]==rank:
                    prefix="delta_vfa_" if ref["method"].startswith("VFA") else "delta_sitreg_"
                    for k in METRICS: rec[prefix+k]=r[k]-ref[k]
            per_case.append(rec)
            for st,flag in zip(r["stages"],stage_mask(r["bounded_last"])):
                info=st["optimizer"]
                stages.append({"case_rank":rank,"bounded_last":r["bounded_last"],"cps":st["cps"],"bounded":flag,
                               "alpha":st["alpha"],**st["metrics"],
                               **{k:info.get(k) for k in ("best_epoch","actual_epochs","training_elapsed_sec","stopping_reason",
                                     "saturated_coefficient_fraction_gt_95pct","max_abs_control_native_voxel")},
                               **st["atomic_corner_audit"]})
    for method in ("VFA (original)","SITReg λ=1 (original)"):
        for subset,ranks in (("ranks0-9",range(10)),("ranks1-9",range(1,10))):
            group=[r for r in manifest["references"] if r["method"]==method and r["rank"] in ranks]
            result={"method":method,"bounded_last":None,"subset":subset,"n":len(group)}
            for k in METRICS:
                result[k+"_mean"]=statistics.mean(r[k] for r in group)
                result[k+"_sd"]=statistics.stdev(r[k] for r in group)
            summary.append(result)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"ACCEPTED.json").exists(): raise RuntimeError("Preserve accepted report")
    c.write_csv(out/"per_case.csv",per_case)
    c.write_csv(out/"summary.csv",summary)
    c.write_csv(out/"stages.csv",stages)
    c.write_csv(out/"references.csv",manifest["references"])
    atomic_json(out/"results.json",{"complete":True,"summary":summary,"per_case":per_case,"candidates":rows,
                                    "common_prefix_audit":common,"provenance":manifest})
    lines=["# First10 selectively bounded B-spline compositions","",
           "Seed3 shuffled ranks0–9; moving→fixed. Four retained increments cps8→4→2→1; λ=[1,1,5,1].",
           "Local NCC9 (1−mean correlation²), unmasked total-field evaluation-grid voxel diffusion, normalized latent LR0.03/0.01/0.003/0.001.",
           "All variants compose at every stage. Only the specified final increments are tanh-bounded; earlier controls are unbounded.",
           "The four previously accepted rank0 outputs are reused unchanged; ranks1–9 are36 fresh identity registrations.",
           "This extends a rank0 exploratory choice: ranks1–9 are shown separately; this is not an unbiased held-out benchmark.",""]
    for subset in ("ranks0-9","ranks1-9"):
        lines += [f"## {subset}: mean ± across-case sample SD","",
                  "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in summary:
            if r["subset"]==subset:
                lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k+'_mean']:.6f} ± {r[k+'_sd']:.6f}" for k in METRICS)+" |")
        lines.append("")
    lines+=["## Per case","",
            "| Rank | Fixed ← moving | Bounded last | Dice | HD95 mm | ASD mm | FG SD(log J) | FG folding % |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in per_case:
        lines.append(f"| {r['case_rank']} | {r['fixed_id']} ← {r['moving_id']} | {r['bounded_last']} | "+
                     " | ".join(f"{r[k]:.6f}" for k in METRICS)+" |")
    lines += ["","## Interpretation and acceptance","",
              "Mean and sample SD use ten case-level metrics, not pooled labels/voxels. FG is fixed raw image>0; canonical cropped-interior positive-J SD(log J) and J≤0 folding%.",
              "Canonical absent-label exclusion is retained per case/stage. Accuracy and regularity are separate outcomes; no zero-folding guarantee is imposed on exported fields.",
              "Only bounded elementary maps must pass their contraction audit. Later bounds cannot undo all folds from unbounded earlier maps.",
              "Each run reloads every checkpoint, reconstructs its controls and field, checks selected loss/history, serialization, all native-cell corners and100k shared off-grid foreground probes.",
              "Common unbounded stages were checked per case within1e-6 loss,1e-4 max/1e-5 RMS voxel tolerance.",
              "Shared-GPU runtime is diagnostic. No historical field/source modification, tuning, or failed/incomplete group aggregation."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for name,count in (("per_case.csv",40),("summary.csv",12),("stages.csv",160)):
        with (out/name).open() as stream: back=list(csv.DictReader(stream))
        if len(back)!=count: raise RuntimeError("CSV roundtrip count mismatch")
    assert_sources(manifest); assert_references(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"production_fits":40,"new_fits":36,"reused_rank0":4,
                                    "completed_at":c.now_text(),"sample_sd_ddof":1,
                                    "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def start_worker(args,manifest_path,rank,last,gpu,smoke,required_mib):
    state=next(r for r in legacy.gpu_snapshot()["rows"] if r["physical_gpu"]==gpu)
    if state["memory_free_mib"]<required_mib: return None
    path=manifest_path.parent/run_name(rank,last,smoke)
    if path.exists(): raise RuntimeError(f"Attempt preserved; audit before retry: {path}")
    cmd=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(manifest_path),
         "--out-dir",str(path),"--case-rank",str(rank),"--bounded-last",str(last),"--gpu",gpu]
    if smoke: cmd.append("--smoke")
    log=(manifest_path.parent/(path.name+".log")).open("x")
    env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
         "OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
    try: process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    except BaseException: log.close(); raise
    print(json.dumps({"launched":path.name,"gpu":gpu,"pid":process.pid,"gpu_before":state}),flush=True)
    return {"process":process,"log":log,"path":path,"rank":rank,"last":last,"gpu":gpu,
            "command":cmd,"started":time.monotonic(),"gpu_before":state}


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require UID:GID1009:1020")
    root=args.artifact_root/"runs"/VARIANT
    out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): reference.accepted_data(out); print("Already accepted"); return
        manifest_path=root/"launch_manifest.json"
        if manifest_path.exists():
            manifest=c.read_json(manifest_path); assert_sources(manifest); assert_references(manifest)
            if manifest["gpus"]!=args.gpus: raise RuntimeError("GPU assignment changed")
        else:
            manifest=preflight(args); atomic_json(manifest_path,manifest)
        required=8192
        # Generalized case handling and both bounding extremes through all4 stages.
        smoke_gpu="3" if "3" in args.gpus else args.gpus[0]
        for last in (1,4):
            path=root/run_name(1,last,True)
            if not (path/"DONE").exists():
                item=start_worker(args,manifest_path,1,last,smoke_gpu,True,required)
                if item is None: raise RuntimeError("Insufficient memory for smoke")
                status=item["process"].wait(); item["log"].close()
                if status: raise RuntimeError(f"Smoke failed: {path}; preserved")
            row=validate(path,manifest,1,last,True)
            required=max(required,int(math.ceil(row["peak_cuda_gib"]*1024))+2048)
            atomic_json(root/f"smoke_last{last}.json",{"passed":True,"required_free_mib":required,"result":row})
        if args.stop_after=="smoke": return
        rows=cohort_rows(manifest)
        todo={}
        for rank in range(1,10):
            todo[rank]=[]
            for last in range(1,5):
                path=root/run_name(rank,last)
                if (path/"DONE").exists(): rows.append(validate(path,manifest,rank,last))
                elif path.exists(): raise RuntimeError(f"Incomplete attempt preserved: {path}")
                else: todo[rank].append(last)
        pending=[rank for rank in range(1,10) if todo[rank]]
        assignments={}; active={}; failures=[]; durations=[]
        started=time.monotonic()
        while pending or assignments or active:
            # Keep a case's variants on one GPU to minimize backend-induced prefix drift.
            if not failures:
                for gpu in args.gpus:
                    if gpu not in assignments and pending: assignments[gpu]=pending.pop(0)
                    if gpu in assignments and gpu not in active:
                        rank=assignments[gpu]
                        if not todo[rank]: del assignments[gpu]; continue
                        item=start_worker(args,manifest_path,rank,todo[rank][0],gpu,False,required)
                        if item is not None: active[gpu]=item
            for gpu,item in list(active.items()):
                status=item["process"].poll()
                if status is None: continue
                item["log"].close(); del active[gpu]
                if status:
                    failures.append({"rank":item["rank"],"bounded_last":item["last"],"path":str(item["path"]),"exit":status})
                    continue
                try: row=validate(item["path"],manifest,item["rank"],item["last"])
                except BaseException as exc:
                    failures.append({"rank":item["rank"],"bounded_last":item["last"],"path":str(item["path"]),"error":str(exc)})
                    continue
                rows.append(row); todo[item["rank"]].pop(0)
                durations.append(time.monotonic()-item["started"])
                print(json.dumps({"completed":len(rows),"of":40,"rank":item["rank"],"bounded_last":item["last"],
                                  **{k:row[k] for k in METRICS}}),flush=True)
            elapsed=time.monotonic()-started
            fresh=len(rows)-4
            eta=None if not fresh else elapsed/fresh*(40-len(rows))
            atomic_json(root/"progress.json",{"phase":"failed_draining" if failures else "production",
                "completed":len(rows),"reused":4,"new_completed":fresh,"total":40,"elapsed_sec":elapsed,
                "eta_sec_observed_throughput":eta,"active":[{"rank":x["rank"],"bounded_last":x["last"],"gpu":g,
                "pid":x["process"].pid,"run_dir":str(x["path"])} for g,x in active.items()],
                "pending_case_ranks":pending,"failures":failures,"updated_at":c.now_text()})
            if failures and not active:
                raise RuntimeError(f"Failures preserved; queue stopped for audit: {failures}")
            if pending or assignments or active: time.sleep(3)
        atomic_json(root/"progress.json",{"phase":"report_validation","completed":len(rows),"total":40,"updated_at":c.now_text()})
        report(rows,manifest,out)
        atomic_json(root/"progress.json",{"phase":"complete","completed":40,"total":40,"updated_at":c.now_text()})
        print(f"Accepted: {out}",flush=True)


if __name__=="__main__":
    args=parse_args()
    try:
        worker(args) if args.command=="worker" else launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise

