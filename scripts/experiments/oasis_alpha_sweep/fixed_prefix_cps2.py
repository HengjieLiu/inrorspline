#!/usr/bin/env python
"""Isolated NCC9 cps2-alpha sweep after immutable alpha=1 cps8/4 prefixes."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
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
import run_oasis_dl_ncc_diffusion_one as legacy
import dl_ncc_diffusion as d
import common as c
from launch_oasis_dl_ncc_diffusion_sweep import validate_result as validate_legacy

VARIANT = "oasis_seed3_first10_mrdbscp84_prefixlambda1_cps2_lambda_sweep_v1"
METHOD = "direct_cp_8_4_lambda1_then_cps2_ncc9_diffusion"
ALPHAS = (.1, .2, .5, 1., 2., 5.)
GPUS = ("0", "1", "3")
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
sha = legacy.sha256_file


@dataclass(frozen=True)
class Spec:
    case_rank: int
    cps2_alpha: float
    prefix_alpha: float = 1.
    seed: int = 1
    smoke: bool = False

    def validate(self):
        if self.case_rank not in range(10) or self.prefix_alpha != 1. or self.cps2_alpha not in ALPHAS or self.seed != 1:
            raise ValueError("Require first10, prefix alpha1, approved cps2 alpha and seed1")

    def prefix(self):
        # A smoke refines the SAME full-budget production prefix.
        return d.FitSpec(self.case_rank, "pyramid", 1., "842")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:10]
        return f"{'smoke_' if self.smoke else ''}rank{self.case_rank:03d}_lambda1_1_{c.format_float(self.cps2_alpha)}_{digest}"


def stage_for(spec):
    spec.validate()
    return d.stages(d.FitSpec(spec.case_rank,"pyramid",1.,"842",smoke=spec.smoke))[-1]


def objective_for(fixed, moving, spec):
    shapes=c.build_nested_cp_shapes(tuple(fixed.shape),8,[8,4,2])
    return d.StageObjective(fixed,moving,spec.cps2_alpha,stage_for(spec),shapes[2])


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("w",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False,default=str); f.write("\n"); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def sources():
    return {"legacy":legacy.source_hashes(),"fixed_prefix_cps2.py":sha(Path(__file__))}


def assert_sources(manifest):
    if sources()!=manifest["source_hashes"]:
        raise RuntimeError("Source changed; preserve this run and audit before resuming")


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch")
    launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--gpus",default="0,1,3")
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    worker=sub.add_parser("worker")
    worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--gpu",choices=GPUS,required=True)
    worker.add_argument("--case-rank",type=int,required=True)
    worker.add_argument("--prefix-alpha",type=float,default=1.)
    worker.add_argument("--cps2-alpha",type=float,choices=ALPHAS,required=True)
    worker.add_argument("--smoke",action="store_true")
    args=p.parse_args(argv)
    if args.command=="worker":
        Spec(args.case_rank,args.cps2_alpha,args.prefix_alpha,smoke=args.smoke).validate()
        if args.out_dir.resolve().is_relative_to(c.REPO_ROOT): p.error("External output required")
    elif args.gpus!="0,1,3" or args.artifact_root.resolve().is_relative_to(c.REPO_ROOT):
        p.error("Approved GPUs are 0,1,3; artifact store must be external")
    return args


def worker(args):
    manifest=c.read_json(args.manifest); assert_sources(manifest)
    spec=Spec(args.case_rank,args.cps2_alpha,args.prefix_alpha,smoke=args.smoke)
    spec.validate()
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require UID:GID1009:1020")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); initial_gpu=legacy.gpu_snapshot()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.set_num_threads(4)
    device=c.set_cuda_device(args.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    for name in ("oasis_root","pair_file","dl_code_root","vfa_root"):
        setattr(args,name,Path(manifest["settings"][name]))
    args.prefix_dir=Path(manifest["prefixes"][str(spec.case_rank)]["run_dir"])
    batch,input_hashes=legacy.load_batch(args,device)
    prefix=manifest["prefixes"][str(spec.case_rank)]
    if input_hashes!=prefix["input_hashes"] or sha(args.pair_file)!=prefix["pair_file_sha256"]:
        raise RuntimeError("Prefix input/split mismatch")
    checkpoint=args.prefix_dir/"stage1/best.pt"
    if sha(checkpoint)!=manifest["cps4_checkpoints"][str(spec.case_rank)]["sha256"]:
        raise RuntimeError("Cps4 checkpoint changed")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["summary"]!=prefix["stages"][1]:
        raise RuntimeError("Cps4 stage record mismatch")
    controls=saved["parameter_norm_xyz"].to(device)
    shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,8,[8,4,2])
    if tuple(controls.shape)!=(*shapes[4],3) or not bool(torch.isfinite(controls).all()):
        raise RuntimeError("Invalid cps4 controls")
    parameter=c.cubic_bspline_dyadic_knot_insert_3d(controls,shapes[2]).detach()
    with torch.no_grad():
        before=d.AlignedBSpline(shapes[4],c.IMAGE_SHAPE,c.IMAGE_SHAPE,4,device)(controls)
        after=d.AlignedBSpline(shapes[2],c.IMAGE_SHAPE,c.IMAGE_SHAPE,2,device)(parameter)
        reconstruction=legacy.verify_init(before,after)
    init_hash=hashlib.sha256(parameter.cpu().numpy().tobytes()).hexdigest()
    if init_hash!=manifest["cps4_checkpoints"][str(spec.case_rank)]["cps2_initial_parameter_sha256"]:
        raise RuntimeError("Cps2 initialization differs from preflight")
    del controls,before,after
    row={"variant":VARIANT,"method":METHOD,"spec":asdict(spec),"run_name":spec.name(),
         "stage_alpha":[1.,1.,spec.cps2_alpha],"protocol":d.PROTOCOL,"source_hashes":manifest["source_hashes"],
         "source_revision":manifest["source_revision"],"input_hashes":input_hashes,
         "pair_file_sha256":sha(args.pair_file),"case_rank":spec.case_rank,
         "fixed_id":batch["fixed_id"],"moving_id":batch["moving_id"],"direction":"moving_to_fixed",
         "sampling":"fixed_to_moving_pull_identity_plus_displacement","gpu":args.gpu,
         "visible_cuda_device":"cuda:0","initial_gpu_state":initial_gpu,
         "cuda_backend":c.cuda_backend_metadata(device),"tf32_matmul":False,"tf32_cudnn":False,
         "uid":os.getuid(),"gid":os.getgid(),"seed":1,
         "container_image":manifest["container_image"],"container_image_id":manifest["container_image_id"],
         "cuda_nondeterminism_caveat":"grid_sample backward may remain nondeterministic",
         "representation":"bspline_control","parameter_units":"normalized_grid_xyz",
         "cps":2,"control_shape":list(shapes[2]),"bspline_order":3,
         "prefix_path":str(args.prefix_dir),"source_842_dvf_sha256":prefix["dvf_sha256"],
         "prefix_model_sha256":sha(checkpoint),"prefix_gpu":prefix["gpu"],
         "cps4_checkpoint_path":str(checkpoint),"initial_parameter_sha256":init_hash,
         "initialization_checks":{"cps4_to_cps2":reconstruction,"exact_clone":True},
         "segmentation_loss":False,"feature_loss":False,"reference_dvf_supervision":False,
         "started_at":c.now_text(),"argv":sys.argv}
    if spec.smoke: row["native_parity"]=legacy.native_parity(device,args.dl_code_root,args.vfa_root)
    atomic_json(args.out_dir/"manifest.json",row)
    objective=objective_for(batch["fixed"],batch["moving"],spec)
    if spec.cps2_alpha==1.:
        initial_values=[float(v) for v in objective(parameter)]
        import csv
        with (args.prefix_dir/"stage2/history.csv").open() as history:
            historical_initial=next(csv.DictReader(history))
        errors={k:abs(v-float(historical_initial[k])) for k,v in zip(("loss","data_loss","reg_loss"),initial_values)}
        if max(errors.values())>1e-6: raise RuntimeError(f"Historical cps2 objective mismatch: {errors}")
        row["lambda1_initial_objective_errors"]=errors
    best_controls,info=d.optimize_stage(parameter,objective,stage_for(spec),args.out_dir/"stage2")
    with torch.no_grad(): best=objective.field(best_controls).detach()
    del objective,parameter,best_controls
    atomic_json(args.out_dir/"progress.json",{"phase":"canonical_metrics","optimization_seconds":info["training_elapsed_sec"]})
    field_path=args.out_dir/f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
    np.save(field_path,best.detach().cpu().numpy())
    restored=torch.from_numpy(np.load(field_path)).to(device)
    if restored.shape!=(*c.IMAGE_SHAPE,3) or not bool(torch.isfinite(restored).all()) or not torch.equal(restored,best):
        raise RuntimeError("Invalid DVF serialization")
    vox=restored*d.voxel_scale(c.IMAGE_SHAPE,device=device)
    metrics,labels=legacy.canonical_metrics(vox,batch)
    labels["canonical_legacy_global_ncc_diagnostic"]=labels.pop("dense_image_loss")
    c.write_csv(args.out_dir/"case_metrics.csv",[labels])
    zyx=vox.detach().cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
    regularity=c.jacobian_metrics_voxel(zyx,batch["fixed_mask"].cpu().numpy().astype(bool))
    diffusion=float(c.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None))
    excluded=[label for label in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    row.update(**metrics,**regularity,diffusion=diffusion,stages=prefix["stages"][:2]+[info],
               best_epoch=info["best_epoch"],best_loss=info["best_loss"],excluded_labels=excluded,
               valid_label_count=len(c.OASIS_LABELS)-len(excluded),dvf_path=str(field_path),dvf_sha256=sha(field_path),
               model_path=str(args.out_dir/"stage2/best.pt"),model_sha256=sha(args.out_dir/"stage2/best.pt"),
               training_elapsed_sec=info["training_elapsed_sec"],pipeline_training_elapsed_sec=sum(s["training_elapsed_sec"] for s in prefix["stages"][:2])+info["training_elapsed_sec"],
               total_elapsed_sec=time.monotonic()-started,run_dir=str(args.out_dir),completed_at=c.now_text())
    assert_sources(manifest)
    if not all(math.isfinite(row[k]) for k in (*METRICS,"diffusion","best_loss")): raise RuntimeError("Nonfinite result")
    atomic_json(args.out_dir/"summary.json",row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def validate(path,spec,manifest):
    if not (path/"DONE").is_file(): raise RuntimeError("Missing DONE")
    row=c.read_json(path/"summary.json"); prefix=manifest["prefixes"][str(spec.case_rank)]
    for key,value in (("spec",asdict(spec)),("stage_alpha",[1.,1.,spec.cps2_alpha]),
                      ("source_hashes",manifest["source_hashes"]),("protocol",d.PROTOCOL),
                      ("method",METHOD),("representation","bspline_control"),("direction","moving_to_fixed"),
                      ("parameter_units","normalized_grid_xyz")):
        if row[key]!=value: raise RuntimeError(f"Invalid {key}")
    if (row["fixed_id"],row["moving_id"])!=(prefix["fixed_id"],prefix["moving_id"]): raise RuntimeError("Case mismatch")
    if row["stages"][:2]!=prefix["stages"][:2] or row["source_842_dvf_sha256"]!=prefix["dvf_sha256"] or row["prefix_model_sha256"]!=manifest["cps4_checkpoints"][str(spec.case_rank)]["sha256"]:
        raise RuntimeError("Prefix mismatch")
    if (row["uid"],row["gid"])!=(1009,1020) or row["gpu"] not in GPUS: raise RuntimeError("Wrong execution identity")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Backend drift")
    if row["cps"]!=2 or row["bspline_order"]!=3 or row["control_shape"]!=[84,116,100] or not row["initialization_checks"]["exact_clone"]:
        raise RuntimeError("Cps2 representation/init mismatch")
    if row["initial_parameter_sha256"]!=manifest["cps4_checkpoints"][str(spec.case_rank)]["cps2_initial_parameter_sha256"]: raise RuntimeError("Initialization mismatch")
    if any(row[k] for k in ("segmentation_loss","feature_loss","reference_dvf_supervision")):
        raise RuntimeError("Unexpected supervision")
    if not all(math.isfinite(row[k]) for k in (*METRICS,"diffusion","best_loss")): raise RuntimeError("Nonfinite result")
    last=row["stages"][-1]
    if last["stage"]!=asdict(stage_for(spec)) or last["alpha"]!=spec.cps2_alpha: raise RuntimeError("Wrong cps2 schedule")
    if not 0<=last["best_epoch"]<=last["actual_epochs"]<=stage_for(spec).max_epochs: raise RuntimeError("Invalid epoch")
    if last["best_loss"]>min(last["initial_loss"],last["last_loss"])+1e-6: raise RuntimeError("Wrong best checkpoint")
    if spec.smoke and not row["native_parity"]["passed"]: raise RuntimeError("Failed native parity")
    field=np.load(row["dvf_path"],mmap_mode="r")
    if field.shape!=(*c.IMAGE_SHAPE,3) or field.dtype!=np.float32 or not np.isfinite(field).all(): raise RuntimeError("Invalid expanded field")
    for key in ("model","dvf"):
        if sha(Path(row[key+"_path"]))!=row[key+"_sha256"]: raise RuntimeError("Output hash mismatch")
    return row


def preflight(args,root):
    torch.set_num_threads(4)
    old_root=args.artifact_root/"runs/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1"
    old=c.read_json(old_root/"launch_manifest.json"); state=c.read_json(old_root/"launcher_state.json")
    if legacy.source_hashes()!=old["numerical_source_hashes"]: raise RuntimeError("Historical numerical source drift")
    if sha(Path(old["args"]["pair_file"]))!=old["pair_file_sha256"]: raise RuntimeError("Split changed")
    prefixes={}
    for rank in range(10):
        spec=d.FitSpec(rank,"pyramid",1.,"842")
        row=validate_legacy(Path(state["results"][spec.name()]),spec,old["numerical_source_hashes"])
        prefixes[str(rank)]=row
    ref_root=args.artifact_root/"repo_experiments/oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1"
    fed_root=args.artifact_root/"repo_experiments/oasis_seed3_first100_fedda_training_lambda_sweep_v1"
    ref_accepted=c.read_json(ref_root/"ACCEPTED.json"); fed_accepted=c.read_json(fed_root/"ACCEPTED.json")
    for folder,record,key in ((ref_root,ref_accepted,"output_sha256"),(fed_root,fed_accepted,"report_sha256")):
        for name,digest in record[key].items():
            if sha(folder/name)!=digest: raise RuntimeError("Reference report hash mismatch")
    refs=[]
    for row in c.read_json(ref_root/"results.json")["per_case"]:
        if row["rank"]<10 and row["method"] in ("VFA (original)","SITReg λ=1 (original)"): refs.append(row)
    for row in c.read_json(fed_root/"results.json")["per_case"]:
        if row["rank"]<10 and row["method"]=="FeDDA-1MF lambda1 historical": refs.append(row)
    if len(refs)!=30: raise RuntimeError("Missing first10 references")
    for r in refs:
        p=prefixes[str(r["rank"])]
        if (r["fixed_id"],r["moving_id"])!=(p["fixed_id"],p["moving_id"]): raise RuntimeError("Reference pair mismatch")
    checkpoints={}
    for rank,p in prefixes.items():
        checkpoint=Path(p["run_dir"])/"stage1/best.pt"
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        controls=saved["parameter_norm_xyz"]
        if saved["summary"]!=p["stages"][1] or controls.shape!=(44,60,52,3) or not bool(torch.isfinite(controls).all()):
            raise RuntimeError("Invalid saved cps4 checkpoint")
        if any(st["alpha"]!=1. for st in p["stages"]) or len(p["stages"])!=3: raise RuntimeError("Not the lambda1 prefix")
        shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,8,[8,4,2])
        initial=c.cubic_bspline_dyadic_knot_insert_3d(controls,shapes[2]).detach()
        checkpoints[rank]={"path":str(checkpoint),"sha256":sha(checkpoint),
            "cps2_initial_parameter_sha256":hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            "stage0_checkpoint_sha256":sha(Path(p["run_dir"])/"stage0/best.pt")}
    return {"variant":VARIANT,"cps4_checkpoints":checkpoints,"source_hashes":sources(),"source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=c.REPO_ROOT,text=True).strip(),
            "settings":old["args"],"prefixes":prefixes,"historical_cps2_lambda1":prefixes,"references":refs,
            "reference_report_hashes":{str(ref_root/"results.json"):sha(ref_root/"results.json"),str(fed_root/"results.json"):sha(fed_root/"results.json")},
            "gpus":list(GPUS),"initial_gpu_state":legacy.gpu_snapshot(),"created_at":c.now_text(),
            "container_image":os.environ["INRORSPLINE_CONTAINER_IMAGE"],"container_image_id":os.environ["INRORSPLINE_CONTAINER_IMAGE_ID"],
            "stage_alpha_vectors":[[1.,1.,a] for a in ALPHAS],"historical_run_manifest_sha256":sha(old_root/"launch_manifest.json")}


def report(results,manifest,out):
    expected={(r,a) for r in range(10) for a in ALPHAS}
    if len(results)!=60 or {(r["case_rank"],r["spec"]["cps2_alpha"]) for r in results}!=expected: raise RuntimeError("Incomplete60-case sweep")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in results:
        stage=r["stages"][-1]
        rows.append({"method":f"CPS2 λ={r['spec']['cps2_alpha']:g}","rank":r["case_rank"],
            "fixed_id":r["fixed_id"],"moving_id":r["moving_id"],**{k:r[k] for k in METRICS},
            "best_epoch":stage["best_epoch"],"stopping_epoch":stage["actual_epochs"],"stopping_reason":stage["stopping_reason"],
            "gpu":r["gpu"],"training_seconds":r["training_elapsed_sec"],"total_seconds":r["total_elapsed_sec"]})
    for rank,p in manifest["prefixes"].items():
        rows.append({"method":"Original 8→4→2 λ=1","rank":int(rank),"fixed_id":p["fixed_id"],"moving_id":p["moving_id"],**{k:p[k] for k in METRICS}})
    for ref in manifest["references"]:
        rows.append({k:ref[k] for k in ("method","rank","fixed_id","moving_id",*METRICS)})
    for row in rows:
        prefix=manifest["prefixes"][str(row["rank"])]
        control=next(r for r in results if r["case_rank"]==row["rank"] and r["spec"]["cps2_alpha"]==1.)
        for k in METRICS:
            row["delta_prefix_"+k]=row[k]-prefix[k]
            row["delta_cps2_1_"+k]=row[k]-control[k]
    means=[]
    groups=[f"CPS2 λ={a:g}" for a in ALPHAS]+["Original 8→4→2 λ=1","VFA (original)","SITReg λ=1 (original)","FeDDA-1MF lambda1 historical"]
    for group in groups:
        subset=[r for r in rows if r["method"]==group]
        if sorted(r["rank"] for r in subset)!=list(range(10)): raise RuntimeError("Incomplete group")
        row={"method":group,"n":10}
        for metric in METRICS:
            vals=[r[metric] for r in subset]
            if not all(math.isfinite(v) for v in vals): raise RuntimeError("Nonfinite metric")
            row[metric+"_mean"],row[metric+"_std"]=statistics.mean(vals),statistics.stdev(vals)
        means.append(row)
    reproduce=[]
    for r in results:
        if r["spec"]["cps2_alpha"]!=1.: continue
        old=manifest["historical_cps2_lambda1"][str(r["case_rank"])]
        reproduce.append({"rank":r["case_rank"],**{f"delta_{k}":r[k]-old[k] for k in METRICS},
            "same_dvf_sha256":r["dvf_sha256"]==old["dvf_sha256"],"old_gpu":old["gpu"],"new_gpu":r["gpu"],
            "old_stopping_epoch":old["stages"][-1]["actual_epochs"],"new_stopping_epoch":r["stages"][-1]["actual_epochs"]})
    selected=min(means[:6],key=lambda r:(-r["dice_mean"],r["asd_mean"],r["hd95_mean"]))
    c.write_csv(out/"per_case.csv",rows); c.write_csv(out/"mean_std.csv",means); c.write_csv(out/"lambda1_reproducibility.csv",reproduce)
    atomic_json(out/"results.json",{"complete":True,"means":means,"per_case":rows,"candidates":results,
                                    "lambda1_reproducibility":reproduce,"selected":selected,"provenance":manifest})
    lines=["# Fixed λ=1 cps8→4 prefix → cps2 λ sweep", "",
        "All60 cps2 fits are fresh. No dense refinement. Same seed3 shuffled first10 pairs; mean ± across-case sample SD (ddof=1).",
        "Local squared NCC9 + unmasked total-field voxel diffusion; normalized cps2 LR0.003. Cps8/cps4 λ is1 in every case.",
        "Original DL references use the same ten cases, not their100-case means. Exploratory test-case tuning; shared-workload timing is diagnostic.","",
        "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in means: lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k+'_mean']:.6f} ± {r[k+'_std']:.6f}" for k in METRICS)+" |")
    lines += ["",f"Dice-first selection: {selected['method']}; ties break by lower ASD, then HD95.",
        "Foreground=fixed raw image>0. SD(log J) uses positive Jacobians; folding measures J≤0 percent.",
        "Per-case delta_prefix columns compare against original 8→4→2 λ1; delta_cps2_1 columns compare against fresh cps2 λ1. Historical λ1 reproducibility is separate and not assumed bitwise."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Verify serialized tables before the acceptance marker.
    import csv
    for name,wanted in (("mean_std.csv",means),("per_case.csv",rows),("lambda1_reproducibility.csv",reproduce)):
        with (out/name).open() as f: restored=list(csv.DictReader(f))
        if len(restored)!=len(wanted): raise RuntimeError("Report row mismatch")
        for a,b in zip(restored,wanted):
            for k,v in b.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool) and not math.isclose(float(a[k]),v,abs_tol=1e-12): raise RuntimeError("CSV/JSON mismatch")
    assert_sources(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"fresh_fits":60,"prefixes":10,"reference_rows":30,
        "case_rows":100,"groups":10,"completed_at":c.now_text(),"source_hashes":sources(),
        "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="ACCEPTED.json"}})


def launch(args):
    root=args.artifact_root/"runs"/VARIANT; out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): print("Already accepted",flush=True); return
        manifest_path=root/"launch_manifest.json"
        if manifest_path.exists(): manifest=c.read_json(manifest_path); assert_sources(manifest)
        else:
            manifest=preflight(args,root); atomic_json(manifest_path,manifest)
        state_path=root/"state.json"
        state=c.read_json(state_path) if state_path.exists() else {"attempts":[],"results":{},"phase":"smoke"}
        def save(): atomic_json(state_path,state)
        def execute_specs(specs,phase,required_mib):
            pending=[]; validated={}; active={}; session_start=time.monotonic(); done_session=0
            for spec in specs:
                attempts=[a for a in state["attempts"] if a["name"]==spec.name()]
                complete=next((a for a in attempts if (Path(a["path"])/"DONE").exists()),None)
                if complete:
                    validated[spec.name()]=validate(Path(complete["path"]),spec,manifest)
                    state["results"][spec.name()]=complete["path"]
                else:
                    if attempts: raise RuntimeError(f"Unfinished prior attempt requires audit: {spec.name()}")
                    pending.append((spec,0,0.))
            state["phase"]=phase; save()
            errors=[]
            while pending or active:
                assert_sources(manifest)
                for gpu,(proc,log,record,spec) in list(active.items()):
                    code=proc.poll()
                    if code is None: continue
                    log.close(); del active[gpu]
                    record.update(returncode=code,finished_at=c.now_text(),wall_seconds=time.time()-record["started_unix"])
                    if code==0:
                        try: result=validate(Path(record["path"]),spec,manifest)
                        except Exception as exc: record["validation_error"]=str(exc); code=1
                    if code==0:
                        validated[spec.name()]=result; state["results"][spec.name()]=record["path"]
                        record["status"]="complete"; done_session+=1
                    else:
                        record["status"]="failed"
                        oom="out of memory" in Path(record["log"]).read_text(errors="replace").lower()
                        if oom and record["attempt"]==0: pending.append((spec,1,time.time()+15))
                        else: errors.append(spec.name())
                    save()
                snapshot=legacy.gpu_snapshot()
                available=sorted([r for r in snapshot["rows"] if r["physical_gpu"] in GPUS and r["physical_gpu"] not in active and r["memory_free_mib"]>=required_mib],key=lambda r:-r["memory_free_mib"])
                for gpu_info in available:
                    if not pending: break
                    eligible=next((i for i,(_,_,ready) in enumerate(pending) if time.time()>=ready),None)
                    if eligible is None: break
                    spec,attempt,_=pending.pop(eligible); gpu=gpu_info["physical_gpu"]
                    path=root/"jobs"/spec.name()/f"attempt{attempt}"; path.parent.mkdir(parents=True,exist_ok=True)
                    log_path=path.parent/f"attempt{attempt}.log"
                    command=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(manifest_path),
                             "--out-dir",str(path),"--gpu",gpu,"--case-rank",str(spec.case_rank),
                             "--prefix-alpha","1","--cps2-alpha",str(spec.cps2_alpha)]
                    if spec.smoke: command.append("--smoke")
                    log=log_path.open("x")
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    record={"name":spec.name(),"spec":asdict(spec),"attempt":attempt,"path":str(path),"log":str(log_path),
                            "gpu":gpu,"pid":proc.pid,"status":"running","command":command,"gpu_before":gpu_info,"started_unix":time.time()}
                    state["attempts"].append(record); active[gpu]=(proc,log,record,spec); save()
                    print(json.dumps({"launched":spec.name(),"gpu":gpu,"phase":phase}),flush=True)
                elapsed=time.monotonic()-session_start
                progress={"phase":phase,"completed":len(validated),"total":len(specs),"active":len(active),"pending":len(pending),
                          "failed":errors,"new_completed_this_session":done_session,"elapsed_seconds":elapsed,
                          "eta_seconds":None if not done_session else (len(specs)-len(validated))*elapsed/done_session,
                          "updated_at":c.now_text()}
                atomic_json(root/"progress.json",progress)
                if pending or active: time.sleep(5)
            if errors: raise RuntimeError(f"Failed candidates preserved: {errors}")
            return [validated[spec.name()] for spec in specs]
        smokes=execute_specs([Spec(0,a,smoke=True) for a in (.1,5.)],"smoke",4096)
        required_mib=max(4096,int(math.ceil(max(r["stages"][-1]["peak_cuda_gib"] for r in smokes)*1024))+2048)
        atomic_json(root/"smoke_results.json",{"passed":True,"results":smokes,"required_free_mib":required_mib})
        if args.stop_after=="smoke": return
        fits=execute_specs([Spec(rank,a) for rank in range(10) for a in ALPHAS],"production",required_mib)
        state["phase"]="reporting"; save()
        report(fits,manifest,out)
        state["phase"]="complete"; state["completed_at"]=c.now_text(); save()
        atomic_json(root/"progress.json",{"phase":"complete","completed":60,"total":60,"updated_at":c.now_text()})
        print(json.dumps({"complete":True,"report":str(out)}),flush=True)


if __name__=="__main__":
    args=parse_args()
    try:
        worker(args) if args.command=="worker" else launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise

