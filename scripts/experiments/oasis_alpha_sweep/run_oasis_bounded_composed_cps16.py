#!/usr/bin/env python
"""P65 rank0 cps16 extension of the immutable bounded-composition pilot."""
from __future__ import annotations

import argparse
import ast
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

import bounded_composed_cps16 as b
import run_oasis_bounded_composed as previous_bounded
import cps2lambda5_cps1 as unbounded

c, d, legacy = b.c, b.d, unbounded.legacy
sha, atomic_json = unbounded.sha, unbounded.atomic_json
METRICS = unbounded.METRICS
VARIANT = "oasis_seed3_rank000_mrdbscp168421_tanh_composed_lambda11151_v1"
METHOD = "direct_cp_16_8_4_2_1_tanh_bounded_composition_ncc9_diffusion"
PROTOCOL = {**d.PROTOCOL, **b.SCHEDULE, "version": 1,
            "geometry": "endpoint_aligned_cubic_native_samples_then_border_trilinear_retained_composition",
            "parameter_units": "normalized_grid_xyz_latent",
            "parameterization": "cap_norm*tanh(latent_norm/cap_norm)",
            "regularizer": "unmasked_total_composed_field_forward_valid_l2_component_mean_axes_mean",
            "stage_alpha": list(b.ALPHAS), "bound_multiplier": .99, "increments_per_stage": 1,
            "composition": "Phi_previous_o_delta_new", "initialization": "identity_zero_latents_each_stage",
            "interstage_knot_insertion": False, "resample_accumulated_map": False,
            "bspline_prefilter": False, "coordinate_clipping": False}


def run_name(smoke=False):
    digest = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "") + f"rank000_cps168421_lambda1_1_1_5_1_tanh099_comp1_{digest}"


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch")
    launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--sitreg-root",type=Path,required=True)
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    worker=sub.add_parser("worker")
    worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--smoke",action="store_true")
    for parser in (launch,worker):
        parser.add_argument("--case-rank",type=int,choices=(0,),default=0)
        parser.add_argument("--gpu",choices=("3",),default="3")
        parser.add_argument("--stage-alpha",type=float,nargs=5,default=list(b.ALPHAS))
        parser.add_argument("--bound-multiplier",type=float,choices=(.99,),default=.99)
        parser.add_argument("--increments-per-stage",type=int,choices=(1,),default=1)
    args=p.parse_args(argv)
    if tuple(args.stage_alpha)!=b.ALPHAS:
        p.error("Approved stage alpha is exactly 1 1 1 5 1")
    output=args.artifact_root if args.command=="launch" else args.out_dir
    if output.resolve().is_relative_to(c.REPO_ROOT):
        p.error("External artifact outputs required")
    return args


def sources(sitreg_root):
    paths=[Path(__file__),Path(b.__file__),Path(b.original.__file__),Path(previous_bounded.__file__),
           sitreg_root/"src/model/sitreg/model.py",
           sitreg_root/"src/algorithm/cubic_b_spline_control_point_upper_bound.py",
           sitreg_root/"src/algorithm/cubic_spline_upsampling.py",
           sitreg_root/"src/util/dimension_order.py"]
    return {"historical":unbounded.sources(),"new_and_sitreg":{str(p):sha(p) for p in paths}}


def assert_sources(manifest):
    if sources(Path(manifest["sitreg_root"]))!=manifest["source_hashes"]:
        raise RuntimeError("Source drift; preserve outputs and audit")


def native_bound_audit(sitreg_root):
    tree=ast.parse((sitreg_root/"src/model/sitreg/model.py").read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_DenseExtractionNetwork")
    assignment=next(n for n in klass.body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=="CONTROL_POINT_UPPER_BOUNDS_LOOKUP" for t in n.targets))
    lookup=ast.literal_eval(assignment.value)
    sys.path.insert(0,str(sitreg_root/"src"))
    module=legacy.import_file("bounded_reference_upper_bound",sitreg_root/"src/algorithm/cubic_b_spline_control_point_upper_bound.py")
    rows=[]
    for cps in b.CPS:
        native=float(module.compute_max_control_point_value([cps]*3,dtype=torch.float64))
        reproduced=b.compute_bound(cps,"sitreg_half_voxel")["bound_control_units"]
        if not all(math.isclose(v,lookup[(cps,)*3],rel_tol=0,abs_tol=1e-12) for v in (native,reproduced)):
            raise RuntimeError("Native SITReg lookup reproduction failed")
        rows.append({"cps":cps,"lookup":lookup[(cps,)*3],"native_algorithm":native,
                     "independent_all_corners_signs":reproduced,"our_geometry":b.compute_bound(cps)})
    return {"passed":True,"rows":rows}


def preflight(args):
    # Validates the read-only P61/P62 references. None of their states initialize us.
    old=unbounded.preflight(args)
    old_root=args.artifact_root/"runs"/unbounded.VARIANT
    old_manifest=c.read_json(old_root/"launch_manifest.json")
    unbounded.assert_sources(old_manifest)
    report_root=args.artifact_root/"repo_experiments"/unbounded.VARIANT
    accepted=unbounded.previous.accepted_data(report_root)
    previous=unbounded.validate(Path(accepted["candidate"]["run_dir"]),old_manifest)
    if previous!=accepted["candidate"]:
        raise RuntimeError("Unbounded report/run mismatch")
    bounded_root=args.artifact_root/"repo_experiments"/previous_bounded.VARIANT
    bounded_data=unbounded.previous.accepted_data(bounded_root)
    bounded_manifest=c.read_json(args.artifact_root/"runs"/previous_bounded.VARIANT/"launch_manifest.json")
    bounded=previous_bounded.validate(Path(bounded_data["candidate"]["run_dir"]),bounded_manifest)
    if bounded!=bounded_data["candidate"]:
        raise RuntimeError("Previous bounded report/run mismatch")
    return {"variant":VARIANT,"method":METHOD,"protocol":PROTOCOL,
            "bounded_reference":bounded,
            "bounded_reference_report_path":str(bounded_root/"results.json"),
            "bounded_reference_report_sha256":sha(bounded_root/"results.json"),
            "settings":old["settings"],"input_hashes":previous["input_hashes"],
            "pair_file_sha256":previous["pair_file_sha256"],"source_hashes":sources(args.sitreg_root),
            "source_revision":old["source_revision"],"source_branch":old["source_branch"],
            "sitreg_root":str(args.sitreg_root),
            "sitreg_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.sitreg_root,text=True).strip(),
            "native_bound_audit":native_bound_audit(args.sitreg_root),
            "bounds":{str(r):b.compute_bound(r) for r in b.CPS},
            "unbounded_reference":previous,"dense_reference":old["dense_reference"],
            "references":[r for r in old["references"] if r["method"] in ("VFA (original)","SITReg λ=1 (original)")],
            "reference_report_path":str(report_root/"results.json"),
            "reference_report_sha256":sha(report_root/"results.json"),
            "gpu":args.gpu,"initial_gpu_state":legacy.gpu_snapshot(),"created_at":c.now_text(),
            "container_image":old["container_image"],"container_image_id":old["container_image_id"]}


def metric_row(field,batch,out):
    vox=field*d.voxel_scale(c.IMAGE_SHAPE,device=field.device)
    acc,labels=legacy.canonical_metrics(vox,batch)
    labels["canonical_legacy_global_ncc_diagnostic"]=labels.pop("dense_image_loss")
    c.write_csv(out/"case_metrics.csv",[labels])
    zyx=vox.cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
    reg=c.jacobian_metrics_voxel(zyx,batch["fixed_mask"].cpu().numpy().astype(bool))
    absent=[label for label in c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    return {**acc,**reg,"diffusion_native":float(c.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None)),
            "excluded_labels":absent,"valid_label_count":len(c.OASIS_LABELS)-len(absent)}


def worker(args):
    manifest=c.read_json(args.manifest); assert_sources(manifest)
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
    if input_hashes!=manifest["input_hashes"] or sha(args.pair_file)!=manifest["pair_file_sha256"]:
        raise RuntimeError("Input or split changed")
    if (batch["fixed_id"],batch["moving_id"])!=("0333","0115"):
        raise RuntimeError("Wrong rank0 pair")
    if not torch.equal(batch["fixed_mask"].bool(),batch["fixed"]>0):
        raise RuntimeError("Wrong foreground mask")
    row={"variant":VARIANT,"method":METHOD,"run_name":run_name(args.smoke),"smoke":args.smoke,
         "protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),"case_rank":0,"fixed_id":"0333","moving_id":"0115",
         "direction":"moving_to_fixed","sampling":"fixed_to_moving_pull","gpu":args.gpu,"visible_cuda_device":"cuda:0",
         "source_hashes":manifest["source_hashes"],"source_revision":manifest["source_revision"],
         "sitreg_revision":manifest["sitreg_revision"],"input_hashes":input_hashes,
         "pair_file_sha256":manifest["pair_file_sha256"],"initial_gpu_state":initial_gpu,
         "cuda_backend":c.cuda_backend_metadata(device),"tf32_matmul":False,"tf32_cudnn":False,
         "uid":os.getuid(),"gid":os.getgid(),"seed":1,
         "container_image":manifest["container_image"],"container_image_id":manifest["container_image_id"],
         "representation":"retained_composition_of_five_bounded_bspline_generated_trilinear_maps",
         "bounds":manifest["bounds"],"segmentation_loss":False,"reference_supervision":False,
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
    for stage,alpha in zip(schedule,b.ALPHAS):
        progress={"phase":"optimization","stage":stage.index,"cps":stage.cps,"updated_at":c.now_text()}
        atomic_json(args.out_dir/"progress.json",progress)
        obj=b.Objective(batch["fixed"],batch["moving"],stage,alpha,fields,manifest["bounds"][str(stage.cps)])
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
            rebuilt=b.capped_controls(checkpoint["latent_norm_xyz"].to(device),obj.cap_norm)
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
        if not topology["bound_below_one"] or topology["nonpositive_corner_count"]:
            raise RuntimeError(f"Elementary-map bound failed: {topology}")
        field_path=stage_dir/"disp_norm_0333_0115.npy"
        np.save(field_path,current.cpu().numpy())
        restored=torch.from_numpy(np.load(field_path)).to(device)
        if not torch.equal(restored,current) or restored.shape!=(*c.IMAGE_SHAPE,3) or not bool(torch.isfinite(restored).all()):
            raise RuntimeError("Invalid field serialization")
        metrics=metric_row(restored,batch,stage_dir)
        record={"index":stage.index,"cps":stage.cps,"alpha":alpha,"optimizer":info,
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
    if probes["retained_composition"]["nonpositive_count"]:
        raise RuntimeError("Retained composition has a nonpositive probed Jacobian; audit required")
    peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
    row.update(**records[-1]["metrics"],stages=records,offgrid_audit=probes,peak_cuda_gib=peak,
               dvf_path=records[-1]["dvf_path"],dvf_sha256=records[-1]["dvf_sha256"],
               training_elapsed_sec=sum(r["optimizer"]["training_elapsed_sec"] for r in records),
               total_elapsed_sec=time.monotonic()-started,run_dir=str(args.out_dir),completed_at=c.now_text())
    assert_sources(manifest)
    atomic_json(args.out_dir/"summary.json",row)
    (args.out_dir/"DONE").write_text(c.now_text()+"\n")


def validate(path,manifest,smoke=False):
    assert_sources(manifest)
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing completion")
    row=c.read_json(path/"summary.json")
    expected={"variant":VARIANT,"method":METHOD,"protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),
              "case_rank":0,"fixed_id":"0333","moving_id":"0115","direction":"moving_to_fixed",
              "gpu":"3","uid":1009,"gid":1020,"seed":1,"smoke":smoke,
              "source_hashes":manifest["source_hashes"],"input_hashes":manifest["input_hashes"],
              "pair_file_sha256":manifest["pair_file_sha256"],"initialization":"identity_no_reused_prefix",
              "segmentation_loss":False,"reference_supervision":False,"analytic_regularizer":False,
              "representation":"retained_composition_of_five_bounded_bspline_generated_trilinear_maps"}
    for key,value in expected.items():
        if row[key]!=value:
            raise RuntimeError(f"Invalid {key}")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong backend")
    if len(row["stages"])!=5 or len(row["bounds"])!=5:
        raise RuntimeError("Wrong map count")
    for record,stage,alpha in zip(row["stages"],b.stages(smoke),b.ALPHAS):
        info=record["optimizer"]
        if info["stage"]!=asdict(stage) or info["alpha"]!=alpha or info["bound"]!=manifest["bounds"][str(stage.cps)]:
            raise RuntimeError("Stage schedule or bound mismatch")
        if not 0<=info["best_epoch"]<=info["actual_epochs"]<=stage.max_epochs:
            raise RuntimeError("Invalid epochs")
        if info["best_loss"]>min(info["initial_loss"],info["last_loss"])+1e-6 or info["max_abs_coefficient_ratio"]>1.+1e-6:
            raise RuntimeError("Invalid state selection or cap")
        check=record["identity_increment_check"]
        if check["max_abs_voxel"]>1e-4 or check["rms_voxel"]>1e-5:
            raise RuntimeError("Nonidentity new increment")
        if not record["atomic_corner_audit"]["bound_below_one"] or record["atomic_corner_audit"]["nonpositive_corner_count"]:
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
    if row["offgrid_audit"]["count"]!=100000 or row["offgrid_audit"]["retained_composition"]["nonpositive_count"]:
        raise RuntimeError("Off-grid audit failure")
    if smoke and not row["native_loss_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    return row


def report(result,manifest,out):
    out.mkdir(parents=True,exist_ok=False)
    un=manifest["unbounded_reference"]; dense=manifest["dense_reference"]; bounded=manifest["bounded_reference"]
    rows=[]
    for name,item in [("Bounded 16→8→4→2→1 [1,1,1,5,1]",result),
                      ("Bounded 8→4→2→1 [1,1,5,1]",bounded),
                      ("Unbounded B-splines [1,1,5,1]",un),("Dense refinement [1,1,5,1]",dense)]+[(r["method"],r) for r in manifest["references"]]:
        row={"method":name,"rank":0,"fixed_id":"0333","moving_id":"0115",**{k:item[k] for k in METRICS}}
        for key in METRICS:
            row["delta_unbounded_"+key]=row[key]-un[key]
            row["delta_dense_"+key]=row[key]-dense[key]
            row["delta_bounded8421_"+key]=row[key]-bounded[key]
        rows.append(row)
    stage_rows=[{"cps":r["cps"],"alpha":r["alpha"],**r["metrics"],
                 **{k:r["optimizer"][k] for k in ("best_epoch","actual_epochs","training_elapsed_sec","stopping_reason","saturated_coefficient_fraction_gt_95pct")},
                 **r["atomic_corner_audit"]} for r in result["stages"]]
    c.write_csv(out/"comparison.csv",rows); c.write_csv(out/"stages.csv",stage_rows)
    atomic_json(out/"results.json",{"complete":True,"comparison":rows,"candidate":result,"provenance":manifest})
    lines=["# Rank0 cps16 bounded compositional B-splines", "",
           "Fixed0333 ← moving0115, shuffled seed3 rank0. One identity-initialized bounded increment per cps16/8/4/2/1; lambda[1,1,1,5,1].",
           "Five retained elementary maps, not a clipped historical prefix or one prolonged cps1 field. Native local squared NCC9 and total-composition evaluation-grid voxel diffusion; no label supervision.","",
           "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k]:.6f}" for k in METRICS)+" |")
    lines += ["", "New minus bounded8421: "+", ".join(f"{k}={rows[0]['delta_bounded8421_'+k]:+.9f}" for k in METRICS)+".",
              "", "New minus unbounded: "+", ".join(f"{k}={rows[0]['delta_unbounded_'+k]:+.9f}" for k in METRICS)+".","",
              "## Per-stage motion bounds and diagnostics", "",
              "| CPS | Control-unit bound | Native-voxel cap/component | Atomic max derivative row sum | Saturation >95% | Best/stop epochs | Optimization s |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["stages"]:
        info=r["optimizer"]; bound=info["bound"]
        lines.append(f"| {r['cps']} | {bound['bound_control_units']:.9f} | {bound['cap_native_voxel']:.9f} | {r['atomic_corner_audit']['max_displacement_jacobian_infinity_norm']:.6f} | {info['saturated_coefficient_fraction_gt_95pct']*100:.3f}% | {info['best_epoch']}/{info['actual_epochs']} | {info['training_elapsed_sec']:.3f} |")
    lines += ["", "## Retained composition versus exported-field interpolation", "",
              "Same100,000 off-grid foreground points, seed1; Jacobians evaluated in FP64. These probes are separate from canonical finite-difference metrics.","",
              "| Representation | Minimum probed Jacobian | Nonpositive count | Nonpositive % |",
              "|---|---:|---:|---:|"]
    for name in ("retained_composition","interpolated_export"):
        x=result["offgrid_audit"][name]
        lines.append(f"| {name} | {x['minimum_jacobian']:.9f} | {x['nonpositive_count']} | {x['nonpositive_percent']:.6f} |")
    lines += ["",f"Optimization {result['training_elapsed_sec']:.3f}s; complete worker {result['total_elapsed_sec']:.3f}s; peak allocated CUDA {result['peak_cuda_gib']:.3f}GiB on physical GPU3.",
              "Foreground=fixed raw image>0; canonical SDlogJ uses positive Jacobians and folding uses J<=0, with the same cropped-interior metric pipeline as references.",
              "One exploratory case, no sample SD or generalization claim. Bounds constrain each elementary sampled map, not every exported-field interpolant or finite-difference metric. No automatic capacity/LR/lambda expansion.",
              "Adding cps16 changes motion capacity, budget and coarsest NCC physical support. Coefficient saturation is diagnostic, not causal proof. Shared workloads make runtime diagnostic only."]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    with (out/"comparison.csv").open() as stream: restored=list(csv.DictReader(stream))
    if len(restored)!=6 or any(not math.isclose(float(a[k]),bb[k],abs_tol=1e-12) for a,bb in zip(restored,rows) for k in METRICS):
        raise RuntimeError("Report validation failed")
    if sha(Path(manifest["bounded_reference_report_path"]))!=manifest["bounded_reference_report_sha256"]:
        raise RuntimeError("Historical bounded reference changed")
    if result["offgrid_audit"]["points_sha256"]!=bounded["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Off-grid probes differ from previous bounded experiment")
    if sha(Path(manifest["reference_report_path"]))!=manifest["reference_report_sha256"]:
        raise RuntimeError("Historical reference changed")
    assert_sources(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"production_fits":1,"case_rank":0,"completed_at":c.now_text(),
                                     "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require UID:GID1009:1020")
    root=args.artifact_root/"runs"/VARIANT; out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): print("Already accepted",flush=True); return
        manifest_path=root/"launch_manifest.json"
        if manifest_path.exists(): manifest=c.read_json(manifest_path); assert_sources(manifest)
        else: manifest=preflight(args); atomic_json(manifest_path,manifest)
        required_mib=8192
        for smoke in (True,False):
            path=root/run_name(smoke)
            if (path/"DONE").exists(): row=validate(path,manifest,smoke)
            else:
                if path.exists(): raise RuntimeError("Incomplete attempt preserved; audit before retry")
                gpu=next(r for r in legacy.gpu_snapshot()["rows"] if r["physical_gpu"]==args.gpu)
                if gpu["memory_free_mib"]<required_mib: raise RuntimeError("Insufficient GPU3 memory")
                command=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(manifest_path),
                         "--out-dir",str(path),"--gpu",args.gpu,"--case-rank","0","--stage-alpha","1","1","1","5","1",
                         "--bound-multiplier","0.99","--increments-per-stage","1"]+(["--smoke"] if smoke else [])
                atomic_json(root/"progress.json",{"phase":"smoke" if smoke else "production","command":command,
                                                  "gpu_before":gpu,"updated_at":c.now_text()})
                print(f"Launching {run_name(smoke)} on GPU3",flush=True)
                with (root/(run_name(smoke)+".log")).open("x") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                row=validate(path,manifest,smoke)
            if smoke:
                required_mib=max(8192,int(math.ceil(row["peak_cuda_gib"]*1024))+2048)
                atomic_json(root/"smoke_results.json",{"passed":True,"required_free_mib":required_mib,"result":row})
                if args.stop_after=="smoke": return
            else: report(row,manifest,out)
        atomic_json(root/"progress.json",{"phase":"complete","completed":1,"total":1,"updated_at":c.now_text()})
        print(f"Accepted report: {out}",flush=True)


if __name__=="__main__":
    args=parse_args()
    try:
        worker(args) if args.command=="worker" else launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise

