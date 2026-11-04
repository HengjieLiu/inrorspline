#!/usr/bin/env python
"""P66: three rank0 selectively bounded retained-composition registrations."""
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

import selective_bounded_composed as b
import run_oasis_bounded_composed as previous_bounded
import cps2lambda5_cps1 as unbounded

c, d, legacy = b.c, b.d, unbounded.legacy
sha, atomic_json = unbounded.sha, unbounded.atomic_json
METRICS = unbounded.METRICS
VARIANT = "oasis_seed3_rank000_mrdbscp8421_selective_tanh_lambda1151_v1"
METHOD = "direct_cp_8_4_2_1_selective_tanh_composition_ncc9_diffusion"
PROTOCOL = {**d.PROTOCOL, "version": 1,
            "geometry": "endpoint_aligned_cubic_native_samples_then_border_trilinear_retained_composition",
            "parameter_units": "normalized_grid_xyz_latent",
            "parameterization": "bounded: cap_norm*tanh(p/cap_norm); unbounded: controls=p",
            "stage_cps": list(b.CPS), "pyramid_factors": [4,2,1,1],
            "stage_bounded_by_last": {str(n):list(b.stage_mask(n)) for n in (1,2,3)},
            "regularizer": "unmasked_total_composed_field_forward_valid_l2_component_mean_axes_mean",
            "stage_alpha": list(b.ALPHAS), "bound_multiplier": .99, "increments_per_stage": 1,
            "composition": "Phi_previous_o_delta_new", "initialization": "identity_zero_latents_each_stage",
            "interstage_knot_insertion": False, "resample_accumulated_map": False,
            "bspline_prefilter": False, "coordinate_clipping": False}


def run_name(bounded_last,smoke=False):
    b.stage_mask(bounded_last)
    digest = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "") + f"rank000_cps8421_lambda1_1_5_1_boundlast{bounded_last}_comp1_{digest}"


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch")
    launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--sitreg-root",type=Path,required=True)
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    launch.add_argument("--bounded-last",type=int,nargs="+",choices=(1,2,3),default=[1,2,3])
    worker=sub.add_parser("worker")
    worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--smoke",action="store_true")
    worker.add_argument("--bounded-last",type=int,choices=(1,2,3),required=True)
    for parser in (launch,worker):
        parser.add_argument("--case-rank",type=int,choices=(0,),default=0)
        parser.add_argument("--gpu",choices=("3",),default="3")
        parser.add_argument("--stage-alpha",type=float,nargs=4,default=list(b.ALPHAS))
        parser.add_argument("--bound-multiplier",type=float,choices=(.99,),default=.99)
        parser.add_argument("--increments-per-stage",type=int,choices=(1,),default=1)
    args=p.parse_args(argv)
    if tuple(args.stage_alpha)!=b.ALPHAS:
        p.error("Approved stage alpha is exactly 1 1 5 1")
    if args.command=="launch" and len(set(args.bounded_last))!=len(args.bounded_last):
        p.error("Duplicate variants")
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
    if bounded!=bounded_data["candidate"]: raise RuntimeError("Bounded reference mismatch")
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
    row={"variant":VARIANT,"method":METHOD,"run_name":run_name(args.bounded_last,args.smoke),"smoke":args.smoke,
         "bounded_last":args.bounded_last,"stage_bounded":list(b.stage_mask(args.bounded_last)),
         "protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),"case_rank":0,"fixed_id":"0333","moving_id":"0115",
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
    for stage,alpha,bounded in zip(schedule,b.ALPHAS,b.stage_mask(args.bounded_last)):
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
        field_path=stage_dir/"disp_norm_0333_0115.npy"
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


def validate(path,manifest,bounded_last,smoke=False):
    assert_sources(manifest)
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing completion")
    row=c.read_json(path/"summary.json")
    expected={"variant":VARIANT,"method":METHOD,"protocol":PROTOCOL,"stage_alpha":list(b.ALPHAS),
              "case_rank":0,"fixed_id":"0333","moving_id":"0115","direction":"moving_to_fixed",
              "gpu":"3","uid":1009,"gid":1020,"seed":1,"smoke":smoke,
              "bounded_last":bounded_last,"stage_bounded":list(b.stage_mask(bounded_last)),
              "source_hashes":manifest["source_hashes"],"input_hashes":manifest["input_hashes"],
              "pair_file_sha256":manifest["pair_file_sha256"],"initialization":"identity_no_reused_prefix",
              "segmentation_loss":False,"reference_supervision":False,"analytic_regularizer":False}
    for key,value in expected.items():
        if row[key]!=value:
            raise RuntimeError(f"Invalid {key}")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong backend")
    if len(row["stages"])!=4 or len(row["bounds"])!=4:
        raise RuntimeError("Wrong map count")
    for record,stage,alpha,bounded in zip(row["stages"],b.stages(smoke),b.ALPHAS,b.stage_mask(bounded_last)):
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
    if row["offgrid_audit"]["count"]!=100000 or row["offgrid_audit"]["points_sha256"]!=manifest["bounded_reference"]["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Off-grid audit failure")
    if smoke and not row["native_loss_parity"]["passed"]:
        raise RuntimeError("Native loss parity failed")
    return row


def common_prefix_audit(results):
    by_last={r["bounded_last"]:r for r in results}
    checks=[]
    for first,second,count in ((1,2,2),(1,3,1),(2,3,1)):
        for i in range(count):
            a,bb=by_last[first]["stages"][i],by_last[second]["stages"][i]
            ea,eb=a["optimizer"],bb["optimizer"]
            loss_diff=abs(ea["best_loss"]-eb["best_loss"])
            if ea["actual_epochs"]!=eb["actual_epochs"] or ea["best_epoch"]!=eb["best_epoch"] or loss_diff>1e-6:
                raise RuntimeError("Common unbounded prefix epochs/loss mismatch")
            va=np.load(a["dvf_path"],mmap_mode="r")
            vb=np.load(bb["dvf_path"],mmap_mode="r")
            residual=(va-vb)*np.asarray([95.5,111.5,79.5],dtype=np.float32)
            maximum=float(np.abs(residual).max())
            rms=float(np.sqrt(np.mean(residual.astype(np.float64)**2)))
            if maximum>1e-4 or rms>1e-5:
                raise RuntimeError("Common unbounded prefix field mismatch")
            checks.append({"last_a":first,"last_b":second,"cps":a["cps"],
                           "max_abs_voxel":maximum,"rms_voxel":rms,"best_loss_abs_diff":loss_diff,
                           "identical_dvf_sha256":a["dvf_sha256"]==bb["dvf_sha256"]})
    return checks


def report(results,manifest,out):
    if sorted(r["bounded_last"] for r in results)!=[1,2,3]:
        raise RuntimeError("Require all three completed fits")
    common=common_prefix_audit(results)
    old=manifest["bounded_reference"]
    rows=[]; stage_rows=[]
    items=[(f"Bound last{r['bounded_last']} (composed)",r) for r in results]
    items += [("Bound all4 (P64 composed)",old),
              ("Unbounded (historical prolonged)",manifest["unbounded_reference"]),
              ("Dense (historical prolonged prefix)",manifest["dense_reference"])]
    items += [(r["method"],r) for r in manifest["references"]]
    for name,item in items:
        row={"method":name,"rank":0,"fixed_id":"0333","moving_id":"0115",
             **{k:item[k] for k in METRICS}}
        for key in METRICS: row["delta_all_bounded_"+key]=row[key]-old[key]
        rows.append(row)
    for r in results:
        for st in r["stages"]:
            info=st["optimizer"]
            stage_rows.append({"bounded_last":r["bounded_last"],"cps":st["cps"],
                               "bounded":st["bounded"],"alpha":st["alpha"],**st["metrics"],
                               **{k:info[k] for k in ("best_epoch","actual_epochs","training_elapsed_sec",
                                  "stopping_reason","saturated_coefficient_fraction_gt_95pct","max_abs_control_native_voxel")},
                               **st["atomic_corner_audit"]})
    out.mkdir(parents=True,exist_ok=False)
    c.write_csv(out/"comparison.csv",rows)
    c.write_csv(out/"stages.csv",stage_rows)
    atomic_json(out/"results.json",{"complete":True,"comparison":rows,"candidates":results,
                                    "common_prefix_audit":common,"provenance":manifest})
    lines=["# Rank0 selective bounding: cps8/4/2/1", "",
           "Fixed0333 ← moving0115, seed3 rank0, stage lambda[1,1,5,1]. Fresh identity increments; composition at every stage.",
           "last1 bounds cps1; last2 bounds cps2/1; last3 bounds cps4/2/1. Earlier coefficients are unbounded c=p, not prolonged historical prefixes.",
           "Same NCC9/evaluation-grid total-field diffusion and P64 schedules. Historical unbounded/dense are contextual, not pure bounding ablations.", "",
           "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k]:.6f}" for k in METRICS)+" |")
    lines += ["", "## Stage diagnostics", "",
              "| Variant | CPS | Bounded | Dice | FG folding % | Atomic max derivative | Saturation >95% | Best/stop epoch |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for st in stage_rows:
        sat=st["saturated_coefficient_fraction_gt_95pct"]
        sat_text="N/A" if sat is None else f"{sat*100:.3f}%"
        lines.append(f"| last{st['bounded_last']} | {st['cps']} | {st['bounded']} | {st['dice']:.6f} | {st['jdet_le0_pct_fg']:.6f} | {st['max_displacement_jacobian_infinity_norm']:.6f} | {sat_text} | {st['best_epoch']}/{st['actual_epochs']} |")
    lines += ["", "## Off-grid topology and runtime", "",
              "Same100,000 seed1 fixed-raw-positive foreground points as P64; FP64 coordinate Jacobians.",
              "| Variant | Chain nonpositive % | Export nonpositive % | Chain minimum J | Export minimum J | Optimization s | Worker s | Peak GiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        chain=r["offgrid_audit"]["retained_composition"]; export=r["offgrid_audit"]["interpolated_export"]
        lines.append(f"| last{r['bounded_last']} | {chain['nonpositive_percent']:.6f} | {export['nonpositive_percent']:.6f} | {chain['minimum_jacobian']:.6f} | {export['minimum_jacobian']:.6f} | {r['training_elapsed_sec']:.3f} | {r['total_elapsed_sec']:.3f} | {r['peak_cuda_gib']:.3f} |")
    lines += ["", "All bounded elementary maps passed their derivative gates. Unbounded maps and the total chain may fold; those measurements are not filtered out.",
              "Common initial-stage histories/fields agree within1e-6 loss and1e-4 max/1e-5 RMS voxel tolerance; exact hashes are recorded.",
              "Foreground=fixed raw image>0, canonical cropped-interior SD(log J) over positive J and folding J<=0 percent. Canonical absent-label handling unchanged.",
              "One exploratory case, no across-case SD, global topology guarantee or generalization claim. Shared-load runtime is diagnostic."]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    with (out/"comparison.csv").open() as stream: restored=list(csv.DictReader(stream))
    if len(restored)!=8 or any(not math.isclose(float(a[k]),bb[k],rel_tol=0,abs_tol=1e-12) for a,bb in zip(restored,rows) for k in METRICS):
        raise RuntimeError("Report roundtrip mismatch")
    for key in ("reference","bounded_reference"):
        if sha(Path(manifest[key+"_report_path"]))!=manifest[key+"_report_sha256"]:
            raise RuntimeError("Historical reference changed")
    assert_sources(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"production_fits":3,"case_rank":0,"completed_at":c.now_text(),
                                     "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require UID:GID1009:1020")
    root=args.artifact_root/"runs"/VARIANT
    out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): print("Already accepted",flush=True); return
        manifest_path=root/"launch_manifest.json"
        if manifest_path.exists(): manifest=c.read_json(manifest_path); assert_sources(manifest)
        else: manifest=preflight(args); atomic_json(manifest_path,manifest)
        required_mib=8192; completed=0
        for smoke in (True,False):
            for last in args.bounded_last:
                path=root/run_name(last,smoke)
                if (path/"DONE").exists(): row=validate(path,manifest,last,smoke)
                else:
                    if path.exists(): raise RuntimeError("Incomplete attempt preserved; audit before retry")
                    gpu=next(r for r in legacy.gpu_snapshot()["rows"] if r["physical_gpu"]==args.gpu)
                    if gpu["memory_free_mib"]<required_mib: raise RuntimeError("Insufficient GPU3 memory")
                    command=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(manifest_path),
                             "--out-dir",str(path),"--gpu",args.gpu,"--case-rank","0","--stage-alpha","1","1","5","1",
                             "--bounded-last",str(last),"--bound-multiplier","0.99","--increments-per-stage","1"]
                    if smoke: command.append("--smoke")
                    atomic_json(root/"progress.json",{"phase":"smoke" if smoke else "production",
                        "bounded_last":last,"completed":completed,"total":3,"command":command,
                        "gpu_before":gpu,"updated_at":c.now_text()})
                    print(f"Launching {run_name(last,smoke)} on GPU3",flush=True)
                    with (root/(run_name(last,smoke)+".log")).open("x") as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    row=validate(path,manifest,last,smoke)
                if smoke:
                    required_mib=max(required_mib,int(math.ceil(row["peak_cuda_gib"]*1024))+2048)
                    atomic_json(root/f"smoke_last{last}.json",{"passed":True,"required_free_mib":required_mib,"result":row})
                else:
                    completed+=1
                    print(json.dumps({"completed":completed,"bounded_last":last,**{k:row[k] for k in METRICS}}),flush=True)
            if smoke and args.stop_after=="smoke": return
        paths=[root/run_name(n) for n in (1,2,3)]
        if all((p/"DONE").exists() for p in paths):
            rows=[validate(p,manifest,n) for p,n in zip(paths,(1,2,3))]
            report(rows,manifest,out)
            atomic_json(root/"progress.json",{"phase":"complete","completed":3,"total":3,"updated_at":c.now_text()})
            print(f"Accepted report: {out}",flush=True)
        else:
            atomic_json(root/"progress.json",{"phase":"partial","completed":completed,"total":3,"updated_at":c.now_text()})


if __name__=="__main__":
    args=parse_args()
    try:
        worker(args) if args.command=="worker" else launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise

