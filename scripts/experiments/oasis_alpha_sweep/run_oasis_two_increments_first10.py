#!/usr/bin/env python
"""P69: immutable P68 two-increment worker, first10 lambda1151/1111 cohort."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import run_oasis_bounded_two_blocks as pilot

c, b, legacy = pilot.c, pilot.b, pilot.legacy
sha, atomic_json, METRICS = pilot.sha, pilot.atomic_json, pilot.METRICS
VARIANT = "oasis_seed3_first10_mrdbscp8421_tanh_two_increments_lambda1151_1111_v1"
VECTORS = ((1.,1.,5.,1.), (1.,1.,1.,1.))
MODE = "two_increments"
PILOT_VARIANT, PILOT_SOURCES = pilot.VARIANT, pilot.sources
BASE_PROTOCOL = dict(pilot.PROTOCOL)
METHOD = pilot.METHODS[MODE]
SUBSETS = (("ranks0-9",tuple(range(10))), ("nonpilot8",(1,2,3,5,6,7,8,9)))


def weights(values):
    result=tuple(float(x) for x in values)
    if result not in VECTORS: raise ValueError("Only stage alpha1,1,5,1 or1,1,1,1 is approved")
    return result


def tag(values): return "_".join(f"{x:g}" for x in weights(values))


def protocol(values):
    return {**BASE_PROTOCOL,"cohort_version":1,"stage_alpha":list(weights(values)),
            "allowed_mode":MODE,"increments_per_stage":2}


def run_name(rank,values,smoke=False):
    if type(rank) is not int or rank not in range(10): raise ValueError("Require rank0..9")
    digest=hashlib.sha256(json.dumps(protocol(values),sort_keys=True).encode()).hexdigest()[:10]
    return ("smoke_" if smoke else "")+f"rank{rank:03d}_cps8421_lambda{tag(values)}_two_increments_{digest}"


def sources(sitreg_root):
    return {"pilot":PILOT_SOURCES(sitreg_root),"cohort_runner":sha(Path(__file__)),
            "cohort_audit":sha(Path(__file__).with_name("audit_oasis_two_increments_first10.py")),
            "cohort_tests":sha(c.REPO_ROOT/"tests/test_oasis_two_increments_first10.py")}


@contextmanager
def configured(values):
    """Only process-local protocol bindings; original worker/optimizer stay identical."""
    values=weights(values)
    changes={"VARIANT":VARIANT,"PROTOCOL":protocol(values),"RANKS":tuple(range(10)),
             "PLACEMENT":(),"METHODS":{MODE:METHOD},"sources":sources,
             "run_name":lambda rank,mode,smoke=False: run_name(rank,values,smoke)
             if mode==MODE else (_ for _ in ()).throw(ValueError("Only two increments"))}
    old={key:getattr(pilot,key) for key in changes}; old_alpha=b.ALPHAS
    try:
        for key,value in changes.items(): setattr(pilot,key,value)
        b.ALPHAS=values
        yield
    finally:
        b.ALPHAS=old_alpha
        for key,value in old.items(): setattr(pilot,key,value)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    launch=sub.add_parser("launch"); launch.add_argument("--artifact-root",type=Path,required=True)
    launch.add_argument("--gpus",default="0,1,2,3",choices=("0,1,2,3",))
    launch.add_argument("--stop-after",choices=("smoke","full"),default="full")
    worker=sub.add_parser("worker"); worker.add_argument("--manifest",type=Path,required=True)
    worker.add_argument("--out-dir",type=Path,required=True)
    worker.add_argument("--case-rank",type=int,choices=range(10),required=True)
    worker.add_argument("--stage-alpha",type=float,nargs=4,required=True)
    worker.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    worker.add_argument("--smoke",action="store_true")
    args=p.parse_args(argv)
    if args.command=="worker":
        try: args.stage_alpha=weights(args.stage_alpha)
        except ValueError as exc: p.error(str(exc))
        args.mode=MODE; output=args.out_dir
    else: args.gpus=args.gpus.split(","); output=args.artifact_root
    if output.resolve().is_relative_to(c.REPO_ROOT): p.error("External artifacts required")
    return args


def assert_manifest(manifest):
    if sources(Path(manifest["sitreg_root"]))!=manifest["source_hashes"]: raise RuntimeError("Source drift")
    pilot.assert_references(manifest)


def check_audit(row):
    path=Path(row["run_dir"])
    audit=c.read_json(path/"independent_audit.json")
    if not audit["passed"] or len(audit["checks"])!=8 or not audit["offgrid_reproduced"]:
        raise RuntimeError("Missing complete eight-block audit")
    if audit["summary_sha256"]!=sha(path/"summary.json"): raise RuntimeError("Audited summary drift")
    if (audit["case_rank"],audit["mode"],audit["gpu"])!=(row["case_rank"],MODE,row["gpu"]):
        raise RuntimeError("Audit identity mismatch")
    return audit


def preflight(args):
    old_root=args.artifact_root/"runs"/PILOT_VARIANT
    old_report=args.artifact_root/"repo_experiments"/PILOT_VARIANT
    old_manifest=c.read_json(old_root/"launch_manifest.json")
    data=pilot.previous.reference.accepted_data(old_report)
    pilot.assert_sources(old_manifest); pilot.assert_references(old_manifest)
    reused=[]
    for row in data["candidates"]:
        if row["mode"]!=MODE: continue
        if pilot.validate(Path(row["run_dir"]),old_manifest,row["case_rank"],MODE)!=row:
            raise RuntimeError("Pilot report/summary mismatch")
        check_audit(row)
        reused.append(row)
    if {r["case_rank"] for r in reused}!={0,4} or len(reused)!=2: raise RuntimeError("Missing pilot rows")
    with configured(VECTORS[0]): manifest=pilot.preflight(args)
    manifest.update(protocol={**BASE_PROTOCOL,"cohort_version":1,"stage_alpha_options":[list(v) for v in VECTORS],
                              "allowed_mode":MODE,"increments_per_stage":2},
                    gpus=args.gpus,reused_pilot=reused,new_production_fits=18,total_production_fits=20,
                    sample_sd_ddof=1,case_queue=[0,1,4,2,3,5,6,7,8,9],
                    worker_configuration="scoped immutable P68 worker; two increments only")
    for path in (old_report/"ACCEPTED.json",old_report/"results.json",old_root/"launch_manifest.json"):
        manifest["reference_hashes"][str(path)]=sha(path)
    for row in reused:
        for name in ("summary.json","independent_audit.json","DONE"):
            path=Path(row["run_dir"])/name; manifest["reference_hashes"][str(path)]=sha(path)
        case=manifest["cases"][str(row["case_rank"])]
        if row["input_hashes"]!=case["input_hashes"] or row["stage_alpha"]!=list(VECTORS[0]):
            raise RuntimeError("Pilot protocol/input mismatch")
    assert_manifest(manifest)
    return manifest


def validate(path,manifest,rank,values,smoke=False):
    with configured(values): row=pilot.validate(path,manifest,rank,MODE,smoke)
    if row["gpu"] not in manifest["gpus"] or row["increments_per_level"]!=2:
        raise RuntimeError("Wrong device or increment count")
    return row


def require_cohort(rows):
    expected={(rank,v) for rank in range(10) for v in VECTORS}
    if len(rows)!=20 or {(r["case_rank"],weights(r["stage_alpha"])) for r in rows}!=expected:
        raise RuntimeError("Require all20 case/vector combinations")
    if any(not math.isfinite(float(r[k])) for r in rows for k in METRICS):
        raise RuntimeError("Nonfinite cohort metric")


def summarize(rows,method,subset,ranks):
    group=[r for r in rows if r["case_rank"] in ranks]
    if {r["case_rank"] for r in group}!=set(ranks) or len(group)!=len(ranks):
        raise RuntimeError("Incomplete comparison subset")
    out={"method":method,"subset":subset,"n":len(group)}
    for key in METRICS:
        values=[float(r[key]) for r in group]
        if not all(math.isfinite(x) for x in values): raise RuntimeError("Nonfinite summary")
        out[key+"_mean"]=statistics.mean(values); out[key+"_sd"]=statistics.stdev(values)
    return out


def common_prefix(a,bb):
    checks=[]
    if a["offgrid_audit"]["points_sha256"]!=bb["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Off-grid point mismatch")
    for step in range(4):
        x,y=a["blocks"][step],bb["blocks"][step]
        checked=pilot.compare_fields(x["dvf_path"],y["dvf_path"])
        for key in ("best_epoch","actual_epochs"):
            if x["optimizer"][key]!=y["optimizer"][key]: raise RuntimeError("Shared prefix epoch drift")
        for key in ("best_loss","best_data_loss","best_reg_loss"):
            if abs(x["optimizer"][key]-y["optimizer"][key])>1e-6: raise RuntimeError("Shared prefix loss drift")
        checks.append({"case_rank":a["case_rank"],"step":step,**checked})
    return checks


def report(rows,manifest,out):
    require_cohort(rows); assert_manifest(manifest)
    rows=sorted(rows,key=lambda r:(r["case_rank"],VECTORS.index(weights(r["stage_alpha"]))))
    audits=[check_audit(r) for r in rows]
    common=[]; pairs=[]; blocks=[]; per_case=[]; second=[]
    for rank in range(10):
        a,bb=[r for r in rows if r["case_rank"]==rank]
        common.extend(common_prefix(a,bb))
        pairs.append({"case_rank":rank,"difference":"1111-minus-1151",**{k:bb[k]-a[k] for k in METRICS}})
    groups=[]
    for v in VECTORS: groups.append(("Two maps lambda"+tag(v),[r for r in rows if weights(r["stage_alpha"])==v]))
    for last in (4,2):
        groups.append(("Historical one map all bounded" if last==4 else "Historical bound last2",
                       [r for r in manifest["baseline_rows"] if r["bounded_last"]==last]))
    for method in ("VFA (original)","SITReg λ=1 (original)"):
        groups.append((method,[{**r,"case_rank":r["rank"]} for r in manifest["references"] if r["method"]==method]))
    summaries=[]
    for method,group in groups:
        for subset,ranks in SUBSETS: summaries.append(summarize(group,method,subset,ranks))
        for r in group:
            per_case.append({"method":method,"case_rank":r["case_rank"],"fixed_id":r["fixed_id"],
                             "moving_id":r["moving_id"],**{k:r[k] for k in METRICS}})
    diagnostics=[]
    for r in rows:
        config=tag(r["stage_alpha"]); rank=r["case_rank"]
        refs={"vfa":next(x for x in manifest["references"] if x["rank"]==rank and x["method"].startswith("VFA")),
              "sitreg":next(x for x in manifest["references"] if x["rank"]==rank and x["method"].startswith("SITReg")),
              "old_all4":pilot.baseline(manifest,rank)}
        rec={"case_rank":rank,"stage_alpha":config,"reused_pilot":r["variant"]==PILOT_VARIANT,
             "gpu":r["gpu"],"run_dir":r["run_dir"],"training_elapsed_sec":r["training_elapsed_sec"],
             "total_elapsed_sec":r["total_elapsed_sec"],"peak_cuda_gib":r["peak_cuda_gib"],
             "chain_nonpositive_percent":r["offgrid_audit"]["retained_composition"]["nonpositive_percent"],
             "export_nonpositive_percent":r["offgrid_audit"]["interpolated_export"]["nonpositive_percent"],
             **{k:r[k] for k in METRICS}}
        for name,ref in refs.items():
            for k in METRICS: rec["delta_"+name+"_"+k]=r[k]-ref[k]
        diagnostics.append(rec)
        for x in r["blocks"]:
            info=x["optimizer"]
            blocks.append({"case_rank":rank,"stage_alpha":config,"step":x["step_index"],"cps":x["cps"],
                           "block":x["block_index"],"alpha":x["alpha"],**x["metrics"],**x["atomic_corner_audit"],
                           **{k:info[k] for k in ("best_epoch","actual_epochs","best_loss","best_data_loss","best_reg_loss",
                             "training_elapsed_sec","stopping_reason","saturated_coefficient_fraction_gt_95pct")}})
            if x["block_index"]:
                first=r["blocks"][x["step_index"]-1]
                second.append({"case_rank":rank,"stage_alpha":config,"cps":x["cps"],
                    "total_loss_change":info["best_loss"]-first["optimizer"]["best_loss"],
                    **{"delta_"+k:x["metrics"][k]-first["metrics"][k] for k in METRICS}})
    out.mkdir(parents=True,exist_ok=False)
    tables={"per_case.csv":per_case,"summary.csv":summaries,"paired_lambda.csv":pairs,
            "blocks.csv":blocks,"candidate_diagnostics.csv":diagnostics,"second_increment_changes.csv":second}
    for name,table in tables.items(): c.write_csv(out/name,table)
    atomic_json(out/"results.json",{"complete":True,"candidates":rows,"summary":summaries,"per_case":per_case,
        "paired_lambda":pairs,"common_prefix_checks":common,"independent_audits":audits,"diagnostics":diagnostics,
        "second_increment_changes":second,"provenance":manifest})
    lines=["# First10: two bounded increments per level, lambda1151 versus1111","",
        "Cps8→4→2→1;8 retained maps, two independently optimized bounded increments per level.",
        "Only cps2 diffusion lambda changes5→1, applied to both increments; other settings match P68.",
        "18 fresh identity registrations plus2 read-only accepted1151 pilot cases (ranks0/4).",
        "Raw squared local NCC9; unmasked total-field evaluation-grid voxel diffusion; normalized LR0.03/0.01/0.003/0.001.",
        "Exploratory test-case tuning; ranks0/4 informed the design. Nonpilot8 are reported separately.",""]
    for subset,_ in SUBSETS:
        lines += [f"## {subset}: mean ± across-case sample SD","",
                  "| Method | Dice ↑ | HD95 mm ↓ | Symmetric ASD mm ↓ | FG SD(log J) | FG folding % |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in summaries:
            if r["subset"]==subset: lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k+'_mean']:.6f} ± {r[k+'_sd']:.6f}" for k in METRICS)+" |")
        lines.append("")
    lines += ["## Per case","","| Rank | Fixed ← moving | Method | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(per_case,key=lambda x:x["case_rank"]):
        lines.append(f"| {r['case_rank']} | {r['fixed_id']} ← {r['moving_id']} | {r['method']} | "+" | ".join(f"{r[k]:.6f}" for k in METRICS)+" |")
    lines += ["","## Paired change:1111 minus1151","","| Rank | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding pp |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in pairs: lines.append(f"| {r['case_rank']} | "+" | ".join(f"{r[k]:+.6f}" for k in METRICS)+" |")
    lines += ["","## Diagnostics and verification","",
              "| Rank | Lambda | Reused | GPU | Optimization s | Worker s | Peak GiB | Off-grid chain nonpositive % | Off-grid export nonpositive % |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in diagnostics:
        lines.append(f"| {r['case_rank']} | {r['stage_alpha']} | {r['reused_pilot']} | {r['gpu']} | {r['training_elapsed_sec']:.2f} | {r['total_elapsed_sec']:.2f} | {r['peak_cuda_gib']:.3f} | {r['chain_nonpositive_percent']:.6f} | {r['export_nonpositive_percent']:.6f} |")
    lines += ["","All20 results are complete and independently audited (18 fresh audits,2 pilot audits).",
              "All160 block checkpoints/fields/losses/per-label metrics and native elementary corners were replayed; same100k off-grid points per case.",
              "Both vectors share the first4 blocks within1e-6 loss,1e-4 max/1e-5 RMS voxel and exact epoch tolerance.",
              "FG=fixed raw image>0. Canonical regularity uses cropped-interior Jacobians; SDlogJ includes positive J only; folding is J≤0%.",
              "Canonical absent-label exclusions are unchanged. SD is across case-level values (ddof1), not pooled labels/voxels.",
              "Elementary bounds do not imply unconditional zero folding of an exported interpolant. Off-grid probe percentages are a separate diagnostic.",
              "Histories, selected/stopping epochs, control saturation, per-increment changes and paired DL differences are saved in companion CSV/JSON files.",
              "Existing GPU jobs were not altered; runtime is diagnostic. No extra configurations or silent incomplete-group aggregation."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for name,count in (("per_case.csv",60),("summary.csv",12),("paired_lambda.csv",10),("blocks.csv",160),
                       ("candidate_diagnostics.csv",20),("second_increment_changes.csv",80)):
        with (out/name).open() as stream:
            if len(list(csv.DictReader(stream)))!=count: raise RuntimeError("Report count mismatch")
    assert_manifest(manifest)
    atomic_json(out/"ACCEPTED.json",{"complete":True,"total_fits":20,"new_fits":18,"reused_pilot":2,
                "independent_audits":20,"common_prefix_checks":40,"sample_sd_ddof":1,"completed_at":c.now_text(),
                "output_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def start(root,rank,values,gpu,required,smoke=False,audit=False):
    state=next(r for r in legacy.gpu_snapshot()["rows"] if r["physical_gpu"]==gpu)
    if state["memory_free_mib"]<required: return None
    path=root/run_name(rank,values,smoke)
    if audit:
        cmd=[sys.executable,"-u","-B",str(Path(__file__).with_name("audit_oasis_two_increments_first10.py")),
             "--manifest",str(root/"launch_manifest.json"),"--run-dir",str(path),"--gpu",gpu]
    else:
        if path.exists(): raise RuntimeError(f"Attempt preserved for audit: {path}")
        cmd=[sys.executable,"-u","-B",str(Path(__file__)),"worker","--manifest",str(root/"launch_manifest.json"),
             "--out-dir",str(path),"--case-rank",str(rank),"--stage-alpha",*[str(v) for v in values],"--gpu",gpu]
        if smoke: cmd.append("--smoke")
    log=(root/(path.name+("_audit" if audit else "")+".log")).open("x")
    env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
    try: proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    except BaseException: log.close(); raise
    print(json.dumps({"launched":path.name,"audit":audit,"gpu":gpu,"pid":proc.pid,"gpu_before":state}),flush=True)
    return {"process":proc,"log":log,"path":path,"rank":rank,"values":values,"gpu":gpu,"started":time.monotonic()}


def launch(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    root=args.artifact_root/"runs"/VARIANT; out=args.artifact_root/"repo_experiments"/VARIANT
    root.mkdir(parents=True,exist_ok=True)
    with (root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"ACCEPTED.json").exists(): pilot.previous.reference.accepted_data(out); print("Already accepted"); return
        mp=root/"launch_manifest.json"
        if mp.exists(): manifest=c.read_json(mp); assert_manifest(manifest)
        else: manifest=preflight(args); atomic_json(mp,manifest)
        if manifest["gpus"]!=args.gpus: raise RuntimeError("GPU configuration drift")
        required=8192
        for values in VECTORS:
            path=root/run_name(1,values,True)
            if not (path/"DONE").exists():
                item=start(root,1,values,"3",required,True)
                if item is None: raise RuntimeError("Insufficient smoke memory")
                code=item["process"].wait(); item["log"].close()
                if code: raise RuntimeError(f"Smoke failed, preserved: {path}")
            row=validate(path,manifest,1,values,True)
            required=max(required,math.ceil(row["peak_cuda_gib"]*1024)+2048)
            atomic_json(root/f"smoke_lambda{tag(values)}.json",{"passed":True,"required_mib":required,"result":row})
        if args.stop_after=="smoke": return
        rows=list(manifest["reused_pilot"]); todo={}
        for rank in range(10):
            todo[rank]=[]
            for values in VECTORS:
                if rank in (0,4) and values==VECTORS[0]: continue
                path=root/run_name(rank,values)
                if (path/"DONE").exists(): rows.append(validate(path,manifest,rank,values))
                elif path.exists(): raise RuntimeError(f"Incomplete attempt preserved: {path}")
                else: todo[rank].append(values)
        pending=[rank for rank in manifest["case_queue"] if todo[rank]]
        assigned={}; active={}; failures=[]; started=time.monotonic(); completed_at_start=len(rows)
        while pending or assigned or active:
            if not failures:
                for gpu in args.gpus:
                    if gpu not in assigned and pending: assigned[gpu]=pending.pop(0)
                    if gpu in assigned and gpu not in active:
                        rank=assigned[gpu]
                        if not todo[rank]: del assigned[gpu]; continue
                        item=start(root,rank,todo[rank][0],gpu,required)
                        if item: active[gpu]=item
            for gpu,item in list(active.items()):
                code=item["process"].poll()
                if code is None: continue
                item["log"].close(); del active[gpu]
                try:
                    if code: raise RuntimeError(f"Worker exit{code}")
                    row=validate(item["path"],manifest,item["rank"],item["values"])
                    rows.append(row); todo[item["rank"]].pop(0)
                    print(json.dumps({"completed":len(rows),"total":20,"rank":item["rank"],"lambda":tag(item["values"]),
                                      **{k:row[k] for k in METRICS}}),flush=True)
                except BaseException as exc: failures.append({"path":str(item["path"]),"error":str(exc)})
            elapsed=time.monotonic()-started; fresh=len(rows)-completed_at_start
            atomic_json(root/"progress.json",{"phase":"failed_draining" if failures else "production","completed":len(rows),
                "total":20,"reused":2,"new_completed":len(rows)-2,"elapsed_sec":elapsed,
                "eta_production_sec":None if not fresh else elapsed/fresh*(20-len(rows)),
                "active":[{"rank":x["rank"],"stage_alpha":tag(x["values"]),"gpu":g,"pid":x["process"].pid,
                           "run_dir":str(x["path"])} for g,x in active.items()],"pending":pending,"failures":failures,"updated_at":c.now_text()})
            if failures and not active: raise RuntimeError(f"Failures preserved, queue stopped: {failures}")
            if pending or assigned or active: time.sleep(3)
        require_cohort(rows)
        audit_queue={gpu:[] for gpu in args.gpus}; done=2; active={}; failures=[]; started=time.monotonic()
        for row in rows:
            if row["variant"]==PILOT_VARIANT: check_audit(row); continue
            if (Path(row["run_dir"])/"independent_audit.json").exists(): check_audit(row); done+=1
            else: audit_queue[row["gpu"]].append(row)
        initially_done=done
        while any(audit_queue.values()) or active:
            if not failures:
                for gpu,queue in audit_queue.items():
                    if queue and gpu not in active:
                        row=queue[0]; item=start(root,row["case_rank"],weights(row["stage_alpha"]),gpu,required,audit=True)
                        if item: active[gpu]=item
            for gpu,item in list(active.items()):
                code=item["process"].poll()
                if code is None: continue
                item["log"].close(); del active[gpu]
                try:
                    if code: raise RuntimeError(f"Audit exit{code}")
                    check_audit(audit_queue[gpu][0]); audit_queue[gpu].pop(0); done+=1
                    print(json.dumps({"audited":done,"total":20,"rank":item["rank"],"lambda":tag(item["values"])}),flush=True)
                except BaseException as exc: failures.append({"path":str(item["path"]),"error":str(exc)})
            elapsed=time.monotonic()-started; fresh=done-initially_done
            atomic_json(root/"progress.json",{"phase":"audit_failed_draining" if failures else "independent_audits",
                "completed":20,"total":20,"audited":done,"eta_audit_sec":None if not fresh else elapsed/fresh*(20-done),
                "active":[{"gpu":g,"rank":x["rank"],"stage_alpha":tag(x["values"]),"pid":x["process"].pid} for g,x in active.items()],
                "failures":failures,"updated_at":c.now_text()})
            if failures and not active: raise RuntimeError(f"Audit failures preserved: {failures}")
            if any(audit_queue.values()) or active: time.sleep(3)
        atomic_json(root/"progress.json",{"phase":"report_validation","completed":20,"audited":20,"updated_at":c.now_text()})
        report(rows,manifest,out)
        atomic_json(root/"progress.json",{"phase":"complete","completed":20,"audited":20,"updated_at":c.now_text()})
        print(f"Accepted: {out}",flush=True)


if __name__=="__main__":
    args=parse_args()
    try:
        if args.command=="worker":
            with configured(args.stage_alpha): pilot.worker(args)
        else: launch(args)
    except BaseException as exc:
        if args.command=="worker" and args.out_dir.exists():
            atomic_json(args.out_dir/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise
