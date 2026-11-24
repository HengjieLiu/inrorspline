"""Paired P84 interpretation from accepted metadata; writes external artifacts only.

Run in the configured Docker runtime. Original runs/reports remain immutable.
Bootstrap resamples the ten paired cases, never labels or case-by-weight rows.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

ARMS=("diffusion_fd","diffusion_analytic","be_fd","be_analytic","kl_c1_fd","kl_c1_analytic")
REPO=Path(__file__).resolve().parents[2]
METRICS=("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","log_jdet_pos_frac_fg",
    "analytic_jacobian_fg_std_logj","analytic_jacobian_fg_fraction_le_zero","ndv_frac_fg",
    "diffusion_native_analytic_fixed_fg_voxel","be_native_analytic_fixed_fg_voxel",
    "fd_jacobian_fg_kl_c1","analytic_jacobian_fg_kl_c1","registration_sec","peak_allocated",
    "peak_reserved","nvml_peak_mib","host_rss_peak","updates","loop_sec_per_update")
HIGHER={"dice","log_jdet_pos_frac_fg"}


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def dump(path,value): Path(path).write_text(json.dumps(clean(value),indent=2,sort_keys=True)+"\n")


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k],(dict,list))))
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        writer.writerows({k:r.get(k) for k in keys} for r in rows)


def stat(values):
    a=np.asarray(values,dtype=float)
    good=np.isfinite(a)
    return dict(n=len(a),finite_n=int(good.sum()),mean=float(a.mean()) if good.all() else None,
                sd=float(a.std(ddof=1)) if good.all() and len(a)>1 else None,
                min=float(a.min()) if good.all() else None,max=float(a.max()) if good.all() else None)


def paired(a,b,key,bootstrap):
    """All deltas are B-A, with case-level percentile bootstrap CI."""
    x=np.array([r.get(key,np.nan) for r in a],dtype=float)
    y=np.array([r.get(key,np.nan) for r in b],dtype=float)
    if len(x)!=10 or len(y)!=10: raise ValueError("Only complete paired first10")
    delta=y-x
    if not np.isfinite(delta).all(): return dict(n=10,finite_n=int(np.isfinite(delta).sum()))
    ci=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
    tie=1e-6 if key=="dice" else 1e-10
    improvement=delta if key in HIGHER else -delta
    result=dict(n=10,finite_n=10,a_mean=float(x.mean()),b_mean=float(y.mean()),mean_delta=float(delta.mean()),
                sd_delta=float(delta.std(ddof=1)),median_delta=float(np.median(delta)),ci95_low=float(ci[0]),
                ci95_high=float(ci[1]),b_better=int((improvement>tie).sum()),ties=int((abs(improvement)<=tie).sum()),
                b_worse=int((improvement<-tie).sum()),delta_min=float(delta.min()),delta_max=float(delta.max()))
    if (x>0).all(): result.update(mean_case_ratio=float(np.mean(y/x)),ratio_of_means=float(y.mean()/x.mean()))
    return result


def analyze(run,review,out):
    if out.resolve().is_relative_to(REPO): raise ValueError("Generated analysis must be external")
    if not read(run/"ACCEPTED.json")["complete"]: raise ValueError("Study not accepted")
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/"inputs"; snapshot.mkdir()
    names=("accuracy_regularity_cases.json","timing_cases.json","stage_metrics.json","setting_resources.json",
           "cohort_summary.json","per_label.json","dice_selected_operating_points.json")
    receipts={}
    for name in names:
        shutil.copyfile(review/name,snapshot/name); receipts[name]=digest(snapshot/name)
    source=run/"manifest.json"; manifest=read(source)
    source_checks={name:digest(REPO/name)==expected for name,expected in manifest["source_hashes"].items()}
    if not all(source_checks.values()): raise ValueError("Run source changed since freeze")
    accepted=read(run/"ACCEPTED.json")
    for name in ("manifest.json","ACCEPTED.json","selection.json","status.json"):
        shutil.copyfile(run/name,snapshot/name); receipts[name]=digest(snapshot/name)
    dump(out/"provenance.json",dict(created_at=datetime.now().astimezone().isoformat(),inputs=receipts,
        source_checks=source_checks,analysis_source_sha256=digest(__file__),bootstrap_seed=20261110,
        bootstrap_replicates=20000,delta_convention="B-A",paired_unit="case; n10",
        inference="exploratory post-selection percentile CIs, no multiplicity correction"))
    acc=read(snapshot/"accuracy_regularity_cases.json"); timing=read(snapshot/"timing_cases.json")
    stages=read(snapshot/"stage_metrics.json"); resources=read(snapshot/"setting_resources.json")
    labels=read(snapshot/"per_label.json"); selected=read(snapshot/"dice_selected_operating_points.json")
    aggregates=[r for r in read(snapshot/"cohort_summary.json") if r["subset"]=="first10"]
    cohorts=[r for r in acc if r["phase"]=="cohort" and r["condition"]=="warm"]
    assert len(acc)==len(timing)==721 and len(cohorts)==550 and len(aggregates)==55
    assert len(list(run.glob("blocks/*/*/attempt*/register_ACCEPTED.json")))==116
    assert len(list(run.glob("blocks/*/*/attempt*/eval_ACCEPTED.json")))==116
    assert not list(run.rglob("*FAILED*")) and not list(run.rglob("*FAILURE*"))
    expected_pairs=re.findall(r"\((\d+)\s*,\s*(\d+)\)",(REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt").read_text())[:10]
    groups=defaultdict(list)
    stage_map=defaultdict(list)
    for row in stages: stage_map[(row["block"],row["condition"],row["case_rank"])].append(row)
    time_map={(r["block"],r["condition"],r["case_rank"]):r for r in timing}
    for row in cohorts:
        assert (row["fixed_id"],row["moving_id"])==expected_pairs[row["case_rank"]]
        key=(row["block"],row["condition"],row["case_rank"])
        ss=stage_map[key]; assert sorted(s["index"] for s in ss)==[0,1,2]
        epochs=sum(s["actual_epochs"] for s in ss)
        merged=dict(row,**{k:v for k,v in time_map[key].items() if k not in row},updates=epochs,
                    loop_sec_per_update=sum(s["optimizer_loop_sec"] for s in ss)/epochs,
                    selected_updates=sum(s["best_epoch"] for s in ss),
                    loop_sec=sum(s["optimizer_loop_sec"] for s in ss),
                    capped_stages=sum(s["stopping_reason"]=="max_epochs" for s in ss))
        groups[(row["arm"],row["weight"])].append(merged)
    for key,rows in groups.items():
        rows.sort(key=lambda r:r["case_rank"]); assert [r["case_rank"] for r in rows]==list(range(10))
        original=next(r for r in aggregates if (r["arm"],r["weight"])==key)
        for metric in METRICS:
            if metric+"_mean" in original:
                actual=stat([r.get(metric,np.nan) for r in rows])
                if actual["mean"] is not None:
                    assert math.isclose(actual["mean"],original[metric+"_mean"],rel_tol=1e-12,abs_tol=1e-12),(key,metric)
    # Verify invariant input/source receipts on every warm case, not only IDs.
    input_hashes={}
    for path in run.glob("blocks/cohort/*/attempt*/block.json"):
        block=read(path)
        for row in block["cases"]:
            assert row["source_hashes"]==manifest["source_hashes"]
            assert row["backend"]["cudnn_benchmark"] is False and row["backend"]["cudnn_deterministic"] is True
            rank=row["case_rank"]
            if rank in input_hashes: assert input_hashes[rank]==row["input_hashes"]
            input_hashes[rank]=row["input_hashes"]
    bootstrap=np.random.default_rng(20261110).integers(0,10,size=(20000,10))
    cases=[]; pairs=[]; common_labels=[]
    label_map=defaultdict(dict)
    for row in labels:
        if row["phase"]=="cohort" and row["condition"]=="warm":
            label_map[(row["arm"],row["weight"],row["case_rank"])][row["label"]]=row

    def compare(name,a_key,b_key,kind):
        a,b=groups[a_key],groups[b_key]
        for metric in METRICS:
            result=paired(a,b,metric,bootstrap)
            pairs.append(dict(comparison=name,kind=kind,a_arm=a_key[0],a_weight=a_key[1],b_arm=b_key[0],b_weight=b_key[1],metric=metric,**result))
        for ar,br in zip(a,b):
            delta={metric:br[metric]-ar[metric] if ar.get(metric) is not None and br.get(metric) is not None else None for metric in METRICS}
            cases.append(dict(comparison=name,kind=kind,case_rank=ar["case_rank"],fixed_id=ar["fixed_id"],moving_id=ar["moving_id"],**delta))
            al=label_map[(*a_key,ar["case_rank"])]; bl=label_map[(*b_key,br["case_rank"])]
            eligible=[i for i in al if al[i]["eligible"] and bl[i]["eligible"]]
            common_labels.append(dict(comparison=name,case_rank=ar["case_rank"],eligible_count=len(eligible),
                **{metric+"_delta":np.mean([bl[i][metric]-al[i][metric] for i in eligible]) for metric in ("dice","hd95","asd_symmetric")}))

    for family in ("diffusion","be","kl_c1"):
        fd,an=family+"_fd",family+"_analytic"
        shared=sorted(set(w for a,w in groups if a==fd)&set(w for a,w in groups if a==an))
        for weight in shared: compare(f"{family}_analytic_minus_fd_w{weight:g}",(fd,weight),(an,weight),"same_weight_backend")
    for family in ("diffusion","be","kl_c1"):
        fd,an=family+"_fd",family+"_analytic"
        compare(f"selected_{family}_analytic_minus_fd",(fd,selected[fd]["weight"]),(an,selected[an]["weight"]),"selected_backend")
    for backend in ("fd","analytic"):
        for a,b in itertools.combinations(("diffusion","be","kl_c1"),2):
            aa,bb=a+"_"+backend,b+"_"+backend
            compare(f"selected_{bb}_minus_{aa}",(aa,selected[aa]["weight"]),(bb,selected[bb]["weight"]),"selected_family")
        compare(f"be10_minus_kl003_{backend}",(f"kl_c1_{backend}",.03),(f"be_{backend}",10.),"observed_alternative")
    for arm in ARMS: compare(f"selected_{arm}_minus_none",("none",0.),(arm,selected[arm]["weight"]),"zero_control")
    for name,rows in (("paired_summary",pairs),("paired_cases",cases),("common_label_sensitivity",common_labels)):
        dump(out/(name+".json"),rows); csv_write(out/(name+".csv"),rows)
    common_summary=[]
    for name in sorted(set(r["comparison"] for r in common_labels)):
        selected_common=sorted([r for r in common_labels if r["comparison"]==name],key=lambda r:r["case_rank"])
        for metric in ("dice","hd95","asd_symmetric"):
            delta=np.array([r[metric+"_delta"] for r in selected_common])
            ci=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
            common_summary.append(dict(comparison=name,metric=metric,mean_delta=delta.mean(),ci95_low=ci[0],ci95_high=ci[1],
                min_shared_labels=min(r["eligible_count"] for r in selected_common),n=10))
    dump(out/"common_label_paired_summary.json",common_summary); csv_write(out/"common_label_paired_summary.csv",common_summary)
    selected_rows=[]
    stage_rows=[]
    for arm in (*ARMS,"none"):
        weight=0. if arm=="none" else selected[arm]["weight"]
        rows=groups[(arm,weight)]
        item=dict(arm=arm,weight=weight,boundary_warning=selected.get(arm,{}).get("boundary_warning",False))
        for metric in METRICS: item.update({metric+"_"+k:v for k,v in stat([r.get(metric,np.nan) for r in rows]).items()})
        item["rank0_dice"]=rows[0]["dice"]; item["ranks1_9_dice_mean"]=np.mean([r["dice"] for r in rows[1:]])
        item["capped_stage_count"]=sum(r["capped_stages"] for r in rows)
        item["eligible_labels_by_case"]=[sum(lr["eligible"] for lr in label_map[(arm,weight,r["case_rank"])].values()) for r in rows]
        resource=next(r for r in resources if r["phase"]=="cohort" and r["arm"]==arm and r["weight"]==weight)
        item["resources"]=resource
        selected_rows.append(item)
        for i in range(3):
            ss=[s for s in stages if s["phase"]=="cohort" and s["condition"]=="warm" and s["arm"]==arm and s["weight"]==weight and s["index"]==i]
            stage_rows.append(dict(arm=arm,weight=weight,stage=i,cps=ss[0]["cps"],
                mean_updates=np.mean([s["actual_epochs"] for s in ss]),mean_selected_updates=np.mean([s["best_epoch"] for s in ss]),
                mean_loop_sec=np.mean([s["optimizer_loop_sec"] for s in ss]),
                mean_sec_per_update=np.mean([s["optimizer_loop_sec"]/s["actual_epochs"] for s in ss]),
                capped=sum(s["stopping_reason"]=="max_epochs" for s in ss)))
    dump(out/"selected_settings.json",selected_rows); csv_write(out/"selected_settings.csv",selected_rows)
    dump(out/"selected_stage_cost.json",stage_rows); csv_write(out/"selected_stage_cost.csv",stage_rows)
    # Pair cold0 with warm0 for each setting; differing update counts are explicit.
    coldwarm=[]
    for resource in resources:
        if resource["phase"] not in ("pilot","cohort"): continue
        both=[r for r in timing if r["block"]==resource["block"] and r["case_rank"]==0]
        cold=next(r for r in both if r["condition"]=="cold"); warm=next(r for r in both if r["condition"]=="warm")
        cold_stages=stage_map[(cold["block"],"cold",0)]; warm_stages=stage_map[(warm["block"],"warm",0)]
        ca=next(r for r in acc if r["block"]==cold["block"] and r["condition"]=="cold" and r["case_rank"]==0)
        wa=next(r for r in acc if r["block"]==warm["block"] and r["condition"]=="warm" and r["case_rank"]==0)
        coldwarm.append(dict(phase=resource["phase"],arm=resource["arm"],weight=resource["weight"],gpu=resource["gpu"],
            cold_sec=cold["registration_sec"],warm0_sec=warm["registration_sec"],
            gap_sec=cold["registration_sec"]-warm["registration_sec"],dice_cold_minus_warm=ca["dice"]-wa["dice"],
            cold_updates=sum(s["actual_epochs"] for s in cold_stages),warm_updates=sum(s["actual_epochs"] for s in warm_stages),
            startup_sec=cold["worker_startup_sec"],load_sec=cold["case_load_sec"],launch_to_dvf_sec=cold["launch_to_dvf_sec"]))
    dump(out/"paired_cold_warm.json",coldwarm); csv_write(out/"paired_cold_warm.csv",coldwarm)
    tuning=[]
    for arm in (*ARMS,"none"):
        for phase in ("pilot","cohort"):
            tr=[r for r in timing if r["phase"]==phase and r["arm"]==arm]
            sr=[s for s in stages if s["phase"]==phase and s["arm"]==arm]
            er=[r for r in acc if r["phase"]==phase and r["arm"]==arm]
            tuning.append(dict(phase=phase,arm=arm,registrations=len(tr),settings=len(set(r["weight"] for r in tr)),
                registration_endpoint_sec=sum(r["registration_sec"] for r in tr),
                evaluation_sec=sum(r["metric_seconds"] for r in er),
                max_allocated=max(r["peak_allocated"] for r in tr),max_reserved=max(r["peak_reserved"] for r in tr),
                capped_stages=sum(s["stopping_reason"]=="max_epochs" for s in sr),stages=len(sr)))
    dump(out/"tuning_cost.json",tuning); csv_write(out/"tuning_cost.csv",tuning)
    dominance=[]
    metrics=("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","analytic_jacobian_fg_std_logj","analytic_jacobian_fg_fraction_le_zero","ndv_frac_fg")
    for backend in ("fd","analytic"):
        target=("kl_c1_"+backend,.03)
        for key,rows in groups.items():
            if key[0] not in ("diffusion_"+backend,"be_"+backend): continue
            means={k:np.mean([r[k] for r in rows]) for k in metrics}; targets={k:np.mean([r[k] for r in groups[target]]) for k in metrics}
            dominant=all(means[k]>=targets[k] if k=="dice" else means[k]<=targets[k] for k in metrics)
            dominance.append(dict(target_arm=target[0],target_weight=.03,alternative_arm=key[0],alternative_weight=key[1],
                all_eight_mean_metrics_better=dominant,**means))
    dump(out/"observed_dominance.json",dominance); csv_write(out/"observed_dominance.csv",dominance)
    eval_resources=[]
    for path in sorted(run.glob("blocks/*/*/attempt*/eval_telemetry.json")):
        block=read(path.parent/"block.json"); job=block["job"]; telemetry=read(path)
        eval_resources.append(dict(phase=job["phase"],arm=job["arm"],weight=job["weight"],gpu=job["gpu"],
            sampled_process_gpu_peak_mib=telemetry["process_gpu_peak_mib"],sampled_host_rss_peak=telemetry["host_rss_peak"],
            samples=telemetry["samples"],contamination_count=len(telemetry["contaminated"])))
    dump(out/"evaluation_resources.json",eval_resources); csv_write(out/"evaluation_resources.csv",eval_resources)
    launch=datetime.fromisoformat(read(run/"controller.json")["started_at"].replace("Z","+00:00"))
    completion=datetime.fromisoformat(accepted["at"].replace("Z","+00:00"))
    quality=dict(accepted=True,timing_rows=len(timing),evaluated_rows=len(acc),warm_first10_rows=len(cohorts),groups=len(groups),
        total_label_rows=len(labels),warm_primary_label_rows=sum(r["phase"]=="cohort" and r["condition"]=="warm" for r in labels),
        sources_unchanged=all(source_checks.values()),v2_failures=0,
        registration_endpoint_hours=sum(r["registration_sec"] for r in timing)/3600,
        total_evaluation_hours=sum(r["metric_seconds"] for r in acc)/3600,
        selected_eligible_counts={r["arm"]:r["eligible_labels_by_case"] for r in selected_rows},
        coldwarm_cohort_max_abs_dice_difference=max(abs(r["dice_cold_minus_warm"]) for r in coldwarm if r["phase"]=="cohort"),
        coldwarm_cohort_changed_update_counts=sum(r["cold_updates"]!=r["warm_updates"] for r in coldwarm if r["phase"]=="cohort"),
        production_wall_seconds=(completion-launch).total_seconds(),
        evaluation_max_sampled_gpu_gib=max(r["sampled_process_gpu_peak_mib"] for r in eval_resources)/1024,
        evaluation_max_sampled_host_rss_gib=max(r["sampled_host_rss_peak"] for r in eval_resources)/1024**3,
        selected_settings=[dict(arm=r["arm"],weight=r["weight"]) for r in selected_rows])
    dump(out/"audit.json",quality)
    make_plots(out,groups,selected,selected_rows)
    make_markdown(out,selected_rows,pairs,stage_rows,quality,tuning,coldwarm,read(run/"selection.json"),common_summary)
    print(json.dumps(clean(dict(out=str(out),audit=quality,selected=[{k:r[k] for k in ("arm","weight","dice_mean","registration_sec_mean","updates_mean")} for r in selected_rows])),indent=2))


def make_plots(out,groups,selected,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"svg.fonttype":"none"})
    colors={"fd":"#0072B2","analytic":"#D55E00"}
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for col,family in enumerate(("diffusion","be","kl_c1")):
        for backend in ("fd","analytic"):
            keys=sorted([k for k in groups if k[0]==family+"_"+backend],key=lambda k:k[1])
            for row,metric in enumerate(("dice","hd95","asd")):
                means=[np.mean([r[metric] for r in groups[k]]) for k in keys]
                axes[row,col].plot([k[1] for k in keys],means,"o-",label=backend,color=colors[backend])
                axes[row,col].set(xscale="log",xlabel="Regularizer weight",ylabel=metric,title=family if row==0 else None)
        axes[0,col].legend()
    fig.suptitle("First10 accuracy: independently tuned regularizers; means over paired cases")
    fig.tight_layout(); fig.savefig(out/"accuracy_weight.png",dpi=180); fig.savefig(out/"accuracy_weight.pdf"); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    palette={"diffusion":"#009E73","be":"#0072B2","kl_c1":"#D55E00"}
    for arm in ARMS:
        family=arm.removesuffix("_analytic").removesuffix("_fd")
        keys=sorted([k for k in groups if k[0]==arm],key=lambda k:k[1])
        line="-" if arm.endswith("analytic") else "--"
        for col,metric in enumerate(("dice","hd95","asd")):
            y=[np.mean([r[metric] for r in groups[k]]) for k in keys]
            for row,xmetric in enumerate(("log_jdet_std_fg","analytic_jacobian_fg_fraction_le_zero")):
                x=[np.mean([r[xmetric] for r in groups[k]])*(100 if row else 1) for k in keys]
                axes[row,col].plot(x,y,"o"+line,color=palette[family],label=arm,markersize=4)
                axes[row,col].set(xlabel="Analytic J≤0 (%)" if row else "Canonical foreground SD(logJ)",ylabel=metric)
                if row: axes[row,col].set_xscale("symlog",linthresh=.01)
    axes[0,0].legend(fontsize=8); fig.suptitle("Accuracy–regularity trade-offs; symlog folding axis retains exact zeros")
    fig.tight_layout(); fig.savefig(out/"accuracy_regularity.png",dpi=180); fig.savefig(out/"accuracy_regularity.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for arm in ARMS:
        rows=groups[(arm,selected[arm]["weight"])]; family=arm.removesuffix("_analytic").removesuffix("_fd")
        marker="s" if arm.endswith("analytic") else "o"
        for col,(x,y,factor) in enumerate((("jdet_le0_pct_fg","analytic_jacobian_fg_fraction_le_zero",100),
            ("log_jdet_std_fg","analytic_jacobian_fg_std_logj",1),("log_jdet_std_fg","ndv_frac_fg",1))):
            axes[col].scatter([r[x] for r in rows],[factor*r[y] for r in rows],color=palette[family],marker=marker,label=arm,s=32)
    axes[0].set(xlabel="Central-difference J≤0 (%)",ylabel="Analytic J≤0 (%)")
    axes[0].plot([0,18],[0,18],color="grey",ls=":")
    axes[1].set(xlabel="Canonical SD(logJ)",ylabel="Analytic SD(logJ)")
    axes[1].plot([0,1.3],[0,1.3],color="grey",ls=":")
    axes[2].set(xlabel="Canonical SD(logJ)",ylabel="NDV index (not folded-voxel fraction)")
    axes[1].legend(fontsize=7); fig.suptitle("Dice-selected settings: Jacobian conventions can disagree")
    fig.tight_layout(); fig.savefig(out/"jacobian_convention_audit.png",dpi=180); fig.savefig(out/"jacobian_convention_audit.pdf"); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,11))
    for i,family in enumerate(("diffusion","be","kl_c1")):
        fd=groups[(family+"_fd",selected[family+"_fd"]["weight"])]; an=groups[(family+"_analytic",selected[family+"_analytic"]["weight"])]
        for col,(metric,factor) in enumerate((("dice",1),("registration_sec",1),("peak_allocated",1/1024**3))):
            x=[r[metric]*factor for r in fd]; y=[r[metric]*factor for r in an]
            axes[i,col].scatter(x,y,color=palette[family]); low=min(x+y); high=max(x+y)
            axes[i,col].plot([low,high],[low,high],color="grey",ls=":")
            for rank,(xx,yy) in enumerate(zip(x,y)): axes[i,col].annotate(str(rank),(xx,yy),fontsize=8)
            axes[i,col].set(xlabel="FD "+metric,ylabel="Analytic "+metric,title=family)
    fig.suptitle("Paired cases at Dice-selected weights; memory in GiB; device/step counts differ")
    fig.tight_layout(); fig.savefig(out/"paired_backend_accuracy_resources.png",dpi=180); fig.savefig(out/"paired_backend_accuracy_resources.pdf"); plt.close(fig)


def make_markdown(out,rows,pairs,stages,audit,tuning,coldwarm,selection,common_summary):
    def f(v,n=4): return "NA" if v is None else f"{v:.{n}f}"
    def mean_sd(r,k,n=4): return f(r[k+"_mean"],n)+" ± "+f(r[k+"_sd"],n)
    lines=["# Detailed paired review: OASIS six-arm regularizer tuning","",
        "All data are from accepted execution oasis_reg6_20261110T0917Z_v2. This review adds analysis without changing the original run or report.","",
        "## Completion and interpretation boundaries","",
        f"{audit['timing_rows']} registration/evaluation rows:6 smokes,110 coarse pilot observations and605 final observations. Final primary accuracy uses550 warm observations:54 positive settings ×10 cases plus one ten-case zero control. All55 groups are complete; no v2 failures. All frozen sources and per-case input identities verified.","",
        "Hyperparameters and comparisons are exploratory first10 test-set selections. The paired sampling unit is the case, n=10. Percentile bootstrap intervals resample cases20,000 times, seed20261110; they are post-selection and unadjusted for multiple comparisons. Labels and weights are not independent replicates.","",
        "## Interpretation","",
        "Diffusion at weight1 is the strongest tested accuracy solution. FD and analytic diffusion have almost identical Dice and surface metrics; analytic derivatives give small, consistent geometric improvements at a memory cost. BE at its Dice-selected weight0.3 suppresses curvature more than diffusion, but allows more deformation-gradient energy, Jacobian variation and folding. Its best HD95 and ASD occur at different weights3 and1, respectively; even those mean accuracy optima remain worse than the corresponding diffusion optima.","",
        "KL-C1 alone lowers canonical positive-J SD(logJ) and central folding at its selected weight0.03, but costs about five Dice percentage points relative to diffusion, with much worse surface distances and much larger diffusion/BE energies. This is consistent with a determinant penalty leaving volume-preserving shear unconstrained. Scalar metrics do not prove the optimization mechanism, but do not support replacing smoothness with standalone KL under this schedule.","",
        "Derivative choice matters most for KL geometry. At weight0.03, central/analytic foreground folding is0.430%/4.021% for KL-FD versus0.507%/0.553% for KL-analytic. Analytic KL reduces analytic folding and NDV in all10 paired cases. The FD result is sensitive to the Jacobian discretization: a low central-difference folding value is not enough to claim a valid continuous field. Analytic voxel-center samples are also not a guarantee between samples.","",
        "The SD(logJ) gain is not a superior observed trade-off at comparable accuracy. Within either backend, BE10 has slightly higher mean Dice than KL0.03, better HD95/ASD, lower canonical and analytic SD(logJ), lower both folding measures and lower NDV. Dice wins are5/10 and its paired Dice CI spans zero, so this is mean operating-point dominance, not every-case Dice superiority. HD95, SD(logJ), analytic folding and NDV improve in all10 cases. No interpolation or equal numerical-weight comparison is needed.","",
        "Cold/warm and method timing require separate interpretations. Rank0 cold/warm Dice and update counts match for every final setting, so the paired rank0 gap is meaningful here. Across methods, KL uses many more adaptive updates; its endpoint runtime does not measure determinant evaluation cost alone. These observations use different physical GPUs for some paired arms and share host resources.","",
        "The next useful hypothesis is analytic KL as an additional term with diffusion retained near its established weight1, initially testing modest KL weights around0.001-0.01. This is a proposal, not an executed combined-loss result. Keep both Jacobian conventions, NDV and label coverage. Freeze settings before evaluating a fresh cohort; first10 selections do not establish generalization.","",
        "## Dice-selected first10 settings","",
        "| Arm | Weight | Dice mean ± SD | HD95 mm | ASD mm | Canonical FG SDlogJ | Central folds % | Analytic folds % | NDV index |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+" | ".join([r["arm"],f"{r['weight']:g}",mean_sd(r,"dice")]+[f(r[k+"_mean"]) for k in
            ("hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg")]+[f(100*r["analytic_jacobian_fg_fraction_le_zero_mean"]),f(r["ndv_frac_fg_mean"])])+" |")
    lines += ["","SD(logJ) uses positive determinants only, crop2, and fixed raw image>0 foreground. Analytic J evaluates continuous B-spline derivatives at voxel centers, with the same crop/foreground. NDV is the existing multi-stencil negative-volume index, not a fraction of folded voxels; it can exceed one. Neither sampled determinant proves off-grid invertibility.","",
        "## Independent tuning and losses","",
        "| Arm | Rank0 coarse winner | First10 winner | First10 grid boundary warning |","|---|---:|---:|---|"]
    for r in rows:
        if r["arm"]=="none": continue
        lines.append(f"| {r['arm']} | {selection[r['arm']]['best']['weight']:g} | {r['weight']:g} | {r['boundary_warning']} |")
    lines += ["","Weights across diffusion, BE and KL are not equal-strength regularization. Each arm used its own coarse tuning and frozen nine-point grid. KL-analytic's population winner0.03 is penultimate in its grid; only0.1 is larger. It is the best tested setting, not a fully established population optimum.","",
        "## Paired backend comparisons at selected weights","",
        "All differences below are analytic minus FD; negative favors analytic except for Dice. CIs are exploratory95% bootstrap intervals.","",
        "| Family | Endpoint | Mean delta | 95% CI | Analytic better/tie/worse cases |","|---|---|---:|---|---|"]
    for family in ("diffusion","be","kl_c1"):
        name=f"selected_{family}_analytic_minus_fd"
        for metric in ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","analytic_jacobian_fg_fraction_le_zero","ndv_frac_fg","registration_sec","peak_allocated","updates","loop_sec_per_update"):
            r=next(x for x in pairs if x["comparison"]==name and x["metric"]==metric)
            scale=100 if metric in ("dice","analytic_jacobian_fg_fraction_le_zero") else 1/1024**3 if metric=="peak_allocated" else 1000 if metric=="loop_sec_per_update" else 1
            unit=" (pp)" if scale==100 else " (GiB)" if metric=="peak_allocated" else " (ms/update)" if metric=="loop_sec_per_update" else ""
            lines.append(f"| {family} | {metric}{unit} | {f(scale*r['mean_delta'],6)} | [{f(scale*r['ci95_low'],6)}, {f(scale*r['ci95_high'],6)}] | {r['b_better']}/{r['ties']}/{r['b_worse']} |")
    lines += ["","All shared-weight comparisons are in paired_summary.csv:9 weights for diffusion,9 for BE,7 for KL. No pooling of250 case×weight pairs as independent subjects. Common-label sensitivity is in common_label_sensitivity.csv.","",
        "The canonical eligibility rule excludes a label missing from fixed, original moving or warped segmentations. KL-FD0.03 loses label34 in rank1 (fixed22 voxels, moving6, warped0); other selected nonzero arms retain all35 labels. Using identical eligible labels within each pair changes the mean analytic-minus-FD KL Dice difference from+0.135pp to+0.335pp. This sensitivity must remain separate from the unchanged canonical metric.","",
        "| Selected backend comparison | Common-label Dice delta pp | Exploratory95% CI pp | Minimum shared labels |","|---|---:|---|---:|"]
    for family in ("diffusion","be","kl_c1"):
        q=next(q for q in common_summary if q["comparison"]==f"selected_{family}_analytic_minus_fd" and q["metric"]=="dice")
        lines.append(f"| {family} | {f(100*q['mean_delta'],4)} | [{f(100*q['ci95_low'],4)}, {f(100*q['ci95_high'],4)}] | {q['min_shared_labels']} |")
    lines += ["","## Same-weight backend differences across the curves","",
        "All deltas are analytic minus FD. This table uses only weights evaluated in both first10 grids.","",
        "| Family | Weight | Dice delta pp | HD95 delta mm | Canonical SDlogJ delta | Analytic folding delta pp | Runtime change % |","|---|---:|---:|---:|---:|---:|---:|"]
    for family in ("diffusion","be","kl_c1"):
        names=sorted(set(q["comparison"] for q in pairs if q["kind"]=="same_weight_backend" and q["a_arm"]==family+"_fd"),key=lambda name:next(q["a_weight"] for q in pairs if q["comparison"]==name))
        for name in names:
            values={q["metric"]:q for q in pairs if q["comparison"]==name}
            lines.append("| "+" | ".join([family,f"{values['dice']['a_weight']:g}",f(100*values["dice"]["mean_delta"],4),f(values["hd95"]["mean_delta"],4),
                f(values["log_jdet_std_fg"]["mean_delta"],4),f(100*values["analytic_jacobian_fg_fraction_le_zero"]["mean_delta"],4),
                f(100*(values["registration_sec"]["ratio_of_means"]-1),2)])+" |")
    lines += ["","## Paired comparisons between Dice-selected regularizers","",
        "Differences are B-A within the same derivative backend. Cases are paired before computing deltas and CIs.","",
        "| A → B | Metric | Mean delta | Exploratory95% CI | B better/tie/worse |","|---|---|---:|---|---|"]
    for name in sorted(set(q["comparison"] for q in pairs if q["kind"]=="selected_family")):
        for metric in ("dice","hd95","asd","log_jdet_std_fg","ndv_frac_fg","registration_sec"):
            q=next(q for q in pairs if q["comparison"]==name and q["metric"]==metric)
            scale=100 if metric=="dice" else 1
            lines.append(f"| {q['a_arm']} → {q['b_arm']} | {metric}{' pp' if scale==100 else ''} | {f(scale*q['mean_delta'],5)} | [{f(scale*q['ci95_low'],5)}, {f(scale*q['ci95_high'],5)}] | {q['b_better']}/{q['ties']}/{q['b_worse']} |")
    lines += ["",
        "## Common analytic energy measures at selected settings","",
        "All rows use the same foreground, voxel-coordinate analytic evaluation regardless of training loss.","",
        "| Arm | Diffusion | BE | Canonical KL-C1 | Analytic KL-C1 | Analytic SDlogJ |","|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+" | ".join([r["arm"]]+[f(r[k+"_mean"],6) for k in
            ("diffusion_native_analytic_fixed_fg_voxel","be_native_analytic_fixed_fg_voxel","fd_jacobian_fg_kl_c1","analytic_jacobian_fg_kl_c1","analytic_jacobian_fg_std_logj")])+" |")
    lines += ["","## Runtime and memory at selected settings","",
        "Headline seconds start after case loading and include optimization, stage work, checkpoint I/O, dense expansion and DVF serialization. Metrics are excluded. Cold is one fresh-worker rank0; warm mean±SD spans ten cases. Cold minus warm cohort mean is not a pure startup estimate; use paired rank0.","",
        "| Arm | Cold registration s | Warm mean ± SD s | Warm0 s | Paired cold-warm0 s | Startup s | Cold launch-to-DVF s | Warm allocated/reserved GiB | Mean sampled NVML GiB |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        q=r["resources"]
        lines.append("| "+" | ".join([r["arm"],f(q["cold_registration_sec"],2),f(q["warm_registration_sec_mean"],2)+" ± "+f(q["warm_registration_sec_std"],2),
            f(q["warm0_registration_sec"],2),f(q["paired_rank0_cold_minus_warm_sec"],2),f(q["cold_worker_startup_sec"],2),f(q["cold_launch_to_dvf_sec"],2),
            f(q["warm_peak_allocated_mean"]/1024**3,3)+" / "+f(q["warm_peak_reserved_mean"]/1024**3,3),f(q["warm_nvml_peak_mib_mean"]/1024,3)])+" |")
    lines += ["","NVML samples are200ms and attributed to exact worker PID; sampled peaks are lower bounds. PyTorch allocated and reserved are different counters. Warm reserved includes allocator cache. Host RSS is available separately. GPU model is identical, but paired backends ran on different physical GPUs and under shared CPU/storage load; no isolated per-operator speed claim follows.","",
        f"Separate saved-field verification/scoring reached sampled process GPU peaks up to{audit['evaluation_max_sampled_gpu_gib']:.3f}GiB and host RSS{audit['evaluation_max_sampled_host_rss_gib']:.3f}GiB. Provisioning the complete pipeline therefore requires more memory than the registration-only rows. Per-block evidence is in evaluation_resources.csv; no PyTorch evaluation-peak counters were recorded, so no such numbers are imputed.","",
        f"All55 final rank0 cold/warm pairs have identical Dice and update counts in this execution (max Dice difference{audit['coldwarm_cohort_max_abs_dice_difference']:g}). This is observed reproducibility, not a universal bitwise determinism guarantee. Selected paired cold/warm registration gaps are1.57-2.32s; import/device startup is another4.07-4.43s, excluded from the historical registration endpoint.","",
        "## Adaptive update counts and measured loop cost","",
        "| Arm | Mean updates | Mean selected updates | Capped stages /30 | cps8 updates /ms per update | cps4 updates /ms per update | cps2 updates /ms per update |","|---|---:|---:|---:|---|---|---|"]
    for r in rows:
        ss=sorted([s for s in stages if s["arm"]==r["arm"]],key=lambda s:s["stage"])
        lines.append("| "+" | ".join([r["arm"],f(r["updates_mean"],1),f(sum(s["mean_selected_updates"] for s in ss),1),str(r["capped_stage_count"])]+[
            f(s["mean_updates"],1)+" / "+f(1000*s["mean_sec_per_update"],2) for s in ss])+" |")
    lines += ["","Stage loop cost includes objective evaluation, backward/update, scalar transfers, logging and stopping checks. It is an observed loop-time/update proxy, not a kernel benchmark. Different stopping/update counts must be considered before comparing whole-case times.","",
        "## Full tuning cost","",
        f"Registration endpoint hours are sums of overlapping worker durations, not elapsed wall time or pure CUDA compute time. Production wall time was{audit['production_wall_seconds']/3600:.4f}h from launch to accepted completion. Registration endpoints total{audit['registration_endpoint_hours']:.4f}worker-hours, including smokes; separate verification/scoring totals{audit['total_evaluation_hours']:.4f}worker-hours, with explicit identical-field metric reuse. The KL grids differ, so aggregate KL tuning cost is not an equal-grid backend-speed comparison.","",
        "| Phase | Arm | Registrations | Registration endpoint h | Evaluation h | Capped stages | Max allocated/reserved GiB |","|---|---|---:|---:|---:|---:|---|"]
    for r in tuning:
        lines.append("| "+" | ".join([r["phase"],r["arm"],str(r["registrations"]),f(r["registration_endpoint_sec"]/3600,3),f(r["evaluation_sec"]/3600,3),
            f"{r['capped_stages']}/{r['stages']}",f(r["max_allocated"]/1024**3,3)+" / "+f(r["max_reserved"]/1024**3,3)])+" |")
    lines += ["","## Observed accuracy/regularity alternatives","",
        "Compare BE10 with KL0.03 within each backend. These are observed operating points; no interpolation or equal numerical-weight claim is used.","",
        "| Backend | Endpoint | BE10 minus KL0.03 | 95% CI | BE better/tie/worse cases |","|---|---|---:|---|---|"]
    for backend in ("fd","analytic"):
        for metric in ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","analytic_jacobian_fg_fraction_le_zero","ndv_frac_fg","registration_sec"):
            r=next(r for r in pairs if r["comparison"]==f"be10_minus_kl003_{backend}" and r["metric"]==metric)
            scale=100 if metric in ("dice","analytic_jacobian_fg_fraction_le_zero") else 1
            lines.append(f"| {backend} | {metric}{' pp' if scale==100 else ''} | {f(scale*r['mean_delta'],6)} | [{f(scale*r['ci95_low'],6)}, {f(scale*r['ci95_high'],6)}] | {r['b_better']}/{r['ties']}/{r['b_worse']} |")
    lines += ["","## Files and verification","",
        "- audit.json and provenance.json: counts, unchanged source/input checks, input snapshots and analysis receipt.",
        "- paired_summary.csv: every same-weight backend, selected backend/family, zero-control and observed-alternative comparison.",
        "- paired_cases.csv: signed differences on each actual case; common_label_sensitivity.csv uses identical eligible labels within each pair.",
        "- selected_settings.json, selected_stage_cost.csv, paired_cold_warm.csv, tuning_cost.csv: detailed accuracy/resource accounting.",
        "- observed_dominance.csv: discrete eight-metric comparisons of diffusion/BE alternatives with selected KL.",
        "- accuracy_weight, accuracy_regularity, jacobian_convention_audit and paired_backend_accuracy_resources: PNG/PDF figures.",""]
    (out/"detailed_paired_review.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--review-root",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True); args=parser.parse_args()
    analyze(args.run_root,args.review_root,args.out_dir)


if __name__=="__main__": main()
