#!/usr/bin/env python
"""P80 complete-group tables, matched depth differences and shared-axis dual ARC."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"experiments/oasis_alpha_sweep"))
import bspline8421_matched_prefix as e
import oasis_first100_five_family_dual_arc as arc


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def grouped_summary(rows):
    grouped=defaultdict(list)
    for row in rows: grouped[row["depth"],row["family"],row["alpha"]].append(row)
    result=[]; coverage=[]
    for key,group in sorted(grouped.items(),key=lambda kv:str(kv[0])):
        ranks=sorted(r["case_rank"] for r in group)
        if len(set(ranks))!=len(ranks): raise RuntimeError("Duplicate paired case")
        coverage.append(dict(zip(("depth","family","alpha"),key),n=len(ranks),missing_ranks=sorted(set(range(100))-set(ranks))))
        if ranks!=list(range(100)): continue
        r=dict(zip(("depth","family","alpha"),key),n=100)
        for metric in e.METRICS:
            values=[x[metric] for x in group]
            if not all(math.isfinite(v) for v in values): raise RuntimeError("Nonfinite aggregate")
            r[metric+"_mean"]=statistics.mean(values); r[metric+"_sd"]=statistics.stdev(values)
        for metric in ("optimizer_sec","new_optimizer_sec","retained_prefix_optimizer_sec","combined_optimizer_sec","worker_sec","audit_sec",
                       "actual_new_adam_updates","peak_allocated_gib","peak_reserved_gib","audit_peak_allocated_gib","audit_peak_reserved_gib"):
            values=[x[metric] for x in group if x.get(metric) is not None]
            if len(values)==100:
                for suffix,value in (("mean",statistics.mean(values)),("sd",statistics.stdev(values)),("min",min(values)),("max",max(values))):
                    r[metric+"_"+suffix]=value
        result.append(r)
    return result,coverage


def depth_pairs(new,old):
    index={(r["family"],r["alpha"],r["case_rank"]):r for r in old}
    result=[]
    for r in new:
        p=index[r["family"],r["alpha"],r["case_rank"]]
        if any(r[k]!=p[k] for k in ("fixed_id","moving_id","direction")): raise RuntimeError("Unpaired depth rows")
        result.append({k:r[k] for k in ("family","alpha","case_rank","fixed_id","moving_id")}|{k:r[k]-p[k] for k in e.METRICS})
    return result


def paired_summary(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[r["family"],r["alpha"]].append(r)
    result=[]
    for (family,alpha),group in sorted(grouped.items()):
        if sorted(r["case_rank"] for r in group)!=list(range(100)): continue
        r={"family":family,"alpha":alpha,"n":100}
        for metric in e.METRICS:
            values=[x[metric] for x in group]; sign=1 if metric=="dice" else -1
            r.update({metric+"_delta_mean":statistics.mean(values),metric+"_delta_sd":statistics.stdev(values),
                      metric+"_improved":sum(sign*x>1e-12 for x in values),metric+"_worsened":sum(sign*x < -1e-12 for x in values),
                      metric+"_tied":sum(abs(x)<=1e-12 for x in values)})
        result.append(r)
    return result


def figures(out,old_rows,new_rows,dl):
    pairs=[(r["fixed_id"],r["moving_id"]) for r in sorted([x for x in dl if x["family"]=="VFA" and x["alpha"]==1],key=lambda x:x["case_rank"])]
    means={"842":arc.summarize(old_rows+dl,pairs),"8421":arc.summarize(new_rows+dl,pairs)}
    all_means=means["842"]+means["8421"]
    floor=arc.folding_floor(all_means)
    original_floor=arc.folding_floor; original_style=arc.STYLES["displacement"]
    arc.folding_floor=lambda _:floor
    diagnostics={}
    try:
        with arc.plt.rc_context({"font.family":"DejaVu Sans","font.size":15,"axes.labelsize":16,"xtick.labelsize":14,
                               "ytick.labelsize":14,"pdf.fonttype":42,"svg.fonttype":"none"}):
            for depth,summary in means.items():
                arc.STYLES["displacement"]=(f"Displacement B-spline {'8→4→2→cps1' if depth=='8421' else '8→4→2'}",*original_style[1:])
                for name,folding in (("fg_stdlogj",False),("fg_folding_log10",True)):
                    fig,_=arc.figure(summary,folding)
                    xkey="folding_ratio_fg" if folding else "log_jdet_std_fg"
                    xmax=max(r[xkey+"_mean"]+r[xkey+"_sd"] for r in all_means)
                    xlim=(floor/2,10.**math.ceil(math.log10(xmax))) if folding else (0,math.ceil((xmax+.04)*10)/10)
                    axes=[]
                    for i,ax in enumerate(fig.axes):
                        metric=list(arc.Y_LABELS)[i%3]; ax.set_xlim(*xlim)
                        if i>=3:
                            lo=min(r[metric+"_mean"]-r[metric+"_sd"] for r in all_means)
                            hi=max(r[metric+"_mean"]+r[metric+"_sd"] for r in all_means); pad=.065*(hi-lo)
                            ax.set_ylim(max(0,lo-pad),min(1,hi+pad) if metric=="dice" else hi+pad)
                        axes.append({"metric":metric,"row":i//3,"xlim":list(ax.get_xlim()),"ylim":list(ax.get_ylim())})
                    fig._suptitle.set_text(fig._suptitle.get_text()+f" · cps{depth}")
                    stem=("arc_" if depth=="8421" else "companion_842_arc_")+name+"_2x3"
                    for ext in ("png","pdf","svg"): fig.savefig(out/f"{stem}.{ext}",dpi=180,facecolor="white")
                    arc.plt.close(fig); diagnostics[depth+"_"+name]=axes
    finally:
        arc.folding_floor=original_floor; arc.STYLES["displacement"]=original_style
    for metric in ("fg_stdlogj","fg_folding_log10"):
        if diagnostics["842_"+metric]!=diagnostics["8421_"+metric]: raise RuntimeError("Depth figure axes differ")
    return {"axes":diagnostics,"folding_display_floor":floor,"means":means}


def table(rows):
    lines=["| Method | Depth | λ | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values=[f"{r[k+'_mean']:.3f} ± {r[k+'_sd']:.3f}" if k!="jdet_le0_pct_fg" else f"{r[k+'_mean']:.3e} ± {r[k+'_sd']:.3e}" for k in e.METRICS]
        lines.append(f"| {r['family']} | {r['depth']} | {r['alpha']:.6g} | "+" | ".join(values)+" |")
    return lines


def reproducibility(root,new,provenance):
    """Historical four-level finals are appendix-only, never fill primary cells."""
    configs=[(e.old.P43,"displacement"),(e.old.P74+"/continuation_v2","svf_velocity"),
             ("oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3","bounded2")]
    selected={(r["family"],r["alpha"],r["case_rank"]):r for r in new}; results=[]
    for key,family in configs:
        folder=root/"repo_experiments"/key
        accepted=e.c.read_json(folder/"ACCEPTED.json")
        if not any(accepted.get(k) for k in ("complete","passed","technical_completion")):
            raise RuntimeError("Unaccepted historical reproduction reference")
        provenance[str(folder/"ACCEPTED.json")]=e.sha(folder/"ACCEPTED.json")
        provenance[str(folder/"results.json")]=e.sha(folder/"results.json")
        for row in e.c.read_json(folder/"results.json")["candidates"]:
            if family=="displacement":
                s=row["spec"]
                if (s["representation"],s["resolution"])!=("8421","pyramid"): continue
                rank,alpha=s["case_rank"],s["alpha"]
            elif family=="svf_velocity":
                s=row["spec"]
                if (s["schedule"],s["regularize"])!=("8421","velocity"): continue
                rank,alpha=s["case_rank"],s["alpha"]
            else:
                if row["depth"]!=4 or row["config_key"]!="bounded_2_compose": continue
                rank,alpha=row["case_rank"],1.
            fresh=selected.get((family,alpha,rank))
            if fresh is None: continue
            results.append({"family":family,"alpha":alpha,"case_rank":rank,
                "historical_source":str(folder),"historical_run":row["run_dir"],"fresh_run":fresh["run_dir"],
                "interpretation":"matched-prefix extension versus historical four-level fit; not guaranteed identical initialization",
                **{metric+"_fresh_minus_historical":fresh[metric]-row[metric] for metric in e.METRICS}})
    return results


def main(args):
    root=args.run_root or args.artifact_root/"runs"/e.VARIANT/"execution_v1"
    m=e.c.read_json(root/"launch_manifest.json"); e.assert_manifest(m,True)
    state=e.c.read_json(root/"production_queue/state.json")
    out=args.artifact_root/"repo_experiments"/e.VARIANT/f"release_{time.time_ns()}"
    out.mkdir(parents=True,exist_ok=False)
    old,_,provenance=arc.load_sources(Path(m["prefix_report"]),Path(m["dl_report"]))
    dl=[r for r in old if r["family"] in ("VFA","SITReg")]
    old=[r for r in old if r["family"] in e.FAMILIES]
    new=[]; stages=[]; labels=[]; sensitivity=[]; failures=[]
    for j in state["jobs"]:
        if j["status"]!="complete": failures.append(j); continue
        spec=e.Spec(**j["spec"]); r=e.validate(j["out"],m,spec)
        a=e.c.read_json(Path(j["out"])/"ACCEPTED.json")
        new.append({"family":spec.family,"alpha":spec.alpha,"depth":"8421",
                    **{k:r[k] for k in (*e.METRICS,"case_rank","fixed_id","moving_id","direction","gpu","run_dir",
                     "new_optimizer_sec","retained_prefix_optimizer_sec","combined_optimizer_sec","retained_prefix_worker_sec",
                     "worker_sec","actual_new_adam_updates","retained_prefix_adam_updates","peak_allocated_gib","peak_reserved_gib",
                     "prefix_summary_sha256","prefix_dvf_sha256","dvf_sha256")},
                    **{k:a[k] for k in ("audit_sec","audit_peak_allocated_gib","audit_peak_reserved_gib")}})
        for b in r["new_blocks"]:
            stages.append({"family":spec.family,"alpha":spec.alpha,"case_rank":spec.case_rank,
                           "cps":1,"map":b["map_index"],"gpu":r["gpu"],**b["optimizer"]})
        for d in r.get("integration_diagnostics",[]):
            sensitivity.append({"family":spec.family,"alpha":spec.alpha,"case_rank":spec.case_rank,**d})
        with (Path(j["out"])/"final_metrics/case_metrics.csv").open() as f: wide=next(csv.DictReader(f))
        for label in e.c.OASIS_LABELS:
            labels.append({"depth":"8421","family":spec.family,"alpha":spec.alpha,"case_rank":spec.case_rank,"label":label,
                           **{k:float(v) for k,v in wide.items() if k.endswith(f"_label_{label:02d}")}})
    old=[dict(r,depth="842") for r in old]; dl=[dict(r,depth="DL") for r in dl]
    for r in old:
        source=m["prefixes"][e.Spec(r["case_rank"],r["family"],r["alpha"]).key()]["accepted_row"]
        for key in ("optimizer_sec","worker_sec","audit_sec","actual_adam_updates","gpu","run_dir",
                    "peak_allocated_gib","peak_reserved_gib","audit_peak_allocated_gib","audit_peak_reserved_gib"):
            r[key]=source.get(key)
    all_rows=old+new+dl
    summaries,coverage=grouped_summary(all_rows)
    present={(r["depth"],r["family"],r["alpha"]) for r in coverage}
    for family in e.FAMILIES:
        for alpha in e.LAMBDAS:
            if ("8421",family,alpha) not in present:
                coverage.append({"depth":"8421","family":family,"alpha":alpha,"n":0,"missing_ranks":list(range(100))})
    paired=depth_pairs(new,old); paired_stats=paired_summary(paired)
    repro=reproducibility(args.artifact_root,new,provenance)
    bysummary={(r["depth"],r["family"],r["alpha"]):r for r in summaries}
    for r in paired_stats:
        before=bysummary["842",r["family"],r["alpha"]]; after=bysummary["8421",r["family"],r["alpha"]]
        for metric in e.METRICS:r[metric+"_across_case_sd_change"]=after[metric+"_sd"]-before[metric+"_sd"]
    complete=len(new)==2100 and len(state["jobs"])==2100 and not failures and len(summaries)==52
    plot=figures(out,old,new,dl) if complete else {"omitted":"Incomplete100-case matrix; no partial curves masquerading as full100"}
    for name,rows in (("per_case",all_rows),("summary",summaries),("coverage",coverage),("paired_depth_differences",paired),
                      ("paired_depth_summary",paired_stats),("new_stage_diagnostics",stages),("per_label_8421",labels),
                      ("integration_sensitivity",sensitivity),("historical_8421_reproducibility",repro)):
        csv_write(out/f"{name}.csv",rows)
    # Keep full prefix label rows alongside, without reinterpreting their canonical absent labels.
    shutil_source=Path(m["prefix_report"])/"per_label.csv"
    import shutil
    shutil.copyfile(shutil_source,out/"per_label_842.csv")
    e.atomic_json(out/"failures.json",{"failures":failures,"all_attempt_failures":[e.c.read_json(p) for p in (root/"production_queue/failures").glob("*.json")]})
    e.atomic_json(out/"results.json",{"complete":complete,"candidate_count":len(new),"per_case":all_rows,"summary":summaries,
                  "coverage":coverage,"paired":paired,"paired_summary":paired_stats,"plot":plot,"failures":failures})
    lines=["# Matched-prefix cps842 → cps8421: OASIS first100","",
           "Status: **ACCEPTED**" if complete else "Status: **INCOMPLETE — deferred failures require review**","",
           f"Fresh accepted four-level outputs: {len(new)}/2100. All tables use complete100-case groups and sample SD.","",
           "The exact accepted prefix is retained per family/λ/case. First10 cases informed design; this is exploratory, not untouched confirmation.","",
           "## All methods and λ settings","",*table(summaries),"",
           "## Within-method depth comparisons","",
           "Differences are cps8421 minus the identical cps842 prefix; positive Dice and negative other metrics favor cps8421.",""]
    for family in e.FAMILIES:
        lines += [f"### {family}","","| λ | ΔDice | ΔHD95 | ΔASD | ΔFG SD(log J) | Δfolding % |","|---:|---:|---:|---:|---:|---:|"]
        for r in paired_stats:
            if r["family"]==family:
                lines.append(f"| {r['alpha']:g} | "+" | ".join(f"{r[k+'_delta_mean']:+.4g} ± {r[k+'_delta_sd']:.4g}" for k in e.METRICS)+" |")
        lines += ["","Per-case improvement/worsening counts and paired sample SD are in paired_depth_summary.csv. Across-case variability changes are available from the two depth summaries.",""]
    lines += ["## Resources and interpretation","",
              "new_optimizer_sec measures cps1 only. combined_optimizer_sec adds saved prefix optimizer time; it is not a fresh end-to-end four-level worker measurement. Worker/audit wall time and allocated/reserved memory are separate, under concurrent workloads.","",
              "The bounded method adds TWO full-schedule cps1 maps; the other methods add one. Equal λ does not impose equal-strength regularity on velocity and displacement.","",
              "ARC top rows use zoomed means; bottom contours show marginal across-case ±1 sample SD, not confidence regions. Folding is percent/100 on a log10 axis; true zeros remain zero in data and use hollow display-floor markers.","",
              "Failures and attempts are preserved separately. Incomplete groups are not aggregated or plotted as100-case results.",""]
    for metric in e.METRICS:
        worst=sorted(paired,key=lambda r:r[metric] if metric=="dice" else -r[metric])[:10]
        csv_write(out/f"largest_regressions_{metric}.csv",worst)
    (out/"report.md").write_text("\n".join(lines)+"\n")
    statistic_checks=0
    with (out/"summary.csv").open() as stream:
        serialized=list(csv.DictReader(stream))
    if len(serialized)!=len(summaries):raise RuntimeError("Serialized summary count mismatch")
    for a,z in zip(serialized,summaries):
        for key,value in z.items():
            if key.endswith(("_mean","_sd")):
                if not math.isclose(float(a[key]),value,rel_tol=0,abs_tol=1e-12):raise RuntimeError("Serialized statistics mismatch")
                statistic_checks+=1
    if any(e.sha(path)!=digest for path,digest in provenance.items()):raise RuntimeError("Source changed during reporting")
    e.assert_manifest(m,True)
    marker={"complete":complete,"candidate_count":len(new),"settings":len(summaries),"failures":len(failures),
            "source_sha256":provenance,"source_hashes":m["source_hashes"],"statistic_checks":statistic_checks,"created_at":e.c.now_text(),
            "output_sha256":{p.name:e.sha(p) for p in out.iterdir() if p.is_file()}}
    e.atomic_json(out/("ACCEPTED.json" if complete else "INCOMPLETE.json"),marker)
    e.atomic_json(root/"LATEST_REPORT.json",{"path":str(out),"complete":complete})
    print(str(out),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
    p.add_argument("--run-root",type=Path)
    main(p.parse_args())
