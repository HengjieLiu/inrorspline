"""External CSV/JSON/Markdown reports and plots from immutable case records."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

from .protocol import ARMS, dump, read, stamp


def flatten(value,prefix=""):
    result={}
    for key,item in value.items():
        name=prefix+key
        if isinstance(item,dict): result.update(flatten(item,name+"_"))
        elif not isinstance(item,list): result[name]=item
    return result


def csv_write(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def stats(values):
    valid=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    n=len(valid)
    return dict(n=n,mean=statistics.mean(valid) if n==len(values) and n else None,
        std=statistics.stdev(valid) if n==len(values) and n>1 else None,
        min=min(valid) if n else None,max=max(valid) if n else None,
        median=statistics.median(valid) if n==len(values) and n else None)


def render(root,out,plots=True):
    out.mkdir(parents=True,exist_ok=True)
    timing=[]; stages=[]; accuracy=[]; labels=[]; resources=[]; cohorts=[]
    for path in sorted((root/"blocks").glob("*/*/attempt*/block.json")):
        if not (path.parent/"register_ACCEPTED.json").exists(): continue
        block=read(path); job=block["job"]
        common=dict(phase=job["phase"],arm=job["arm"],weight=job["weight"],gpu=job["gpu"],gpu_uuid=job["gpu_uuid"],block=str(path.parent))
        for case in block["cases"]:
            row=dict(common,case_rank=case["case_rank"],condition=case["condition"],worker_pid=case["worker_pid"],
                **{key:case.get(key) for key in ("registration_sec","worker_startup_sec","launch_to_dvf_sec","case_load_sec",
                    "case_to_dvf_sec","finalize_dvf_sec","peak_allocated","peak_reserved","nvml_peak_mib","host_rss_peak","telemetry_samples")},
                baseline_reserved=case["baseline_memory"]["reserved"],baseline_allocated=case["baseline_memory"]["allocated"])
            timing.append(row)
            for stage in case["stages"]:
                stages.append(dict(common,case_rank=case["case_rank"],condition=case["condition"],**flatten(stage)))
            metric_path=Path(case["dvf_path"]).parent/"evaluation.json"
            if metric_path.exists() and (path.parent/"eval_ACCEPTED.json").exists():
                metrics=read(metric_path)
                metric=dict(common,case_rank=case["case_rank"],condition=case["condition"],
                    fixed_id=case["fixed_id"],moving_id=case["moving_id"],dvf_sha256=case["dvf_sha256"],
                    dice=metrics["dice"],hd95=metrics["hd95"],asd=metrics["asd"],ncc9=metrics["ncc9"],
                    **flatten(metrics["regularity"]),**flatten(metrics["fd_jacobian"],"fd_jacobian_"),
                    **flatten(metrics["analytic_jacobian"],"analytic_jacobian_"),
                    metric_seconds=metrics["metric_seconds"],metric_alias_of=metrics.get("alias_of"))
                accuracy.append(metric)
                for label in metrics["per_label"]:
                    labels.append(dict(common,case_rank=case["case_rank"],condition=case["condition"],**flatten(label),
                        exclusion_reason=";".join(label["exclusion_reason"])))
        warm=[r for r in timing if r["block"]==str(path.parent) and r["condition"]=="warm"]
        cold=next(r for r in timing if r["block"]==str(path.parent) and r["condition"]=="cold")
        resource=dict(common,cold_registration_sec=cold["registration_sec"],cold_launch_to_dvf_sec=cold["launch_to_dvf_sec"],
            cold_worker_startup_sec=cold["worker_startup_sec"],warm_n=len(warm))
        for key in ("registration_sec","case_load_sec","case_to_dvf_sec","finalize_dvf_sec","peak_allocated","peak_reserved", "nvml_peak_mib","host_rss_peak","baseline_reserved"):
            resource["cold_"+key]=cold[key]
            if warm: resource.update({"warm_"+key+"_"+k:v for k,v in stats([r[key] for r in warm]).items()})
        if warm:
            w0=next(r for r in warm if r["case_rank"]==0)
            resource.update(warm0_registration_sec=w0["registration_sec"],paired_rank0_cold_minus_warm_sec=cold["registration_sec"]-w0["registration_sec"],
                paired_rank0_cold_warm_ratio=cold["registration_sec"]/w0["registration_sec"])
        resources.append(resource)
        group=[r for r in accuracy if r["block"]==str(path.parent) and r["condition"]=="warm"]
        if job["phase"]=="cohort" and sorted(r["case_rank"] for r in group)==list(range(10)):
            for subset,ranks in (("first10",range(10)),("rank0",[0]),("ranks1_9",range(1,10))):
                selected=[r for r in group if r["case_rank"] in ranks]
                summary=dict(common,subset=subset,n=len(selected))
                keys=[k for k,v in selected[0].items() if isinstance(v,(float,int)) and k not in ("weight","case_rank")]
                # Include metrics whose first case is undefined as well.
                keys+= [k for k,v in selected[0].items() if v is None and k not in ("metric_alias_of",)]
                for key in keys: summary.update({key+"_"+k:v for k,v in stats([r.get(key) for r in selected]).items()})
                cohorts.append(summary)
    for name,rows in (("timing_cases",timing),("stage_metrics",stages),("accuracy_regularity_cases",accuracy),
                      ("per_label",labels),("setting_resources",resources),("cohort_summary",cohorts)):
        csv_write(out/(name+".csv"),rows); dump(out/(name+".json"),rows)
    selection=read(root/"selection.json") if (root/"selection.json").exists() else {}
    status=read(root/"status.json")
    lines=["# OASIS six-arm regularizer sweep", "",f"Updated: {stamp()}","",f"Controller phase: {status['phase']}","",
        "Exploratory seed3 first10; NCC9, prolonged cps8/4/2, fixed stage rates and adaptive stopping.","",
        "## Progress and selection","",f"Accepted timing rows: {len(timing)}; evaluated case rows: {len(accuracy)}.","",
        "The planned final matrix is55 settings /605 timing rows /550 warm accuracy rows; pilot and smoke rows are additional.","",
        "| Arm | Pilot decision | Coarse best weight | Frozen first10 grid |","|---|---|---:|---|"]
    for arm in ARMS:
        decision=selection.get(arm,{})
        lines.append(f"| {arm} | {decision.get('status','pending')} | {decision.get('best',{}).get('weight','')} | {decision.get('first10_grid','')} |")
    lines += ["","## Cold/warm time and memory","",
        "Times are case_load_complete to dvf_file_ready and exclude evaluation. Cold is the first registration of a fresh process, not cold filesystem caches. One cold sample has no SD. Warm means span ten different cases in cohort blocks; the paired rank0 gap isolates case identity.","",
        "All byte-valued memory columns are bytes; NVML columns are MiB. Reserved memory includes allocator cache. NVML/RSS are sampled peaks; PyTorch peaks are allocator counters. All raw measurements, startup/load/stage/finalization times, standard deviations and warm memory baselines are in setting_resources.csv and timing_cases.csv.","",
        "| Phase | Arm | Weight | GPU | Cold s | Warm s mean ± SD | Paired gap s | Cold alloc/res GiB | Warm max alloc/res GiB | Cold/warm max NVML MiB |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value): return "NA" if value is None else f"{value:.4g}"
    def gib(value): return None if value is None else value/1024**3
    for row in resources:
        lines.append("| "+" | ".join([row["phase"],row["arm"],f"{row['weight']:g}",str(row["gpu"]),fmt(row["cold_registration_sec"]),
            fmt(row.get("warm_registration_sec_mean"))+" ± "+fmt(row.get("warm_registration_sec_std")),fmt(row.get("paired_rank0_cold_minus_warm_sec")),
            fmt(gib(row["cold_peak_allocated"]))+" / "+fmt(gib(row["cold_peak_reserved"])),
            fmt(gib(row.get("warm_peak_allocated_max")))+" / "+fmt(gib(row.get("warm_peak_reserved_max"))),
            fmt(row["cold_nvml_peak_mib"])+" / "+fmt(row.get("warm_nvml_peak_mib_max"))])+" |")
    lines += ["","## Complete first10 accuracy and regularity","",
        "Canonical SD(logJ) uses positive determinants only, crop2, fixed raw image>0 foreground, population within-volume SD. Across-case SD uses ddof1. Inspect folding, coverage, NDV, and both FD/analytic energies alongside it; no undefined case is silently removed.","",
        "| Arm | Weight | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folds % |","|---|---:|---:|---:|---:|---:|---:|"]
    primary=[r for r in cohorts if r["subset"]=="first10"]
    for row in primary:
        lines.append("| "+" | ".join([row["arm"],f"{row['weight']:g}"]+[fmt(row.get(k+"_mean")) for k in
                     ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg")])+" |")
    winners={}
    for arm in ARMS:
        candidates=[r for r in primary if r["arm"]==arm and r.get("dice_mean") is not None]
        if candidates:
            best=min(candidates,key=lambda r:(-r["dice_mean"],r["asd_mean"],r["hd95_mean"],r["weight"]))
            weights=sorted(r["weight"] for r in candidates)
            winners[arm]=dict(weight=best["weight"],dice=best["dice_mean"],boundary_warning=weights.index(best["weight"])<2 or weights.index(best["weight"])>=len(weights)-2)
    dump(out/"dice_selected_operating_points.json",winners)
    lines += ["","## Files","","- setting_resources.csv: every setting's cold/warm runtime and GPU/host memory.",
        "- timing_cases.csv and stage_metrics.csv: case/stage time, memory, objective and stopping records.",
        "- accuracy_regularity_cases.csv, per_label.csv, cohort_summary.csv: complete scalar and label metrics.",
        "- JSON companions retain the same records; selection.json in the run root records bracketing evidence.",
        "- Failures/monitor traces/source receipts remain in the run root. Historical results are not substituted for fresh resource observations.",""]
    (out/"report.md").write_text("\n".join(lines))
    dump(out/"report_status.json",dict(at=stamp(),timing_rows=len(timing),evaluated_rows=len(accuracy),complete_first10_groups=len(primary),winners=winners))
    if plots and accuracy:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for phase in ("pilot","cohort"):
            data=[r for r in accuracy if r["condition"]=="warm" and (r["phase"].startswith("pilot") if phase=="pilot" else r["phase"]=="cohort")]
            if not data: continue
            fig,axes=plt.subplots(2,3,figsize=(15,8))
            for arm in ARMS:
                grouped=[]
                for weight in sorted(set(r["weight"] for r in data if r["arm"]==arm)):
                    group=[r for r in data if r["arm"]==arm and r["weight"]==weight]
                    if phase=="cohort" and len(group)!=10: continue
                    grouped.append(dict(weight=weight,**{key:stats([r.get(key) for r in group])["mean"] for key in
                        ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg")}))
                for col,key in enumerate(("dice","hd95","asd")):
                    axes[0,col].plot([r["weight"] for r in grouped],[r[key] for r in grouped],"o-",label=arm)
                    axes[0,col].set_xscale("log"); axes[0,col].set(xlabel="Regularizer weight",ylabel=key)
                    axes[1,col].plot([r["log_jdet_std_fg"] for r in grouped],[r[key] for r in grouped],"o-",label=arm)
                    axes[1,col].set(xlabel="Foreground SD(logJ), positive J",ylabel=key)
            axes[0,0].legend(fontsize=8); fig.suptitle(phase+" exploratory regularizer sweep")
            fig.tight_layout(); fig.savefig(out/(phase+"_accuracy_sdlogj.png"),dpi=160); plt.close(fig)
    return dict(timing_rows=len(timing),accuracy_rows=len(accuracy),groups=len(primary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); print(render(args.root,args.out),flush=True)


if __name__=="__main__": main()
