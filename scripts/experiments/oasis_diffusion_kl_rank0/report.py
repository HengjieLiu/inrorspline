"""Rank0-only combined-loss gamma curves, all metrics and explicit baseline reuse."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts.experiments.oasis_regularizer_sweep.protocol import read,dump,sha256,stamp
from scripts.experiments.oasis_regularizer_sweep.report import flatten,csv_write
from .protocol import BACKENDS,GAMMAS

REPO=Path(__file__).resolve().parents[3]


def render(root,out):
    out.mkdir(parents=True,exist_ok=True)
    records=[]; eval_resources=[]
    for backend in BACKENDS:
        path=root/"baselines"/f"{backend}.json"
        if path.exists():
            ref=read(path)
            for item in ref["cases"]:
                records.append(dict(backend=backend,gamma=0.,phase="reference",source_kind="historical_gamma0",
                                    case=item["case"],metrics=item["metrics"],source=ref["source_block"]))
    for path in sorted(root.glob("blocks/*/*/attempt*/block.json")):
        if not (path.parent/"register_ACCEPTED.json").exists(): continue
        block=read(path); job=block["job"]; backend=job["arm"].removeprefix("diffusion_")
        for case in block["cases"]:
            if case["case_rank"]!=0 or (case["fixed_id"],case["moving_id"])!=("0333","0115"): raise RuntimeError("Rank0-only scope failed")
            metric_path=Path(case["dvf_path"]).parent/"evaluation.json"
            metrics=read(metric_path) if metric_path.exists() and (path.parent/"eval_ACCEPTED.json").exists() else None
            records.append(dict(backend=backend,gamma=job["gamma"],phase=job["phase"],source_kind="fresh_combined",
                                case=case,metrics=metrics,source=str(path.parent)))
        if (path.parent/"eval_telemetry.json").exists():
            eval_resources.append(dict(backend=backend,gamma=job["gamma"],phase=job["phase"],**read(path.parent/"eval_telemetry.json")))
    all_rows=[]; label_rows=[]; stage_rows=[]; primary=[]
    for rec in records:
        case,metric=rec["case"],rec["metrics"]
        meta={k:rec[k] for k in ("backend","gamma","phase","source_kind","source")}
        meta.update(case_rank=0,condition=case["condition"],fixed_id=case["fixed_id"],moving_id=case["moving_id"],gpu=case["gpu"],
            lambda_diffusion=1.,kl_backend="analytic",registration_sec=case["registration_sec"],
            worker_startup_sec=case["worker_startup_sec"],launch_to_dvf_sec=case.get("launch_to_dvf_sec"),case_load_sec=case["case_load_sec"],
            peak_allocated=case["peak_allocated"],peak_reserved=case["peak_reserved"],nvml_peak_mib=case.get("nvml_peak_mib"),
            host_rss_peak=case.get("host_rss_peak"),updates=sum(s["actual_epochs"] for s in case["stages"]),
            selected_updates=sum(s["best_epoch"] for s in case["stages"]),
            capped_stages=sum(s["stopping_reason"]=="max_epochs" for s in case["stages"]))
        for stage in case["stages"]: stage_rows.append(dict(meta,**flatten(stage,"stage_")))
        if metric is None: continue
        if metric["input_hashes"]!=case["input_hashes"] or metric["dvf_sha256"]!=case["dvf_sha256"]: raise RuntimeError("Metric identity mismatch")
        row=dict(meta,dice=metric["dice"],hd95=metric["hd95"],asd=metric["asd"],ncc9=metric["ncc9"],
                 eligible_labels=sum(p["eligible"] for p in metric["per_label"]),
                 **flatten(metric["regularity"]),**flatten(metric["fd_jacobian"],"fd_jacobian_"),
                 **flatten(metric["analytic_jacobian"],"analytic_jacobian_"))
        all_rows.append(row)
        for label in metric["per_label"]: label_rows.append(dict(meta,**flatten(label),exclusion_reason=";".join(label["exclusion_reason"])))
        if case["condition"]=="warm" and rec["phase"]!="smoke": primary.append(row)
    primary.sort(key=lambda r:(r["backend"],r["gamma"]))
    for backend in BACKENDS:
        selected=[r for r in primary if r["backend"]==backend]
        if selected:
            baseline=next(r for r in selected if r["gamma"]==0)
            ref=next(r for r in records if r["backend"]==backend and r["gamma"]==0 and r["case"]["condition"]=="warm")
            for rec in records:
                if rec["backend"]==backend and rec["case"]["input_hashes"]!=ref["case"]["input_hashes"]: raise RuntimeError("New/baseline inputs differ")
            for row in selected:
                for key in ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","ndv_frac_fg",
                            "analytic_jacobian_fg_std_logj","analytic_jacobian_fg_fraction_le_zero","registration_sec"):
                    row["delta_vs_gamma0_"+key]=row[key]-baseline[key]
    backend_deltas=[]
    for gamma in GAMMAS:
        rows=[r for r in primary if r["gamma"]==gamma]
        if len(rows)==2:
            fd=next(r for r in rows if r["backend"]=="fd"); an=next(r for r in rows if r["backend"]=="analytic")
            backend_deltas.append(dict(gamma=gamma,**{key:an[key]-fd[key] for key in
                ("dice","hd95","asd","log_jdet_std_fg","jdet_le0_pct_fg","analytic_jacobian_fg_std_logj",
                 "analytic_jacobian_fg_fraction_le_zero","ndv_frac_fg","registration_sec","peak_allocated")}))
    for name,rows in (("all_metrics",all_rows),("primary_rank0",primary),("per_label",label_rows),
                      ("stage_components",stage_rows),("backend_deltas",backend_deltas),("evaluation_resources",eval_resources)):
        dump(out/(name+".json"),rows); csv_write(out/(name+".csv"),rows)
    dump(out/"evaluation_payloads.json",records)
    def f(v,n=4): return "NA" if v is None else f"{v:.{n}f}"
    lines=["# Rank0: diffusion(lambda1) + analytic KL-C1 gamma sweep","",f"Updated: {stamp()}","",
        "Fixed0333 / moving0115 only. No first10 extension. Both FD and analytic diffusion are evaluated; the KL derivative is analytic in both arms. Epsilon0.05/kappa120, NCC9 and the P84 backbone/optimizer/stopping remain fixed.","",
        "Gamma0 is a hash/protocol/input-audited historical reference from P84. Its cold/warm time and memory are historical. Positive-gamma fields start independently from identity and have fresh cold/warm measurements. Warm rank0 is the primary scientific endpoint; cold is QA, not a second subject.","",
        f"Available primary settings: {len(primary)}/12; scored observations including smokes/references: {len(all_rows)}. Full target:20 new production +4 reference +4 smoke observations.","",
        "## Accuracy and baseline changes","",
        "| Diffusion | Gamma | Dice | Delta Dice pp | HD95 mm | Delta HD95 | ASD mm | Delta ASD | Labels |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append("| "+" | ".join([r["backend"],f"{r['gamma']:g}",f(r["dice"],6),f(100*r["delta_vs_gamma0_dice"],3),
            f(r["hd95"]),f(r["delta_vs_gamma0_hd95"]),f(r["asd"]),f(r["delta_vs_gamma0_asd"]),str(r["eligible_labels"])])+" |")
    lines += ["","## Foreground geometry","",
        "Both SD(logJ) measures use positive determinants only. Central/analytic Jacobians use crop2. NDV is the discrete negative-volume index with crop1, not a folded-voxel percentage. All complete scalar definitions, full-volume versions, J quantiles/coverage, and FD/analytic energies are in all_metrics.csv and evaluation_payloads.json.","",
        "| Diffusion | Gamma | Central SDlogJ | Analytic SDlogJ | Central folds % | Analytic folds % | NDV | Analytic diffusion | Analytic BE |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append("| "+" | ".join([r["backend"],f"{r['gamma']:g}",f(r["log_jdet_std_fg"]),f(r["analytic_jacobian_fg_std_logj"]),
            f(r["jdet_le0_pct_fg"]),f(100*r["analytic_jacobian_fg_fraction_le_zero"]),f(r["ndv_frac_fg"],6),
            f(r["diffusion_native_analytic_fixed_fg_voxel"],6),f(r["be_native_analytic_fixed_fg_voxel"],6)])+" |")
    lines += ["","## Cold/warm performance","",
        "No sample SD or population CI is inferred from a single case. Registration spans case_load_complete through dvf_file_ready, excluding scoring. NVML/RSS are200ms sampled process peaks; Torch allocated/reserved are separate counters. Baseline timings were collected earlier, and some settings use different physical GPUs, so timing differences are observational.","",
        "| Diffusion | Gamma | Provenance | GPU | Cold s | Warm s | Paired cold-warm s | Warm updates | Capped stages | Warm allocated/reserved GiB | Sampled process GPU GiB |","|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        cold=next(q for q in all_rows if q["backend"]==r["backend"] and q["gamma"]==r["gamma"] and q["condition"]=="cold" and q["phase"]!="smoke")
        lines.append("| "+" | ".join([r["backend"],f"{r['gamma']:g}",r["source_kind"],str(r["gpu"]),f(cold["registration_sec"],2),f(r["registration_sec"],2),
            f(cold["registration_sec"]-r["registration_sec"],2),str(r["updates"]),str(r["capped_stages"]),
            f(r["peak_allocated"]/1024**3,3)+" / "+f(r["peak_reserved"]/1024**3,3),f(r["nvml_peak_mib"]/1024,3)])+" |")
    choices={}
    lines += ["","## Best tested settings and interpretation","","Best Dice is a single-case tuning choice; do not choose checkpoints or repeats using labels.",""]
    for backend in BACKENDS:
        rows=[r for r in primary if r["backend"]==backend]
        if not rows: continue
        best=min(rows,key=lambda r:(-r["dice"],r["asd"],r["hd95"],r["gamma"]))
        choices[backend]=dict(gamma=best["gamma"],dice=best["dice"],hd95=best["hd95"],asd=best["asd"],
            delta_dice=best["delta_vs_gamma0_dice"],central_sdlogj=best["log_jdet_std_fg"],analytic_sdlogj=best["analytic_jacobian_fg_std_logj"],
            central_folds_pct=best["jdet_le0_pct_fg"],analytic_folds_pct=100*best["analytic_jacobian_fg_fraction_le_zero"],ndv=best["ndv_frac_fg"])
        lines.append(f"- {backend} diffusion: best tested gamma={best['gamma']:g}, Dice={best['dice']:.6f}, change={100*best['delta_vs_gamma0_dice']:+.3f}pp; HD95={best['hd95']:.4f}mm, ASD={best['asd']:.4f}mm.")
    lines += ["","The all-gamma tables expose accuracy-regularity trade-offs even when the best-Dice setting is not the smoothest. A gamma0 winner or boundary winner is a valid result, not a reason to expand this explicitly fixed pilot automatically. Further gamma refinement and first10 testing await a separate decision.","",
        "## Detailed files","","- primary_rank0.csv/json: all12 warm setting endpoints and signed baseline changes.",
        "- all_metrics.csv/json and evaluation_payloads.json: complete case metrics and original nested evaluation records.",
        "- per_label.csv/json: all35 label values/counts/eligibility reasons, including cold QA.",
        "- stage_components.csv/json: selected/actual epochs, stopping, separate loss terms, time and memory.",
        "- backend_deltas.csv/json: analytic-minus-FD diffusion at matching gamma.",
        "- evaluation_resources.csv/json: separately monitored scoring memory.",
        "- Each fresh run retains complete per-step component histories, best/last checkpoints and exact source/input/field receipts.",""]
    (out/"report.md").write_text("\n".join(lines)); dump(out/"best_tested.json",choices)
    complete=all(sorted(r["gamma"] for r in primary if r["backend"]==backend)==list(GAMMAS) for backend in BACKENDS)
    dump(out/"report_status.json",dict(at=stamp(),complete=complete,primary_settings=len(primary),scored_observations=len(all_rows),first10_extension=False))
    if primary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        specs=(("dice","Dice"),("hd95","HD95 mm"),("asd","ASD mm"),("log_jdet_std_fg","Central SDlogJ"),
               ("analytic_jacobian_fg_fraction_le_zero","Analytic folds %"),("ndv_frac_fg","NDV index"))
        for backend in BACKENDS:
            rows=sorted([r for r in primary if r["backend"]==backend],key=lambda r:r["gamma"])
            x=[r["gamma"] for r in rows]
            for ax,(key,label) in zip(axes.flat,specs):
                ax.plot(x,[r[key]*(100 if key=="analytic_jacobian_fg_fraction_le_zero" else 1) for r in rows],"o-",label=backend+" diffusion")
                ax.set(xlabel="Gamma (diffusion lambda=1)",ylabel=label,xscale="symlog"); ax.set_xscale("symlog",linthresh=1e-4)
        axes[0,0].legend(); fig.suptitle("Rank0 only: diffusion1 + analytic KL-C1; gamma0 references retained")
        fig.tight_layout(); fig.savefig(out/"gamma_accuracy_regularity.png",dpi=170); fig.savefig(out/"gamma_accuracy_regularity.pdf"); plt.close(fig)
    if (root/"ACCEPTED.json").exists() and not complete: raise RuntimeError("Accepted marker with incomplete report")
    print(dict(complete=complete,primary_settings=len(primary),choices=choices),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); render(a.root,a.out)
