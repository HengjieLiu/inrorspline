"""CPU-only final plots and independent audit of the completed overnight study."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def stage_diagnostics(rows):
    """Read retained endpoints only; no new registration or metric computation."""
    result=[]
    valid=True
    for row in rows:
        summary=json.loads(Path(row["summary"]).read_text())
        history=json.loads(Path(row["history"]).read_text())
        valid &= len(history)==sum(row["steps"])
        stages=summary["stage_summaries"]
        valid &= len(stages)==len(row["steps"])
        for stage in stages:
            index=int(stage["stage_idx"])
            values=[h for h in history if int(h["stage_idx"])==index]
            valid &= len(values)==row["steps"][index]==stage["actual_epochs"]
            if not values:
                continue
            last=values[-1]
            result.append(dict(id=row["id"],variant=row["variant"],rank=row["rank"],
                condition=row["condition"],stage=index,full_cps=stage["full_cps"],
                downsample=stage["downsample"],steps=stage["actual_epochs"],
                stop_reason=stage["stop_reason"],loop_and_postloop_wall_sec=stage["elapsed_sec"],
                last_preupdate_total=last["loss"],last_preupdate_similarity=last["data_loss"],
                last_preupdate_be=last["reg_loss"],last_preupdate_weighted_be=last["alpha"]*last["reg_loss"],
                first_effective_lr=values[0]["effective_lr"],last_effective_lr=last["effective_lr"]))
    return result,bool(valid)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--review-root",type=Path,required=True)
    a=p.parse_args()
    artifact=Path(os.environ["ARTIFACT_ROOT"]).resolve()
    if any(x.resolve()==artifact or not x.resolve().is_relative_to(artifact) for x in (a.run_root,a.review_root)):
        raise ValueError("External artifact child roots required")
    if not (a.run_root/"controller_EXIT.json").exists():
        raise ValueError("Wait for controller exit before finalizing its report")
    load=lambda p:json.loads(Path(p).read_text())
    result=load(a.review_root/"results.json")
    rows=load(a.run_root/"primary_rows.json")
    metrics=load(a.run_root/"quality_rows.json") if (a.run_root/"quality_rows.json").exists() else []
    accepted=[r for r in rows if r.get("phase")=="headline"]
    complete=[s for s in result["summaries"] if s["complete"]]
    reference=next((s for s in complete if s["variant"]=="reference_cpbe"),None)
    if complete:
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
        measures=(("dice","Mean Dice ↑"),("hd95","Mean HD95 (mm) ↓"),
                  ("ncc","Full-volume NCC ↑"),("folding","Foreground folding (%) ↓"))
        for ax,(key,label) in zip(axes.ravel(),measures):
            for i,s in enumerate(complete,1):
                color="#1b9e77" if s.get("all_gates_pass") else "#e6ab02" if s.get("primary_pass") else "#d95f02"
                ax.scatter(s["warm_seconds"]["mean"],s["metrics"][key]["mean"],c=color,s=45,zorder=3)
                ax.annotate(str(i),(s["warm_seconds"]["mean"],s["metrics"][key]["mean"]),xytext=(4,4),textcoords="offset points",fontsize=8)
            if reference:
                ax.axhline(reference["metrics"][key]["mean"],color="gray",ls="--",lw=.8)
                ax.axvline(reference["warm_seconds"]["mean"],color="gray",ls="--",lw=.8)
            ax.set_xlabel("Warm endpoint time (s)")
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        fig.suptitle("OASIS first10 tuning screen · CP-BE accuracy / speed\nGreen: all guards; yellow: Dice/HD95 guards only; orange: primary guard failure",fontsize=13)
        fig.savefig(a.review_root/"accuracy_speed.png",dpi=160)
        plt.close(fig)
    frozen=load(a.run_root/"formal_source_sha256.json")
    checks=dict(source_freeze=all(digest(path)==h for path,h in frozen.items()),
        clean_primary=all(r["clean"] for r in accepted),
        finite_float32_shape_validated=all(r["dvf_valid"] for r in accepted),
        expected_backend=all(r["backend"]["cudnn_benchmark"] is False and r["backend"]["cudnn_deterministic"] is True
                             and r["backend_extra"]["physical_gpu"]==0 for r in accepted),
        clean_quality=all(r["clean"] for r in metrics),
        label_accounting=all(r["hd95_nan_pattern_matches_dice"] and r["valid_labels"]>0 for r in metrics))
    endpoints=True
    for row in accepted:
        timing=load(row["timing"])
        marks={m["name"]:m["offset_sec"] for m in timing["marks"]}
        endpoints &= abs(row["elapsed_sec"]-(marks["dvf_file_ready"]-marks["case_load_complete"]))<1e-9
    checks["endpoint_deltas_match"]=bool(endpoints)
    stages,checks["history_steps_match"]=stage_diagnostics(accepted)
    if stages:
        with (a.review_root/"stage_diagnostics.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(stages[0]))
            writer.writeheader()
            writer.writerows(stages)
    quality_lookup={(r["variant"],r["rank"]):r for r in metrics}
    confirmations=[]
    for row in rows:
        if not row.get("phase","").startswith("confirmation"):
            continue
        ref=quality_lookup.get((row["variant"],row["rank"]))
        field=np.load(row["dvf"],mmap_mode="r")
        h=hashlib.sha256()
        for slab in field:
            h.update(memoryview(np.ascontiguousarray(slab)))
        confirmations.append(dict(id=row["id"],rank=row["rank"],sha256=h.hexdigest(),
            matches_headline=bool(ref and ref["dvf_sha256"]==h.hexdigest())))
    checks["confirmation_fields_equal_headline"]=all(r["matches_headline"] for r in confirmations)
    state=load(a.run_root/"state.json")
    core_complete=(a.run_root/"main_DONE.json").exists()
    summary=dict(checks=checks,accepted_checks_pass=all(checks.values()),main_complete=core_complete,
        controller_state=state,headline_count=len(accepted),quality_count=len(metrics),
        complete_candidates=len(complete),confirmation_fields=confirmations,
        report_source_sha256=digest(__file__))
    (a.review_root/"delivery_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
    shutil.copy2(__file__,a.review_root/"delivery_source.py")
    report=a.review_root/"report.md"
    extra="\n\n## Final independent delivery audit\n\n"
    extra+=f"Accepted-row checks passed: {summary['accepted_checks_pass']}; main experiment complete: {core_complete}. Full verdicts: `delivery_validation.json`.\n\n"
    if confirmations:
        extra+="Confirmation-field equality to headline results is checked using exact array SHA256; mismatches are flagged, not treated as confirmed quality equivalence.\n\n"
    if complete:
        extra+="![Accuracy and speed](accuracy_speed.png)\n\nPlot IDs:\n\n"
        extra+="\n".join(f"{i}. `{s['variant']}`" for i,s in enumerate(complete,1))+"\n"
    if stages:
        extra+="\n## Actual stage work and stopping\n\n"
        extra+="Warm cases only; each step count is mean [min, max]. The legacy stage wall clock starts AFTER volume/grid/optimizer setup and includes the loop plus post-loop statistics and any final expansion; it is not pure GPU kernel time or a complete stage-preparation timer.\n\n"
        extra+="| Candidate | Full cps | Mean steps [range] | Loop + post-loop wall s, mean ± SD | Stop counts |\n|---|---:|---:|---:|---|\n"
        for name in dict.fromkeys(s["variant"] for s in stages):
            for index in sorted({s["stage"] for s in stages if s["variant"]==name}):
                values=[s for s in stages if s["variant"]==name and s["stage"]==index and s["condition"]=="warm"]
                if not values:
                    continue
                counts=[s["steps"] for s in values]
                elapsed=[s["loop_and_postloop_wall_sec"] for s in values]
                sd=f"{statistics.stdev(elapsed):.4f}" if len(elapsed)>1 else "n/a"
                reasons="; ".join(f"{k}: {v}" for k,v in Counter(s["stop_reason"] for s in values).items())
                extra+=f"| {name} | {values[0]['full_cps']} | {statistics.mean(counts):.1f} [{min(counts)}, {max(counts)}] | {statistics.mean(elapsed):.4f} ± {sd} | {reasons} |\n"
        extra+="\n`stage_diagnostics.csv` retains cold/warm stage rows, first/last effective LR, and last PRE-update similarity, BE, weighted BE and total loss. These are not losses recomputed on the final post-update field; cross-resolution loss values are not interchangeable.\n"
        extra+="\n## Cold gap and setup outside the endpoint\n\n| Candidate | Cold − warm s | Cold setup s (excluded) | Warm setup mean s (excluded) |\n|---|---:|---:|---:|\n"
        for s in complete:
            own=[r for r in accepted if r["variant"]==s["variant"]]
            cold=next(r for r in own if r["condition"]=="cold")
            warm=[r["setup_device_sec"] for r in own if r["condition"]=="warm"]
            extra+=f"| {s['variant']} | {s['cold_warm_gap_seconds']:.4f} | {cold['setup_device_sec']:.4f} | {statistics.mean(warm):.4f} |\n"
    report.write_text(report.read_text()+extra)
    hashes={str(p.relative_to(a.review_root)):digest(p) for p in a.review_root.rglob("*") if p.is_file() and p.name!="output_sha256.json"}
    (a.review_root/"output_sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(summary))
    if not summary["accepted_checks_pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
