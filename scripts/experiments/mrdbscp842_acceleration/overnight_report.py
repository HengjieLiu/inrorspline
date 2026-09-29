"""Incremental/final accuracy-first report. Never includes metric time in runtime."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .overnight_candidates import GATES
from .run_suite import dump


def stats(values):
    values = [float(v) for v in values]
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,n=len(values)) if values else dict(mean=None,sd=None,n=0)


def table(path, rows):
    if not rows:
        return
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def build(root, review):
    review.mkdir(parents=True,exist_ok=True)
    registrations = load(root/"primary_rows.json",[])
    metrics = load(root/"quality_rows.json",[])
    specs = load(root/"candidate_manifest.json",[])
    metric_map = {r["registration_id"]:r for r in metrics if r["clean"]}
    cases = []
    for r in registrations:
        if r.get("phase") != "headline":
            continue
        m = metric_map.get(r["id"])
        if not m:
            continue
        reg = m["regularity"]
        cases.append(dict(variant=r["variant"],rank=r["rank"],seconds=r["elapsed_sec"],
            dice=m["mean_dice"],hd95=m["surface_metrics"]["hd95_mean"],ncc=m["ncc_full"],
            folding=reg["jdet_le0_pct_fg"],sdlogj=reg["log_jdet_std_fg"],
            be=reg["be_fd_forward_fixed_fg_voxel"],diffusion=reg["diffusion_l2_fg"],
            folding_full=reg["jdet_le0_pct_full"],sdlogj_full=reg["log_jdet_std_full"],
            positive_j_fg=reg["log_jdet_pos_frac_fg"],positive_j_full=reg["log_jdet_pos_frac_full"],
            peak_gib=r["peak_gib"],steps=r["steps"],valid_labels=m["valid_labels"],id=r["id"]))
    reference = {r["rank"]:r for r in cases if r["variant"]=="reference_cpbe"}
    paired,labels,summaries = [],[],[]
    all_metrics = ("seconds","dice","hd95","ncc","folding","sdlogj","be","diffusion","peak_gib")
    for spec in specs:
        name = spec["name"]
        rows = [r for r in cases if r["variant"]==name]
        times = [r for r in registrations if r["variant"]==name and r.get("phase")=="headline"]
        cold = [r for r in times if r["condition"]=="cold"]
        warm = [r for r in times if r["condition"]=="warm"]
        summary = dict(variant=name,complete=len(rows)==10 and sorted(r["rank"] for r in rows)==list(range(10)) and len(cold)==1 and len(warm)==10,
            timed=len(times),evaluated=len(rows),cold_seconds=cold[0]["elapsed_sec"] if cold else None,
            warm_seconds=stats([r["elapsed_sec"] for r in warm]),
            metrics={k:stats([r[k] for r in rows]) for k in all_metrics},candidate=spec)
        if cold and warm:
            summary["cold_warm_gap_seconds"] = cold[0]["elapsed_sec"]-summary["warm_seconds"]["mean"]
        deltas = []
        same_labels = True
        for r in rows:
            if r["rank"] not in reference:
                continue
            ref = reference[r["rank"]]
            diff = {k:r[k]-ref[k] for k in all_metrics}
            paired.append(dict(variant=name,rank=r["rank"],**diff))
            deltas.append(diff)
            a,b = metric_map[r["id"]],metric_map[ref["id"]]
            for label,(ad,bd,ah,bh) in enumerate(zip(a["dice"],b["dice"],a["surface_metrics"]["hd95"],b["surface_metrics"]["hd95"]),1):
                same_labels &= math.isfinite(ad)==math.isfinite(bd) and math.isfinite(ah)==math.isfinite(bh)
                labels.append(dict(variant=name,rank=r["rank"],label=label,dice_delta=ad-bd,
                    hd95_delta_mm=ah-bh,reference_dice=bd,candidate_dice=ad,
                    reference_hd95_mm=bh,candidate_hd95_mm=ah,
                    moving_voxels=a["label_voxel_counts"]["moving"][label-1],
                    fixed_voxels=a["label_voxel_counts"]["fixed"][label-1],
                    warped_voxels=a["label_voxel_counts"]["warped"][label-1]))
        if summary["complete"] and len(deltas)==10:
            avg = lambda k: statistics.mean(d[k] for d in deltas)
            means = summary["metrics"]
            refmean = lambda k: statistics.mean(r[k] for r in reference.values())
            primary = dict(dice_mean=avg("dice")>=-GATES["dice_mean_drop"],
                dice_worst=min(d["dice"] for d in deltas)>=-GATES["dice_worst_drop"],
                hd95_mean=avg("hd95")<=GATES["hd95_mean_increase_mm"],
                hd95_worst=max(d["hd95"] for d in deltas)<=GATES["hd95_worst_increase_mm"],
                same_valid_labels=bool(same_labels))
            secondary = dict(ncc=avg("ncc")>=-GATES["ncc_mean_drop"],
                folding=avg("folding")<=GATES["folding_fg_increase_pp"],
                sdlogj=means["sdlogj"]["mean"]<=refmean("sdlogj")*GATES["sdlogj_fg_ratio"],
                be=means["be"]["mean"]<=refmean("be")*GATES["fd_be_fg_ratio"],
                diffusion=means["diffusion"]["mean"]<=refmean("diffusion")*GATES["diffusion_fg_ratio"])
            summary.update(primary_gates=primary,secondary_gates=secondary,
                primary_pass=all(primary.values()),all_gates_pass=all(primary.values()) and all(secondary.values()),
                paired_differences={k:stats([d[k] for d in deltas]) for k in all_metrics},
                warm_change_percent=100*(summary["warm_seconds"]["mean"]/refmean("seconds")-1))
        summaries.append(summary)
    complete = [s for s in summaries if s["complete"]]
    eligible = [s for s in complete if s.get("all_gates_pass")]
    best = min(eligible,key=lambda s:s["warm_seconds"]["mean"])["variant"] if eligible else None
    confirmations=[]
    for name in sorted({r["variant"] for r in registrations if r.get("phase","").startswith("confirmation")}):
        for phase in ("confirmation0","confirmation1"):
            rows=[r for r in registrations if r["variant"]==name and r.get("phase")==phase]
            confirmations.append(dict(variant=name,phase=phase,rows=len(rows),
                warm_seconds=stats([r["elapsed_sec"] for r in rows if r["condition"]=="warm"]),
                cold_seconds=next((r["elapsed_sec"] for r in rows if r["condition"]=="cold"),None)))
    # Accuracy/runtime Pareto: no weighted composite can hide a quality loss.
    for s in complete:
        point = (s["warm_seconds"]["mean"],-s["metrics"]["dice"]["mean"],s["metrics"]["hd95"]["mean"])
        s["pareto_accuracy_time"] = not any(
            all(a<=b for a,b in zip((t["warm_seconds"]["mean"],-t["metrics"]["dice"]["mean"],t["metrics"]["hd95"]["mean"]),point)) and
            any(a<b for a,b in zip((t["warm_seconds"]["mean"],-t["metrics"]["dice"]["mean"],t["metrics"]["hd95"]["mean"]),point)) for t in complete)
    payload = dict(summaries=summaries,confirmations=confirmations,fastest_all_gate_candidate=best,gates=GATES,
        state=load(root/"state.json",{}),candidate_count=len(specs),complete_candidates=len(complete),
        timing_rows=len(registrations),metric_rows=len(metrics),
        limitations="Exploratory first10 tuning; NOT held-out equivalence or publication cohort evidence. CP-BE prior FP32 non-equivalence retained.")
    dump(review/"results.json",payload)
    dump(review/"evaluations.json",metrics)
    table(review/"case_metrics.csv",cases)
    table(review/"paired_differences.csv",paired)
    table(review/"label_differences.csv",labels)
    table(review/"timings.csv",[dict(id=r["id"],variant=r["variant"],rank=r["rank"],phase=r.get("phase"),condition=r["condition"],
        seconds=r["elapsed_sec"],peak_gib=r["peak_gib"],steps=r["steps"],stop_reasons=r["stop_reasons"],
        setup_device_sec=r["setup_device_sec"],history=r["history"],dvf=r["dvf"]) for r in registrations])
    f = lambda v: "—" if v is None else f"{v:.4f}"
    lines = ["# OASIS first10 accuracy-first CP-BE speed screen", "",
        payload["limitations"], "", "## Status and protocol", "",
        f"Completed candidates: {len(complete)}/{len(specs)}. State: `{payload['state']}`.",
        "Physical GPU 0, sequential Docker workers, seed-3 ranks 0–9, alpha1000 seed1 float32, cuDNN F/T, six CPU threads.",
        "Per candidate: fresh worker rank0 cold, rank0 warm then ranks1–9 warm. Only ten warm fields form cohort metrics.",
        "Endpoint case_load_complete→dvf_file_ready includes optimization and field writing; excludes loading/device setup and every metric.",
        "NCC is recomputed on the FINAL full-resolution warped MRI; no comparison of differently downsampled last-step losses.",
        "Regularity: voxel-unit FD BE/diffusion (not training CP-BE), fixed raw>0 foreground; float64 Jacobian crop2; positive-J natural-log population SD plus folding/positive fractions.",
        "Dice/HD95: 35 labels and existing missing-label rules, area-weighted surface HD95 at1mm. Label-mask mismatches fail promotion.",
        "Losses are pre-update, final field/quality post-update. Base LR and actual per-step LR are both retained in history.json.",
        "No segmentation-driven per-case optimizer/stop. First10 selects candidates, so independent validation is still required.", "",
        "## Accuracy and speed", "", "| Candidate | Cases | Cold s | Warm s ± SD | Dice | HD95 mm | Δ warm % | Primary gates | All gates |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for s in summaries:
        w,m=s["warm_seconds"],s["metrics"]
        lines.append(f"| {s['variant']} | {s['evaluated']}/10 | {f(s['cold_seconds'])} | {f(w['mean'])} ± {f(w['sd'])} | {f(m['dice']['mean'])} | {f(m['hd95']['mean'])} | {f(s.get('warm_change_percent'))} | {s.get('primary_pass','pending')} | {s.get('all_gates_pass','pending')} |")
    lines += ["", "## Similarity and regularity", "", "| Candidate | NCC ↑ | FG folding % ↓ | FG SDlogJ | FG FD BE | FG diffusion | Peak GiB mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        m=s["metrics"]
        lines.append("| "+s["variant"]+" | "+" | ".join(f(m[k]["mean"]) for k in ("ncc","folding","sdlogj","be","diffusion","peak_gib"))+" |")
    lines += ["", "## Prespecified engineering gates", "", "```json",json.dumps(GATES,indent=2),"```", "",
        "Gate limits are exploratory screening tolerances, not established clinical margins. Reports retain all tradeoffs, rejected candidates, per-case differences and per-label outliers.",
        f"Fastest complete candidate passing every gate: **{best or 'none yet'}**. This is not a declaration of equivalence.",
        "Cold has one observation per candidate; no cold SD or strong cold inference. Warm SD is sample SD across ten cases, not pure timer noise.", "",
        "## Reproduction / audit", "",
        f"Run root: `{root}`. Candidate JSON, exact worker argv, source snapshots/hash freeze, Git dirty provenance, Docker/data manifests, GPU monitor and all retries retained there.",
        "`results.json` includes individual gate verdicts and 3-objective Pareto flags. `case_metrics.csv`, `paired_differences.csv`, `label_differences.csv` and `timings.csv` are machine-readable.",
        "Interrupted/contaminated attempts are not silently substituted into primary summaries. Missing candidates remain marked incomplete."]
    lines += ["", "## Candidate-specific quality verdicts", ""]
    for s in complete:
        if "primary_gates" not in s:
            continue
        failed=[k for k,v in {**s["primary_gates"],**s["secondary_gates"]}.items() if not v]
        d=s["paired_differences"]
        lines.append(f"- {s['variant']}: paired Dice {d['dice']['mean']:+.5f}, HD95 {d['hd95']['mean']:+.4f} mm; failed guards: {', '.join(failed) or 'none'}; accuracy/time Pareto: {s['pareto_accuracy_time']}.")
    if confirmations:
        lines += ["", "## Fresh-worker timing confirmation", "", "| Candidate | Repeat | Rows | Warm s ± SD | Cold s |", "|---|---|---:|---:|---:|"]
        for r in confirmations:
            lines.append(f"| {r['variant']} | {r['phase']} | {r['rows']}/11 | {f(r['warm_seconds']['mean'])} ± {f(r['warm_seconds']['sd'])} | {f(r['cold_seconds'])} |")
    provenance=load(root/"latest_provenance.json",{})
    if provenance:
        manifest=load(Path(provenance["path"])/"manifest.json",{})
        lines += ["", f"Git HEAD: `{manifest.get('git_head')}`. Dirty status was recorded: `{bool(manifest.get('git_status'))}`."]
    (review/"report.md").write_text("\n".join(lines)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--review",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.root,a.review),default=str))


if __name__ == "__main__":
    main()
