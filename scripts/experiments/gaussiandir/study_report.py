"""Complete external report for the sampled GaussianDIR convergence study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import atomic_json, external, sha256
from .report import flatten, write_csv

VARIANTS = ("released_control", "paper_reconstruction")
NAMES = {"released_control":"Released control", "paper_reconstruction":"Corrected reconstruction"}
COLORS = {"released_control":"#1f77b4", "paper_reconstruction":"#d6611c"}


def stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values), maximum=max(values), n=len(values))


def load_data(review):
    manifest = json.loads((review/"evaluation_manifest.json").read_text())
    if not (review/"evaluation_completed.json").exists():
        raise RuntimeError("Complete all scheduled posthoc evaluations first")
    records = []
    for job in manifest["jobs"]+manifest["aliases"]:
        key = job.get("metrics_from", job["key"])
        directory = review/"evaluations"/key
        result = json.loads((directory/"metrics.json").read_text())
        receipt = json.loads((directory/"receipt.json").read_text())
        resources = json.loads((directory/"resources.json").read_text())
        if "metrics_from" not in job and receipt["source_sha256"] != sha256(job["source"]):
            raise RuntimeError(f"Evaluation source drift: {key}")
        if "metrics_from" in job and job["field_sha256"] != sha256(job["source"]):
            raise RuntimeError("Aliased selected field changed")
        if result["eligible_labels"] != 35:
            raise RuntimeError(f"Unexpected missing labels: {key}")
        summary = json.loads((Path(job["run"])/"summary.json").read_text())
        if summary["kind"] == "convergence":
            # Selected-step optimizer time differs from all work charged to stop.
            selected = torch.load(Path(job["run"])/"selected.pt", map_location="cpu", weights_only=False)
            job["optimization_seconds"] = selected["optimization_seconds"]
            job["selected_charged_seconds"] = selected["charged_seconds"]
        phases = summary["resources"]["phases"]
        charged = sum(phases.get(k,{}).get("wall_seconds",0.) for k in
                      ("initialization", "optimization", "monitoring", "output_generation"))
        if abs(charged+summary["nonphase_registration_seconds"]-summary["registration_seconds"]) > 1e-6:
            raise RuntimeError("Registration timing components do not reconcile")
        if summary["checkpoint_replay_max_error"] != 0:
            raise RuntimeError("Checkpoint replay discrepancy")
        records.append(dict(job=job, metrics=result, training=summary, posthoc=resources,
                            metrics_sha256=sha256(directory/"metrics.json")))
    return manifest, records


def reconcile_boost_memory(review):
    """Recover NVML host PIDs for posthoc workers sharing each review GPU."""
    mapping_path = review/"posthoc_boost_pid_map.json"
    if not mapping_path.exists():
        return 0
    mapping = json.loads(mapping_path.read_text())["mapping"]
    boost = json.loads((review/"posthoc_boost_manifest.json").read_text())
    updated = 0
    for worker in boost["plan"]:
        name, pid = worker["name"], int(mapping[worker["name"]])
        for key in (j["key"] for j in json.loads((review/f"{name}.json").read_text())):
            folder = review/"evaluations"/key
            raw_path, resource_path = folder/"process_memory.json", folder/"resources.json"
            if not raw_path.exists() or not resource_path.exists():
                continue
            rows = json.loads(raw_path.read_text())
            own = [(sample, process["mib"]) for sample in rows for process in sample["processes"]
                   if int(process["pid"]) == pid]
            if not own:
                raise RuntimeError(f"NVML did not observe scorer PID {pid}: {key}")
            process_memory = json.loads(resource_path.read_text())["process_memory"]
            baseline = next((mib for sample,mib in own if sample["phase"] == "setup"), own[0][1])
            phases = {}
            own_times = set()
            for sample,mib in own:
                own_times.add(sample["time"])
                phases[sample["phase"]] = max(phases.get(sample["phase"], 0.), mib)
            siblings = sorted({int(process["pid"]) for sample in rows for process in sample["processes"]
                               if int(process["pid"]) != pid})
            process_memory.update(host_pid=pid, peak_process_mib_by_phase=phases,
                increase_above_loaded_baseline_mib={k:max(0.,v-baseline) for k,v in phases.items()},
                loaded_baseline_mib=baseline, competing_compute_pids=siblings,
                note="Concurrent posthoc processes were separated by NVML PID mapping;100ms samples can miss peaks.")
            details = json.loads(resource_path.read_text())
            details["posthoc_pid_mapping_source"] = "posthoc_boost_pid_map.json / docker top command line"
            atomic_json(resource_path, details)
            updated += 1
    return updated


def build(args):
    review, out = external(args.review), external(args.out)
    # Report generation is deterministic and overwrites its known outputs, so
    # an interrupted prior attempt can be resumed in the same external folder.
    out.mkdir(parents=True, exist_ok=True)
    reconciled = reconcile_boost_memory(review)
    manifest, records = load_data(review)
    boost_path = review/"posthoc_boost_manifest.json"
    boost = json.loads(boost_path.read_text()) if boost_path.exists() else None
    reference = {k:json.loads((args.references/k/"metrics.json").read_text()) for k in ("identity", "sitreg")}
    rows, labels, energy = [], [], []
    for record in records:
        job, result = record["job"], record["metrics"]
        row = dict(**job, **flatten({k:v for k,v in result.items() if k not in
                   ("input_receipts", "field_receipt", "conventions")}),
                   **flatten(record["training"], "training"), **flatten(record["posthoc"], "posthoc"))
        rows.append(row)
        labels += [dict(key=job["key"], **flatten(r)) for r in result["per_label"]]
        energy += [dict(key=job["key"], metric=kind, **r) for kind in ("bending_energy_rows", "diffusion_rows")
                   for r in result[kind]]
    for key,result in reference.items():
        rows.append(dict(key=key, variant=key, run_kind="saved_reference", **flatten(result)))
        labels += [dict(key=key, **flatten(r)) for r in result["per_label"]]
        energy += [dict(key=key, metric=kind, **r) for kind in ("bending_energy_rows", "diffusion_rows") for r in result[kind]]
    write_csv(out/"all_metrics.csv", rows)
    write_csv(out/"per_label.csv", labels)
    write_csv(out/"energy_metrics.csv", energy)

    groups, timing = [], []
    for variant in VARIANTS:
        for budget in (200, 500, 2000):
            group = [r for r in records if r["job"]["variant"] == variant and
                     r["job"]["run_kind"] == "fixed" and r["job"]["budget"] == budget]
            if len(group) != 3:
                raise RuntimeError("Missing one of three timing repetitions")
            summaries = [r["training"] for r in group]
            if any(s["process_memory"]["competing_compute_pids"] for s in summaries):
                raise RuntimeError("Contaminated timing repetitions")
            times = stats([s["registration_seconds"] for s in summaries])
            item = dict(variant=variant, budget=budget, kind="fixed", registration_seconds=times,
                optimization_seconds=stats([s["resources"]["phases"]["optimization"]["wall_seconds"] for s in summaries]),
                dice=stats([r["metrics"]["accuracy"]["dice_mean"] for r in group]),
                hd95=stats([r["metrics"]["accuracy"]["hd95_mean"] for r in group]),
                asd=stats([r["metrics"]["accuracy"]["asd_symmetric_mean"] for r in group]),
                folding=stats([r["metrics"]["regularity"]["jdet_le0_pct_fg"] for r in group]),
                sdlogj=stats([r["metrics"]["regularity"]["log_jdet_std_fg"] for r in group]),
                unique_field_hashes=len({s["field_sha256"] for s in summaries}),
                representative_key=group[0]["job"]["key"])
            groups.append(item)
            timing.append(flatten(item))
        convergence = next(r for r in records if r["job"]["variant"] == variant and r["job"]["run_kind"] == "convergence")
        result, summary = convergence["metrics"], convergence["training"]
        groups.append(dict(variant=variant, budget=summary["total_steps"], kind="convergence",
            registration_seconds=stats([summary["registration_seconds"]]),
            optimization_seconds=stats([summary["resources"]["phases"]["optimization"]["wall_seconds"]]),
            dice=stats([result["accuracy"]["dice_mean"]]), hd95=stats([result["accuracy"]["hd95_mean"]]),
            asd=stats([result["accuracy"]["asd_symmetric_mean"]]),
            folding=stats([result["regularity"]["jdet_le0_pct_fg"]]), sdlogj=stats([result["regularity"]["log_jdet_std_fg"]]),
            representative_key=convergence["job"]["key"], outcome=summary["outcome"], selected_step=summary["selected_step"]))
    write_csv(out/"timing_repeats_summary.csv", timing)
    frontier = []
    for metric, sign in (("dice", 1), ("hd95", -1), ("asd", -1)):
        for item in groups:
            t, value = item["registration_seconds"]["mean"], sign*item[metric]["mean"]
            dominated = any(other["registration_seconds"]["mean"] <= t and sign*other[metric]["mean"] >= value and
                (other["registration_seconds"]["mean"] < t or sign*other[metric]["mean"] > value) for other in groups)
            frontier.append(dict(metric=metric, key=item["representative_key"], seconds=t,
                value=sign*value, nondominated=not dominated, folding_fg_pct=item["folding"]["mean"]))
    write_csv(out/"descriptive_frontier.csv", frontier)
    atomic_json(out/"comparison.json", dict(records=records, reference=reference, groups=groups,
        frontier=frontier, posthoc_workers=boost))
    atomic_json(out/"provenance.json", dict(evaluation_manifest_sha256=sha256(review/"evaluation_manifest.json"),
        report_source_sha256=sha256(__file__), posthoc_boost_manifest_sha256=sha256(boost_path) if boost_path.exists() else None,
        posthoc_pid_map_sha256=sha256(review/"posthoc_boost_pid_map.json") if (review/"posthoc_boost_pid_map.json").exists() else None,
        reconciled_posthoc_memory_rows=reconciled,
        references={k:sha256(args.references/k/"metrics.json") for k in reference},
        metrics={r["job"]["key"]:r["metrics_sha256"] for r in records}))
    (out/"report_source.py").write_bytes(Path(__file__).read_bytes())
    plots(out, records, reference, groups)
    markdown(out, records, reference, groups, manifest,
             boosted=(review/"posthoc_boost_manifest.json").exists())
    print(json.dumps(dict(report=str(out/"report.md"), rows=len(rows), labels=len(labels), groups=len(groups))), flush=True)


def plots(out, records, reference, groups):
    plt.rcParams.update({"font.size":10, "axes.grid":True, "grid.alpha":.2})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col,(short, metric, title) in enumerate((("dice", "dice_mean", "Dice ↑"),
            ("hd95", "hd95_mean", "HD95 mm ↓"), ("asd", "asd_symmetric_mean", "Symmetric ASD mm ↓"))):
        for variant in VARIANTS:
            trajectory = sorted([r for r in records if r["job"]["variant"] == variant and
                                 r["job"]["run_kind"] == "trajectory"], key=lambda r:r["job"]["step"])
            axes[0,col].plot([r["job"]["optimization_seconds"] for r in trajectory],
                [r["metrics"]["accuracy"][metric] for r in trajectory], ".-", ms=3,
                color=COLORS[variant], label=NAMES[variant]+" trajectory")
            fixed = [g for g in groups if g["variant"] == variant and g["kind"] == "fixed"]
            axes[0,col].scatter([g["optimization_seconds"]["mean"] for g in fixed],
                [g[short]["mean"] for g in fixed], marker="s", color=COLORS[variant],
                edgecolor="black", label=NAMES[variant]+" fresh short/anchor")
            chosen = next(r for r in records if r["job"]["variant"] == variant and
                          r["job"]["run_kind"] == "convergence")
            axes[0,col].scatter(chosen["job"]["optimization_seconds"],
                chosen["metrics"]["accuracy"][metric], marker="*", s=125,
                color=COLORS[variant], edgecolor="black", zorder=5,
                label=NAMES[variant]+" selected checkpoint")
            axes[0,col].annotate("selected", (chosen["job"]["optimization_seconds"],
                chosen["metrics"]["accuracy"][metric]), xytext=(4,5), textcoords="offset points", fontsize=8)
            selected = [g for g in groups if g["variant"] == variant]
            for g in selected:
                axes[1,col].errorbar(g["registration_seconds"]["mean"], g[short]["mean"],
                    xerr=g["registration_seconds"]["sample_sd"] or 0., fmt="*" if g["kind"] == "convergence" else "o",
                    color=COLORS[variant], ms=10 if g["kind"] == "convergence" else 6)
                axes[1,col].annotate("polish" if g["kind"] == "convergence" else str(g["budget"]),
                    (g["registration_seconds"]["mean"], g[short]["mean"]), xytext=(4,5),
                    textcoords="offset points", fontsize=8, color=COLORS[variant])
        for row in range(2):
            axes[row,col].axhline(reference["sitreg"]["accuracy"][metric], color="black", ls="--", label="SITReg accuracy")
            axes[row,col].set_xscale("symlog", linthresh=1.)
            axes[row,col].set_ylabel(title)
            axes[row,col].set_xlabel("Cumulative optimization seconds" if row == 0 else "Usable-output registration seconds")
        axes[0,col].set_title(title)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("One exploratory OASIS pair · timing SD across same-seed repeats · polish includes all stopping work")
    fig.savefig(out/"accuracy_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    process_fig, process_axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for col,variant in enumerate(VARIANTS):
        record = next(r for r in records if r["job"]["variant"] == variant and r["job"]["run_kind"] == "convergence")
        run = Path(record["job"]["run"])
        history = json.loads((run/"history.json").read_text())
        monitoring = json.loads((run/"monitoring.json").read_text())
        steps = [r["total_steps"] for r in history]
        axes[0,col].plot(steps, [r["loss"] for r in history], lw=.8, label="Training batch total")
        for bank in range(2):
            axes[1,col].plot([r["step"] for r in monitoring], [r["scores"][bank] for r in monitoring], label=f"Fixed bank {bank+1}")
        axes[2,col].plot(steps, [r["lr_multiplier"] for r in history], color="purple", label="LR multiplier")
        axes[2,col].set_yscale("log")
        twin = axes[2,col].twinx()
        twin.plot(steps, [r["nodes"] for r in history], color="green", alpha=.65)
        twin.set_ylabel("Gaussian nodes", color="green")
        for key,label in (("allocated_mib", "Torch allocated"), ("reserved_mib", "Torch reserved")):
            axes[3,col].plot(steps, [r[key] for r in history], label=label)
        for row in range(4):
            axes[row,col].axvline(2000, color="grey", ls="--")
            axes[row,col].set_xlabel("Completed updates")
            axes[row,col].legend(fontsize=8)
        axes[0,col].set_title(NAMES[variant])
        axes[0,col].set_ylabel("Signed NCC + 8 TV")
        axes[1,col].set_ylabel("Fixed-bank mean objective")
        axes[2,col].set_ylabel("Fraction of native group LR")
        axes[3,col].set_ylabel("MiB")
        memory = json.loads((run/"process_memory.json").read_text())
        pid = record["training"]["process_memory"]["host_pid"]
        values = [(r["time"]-memory[0]["time"], next((p["mib"] for p in r["processes"] if p["pid"]==pid), 0.)) for r in memory]
        process_axes[col].plot([t for t,v in values], [v for t,v in values])
        process_axes[col].set(title=NAMES[variant], xlabel="Worker seconds (includes snapshots/setup)", ylabel="Sampled process GPU MiB")
    fig.savefig(out/"optimization_traces.png", dpi=150)
    process_fig.savefig(out/"process_memory.png", dpi=150)
    plt.close(fig); plt.close(process_fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    keys = ("jdet_le0_pct_fg", "diffusion_l2_fg", "be_fd_forward_fixed_fg_voxel")
    titles = ("Foreground folds %", "Foreground diffusion", "Foreground bending energy")
    for variant in VARIANTS:
        trajectory = sorted([r for r in records if r["job"]["variant"]==variant and
                             r["job"]["run_kind"]=="trajectory"], key=lambda r:r["job"]["step"])
        for ax,key,title in zip(axes,keys,titles):
            ax.plot([r["job"]["optimization_seconds"] for r in trajectory],
                    [r["metrics"]["regularity"][key] for r in trajectory], ".-", ms=3,
                    color=COLORS[variant], label=NAMES[variant])
            ax.axhline(reference["sitreg"]["regularity"][key], color="black", ls="--")
            ax.set(xlabel="Cumulative optimization seconds", ylabel=title, xscale="symlog")
    axes[0].legend(fontsize=8)
    fig.savefig(out/"regularity_time.png", dpi=150); plt.close(fig)


def markdown(out, records, reference, groups, manifest, boosted=False):
    sit = reference["sitreg"]
    lines = ["# GaussianDIR: paper loss, convergence, latency and GPU memory", "",
        "Fixed 0333 / moving 0115, seed-3 shuffled OASIS rank 0. Sampled signed NCC + TV, journal lambda 8. "
        "One exploratory pair; three same-seed timing repeats measure runtime variation, not scientific uncertainty.", "",
        "## Accuracy, regularity and usable-output latency", "",
        "| Configuration | Steps | Latency mean ± SD s | Dice | HD95 mm | ASD mm | FG folds % | FG SDlogJ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        t = g["registration_seconds"]
        label = NAMES[g["variant"]]+(" — selected polish" if g["kind"]=="convergence" else "")
        time_text = f"{t['mean']:.3f}"+(f" ± {t['sample_sd']:.3f}" if t["sample_sd"] is not None else "")
        lines.append(f"| {label} | {g['budget']:,} | {time_text} | {g['dice']['mean']:.6f} | {g['hd95']['mean']:.4f} | "
                     f"{g['asd']['mean']:.4f} | {g['folding']['mean']:.6g} | {g['sdlogj']['mean']:.6f} |")
    a,reg = sit["accuracy"],sit["regularity"]
    lines.append(f"| Original SITReg (saved reference) | — | unmeasured | {a['dice_mean']:.6f} | {a['hd95_mean']:.4f} | "
                 f"{a['asd_symmetric_mean']:.4f} | {reg['jdet_le0_pct_fg']:.6g} | {reg['log_jdet_std_fg']:.6f} |")
    lines += ["", "Both GaussianDIR variants start from identity. Loss reductions, population rules and capacity "
        "differ. Corrected reconstruction includes explicitly specified two-scale details; joint polish is a further "
        "extension. These rows do not isolate anisotropy or prove a global optimum.", "",
        "## Convergence disposition", ""]
    for variant in VARIANTS:
        r = next(r for r in records if r["job"]["variant"]==variant and r["job"]["run_kind"]=="convergence")
        s = r["training"]
        anchor = next(g for g in groups if g["variant"]==variant and g["kind"]=="fixed" and g["budget"]==2000)
        change = r["metrics"]["accuracy"]["dice_mean"]-anchor["dice"]["mean"]
        lines += [f"- **{NAMES[variant]}:** `{s['outcome']}` after {s['total_steps']:,} updates; selected step "
            f"{s['selected_step']:,}; {len(s['lr_reductions'])} LR reductions; "
            f"{s['registration_seconds']:.3f} s charged, {s['operation_overshoot_seconds']:.3f} s overshoot. "
            f"Selected Dice change versus the 2,000-step anchor: {change:+.6f}. "
            f"Final frozen populations: {s['population_frozen_in_refinement']}."]
    lines += ["", "Selection minimizes the primary fixed bank; both banks and 8,192 foreground probe points determine "
        "plateau. All work until stopping and selected-output generation is charged. A time limit without meeting "
        "the criteria means convergence remains unresolved. Terminal metrics and every archived checkpoint are "
        "included in the complete tables; labels never influence optimization, selection or stopping.", "",
        "## Timing and GPU memory", "",
        "| Run | Setup s | Init s | Opt s | Monitor s | Output s | Snapshot/analysis s | Torch opt alloc/reserved MiB | Torch output alloc/reserved MiB | Process peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        j,s = r["job"],r["training"]
        if j["run_kind"]=="trajectory": continue
        phases = s["resources"]["phases"]
        def sec(name): return phases.get(name,{}).get("wall_seconds",0.)
        def memory(name):
            p=phases[name]
            return f"{p['peak_allocated_mib']:.1f}/{p['peak_reserved_mib']:.1f}"
        process = max(v for k,v in s["process_memory"]["peak_process_mib_by_phase"].items()
                      if k not in ("analysis_storage", "setup"))
        lines.append(f"| {j['key']} | {s['setup_seconds']:.3f} | {sec('initialization'):.3f} | {sec('optimization'):.3f} | "
            f"{sec('monitoring'):.3f} | {sec('output_generation'):.3f} | {sec('analysis_storage'):.3f} | "
            f"{memory('optimization')} | {memory('output_generation')} | {process:.0f} |")
    lines += ["", "Registration latency is synchronized wall time from GPU-ready inputs to full DVF and warped image, "
        "including initialization, optimization, required monitoring and control/instrumentation. Setup, snapshots, "
        "serialization and posthoc labels/surfaces are separate. CUDA-event intervals may include GPU idle gaps during "
        "CPU work. The tables retain phase absolute/incremental Torch peaks and sampled NVML process peaks; NVML "
        "sampling every 100 ms can miss brief peaks. Posthoc peaks are measured separately in evaluation resources.", ""]
    if boosted:
        lines += ["Posthoc scoring used additional workers on disjoint later checkpoint ranges to reduce review time. "
            "Workers shared review GPUs; synchronized Torch allocated/reserved peaks are recorded per evaluation. "
            "NVML records process lists, but the container PID namespace prevents separate NVML peak attribution "
            "for each simultaneous scorer. Registration phase attribution used one worker per GPU and remains intact. "
            "Posthoc concurrency did not overlap registration timing.", ""]
    for r in records:
        if r["job"]["run_kind"] == "startup":
            lines.append(f"- {NAMES[r['job']['variant']]} fresh-process 2,000-step latency: "
                         f"{r['training']['fresh_process_seconds']:.3f} s (imports, verification and provenance included).")
    example = json.loads((Path(records[0]["job"]["run"])/"manifest.json").read_text())
    env = example["environment"]
    lines += ["", f"Hardware/runtime: {env['device']}; PyTorch {env['torch']}, CUDA build {env['cuda_build']}, "
        f"PyTorch3D {env['pytorch3d']}. Requested/effective cuDNN benchmark=False, deterministic=True, "
        "matmul TF32=False and cuDNN TF32=False. The GPU was checked empty before every timing worker; "
        "sampled competing-compute PID records are retained.", "",
        "| Configuration | Initial nodes / parameters | Selected nodes / parameters | Peak nodes / parameters | Posthoc allocated / reserved MiB |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for g in groups:
        r = next(r for r in records if r["job"]["key"]==g["representative_key"])
        s = r["training"]
        post = r["posthoc"]["resources"]["phases"]["posthoc_evaluation"]
        lines.append(f"| {NAMES[g['variant']]} {g['kind']} {g['budget']} | "
            f"{s['initial_counts']['nodes']} / {s['initial_counts']['parameters']} | "
            f"{s['selected_counts']['nodes']} / {s['selected_counts']['parameters']} | "
            f"{s['peak_nodes']} / {s['peak_parameters']} | "
            f"{post['peak_allocated_mib']:.1f} / {post['peak_reserved_mib']:.1f} |")
    lines += ["", "No SITReg inference time is claimed. Loading its saved field is only reference evaluation. "
        "The paper's DIR-Lab2.5s and supplement5.1s on RTX4090 are different operating points and timer boundaries; "
        "the released 2,000-step loop has no automatic early stopping.", "",
        "## Review artifacts", "",
        "![Accuracy and latency](accuracy_time.png)", "", "![Loss, LR and populations](optimization_traces.png)", "",
        "![Process GPU memory](process_memory.png)", "", "![Regularity over optimization](regularity_time.png)", "",
        "- `all_metrics.csv`: all accuracy, full/foreground regularity, field/image fidelity and resource scalars.",
        "- `per_label.csv`: all 35 labels, eligibility/counts, Dice, HD95 and symmetric/directional ASD.",
        "- `energy_metrics.csv`: all diffusion/BE units, formulas and domains.",
        "- `timing_repeats_summary.csv`: same-seed mean/sample SD/range and distinct field hashes.",
        "- `descriptive_frontier.csv`: non-dominated measured latency/accuracy points, separately for Dice/HD95/ASD, "
        "with folding companion. It is descriptive, single-case and not a tuned held-out frontier.",
        "- `comparison.json`, `provenance.json`: complete inputs, selections, checksums and results.", "",
        f"Posthoc scoring covered {len(manifest['jobs'])} distinct exports/checkpoints plus {len(manifest['aliases'])} "
        "byte-identical field aliases. Metric reuse requires exact saved-field SHA-256 equality. No approximate reuse.", "",
        "Original pilot and standardized-loss prototype artifacts remain unchanged. LNCC²/diffusion, dense-gradient "
        "optimization, new datasets and lambda sweeps are deferred."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args())
