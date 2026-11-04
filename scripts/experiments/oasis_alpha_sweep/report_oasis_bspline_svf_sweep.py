"""P74 complete-cohort metrics, resources, paired comparisons and line-only ARC."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import bspline_svf_sweep as s

c = s.c
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
RESOURCES = ("optimizer_sec", "worker_sec", "metrics_diagnostics_sec", "audit_sec",
             "actual_adam_updates", "peak_cuda_gib", "peak_reserved_gib",
             "audit_peak_cuda_gib", "audit_peak_reserved_gib")


def summarize(rows, keys=(*METRICS, *RESOURCES)):
    if {x["case_rank"] for x in rows} != set(range(10)) or len(rows) != 10:
        raise ValueError("Only a complete unique ten-case cohort can be summarized")
    result = {"n": 10}
    for key in keys:
        values = [float(x[key]) for x in rows if x.get(key) not in (None, "", "n/a")]
        if values and (len(values) != 10 or not all(math.isfinite(v) for v in values)):
            raise ValueError(f"Partial or nonfinite metric:{key}")
        if not values and key in METRICS: raise ValueError(f"Missing primary metric:{key}")
        for suffix, value in (("mean", statistics.mean(values) if values else None),
                              ("sd", statistics.stdev(values) if values else None),
                              ("min", min(values) if values else None), ("max", max(values) if values else None)):
            result[f"{key}_{suffix}"] = value
    return result


def family(schedule, regularize):
    return f"Refined SVF {'→'.join(schedule)} / {regularize}"


def pareto(rows):
    def cost(row): return [-row["dice_mean"], *[row[f"{k}_mean"] for k in METRICS[1:]]]
    return [x for x in rows if not any(all(a <= b for a,b in zip(cost(y),cost(x)))
            and any(a < b for a,b in zip(cost(y),cost(x))) for y in rows if y is not x)]


def fmt(row, key, digits=6):
    a, b = row.get(f"{key}_mean"), row.get(f"{key}_sd")
    return "n/a" if a is None else f"{a:.{digits}f} ± {b:.{digits}f}"


def generate(rows, manifest, out):
    if (out/"ACCEPTED.json").exists(): raise RuntimeError("Accepted report is immutable")
    expected = {x.name() for x in s.matrix()}
    if len(rows) != 280 or {r["run_name"] for r in rows} != expected:
        raise RuntimeError("Require exactly280 independently audited registrations")
    out.mkdir(parents=True, exist_ok=True)
    groups, flat, stage_rows, sensitivity, labels = defaultdict(list), [], [], [], []
    for r in rows:
        spec = s.Spec(**r["spec"])
        base = {"run_name": r["run_name"], "case_rank": spec.case_rank, "fixed_id": r["fixed_id"],
                "moving_id": r["moving_id"], "schedule": spec.schedule, "regularize": spec.regularize,
                "alpha": spec.alpha, "method": family(spec.schedule, spec.regularize), "source_kind": "new_svf"}
        x = {**base, **{k: r[k] for k in (*METRICS, *RESOURCES, "gpu", "run_dir", "valid_label_count")},
             "velocity_diffusion_native": r["blocks"][-1]["velocity_diffusion_native"],
             "displacement_diffusion_native": r["blocks"][-1]["displacement_diffusion_native"],
             "initial_derivative_bound_max": max(b["scaled_initial_map_corners"]["max_displacement_jacobian_infinity_norm"] for b in r["blocks"])}
        flat.append(x); groups[(spec.schedule, spec.regularize, spec.alpha)].append(x)
        for block in r["blocks"]:
            stage_rows.append({**base, "cps": block["cps"], **block["optimizer"],
                               "initial_derivative_bound": block["scaled_initial_map_corners"]["max_displacement_jacobian_infinity_norm"],
                               "bound_below_one": block["scaled_initial_map_corners"]["bound_below_one"]})
        for diag in r["integration_diagnostics"]:
            sensitivity.append({**base, "integration_steps": diag["integration_steps"], **{k: diag["metrics"][k] for k in METRICS},
                                **{k:v for k,v in diag.items() if k.startswith("delta_from")},
                                **{f"delta_ss7_{k}": v for k,v in diag["delta_primary_metrics"].items()},
                                **{f"inverse_{order}_{key}": value for order, vals in diag["inverse_consistency"].items()
                                   if isinstance(vals, dict) for key,value in vals.items()}})
        with (Path(r["run_dir"])/"final_metrics/case_metrics.csv").open() as stream:
            lr = next(csv.DictReader(stream))
        if lr["fixed_id"] != r["fixed_id"] or lr["moving_id"] != r["moving_id"]:
            raise RuntimeError("Label table subject mismatch")
        for label in c.OASIS_LABELS:
            vals = {k: float(lr[f"{src}_label_{label:02d}"]) for k,src in
                    (("dice", "dice"), ("hd95", "hd95"), ("asd", "asd_symmetric"))}
            valid = all(math.isfinite(v) for v in vals.values())
            labels.append({**base, "label": label, "valid": valid,
                           **{k: v if math.isfinite(v) else None for k,v in vals.items()}})
    summaries = []
    for key in sorted(groups):
        schedule, regularize, alpha = key
        summaries.append({"schedule": schedule, "regularize": regularize, "alpha": alpha,
                          "method": family(schedule, regularize), "source_kind": "new_svf", **summarize(groups[key])})
    references, refgroups = [], defaultdict(list)
    for row in manifest["references"]:
        x = {k: row[k] for k in ("case_rank", "fixed_id", "moving_id", "method", *METRICS)}
        x.update(source_kind="original_DL", **{k: None for k in RESOURCES})
        references.append(x); refgroups[x["method"]].append(x)
    reference_summaries = [{"method": method, "source_kind": "original_DL", **summarize(items)} for method,items in refgroups.items()]
    if len(reference_summaries) != 2: raise RuntimeError("Require two original DL references")
    lookup = {(x["schedule"],x["regularize"],x["alpha"],x["case_rank"]): x for x in flat}
    paired = []
    for x in flat:
        opponents = [(r["method"],r) for r in references if r["case_rank"] == x["case_rank"]]
        opponents.append(("same setting lambda1", lookup[(x["schedule"],x["regularize"],1.,x["case_rank"])]))
        opponents.append(("other penalty same lambda", lookup[(x["schedule"],"displacement" if x["regularize"] == "velocity" else "velocity",x["alpha"],x["case_rank"])]))
        opponents.append(("other depth same lambda", lookup[("8421" if x["schedule"] == "842" else "842",x["regularize"],x["alpha"],x["case_rank"])]))
        for name, ref in opponents:
            paired.append({"run_name": x["run_name"], "case_rank": x["case_rank"], "schedule": x["schedule"],
                           "regularize": x["regularize"], "alpha": x["alpha"], "reference": name,
                           **{f"delta_{k}": x[k]-ref[k] for k in METRICS}})
    pilot = []
    for ref in manifest["pilot_rank0"]:
        old = ref["spec"]
        now = lookup[(old["schedule"],old["regularize"],1.,0)]
        pilot.append({"schedule": old["schedule"], "regularize": old["regularize"],
                      **{f"new_minus_pilot_{k}": now[k]-ref[k] for k in METRICS}})
    selected, resource_methods = [], []
    for schedule in ("842", "8421"):
        for reg in ("velocity", "displacement"):
            candidates = [x for x in summaries if x["schedule"] == schedule and x["regularize"] == reg]
            selected.append(min(candidates, key=lambda x: (-x["dice_mean"],x["asd_mean"],x["hd95_mean"])))
            pool = [x for x in flat if x["schedule"] == schedule and x["regularize"] == reg]
            stats = {"method": family(schedule,reg), "n": len(pool), "scope": "pooled over seven lambdas and ten cases"}
            for key in RESOURCES:
                vals = [x[key] for x in pool]
                stats.update({f"{key}_mean": statistics.mean(vals), f"{key}_sd": statistics.stdev(vals),
                              f"{key}_min": min(vals), f"{key}_max": max(vals)})
            resource_methods.append(stats)
    if (len(summaries), len(stage_rows), len(sensitivity), len(labels), len(paired)) != (28,980,840,9800,1400):
        raise RuntimeError("Incomplete report row counts")
    outputs = {"per_case.csv": flat+references, "summary.csv": summaries+reference_summaries,
               "paired_differences.csv": paired, "per_label.csv": labels, "stage_diagnostics.csv": stage_rows,
               "integration_sensitivity.csv": sensitivity, "resource_by_method.csv": resource_methods,
               "rank0_pilot_reproducibility.csv": pilot, "dice_first.csv": selected,
               "pareto.csv": pareto(summaries)}
    for name, values in outputs.items(): c.write_csv(out/name, values)
    payload = {"complete": True, "variant": s.VARIANT, "candidates": rows, "per_case": flat+references,
               "summary": summaries+reference_summaries, "paired_differences": paired, "per_label": labels,
               "stage_diagnostics": stage_rows, "integration_sensitivity": sensitivity,
               "resource_by_method": resource_methods, "dice_first": selected, "pareto": pareto(summaries),
               "rank0_pilot_reproducibility": pilot, "provenance": manifest}
    (out/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    text = ["# First10 refined B-spline SVF lambda sweep", "",
            "Exploratory seed-3 shuffled ranks0–9.280 fresh registrations; each lambda applies at every active stage. Primary integration: seven native-resolution scaling/squaring steps. No segmentation checkpoint selection.", "",
            "All ± values are across-case sample SD (n=10). Foreground is fixed raw image>0; SD(log J) uses positive Jacobians and folding counts J≤0. Velocity/displacement lambda values do not impose equal-strength constraints.", ""]
    for schedule in ("842", "8421"):
        for reg in ("velocity", "displacement"):
            text += [f"## {family(schedule,reg)}", "",
                     "| Setting | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % | Optimizer s | Worker s | Peak allocated GiB | Peak reserved GiB |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in [x for x in summaries if x["schedule"] == schedule and x["regularize"] == reg]+reference_summaries:
                label = f"λ={row['alpha']:g}" if row.get("alpha") is not None else row["method"]
                text.append("| "+label+" | "+" | ".join(fmt(row,k,6 if k in METRICS else 2) for k in
                            (*METRICS,"optimizer_sec","worker_sec","peak_cuda_gib","peak_reserved_gib"))+" |")
            text += [""]
    text += ["## Timing and memory interpretation", "",
             "One new worker/audit per GPU0–3, with other workloads untouched. Optimizer time sums actual stage optimization; worker time includes loading, exports, metrics and integration diagnostics, but excludes separate independent replay. Audit time is separately recorded. Timings are shared-load diagnostics, not isolated benchmarks.", "",
             "Memory is the worker process PyTorch peak allocated/reserved memory, not whole-device usage or other jobs. Stage and independent-audit peaks are separate. summary.csv provides mean/sampleSD/min/max for every resource metric per setting/lambda; resource_by_method.csv additionally pools all70 runs of each method. GPU snapshots and the minimum-free-memory admission gate are recorded in the run artifacts. DL timing/memory is n/a.", "",
             "## Resource ranges by setting and lambda", "",
             "Ranges below are across the ten cases. Peak memory maxima are useful for capacity planning; keep admission headroom for CUDA context and concurrent allocations.", "",
             "| Method | λ | Optimizer s, mean [min,max] | Worker s, mean [min,max] | Audit s, mean | Max allocated GiB | Max reserved GiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        ranges = [f"{row[k+'_mean']:.2f} [{row[k+'_min']:.2f}, {row[k+'_max']:.2f}]" for k in ("optimizer_sec","worker_sec")]
        text.append(f"| {row['method']} | {row['alpha']:g} | {' | '.join(ranges)} | {row['audit_sec_mean']:.2f} | {row['peak_cuda_gib_max']:.3f} | {row['peak_reserved_gib_max']:.3f} |")
    text += ["", "## Selection and diagnostics", "",
             "Dice-first choices use mean Dice, lower ASD then HD95. pareto.csv is nondominated across all five accuracy/regularity means; it does not collapse folding into SD(log J). No boundary-driven automatic search expansion.", ""]
    for row in selected:
        text.append(f"- {row['method']}: Dice-first λ={row['alpha']:g}, Dice={row['dice_mean']:.6f}, folding={row['jdet_le0_pct_fg_mean']:.6f}%.")
    text += ["", "Seven-step fields remain primary. integration_sensitivity.csv contains8/10-step changes and forward/reverse inverse residuals without optimization. Initial scaled-map derivative-bound violations and measured folding are reported outcomes, not discarded cases. The rank0 appendix reports fresh-versus-pilot differences without demanding bitwise CUDA reproducibility.", "",
             "![Accuracy–regularity sweeps](arc_lines.png)", ""]
    (out/"report.md").write_text("\n".join(text))
    plot(summaries, reference_summaries, out)
    # Independently read back tabular outputs before creating acceptance.
    with (out/"summary.csv").open() as stream: saved = list(csv.DictReader(stream))
    if len(saved) != 30: raise RuntimeError("Missing summary rows")
    for a,b in zip(saved,summaries+reference_summaries):
        for key in METRICS:
            if abs(float(a[f"{key}_mean"])-b[f"{key}_mean"]) > 1e-12: raise RuntimeError("CSV/JSON metric mismatch")
    names = [*outputs,"results.json","report.md","arc_lines.png","arc_lines.svg","arc_lines.pdf"]
    accepted = {"passed": True, "candidate_count": 280, "independent_replay_count": 280,
                "stage_count": 980, "integration_evaluations": 840,
                "output_sha256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in names},
                "completed_at": c.now_text()}
    (out/"ACCEPTED.json").write_text(json.dumps(accepted, indent=2)+"\n")


def plot(summaries, refs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    families = [(schedule,reg) for schedule in ("842","8421") for reg in ("velocity","displacement")]
    for row,(xkey,xlabel) in enumerate((("log_jdet_std_fg","FG SD(log J), positive J"),("jdet_le0_pct_fg","FG folding (%)"))):
        for col,(ykey,ylabel) in enumerate((("dice","Dice ↑"),("hd95","HD95 (mm) ↓"),("asd","ASD (mm) ↓"))):
            ax = axes[row,col]
            for i,(schedule,reg) in enumerate(families):
                points = sorted([x for x in summaries if x["schedule"] == schedule and x["regularize"] == reg],key=lambda x:x["alpha"])
                ax.plot([x[f"{xkey}_mean"] for x in points],[x[f"{ykey}_mean"] for x in points],"-o",color=f"C{i}",label=family(schedule,reg),markersize=4)
                for x in points: ax.annotate(f"{x['alpha']:g}",(x[f"{xkey}_mean"],x[f"{ykey}_mean"]),xytext=(3,3),textcoords="offset points",fontsize=7,color=f"C{i}")
            for i,ref in enumerate(refs):
                ax.scatter(ref[f"{xkey}_mean"],ref[f"{ykey}_mean"],marker=("*","X")[i],s=90,color=("black","gray")[i],label=ref["method"],zorder=5)
            ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,bbox_to_anchor=(.5,.005))
    fig.suptitle("First10 exploratory refined-SVF sweeps · point labels are λ · means, no ellipses")
    fig.tight_layout(rect=(0,.09,1,.96))
    for suffix in ("png","svg","pdf"): fig.savefig(out/f"arc_lines.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
