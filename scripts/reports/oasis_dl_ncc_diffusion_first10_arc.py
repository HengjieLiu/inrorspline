#!/usr/bin/env python
"""Strict complete-cohort aggregation for the six-way NCC9/diffusion sweep."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

SWEEP = Path(__file__).resolve().parents[1]/"experiments/oasis_alpha_sweep"
sys.path.insert(0, str(SWEEP))
import common as c
import dl_ncc_diffusion as d
from run_oasis_saved_field_bspline_fit_one import sha256_file

METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg", "diffusion")
LABELS = {"842": "8→4→2", "8421": "8→4→2→cps1", "842_dense": "8→4→2→dense"}
COLORS = {"842": "#D39A20", "8421": "#009E73", "842_dense": "#D55E00", "VFA": "#0072B2", "SITReg": "#8E63CE"}
ELLIPSE_DEFINITION = "Axis-aligned +/-1 across-case sample SD (ddof=1); width/height=2*SD; not a confidence region"


def load_baselines(root):
    accepted = c.read_json(root/"ACCEPTED.json")
    if not accepted.get("complete") or accepted.get("fresh_baseline_evaluations") != 20:
        raise RuntimeError("Require accepted20-case VFA/SITReg reference report")
    rows, hashes = [], {str(root/"ACCEPTED.json"): sha256_file(root/"ACCEPTED.json")}
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    for target in ("vfa", "sitreg"):
        for rank in range(10):
            path = root/"baseline_cases"/f"{target}_rank{rank:03d}.json"
            saved = c.read_json(path)
            row = dict(saved["row"])
            if row["case_rank"] != rank or (row["fixed_id"], row["moving_id"]) != pairs[rank]:
                raise RuntimeError("Baseline cohort identity mismatch")
            if row["direction"] != "moving_to_fixed":
                raise RuntimeError("Baseline direction mismatch")
            for key, filename in (("metric_code_sha256", "common.py"),
                                  ("evaluator_code_sha256", "run_oasis_vfa_feature_dense_one.py")):
                if saved[key] != sha256_file(SWEEP/filename):
                    raise RuntimeError("Canonical metric source changed since reference acceptance")
            if sha256_file(Path(row["source_path"])) != row["source_sha256"]:
                raise RuntimeError("Original reference DVF changed")
            hashes[str(path)] = sha256_file(path)
            hashes[row["source_path"]] = row["source_sha256"]
            row.update(representation=row["group"], resolution="reference", source_kind="original_saved_DL")
            rows.append(row)
    return rows, hashes


def flat_row(row):
    spec = row["spec"]
    values = {key: row[key] for key in (*METRICS, "case_rank", "fixed_id", "moving_id", "direction",
                                       "gpu", "prefix_gpu", "best_epoch", "best_loss", "valid_label_count",
                                       "training_elapsed_sec", "pipeline_training_elapsed_sec", "run_dir",
                                       "dvf_sha256", "prefix_dvf_sha256")}
    return {**values, "resolution": spec["resolution"], "representation": spec["representation"],
            "alpha": spec["alpha"], "group": f"{spec['resolution']} {LABELS[spec['representation']]} λ{spec['alpha']:g}",
            "stage_actual_epochs": "/".join(str(s["actual_epochs"]) for s in row["stages"]),
            "stage_best_epochs": "/".join(str(s["best_epoch"]) for s in row["stages"]),
            "stage_stopping_reasons": "/".join(s["stopping_reason"] for s in row["stages"]),
            "excluded_labels": ",".join(map(str, row["excluded_labels"])), "source_kind": "new_fit"}


def summarize(rows):
    keys = [(mode, rep, alpha) for mode in d.RESOLUTIONS for rep in d.REPRESENTATIONS for alpha in d.LAMBDAS]
    keys += [("reference", name, None) for name in ("VFA", "SITReg")]
    means = []
    for mode, rep, alpha in keys:
        selected = [r for r in rows if (r["resolution"], r["representation"], r["alpha"]) == (mode, rep, alpha)]
        if sorted(r["case_rank"] for r in selected) != list(range(10)):
            raise RuntimeError(f"Incomplete/duplicate cohort: {mode}/{rep}/{alpha}")
        out = {"resolution": mode, "representation": rep, "alpha": alpha, "n": 10,
               "group": selected[0]["group"]}
        for metric in METRICS:
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric}")
            out[metric+"_mean"], out[metric+"_std"] = float(values.mean()), float(values.std(ddof=1))
        means.append(out)
    if len(rows) != 440:
        raise RuntimeError("Require420 fits plus20 original references")
    return means


def ellipse(ax, row, x, y, color):
    patch = Ellipse((row[x+"_mean"], row[y+"_mean"]), width=2*row[x+"_std"], height=2*row[y+"_std"],
                    facecolor=color, edgecolor=color, alpha=.12, linewidth=1)
    ax.add_patch(patch)
    return patch


def make_plot(means, root, combined=False, folding=False):
    x = "jdet_le0_pct_fg" if folding else "log_jdet_std_fg"
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1 if combined else 2, 2, figsize=(15, 6.6 if combined else 11), squeeze=False)
        for ridx, mode in enumerate(("combined",) if combined else d.RESOLUTIONS):
            for col, (y, label) in enumerate((("dice", "Dice ↑"), ("hd95", "HD95 (mm) ↓"))):
                ax = axes[ridx, col]
                for ref in ("VFA", "SITReg"):
                    row = next(r for r in means if r["representation"] == ref)
                    ellipse(ax, row, x, y, COLORS[ref])
                    ax.scatter(row[x+"_mean"], row[y+"_mean"], c=COLORS[ref], marker="D" if ref == "VFA" else "s", s=75, zorder=5)
                for resolution in d.RESOLUTIONS if combined else (mode,):
                    for rep in d.REPRESENTATIONS:
                        rows = sorted([r for r in means if r["representation"] == rep and r["resolution"] == resolution], key=lambda r:r["alpha"])
                        color = COLORS[rep]
                        for row in rows:
                            ellipse(ax, row, x, y, color)
                        ax.plot([r[x+"_mean"] for r in rows], [r[y+"_mean"] for r in rows],
                                color=color, linestyle="-" if resolution == "pyramid" else "--", marker="o", markersize=5)
                        for i, row in enumerate(rows):
                            ax.annotate(f"{row['alpha']:g}", (row[x+"_mean"], row[y+"_mean"]),
                                        xytext=(4, 6 if i%2 else -12), textcoords="offset points", color=color, fontsize=8)
                ax.set_xlabel("Foreground J ≤ 0 (%) ↓" if folding else "Foreground SD(log J) ↓")
                ax.set_ylabel(label)
                ax.set_title("Both resolutions" if combined else ("Pyramid 4→2→1" if mode == "pyramid" else "Full-resolution loss throughout"))
                ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
                ax.margins(x=.12, y=.15)
        if not combined:
            for col in range(2):
                xs = [ax.get_xlim() for ax in axes[:, col]]
                ys = [ax.get_ylim() for ax in axes[:, col]]
                for ax in axes[:, col]:
                    ax.set_xlim(min(p[0] for p in xs), max(p[1] for p in xs))
                    ax.set_ylim(min(p[0] for p in ys), max(p[1] for p in ys))
        handles = [Line2D([], [], color=COLORS[r], marker="o", label=LABELS[r]) for r in d.REPRESENTATIONS]
        handles += [Line2D([], [], color=COLORS[r], marker="D" if r == "VFA" else "s", linestyle="none", label=r) for r in ("VFA", "SITReg")]
        if combined:
            handles += [Line2D([], [], color="gray", linestyle=style, label=mode) for mode,style in (("Pyramid","-"),("Full resolution","--"))]
        fig.suptitle("OASIS first10: DL NCC9 + diffusion", fontsize=16)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=len(handles), frameon=False)
        fig.text(.5,.035,"Labels: λ. Ellipses: ±1 across-case sample SD, not confidence intervals. Exploratory test-case tuning.",ha="center")
        fig.text(.5,.012,"SD(log J) excludes nonpositive-J foreground voxels; folding is reported separately.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.87))
        stem = "tradeoff_folding" if folding else ("arc_sdlogj_combined" if combined else "arc_sdlogj_faceted")
        for suffix in ("png","pdf","svg"):
            fig.savefig(root/f"{stem}.{suffix}", dpi=200)
        plt.close(fig)


def write_report(results, references, root, baseline_hashes):
    expected = {(rank, mode, rep, alpha) for rank in range(10) for mode in d.RESOLUTIONS
                for rep in d.REPRESENTATIONS for alpha in d.LAMBDAS}
    observed = {(r["case_rank"], r["spec"]["resolution"], r["spec"]["representation"], r["spec"]["alpha"]) for r in results}
    if len(results) != 420 or observed != expected or any(r["spec"]["smoke"] for r in results):
        raise RuntimeError("Require exactly420 successful production results")
    for path, digest in baseline_hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("Immutable reference changed during experiment")
    rows = [flat_row(r) for r in results]+references
    for row in rows:
        for ref in ("VFA", "SITReg"):
            baseline = next(r for r in references if r["representation"] == ref and r["case_rank"] == row["case_rank"])
            for metric in ("dice", "hd95", "asd"):
                row[f"delta_{ref.lower()}_{metric}"] = row[metric]-baseline[metric]
    means = summarize(rows)
    for row in means:
        for ref in ("VFA", "SITReg"):
            baseline=next(r for r in means if r["representation"]==ref)
            for metric in ("dice","hd95","asd"):
                row[f"delta_{ref.lower()}_{metric}_mean"]=row[metric+"_mean"]-baseline[metric+"_mean"]
    selections, metric_winners, pareto = [], [], []
    for mode in d.RESOLUTIONS:
        for rep in d.REPRESENTATIONS:
            group = [r for r in means if r["resolution"] == mode and r["representation"] == rep]
            selections.append(min(group, key=lambda r:(-r["dice_mean"],r["asd_mean"],r["hd95_mean"])))
            for metric in ("dice", "hd95", "asd"):
                winner = min(group, key=lambda r:(-1 if metric == "dice" else 1)*r[metric+"_mean"])
                metric_winners.append({"resolution":mode,"representation":rep,"metric":metric,"alpha":winner["alpha"]})
            for row in group:
                v=(-row["dice_mean"],row["hd95_mean"],row["asd_mean"],row["log_jdet_std_fg_mean"])
                if not any(all(a<=b for a,b in zip((-s["dice_mean"],s["hd95_mean"],s["asd_mean"],s["log_jdet_std_fg_mean"]),v)) and
                           any(a<b for a,b in zip((-s["dice_mean"],s["hd95_mean"],s["asd_mean"],s["log_jdet_std_fg_mean"]),v)) for s in group):
                    pareto.append(row)
    root.mkdir(parents=True,exist_ok=True)
    c.write_csv(root/"case_metrics.csv", rows)
    c.write_csv(root/"mean_std.csv", means)
    c.write_csv(root/"selected_configurations.csv", selections)
    c.write_json(root/"results.json", {"complete":True,"variant":d.VARIANT,"protocol":d.PROTOCOL,"case_metrics":rows,
                 "means":means,"selected":selections,"metric_winners":metric_winners,"pareto":pareto,
                 "ellipse_definition":ELLIPSE_DEFINITION,"baseline_hashes":baseline_hashes,"candidates":results})
    lines=["# OASIS first10: six-way DL NCC9/diffusion sweep", "", "All420 candidates complete. All first10 cases are exploratory test-case tuning.",
           "Timings are diagnostic under shared GPU workloads. Pipeline optimization budgets differ.", "",
           "Normalized parameters; LR0.03/0.01/0.003/0.001; raw intensities; border image sampling; voxel diffusion.",
           "Pyramid uses corrected endpoint-aligned continuous cubic evaluation. Best total loss selects every stage checkpoint.", "",
           "| Resolution | Representation | λ | Dice ± SD | HD95 ± SD | ASD ± SD | SD(log J) ± SD |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in means:
        vals=[f"{row[k+'_mean']:.6f} ± {row[k+'_std']:.6f}" for k in ("dice","hd95","asd","log_jdet_std_fg")]
        lines.append(f"| {row['resolution']} | {LABELS.get(row['representation'],row['representation'])} | {row['alpha'] if row['alpha'] is not None else '—'} | "+" | ".join(vals)+" |")
    lines += ["", "## Dice-first selected configurations", ""]
    lines += [f"- {r['resolution']} {LABELS[r['representation']]}: λ={r['alpha']:g}, Dice={r['dice_mean']:.6f}, HD95={r['hd95_mean']:.6f}, ASD={r['asd_mean']:.6f}." for r in selections]
    lines += ["", ELLIPSE_DEFINITION+".", "SD(log J) uses positive-J foreground only; inspect folding and excluded-label diagnostics alongside accuracy."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    make_plot(means,root); make_plot(means,root,combined=True); make_plot(means,root,folding=True)
    c.write_json(root/"ACCEPTED.json", {"technical_completion":True,"candidate_count":420,"configuration_count":42,
                 "reference_count":20,"scientific_claim":"exploratory_pipeline_comparison","completed_at":c.now_text()})
