#!/usr/bin/env python
"""Complete-cohort five-loss report with immutable NCC/VFA/SITReg references."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path(__file__).resolve().parents[1]/"experiments/oasis_alpha_sweep"
sys.path.insert(0, str(SWEEP))
import common as c
import dl_loss_sweep as d
import oasis_dl_ncc_diffusion_first10_arc as old
from run_oasis_saved_field_bspline_fit_one import sha256_file

METRICS = old.METRICS
COLORS = dict(mse="#D55E00", mind="#009E73", rc="#CC79A7", ngf="#E69F00",
              femse="#56B4E9", ncc="#222222", VFA="#0072B2", SITReg="#7A3DB8")


def load_references(baseline_root, ncc_root):
    rows, hashes = old.load_baselines(baseline_root)
    for row in rows:
        row.update(loss=row["representation"], effective_lambda=None, similarity_scale=None,
                   actual_regularizer_coefficient=None, source_kind="original_DL")
    path = ncc_root/"results.json"
    accepted = c.read_json(ncc_root/"ACCEPTED.json")
    previous = c.read_json(path)
    if not accepted.get("technical_completion") or accepted.get("candidate_count") != 420 or not previous.get("complete"):
        raise RuntimeError("Require complete accepted P43 NCC sweep")
    hashes[str(path)] = sha256_file(path)
    hashes[str(ncc_root/"ACCEPTED.json")] = sha256_file(ncc_root/"ACCEPTED.json")
    import launch_oasis_dl_ncc_diffusion_sweep as old_launch
    from run_oasis_dl_ncc_diffusion_one import source_hashes as old_hashes
    import dl_ncc_diffusion as ncc
    expected_hashes = old_hashes()
    count = 0
    for candidate in previous["candidates"]:
        spec = candidate["spec"]
        if spec["resolution"] != "full" or spec["representation"] != "842_dense":
            continue
        original = old_launch.validate_result(Path(candidate["run_dir"]), ncc.FitSpec(**spec), expected_hashes)
        if original != candidate:
            raise RuntimeError("NCC report disagrees with worker summary")
        row = old.flat_row(candidate)
        row.update(loss="ncc", effective_lambda=spec["alpha"], similarity_scale=1.,
                   actual_regularizer_coefficient=spec["alpha"], source_kind="read_only_P43_full_dense")
        rows.append(row); count += 1
        for key in ("model_path", "dvf_path"):
            hashes[candidate[key]] = candidate[key.replace("_path", "_sha256")]
        for name in ("summary.json", "DONE"):
            artifact = Path(candidate["run_dir"])/name
            hashes[str(artifact)] = sha256_file(artifact)
    if count != 70:
        raise RuntimeError("Require 70 historical full-resolution dense NCC outputs")
    return rows, hashes


def flat(row):
    keys = (*METRICS, "case_rank", "fixed_id", "moving_id", "direction", "gpu", "best_epoch", "best_loss",
            "effective_lambda", "similarity_scale", "actual_regularizer_coefficient", "training_elapsed_sec",
            "total_elapsed_sec", "valid_label_count", "run_dir", "dvf_sha256")
    return {**{k: row[k] for k in keys}, "loss": row["spec"]["loss"], "multiplier": row["spec"]["multiplier"],
            "source_kind": "new_fit", "group": f"{row['spec']['loss']} λ{row['effective_lambda']:.12g}",
            "stage_actual_epochs": "/".join(str(s["actual_epochs"]) for s in row["stages"]),
            "stage_best_epochs": "/".join(str(s["best_epoch"]) for s in row["stages"]),
            "stage_stopping_reasons": "/".join(s["stopping_reason"] for s in row["stages"]),
            "excluded_labels": ",".join(map(str, row["excluded_labels"]))}


def summarize(rows):
    keys = [(loss, d.CENTERS[loss]*m) for loss in d.LOSSES for m in d.MULTIPLIERS]
    keys += [("ncc", a) for a in (.01, .1, .3, 1., 3., 10., 100.)]+[("VFA", None), ("SITReg", None)]
    means = []
    if len(rows) != 440:
        raise RuntimeError("Require 350 new, 70 NCC and 20 original DL rows")
    for loss, alpha in keys:
        group = [r for r in rows if (r["loss"], r["effective_lambda"]) == (loss, alpha)]
        if sorted(r["case_rank"] for r in group) != list(range(10)):
            raise RuntimeError(f"Incomplete/duplicate cohort {loss}/{alpha}")
        row = {"loss": loss, "effective_lambda": alpha, "n": 10,
               "similarity_scale": group[0]["similarity_scale"],
               "actual_regularizer_coefficient": group[0]["actual_regularizer_coefficient"]}
        for metric in METRICS:
            values = np.array([r[metric] for r in group], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric}")
            row[metric+"_mean"], row[metric+"_std"] = float(values.mean()), float(values.std(ddof=1))
        means.append(row)
    return means


def plot(means, root, folding=False):
    x = "jdet_le0_pct_fg" if folding else "log_jdet_std_fg"
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ridx, mode in enumerate(("all", "selected")):
        for ax, metric in zip(axes[ridx], ("dice", "hd95", "asd")):
            for loss in (*d.LOSSES, "ncc", "VFA", "SITReg"):
                group = [r for r in means if r["loss"] == loss]
                group.sort(key=lambda r: r["effective_lambda"] or 0)
                if mode == "selected":
                    group = [min(group, key=lambda r: (-r["dice_mean"], r["asd_mean"], r["hd95_mean"]))]
                for row in group:
                    old.ellipse(ax, row, x, metric, COLORS[loss])
                ax.plot([r[x+"_mean"] for r in group], [r[metric+"_mean"] for r in group],
                        color=COLORS[loss], marker="D" if loss == "VFA" else "s" if loss == "SITReg" else "o",
                        linestyle="--" if loss == "ncc" else "-", label=loss)
                for row in group:
                    if row["effective_lambda"] is not None:
                        ax.annotate(f"{row['effective_lambda']:.3g}", (row[x+"_mean"], row[metric+"_mean"]),
                                    xytext=(3, 4), textcoords="offset points", fontsize=7, color=COLORS[loss])
            ax.set_xlabel("Foreground J ≤ 0 (%)" if folding else "Foreground SD(log J)")
            ax.set_ylabel(metric+(" ↑" if metric == "dice" else " (mm) ↓"))
            ax.set_title("All weights" if mode == "all" else "Dice-first selection per loss")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=8, loc="upper center")
    fig.text(.5, .02, "Full-resolution 8→4→2→dense; labels: effective λ. Ellipses: ±1 sample SD, not confidence intervals.\n"
             "All first10 are exploratory. Positive-J-only SD(log J); inspect folding separately.", ha="center")
    fig.tight_layout(rect=(0, .07, 1, .94))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root/f"{'tradeoff_folding' if folding else 'arc_sdlogj'}.{ext}", dpi=200)
    plt.close(fig)


def write_report(results, references, root, hashes):
    expected = {s.name() for s in d.specs()}
    if len(results) != 350 or {r["run_name"] for r in results} != expected:
        raise RuntimeError("Require exactly 350 successful production registrations")
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Read-only reference changed: {path}")
    rows = [flat(r) for r in results]+references
    for row in rows:
        for ref in ("VFA", "SITReg"):
            target = next(r for r in references if r["loss"] == ref and r["case_rank"] == row["case_rank"])
            for metric in ("dice", "hd95", "asd"):
                row[f"delta_{ref.lower()}_{metric}"] = row[metric]-target[metric]
    means = summarize(rows)
    selected, winners, pareto = [], [], []
    for loss in (*d.LOSSES, "ncc"):
        group = [r for r in means if r["loss"] == loss]
        best = dict(min(group, key=lambda r: (-r["dice_mean"], r["asd_mean"], r["hd95_mean"])))
        best["boundary_optimum"] = best["effective_lambda"] in (min(r["effective_lambda"] for r in group), max(r["effective_lambda"] for r in group))
        selected.append(best)
        for metric in ("dice", "hd95", "asd"):
            winner = min(group, key=lambda r: (-1 if metric == "dice" else 1)*r[metric+"_mean"])
            winners.append({"loss": loss, "metric": metric, "effective_lambda": winner["effective_lambda"]})
        def vector(r):
            return (-r["dice_mean"], r["hd95_mean"], r["asd_mean"], r["jdet_le0_pct_fg_mean"])
        for row in group:
            if not any(all(a <= b for a, b in zip(vector(s), vector(row))) and
                       any(a < b for a, b in zip(vector(s), vector(row))) for s in group):
                pareto.append(row)
    root.mkdir(parents=True, exist_ok=True)
    c.write_csv(root/"case_metrics.csv", rows); c.write_csv(root/"mean_std.csv", means)
    c.write_csv(root/"selected_configurations.csv", selected)
    c.write_json(root/"results.json", {"complete": True, "variant": d.VARIANT, "protocol": d.PROTOCOL,
                 "case_metrics": rows, "means": means, "selected": selected, "metric_winners": winners,
                 "pareto_accuracy_folding": pareto, "reference_hashes": hashes, "candidates": results,
                 "ellipse_definition": old.ELLIPSE_DEFINITION})
    lines = ["# OASIS first10: native DL-loss 8→4→2→dense sweep", "",
             "350 new fits; 70 read-only full-resolution NCC fits; 20 original VFA/SITReg references.",
             "All first10 are exploratory tuning. Runtime is diagnostic under shared workloads.",
             "Feature-MSE: frozen FEDD-A first-level encoder on the warped image, not VFA projections.",
             "DL anchors have different architectures/budgets; they supply weights, not a controlled DL-loss comparison.", "",
             "| Loss | Effective λ | Similarity scale | Actual reg coefficient | Dice ± SD | HD95 ± SD | ASD ± SD | FG SD(log J) ± SD | FG folding % ± SD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in means:
        values = [f"{row[k+'_mean']:.6f} ± {row[k+'_std']:.6f}" for k in METRICS[:-1]]
        lines.append(f"| {row['loss']} | {row['effective_lambda']} | {row['similarity_scale']} | {row['actual_regularizer_coefficient']} | "+" | ".join(values)+" |")
    lines += ["", "## Dice-first selections", ""]
    lines += [f"- {r['loss']}: λ={r['effective_lambda']:.12g}; boundary optimum={r['boundary_optimum']}." for r in selected]
    lines += ["", old.ELLIPSE_DEFINITION+".", "Positive-J foreground only for SD(log J); nonpositive-J fraction is reported separately."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    plot(means, root); plot(means, root, folding=True)
    c.write_json(root/"ACCEPTED.json", {"technical_completion": True, "candidate_count": 350,
                 "configuration_count": 35, "historical_ncc_count": 70, "original_reference_count": 20,
                 "completed_at": c.now_text(), "scientific_claim": "exploratory_test_case_tuning"})
