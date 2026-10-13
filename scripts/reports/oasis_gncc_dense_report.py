"""Three-NCC tables and paired2x3 ARC panels; accepted references are read-only."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path(__file__).resolve().parents[1]/"experiments/oasis_alpha_sweep"
sys.path.insert(0, str(SWEEP))
import common as c
import gncc_dense as d
import oasis_dl_loss_first10_arc as previous
import oasis_dl_ncc_diffusion_first10_arc as old
from run_oasis_saved_field_bspline_fit_one import sha256_file

NCC_KEYS = (*d.LOSSES, "lncc")
LABELS = {"gncc": "Global NCC: 1−ρ", "gncc2": "Global NCC²: 1−ρ²", "lncc": "Local NCC9: 1−mean(ρ²)",
          "VFA": "VFA", "SITReg": "SITReg", "mse": "MSE", "mind": "MIND", "rc": "RC", "ngf": "NGF", "femse": "Feature-MSE"}
COLORS = {"gncc": "#D55E00", "gncc2": "#009E73", "lncc": "#222222", "VFA": "#0072B2", "SITReg": "#7A3DB8",
          "mse": "#E69F00", "mind": "#56B4E9", "rc": "#CC79A7", "ngf": "#8C564B", "femse": "#7F7F7F"}


def load_references(baseline_root, ncc_root):
    rows, hashes = previous.load_references(baseline_root, ncc_root)
    for row in rows:
        if row["loss"] == "ncc":
            row["loss"] = "lncc"
    return rows, hashes


def flat(row):
    result = {key: row[key] for key in (*d.METRICS, "case_rank", "fixed_id", "moving_id", "direction",
              "gpu", "effective_lambda", "training_elapsed_sec", "total_elapsed_sec", "best_epoch", "best_loss",
              "valid_label_count", "run_dir", "dvf_sha256")}
    result.update(loss=row["spec"]["loss"], similarity_scale=1., actual_regularizer_coefficient=row["effective_lambda"],
                  source_kind="new_GNCC", stage_actual_epochs="/".join(str(s["actual_epochs"]) for s in row["stages"]),
                  stage_best_epochs="/".join(str(s["best_epoch"]) for s in row["stages"]),
                  stage_stopping_reasons="/".join(s["stopping_reason"] for s in row["stages"]))
    return result


def summarize(rows, expected_keys=None):
    keys = expected_keys or [(key, a) for key in NCC_KEYS for a in d.LAMBDAS]+[("VFA", None), ("SITReg", None)]
    if len(rows) != 10*len(keys):
        raise RuntimeError("Unexpected case count")
    output = []
    for key, alpha in keys:
        group = [r for r in rows if (r["loss"], r["effective_lambda"]) == (key, alpha)]
        if sorted(r["case_rank"] for r in group) != list(range(10)):
            raise RuntimeError(f"Incomplete/duplicate cohort: {key}/{alpha}")
        row = {"loss": key, "effective_lambda": alpha, "n": 10}
        for metric in d.METRICS:
            values = np.asarray([r[metric] for r in group], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite {metric}")
            row[metric+"_mean"], row[metric+"_std"] = float(values.mean()), float(values.std(ddof=1))
        output.append(row)
    return output


def plot(means, root, keys=NCC_KEYS, filename="ncc3_arc_2x3", x="log_jdet_std_fg"):
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
        for r in range(2):
            for ax, metric in zip(axes[r], ("dice", "hd95", "asd")):
                for key in (*keys, "VFA", "SITReg"):
                    group = sorted([v for v in means if v["loss"] == key], key=lambda v: v["effective_lambda"] or 0)
                    if r == 0:
                        for row in group:
                            old.ellipse(ax, row, x, metric, COLORS[key])
                    ax.plot([v[x+"_mean"] for v in group], [v[metric+"_mean"] for v in group],
                            color=COLORS[key], marker="D" if key == "VFA" else "s" if key == "SITReg" else "o",
                            markersize=6, linestyle="None" if key in ("VFA", "SITReg") else "-", label=LABELS[key])
                    for j, row in enumerate(group):
                        if row["effective_lambda"] is not None:
                            ax.annotate(f"{row['effective_lambda']:.3g}", (row[x+"_mean"], row[metric+"_mean"]),
                                        xytext=(3, 5 if j%2 else -11), textcoords="offset points", fontsize=7, color=COLORS[key])
                ax.set_xlabel("Foreground SD(log J) ↓" if x == "log_jdet_std_fg" else "Foreground folding (%) ↓")
                ax.set_ylabel({"dice": "Dice ↑", "hd95": "HD95 (mm) ↓", "asd": "Symmetric ASD (mm) ↓"}[metric])
                ax.set_title("With ±1 sample-SD ellipses" if r == 0 else "Lines only — same operating points")
                ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
                ax.margins(.1)
        for col in range(3):
            for setter, getter in (("set_xlim", "get_xlim"), ("set_ylim", "get_ylim")):
                limits = [getattr(ax, getter)() for ax in axes[:, col]]
                bound = (min(v[0] for v in limits), max(v[1] for v in limits))
                for ax in axes[:, col]:
                    getattr(ax, setter)(bound)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), fontsize=10)
        fig.text(.5, .012, "Full-resolution 8→4→2→dense; labels are training λ. All first10 are exploratory.\n"
                 "Ellipses: axis-aligned marginal ±1 across-case sample SD, not confidence regions. SD(log J) excludes nonpositive J; inspect folding.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .92))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"{filename}.{ext}", dpi=200)
        plt.close(fig)


def table(rows):
    lines = ["| Loss | λ | Dice ± SD | HD95 mm ± SD | ASD mm ± SD | FG stdlogJ ± SD | FG folding % ± SD |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [f"{row[m+'_mean']:.6f} ± {row[m+'_std']:.6f}" for m in d.METRICS[:-1]]
        alpha = "—" if row["effective_lambda"] is None else f"{row['effective_lambda']:g}"
        lines.append(f"| {LABELS[row['loss']]} | {alpha} | "+" | ".join(values)+" |")
    return lines


def write_report(results, references, root, hashes, loss_root):
    if len(results) != 140 or {r["run_name"] for r in results} != {s.name() for s in d.specs()}:
        raise RuntimeError("Require all140 successful GNCC candidates")
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Reference changed: {path}")
    rows = [flat(r) for r in results]+references
    for row in rows:
        for ref in ("VFA", "SITReg"):
            baseline = next(r for r in references if r["loss"] == ref and r["case_rank"] == row["case_rank"])
            for metric in d.METRICS[:-1]:
                row[f"delta_{ref}_{metric}"] = row[metric]-baseline[metric]
        if row["loss"] in d.LOSSES:
            baseline = next(r for r in references if r["loss"] == "lncc" and r["case_rank"] == row["case_rank"] and r["effective_lambda"] == row["effective_lambda"])
            for metric in d.METRICS[:-1]:
                row[f"delta_LNCC_same_lambda_{metric}"] = row[metric]-baseline[metric]
    means = summarize(rows)
    selections, winners = [], []
    for key in NCC_KEYS:
        group = [r for r in means if r["loss"] == key]
        best = dict(min(group, key=lambda r: (-r["dice_mean"], r["asd_mean"], r["hd95_mean"])))
        best["boundary_optimum"] = best["effective_lambda"] in (min(d.LAMBDAS), max(d.LAMBDAS))
        selections.append(best)
        for metric in ("dice", "hd95", "asd"):
            chosen = min(group, key=lambda r: (-1 if metric == "dice" else 1)*r[metric+"_mean"])
            winners.append({"loss": key, "metric": metric, "lambda": chosen["effective_lambda"]})
    labels = [label for r in results for label in c.read_json(Path(r["run_dir"])/"per_label.json")]
    if len(labels) != 4900:
        raise RuntimeError("Require4900 new per-label records")
    by_lambda = sorted([r for r in means if r["loss"] in NCC_KEYS], key=lambda r: (r["effective_lambda"], NCC_KEYS.index(r["loss"])))
    selected_table = selections+[r for r in means if r["loss"] in ("VFA", "SITReg")]
    c.write_csv(root/"case_metrics.csv", rows); c.write_json(root/"per_case.json", rows)
    c.write_csv(root/"mean_std.csv", means); c.write_csv(root/"comparison_by_lambda.csv", by_lambda)
    c.write_csv(root/"selected_comparison.csv", selected_table)
    c.write_csv(root/"per_label_per_case.csv", labels); c.write_json(root/"per_label_per_case.json", labels)
    calibration = c.read_json(root/"calibration.json")
    if not calibration["complete"]:
        raise RuntimeError("Calibration incomplete")
    c.write_json(root/"results.json", {"complete": True, "variant": d.VARIANT, "protocol": d.PROTOCOL,
                 "case_metrics": rows, "means": means, "selected": selections, "metric_winners": winners,
                 "candidates": results, "reference_hashes": hashes, "calibration": calibration,
                 "ellipse_definition": old.ELLIPSE_DEFINITION})
    lines = ["# OASIS first10: GNCC, GNCC² and LNCC9", "",
             "140 new fits;70 read-only full-resolution LNCC fits;20 original VFA/SITReg references.",
             "All first10 cases are exploratory tuning. Runtime is diagnostic under shared workloads.",
             "Both GNCC losses are unscaled. Calibration is diagnostic only; equal λ does not imply equal effective regularization.",
             "Full-resolution loss at all stages; normalized LR .03/.01/.003/.001; total voxel-field diffusion.", "",
             "## Dice-first selected comparison", "", *table(selected_table), "", "## All weights grouped by lambda", "",
             *table(by_lambda), "", "## All weights grouped by loss", "", *table(means), "",
             "## Calibration", "", "Local/global voxel-field gradient RMS ratios on common identity/cps2/dense fields:", ""]
    lines += [f"- {LABELS[k]}: median scale {v['median']:.6g}; p10–p90 {v['p10']:.6g}–{v['p90']:.6g}. Approximate LNCC-equivalent λ = raw GNCC λ × this scale; not an equivalence of objectives." for k, v in calibration["scales"].items()]
    lines += ["", "## Selection and limits", ""]+[f"- {LABELS[r['loss']]}: λ={r['effective_lambda']:g}; boundary optimum={r['boundary_optimum']}." for r in selections]
    lines += ["", old.ELLIPSE_DEFINITION+".", "Foreground stdlogJ uses positive Jacobians only; folding is reported separately.",
              "", "![Three NCC ARC comparison](ncc3_arc_2x3.png)", "", "![Folding comparison](ncc3_folding_2x3.png)"]
    plot(means, root); plot(means, root, filename="ncc3_folding_2x3", x="jdet_le0_pct_fg")
    accepted = c.read_json(loss_root/"ACCEPTED.json")
    historical = c.read_json(loss_root/"results.json")
    if not accepted.get("technical_completion") or accepted.get("candidate_count") != 350 or not historical["complete"]:
        raise RuntimeError("Eight-loss context requires accepted P44")
    extra = [r for r in historical["case_metrics"] if r["loss"] not in ("ncc", "VFA", "SITReg")]
    keys = [(k, a) for k, a in dict.fromkeys((r["loss"], r["effective_lambda"]) for r in rows+extra)]
    extended = summarize(rows+extra, keys)
    c.write_csv(root/"eight_loss_mean_std.csv", extended)
    plot(extended, root, keys=(*NCC_KEYS, "mse", "mind", "rc", "ngf", "femse"), filename="eight_loss_arc_2x3")
    lines += ["", "## Eight-loss context", "", "![Eight-loss context](eight_loss_arc_2x3.png)", "",
              "Other DL-loss rows retain their original similarity scales and effective-lambda conventions; see the source report."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    output_hashes = {p.name: sha256_file(p) for p in root.iterdir() if p.is_file() and p.name not in ("ACCEPTED.json", "DONE")}
    c.write_json(root/"ACCEPTED.json", {"technical_completion": True, "candidate_count": 140, "historical_lncc_count": 70,
                 "original_reference_count": 20, "new_label_records": 4900, "completed_at": c.now_text(),
                 "scientific_claim": "exploratory_test_case_tuning", "outputs_sha256": output_hashes})
    (root/"DONE").write_text(c.now_text()+"\n")
