#!/usr/bin/env python
"""Render the accepted SITReg NCC comparison, lines top and SD ellipses bottom.

Presentation-only export: never modify hash-locked experiment/report sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO = Path(__file__).resolve().parents[2]
X = "log_jdet_std_fg"
METRICS = ("dice", "hd95", "asd", X, "jdet_le0_pct_fg")
Y = (("dice", "Dice ↑"), ("hd95", "HD95 (mm) ↓"),
     ("asd", "Symmetric ASD (mm) ↓"))
SERIES = {"gncc": [("3", 3.), ("1", 1.), ("1over3", 1/3), ("0.1", .1)],
          "lncc9": [("10", 10.), ("3", 3.), ("1", 1.), ("1over3", 1/3), ("0.1", .1)]}
COLORS = {"gncc": "#D55E00", "lncc9": "#0072B2"}
STEM = "sitreg_gncc_vs_lncc_arc_lines_top_2x3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validated_means(data):
    """Reconstruct all nine means/sample SDs from the same 100 case pairs."""
    if data.get("complete") is not True:
        raise ValueError("Require complete experiment report")
    expected = {(loss, key) for loss, group in SERIES.items() for key, _ in group}
    rows, cases = data["means"], data["per_case"]
    means = {(r["loss"], r["lambda_key"]): r for r in rows}
    if len(rows) != 9 or set(means) != expected or len(cases) != 900:
        raise ValueError("Require all nine settings and 900 complete case rows")
    if {(r["loss"], r["lambda_key"]) for r in cases} != expected:
        raise ValueError("Unexpected per-case setting")
    pair_ids = None
    for loss, group in SERIES.items():
        for key, alpha in group:
            row = means[loss, key]
            historical = loss == "lncc9" and key == "1"
            if row["n"] != 100 or row["alpha"] != alpha or row["historical"] != historical:
                raise ValueError("Wrong cohort size, lambda or historical tag")
            subset = sorted((r for r in cases if (r["loss"], r["lambda_key"]) == (loss, key)),
                            key=lambda r: r["rank"])
            if [r["rank"] for r in subset] != list(range(100)):
                raise ValueError("Missing or duplicate case ranks")
            current_ids = [(r["fixed_id"], r["moving_id"]) for r in subset]
            if pair_ids is None:
                pair_ids = current_ids
            elif current_ids != pair_ids:
                raise ValueError("Mismatched case identity or direction")
            for metric in METRICS:
                values = [r[metric] for r in subset]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite case metric")
                for suffix, value in (("mean", statistics.mean(values)),
                                      ("std", statistics.stdev(values))):
                    if not math.isclose(value, row[metric+"_"+suffix], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Saved statistics do not match per-case metrics")
    return means


def build_figure(means):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey="col")
    for row_index in range(2):
        for col, (metric, title) in enumerate(Y):
            ax = axes[row_index, col]
            for loss, group in SERIES.items():
                color = COLORS[loss]
                for (left, _), (right, _) in zip(group, group[1:]):
                    historic_segment = loss == "lncc9" and "1" in (left, right)
                    ax.plot([means[loss, k][X+"_mean"] for k in (left, right)],
                            [means[loss, k][metric+"_mean"] for k in (left, right)],
                            color=color, lw=2.2, ls="--" if historic_segment else "-", zorder=3)
                for key, alpha in group:
                    point = means[loss, key]
                    x, y = point[X+"_mean"], point[metric+"_mean"]
                    historical = point["historical"]
                    if row_index == 1:
                        ax.add_patch(Ellipse((x, y), 2*point[X+"_std"], 2*point[metric+"_std"],
                                             facecolor=color, edgecolor=color, alpha=.13, lw=1.1, zorder=1))
                    ax.plot(x, y, marker="D" if historical else "o" if loss == "gncc" else "s",
                            color=color, markerfacecolor="white" if historical else color,
                            ms=6.5, markeredgewidth=1.4, ls="none", zorder=4)
                    label = "1/3" if key == "1over3" else key
                    offset = (6, 9) if loss == "gncc" else (6, -16)
                    ax.annotate("λ="+label+("*" if historical else ""), (x, y), xytext=offset,
                                textcoords="offset points", fontsize=8.5, color=color, zorder=5)
            ax.set_title(f"{'abcdef'[row_index*3+col]}) {title}", loc="left", fontsize=13,
                         fontweight="semibold")
            ax.set_xlabel("Foreground SD(log J) ↓", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.grid(alpha=.18, lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelbottom=True)
            for axis, field in (("x", X), ("y", metric)):
                low = min(r[field+"_mean"]-r[field+"_std"] for r in means.values())
                high = max(r[field+"_mean"]+r[field+"_std"] for r in means.values())
                margin = max(high-low, 1e-6)*.11
                getattr(ax, "set_"+axis+"lim")(max(0, low-margin), high+margin)
    legend = [Line2D([], [], color=COLORS["gncc"], marker="o", lw=2.2, label="SITReg · 1−GNCC"),
              Line2D([], [], color=COLORS["lncc9"], marker="s", lw=2.2,
                     label="SITReg · 1−LNCC² (9³ window)"),
              Line2D([], [], color=COLORS["lncc9"], marker="D", markerfacecolor="white", ls="--",
                     label="LNCC² λ=1* · historical checkpoint")]
    fig.suptitle("SITReg · global versus local NCC · OASIS first100", fontsize=19, y=.985)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .948), ncol=3,
               frameon=False, fontsize=11)
    fig.text(.5, .892, "TOP: mean curves only", ha="center", fontsize=12, fontweight="semibold")
    fig.text(.5, .486, "BOTTOM: same curves with ±1 across-case SD ellipses", ha="center",
             fontsize=12, fontweight="semibold")
    fig.text(.5, .044, "100 paired cases per point · Ellipses: marginal sample SD (ddof=1), not confidence regions or covariance ellipses.",
             ha="center", fontsize=10)
    fig.text(.5, .018, "Checkpoints selected by validation Dice. Foreground: fixed image > 0; SD(log J) excludes J ≤ 0. Folding is reported separately.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.06, right=.985, bottom=.12, top=.85, wspace=.26, hspace=.51)
    return fig, axes


def export(report_root, out_dir):
    report_root, out_dir = Path(report_root), Path(out_dir)
    if out_dir.resolve().is_relative_to(REPO) or out_dir.resolve() == report_root.resolve():
        raise ValueError("Use a distinct external artifact directory")
    if not (report_root/"DONE").is_file():
        raise ValueError("Wait for all four evaluations and final report acceptance")
    accepted = read(report_root/"ACCEPTED.json")
    if not accepted.get("complete") or accepted.get("new_cases") != 400 or accepted.get("reference_cases") != 500:
        raise ValueError("Require accepted 400-new/500-reference comparison")
    inputs = accepted["outputs_sha256"]
    if "results.json" not in inputs:
        raise ValueError("Missing results checksum")
    for name, digest in inputs.items():
        if sha(report_root/name) != digest:
            raise ValueError("Source checksum mismatch: "+name)
    acceptance_hash = sha(report_root/"ACCEPTED.json")
    means = validated_means(read(report_root/"results.json"))
    out_dir.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, _ = build_figure(means)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir/f"{STEM}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
    rows = [means[loss, key] for loss, group in SERIES.items() for key, _ in group]
    with (out_dir/"plotted_mean_std.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if any(sha(report_root/name) != digest for name, digest in inputs.items()):
        raise ValueError("Source changed during plotting")
    manifest = dict(complete=True, cases_per_point=100, points=9, layout="2x3",
                    top="means_only", bottom="axis_aligned_plus_minus_one_sample_SD",
                    ellipse_width_height="2 * marginal across-case sample SD; ddof=1; not CI",
                    connection_order="decreasing_lambda; straight segments, no smoothing",
                    historical_lncc_lambda1="hollow diamond; dashed adjacent segments",
                    source_sha256=inputs, acceptance_sha256=acceptance_hash,
                    script_sha256=sha(__file__),
                    output_sha256={p.name: sha(p) for p in out_dir.iterdir() if p.is_file()})
    (out_dir/"plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.report_root, args.out_dir), flush=True)


if __name__ == "__main__":
    main()
