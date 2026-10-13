#!/usr/bin/env python
"""Validated 2x3 accuracy–regularity comparison, with/without SD ellipses."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import oasis_first100_with_sitreg_sweep as source

REPO = Path(__file__).resolve().parents[2]
X = "log_jdet_std_fg"
SERIES = {
    "MR-D-BSCP": [(3., "MR-D-BSCP λ=3"), (1., "MR-D-BSCP λ=1"),
                  (.3, "MR-D-BSCP λ=0.3"), (.1, "MR-D-BSCP λ=0.1")],
    "SITReg": [(10., "SITReg λ=10"), (3., "SITReg λ=3"),
               (1., "SITReg λ=1 (original)"), (1/3, "SITReg λ=1/3"), (.1, "SITReg λ=0.1")],
}
COLORS = {"MR-D-BSCP": "#D55E00", "SITReg": "#0072B2"}
Y = (("dice", "Dice ↑"), ("hd95", "HD95 (mm) ↓"), ("asd", "Symmetric ASD (mm) ↓"))


def validated_means(data):
    reproduced = source.summarize(data["per_case"])
    saved = {r["method"]: r for r in data["means"]}
    if len(saved) != 10 or len(reproduced) != 10:
        raise ValueError("Incomplete source matrix")
    for row in reproduced:
        expected = saved[row["method"]]
        for key, value in row.items():
            if key != "method" and not math.isclose(value, expected[key], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Source summary differs from case values")
    selected = {name for group in SERIES.values() for _, name in group}
    if not selected.issubset(saved):
        raise ValueError("Missing sweep setting")
    return {name: saved[name] for name in selected}


def build_figure(means):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey="col")
    for row_index in range(2):
        for col, (metric, title) in enumerate(Y):
            ax = axes[row_index, col]
            for method, group in SERIES.items():
                color = COLORS[method]
                for (_, left), (_, right) in zip(group, group[1:]):
                    historical = "original" in left or "original" in right
                    ax.plot([means[n][X+"_mean"] for n in (left, right)],
                            [means[n][metric+"_mean"] for n in (left, right)],
                            color=color, lw=2.2, linestyle="--" if historical else "-", zorder=3)
                for alpha, name in group:
                    point = means[name]
                    x, y = point[X+"_mean"], point[metric+"_mean"]
                    historical = "original" in name
                    if row_index == 1:
                        ax.add_patch(Ellipse((x, y), width=2*point[X+"_std"], height=2*point[metric+"_std"],
                                             facecolor=color, edgecolor=color, alpha=.13, lw=1.1, zorder=1))
                    ax.plot(x, y, marker="D" if historical else "o" if method == "MR-D-BSCP" else "s",
                            ms=6.5, color=color, markerfacecolor="white" if historical else color,
                            markeredgewidth=1.4, linestyle="none", zorder=4)
                    label = "1/3" if alpha == 1/3 else f"{alpha:g}"
                    # Opposite label directions separate the two nearly equal HD95 points.
                    offset = (6, 9) if method == "MR-D-BSCP" else (-6, -15)
                    if method == "SITReg" and alpha == 10:
                        offset = (6, -15)
                    ax.annotate("λ="+label+("*" if historical else ""), (x, y), xytext=offset,
                                textcoords="offset points", ha="left" if offset[0] > 0 else "right",
                                fontsize=8.5, color=color, zorder=5)
            ax.set_title(f"{'abcdef'[row_index*3+col]}) {title}", loc="left", fontsize=13, fontweight="semibold")
            ax.set_xlabel("Foreground SD(log J) ↓", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.grid(alpha=.18, lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelbottom=True)
            ax.set_xlim(.14, 1.19)
            values = [(r[metric+"_mean"]-r[metric+"_std"], r[metric+"_mean"]+r[metric+"_std"]) for r in means.values()]
            low, high = min(v[0] for v in values), max(v[1] for v in values)
            margin = (high-low)*.09
            ax.set_ylim(max(0, low-margin), high+margin)
    legend = [Line2D([], [], color=COLORS["MR-D-BSCP"], marker="o", lw=2.2,
                     label="MR-D-BSCP · pyramid 8→4→2→dense"),
              Line2D([], [], color=COLORS["SITReg"], marker="s", lw=2.2, label="SITReg"),
              Line2D([], [], color=COLORS["SITReg"], marker="D", markerfacecolor="white", ls="--",
                     label="SITReg λ=1* · historical checkpoint")]
    fig.suptitle("OASIS first100 · accuracy–regularity tradeoffs", fontsize=19, y=.985)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .948), ncol=3, frameon=False, fontsize=11)
    fig.text(.5, .892, "TOP: mean curves only", ha="center", fontsize=12, fontweight="semibold")
    fig.text(.5, .486, "BOTTOM: same curves with ±1 across-case SD ellipses", ha="center", fontsize=12, fontweight="semibold")
    fig.text(.5, .044, "100 paired cases per point · Ellipses: axis-aligned marginal sample SD (ddof=1), not confidence regions or covariance ellipses.", ha="center", fontsize=10)
    fig.text(.5, .018, "λ=1* was not retrained with the SITReg sweep; adjacent segments are dashed. SD(log J) excludes J≤0 and does not measure folding.", ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.06, right=.985, bottom=.12, top=.85, wspace=.26, hspace=.51)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.resolve().is_relative_to(REPO):
        raise ValueError("Generated figures must be outside Git")
    accepted = source.read(args.report_root/"ACCEPTED.json")
    if not accepted["complete"] or accepted["groups"] != 10:
        raise ValueError("Require accepted combined100-case report")
    hashes = {name: source.sha(args.report_root/name) for name in accepted["output_sha256"]}
    if hashes != accepted["output_sha256"]:
        raise ValueError("Source checksum mismatch")
    means = validated_means(source.read(args.report_root/"results.json"))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42, "svg.fonttype":"none"}):
        fig, _ = build_figure(means)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.out_dir/f"arc_mrdbscp_sitreg_first100_2x3.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
    source.write_csv(args.out_dir/"plotted_mean_std.csv", [means[n] for g in SERIES.values() for _, n in g])
    if any(source.sha(args.report_root/name) != value for name, value in hashes.items()):
        raise ValueError("Source changed during plotting")
    source.write_json(args.out_dir/"plot_manifest.json", {
        "complete":True, "cases_per_point":100, "points":9, "layout":"2x3",
        "top":"means_only", "bottom":"axis_aligned_plus_minus_one_sample_SD",
        "ellipse_width_height":"2 * marginal across-case sample SD; ddof=1; not CI",
        "connection_order":"decreasing_lambda; straight segments, no smoothing",
        "historical_sitreg_lambda1":"hollow diamond, dashed adjacent segments",
        "source_sha256":hashes, "script_sha256":source.sha(Path(__file__)),
        "output_sha256":{p.name:source.sha(p) for p in args.out_dir.iterdir() if p.is_file()}})
    print(args.out_dir, flush=True)


if __name__ == "__main__":
    main()
