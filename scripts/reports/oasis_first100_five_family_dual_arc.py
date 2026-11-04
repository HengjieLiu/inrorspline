#!/usr/bin/env python
"""Read-only 100-case, five-family ARC with linear SDlogJ and log folding ratio."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator
import numpy as np

REPO = Path(__file__).resolve().parents[2]
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
STYLES = {
    "SITReg": ("SITReg", "#332288", "D", "-"),
    "VFA": ("VFA", "#AA4499", "P", "--"),
    "displacement": ("Displacement B-spline 8→4→2", "#D55E00", "o", "-"),
    "bounded2": ("Bounded, 2 maps/level", "#0072B2", "s", (0, (5, 2))),
    "svf_velocity": ("Refined SVF / velocity", "#009E73", "^", "-."),
}
OPT_LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
DL_LAMBDAS = (.1, 1/3, 1., 3., 10.)
TOP = {"dice": (.76, .84, .02), "hd95": (1.7, 2.7, .2), "asd": (.30, .60, .05)}
Y_LABELS = {"dice": "Dice ↑", "hd95": "HD95 (mm) ↓", "asd": "Symmetric ASD (mm) ↓"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_source(root, marker, provenance):
    record = read(root/marker)
    if not record.get("complete"):
        raise ValueError("Incomplete source report")
    provenance[str(root/marker)] = sha(root/marker)
    for name, digest in record["output_sha256"].items():
        if sha(root/name) != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
        provenance[str(root/name)] = digest
    return record


def summarize(rows, pairs):
    grouped = defaultdict(list)
    for row in rows:
        rank = row["case_rank"]
        if (not 0 <= rank < 100 or (row["fixed_id"], row["moving_id"]) != pairs[rank]
                or row["direction"] != "moving_to_fixed"):
            raise ValueError("Case identity/direction mismatch")
        if any(not math.isfinite(row[k]) for k in METRICS):
            raise ValueError("Nonfinite metric")
        if not 0 <= row["jdet_le0_pct_fg"] <= 100:
            raise ValueError("Invalid folding percentage")
        grouped[row["family"], row["alpha"]].append(row)
    expected = {(family, a) for family in STYLES for a in
                (DL_LAMBDAS if family in ("SITReg", "VFA") else OPT_LAMBDAS)}
    if set(grouped) != expected:
        raise ValueError("Require full 31-setting matrix")
    result = []
    for family in STYLES:
        for key in sorted(k for k in grouped if k[0] == family):
            group = grouped[key]
            if sorted(r["case_rank"] for r in group) != list(range(100)):
                raise ValueError("Incomplete or duplicate cohort")
            out = {"family": family, "method": STYLES[family][0], "alpha": key[1], "n": 100}
            for metric in (*METRICS, "folding_ratio_fg"):
                values = [r[metric] if metric != "folding_ratio_fg" else r["jdet_le0_pct_fg"]/100 for r in group]
                out[metric+"_mean"] = statistics.mean(values)
                out[metric+"_sd"] = statistics.stdev(values)
            result.append(out)
    return result


def load_sources(opt, dl):
    provenance = {}
    accepted = verify_source(opt, "ACCEPTED.json", provenance)
    dl_accepted = verify_source(dl, "plot_manifest.json", provenance)
    if accepted["candidate_count"] != 2100 or accepted["failures"] != 0:
        raise ValueError("Need accepted 2100-case P79 report")
    if (dl_accepted["points"], dl_accepted["case_rows"], dl_accepted["cases_per_point"]) != (19, 1900, 100):
        raise ValueError("Wrong accepted DL sweep report")
    split = REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    provenance[str(split)] = sha(split)
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", split.read_text())[:100]
    if len(pairs) != 100:
        raise ValueError("Wrong pair list")
    dl_stats = {r["method"]: r for r in csv_rows(dl/"plotted_mean_std.csv")}
    rows, expected, references = [], {}, {}
    for r in csv_rows(opt/"summary.csv"):
        if r["cohort"] == "first100" and r["family"] not in ("SITReg", "VFA"):
            expected[r["family"], float(r["alpha"])] = (r, "sd")
    for r in csv_rows(opt/"per_case.csv"):
        if r["family"] in ("SITReg", "VFA"):
            references[r["family"], int(r["case_rank"])] = r
            continue
        if r["family"] not in STYLES or r["direction"] != "moving_to_fixed":
            raise ValueError("Wrong optimization representation/direction")
        rows.append({"family": r["family"], "alpha": float(r["alpha"]), "case_rank": int(r["case_rank"]),
                     "fixed_id": r["fixed_id"], "moving_id": r["moving_id"], "direction": r["direction"],
                     **{k: float(r[k]) for k in METRICS}, "source": str(opt/"per_case.csv"),
                     "source_method": r["method"], "historical_dl_lambda1": False})
    for r in csv_rows(dl/"per_case.csv"):
        stat = dl_stats[r["method"]]
        family = stat["family"]
        if family not in ("SITReg", "VFA"):
            continue  # Do not substitute the old dense MR-D-BSCP curve or FeDDA.
        alpha, rank = float(stat["alpha"]), int(r["rank"])
        if alpha == 1:
            ref = references[family, rank]
            if (r["fixed_id"], r["moving_id"]) != (ref["fixed_id"], ref["moving_id"]):
                raise ValueError("Original DL pair mismatch")
            if any(not math.isclose(float(r[k]), float(ref[k]), rel_tol=0, abs_tol=1e-12) for k in METRICS):
                raise ValueError("Original DL reference metrics differ")
        expected[family, alpha] = (stat, "std")
        rows.append({"family": family, "alpha": alpha, "case_rank": rank, "fixed_id": r["fixed_id"],
                     "moving_id": r["moving_id"], "direction": "moving_to_fixed",
                     **{k: float(r[k]) for k in METRICS}, "source": str(dl/"per_case.csv"),
                     "source_method": r["method"], "historical_dl_lambda1": alpha == 1})
    means = summarize(rows, pairs)
    for row in means:
        old, suffix = expected[row["family"], row["alpha"]]
        for metric in METRICS:
            for new, source_suffix in (("mean", "mean"), ("sd", suffix)):
                if not math.isclose(row[metric+"_"+new], float(old[metric+"_"+source_suffix]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Recomputed/source mean or sample SD mismatch")
    return rows, means, provenance


def folding_floor(means):
    values = [r["folding_ratio_fg_mean"] for r in means if r["folding_ratio_fg_mean"] > 0]
    if not values:
        raise ValueError("No positive folding values for log scale")
    return 10.**(math.floor(math.log10(min(values)))-1)


def sd_contour(x, y, sx, sy, floor):
    """Intersect an arithmetic marginal-SD ellipse with x>=floor, before log."""
    theta = np.linspace(0, 2*np.pi, 721)
    polygon = np.column_stack((x+sx*np.cos(theta), y+sy*np.sin(theta)))
    clipped, previous = [], polygon[-1]
    for current in polygon:
        a, b = previous[0] >= floor, current[0] >= floor
        if a != b:
            t = (floor-previous[0])/(current[0]-previous[0])
            clipped.append([floor, previous[1]+t*(current[1]-previous[1])])
        if b:
            clipped.append(current.tolist())
        previous = current
    return np.asarray(clipped)


def figure(means, folding=False):
    xkey = "folding_ratio_fg" if folding else "log_jdet_std_fg"
    floor = folding_floor(means) if folding else None
    xmax = max(r[xkey+"_mean"]+r[xkey+"_sd"] for r in means)
    xlim = (floor/2, 10.**math.ceil(math.log10(xmax))) if folding else (0, math.ceil((xmax+.04)*10)/10)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.subplots_adjust(left=.065, right=.982, bottom=.19, top=.79, wspace=.27, hspace=.50)
    title = "Foreground folding · log₁₀ axis" if folding else "Foreground SD(log J)"
    fig.suptitle(f"OASIS · 100 paired cases · {title}", fontsize=24, weight="semibold", y=.98)
    handles = [Line2D([], [], label=name, color=color, marker=marker, linestyle=style,
                      lw=2.7, markersize=8) for name, color, marker, style in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3,
               frameon=False, fontsize=16, columnspacing=2.2, handlelength=3.1)
    fig.text(.5, .824, "TOP: accuracy zoom · means only     |     BOTTOM: full sweep · means ± 1 sample SD",
             ha="center", fontsize=16)
    diagnostics = []
    for band in range(2):
        for column, (metric, label) in enumerate(Y_LABELS.items()):
            ax = axes[band, column]
            if folding:
                ax.set_xscale("log", base=10)
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.axvline(floor, color=".5", ls=":", lw=1.2, zorder=0)
                ax.text(floor, .98, "0†", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12, color=".35")
                ax.minorticks_off()
            else:
                ax.xaxis.set_major_locator(MultipleLocator(.3))
            for family, (_, color, marker, style) in STYLES.items():
                group = sorted((r for r in means if r["family"] == family), key=lambda r: r["alpha"], reverse=True)
                xx = [max(r[xkey+"_mean"], floor) if folding else r[xkey+"_mean"] for r in group]
                yy = [r[metric+"_mean"] for r in group]
                if band:
                    for x, r in zip(xx, group):
                        sx, sy, y = r[xkey+"_sd"], r[metric+"_sd"], r[metric+"_mean"]
                        if not folding:
                            patch = Ellipse((x, y), 2*sx, 2*sy)
                        elif sx == 0:
                            ax.vlines(x, y-sy, y+sy, color=color, alpha=.4, lw=2, zorder=1)
                            continue
                        else:
                            patch = Polygon(sd_contour(r[xkey+"_mean"], y, sx, sy, floor), closed=True)
                        patch.set(facecolor=color, edgecolor=color, alpha=.10, linewidth=1, zorder=1)
                        ax.add_patch(patch)
                ax.plot(xx, yy, color=color, marker=marker, linestyle=style, lw=2.5, markersize=7,
                        markeredgewidth=.6, markeredgecolor="white", zorder=4)
                if folding:
                    for x, y, r in zip(xx, yy, group):
                        if r[xkey+"_mean"] == 0:
                            ax.plot(x, y, marker=marker, mfc="white", mec=color, mew=1.5, ms=7, ls="none", zorder=5)
            lo, hi, step = TOP[metric]
            if band:
                lower = min(r[metric+"_mean"]-r[metric+"_sd"] for r in means)
                upper = max(r[metric+"_mean"]+r[metric+"_sd"] for r in means)
                pad = .065*(upper-lower)
                lo, hi = max(0, lower-pad), min(1, upper+pad) if metric == "dice" else upper+pad
            else:
                ax.yaxis.set_major_locator(MultipleLocator(step))
                count = sum(lo <= r[metric+"_mean"] <= hi for r in means)
                ax.text(.02, .025, f"Zoom: {count}/{len(means)} means in view", transform=ax.transAxes,
                        fontsize=11, color=".35", bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
            ax.set(xlim=xlim, ylim=(lo, hi), ylabel=label,
                   xlabel="Foreground folding ratio (log₁₀ scale) ↓" if folding else "Foreground SD(log J) ↓")
            ax.set_title(f"{'abcdef'[band*3+column]}) {label}", loc="left", fontsize=18, pad=10)
            ax.grid(which="major", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            diagnostics.append({"row": band, "metric": metric, "xlim": list(ax.get_xlim()),
                                "ylim": list(ax.get_ylim()), "scale": ax.get_xscale()})
    fig.text(.5, .124, "λ: SITReg / VFA = {0.1, 1/3, 1, 3, 10}; each B-spline family = {0.01, 0.1, 0.3, 1, 3, 10, 100}.", ha="center", fontsize=14)
    if folding:
        fig.text(.5, .089, f"† Exact-zero means: hollow markers at the display floor {floor:g}; saved values remain zero. Ratio = percentage / 100.", ha="center", fontsize=14)
        fig.text(.5, .057, "Bottom: arithmetic ±1 sample-SD contours, clipped at the display floor then log-transformed; not log-space SD or confidence regions.", ha="center", fontsize=13)
    else:
        fig.text(.5, .089, "Ellipses: marginal ±1 across-case sample SD (n = 100), not confidence or covariance regions. Bottom retains all 31 settings.", ha="center", fontsize=14)
        fig.text(.5, .057, "Foreground = fixed raw image > 0; SD(log J) uses positive Jacobians. Folding is shown separately in the companion figure.", ha="center", fontsize=13)
    fig.text(.5, .025, "Uniform DL curve styling; λ=1 uses the original accepted checkpoints. Exploratory results; equal λ does not imply equal regularization strength.", ha="center", fontsize=12, color=".35")
    return fig, diagnostics


def main(args):
    if args.out.resolve().is_relative_to(REPO):
        raise ValueError("Generated artifacts must be external")
    rows, means, provenance = load_sources(args.optimization, args.dl)
    args.out.mkdir(parents=True, exist_ok=False)
    diagnostics = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        for key, folding in (("fg_stdlogj", False), ("fg_folding_log10", True)):
            fig, diagnostics[key] = figure(means, folding)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(args.out/f"arc_{key}_2x3.{ext}", dpi=180, facecolor="white")
            plt.close(fig)
    write_csv(args.out/"per_case.csv", rows)
    write_csv(args.out/"plotted_mean_sd.csv", means)
    (args.out/"results.json").write_text(json.dumps({"per_case": rows, "summary": means}, indent=2)+"\n")
    floor = folding_floor(means)
    (args.out/"README.md").write_text(
        "# Five-family OASIS first100 ARC\n\n"
        "Two 2×3 figures: Dice, HD95, symmetric ASD; top accuracy zoom without ellipses, bottom full sweeps with sample-SD contours.\n\n"
        "31 settings × 100 identical seed-3 pairs: SITReg/VFA NCC9 sweeps (5 each) and displacement/bounded-two/refined-SVF-velocity 842 sweeps (7 each). "
        "Original accepted DL lambda1 checkpoints use the same family styling as newer sweep points. The old dense MR-D-BSCP and FeDDA curves are excluded.\n\n"
        "Means and sample SD were recomputed from 3100 per-case rows and matched against accepted source summaries (310 statistic checks). "
        "DL lambda1 rows also match all 100 P79 reference rows for each model. No registration, field-metric recomputation, or GPU work.\n\n"
        f"Folding is a ratio, not percent. Arithmetic mean/SD are divided by 100; no log-space averaging or pseudocounts. Zero means use hollow markers at {floor:g} "
        "on the log axis for display only. Arithmetic marginal-SD ellipses are clipped at this floor and transformed by the log axis, so are not visually elliptical there. "
        "They are not confidence or covariance regions. Raw values remain unchanged.\n\n"
        "Top y windows: Dice [0.76, 0.84], HD95 [1.7, 2.7] mm, ASD [0.30, 0.60] mm; out-of-window settings are retained in the bottom row and CSV. "
        "The two figures share corresponding y limits, and all panels within a figure share x limits. Fixed raw-positive foreground; positive-J SD(log J); J≤0 folding. "
        "Exploratory test data, not an untouched confirmatory benchmark.\n")
    if any(sha(path) != digest for path, digest in provenance.items()):
        raise RuntimeError("Accepted source changed during rendering")
    manifest = {"complete": True, "settings": 31, "case_rows": len(rows), "cases_per_setting": 100,
                "source_statistic_checks": 310, "source_sha256": provenance, "axes": diagnostics,
                "folding_ratio_display_floor": floor, "zero_mean_settings": [{k:r[k] for k in ("family", "alpha")} for r in means if r["folding_ratio_fg_mean"] == 0],
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "renderer_sha256": sha(__file__), "created_at": datetime.now(timezone.utc).isoformat(),
                "output_sha256": {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}}
    (args.out/"ACCEPTED.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"complete": True, "out": str(args.out), "settings": 31, "case_rows": len(rows), "floor": floor}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--dl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
