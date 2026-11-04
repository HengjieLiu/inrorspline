#!/usr/bin/env python
"""SITReg-only F/G presentation revision; accepted metrics remain read-only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import subprocess
import sys

import oasis_first10_family_arc as base

PANELS = {"F": "8→4→2", "G": "8→4→2→cps1"}
SERIES = ("sitreg", "displacement", "bounded2", "svf")
STYLES = {
    "sitreg": ("SITReg", "#0072B2", "o", "-"),
    "displacement": ("Displacement B-spline", "#D55E00", "s", "--"),
    "bounded2": ("Bounded · 2 maps/level", "#333333", "D", "none"),
    "svf": ("SVF · velocity penalty", "#AA4499", "^", "-."),
}
# Shared across F/G. Tightened to the full range of means, not selected lambdas.
TOP_AXES = {
    "dice": {**base.AXES["dice"], "limits": (.62, .85), "ticks": (.65, .70, .75, .80, .85)},
    "hd95": {**base.AXES["hd95"], "limits": (1.6, 5.), "ticks": (2., 2.5, 3., 3.5, 4., 4.5, 5.), "format": "%.1f"},
    "asd": {**base.AXES["asd"], "limits": (.30, 1.12), "ticks": (.3, .5, .7, .9, 1.1)},
}


def expected_groups():
    return {(p, s, a) for p in PANELS for s in SERIES
            for a in (base.DL_LAMBDAS if s == "sitreg" else (1.,) if s == "bounded2" else base.LAMBDAS)}


def select(rows):
    selected = [{**r, "method": STYLES[r["series"]][0],
                 "alpha": float(r["alpha"]), "case_rank": int(r["case_rank"]),
                 **{m: float(r[m]) for m in base.METRICS}}
                for r in rows if r["panel"] in PANELS and r["series"] in SERIES]
    return sorted(selected, key=lambda r: (r["panel"], SERIES.index(r["series"]), r["alpha"], r["case_rank"]))


def summarize(rows, pairs):
    groups = defaultdict(list)
    for r in rows:
        rank = r["case_rank"]
        if not 0 <= rank < 10 or (r["fixed_id"], r["moving_id"]) != tuple(pairs[rank]) or r["direction"] != "moving_to_fixed":
            raise ValueError("Case identity/direction mismatch")
        groups[(r["panel"], r["series"], r["alpha"])].append(r)
    if len(rows) != 400 or set(groups) != expected_groups():
        raise ValueError("Require 40 settings / 400 displayed case records")
    summaries = []
    for (panel, series, alpha), cohort in sorted(groups.items()):
        if sorted(r["case_rank"] for r in cohort) != list(range(10)):
            raise ValueError("Incomplete/duplicate cohort")
        schedule = "842" if panel == "F" else "8421"
        origin = {"sitreg": ("A", "sitreg"), "displacement": ("B", schedule),
                  "bounded2": ("C" if panel == "F" else "D", "bounded_2_compose"),
                  "svf": ("E", schedule + "_velocity")}[series]
        if any((r["source_panel"], r["source_series"]) != origin for r in cohort):
            raise ValueError("Wrong representation/source mapping")
        result = dict(panel=panel, series=series, method=STYLES[series][0], alpha=alpha, n=10,
                      source_panel=origin[0], source_series=origin[1])
        for metric in base.METRICS:
            values = [r[metric] for r in cohort]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metrics")
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_sd"] = statistics.stdev(values)
        summaries.append(result)
    unique = {(r["source_panel"], r["source_series"], r["alpha"], r["case_rank"]) for r in rows}
    if len(unique) != 350:
        raise ValueError("Expected 350 unique source cases; SITReg repeats across F/G")
    return summaries


def check_top_extents(summaries):
    for row in summaries:
        for metric, spec in TOP_AXES.items():
            if not spec["limits"][0] < row[metric + "_mean"] < spec["limits"][1]:
                raise ValueError(f"Mean outside tightened axes: {row['panel']}/{row['series']}/{metric}")


def figure(panel, summaries):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FormatStrFormatter

    check_top_extents(summaries)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=150)
    fig.subplots_adjust(left=.083, right=.982, bottom=.15, top=.78, wspace=.29, hspace=.62)
    fig.suptitle(f"{panel}  |  SITReg and {PANELS[panel]} optimization", fontsize=24, fontweight="semibold", y=.976)
    fig.text(.5, .931, "OASIS first 10 · NCC9 + diffusion · Top: tighter y scales; bottom: full SD ranges",
             ha="center", fontsize=16)
    handles = [Line2D([], [], label=STYLES[s][0], color=STYLES[s][1], marker=STYLES[s][2],
                      linestyle=STYLES[s][3], linewidth=2.8, markersize=10 if s == "bounded2" else 7,
                      markeredgecolor="white", markeredgewidth=.7) for s in SERIES]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895),
                        ncol=4, frameon=False, fontsize=16, columnspacing=1.9, handlelength=3.2)
    fig.text(.012, .66, "Means · tighter y scale", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.012, .27, "Means + SD ellipses", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.5, .063, "λ: SITReg {0.1, 1/3, 1, 3, 10}; displacement / SVF {0.01, 0.1, 0.3, 1, 3, 10, 100}; bounded: 1 only.", ha="center", fontsize=14)
    fig.text(.5, .032, "All sweep points retained. Ellipses: ±1 sample SD (n = 10), not confidence regions. Fixed raw foreground; J > 0. Exploratory results.", ha="center", fontsize=14)
    diagnostics = []
    for i in range(2):
        for j, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[i, j]
            for series in SERIES:
                rows = sorted((r for r in summaries if r["panel"] == panel and r["series"] == series), key=lambda r: r["alpha"])
                _, color, marker, style = STYLES[series]
                if i:
                    for row in rows:
                        ax.add_patch(Ellipse((row["log_jdet_std_fg_mean"], row[metric + "_mean"]),
                                             2 * row["log_jdet_std_fg_sd"], 2 * row[metric + "_sd"],
                                             facecolor=color, edgecolor=color, linewidth=1., alpha=.12, zorder=1))
                ax.plot([r["log_jdet_std_fg_mean"] for r in rows], [r[metric + "_mean"] for r in rows],
                        color=color, marker=marker, linestyle=style, linewidth=2.8,
                        markersize=10 if series == "bounded2" else 7, markeredgecolor="white", markeredgewidth=.7,
                        zorder=5 if series == "bounded2" else 3, label=series)
            yspec = TOP_AXES[metric] if not i else base.AXES[metric]
            ax.set_xlim(base.AXES["x"]["limits"]); ax.set_xticks(base.AXES["x"]["ticks"])
            ax.set_ylim(yspec["limits"]); ax.set_yticks(yspec["ticks"])
            ax.xaxis.set_major_formatter(FormatStrFormatter(base.AXES["x"]["format"]))
            ax.yaxis.set_major_formatter(FormatStrFormatter(yspec["format"]))
            ax.set_xlabel(base.AXES["x"]["label"], fontsize=19, labelpad=9)
            ax.set_ylabel(yspec["label"], fontsize=19, labelpad=10)
            if not i:
                ax.set_title(("Dice", "HD95", "Symmetric ASD")[j], fontsize=21, pad=14)
            ax.tick_params(axis="both", labelsize=16, length=5, pad=6)
            ax.set_axisbelow(True); ax.grid(color="#D8DDE3", linewidth=.7, alpha=.8)
            ax.spines[["top", "right"]].set_visible(False)
            diagnostics.append(dict(panel=panel, row=i, metric=metric,
                                    xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                                    xticks=list(ax.get_xticks()), yticks=list(ax.get_yticks()),
                                    xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                                    point_count=sum(len(line.get_xdata()) for line in ax.lines),
                                    ellipse_count=len(ax.patches)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes.flat:
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    for artist in [*texts, legend]:
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError("Text/legend outside canvas")
    if legend.get_window_extent(renderer).y0 < max(ax.get_tightbbox(renderer).y1 for ax in axes[0]):
        raise ValueError("Legend overlaps axes")
    for top, bottom in zip(axes[0], axes[1]):
        for a, b in zip(top.lines, bottom.lines):
            if list(a.get_xdata()) != list(b.get_xdata()) or list(a.get_ydata()) != list(b.get_ydata()):
                raise ValueError("Top/bottom mean mismatch")
        if top.patches or len(bottom.patches) != 20:
            raise ValueError("Wrong ellipse count")
    return fig, diagnostics


def run(args):
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing release")
    manifest = base.read_json(args.source / "manifest.json")
    accepted = base.read_json(args.source / "ACCEPTED.json")
    if not accepted.get("complete"):
        raise ValueError("Unaccepted F/G source")
    hashes = {**manifest["source_sha256"],
              **{str(args.source / name): digest for name, digest in accepted["output_sha256"].items()},
              str(args.source / "ACCEPTED.json"): base.sha(args.source / "ACCEPTED.json")}
    for path, digest in hashes.items():
        if base.sha(path) != digest:
            raise ValueError(f"Source hash mismatch: {path}")
    rows = select(base.read_csv(args.source / "per_case.csv"))
    summaries = summarize(rows, manifest["pairs"])
    old = {(r["panel"], r["series"], float(r["alpha"])): r for r in base.read_csv(args.source / "summary.csv")}
    for row in summaries:
        for metric in base.METRICS:
            for suffix in ("mean", "sd"):
                k = metric + "_" + suffix
                if abs(row[k] - float(old[(row["panel"], row["series"], row["alpha"])][k])) > 1e-12:
                    raise ValueError("Original/recomputed statistic mismatch")
    extents = base.check_extents(summaries)
    check_top_extents(summaries)
    args.out.mkdir(parents=True, exist_ok=False)
    base.write_csv(args.out / "summary.csv", summaries)
    base.write_csv(args.out / "per_case.csv", rows)
    base.write_json(args.out / "results.json", {"summary": summaries, "per_case": rows, "pairs": manifest["pairs"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    records = []
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}), \
            PdfPages(args.out / "arc_panels_F_to_G.pdf") as book:
        for panel in PANELS:
            fig, diagnostics = figure(panel, summaries)
            records.extend(diagnostics)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(args.out / f"panel_{panel}.{ext}", dpi=200)
            book.savefig(fig); plt.close(fig)
            print(f"Rendered {panel}: 20 points, 4 methods; tighter top y axes, full SD bottom", flush=True)
    for path, digest in hashes.items():
        if base.sha(path) != digest:
            raise ValueError(f"Source changed during rendering: {path}")
    base.write_json(args.out / "manifest.json", {
        "source": str(args.source), "source_sha256": hashes, "script_sha256": base.sha(__file__),
        "base_script_sha256": base.sha(base.__file__), "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=base.REPO, text=True),
        "python": sys.version, "matplotlib": matplotlib.__version__, "command": sys.argv,
        "axes": base.AXES, "top_axes": TOP_AXES, "font_sizes_pt": base.FONT, "rendered_axes": records,
        "sd_extents": extents, "styles": STYLES, "pairs": manifest["pairs"], "points_per_panel": 20,
        "displayed_case_records": 400, "unique_source_case_records": 350, "all_means_visible": True,
        "source_files_unchanged": True, "registration_or_metric_reevaluation": False,
        "visual_review": "pending", "combined_pdf": "pending host pdfunite; original A–E pages plus revised F–G"})
    (args.out / "README.md").write_text(
        "# F/G revision: SITReg and B-spline optimization\n\n"
        "[F](panel_F.png) · [G](panel_G.png) · [F–G PDF](arc_panels_F_to_G.pdf) · [A–G PDF](arc_panels_A_to_G.pdf)\n\n"
        "F: SITReg NCC9, prolonged displacement B-spline, bounded two-map composition, and refined SVF with velocity diffusion at 8→4→2. G: same families at 8→4→2→cps1, not dense refinement. VFA and FeDDA are removed from F/G only. Original A–E and old F/G remain unchanged.\n\n"
        "Top y scales are tightened to Dice 0.62–0.85, HD95 1.6–5.0 mm, ASD 0.30–1.12 mm, identically in F/G. Every operating point remains visible. Bottom y scales retain full SD ranges; x axes remain unchanged throughout. No smoothing or changes to data.\n\n"
        "SITReg: solid blue/circles; displacement: dashed vermillion/squares; SVF: dash-dot purple/triangles; bounded two-map: black diamond, lambda1 only (no sweep curve). One-row legend; historical SITReg lambda1 has uniform styling.\n\n"
        "Each figure contains 20 settings: five SITReg lambdas, seven displacement lambdas, seven SVF lambdas and one bounded-two point. There are 400 displayed case records, 350 unique source records (SITReg repeats across F/G), not new registrations. All active stages use the displayed lambda.\n\n"
        "Ellipses are axis-aligned marginal ±1 across-case sample SD (ddof1), not covariance/confidence regions. Canonical Dice, HD95 and pooled-area symmetric ASD; fixed raw-image-positive foreground, positive Jacobians for SD(log J). Folding accompanies the CSV/JSON and cannot be inferred from SD(log J). Two-map methods have larger optimizer budgets; equal lambda does not imply equal-strength regularization. These first-ten results remain exploratory.\n\n"
        "[Summary CSV](summary.csv) · [Per-case CSV](per_case.csv) · [JSON](results.json). ACCEPTED.json records final verification/visual review; manifest and RENDERED.json preserve pre-review provenance.\n", encoding="utf-8")
    base.write_json(args.out / "RENDERED.json", {"render_checks_passed": True,
                    "output_sha256": {p.name: base.sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Accepted panels_FG_v1 directory")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
