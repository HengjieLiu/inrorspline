#!/usr/bin/env python
"""P75 F/G extension: accepted saved metrics only, no numerical experiments."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import oasis_first10_family_arc as base

PANELS = {"F": "8→4→2", "G": "8→4→2→1"}
SERIES = ("sitreg", "vfa", "fedda", "displacement", "bounded2", "svf")
STYLES = {
    **{s: base.STYLES[s] for s in SERIES[:3]},
    "displacement": ("Displacement B-spline", "#AA4499", "D", "--"),
    "bounded2": ("Bounded · 2 maps/level", "#333333", "P", "none"),
    "svf": ("SVF · velocity penalty", "#B58900", "v", "-."),
}


def mapping(panel):
    schedule = "842" if panel == "F" else "8421"
    return {**{("A", s): s for s in SERIES[:3]},
            ("B", schedule): "displacement",
            ("C" if panel == "F" else "D", "bounded_2_compose"): "bounded2",
            ("E", schedule + "_velocity"): "svf"}


def expected_groups():
    return {(p, s, alpha) for p in PANELS for s in SERIES
            for alpha in (base.DL_LAMBDAS if s in SERIES[:3] else (1.,) if s == "bounded2" else base.LAMBDAS)}


def select(rows):
    result = []
    for panel in PANELS:
        wanted = mapping(panel)
        for row in rows:
            key = row["panel"], row["series"]
            if key not in wanted:
                continue
            series = wanted[key]
            if series == "bounded2" and float(row["alpha"]) != 1.:
                raise ValueError("Bounded two-map operating point must be lambda1")
            result.append({**row, "source_panel": row["panel"], "source_series": row["series"],
                           "panel": panel, "series": series, "method": STYLES[series][0],
                           "alpha": float(row["alpha"]), "case_rank": int(row["case_rank"]),
                           **{m: float(row[m]) for m in base.METRICS}})
    return sorted(result, key=lambda r: (r["panel"], SERIES.index(r["series"]), r["alpha"], r["case_rank"]))


def summarize(rows, pairs):
    grouped = defaultdict(list)
    for row in rows:
        if (row["fixed_id"], row["moving_id"]) != tuple(pairs[row["case_rank"]]) or row["direction"] != "moving_to_fixed":
            raise ValueError("Case identity/direction mismatch")
        grouped[(row["panel"], row["series"], row["alpha"])].append(row)
    if set(grouped) != expected_groups() or len(rows) != 600:
        raise ValueError("Require exactly 60 plotted settings/600 displayed case records")
    summaries = []
    for (panel, series, alpha), cohort in sorted(grouped.items()):
        if sorted(r["case_rank"] for r in cohort) != list(range(10)):
            raise ValueError("Incomplete/duplicate cohort")
        row = dict(panel=panel, series=series, method=STYLES[series][0], alpha=alpha, n=10,
                   source_panel=cohort[0]["source_panel"], source_series=cohort[0]["source_series"])
        for metric in base.METRICS:
            values = [r[metric] for r in cohort]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metrics")
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values)
        summaries.append(row)
    unique = {(r["source_panel"], r["source_series"], r["alpha"], r["case_rank"]) for r in rows}
    if len(unique) != 450:
        raise ValueError("Expected 450 unique source case records; DL is repeated across F/G")
    return summaries


def figure(panel, summaries):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FormatStrFormatter

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=150)
    fig.subplots_adjust(left=.083, right=.982, bottom=.15, top=.78, wspace=.29, hspace=.62)
    fig.suptitle(f"{panel}  |  DL and {PANELS[panel]} optimization", fontsize=24, fontweight="semibold", y=.976)
    fig.text(.5, .931, "OASIS first 10 · NCC9 + diffusion · B-spline displacement, bounded composition and velocity-regularized SVF",
             ha="center", fontsize=16)
    handles = [Line2D([], [], label=STYLES[s][0], color=STYLES[s][1], marker=STYLES[s][2],
                      linestyle=STYLES[s][3], linewidth=2.5, markersize=7) for s in SERIES]
    # Matplotlib fills legend columns first; permute for the requested row order.
    legend = fig.legend(handles=[handles[i] for i in (0, 3, 1, 4, 2, 5)], loc="upper center",
                        bbox_to_anchor=(.5, .904), ncol=3, frameon=False,
                        fontsize=16, columnspacing=2., handlelength=2.6)
    fig.text(.012, .66, "Means only", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.012, .27, "Means + SD ellipses", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.5, .063, "λ: DL {0.1, 1/3, 1, 3, 10}; displacement / SVF {0.01, 0.1, 0.3, 1, 3, 10, 100}; bounded: 1 only.", ha="center", fontsize=14)
    fig.text(.5, .032, "Ellipses: ±1 sample SD (n = 10), not confidence regions. Fixed raw foreground; J > 0 for SD(log J). Exploratory first-ten results.", ha="center", fontsize=14)
    diagnostics = []
    for i in range(2):
        for j, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[i, j]
            for series in SERIES:
                rows = sorted((r for r in summaries if r["panel"] == panel and r["series"] == series), key=lambda r: r["alpha"])
                _, color, marker, style = STYLES[series]
                if i:
                    for row in rows:
                        ax.add_patch(Ellipse((row["log_jdet_std_fg_mean"], row[metric+"_mean"]),
                                             2*row["log_jdet_std_fg_sd"], 2*row[metric+"_sd"],
                                             facecolor=color, edgecolor=color, linewidth=1., alpha=.16, zorder=1))
                ax.plot([r["log_jdet_std_fg_mean"] for r in rows], [r[metric+"_mean"] for r in rows],
                        color=color, marker=marker, linestyle=style, linewidth=2.5,
                        markersize=9 if series == "bounded2" else 7, markeredgecolor="white", markeredgewidth=.65,
                        zorder=5 if series == "bounded2" else 3, label=series)
            ax.set_xlim(base.AXES["x"]["limits"]); ax.set_xticks(base.AXES["x"]["ticks"])
            ax.set_ylim(base.AXES[metric]["limits"]); ax.set_yticks(base.AXES[metric]["ticks"])
            ax.xaxis.set_major_formatter(FormatStrFormatter(base.AXES["x"]["format"]))
            ax.yaxis.set_major_formatter(FormatStrFormatter(base.AXES[metric]["format"]))
            ax.set_xlabel(base.AXES["x"]["label"], fontsize=19, labelpad=9)
            ax.set_ylabel(base.AXES[metric]["label"], fontsize=19, labelpad=10)
            if not i: ax.set_title(("Dice", "HD95", "Symmetric ASD")[j], fontsize=21, pad=14)
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
        if top.patches or len(bottom.patches) != 30:
            raise ValueError("Wrong ellipse count")
    return fig, diagnostics


def run(args):
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing release")
    manifest = base.read_json(args.source / "manifest.json")
    accepted = base.read_json(args.source / "ACCEPTED.json")
    if not accepted.get("complete"):
        raise ValueError("Unaccepted source")
    hashes = {str(args.source / name): digest for name, digest in accepted["output_sha256"].items()}
    hashes[str(args.source / "ACCEPTED.json")] = base.sha(args.source / "ACCEPTED.json")
    table = args.source / "lambda1_table_v1"
    for path in sorted(table.iterdir()):
        if path.is_file(): hashes[str(path)] = base.sha(path)
    for path, digest in hashes.items():
        if base.sha(path) != digest: raise ValueError(f"Source hash mismatch: {path}")
    source_rows = base.read_csv(args.source / "per_case.csv")
    rows = select(source_rows)
    summary = summarize(rows, manifest["pairs"])
    old_summary = {(r["panel"], r["series"], float(r["alpha"])): r for r in base.read_csv(args.source / "summary.csv")}
    for row in summary:
        original = old_summary[(row["source_panel"], row["source_series"], row["alpha"])]
        for metric in base.METRICS:
            for suffix in ("mean", "sd"):
                if abs(row[metric+"_"+suffix]-float(original[metric+"_"+suffix])) > 1e-12:
                    raise ValueError("Source/recomputed summary mismatch")
    extents = base.check_extents(summary)
    args.out.mkdir(parents=True, exist_ok=False)
    base.write_csv(args.out / "summary.csv", summary)
    base.write_csv(args.out / "per_case.csv", rows)
    base.write_json(args.out / "results.json", {"summary": summary, "per_case": rows, "pairs": manifest["pairs"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    records = []
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}), \
            PdfPages(args.out / "arc_panels_F_to_G.pdf") as book:
        for panel in PANELS:
            fig, diagnostics = figure(panel, summary)
            records.extend(diagnostics)
            for ext in ("png", "pdf", "svg"): fig.savefig(args.out / f"panel_{panel}.{ext}", dpi=200)
            book.savefig(fig); plt.close(fig)
            print(f"Rendered {panel}: 30 settings, 6 methods; layout/axes/ellipse checks passed", flush=True)
    for path, digest in hashes.items():
        if base.sha(path) != digest: raise ValueError(f"Source changed during rendering: {path}")
    base.write_json(args.out / "manifest.json", {
        "source": str(args.source), "source_sha256": hashes, "script_sha256": base.sha(__file__),
        "base_script_sha256": base.sha(base.__file__), "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=base.REPO, text=True),
        "python": sys.version, "matplotlib": matplotlib.__version__, "command": sys.argv,
        "axes": base.AXES, "font_sizes_pt": base.FONT, "rendered_axes": records, "sd_extents": extents,
        "styles": STYLES, "pairs": manifest["pairs"], "points_per_panel": 30,
        "displayed_case_records": 600, "unique_source_case_records": 450,
        "source_files_unchanged": True, "registration_or_metric_reevaluation": False,
        "visual_review": "pending", "combined_pdf": "pending host pdfunite; append original pages without rerendering"})
    (args.out / "README.md").write_text(
        "# F/G: DL and B-spline optimization ARC comparisons\n\n"
        "[Panel F](panel_F.png) · [Panel G](panel_G.png) · [F–G PDF](arc_panels_F_to_G.pdf) · [A–G PDF](arc_panels_A_to_G.pdf)\n\n"
        "F compares SITReg/VFA/FeDDA-1MF NCC9 with 8→4→2 prolonged displacement, bounded-two composition and velocity-regularized refined SVF. G uses 8→4→2→cps1 for the three optimization families, not dense refinement.\n\n"
        "Each figure contains all five DL lambdas per model, seven displacement/SVF lambdas, and one bounded lambda1 point: 30 settings/300 case records per figure. The same DL data appear in both, so there are450 unique source records, not600 independent experiments. All stages have the displayed lambda; bounded vectors are111/1111.\n\n"
        "Top: mean-only. Bottom: identical means with axis-aligned marginal ±1 across-case sample SD (ddof1), not covariance/confidence regions. Same first10 pairs, masks, labels and metric conventions as accepted A–E. Shared ranges/ticks/fonts are unchanged. Folding is included in CSV/JSON, not inferred from positive-J foreground SD(log J).\n\n"
        "DL lambda1 is historical but has uniform visual styling. Two-map methods have larger optimizer budgets. Equal lambda is not equal-strength regularization across representations. This remains exploratory first-ten analysis. No registration, inference or field-metric reevaluation occurred. Original A–E/table files stay unchanged.\n\n"
        "[Summary CSV](summary.csv) · [Per-case CSV](per_case.csv) · [JSON](results.json). See ACCEPTED.json for final visual/PDF/statistical verification; manifest records the earlier rendering stage.\n", encoding="utf-8")
    base.write_json(args.out / "RENDERED.json", {"render_checks_passed": True,
                    "output_sha256": {p.name: base.sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
