#!/usr/bin/env python
"""P76 presentation: bounded-two lambda curve in C/F; four-level G stays lambda1."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import oasis_first10_family_arc as base
import oasis_first10_sitreg_arc as old

KEY = "oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1"
PANEL_SERIES = {"C": ("unbounded_1", "bounded_1", "bounded2"), "F": old.SERIES, "G": old.SERIES}
TITLES = {"C": "B-spline composition · 8→4→2", "F": "SITReg and 8→4→2 optimization",
          "G": "SITReg and 8→4→2→cps1 optimization"}
CAPTIONS = {
 "C": "λ: bounded 2 maps {0.01, 0.1, 0.3, 1, 3, 10, 100}; unbounded / bounded 1 map: 1 only.",
 "F": "λ: SITReg {0.1, 1/3, 1, 3, 10}; displacement / bounded 2 maps / SVF {0.01, 0.1, 0.3, 1, 3, 10, 100}.",
 "G": "λ: SITReg {0.1, 1/3, 1, 3, 10}; displacement / SVF {0.01, 0.1, 0.3, 1, 3, 10, 100}; bounded: 1 only."}


def panel_styles(panel):
    styles = {**old.STYLES, "unbounded_1": ("Unbounded · 1 map/level", "#D55E00", "s", "none"),
              "bounded_1": ("Bounded · 1 map/level", "#009E73", "^", "none")}
    if panel != "G":
        styles["bounded2"] = ("Bounded · 2 maps/level", "#333333", "D", ":")
    return styles


def select(original, fg, sweep):
    result = []
    def add(row, panel, series, source):
        result.append({**row, "panel": panel, "series": series, "method": panel_styles(panel)[series][0],
                       "alpha": float(row["alpha"]), "case_rank": int(row["case_rank"]),
                       **{k: float(row[k]) for k in base.METRICS}, "plot_source": source})
    for row in original:
        if row["panel"] == "C" and row["series"] in ("unbounded_1", "bounded_1"):
            add(row, "C", row["series"], "P75")
    for row in fg:
        if row["panel"] == "G" or (row["panel"] == "F" and row["series"] != "bounded2"):
            add(row, row["panel"], row["series"], "P75_FG_sitreg_v2")
    for row in sweep:
        for panel in ("C", "F"):
            add(row, panel, "bounded2", "P76")
    return sorted(result, key=lambda r: (r["panel"], PANEL_SERIES[r["panel"]].index(r["series"]), r["alpha"], r["case_rank"]))


def expected_groups():
    return {(panel, s, a) for panel, series in PANEL_SERIES.items() for s in series
            for a in (base.DL_LAMBDAS if s == "sitreg" else
                      (1.,) if s in ("unbounded_1", "bounded_1") or (panel == "G" and s == "bounded2") else base.LAMBDAS)}


def summarize(rows, pairs):
    groups = defaultdict(list)
    for row in rows:
        rank = row["case_rank"]
        if not 0 <= rank < 10 or (row["fixed_id"], row["moving_id"]) != tuple(pairs[rank]) or row["direction"] != "moving_to_fixed":
            raise ValueError("Wrong case identity/direction")
        groups[(row["panel"], row["series"], row["alpha"])].append(row)
    if len(rows) != 550 or set(groups) != expected_groups():
        raise ValueError("Require 55 settings/550 displayed case rows")
    summary = []
    for (panel, s, a), cohort in sorted(groups.items()):
        if sorted(r["case_rank"] for r in cohort) != list(range(10)):
            raise ValueError("Incomplete/duplicate cohort")
        rec = {"panel": panel, "series": s, "alpha": a, "method": panel_styles(panel)[s][0], "n": 10}
        for metric in base.METRICS:
            values = [r[metric] for r in cohort]
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Nonfinite plot value")
            rec[metric+"_mean"], rec[metric+"_sd"] = statistics.mean(values), statistics.stdev(values)
        summary.append(rec)
    return summary


def derive_axes(summary):
    axes_spec, top_spec = copy.deepcopy(base.AXES), copy.deepcopy(old.TOP_AXES)
    changes = []
    for key, spec in axes_spec.items():
        metric = "log_jdet_std_fg" if key == "x" else key
        lo = min(r[metric+"_mean"]-r[metric+"_sd"] for r in summary)
        hi = max(r[metric+"_mean"]+r[metric+"_sd"] for r in summary)
        step = spec["ticks"][1]-spec["ticks"][0]
        limits = list(spec["limits"])
        if lo < limits[0] or hi > limits[1]:
            limits = [min(limits[0], math.floor(lo/step)*step), max(limits[1], math.ceil(hi/step)*step)]
            spec["limits"] = limits
            spec["ticks"] = np.arange(limits[0], limits[1]+step/2, step).round(10).tolist()
            changes.append({"axis": key, "scope": "full_SD", "limits": limits})
    for metric, spec in top_spec.items():
        relevant = [r for r in summary if r["panel"] in ("F", "G")]
        lo, hi = min(r[metric+"_mean"] for r in relevant), max(r[metric+"_mean"] for r in relevant)
        limits = list(spec["limits"])
        if lo <= limits[0] or hi >= limits[1]:
            step = spec["ticks"][1]-spec["ticks"][0]
            limits = [min(limits[0], math.floor((lo-step*.1)/step)*step), max(limits[1], math.ceil((hi+step*.1)/step)*step)]
            spec["limits"] = limits
            spec["ticks"] = np.arange(math.ceil(limits[0]/step)*step, limits[1]+step/2, step).round(10).tolist()
            changes.append({"axis": metric, "scope": "tight_top_FG", "limits": limits})
    return axes_spec, top_spec, changes


def figure(panel, summaries, axes_spec, top_spec):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FormatStrFormatter

    series_list = PANEL_SERIES[panel]
    styles = panel_styles(panel)
    point_count = sum(r["panel"] == panel for r in summaries)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=150)
    fig.subplots_adjust(left=.083, right=.982, bottom=.15, top=.78, wspace=.29, hspace=.62)
    fig.suptitle(f"{panel}  |  {TITLES[panel]}", fontsize=24, fontweight="semibold", y=.976)
    fig.text(.5, .931, ("OASIS first 10 · NCC9 + diffusion · Bounded two-map λ sweep; one-map references at λ = 1" if panel == "C" else "OASIS first 10 · NCC9 + diffusion · Top: tighter y scales; bottom: full SD ranges"),
             ha="center", fontsize=16)
    handles = [Line2D([], [], label=styles[s][0], color=styles[s][1], marker=styles[s][2],
                      linestyle=styles[s][3], linewidth=2.8, markersize=(10 if panel == "G" else 8) if s == "bounded2" else 7,
                      markeredgecolor="white", markeredgewidth=.7) for s in series_list]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895),
                        ncol=len(series_list), frameon=False, fontsize=16, columnspacing=1.9, handlelength=3.2)
    fig.text(.012, .66, ("Means only" if panel == "C" else "Means · tighter y scale"), rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.012, .27, "Means + SD ellipses", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.5, .063, CAPTIONS[panel], ha="center", fontsize=14)
    fig.text(.5, .032, "All sweep points retained. Ellipses: ±1 sample SD (n = 10), not confidence regions. Fixed raw foreground; J > 0. Exploratory results.", ha="center", fontsize=14)
    diagnostics = []
    for i in range(2):
        for j, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[i, j]
            for series in series_list:
                rows = sorted((r for r in summaries if r["panel"] == panel and r["series"] == series), key=lambda r: r["alpha"])
                _, color, marker, style = styles[series]
                if i:
                    for row in rows:
                        ax.add_patch(Ellipse((row["log_jdet_std_fg_mean"], row[metric + "_mean"]),
                                             2 * row["log_jdet_std_fg_sd"], 2 * row[metric + "_sd"],
                                             facecolor=color, edgecolor=color, linewidth=1., alpha=.12, zorder=1))
                ax.plot([r["log_jdet_std_fg_mean"] for r in rows], [r[metric + "_mean"] for r in rows],
                        color=color, marker=marker, linestyle=style, linewidth=2.8,
                        markersize=(10 if panel == "G" else 8) if series == "bounded2" else 7, markeredgecolor="white", markeredgewidth=.7,
                        zorder=5 if series == "bounded2" else 3, label=series)
            yspec = top_spec[metric] if not i and panel != "C" else axes_spec[metric]
            ax.set_xlim(axes_spec["x"]["limits"]); ax.set_xticks(axes_spec["x"]["ticks"])
            ax.set_ylim(yspec["limits"]); ax.set_yticks(yspec["ticks"])
            ax.xaxis.set_major_formatter(FormatStrFormatter(axes_spec["x"]["format"]))
            ax.yaxis.set_major_formatter(FormatStrFormatter(yspec["format"]))
            ax.set_xlabel(axes_spec["x"]["label"], fontsize=19, labelpad=9)
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
        if top.patches or len(bottom.patches) != point_count:
            raise ValueError("Wrong ellipse count")
    return fig, diagnostics


def run(root):
    source = root / "repo_experiments" / base.KEY
    fg_source = source / "panels_FG_sitreg_v2"
    sweep_source = root / "repo_experiments" / KEY
    out = sweep_source / "arc_panels_v1"
    if out.exists():
        raise FileExistsError("Preserve existing plot release")
    hashes = {}
    for directory in (source, fg_source, sweep_source):
        accepted = base.read_json(directory / "ACCEPTED.json")
        if not accepted["complete"]:
            raise ValueError("Unaccepted plot source")
        for name, digest in accepted["output_sha256"].items():
            if base.sha(directory / name) != digest:
                raise ValueError("Plot source hash mismatch")
            hashes[str(directory / name)] = digest
        hashes[str(directory / "ACCEPTED.json")] = base.sha(directory / "ACCEPTED.json")
    pairs = base.read_json(source / "manifest.json")["pairs"]
    rows = select(base.read_csv(source / "per_case.csv"), base.read_csv(fg_source / "per_case.csv"),
                  base.read_csv(sweep_source / "per_case.csv"))
    summary = summarize(rows, pairs)
    axes_spec, top_spec, changes = derive_axes(summary)
    out.mkdir()
    # Different historical row schemas are intentionally normalized for display data.
    fields = ("panel", "series", "method", "alpha", "case_rank", "fixed_id", "moving_id", "direction", *base.METRICS, "plot_source")
    rows = [{k: r[k] for k in fields} for r in rows]
    base.write_csv(out / "per_case.csv", rows); base.write_csv(out / "summary.csv", summary)
    base.write_json(out / "results.json", {"per_case": rows, "summary": summary, "pairs": pairs})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    diagnostics = []
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}), PdfPages(out / "arc_panels_F_to_G.pdf") as book:
        for panel in ("C", "F", "G"):
            fig, records = figure(panel, summary, axes_spec, top_spec)
            diagnostics.extend(records)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(out / f"panel_{panel}.{ext}", dpi=200)
            if panel in ("F", "G"):
                book.savefig(fig)
            plt.close(fig)
            print(f"Rendered {panel}: {sum(r['panel']==panel for r in summary)} operating points", flush=True)
    for path, digest in hashes.items():
        if base.sha(path) != digest:
            raise ValueError("Source changed during rendering")
    base.write_json(out / "manifest.json", {"source_sha256": hashes, "source_p75": str(source), "source_sweep": str(sweep_source),
                    "script_sha256": base.sha(__file__), "base_script_sha256": base.sha(base.__file__),
                    "prior_renderer_sha256": base.sha(old.__file__), "axes": axes_spec, "top_axes": top_spec,
                    "axis_changes": changes, "styles": {p: panel_styles(p) for p in PANEL_SERIES},
                    "rendered_axes": diagnostics, "points": {"C": 9, "F": 26, "G": 20},
                    "displayed_case_rows": 550, "pairs": pairs, "font_sizes_pt": base.FONT,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "visual_review": "pending", "combined_pdf": "pending host merge of original A/B/D/E and revised C/F/G"})
    (out / "README.md").write_text(
        "# ARC update: bounded two-map 842 lambda sweep\n\n"
        "[C](panel_C.png) · [F](panel_F.png) · [G](panel_G.png) · [F–G PDF](arc_panels_F_to_G.pdf) · [A–G PDF](arc_panels_A_to_G.pdf)\n\n"
        "C/F now contain seven fresh bounded-two 8→4→2 lambda points, black dotted/diamond. C retains unbounded-one and bounded-one atlambda1. G remains 8→4→2→cps1 with the historical bounded-two lambda1 point, not this three-level sweep. F/G retain only SITReg among DL models. No other curve data changed.\n\n"
        "F/G top rows have tighter shared y scales; lower rows retain full SD ranges. C keeps the original family-panel scale convention. All x axes are shared. Any necessary axis expansions are recorded in manifest; no points are silently clipped. Ellipses are marginal ±1 across-case sample SD (n10, ddof1), not confidence regions. Fixed raw-positive foreground, positive J for SD(log J). Folding is separately tabulated.\n\n"
        "55 displayed operating points / 550 displayed case records. C/F repeat the same70 bounded case records; F/G repeat50 SITReg records:430 unique source records, not550 independent experiments. These first-ten tuned results remain exploratory. Original figure releases remain read-only.\n", encoding="utf-8")
    base.write_json(out / "RENDERED.json", {"passed": True,
                    "output_sha256": {p.name: base.sha(p) for p in out.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    run(parser.parse_args().artifact_root)

