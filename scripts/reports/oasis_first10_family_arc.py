#!/usr/bin/env python
"""Read-only aggregation of accepted first10 metrics into five shared-axis ARCs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
KEY = "oasis_seed3_first10_ncc9_method_family_arc_panels_v1"
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
DL_LAMBDAS = (.1, 1 / 3, 1., 3., 10.)
SOURCES = {
    "dl": ("oasis_seed3_first100_four_method_sweeps_arc_v1", "per_case.csv", "per_case", "plot_manifest.json"),
    "direct": ("oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1", "case_metrics.csv", "case_metrics", "ACCEPTED.json"),
    "compose": ("oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3", "per_case.csv", "per_case", "ACCEPTED.json"),
    "svf": ("oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1/continuation_v2", "per_case.csv", "per_case", "ACCEPTED.json"),
}
# Explicit aliases preserve exact one-third rather than the old rounded display.
DL_NAMES = {
    "SITReg λ=0.1": ("sitreg", .1), "SITReg λ=1/3": ("sitreg", 1/3),
    "SITReg λ=1 (original)": ("sitreg", 1.), "SITReg λ=3": ("sitreg", 3.),
    "SITReg λ=10": ("sitreg", 10.),
    "VFA λ=0.1": ("vfa", .1), "VFA λ=1/3": ("vfa", 1/3),
    "VFA (original)": ("vfa", 1.), "VFA λ=3": ("vfa", 3.), "VFA λ=10": ("vfa", 10.),
    "FeDDA-1MF lambda0.1": ("fedda", .1), "FeDDA-1MF lambda0.333333": ("fedda", 1/3),
    "FeDDA-1MF lambda1 historical": ("fedda", 1.),
    "FeDDA-1MF lambda3": ("fedda", 3.), "FeDDA-1MF lambda10": ("fedda", 10.),
}
AXES = {
    "x": {"label": "Foreground SD(log J) ↓", "limits": (0., 2.), "ticks": (0., .4, .8, 1.2, 1.6, 2.), "format": "%.1f"},
    "dice": {"label": "Dice ↑", "limits": (.5, .9), "ticks": (.5, .6, .7, .8, .9), "format": "%.2f"},
    "hd95": {"label": "HD95 (mm) ↓", "limits": (1., 7.), "ticks": (1., 2., 3., 4., 5., 6., 7.), "format": "%.0f"},
    "asd": {"label": "Symmetric ASD (mm) ↓", "limits": (0., 1.6), "ticks": (0., .4, .8, 1.2, 1.6), "format": "%.1f"},
}
FONT = {"title": 24, "column": 21, "label": 19, "tick": 16, "legend": 16, "caption": 14}
FIGURES = {
    "A": ("Deep-learning methods", ("sitreg", "vfa", "fedda")),
    "B": ("Direct displacement representations", ("842", "8421", "842_dense")),
    "C": ("B-spline composition · 8→4→2", ("unbounded_1", "bounded_1", "bounded_2_compose")),
    "D": ("B-spline composition · 8→4→2→1", ("unbounded_1", "bounded_1", "bounded_2_compose")),
    "E": ("Refined B-spline SVFs", ("842_velocity", "842_displacement", "8421_velocity", "8421_displacement")),
}
# label, color, marker, linestyle. DL lambda1 is not a separate visual category.
STYLES = {
    "sitreg": ("SITReg", "#0072B2", "o", "-"),
    "vfa": ("VFA", "#D55E00", "s", "-"),
    "fedda": ("FeDDA-1MF", "#009E73", "^", "-"),
    "842": ("8→4→2", "#0072B2", "o", "-"),
    "8421": ("8→4→2→cps1", "#D55E00", "s", "--"),
    "842_dense": ("8→4→2→dense", "#009E73", "^", ":"),
    "unbounded_1": ("Unbounded · 1 map/level", "#555555", "s", "none"),
    "bounded_1": ("Bounded · 1 map/level", "#0072B2", "o", "none"),
    "bounded_2_compose": ("Bounded · 2 maps/level", "#009E73", "^", "none"),
    "842_velocity": ("8→4→2 · velocity penalty", "#0072B2", "o", "-"),
    "842_displacement": ("8→4→2 · displacement penalty", "#D55E00", "s", "-"),
    "8421_velocity": ("8→4→2→1 · velocity penalty", "#0072B2", "^", "--"),
    "8421_displacement": ("8→4→2→1 · displacement penalty", "#D55E00", "D", "--"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_pairs(path):
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", path.read_text())[:10]
    if len(pairs) != 10 or len(set(pairs)) != 10:
        raise ValueError("Require ten unique canonical pairs")
    return pairs


def classify(source, row):
    """Return the prespecified figure/series/lambda or exclude contextual rows."""
    if source == "dl":
        return ("A", *DL_NAMES[row["method"]]) if row["method"] in DL_NAMES else None
    if source == "direct":
        if row["resolution"] == "pyramid" and row["representation"] in FIGURES["B"][1]:
            return "B", row["representation"], float(row["alpha"])
    if source == "compose":
        if str(row["depth"]) in ("3", "4") and row["config_key"] in FIGURES["C"][1]:
            return ("C" if int(row["depth"]) == 3 else "D"), row["config_key"], 1.
    if source == "svf" and row["source_kind"] == "new_svf":
        return "E", f"{row['schedule']}_{row['regularize']}", float(row["alpha"])
    return None


def expected_groups():
    return {(panel, series, alpha) for panel, (_, series_list) in FIGURES.items()
            for series in series_list
            for alpha in (DL_LAMBDAS if panel == "A" else (1.,) if panel in "CD" else LAMBDAS)}


def identity(source, row):
    return classify(source, row), int(row.get("case_rank", row.get("rank")))


def load_sources(root, pairs):
    rows, fingerprints, checks = [], {}, []
    for source, (folder, csv_name, json_key, accepted_name) in SOURCES.items():
        base = root / folder
        accepted = read_json(base / accepted_name)
        if not any(accepted.get(k) is True for k in ("complete", "technical_completion", "passed")):
            raise ValueError(f"Unaccepted source: {source}")
        result = read_json(base / "results.json")
        paths = [base / accepted_name, base / csv_name, base / "results.json"]
        summary_name = "mean_std.csv" if source == "direct" else "summary.csv"
        if source != "dl":
            paths.append(base / summary_name)
        for path in paths:
            fingerprints[str(path)] = sha(path)
            expected = accepted.get("output_sha256", {}).get(path.name)
            if expected and fingerprints[str(path)] != expected:
                raise ValueError(f"Accepted source hash changed: {path}")
        raw_rows = read_csv(base / csv_name)
        json_rows = result[json_key]
        index = {}
        for row in json_rows:
            key = identity(source, row)
            if key[0] is None or not 0 <= key[1] < 10:
                continue
            if key in index:
                raise ValueError("Duplicate source JSON case")
            index[key] = row
        selected = []
        for raw in raw_rows:
            group, rank = identity(source, raw)
            if group is None or not 0 <= rank < 10:
                continue
            if raw.get("direction") not in (None, "", "moving_to_fixed"):
                raise ValueError("Registration direction mismatch")
            fixed, moving = (raw[k].removeprefix("OASIS_OAS1_") for k in ("fixed_id", "moving_id"))
            if (fixed, moving) != pairs[rank]:
                raise ValueError(f"Case identity mismatch: {source}, {rank}")
            original = index[(group, rank)]
            if (original["fixed_id"], original["moving_id"]) != (fixed, moving):
                raise ValueError("CSV/JSON case identity disagreement")
            for metric in METRICS:
                if not math.isfinite(float(raw[metric])) or float(raw[metric]) != float(original[metric]):
                    raise ValueError(f"Nonfinite or CSV/JSON metric disagreement: {metric}")
            panel, series, alpha = group
            selected.append(dict(panel=panel, series=series, method=STYLES[series][0], alpha=alpha,
                                 case_rank=rank, fixed_id=fixed, moving_id=moving, direction="moving_to_fixed",
                                 **{m: float(raw[m]) for m in METRICS}, source=source,
                                 source_file=str(base / csv_name), source_method=raw.get("method", raw.get("group", "")),
                                 source_run=raw.get("run_dir", ""), historical_dl_lambda1=panel == "A" and alpha == 1.))
        if len(selected) != len(index):
            raise ValueError("CSV/JSON selected coverage mismatch")
        if source != "dl":
            summaries = {classify(source, r): r for r in read_csv(base / summary_name) if classify(source, r)}
            for row in summarize(selected, complete=False):
                prior = summaries[(row["panel"], row["series"], row["alpha"])]
                if int(prior["n"]) != 10:
                    raise ValueError("Wrong source cohort size")
                for metric in METRICS:
                    for new, old in (("mean", "mean"), ("sd", "std" if source == "direct" else "sd")):
                        if not math.isclose(row[f"{metric}_{new}"], float(prior[f"{metric}_{old}"]), rel_tol=0., abs_tol=1e-12):
                            raise ValueError("Recomputed/source summary disagreement")
        rows.extend(selected)
        checks.append({"source": source, "selected_cases": len(selected), "csv_json_equal": True,
                       "source_summary_equal": source != "dl", "direction": "moving_to_fixed from accepted evaluation; explicit row checked when present"})
    return rows, fingerprints, checks


def summarize(rows, complete=True):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["series"], row["alpha"])].append(row)
    if complete and (set(groups) != expected_groups() or len(rows) != 700):
        raise ValueError("Require exactly 70 prespecified settings and 700 cases")
    summaries = []
    for (panel, series, alpha), cohort in sorted(groups.items()):
        if sorted(r["case_rank"] for r in cohort) != list(range(10)):
            raise ValueError("Incomplete/duplicate first10 cohort")
        row = dict(panel=panel, series=series, method=STYLES[series][0], alpha=alpha, n=10)
        for metric in METRICS:
            values = [r[metric] for r in cohort]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metric")
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values)
        summaries.append(row)
    return summaries


def check_extents(summaries):
    extents = {}
    for metric in ("log_jdet_std_fg", "dice", "hd95", "asd"):
        lo = min(r[metric + "_mean"] - r[metric + "_sd"] for r in summaries)
        hi = max(r[metric + "_mean"] + r[metric + "_sd"] for r in summaries)
        bounds = AXES["x" if metric == "log_jdet_std_fg" else metric]["limits"]
        if lo < bounds[0] or hi > bounds[1]:
            raise ValueError(f"Approved axis would clip SD ellipse: {metric}")
        extents[metric] = {"minimum_mean_minus_sd": lo, "maximum_mean_plus_sd": hi}
    return extents


def make_figure(panel, summaries):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FormatStrFormatter

    title, series_list = FIGURES[panel]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=150)
    fig.subplots_adjust(left=.083, right=.982, bottom=.15, top=.78, wspace=.29, hspace=.62)
    fig.suptitle(f"{panel}  |  {title}", fontsize=FONT["title"], fontweight="semibold", y=.976)
    weight_note = ("λ = 0.1, 1/3, 1, 3, 10" if panel == "A" else
                   "λ = 1 at every level · fixed-setting comparison" if panel in "CD" else
                   "λ = 0.01, 0.1, 0.3, 1, 3, 10, 100 at every level")
    fig.text(.5, .931, f"OASIS first 10 · NCC9 + diffusion · {weight_note}", ha="center", fontsize=16)
    handles = [Line2D([], [], label=STYLES[s][0], color=STYLES[s][1], marker=STYLES[s][2],
                      linestyle=STYLES[s][3], linewidth=2.5, markersize=7) for s in series_list]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .904),
                        ncol=2 if panel == "E" else 3, frameon=False, fontsize=FONT["legend"],
                        columnspacing=2., handlelength=2.6)
    fig.text(.012, .66, "Means only", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.012, .27, "Means + SD ellipses", rotation=90, va="center", ha="center", fontsize=16)
    fig.text(.5, .063, "Ellipses: ±1 across-case sample SD (n = 10); not confidence regions.", ha="center", fontsize=FONT["caption"])
    fig.text(.5, .032, "Foreground: fixed raw image > 0; SD(log J) uses J > 0. Folding is tabulated separately. Exploratory test-case tuning.", ha="center", fontsize=FONT["caption"])

    diagnostics = []
    for row_index in range(2):
        for col, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[row_index, col]
            for series in series_list:
                rs = sorted((r for r in summaries if r["panel"] == panel and r["series"] == series), key=lambda r: r["alpha"])
                _, color, marker, style = STYLES[series]
                if row_index:
                    for r in rs:
                        ax.add_patch(Ellipse((r["log_jdet_std_fg_mean"], r[metric + "_mean"]),
                                             2*r["log_jdet_std_fg_sd"], 2*r[metric + "_sd"],
                                             facecolor=color, edgecolor=color, linewidth=1., alpha=.16, zorder=1))
                ax.plot([r["log_jdet_std_fg_mean"] for r in rs], [r[metric + "_mean"] for r in rs],
                        color=color, marker=marker, linestyle=style, linewidth=2.5, markersize=7,
                        markeredgecolor="white", markeredgewidth=.65, zorder=3, label=series)
            ax.set_xlim(AXES["x"]["limits"]); ax.set_xticks(AXES["x"]["ticks"])
            ax.set_ylim(AXES[metric]["limits"]); ax.set_yticks(AXES[metric]["ticks"])
            ax.xaxis.set_major_formatter(FormatStrFormatter(AXES["x"]["format"]))
            ax.yaxis.set_major_formatter(FormatStrFormatter(AXES[metric]["format"]))
            ax.set_xlabel(AXES["x"]["label"], fontsize=FONT["label"], labelpad=9)
            ax.set_ylabel(AXES[metric]["label"], fontsize=FONT["label"], labelpad=10)
            if row_index == 0:
                ax.set_title(("Dice", "HD95", "Symmetric ASD")[col], fontsize=FONT["column"], pad=14)
            ax.tick_params(axis="both", labelsize=FONT["tick"], length=5, pad=6)
            ax.set_axisbelow(True); ax.grid(color="#D8DDE3", linewidth=.7, alpha=.8)
            ax.spines[["top", "right"]].set_visible(False)
            diagnostics.append({"panel": panel, "row": row_index, "metric": metric,
                                "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                                "xticks": list(ax.get_xticks()), "yticks": list(ax.get_yticks()),
                                "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                                "point_count": sum(len(line.get_xdata()) for line in ax.lines),
                                "ellipse_count": len(ax.patches)})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Ensure all titles/labels/ticks/legend are inside the saved page.
    texts = list(fig.texts)
    for ax in axes.flat:
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    for artist in [*texts, legend]:
        if not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.width or bbox.y1 > fig.bbox.height:
            raise ValueError(f"Text/legend outside figure: {getattr(artist, 'get_text', lambda: 'legend')()}")
    if legend.get_window_extent(renderer).y0 < max(ax.get_tightbbox(renderer).y1 for ax in axes[0]):
        raise ValueError("Legend overlaps plot titles")
    for top, bottom in zip(axes[0], axes[1]):
        for a, b in zip(top.lines, bottom.lines):
            if list(a.get_xdata()) != list(b.get_xdata()) or list(a.get_ydata()) != list(b.get_ydata()):
                raise ValueError("Top/bottom means differ")
        if top.patches or len(bottom.patches) != sum(len(line.get_xdata()) for line in top.lines):
            raise ValueError("Wrong ellipse coverage")
    return fig, diagnostics


def readme(summaries):
    lines = ["# First10 NCC9 method-family ARC figures", "",
             "Five figures, each 2×3: mean-only top, the identical means with marginal ±1 sample-SD ellipses below.",
             "Columns are Dice, HD95 and pooled-area symmetric ASD. All figures share the same x scale and corresponding y scales.", "",
             "## Files", "", "[All five figures (PDF)](arc_panels_A_to_E.pdf)", ""]
    for panel, (title, _) in FIGURES.items():
        lines.append(f"- {panel}: {title} — [PNG](panel_{panel}.png), [PDF](panel_{panel}.pdf), [SVG](panel_{panel}.svg)")
    lines += ["", "## Interpretation and provenance", "",
              "- Same seed3 shuffled ranks0–9, moving→fixed; n=10 per operating point. All means and sample SDs (ddof=1) are recomputed from saved per-case metrics, not 100-case summaries.",
              "- A uses SITReg, VFA and original 200-epoch FeDDA-1MF NCC9. Their lambda1 checkpoints are historical, not newly matched-seed trainings; all lambdas intentionally share one visual format.",
              "- B uses pyramid-loss prolonged displacement B-splines or direct dense refinement, with a shared lambda at every active stage.",
              "- C/D use the matched P70 factorial: retained composition with one unbounded, one bounded, or two bounded maps per level. All stage lambdas are1. These are fixed-setting markers, not lambda curves. The unbounded comparator is NOT the prolonged spline in B. No1151 or prior P69 duplicate runs are substituted.",
              "- E uses refined single SVFs: native seven-step integration, with diffusion on velocity or integrated displacement. Eight-/ten-step diagnostic fields are excluded. The complete continuation_v2 release supplies all280 results.",
              "- Ellipses are axis-aligned, width=2×sample SD(x), height=2×sample SD(y). They show marginal across-case spread, not covariance contours, confidence regions or uncertainty of the mean.",
              "- Foreground is fixed raw image>0. SD(log J) uses positive Jacobians only; folding (J≤0, percent) is retained in CSV/JSON, not encoded by SD(log J). Canonical absent-label and pooled-area symmetric-ASD conventions are inherited from accepted evaluations.",
              "- Shared broad limits are intentional and contain all mean±SD extents. No clipping, zooming, smoothing or inferred operating points. Lines connect increasing lambdas; C/D have no connecting lines.",
              "- Two maps receive more optimization budget; equal lambda on different representations/penalties does not imply equal-strength regularization. This is exploratory test-case analysis, not an unbiased benchmark or a topology guarantee.",
              "- No registration, inference, field-metric reevaluation or GPU work was performed. Original reports remain unchanged. manifest.json records source hashes, source revision and figure-axis checks.", "",
              "## Data", "", "[Per-case CSV](per_case.csv) · [Mean/sample-SD CSV](summary.csv) · [JSON](results.json)", "",
              "| Panel | Method | λ | Dice | HD95 mm | Symmetric ASD mm | FG SD(log J) | FG folding % |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        label = "1/3" if r["alpha"] == 1/3 else f"{r['alpha']:g}"
        lines.append("| " + " | ".join([r["panel"], r["method"], label,
                       *[f"{r[m+'_mean']:.6f} ± {r[m+'_sd']:.6f}" for m in METRICS]]) + " |")
    return "\n".join(lines) + "\n"


def run(args):
    if args.out.exists():
        raise FileExistsError("Output directory exists; refusing to overwrite any release")
    pairs = case_pairs(args.pairs)
    rows, fingerprints, checks = load_sources(args.reports, pairs)
    summaries = summarize(rows)
    extents = check_extents(summaries)
    fingerprints[str(args.pairs)] = sha(args.pairs)
    fingerprints[str(Path(__file__))] = sha(__file__)
    args.out.mkdir(parents=True, exist_ok=False)
    rows.sort(key=lambda r: (r["panel"], r["series"], r["alpha"], r["case_rank"]))
    write_csv(args.out / "per_case.csv", rows)
    write_csv(args.out / "summary.csv", summaries)
    write_json(args.out / "results.json", {"per_case": rows, "summary": summaries, "case_pairs": pairs})
    (args.out / "README.md").write_text(readme(summaries), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    axis_records = []
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False}), \
            PdfPages(args.out / "arc_panels_A_to_E.pdf") as book:
        for panel in FIGURES:
            fig, diagnostics = make_figure(panel, summaries)
            axis_records.extend(diagnostics)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(args.out / f"panel_{panel}.{ext}", dpi=200)
            book.savefig(fig)
            plt.close(fig)
            print(f"Rendered panel {panel}; shared-axis/layout/ellipse checks passed", flush=True)
    for path, digest in fingerprints.items():
        if sha(path) != digest:
            raise ValueError(f"Source changed during rendering: {path}")
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(), "key": KEY,
        "source_sha256": fingerprints, "source_checks": checks,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "python": sys.version, "matplotlib": matplotlib.__version__, "command": sys.argv,
        "pairs": pairs, "direction": "moving_to_fixed", "foreground": "fixed_raw_image_positive",
        "ellipse": "axis-aligned marginal ±1 across-case sample SD, ddof1; not CI",
        "font_sizes_pt": FONT, "figure_inches": [20, 11], "axes": AXES,
        "sd_extents": extents, "rendered_axes": axis_records,
        "points_by_panel": dict(Counter(r["panel"] for r in summaries)),
        "source_files_unchanged": True, "no_gpu_or_registration": True,
        "visual_review": "pending separate review", "historical_dl_lambda1_styling": "uniform",
    }
    write_json(args.out / "manifest.json", provenance)
    write_json(args.out / "RENDERED.json", {
        "render_checks_passed": True, "case_count": len(rows), "point_count": len(summaries),
        "axis_count": len(axis_records), "visual_review_pending": True,
        "output_sha256": {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.is_file()},
    })
    print(f"Rendered all five figures: {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt")
    run(parser.parse_args())
