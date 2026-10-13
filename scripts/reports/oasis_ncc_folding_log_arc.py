#!/usr/bin/env python
"""Presentation-only log-folding export; preserve accepted linear ARC artifacts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox

import oasis_ncc_three_regularity_arc as base

X = "jdet_le0_pct_fg"


def label_value(value):
    if value < 0 or not math.isfinite(value):
        raise ValueError("Folding must be finite and nonnegative")
    return "0%" if value == 0 else f"{value:.4g}%"


def display_floor(rows):
    positive = [r[X + "_mean"] for r in rows if r[X + "_mean"] > 0]
    if not positive:
        raise ValueError("Need positive folding values for a logarithmic axis")
    return 10. ** (math.floor(math.log10(min(positive))) - 1)


def sd_contour(x, y, sx, sy, floor):
    """Clip an arithmetic +/-1-SD ellipse before applying the log-axis transform.

    Do not compute log-space SD, replace casewise zeros, or pretend the contour
    remains an ellipse in display coordinates. Intersect with x >= floor.
    """
    theta = np.linspace(0, 2 * np.pi, 721)
    polygon = np.column_stack((x + sx * np.cos(theta), y + sy * np.sin(theta)))
    clipped = []
    previous = polygon[-1]
    for current in polygon:
        prev_in, curr_in = previous[0] >= floor, current[0] >= floor
        if prev_in != curr_in:
            t = (floor - previous[0]) / (current[0] - previous[0])
            clipped.append([floor, previous[1] + t * (current[1] - previous[1])])
        if curr_in:
            clipped.append(current.tolist())
        previous = current
    return np.asarray(clipped)


def overlap(a, b):
    return max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))


def place_labels(fig, axes):
    """Deterministic screen-space placement; leader lines identify crowded points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        bounds, placed = ax.get_window_extent().padded(-3), []
        annotations = list(ax.texts)
        markers = [Bbox.from_bounds(*(ax.transData.transform(a.xy) - 5), 10, 10) for a in annotations]
        for annotation in annotations:
            prefer_up = annotation.get_color() == base.old.COLORS["gncc"]
            offsets = [(dx, dy) for dy in ([12, 30, -12, -30, 48, -48, 66, -66] if prefer_up else [-12, -30, 12, 30, -48, 48, -66, 66])
                       for dx in [7, -7, 26, -26, 48, -48, 72, -72]]
            best = None
            for index, (dx, dy) in enumerate(offsets):
                annotation.set_position((dx, dy))
                annotation.set_ha("left" if dx > 0 else "right")
                annotation.set_va("bottom" if dy > 0 else "top")
                annotation.update_positions(renderer)
                annotation.update_bbox_position_size(renderer)
                box = annotation.get_bbox_patch().get_window_extent(renderer).padded(2)
                outside = max(0, box.width * box.height - overlap(box, bounds))
                score = 100 * outside + 20 * sum(overlap(box, b) for b in placed) + 5 * sum(overlap(box, b) for b in markers) + .03 * (dx * dx + dy * dy) + index * .01
                if best is None or score < best[0]:
                    best = score, dx, dy, box
            _, dx, dy, box = best
            annotation.set_position((dx, dy))
            annotation.set_ha("left" if dx > 0 else "right")
            annotation.set_va("bottom" if dy > 0 else "top")
            placed.append(box)


def build_figure(model, means, all_means, floor):
    fig, axes = base.build_figure(model, means, X, all_means)
    upper = 10. ** math.ceil(math.log10(max(r[X + "_mean"] + r[X + "_std"] for r in all_means)))
    for ri in range(2):
        for ci, (metric, _) in enumerate(base.old.Y):
            ax = axes[ri, ci]
            for artist in list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.texts):
                artist.remove()
            ax.set_xscale("log")
            ax.set_xlim(floor / 2, upper)
            ax.set_xlabel("Foreground folding (%) · logarithmic scale ↓", fontsize=11)
            ax.axvline(floor, color="#777777", ls=":", lw=.8, alpha=.7)
            ax.grid(visible=True, which="major", alpha=.18)
            ax.tick_params(axis="x", which="minor", bottom=False)
            for loss, group in base.old.SERIES.items():
                color = base.old.COLORS[loss]
                for (ka, _), (kb, _) in zip(group, group[1:]):
                    points = [means[loss, k] for k in (ka, kb)]
                    ax.plot([max(r[X + "_mean"], floor) for r in points], [r[metric + "_mean"] for r in points],
                            color=color, lw=2.2, ls="--" if any(r["historical"] for r in points) else "-", zorder=3)
                for key, _ in group:
                    r = means[loss, key]
                    actual, y = r[X + "_mean"], r[metric + "_mean"]
                    x = max(actual, floor)
                    if ri:
                        sx, sy = r[X + "_std"], r[metric + "_std"]
                        if sx == 0:
                            ax.vlines(x, y - sy, y + sy, color=color, alpha=.35, lw=2)
                        else:
                            contour = sd_contour(actual, y, sx, sy, floor)
                            ax.add_patch(Polygon(contour, closed=True, facecolor=color, edgecolor=color, alpha=.13, lw=1.1, zorder=1))
                    ax.plot(x, y, marker="D" if r["historical"] else "o" if loss == "gncc" else "s",
                            color=color, markerfacecolor="white" if r["historical"] else color,
                            ms=6.5, markeredgewidth=1.4, ls="none", zorder=4)
                    key_label = "1/3" if key == "1over3" else key
                    text = f"λ={key_label}{'*' if r['historical'] else ''}\n{label_value(actual)}" + ("†" if actual == 0 else "")
                    ax.annotate(text, (x, y), xytext=(7, 12), textcoords="offset points", color=color,
                                fontsize=8.3, zorder=5,
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=.8),
                                arrowprops=dict(arrowstyle="-", color=color, lw=.55, alpha=.7, shrinkA=2, shrinkB=5))
    fig._suptitle.set_text(f"{base.NAMES[model]} · OASIS first100 · Foreground folding (log scale)")
    for text in fig.texts:
        if text.get_text().startswith("TOP:"):
            text.set_text("TOP: mean curves · each point labeled with λ and mean folding (%)")
        elif text.get_text().startswith("BOTTOM:"):
            text.set_text("BOTTOM: same means with arithmetic ±1 SD contours transformed onto the log axis")
        elif text.get_position()[1] == .044:
            text.set_text(f"† Exact-zero means are displayed at {floor:g}% (dotted line), not replaced in the data. All nonzero point labels show their actual mean values.")
        elif text.get_position()[1] == .018:
            text.set_text("100 paired cases/point · Sample SD, not confidence regions · SD contours below the display floor are clipped; no log-space SD or pseudocounts.")
    place_labels(fig, axes)
    return fig, axes


def export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(base.REPO):
        raise ValueError("Use a distinct external presentation directory")
    accepted = base.read(root / "ACCEPTED.json")
    if not accepted.get("complete") or accepted.get("cases") != 1800 or not (root / "DONE").is_file():
        raise ValueError("Require completed 1800-field analysis")
    for name, digest in accepted["outputs_sha256"].items():
        if base.sha(root / name) != digest:
            raise ValueError("Changed accepted source: " + name)
    means = base.summarize(base.read(root / "per_case.json"))
    all_means = [r for model in means.values() for r in model.values()]
    floor = display_floor(all_means)
    out.mkdir(parents=True, exist_ok=False)
    points = []
    for model in means:
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
            fig, _ = build_figure(model, means[model], all_means, floor)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(out / f"{model}_gncc_vs_lncc_folding_log_values_2x3.{ext}", dpi=180, facecolor="white")
            plt.close(fig)
        for r in means[model].values():
            points.append({**r, "folding_label": label_value(r[X + "_mean"]),
                           "display_x_pct": max(r[X + "_mean"], floor), "zero_display_only": r[X + "_mean"] == 0})
    base.write_csv(out / "plotted_mean_std.csv", points)
    for name, digest in accepted["outputs_sha256"].items():
        if base.sha(root / name) != digest:
            raise ValueError("Source changed during export")
    base.write_json(out / "plot_manifest.json", dict(complete=True, figures=2, points=18, cases_per_point=100,
        xscale="log", folding_unit="percent", display_floor_pct=floor,
        zero_rule="Exact-zero means displayed at floor and labeled 0% dagger; source measurements unchanged",
        uncertainty="Arithmetic marginal +/-1 sample SD contours; clipped x>=floor then log-transformed; not log-SD, CI or covariance ellipses",
        source_acceptance_sha256=base.sha(root / "ACCEPTED.json"), source_outputs_sha256=accepted["outputs_sha256"],
        script_sha256={str(p): base.sha(p) for p in (Path(__file__), Path(base.__file__), Path(base.old.__file__))},
        outputs_sha256={p.name: base.sha(p) for p in out.iterdir() if p.is_file()}))
    print("COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.out)
