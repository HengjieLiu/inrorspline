#!/usr/bin/env python
"""Line-only six-loss ARC plots from the accepted P44 first10 report.

No registrations, metric recomputation from images, or changes to frozen P44
sources/results. Reproduce every plotted mean from the 440 accepted case rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAIR_FILE = REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
VARIANT = "oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1"
LOSSES = ("ncc", "mse", "mind", "rc", "ngf", "femse")
REFERENCES = ("VFA", "SITReg")
LABELS = dict(ncc="NCC (window 9)", mse="MSE", mind="MIND", rc="RC", ngf="NGF",
              femse="Feature-MSE", VFA="VFA", SITReg="SITReg")
COLORS = dict(ncc="#222222", mse="#D55E00", mind="#009E73", rc="#CC79A7",
              ngf="#B48A00", femse="#56B4E9", VFA="#0072B2", SITReg="#7A3DB8")
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg", "diffusion")
CENTERS = dict(mse=.005, mind=.04, rc=.1, ngf=.2, femse=.2)
MULTIPLIERS = (.01, .1, 1/3, 1., 3., 10., 100.)
GRIDS = {loss: tuple(center*m for m in MULTIPLIERS) for loss, center in CENTERS.items()}
GRIDS["ncc"] = (.01, .1, .3, 1., 3., 10., 100.)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs():
    return re.findall(r"\((\d{4}),\s*(\d{4})\)", PAIR_FILE.read_text())[:10]


def validate_payload(data, accepted, expected_pairs):
    counts = {"candidate_count": 350, "configuration_count": 35,
              "historical_ncc_count": 70, "original_reference_count": 20}
    if not accepted.get("technical_completion") or any(accepted.get(k) != v for k, v in counts.items()):
        raise ValueError("Require accepted 350-fit report plus70 NCC and20 DL references")
    if not data.get("complete") or data.get("variant") != VARIANT:
        raise ValueError("Wrong or incomplete experiment")
    rows, means = data["case_metrics"], data["means"]
    if len(rows) != 440 or len(means) != 44 or len(data["candidates"]) != 350:
        raise ValueError("Wrong case/mean/candidate counts")
    if len(expected_pairs) != 10:
        raise ValueError("Require audited first10 pairs")
    expected_keys = {(loss, alpha) for loss in LOSSES for alpha in GRIDS[loss]}
    expected_keys |= {(ref, None) for ref in REFERENCES}
    mean_map = {(r["loss"], r["effective_lambda"]): r for r in means}
    if len(mean_map) != 44 or set(mean_map) != expected_keys:
        raise ValueError("Missing, duplicate or wrong lambda cohort")
    if {(r["loss"], r["effective_lambda"]) for r in rows} != expected_keys:
        raise ValueError("Wrong case cohorts")
    checked = []
    for loss in (*LOSSES, *REFERENCES):
        for alpha in GRIDS[loss] if loss in GRIDS else (None,):
            group = [r for r in rows if (r["loss"], r["effective_lambda"]) == (loss, alpha)]
            if sorted(r["case_rank"] for r in group) != list(range(10)):
                raise ValueError("Incomplete or duplicate first10 cohort")
            for row in group:
                if (row["fixed_id"], row["moving_id"]) != expected_pairs[row["case_rank"]] or row["direction"] != "moving_to_fixed":
                    raise ValueError("Case identity/direction mismatch")
                if loss == "ncc" and (row.get("resolution"), row.get("representation")) != ("full", "842_dense"):
                    raise ValueError("NCC reference must be full-resolution 8-4-2-to-dense")
            summary = dict(mean_map[(loss, alpha)])
            if summary["n"] != 10:
                raise ValueError("Wrong mean sample size")
            for metric in METRICS:
                values = np.asarray([r[metric] for r in group], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite case metric")
                for suffix, recomputed in (("mean", values.mean()), ("std", values.std(ddof=1))):
                    if not math.isclose(summary[f"{metric}_{suffix}"], recomputed, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Accepted summary disagrees with case-level values")
            checked.append(summary)
    return checked


def build_figure(means):
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.5), sharex=True)
    titles = (("dice", "Dice ↑"), ("hd95", "HD95 (mm) ↓"), ("asd", "Symmetric ASD (mm) ↓"))
    for ax, (metric, label) in zip(axes, titles):
        for loss in LOSSES:
            # Connect the measured settings in lambda order, not a fitted/smoothed envelope.
            group = sorted((r for r in means if r["loss"] == loss), key=lambda r: r["effective_lambda"], reverse=True)
            ax.plot([r["log_jdet_std_fg_mean"] for r in group], [r[metric+"_mean"] for r in group],
                    color=COLORS[loss], linewidth=2.3, linestyle="-", marker=None, label=LABELS[loss])
        for ref, marker in (("VFA", "D"), ("SITReg", "s")):
            row = next(r for r in means if r["loss"] == ref)
            ax.plot(row["log_jdet_std_fg_mean"], row[metric+"_mean"], marker=marker,
                    markersize=8, color=COLORS[ref], linestyle="none", markeredgecolor="white",
                    markeredgewidth=.8, label=ref, zorder=6)
        ax.set_xlabel("Foreground SD(log J)")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="semibold", pad=12)
        ax.grid(alpha=.18, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=0)
        ax.margins(x=.03, y=.09)
        ax.tick_params(labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("OASIS first10 · full-resolution 8→4→2→dense", fontsize=17, y=.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .927), ncol=8,
               frameon=False, fontsize=10.5, handlelength=2.5, columnspacing=1.6)
    fig.text(.5, .055, "Means across 10 cases · Six curves connect seven λ settings each · VFA/SITReg: original-field references",
             ha="center", fontsize=10)
    fig.text(.5, .019, "No ellipses or error bars. SD(log J) excludes J ≤ 0; it does not measure folding. Exploratory test-case tuning.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(.005, .11, .995, .83), w_pad=2.6)
    return fig, axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.report_root.resolve()
    out = args.out_dir.resolve() if args.out_dir else source/"lines_only_v1"
    if out.is_relative_to(REPO) or source.is_relative_to(out):
        parser.error("Output must be external and must not replace source report")
    files = [source/"results.json", source/"ACCEPTED.json", source/"mean_std.csv", source/"case_metrics.csv", PAIR_FILE]
    source_hashes = {str(p): sha256(p) for p in files}
    data = json.loads((source/"results.json").read_text())
    accepted = json.loads((source/"ACCEPTED.json").read_text())
    means = validate_payload(data, accepted, pairs())
    out.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, _ = build_figure(means)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out/f"arc_six_losses_lines_only.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
    with (out/"plotted_mean_std.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader(); writer.writerows(means)
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Read-only source changed while plotting")
    outputs = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    manifest = {"complete": True, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_hashes": source_hashes, "script_sha256": sha256(Path(__file__)),
                "outputs": outputs, "cohorts": 44, "case_rows": 440, "new_fits": 350,
                "historical_ncc_fits": 70, "original_dl_references": 20, "cases_per_point": 10,
                "x": "mean_foreground_positive_J_SD_logJ", "y": ["dice", "hd95", "symmetric_asd"],
                "sweep_lines": list(LOSSES), "reference_points": list(REFERENCES),
                "connection_order": "decreasing_lambda", "smoothing": False,
                "ellipses": False, "error_bars": False, "sweep_markers": False,
                "all_44_means_and_sample_sds_reproduced_from_case_rows": True,
                "accepted_at": accepted["completed_at"]}
    (out/"plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"complete": True, "out_dir": str(out), "outputs": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
