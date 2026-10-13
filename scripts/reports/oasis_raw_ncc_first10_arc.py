#!/usr/bin/env python
"""Canonical first10 VFA/SITReg evaluation and raw-NCC ARC plots, no training."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "scripts/experiments/oasis_alpha_sweep"
sys.path.insert(0, str(SWEEP))
import common as c
import vfa_feature_refinement as f
from run_oasis_vfa_feature_dense_one import canonical_metrics
from run_oasis_saved_field_bspline_fit_one import load_target, sha256_file, gpu_snapshot, target_path
from launch_oasis_mrdbscp842_dense_lambda_sweep import source_metadata
from launch_oasis_raw_ncc_lambda_extension import VARIANT as EXTENSION

VARIANT = "oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1"
GROUPS = ("Raw NCC λ30", "Raw NCC λ100", "Raw NCC λ300", "Raw NCC λ1000", "VFA", "SITReg")
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg", "diffusion")
COLORS = {"Raw NCC": "#D39A20", "VFA": "#0072B2", "SITReg": "#8E63CE"}
ELLIPSE_DEFINITION = "Axis-aligned +/-1 across-case sample SD (ddof=1); width/height=2*SD; not a confidence region"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser.add_argument("--extension-review", type=Path, default=root / "repo_experiments" / EXTENSION)
    parser.add_argument("--prior-runs", type=Path, default=root / "runs" / f.VARIANT)
    parser.add_argument("--out-root", type=Path, default=root / "repo_experiments" / VARIANT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Evaluate rank0 only; do not aggregate")
    args = parser.parse_args(argv)
    for source in (args.extension_review, args.prior_runs, REPO):
        if args.out_root.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.out_root.resolve()):
            parser.error("Report output must not overlap source artifacts or the repository")
    if args.cpu_threads < 1 or args.gpu not in ("0", "1", "2", "3"):
        parser.error("Require positive CPU threads and GPU0/1/2/3")
    return args


def summarize(rows):
    result = []
    for first in (0, 1):
        for group in GROUPS:
            selected = [r for r in rows if r["group"] == group and r["case_rank"] >= first]
            if sorted(r["case_rank"] for r in selected) != list(range(first, 10)):
                raise RuntimeError(f"Require exactly the same cases for every group: {group}")
            row = {"subset": f"ranks_{first}_9", "group": group, "n": len(selected),
                   "alpha": selected[0].get("alpha")}
            for key in METRICS:
                values = np.asarray([r[key] for r in selected], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite metric: {group}/{key}")
                row[key + "_mean"] = float(values.mean())
                row[key + "_std"] = float(values.std(ddof=1))
            result.append(row)
    return result


def add_sd_ellipse(ax, row, xkey, ykey, color):
    ellipse = Ellipse((row[xkey + "_mean"], row[ykey + "_mean"]),
                      width=2 * row[xkey + "_std"], height=2 * row[ykey + "_std"],
                      facecolor=to_rgba(color, .10), edgecolor=to_rgba(color, .38),
                      linewidth=1.1, zorder=1)
    ax.add_patch(ellipse)
    return ellipse


def make_plot(means, out_root, *, xkey="log_jdet_std_fg"):
    rows = [r for r in means if r["subset"] == "ranks_0_9"]
    raw = sorted([r for r in rows if r["group"].startswith("Raw NCC")], key=lambda r: r["alpha"])
    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.1))
        fig.subplots_adjust(left=.065, right=.98, bottom=.20, top=.77, wspace=.28)
        for ax, ykey, label in zip(axes, ("dice", "hd95", "asd"), ("Dice ↑", "HD95 (mm) ↓", "Symmetric ASD (mm) ↓")):
            for row in rows:
                family = "Raw NCC" if row["group"].startswith("Raw NCC") else row["group"]
                add_sd_ellipse(ax, row, xkey, ykey, COLORS[family])
                if family != "Raw NCC":
                    ax.scatter(row[xkey + "_mean"], row[ykey + "_mean"], color=COLORS[family],
                               marker="D" if family == "VFA" else "s", s=82,
                               edgecolor="white", linewidth=.8, zorder=5)
            ax.plot([r[xkey + "_mean"] for r in raw], [r[ykey + "_mean"] for r in raw],
                    color=COLORS["Raw NCC"], marker="o", markersize=7, linewidth=2.2,
                    markeredgecolor="white", markeredgewidth=.7, zorder=4)
            for row in raw:
                offset = (8, -18) if row["alpha"] == 30 else (8, 9)
                ax.annotate(f"λ={row['alpha']:g}", (row[xkey + "_mean"], row[ykey + "_mean"]),
                            xytext=offset, textcoords="offset points", fontsize=10,
                            color="#71500A", zorder=6)
            ax.set_xlabel("Foreground SD(log J) ↓" if xkey == "log_jdet_std_fg" else "Foreground J ≤ 0 (%) ↓")
            ax.set_ylabel(label)
            ax.set_title("Overlap" if ykey == "dice" else "Surface distance")
            ax.grid(alpha=.18, linewidth=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=.16, y=.22)
            # Leave visual room for near-zero SITReg folding markers; a hard
            # zero boundary clips half their glyph. This is axis padding only,
            # not negative observed folding. Marginal SD ellipses are unbounded.
            if xkey == "jdet_le0_pct_fg":
                ax.set_xlim(left=-.04 * ax.get_xlim()[1])
            else:
                ax.set_xlim(left=max(0, ax.get_xlim()[0]))
        title = "OASIS first10: raw-NCC/diffusion sweep vs VFA and SITReg"
        fig.suptitle(title, y=.97, fontsize=17, fontweight="semibold")
        handles = [Line2D([], [], color=COLORS["Raw NCC"], marker="o", linewidth=2.2, label="8→4→2→dense: raw NCC + diffusion"),
                   Line2D([], [], color=COLORS["VFA"], marker="D", linestyle="none", markersize=8, label="VFA"),
                   Line2D([], [], color=COLORS["SITReg"], marker="s", linestyle="none", markersize=8, label="SITReg")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .89), ncol=3, frameon=False)
        fig.text(.5, .075, "Means over the same 10 cases; ellipses = ±1 across-case sample SD on each axis (not confidence intervals).",
                 ha="center", fontsize=11)
        fig.text(.5, .025, "Exploratory test-case tuning. SD(log J) uses positive-J foreground voxels; folding is reported separately.",
                 ha="center", fontsize=10, color="#555555")
        stem = "arc_sdlogj_first10" if xkey == "log_jdet_std_fg" else "tradeoff_folding_first10"
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out_root / f"{stem}.{suffix}", dpi=200, facecolor="white")
        plt.close(fig)


def baseline_row(args, prior_args, rank, target, batch, device, vfa_reference):
    target_args = argparse.Namespace(target=target, dl_root=Path(prior_args["dl_root"]))
    path = target_path(target_args, str(batch["fixed_id"]), str(batch["moving_id"]))
    digest = sha256_file(path)
    cache_path = args.out_root / "baseline_cases" / f"{target}_rank{rank:03d}.json"
    if cache_path.exists():
        saved = c.read_json(cache_path)
        if saved["audit"]["source_sha256"] != digest or saved["row"]["case_rank"] != rank:
            raise RuntimeError("Saved baseline changed since this report was started")
        if (saved["metric_code_sha256"] != sha256_file(SWEEP / "common.py") or
                saved["evaluator_code_sha256"] != sha256_file(SWEEP / "run_oasis_vfa_feature_dense_one.py")):
            raise RuntimeError("Baseline evaluator changed since cached evaluation")
        row = saved["row"]
    else:
        norm, audit = load_target(target_args, fixed_id=str(batch["fixed_id"]), moving_id=str(batch["moving_id"]), device=device)
        field = norm * f.voxel_scale(norm)
        metrics, labels = canonical_metrics(field, batch)
        zyx = field.detach().cpu().numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
        mask = batch["fixed_mask"].cpu().numpy().astype(bool)
        regularity = c.jacobian_metrics_voxel(zyx, mask)
        row = {"group": "VFA" if target == "vfa" else "SITReg", "alpha": None, "case_rank": rank,
               "fixed_id": batch["fixed_id"], "moving_id": batch["moving_id"], "direction": "moving_to_fixed",
               **metrics, **regularity, "diffusion": c.diffusion_loss_voxel(zyx),
               "source_path": str(path), "source_sha256": digest}
        c.write_csv(args.out_root / "baseline_cases" / f"{target}_rank{rank:03d}_labels.csv", [labels])
        c.write_json(cache_path, {"row": row, "audit": audit, "metric_code_sha256": sha256_file(SWEEP / "common.py"),
                                 "evaluator_code_sha256": sha256_file(SWEEP / "run_oasis_vfa_feature_dense_one.py")})
    if target == "vfa" and any(abs(row[k] - vfa_reference[k]) > 1e-6 for k in ("dice", "hd95", "asd")):
        raise RuntimeError(f"Fresh VFA metrics disagree with P40: rank{rank}")
    if rank == 0 and target == "sitreg":
        expected = {"dice": .7987193557676338, "hd95": 2.061319336972447, "asd": .42306412147236566}
        if any(abs(row[k] - expected[k]) > 1e-6 for k in expected):
            raise RuntimeError("Rank0 SITReg identity/canonical-metric mismatch")
    if (row["fixed_id"], row["moving_id"]) != (batch["fixed_id"], batch["moving_id"]):
        raise RuntimeError("Wrong baseline case identity")
    return row


def run(args):
    args.out_root.mkdir(parents=True, exist_ok=True)
    acceptance = c.read_json(args.extension_review / "ACCEPTED.json")
    if not acceptance["technical_completion"] or acceptance["new_fit_count"] != 30:
        raise RuntimeError("Require completed/accepted lambda extension")
    source_report = args.extension_review / "report.json"
    report_hash = sha256_file(source_report)
    original = c.read_json(source_report)
    old_args = c.read_json(args.prior_runs / "launch_manifest.json")["args"]
    prior_state = c.read_json(args.prior_runs / "launcher_state.json")
    manifest = {"variant": VARIANT, "args": vars(args), "started_at": c.now_text(),
                "source": source_metadata(), "source_report": str(source_report), "source_report_sha256": report_hash,
                "source_hashes": {str(p): sha256_file(p) for p in (Path(__file__), SWEEP / "common.py", SWEEP / "run_oasis_vfa_feature_dense_one.py")},
                "ellipse_definition": ELLIPSE_DEFINITION, "initial_gpu_state": gpu_snapshot(),
                "uid": os.getuid(), "gid": os.getgid(), "no_registration_optimization": True,
                "foreground": "fixed raw image > 0; 2-voxel Jacobian boundary crop; log only for J>0",
                "across_case_sd_ddof": 1, "within_case_log_jacobian_sd_ddof": 0,
                "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "unrecorded"),
                "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "unrecorded")}
    c.write_json(args.out_root / ("smoke_manifest.json" if args.smoke else "manifest.json"), manifest)
    torch.set_num_threads(args.cpu_threads)
    device = c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    rows = []
    for rank in range(1 if args.smoke else 10):
        batch = c.load_oasis_case(rank=rank, device=device, oasis_root=Path(old_args["oasis_root"]), pair_file=Path(old_args["pair_file"]))
        reference = next(r for r in original["case_metrics"] if r["method"] == "VFA" and r["case_rank"] == rank)
        for target in ("vfa", "sitreg"):
            row = baseline_row(args, old_args, rank, target, batch, device, reference)
            rows.append(row)
            print({"complete": row["group"], "rank": rank, **{k: row[k] for k in ("dice", "hd95", "asd")}}, flush=True)
        for alpha in (30., 100., 300., 1000.):
            matches = [r for r in original["case_metrics"] if r["case_rank"] == rank and r["method"] == f"Raw NCC + diffusion lambda{alpha:g}"]
            if len(matches) != 1:
                raise RuntimeError("Missing or duplicate raw-NCC operating point")
            source = matches[0]
            source_dir = Path(source["run_dir"])
            if not (source_dir / "DONE").is_file():
                raise RuntimeError("Incomplete raw-NCC source run")
            summary = c.read_json(source_dir / "summary.json")
            for key in ("dice", "hd95", "asd"):
                if summary[key] != source[key]:
                    raise RuntimeError("Raw-NCC source metric mismatch")
            if source["log_jdet_std_fg"] != summary["regularity"]["log_jdet_std_fg"]:
                raise RuntimeError("Raw-NCC regularity mismatch")
            prepared = c.read_json(Path(prior_state["results"][f"prepare_rank{rank:03d}"]) / "summary.json")
            if summary["prefix_sha256"] != prepared["prefix_norm_sha256"]:
                raise RuntimeError("Raw-NCC prefix mismatch")
            if summary["source_hashes"][str(SWEEP / "common.py")] != sha256_file(SWEEP / "common.py"):
                raise RuntimeError("Accuracy/regularity implementation differs from raw-NCC fits")
            if (source["fixed_id"], source["moving_id"]) != (batch["fixed_id"], batch["moving_id"]):
                raise RuntimeError("Raw-NCC case mismatch")
            rows.append({"group": f"Raw NCC λ{alpha:g}", "alpha": alpha, "case_rank": rank,
                         "fixed_id": source["fixed_id"], "moving_id": source["moving_id"], "direction": "moving_to_fixed",
                         **{k: source[k] for k in METRICS}, "source_path": str(source_dir / "summary.json"),
                         "source_sha256": sha256_file(source_dir / "summary.json")})
        del batch
        torch.cuda.empty_cache()
    if args.smoke:
        c.write_json(args.out_root / "SMOKE_PASSED.json", {"rank": 0, "rows": rows, "time": c.now_text()})
        return
    if sha256_file(source_report) != report_hash:
        raise RuntimeError("Source report changed during evaluation")
    means = summarize(rows)
    c.write_csv(args.out_root / "case_metrics.csv", rows)
    c.write_csv(args.out_root / "mean_std.csv", means)
    c.write_json(args.out_root / "results.json", {"case_metrics": rows, "mean_std": means, "manifest": manifest})
    make_plot(means, args.out_root)
    make_plot(means, args.out_root, xkey="jdet_le0_pct_fg")
    lines = ["# OASIS first10: raw-NCC sweep, VFA and SITReg", "",
             "Same seed3 shuffled first10 pairs. Original saved best-Dice VFA/SITReg fields, not B-spline fits.", "",
             "## Means and across-case sample SD", "",
             "| Method | Dice | HD95 (mm) | ASD (mm) | Foreground SD(log J) | Foreground J <= 0 (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in means:
        if r["subset"] == "ranks_0_9":
            values = [f"{r[k + '_mean']:.6f} ± {r[k + '_std']:.6f}" for k in METRICS[:-1]]
            lines.append("| " + r["group"] + " | " + " | ".join(values) + " |")
    lines += ["", "![ARC plots](arc_sdlogj_first10.png)", "", ELLIPSE_DEFINITION + ".",
              "Each x coordinate is within-case foreground SD(log J), averaged over10 cases.",
              "Ellipse x radius is the sample SD of those10 case-level values, not within-image SD again.",
              "SD(log J) uses only positive-J fixed-image-positive voxels and does not quantify folding by itself.",
              "![Folding trade-off](tradeoff_folding_first10.png)", "",
              "Raw-NCC first10 is exploratory test-case tuning, not an unbiased held-out comparison.",
              "The curve contains lambda30/100/300/1000 only: lower lambdas have no complete first10 cohort.",
              "Per-case rows, ranks1–9 descriptive means, source hashes and full precision are in CSV/JSON."]
    (args.out_root / "report.md").write_text("\n".join(lines) + "\n")
    c.write_json(args.out_root / "ACCEPTED.json", {"complete": True, "groups": 6, "cases_per_group": 10,
                 "fresh_baseline_evaluations": 20, "raw_ncc_rows_reused": 40,
                 "vfa_matches_previous_metrics": True, "ellipse_definition": ELLIPSE_DEFINITION})


if __name__ == "__main__":
    args = parse_args()
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Use the approved container UID:GID1009:1020")
    try:
        run(args)
    except Exception as exc:
        c.write_json(args.out_root / "FAILED.json", {"time": c.now_text(), "error": repr(exc)})
        raise
