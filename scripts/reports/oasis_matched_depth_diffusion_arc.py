#!/usr/bin/env python
"""Posthoc final-displacement foreground diffusion for accepted P80/P79 ARC.

Never change registration sources/results. Training diffusion was unmasked;
the SVF arm penalized velocity, not the displacement measured here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np
import oasis_first100_five_family_dual_arc as arc
import oasis_ncc_three_regularity_arc as fd

REPO = Path(__file__).resolve().parents[2]
X = "diffusion_l2_fg"
MEASURES = (*arc.METRICS, X, "diffusion_l2_full")
VARIANT = "oasis_seed3_first100_bspline842_8421_foreground_diffusion_arc_v1"
CONVENTION = (
    "Final total displacement, voxel units; forward-valid squared differences; "
    "component mean then equal axis mean; both edge endpoints in fixed raw image > 0; "
    "no crop, no lambda weighting; FP32 differences/squares, FP64 accumulation. "
    "Posthoc diagnostic, NOT the unmasked training loss; SVF training penalized velocity."
)


def read_checked(path, inputs, expected=None):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Changed source: {path}")
    inputs[str(path)] = digest
    return json.loads(payload)


def identity(row):
    return tuple(row[k] for k in ("depth", "family", "alpha", "case_rank"))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def check_metrics(row, source, alternate=False):
    if (row["fixed_id"], row["moving_id"], row["direction"]) != (
            source["fixed_id"], source["moving_id"], source["direction"]):
        raise ValueError("Field identity/direction mismatch")
    keys = ("dice_mean", "hd95_mean", "asd_symmetric_mean", *arc.METRICS[3:]) if alternate else arc.METRICS
    for metric, key in zip(arc.METRICS, keys):
        if not math.isclose(row[metric], source[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Accuracy/regularity source mismatch: {metric}")


def collect(args):
    inputs = {}
    accepted = read_checked(args.report / "ACCEPTED.json", inputs)
    if (accepted["complete"], accepted["candidate_count"], accepted["failures"]) != (True, 2100, 0):
        raise ValueError("Require accepted full100 matched-prefix release")
    data = read_checked(args.report / "results.json", inputs, accepted["output_sha256"]["results.json"])
    split = REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    inputs[str(split)] = arc.sha(split)
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", split.read_text())[:100]
    manifest = read_checked(args.run_root / "launch_manifest.json", inputs)
    prefixes = {v["summary_path"]: v for v in manifest["prefixes"].values()}
    dl_marker = read_checked(Path(manifest["dl_report"]) / "plot_manifest.json", inputs)
    dl_paths = dict(dl_marker["source_sha256"])
    # SITReg detailed sources are recorded by the intermediate accepted report.
    for path, digest in list(dl_paths.items()):
        if path.endswith("results.json") and "with_sitreg" in path:
            nested = read_checked(path, inputs, digest)
            dl_paths.update(nested.get("input_sha256", {}))
    dl_details = {}
    for family in ("SITReg", "VFA"):
        for alpha in arc.DL_LAMBDAS:
            if alpha == 1:
                continue
            key = "1over3" if alpha == 1/3 else f"{alpha:g}"
            candidates = [p for p in dl_paths if p.endswith("/per_case.json") and
                          f"/{family.lower()}_lambda_{key}_" in p]
            if family == "SITReg" and not candidates:
                folder = args.artifact_root / "runs/oasis_seed3_first100_sitreg_training_lambda_eval_v1" / f"sitreg_lambda_{key}_seed42"
                candidates = [str(folder / "per_case.json")]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous DL detailed source: {family}, {alpha}: {candidates}")
            path = Path(candidates[0])
            if not (path.parent / "COMPLETED").is_file():
                raise ValueError("Incomplete DL evaluation")
            m = read_checked(path.parent / "manifest.json", inputs)
            if (m["direction"], m["foreground"], m["field_storage"]) != (
                    "moving_to_fixed", "fixed_raw_image_positive", "channel_first_zyx_voxel"):
                raise ValueError("Wrong DL field convention")
            rows = read_checked(path, inputs, dl_paths.get(str(path)))
            if sorted(r["rank"] for r in rows) != list(range(100)):
                raise ValueError("Incomplete DL ranks")
            for row in rows:
                dl_details[family, alpha, row["rank"]] = row
    historical = args.artifact_root / "repo_experiments/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1/results.json"
    hist = read_checked(historical, inputs)
    for row in hist["case_metrics"]:
        if row["representation"] in ("SITReg", "VFA"):
            dl_details[row["representation"], 1., row["case_rank"]] = row
    def resolve(row):
        if (row["fixed_id"], row["moving_id"]) != pairs[row["case_rank"]] or row["direction"] != "moving_to_fixed":
            raise ValueError("Wrong split/direction")
        if row["depth"] == "DL":
            source = dl_details[row["family"], row["alpha"], row["case_rank"]]
            historical = row["alpha"] == 1
            check_metrics(row, source, alternate=not historical)
            path = source["source_path" if historical else "field_path"]
            digest = source["source_sha256" if historical else "field_sha256"]
            full = source["diffusion" if historical else "diffusion_l2_full"]
            layout, input_hash = "channel_first_zyx_voxel", None
        else:
            summary_path = str(Path(row["run_dir"]) / "summary.json")
            if row["depth"] == "842":
                expected = prefixes[summary_path]["summary_sha256"]
            elif row["depth"] == "8421":
                receipt = read_checked(Path(row["run_dir"]) / "ACCEPTED.json", inputs)
                if not receipt["passed"]:
                    raise ValueError("Unaccepted new registration")
                expected = receipt["summary_sha256"]
            else:
                raise ValueError("Unexpected depth")
            source = read_checked(summary_path, inputs, expected)
            check_metrics(row, source)
            path, digest = source["dvf_path"], source["dvf_sha256"]
            if row.get("dvf_sha256", digest) != digest:
                raise ValueError("New output DVF hash differs")
            full = source.get("diffusion_native", source.get("diffusion"))
            if full is None:
                raise ValueError("Missing full displacement diffusion reference")
            layout = "normalized_grid_xyz"
            input_hash = source["input_hashes"][f"img{row['fixed_id']}.nii.gz"]
        return {**row, "field_path": path, "field_sha256": digest, "field_layout": layout,
                "reference_diffusion_l2_full": full, "fixed_input_sha256": input_hash}
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, job in enumerate(executor.map(resolve, data["per_case"]), 1):
            jobs.append(job)
            if i % 1000 == 0:
                print(f"Source metadata verified: {i}/5200", flush=True)
    if len(jobs) != 5200 or len({identity(r) for r in jobs}) != 5200:
        raise ValueError("Require 5200 unique rows (two depths plus shared DL)")
    for depth in ("842", "8421"):
        arc.summarize([r for r in jobs if r["depth"] in (depth, "DL")], pairs)
    return jobs, inputs, data


def normalized_to_voxel(field):
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("Expected DHW3 normalized xyz field")
    scales = np.asarray([(n - 1)/2 for n in field.shape[:3][::-1]], dtype=np.float32)
    return np.ascontiguousarray(np.moveaxis(field * scales, -1, 0))


def measure(job, mask, mask_hash, source_hash, out):
    key = hashlib.sha256(json.dumps(identity(job)).encode()).hexdigest()[:20]
    path = out / "cases" / f"{key}.json"
    provenance = {k: job[k] for k in ("field_path", "field_sha256", "field_layout")}
    provenance.update(case=list(identity(job)), mask_sha256=mask_hash, analysis_sha256=source_hash)
    if job["fixed_input_sha256"] not in (None, mask_hash):
        raise ValueError("Fixed mask input changed")
    payload = Path(job["field_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != job["field_sha256"]:
        raise ValueError("Field checksum mismatch")
    if path.exists():
        cached = arc.read(path)
        if cached["provenance"] != provenance:
            raise ValueError("Cache provenance mismatch")
        return cached["row"]
    if job["field_layout"] == "normalized_grid_xyz":
        field = np.load(io.BytesIO(payload), allow_pickle=False)
        if field.shape != (*fd.SHAPE, 3) or field.dtype != np.float32:
            raise ValueError("Invalid normalized DVF geometry/dtype")
        field = normalized_to_voxel(field)
    elif job["field_layout"] == "channel_first_zyx_voxel":
        image = nib.Nifti1Image.from_bytes(gzip.decompress(payload))
        field = np.ascontiguousarray(image.dataobj, dtype=np.float32)
    else:
        raise ValueError("Unknown displacement units/layout")
    del payload
    if field.shape != (3, *fd.SHAPE):
        raise ValueError("Invalid voxel DVF geometry")
    metrics = fd.diffusion(field, mask)
    reference = job["reference_diffusion_l2_full"]
    if not math.isclose(reference, metrics["diffusion_l2_full"], rel_tol=2e-6, abs_tol=1e-8):
        raise ValueError(f"Full diffusion parity mismatch {identity(job)}: {reference} versus {metrics['diffusion_l2_full']}")
    row = {**job, **metrics, "mask_sha256": mask_hash, "foreground_voxels": int(mask.sum()),
           "full_parity_abs_error": abs(reference-metrics["diffusion_l2_full"])}
    fd.write_json(path, dict(provenance=provenance, row=row))
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["depth"], row["family"], row["alpha"]].append(row)
    result = []
    for (depth, family, alpha), group in sorted(groups.items()):
        if sorted(r["case_rank"] for r in group) != list(range(100)):
            raise ValueError("Incomplete/duplicate 100-case group")
        r = dict(depth=depth, family=family, alpha=alpha, n=100)
        for metric in MEASURES:
            values = [g[metric] for g in group]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite aggregate")
            r[metric+"_mean"], r[metric+"_sd"] = statistics.mean(values), statistics.stdev(values)
        result.append(r)
    return result


def build_figure(depth, means, source_axes, xlim):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    fig, axes = arc.plt.subplots(2, 3, figsize=(21, 12))
    fig.subplots_adjust(left=.065, right=.982, bottom=.19, top=.79, wspace=.27, hspace=.50)
    schedule = "8→4→2" if depth == "842" else "8→4→2→cps1"
    fig.suptitle(f"OASIS · 100 paired cases · Foreground diffusion energy · {schedule}", fontsize=24, weight="semibold", y=.98)
    handles = []
    for family, (name, color, marker, style) in arc.STYLES.items():
        if family == "displacement":
            name = f"Displacement B-spline {schedule}"
        handles.append(Line2D([], [], label=name, color=color, marker=marker, linestyle=style, lw=2.7, markersize=8))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3,
               frameon=False, fontsize=16, columnspacing=2.2, handlelength=3.1)
    fig.text(.5, .824, "TOP: accuracy zoom · means only     |     BOTTOM: full sweep · means ± 1 sample SD", ha="center", fontsize=16)
    diagnostic = []
    for i, ax in enumerate(axes.flat):
        band, col = divmod(i, 3)
        metric, label = list(arc.Y_LABELS.items())[col]
        for family, (_, color, marker, style) in arc.STYLES.items():
            group = sorted((r for r in means if r["family"] == family), key=lambda r: r["alpha"], reverse=True)
            xx, yy = [r[X+"_mean"] for r in group], [r[metric+"_mean"] for r in group]
            if band:
                for x, y, r in zip(xx, yy, group):
                    ax.add_patch(Ellipse((x, y), 2*r[X+"_sd"], 2*r[metric+"_sd"],
                                        facecolor=color, edgecolor=color, alpha=.10, linewidth=1, zorder=1))
            ax.plot(xx, yy, color=color, marker=marker, linestyle=style, lw=2.5, markersize=7,
                    markeredgewidth=.6, markeredgecolor="white", zorder=4)
        ylim = source_axes[i]["ylim"]
        if not band:
            ax.yaxis.set_major_locator(MultipleLocator(arc.TOP[metric][2]))
            count = sum(ylim[0] <= r[metric+"_mean"] <= ylim[1] for r in means)
            ax.text(.02, .025, f"Zoom: {count}/{len(means)} means in view", transform=ax.transAxes,
                    fontsize=11, color=".35", bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.set(xlim=xlim, ylim=ylim, ylabel=label, xlabel="Foreground displacement diffusion energy ↓")
        ax.set_title(f"{'abcdef'[i]}) {label}", loc="left", fontsize=18, pad=10)
        ax.grid(which="major", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        diagnostic.append(dict(row=band, metric=metric, xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                               xticks=ax.get_xticks().tolist(), yticks=ax.get_yticks().tolist()))
    fig.text(.5, .124, "λ: SITReg / VFA = {0.1, 1/3, 1, 3, 10}; each B-spline family = {0.01, 0.1, 0.3, 1, 3, 10, 100}.", ha="center", fontsize=14)
    fig.text(.5, .089, "Ellipses: marginal ±1 across-case sample SD (n = 100), not confidence regions. Bottom retains all 31 settings.", ha="center", fontsize=14)
    fig.text(.5, .057, "Final displacement · voxel-unit forward differences · both edge endpoints in fixed raw image > 0 · no λ weighting.", ha="center", fontsize=13)
    fig.text(.5, .025, "Post-hoc foreground metric: training diffusion was UNMASKED; refined SVF training regularized VELOCITY, not displacement.", ha="center", fontsize=12, color=".35")
    return fig, diagnostic


def run(args):
    out = args.out.resolve()
    if out.is_relative_to(REPO) or out == args.artifact_root.resolve():
        raise ValueError("Use a distinct external output directory")
    if (out / "ACCEPTED.json").exists() or (out.exists() and not args.resume):
        raise FileExistsError("Refuse to overwrite existing analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    started = time.time()
    jobs, inputs, source_data = collect(args)
    if args.smoke:
        jobs = [j for j in jobs if j["case_rank"] == 0]
    print(f"Validated provenance for {len(jobs)} fields", flush=True)
    for path in (Path(__file__), Path(fd.__file__), Path(arc.__file__)):
        inputs[str(path)] = arc.sha(path)
    source_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    masks, mask_hashes = {}, {}
    for fixed in sorted({j["fixed_id"] for j in jobs}):
        path = args.data_root / "test" / f"img{fixed}.nii.gz"
        masks[fixed], mask_hashes[fixed] = fd.load_mask(path)
        inputs[str(path)] = mask_hashes[fixed]
    fd.write_json(out / "manifest.json", dict(convention=CONVENTION, input_sha256=inputs,
        analysis_sha256=source_hash, source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        cpu_only=True, unchanged_accuracy=True, original_outputs_read_only=True,
        started_at=datetime.now(timezone.utc).isoformat()))
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(measure, j, masks[j["fixed_id"]], mask_hashes[j["fixed_id"]], source_hash, out): j for j in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(dict(case=identity(futures[future]), error=repr(exc)))
                fd.write_json(out / "failures.json", failures)
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time()-started
                progress = dict(completed=len(rows), failed=len(failures), total=len(jobs), elapsed_seconds=elapsed,
                                eta_seconds=elapsed/i*(len(jobs)-i))
                fd.write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    rows.sort(key=identity)
    fd.write_json(out / "per_case.json", rows)
    fd.write_json(out / "failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} field analyses failed; preserved for review; no partial plots")
    write_csv(out / "per_case.csv", rows)
    if args.smoke:
        fd.write_json(out / "SMOKE_PASSED.json", dict(cases=len(rows), full_parity_max_abs_error=max(r["full_parity_abs_error"] for r in rows)))
        return
    means = summarize(rows)
    if len(means) != 52 or len(rows) != 5200:
        raise ValueError("Wrong complete matrix")
    original = {(r["depth"], r["family"], r["alpha"]): r for r in source_data["summary"]}
    checks = 0
    for r in means:
        before = original[r["depth"], r["family"], r["alpha"]]
        for k in arc.METRICS:
            for suffix in ("_mean", "_sd"):
                if not math.isclose(r[k+suffix], before[k+suffix], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Original accuracy/regularity aggregation changed")
                checks += 1
    write_csv(out / "summary.csv", means)
    fd.write_json(out / "results.json", dict(summary=means, per_case=rows, convention=CONVENTION))
    maximum = max(r[X+"_mean"]+r[X+"_sd"] for r in means)
    xlim = (0, maximum*1.06)
    axes = {}
    with arc.plt.rc_context({"font.family":"DejaVu Sans", "font.size":15, "axes.labelsize":16,
                             "xtick.labelsize":14, "ytick.labelsize":14, "pdf.fonttype":42, "svg.fonttype":"none"}):
        for depth in ("842", "8421"):
            subset = [r for r in means if r["depth"] in (depth, "DL")]
            source_axes = source_data["plot"]["axes"][depth+"_fg_stdlogj"]
            fig, axes[depth] = build_figure(depth, subset, source_axes, xlim)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(out/f"arc_{depth}_fg_diffusion_2x3.{ext}", dpi=180, facecolor="white")
            arc.plt.close(fig)
    if axes["842"] != axes["8421"]:
        raise ValueError("Depth axes/ticks differ")
    (out / "README.md").write_text(
        "# Foreground displacement diffusion ARC: 842 and 8421\n\n"
        "Two 2×3 figures, 31 settings × 100 paired cases in each; 5,200 unique fields (DL shared). "
        "Top: zoomed accuracy means; bottom: marginal ±1 sample SD, not confidence/covariance regions. "
        "Both depths share axes/ticks, colors and line styles; y limits equal the accepted SD(log J) figures.\n\n"
        + CONVENTION + "\n\n"
        "For each spatial axis a, E_a contains forward-valid edges whose BOTH endpoints are foreground. "
        "R_FG(u) = (1/3) sum_a [sum_(x in E_a,c) (u_c(x+e_a)-u_c(x))² / (3 |E_a|)]. "
        "Each case contributes equally to cohort means and sample SD (ddof=1). Unit 1mm voxels; no segmentation mask.\n\n"
        "All fields and fixed inputs were checksum-verified. Unmasked recomputation matched existing full-volume displacement diffusion "
        "within rtol=2e-6/atol=1e-8. Original Dice/HD95/symmetric ASD and Jacobian values remain unchanged. "
        "No training, registration, GPU computation, or historical output changes. First-ten-informed exploratory analysis.\n")
    def unchanged(item):
        path, digest = item
        if arc.sha(path) != digest:
            raise ValueError("Analysis source changed during processing: " + path)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(unchanged, inputs.items()))
    fd.write_json(out / "ACCEPTED.json", dict(complete=True, unique_fields=5200, settings=52,
        figures=2, plotted_settings_per_depth=31, cases_per_setting=100, source_statistic_checks=checks,
        all_field_hashes_verified=True, all_full_diffusion_parity_passed=True,
        full_diffusion_max_abs_error=max(r["full_parity_abs_error"] for r in rows),
        axes=axes, elapsed_seconds=time.time()-started, convention=CONVENTION,
        output_sha256={p.name:arc.sha(p) for p in out.iterdir() if p.is_file()},
        completed_at=datetime.now(timezone.utc).isoformat()))
    print("COMPLETE "+str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "run-root", "report", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        p.error("Use 1–16 CPU threads")
    run(args)


if __name__ == "__main__":
    main()
