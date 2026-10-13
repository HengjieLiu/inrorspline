#!/usr/bin/env python
"""Read-only saved-field analysis: three foreground ARC figures per DL model.

Training used full-volume Grad3d L2, not a masked regularizer. The new foreground
metric retains its voxel-unit, component/axis-mean, forward-valid definition,
restricted to edges with BOTH endpoints in fixed raw image > 0. No crop for
diffusion; accepted Jacobian metrics retain their canonical two-voxel crop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np
import oasis_sitreg_ncc_arc_lines_top as old

REPO = Path(__file__).resolve().parents[2]
REPORTS = {
    "sitreg": "oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1",
    "fedda": "oasis_seed3_first100_fedda_gncc_vs_lncc_v1",
}
NAMES = {"sitreg": "SITReg", "fedda": "FeDDA-1MF"}
AXES = {
    "diffusion_l2_fg": ("diffusion", "Foreground diffusion energy ↓"),
    "log_jdet_std_fg": ("stdlogj", "Foreground SD(log J) ↓"),
    "jdet_le0_pct_fg": ("folding", "Foreground folding (%) ↓"),
}
METRICS = (*old.METRICS, "diffusion_l2_fg", "diffusion_l2_full")
SHAPE = (160, 224, 192)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def diffusion(field, mask):
    """Channel-first voxel DVF; axis/component mean, FP64 scalar accumulation."""
    if field.ndim != 4 or field.shape[0] != 3 or mask.shape != field.shape[1:]:
        raise ValueError("Require channel-first 3D displacement and matching mask")
    if mask.dtype != bool or not np.isfinite(field).all():
        raise ValueError("Require boolean mask and finite field")
    full, fg, counts = [], [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        edges = mask[tuple(lo)] & mask[tuple(hi)]
        count = int(edges.sum())
        if not count:
            raise ValueError("No valid foreground edges on an axis")
        delta = np.diff(field, axis=axis + 1)
        np.square(delta, out=delta)
        full.append(float(delta.mean(dtype=np.float64)))
        fg.append(float(np.sum(delta, where=edges[None], dtype=np.float64) / (3 * count)))
        counts.append(count)
    return dict(diffusion_l2_full=float(np.mean(full)), diffusion_l2_fg=float(np.mean(fg)),
                foreground_edges_by_array_axis=counts)


def load_mask(path):
    payload = Path(path).read_bytes()
    source = nib.Nifti1Image.from_bytes(gzip.decompress(payload))
    if nib.aff2axcodes(source.affine) != ("L", "I", "A"):
        raise ValueError("Unexpected raw image orientation")
    if not np.allclose(source.header.get_zooms()[:3], 1):
        raise ValueError("Expected 1mm spacing")
    raw = np.asarray(source.dataobj, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input image")
    mask = np.ascontiguousarray(np.flip(np.transpose(raw, (0, 2, 1)), (0, 2)) > 0)
    canonical = np.asarray(nib.as_closest_canonical(source).dataobj) > 0
    if mask.shape != SHAPE or not np.array_equal(mask, canonical):
        raise ValueError("Foreground orientation/geometry mismatch")
    return mask, hashlib.sha256(payload).hexdigest()


def checked_read(path, inputs, expected=None):
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Input checksum mismatch: " + str(path))
    inputs[str(path)] = digest
    return read(path)


def collect_jobs(artifact, inputs):
    """Resolve fields from detailed evaluations, joined to accepted plot rows."""
    reports, jobs = {}, []
    pairs = None
    for model, name in REPORTS.items():
        root = artifact / "repo_experiments" / name
        accepted = checked_read(root / "ACCEPTED.json", inputs)
        if not accepted.get("complete") or not (root / "DONE").is_file():
            raise ValueError("Incomplete source report")
        for filename, digest in accepted["outputs_sha256"].items():
            if sha(root / filename) != digest:
                raise ValueError("Changed accepted report: " + filename)
        data = checked_read(root / "results.json", inputs, accepted["outputs_sha256"]["results.json"])
        old.validated_means(data)
        reports[model] = data
        for loss, group in old.SERIES.items():
            for key, _ in group:
                historical = loss == "lncc9" and key == "1"
                if historical and model == "sitreg":
                    path = artifact / "repo_experiments/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1/results.json"
                    source = checked_read(path, inputs)
                    details = [{**r, "rank": r["case_rank"], "field_path": r["source_path"],
                                "field_sha256": r["source_sha256"], "diffusion_l2_full": r["diffusion"]}
                               for r in source["case_metrics"] if r["representation"] == "SITReg"]
                elif historical:
                    path = artifact / "repo_experiments/oasis_seed3_first100_fedda_training_lambda_sweep_v1/historical_lambda1/per_case.json"
                    details = checked_read(path, inputs, data["input_sha256"][str(path)])
                else:
                    if model == "sitreg":
                        variant = "sitreg_gncc" if loss == "gncc" else "sitreg"
                        directory = artifact / "runs" / f"oasis_seed3_first100_{variant}_training_lambda_eval_v1" / f"{variant}_lambda_{key}_seed42"
                    elif loss == "gncc":
                        directory = artifact / "runs/oasis_seed3_first100_fedda_gncc_training_lambda_eval_v1" / f"fedda_1mf_gncc_lambda_{key}_seed42"
                    else:
                        directory = artifact / "runs/oasis_fedda_ncc9_diffusion_training_lambda_sweep_v1/first100" / f"fedda_1mf_lambda_{key}_seed42_attempt01"
                    if not (directory / "COMPLETED").is_file():
                        raise ValueError("Incomplete detailed evaluation")
                    manifest = checked_read(directory / "manifest.json", inputs)
                    if (manifest["direction"], manifest["foreground"], manifest["field_storage"]) != (
                            "moving_to_fixed", "fixed_raw_image_positive", "channel_first_zyx_voxel"):
                        raise ValueError("Evaluation convention mismatch")
                    details = checked_read(directory / "per_case.json", inputs)
                if sorted(r["rank"] for r in details) != list(range(100)):
                    raise ValueError("Missing/duplicate detailed cases")
                detailed = {r["rank"]: r for r in details}
                rows = sorted((r for r in data["per_case"] if (r["loss"], r["lambda_key"]) == (loss, key)), key=lambda r: r["rank"])
                ids = [(r["fixed_id"], r["moving_id"]) for r in rows]
                if pairs is None:
                    pairs = ids
                if pairs != ids:
                    raise ValueError("Cross-model case identity mismatch")
                for row in rows:
                    source = detailed[row["rank"]]
                    if (source["fixed_id"], source["moving_id"], source["direction"]) != (row["fixed_id"], row["moving_id"], "moving_to_fixed"):
                        raise ValueError("Field case/direction mismatch")
                    for metric, alternate in zip(old.METRICS, ("dice_mean", "hd95_mean", "asd_symmetric_mean", old.X, "jdet_le0_pct_fg")):
                        value = source.get(metric, source.get(alternate))
                        if value is None or not math.isclose(row[metric], value, abs_tol=1e-12, rel_tol=1e-12):
                            raise ValueError("Detailed metrics differ from accepted source")
                    jobs.append({**row, "model": model, "historical": historical,
                                 "field_path": source["field_path"], "field_sha256": source["field_sha256"],
                                 "reference_diffusion_l2_full": source.get("diffusion_l2_full")})
    assert len(jobs) == 1800
    return reports, jobs


def measure(job, mask, mask_sha, source_sha, out):
    cache = out / "cases" / f"{job['model']}_{job['loss']}_{job['lambda_key']}_{job['rank']:03d}.json"
    identity = {k: job[k] for k in ("model", "loss", "lambda_key", "rank", "fixed_id", "moving_id", "field_path", "field_sha256")}
    identity.update(mask_sha256=mask_sha, analysis_source_sha256=source_sha)
    if cache.exists():
        previous = read(cache)
        if previous.get("identity") != identity:
            raise ValueError("Cache provenance differs")
        if sha(job["field_path"]) != job["field_sha256"]:
            raise ValueError("Cached field changed")
        return previous["row"]
    payload = Path(job["field_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != job["field_sha256"]:
        raise ValueError("DVF checksum mismatch")
    field = np.ascontiguousarray(nib.Nifti1Image.from_bytes(gzip.decompress(payload)).dataobj, dtype=np.float32)
    if field.shape != (3, *SHAPE):
        raise ValueError("Wrong canonical saved DVF geometry")
    metrics = diffusion(field, mask)
    reference = job["reference_diffusion_l2_full"]
    if reference is not None and not math.isclose(reference, metrics["diffusion_l2_full"], rel_tol=2e-6, abs_tol=1e-8):
        raise ValueError("Full-volume diffusion failed parity with canonical metric")
    row = {**job, **metrics, "mask_sha256": mask_sha,
           "foreground_voxels": int(mask.sum()),
           "diffusion_full_parity_abs_difference": None if reference is None else abs(reference - metrics["diffusion_l2_full"])}
    write_json(cache, dict(identity=identity, row=row))
    return row


def summarize(rows):
    result = {}
    for model in REPORTS:
        result[model] = {}
        for loss, group in old.SERIES.items():
            for key, alpha in group:
                cases = [r for r in rows if (r["model"], r["loss"], r["lambda_key"]) == (model, loss, key)]
                if sorted(r["rank"] for r in cases) != list(range(100)):
                    raise ValueError("Require all 100 unique cases per operating point")
                row = dict(model=model, loss=loss, lambda_key=key, alpha=alpha, n=100,
                           historical=loss == "lncc9" and key == "1")
                for metric in METRICS:
                    values = [r[metric] for r in cases]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError("Nonfinite metric")
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_std"] = statistics.stdev(values)
                result[model][loss, key] = row
    return result


def limits(rows, metric):
    lo = min(r[metric + "_mean"] - r[metric + "_std"] for r in rows)
    hi = max(r[metric + "_mean"] + r[metric + "_std"] for r in rows)
    pad = max(hi - lo, 1e-9) * .11
    return max(0, lo - pad), hi + pad


def build_figure(model, means, xmetric, all_means):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey="col")
    for ri in range(2):
        for ci, (metric, title) in enumerate(old.Y):
            ax = axes[ri, ci]
            for loss, group in old.SERIES.items():
                color = old.COLORS[loss]
                for (ka, _), (kb, _) in zip(group, group[1:]):
                    ax.plot([means[loss, k][xmetric + "_mean"] for k in (ka, kb)],
                            [means[loss, k][metric + "_mean"] for k in (ka, kb)],
                            color=color, lw=2.2, ls="--" if loss == "lncc9" and "1" in (ka, kb) else "-", zorder=3)
                for key, _ in group:
                    r = means[loss, key]
                    x, y = r[xmetric + "_mean"], r[metric + "_mean"]
                    if ri:
                        ax.add_patch(Ellipse((x, y), 2 * r[xmetric + "_std"], 2 * r[metric + "_std"],
                                             facecolor=color, edgecolor=color, alpha=.13, lw=1.1, zorder=1))
                    # A zero-SD folding point has a degenerate ellipse: show its vertical extent.
                    if ri and r[xmetric + "_std"] == 0:
                        ax.vlines(x, y - r[metric + "_std"], y + r[metric + "_std"], color=color, alpha=.35, lw=2)
                    ax.plot(x, y, marker="D" if r["historical"] else "o" if loss == "gncc" else "s",
                            color=color, markerfacecolor="white" if r["historical"] else color,
                            ms=6.5, markeredgewidth=1.4, ls="none", zorder=4, clip_on=False)
                    label = "1/3" if key == "1over3" else key
                    ax.annotate("λ=" + label + ("*" if r["historical"] else ""), (x, y),
                                xytext=(6, 9) if loss == "gncc" else (6, -16), textcoords="offset points",
                                fontsize=8.5, color=color, zorder=5)
            ax.set_title(f"{'abcdef'[ri * 3 + ci]}) {title}", loc="left", fontsize=13, fontweight="semibold")
            ax.set_xlabel(AXES[xmetric][1], fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_xlim(*limits(list(means.values()), xmetric))
            ax.set_ylim(*limits(all_means, metric))
            ax.grid(alpha=.18, lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelbottom=True)
    handles = [Line2D([], [], color=old.COLORS["gncc"], marker="o", lw=2.2, label="1−GNCC"),
               Line2D([], [], color=old.COLORS["lncc9"], marker="s", lw=2.2, label="1−LNCC² (9³ window)"),
               Line2D([], [], color=old.COLORS["lncc9"], marker="D", markerfacecolor="white", ls="--",
                      label="LNCC² λ=1* · historical checkpoint")]
    fig.suptitle(f"{NAMES[model]} · OASIS first100 · {AXES[xmetric][1].removesuffix(' ↓')}", fontsize=19, y=.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .948), ncol=3, frameon=False, fontsize=11)
    fig.text(.5, .892, "TOP: mean curves only", ha="center", fontsize=12, fontweight="semibold")
    fig.text(.5, .486, "BOTTOM: same curves with ±1 across-case SD ellipses", ha="center", fontsize=12, fontweight="semibold")
    fig.text(.5, .044, "100 paired cases/point · Marginal sample SD (ddof=1), not confidence/covariance regions · Linear axes; negative SD regions clipped at zero.", ha="center", fontsize=9.5)
    foot = {"diffusion_l2_fg": "Foreground: fixed raw image > 0; both edge endpoints inside. Voxel-unit forward differences; component/axis means. Training used UNMASKED full-volume diffusion.",
            "log_jdet_std_fg": "Foreground: fixed raw image > 0; canonical crop2; positive Jacobians only. Checkpoints selected by validation Dice; no new training.",
            "jdet_le0_pct_fg": "Foreground: fixed raw image > 0; canonical crop2; 100 × fraction(J ≤ 0). Exact zero retained; zero horizontal SD shown as a vertical line."}[xmetric]
    fig.text(.5, .018, foot, ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.06, right=.985, bottom=.12, top=.85, wspace=.26, hspace=.51)
    return fig, axes


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out = args.out.resolve()
    if out.is_relative_to(REPO) or out == args.artifact_root.resolve():
        raise ValueError("Use a distinct external artifact directory")
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if (out / "DONE").exists():
        raise ValueError("Refuse to overwrite completed analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    started = time.time()
    inputs = {str(Path(__file__)): sha(__file__), str(Path(old.__file__)): sha(old.__file__)}
    reports, jobs = collect_jobs(args.artifact_root, inputs)
    if args.smoke:
        jobs = [j for j in jobs if j["rank"] == 0]
    masks, mask_hashes = {}, {}
    for fixed in sorted({j["fixed_id"] for j in jobs}):
        path = args.data_root / "test" / f"img{fixed}.nii.gz"
        masks[fixed], mask_hashes[fixed] = load_mask(path)
        inputs[str(path)] = mask_hashes[fixed]
    source_sha = sha(__file__)
    provenance = dict(source_sha256=inputs, source_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        training_regularizer="unmasked full-volume Grad3d L2, voxel units, axis/component means",
        foreground_diffusion="fixed raw image > 0; forward valid; BOTH edge endpoints; no boundary-crossing edges; no extra crop; component mean then equal axis mean",
        foreground_jacobian="unchanged accepted fixed raw image > 0; canonical crop2; std(log J) on positive J; folding=100*fraction(J<=0)",
        computation="CPU only; saved fields; FP32 forward differences and squares, FP64 accumulation; no lambda weighting",
        plot="six 2x3 figures; top lines; bottom marginal +/-1 sample SD; linear axes; exact folding zeros retained; common accuracy limits across models",
        numpy=np.__version__, nibabel=nib.__version__, started_unix=started)
    write_json(out / "manifest.json", provenance)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(measure, j, masks[j["fixed_id"]], mask_hashes[j["fixed_id"]], source_sha, out) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(jobs):
                progress = dict(completed=len(rows), total=len(jobs), elapsed_seconds=time.time() - started)
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    rows.sort(key=lambda r: (r["model"], r["loss"], r["alpha"], r["rank"]))
    write_json(out / "per_case.json", rows)
    write_csv(out / "per_case.csv", rows)
    if args.smoke:
        write_json(out / "SMOKE_PASSED.json", dict(cases=len(rows), all_fields_hash_verified=True,
                   full_diffusion_max_parity_error=max(r["diffusion_full_parity_abs_difference"] or 0 for r in rows)))
        return
    means = summarize(rows)
    flattened = [r for group in means.values() for r in group.values()]
    # Existing accuracy/Jacobian means and sample SD must be reproduced exactly.
    for model in REPORTS:
        for row in reports[model]["means"]:
            actual = means[model][row["loss"], row["lambda_key"]]
            for metric in old.METRICS:
                for suffix in ("_mean", "_std"):
                    if not math.isclose(row[metric + suffix], actual[metric + suffix], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Original plotted statistics changed")
    write_csv(out / "mean_std.csv", flattened)
    write_json(out / "mean_std.json", flattened)
    import matplotlib.pyplot as plt
    for model in REPORTS:
        for metric, (stem, _) in AXES.items():
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
                fig, _ = build_figure(model, means[model], metric, flattened)
                for ext in ("png", "pdf", "svg"):
                    fig.savefig(out / f"{model}_gncc_vs_lncc_{stem}_arc_2x3.{ext}", dpi=180, facecolor="white")
                plt.close(fig)
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("Input changed during analysis: " + path)
    write_json(out / "ACCEPTED.json", dict(complete=True, cases=1800, settings=18, figures=6,
        original_metrics_unchanged=True, all_fields_sha256_verified=True,
        full_diffusion_parity_checked=sum(r["reference_diffusion_l2_full"] is not None for r in rows),
        full_diffusion_max_parity_error=max(r["diffusion_full_parity_abs_difference"] or 0 for r in rows),
        elapsed_seconds=time.time() - started,
        outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "ACCEPTED.json"}))
    (out / "DONE").write_text("All 1800 fields validated; six foreground ARC figures complete.\n")
    print("COMPLETE " + str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use 1 to 16 CPU workers")
    run(args)


if __name__ == "__main__":
    main()
