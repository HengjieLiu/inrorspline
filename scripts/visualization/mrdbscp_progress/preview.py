"""Create the case-selection gate from accepted CP-BE warm outputs, on CPU only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np
from PIL import Image
import torch

from scripts.experiments.oasis_alpha_sweep import common as c
from .config import DEFAULT
from .geometry import preview_planes


# Standard named-color palette, label order 0..35, matching the reference demo.
# Rendering and geometry are implemented independently; no external repo import.
COLORS = ("black red lime blue yellow cyan magenta papayawhip mediumblue peru tan "
          "mediumaquamarine navy darkcyan seagreen mistyrose slateblue plum darksalmon "
          "brown snow mediumpurple orchid indigo lightpink mediumseagreen blanchedalmond "
          "bisque goldenrod teal rosybrown hotpink peachpuff burlywood chartreuse saddlebrown").split()
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-.5, 36.5), len(COLORS))
REPO = Path(__file__).resolve().parents[3]
MODULE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_output(output, artifact_root):
    output, artifact_root = output.resolve(), artifact_root.resolve()
    if output == artifact_root or not output.is_relative_to(artifact_root):
        raise ValueError("Output must be a new child of ARTIFACT_ROOT")
    if output.is_relative_to(REPO) or artifact_root.is_relative_to(REPO):
        raise ValueError("Generated artifacts cannot be written inside the source repository")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    return output


def warm_rows(rows):
    selected = [r for r in rows if r["variant"] == "cpbe" and r["condition"] == "warm"
                and r["id"].startswith("headline_")]
    selected.sort(key=lambda r: int(r["rank"]))
    if [int(r["rank"]) for r in selected] != list(range(10)):
        raise ValueError("Require exactly one CP-BE headline warm output for each rank 0..9")
    return selected


def validate_field(path, expected_digest):
    field = np.load(path, mmap_mode="r", allow_pickle=False)
    if field.dtype != np.float32 or field.shape != (*c.IMAGE_SHAPE, 3) or not field.flags.c_contiguous:
        raise ValueError(f"Invalid DVF format: {path}")
    # The evaluator hashes array bytes, not the .npy header.
    h = hashlib.sha256()
    for slab in field:
        if not np.isfinite(slab).all():
            raise ValueError(f"Nonfinite DVF: {path}")
        h.update(memoryview(np.ascontiguousarray(slab)))
    if h.hexdigest() != expected_digest:
        raise ValueError(f"Field does not match evaluated bytes: {path}")
    return field


def image_metadata(path):
    image = nib.load(str(path))
    if image.shape != (160, 192, 224) or nib.aff2axcodes(image.affine) != ("L", "I", "A"):
        raise ValueError(f"Input does not match the loader's LIA reorientation: {path}")
    if not np.allclose(image.header.get_zooms()[:3], (1., 1., 1.)):
        raise ValueError("Preview assumes the accepted 1mm OASIS geometry")
    return dict(path=str(path), sha256=digest(path), shape=list(image.shape),
                input_axis_codes=list(nib.aff2axcodes(image.affine)),
                affine=image.affine.tolist(), spacing_mm=list(map(float, image.header.get_zooms()[:3])))


def show(ax, value, kind="image", title=None, orientation=False):
    settings = dict(origin="lower", interpolation="nearest", aspect="equal")
    if kind == "labels":
        settings.update(cmap=CMAP, norm=NORM)
    else:
        settings.update(cmap="gray", vmin=0., vmax=.75 if kind == "image" else 1.)
    ax.imshow(value, **settings)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=15, pad=7)
    if orientation:
        for x, y, label in ((.04, .5, "R"), (.96, .5, "L"), (.5, .97, "A"), (.5, .03, "P")):
            ax.text(x, y, label, transform=ax.transAxes, color="white", fontsize=9,
                    ha="center", va="center", bbox=dict(facecolor="black", alpha=.55, pad=1, edgecolor="none"))
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_case(planes, row, output, slice_index):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 4, left=.035, right=.975, bottom=.12, top=.86, wspace=.13, hspace=.15)
    for column, (key, label) in enumerate((("moving", "Moving"), ("fixed", "Fixed"), ("warped", "Warped"))):
        show(fig.add_subplot(gs[0, column]), planes[key], title=label, orientation=True)
        show(fig.add_subplot(gs[1, column]), planes[key + "_seg"], "labels")
    show(fig.add_subplot(gs[0, 3]), planes["grid"], "grid", "Warped reference grid")
    text_ax = fig.add_subplot(gs[1, 3])
    text_ax.axis("off")
    text_ax.text(0., .94,
        f"Whole-volume metrics\n\n3D Dice       {row['mean_dice']:.4f}\n"
        f"HD95          {row['hd95_mm']:.3f} mm\n"
        f"FG SDlogJ    {row['log_jdet_std_fg']:.4f}\n"
        f"FG folding   {row['jdet_le0_pct_fg']:.3f}%\n\n"
        f"Warm time   {row['elapsed_sec']:.3f} s\n"
        f"Stage steps  {' / '.join(map(str, row['steps']))}\n\n"
        "Metrics reused from accepted\nfull-volume evaluation, not this slice.",
        fontsize=14, va="top", linespacing=1.5)
    fig.suptitle(f"Case {row['rank'] + 1:02d}  |  seed-3 rank {row['rank']}  |  "
                 f"Fixed {row['fixed_id']}  /  Moving {row['moving_id']}", fontsize=24, y=.966)
    fig.text(.5, .904, f"MR-D-BSCP fast 8-4-2 + control-space BE  |  Axial slice {slice_index} (zero-based)",
             ha="center", fontsize=16, color="#444444")
    fig.text(.04, .07, "MRI: grayscale, shared window [0, 0.75]  |  Labels: fixed 35-label palette  |  Grid: 8-voxel spacing",
             fontsize=13)
    fig.text(.04, .038, "Backward (pull) grid, not control lattice  |  CP-BE is not numerically equivalent to original BE  |  Case-selection preview",
             fontsize=12, color="#555555")
    fig.savefig(output, dpi=100)
    plt.close(fig)


def render_overview(cases, slices, path):
    fig = plt.figure(figsize=(20, 17), dpi=120, facecolor="white")
    outer = fig.add_gridspec(5, 2, left=.035, right=.985, top=.94, bottom=.04, wspace=.1, hspace=.35)
    for i, row in enumerate(cases):
        inner = outer[i // 2, i % 2].subgridspec(1, 5, wspace=.04)
        for col, (key, title, kind) in enumerate((("moving", "Moving", "image"), ("fixed", "Fixed", "image"),
                  ("warped", "Warped", "image"), ("warped_seg", "Warped labels", "labels"), ("grid", "Pull grid", "grid"))):
            ax = fig.add_subplot(inner[0, col])
            show(ax, slices[i][key], kind)
            if i < 2:
                ax.set_title(title, fontsize=11, pad=30)
        box = outer[i // 2, i % 2].get_position(fig)
        fig.text(box.x0, box.y1 + .005,
                 f"Case {i + 1:02d} / rank {row['rank']}   F{row['fixed_id']} / M{row['moving_id']}"
                 f"   Dice {row['mean_dice']:.3f}   warm {row['elapsed_sec']:.2f}s", fontsize=12)
    fig.suptitle("Choose a case  |  CP-BE fast 8-4-2  |  Seed-3 first10", fontsize=23, y=.99)
    fig.text(.5, .015, "Same central axial slice and fixed display settings for every case. Dice is whole-volume. Open individual panels for details.",
             ha="center", fontsize=13)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_html(cases, path):
    cards = []
    for r in cases:
        caption = html.escape(f"Case {r['rank'] + 1:02d} / rank {r['rank']} — Fixed {r['fixed_id']}, Moving {r['moving_id']}")
        cards.append(f'<section id="rank{r["rank"]}"><h2>{caption}</h2>'
                     f'<a href="{r["figure"]}"><img loading="lazy" src="{r["figure"]}" alt="{caption}"></a>'
                     f'<p>Choose by replying with <b>rank {r["rank"]}</b> (case {r["rank"] + 1:02d}). '
                     'This page does not submit a selection or start any job.</p></section>')
    navigation = ' '.join(f'<a href="#rank{i}">Case {i+1:02d}</a>' for i in range(10))
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>MR-D-BSCP: choose an animation case</title><style>'
        'body{font:18px system-ui;margin:24px auto;max-width:1440px;padding:0 16px;background:#f1f4f7;color:#182333}'
        'img{width:100%;height:auto}section{background:white;margin:24px 0;padding:18px;border-radius:10px}'
        'nav{position:sticky;top:0;background:#182333;padding:14px;z-index:1}nav a{color:white;margin:0 10px}'
        'a{color:#075d94}</style><h1>MR-D-BSCP fast + control-space BE</h1>'
        '<p>Case-selection gate: no optimization rerun and no animation capture yet. '
        'Existing accepted CP-BE warm results; ranks follow seed-3 first10 order.</p>'
        '<p>Axial: radiological display (left R, right L, top A). Metrics are 3D, computed offline. '
        'Small mean quality differences do not establish CP-BE equivalence to original BE.</p>'
        f'<nav>{navigation}</nav><a href="overview.png"><img src="overview.png" alt="All ten cases"></a>'
        + ''.join(cards) + '<p><a href="report.md">Report</a> · <a href="cases.csv">Case metrics</a> · '
        '<a href="manifest.json">Provenance</a></p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=DEFAULT.slice_index)
    args = parser.parse_args()
    if not 0 <= args.slice_index < c.IMAGE_SHAPE[2]:
        parser.error("Axial slice must be in 0..191")
    output = validate_output(args.output, Path(os.environ["ARTIFACT_ROOT"]))
    if torch.cuda.is_initialized():
        raise RuntimeError("Preview must not initialize CUDA")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    accept = json.loads((args.source_review / "acceptance.json").read_text())
    if accept.get("complete") is not True or not all(r["passed"] for r in accept["checks"]):
        raise ValueError("Source experiment is not accepted")
    with (args.source_review / "timings_metrics.csv").open(newline="") as stream:
        selected = warm_rows(list(csv.DictReader(stream)))
    evaluated = {r["id"]: r for r in json.loads((args.source_review / "evaluations.json").read_text())["rows"]}
    primary = {r["id"]: r for r in json.loads((args.source_run / "primary_rows.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    (output / "panels").mkdir()
    (output / "slices").mkdir()
    (output / "source_snapshot").mkdir()
    snapshot_sources = sorted(MODULE.glob("*.py")) + [MODULE / "README.md", Path(c.__file__), args.pair_file]
    sources = {}
    for source in snapshot_sources:
        rel = source.resolve().relative_to(REPO).as_posix()
        target = output / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[rel] = digest(source)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO, text=True)
    manifest = dict(status="in_progress", created_utc=datetime.now(timezone.utc).isoformat(),
        configuration={**DEFAULT.to_dict(), "slice_index": args.slice_index}, argv=sys.argv,
        repo_commit=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
        sources=sources, source_review=str(args.source_review), source_run=str(args.source_run),
        source_manifest_hashes={name: digest(args.source_review / name) for name in
                                ("acceptance.json", "evaluations.json", "timings_metrics.csv")},
        primary_rows_sha256=digest(args.source_run / "primary_rows.json"),
        pair_file_sha256=digest(args.pair_file), sorted_pairs_sha256=digest(args.oasis_root / "pair_test_200.txt"),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                     matplotlib=matplotlib.__version__, nibabel=nib.__version__,
                     device="cpu", uid=os.getuid(), gid=os.getgid(), cuda_initialized=False),
        display=dict(axis=2, index_zero_based=args.slice_index, transform="transpose then flip horizontal; origin lower",
                     screen_directions=dict(left="R", right="L", top="A", bottom="P"),
                     image_window=[0., .75], grid="3D pull-warped axial reference grid, 8 voxel spacing"),
        metric_policy="Reuse accepted whole-volume evaluation only after array SHA256 verification",
        next_action="Await explicit user case selection; do not start registration")
    (output / "dirty.patch").write_text(git("diff", "--binary", "HEAD"))
    write_json(output / "manifest.json", manifest)
    cases, slices, input_metadata = [], [], {}
    for source in selected:
        rank = int(source["rank"])
        record = primary[source["id"]]
        if (not record["clean"] or record["features"] != ["cpbe"]
                or record["backend"]["cudnn_benchmark"] is not False
                or record["backend"]["cudnn_deterministic"] is not True
                or record["dvf"] != source["dvf"] or int(record["rank"]) != rank):
            raise ValueError("Unclean or mismatched source configuration")
        field = validate_field(Path(source["dvf"]), evaluated[source["id"]]["dvf_sha256"])
        batch = c.load_oasis_case(rank=rank, device=torch.device("cpu"), oasis_root=args.oasis_root, pair_file=args.pair_file)
        affines = []
        for subject in (batch["fixed_id"], batch["moving_id"]):
            for prefix in ("img", "seg"):
                path = args.oasis_root / "test" / f"{prefix}{subject}.nii.gz"
                key = path.name
                if key not in input_metadata:
                    input_metadata[key] = image_metadata(path)
                affines.append(input_metadata[key]["affine"])
        if not all(np.allclose(affines[0], a) for a in affines):
            raise ValueError("Expected matching OASIS image/label affines")
        expected_name = f"disp_norm_{batch['fixed_id']}_{batch['moving_id']}.npy"
        if Path(source["dvf"]).name != expected_name:
            raise ValueError("Field IDs do not match pair file")
        row = {key: float(source[key]) for key in ("elapsed_sec", "mean_dice", "hd95_mm", "log_jdet_std_fg", "jdet_le0_pct_fg")}
        if not all(np.isfinite(v) for v in row.values()):
            raise ValueError("Nonfinite source metrics")
        e = evaluated[source["id"]]
        expected_metrics = dict(mean_dice=e["mean_dice"], hd95_mm=e["surface_metrics"]["hd95_mean"],
            log_jdet_std_fg=e["regularity"]["log_jdet_std_fg"],
            jdet_le0_pct_fg=e["regularity"]["jdet_le0_pct_fg"], elapsed_sec=record["elapsed_sec"])
        if not all(np.isclose(row[key], value, atol=1e-12, rtol=0) for key, value in expected_metrics.items()):
            raise ValueError("Metric/timing summary differs from original record")
        if json.loads(source["steps"]) != record["steps"] or int(e["rank"]) != rank:
            raise ValueError("Step counts or evaluation rank differ from original record")
        row.update(rank=rank, case_number=rank + 1, fixed_id=batch["fixed_id"], moving_id=batch["moving_id"],
            source_id=source["id"], dvf=source["dvf"], dvf_array_sha256=e["dvf_sha256"],
            steps=json.loads(source["steps"]), valid_dice_labels=int(np.isfinite(e["dice"]).sum()),
            figure=f"panels/rank{rank:02d}.png", slices=f"slices/rank{rank:02d}.npz")
        with torch.no_grad():
            planes = preview_planes(batch, field, args.slice_index, DEFAULT.grid_spacing)
        np.savez_compressed(output / row["slices"], **planes)
        render_case(planes, row, output / row["figure"], args.slice_index)
        cases.append(row)
        slices.append(planes)
        print(json.dumps(dict(completed=rank + 1, total=10, rank=rank, figure=row["figure"])), flush=True)
        del batch, field
    render_overview(cases, slices, output / "overview.png")
    render_html(cases, output / "index.html")
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    write_json(output / "cases.json", cases)
    manifest.update(status="awaiting_case_selection", cases=cases, input_metadata=input_metadata)
    manifest["outputs_sha256"] = {p.relative_to(output).as_posix(): digest(p) for p in
        [output / "overview.png", output / "index.html", output / "cases.csv", output / "cases.json"]
        + sorted((output / "panels").glob("*.png")) + sorted((output / "slices").glob("*.npz"))}
    write_json(output / "manifest.json", manifest)
    checks = dict(ten_unique_cpbe_warm_cases=len(cases) == 10,
                  ranks_in_protocol_order=[r["rank"] for r in cases] == list(range(10)),
                  all_panels_1080p=all(Image.open(output / r["figure"]).size == (1920, 1080) for r in cases),
                  all_dvf_hashes_verified=True, all_raw_inputs_geometry_verified=True,
                  cuda_not_initialized=not torch.cuda.is_initialized(), registration_launched=False)
    passed = all(v for k, v in checks.items() if k != "registration_launched")
    write_json(output / "validation.json", dict(passed=passed, checks=checks))
    lines = ["# CP-BE first10 animation case-selection preview", "",
        "Status: awaiting user case selection. No registration, trajectory capture, or animation encoding was run.", "",
        "Open `index.html` for the offline gallery or `overview.png` for all ten cases.", "",
        "## Protocol and interpretation", "",
        "- Existing accepted CP-BE-only headline warm fields, seed-3 ranks 0–9; not cache+batch+CP-BE.",
        "- CPU-only 3-D pull sampling; full through-plane displacement is retained. Labels use nearest neighbor.",
        f"- Axial axis 2, index {args.slice_index}; RAS arrays displayed radiologically (left R, top A).",
        "- Shared [0, .75] intensity window after original pairwise normalization; no per-frame contrast adjustment.",
        "- Dice/HD95/SDlogJ/folding are reused whole-volume metrics, not new 2-D slice metrics.",
        "- Warm seconds are original case-load-complete to DVF-file-ready measurements, not preview generation time.",
        "- Grid is a pull-warped reference texture, not the control lattice or a forward deformation map.",
        "- CP-BE has a prior strict numerical-equivalence failure; quality metrics do not prove equivalence.", "",
        "## Cases", "", "| Case / rank | Fixed / moving | Dice | HD95 mm | FG SDlogJ | FG folding % | Warm s |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in cases:
        lines.append(f"| {r['case_number']:02d} / {r['rank']} | {r['fixed_id']} / {r['moving_id']} | "
            f"{r['mean_dice']:.4f} | {r['hd95_mm']:.3f} | {r['log_jdet_std_fg']:.4f} | "
            f"{r['jdet_le0_pct_fg']:.3f} | {r['elapsed_sec']:.3f} |")
    lines += ["", "## Verification and next gate", "",
        f"Automatic verification passed: {passed}. Source, input and DVF hashes are in manifest.json.",
        "The module source snapshot, exact Python argv, Git status and dirty patch are retained here.",
        "Reply with a rank (0–9) to select a case. Complete trajectory capture is deliberately not launched.", ""]
    (output / "report.md").write_text("\n".join(lines))
    if not passed:
        raise RuntimeError("Preview validation failed")
    print(json.dumps(dict(status="awaiting_case_selection", output=str(output), validation_passed=True)))


if __name__ == "__main__":
    main()
