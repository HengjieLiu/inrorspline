#!/usr/bin/env python
"""Read-only, orientation-audited rank4 review using the external DL demo.

No registration is run. Generated figures and provenance go to --out-root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/experiments/oasis_alpha_sweep"))
import common as c

VIEWS = ((0, 70, "Sagittal", ("A", "P", "I", "S")),
         (1, 112, "Coronal", ("R", "L", "I", "S")),
         (2, 96, "Axial", ("R", "L", "P", "A")))
LAMBDAS = (0.1, 0.3, 1, 3)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_slice(volume, axis, index):
    """Exactly the demo: transpose then horizontal flip; origin must be lower."""
    return np.fliplr(np.take(volume, index, axis=axis).T)


def normalized_from_axis_voxels(raw):
    """Saved DL channels follow array axes; grid_sample xyz is reverse order."""
    if raw.shape != (3, *c.IMAGE_SHAPE):
        raise ValueError(f"Invalid saved DL layout: {raw.shape}")
    scale = np.asarray([(s - 1) / 2 for s in c.IMAGE_SHAPE[::-1]], np.float32)
    return np.ascontiguousarray(raw[::-1].transpose(1, 2, 3, 0) / scale)


def select_rows(report):
    if not report.get("complete"):
        raise ValueError("Unaccepted/incomplete sweep")
    selected = []
    for alpha in LAMBDAS:
        rows = [r for r in report["case_metrics"] if r["case_rank"] == 4
                and r["resolution"] == "full" and r["representation"] == "842_dense"
                and r["alpha"] == alpha]
        if len(rows) != 1:
            raise ValueError(f"Require one full-resolution dense lambda={alpha} row")
        selected += rows
    for name in ("VFA", "SITReg"):
        rows = [r for r in report["case_metrics"] if r["case_rank"] == 4 and r["group"] == name]
        if len(rows) != 1:
            raise ValueError(f"Require one original {name} row")
        selected += rows
    for row in selected:
        if (row["fixed_id"], row["moving_id"], row["direction"]) != ("0084", "0108", "moving_to_fixed"):
            raise ValueError("Case/direction mismatch")
    return selected


def load_demo(root):
    path = root / "demo/example1_visualize_registration_results/utils_display.py"
    spec = importlib.util.spec_from_file_location("rank4_external_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_image(path, demo):
    source = nib.load(path)
    raw = source.get_fdata(dtype=np.float32)
    if nib.aff2axcodes(source.affine) != ("L", "I", "A"):
        raise ValueError(f"Expected source LIA affine: {path}")
    canonical = nib.as_closest_canonical(source)
    arr = np.ascontiguousarray(demo.reorient_OASIS_to_RAS(raw))
    if arr.shape != c.IMAGE_SHAPE or not np.isfinite(arr).all():
        raise ValueError("Invalid reoriented image")
    np.testing.assert_array_equal(arr, canonical.get_fdata(dtype=np.float32))
    np.testing.assert_allclose(nib.affines.voxel_sizes(canonical.affine), (1, 1, 1), atol=1e-6)
    return arr, {"path": str(path), "sha256": sha256(path), "source_shape": list(raw.shape),
                 "source_axcodes": list(nib.aff2axcodes(source.affine)),
                 "source_affine": source.affine.tolist(), "ras_affine": canonical.affine.tolist(),
                 "ras_shape": list(arr.shape), "demo_equals_affine_canonical_exactly": True}


def add_orientation(ax, directions):
    for text, x, y in zip(directions, (.025, .975, .5, .5), (.5, .5, .025, .975)):
        ax.text(x, y, text, color="white", fontsize=8, weight="bold", ha="center", va="center",
                transform=ax.transAxes, bbox={"facecolor": "black", "alpha": .65, "pad": 1.2, "edgecolor": "none"})


def verify_synthetic(demo):
    shape = (7, 9, 11)
    axes = np.indices(shape).astype(np.float32)
    transformer = demo.SpatialTransformer(shape)
    for component in range(3):
        flow = torch.zeros((1, 3, *shape))
        flow[:, component] = 1
        ramp = torch.from_numpy(axes[component])[None, None]
        warped = transformer(ramp, flow, padding_mode="zeros")[0, 0].numpy()
        crop = tuple(slice(1, -1) for _ in shape)
        np.testing.assert_allclose(warped[crop], axes[component][crop] + 1, atol=1e-6)
    for axis, _, _, _ in VIEWS:
        remaining = [i for i in range(3) if i != axis]
        horizontal = display_slice(axes[remaining[0]], axis, shape[axis] // 2)
        vertical = display_slice(axes[remaining[1]], axis, shape[axis] // 2)
        assert np.all(np.diff(horizontal, axis=1) == -1)
        assert np.all(np.diff(vertical, axis=0) == 1)
    zero = transformer(torch.ones((1, 1, *shape)), torch.zeros((1, 3, *shape)), padding_mode="zeros")
    assert torch.equal(zero, torch.ones_like(zero))
    return {"three_component_positive_pull_tests": True, "display_axis_direction_tests": True,
            "zero_field_identity": True}


def save_figure(fig, root, stem, dpi=200):
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render(demo, root, views, names, rows):
    metrics = [f"D {r['dice']:.3f} | HD {r['hd95']:.2f}\nASD {r['asd']:.3f} | Fold {r['jdet_le0_pct_fg']:.3f}%" for r in rows]
    titles = ["Fixed 0084", "Moving 0108\n(unregistered)"] + [f"{n}\n{m}" for n, m in zip(names, metrics)]
    for axis, index, name, directions in VIEWS:
        v = views[axis]
        def settings(cmap, low=None, high=None, norm=None):
            params = {"origin": "lower", "cmap": cmap, "interpolation": "nearest"}
            params.update({"norm": norm} if norm is not None else {"vmin": low, "vmax": high})
            return {"plot_type": "imshow", "imshow_params": params}
        jac_setting = settings("bwr", -1, 3)
        jac_setting["contour_overlay"] = {"show": True, "level": 0., "color": "black", "linewidth": .5}
        data_rows = [
            {"category": "deformed", "row_title": "Images", "data": v["images"],
             "disp_settings": [settings("gray", 0, .75)]},
            {"category": "diff", "row_title": "Warped − fixed", "data": v["differences"],
             "disp_settings": [settings("coolwarm", -.1, .1)],
             "shared_row_colorbar": {"show": True, "label": "Intensity\ndifference"}},
            {"category": "deformed", "row_title": "Labels", "data": v["labels"],
             "disp_settings": [settings(demo.cmap_seg_oasis, norm=demo.norm_seg_oasis)]},
            {"category": "deformation", "row_title": "Pull grid (8 mm)", "data": v["grids"],
             "disp_settings": [settings("gray", 0, 1)]},
            {"category": "deformation", "row_title": "Jacobian determinant", "data": v["jacobians"],
             "disp_settings": [jac_setting],
             "shared_row_colorbar": {"show": True, "label": "J; black\ncontour: J=0"}},
        ]
        height, width = v["images"][0].shape
        demo.plot_registration_results(
            data_rows=data_rows, num_methods=6, col_titles=titles, subplot_width=2.8,
            subplot_aspect_ratio=height/width, col_title_fontsize=10, row_title_fontsize=12,
            tick_label_fontsize=10, suptitle_fontsize=15, dpi=100, top_margin_factor=.88,
            right_margin_factor=.96, wspace_factor=.055, hspace_factor=.06,
            suptitle_text=f"Rank 4 · moving 0108 → fixed 0084 · {name} ({index+1}/{c.IMAGE_SHAPE[axis]})\n"
                          "8→4→2→dense: full-resolution NCC9 + diffusion vs original VFA/SITReg · metrics are whole-volume",
            show_figure=True)
        fig = plt.gcf()
        for ax in fig.axes:
            if ax.images:
                add_orientation(ax, directions)
        fig.text(.5, .012, "Shared display scales. RAS volumes; explicit anatomical edge labels. "
                 "Images: linear interpolation; labels: nearest. Grid is a 2D projection of the 3D pull field.",
                 ha="center", fontsize=10)
        save_figure(fig, root, name.lower())
    # A compact three-view overview with anatomically relevant fixed/warped outlines.
    fig, axes = plt.subplots(3, 8, figsize=(23, 10.5))
    for axis, index, name, directions in VIEWS:
        v = views[axis]
        for col, ax in enumerate(axes[axis]):
            ax.imshow(v["images"][col], cmap="gray", vmin=0, vmax=.75, origin="lower", interpolation="nearest")
            fixed_roi = np.isin(v["labels"][0], (3, 8, 22, 27))
            warped_roi = np.isin(v["labels"][col], (3, 8, 22, 27))
            if col > 0 and warped_roi.any():
                ax.contour(warped_roi, levels=[.5], colors=["#ff9d28"], linewidths=.65, origin="lower")
            if fixed_roi.any():
                ax.contour(fixed_roi, levels=[.5], colors=["#00e5ff"], linewidths=.65, origin="lower")
            add_orientation(ax, directions)
            ax.set_xticks([]); ax.set_yticks([])
            if axis == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}\nslice {index+1}", fontsize=12)
    fig.suptitle("Rank 4: NCC9/diffusion 8→4→2→dense sweep vs VFA/SITReg\n"
                 "Lateral ventricles + caudates: fixed contours cyan; moving/warped contours orange", fontsize=16)
    fig.tight_layout(rect=(0, .02, 1, .91))
    save_figure(fig, root, "three_view_overview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-repo", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--sweep-report", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError("Use a fresh output directory; prior review outputs are immutable")
    if args.out_root.resolve().is_relative_to(REPO):
        raise ValueError("Generated artifacts must be outside the source worktree")
    torch.set_num_threads(4)
    demo, demo_path = load_demo(args.demo_repo)
    checks = verify_synthetic(demo)
    report = json.loads(args.sweep_report.read_text())
    rows = select_rows(report)
    assert c.parse_pair_file(c.DEFAULT_PAIR_FILE)[4] == ("0084", "0108")
    inputs, arrays = [], {}
    for key, filename in (("fixed", "img0084"), ("moving", "img0108"),
                          ("fixed_seg", "seg0084"), ("moving_seg", "seg0108")):
        arrays[key], audit = load_image(args.oasis_root / "test" / f"{filename}.nii.gz", demo)
        inputs.append(audit)
    for audit in inputs[1:]:
        np.testing.assert_allclose(audit["ras_affine"], inputs[0]["ras_affine"], atol=1e-6)
    print("Affine RAS parity and synthetic component/orientation checks passed", flush=True)
    fixed, moving = arrays["fixed"], arrays["moving"]
    fixed_seg, moving_seg = arrays["fixed_seg"], arrays["moving_seg"]
    views = {axis: {"images": [display_slice(a, axis, idx) for a in (fixed, moving)],
                    "differences": [display_slice(moving-fixed, axis, idx)],
                    "labels": [display_slice(a, axis, idx) for a in (fixed_seg, moving_seg)],
                    "grids": [], "jacobians": []} for axis, idx, _, _ in VIEWS}
    identity = c.make_coordinate_grid(c.IMAGE_SHAPE, torch.device("cpu"))
    scale = torch.tensor([(s-1)/2 for s in c.IMAGE_SHAPE[::-1]], dtype=torch.float32)
    moving_t, moving_seg_t = torch.from_numpy(moving), torch.from_numpy(moving_seg)
    fixed_seg_t = torch.from_numpy(fixed_seg)
    transformer = demo.SpatialTransformer(c.IMAGE_SHAPE)
    audits, names = [], []
    for row in rows:
        is_dl = row["resolution"] == "reference"
        path = Path(row["source_path"]) if is_dl else Path(row["run_dir"]) / "disp_norm_0084_0108.npy"
        expected_hash = row["source_sha256"] if is_dl else row["dvf_sha256"]
        assert sha256(path) == expected_hash, f"Source field changed: {path}"
        if is_dl:
            norm = normalized_from_axis_voxels(nib.load(path).get_fdata(dtype=np.float32))
            name = row["group"]
        else:
            assert (path.parent / "DONE").is_file()
            norm = np.load(path, allow_pickle=False)
            name = f"NCC λ={row['alpha']:g}"
        assert norm.shape == (*c.IMAGE_SHAPE, 3) and np.isfinite(norm).all()
        # Reproduce the saved evaluator's norm->voxel->norm arithmetic exactly.
        vox = torch.from_numpy(norm) * scale
        coords = identity + vox / scale
        axis_vox = vox.numpy()[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
        with torch.no_grad():
            warped = c.warp_full(moving_t, coords).numpy()
            warped_seg_t = c.warp_full(moving_seg_t, coords, mode="nearest")
            warped_seg = warped_seg_t.numpy()
            demo_warp = transformer(moving_t[None, None], torch.from_numpy(axis_vox), padding_mode="zeros")[0, 0].numpy()
        warp_difference = float(np.max(np.abs(warped-demo_warp)))
        if warp_difference > 1e-4:
            raise ValueError(f"Demo/canonical coordinate convention mismatch: {warp_difference}")
        dice = float(np.nanmean(c.dice_per_label(warped_seg_t, fixed_seg_t, moving_seg_t)))
        if abs(dice-row["dice"]) > 5e-6:
            raise ValueError(f"Rendered label Dice disagrees: {dice} vs {row['dice']}")
        jdet = c.jacobian_determinant_zyx(axis_vox.astype(np.float64))
        demo_jdet, _, _ = demo.JDet()(axis_vox.astype(np.float64))
        np.testing.assert_array_equal(jdet, demo_jdet)
        values = jdet[(fixed > 0)[2:-2, 2:-2, 2:-2]]
        folding, sdlogj = float(np.mean(values <= 0) * 100), c.log_std_positive(values)
        np.testing.assert_allclose([folding, sdlogj], [row["jdet_le0_pct_fg"], row["log_jdet_std_fg"]], atol=1e-6, rtol=0)
        padded = np.pad(jdet, 2, constant_values=np.nan)  # no fabricated boundary Jacobians
        for axis, index, _, _ in VIEWS:
            v = views[axis]
            v["images"].append(display_slice(warped, axis, index))
            v["labels"].append(display_slice(warped_seg, axis, index))
            v["differences"].append(display_slice(warped-fixed, axis, index))
            v["jacobians"].append(display_slice(padded, axis, index))
            # Demo's default to_tensor=True implicitly uses CUDA; explicitly stay on CPU.
            grid = torch.from_numpy(demo.mk_grid_img_3d(8, grid_size=c.IMAGE_SHAPE, axis_to_slice=axis, to_tensor=False))[0, 0]
            with torch.no_grad():
                warped_grid = c.warp_full(grid, coords).numpy()
            v["grids"].append(display_slice(warped_grid, axis, index))
        audits.append({"method": name, "path": str(path), "sha256": expected_hash,
                       "layout": "channel_first_array_axis_voxels" if is_dl else "channel_last_normalized_grid_xyz",
                       "rendered_dice": dice, "saved_dice": row["dice"], "dice_difference": dice-row["dice"],
                       "demo_canonical_warp_max_abs": warp_difference,
                       "demo_canonical_jdet_exact": True, "fg_folding_pct": folding, "fg_sdlogj": sdlogj})
        names.append(name)
        print(f"Verified {name}: Dice={dice:.9f}, folding={folding:.6f}%, SDlogJ={sdlogj:.6f}", flush=True)
    args.out_root.mkdir(parents=True)
    render(demo, args.out_root, views, names, rows)
    manifest = {"case_rank": 4, "fixed_id": "0084", "moving_id": "0108", "direction": "moving_to_fixed",
                "scope": "full-resolution NCC9 + diffusion 842->dense, lambda at all stages; original DL fields",
                "source_report": str(args.sweep_report), "source_report_sha256": sha256(args.sweep_report),
                "script_sha256": sha256(__file__), "demo_path": str(demo_path), "demo_sha256": sha256(demo_path),
                "repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "device": "CPU; no GPU access", "inputs": inputs, "checks": checks, "fields": audits,
                "views": [{"axis": a, "zero_based_index": i, "name": n, "left_right_bottom_top": o} for a, i, n, o in VIEWS],
                "sampling": "canonical pull grid, align_corners=True, zeros padding; image linear, labels nearest",
                "display": "demo transpose+fliplr, origin=lower on every row; fixed shared ranges",
                "jacobian": "canonical central FD, crop2, undefined border NaN; black zero contour; no log(J+3) statistic",
                "metrics": "titles use accepted whole-volume metrics, not slice scores; Dice and regularity revalidated"}
    for entry in inputs + audits:
        assert sha256(entry["path"]) == entry["sha256"], "Read-only source changed"
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    keys = ("group", "dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
    with (args.out_root / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows({k: r[k] for k in keys} for r in rows)
    links = "\n".join(f'<h2>{name}</h2><a href="{stem}.pdf">PDF</a><br><a href="{stem}.png"><img src="{stem}.png" style="width:100%"></a>'
                      for name, stem in (("Three-view outline overview", "three_view_overview"),
                                         ("Sagittal", "sagittal"), ("Coronal", "coronal"), ("Axial", "axial")))
    (args.out_root / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>OASIS rank4 registration review</title>'
        '<body style="font-family:system-ui;margin:2em"><h1>Rank 4: fixed 0084, moving 0108</h1>'
        '<p>Full-resolution NCC9 + diffusion: 8→4→2→dense at λ=0.1/0.3/1/3; original VFA/SITReg. No new registrations.</p>'
        '<p>Demo slice positions, shared scales, RAS reorientation verified against NIfTI affines. Anatomical edges are labeled. '
        'Metrics in titles are whole-volume. Overview: cyan fixed / orange warped lateral-ventricle and caudate contours. '
        'Jacobian maps: black contour at J=0. Two-voxel excluded borders are blank. '
        'Grid rows show in-plane projections, not the complete 3D deformation.</p>'
        '<p><a href="manifest.json">Orientation/provenance audit</a> · <a href="metrics.csv">Metrics CSV</a></p>' + links + '</body>')
    (args.out_root / "ACCEPTED.json").write_text(json.dumps({"complete": True, "method_count": 6, "view_count": 3,
        "rendering": "CPU only", "figures": sorted(p.name for p in args.out_root.glob("*.png")),
        "source_hashes_unchanged": True, "orientation_and_metric_checks_passed": True}, indent=2) + "\n")
    print(f"Complete: {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
