"""P81 immutable source selection, physical-coordinate adapter and input audit.

Scalar arrays retain the repository's reoriented D,H,W layout. Vectors exported
here are world RAS millimetres, never PyTorch grid XYZ. All artifacts are external.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
SPLIT = REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
SPLIT_SHA256 = "40ff80b23407c37ed8a20092f018fd9588e5df3eb95628069ec42da698aac3a0"
KEY = "oasis_seed3_first10_adaptive_local_affine_fieldfit_v1"
REPORTS = {
    "p43": "oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1",
    "p76": "oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1",
    "p74": "oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1/continuation_v2",
}
SOURCES = ("sitreg", "unbounded", "bounded2", "svf")
SOURCE_REPORT = dict(sitreg="p43", unbounded="p43", bounded2="p76", svf="p74")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def reorient(array, affine):
    """Exact legacy transpose/flip, with the missing affine update made explicit."""
    shape = array.shape
    if array.ndim != 3:
        raise ValueError("Expected a scalar 3D volume")
    # raw i = (N0-1-d, N1-1-w, h)
    transform = np.array([[-1, 0, 0, shape[0]-1],
                          [0, 0, -1, shape[1]-1],
                          [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    return np.ascontiguousarray(array.transpose(0, 2, 1)[::-1, :, ::-1]), np.asarray(affine) @ transform


def indices_to_mm(indices, shape, affine):
    ijk = np.stack(np.unravel_index(np.asarray(indices, dtype=np.int64), tuple(shape)), -1)
    h = np.asarray(affine, dtype=np.float64)
    return ijk @ h[:3, :3].T + h[:3, 3]


def normalized_to_mm(displacement, fixed_affine, moving_affine):
    """Convert normalized-grid XYZ pull displacement to physical endpoint DVF."""
    shape = displacement.shape[:3]
    scale = (np.asarray(shape[::-1], dtype=np.float64)-1)/2
    h_f, h_m = np.asarray(fixed_affine), np.asarray(moving_affine)
    result = np.empty(displacement.shape, dtype=np.float32)
    for start in range(0, shape[0], 8):
        end = min(start + 8, shape[0])
        index = np.stack(np.meshgrid(np.arange(start, end), np.arange(shape[1]),
                                    np.arange(shape[2]), indexing="ij"), -1)
        delta = (np.asarray(displacement[start:end], np.float64) * scale)[..., ::-1]
        result[start:end] = ((index + delta) @ h_m[:3, :3].T + h_m[:3, 3]
                             - (index @ h_f[:3, :3].T + h_f[:3, 3]))
    return result


def mm_to_normalized(displacement, fixed_affine, moving_affine):
    shape = displacement.shape[:3]
    scale = (np.asarray(shape[::-1], dtype=np.float64)-1)/2
    h_f, h_m = np.asarray(fixed_affine), np.asarray(moving_affine)
    inverse = np.linalg.inv(h_m[:3, :3])
    result = np.empty(displacement.shape, dtype=np.float32)
    for start in range(0, shape[0], 8):
        end = min(start+8, shape[0])
        index = np.stack(np.meshgrid(np.arange(start, end), np.arange(shape[1]),
                                    np.arange(shape[2]), indexing="ij"), -1)
        x = index @ h_f[:3, :3].T + h_f[:3, 3]
        moving = (x + np.asarray(displacement[start:end], np.float64)-h_m[:3, 3]) @ inverse.T
        result[start:end] = (moving-index)[..., ::-1]/scale
    return result


def stratified_samples(mask, seed=0, caps=(65536, 16384, 65536), bins=8):
    """Proportionally sample spatial strata once, then divide into disjoint sets."""
    ids = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(ids) < sum(caps):
        counts = np.floor(len(ids) * np.asarray(caps)/sum(caps)).astype(int)
        for i in range(len(ids)-int(counts.sum())):
            counts[i % 3] += 1
    else:
        counts = np.asarray(caps, dtype=int)
    count = int(counts.sum())
    coords = np.stack(np.unravel_index(ids, mask.shape), -1)
    cell = np.minimum(coords*bins//np.asarray(mask.shape), bins-1)
    keys = (cell[:, 0]*bins+cell[:, 1])*bins+cell[:, 2]
    order = np.argsort(keys, kind="stable")
    keys, ids = keys[order], ids[order]
    cuts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1, len(ids)]
    sizes = np.diff(cuts)
    exact = sizes * count/max(len(ids), 1)
    take = np.floor(exact).astype(int)
    remainder = count-int(take.sum())
    # Largest remainder makes allocation reproducible and proportional.
    take[np.argsort(-(exact-take), kind="stable")[:remainder]] += 1
    selected = np.concatenate([rng.permutation(ids[a:b])[:n]
                               for a, b, n in zip(cuts[:-1], cuts[1:], take)])
    rng.shuffle(selected)
    first, second = int(counts[0]), int(counts[0]+counts[1])
    return dict(fit=selected[:first], val=selected[first:second], quick=selected[second:])


def select_sources(reports, pairs):
    selected = {}
    for source in SOURCES:
        report = reports[SOURCE_REPORT[source]]
        rows = report["case_metrics"] if source == "sitreg" else report["candidates"]
        for row in rows:
            s = row.get("spec", row)
            if source == "sitreg":
                use = row.get("representation") == "SITReg"
            elif source == "unbounded":
                use = (s.get("representation") == "842" and s.get("resolution") == "pyramid"
                       and s.get("alpha") == 1)
            elif source == "bounded2":
                use = row.get("alpha") == 1 and row.get("maps_per_level") == 2
            else:
                use = s.get("schedule") == "842" and s.get("regularize") == "velocity" and s.get("alpha") == 1
            rank = row.get("case_rank")
            if not use or rank not in range(10):
                continue
            if (row["fixed_id"], row["moving_id"]) != tuple(pairs[rank]):
                raise ValueError(f"Wrong pair identity {source}/{rank}")
            if row.get("direction") != "moving_to_fixed":
                raise ValueError(f"Wrong direction {source}/{rank}")
            key = (source, rank)
            if key in selected:
                raise ValueError(f"Duplicate selected row {key}")
            selected[key] = row
    return selected


def warp_audit(moving, segmentation, source_norm, recovered_norm):
    """Independent legacy normalized path versus physical-adapter round trip."""
    shape = moving.shape
    axes = [torch.linspace(-1, 1, n) for n in shape]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    identity = torch.stack((xx, yy, zz), -1)
    original = identity + torch.from_numpy(np.ascontiguousarray(source_norm))
    restored = identity + torch.from_numpy(np.ascontiguousarray(recovered_norm))
    volume = torch.from_numpy(np.ascontiguousarray(moving, dtype=np.float32))[None, None]
    labels = torch.from_numpy(np.ascontiguousarray(segmentation, dtype=np.float32))[None, None]
    scale = max(float(np.max(moving)-np.min(moving)), 1e-12)
    stats = {}
    for padding in ("zeros", "border"):
        old = F.grid_sample(volume, original[None], align_corners=True, padding_mode=padding)
        new = F.grid_sample(volume, restored[None], align_corners=True, padding_mode=padding)
        difference = (old-new).double()/scale
        rmse = float(torch.sqrt(difference.square().mean()))
        maximum = float(difference.abs().max())
        old_labels = F.grid_sample(labels, original[None], mode="nearest", align_corners=True, padding_mode=padding)
        new_labels = F.grid_sample(labels, restored[None], mode="nearest", align_corners=True, padding_mode=padding)
        mismatch = float((old_labels != new_labels).double().mean())
        stats[padding] = dict(image_range_normalized_rmse=rmse,
                              image_range_normalized_max=maximum, label_mismatch_fraction=mismatch)
        if rmse > 1e-5 or maximum > 1e-3 or mismatch > 1e-4:
            raise ValueError(f"Warp round-trip tolerance failed: {padding}: {stats[padding]}")
    return stats


def _load_report(artifact_root, key):
    folder = artifact_root/"repo_experiments"/key
    accepted_path, path = folder/"ACCEPTED.json", folder/"results.json"
    accepted = json.loads(accepted_path.read_text())
    if not any(accepted.get(k) is True for k in ("complete", "passed", "technical_completion")):
        raise ValueError(f"Source report not accepted: {key}")
    digest = sha256(path)
    expected = accepted.get("output_sha256", accepted.get("report_sha256", {})).get("results.json")
    if expected is not None and digest != expected:
        raise ValueError(f"Accepted report checksum mismatch: {key}")
    return json.loads(path.read_text()), dict(path=str(path), sha256=digest,
                                             accepted_path=str(accepted_path), accepted_sha256=sha256(accepted_path),
                                             embedded_results_hash_available=expected is not None)


def prepare(root, artifact_root=None, oasis_root=None):
    root = Path(root).resolve()
    artifact_root = Path(artifact_root or os.environ["ARTIFACT_ROOT"]).resolve()
    oasis_root = Path(oasis_root or os.environ.get("OASIS_ROOT", os.environ.get("INRORSPLINE_OASIS_ROOT", "")))
    if not root.is_relative_to(artifact_root) or root.is_relative_to(REPO):
        raise ValueError("Prepared inputs must live under the external ARTIFACT_ROOT")
    if (root/"input_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing input manifest")
    root.mkdir(parents=True, exist_ok=True)
    inputs = root/"inputs"
    inputs.mkdir(exist_ok=False)
    (inputs/"pairs").mkdir()
    (inputs/"targets").mkdir()
    if sha256(SPLIT) != SPLIT_SHA256:
        raise ValueError("Canonical seed3 split hash changed")
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", SPLIT.read_text())[:10]
    reports, receipts, report_errors = {}, {}, {}
    for name, key in REPORTS.items():
        try:
            reports[name], receipts[name] = _load_report(artifact_root, key)
        except Exception as exc:
            report_errors[name] = f"{type(exc).__name__}: {exc}"
            reports[name] = dict(candidates=[], case_metrics=[])
    selected = select_sources(reports, pairs)
    pair_records, targets, failures = {}, [], []
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for rank, (fixed, moving) in enumerate(pairs):
        pair_dir = inputs/"pairs"/f"rank{rank:03d}"
        pair_dir.mkdir()
        try:
            images, affines, input_hashes, raw_headers = {}, {}, {}, {}
            for prefix, ident, name in (("img", fixed, "fixed_image"), ("img", moving, "moving_image"),
                                      ("seg", fixed, "fixed_seg"), ("seg", moving, "moving_seg")):
                path = oasis_root/"test"/f"{prefix}{ident}.nii.gz"
                volume = nib.load(path)
                arr = np.asarray(volume.dataobj, dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise ValueError(f"Nonfinite input {path.name}")
                images[name], affines[name] = reorient(arr, volume.affine)
                if prefix == "seg":
                    images[name] = images[name].astype(np.int16)
                input_hashes[path.name] = sha256(path)
                raw_headers[name] = dict(path=str(path), shape=list(volume.shape), affine=volume.affine.tolist(),
                                         orientation=list(nib.aff2axcodes(volume.affine)), spacing=list(map(float,volume.header.get_zooms())))
            shape = images["fixed_image"].shape
            if any(a.shape != shape for a in images.values()) or shape != (160,224,192):
                raise ValueError("Unexpected canonical scalar geometry")
            for stem in ("fixed", "moving"):
                if not np.allclose(affines[stem+"_image"], affines[stem+"_seg"], atol=1e-7, rtol=0):
                    raise ValueError("Image/segmentation physical geometry mismatch")
            # Current saved fields presume common atlas index grids. Do not silently
            # reinterpret a differently processed pair as a compatible reference.
            if not np.allclose(affines["fixed_image"], affines["moving_image"], atol=1e-7, rtol=0):
                raise ValueError("Saved-field atlas convention requires matched fixed/moving geometry")
            for source in SOURCES:
                row = selected.get((source, rank), {})
                for name, expected in row.get("input_hashes", {}).items():
                    if Path(name).name in input_hashes and input_hashes[Path(name).name] != expected:
                        raise ValueError(f"Historical input checksum mismatch: {source}/{Path(name).name}")
            for name, array in images.items():
                np.save(pair_dir/f"{name}.npy", array)
            mask = images["fixed_image"] > 0
            np.save(pair_dir/"mask.npy", mask)
            sample_arrays = stratified_samples(mask, seed=rank)
            # seed=0 + rank gives deterministic independent pair streams.
            sample_paths = {}
            for name, ids in sample_arrays.items():
                path = pair_dir/f"{name}_indices.npy"
                np.save(path, ids)
                sample_paths[name] = str(path)
            corners = np.array(np.meshgrid(*[[0,n-1] for n in shape], indexing="ij")).reshape(3,-1).T
            affine = affines["fixed_image"]
            corners_mm = corners @ affine[:3,:3].T + affine[:3,3]
            info = dict(pair_rank=rank, fixed_id=fixed, moving_id=moving,
                        pair_directory=str(pair_dir), shape=list(shape), affine=affine.tolist(),
                        fixed_affine=affine.tolist(), moving_affine=affines["moving_image"].tolist(),
                        domain_min=corners_mm.min(0).tolist(), domain_max=corners_mm.max(0).tolist(),
                        mask_path=str(pair_dir/"mask.npy"), sample_paths=sample_paths,
                        sample_hashes={k:sha256(v) for k,v in sample_paths.items()},
                        sample_counts={k:len(v) for k,v in sample_arrays.items()},
                        sampling_seed=0, pair_seed_offset=rank, foreground_voxels=int(mask.sum()),
                        input_hashes=input_hashes, raw_headers=raw_headers,
                        input_hash_validation="matched_available_source_input_hashes",
                        status="passed")
            atomic_json(pair_dir/"pair.json", info)
            pair_records[str(rank)] = info
        except Exception as exc:
            info = dict(pair_rank=rank, fixed_id=fixed, moving_id=moving, status="failed",
                        failure=f"{type(exc).__name__}: {exc}")
            pair_records[str(rank)] = info
            failures.append(dict(kind="pair", **info))
            atomic_json(pair_dir/"FAILED.json", dict(**info, traceback=traceback.format_exc()))
        for source in SOURCES:
            target_id = f"rank{rank:03d}_{source}"
            target_dir = inputs/"targets"/target_id
            target_dir.mkdir()
            record = dict(id=target_id, target_id=target_id, pair_rank=rank, source_method=source,
                          fixed_id=fixed, moving_id=moving, status="pending")
            try:
                if info["status"] != "passed":
                    raise ValueError("Pair input audit failed")
                row = selected.get((source,rank))
                if row is None:
                    raise ValueError(f"Missing selected source row; {report_errors.get(SOURCE_REPORT[source], '')}")
                path = Path(row["source_path"] if source == "sitreg" else row["dvf_path"])
                expected = row["source_sha256"] if source == "sitreg" else row["dvf_sha256"]
                actual = sha256(path)
                if actual != expected:
                    raise ValueError("Source field checksum mismatch")
                if source == "sitreg":
                    raw = np.asarray(nib.load(path).dataobj,dtype=np.float32)
                    if raw.shape != (3,*shape):
                        raise ValueError("Wrong SITReg channel-first voxel shape")
                    norm = np.moveaxis(raw,0,-1)[...,::-1]/((np.array(shape[::-1])-1)/2).astype(np.float32)
                    storage = "channel_first_zyx_voxel"
                else:
                    raw = np.load(path,mmap_mode="r")
                    if raw.shape != (*shape,3):
                        raise ValueError("Wrong normalized spline field shape")
                    norm = np.asarray(raw,dtype=np.float32)
                    storage = "channel_last_xyz_normalized_grid"
                if not all(np.isfinite(norm[a:a+8]).all() for a in range(0,shape[0],8)):
                    raise ValueError("Source field has nonfinite values")
                field = normalized_to_mm(norm,info["fixed_affine"],info["moving_affine"])
                recovered = mm_to_normalized(field,info["fixed_affine"],info["moving_affine"])
                error = np.abs(recovered.astype(np.float64)-norm)
                roundtrip = dict(normalized_max=float(error.max()), normalized_rmse=float(np.sqrt(np.mean(error*error))))
                if roundtrip["normalized_max"] > 2e-7:
                    raise ValueError(f"Coordinate round trip failed: {roundtrip}")
                warp = warp_audit(images["moving_image"],images["moving_seg"],norm,recovered) if rank == 0 else None
                canonical_path = target_dir/"field_mm.npy"
                np.save(canonical_path,field)
                # Bound statistics are input descriptions, not a topology certificate.
                norms = np.linalg.norm(field[mask],axis=-1)
                baseline = dict(displacement_rms_mm=float(np.sqrt(np.mean(norms.astype(np.float64)**2))),
                                displacement_p95_mm=float(np.percentile(norms,95)),
                                displacement_max_mm=float(norms.max()),
                                historical_foreground_folding_pct=row.get("jdet_le0_pct_fg"))
                native_cp = row.get("model_path")
                if source == "svf" and row.get("blocks"):
                    native_cp=row["blocks"][-1].get("model_path")
                record.update({k:info[k] for k in ("pair_directory","shape","affine","fixed_affine","moving_affine",
                                                      "domain_min","domain_max","mask_path","sample_paths","sample_hashes")})
                record.update(source_path=str(path),source_sha256=actual,source_config_path=receipts[SOURCE_REPORT[source]]["path"],
                              source_report=SOURCE_REPORT[source],source_report_sha256=receipts[SOURCE_REPORT[source]]["sha256"],
                              source_shape=list(raw.shape),tensor_layout=storage,source_dtype=str(raw.dtype),
                              field_kind="displacement",mapping_direction="fixed_mm_to_moving_mm_pullback",pullback_or_pushforward="pullback",
                              coordinate_units="mm",canonical_path=str(canonical_path),canonical_sha256=sha256(canonical_path),
                              canonical_shape=list(field.shape),canonical_grid_affine=info["affine"],
                              vector_channel_order="world_RAS",align_corners=True,interpolation="trilinear",
                              padding_mode="zero_image_border_field",includes_global_affine="not_separately_identified_in_saved_endpoint",
                              composition_order="previous_o_increment_six_maps" if source=="bounded2" else "single_final_endpoint",
                              native_velocity_available="checkpoint_replay_pending" if source=="svf" else False,
                              native_control_points_available=bool(native_cp and Path(native_cp).is_file()),
                              native_checkpoint_path=native_cp,source_code_revision=row.get("source_revision"),
                              conversion_code_revision=revision,conversion_code_sha256=sha256(__file__),
                              conversion_status="passed",source_regularization_weight=1,
                              source_regularization="velocity_diffusion" if source=="svf" else "historical_source_protocol",
                              roundtrip=roundtrip,warp_audit=warp,reference_statistics=baseline,
                              metric_replay_status="canonical_map_and_warp_parity_checked; historical_Dice_HD95_not_recomputed",
                              warnings=["includes_global_affine not separately identified; final endpoint retained unchanged"],status="passed")
                atomic_json(target_dir/"target.json",record)
                print(json.dumps(dict(target_id=target_id,status="passed",roundtrip=roundtrip)),flush=True)
            except Exception as exc:
                record.update(status="failed",failure=f"{type(exc).__name__}: {exc}")
                failures.append(dict(kind="target",**record))
                atomic_json(target_dir/"FAILED.json",dict(**record,traceback=traceback.format_exc()))
                print(json.dumps(dict(target_id=target_id,status="failed",failure=record["failure"])),flush=True)
            targets.append(record)
            atomic_json(root/"input_audit_progress.json",dict(targets=targets,pairs=pair_records,failures=failures))
    manifest=dict(version=1,key=KEY,run_root=str(root),created_unix=time.time(),source_revision=revision,
                  source_code_sha256=sha256(__file__),split_path=str(SPLIT),split_sha256=sha256(SPLIT),
                  source_reports=receipts,source_report_failures=report_errors,pairs=pair_records,targets=targets,
                  failures=failures,expected_targets=40,passed_targets=sum(r["status"]=="passed" for r in targets),
                  complete=len(targets)==40 and not failures,coordinate_convention="physical_RAS_mm_on_reoriented_fixed_DHW_grid",
                  limitations=["Historical Dice/HD95 source scores are retained, not recalculated during input preparation.",
                               "Native velocity replay is a separate diagnostic; saved endpoint is the immutable target."])
    atomic_json(root/"input_manifest.json",manifest)
    return root/"input_manifest.json"


def load_target(record, mmap_mode="r"):
    if record.get("status") != "passed":
        raise ValueError("Cannot load target that failed input audit")
    field=np.load(record["canonical_path"],mmap_mode=mmap_mode)
    mask=np.load(record["mask_path"],mmap_mode=mmap_mode)
    ids={k:np.load(v,mmap_mode=mmap_mode) for k,v in record["sample_paths"].items()}
    shape=tuple(record["shape"])
    return dict(field=field,mask=mask,affine=np.asarray(record["affine"]),shape=shape,
                fit_indices=ids["fit"],val_indices=ids["val"],quick_indices=ids["quick"],
                points_mm=indices_to_mm(ids["fit"],shape,record["affine"]).astype(np.float32),
                targets_mm=np.asarray(field.reshape(-1,3)[ids["fit"]]),
                fixed_image_path=str(Path(record["pair_directory"])/"fixed_image.npy"),
                moving_image_path=str(Path(record["pair_directory"])/"moving_image.npy"),
                fixed_seg_path=str(Path(record["pair_directory"])/"fixed_seg.npy"),
                moving_seg_path=str(Path(record["pair_directory"])/"moving_seg.npy"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    print(prepare(args.root),flush=True)
