#!/usr/bin/env python
"""P79 process-local cohort adapters over immutable P43/P74/P76 workers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import oasis_bspline842_first100 as e
import dl_ncc_diffusion as d
import run_oasis_dl_ncc_diffusion_one as displacement
import run_oasis_bounded2_842_sweep as bounded
import run_oasis_bspline_svf_sweep_recovery as svf
import bspline_svf_sweep as s
from oasis_ncc9_first100 import validate_first100


class First100SVFSpec(s.Spec):
    def validate(self):
        e.Spec(self.case_rank, "svf_velocity", self.alpha, self.smoke).validate()
        if (self.schedule, self.regularize, self.formulation, self.integration_steps) != ("842", "velocity", "refine", 7):
            raise ValueError("Only842 refined total SVF / velocity / seven steps")

    def name(self):
        self.validate()
        return e.Spec(self.case_rank, "svf_velocity", self.alpha, self.smoke).name()


def install():
    # Module mutation is local to this new process; historical source bytes and
    # imported objective/optimizer function bodies are never edited.
    d.FitSpec.validate = validate_first100
    d.VARIANT = e.VARIANT
    bounded.VARIANT = e.VARIANT
    bounded.run_name = lambda rank, alpha, smoke=False: e.Spec(rank, "bounded2", alpha, smoke).name()
    bounded.assert_manifest = e.assert_manifest
    s.Spec = First100SVFSpec
    s.VARIANT = e.VARIANT
    s.PROTOCOL = {**s.PROTOCOL, "ranks": list(range(100)), "cohort_extension": e.VARIANT}
    svf.assert_manifest = e.assert_manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("worker", "audit"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gpu", choices=("0","1","2","3"), required=True)
    p.add_argument("--case-rank", type=int, required=True)
    p.add_argument("--family", choices=e.FAMILIES, required=True)
    p.add_argument("--alpha", type=float, choices=e.LAMBDAS, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    e.Spec(a.case_rank, a.family, a.alpha, a.smoke).validate()
    if a.out_dir.resolve().is_relative_to(e.c.REPO_ROOT):
        p.error("External artifacts only")
    return a


def audit_displacement(args, manifest, spec):
    started = time.monotonic()
    row = e.c.read_json(args.out_dir / "summary.json")
    e.validate_identity(row, spec, manifest)
    if row["gpu"] != args.gpu:
        raise RuntimeError("Audit must retain worker GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    device = e.c.set_cuda_device(args.gpu, cudnn_benchmark=False, cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch, hashes = displacement.load_batch(args, device)
    if hashes != row["input_hashes"]:
        raise RuntimeError("Replay input hashes differ")
    checks, previous_controls, previous_field = [], None, None
    ds = d.FitSpec(spec.case_rank, "pyramid", spec.alpha, "842", 1, spec.smoke)
    for stage, info in zip(d.stages(ds), row["stages"]):
        folder = args.out_dir / f"stage{stage.index}"
        state = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        controls = state["parameter_norm_xyz"]
        obj = d.StageObjective(batch["fixed"], batch["moving"], spec.alpha, stage, controls.shape[:3])
        initial = torch.zeros_like(controls) if previous_controls is None else e.c.cubic_bspline_dyadic_knot_insert_3d(previous_controls, controls.shape[:3])
        with torch.no_grad():
            values = [float(v) for v in obj(controls)]
            initial_values = [float(v) for v in obj(initial)]
            if info["stage"] != asdict(stage) or state["summary"] != info:
                raise RuntimeError("Checkpoint stage metadata mismatch")
            if any(not math.isclose(v, info[k], rel_tol=1e-6, abs_tol=1e-7)
                   for v,k in zip(values, ("best_loss","best_data_loss","best_reg_loss"))):
                raise RuntimeError("Saved checkpoint loss differs")
            if not math.isclose(initial_values[0], info["initial_loss"], rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError("Initialization loss differs")
            expand = d.AlignedBSpline(controls.shape[:3], e.c.IMAGE_SHAPE, e.c.IMAGE_SHAPE, stage.cps, device)
            if previous_field is not None:
                delta = displacement.verify_init(previous_field, expand(initial))
                if delta != {k:v for k,v in row["initialization_checks"][stage.index-1].items() if k != "stage"}:
                    raise RuntimeError("Prolongation replay differs")
            current = expand(controls).detach()
        e.check_history(folder / "history.csv", info, spec.alpha)
        checks.append({"cps": stage.cps, "loss_history_verified": True, "selected_loss_data_reg": values})
        previous_controls, previous_field = controls, current
        del obj, initial, expand, state
    saved = np.load(row["dvf_path"])
    if not np.array_equal(current.cpu().numpy(), saved):
        raise RuntimeError("Independent displacement replay differs")
    vox = current * d.voxel_scale(e.c.IMAGE_SHAPE, device=device)
    accuracy, labels = displacement.canonical_metrics(vox, batch)
    regularity = e.c.jacobian_metrics_voxel(vox.cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy(), batch["fixed_mask"].cpu().numpy().astype(bool))
    metrics = {**accuracy, **regularity}
    if any(not math.isclose(metrics[k], row[k], rel_tol=0, abs_tol=1e-12) for k in e.METRICS):
        raise RuntimeError("Canonical replay differs")
    with (args.out_dir / "case_metrics.csv").open() as stream:
        old = next(csv.DictReader(stream))
    for key, value in labels.items():
        if "_label_" not in key:
            continue
        expected = float(old[key])
        if math.isnan(value) and math.isnan(expected):
            continue
        if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Per-label replay differs: {key}")
    absent = [label for label in e.c.OASIS_LABELS if not math.isfinite(labels[f"dice_label_{label:02d}"])]
    if absent != row["excluded_labels"] or len(e.c.OASIS_LABELS)-len(absent) != row["valid_label_count"]:
        raise RuntimeError("Absent-label handling differs")
    e.atomic_json(args.out_dir / "independent_audit.json", {"passed": True, "checks": checks,
        "all_label_metrics_reproduced": True, "summary_sha256": e.sha(args.out_dir / "summary.json"),
        "audit_sec": time.monotonic()-started, "audit_peak_cuda_gib": torch.cuda.max_memory_allocated()/2**30,
        "audit_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "gpu": args.gpu, "completed_at": e.c.now_text()})


def main(args):
    if (os.getuid(), os.getgid()) != (1009,1020):
        raise RuntimeError("Require UID:GID1009:1020")
    manifest = e.c.read_json(args.manifest)
    e.assert_manifest(manifest)
    spec = e.Spec(args.case_rank, args.family, args.alpha, args.smoke)
    install()
    for key in ("oasis_root", "pair_file", "dl_code_root", "vfa_root"):
        setattr(args, key, Path(manifest["settings"][key]))
    args.depth, args.config = 3, bounded.CONFIG
    args.schedule, args.regularize, args.integration_steps = "842", "velocity", 7
    args.resolution, args.representation, args.seed = "pyramid", "842", 1
    args.cpu_threads, args.prefix_dir = 4, None
    args.source_revision = manifest["source_revision"]
    os.environ["INRORSPLINE_CONTAINER_IMAGE"] = manifest["container_image"]
    os.environ["INRORSPLINE_CONTAINER_IMAGE_ID"] = manifest["container_image_id"]
    if args.action == "worker":
        {"displacement": displacement.run, "bounded2": bounded.worker, "svf_velocity": svf.worker}[spec.family](args)
        e.validate_identity(e.c.read_json(args.out_dir / "summary.json"), spec, manifest)
        e.atomic_json(args.out_dir / "extension_provenance.json", {
            "variant": e.VARIANT, "spec": asdict(spec), "manifest_sha256": e.sha(args.manifest),
            "source_hashes": manifest["source_hashes"], "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
            "gpu": args.gpu, "visible_cuda_device": "cuda:0"})
    else:
        if (args.out_dir / "independent_audit.json").exists():
            # A crash after a successful replay can safely finish its receipt,
            # but never rerun an auditor over existing outputs.
            audit = e.c.read_json(args.out_dir / "independent_audit.json")
            if not audit.get("passed") or audit["summary_sha256"] != e.sha(args.out_dir / "summary.json"):
                raise RuntimeError("Preserve stale/failed audit for review")
        elif spec.family == "bounded2":
            import audit_oasis_bounded2_842_sweep as audit
            sys.argv = [str(Path(audit.__file__)), "--manifest", str(args.manifest), "--run-dir", str(args.out_dir), "--gpu", args.gpu]
            audit.main()
        elif spec.family == "svf_velocity":
            svf.audit(args)
        else:
            audit_displacement(args, manifest, spec)
        row = e.c.read_json(args.out_dir / "summary.json")
        receipt = e.artifact_receipt(row, spec, manifest)
        receipt["files"][str(args.out_dir / "extension_provenance.json")] = {
            **e.fingerprint(args.out_dir / "extension_provenance.json"), "sha256": e.sha(args.out_dir / "extension_provenance.json")}
        e.atomic_json(args.out_dir / "extension_receipt.json", receipt)
        e.validate(args.out_dir, manifest, spec)
    e.assert_manifest(manifest)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception:
        if args.out_dir.is_dir():
            e.atomic_json(args.out_dir / f"FAILED_{args.action}_{time.time_ns()}.json", {"traceback": traceback.format_exc(), "at": e.c.now_text()})
        raise
