"""Persistent single-GPU worker. CUDA is forbidden before host PID acknowledgement."""
import argparse
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True, default=str)+"\n")
    temporary.replace(path)


def registration(command, backend, args):
    import numpy as np
    import torch
    from scripts.experiments.oasis_alpha_sweep import run_one as run
    c = run.c
    from scripts.experiments.oasis_alpha_sweep import launch_oasis_unified_acceleration_sweep as unified
    out = Path(command["output"])
    candidate = unified.candidate_map()["MR_A_fast_decay"]
    if command.get("candidate_overrides"):
        allowed = {"name", "epochs", "stage_epochs", "stage_full_cps", "stage_downsample",
                   "stage_lr", "early_stop_mode", "stage_min_epochs", "stage_patience",
                   "stage_check_interval", "stage_rel_delta_total", "stage_rel_delta_data"}
        if set(command["candidate_overrides"]) - allowed:
            raise ValueError("Unsupported experiment candidate override")
        candidate = replace(candidate, **command["candidate_overrides"])
    if command.get("steps"):
        steps = command["steps"]
        candidate = replace(candidate, stage_epochs=",".join(map(str, steps)),
                            epochs=sum(steps), early_stop_mode="none")
    ns = argparse.Namespace(phase="acceleration", alpha=1000., seed=1, eval_interval=100,
                            timing_detail=command.get("detail", "endpoints"), variant="default",
                            out_root=out, oasis_root=args.oasis_root, pair_file=args.pair_file,
                            overwrite=False, epoch_override=1 if command.get("smoke") else 0)
    argv = unified.run_one_argv(ns, candidate, int(command["rank"]), str(args.gpu))
    argv += ["--output-mode", "dvf_only", "--save-final-dense", "--cudnn-benchmark", "false",
             "--cudnn-deterministic", "true", "--loss-history-mode",
             "gpu_buffer" if "buffer" in backend.features else "cpu"]
    write_json(out / "argv.json", argv)
    old_argv, parse = sys.argv, run.parse_args
    sys.argv = argv
    if command.get("snapshots"):
        desired = command["snapshots"]
        def snapshot(stage, epoch, control, cs, shape, spacing, final):
            if final or epoch in desired[str(stage)]:
                dest = out / "controls" / f"stage{stage}_epoch{epoch:04d}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez(dest, controls=control.detach().cpu().numpy().reshape(*cs, 3),
                         image_shape=shape, cps=spacing, stage=stage, epoch=epoch)
        def parsed():
            parsed_args = parse()
            parsed_args._control_snapshot_callback = snapshot
            return parsed_args
        run.parse_args = parsed
    backend.clear_case()
    profile = nullcontext()
    if command.get("profile"):
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
    try:
        with profile as prof:
            run.main()
        if command.get("profile"):
            prof.export_chrome_trace(str(out / "profile.json"))
            (out / "profile_summary.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
    finally:
        sys.argv, run.parse_args = old_argv, parse
        backend.clear_case()
    summaries = list(out.glob("*/*/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one summary, found {summaries}")
    summary = json.loads(summaries[0].read_text())
    timing = json.loads((summaries[0].parent / "timing.json").read_text())
    marks = {m["name"]: m["offset_sec"] for m in timing["marks"]}
    dvf = np.load(summary["dvf_file_path"], mmap_mode="r")
    valid = dvf.dtype == np.float32 and dvf.shape == (160, 224, 192, 3) and bool(np.isfinite(dvf).all())
    if not valid or timing["cudnn_requested"] != {"benchmark": False, "deterministic": True}:
        raise RuntimeError("DVF/backend validation failed")
    if (timing["cuda_backend"]["cudnn_benchmark"] is not False
            or timing["cuda_backend"]["cudnn_deterministic"] is not True
            or torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32
            or torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("Unexpected effective backend settings")
    result = dict(rank=command["rank"], elapsed_sec=marks["dvf_file_ready"]-marks["case_load_complete"],
                  steps=summary["stage_actual_epochs"], stop_reasons=summary.get("stage_stop_reasons"),
                  peak_gib=summary["training_peak_mem_gb"], dvf=summary["dvf_file_path"],
                  summary=str(summaries[0]), timing=str(summaries[0].parent / "timing.json"),
                  backend=timing["cuda_backend"], dvf_valid=bool(valid),
                  setup_device_sec=sum(e["elapsed_sec"] for e in timing["events"] if e["name"] == "setup.device"))
    result["features"] = backend.features
    result["backend_extra"] = dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                                   cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                                   deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                                   num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
                                   physical_gpu=args.gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   torch_seed=1, numpy_seed=1)
    result["effective_functions"] = {name: getattr(c, name).__qualname__ for name in
                                     ("conv1d", "cubic_bspline1d", "ncc_loss", "analytic_bspline_sampled_be")}
    gc.collect()
    return result


def microbench(args, output):
    import numpy as np
    import torch
    from .backends import Backend, VARIANTS
    from .backends import c
    c.set_cuda_device(str(args.gpu), cudnn_benchmark=False, cudnn_deterministic=True)
    torch.manual_seed(19)
    rows = []
    for shape in ((40, 56, 48), (80, 112, 96), (160, 224, 192)):
        cs = tuple(n//2+4 for n in shape)
        controls = (torch.randn((*cs, 3), device="cuda")*.01).requires_grad_()
        for variant in ("reference", "cache", "batch", "cpbe"):
            with Backend(VARIANTS[variant]).activate():
                for operator in ("expansion", "BE"):
                    if variant == "cpbe" and operator == "expansion":
                        continue
                    def call():
                        y = (c.expand_bspline_controls(controls, cs, shape, 2).square().mean()
                             if operator == "expansion" else c.analytic_bspline_sampled_be(controls, shape, 2))
                        torch.autograd.grad(y, controls)
                    for _ in range(10):
                        call()
                    groups = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(25):
                            call()
                        torch.cuda.synchronize()
                        groups.append((time.perf_counter()-started)/25)
                    rows.append(dict(variant=variant, operator=operator, shape=shape, control_shape=cs,
                                     group_sec=groups, median_sec=float(np.median(groups))))
    return {"rows": rows, "backend": c.cuda_backend_metadata()}


def evaluate(command, args):
    import numpy as np
    import torch
    from scripts.experiments.oasis_alpha_sweep import common as c
    device = c.set_cuda_device(str(args.gpu), cudnn_benchmark=False, cudnn_deterministic=True)
    rows = []
    cache = {}
    for item in command["items"]:
        rank = int(item["rank"])
        field = np.load(item["dvf"])
        if field.shape != (160, 224, 192, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
            raise RuntimeError("Invalid evaluation DVF")
        digest = hashlib.sha256(memoryview(field)).hexdigest()
        want_surface = bool(item.get("hd95", False))
        want_regularity = bool(item.get("regularity",False))
        key = (rank, digest, want_surface, want_regularity)
        cached = key in cache
        if not cached:
            batch = c.load_oasis_case(rank=rank, device=device, oasis_root=Path(args.oasis_root), pair_file=Path(args.pair_file))
            coords = c.make_coordinate_grid(c.IMAGE_SHAPE, device) + torch.from_numpy(field).to(device)
            warped_seg = c.warp_full(batch["moving_seg"], coords, mode="nearest")
            dice = c.dice_per_label(warped_seg, batch["fixed_seg"], batch["moving_seg"])
            vox = c.dhw3_norm_to_zyx_voxel_ncdhw(field)
            jd = c.jacobian_determinant_zyx(vox.astype(np.float64, copy=False))
            cache[key] = dict(dice=dice, mean_dice=float(np.nanmean(dice)),
                              nonpositive_jacobian_fraction=float(np.mean(jd <= 0)),
                              jacobian_mean=float(np.mean(jd)))
            if want_regularity:
                cache[key]["regularity"] = c.jacobian_metrics_voxel(
                    vox,batch["fixed_mask"].detach().cpu().bool().numpy())
            wc = warped_seg.detach().cpu().numpy().astype(np.int16)
            fc = batch["fixed_seg"].detach().cpu().numpy().astype(np.int16)
            mc = batch["moving_seg"].detach().cpu().numpy().astype(np.int16)
            cache[key]["label_voxel_counts"] = dict(
                labels=list(range(1,36)),
                warped=np.bincount(wc.ravel(),minlength=36)[1:36].tolist(),
                fixed=np.bincount(fc.ravel(),minlength=36)[1:36].tolist(),
                moving=np.bincount(mc.ravel(),minlength=36)[1:36].tolist(),
                intersection=np.bincount(fc[fc == wc],minlength=36)[1:36].tolist())
            del wc,fc,mc
            if want_surface:
                cache[key]["surface_metrics"] = c.surface_metrics_per_label(
                    warped_seg, batch["fixed_seg"], batch["moving_seg"])
                cache[key]["hd95_nan_pattern_matches_dice"] = bool(np.array_equal(
                    np.isnan(cache[key]["surface_metrics"]["hd95"]), np.isnan(dice)))
            del batch, coords, warped_seg, vox, jd
        row = dict(id=item["id"], rank=rank, dvf_sha256=digest, reused_identical_dvf_metrics=cached, **cache[key])
        if item.get("reference_dvf"):
            ref = np.load(item["reference_dvf"], mmap_mode="r")
            delta = (field.astype(np.float64)-ref)*np.array([191/2, 223/2, 159/2])
            norm2 = np.sum(delta*delta, axis=-1)
            row.update(dvf_vector_rms_voxel=float(np.sqrt(norm2.mean())),
                       dvf_vector_max_voxel=float(np.sqrt(norm2.max())))
        rows.append(row)
        write_json(Path(command["output"])/"evaluation_partial.json", rows)
        del field
        gc.collect()
    return {"rows": rows, "backend": c.cuda_backend_metadata()}


def environment_manifest(args, ranks=None):
    import importlib.metadata
    import nibabel as nib
    from .backends import c
    root = Path(args.oasis_root)
    ranks = list(range(10)) if ranks is None else [int(value) for value in ranks]
    cases = [c.resolve_case(rank=r,pair_file=Path(args.pair_file),oasis_root=root) for r in ranks]
    ids = sorted({value for case in cases for value in (case.fixed_id,case.moving_id)})
    files = []
    for identifier in ids:
        for prefix in ("img","seg"):
            path = root/"test"/f"{prefix}{identifier}.nii.gz"
            image = nib.load(str(path))
            if image.shape != (160,192,224):
                raise RuntimeError(f"Unexpected source image shape: {path}: {image.shape}")
            files.append(dict(relative_path=str(path.relative_to(root)),bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              nifti_shape=image.shape,zooms_mm=list(map(float,image.header.get_zooms()))))
    versions = {}
    for name in ("torch","numpy","scipy","nibabel","surface-distance","triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    counts = dict(sorted_pairs=len(c.parse_pair_file(root/"pair_test_200.txt")),
                  shuffled_pairs=len(c.parse_pair_file(Path(args.pair_file))),
                  images=len(list((root/"test").glob("img*.nii.gz"))),
                  segmentations=len(list((root/"test").glob("seg*.nii.gz"))))
    if counts != dict(sorted_pairs=200,shuffled_pairs=200,images=84,segmentations=84):
        raise RuntimeError(f"Unexpected dataset counts: {counts}")
    return dict(counts=counts,cases=[case.__dict__ for case in cases],files=files,versions=versions,
                installed_distributions=sorted({(d.metadata.get("Name", "unknown"),d.version)
                                                for d in importlib.metadata.distributions()}),
                sorted_pairs_sha256=hashlib.sha256((root/"pair_test_200.txt").read_bytes()).hexdigest(),
                shuffled_pairs_sha256=hashlib.sha256(Path(args.pair_file).read_bytes()).hexdigest(),
                surface_backend_available=c.surface_distance_metrics is not None,uid=os.getuid(),gid=os.getgid(),
                requested_ranks=ranks,reoriented_shape=[160,224,192],foreground="raw fixed image > 0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-token", required=True)
    p.add_argument("--mailbox", required=True, type=Path)
    p.add_argument("--gpu", required=True, type=int)
    p.add_argument("--oasis-root", required=True)
    p.add_argument("--pair-file", required=True)
    p.add_argument("--features", default="")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    write_json(args.mailbox/"ready.json", dict(container_pid=os.getpid(), token=args.job_token))
    deadline = time.monotonic()+120
    while not (args.mailbox/"host_ack.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("No verified host PID acknowledgement")
        time.sleep(.1)
    import torch
    from .backends import Backend
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    backend = Backend(tuple(filter(None, args.features.split(","))))
    with backend.activate():
        index = 0
        while True:
            path = args.mailbox/f"command_{index:04d}.json"
            while not path.exists():
                time.sleep(.1)
            command = json.loads(path.read_text())
            if command["action"] == "exit":
                break
            out = Path(command["output"])
            out.mkdir(parents=True, exist_ok=True)
            started = time.time()
            try:
                with (out/"worker.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                    if command["action"] == "register":
                        result = registration(command, backend, args)
                    elif command["action"] == "schedule_register":
                        from .overnight_actions import registration as schedule_registration
                        result = schedule_registration(command, backend, args)
                    elif command["action"] == "schedule_evaluate":
                        from .overnight_actions import evaluate as schedule_evaluate
                        result = schedule_evaluate(command, args)
                    elif command["action"] == "progress_register":
                        from scripts.visualization.mrdbscp_progress.capture import registration as progress_registration
                        result = progress_registration(command, backend, args)
                    elif command["action"] == "progress_evaluate":
                        from scripts.visualization.mrdbscp_progress.offline import evaluate_progress
                        result = evaluate_progress(command, args)
                    elif command["action"] == "checks":
                        from .checks import run_checks
                        result = run_checks("cuda", True, command.get("only_real", False))
                    elif command["action"] == "micro":
                        result = microbench(args, out)
                    elif command["action"] == "snapchecks":
                        from .checks import snapshot_checks
                        result = snapshot_checks(command["paths"],args.oasis_root,args.pair_file,
                                                 args.gpu,command["feature_sets"])
                    elif command["action"] == "evaluate":
                        result = evaluate(command, args)
                    elif command["action"] == "environment":
                        result = environment_manifest(args, command.get("ranks"))
                    else:
                        raise ValueError(command["action"])
                payload = dict(ok=True, result=result)
            except Exception:
                payload = dict(ok=False, traceback=traceback.format_exc())
            payload.update(start_unix=started, end_unix=time.time(), command=command)
            write_json(out/"result.json", payload)
            write_json(args.mailbox/f"result_{index:04d}.json", payload)
            index += 1


if __name__ == "__main__":
    main()
