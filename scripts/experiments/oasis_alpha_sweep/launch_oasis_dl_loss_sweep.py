#!/usr/bin/env python
"""Four-GPU gated scheduler for 350 independent native DL-loss registrations."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]/"reports"))
import common as c
import dl_loss_sweep as d
import oasis_dl_loss_first10_arc as report
from launch_oasis_mrdbscp842_dense_lambda_sweep import source_metadata
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file
from run_oasis_dl_loss_one import source_hashes, native_hashes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    p.add_argument("--out-root", type=Path, default=root/"runs"/d.VARIANT)
    p.add_argument("--review-root", type=Path, default=root/"repo_experiments"/d.VARIANT)
    p.add_argument("--baseline-root", type=Path, default=root/"repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1")
    p.add_argument("--ncc-root", type=Path, default=root/"repo_experiments/oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1")
    for name in ("oasis-root", "wavereg-code-root", "vfa-root"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--feature-checkpoint", type=Path, default=d.CHECKPOINT)
    p.add_argument("--feature-config", type=Path, default=d.CONFIG)
    p.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.gpu_list = args.gpus.split(",")
    if (len(set(args.gpu_list)) != len(args.gpu_list) or any(g not in ("0", "1", "2", "3") for g in args.gpu_list)
            or not 1 <= args.max_concurrent <= min(4, len(args.gpu_list)) or args.cpu_threads < 1):
        p.error("Require unique GPUs0–3 and at most one new worker per GPU")
    if args.feature_checkpoint is None or args.feature_config is None:
        p.error("--feature-checkpoint and --feature-config are required for native Feature-MSE")
    sources = [c.REPO_ROOT.resolve(), args.baseline_root.resolve(), args.ncc_root.resolve(),
               args.oasis_root.resolve(), args.wavereg_code_root.resolve(), args.vfa_root.resolve(),
               args.feature_config.resolve(), args.feature_checkpoint.resolve()]
    outputs = [args.out_root.resolve(), args.review_root.resolve()]
    for out in outputs:
        if any(out.is_relative_to(src) or src.is_relative_to(out) for src in sources):
            p.error("Output overlaps protected input")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        p.error("Run and review roots must be disjoint")
    if args.pair_file.resolve() != c.DEFAULT_PAIR_FILE.resolve():
        p.error("This benchmark requires the audited seed3 first10 pair file")
    return args


def validate_result(path, spec, hashes, native):
    if not (path/"DONE").is_file():
        raise RuntimeError("Missing completion marker")
    row = c.read_json(path/"summary.json")
    for key, expected in (("spec", asdict(spec)), ("protocol", d.PROTOCOL), ("source_hashes", hashes),
                          ("native_source_hashes", native), ("method", d.METHOD)):
        if row[key] != expected:
            raise RuntimeError(f"Result invariant failed: {key}")
    pair = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec.case_rank]
    if (row["fixed_id"], row["moving_id"]) != pair or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Wrong case/direction")
    if row["parameter_units"] != "normalized_grid_xyz" or row["representation"] != "dense_voxel":
        raise RuntimeError("Wrong representation/units")
    if any(row[k] is not None for k in ("cps", "bspline_order", "control_shape")):
        raise RuntimeError("Final field is not directly dense")
    if (row["uid"], row["gid"]) != (1009, 1020) or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong UID/precision")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"]:
        raise RuntimeError("Wrong backend flags")
    if row["segmentation_loss"] or row["reference_dvf_supervision"] or row["feature_loss"] != (spec.loss == "femse"):
        raise RuntimeError("Unexpected supervision")
    if row["objective_support"] != "full_native_volume" or len(row["stages"]) != 4:
        raise RuntimeError("Wrong objective support/stage count")
    for actual, stage in zip(row["stages"], d.stages(spec)):
        if actual["stage"] != asdict(stage) or actual["objective_voxel_count"] != int(np.prod(c.IMAGE_SHAPE)):
            raise RuntimeError("Stage schedule or voxel count changed")
        if actual["alpha"] != spec.scale*spec.alpha or actual["loss_key"] != spec.loss:
            raise RuntimeError("Loss coefficients changed")
    if not all(math.isfinite(float(row[k])) for k in (*report.METRICS, "best_loss")):
        raise RuntimeError("Nonfinite metrics")
    if spec.smoke and (not row["native_parity"]["passed"] or not row["diffusion_parity_passed"]):
        raise RuntimeError("Failed native parity")
    field = np.load(row["dvf_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid field")
    for key in ("model_path", "dvf_path"):
        if sha256_file(Path(row[key])) != row[key.replace("_path", "_sha256")]:
            raise RuntimeError("Output hash mismatch")
    return row


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.path = args.out_root/"launcher_state.json"
        self.state = c.read_json(self.path) if self.path.exists() else {"phase": "preflight", "results": {}, "attempts": []}
        self.hashes, self.native = source_hashes(), native_hashes(args)
        self.revision = source_metadata()["source_revision"]
        for key, value in (("source_hashes", self.hashes), ("native_source_hashes", self.native)):
            if key in self.state and self.state[key] != value:
                raise RuntimeError("Source drift: use distinct reviewed attempt root")
            self.state[key] = value

    def save(self):
        temp = self.path.with_suffix(".tmp")
        c.write_json(temp, self.state); temp.replace(self.path)

    def event(self, event, **details):
        row = {"time": c.now_text(), "event": event, **details}
        print(json.dumps(row), flush=True)
        with (self.args.out_root/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")

    def command(self, spec, gpu, path):
        cmd = [sys.executable, "-B", str(HERE/"run_oasis_dl_loss_one.py"), "--gpu", gpu,
               "--case-rank", str(spec.case_rank), "--loss", spec.loss, "--multiplier", repr(spec.multiplier),
               "--out-dir", str(path), "--source-revision", self.revision, "--cpu-threads", str(self.args.cpu_threads)]
        for key in ("oasis_root", "wavereg_code_root", "vfa_root", "pair_file", "feature_checkpoint", "feature_config"):
            cmd += ["--"+key.replace("_", "-"), str(getattr(self.args, key))]
        if spec.smoke:
            cmd.append("--smoke")
        return cmd

    def run(self, requested):
        pending = []
        for spec in requested:
            if spec.name() in self.state["results"]:
                validate_result(Path(self.state["results"][spec.name()]), spec, self.hashes, self.native)
                continue
            attempts = [a for a in self.state["attempts"] if a["key"] == spec.name()]
            complete = next((a for a in attempts if (Path(a["path"])/"DONE").is_file()), None)
            if complete:
                validate_result(Path(complete["path"]), spec, self.hashes, self.native)
                self.state["results"][spec.name()] = complete["path"]
                complete["status"] = "complete"
            elif attempts:
                raise RuntimeError(f"Unresolved attempt requires review: {spec.name()}")
            else:
                pending.append((spec, 0, 0))
        active, finished, failures = {}, 0, []
        self.save()
        while pending or active:
            for gpu, info in list(active.items()):
                code = info["process"].poll()
                if code is None:
                    continue
                info["log"].close(); del active[gpu]; finished += 1
                spec, record = info["spec"], info["record"]
                try:
                    if code:
                        raise RuntimeError(f"Worker exit {code}")
                    row = validate_result(Path(record["path"]), spec, self.hashes, self.native)
                    self.state["results"][spec.name()] = record["path"]
                    record["status"] = "complete"
                    if spec.smoke:
                        self.state.setdefault("smoke_peak_gib", {})[spec.loss] = max(s["peak_cuda_gib"] for s in row["stages"])
                    self.event("complete", key=spec.name(), gpu=gpu, dice=row["dice"], hd95=row["hd95"], asd=row["asd"])
                except Exception as exc:
                    record.update(status="failed", exit_code=code, error=str(exc))
                    log = Path(record["log_path"]).read_text(errors="replace").lower()
                    if record["attempt"] == 0 and "cuda" in log and "out of memory" in log:
                        pending.append((spec, 1, finished+(1 if active else 0)))
                        self.event("oom_retry_queued", key=spec.name())
                    else:
                        failures.append({"key": spec.name(), "error": str(exc), "log": record["log_path"]})
                        self.event("failure", **failures[-1])
                self.save()
            if failures:
                pending.clear()  # Let already-running workers finish; never kill GPU jobs.
            free = [g for g in self.args.gpu_list if g not in active]
            while pending and free and len(active) < self.args.max_concurrent:
                eligible = [i for i, (_, _, ready) in enumerate(pending) if finished >= ready]
                if not eligible:
                    break
                index = eligible[0]
                spec, attempt, _ = pending[index]
                snapshot = gpu_snapshot()
                if snapshot["error"]:
                    raise RuntimeError(snapshot["error"])
                memory = {r["physical_gpu"]: r["memory_free_mib"] for r in snapshot["rows"]}
                needed = max(8192, 1024*self.state.get("smoke_peak_gib", {}).get(spec.loss, 0)+4096)
                sufficient = [g for g in free if memory.get(g, 0) >= needed]
                if not sufficient:
                    break
                gpu = max(sufficient, key=lambda g: memory[g]) if attempt else sufficient[0]
                if source_hashes() != self.hashes or native_hashes(self.args) != self.native:
                    raise RuntimeError("Source changed while scheduler was running")
                pending.pop(index); free.remove(gpu)
                path = self.args.out_root/"jobs"/spec.name()/f"attempt{attempt}"
                log_path = self.args.out_root/"logs"/f"{spec.name()}_attempt{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or log_path.exists():
                    raise FileExistsError("Refusing to overwrite attempt")
                cmd = self.command(spec, gpu, path)
                handle = log_path.open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS=str(self.args.cpu_threads), MKL_NUM_THREADS=str(self.args.cpu_threads))
                process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
                record = {"key": spec.name(), "spec": asdict(spec), "attempt": attempt, "gpu": gpu,
                          "pid": process.pid, "path": str(path), "log_path": str(log_path), "status": "running",
                          "command": cmd, "initial_gpu_state": snapshot, "started_at": c.now_text()}
                self.state["attempts"].append(record)
                active[gpu] = dict(process=process, record=record, spec=spec, log=handle)
                self.event("launched", key=spec.name(), gpu=gpu, pid=process.pid, attempt=attempt)
                self.save()
            if pending or active:
                time.sleep(3)
        if failures:
            raise RuntimeError(f"Failures preserved; no incomplete aggregation: {failures}")
        return [validate_result(Path(self.state["results"][s.name()]), s, self.hashes, self.native) for s in requested]


def execute(args):
    if args.dry_run:
        print(json.dumps({"production_count": len(d.specs()), "smoke_count": len(d.specs(True)),
                          "names": [s.name() for s in d.specs()], "protocol": d.PROTOCOL}, indent=2))
        return
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True); args.review_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = Scheduler(args)
        refs, hashes = report.load_references(args.baseline_root, args.ncc_root)
        manifest = {"args": vars(args), "protocol": d.PROTOCOL, "source": source_metadata(),
                    "source_hashes": scheduler.hashes, "native_source_hashes": scheduler.native,
                    "reference_hashes": hashes, "initial_gpu_state": gpu_snapshot(),
                    "uid": os.getuid(), "gid": os.getgid(), "started_at": c.now_text(), "production_count": 350}
        manifest_path = args.out_root/"launch_manifest.json"
        if manifest_path.exists():
            old = c.read_json(manifest_path)
            if any(old[k] != manifest[k] for k in ("source_hashes", "native_source_hashes", "reference_hashes")):
                raise RuntimeError("Launch source/reference drift")
        else:
            c.write_json(manifest_path, manifest)
        (args.out_root/"launcher.pid").write_text(str(os.getpid())+"\n")
        scheduler.state["phase"] = "smoke"; scheduler.save()
        smoke = scheduler.run(d.specs(True))
        c.write_json(args.review_root/"smoke_results.json", {"passed": True, "count": 5, "results": smoke})
        scheduler.state["phase"] = "smoke_passed"; scheduler.save()
        if args.stop_after == "smoke":
            return
        scheduler.state["phase"] = "production"; scheduler.save()
        results = scheduler.run(d.specs())
        report.write_report(results, refs, args.review_root, hashes)
        scheduler.state["phase"] = "complete"; scheduler.save()
        scheduler.event("experiment_complete", outputs=350, report=str(args.review_root/"report.md"))


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        execute(args)
    else:
        args.out_root.mkdir(parents=True, exist_ok=True)
        with (args.out_root/f"launcher_{time.time_ns()}.log").open("w", buffering=1) as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                try:
                    execute(args)
                except Exception:
                    traceback.print_exc()
                    raise
