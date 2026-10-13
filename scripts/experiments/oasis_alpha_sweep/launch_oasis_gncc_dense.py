"""Gated four-GPU GNCC sweep with automatic calibration, tables and ARC plots."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"reports"))
import common as c
import gncc_dense as d
import oasis_gncc_dense_report as report
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    p.add_argument("--out-root", type=Path, default=root/"runs"/d.VARIANT)
    p.add_argument("--review-root", type=Path, default=root/"repo_experiments"/d.VARIANT)
    p.add_argument("--ncc-root", type=Path, default=root/"repo_experiments/oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1")
    p.add_argument("--baseline-root", type=Path, default=root/"repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1")
    p.add_argument("--loss-root", type=Path, default=root/"repo_experiments/oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1/verified_environment_v1")
    p.add_argument("--oasis-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--stop-after", choices=("smoke", "qualification", "full"), default="full")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.gpu_list = args.gpus.split(",")
    if len(set(args.gpu_list)) != len(args.gpu_list) or any(g not in ("0", "1", "2", "3") for g in args.gpu_list) or not 1 <= args.max_concurrent <= len(args.gpu_list) or args.cpu_threads < 1:
        p.error("Unique GPUs0-3, one worker/GPU, positive threads required")
    outputs = [args.out_root.resolve(), args.review_root.resolve()]
    sources = [c.REPO_ROOT.resolve()]+[getattr(args, k).resolve() for k in ("oasis_root", "ncc_root", "baseline_root", "loss_root")]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a in outputs for b in sources) or outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        p.error("Output overlaps protected input or other output")
    return args


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.path = args.out_root/"launcher_state.json"
        self.hashes = d.source_hashes()
        self.state = c.read_json(self.path) if self.path.exists() else {"phase": "preflight", "results": {}, "attempts": [], "started_unix": time.time()}
        if self.state.get("source_hashes", self.hashes) != self.hashes:
            raise RuntimeError("Source drift: require a separately reviewed output root")
        self.state["source_hashes"] = self.hashes

    def save(self):
        self.state["updated_unix"] = time.time()
        temp = self.path.with_suffix(".tmp")
        c.write_json(temp, self.state); temp.replace(self.path)

    def event(self, event, **details):
        row = {"time": c.now_text(), "event": event, **details}
        print(json.dumps(row), flush=True)
        with (self.args.out_root/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")

    def run(self, specs):
        pending = []
        for spec in specs:
            if spec.name() in self.state["results"]:
                d.validate_result(Path(self.state["results"][spec.name()]), spec, self.hashes)
                continue
            attempts = [a for a in self.state["attempts"] if a["key"] == spec.name()]
            done = next((a for a in attempts if (Path(a["path"])/"DONE").exists()), None)
            if done:
                d.validate_result(Path(done["path"]), spec, self.hashes)
                done["status"] = "complete"; self.state["results"][spec.name()] = done["path"]
            elif attempts:
                raise RuntimeError(f"Unresolved attempt needs review: {spec.name()}")
            else:
                pending.append((spec, 0, 0))
        active, failures, finished, heartbeat = {}, [], 0, 0.
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
                    row = d.validate_result(Path(record["path"]), spec, self.hashes)
                    record.update(status="complete", completed_unix=time.time())
                    self.state["results"][spec.name()] = record["path"]
                    if spec.smoke:
                        self.state.setdefault("peak_gib", {})[spec.loss] = max(row["peak_reserved_gib"], max(s["peak_cuda_gib"] for s in row["stages"]))
                    self.event("complete", key=spec.name(), gpu=gpu, dice=row["dice"], hd95=row["hd95"], asd=row["asd"])
                except Exception as exc:
                    record.update(status="failed", error=str(exc), completed_unix=time.time())
                    log = Path(record["log_path"]).read_text(errors="replace").lower()
                    if record["attempt"] == 0 and "cuda" in log and "out of memory" in log:
                        pending.append((spec, 1, finished+(1 if active else 0)))
                        self.event("oom_retry_queued", key=spec.name())
                    else:
                        failures.append(record); self.event("failure", key=spec.name(), error=str(exc))
                self.save()
            if failures:
                pending.clear()  # Let current workers finish; never kill existing GPU jobs.
            free = [g for g in self.args.gpu_list if g not in active]
            while pending and free and len(active) < self.args.max_concurrent:
                index = next((i for i, (_, _, ready) in enumerate(pending) if finished >= ready), None)
                if index is None:
                    break
                spec, attempt, _ = pending[index]
                state = gpu_snapshot()
                if state["error"]:
                    raise RuntimeError(state["error"])
                memory = {r["physical_gpu"]: r["memory_free_mib"] for r in state["rows"]}
                needed = max(8192, 1024*self.state.get("peak_gib", {}).get(spec.loss, 0)+4096)
                available = [g for g in free if memory.get(g, 0) >= needed]
                if not available:
                    break
                gpu = max(available, key=lambda g: memory[g]) if attempt else available[0]
                d.verify_source_hashes(self.hashes)
                pending.pop(index); free.remove(gpu)
                path = self.args.out_root/"jobs"/spec.name()/f"attempt{attempt}"
                log_path = self.args.out_root/"logs"/f"{spec.name()}_attempt{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or log_path.exists():
                    raise FileExistsError("Refusing to overwrite an attempt")
                command = [sys.executable, "-B", str(HERE/"run_oasis_gncc_dense_one.py"), "--case-rank", str(spec.case_rank),
                           "--loss", spec.loss, "--alpha", repr(spec.alpha), "--gpu", gpu, "--oasis-root", str(self.args.oasis_root),
                           "--out-dir", str(path), "--cpu-threads", str(self.args.cpu_threads)]+(["--smoke"] if spec.smoke else [])
                handle = log_path.open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS=str(self.args.cpu_threads), MKL_NUM_THREADS=str(self.args.cpu_threads))
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                record = {"key": spec.name(), "spec": asdict(spec), "gpu": gpu, "attempt": attempt, "pid": process.pid,
                          "path": str(path), "log_path": str(log_path), "status": "running", "command": command,
                          "initial_gpu_state": state, "started_unix": time.time()}
                self.state["attempts"].append(record)
                active[gpu] = dict(process=process, log=handle, record=record, spec=spec)
                self.event("launched", key=spec.name(), gpu=gpu, pid=process.pid); self.save()
            if time.monotonic()-heartbeat > 15:
                self.state.update(active_workers=len(active), pending_workers=len(pending))
                self.save(); heartbeat = time.monotonic()
            if pending or active:
                time.sleep(3)
        if failures:
            raise RuntimeError("Failures preserved; incomplete results cannot be aggregated")
        return [d.validate_result(Path(self.state["results"][s.name()]), s, self.hashes) for s in specs]


def qualify(row, args):
    candidates = c.read_json(args.ncc_root/"results.json")["candidates"]
    ref = next(r for r in candidates if r["case_rank"] == 0 and r["spec"]["resolution"] == "full" and r["spec"]["representation"] == "842_dense" and r["spec"]["alpha"] == 1.)
    for key in ("actual_epochs", "best_epoch"):
        if [s[key] for s in row["stages"]] != [s[key] for s in ref["stages"]]:
            raise RuntimeError(f"LNCC reproduction stage mismatch: {key}")
    scale = np.asarray([(c.IMAGE_SHAPE[2]-1)/2, (c.IMAGE_SHAPE[1]-1)/2, (c.IMAGE_SHAPE[0]-1)/2])
    a, b = np.load(row["dvf_path"], mmap_mode="r"), np.load(ref["dvf_path"], mmap_mode="r")
    max_voxel = max(float(np.max(np.abs((a[z]-b[z])*scale))) for z in range(len(a)))
    deltas = {k: row[k]-ref[k] for k in ("dice", "hd95", "asd")}
    passed = max_voxel <= .01 and all(abs(deltas[k]) <= t for k, t in (("dice", 1e-4), ("hd95", .01), ("asd", .001)))
    check = {"passed": passed, "max_abs_voxel": max_voxel, "metric_deltas": deltas, "reference": ref["run_dir"], "qualification": row["run_dir"]}
    c.write_json(args.review_root/"lncc_reproduction.json", check)
    if not passed:
        raise RuntimeError(f"LNCC reproduction gate failed: {check}")


def execute(args):
    if args.dry_run:
        print(json.dumps({"production_count": len(d.specs()), "smoke_count": len(d.specs(True)), "protocol": d.PROTOCOL}, indent=2)); return
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True); args.review_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root/"launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = Scheduler(args)
        refs, hashes = report.load_references(args.baseline_root, args.ncc_root)
        for name in ("results.json", "ACCEPTED.json"):
            hashes[str(args.loss_root/name)] = sha256_file(args.loss_root/name)
        if s.state.get("reference_hashes", hashes) != hashes:
            raise RuntimeError("Reference drift")
        s.state["reference_hashes"] = hashes
        s.state["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        s.state["pid"] = os.getpid(); s.save()
        (args.out_root/"launcher.pid").write_text(str(os.getpid())+"\n")
        s.state["phase"] = "smoke"; s.save()
        smoke = s.run(d.specs(True))
        c.write_json(args.review_root/"smoke_results.json", {"passed": True, "results": smoke})
        if args.stop_after == "smoke":
            s.state["phase"] = "smoke_passed"; s.save(); return
        s.state["phase"] = "lncc_qualification"; s.save()
        qualification = s.run([d.FitSpec(0, "lncc", 1.)])[0]
        qualify(qualification, args)
        if args.stop_after == "qualification":
            s.state["phase"] = "qualification_passed"; s.save(); return
        s.state["phase"] = "calibration"; s.save()
        cal = args.review_root/"calibration.json"
        if cal.exists():
            if c.read_json(cal)["source_hashes"] != s.hashes or not c.read_json(cal)["complete"]:
                raise RuntimeError("Stale/incomplete calibration")
        else:
            command = [sys.executable, "-B", str(HERE/"calibrate_oasis_gncc_dense.py"), "--ncc-root", str(args.ncc_root),
                       "--oasis-root", str(args.oasis_root), "--review-root", str(args.review_root), "--gpu", args.gpu_list[0]]
            with (args.out_root/"calibration.log").open("x") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        s.state["phase"] = "production"; s.save()
        results = s.run(d.specs())
        s.state["phase"] = "reporting"; s.save()
        report.write_report(results, refs, args.review_root, hashes, args.loss_root)
        s.state["phase"] = "complete"; s.state["completed_unix"] = time.time(); s.save()
        s.event("experiment_complete", results=140, report=str(args.review_root/"report.md"))


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        execute(args)
    else:
        args.out_root.mkdir(parents=True, exist_ok=True)
        with (args.out_root/f"launcher_{time.time_ns()}.log").open("w", buffering=1) as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                try:
                    execute(args)
                except Exception:
                    traceback.print_exc(); raise
