"""Posthoc full-metric scoring; timed registration must finish before this queue."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .data import atomic_json, external, load_inputs, sha256
from .evaluate import evaluate
from .model import GaussianModel, REPO
from .resources import NvmlSampler, Resources
from .run import render
from .study import configure, load_configuration


def discover(run_root):
    jobs, aliases, hashes = [], [], {}
    for path in sorted(run_root.glob("**/summary.json")):
        if "source_snapshot" in path.parts:
            continue
        run = path.parent
        summary = json.loads(path.read_text())
        if summary["smoke"]:
            raise RuntimeError("Production scoring cannot contain smoke rows")
        key = run.relative_to(run_root).as_posix().replace("/", "__")
        field = run/"field_false_xyz.npy"
        digest = sha256(field)
        if digest != summary["field_sha256"]:
            raise RuntimeError("Selected field changed")
        entry = dict(key=key, source=str(field), kind="field", run=str(run),
            variant=summary["variant"], budget=summary["budget"], run_kind=summary["kind"],
            repeat=summary["repeat"], step=summary["selected_step"],
            optimization_seconds=summary["resources"]["phases"]["optimization"]["wall_seconds"],
            registration_seconds=summary["registration_seconds"], field_sha256=digest)
        if digest in hashes:
            aliases.append(dict(**entry, metrics_from=hashes[digest]))
        else:
            hashes[digest] = key
            jobs.append(entry)
        if summary["kind"] != "convergence":
            continue
        # Every archived scientific timepoint, plus terminal, scored independently.
        seen = set()
        for snapshot in json.loads((run/"snapshots.json").read_text()):
            if snapshot["step"] in seen and snapshot["tag"] != "terminal":
                continue
            seen.add(snapshot["step"])
            jobs.append(dict(key=key+"__"+snapshot["tag"], source=str(run/snapshot["path"]),
                kind="checkpoint", run=str(run), variant=summary["variant"], budget=summary["budget"],
                run_kind="trajectory", repeat=1, step=snapshot["step"],
                optimization_seconds=snapshot["optimization_seconds"], registration_seconds=None,
                charged_seconds=snapshot["charged_seconds"]))
    return jobs, aliases


def worker(args):
    configure()
    base, _ = load_configuration(args.config)
    inputs = load_inputs(base)
    jobs = json.loads(args.jobs.read_text())
    nvml = NvmlSampler(args.gpu_uuid)
    try:
        for job in jobs:
            out = external(args.out/job["key"])
            if (out/"metrics.json").exists():
                old = json.loads((out/"receipt.json").read_text())
                if old["source_sha256"] != sha256(job["source"]):
                    raise RuntimeError("Completed evaluation source drift")
                continue
            out.mkdir(parents=True, exist_ok=False)
            start_row = len(nvml.rows)
            resources = Resources(nvml)
            try:
                if job["kind"] == "field":
                    prediction = np.load(job["source"])
                else:
                    with resources.phase("posthoc_render"):
                        checkpoint = torch.load(job["source"], map_location="cpu", weights_only=False)
                        model = GaussianModel.restore(checkpoint["model"], "cuda")
                        prediction = render(model, inputs["mask"].shape, base["render_chunk"])
                        del model, checkpoint
                with resources.phase("posthoc_evaluation"):
                    result, warped, labels = evaluate(prediction, inputs)
                result["input_receipts"] = inputs["receipts"]
                result["field_receipt"] = dict(kind=job["kind"], source=job["source"],
                    source_sha256=sha256(job["source"]),
                    raw_array_sha256=hashlib.sha256(prediction.tobytes()).hexdigest())
                with resources.phase("analysis_storage"):
                    atomic_json(out/"metrics.json", result)
                    atomic_json(out/"receipt.json", dict(job=job, source_sha256=sha256(job["source"]),
                        evaluator_sha256=sha256(Path(__file__).with_name("evaluate.py")),
                        worker_sha256=sha256(__file__)))
                nvml.sample()
                # Snapshot the list without replacing the monitor thread's append target.
                view = copy.copy(nvml)
                view.rows = nvml.rows[start_row:]
                memory = view.summary()
                atomic_json(out/"resources.json", dict(resources=resources.summary(), process_memory=memory,
                    gpu_uuid=args.gpu_uuid, note="Posthoc only; excluded from registration latency."))
                print(json.dumps(dict(key=job["key"], dice=result["accuracy"]["dice_mean"],
                    seconds=result["metric_seconds"], labels=result["eligible_labels"])), flush=True)
                del prediction, result, warped, labels
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                atomic_json(out/"failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
                raise
    finally:
        nvml.close()


def orchestrate(args):
    if not (args.run_root/"completed.json").exists():
        raise RuntimeError("Finish all timed registration before posthoc GPU/CPU work")
    out = external(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs, aliases = discover(args.run_root)
    atomic_json(out/"evaluation_manifest.json", dict(jobs=jobs, aliases=aliases,
        run_root=str(args.run_root), queue_completed_sha256=sha256(args.run_root/"completed.json")))
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    from .study_queue import gpu_state
    available = [uuid.strip() for uuid in devices if not gpu_state(uuid.strip())["processes"]]
    if not available:
        raise RuntimeError("No idle GPU for posthoc scoring")
    available = available[:args.workers]
    def shard(index, uuid):
        path = out/f"jobs_{index}.json"
        atomic_json(path, jobs[index::len(available)])
        command = [sys.executable, "-B", "-m", "scripts.experiments.gaussiandir.study_evaluate",
            "--config", str(args.config), "--out", str(out/"evaluations"),
            "--jobs", str(path), "--gpu-uuid", uuid]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        with (out/f"worker_{index}.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Evaluation worker{index} failed; see retained log")
        return dict(worker=index, gpu_uuid=uuid, jobs=len(jobs[index::len(available)]))
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        results = list(pool.map(lambda pair:shard(*pair), enumerate(available)))
    atomic_json(out/"evaluation_completed.json", dict(workers=results, jobs=len(jobs), aliases=len(aliases)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=Path)
    p.add_argument("--gpu-uuid")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    worker(args) if args.jobs else orchestrate(args)


if __name__ == "__main__":
    main()
