"""Sequential, empty-GPU-gated paper-loss timing matrix and convergence runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import atomic_json, external, sha256
from .model import REPO
from .run import source_receipts
from .study import load_configuration


def gpu_state(uuid):
    raw = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-gpu=uuid,name,memory.free,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True).strip()
    values = [s.strip() for s in raw.split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    return dict(uuid=values[0], name=values[1], free_mib=int(values[2]), used_mib=int(values[3]),
        utilization=int(values[4]), driver=values[5],
        processes=[p for p in processes if p.startswith(uuid)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smokes", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--validation", type=Path, required=True)
    args = p.parse_args()
    _, settings = load_configuration(args.config)
    validation = json.loads(args.validation.read_text())
    if validation.get("tests_passed") != 24:
        raise RuntimeError("Missing approved numerical test receipt")
    smokes = {}
    for variant in ("released_control", "paper_reconstruction"):
        summary = json.loads((args.smokes/variant/"summary.json").read_text())
        if (summary["outcome"] != "smoke_refinement_limit" or summary["refinement_steps"] != 8 or
                summary["checkpoint_replay_max_error"] != 0 or
                not any(t["kind"] == "joint_refinement" and t["max_error"] == 0 for t in summary["transitions"])):
            raise RuntimeError(f"Incomplete native smoke: {variant}")
        if variant == "paper_reconstruction" and len(summary["transitions"]) != 2:
            raise RuntimeError("Missing coarse/fine smoke")
        if summary["process_memory"]["competing_compute_pids"]:
            raise RuntimeError("Smoke process-memory identity was contaminated")
        smokes[variant] = summary
    out = external(args.out)
    out.mkdir(parents=True, exist_ok=False)
    frozen = source_receipts()
    config_hash = sha256(args.config)
    atomic_json(out/"queue_manifest.json", dict(sources=frozen, config_sha256=config_hash,
        validation=validation, smokes=smokes, gpu_uuid=args.gpu_uuid, command=sys.argv))
    for relative in frozen:
        path = out/"source_snapshot"/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO/relative).read_bytes())
    jobs = []
    # Anchor before convergence supplies a measured output reserve. All timed runs are sequential.
    for variant in ("released_control", "paper_reconstruction"):
        for budget in (200, 500, 2000):
            jobs.append(dict(variant=variant, kind="fixed", budget=budget, repeats=3,
                             name=f"{variant}_{budget}"))
        jobs.append(dict(variant=variant, kind="startup", budget=2000, repeats=1,
                         name=f"{variant}_startup"))
    for variant in ("released_control", "paper_reconstruction"):
        jobs.append(dict(variant=variant, kind="convergence", budget=2000, repeats=1,
                         name=f"{variant}_convergence"))
    results = []
    for job in jobs:
        if any(sha256(REPO/path) != digest for path,digest in frozen.items()) or sha256(args.config) != config_hash:
            raise RuntimeError("Source/config drift since queue freeze")
        state = gpu_state(args.gpu_uuid)
        if state["processes"] or state["free_mib"] < settings["minimum_free_gpu_mib"] or state["utilization"] > 5:
            atomic_json(out/"blocked.json", dict(job=job, gpu=state))
            raise RuntimeError(f"Timing GPU is not idle: {state}")
        reserve = 10.
        if job["kind"] == "convergence":
            reserve = max(json.loads(path.read_text())["resources"]["phases"]["output_generation"]["wall_seconds"]
                          for path in (out/f"{job['variant']}_2000").glob("repeat*/summary.json"))
        command = [sys.executable, "-B", "-m", "scripts.experiments.gaussiandir.study",
            "--config", str(args.config), "--variant", job["variant"], "--kind", job["kind"],
            "--budget", str(job["budget"]), "--repeats", str(job["repeats"]),
            "--out", str(out/job["name"]), "--gpu-uuid", args.gpu_uuid, "--export-reserve", str(reserve)]
        started = time.monotonic()
        command += ["--launch-monotonic", str(started)]
        atomic_json(out/"queue_progress.json", dict(current=job, completed=results, gpu_before=state))
        print(json.dumps(dict(start=job, gpu=state)), flush=True)
        with (out/(job["name"]+".log")).open("w") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        record = dict(**job, exitcode=result.returncode, worker_wall_seconds=time.monotonic()-started,
                      gpu_before=state, gpu_after=gpu_state(args.gpu_uuid))
        results.append(record)
        atomic_json(out/"queue_progress.json", dict(current=None, completed=results))
        if result.returncode:
            raise RuntimeError(f"Worker failed, retained log: {job['name']}")
        print(json.dumps(dict(completed=record)), flush=True)
    atomic_json(out/"completed.json", dict(jobs=results,
        completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


if __name__ == "__main__":
    main()
