"""Add concurrent posthoc scorers for disjoint, later checkpoint ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-root", type=Path, required=True)
    p.add_argument("--gpu-uuids", nargs=4, required=True)
    p.add_argument("--start-index", type=int, default=56)
    args = p.parse_args()
    scoring = args.review_root/"scoring"
    evaluations = scoring/"evaluations"
    workers, manifest = [], []
    for shard, uuid in enumerate(args.gpu_uuids):
        source = json.loads((scoring/f"jobs_{shard}.json").read_text())
        for parity in (0, 1):
            part = source[args.start_index+parity::2]
            part = [job for job in part if not (evaluations/job["key"]/"metrics.json").exists()]
            name = f"boost_{shard}_{parity}"
            jobfile = scoring/f"{name}.json"
            jobfile.write_text(json.dumps(part))
            manifest.append(dict(name=name, gpu_uuid=uuid, jobs=len(part), keys=[j["key"] for j in part]))
            if part:
                workers.append((name, uuid, jobfile, len(part)))
    source_path = Path(__file__).with_name("resources.py")
    receipt = dict(plan=manifest, resource_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        note="Posthoc scoring only. These disjoint later ranges finish before the sequential scorers reach them.")
    (scoring/"posthoc_boost_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    state, lock = dict(completed=[], failures=[]), threading.Lock()

    def save_state():
        temporary = scoring/"posthoc_boost_progress.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(scoring/"posthoc_boost_progress.json")

    def run(item):
        name, uuid, jobfile, count = item
        command = ["bash", "scripts/experiments/gaussiandir/python_env.sh", "-m",
            "scripts.experiments.gaussiandir.study_evaluate", "--config",
            "experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_convergence_v1.json",
            "--out", str(evaluations), "--jobs", str(jobfile), "--gpu-uuid", uuid]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        with (scoring/f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[3],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(worker=name, gpu_uuid=uuid, jobs=count, exitcode=result.returncode)
        with lock:
            (state["failures"] if result.returncode else state["completed"]).append(row)
            save_state()
        return row

    save_state()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(run, workers):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
