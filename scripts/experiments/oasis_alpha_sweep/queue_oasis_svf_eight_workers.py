#!/usr/bin/env python
"""P74 orchestration-only handover: two unchanged workers/auditors per GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_oasis_bspline_svf_sweep_recovery as r

GPUS = ("0", "1", "2", "3")
PER_GPU = 2
MAX_ACTIVE = 8
FOLDER = "parallel8_v1"
WORKER = Path(r.__file__).resolve()


def sources():
    return {Path(__file__).name: r.sha(__file__),
            "test_oasis_svf_eight_workers.py": r.sha(r.c.REPO_ROOT/"tests/test_oasis_svf_eight_workers.py")}


def process_info(pid):
    try:
        path = Path("/proc")/str(pid)
        fields = (path/"stat").read_text().rsplit(")", 1)[1].split()
        argv = [x.decode() for x in (path/"cmdline").read_bytes().split(b"\0") if x]
        return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0],
                "argv": argv, "uid": path.stat().st_uid}
    except (OSError, UnicodeError, IndexError):
        return None


def same_process(record):
    current = process_info(record["pid"])
    return bool(current and current["start_ticks"] == record["start_ticks"]
                and current["state"] != "Z")


def option(argv, key, default=None):
    return argv[argv.index(key)+1] if key in argv else default


def experiment_processes(root):
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit(): continue
        rec = process_info(path.name)
        if not rec or rec["uid"] != 1009 or str(WORKER) not in rec["argv"]: continue
        argv = rec["argv"]; action = argv[argv.index(str(WORKER))+1]
        if action == "launch":
            if Path(option(argv, "--artifact-root", "/database/inrorspline"))/"runs"/r.s.VARIANT != root:
                continue
        elif action in ("worker", "audit"):
            out = Path(option(argv, "--out-dir", "/"))
            if not out.is_relative_to(root): continue
            rec.update(out=str(out), gpu=option(argv, "--gpu"))
        else: continue
        rec["action"] = action
        found.append(rec)
    return found


def handover(root, allow):
    controllers = [x for x in experiment_processes(root) if x["action"] == "launch"]
    if len(controllers) > 1: raise RuntimeError("Multiple old controllers; audit before takeover")
    if controllers and not allow: raise RuntimeError("Use --take-over to replace this sweep controller")
    folder = root/FOLDER; folder.mkdir(exist_ok=True)
    record = {"created_at": r.c.now_text(), "controllers": controllers,
              "processes_before": experiment_processes(root),
              "progress_before": r.c.read_json(root/"progress.json"),
              "action": "SIGTERM controller PID only; workers/auditors continue"}
    r.atomic_json(folder/f"handover_{time.time_ns()}.json", record)
    for controller in controllers:
        current = process_info(controller["pid"])
        if (not current or current["start_ticks"] != controller["start_ticks"]
                or current["argv"] != controller["argv"]):
            raise RuntimeError("Controller identity changed during takeover")
        os.kill(controller["pid"], signal.SIGTERM)
        deadline = time.monotonic()+10
        while same_process(controller) and time.monotonic() < deadline: time.sleep(.2)
        if same_process(controller): raise RuntimeError("Old controller did not exit; workers left untouched")


def resume_jobs(root, manifest, live):
    by_path = {}
    for proc in live:
        if proc["action"] == "launch": raise RuntimeError("Old controller still dispatching")
        if proc["out"] in by_path: raise RuntimeError("Duplicate workers for one attempt")
        by_path[proc["out"]] = proc
    jobs, active, affinity = [], {}, {}
    approved = manifest["recovery"]["approved_retry"]
    for spec in r.s.matrix():
        job = {"spec": spec, "status": "pending", "gpu": None, "attempt": 0, "row": None}
        attempts = sorted((root/spec.name()).glob("attempt*"))
        attempts = [x for x in attempts if x.is_dir()]
        accepted = [x for x in attempts if (x/"independent_audit.json").is_file()]
        if len(accepted) > 1: raise RuntimeError("Duplicate accepted candidate")
        if accepted:
            path = accepted[0]
            row = r.validate(path, manifest, spec)
            job.update(status="complete", out=path, attempt=int(path.name[7:]), gpu=row["gpu"], row=row)
            proc = by_path.pop(str(path), None)
            if proc:
                if proc["action"] != "audit": raise RuntimeError("Accepted run has an active optimizer")
                job.update(status="auditing", gpu=proc["gpu"])
                active[proc["pid"]] = {"record": proc, "process": None, "job": job,
                                       "action": "audit", "log": None, "adopted": True}
        elif attempts:
            path = max(attempts, key=lambda x: int(x.name[7:]))
            job.update(out=path, attempt=int(path.name[7:]))
            proc = by_path.pop(str(path), None)
            if proc:
                job.update(status="running" if proc["action"] == "worker" else "auditing", gpu=proc["gpu"])
                active[proc["pid"]] = {"record": proc, "process": None, "job": job,
                                       "action": proc["action"], "log": None, "adopted": True}
            elif (path/"DONE").is_file():
                job.update(status="audit_pending", gpu=r.c.read_json(path/"summary.json")["gpu"])
            elif str(path) == approved["path"] and r.sha(path/"FAILED_worker.json") == approved["failure_sha256"]:
                job.update(attempt=job["attempt"]+1)
            else:
                raise RuntimeError(f"Unexplained incomplete attempt preserved:{path}")
        if job["gpu"] is not None: affinity.setdefault(spec.case_rank, job["gpu"])
        jobs.append(job)
    if by_path: raise RuntimeError(f"Unrecognized active processes:{by_path}")
    counts = Counter(x["job"]["gpu"] for x in active.values())
    if len(active) > MAX_ACTIVE or any(n > PER_GPU for n in counts.values()):
        raise RuntimeError("Existing sweep processes exceed approved concurrency")
    return jobs, active, affinity


def has_capacity(gpu, active, free_mib, required_mib):
    counts = Counter(x["job"]["gpu"] for x in active.values())
    return (gpu in GPUS and len(active) < MAX_ACTIVE and counts[gpu] < PER_GPU
            and free_mib >= required_mib)


def choose_job(jobs, gpu, affinity, failures, completed):
    candidates = []
    for index, job in enumerate(jobs):
        if job["status"] not in ("pending", "retry", "audit_pending"): continue
        if failures and job["status"] != "audit_pending": continue
        if job.get("retry_after", -1) >= completed: continue
        audit = job["status"] == "audit_pending"
        preferred = job["gpu"] if audit else affinity.get(job["spec"].case_rank)
        candidates.append(((not audit, preferred != gpu, preferred is not None, index), job))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def finished_code(item):
    if item["process"] is not None: return item["process"].poll()
    if same_process(item["record"]): return None
    marker = "DONE" if item["action"] == "worker" else "independent_audit.json"
    return 0 if (item["job"]["out"]/marker).is_file() else 1


def run_queue(root, manifest, jobs, active, affinity, required):
    started = time.monotonic()
    at_resume = sum(j["status"] == "complete" for j in jobs)
    failures, last_sample = [], 0.
    while True:
        for pid, item in list(active.items()):
            code = finished_code(item)
            if code is None: continue
            if item["log"]: item["log"].close()
            job = item["job"]
            if code:
                logs = list(job["out"].parent.glob(f"attempt{job['attempt']}_{item['action']}_*.log"))
                content = "\n".join(x.read_text(errors="replace")[-16000:] for x in logs)
                oom = "out of memory" in content.lower() and "cuda" in content.lower()
                failure = {"spec": asdict(job["spec"]), "action": item["action"], "gpu": job["gpu"],
                           "attempt": job["attempt"], "returncode": code, "oom": oom,
                           "logs": [str(x) for x in logs], "created_at": r.c.now_text()}
                r.atomic_json(root/FOLDER/f"failure_{time.time_ns()}.json", failure)
                if oom and not job.get("oom_retried"):
                    job.update(oom_retried=True, retry_after=sum(j["status"] == "complete" for j in jobs),
                               status="retry" if item["action"] == "worker" else "audit_pending")
                    if item["action"] == "worker": job["attempt"] += 1
                else:
                    failures.append(failure); job["status"] = "failed"
            elif item["action"] == "worker":
                job["status"] = "audit_pending"
            else:
                job["row"] = r.validate(job["out"], manifest, job["spec"])
                job["status"] = "complete"
            del active[pid]
        completed = sum(j["status"] == "complete" for j in jobs)
        if failures:
            for job in jobs:
                if job["status"] in ("pending", "retry"): job["status"] = "blocked"
        snapshot = r.legacy.gpu_snapshot()
        free = {str(x["physical_gpu"]): x["memory_free_mib"] for x in snapshot.get("rows", [])}
        if time.monotonic()-last_sample >= 30:
            with (root/FOLDER/"gpu_load.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": r.c.now_text(), "snapshot": snapshot})+"\n")
            last_sample = time.monotonic()
        for _ in range(PER_GPU):
            for gpu in GPUS:
                if not has_capacity(gpu, active, free.get(gpu, 0), required): continue
                job = choose_job(jobs, gpu, affinity, failures, completed)
                if job is None: continue
                action = "audit" if job["status"] == "audit_pending" else "worker"
                spec = job["spec"]
                path = job["out"] if action == "audit" else root/spec.name()/f"attempt{job['attempt']}"
                if action == "worker" and path.exists(): raise RuntimeError(f"Refuse overwrite:{path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", "-B", str(WORKER), action, "--manifest",
                           str(root/FOLDER/"launch_manifest.json"), "--out-dir", str(path), "--gpu", gpu]
                if action == "worker":
                    command += ["--case-rank", str(spec.case_rank), "--schedule", spec.schedule,
                                "--regularize", spec.regularize, "--alpha", str(spec.alpha)]
                log_path = path.parent/f"attempt{job['attempt']}_{action}_parallel8_{time.time_ns()}.log"
                log = log_path.open("x")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
                try: proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close(); raise
                job.update(status="running" if action == "worker" else "auditing", out=path, gpu=gpu)
                job.pop("retry_after", None)
                affinity.setdefault(spec.case_rank, gpu)
                active[proc.pid] = {"record": process_info(proc.pid), "process": proc, "job": job,
                                    "action": action, "log": log, "adopted": False}
                event = {"run": spec.name(), "action": action, "gpu": gpu, "pid": proc.pid,
                         "out": str(path), "log_path": str(log_path), "time": r.c.now_text(),
                         "free_mib_before_admission": free[gpu], "required_free_mib": required,
                         "max_concurrent": MAX_ACTIVE, "workers_per_gpu": PER_GPU,
                         "concurrency_on_gpu_after_launch": sum(x["job"]["gpu"] == gpu for x in active.values())}
                with (root/FOLDER/"launch_events.jsonl").open("a") as stream: stream.write(json.dumps(event)+"\n")
                print(json.dumps(event), flush=True)
                free[gpu] -= required  # Reserve capacity for workers still loading.
        elapsed = time.monotonic()-started
        new_done = completed-at_resume
        eta = elapsed/new_done*(len(jobs)-completed) if new_done and not failures else None
        phases = []
        for item in active.values():
            job = item["job"]; path = job["out"]/"progress.json"
            detail = r.c.read_json(path) if path.exists() else {"phase": "loading"}
            phases.append({"gpu": job["gpu"], "action": item["action"], "run": job["spec"].name(),
                           "adopted": item["adopted"], "pid": item["record"]["pid"], **detail})
        status = {"phase": ("draining_for_audit" if active else "stopped_for_audit") if failures else "production",
                  "completed": completed, "total": len(jobs), "failed": failures, "active_phases": phases,
                  "max_concurrent": MAX_ACTIVE, "workers_per_gpu": PER_GPU, "elapsed_sec": elapsed,
                  "eta_sec_observed_audited_throughput": eta, "completed_at_handover": at_resume,
                  "required_free_mib": required, "updated_at": r.c.now_text(), "case_gpu_affinity": affinity,
                  "workers_finished": sum(j["status"] in ("complete", "audit_pending", "auditing") for j in jobs),
                  "jobs": [{"spec": asdict(j["spec"]), "status": j["status"], "attempt": j["attempt"],
                            "gpu": j["gpu"], "out": str(j.get("out", ""))} for j in jobs]}
        r.atomic_json(root/"progress.json", status)
        if not active and (completed == len(jobs) or failures): break
        if not active:
            # No other work can release a slot at the tail; an OOM retry may now proceed.
            for job in jobs: job.pop("retry_after", None)
        time.sleep(5)
    if failures: raise RuntimeError("Failure preserved; stopped new dispatch for audit")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    p.add_argument("--gpus", choices=("0,1,2,3",), default="0,1,2,3")
    p.add_argument("--max-concurrent", type=int, choices=(8,), default=8)
    p.add_argument("--workers-per-gpu", type=int, choices=(2,), default=2)
    p.add_argument("--take-over", action="store_true")
    a = p.parse_args(argv)
    if a.artifact_root.resolve().is_relative_to(r.c.REPO_ROOT): p.error("External artifacts only")
    return a


def main(args):
    if (os.getuid(), os.getgid()) != (1009, 1020): raise RuntimeError("Require UID:GID1009:1020")
    root = args.artifact_root/"runs"/r.s.VARIANT
    old_path = root/"recovery_v1/launch_manifest.json"
    manifest = r.c.read_json(old_path)
    r.assert_manifest(manifest)
    if (root/"COMPLETE.json").exists():
        print("All fits already complete; nothing to restart", flush=True); return
    folder = root/FOLDER; folder.mkdir(exist_ok=True)
    with (folder/"handover.lock").open("a") as handover_lock:
        fcntl.flock(handover_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        handover(root, args.take_over)
        with (root/"controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            path = folder/"launch_manifest.json"
            if path.exists():
                manifest = r.c.read_json(path)
                if manifest["execution"]["source_hashes"] != sources(): raise RuntimeError("Queue source drift")
            else:
                manifest = {**manifest, "max_concurrent": MAX_ACTIVE, "workers_per_gpu": PER_GPU,
                            "execution": {"source_hashes": sources(), "changed": "scheduler_only",
                                          "parent_manifest_sha256": r.sha(old_path), "created_at": r.c.now_text(),
                                          "gpus": list(GPUS), "max_concurrent": MAX_ACTIVE,
                                          "workers_per_gpu": PER_GPU, "initial_gpu_state": r.legacy.gpu_snapshot()}}
                r.atomic_json(path, manifest)
            r.assert_manifest(manifest)
            jobs, active, affinity = resume_jobs(root, manifest, experiment_processes(root))
            r.atomic_json(folder/"adopted_processes.json", {"processes": [x["record"] for x in active.values()],
                          "completed": sum(j["status"] == "complete" for j in jobs), "created_at": r.c.now_text()})
            required = max(8192, r.c.read_json(root/"recovery_v1/smoke_passed.json")["required_free_mib"])
            run_queue(root, manifest, jobs, active, affinity, required)
            r.report(args.artifact_root)
            out = args.artifact_root/"repo_experiments"/r.s.VARIANT
            r.atomic_json(out/"execution_concurrency.json", manifest["execution"])
            (out/"execution_concurrency.md").write_text(
                "# Execution concurrency\n\nUser approved two workers/auditors per GPU (eight total) partway through the sweep. "
                "Numerical worker and audit sources are unchanged. Existing children were adopted without interruption; "
                "completed runs were not rerun. Timing is mixed-concurrency, shared-load diagnostic, not an isolated "
                "speed benchmark. See parallel8_v1 launch events for per-run placement and transition timing.\n")


if __name__ == "__main__":
    args = parse_args()
    try: main(args)
    except BaseException:
        root = args.artifact_root/"runs"/r.s.VARIANT
        if (root/FOLDER).exists():
            r.atomic_json(root/FOLDER/f"controller_failure_{time.time_ns()}.json",
                          {"traceback": traceback.format_exc(), "created_at": r.c.now_text()})
        raise
