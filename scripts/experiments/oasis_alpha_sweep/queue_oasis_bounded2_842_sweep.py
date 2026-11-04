#!/usr/bin/env python
"""P76 durable, continue-on-individual-error, memory-gated eight-slot queue."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import run_oasis_bounded2_842_sweep as r

GPUS = ("0", "1", "2", "3")
POLICY = {"max_concurrent": 8, "workers_per_gpu": 2, "min_free_mib_per_slot": 8192,
          "oom_retries": 1, "min_host_available_gib": 32, "min_disk_available_gib": 200,
          "individual_failure": "preserve and defer; continue unrelated jobs",
          "shared_failure": "pause admissions; keep polling running children",
          "retry": "OOM once, after ordinary work; never repair numerics automatically"}


@contextmanager
def deadline(seconds=30):
    def expired(*_):
        raise TimeoutError("Controller I/O deadline")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def matrix(smoke=False):
    weights = (.01, 1., 100.) if smoke else r.LAMBDAS
    # Lambda-major order distributes the first cohort across all GPUs.
    return [{"rank": rank, "alpha": a, "smoke": smoke, "name": r.run_name(rank, a, smoke),
             "status": "pending", "attempt": 0, "oom_retries": 0}
            for a in weights for rank in (range(1) if smoke else range(10))]


def capacity(gpu, active_gpus, free_mib, required_mib):
    count = active_gpus.count(gpu)
    return gpu in GPUS and len(active_gpus) < 8 and count < 2 and free_mib - count * required_mib >= required_mib


def next_status(action, code, marker_exists, tail, retries):
    if code == 0 and marker_exists:
        return "audit_pending" if action == "worker" else "complete"
    oom = "cuda" in tail.lower() and "out of memory" in tail.lower()
    return "oom_retry" if oom and retries < 1 else "failed_deferred"


def process_record(pid):
    try:
        path = Path("/proc") / str(pid)
        stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19], "uid": path.stat().st_uid,
                "argv": (path / "cmdline").read_bytes().split(b"\0")[:-1], "state": stat[0]}
    except (OSError, IndexError):
        return None


def live(record):
    current = process_record(record["pid"])
    return current and current["state"] != "Z" and current["start_ticks"] == record["start_ticks"] and current["uid"] == 1009


def experiment_children(root):
    """Adopt only this exact experiment's children, including a spawn/save crash gap."""
    found = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        rec = process_record(int(path.name))
        if not rec or rec["uid"] != 1009 or rec["state"] == "Z":
            continue
        argv = [s.decode(errors="replace") for s in rec["argv"]]
        action = next((kind for script, kind in (("run_oasis_bounded2_842_sweep.py", "worker"),
                       ("audit_oasis_bounded2_842_sweep.py", "audit")) if any(a.endswith("/" + script) for a in argv)), None)
        if action is None:
            continue
        option = "--out-dir" if action == "worker" else "--run-dir"
        if option not in argv or "--gpu" not in argv:
            continue
        out = Path(argv[argv.index(option) + 1])
        if not out.is_relative_to(root):
            continue
        if str(out) in found:
            raise RuntimeError("Duplicate live workers for one output")
        found[str(out)] = {"record": {k: v for k, v in rec.items() if k != "argv"},
                           "action": action, "gpu": argv[argv.index("--gpu") + 1]}
    return found


def tail(path):
    try:
        with deadline(5), Path(path).open("rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 32768))
            return f.read().decode(errors="replace")
    except (OSError, ValueError, TimeoutError):
        return "[log temporarily unavailable]"


def telemetry():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=15, check=True)
    rows = {}
    for line in proc.stdout.splitlines():
        gpu, free, used, utilization = [v.strip() for v in line.split(",")]
        if gpu in GPUS:
            rows[gpu] = {"free_mib": int(free), "used_mib": int(used), "utilization": int(utilization)}
    if set(rows) != set(GPUS):
        raise RuntimeError("Incomplete GPU telemetry")
    return rows


class Queue:
    def __init__(self, root, manifest, smoke, required):
        self.root, self.manifest, self.smoke, self.required = root, manifest, smoke, required
        self.state_path = root / ("smoke_state.json" if smoke else "production_state.json")
        self.jobs, self.active = matrix(smoke), {}
        self.started = time.time(); self.last_check = 0.; self.shared_error = None

    def save(self):
        with deadline():
            r.atomic_json(self.state_path, {"jobs": self.jobs, "policy": POLICY,
                          "required_mib": self.required, "updated_at": r.c.now_text()})

    def reconcile(self):
        children = experiment_children(self.root)
        if not self.state_path.exists() and not children:
            self.save(); return
        saved = r.c.read_json(self.state_path) if self.state_path.exists() else {"policy": POLICY, "jobs": self.jobs}
        if saved["policy"] != POLICY or {j["name"] for j in saved["jobs"]} != {j["name"] for j in self.jobs}:
            raise RuntimeError("Queue matrix/policy changed")
        self.jobs = saved["jobs"]
        for job in self.jobs:
            matches = [(out, child) for out, child in children.items()
                       if Path(out).parent == self.root / job["name"]]
            if len(matches) > 1:
                raise RuntimeError("Multiple live attempts for one configuration")
            if matches:
                out, child = matches[0]
                job.update(out=out, **child, status="running" if child["action"] == "worker" else "auditing")
                try:
                    job["log"] = os.readlink(f"/proc/{child['record']['pid']}/fd/1")
                except OSError:
                    job.setdefault("log", "")
                children.pop(out)
            if job.get("record") and live(job["record"]):
                record = process_record(job["record"]["pid"])
                argv = [x.decode() for x in record["argv"]]
                if job.get("out") not in argv or not any("oasis_bounded2_842_sweep.py" in a for a in argv):
                    raise RuntimeError("Live PID ownership/command mismatch")
                self.active[job["name"]] = {"job": job, "process": None}
            elif job["status"] in ("launching", "running", "auditing"):
                marker = "DONE" if job.get("action") == "worker" else "independent_audit.json"
                if (Path(job["out"]) / marker).exists():
                    job["status"] = "audit_pending" if job["action"] == "worker" else "complete"
                else:
                    self.failure(job, "Interrupted process; preserve incomplete attempt", tail(job.get("log", "")))
            if job["status"] == "complete":
                try:
                    row = r.validate(Path(job["out"]), self.manifest, job["rank"], job["alpha"], job["smoke"])
                    r.audit_ok(row)
                except Exception:
                    self.failure(job, "Resume verification: " + traceback.format_exc(), "")
        counts = Counter(x["job"]["gpu"] for x in self.active.values())
        # Other phase children may exist after controller restart. Never admit a
        # new phase while unknown live children consume its capacity.
        if children:
            raise RuntimeError("Other-phase experiment children still live; wait before restart")
        if len(self.active) > 8 or any(n > 2 for n in counts.values()):
            raise RuntimeError("Adopted jobs exceed approved capacity")
        self.save()

    def failure(self, job, reason, log):
        disposition = next_status(job.get("action", "worker"), 1, False, log, job["oom_retries"])
        event = {"name": job["name"], "action": job.get("action"), "reason": reason, "log_tail": log,
                 "out": job.get("out"), "gpu": job.get("gpu"), "attempt": job["attempt"],
                 "disposition": disposition, "at": r.c.now_text()}
        with deadline():
            r.atomic_json(self.root / "failures" / f"{time.time_ns()}.json", event)
        if disposition == "oom_retry":
            job["oom_retries"] += 1
        job.update(status=disposition, failure=event, retry_action=job.get("action", "worker"))
        self.save()
        print(json.dumps({"failure": event}), flush=True)

    def poll(self):
        for name, item in list(self.active.items()):
            job, process = item["job"], item["process"]
            code = process.poll() if process else (None if live(job["record"]) else 0)
            if code is None:
                continue
            marker = "DONE" if job["action"] == "worker" else "independent_audit.json"
            try:
                with deadline():
                    exists = (Path(job["out"]) / marker).is_file()
                    if code or not exists:
                        self.failure(job, f"{job['action']} exit {code}, marker={exists}", tail(job["log"]))
                    elif job["action"] == "worker":
                        job["status"] = "audit_pending"
                    else:
                        row = r.c.read_json(Path(job["out"]) / "summary.json")
                        audit = r.audit_ok(row)
                        job.update(status="complete", optimizer_sec=row["training_elapsed_sec"],
                                   worker_sec=row["total_elapsed_sec"], audit_sec=audit["audit_elapsed_sec"])
                del self.active[name]
                self.save()
                print(json.dumps({"finished": name, "status": job["status"], "gpu": job["gpu"]}), flush=True)
            except (OSError, TimeoutError):
                # Retain slot and retry observation after shared storage recovers.
                self.shared_error = traceback.format_exc()
            except Exception:
                self.failure(job, "Audit validation: " + traceback.format_exc(), "")
                del self.active[name]

    def infrastructure(self):
        with deadline():
            if time.time() - self.last_check > 60:
                r.assert_manifest(self.manifest)
                self.last_check = time.time()
            if shutil.disk_usage(self.root).free / 2**30 < POLICY["min_disk_available_gib"]:
                raise RuntimeError("Artifact disk reserve below 200 GiB")
            available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            if available / 2**20 < POLICY["min_host_available_gib"]:
                raise RuntimeError("Available host RAM below 32 GiB")
        return telemetry()

    def start(self, job, gpu, snapshot):
        retry = job["status"] == "oom_retry"
        action = job.get("retry_action", "worker") if retry else ("audit" if job["status"] == "audit_pending" else "worker")
        if retry and action == "worker":
            job["attempt"] += 1
        path = Path(job["out"]) if action == "audit" else self.root / job["name"] / f"attempt{job['attempt']}"
        with deadline():
            if action == "worker" and path.exists():
                self.failure(job, "Refusing existing attempt output", ""); return
            if action == "audit" and (path / "independent_audit.json").exists():
                self.failure(job, "Refusing existing independent audit", ""); return
            path.parent.mkdir(parents=True, exist_ok=True)
            invocation = str(time.time_ns())
            logfile = path.parent / f"attempt{job['attempt']}_{action}_{invocation}.log"
            stream = logfile.open("x")
        job.update(out=str(path), gpu=gpu, action=action, log=str(logfile), status="launching", launch_at=time.time())
        self.save()  # durable launch intent before spawning
        script = "run_oasis_bounded2_842_sweep.py" if action == "worker" else "audit_oasis_bounded2_842_sweep.py"
        command = [sys.executable, "-u", "-B", str(Path(__file__).with_name(script))]
        if action == "worker":
            command += ["worker", "--out-dir", str(path), "--case-rank", str(job["rank"]), "--alpha", str(job["alpha"])]
            if job["smoke"]:
                command.append("--smoke")
        else:
            command += ["--run-dir", str(path)]
        command += ["--manifest", str(self.root / "launch_manifest.json"), "--gpu", gpu]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        except OSError:
            stream.close(); self.failure(job, "Process spawn: " + traceback.format_exc(), ""); return
        stream.close()
        rec = process_record(process.pid)
        if rec:
            rec = {k: v for k, v in rec.items() if k != "argv"}
        else:
            rec = {"pid": process.pid, "start_ticks": "exited", "uid": 1009}
        job.update(status="running" if action == "worker" else "auditing", record=rec)
        self.active[job["name"]] = {"job": job, "process": process}
        with deadline():
            r.atomic_json(self.root / "launches" / f"{invocation}.json", {"name": job["name"], "action": action,
                          "gpu": gpu, "pid": process.pid, "command": command, "gpu_snapshot": snapshot,
                          "required_mib": self.required, "at": r.c.now_text()})
        self.save()
        print(json.dumps({"launched": job["name"], "action": action, "gpu": gpu, "pid": process.pid}), flush=True)

    def progress(self):
        counts = Counter(j["status"] for j in self.jobs)
        complete = [j for j in self.jobs if j["status"] == "complete"]
        elapsed = time.time() - self.started
        remaining = len(self.jobs) - counts["complete"] - counts["failed_deferred"]
        value = {"phase": "smoke" if self.smoke else "production_and_audit", "counts": dict(counts),
                 "total": len(self.jobs), "completed": len(complete), "elapsed_sec": elapsed,
                 "throughput_eta_sec": elapsed / len(complete) * remaining if complete else None,
                 "active": [{"name": j["name"], "gpu": j["gpu"], "action": j["action"], "pid": j["record"]["pid"]}
                            for j in self.jobs if j["name"] in self.active],
                 "shared_admission_error": self.shared_error, "updated_at": r.c.now_text()}
        with deadline():
            r.atomic_json(self.root / "progress.json", value)

    def run(self):
        self.reconcile()
        while any(j["status"] not in ("complete", "failed_deferred") for j in self.jobs) or self.active:
            try:
                self.poll()
                snapshot = self.infrastructure()
                self.shared_error = None
                ordinary = any(j["status"] in ("pending", "running", "audit_pending", "auditing", "launching") for j in self.jobs)
                for _ in range(8):
                    pending = [j for j in self.jobs if j["status"] in ("audit_pending", "pending") or (not ordinary and j["status"] == "oom_retry")]
                    if not pending:
                        break
                    assignments = []
                    active_gpus = [x["job"]["gpu"] for x in self.active.values()]
                    for job in pending:
                        audit = job["status"] == "audit_pending" or (job["status"] == "oom_retry" and job.get("retry_action") == "audit")
                        for gpu in ([job["gpu"]] if audit else GPUS):
                            if capacity(gpu, active_gpus, snapshot[gpu]["free_mib"], self.required):
                                assignments.append(((not audit, active_gpus.count(gpu), -snapshot[gpu]["free_mib"], pending.index(job), gpu), job, gpu))
                    if not assignments:
                        break
                    _, job, gpu = min(assignments, key=lambda x: x[0])
                    self.start(job, gpu, snapshot)
                    snapshot = telemetry()  # do not reuse stale launch memory
                self.progress()
            except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError, ValueError):
                self.shared_error = traceback.format_exc()
                print(json.dumps({"admissions_paused": self.shared_error, "at": r.c.now_text()}), flush=True)
                try:
                    self.progress()
                except Exception:
                    pass
            time.sleep(3)
        self.save(); self.progress()
        return all(j["status"] == "complete" for j in self.jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    if args.artifact_root.resolve().is_relative_to(r.c.REPO_ROOT):
        raise RuntimeError("External artifact root required")
    root = args.artifact_root / "runs" / r.VARIANT
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = r.prepare(args.artifact_root)
        smoke = Queue(root, manifest, True, 8192)
        if (root / "SMOKES_PASSED.json").exists():
            smoke.jobs = r.c.read_json(smoke.state_path)["jobs"]
            if {j["name"] for j in smoke.jobs} != {j["name"] for j in matrix(True)} or any(j["status"] != "complete" for j in smoke.jobs):
                raise RuntimeError("Completed smoke matrix changed")
        elif not smoke.run():
            raise RuntimeError("Smoke failure: preserved; production not admitted")
        peaks = []
        for job in smoke.jobs:
            row = r.validate(Path(job["out"]), manifest, job["rank"], job["alpha"], True)
            audit = r.audit_ok(row)
            peaks += [row[k] for k in ("peak_cuda_gib", "peak_reserved_gib")]
            peaks += [audit[k] for k in ("peak_cuda_gib", "peak_reserved_gib")]
        required = max(8192, math.ceil((max(peaks) + 2) * 1024))
        r.atomic_json(root / "SMOKES_PASSED.json", {"passed": True, "count": 3, "maximum_peak_gib": max(peaks),
                      "required_mib": required, "source_hashes": manifest["source_hashes"], "at": r.c.now_text()})
        if args.stop_after == "smoke":
            return
        production = Queue(root, manifest, False, required)
        passed = production.run()
        r.atomic_json(root / "QUEUE_FINISHED.json", {"complete": passed,
                      "completed": sum(j["status"] == "complete" for j in production.jobs), "total": 70,
                      "failures": [j for j in production.jobs if j["status"] != "complete"], "at": r.c.now_text()})
        print(json.dumps({"queue_finished": True, "complete": passed}), flush=True)


if __name__ == "__main__":
    main()
