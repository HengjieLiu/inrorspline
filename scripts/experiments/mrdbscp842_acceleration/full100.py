"""OASIS100 fixed-50 CP-BE timing plus adaptive 16/32-way evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .overnight_candidates import candidate, optional_candidates
from .run_be_combo_first10 import ComboSuite
from .run_suite import Worker, dump, process_identity

EXPERIMENT = "2026-09-24_oasis_cpbe_fixed50_first100"


def candidates():
    base = candidate("fixed_50_50_50", (50, 50, 50))
    values = {value["name"]: value for value in optional_candidates(base)}
    return [values["fixed_50_50_50_cached_batch"], values["fixed_50_50_50_amsgrad"]]


def gpu_count():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True).splitlines()
    return len(lines)


def gpu_is_clear(index):
    text = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True).strip()
    return not text


def clear_gpus(samples=1):
    common = {index for index in range(gpu_count()) if gpu_is_clear(index)}
    for _ in range(samples - 1):
        time.sleep(1)
        common &= {index for index in range(gpu_count()) if gpu_is_clear(index)}
    return sorted(common)


def mem_available_gib():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0])
    return fields["MemAvailable"] / 1024 / 1024


def gpu_used_gib(indices):
    result = {}
    for index in indices:
        value = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
        result[index] = float(value) / 1024
    return result


class EvalWorker:
    """Persistent untimed evaluator. Peer evaluators on the same GPU are allowed."""

    def __init__(self, suite, gpu, ordinal):
        self.suite, self.gpu, self.index = suite, int(gpu), 0
        self.name = f"parallel_eval_g{gpu}_{ordinal:02d}_{uuid.uuid4().hex[:8]}"
        self.mail = suite.root / "evaluation_workers" / self.name
        self.mail.mkdir(parents=True)
        token = uuid.uuid4().hex
        args = suite.args
        self.argv = [
            "docker", "exec", "-u", "1009:1009", "-w", args.repo_container,
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=6",
            "-e", "MKL_NUM_THREADS=6", "-e", "OPENBLAS_NUM_THREADS=6",
            "-e", f"CUDA_VISIBLE_DEVICES={gpu}", args.container, "python", "-B", "-m",
            "scripts.experiments.mrdbscp842_acceleration.worker",
            "--job-token", token, "--mailbox", suite.container(self.mail),
            "--gpu", str(gpu), "--features", "", "--oasis-root", args.oasis_root,
            "--pair-file",
            args.repo_container + "/configs/splits/oasis/pair_test_200_shuffled_seed000003.txt",
        ]
        dump(self.mail / "launch.json", self.argv)
        self.log = (self.mail / "process.log").open("w")
        self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT)
        self.wait(self.mail / "ready.json", 180)
        ready = json.loads((self.mail / "ready.json").read_text())
        matches = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                ident = process_identity(int(proc.name))
                if (token in ident["argv"]
                        and "scripts.experiments.mrdbscp842_acceleration.worker" in ident["argv"]
                        and Path(ident["argv"][0]).name.startswith("python")):
                    matches.append(ident)
            except (OSError, ValueError):
                pass
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify evaluation worker: {matches}")
        status = Path(f"/proc/{matches[0]['pid']}/status").read_text()
        ns = next((line.split()[1:] for line in status.splitlines()
                   if line.startswith("NSpid:")), [])
        if not ns or int(ns[-1]) != ready["container_pid"]:
            raise RuntimeError("Evaluation worker PID namespace mismatch")
        dump(self.mail / "host_ack.json", matches[0])

    def wait(self, path, timeout):
        start = time.monotonic()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Evaluation worker exited; see {self.mail}")
            if time.monotonic() - start > timeout:
                raise TimeoutError(str(path))
            time.sleep(.2)

    def execute(self, item):
        base = f"quality_{item['variant']}_rank{int(item['rank']):03d}"
        attempt = 0
        while (self.suite.root / "jobs" / f"{base}_attempt{attempt}").exists():
            attempt += 1
        label = f"{base}_attempt{attempt}"
        output = self.suite.root / "jobs" / label
        output.mkdir(parents=True)
        command = {
            "action": "schedule_evaluate",
            "output": self.suite.container(output),
            "item": item,
        }
        dump(self.mail / f"command_{self.index:04d}.json", command)
        result_path = self.mail / f"result_{self.index:04d}.json"
        self.wait(result_path, 1200)
        payload = json.loads(result_path.read_text())
        self.index += 1
        if not payload["ok"]:
            raise RuntimeError(payload["traceback"])
        value = payload["result"]
        value.update(id=label, worker=self.name, physical_gpu=self.gpu, clean=True)
        dump(output / "accepted_result.json", value)
        return value

    def close(self):
        if self.process.poll() is None:
            dump(self.mail / f"command_{self.index:04d}.json", {"action": "exit"})
            self.process.wait(timeout=180)
        self.log.close()


class Full100Suite(ComboSuite):
    def registration_block(self, spec):
        name = spec["name"]
        ranks = [0, 0] + list(range(1, 100))
        worker = None
        try:
            for index, rank in enumerate(ranks):
                prefix = f"headline_{name}_{index:03d}_attempt"
                prior = [
                    row for row in self.rows
                    if row.get("block") == "headline_" + name and row["id"].startswith(prefix)
                ]
                if prior:
                    if len(prior) != 1 or int(prior[0]["rank"]) != rank:
                        raise RuntimeError("Registration resume manifest mismatch")
                    continue
                attempt = 0
                while (self.root / "jobs" / f"{prefix}{attempt}").exists():
                    attempt += 1
                while True:
                    if worker is None:
                        worker = Worker(self, "headline_" + name, spec["features"])
                        if index > 0:
                            prime = worker.execute(
                                "schedule_register", f"prime_{name}_{uuid.uuid4().hex[:8]}",
                                rank=rank, candidate=spec, command_timeout=900)
                            if not prime["clean"]:
                                worker.close()
                                worker = None
                                continue
                    label = f"{prefix}{attempt}"
                    row = worker.execute(
                        "schedule_register", label, rank=rank, candidate=spec,
                        command_timeout=900)
                    if row["clean"]:
                        self.record(
                            row, variant=name, phase="headline", block="headline_" + name,
                            job_index=index, condition="cold" if index == 0 else "warm",
                            primary=True)
                        print(json.dumps({
                            "completed": label, "seconds": row["elapsed_sec"],
                            "rank": rank,
                        }), flush=True)
                        break
                    attempt += 1
                    if index == 0:
                        worker.close()
                        worker = None
                    else:
                        self.monitor.wait_clear()
        finally:
            if worker:
                worker.close()

    def preflight_full100(self, specs):
        self.source_checks("preflight_source_checks")
        environment = self.auxiliary(
            "environment", "data_environment", ranks=list(range(100)))
        dump(self.root / "data_environment.json", environment)
        for spec in specs:
            worker = Worker(self, "smoke_" + spec["name"], spec["features"])
            label = "smoke_" + spec["name"]
            try:
                row = worker.execute(
                    "schedule_register", label, rank=0, candidate=spec,
                    smoke=True, command_timeout=900)
            finally:
                worker.close()
            if not row["clean"] or list(row["features"]) != list(spec["features"]):
                raise RuntimeError("Candidate smoke or feature activation failed")
        dump(self.root / "preflight_DONE.json", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(specs),
        })

    def registration_rows(self):
        return [
            row for row in self.rows
            if row.get("phase") == "headline" and row.get("primary")
        ]

    def _new_eval_workers(self, count, gpus, offset=0):
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(EvalWorker, self, gpus[i % len(gpus)], offset + i)
                for i in range(count)
            ]
            return [future.result() for future in futures]

    def _eval_wave(self, workers, items, label):
        chunks = [items[i::len(workers)] for i in range(len(workers))]
        lock = threading.Lock()
        started = time.monotonic()

        def consume(worker, chunk):
            rows = []
            for item in chunk:
                row = worker.execute(item)
                rows.append(row)
                with lock:
                    self.quality.append(row)
                    dump(self.root / "quality_rows.json", self.quality)
            return rows

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [
                pool.submit(consume, worker, chunk)
                for worker, chunk in zip(workers, chunks)
            ]
            results = [future.result() for future in futures]
        elapsed = time.monotonic() - started
        snapshot = {
            "label": label,
            "cases": len(items),
            "workers": len(workers),
            "elapsed_sec": elapsed,
            "cases_per_sec": len(items) / elapsed,
            "mem_available_gib": mem_available_gib(),
            "gpu_used_gib": gpu_used_gib(sorted({worker.gpu for worker in workers})),
        }
        self.eval_calibration.append(snapshot)
        dump(self.root / "evaluation_parallelism.json", self.eval_calibration)
        return [row for group in results for row in group]

    def evaluate_parallel(self):
        warm = [
            row for row in self.registration_rows() if row["condition"] == "warm"
        ]
        if len(warm) != 200:
            raise RuntimeError(f"Expected 200 warm fields, got {len(warm)}")
        self.quality = (
            json.loads((self.root / "quality_rows.json").read_text())
            if (self.root / "quality_rows.json").exists() else []
        )
        done = {row["registration_id"] for row in self.quality}
        items = [
            {
                "id": row["id"], "rank": row["rank"],
                "variant": row["variant"], "dvf": row["dvf"],
            }
            for row in warm if row["id"] not in done
        ]
        if not items:
            return
        gpus = clear_gpus(samples=3)
        if not gpus:
            raise RuntimeError("No idle GPU available for evaluation")
        self.eval_calibration = (
            json.loads((self.root / "evaluation_parallelism.json").read_text())
            if (self.root / "evaluation_parallelism.json").exists() else []
        )
        workers = self._new_eval_workers(min(16, len(items)), gpus)
        all_workers = list(workers)
        try:
            wave16 = items[:min(32, len(items))]
            self._eval_wave(workers, wave16, "pilot_16")
            items = items[len(wave16):]
            active = workers
            if items:
                extra = self._new_eval_workers(min(16, len(items)), gpus, len(workers))
                all_workers.extend(extra)
                workers32 = workers + extra
                wave32 = items[:min(64, len(items))]
                self._eval_wave(workers32, wave32, "pilot_32")
                items = items[len(wave32):]
                a, b = self.eval_calibration[-2:]
                keep32 = (
                    b["cases_per_sec"] >= 1.10 * a["cases_per_sec"]
                    and b["mem_available_gib"] >= 64
                    and max(b["gpu_used_gib"].values(), default=0) < 40
                )
                dump(self.root / "evaluation_parallelism_decision.json", {
                    "keep_32": keep32,
                    "throughput_ratio": b["cases_per_sec"] / a["cases_per_sec"],
                    "criteria": (
                        "throughput >=1.10x, MemAvailable >=64 GiB, "
                        "each GPU <40 GiB"
                    ),
                })
                active = workers32 if keep32 else workers
                if not keep32:
                    for worker in extra:
                        worker.close()
                        all_workers.remove(worker)
                if items:
                    self._eval_wave(active, items, "remainder")
        finally:
            for worker in all_workers:
                worker.close()

    def report_full100(self):
        self.helper(
            "report", "scripts.experiments.mrdbscp842_acceleration.full100_report",
            "--root", self.container(self.root),
            "--review", self.container(self.review),
            "--artifact-root", self.args.artifact_container,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "container", "repo-container", "artifact-host", "artifact-container",
        "oasis-root", "run-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument(
        "--phase", choices=["dry-run", "preflight", "all", "evaluate", "report"],
        default="all")
    args = parser.parse_args()
    args.experiment_key = EXPERIMENT

    if args.gpu == "auto":
        available = clear_gpus(samples=3)
        if not available:
            raise RuntimeError("No GPU stayed idle for three checks")
        args.gpu_requested, args.gpu = "auto", available[0]
    else:
        args.gpu_requested, args.gpu = args.gpu, int(args.gpu)
        if args.gpu not in clear_gpus(samples=3):
            raise RuntimeError(f"Requested GPU {args.gpu} is not idle")

    artifact, repo = Path(args.artifact_host).resolve(), Path.cwd().resolve()
    if (
        not artifact.is_dir() or artifact in (Path("/"), Path.home())
        or artifact == repo or repo in artifact.parents or artifact in repo.parents
        or "/" in args.run_id
        or not args.run_id.startswith("oasis_cpbe_fixed50_first100_")
    ):
        raise ValueError("Invalid external artifact root or run ID")

    specs = candidates()
    suite = Full100Suite(args)
    suite.provenance()
    dump(suite.root / "candidate_manifest.json", specs)
    dump(suite.root / "job_manifest.json", {
        "timing_rows": 202,
        "warm_fields": 200,
        "evaluation_rows": 200,
        "ranks": list(range(100)),
        "registration_gpu": args.gpu,
        "evaluation_parallelism": "adaptive 16 then 32",
    })
    if args.phase == "dry-run":
        print(json.dumps({
            "timing_rows": 202, "evaluation_rows": 200,
            "gpu": args.gpu, "root": str(suite.root),
        }))
        return
    if args.phase == "evaluate":
        dump(suite.root / "evaluation_source_amendment.json", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "evaluation orchestration only; 202 accepted timing rows remain frozen",
            "reason": (
                "PID matcher initially included both docker-exec client and Python worker; "
                "continuation restricts ownership match to Python exactly as accepted Worker"
            ),
        })
        suite.evaluate_parallel()
        suite.source_checks("evaluation_source_checks")
        suite.report_full100()
        dump(suite.root / "evaluate_DONE.json", {
            "utc": datetime.now(timezone.utc).isoformat(),
        })
        return


    suite.monitor.thread.start()
    try:
        if (
            args.phase in ("preflight", "all")
            and not (suite.root / "preflight_DONE.json").exists()
        ):
            suite.preflight_full100(specs)
        if args.phase == "all":
            suite.frozen()
            for spec in specs:
                suite.registration_block(spec)
            dump(suite.root / "registrations_DONE.json", {
                "utc": datetime.now(timezone.utc).isoformat(),
            })
            suite.monitor.done.set()
            suite.monitor.thread.join(timeout=5)
            suite.evaluate_parallel()
            suite.frozen()
            suite.source_checks("final_source_checks")
            suite.report_full100()
        elif args.phase == "report":
            suite.report_full100()
        dump(suite.root / f"{args.phase}_DONE.json", {
            "utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        dump(suite.root / f"{args.phase}_ERROR.json", {"error": repr(exc)})
        raise
    finally:
        suite.monitor.done.set()
        if suite.monitor.thread.is_alive():
            suite.monitor.thread.join(timeout=5)


if __name__ == "__main__":
    main()
