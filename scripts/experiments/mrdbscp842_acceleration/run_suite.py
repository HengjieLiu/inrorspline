#!/usr/bin/env python3
"""Host-only orchestration, provenance and GPU contention monitoring.

No host torch/data dependencies. Workers run sequentially in the specified Docker.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid
from .nvml_monitor import NVML

FEATURES = {"reference": (), "cache": ("cache",), "batch": ("batch",),
            "ncc": ("ncc",), "buffer": ("buffer",), "cpbe": ("cpbe",)}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True)+"\n")
    tmp.replace(path)


def cmd(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def process_identity(pid):
    path = Path(f"/proc/{pid}")
    stat = (path/"stat").read_text().rsplit(")", 1)[1].split()
    argv = (path/"cmdline").read_bytes().split(b"\0")
    return dict(pid=int(pid), parent=int(stat[1]), start_ticks=int(stat[19]),
                argv=[v.decode(errors="replace") for v in argv if v])


class Monitor:
    def __init__(self, gpu, path):
        self.gpu = gpu
        self.path = path
        self.uuid = cmd(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"])
        self.nvml = NVML(gpu, self.uuid)
        self.owner = None
        self.records = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        now = time.time()
        try:
            apps = self.nvml.processes()
            allowed = set()
            owner = self.owner
            if owner:
                current = process_identity(owner["pid"])
                if current["start_ticks"] != owner["start_ticks"]:
                    raise RuntimeError("Worker PID reused")
                allowed.add(owner["pid"])
                for app in apps:
                    pid = app["pid"]
                    for _ in range(32):
                        if pid in allowed:
                            allowed.add(app["pid"])
                            break
                        if pid <= 1:
                            break
                        pid = process_identity(pid)["parent"]
            record = dict(unix=now, source="NVML_v3", apps=apps, allowed_pids=sorted(allowed),
                          foreign=[a for a in apps if a["pid"] not in allowed], error=None)
        except Exception as exc:
            record = dict(unix=now, apps=[], allowed_pids=[], foreign=[], error=repr(exc))
        with self.lock:
            self.records.append(record)
            with self.path.open("a") as f:
                f.write(json.dumps(record)+"\n")
        return record

    def loop(self):
        while not self.done.is_set():
            self.sample()
            self.done.wait(1.)

    def wait_clear(self):
        clean = 0
        announced = False
        while clean < 3:
            record = self.sample()
            if record["error"] or record["foreign"]:
                clean = 0
                if not announced:
                    print("Waiting for GPU to clear / monitor to recover", flush=True)
                    announced = True
            else:
                clean += 1
            time.sleep(1)

    def interval(self, start, end):
        with self.lock:
            records = [r for r in self.records if start <= r["unix"] <= end]
        times = sorted([start,end]+[r["unix"] for r in records])
        return dict(clean=bool(records) and all(not r["error"] and not r["foreign"] for r in records),
                    start_unix=start,end_unix=end,max_sample_gap_sec=max(b-a for a,b in zip(times,times[1:])),
                    sample_count=len(records), bad_samples=[r for r in records if r["error"] or r["foreign"]])


class Worker:
    def __init__(self, suite, name, features=()):
        self.suite = suite
        self.name = name + "_" + uuid.uuid4().hex[:10]
        self.mail = suite.root/"workers"/self.name
        self.mail.mkdir(parents=True)
        self.index = 0
        suite.monitor.wait_clear()
        token = uuid.uuid4().hex
        args = suite.args
        self.argv = ["docker", "exec", "-u", "1009:1009", "-w", args.repo_container,
                     "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=6",
                     "-e", "MKL_NUM_THREADS=6", "-e", "OPENBLAS_NUM_THREADS=6",
                     "-e", "CUDA_VISIBLE_DEVICES="+str(args.gpu),
                     "-e", "TMPDIR="+suite.container(suite.root/"tmp"),
                     "-e", "TORCHINDUCTOR_CACHE_DIR="+suite.container(suite.root/"cache"),
                     args.container, "python", "-B", "-m", "scripts.experiments.mrdbscp842_acceleration.worker",
                     "--job-token", token, "--mailbox", suite.container(self.mail), "--gpu", str(args.gpu),
                     "--features", ",".join(features), "--oasis-root", args.oasis_root,
                     "--pair-file", args.repo_container+"/configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"]
        dump(self.mail/"launch.json", self.argv)
        source_hashes = {}
        for source in list(Path("scripts/experiments/mrdbscp842_acceleration").glob("*.py")) + [
                Path("scripts/experiments/oasis_alpha_sweep/run_one.py"),Path("scripts/experiments/oasis_alpha_sweep/common.py")]:
            data = source.read_bytes()
            source_hashes[str(source)] = hashlib.sha256(data).hexdigest()
            copy = self.mail/"source"/source
            copy.parent.mkdir(parents=True,exist_ok=True)
            copy.write_bytes(data)
        dump(self.mail/"source_sha256.json",source_hashes)
        self.log = (self.mail/"process.log").open("w")
        self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT)
        self.wait_file(self.mail/"ready.json", timeout=120)
        ready = json.loads((self.mail/"ready.json").read_text())
        matches = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                ident = process_identity(int(proc.name))
                argv = ident["argv"]
                if (token in argv and "scripts.experiments.mrdbscp842_acceleration.worker" in argv
                        and Path(argv[0]).name.startswith("python")):
                    matches.append(ident)
            except (OSError, IndexError, ValueError):
                continue
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely identify worker host PID: {matches}")
        ident = matches[0]
        status = Path(f"/proc/{ident['pid']}/status").read_text()
        ns = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
        if not ns or int(ns[-1]) != ready["container_pid"]:
            raise RuntimeError("Host/container PID mapping mismatch")
        suite.monitor.owner = ident
        dump(self.mail/"host_ack.json", ident)

    def wait_file(self, path, timeout=3600):
        started = time.monotonic()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Worker exited {self.process.returncode}; see {self.mail}")
            if time.monotonic()-started > timeout:
                raise TimeoutError(str(path))
            time.sleep(.2)

    def execute(self, action, label, *, command_timeout=3600, **kwargs):
        output = self.suite.root/"jobs"/label
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        command = dict(action=action, output=self.suite.container(output), **kwargs)
        start = time.time()
        self.suite.monitor.sample()
        dump(self.mail/f"command_{self.index:04d}.json", command)
        result_path = self.mail/f"result_{self.index:04d}.json"
        self.wait_file(result_path, timeout=command_timeout)
        self.suite.monitor.sample()
        end = time.time()
        monitor = self.suite.monitor.interval(start, end)
        result = json.loads(result_path.read_text())
        self.index += 1
        dump(output/"monitor_verdict.json", monitor)
        if not result["ok"]:
            raise RuntimeError(result["traceback"])
        value = result["result"]
        value.update(id=label, action=action, clean=monitor["clean"], monitor=monitor,
                     worker=self.name, command=command)
        dump(output/"accepted_result.json", value)
        self.suite.attempts.append(value)
        dump(self.suite.root/"attempts.json", self.suite.attempts)
        return value

    def close(self):
        if self.process.poll() is None:
            dump(self.mail/f"command_{self.index:04d}.json", {"action": "exit"})
            self.process.wait(timeout=120)
        self.suite.monitor.owner = None
        self.log.close()


class Suite:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.artifact_host)/"runs"/args.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/"tmp").mkdir(exist_ok=True)
        (self.root/"cache").mkdir(exist_ok=True)
        self.review = Path(args.artifact_host)/"repo_experiments"/getattr(args,"experiment_key","2026-09-19_mrdbscp842_acceleration_gpu0")/args.run_id
        self.review.mkdir(parents=True, exist_ok=True)
        self.attempts = json.loads((self.root/"attempts.json").read_text()) if (self.root/"attempts.json").exists() else []
        self.rows = json.loads((self.root/"primary_rows.json").read_text()) if (self.root/"primary_rows.json").exists() else []
        self.monitor = Monitor(args.gpu, self.root/"gpu_monitor.jsonl")

    def container(self, path):
        return str(Path(self.args.artifact_container)/Path(path).relative_to(self.args.artifact_host))

    def provenance(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = self.root/"provenance"/stamp
        dest.mkdir(parents=True)
        paths = [Path("AGENTS.md"), Path("configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"),
                 Path("scripts/experiments/artifact_paths.py"),
                 Path("docs/implementation/mr_d_bscp_fast_842.md"),
                 Path("docs/codex_prompts/MR_D_BSCP_842_Codex_Handoff_ZH.zip")]
        paths += list(Path("scripts/experiments/mrdbscp842_acceleration").glob("*.py"))
        paths += list(Path("scripts/experiments/oasis_alpha_sweep").glob("*.py"))
        paths += list(Path("experiments/2026-09-19_mrdbscp842_acceleration_gpu0").glob("*.md"))
        if getattr(self.args,"experiment_key",None):
            paths += list((Path("experiments")/self.args.experiment_key).glob("*.md"))
        hashes = {}
        for path in paths:
            data = path.read_bytes()
            target = dest/"source"/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
        (dest/"dirty.patch").write_text(cmd(["git", "diff", "--binary", "HEAD"]))
        manifest = dict(args=vars(self.args), command=sys.argv, git_head=cmd(["git", "rev-parse", "HEAD"]),
                        git_status=cmd(["git", "status", "--porcelain=v1"]), source_sha256=hashes,
                        docker=json.loads(cmd(["docker", "inspect", self.args.container])),
                        gpu=cmd(["nvidia-smi", "-i", str(self.args.gpu), "-q"]),
                        host_uid=os.getuid(), host_gid=os.getgid(), utc=stamp)
        dump(dest/"manifest.json", manifest)
        dump(self.root/"latest_provenance.json", {"path": str(dest)})

    def record(self, row, **tags):
        row.update(tags)
        self.rows.append(row)
        dump(self.root/"primary_rows.json", self.rows)

    def block(self, name, variant, jobs, *, warmup=True, features=None):
        """Fresh worker per block. Retry warm in worker; retry cold in fresh worker."""
        features = FEATURES[variant] if features is None else features
        worker = None
        results = []
        try:
            for index, spec in enumerate(jobs):
                prefix = f"{name}_{index:03d}_attempt"
                accepted = [r for r in self.rows if r["block"] == name and r["id"].startswith(prefix)]
                if accepted:
                    if len(accepted) != 1 or accepted[0]["rank"] != spec["rank"]:
                        raise RuntimeError("Resume manifest mismatch")
                    results.append(accepted[0])
                    continue
                attempt = 0
                while (self.root/"jobs"/f"{prefix}{attempt}").exists():
                    attempt += 1
                while True:
                    if worker is None:
                        worker = Worker(self, name, features)
                        if warmup or index > 0:
                            warm = worker.execute("register", name+"_warmup_"+uuid.uuid4().hex[:6], rank=spec["rank"])
                            if not warm["clean"]:
                                worker.close()
                                worker = None
                                continue
                    label = f"{name}_{index:03d}_attempt{attempt}"
                    row = worker.execute("register", label, **spec)
                    if row["clean"]:
                        self.record(row, block=name, variant=variant, primary=True,
                                    job_index=index,
                                    condition="cold" if not warmup and index == 0 else "warm")
                        results.append(row)
                        print(json.dumps({"completed":label,"seconds":row["elapsed_sec"],"steps":row["steps"]}), flush=True)
                        break
                    attempt += 1
                    if not warmup and index == 0:
                        worker.close()
                        worker = None
                    else:
                        self.monitor.wait_clear()
        finally:
            if worker:
                worker.close()
        return results

    def auxiliary(self, action, name, **kwargs):
        attempt = 0
        while (self.root/"jobs"/f"{name}_attempt{attempt}").exists():
            attempt += 1
        while True:
            worker = Worker(self, name)
            try:
                row = worker.execute(action, f"{name}_attempt{attempt}", **kwargs)
            finally:
                worker.close()
            if row["clean"]:
                return row
            attempt += 1

    def preflight(self):
        checks = self.auxiliary("checks", "gpu_checks")
        dump(self.root/"gpu_checks.json", checks)
        variants = [v for v in FEATURES if checks["passed"].get(v) or v == "cpbe"]
        for variant in variants:
            self.block("smoke_"+variant, variant, [dict(rank=0, smoke=True)], warmup=False)
        micro = self.auxiliary("micro", "microbenchmark")
        dump(self.root/"microbenchmark.json", micro)

    def baseline(self):
        contracts = subprocess.run(["docker","exec","-u","1009:1009","-w",self.args.repo_container,
                    "-e","PYTHONDONTWRITEBYTECODE=1","-e","OMP_NUM_THREADS=6",self.args.container,
                    "python","-B","-m","unittest","scripts.experiments.mrdbscp842_acceleration.test_contracts","-v"],
                    capture_output=True,text=True,timeout=180)
        dump(self.root/"contract_tests.json",dict(command=contracts.args,returncode=contracts.returncode,
                                               stdout=contracts.stdout,stderr=contracts.stderr))
        contracts.check_returncode()
        # Retain initial generic-grid preflight; formal MR checks use the exact
        # nested CP_MARGIN=4 dimensions and explicitly record those dimensions.
        micro = json.loads((self.root/"microbenchmark.json").read_text())
        if not all("control_shape" in r for r in micro["rows"]):
            for stem in ("gpu_checks", "microbenchmark"):
                dump(self.root/(stem+"_initial_generic_grid.json"),json.loads((self.root/(stem+".json")).read_text()))
            checks = self.auxiliary("checks","gpu_checks_nested",only_real=True)
            dump(self.root/"gpu_checks.json",checks)
            micro = self.auxiliary("micro","microbenchmark_nested")
            dump(self.root/"microbenchmark.json",micro)
        if not any(r["block"] == "smoke_cpbe" for r in self.rows):
            self.block("smoke_cpbe", "cpbe", [dict(rank=0, smoke=True)], warmup=False)
        self.block("reference_AA", "reference", [dict(rank=0) for _ in range(6)], warmup=False)
        baseline = self.block("reference_first10", "reference", [dict(rank=r) for r in [0]+list(range(10))], warmup=False)
        dump(self.root/"baseline.json", baseline)
        steps = baseline[1]["steps"]
        snapshots = {str(i): sorted(set((0, 1, n//2, n-25, n))) for i,n in enumerate(steps)}
        self.block("reference_diagnostic", "reference", [dict(rank=0, detail="deep", snapshots=snapshots)], warmup=False)
        self.block("reference_profile", "reference", [dict(rank=0, smoke=True, profile=True)], warmup=False)

    def pilot(self):
        checks = json.loads((self.root/"gpu_checks.json").read_text())
        baseline = json.loads((self.root/"baseline.json").read_text())
        steps = {r["rank"]:r["steps"] for r in baseline}
        results = {}
        for variant in ("cache", "batch", "ncc", "buffer", "cpbe"):
            if not checks["passed"].get(variant) and variant != "cpbe":
                results[variant] = {"excluded": "numerical gate"}
                continue
            current = []
            for mode in ("fixed", "adaptive"):
                for repeat in range(3):
                    order = ["reference", variant] if repeat % 2 == 0 else [variant, "reference"]
                    for arm in order:
                        jobs = [dict(rank=r, **({"steps":steps[r]} if mode == "fixed" else {})) for r in (9,0,7)]
                        rows = self.block(f"pilot_{variant}_{mode}_rep{repeat}_{arm}", arm, jobs)
                        for row in rows:
                            row.update(comparison=variant, mode=mode, repeat=repeat)
                        current.extend(rows)
                        dump(self.root/"primary_rows.json", self.rows)
            results[variant] = {"rows":current}
            dump(self.root/"pilots.json", results)
        dump(self.root/"pilots.json", results)

    def evaluate(self):
        # Reproduce pairing without importing NumPy on the host.
        index = {(r.get("comparison"),r.get("mode"),r.get("repeat"),r["rank"],r["block"].split("_",1)[0]):r
                 for r in self.rows if r["variant"] == "reference" and r.get("comparison")}
        aa = [r for r in self.rows if r["block"] == "reference_AA"]
        items = []
        for row in self.rows:
            if row["block"].startswith(("smoke_","reference_diagnostic","reference_profile")):
                continue
            item = dict(id=row["id"],rank=row["rank"],dvf=row["dvf"])
            if row["block"] == "reference_AA" and row != aa[0]:
                item["reference_dvf"] = aa[0]["dvf"]
            elif row.get("comparison") and row["variant"] != "reference":
                key = (row["comparison"],row["mode"],row["repeat"],row["rank"],row["block"].split("_",1)[0])
                item["reference_dvf"] = index[key]["dvf"]
            # HD95 for BE characterization and matched references; Dice/Jac for all.
            item["hd95"] = "cpbe" in row.get("comparison", "")
            items.append(item)
        existing = {}
        for path in self.root.glob("evaluation_*.json"):
            for row in json.loads(path.read_text()).get("rows",[]):
                existing[row["id"]] = row
        items = [item for item in items if item["id"] not in existing]
        if items:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            result = self.auxiliary("evaluate", "evaluation_"+stamp, items=items)
            dump(self.root/("evaluation_"+stamp+".json"),result)
        self.report()

    def report(self):
        if (self.root/"characterize_DONE.json").exists() and not (self.review/"field_region_errors.json").exists():
            regions = ["docker","exec","-u","1009:1009","-w",self.args.repo_container,
                       "-e","PYTHONDONTWRITEBYTECODE=1",self.args.container,"python","-B","-m",
                       "scripts.experiments.mrdbscp842_acceleration.field_regions","--root",self.container(self.root),
                       "--output",self.container(self.review/"field_region_errors.json")]
            with (self.root/"field_regions.log").open("w") as log:
                subprocess.run(regions,check=True,stdout=log,stderr=subprocess.STDOUT)
        argv = ["docker","exec","-u","1009:1009","-w",self.args.repo_container,
                "-e","PYTHONDONTWRITEBYTECODE=1",self.args.container,"python","-B","-m",
                "scripts.experiments.mrdbscp842_acceleration.report","--root",self.container(self.root),
                "--review",self.container(self.review)]
        subprocess.run(argv,check=True)

    def characterize(self):
        if not (self.root/"data_environment.json").exists():
            data = self.auxiliary("environment","data_environment")
            dump(self.root/"data_environment.json",data)
        baseline = json.loads((self.root/"baseline.json").read_text())
        steps = {r["rank"]:r["steps"] for r in baseline}
        for mode in ("fixed","adaptive"):
            order = ["reference","cpbe"] if mode == "fixed" else ["cpbe","reference"]
            for arm in order:
                jobs = [dict(rank=r, **({"steps":steps[r]} if mode == "fixed" else {})) for r in range(10)]
                rows = self.block(f"characterize_cpbe_{mode}_rep0_{arm}",arm,jobs)
                for row in rows:
                    row.update(comparison="cpbe",mode=mode,repeat=0)
                dump(self.root/"primary_rows.json",self.rows)

    def snapchecks(self):
        paths = sorted((self.root/"jobs").glob("reference_diagnostic_*/controls/*.npz"))
        if len(paths) != 15:
            raise RuntimeError(f"Expected 15 real control snapshots, found {len(paths)}")
        feature_sets = {k:v for k,v in FEATURES.items() if k != "reference"}
        if (self.root/"selection.json").exists():
            selection = json.loads((self.root/"selection.json").read_text())
            if selection["variant"]:
                feature_sets[selection["variant"]] = selection["features"]
        result = self.auxiliary("snapchecks","real_control_checks",paths=[self.container(p) for p in paths],
                                feature_sets=feature_sets)
        dump(self.root/"real_control_checks.json",result)

    def bridge(self):
        jobs = [dict(rank=0,detail="endpoints")]
        for repeat in range(3):
            order = ("deep","endpoints") if repeat % 2 == 0 else ("endpoints","deep")
            jobs.extend(dict(rank=0,detail=detail) for detail in order)
        self.block("reference_bridge","reference",jobs,warmup=False)

    def confirm(self):
        selection = json.loads((self.root/"selection.json").read_text())
        variant, features = selection["variant"], selection["features"]
        if not variant or not features:
            raise RuntimeError("No eligible candidate; retain the negative pilot result")
        checks = json.loads((self.root/"gpu_checks.json").read_text())["passed"]
        states = [r for r in json.loads((self.root/"real_control_checks.json").read_text())["rows"]
                  if r["variant"] == variant and r["features"] == features]
        if (not all(checks.get(f,False) for f in features) or len(states) != 15
                or not all(r["gradient_error"]["relative_l2"] <= 1e-4 for r in states)):
            raise RuntimeError("Selected candidate has not passed operator and 15-state gradient checks")
        baseline = json.loads((self.root/"baseline.json").read_text())
        steps = {r["rank"]:r["steps"] for r in baseline}
        self.block("smoke_"+variant,variant,[dict(rank=0,smoke=True)],warmup=False,features=features)
        for mode in ("fixed","adaptive"):
            for repeat in range(5):
                order = ["reference",variant] if repeat % 2 == 0 else [variant,"reference"]
                for arm in order:
                    jobs = [dict(rank=r, **({"steps":steps[r]} if mode == "fixed" else {})) for r in range(10)]
                    rows = self.block(f"confirm_{variant}_{mode}_rep{repeat}_{arm}",arm,jobs,
                                      features=() if arm == "reference" else features)
                    for row in rows:
                        row.update(comparison=variant,mode=mode,repeat=repeat)
                    dump(self.root/"primary_rows.json",self.rows)
        for repeat in range(3):
            for arm in ("reference",variant):
                self.block(f"cold_{variant}_rep{repeat}_{arm}",arm,[dict(rank=0)],warmup=False,
                           features=() if arm == "reference" else features)

    def select(self):
        report = self.review/"decisions.json"
        decisions = json.loads(report.read_text())
        if sum(r["block"].startswith("pilot_") for r in self.rows) != 180:
            raise RuntimeError("Selection requires all 180 pilot measurements")
        if not all(decisions["decisions"].get(name,{}).get("pilot",{}).get("performance",{}).get(mode,{}).get("complete",False)
                   for name in ("cache","batch","ncc","buffer","cpbe") for mode in ("fixed","adaptive")):
            raise RuntimeError("Selection requires complete pilot reporting in both modes")
        features = [name for name in ("cache","batch","ncc","buffer")
                    if decisions["decisions"].get(name,{}).get("pilot",{}).get("strict_pass",False)]
        selection = dict(features=features,
                         variant=(features[0] if len(features)==1 else "combo_"+"_".join(features)),
                         rule="Combine only isolated variants passing all preregistered numerical, quality, memory and speed gates in both modes.",
                         exploratory_BE_excluded_from_strict_combination=True,
                         decisions_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                         isolated_decisions=decisions["decisions"])
        if not features:
            selection["variant"] = None
            selection["reason"] = "No isolated candidate passed the strict confirmation-entry rule"
        dump(self.root/"selection.json",selection)
        print(json.dumps({"selection":selection["variant"],"features":features}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--container", required=True)
    p.add_argument("--repo-container", required=True)
    p.add_argument("--artifact-host", required=True)
    p.add_argument("--artifact-container", required=True)
    p.add_argument("--oasis-root", required=True)
    p.add_argument("--gpu", type=int, default=0, choices=[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--phase", choices=["preflight", "baseline", "pilot", "characterize", "confirm", "evaluate", "report", "snapchecks", "bridge", "select"], default="preflight")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if "/" in args.run_id or args.run_id in ("", ".", ".."):
        raise ValueError("run-id must be a simple new directory name")
    artifact = Path(args.artifact_host).resolve()
    repo = Path.cwd().resolve()
    if (not Path(args.artifact_host).is_absolute() or not artifact.is_dir()
            or ".." in Path(args.artifact_host).parts or artifact == repo or repo in artifact.parents
            or artifact in repo.parents or artifact in (Path("/"),Path.home())):
        raise ValueError("Artifact root must be a dedicated external directory")
    suite = Suite(args)
    suite.provenance()
    if args.dry_run:
        dump(suite.root/"dry_run.json", dict(phases=["preflight","baseline","pilot","confirmation","evaluation","report"],
                                          features=FEATURES, args=vars(args)))
        print(suite.root, flush=True)
        return
    suite.monitor.thread.start()
    try:
        getattr(suite, args.phase)()
        dump(suite.root/(args.phase+"_DONE.json"), {"completed_utc":datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        dump(suite.root/(args.phase+"_ERROR.json"), {"error":repr(exc)})
        raise
    finally:
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=35)


if __name__ == "__main__":
    main()
