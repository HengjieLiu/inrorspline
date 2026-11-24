"""CPU-only four-GPU orchestration; all image/metric Python executes in Docker."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from .protocol import ARMS, COARSE, PROTOCOL, bracket, dump, read, refined, sequence, setting, sha256, stamp
from scripts.experiments.mrdbscp842_acceleration.nvml_monitor import NVML

REPO=Path(__file__).resolve().parents[3]


def command(argv): return subprocess.check_output(list(map(str,argv)),text=True).strip()


def gpu_inventory():
    lines=command(["nvidia-smi","--query-gpu=index,uuid,name,memory.total","--format=csv,noheader,nounits"]).splitlines()
    return [dict(zip(("gpu","uuid","name","memory_mib"),[s.strip() for s in line.split(",")])) for line in lines]


def identity(pid):
    root=Path("/proc")/str(pid)
    text=(root/"stat").read_text(); fields=text[text.rfind(")")+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),argv=(root/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace"))


def find_worker(job_path,container_pid):
    matches=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            item=identity(int(proc.name))
            if "scripts.experiments.oasis_regularizer_sweep.worker" not in item["argv"] or job_path not in item["argv"]: continue
            if not Path(item["argv"].split()[0]).name.startswith("python"): continue
            ns=next(line.split()[1:] for line in (proc/"status").read_text().splitlines() if line.startswith("NSpid:"))
            if int(ns[-1])==container_pid: matches.append(item)
        except (OSError,ValueError,IndexError,StopIteration): continue
    if len(matches)!=1: raise RuntimeError(f"Cannot identify unique host worker: {matches}")
    return matches[0]


class DeviceMemory(ct.Structure):
    _fields_=[("total",ct.c_ulonglong),("free",ct.c_ulonglong),("used",ct.c_ulonglong)]


def device_memory(nvml):
    info=DeviceMemory(); nvml.check(nvml.lib.nvmlDeviceGetMemoryInfo(nvml.handle,ct.byref(info)))
    return dict(total=info.total,free=info.free,used=info.used)


class Monitor:
    def __init__(self,device,path):
        self.nvml=NVML(int(device["gpu"]),device["uuid"]); self.path=path
        self.owner=None; self.stop_event=threading.Event(); self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True)

    def loop(self):
        try:
            with self.path.open("w") as stream:
                while not self.stop_event.is_set():
                    row=dict(monotonic_ns=time.monotonic_ns(),device=device_memory(self.nvml),processes=self.nvml.processes())
                    if self.owner:
                        try:
                            current=identity(self.owner["pid"])
                            if current["start_ticks"]!=self.owner["start_ticks"]: raise RuntimeError("Worker PID reused")
                            status=Path(f"/proc/{self.owner['pid']}/status").read_text()
                            rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith("VmRSS:")),None)
                            if rss is not None: row["rss_bytes"]=rss
                        except (FileNotFoundError,ProcessLookupError): pass
                    stream.write(json.dumps(row)+"\n"); stream.flush()
                    self.stop_event.wait(.2)
        except BaseException as error: self.error=repr(error)

    def close(self):
        self.stop_event.set(); self.thread.join(10)
        if self.thread.is_alive() or self.error: raise RuntimeError(f"Telemetry failed: {self.error}")


class Study:
    def __init__(self,args):
        self.args=args; self.root=args.artifact_host/"runs"/args.run_id
        self.review=args.artifact_host/"repo_experiments"/"oasis_first10_regularizer_fd_analytic_v1"/args.run_id
        self.devices=gpu_inventory(); self.lock=threading.RLock(); self.active={}
        self.root.mkdir(parents=True,exist_ok=True); self.review.mkdir(parents=True,exist_ok=True)
        self.filelock=(self.root/"controller.lock").open("a")
        fcntl.flock(self.filelock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.source_files=list((REPO/"scripts/experiments/oasis_regularizer_sweep").glob("*.py"))+[
            REPO/"scripts/experiments/oasis_alpha_sweep/common.py",REPO/"scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py",
            REPO/"scripts/experiments/mrdbscp842_acceleration/nvml_monitor.py",
            REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt",REPO/"tests/test_oasis_regularizer_sweep.py"]
        self.hashes={str(p.relative_to(REPO)):sha256(p) for p in self.source_files}
        manifest=self.root/"manifest.json"
        if manifest.exists():
            old=read(manifest)
            if old["source_hashes"]!=self.hashes or old["protocol"]!=PROTOCOL: raise RuntimeError("Frozen source/protocol changed; use a new execution")
        else:
            inspect=json.loads(command(["docker","inspect",args.container]))[0]
            dump(manifest,dict(created_at=stamp(),protocol=PROTOCOL,source_hashes=self.hashes,
                git_head=command(["git","rev-parse","HEAD"]),git_branch=command(["git","branch","--show-current"]),
                git_status=command(["git","status","--short"]),devices=self.devices,
                container=dict(image=inspect["Config"]["Image"],image_id=inspect["Image"],user=inspect["Config"]["User"],mounts=inspect["Mounts"]),
                args=vars(args),command=sys.argv))
            for path in self.source_files:
                target=self.root/"source"/path.relative_to(REPO); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
            (self.root/"source"/"dirty.patch").write_text(command(["git","diff","HEAD"]))
        dump(self.root/"controller.json",dict(identity=identity(os.getpid()),started_at=stamp(),command=sys.argv))
        self.update_status("initializing")

    def container_path(self,path): return str(self.args.artifact_container/Path(path).relative_to(self.args.artifact_host))

    def frozen(self):
        for name,expected in self.hashes.items():
            if sha256(REPO/name)!=expected: raise RuntimeError(f"Frozen source changed: {name}")

    def update_status(self,phase):
        with self.lock:
            blocks=list((self.root/"blocks").glob("*/*/attempt*/block.json"))
            complete=sum(bool(read(p).get("complete")) for p in blocks)
            evaluated=len(list((self.root/"blocks").glob("*/*/attempt*/EVALUATED.json")))
            status=dict(updated_at=stamp(),phase=phase,active=self.active,completed_blocks=complete,evaluated_blocks=evaluated,
                        root=str(self.root),report=str(self.review/"report.md"))
            dump(self.root/"status.json",status)
            (self.review/"progress.md").write_text(f"# Six-arm OASIS experiment\n\nUpdated: {status['updated_at']}\n\nPhase: {phase}\n\nCompleted registration blocks: {complete}; evaluated blocks: {evaluated}.\n\nActive GPU jobs:\n\n"+"\n".join(f"- GPU {g}: {v}" for g,v in self.active.items())+"\n")

    def admission(self,device):
        nvml=NVML(int(device["gpu"]),device["uuid"])
        reserve=8*1024**3
        if (self.root/"PREFLIGHT_ACCEPTED.json").exists(): reserve=max(reserve,read(self.root/"PREFLIGHT_ACCEPTED.json")["peak_reserved"]+2*1024**3)
        while nvml.processes() or device_memory(nvml)["free"]<reserve:
            self.update_status("waiting_for_gpu_resources"); time.sleep(5)
        if shutil.disk_usage(self.root).free<200*1024**3: raise RuntimeError("Less than200GiB artifact capacity")

    def run_process(self,job,device,evaluate=False):
        self.frozen(); self.admission(device)
        root=Path(job["host_block_dir"]); root.mkdir(parents=True,exist_ok=True)
        marker="eval" if evaluate else "register"; gpu=device["gpu"]
        if (root/f"{marker}_boot.json").exists():
            raise RuntimeError("Prior incomplete worker handshake exists; preserve it and use a new attempt/execution")
        job=dict(job,gpu=gpu,gpu_uuid=device["uuid"],source_hashes=self.hashes,
                 parent_launch_ns=time.monotonic_ns(),oasis_root=self.args.oasis_root)
        path=root/f"{marker}_job.json"; dump(path,job)
        argv=["docker","exec","-w","/workspace/repo","-e",f"CUDA_VISIBLE_DEVICES={gpu}",
              "-e","OMP_NUM_THREADS=1","-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1",
              self.args.container,"python","-B","-m","scripts.experiments.oasis_regularizer_sweep.worker",
              "--job",self.container_path(path)]+(["--evaluate"] if evaluate else [])
        dump(root/f"{marker}_command.json",argv)
        monitor=Monitor(device,root/f"{marker}_monitor.jsonl")
        with self.lock: self.active[gpu]=f"{marker}: {job['phase']}/{job['arm']}/{job['weight']:g}"
        self.update_status(job["phase"])
        with (root/f"{marker}.log").open("w") as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            monitor.thread.start()
            try:
                deadline=time.monotonic()+180
                boot=root/f"{marker}_boot.json"
                while not boot.exists():
                    if process.poll() is not None: raise RuntimeError("Worker exited before boot handshake")
                    if time.monotonic()>deadline: raise TimeoutError("Worker boot timed out")
                    time.sleep(.1)
                owner=find_worker(self.container_path(path),read(boot)["container_pid"])
                monitor.owner=owner; dump(root/f"{marker}_host_identity.json",owner)
                dump(root/f"{marker}_ack.json",owner)
                # Large HD95/NDV passes may be slow. Polling preserves monitoring.
                deadline=time.monotonic()+24*3600
                while process.poll() is None:
                    if monitor.error: raise RuntimeError(monitor.error)
                    if time.monotonic()>deadline: raise TimeoutError("Block exceeded24h watchdog")
                    time.sleep(.5)
                if process.returncode: raise RuntimeError(f"Worker exit{process.returncode}; inspect {root/ (marker+'.log')}")
            finally:
                if process.poll() is None:
                    # docker exec termination alone does not reliably stop its child.
                    # Leave an active identity recorded instead of orphaning/retrying it.
                    dump(root/"ACTIVE_WORKER_REQUIRES_RECONCILIATION.json",dict(exec_pid=process.pid,owner=monitor.owner))
                try: monitor.close()
                finally:
                    with self.lock: self.active.pop(gpu,None)
                    self.update_status(job["phase"])
        self.summarize_monitor(root,marker,monitor.owner)
        if not evaluate:
            block=read(root/"block.json")
            if not block["complete"] or [(r["case_rank"],r["condition"]) for r in block["cases"]]!=sequence(job["phase"]):
                raise RuntimeError("Incomplete/out-of-order cold/warm sequence")
        elif not (root/"EVALUATED.json").exists(): raise RuntimeError("Missing evaluation completion")
        dump(root/f"{marker}_ACCEPTED.json",dict(complete=True,at=stamp(),host_identity=monitor.owner))

    def summarize_monitor(self,root,marker,owner):
        peaks=[]; contaminated=[]; rss=[]
        rows=[]
        with (root/f"{marker}_monitor.jsonl").open() as stream:
            for line in stream:
                row=json.loads(line); rows.append(row)
                foreign=[p for p in row["processes"] if p["pid"]!=owner["pid"]]
                if foreign: contaminated.append(dict(at=row["monotonic_ns"],processes=foreign))
                peaks += [p["memory_mib"] for p in row["processes"] if p["pid"]==owner["pid"]]
                rss.append(row.get("rss_bytes",0))
        dump(root/f"{marker}_telemetry.json",dict(samples=len(rows),sample_seconds=.2,process_gpu_peak_mib=max(peaks,default=None),
            host_rss_peak=max(rss,default=None),contaminated=contaminated))
        if contaminated: raise RuntimeError("Competing GPU process contaminated block")
        if marker=="register":
            block=read(root/"block.json")
            for case in block["cases"]:
                samples=[r for r in rows if case["case_load_complete_ns"]<=r["monotonic_ns"]<=case["dvf_file_ready_ns"]]
                memory_values=[p["memory_mib"] for r in samples for p in r["processes"] if p["pid"]==owner["pid"]]
                case["nvml_peak_mib"]=max(memory_values,default=None)
                case["host_rss_peak"]=max((r.get("rss_bytes",0) for r in samples),default=None)
                case["telemetry_samples"]=len(samples)
                if not memory_values: raise RuntimeError("No attributed NVML sample in registration window")
            dump(root/"block.json",block)

    def job(self,phase,arm,weight):
        root=self.root/"blocks"/phase/setting(arm,weight)
        attempts=sorted(root.glob("attempt*"))
        for attempt in attempts:
            if (attempt/"ACTIVE_WORKER_REQUIRES_RECONCILIATION.json").exists(): raise RuntimeError("Active orphan requires identity audit")
            if (attempt/"register_ACCEPTED.json").exists():
                return dict(phase=phase,arm=arm,weight=weight,host_block_dir=str(attempt),block_dir=self.container_path(attempt))
        attempt=root/f"attempt{len(attempts)}"
        return dict(phase=phase,arm=arm,weight=weight,host_block_dir=str(attempt),block_dir=self.container_path(attempt))

    def wave(self,jobs,evaluate=False):
        # A deterministic per-GPU queue ensures one child per GPU and same-GPU replay.
        queues={v["gpu"]:[] for v in self.devices}
        for idx,job in enumerate(jobs):
            block=Path(job["host_block_dir"])
            if evaluate:
                gpu=read(block/"block.json")["job"]["gpu"]
                if (block/"eval_ACCEPTED.json").exists(): continue
            else:
                if (block/"register_ACCEPTED.json").exists(): continue
                gpu=self.devices[idx%len(self.devices)]["gpu"]
            queues[gpu].append(job)
        failures=[]
        def loop(device):
            for job in queues[device["gpu"]]:
                try: self.run_process(job,device,evaluate)
                except BaseException as error:
                    record=dict(job=job,error=repr(error),traceback=traceback.format_exc(),at=stamp())
                    dump(Path(job["host_block_dir"])/("eval" if evaluate else "register")/"controller_FAILURE.json",record)
                    with self.lock: failures.append(record)
                    # Active workers must retain the slot. Do not launch another job here.
                    if (Path(job["host_block_dir"])/"ACTIVE_WORKER_REQUIRES_RECONCILIATION.json").exists(): break
        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            list(pool.map(loop,self.devices))
        if failures:
            dump(self.root/f"failures_{time.time_ns()}.json",failures)
            raise RuntimeError(f"{len(failures)} blocks failed; retained failure evidence")

    def report(self):
        argv=["docker","exec","-w","/workspace/repo",self.args.container,"python","-B","-m",
              "scripts.experiments.oasis_regularizer_sweep.report","--root",self.container_path(self.root),
              "--out",self.container_path(self.review)]
        subprocess.run(argv,check=True)

    def preflight(self):
        if (self.root/"PREFLIGHT_ACCEPTED.json").exists(): return
        result=subprocess.run(["docker","exec","-w","/workspace/repo",self.args.container,"python","-B","-m",
            "unittest","discover","-s","tests","-p","test_oasis_regularizer_sweep.py","-v"],capture_output=True,text=True)
        (self.root/"unit_tests.log").write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError("Unit preflight failed")
        jobs=[self.job("smoke",arm,1.) for arm in ARMS]
        self.wave(jobs); self.wave(jobs,evaluate=True)
        peak=max(row["peak_reserved"] for job in jobs for row in read(Path(job["host_block_dir"])/"block.json")["cases"])
        sizes=[sum(p.stat().st_size for p in Path(job["host_block_dir"]).rglob("*") if p.is_file()) for job in jobs]
        projected=max(sizes)*715*1.3
        if shutil.disk_usage(self.root).free<projected: raise RuntimeError("Insufficient capacity for measured artifact projection plus30%")
        dump(self.root/"PREFLIGHT_ACCEPTED.json",dict(at=stamp(),smokes=6,peak_reserved=peak,
            projected_retained_bytes=projected,source_hashes=self.hashes))
        self.report()

    def run(self):
        if self.args.phase=="report": self.report(); return
        self.preflight()
        if self.args.phase=="preflight": self.update_status("preflight_complete"); return
        jobs=[self.job("pilot",arm,weight) for weight in COARSE for arm in ARMS]+[self.job("pilot","none",0.)]
        self.wave(jobs); self.wave(jobs,evaluate=True); self.report()
        decisions={}
        for round_index in range(7):
            extensions=[]
            for arm in ARMS:
                rows=[]
                for path in (self.root/"blocks").glob(f"pilot*/{arm}_w*/attempt*/block.json"):
                    if not (path.parent/"eval_ACCEPTED.json").exists(): continue
                    block=read(path)
                    warm=next(r for r in block["cases"] if r["condition"]=="warm")
                    metric=read(Path(warm["dvf_path"].replace(str(self.args.artifact_container),str(self.args.artifact_host),1)).parent/"evaluation.json")
                    rows.append(dict(weight=block["job"]["weight"],**{k:metric[k] for k in ("dice","hd95","asd")}))
                decisions[arm]=bracket(rows)
                for weight in decisions[arm]["extensions"]:
                    extensions.append(self.job(f"pilot_extension{round_index}",arm,float(f"{weight:.12g}")))
            dump(self.root/"selection.json",decisions)
            if not extensions: break
            self.wave(extensions); self.wave(extensions,evaluate=True); self.report()
        final=[]
        for arm,decision in decisions.items():
            if decision["status"]=="bracketed":
                decision["first10_grid"]=refined(decision["best"]["weight"])
                final += [self.job("cohort",arm,w) for w in decision["first10_grid"]]
        dump(self.root/"selection.json",decisions)
        final.append(self.job("cohort","none",0.))
        self.wave(final); self.wave(final,evaluate=True); self.report()
        complete=all(v["status"]=="bracketed" for v in decisions.values())
        dump(self.root/("ACCEPTED.json" if complete else "PARTIAL.json"),dict(complete=complete,at=stamp(),
             cohort_settings=len(final),expected_registration_rows=len(final)*11,decisions=decisions))
        self.update_status("complete" if complete else "partial_unbracketed_arms")
        self.report()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container",default="py310sitreg_v1_artifact_store")
    p.add_argument("--artifact-host",type=Path,required=True); p.add_argument("--artifact-container",type=Path,default=Path("/database/inrorspline"))
    p.add_argument("--oasis-root",required=True); p.add_argument("--run-id",required=True)
    p.add_argument("--phase",choices=("preflight","all","report"),default="all")
    args=p.parse_args()
    if "/" in args.run_id or args.run_id in (".",".."): p.error("Run ID must be a single safe path component")
    study=Study(args)
    try: study.run()
    except BaseException as error:
        dump(study.root/"CONTROLLER_FAILED.json",dict(at=stamp(),error=repr(error),traceback=traceback.format_exc()))
        study.update_status("failed"); raise


if __name__=="__main__": main()
