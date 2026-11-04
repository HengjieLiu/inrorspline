#!/usr/bin/env python
"""Durable eight-slot P80 queue: case-local failure isolation, smoke-gated launch."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict

import bspline8421_matched_prefix as e
import queue_oasis_svf_continue_on_error as base

WORKER=Path(__file__).with_name("run_oasis_bspline8421_matched_prefix.py")
REPORT=e.c.REPO_ROOT/"scripts/reports/report_oasis_bspline8421_matched_prefix.py"
TERMINAL={"complete","failed_deferred"}
POLICY={"max_concurrent":8,"workers_per_gpu":2,"oom_retries":1,"min_free_gib":8,
        "min_host_available_gib":32,"min_artifact_free_gib":400,"affinity":"original_prefix_gpu",
        "idle_warning_sec":1800,"idle_terminate_sec":3600,"terminate_grace_sec":60,
        "phase_timeout_min_sec":21600,"phase_timeout_smoke_multiplier":10,
        "failures":"defer cell and continue; shared infrastructure pauses admissions"}


def choose_job(jobs,gpu,active):
    ordinary=any(j["status"] in ("pending","running","launching","audit_pending","auditing") for j in jobs)
    for status in ("audit_pending","pending","oom_retry"):
        for j in jobs:
            if j["status"]!=status or j["gpu"]!=gpu: continue
            if status=="oom_retry" and (ordinary or any(x["job"]["gpu"]==gpu for x in active.values())): continue
            return j


def activity(record,job):
    values=base.activity(record,job)
    for p in Path(job["out"]).glob("level*/map*/history*"):
        try:
            st=p.stat(); values.append((str(p),st.st_mtime_ns,st.st_size))
        except OSError: pass
    return values


def capacity(gpu,active,free,required):
    same=[x for x in active.values() if x["job"]["gpu"]==gpu]
    return len(active)<8 and len(same)<2 and free-sum(x["required_mib"] for x in same)>=required


class Queue:
    def __init__(self,root,m,smoke=False,admission=None):
        self.root,self.m,self.smoke=Path(root),m,smoke
        self.folder=self.root/("smoke_queue" if smoke else "production_queue")
        self.folder.mkdir(parents=True,exist_ok=True)
        self.manifest=self.root/"launch_manifest.json"
        self.admission=admission or {f:{"required_mib":8192,"worker_timeout":21600,"audit_timeout":21600} for f in e.FAMILIES}
        self.jobs=[]; self.active={}; self.started=time.time(); self.last_source_check=0.; self.last_print=0.
        self.reconcile()

    def save(self):
        with base.deadline():
            e.atomic_json(self.folder/"state.json",{"jobs":self.jobs,"policy":POLICY,"updated_at":e.c.now_text()})

    def fail(self,j,action,reason,tail="",allow_oom=False):
        oom=allow_oom and "cuda" in tail.lower() and "out of memory" in tail.lower()
        retry=oom and j.get("oom_retries",0)<1
        event={"at":e.c.now_text(),"spec":j["spec"],"action":action,"reason":reason,"tail":tail,
               "gpu":j["gpu"],"out":j.get("out"),"log":j.get("log"),"oom":oom,
               "progress":base.safe_progress(Path(j["out"])/"progress.json") if j.get("out") else None}
        e.atomic_json(self.folder/"failures"/f"{time.time_ns()}.json",event)
        j.update(status="oom_retry" if retry else "failed_deferred",failure=event,retry_action=action,
                 oom_retries=j.get("oom_retries",0)+int(retry))
        self.save(); print(json.dumps({"failure":j["name"],"status":j["status"],"reason":reason}),flush=True)

    def reconcile(self):
        old=e.c.read_json(self.folder/"state.json")["jobs"] if (self.folder/"state.json").exists() else []
        byname={j["name"]:j for j in old}
        # Also inspect live processes to cover a crash between Popen and durable PID recording.
        live={}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit(): continue
            try:
                rec=base.process_info(int(path.name))
                if not rec or str(WORKER) not in rec["argv"]: continue
                out=base.option(rec["argv"],"--out-dir")
                if out and Path(out).is_relative_to(self.root):
                    if out in live: raise RuntimeError("Duplicate live worker/audit for one output")
                    live[out]=rec
            except (OSError,ProcessLookupError,PermissionError): continue
        for spec in e.matrix(self.smoke):
            j={"name":spec.name(),"spec":asdict(spec),"status":"pending","attempt":0,"oom_retries":0,
               "gpu":self.m["prefixes"][spec.key()]["gpu"],**byname.get(spec.name(),{})}
            j.setdefault("out",str(self.root/("smokes" if self.smoke else "jobs")/spec.name()/"attempt0"))
            self.jobs.append(j); out=Path(j["out"])
            rec=live.pop(str(out),None)
            if rec:
                action="audit" if "audit" in rec["argv"] else "worker"
                j.update(status="auditing" if action=="audit" else "running",record=rec)
                self.active[rec["pid"]]={"job":j,"record":rec,"process":None,"action":action,
                    "started":j.get("launched_at",time.time()),"last_activity":time.time(),"signature":None,
                    "required_mib":self.admission[spec.family]["required_mib"]}
            elif (out/"ACCEPTED.json").is_file():
                try:
                    e.validate(out,self.m,spec); j["status"]="complete"
                except Exception: self.fail(j,"resume_validation",traceback.format_exc())
            elif j["status"] in TERMINAL|{"oom_retry"}: pass
            elif (out/"DONE").is_file(): j["status"]="audit_pending"
            elif out.exists() or j["status"] in ("launching","running","auditing"):
                self.fail(j,"interrupted_attempt","No surviving process or completion; preserved for review")
        if live: raise RuntimeError("Unrecognized live experiment process")
        if len(self.active)>8 or any(n>2 for n in Counter(x["job"]["gpu"] for x in self.active.values()).values()):
            raise RuntimeError("Existing children exceed concurrency")
        self.save()

    def launch(self,j,snapshot):
        retry=j["status"]=="oom_retry"
        action="audit" if j["status"]=="audit_pending" or (retry and j["retry_action"]=="audit") else "worker"
        if retry and action=="worker":
            j["attempt"]+=1; j["out"]=str(Path(j["out"]).parent/f"attempt{j['attempt']}")
        spec=e.Spec(**j["spec"]); invocation=str(time.time_ns())
        log=self.folder/"logs"/f"{j['name']}_{action}_{invocation}.log"; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u","-B",str(WORKER),action,"--manifest",str(self.manifest),"--out-dir",j["out"],
             "--gpu",j["gpu"],"--case-rank",str(spec.case_rank),"--family",spec.family,"--alpha",str(spec.alpha)]
        if spec.smoke: cmd.append("--smoke")
        if action=="audit": cmd.extend(["--audit-dir",str(Path(j["out"])/f"audit_attempt_{invocation}")])
        j.update(status="launching",log=str(log),command=cmd,launched_at=time.time(),action=action)
        self.save()  # Intent must precede process creation.
        env={**os.environ,"CUDA_VISIBLE_DEVICES":j["gpu"],"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
             "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
        try:
            with log.open("x") as stream:
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        except Exception:
            self.fail(j,"spawn",traceback.format_exc()); return
        rec=base.process_info(proc.pid) or {"pid":proc.pid,"argv":cmd,"uid":1009,"start_ticks":None}
        j.update(status="running" if action=="worker" else "auditing",record=rec)
        self.active[proc.pid]={"job":j,"record":rec,"process":proc,"action":action,"started":time.time(),
             "last_activity":time.time(),"signature":None,"required_mib":self.admission[spec.family]["required_mib"]}
        e.atomic_json(self.folder/"launches"/f"{invocation}.json",{"command":cmd,"record":rec,"snapshot":snapshot,
                     "required_mib":self.admission[spec.family]["required_mib"],"retry":retry})
        self.save()

    def watchdog(self,x):
        now=time.time(); rec=x["record"]
        if not rec.get("start_ticks"):
            current=base.process_info(rec["pid"])
            if current: x["record"]=current
            return
        try:
            with base.deadline(5): signature=activity(rec,x["job"])
        except (OSError,TimeoutError): return
        if signature!=x["signature"]: x.update(signature=signature,last_activity=now)
        idle=now-x["last_activity"]
        limit=self.admission[x["job"]["spec"]["family"]][x["action"]+"_timeout"]
        if idle>=1800: x["warning"]=f"No observed activity for {idle:.0f}s"
        overdue=now-x["started"]>limit
        if (idle>=3600 or overdue) and not x.get("terminated"):
            current=base.process_info(rec["pid"])
            if current and current["uid"]==1009 and current["argv"]==rec["argv"] and base.same_process(rec):
                os.kill(rec["pid"],signal.SIGTERM)
                x.update(terminated="phase timeout" if overdue else "60min confirmed inactivity",terminated_at=now)
        if x.get("terminated") and now-x["terminated_at"]>=60 and base.same_process(rec):
            os.kill(rec["pid"],signal.SIGKILL)

    def poll(self):
        for pid,x in list(self.active.items()):
            j=x["job"]
            try:
                code=x["process"].poll() if x["process"] is not None else (None if base.same_process(x["record"]) else 0)
                if code is None:
                    self.watchdog(x); continue
                if code or x.get("terminated"):
                    self.fail(j,x["action"],x.get("terminated",f"Process exit {code}"),base.log_tail(j["log"]),not x.get("terminated"))
                elif x["action"]=="worker":
                    if not (Path(j["out"])/"DONE").is_file(): self.fail(j,"worker","Exit without DONE")
                    else: j["status"]="audit_pending"
                else:
                    e.validate(j["out"],self.m,e.Spec(**j["spec"])); j["status"]="complete"
                del self.active[pid]; self.save()
            except Exception:
                # A parent-side per-cell validation error must not escape into the shared pause.
                if not base.same_process(x["record"]):
                    self.fail(j,"parent_validation",traceback.format_exc()); self.active.pop(pid,None); self.save()
                else: x["warning"]=traceback.format_exc()

    def infrastructure(self):
        with base.deadline(45):
            if time.time()-self.last_source_check>60:
                e.assert_manifest(self.m); self.last_source_check=time.time()
            if shutil.disk_usage(self.root).free/2**30<400: raise RuntimeError("Artifact free space below400GiB")
            avail=next(int(x.split()[1])/2**20 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
            if avail<32: raise RuntimeError("Host available memory below32GiB")
            self.save()

    def progress(self,snapshot,error):
        counts=Counter(j["status"] for j in self.jobs); timings=defaultdict(list)
        optimized=0
        for j in self.jobs:
            if j["status"] in ("complete","audit_pending","auditing"): optimized+=1
            if j["status"]=="complete":
                r=e.c.read_json(Path(j["out"])/"summary.json"); a=e.c.read_json(Path(j["out"])/"ACCEPTED.json")
                timings[j["spec"]["family"]].append((r["worker_sec"],a["audit_sec"]))
        per_gpu=defaultdict(float)
        for j in self.jobs:
            if j["status"] in TERMINAL: continue
            times=timings[j["spec"]["family"]]
            if len(times)>=2:
                per_gpu[j["gpu"]]+=statistics.median(sum(t) for t in times)
        eta=max(per_gpu.values())/2 if per_gpu and all(len(timings[f])>=2 for f in e.FAMILIES) else None
        value={"at":e.c.now_text(),"phase":"paused_infrastructure" if error else "smoke" if self.smoke else "production",
               "total":len(self.jobs),"optimization_finished":optimized,"accepted":counts["complete"],
               "failed_deferred":counts["failed_deferred"],"counts":dict(counts),"gpu_state":snapshot,
               "infrastructure_error":error,"eta_remaining_worker_and_audit_sec":eta,
               "eta_note":"Successful same-family throughput; report and deferred repairs excluded; shared-load estimate",
               "active":[{"name":x["job"]["name"],"gpu":x["job"]["gpu"],"action":x["action"],
                          "pid":x["record"]["pid"],"warning":x.get("warning"),
                          "detail":base.safe_progress(Path(x["job"]["out"])/"progress.json")} for x in self.active.values()]}
        e.atomic_json(self.folder/"progress.json",value)
        if time.time()-self.last_print>60:
            print(json.dumps({k:v for k,v in value.items() if k not in ("active","gpu_state")}),flush=True); self.last_print=time.time()

    def run(self):
        while True:
            error=None
            try: self.poll()
            except Exception: error=traceback.format_exc()
            try: self.infrastructure()
            except Exception: error=traceback.format_exc()
            snapshot=base.gpu_snapshot()
            if error is None:
                for _ in range(2):
                    for gpu in ("0","1","2","3"):
                        j=choose_job(self.jobs,gpu,self.active)
                        if not j or gpu not in snapshot["rows"]: continue
                        required=self.admission[j["spec"]["family"]]["required_mib"]
                        if capacity(gpu,self.active,snapshot["rows"][gpu]["memory_free_mib"],required):
                            try: self.launch(j,snapshot)
                            except Exception: error=traceback.format_exc(); break
            try: self.progress(snapshot,error)
            except Exception: print(json.dumps({"progress_error":traceback.format_exc()}),flush=True)
            if not self.active and all(j["status"] in TERMINAL for j in self.jobs) and error is None:
                e.atomic_json(self.folder/"QUEUE_FINISHED.json",{"at":e.c.now_text(),"accepted":sum(j["status"]=="complete" for j in self.jobs),
                     "failed_deferred":sum(j["status"]=="failed_deferred" for j in self.jobs),
                     "scientific_acceptance":all(j["status"]=="complete" for j in self.jobs)})
                return
            time.sleep(10)


def admission_from_smokes(jobs):
    result={}
    for family in e.FAMILIES:
        rows=[e.c.read_json(Path(j["out"])/"summary.json") for j in jobs if j["spec"]["family"]==family]
        audits=[e.c.read_json(Path(j["out"])/"ACCEPTED.json") for j in jobs if j["spec"]["family"]==family]
        peak=max([r[k] for r in rows for k in ("peak_allocated_gib","peak_reserved_gib")]+
                 [a[k] for a in audits for k in ("audit_peak_allocated_gib","audit_peak_reserved_gib")])
        projected=max(r["worker_sec"]-r["new_optimizer_sec"]+r["new_optimizer_sec"]*2000 for r in rows)
        result[family]={"required_mib":math.ceil(max(8,peak+2)*1024),"measured_peak_gib":peak,
                       "worker_timeout":max(21600,10*projected),"audit_timeout":max(21600,10*max(a["audit_sec"] for a in audits))}
    return result


def finish_report(root,artifact_root):
    log=root/f"report_{time.time_ns()}.log"
    try:
        with log.open("x") as f:
            subprocess.run([sys.executable,"-u","-B",str(REPORT),"--artifact-root",str(artifact_root),"--run-root",str(root)],
                           stdout=f,stderr=subprocess.STDOUT,check=True,timeout=21600)
    except Exception:
        e.atomic_json(root/f"REPORT_FAILED_{time.time_ns()}.json",{"traceback":traceback.format_exc(),"log":str(log)})


def main(args):
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require UID:GID1009:1020")
    if not args.execution.startswith("execution_") or not args.execution.replace("_","").isalnum():
        raise ValueError("Execution must be a simple execution_* name")
    root=args.artifact_root/"runs"/e.VARIANT/args.execution; root.mkdir(parents=True,exist_ok=True)
    if root.resolve().is_relative_to(e.c.REPO_ROOT): raise RuntimeError("External artifacts only")
    with (root/"controller.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=e.prepare(args.artifact_root,root)
        for rank in (29,44):
            spec=e.Spec(rank,"svf_velocity",.01); gpu=m["prefixes"][spec.key()]["gpu"]
            folder=root/"regression_fixtures"/f"rank{rank}"
            if (folder/"ACCEPTED.json").is_file():
                receipt=e.c.read_json(folder/"ACCEPTED.json")
                if not receipt["passed"] or receipt["source_hashes"]!=m["source_hashes"]:
                    raise RuntimeError("Stale historical regression fixture")
                continue
            while True:
                snapshot=base.gpu_snapshot()
                if snapshot["rows"].get(gpu,{}).get("memory_free_mib",0)>=8192: break
                print(json.dumps({"waiting_for_fixture_gpu":gpu,"rank":rank}),flush=True);time.sleep(10)
            log=root/f"fixture_rank{rank}_{time.time_ns()}.log"
            with log.open("x") as stream:
                subprocess.run([sys.executable,"-u","-B",str(WORKER),"fixture","--manifest",str(root/"launch_manifest.json"),
                                "--out-dir",str(folder),"--gpu",gpu,"--family","svf_velocity","--alpha","0.01","--case-rank",str(rank)],
                               stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3600,
                               env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"PYTHONDONTWRITEBYTECODE":"1"})
        smoke=Queue(root,m,True); smoke.run()
        if not all(j["status"]=="complete" for j in smoke.jobs):
            raise RuntimeError("Smoke gate failed; preserve failures, no production launch")
        admission=admission_from_smokes(smoke.jobs)
        e.atomic_json(root/"SMOKES_ACCEPTED.json",{"count":9,"admission":admission,"at":e.c.now_text()})
        bytes_per={f:max(sum(p.stat().st_size for p in Path(j["out"]).rglob("*") if p.is_file()) for j in smoke.jobs if j["spec"]["family"]==f) for f in e.FAMILIES}
        projected=sum(bytes_per.values())*700*1.3
        e.atomic_json(root/"storage_projection.json",{"bytes_per_family_smoke":bytes_per,"projected_bytes_with_30pct_margin":projected})
        if shutil.disk_usage(root).free<projected+400*2**30: raise RuntimeError("Insufficient projected artifact capacity")
        if args.stop_after=="smoke": return
        production=Queue(root,m,False,admission); production.run()
        finish_report(root,args.artifact_root)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
    p.add_argument("--stop-after",choices=("smoke","full"),default="full")
    p.add_argument("--execution",default="execution_v1")
    main(p.parse_args())
