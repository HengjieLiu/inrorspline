#!/usr/bin/env python
"""P79 durable eight-slot extension of the audited continue-on-error scheduler."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import oasis_bspline842_first100 as e
import queue_oasis_svf_continue_on_error as base
from queue_oasis_svf_eight_workers import process_info, option

WORKER = Path(__file__).with_name("run_oasis_bspline842_first100.py")
REPORT = e.c.REPO_ROOT / "scripts/reports/report_oasis_bspline842_first100.py"
POLICY = {**base.POLICY, "min_artifact_free_gib": 100,
          "initial_required_free_gib": 800, "audit_gpu": "same physical GPU as registration",
          "scientific_validation": "unchanged family objectives plus independent replay and artifact receipts"}


def choose_job(jobs, gpu, affinity):
    ordinary = any(j["status"] in ("pending","running","launching","audit_pending","auditing") for j in jobs)
    choices = []
    for i,j in enumerate(jobs):
        status = j["status"]
        if status not in ("pending","audit_pending","oom_retry") or (status == "oom_retry" and ordinary):
            continue
        audit = status == "audit_pending" or (status == "oom_retry" and j.get("retry_action") == "audit")
        if audit and j.get("gpu") != gpu:
            continue
        choices.append(((not audit, affinity.get(j["spec"].case_rank) != gpu, i), j))
    return min(choices, key=lambda x: x[0])[1] if choices else None


def live_processes(root):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        rec = process_info(p.name)
        if not rec or rec["uid"] != 1009 or str(WORKER) not in rec["argv"] or rec["state"] == "Z":
            continue
        argv = rec["argv"]
        out = Path(option(argv, "--out-dir", "/"))
        if not out.is_relative_to(root):
            continue
        action = argv[argv.index(str(WORKER))+1]
        if action in ("worker","audit"):
            rec.update(action=action, out=str(out), gpu=option(argv,"--gpu")); found.append(rec)
    return found


class Queue(base.Queue):
    def __init__(self, root, manifest, specs, required=8192):
        # Reuse durable journal, PID/start-time adoption, isolated error/OOM
        # handling, resource guards and watchdog without changing old sources.
        base.FOLDER = "controller"
        base.POLICY = POLICY
        base.choose_job = choose_job
        base.r = SimpleNamespace(c=e.c, atomic_json=e.atomic_json, sha=e.sha,
            validate=e.validate, assert_manifest=e.assert_manifest,
            s=SimpleNamespace(Spec=e.Spec, matrix=lambda: specs))
        base.sources = lambda: e.source_hashes(manifest["sitreg_root"])["extension"]
        super().__init__(root, manifest, required)
        self.folder.mkdir(parents=True, exist_ok=True)
        for child in ("failures", "launches"):
            (self.folder / child).mkdir(exist_ok=True)
        self.smoke = bool(specs and specs[0].smoke)

    def launch(self, job, gpu, snapshot):
        retry = job["status"] == "oom_retry"
        action = job.get("retry_action", "worker") if retry else ("audit" if job["status"] == "audit_pending" else "worker")
        if retry and action == "worker":
            job["attempt"] += 1
        path = Path(job["out"]) if action == "audit" else self.root/job["spec"].name()/f"attempt{job['attempt']}"
        job.update(out=str(path), gpu=gpu, action=action)
        try:
            if action == "worker" and path.exists():
                raise FileExistsError(f"Preserve existing attempt: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if retry and action == "audit":
                archive = path.parent/f"audit_oom_preserved_{time.time_ns()}"
                archive.mkdir()
                # Preserve partial replay exports; the unchanged SVF auditor
                # creates metric directories with exist_ok=True.
                for p in list(path.glob("independent_*"))+list(path.glob("FAILED_audit*")):
                    if p.is_dir(): shutil.copytree(p, archive/p.name)
                    else: shutil.copy2(p, archive/p.name)
            stamp = str(time.time_ns())
            log_path = path.parent/f"attempt{job['attempt']}_{action}_{stamp}.log"
            log = log_path.open("x")
        except (OSError,ValueError):
            self.failure(job,"launch_preparation",traceback.format_exc()); return
        job.update(status="launching", log_path=str(log_path), invocation=stamp, launch_at=time.time())
        try:
            self.save()
            spec = job["spec"]
            command = [sys.executable,"-u","-B",str(WORKER),action,"--manifest",str(self.folder/"launch_manifest.json"),
                "--out-dir",str(path),"--gpu",gpu,"--case-rank",str(spec.case_rank),
                "--family",spec.family,"--alpha",str(spec.alpha)]
            if spec.smoke: command.append("--smoke")
            env = {**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
                   "OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","PYTHONDONTWRITEBYTECODE":"1"}
            try:
                proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            except OSError:
                self.failure(job,"process_spawn",traceback.format_exc()); return
            rec = process_info(proc.pid) or {"pid":proc.pid,"start_ticks":None,"argv":command,"uid":os.getuid()}
            job.update(status="running" if action=="worker" else "auditing",record=rec)
            self.active[proc.pid] = {"record":rec,"process":proc,"job":job,"action":action,
                                    "signature":None,"last_activity":time.time()}
            self.affinity.setdefault(spec.case_rank,gpu)
            event={"at":e.c.now_text(),"run":spec.name(),"action":action,"gpu":gpu,"pid":proc.pid,
                   "log_path":str(log_path),"snapshot":snapshot,"required_free_mib":self.required,"retry":retry}
            e.atomic_json(self.folder/"launches"/f"{stamp}.json",event)
            self.save(); print(json.dumps(event),flush=True)
        finally:
            log.close()

    def reconcile(self, live):
        super().reconcile(live)
        # Original controller recognizes a completed replay before its receipt;
        # P79 can finish receipt creation idempotently after a late crash.
        for j in self.jobs:
            if j["status"] == "failed_deferred" and j.get("failure",{}).get("action") == "resume_validation":
                p = Path(j["out"])
                if not (p/"extension_receipt.json").exists() and (p/"independent_audit.json").exists():
                    audit=e.c.read_json(p/"independent_audit.json")
                    if audit.get("passed") and audit["summary_sha256"]==e.sha(p/"summary.json"):
                        j.update(status="audit_pending",gpu=e.c.read_json(p/"summary.json")["gpu"])
            if j["status"]=="audit_pending" and j.get("gpu") is None:
                j["gpu"]=e.c.read_json(Path(j["out"])/"summary.json")["gpu"]
        self.save()

    def progress(self, snapshot, infrastructure_error=None):
        result=super().progress(snapshot,infrastructure_error)
        completed=[j for j in self.jobs if j["status"]=="complete"]
        optimized=[j for j in self.jobs if j["status"] in ("complete","audit_pending","auditing")]
        result.update(reused_count=0 if self.smoke else 570, optimization_complete=len(optimized),
            independently_validated=len(completed), reporting="pending", phase_kind="smoke" if self.smoke else "production",
            completed_by_family=dict(Counter(j["spec"].family for j in completed)),
            total_accepted_with_reuse=len(completed)+(0 if self.smoke else 570))
        e.atomic_json(self.root/"progress.json",result)
        return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
    p.add_argument("--gpus",choices=("0,1,2,3",),default="0,1,2,3")
    p.add_argument("--max-concurrent",type=int,choices=(8,),default=8)
    p.add_argument("--workers-per-gpu",type=int,choices=(2,),default=2)
    p.add_argument("--container-image",required=True)
    p.add_argument("--container-image-id",required=True)
    p.add_argument("--stop-after",choices=("prepare","smoke","full"),default="full")
    p.add_argument("--execution",choices=e.EXECUTIONS,default="original")
    a=p.parse_args(argv)
    if a.artifact_root.resolve().is_relative_to(e.c.REPO_ROOT):p.error("External artifacts only")
    return a


def phase(root, manifest, smoke, required):
    specs=e.matrix(smoke=True) if smoke else e.matrix(missing=True)
    q=Queue(root,manifest,specs,required)
    mp=q.folder/"launch_manifest.json"
    if not mp.exists():e.atomic_json(mp,manifest)
    elif e.c.read_json(mp)!=manifest:raise RuntimeError("Controller manifest changed")
    q.reconcile(live_processes(root))
    q.infrastructure()
    print(json.dumps({"phase":"smoke" if smoke else "production","counts":dict(Counter(j['status'] for j in q.jobs))}),flush=True)
    q.run()
    return q


def main(args):
    if (os.getuid(),os.getgid())!=(1009,1020):raise RuntimeError("Require UID:GID1009:1020")
    root=e.run_root(args.artifact_root,args.execution)
    root.mkdir(parents=True,exist_ok=True)
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=e.prepare(args.artifact_root,args.container_image,args.container_image_id,args.execution)
        m={**m,"execution":{"policy":POLICY,"source_hashes":m['source_hashes']['extension']}}
        if args.stop_after=="prepare":return
        smoke=phase(root/"smokes",m,True,8192)
        if any(j['status']!='complete' for j in smoke.jobs):
            raise RuntimeError("Smoke failure: preserve attempts; production not authorized by verification gate")
        peaks=[]
        for j in smoke.jobs:
            p=Path(j['out']);row=e.c.read_json(p/'summary.json');audit=e.c.read_json(p/'independent_audit.json')
            resource=e.c.read_json(p/'extension_provenance.json')
            peaks.extend(float(x[k]) for x in (row,audit,resource) for k in
                ('peak_cuda_gib','peak_allocated_gib','peak_reserved_gib','audit_peak_cuda_gib','audit_peak_reserved_gib') if k in x)
        required=max(8192,math_ceil_mib(max(peaks)+2))
        e.atomic_json(root/'SMOKES_PASSED.json',{'passed':True,'count':6,'required_free_mib':required,'max_peak_gib':max(peaks),'at':e.c.now_text()})
        if args.stop_after=='smoke':return
        production=phase(root,m,False,required)
        logpath=root/f'report_{time.time_ns()}.log'
        try:
            with logpath.open('x') as log:
                result=subprocess.run([sys.executable,'-u','-B',str(REPORT),'--artifact-root',str(args.artifact_root),
                                       '--execution',args.execution],
                                      stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            if result.returncode:raise RuntimeError(f'Report exited {result.returncode}')
        except Exception:
            e.atomic_json(root/'REPORT_FAILED.json',{'log':str(logpath),'traceback':traceback.format_exc(),'at':e.c.now_text()})
            raise
        print(json.dumps({'queue_finished':True,'accepted':sum(j['status']=='complete' for j in production.jobs),
                          'failures':sum(j['status']=='failed_deferred' for j in production.jobs),'report_log':str(logpath)}),flush=True)


def math_ceil_mib(gib):
    import math
    return math.ceil(gib*1024)


if __name__=='__main__':
    main(parse_args())
