#!/usr/bin/env python
"""One-slot-per-GPU Phase-A queue; only explicit smoke/production phases."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_oasis_symmetric_svf as r


def main(args):
    root = args.artifact_root/"runs"/r.m.VARIANT/"phase_a"
    lock = (root/"queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = r.c.read_json(root/"launch_manifest.json")
    r.assert_manifest(manifest)
    # Never duplicate children left by a previous controller.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try: cmd = (proc/"cmdline").read_bytes().split(b"\0")
        except OSError: continue
        if str(root/"launch_manifest.json").encode() in cmd and any(x.endswith(b"/run_oasis_symmetric_svf.py") for x in cmd):
            raise RuntimeError("An existing worker/auditor remains active; preserve it and audit controller recovery")
    required = 8192.
    if args.phase == "production":
        gate = r.c.read_json(root/"SMOKES_PASSED.json")
        if not gate["passed"] or gate["source_hashes"] != r.source_hashes():
            raise RuntimeError("Full-size smokes required")
        for item in gate["runs"]:
            r.validate(Path(item["path"]), manifest, r.m.Spec(**item["spec"]))
        required = max(required, (gate["peak_gib"]+2)*1024)
    jobs=[]
    for spec in r.m.matrix(args.phase == "smoke"):
        path=root/spec.name()/"attempt0"
        status="pending"
        if (path/"independent_audit.json").is_file():
            r.validate(path,manifest,spec);status="complete"
        elif (path/"DONE").is_file(): status="audit_pending"
        elif path.exists():
            raise RuntimeError(f"Preserve incomplete attempt for audit: {path}")
        jobs.append({"spec":spec,"path":path,"status":status,"gpu":None})
    active={};failures=[];started=time.monotonic();last_print=0.
    while any(j["status"] not in ("complete","failed") for j in jobs):
        for gpu,item in list(active.items()):
            proc,job,action=item["process"],item["job"],item["action"]
            code=proc.poll()
            if code is None:continue
            item["log"].close()
            if code:
                job["status"]="failed"
                failures.append({"arm":job["spec"].arm,"action":action,"returncode":code,"log":str(item["log_path"])})
            elif action=="worker":job["status"]="audit_pending"
            else:
                r.validate(job["path"],manifest,job["spec"]);job["status"]="complete"
            del active[gpu]
        state=r.p.legacy.gpu_snapshot()
        available={str(x["physical_gpu"]):x for x in state.get("rows",[]) if str(x["physical_gpu"]) in manifest["gpus"]
                   and str(x["physical_gpu"]) not in active and x["memory_free_mib"]>=required}
        for job in sorted(jobs,key=lambda j:j["status"]!="audit_pending"):
            if job["status"] not in ("pending","audit_pending") or not available:continue
            r.assert_manifest(manifest)
            gpu=job["gpu"] if job["gpu"] in available else max(available,key=lambda k:available[k]["memory_free_mib"])
            action="audit" if job["status"]=="audit_pending" else "worker"
            path=job["path"];path.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,"-u","-B",str(Path(r.__file__)),action,"--manifest",str(root/"launch_manifest.json"),
                     "--out-dir",str(path),"--gpu",gpu]
            if action=="worker":
                command += ["--arm",job["spec"].arm]
                if job["spec"].smoke:command.append("--smoke")
            log_path=path.parent/f"{action}_{time.time_ns()}.log"
            stream=log_path.open("x")
            env={**os.environ,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":gpu,
                 "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4",
                 "MPLCONFIGDIR":"/tmp/inrorspline_p77_mpl"}
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            job.update(status="running",gpu=gpu)
            active[gpu]={"process":process,"job":job,"action":action,"log":stream,"log_path":log_path}
            event={"time":r.c.now_text(),"phase":args.phase,"arm":job["spec"].arm,"action":action,"gpu":gpu,
                   "pid":process.pid,"gpu_before":available[gpu],"log":str(log_path),"command":command}
            with (root/"launch_events.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
            print(json.dumps(event),flush=True)
            del available[gpu]
        completed=sum(j["status"]=="complete" for j in jobs)
        status={"phase":args.phase,"completed":completed,"total":4,"failures":failures,
                "elapsed_sec":time.monotonic()-started,"required_free_mib":required,"updated_at":r.c.now_text(),
                "jobs":[{"spec":asdict(j["spec"]),"path":str(j["path"]),"status":j["status"],"gpu":j["gpu"]} for j in jobs],
                "active":[{"gpu":g,"action":v["action"],"arm":v["job"]["spec"].arm,"pid":v["process"].pid} for g,v in active.items()]}
        r.atomic_json(root/f"{args.phase}_progress.json",status)
        if time.monotonic()-last_print>=30:
            print(json.dumps(status),flush=True);last_print=time.monotonic()
        if any(j["status"] not in ("complete","failed") for j in jobs):time.sleep(5)
    result={"passed":not failures,"source_hashes":r.source_hashes(),"queue_source_sha256":r.sha(__file__),
            "completed":sum(j["status"]=="complete" for j in jobs),"failures":failures,
            "elapsed_sec":time.monotonic()-started,"completed_at":r.c.now_text(),"runs":[],"peak_gib":0.}
    for job in jobs:
        if job["status"]!="complete":continue
        row=r.validate(job["path"],manifest,job["spec"])
        result["runs"].append({"spec":asdict(job["spec"]),"path":str(job["path"]),"gpu":row["gpu"]})
        result["peak_gib"]=max(result["peak_gib"],row["peak_cuda_gib"],row["peak_reserved_gib"],
                                row["audit"]["audit_peak_cuda_gib"],row["audit"]["audit_peak_reserved_gib"])
    r.atomic_json(root/("SMOKES_PASSED.json" if args.phase=="smoke" and not failures else args.phase.upper()+"_FINISHED.json"),result)
    print(json.dumps(result),flush=True)
    if failures:raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
    parser.add_argument("--phase",choices=("smoke","production"),required=True)
    main(parser.parse_args())
