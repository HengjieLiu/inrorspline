"""All-GPU rank0 combined-loss follow-up, reusing P84 scheduling/telemetry guards."""
from __future__ import annotations

import argparse
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

from scripts.experiments.oasis_regularizer_sweep import controller as base
from scripts.experiments.oasis_regularizer_sweep.protocol import dump,read,sha256,stamp,sequence
from .protocol import BACKENDS,GAMMAS,KEY,PROTOCOL,setting

REPO=base.REPO
WORKER="scripts.experiments.oasis_diffusion_kl_rank0.worker"


def find_worker(path,container_pid):
    matches=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            item=base.identity(int(proc.name))
            if WORKER not in item["argv"] or path not in item["argv"]: continue
            if not Path(item["argv"].split()[0]).name.startswith("python"): continue
            ns=next(line.split()[1:] for line in (proc/"status").read_text().splitlines() if line.startswith("NSpid:"))
            if int(ns[-1])==container_pid: matches.append(item)
        except (OSError,ValueError,IndexError,StopIteration): continue
    if len(matches)!=1: raise RuntimeError(f"Cannot identify unique combined worker: {matches}")
    return matches[0]


class Study(base.Study):
    def __init__(self,args):
        self.args=args; self.root=args.artifact_host/"runs"/args.run_id
        self.review=args.artifact_host/"repo_experiments"/KEY/args.run_id
        self.devices=base.gpu_inventory(); self.lock=threading.RLock(); self.active={}
        self.root.mkdir(parents=True,exist_ok=True); self.review.mkdir(parents=True,exist_ok=True)
        self.filelock=(self.root/"controller.lock").open("a"); fcntl.flock(self.filelock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=[]
        for folder in ("scripts/experiments/oasis_diffusion_kl_rank0","scripts/experiments/oasis_regularizer_sweep"):
            sources+=list((REPO/folder).glob("*.py"))
        sources += [REPO/name for name in ("scripts/experiments/oasis_alpha_sweep/common.py",
            "scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py","scripts/experiments/mrdbscp842_acceleration/nvml_monitor.py",
            "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt","tests/test_oasis_regularizer_sweep.py",
            "tests/test_oasis_diffusion_kl_rank0.py","experiments/2026-11-10_oasis_rank000_diffusion_kl_c1/codex_execution_spec.md")]
        self.hashes={str(p.relative_to(REPO)):sha256(p) for p in sources}
        manifest=self.root/"manifest.json"
        if manifest.exists():
            old=read(manifest)
            if old["source_hashes"]!=self.hashes or old["protocol"]!=PROTOCOL: raise RuntimeError("Frozen source/protocol drift")
        else:
            inspect=json.loads(base.command(["docker","inspect",args.container]))[0]
            dump(manifest,dict(created_at=stamp(),protocol=PROTOCOL,source_hashes=self.hashes,git_head=base.command(["git","rev-parse","HEAD"]),
                git_branch=base.command(["git","branch","--show-current"]),git_status=base.command(["git","status","--short"]),
                devices=self.devices,container=dict(image=inspect["Config"]["Image"],image_id=inspect["Image"],user=inspect["Config"]["User"],mounts=inspect["Mounts"]),args=vars(args)))
            for path in sources:
                target=self.root/"source"/path.relative_to(REPO); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
            (self.root/"source"/"dirty.patch").write_text(base.command(["git","diff","HEAD"]))
        dump(self.root/"controller.json",dict(identity=base.identity(os.getpid()),started_at=stamp(),command=sys.argv))
        self.update_status("initializing")

    def update_status(self,phase):
        with self.lock:
            files=list((self.root/"blocks").glob("*/*/attempt*/block.json"))
            status=dict(updated_at=stamp(),phase=phase,active=self.active,
                completed_blocks=sum(read(p).get("complete",False) for p in files),
                evaluated_blocks=len(list((self.root/"blocks").glob("*/*/attempt*/EVALUATED.json"))),
                first10_extension=False,report=str(self.review/"report.md"))
            dump(self.root/"status.json",status)
            (self.review/"progress.md").write_text("# Rank0 diffusion1 + analytic KL-C1\n\n"+
                f"Updated: {stamp()}\n\nPhase: {phase}; only case0.\n\n"+
                "\n".join(f"- GPU {g}: {v}" for g,v in self.active.items())+"\n")

    def job(self,phase,backend,gamma):
        root=self.root/"blocks"/phase/setting(backend,gamma)
        attempts=sorted(root.glob("attempt*"))
        for attempt in attempts:
            if (attempt/"ACTIVE_WORKER_REQUIRES_RECONCILIATION.json").exists(): raise RuntimeError("Active orphan requires reconciliation")
            if (attempt/"register_ACCEPTED.json").exists():
                return dict(phase=phase,arm="diffusion_"+backend,weight=1.,gamma=gamma,host_block_dir=str(attempt),block_dir=self.container_path(attempt))
        root=root/f"attempt{len(attempts)}"
        return dict(phase=phase,arm="diffusion_"+backend,weight=1.,gamma=gamma,host_block_dir=str(root),block_dir=self.container_path(root))

    def run_process(self,job,device,evaluate=False):
        self.frozen(); self.admission(device)
        root=Path(job["host_block_dir"]); root.mkdir(parents=True,exist_ok=True)
        marker="eval" if evaluate else "register"; gpu=device["gpu"]
        if (root/f"{marker}_boot.json").exists(): raise RuntimeError("Prior incomplete handshake; preserve evidence and use a new attempt")
        job=dict(job,gpu=gpu,gpu_uuid=device["uuid"],source_hashes=self.hashes,parent_launch_ns=time.monotonic_ns(),oasis_root=self.args.oasis_root)
        path=root/f"{marker}_job.json"; dump(path,job)
        argv=["docker","exec","-w","/workspace/repo","-e",f"CUDA_VISIBLE_DEVICES={gpu}","-e","OMP_NUM_THREADS=1",
            "-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1",self.args.container,"python","-B","-m",WORKER,"--job",self.container_path(path)]
        if evaluate: argv.append("--evaluate")
        dump(root/f"{marker}_command.json",argv)
        monitor=base.Monitor(device,root/f"{marker}_monitor.jsonl")
        with self.lock: self.active[gpu]=f"{marker}/{job['phase']}/{job['arm']}/gamma{job['gamma']:g}"
        self.update_status(job["phase"])
        with (root/f"{marker}.log").open("w") as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT); monitor.thread.start()
            try:
                deadline=time.monotonic()+180; boot=root/f"{marker}_boot.json"
                while not boot.exists():
                    if process.poll() is not None: raise RuntimeError("Worker exited before handshake")
                    if time.monotonic()>deadline: raise TimeoutError("Boot timed out")
                    time.sleep(.1)
                owner=find_worker(self.container_path(path),read(boot)["container_pid"])
                monitor.owner=owner; dump(root/f"{marker}_host_identity.json",owner); dump(root/f"{marker}_ack.json",owner)
                deadline=time.monotonic()+4*3600
                while process.poll() is None:
                    if monitor.error: raise RuntimeError(monitor.error)
                    if time.monotonic()>deadline: raise TimeoutError("Rank0 block exceeded4h watchdog")
                    time.sleep(.5)
                if process.returncode: raise RuntimeError(f"Combined worker exit{process.returncode}; inspect {root/(marker+'.log')}")
            finally:
                if process.poll() is None:
                    dump(root/"ACTIVE_WORKER_REQUIRES_RECONCILIATION.json",dict(exec_pid=process.pid,owner=monitor.owner))
                try: monitor.close()
                finally:
                    with self.lock: self.active.pop(gpu,None)
                    self.update_status(job["phase"])
        self.summarize_monitor(root,marker,monitor.owner)
        if evaluate:
            if not (root/"EVALUATED.json").exists(): raise RuntimeError("Missing evaluation marker")
        else:
            block=read(root/"block.json")
            if not block["complete"] or [(r["case_rank"],r["condition"]) for r in block["cases"]]!=sequence(job["phase"]):
                raise RuntimeError("Incorrect rank0 sequence")
            for row in block["cases"]:
                if row["lambda_diffusion"]!=1 or row["lambda_kl"]!=job["gamma"] or row["kl_backend"]!="analytic":
                    raise RuntimeError("Coefficient/backend metadata mismatch")
        dump(root/f"{marker}_ACCEPTED.json",dict(complete=True,at=stamp(),host_identity=monitor.owner))

    def report(self):
        subprocess.run(["docker","exec","-w","/workspace/repo",self.args.container,"python","-B","-m",
            "scripts.experiments.oasis_diffusion_kl_rank0.report","--root",self.container_path(self.root),"--out",self.container_path(self.review)],check=True)

    def preflight(self):
        if (self.root/"PREFLIGHT_ACCEPTED.json").exists(): return
        for pattern in ("test_oasis_regularizer_sweep.py","test_oasis_diffusion_kl_rank0.py"):
            result=subprocess.run(["docker","exec","-w","/workspace/repo",self.args.container,"python","-B","-m",
                "unittest","discover","-s","tests","-p",pattern,"-v"],capture_output=True,text=True)
            (self.root/(pattern+".log")).write_text(result.stdout+result.stderr)
            if result.returncode: raise RuntimeError(f"Failed {pattern}")
        subprocess.run(["docker","exec","-w","/workspace/repo",self.args.container,"python","-B","-m",
            "scripts.experiments.oasis_diffusion_kl_rank0.baselines","--prior-root",self.args.prior_root,
            "--out",self.container_path(self.root/"baselines"),"--oasis-root",self.args.oasis_root],check=True)
        jobs=[self.job("smoke",backend,gamma) for gamma in (1e-4,1.) for backend in BACKENDS]
        self.wave(jobs); self.wave(jobs,evaluate=True)
        peak=max(r["peak_reserved"] for job in jobs for r in read(Path(job["host_block_dir"])/"block.json")["cases"])
        dump(self.root/"PREFLIGHT_ACCEPTED.json",dict(complete=True,at=stamp(),smokes=4,peak_reserved=peak,source_hashes=self.hashes))
        self.report()

    def run(self):
        if self.args.phase=="report": self.report(); return
        self.preflight()
        if self.args.phase=="preflight": self.update_status("preflight_complete"); return
        jobs=[self.job("pilot",backend,gamma) for gamma in GAMMAS[1:] for backend in BACKENDS]
        self.wave(jobs); self.wave(jobs,evaluate=True)
        rows=[r for job in jobs for r in read(Path(job["host_block_dir"])/"block.json")["cases"]]
        assert len(rows)==20 and all(r["case_rank"]==0 for r in rows)
        dump(self.root/"ACCEPTED.json",dict(complete=True,at=stamp(),new_settings=10,new_observations=20,
            reused_settings=2,reused_observations=4,smoke_observations=4,first10_extension=False,source_hashes=self.hashes))
        self.update_status("complete_rank0_only"); self.report()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container",required=True); p.add_argument("--artifact-host",type=Path,required=True)
    p.add_argument("--artifact-container",type=Path,default=Path("/database/inrorspline"))
    p.add_argument("--oasis-root",required=True); p.add_argument("--prior-root",required=True)
    p.add_argument("--run-id",required=True); p.add_argument("--phase",choices=("preflight","all","report"),default="all")
    args=p.parse_args()
    if "/" in args.run_id or args.run_id in (".",".."): p.error("Safe single-component run ID required")
    study=Study(args)
    try: study.run()
    except BaseException as error:
        dump(study.root/"CONTROLLER_FAILED.json",dict(at=stamp(),error=repr(error),traceback=traceback.format_exc()))
        study.update_status("failed"); raise


if __name__=="__main__": main()
