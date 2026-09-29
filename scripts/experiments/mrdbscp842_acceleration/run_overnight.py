"""Deadline-bounded GPU-0 accuracy/speed study; sequential, resumable, monitored."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from .run_be_combo_first10 import ComboSuite
from .run_suite import Worker, dump, process_identity
from .overnight_candidates import EXPERIMENT, GATES, main_candidates, optional_candidates
from .overnight_report import build


class BudgetEnd(RuntimeError):
    pass


class TimedWorker(Worker):
    def __init__(self,suite,name,features=()):
        self.guard_done = threading.Event()
        self.force_lock = threading.Lock()
        super().__init__(suite,name,features)
        self.guard = threading.Thread(target=self.watch_deadline,daemon=True)
        self.guard.start()

    def watch_deadline(self):
        while not self.guard_done.wait(1):
            if time.time() >= self.suite.gpu_deadline:
                self.terminate_owned("GPU deadline")
                return

    def terminate_owned(self,reason):
        """Never touch foreign jobs: verify token/PID/start ticks before own-worker signal."""
        with self.force_lock:
            ack = self.mail/"host_ack.json"
            if not ack.exists() or self.process.poll() is not None:
                return
            ident=json.loads(ack.read_text())
            try:
                current=process_identity(ident["pid"])
            except FileNotFoundError:
                return
            token=self.argv[self.argv.index("--job-token")+1]
            if current["start_ticks"] != ident["start_ticks"] or token not in current["argv"]:
                raise RuntimeError("Refuse to signal unverified/reused worker PID")
            ready=json.loads((self.mail/"ready.json").read_text())
            dump(self.mail/"deadline_stop.json",dict(reason=reason,identity=ident,unix=time.time()))
            os.kill(ident["pid"],signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                current=process_identity(ident["pid"])
                if current["start_ticks"]==ident["start_ticks"] and token in current["argv"]:
                    os.kill(ident["pid"],signal.SIGKILL)

    def wait_file(self,path,timeout=3600):
        started=time.monotonic()
        while not path.exists():
            if time.time() >= self.suite.gpu_deadline:
                self.terminate_owned("GPU deadline")
                raise BudgetEnd("GPU cutoff reached")
            if self.process.poll() is not None:
                raise RuntimeError(f"Worker exited; see {self.mail}")
            if time.monotonic()-started > timeout:
                self.terminate_owned("per-job timeout")
                raise TimeoutError(str(path))
            time.sleep(.2)

    def close(self):
        try:
            if self.process.poll() is None:
                dump(self.mail/f"command_{self.index:04d}.json",dict(action="exit"))
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.terminate_owned("owned worker failed to close")
        finally:
            self.guard_done.set()
            self.suite.monitor.owner=None
            self.log.close()


class OvernightSuite(ComboSuite):
    def __init__(self,args):
        self.deadline=datetime.fromisoformat(args.deadline).timestamp()
        self.gpu_deadline=self.deadline-600
        self.launch_deadline=self.deadline-1200
        super().__init__(args)
        self.quality=json.loads((self.root/"quality_rows.json").read_text()) if (self.root/"quality_rows.json").exists() else []
        self.monitor.wait_clear=self.wait_clear

    def budget(self,seconds=0):
        if time.time()+seconds >= self.launch_deadline:
            raise BudgetEnd("Reserved final evaluation/report window reached")

    def wait_clear(self):
        clean=0
        while clean<3:
            if time.time() >= self.gpu_deadline:
                raise BudgetEnd("GPU unavailable before cutoff")
            sample=self.monitor.sample()
            clean=clean+1 if not sample["error"] and not sample["foreign"] else 0
            if not clean:
                self.state("waiting_for_GPU0", foreign=sample["foreign"],error=sample["error"])
            time.sleep(1)

    def state(self,status,**kw):
        dump(self.root/"state.json",dict(status=status,utc=datetime.now(timezone.utc).isoformat(),
            deadline=self.args.deadline,**kw))

    def execute(self,worker,action,label,**kw):
        row=worker.execute(action,label,command_timeout=900,**kw)
        row["clean"] &= row["monitor"]["max_sample_gap_sec"] <= 5.
        row["monitor"]["max_gap_pass"] = row["monitor"]["max_sample_gap_sec"] <= 5.
        row["monitor"]["clean"] = row["clean"]
        dump(self.root/"attempts.json",self.attempts)
        dump(self.root/"jobs"/label/"accepted_result.json",row)
        return row

    def registration_block(self,spec,phase="headline",smoke=False):
        self.frozen()
        worker=None
        name=spec["name"]
        ranks=[0] if smoke else [0,0,1,2,3,4,5,6,7,8,9]
        try:
            for index,rank in enumerate(ranks):
                self.budget(60)
                existing=[r for r in self.rows if r["variant"]==name and r.get("phase")==phase and r["job_index"]==index]
                if existing:
                    continue
                attempt=0
                while True:
                    if worker is None:
                        worker=TimedWorker(self,name,spec["features"])
                        if index>0:
                            # Resume cannot call an unprimed context warm.
                            priming=self.execute(worker,"schedule_register",f"prime_{name}_{uuid.uuid4().hex[:8]}",
                                rank=rank,candidate=spec)
                            if not priming["clean"]:
                                worker.close(); worker=None
                                continue
                    label=f"{phase}_{name}_{index:03d}_attempt{attempt}"
                    if (self.root/"jobs"/label).exists():
                        attempt+=1
                        continue
                    self.state("registration",candidate=name,index=index,rank=rank,phase=phase)
                    row=self.execute(worker,"schedule_register",label,rank=rank,candidate=spec,smoke=smoke)
                    if row["clean"]:
                        self.record(row,variant=name,phase=phase,block=phase+"_"+name,job_index=index,
                            condition="cold" if index==0 else "warm",primary=not smoke)
                        print(json.dumps(dict(completed=label,seconds=row["elapsed_sec"],steps=row["steps"])),flush=True)
                        break
                    attempt+=1
                    if index==0:
                        worker.close();worker=None
                    else:
                        self.monitor.wait_clear()
        finally:
            if worker:
                worker.close()

    def evaluate_candidate(self,spec):
        self.frozen()
        name=spec["name"]
        items=[r for r in self.rows if r["variant"]==name and r.get("phase")=="headline" and r["condition"]=="warm"]
        done={r["registration_id"] for r in self.quality}
        items=[r for r in items if r["id"] not in done]
        if not items:
            return
        worker=TimedWorker(self,"quality_"+name)
        try:
            for item in items:
                self.budget(300)
                self.state("evaluation",candidate=name,rank=item["rank"])
                attempt=0
                while True:
                    label=f"quality_{name}_rank{item['rank']:03d}_attempt{attempt}"
                    if (self.root/"jobs"/label).exists():
                        attempt+=1
                        continue
                    row=self.execute(worker,"schedule_evaluate",label,item=item)
                    if row["clean"]:
                        self.quality.append(row)
                        dump(self.root/"quality_rows.json",self.quality)
                        print(json.dumps(dict(evaluated=name,rank=item["rank"],dice=row["mean_dice"],hd95=row["surface_metrics"]["hd95_mean"],metric_seconds=row["metric_seconds"])),flush=True)
                        build(self.root,self.review)
                        break
                    attempt+=1
                    self.monitor.wait_clear()
        finally:
            worker.close()

    def preflight(self,specs):
        self.helper("contracts","scripts.experiments.mrdbscp842_acceleration.test_overnight",
                    "--output",self.container(self.root/"contracts.json"))
        w=TimedWorker(self,"environment")
        try:
            value=self.execute(w,"environment","environment_attempt0")
        finally:
            w.close()
        if not value["clean"]:
            raise RuntimeError("Preflight contaminated; resume after GPU clears")
        dump(self.root/"data_environment.json",value)
        # Smoke is correctness-only: one persistent worker avoids redundant CUDA
        # setup; formal candidate cold/warm blocks still use fresh workers.
        self.frozen()
        w=TimedWorker(self,"schedule_smokes",("cpbe",))
        try:
            for spec in specs:
                self.budget(60)
                self.state("smoke",candidate=spec["name"])
                if any(r.get("phase")=="smoke" and r["variant"]==spec["name"] for r in self.rows):
                    continue
                label="smoke_"+spec["name"]+"_attempt0"
                row=self.execute(w,"schedule_register",label,rank=0,candidate=spec,smoke=True)
                if not row["clean"]:
                    raise RuntimeError("Contaminated correctness smoke; retry preflight")
                self.record(row,variant=spec["name"],phase="smoke",block="smoke",job_index=0,condition="smoke",primary=False)
                print(json.dumps(dict(smoke=spec["name"],steps=row["steps"])),flush=True)
        finally:
            w.close()
        # Check fixed-schedule reference bridge using identical warm trajectories.
        # Fresh reference formal run follows; no smoke measurement enters headline.
        smoke=next(r for r in self.rows if r["phase"]=="smoke")
        w=TimedWorker(self,"metric_smoke")
        try:
            metric=self.execute(w,"schedule_evaluate","metric_smoke_attempt0",item=smoke)
        finally:
            w.close()
        if not metric["clean"] or not metric["hd95_nan_pattern_matches_dice"]:
            raise RuntimeError("Metric smoke failed")
        dump(self.root/"preflight_DONE.json",dict(utc=datetime.now(timezone.utc).isoformat(),metric_smoke=metric))

    def run(self,specs):
        if not (self.root/"preflight_DONE.json").exists():
            self.preflight(specs)
        for spec in specs:
            # Conservative rolling estimate keeps time for whole-arm quality.
            durations=[r["metric_seconds"] for r in self.quality[-10:]]
            estimate=max(900,10*(max(durations) if durations else 60)+180)
            self.budget(estimate)
            self.registration_block(spec)
            self.evaluate_candidate(spec)
        main=build(self.root,self.review)
        if sum(s["complete"] for s in main["summaries"] if s["variant"] in {v["name"] for v in specs}) != len(specs):
            raise RuntimeError("Main cohort incomplete; do not start extensions")
        dump(self.root/"main_DONE.json",dict(utc=datetime.now(timezone.utc).isoformat()))
        if self.args.extensions:
            best=next(s for s in specs if s["name"]==main["fastest_all_gate_candidate"])
            for spec in optional_candidates(best):
                self.budget(max(1800,10*max(r["metric_seconds"] for r in self.quality[-10:])+180))
                if spec["name"] not in {s["name"] for s in specs}:
                    specs.append(spec)
                dump(self.root/"candidate_manifest.json",specs)
                self.registration_block(spec,"smoke",smoke=True)
                self.registration_block(spec)
                self.evaluate_candidate(spec)
        # Fresh-worker timing repeats for the selected winner and reference;
        # metrics are not silently reused for any changed field.
        latest=build(self.root,self.review)
        winner=latest["fastest_all_gate_candidate"]
        chosen=[s for s in specs if s["name"] in {"reference_cpbe",winner}]
        for repeat in range(2):
            self.budget(600)
            for spec in chosen[::1 if repeat%2==0 else -1]:
                self.registration_block(spec,phase=f"confirmation{repeat}")
        self.frozen()
        self.state("completed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("container","repo-container","artifact-host","artifact-container","oasis-root","run-id","deadline"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--gpu",type=int,default=0,choices=[0])
    p.add_argument("--phase",choices=["dry-run","preflight","all","report"],default="all")
    p.add_argument("--extensions",action="store_true")
    a=p.parse_args()
    a.experiment_key=EXPERIMENT
    artifact,repo=Path(a.artifact_host).resolve(),Path.cwd().resolve()
    if (not artifact.is_dir() or not Path(a.artifact_host).is_absolute() or artifact in (Path("/"),Path.home())
            or artifact==repo or repo in artifact.parents or artifact in repo.parents or "/" in a.run_id
            or not a.run_id.startswith("oasis_cpbe_accuracy_speed_")):
        raise ValueError("Invalid external artifact root or run ID")
    suite=OvernightSuite(a)
    suite.provenance()
    manifest=suite.root/"candidate_manifest.json"
    specs=main_candidates()
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous[:len(specs)] != specs:
            raise ValueError("Candidate manifest drift")
    else:
        dump(manifest,specs)
    dump(suite.root/"plan.json",dict(main_candidates=12,main_timing_rows=132,main_quality_rows=120,
        ranks=list(range(10)),deadline=a.deadline,gpu_cutoff_utc=datetime.fromtimestamp(suite.gpu_deadline,timezone.utc).isoformat(),
        new_work_cutoff_utc=datetime.fromtimestamp(suite.launch_deadline,timezone.utc).isoformat(),gates=GATES,
        extensions=a.extensions,weekly_quota_limit="removed by user",confirmation_repeats=2))
    if a.phase=="dry-run":
        print(json.dumps(dict(root=str(suite.root),candidates=len(specs),registration_jobs=132,quality_jobs=120)))
        return
    if a.phase=="report":
        build(suite.root,suite.review)
        return
    suite.monitor.thread.start()
    try:
        if a.phase=="preflight":
            suite.preflight(specs)
        else:
            suite.run(specs)
    except BudgetEnd as exc:
        suite.state("deadline_limited",reason=str(exc))
    except Exception as exc:
        suite.state("error",error=repr(exc))
        raise
    finally:
        build(suite.root,suite.review)
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=5)
        dump(suite.root/"controller_EXIT.json",dict(utc=datetime.now(timezone.utc).isoformat()))


if __name__=="__main__":
    main()
