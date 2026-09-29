"""Sequential completion queue; wait for an exact existing controller to exit.

No registration or evaluation overlaps the predecessor. All phases retain their
own provenance. A failed phase stops the queue; it never relaxes a gate or deletes
an attempt. The optional wait process must belong to this exact run ID.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from .run_suite import dump,process_identity


def main():
    p = argparse.ArgumentParser()
    for flag in ("container","repo-container","artifact-host","artifact-container","oasis-root","run-id"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--gpu",type=int,choices=[0],default=0)
    p.add_argument("--after-pid",type=int)
    a = p.parse_args()
    root = Path(a.artifact_host)/"runs"/a.run_id
    if not (root/"primary_rows.json").exists():
        raise RuntimeError("Completion queue requires an existing study")
    review = Path(a.artifact_host)/"repo_experiments"/"2026-09-19_mrdbscp842_acceleration_gpu0"/a.run_id
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(),complete=False,phases=[])
    status_path = root/"completion_queue.json"
    if status_path.exists():
        raise RuntimeError("A completion queue record already exists; inspect it before resuming")
    if a.after_pid:
        previous = process_identity(a.after_pid)
        if a.run_id not in " ".join(previous["argv"]) or "mrdbscp842_acceleration.run_suite" not in " ".join(previous["argv"]):
            raise RuntimeError("The wait PID is not a controller for this run")
        state["predecessor"] = previous
        state["status"] = "waiting_for_predecessor"
        dump(status_path,state)
        print(json.dumps(dict(waiting_for_pid=a.after_pid,start_ticks=previous["start_ticks"])),flush=True)
        while True:
            try:
                if process_identity(a.after_pid)["start_ticks"] != previous["start_ticks"]:
                    break
            except (OSError,IndexError):
                break
            time.sleep(10)
    common = []
    for name in ("container","repo_container","artifact_host","artifact_container","oasis_root","run_id","gpu"):
        common += ["--"+name.replace("_","-"),str(getattr(a,name))]
    docker = ["docker","exec","-u","1009:1009","-w",a.repo_container,
              "-e","PYTHONDONTWRITEBYTECODE=1","-e","OMP_NUM_THREADS=6",
              "-e","MKL_NUM_THREADS=6","-e","OPENBLAS_NUM_THREADS=6",a.container,"python","-B"]
    def container_path(path):
        return str(Path(a.artifact_container)/path.relative_to(Path(a.artifact_host)))
    def run(label,argv):
        phase = dict(name=label,argv=argv,started_utc=datetime.now(timezone.utc).isoformat())
        state.update(status="running",active_phase=label)
        state["phases"].append(phase)
        dump(status_path,state)
        with (root/("completion_"+label+".log")).open("x") as log:
            process = subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line,end="",flush=True)
            phase["returncode"] = process.wait()
        phase["finished_utc"] = datetime.now(timezone.utc).isoformat()
        dump(status_path,state)
        if phase["returncode"] != 0:
            raise RuntimeError(f"{label} failed; inspect its retained log")
    try:
        if not (root/"pilot_DONE.json").exists():
            raise RuntimeError("Predecessor exited before pilots completed")
        # Rebuild evidence before selection; the predecessor already evaluates.
        # This also computes deferred CPU-only field localization without overlap.
        for phase in ("report","select","snapchecks","bridge","confirm","evaluate"):
            run(phase,[sys.executable,"-B","-m","scripts.experiments.mrdbscp842_acceleration.run_suite",
                       *common,"--phase",phase])
        run("final_checks",docker+["-m","scripts.experiments.mrdbscp842_acceleration.final_checks",
                                    "--output",container_path(root/"final_source_checks.json")])
        run("validate",docker+["-m","scripts.experiments.mrdbscp842_acceleration.validate",
                               "--root",container_path(root),"--review",container_path(review),"--scan-fields"])
        run("final_report",[sys.executable,"-B","-m","scripts.experiments.mrdbscp842_acceleration.run_suite",
                            *common,"--phase","report"])
        state.update(complete=True,status="complete",finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        state.update(status="failed",error=repr(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        dump(status_path,state)
        raise
    dump(status_path,state)


if __name__ == "__main__":
    main()
