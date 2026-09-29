"""Host stdlib-only completion queue: wait for own controller, run CPU finalizer."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run-host","run-container","review-container","container","repo-container","deadline"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    root=Path(a.run_host)
    deadline=datetime.fromisoformat(a.deadline).timestamp()
    while not (root/"controller_EXIT.json").exists():
        if time.time()>deadline:
            raise TimeoutError("Study controller has not exited by final deadline")
        time.sleep(5)
    argv=["docker","exec","--user","1009:1009","-e","CUDA_VISIBLE_DEVICES=",
          "-e","MPLCONFIGDIR=/tmp/oasis_cpbe_accuracy_mpl","-w",a.repo_container,a.container,
          "python","-B","scripts/reports/oasis_cpbe_overnight_delivery.py",
          "--run-root",a.run_container,"--review-root",a.review_container]
    (root/"delivery_command.json").write_text(json.dumps(argv,indent=2))
    with (root/"delivery.log").open("w") as log:
        done=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
    (root/"delivery_EXIT.json").write_text(json.dumps(dict(returncode=done.returncode,utc=datetime.now(timezone.utc).isoformat()),indent=2))
    raise SystemExit(done.returncode)


if __name__=="__main__":
    main()
