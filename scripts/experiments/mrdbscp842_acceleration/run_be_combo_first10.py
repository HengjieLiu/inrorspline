"""GPU-0 three-arm cold + ten-warm benchmark, separate components and metrics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
from .run_suite import Suite,dump

ARMS = {"reference":(),"cpbe":("cpbe",),"combo_cache_batch_cpbe":("cache","batch","cpbe")}
EXPERIMENT = "2026-09-20_mrdbscp842_cache_batch_cpbe_gpu0"


class ComboSuite(Suite):
    def docker_command(self,module,*extra):
        return ["docker","exec","-u","1009:1009","-w",self.args.repo_container,
                "-e","PYTHONDONTWRITEBYTECODE=1","-e","OMP_NUM_THREADS=6","-e","MKL_NUM_THREADS=6",
                "-e","OPENBLAS_NUM_THREADS=6",self.args.container,"python","-B","-m",module,*extra]

    def helper(self,label,module,*extra):
        argv = self.docker_command(module,*extra)
        dump(self.root/(label+"_command.json"),argv)
        with (self.root/(label+".log")).open("w") as log:
            subprocess.run(argv,check=True,stdout=log,stderr=subprocess.STDOUT)

    def hashes(self):
        paths = list(Path("scripts/experiments/mrdbscp842_acceleration").glob("*.py"))
        paths += list(Path("scripts/experiments/oasis_alpha_sweep").glob("*.py"))
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def frozen(self):
        path = self.root/"formal_source_sha256.json"
        value = self.hashes()
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise RuntimeError("Source changed after formal run started; use a new run ID")
        else:
            dump(path,value)

    def source_checks(self,label):
        self.helper(label,"scripts.experiments.mrdbscp842_acceleration.final_checks",
                    "--output",self.container(self.root/(label+".json")))

    def preflight(self):
        self.source_checks("preflight_source_checks")
        self.monitor.wait_clear()
        environment = self.auxiliary("environment","data_environment")
        dump(self.root/"data_environment.json",environment)
        for detail in ("endpoints","components"):
            for arm,features in ARMS.items():
                self.block(f"smoke_{detail}_{arm}",arm,[dict(rank=0,smoke=True,detail=detail)],
                           warmup=False,features=features)
        self.helper("smoke_validation","scripts.experiments.mrdbscp842_acceleration.be_combo_report",
                    "--root",self.container(self.root),"--review",self.container(self.review),"--smoke")
        dump(self.root/"preflight_DONE.json",dict(utc=datetime.now(timezone.utc).isoformat()))

    def registrations(self):
        for phase,detail in (("headline","endpoints"),("diagnostic","components")):
            for arm,features in ARMS.items():
                self.frozen()
                jobs = [dict(rank=r,detail=detail) for r in (0,0,1,2,3,4,5,6,7,8,9)]
                self.block(f"{phase}_{arm}",arm,jobs,warmup=False,features=features)
            dump(self.root/(phase+"_DONE.json"),dict(utc=datetime.now(timezone.utc).isoformat()))

    def evaluate(self):
        items = [dict(id=r["id"],rank=r["rank"],dvf=r["dvf"],hd95=True,regularity=True)
                 for r in self.rows if r["block"].startswith("headline_")]
        if len(items) != 33:
            raise RuntimeError("Expected 33 headline outputs before evaluation")
        path = self.root/"evaluation.json"
        if not path.exists():
            result = self.auxiliary("evaluate","evaluation",items=items,
                                    command_timeout=getattr(self.args,"evaluation_timeout_seconds",14400))
            dump(path,result)

    def report(self):
        self.helper("report","scripts.experiments.mrdbscp842_acceleration.be_combo_report",
                    "--root",self.container(self.root),"--review",self.container(self.review))


def main():
    p = argparse.ArgumentParser()
    for name in ("container","repo-container","artifact-host","artifact-container","oasis-root","run-id"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--gpu",type=int,choices=[0],default=0)
    p.add_argument("--phase",choices=["dry-run","preflight","all","report"],default="all")
    p.add_argument("--evaluation-timeout-seconds",type=int,default=14400)
    args = p.parse_args()
    if args.evaluation_timeout_seconds <= 0:
        p.error("--evaluation-timeout-seconds must be positive")
    args.experiment_key = EXPERIMENT
    artifact,repo = Path(args.artifact_host).resolve(),Path.cwd().resolve()
    if (not Path(args.artifact_host).is_absolute() or not artifact.is_dir()
            or artifact == repo or repo in artifact.parents or artifact in repo.parents
            or artifact in (Path("/"),Path.home()) or "/" in args.run_id
            or not args.run_id.startswith("mrdbscp842_be_combo_")):
        raise ValueError("Use a new dedicated external run root and expected run-ID prefix")
    suite = ComboSuite(args)
    suite.provenance()
    jobs = [dict(phase=phase,variant=arm,features=list(features),index=i,rank=r,
                 condition="cold" if i == 0 else "warm",detail=detail)
            for phase,detail in (("headline","endpoints"),("diagnostic","components"))
            for arm,features in ARMS.items() for i,r in enumerate((0,0,1,2,3,4,5,6,7,8,9))]
    dump(suite.root/"job_manifest.json",dict(jobs=jobs,expected_headline=33,expected_diagnostic=33,
          expected_evaluations=33,protocol="seed3 OASIS first10 MR_A_fast_decay alpha1000 seed1 F/T GPU0",
          BE_warning="Prior FP32/Adam equivalence failure retained; empirical exploratory comparison"))
    if args.phase == "dry-run":
        print(json.dumps(dict(jobs=len(jobs),root=str(suite.root))),flush=True)
        return
    suite.monitor.thread.start()
    try:
        if args.phase in ("all","preflight") and not (suite.root/"preflight_DONE.json").exists():
            suite.preflight()
        if args.phase == "all":
            suite.registrations()
            suite.evaluate()
            suite.frozen()
            suite.source_checks("final_source_checks")
            suite.report()
        elif args.phase == "report":
            suite.report()
        dump(suite.root/(args.phase+"_DONE.json"),dict(utc=datetime.now(timezone.utc).isoformat()))
    except Exception as exc:
        dump(suite.root/(args.phase+"_ERROR.json"),dict(error=repr(exc)))
        raise
    finally:
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=35)


if __name__ == "__main__":
    main()
