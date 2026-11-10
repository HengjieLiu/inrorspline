"""Gated sequential launch of the four frozen single-case endpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .data import atomic_json, external, sha256
from .run import source_receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    parser.add_argument("--smokes",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    keys=[f"{variant}_{mode}" for mode in ("supervised","unsupervised")
          for variant in ("released_control","paper_reconstruction")]
    preflight=json.loads((args.references/"preflight.json").read_text())
    if preflight["status"]!="passed":
        raise ValueError("Reference preflight has not passed")
    gates={}
    for key in keys:
        summary=json.loads((args.smokes/key/"summary.json").read_text())
        metrics=json.loads((args.smokes/key/"evaluation/metrics.json").read_text())
        if not summary["smoke"] or summary["outcome"]!="completed" or summary["checkpoint_replay_max_error"]>1e-7:
            raise ValueError(f"Smoke gate failed: {key}")
        if metrics["eligible_labels"]!=35:
            raise ValueError(f"Unexpected label exclusion in smoke {key}")
        gates[key]=dict(summary_sha256=sha256(args.smokes/key/"summary.json"),
                       metrics_sha256=sha256(args.smokes/key/"evaluation/metrics.json"))
    out=external(args.out)
    out.mkdir(parents=True,exist_ok=False)
    sources=source_receipts()
    atomic_json(out/"execution_manifest.json",dict(config=json.loads(args.config.read_text()),
                command=sys.argv,sources=sources,smoke_gates=gates,
                preflight_sha256=sha256(args.references/"preflight.json"),
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                execution="four sequential independent jobs on CUDA_VISIBLE_DEVICES; other host jobs may remain active"))
    for key in keys:
        if source_receipts()!=sources:
            raise ValueError("Source drift during pilot queue")
        free,_=torch.cuda.mem_get_info()
        if free<8*2**30:
            raise RuntimeError(f"Require 8GiB free before launching {key}; no other job will be stopped")
        variant,mode=key.rsplit("_",1)
        print(f"Launching {key}",flush=True)
        command=[sys.executable,"-B","-m","scripts.experiments.gaussiandir.run",
                 "--config",str(args.config),"--variant",variant,"--mode",mode,"--out",str(out/key)]
        with (out/f"{key}.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/key/"summary.json").read_text())
        print(json.dumps(dict(endpoint=key,outcome=result["outcome"],steps=result["total_steps"],
                              active_seconds=result["active_seconds"])),flush=True)
    atomic_json(out/"queue_complete.json",dict(status="completed",endpoints=keys,
                completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))


if __name__=="__main__":
    main()
