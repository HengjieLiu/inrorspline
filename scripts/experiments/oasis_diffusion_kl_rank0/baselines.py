"""Read-only gamma0 reuse audit; writes receipts only in the new execution."""
import argparse
from pathlib import Path

import numpy as np

from scripts.experiments.oasis_regularizer_sweep.protocol import PROTOCOL as BASE,read,dump,sha256
from .protocol import BACKENDS

REPO=Path(__file__).resolve().parents[3]


def audit(prior,out,oasis):
    if not read(prior/"ACCEPTED.json")["complete"]: raise RuntimeError("Unaccepted baseline execution")
    sources=read(prior/"manifest.json")["source_hashes"]
    for name,expected in sources.items():
        if sha256(REPO/name)!=expected: raise RuntimeError(f"Baseline numerical source changed: {name}")
    out.mkdir(parents=True,exist_ok=True)
    refs=[]
    for backend in BACKENDS:
        root=prior/"blocks"/"cohort"/f"diffusion_{backend}_w1"/"attempt0"
        for marker in ("register_ACCEPTED.json","eval_ACCEPTED.json","EVALUATED.json"):
            if not read(root/marker)["complete"]: raise RuntimeError("Incomplete reference block")
        block=read(root/"block.json"); cases=block["cases"][:2]
        if [(r["case_rank"],r["condition"]) for r in cases]!=[(0,"cold"),(0,"warm")]: raise RuntimeError("Reference is not rank0 cold/warm")
        rows=[]
        for row in cases:
            if row["protocol"]!=BASE or row["weight"]!=1 or row["arm"]!=f"diffusion_{backend}": raise RuntimeError("Reference protocol differs")
            if (row["fixed_id"],row["moving_id"])!=("0333","0115"): raise RuntimeError("Reference pair differs")
            if row["source_hashes"]!=sources: raise RuntimeError("Reference sources differ")
            for field in ("dvf","model"):
                if sha256(row[field+"_path"])!=row[field+"_sha256"]: raise RuntimeError("Reference field/model checksum differs")
            for filename,expected in row["input_hashes"].items():
                if sha256(oasis/"test"/filename)!=expected: raise RuntimeError("Input data changed")
            field=np.load(row["dvf_path"],mmap_mode="r")
            if field.shape!=(160,224,192,3) or field.dtype!=np.float32 or not np.isfinite(field).all(): raise RuntimeError("Invalid reference field")
            metric_path=Path(row["dvf_path"]).parent/"evaluation.json"
            metrics=read(metric_path)
            if metrics["dvf_sha256"]!=row["dvf_sha256"] or metrics["input_hashes"]!=row["input_hashes"]: raise RuntimeError("Metric receipt mismatch")
            if row["backend"]["cudnn_benchmark"] or not row["backend"]["cudnn_deterministic"]: raise RuntimeError("Backend differs")
            rows.append(dict(case=row,metrics=metrics,metric_path=str(metric_path),metric_sha256=sha256(metric_path)))
        receipt=dict(complete=True,gamma=0.,lambda_diffusion=1.,diffusion_backend=backend,kl_backend="analytic",
                     source_execution=str(prior),source_block=str(root),cases=rows,
                     provenance="reused_accepted_P84_rank0_only; historical timing/memory")
        dump(out/f"{backend}.json",receipt); refs.append(receipt)
    dump(out/"ACCEPTED.json",dict(complete=True,backends=list(BACKENDS),reported_observations=4,source_hashes=sources))
    return refs


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-root",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--oasis-root",type=Path,required=True); a=p.parse_args()
    print([(r["diffusion_backend"],r["cases"][1]["metrics"]["dice"]) for r in audit(a.prior_root,a.out,a.oasis_root)])
