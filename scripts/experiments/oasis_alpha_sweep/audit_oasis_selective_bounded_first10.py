#!/usr/bin/env python
"""Independent checkpoint reconstruction and final-metric replay for P67."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_oasis_selective_bounded_first10 as r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path,required=True)
    parser.add_argument("--gpu",choices=("0","1","2","3"))
    args=parser.parse_args()
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    root=args.artifact_root/"runs"/r.VARIANT
    manifest=r.c.read_json(root/"launch_manifest.json")
    r.assert_sources(manifest); r.assert_references(manifest)
    data=r.reference.accepted_data(args.artifact_root/"repo_experiments"/r.VARIANT)
    if args.gpu is None:
        if (root/"independent_audit.json").exists(): raise RuntimeError("Preserve existing audit")
        gpus=sorted({x["gpu"] for x in data["candidates"]})
        def run(gpu):
            target=root/f"independent_audit_gpu{gpu}.json"
            if target.exists(): return r.c.read_json(target)
            state=next(x for x in r.legacy.gpu_snapshot()["rows"] if x["physical_gpu"]==gpu)
            if state["memory_free_mib"]<8192: raise RuntimeError("Insufficient audit memory")
            cmd=[sys.executable,"-u","-B",str(Path(__file__)),"--artifact-root",str(args.artifact_root),"--gpu",gpu]
            with (root/f"independent_audit_gpu{gpu}.log").open("x") as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"CUDA_DEVICE_ORDER":"PCI_BUS_ID"})
            return r.c.read_json(target)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool: results=list(pool.map(run,gpus))
        checked=[x for result in results for x in result["checks"]]
        if len(checked)!=40 or {(x["case_rank"],x["bounded_last"]) for x in checked}!={(rank,n) for rank in range(10) for n in range(1,5)}:
            raise RuntimeError("Incomplete independent audit")
        r.assert_sources(manifest); r.assert_references(manifest)
        r.atomic_json(root/"independent_audit.json",{"passed":True,"completed_at":r.c.now_text(),
                      "cases":40,"stage_checkpoint_reconstructions":160,"audit_source_sha256":r.sha(Path(__file__)),
                      "checks":checked,"report_acceptance_sha256":r.sha(args.artifact_root/"repo_experiments"/r.VARIANT/"ACCEPTED.json")})
        print("Independent40-field/160-checkpoint audit passed",flush=True)
        return
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.set_num_threads(4)
    device=r.c.set_cuda_device(args.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for key in ("oasis_root","pair_file","dl_code_root","vfa_root"):
        setattr(args,key,Path(manifest["settings"][key]))
    checks=[]; previous_rank=None
    for row in sorted((x for x in data["candidates"] if x["gpu"]==args.gpu),key=lambda x:(x["case_rank"],x["bounded_last"])):
        rank=row["case_rank"]
        if previous_rank!=rank:
            args.case_rank=rank; batch,hashes=r.legacy.load_batch(args,device); previous_rank=rank
        if hashes!=row["input_hashes"]: raise RuntimeError("Input changed")
        identity=r.c.make_coordinate_grid(r.c.IMAGE_SHAPE,device)
        scale=r.d.voxel_scale(r.c.IMAGE_SHAPE,device=device)
        fields=[]; stage_checks=[]
        for record,stage,alpha,bounded in zip(row["stages"],r.b.stages(),r.b.ALPHAS,r.stage_mask(row["bounded_last"])):
            if r.sha(Path(record["model_path"]))!=record["model_sha256"] or r.sha(Path(record["dvf_path"]))!=record["dvf_sha256"]:
                raise RuntimeError("Checkpoint/field changed")
            checkpoint=torch.load(record["model_path"],map_location=device,weights_only=False)
            obj=r.b.Objective(batch["fixed"],batch["moving"],stage,alpha,fields,row["bounds"][str(stage.cps)],bounded)
            with torch.no_grad():
                controls=obj.controls(checkpoint["latent_norm_xyz"])
                if not torch.equal(controls,checkpoint["controls_norm_xyz"]): raise RuntimeError("Controls mismatch")
                values=[float(v) for v in obj(checkpoint["latent_norm_xyz"])]
                for value,key in zip(values,("best_loss","best_data_loss","best_reg_loss")):
                    if not math.isclose(value,record["optimizer"][key],rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Loss mismatch")
                atomic=obj.expand(controls).detach()
                field=r.b.compose([*fields,atomic],identity)-identity
                saved=torch.from_numpy(np.load(record["dvf_path"])).to(device)
                if not torch.equal(field,saved): raise RuntimeError("Field reconstruction mismatch")
            fields.append(atomic)
            stage_checks.append({"cps":stage.cps,"exact_reconstruction":True,"loss_data_reg":values})
            del checkpoint,obj,controls,saved
        with torch.no_grad():
            vox=field*scale
            accuracy,labels=r.legacy.canonical_metrics(vox,batch)
            with (Path(row["stages"][-1]["model_path"]).parent/"case_metrics.csv").open() as stream:
                saved_labels=next(csv.DictReader(stream))
            for label in r.c.OASIS_LABELS:
                for metric in ("dice","hd95","asd_warped_to_fixed","asd_fixed_to_warped","asd_symmetric"):
                    key=f"{metric}_label_{label:02d}"; a,e=labels[key],float(saved_labels[key])
                    if not (math.isnan(a) and math.isnan(e)) and not math.isclose(a,e,rel_tol=0,abs_tol=1e-12):
                        raise RuntimeError("Per-label metric mismatch")
            zyx=vox.cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
            metrics={**accuracy,**r.c.jacobian_metrics_voxel(zyx,batch["fixed_mask"].cpu().numpy().astype(bool))}
            for key in r.METRICS:
                if not math.isclose(metrics[key],row[key],rel_tol=0,abs_tol=1e-12): raise RuntimeError("Metric mismatch")
        checks.append({"case_rank":rank,"bounded_last":row["bounded_last"],"gpu":args.gpu,
                       "stage_checks":stage_checks,"final_metrics_reproduced":True,
                       "all_label_metrics_and_absence_reproduced":True})
        print(f"Audited rank{rank} last{row['bounded_last']}",flush=True)
        del fields,field,vox,atomic,identity
    r.atomic_json(root/f"independent_audit_gpu{args.gpu}.json",{"passed":True,"checks":checks,
                  "audit_source_sha256":r.sha(Path(__file__)),"completed_at":r.c.now_text()})


if __name__=="__main__": main()
