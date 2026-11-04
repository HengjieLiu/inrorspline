#!/usr/bin/env python
"""Independent eight-block checkpoint, metric and topology replay for P68."""
import argparse
import csv
import math
import os
from pathlib import Path
import numpy as np
import torch
import run_oasis_bounded_two_blocks as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    args=p.parse_args()
    if (os.getuid(),os.getgid())!=(1009,1020): raise RuntimeError("Require1009:1020")
    out=args.run_dir/"independent_audit.json"
    if out.exists(): raise RuntimeError("Preserve prior audit")
    manifest=r.c.read_json(args.manifest)
    row=r.c.read_json(args.run_dir/"summary.json")
    r.validate(args.run_dir,manifest,row["case_rank"],row["mode"],row["smoke"])
    if row["gpu"]!=args.gpu: raise RuntimeError("Audit must use original physical GPU")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.set_num_threads(4)
    device=r.c.set_cuda_device(args.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for key in ("oasis_root","pair_file","dl_code_root","vfa_root"):
        setattr(args,key,Path(manifest["settings"][key]))
    args.case_rank=row["case_rank"]
    batch,hashes=r.legacy.load_batch(args,device)
    if hashes!=row["input_hashes"]: raise RuntimeError("Input drift")
    identity=r.c.make_coordinate_grid(r.c.IMAGE_SHAPE,device)
    scale=r.d.voxel_scale(r.c.IMAGE_SHAPE,device=device)
    atomics=[]; checks=[]; previous_field=torch.zeros_like(identity); previous_best=None
    for step,record in enumerate(row["blocks"]):
        stage=r.b.stages(row["smoke"])[step//2]
        frozen=[atomics[i] for i in r.b.frozen_indices(row["mode"],step)]
        obj=r.b.Objective(batch["fixed"],batch["moving"],stage,record["alpha"],frozen,row["bounds"][str(stage.cps)],True)
        initial=r.b.initial_parameter(row["mode"],step,previous_best)
        seed=torch.zeros(*obj.control_shape,3,device=device) if initial is None else initial
        checkpoint=torch.load(record["model_path"],map_location=device,weights_only=False)
        with torch.no_grad():
            init_field=r.b.compose([*frozen,obj.atomic(seed)],identity)-identity
            init_check=r.legacy.verify_init(previous_field,init_field)
            if init_check!=record["initialization_check"]: raise RuntimeError("Initialization check differs")
            initial_values=[float(x) for x in obj(seed)]
            for a,e in zip(initial_values,record["initial_loss_data_reg"]):
                if not math.isclose(a,e,rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Initial objective mismatch")
            controls=obj.controls(checkpoint["latent_norm_xyz"])
            if not torch.equal(controls,checkpoint["controls_norm_xyz"]): raise RuntimeError("Controls mismatch")
            atomic=obj.expand(controls).detach()
            field=r.b.compose([*frozen,atomic],identity)-identity
            restored=torch.from_numpy(np.load(record["dvf_path"])).to(device)
            if not torch.equal(field,restored): raise RuntimeError("Field reconstruction mismatch")
            values=[float(x) for x in obj(checkpoint["latent_norm_xyz"])]
            for value,key in zip(values,("best_loss","best_data_loss","best_reg_loss")):
                if not math.isclose(value,record["optimizer"][key],rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Selected objective mismatch")
            topology=r.b.corner_diagnostics(atomic*scale)
            if topology!=record["atomic_corner_audit"]: raise RuntimeError("Corner replay differs")
            vox=field*scale; acc,labels=r.legacy.canonical_metrics(vox,batch)
            with (Path(record["model_path"]).parent/"case_metrics.csv").open() as stream:
                saved_labels=next(csv.DictReader(stream))
            for label in r.c.OASIS_LABELS:
                for metric in ("dice","hd95","asd_warped_to_fixed","asd_fixed_to_warped","asd_symmetric"):
                    key=f"{metric}_label_{label:02d}"; a,e=labels[key],float(saved_labels[key])
                    if not (math.isnan(a) and math.isnan(e)) and not math.isclose(a,e,rel_tol=0,abs_tol=1e-12):
                        raise RuntimeError("Per-label metric mismatch")
            zyx=vox.cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
            metrics={**acc,**r.c.jacobian_metrics_voxel(zyx,batch["fixed_mask"].cpu().numpy().astype(bool))}
            for key in r.METRICS:
                if not math.isclose(metrics[key],record["metrics"][key],rel_tol=0,abs_tol=1e-12): raise RuntimeError("Metric mismatch")
        previous_best=checkpoint["latent_norm_xyz"].detach().clone()
        previous_field=field.detach(); atomics.append(atomic)
        checks.append({"step":step,"level":step//2,"block":step%2,"exact_field_reconstruction":True,
                       "selected_loss_data_reg":values,"initial_objective":initial_values,
                       "all_label_and_five_metrics_reproduced":True,"corner_audit_reproduced":True})
        del obj,initial,seed,checkpoint,controls,restored,init_field,vox,zyx
        print(f"Audited rank{row['case_rank']} {row['mode']} block{step}",flush=True)
    retained=[atomics[i] for i in row["final_retained_block_indices"]]
    probes=r.b.offgrid_diagnostics(retained,field,batch["fixed_mask"],count=100000,seed=1)
    if probes!=row["offgrid_audit"]: raise RuntimeError("Off-grid replay mismatch")
    r.assert_sources(manifest); r.assert_references(manifest)
    r.atomic_json(out,{"passed":True,"case_rank":row["case_rank"],"mode":row["mode"],"gpu":args.gpu,
                  "checks":checks,"offgrid_reproduced":True,"audit_source_sha256":r.sha(Path(__file__)),
                  "summary_sha256":r.sha(args.run_dir/"summary.json"),"completed_at":r.c.now_text()})


if __name__=="__main__": main()
