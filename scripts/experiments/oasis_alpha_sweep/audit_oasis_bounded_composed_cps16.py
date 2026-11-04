#!/usr/bin/env python
"""Read-only reconstruction of accepted P65 artifacts; write a separate audit."""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch

import run_oasis_bounded_composed_cps16 as r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path,required=True)
    args=parser.parse_args()
    if (os.getuid(),os.getgid())!=(1009,1020):
        raise RuntimeError("Require UID:GID1009:1020")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    torch.set_num_threads(4)
    device=r.c.set_cuda_device("3",cudnn_benchmark=False,cudnn_deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=args.artifact_root/"runs"/r.VARIANT
    output=root/"independent_audit.json"
    if output.exists(): raise RuntimeError("Preserve existing audit")
    manifest=r.c.read_json(root/"launch_manifest.json")
    accepted=r.unbounded.previous.accepted_data(args.artifact_root/"repo_experiments"/r.VARIANT)
    row=r.validate(root/r.run_name(),manifest)
    if accepted["candidate"]!=row: raise RuntimeError("Report candidate mismatch")
    for key in ("oasis_root","pair_file","dl_code_root","vfa_root"):
        setattr(args,key,Path(manifest["settings"][key]))
    args.case_rank=0
    batch,hashes=r.legacy.load_batch(args,device)
    if hashes!=row["input_hashes"] or r.sha(args.pair_file)!=row["pair_file_sha256"]:
        raise RuntimeError("Inputs changed")
    identity=r.c.make_coordinate_grid(r.c.IMAGE_SHAPE,device)
    scale=r.d.voxel_scale(r.c.IMAGE_SHAPE,device=device)
    fields=[]; checks=[]
    for record,stage,alpha in zip(row["stages"],r.b.stages(),r.b.ALPHAS):
        checkpoint=torch.load(record["model_path"],map_location=device,weights_only=False)
        obj=r.b.Objective(batch["fixed"],batch["moving"],stage,alpha,fields,row["bounds"][str(stage.cps)])
        with torch.no_grad():
            latent=checkpoint["latent_norm_xyz"]
            controls=r.b.capped_controls(latent,obj.cap_norm)
            if not torch.equal(controls,checkpoint["controls_norm_xyz"]):
                raise RuntimeError("Control reconstruction differs")
            values=[float(x) for x in obj(latent)]
            for value,key in zip(values,("best_loss","best_data_loss","best_reg_loss")):
                if not math.isclose(value,record["optimizer"][key],rel_tol=1e-6,abs_tol=1e-7):
                    raise RuntimeError("Checkpoint objective differs")
            atomic=obj.expand(controls)
            field=r.b.compose([*fields,atomic],identity)-identity
            saved=torch.from_numpy(np.load(record["dvf_path"])).to(device)
            if not torch.equal(field,saved): raise RuntimeError("Field reconstruction differs")
            topology=r.b.corner_diagnostics(atomic*scale)
            if topology!=record["atomic_corner_audit"]: raise RuntimeError("Topology audit differs")
            acc,labels=r.legacy.canonical_metrics(field*scale,batch)
            with (Path(record["model_path"]).parent/"case_metrics.csv").open() as stream:
                saved_labels=next(csv.DictReader(stream))
            valid=0
            for label in r.c.OASIS_LABELS:
                for metric in ("dice","hd95","asd_warped_to_fixed","asd_fixed_to_warped","asd_symmetric"):
                    key=f"{metric}_label_{label:02d}"
                    a,e=labels[key],float(saved_labels[key])
                    if not (math.isnan(a) and math.isnan(e)) and not math.isclose(a,e,rel_tol=0,abs_tol=1e-12):
                        raise RuntimeError(f"Label metric differs: {key}")
                valid+=int(math.isfinite(labels[f"dice_label_{label:02d}"]))
            if valid!=record["metrics"]["valid_label_count"]: raise RuntimeError("Absent-label mismatch")
            zyx=(field*scale).cpu().numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
            reg=r.c.jacobian_metrics_voxel(zyx,batch["fixed_mask"].cpu().numpy().astype(bool))
            metrics={**acc,**reg}
            for key in r.METRICS:
                if not math.isclose(metrics[key],record["metrics"][key],rel_tol=0,abs_tol=1e-12):
                    raise RuntimeError(f"Stage metric differs: {key}")
        fields.append(atomic.detach())
        checks.append({"cps":stage.cps,"exact_field_reconstruction":True,
                       "loss_data_reg":values,"valid_labels":valid,"metrics_recomputed":True,
                       "corner_audit_reproduced":True})
        del obj,latent,checkpoint,controls,saved
        print(f"Audited cps{stage.cps}",flush=True)
    probes=r.b.offgrid_diagnostics(fields,field,batch["fixed_mask"],count=100000,seed=1)
    if probes!=row["offgrid_audit"]: raise RuntimeError("Off-grid replay differs")
    if probes["points_sha256"]!=manifest["bounded_reference"]["offgrid_audit"]["points_sha256"]:
        raise RuntimeError("Old/new probes differ")
    # Independent FP64 central differences check coordinate autograd on a
    # small set of interior raw-positive foreground points, with native fields.
    locations=torch.nonzero(batch["fixed_mask"].bool(),as_tuple=False).cpu().numpy()
    shape=np.asarray(r.c.IMAGE_SHAPE)
    locations=locations[((locations>2)&(locations<shape-3)).all(axis=1)]
    rng=np.random.default_rng(1)
    xyz=locations[rng.choice(len(locations),128,replace=False)][:,::-1].copy()
    xyz=xyz+rng.uniform(-.4,.4,(128,3))
    points=torch.tensor(xyz,device=device,dtype=torch.float64)/scale.double()-1
    points.requires_grad_(True)
    parts=[x.double() for x in fields]
    mapped=r.b.compose(parts,points)
    jac=torch.stack([torch.autograd.grad(mapped[:,j].sum(),points,retain_graph=j<2)[0]
                     for j in range(3)],dim=-2)
    finite=[]; eps=1e-7
    with torch.no_grad():
        for axis in range(3):
            delta=torch.zeros_like(points); delta[:,axis]=eps
            finite.append((r.b.compose(parts,points+delta)-r.b.compose(parts,points-delta))/(2*eps))
    error=float((jac-torch.stack(finite,dim=-1)).abs().max())
    if error>1e-5: raise RuntimeError(f"Off-grid finite-difference gradient mismatch: {error}")
    r.assert_sources(manifest)
    r.unbounded.previous.accepted_data(args.artifact_root/"repo_experiments"/r.VARIANT)
    result={"passed":True,"completed_at":r.c.now_text(),"source_sha256":r.sha(Path(__file__)),
            "stage_checks":checks,"offgrid_exact_replay":True,"same_points_as_P64":True,
            "coordinate_finite_difference_points":128,"coordinate_finite_difference_step":eps,
            "coordinate_finite_difference_max_error":error,
            "source_reference_and_report_hashes_unchanged":True}
    r.atomic_json(output,result)
    print(result,flush=True)


if __name__=="__main__": main()
