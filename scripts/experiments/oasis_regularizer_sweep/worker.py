"""Fresh CUDA worker: one cold/warm block, or separate saved-field evaluation."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from .protocol import PROTOCOL, dump, read, sequence, sha256, stamp


def runtime(gpu):
    global torch,np,l,c,d
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    import numpy as np
    import torch
    from . import losses as l
    c,d=l.c,l.d
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def load_case(job,rank,device):
    import nibabel as nib
    root=Path(job["oasis_root"])
    info=c.resolve_case(rank=rank,pair_file=c.DEFAULT_PAIR_FILE,oasis_root=root)
    batch=dict(case_rank=rank,fixed_id=info.fixed_id,moving_id=info.moving_id,
               original_case_index=info.original_case_index,input_hashes={})
    for role in ("fixed","moving"):
        for prefix in ("img","seg"):
            path=root/"test"/f"{prefix}{batch[role+'_id']}.nii.gz"
            image=nib.load(str(path))
            if image.shape!=(160,192,224) or tuple(image.header.get_zooms()[:3])!=(1.,1.,1.):
                raise ValueError("Unexpected OASIS geometry")
            arr=c.load_nifti_reoriented(path,dtype=np.float32)
            if arr.shape!=c.IMAGE_SHAPE or not np.isfinite(arr).all(): raise ValueError("Invalid input")
            batch[role if prefix=="img" else role+"_seg"]=torch.from_numpy(arr).to(device)
            if role=="fixed" and prefix=="img": batch["fixed_mask"]=torch.from_numpy(arr>0).to(device)
            batch["input_hashes"][path.name]=sha256(path)
    return batch


def memory():
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())


def stage_spec(i,smoke=False):
    return d.StageSpec(i,PROTOCOL["cps"][i],PROTOCOL["factors"][i],PROTOCOL["lr"][i],
        1 if smoke else PROTOCOL["max_updates"][i],1 if smoke else PROTOCOL["min_updates"][i],
        PROTOCOL["patience"][i],0)


def optimize(initial,objective,stage,out):
    out.mkdir(parents=True,exist_ok=False)
    param=torch.nn.Parameter(initial.detach().clone()); best=param.detach().clone()
    optimizer=torch.optim.Adam([param],lr=stage.lr,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    rows=[]; best_loss=math.inf; best_epoch=0; reason="max_epochs"
    torch.cuda.synchronize(); start=time.monotonic()
    with (out/"history.jsonl").open("w") as stream:
        for epoch in range(stage.max_epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss,data,reg=objective(param)
            values=torch.stack((loss,data,reg)).detach().cpu().tolist()
            if not all(math.isfinite(x) for x in values): raise FloatingPointError("Nonfinite objective")
            if values[0]<best_loss:
                best_loss=values[0]; best_epoch=epoch; best.copy_(param.detach())
            row=dict(epoch=epoch,optimizer_updates_completed=epoch,loss=values[0],data_loss=values[1],
                     reg_loss=values[2],weighted_reg_loss=objective.alpha*values[2],lr=stage.lr,
                     best_epoch=best_epoch,elapsed_sec=time.monotonic()-start)
            rows.append(row); stream.write(__import__("json").dumps(row)+"\n")
            if epoch%25==0:
                stream.flush(); print(__import__("json").dumps(dict(stage=stage.index,**row)),flush=True)
            if epoch==stage.max_epochs: break
            if d.plateau(rows,stage,0): reason="adaptive_plateau"; break
            loss.backward()
            if param.grad is None or not bool(torch.isfinite(param.grad).all()): raise FloatingPointError("Invalid gradient")
            optimizer.step()
    torch.cuda.synchronize(); elapsed=time.monotonic()-start
    with torch.no_grad(): selected=[float(v) for v in objective(best)]
    if not math.isclose(selected[0],best_loss,rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Checkpoint loss replay failed")
    meta=dict(cps=stage.cps,index=stage.index,factor=stage.factor,lr=stage.lr,
        max_updates=stage.max_epochs,min_updates=stage.min_epochs,requested_patience=stage.requested_patience,
        effective_patience=stage.patience,actual_epochs=rows[-1]["epoch"],best_epoch=best_epoch,
        best_loss=selected[0],best_data_loss=selected[1],best_reg_loss=selected[2],
        stopping_reason=reason,lr_reductions=0,optimizer_loop_sec=elapsed,
        arm=objective.arm,weight=objective.alpha,evaluation_shape=list(objective.eval_shape),
        control_shape=list(initial.shape[:3]),voxel_scale_xyz=objective.scale.tolist())
    d.atomic_torch_save(dict(parameter_norm_xyz=best.cpu(),summary=meta),out/"best.pt")
    d.atomic_torch_save(dict(parameter_norm_xyz=param.detach().cpu(),optimizer=optimizer.state_dict(),summary=meta),out/"last.pt")
    c.write_csv(out/"history.csv",rows)
    dump(out/"summary.json",meta)
    return best.detach(),meta


def register(job,rank,condition,index,device,backend):
    out=Path(job["block_dir"])/f"{index:02d}_{condition}_rank{rank:03d}"
    out.mkdir(parents=True,exist_ok=False)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    load_start=time.monotonic_ns(); batch=load_case(job,rank,device)
    torch.cuda.synchronize(); start=time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(); baseline=memory(); case_peaks=[baseline]
    shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,8,[8,4,2])
    parameter=torch.zeros((*shapes[8],3),device=device)
    stages=[]
    for i in range(3):
        case_peaks.append(memory()); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); prep=time.monotonic()
        spec=stage_spec(i,job["phase"]=="smoke")
        if i: parameter=c.cubic_bspline_dyadic_knot_insert_3d(parameter,shapes[spec.cps]).detach()
        objective=l.Objective(batch["fixed"],batch["moving"],job["weight"],spec,shapes[spec.cps],job["arm"])
        torch.cuda.synchronize(); prep_sec=time.monotonic()-prep
        parameter,meta=optimize(parameter,objective,spec,out/f"stage{i}")
        with torch.no_grad():
            j=l.fd_jacobian(objective.field(parameter)*objective.scale)
            meta["fd_j_diagnostics"]=dict(min=float(j.min()),fraction_le_zero=float((j<=0).float().mean()),
                fraction_lt_epsilon=float((j<.05).float().mean()),kl_c1=float(l.kl_loss(j)))
        meta.update(preparation_sec=prep_sec,memory=memory())
        case_peaks.append(meta["memory"]); stages.append(meta); dump(out/f"stage{i}"/"summary.json",meta)
        del objective,j
    case_peaks.append(memory()); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); final_start=time.monotonic_ns()
    with torch.no_grad(): field=d.AlignedBSpline(shapes[2],c.IMAGE_SHAPE,c.IMAGE_SHAPE,2,device)(parameter)
    array=field.cpu().numpy().astype(np.float32,copy=False)
    if array.shape!=(*c.IMAGE_SHAPE,3) or not np.isfinite(array).all(): raise FloatingPointError("Invalid final field")
    np.save(out/"disp_norm_xyz.npy",array)
    torch.cuda.synchronize(); ready=time.monotonic_ns()
    case_peaks.append(memory())
    meta=dict(arm=job["arm"],weight=job["weight"],phase=job["phase"],case_rank=rank,condition=condition,index=index,
        fixed_id=batch["fixed_id"],moving_id=batch["moving_id"],input_hashes=batch["input_hashes"],
        direction="moving_to_fixed",sampling="fixed_to_moving_pull",units="normalized_grid_XYZ",
        gpu=job["gpu"],gpu_uuid=job["gpu_uuid"],worker_pid=os.getpid(),backend=backend,protocol=PROTOCOL,
        source_hashes=job["source_hashes"],load_start_ns=load_start,case_load_complete_ns=start,dvf_file_ready_ns=ready,
        case_load_sec=(start-load_start)/1e9,registration_sec=(ready-start)/1e9,
        case_to_dvf_sec=(ready-load_start)/1e9,finalize_dvf_sec=(ready-final_start)/1e9,
        baseline_memory=baseline,peak_allocated=max(x["peak_allocated"] for x in case_peaks),
        peak_reserved=max(x["peak_reserved"] for x in case_peaks),end_memory=memory(),stages=stages,
        dvf_path=str(out/"disp_norm_xyz.npy"),dvf_sha256=sha256(out/"disp_norm_xyz.npy"),
        model_path=str(out/"stage2"/"best.pt"),model_sha256=sha256(out/"stage2"/"best.pt"),completed_at=stamp())
    for loss in ("diffusion","be","kl"):
        meta["lambda_"+loss]=job["weight"] if job["arm"].startswith("kl_c1" if loss=="kl" else loss+"_") else 0.
    dump(out/"result.json",meta)
    (out/"DONE").write_text(stamp()+"\n")
    del parameter,field,array,batch
    gc.collect()
    return meta


def region_stats(j,mask):
    result={}
    for region,values in (("full",j.reshape(-1)),("fg",j[mask])):
        count=values.numel(); positive=values[values>0]
        if not count: result[region]=dict(count=0,undefined="empty_region"); continue
        quantiles=torch.quantile(values,torch.tensor([.01,.5,.99],dtype=values.dtype,device=values.device))
        result[region]=dict(count=count,positive_count=positive.numel(),positive_fraction=positive.numel()/count,
            min=float(values.min()),p01=float(quantiles[0]),median=float(quantiles[1]),p99=float(quantiles[2]),
            mean=float(values.mean()),fraction_le_zero=float((values<=0).double().mean()),
            fraction_lt_epsilon=float((values<.05).double().mean()),kl_c1=float(l.kl_loss(values)),
            std_logj=float(positive.log().std(unbiased=False)) if positive.numel() else None)
    return result


@__import__("contextlib").contextmanager
def no_grad():
    with torch.no_grad(): yield


def evaluate_one(row,job,device):
    started=time.monotonic(); batch=load_case(job,row["case_rank"],device)
    if batch["input_hashes"]!=row["input_hashes"]: raise RuntimeError("Inputs changed")
    out=Path(row["dvf_path"]).parent
    if sha256(row["dvf_path"])!=row["dvf_sha256"] or sha256(row["model_path"])!=row["model_sha256"]:
        raise RuntimeError("Candidate checksum changed")
    replay=[]
    with no_grad():
        for i in range(3):
            saved=torch.load(out/f"stage{i}"/"best.pt",map_location=device,weights_only=False)
            controls=saved["parameter_norm_xyz"]
            objective=l.Objective(batch["fixed"],batch["moving"],row["weight"],stage_spec(i,row["phase"]=="smoke"),controls.shape[:3],row["arm"])
            values=[float(v) for v in objective(controls)]
            for actual,key in zip(values,("best_loss","best_data_loss","best_reg_loss")):
                if not math.isclose(actual,row["stages"][i][key],rel_tol=1e-5,abs_tol=2e-7): raise RuntimeError("Independent loss replay mismatch")
            replay.append(values); del objective
        field=torch.from_numpy(np.load(row["dvf_path"])).to(device)
        reproduced=d.AlignedBSpline(controls.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,2,device)(controls)
        delta=(field-reproduced)*d.voxel_scale(c.IMAGE_SHAPE,device=device)
        error=dict(max_voxel=float(delta.abs().max()),rms_voxel=float(delta.square().mean().sqrt()))
        if error["max_voxel"]>1e-4 or error["rms_voxel"]>1e-5: raise RuntimeError("Independent field replay mismatch")
        del reproduced,delta
        coords=c.make_coordinate_grid(c.IMAGE_SHAPE,device)+field
        metrics=c.evaluate_from_dense_coords(coords,batch)
        accuracy=c.accuracy_row(metrics,batch["fixed_id"],batch["moving_id"])
        labels=[]
        for idx,label in enumerate(c.OASIS_LABELS):
            counts={name:int((seg==label).sum()) for name,seg in (("fixed",batch["fixed_seg"]),("moving",batch["moving_seg"]),("warped",metrics["warped_seg"]))}
            labels.append(dict(label=label,counts=counts,eligible=all(counts.values()),
                exclusion_reason=[k for k,v in counts.items() if not v],
                **{key:float(metrics[key][idx]) for key in ("dice","hd95","asd_symmetric","asd_warped_to_fixed","asd_fixed_to_warped")}))
        for label in labels:
            vals=[label[k] for k in ("dice","hd95","asd_symmetric")]
            if label["eligible"]!=all(math.isfinite(v) for v in vals): raise RuntimeError("Invalid missing-label handling")
        del coords,metrics
        regularity,be,diffusion=c.regularity_metrics(field,batch["fixed_mask"],control_disp_norm=controls,cps=2)
        vox=field*d.voxel_scale(c.IMAGE_SHAPE,device=device)
        fd=l.fd_jacobian(vox.double())
        mask=batch["fixed_mask"][2:-2,2:-2,2:-2]
        fd_stats=region_stats(fd,mask)
        for region,col in (("full","full"),("fg","fg")):
            a=fd_stats[region]["std_logj"]; b=regularity["log_jdet_std_"+col]
            if a is not None and not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10): raise RuntimeError("Canonical SDlogJ mismatch")
        del fd
        operator=l.Derivatives(controls.shape[:3],c.IMAGE_SHAPE,c.IMAGE_SHAPE,2,device,torch.float64)
        analytic_stats=region_stats(operator.jacobian(controls.double()),mask)
        objective=l.Objective(batch["fixed"],batch["moving"],row["weight"],stage_spec(2),controls.shape[:3],row["arm"])
        ncc9=float(objective(controls)[1])
    short=dict(dice=float(accuracy["dice_mean"]),hd95=float(accuracy["hd95_mean"]),asd=float(accuracy["asd_symmetric_mean"]))
    if not all(math.isfinite(v) for v in short.values()): raise RuntimeError("Nonfinite accuracy")
    result=dict(**short,accuracy=accuracy,per_label=labels,regularity=regularity,be_rows=be,diffusion_rows=diffusion,
        fd_jacobian=fd_stats,analytic_jacobian=analytic_stats,ncc9=ncc9,
        replay=dict(losses=replay,field_error=error),dvf_sha256=row["dvf_sha256"],input_hashes=row["input_hashes"],
        metric_seconds=time.monotonic()-started,metric_protocol="canonical_35_labels_raw_fixed_FG_crop2_positiveJ_sd_ddof0")
    dump(out/"evaluation.json",result)
    del batch,controls,field,vox,operator,objective
    gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True); parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args(); job=read(args.job); root=Path(job["block_dir"])
    marker="eval" if args.evaluate else "register"
    try:
        dump(root/f"{marker}_boot.json",dict(container_pid=os.getpid(),boot_ns=time.monotonic_ns()))
        deadline=time.monotonic()+120
        while not (root/f"{marker}_ack.json").exists():
            if time.monotonic()>deadline: raise TimeoutError("Host identity acknowledgement missing")
            time.sleep(.05)
        device=runtime(job["gpu"])
        backend=c.cuda_backend_metadata(device)
        backend.update(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
        ready=time.monotonic_ns()
        dump(root/f"{marker}_ready.json",dict(ready_ns=ready,backend=backend,pid=os.getpid(),uid=os.getuid(),gid=os.getgid()))
        if args.evaluate:
            results=read(root/"block.json")["cases"]
            # Explicit exact-field evaluation reuse within a block, with input identity.
            cache={}
            for row in results:
                key=(row["dvf_sha256"],row["case_rank"])
                if key in cache:
                    original=cache[key]
                    if row["input_hashes"]!=original["input_hashes"]: raise RuntimeError("Invalid metric alias")
                    result=dict(original,alias_of=original["evaluated_path"],metric_seconds=0.)
                    dump(Path(row["dvf_path"]).parent/"evaluation.json",result)
                else:
                    result=evaluate_one(row,job,device); result["evaluated_path"]=row["dvf_path"]
                    cache[key]=result
            dump(root/"EVALUATED.json",dict(complete=True,cases=len(results),completed_at=stamp()))
        else:
            rows=[]
            for index,(rank,condition) in enumerate(sequence(job["phase"])):
                row=register(job,rank,condition,index,device,backend)
                row["worker_startup_sec"]=(ready-job["parent_launch_ns"])/1e9
                if condition=="cold": row["launch_to_dvf_sec"]=(row["dvf_file_ready_ns"]-job["parent_launch_ns"])/1e9
                rows.append(row); dump(root/"block.json",dict(complete=False,cases=rows,job=job))
            dump(root/"block.json",dict(complete=True,cases=rows,job=job))
    except BaseException as error:
        dump(root/f"{marker}_FAILED.json",dict(error=repr(error),traceback=traceback.format_exc(),at=stamp()))
        raise


if __name__=="__main__": main()
