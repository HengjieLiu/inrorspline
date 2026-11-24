"""Opt-in combined objective and component-aware copy of the frozen optimizer."""
from contextlib import contextmanager
import json
import math
import time

import torch

from scripts.experiments.oasis_regularizer_sweep import losses as l
from scripts.experiments.oasis_regularizer_sweep.protocol import dump
from .protocol import PROTOCOL

BASE_OBJECTIVE=l.Objective


class CombinedObjective(BASE_OBJECTIVE):
    def __init__(self,fixed,moving,weight,stage,control_shape,arm,*,gamma):
        if weight!=1 or arm not in ("diffusion_fd","diffusion_analytic"):
            raise ValueError("Require fixed diffusion lambda1 and an explicit diffusion backend")
        if not math.isfinite(gamma) or gamma<0: raise ValueError("Invalid gamma")
        super().__init__(fixed,moving,weight,stage,control_shape,arm)
        self.gamma=float(gamma)
        self.kl_derivatives=(self.derivatives if self.derivatives is not None else
            l.Derivatives(control_shape,self.image_shape,self.eval_shape,stage.cps,fixed.device)) if gamma else None

    def regularizer(self,parameter,q):
        if self.gamma and self.arm=="diffusion_analytic":
            # Reuse analytic first derivatives for both terms, preserving units/reduction.
            dz,dy,dx=[self.kl_derivatives(parameter,o) for o in ((1,0,0),(0,1,0),(0,0,1))]
            diff=(dz.square().mean()+dy.square().mean()+dx.square().mean())/3
            jac=l.determinant(tuple(v[2:-2,2:-2,2:-2] for v in (dx,dy,dz)))
            kl=l.kl_loss(jac)
        else:
            diff=BASE_OBJECTIVE.regularizer(self,parameter,q)
            kl=l.kl_loss(self.kl_derivatives.jacobian(parameter)) if self.gamma else diff.new_zeros(())
        self.diffusion_component=diff; self.kl_component=kl
        return diff+self.gamma*kl if self.gamma else diff


def optimize(initial,objective,stage,out):
    from scripts.experiments.oasis_regularizer_sweep import worker as w
    out.mkdir(parents=True,exist_ok=False)
    parameter=torch.nn.Parameter(initial.detach().clone()); best=parameter.detach().clone()
    optimizer=torch.optim.Adam([parameter],lr=stage.lr,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    rows=[]; best_loss=math.inf; best_epoch=0; reason="max_epochs"
    if parameter.is_cuda: torch.cuda.synchronize()
    start=time.monotonic()
    with (out/"history.jsonl").open("w") as stream:
        for epoch in range(stage.max_epochs+1):
            optimizer.zero_grad(set_to_none=True)
            total,data,combined=objective(parameter)
            values=torch.stack((total,data,combined,objective.diffusion_component,objective.kl_component)).detach().cpu().tolist()
            if not all(math.isfinite(v) for v in values): raise FloatingPointError("Nonfinite combined objective")
            if values[0]<best_loss:
                best_loss=values[0]; best_epoch=epoch; best.copy_(parameter.detach())
            row=dict(epoch=epoch,optimizer_updates_completed=epoch,loss=values[0],data_loss=values[1],reg_loss=values[2],
                weighted_reg_loss=values[2],diffusion_loss=values[3],kl_c1_loss=values[4] if objective.gamma else None,
                kl_c1_evaluated=bool(objective.gamma),lambda_diffusion=1.,gamma=objective.gamma,
                weighted_diffusion_loss=values[3],weighted_kl_c1_loss=objective.gamma*values[4],
                lr=stage.lr,best_epoch=best_epoch,elapsed_sec=time.monotonic()-start)
            if not math.isclose(values[0],values[1]+values[3]+objective.gamma*values[4],rel_tol=1e-6,abs_tol=2e-7):
                raise RuntimeError("Objective component accounting mismatch")
            rows.append(row); stream.write(json.dumps(row)+"\n")
            if epoch%25==0: stream.flush(); print(json.dumps(dict(stage=stage.index,**row)),flush=True)
            if epoch==stage.max_epochs: break
            if w.d.plateau(rows,stage,0): reason="adaptive_plateau"; break
            total.backward()
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError("Invalid combined gradient")
            optimizer.step()
    if parameter.is_cuda: torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    with torch.no_grad():
        selected=[float(v) for v in objective(best)]
        diffusion=float(objective.diffusion_component); kl=float(objective.kl_component)
    if not math.isclose(selected[0],best_loss,rel_tol=1e-6,abs_tol=1e-7): raise RuntimeError("Best objective replay mismatch")
    meta=dict(cps=stage.cps,index=stage.index,factor=stage.factor,lr=stage.lr,max_updates=stage.max_epochs,
        min_updates=stage.min_epochs,requested_patience=stage.requested_patience,effective_patience=stage.patience,
        actual_epochs=rows[-1]["epoch"],best_epoch=best_epoch,best_loss=selected[0],best_data_loss=selected[1],
        best_reg_loss=selected[2],best_diffusion_loss=diffusion,best_kl_c1_loss=kl if objective.gamma else None,
        best_weighted_kl_c1_loss=objective.gamma*kl,lambda_diffusion=1.,gamma=objective.gamma,lambda_kl=objective.gamma,
        kl_backend="analytic",regularizer="diffusion_plus_analytic_kl_c1",stopping_reason=reason,lr_reductions=0,
        optimizer_loop_sec=elapsed,arm=objective.arm,weight=1.,evaluation_shape=list(objective.eval_shape),
        control_shape=list(initial.shape[:3]),voxel_scale_xyz=objective.scale.tolist())
    w.d.atomic_torch_save(dict(parameter_norm_xyz=best.cpu(),summary=meta),out/"best.pt")
    w.d.atomic_torch_save(dict(parameter_norm_xyz=parameter.detach().cpu(),optimizer=optimizer.state_dict(),summary=meta),out/"last.pt")
    w.c.write_csv(out/"history.csv",rows); dump(out/"summary.json",meta)
    return best.detach(),meta


@contextmanager
def activate(worker,gamma):
    """Process-local adapter; frozen P84 sources and historical artifacts untouched."""
    originals=(worker.l.Objective,worker.optimize,worker.register,worker.PROTOCOL)
    original_register=worker.register
    def factory(*args,**kwargs): return CombinedObjective(*args,**kwargs,gamma=gamma)
    def register(job,rank,condition,index,device,backend):
        if rank!=0: raise ValueError("Rank0 only; first10 extension is not authorized")
        result=original_register(job,rank,condition,index,device,backend)
        result.update(gamma=gamma,lambda_diffusion=1.,lambda_kl=gamma,kl_backend="analytic",
                      regularizer="diffusion_plus_analytic_kl_c1")
        from pathlib import Path
        dump(Path(result["dvf_path"]).parent/"result.json",result)
        return result
    worker.l.Objective=factory; worker.optimize=optimize; worker.register=register
    worker.PROTOCOL=dict(PROTOCOL,gamma=gamma)
    try: yield
    finally: worker.l.Objective,worker.optimize,worker.register,worker.PROTOCOL=originals
