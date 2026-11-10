"""GPU-native independent-component LSQR with deadline-safe finite checkpoints."""
from __future__ import annotations
import math
import torch
from common import Deadline

def solve(op,target,initial=None,check=lambda:None,callback=None,iterations=20000):
    """Paige/Saunders recurrences; columns solved together but stopped separately.

    Column scaling is exact full-box feature L2 norm. Zero columns remain zero.
    Best iterate is judged by LSQR residual norm, with explicit residuals checked
    every ten steps and at convergence. No regularizer is introduced.
    """
    channels=target.shape[-1]
    best=target.new_zeros((op.m,channels)) if initial is None else initial.clone()
    meta=dict(converged=False,iterations=0,reason='not_started',components=[])
    try:
        check(); scales=op.norms();scales=torch.where(scales>0,scales,torch.ones_like(scales))
        def A(x):return op.forward(x/scales[:,None])
        def AT(y):return op.adjoint(y)/scales[:,None]
        x=best*scales[:,None]
        residual=target-A(x);best_mse=float(residual.square().sum(-1).mean())
        meta['mse_mm2']=best_mse
        u=residual; axes=(0,1,2)
        beta=u.square().sum(axes).sqrt();bnorm=beta.clone()
        u=u/beta.clamp_min(1e-300);v=AT(u);alpha=v.square().sum(0).sqrt()
        v=v/alpha.clamp_min(1e-300);w=v.clone();phibar=beta.clone();rhobar=alpha.clone()
        anorm=torch.zeros_like(alpha);ddnorm=torch.zeros_like(alpha)
        active=(beta>1e-14)&(alpha>1e-14)
        reasons=['exact_initial' if not bool(a) else 'iteration_limit' for a in active]
        final_checks={}
        for it in range(1,iterations+1):
            check()
            if not bool(active.any()):break
            u=A(v)-u*alpha;beta=u.square().sum(axes).sqrt();u=u/beta.clamp_min(1e-300)
            anorm=(anorm.square()+alpha.square()+beta.square()).sqrt()
            v=AT(u)-v*beta;alpha=v.square().sum(0).sqrt();v=v/alpha.clamp_min(1e-300)
            rho=(rhobar.square()+beta.square()).sqrt().clamp_min(1e-300)
            cs=rhobar/rho;sn=beta/rho;theta=sn*alpha;rhobar=-cs*alpha
            phi=cs*phibar;phibar=sn*phibar;tau=sn*phi
            d=w/rho;ddnorm+=d.square().sum(0)
            x=x+d*(phi*active);w=v-w*(theta/rho)
            rnorm=phibar.abs();arnorm=alpha*tau.abs();acond=anorm*ddnorm.sqrt()
            xnorm=x.square().sum(0).sqrt()
            stop_res=rnorm<=1e-6*bnorm+1e-6*anorm*xnorm
            stop_normal=arnorm<=1e-6*anorm*rnorm.clamp_min(1e-300)
            stop_cond=acond>=1e12
            candidate_stop=active&(stop_res|stop_normal|stop_cond)
            if it%10==0 or bool(candidate_stop.any()) or it==iterations:
                check();candidate=x/scales[:,None]
                if not bool(torch.isfinite(candidate).all()):raise FloatingPointError('Nonfinite LSQR coefficients')
                real=target-op.forward(candidate);mse=float(real.square().sum(-1).mean())
                grad=AT(real);normal=grad.square().sum(0).sqrt()
                true_r=real.square().sum(axes).sqrt()
                verified=(true_r<=1e-6*bnorm+1e-6*anorm*xnorm)|(normal<=1e-6*anorm*true_r.clamp_min(1e-300))
                if mse<=best_mse:
                    best=candidate.clone();best_mse=mse
                for ch in range(channels):
                    if bool(active[ch] and verified[ch]):reasons[ch]='converged';active[ch]=False
                    elif bool(active[ch] and stop_cond[ch]):reasons[ch]='condition_limit';active[ch]=False
                final_checks=dict(residual_norm=true_r.tolist(),normal_residual_norm=normal.tolist(),condition_estimate=acond.tolist())
                meta.update(iterations=it,mse_mm2=best_mse,components=list(reasons),**final_checks)
                if callback:callback(best,dict(meta))
        meta.update(converged=all(r in ('converged','exact_initial') for r in reasons),components=reasons,**final_checks)
        meta['reason']='converged' if meta['converged'] else 'condition_or_iteration_limit'
    except Deadline:
        meta.update(converged=False,reason='time_limit')
    return best,meta
