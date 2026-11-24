"""Differentiable KL continuation and continuous, endpoint-aligned derivatives."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

LEGACY=Path(__file__).resolve().parents[1]/"oasis_alpha_sweep"
if str(LEGACY) not in sys.path: sys.path.insert(0,str(LEGACY))
import common as c
import dl_ncc_diffusion as d


def kl_density(j, epsilon=.05, kappa=120.):
    if not 0<epsilon<1 or not math.isfinite(kappa) or kappa<=0: raise ValueError("Invalid KL parameters")
    if not j.is_floating_point(): raise TypeError("Floating determinant required")
    j=j.float() if j.dtype in (torch.float16,torch.bfloat16) else j
    safe=j.clamp_min(epsilon)
    raw=(safe-1)*safe.log()
    delta=j-epsilon
    extended=(epsilon-1)*math.log(epsilon)+(1+math.log(epsilon)-1/epsilon)*delta+.5*kappa*delta.square()
    return torch.where(j>=epsilon,raw,extended)


def kl_loss(j, epsilon=.05, kappa=120., mask=None):
    density=kl_density(j,epsilon,kappa)
    if mask is None: return density.mean()
    if mask.shape!=j.shape: raise ValueError("Mask must exactly match determinant shape")
    weights=mask.to(device=j.device,dtype=density.dtype)
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()) or not bool(weights.sum()>0):
        raise ValueError("Mask weights must be finite, nonnegative, with positive sum")
    return (density*weights).sum()/weights.sum()


def determinant(columns):
    # columns are derivatives w.r.t. x,y,z; components are x,y,z.
    a,b,e=columns
    a0,a1,a2=a.unbind(-1); b0,b1,b2=b.unbind(-1); e0,e1,e2=e.unbind(-1)
    return (1+a0)*((1+b1)*(1+e2)-e1*b2)-b0*(a1*(1+e2)-e1*a2)+e0*(a1*b2-(1+b1)*a2)


def fd_jacobian(vox):
    if vox.ndim!=4 or vox.shape[-1]!=3 or min(vox.shape[:3])<5:
        raise ValueError("Need [D,H,W,3] with spatial dimensions>=5")
    dx=.5*(vox[2:-2,2:-2,3:-1]-vox[2:-2,2:-2,1:-3])
    dy=.5*(vox[2:-2,3:-1,2:-2]-vox[2:-2,1:-3,2:-2])
    dz=.5*(vox[3:-1,2:-2,2:-2]-vox[1:-3,2:-2,2:-2])
    return determinant((dx,dy,dz))


def basis(t, order):
    a=t.abs()
    if order==0: return d.cubic_basis(t)
    if order==1:
        return torch.where(a<1,(1.5*a-2)*t,
            torch.where(a<2,-.5*(2-a).square()*torch.sign(t),torch.zeros_like(t)))
    if order==2:
        return torch.where(a<1,3*a-2,torch.where(a<2,2-a,torch.zeros_like(t)))
    raise ValueError("Cubic derivatives only support orders0,1,2")


class Derivatives:
    """Normalized controls -> derivatives of voxel displacement / stage voxels.

    Derivative order tuple is spatial ZYX; output components remain XYZ.
    No dense finite-differencing or control-grid resizing in this operator.
    """
    def __init__(self,control_shape,native_shape,eval_shape,cps,device,dtype=torch.float32):
        self.control_shape=tuple(control_shape); self.native=tuple(native_shape)
        self.shape=tuple(eval_shape); self.cps=cps; self.dtype=dtype; self.device=device
        self.full=self.native==self.shape
        self.scale=d.voxel_scale(self.shape,device=device,dtype=dtype)
        self.stencils=[]
        for ncp,native,nout in zip(self.control_shape,self.native,self.shape):
            loc=torch.linspace(0.,float(native-1),nout,device=device,dtype=dtype)/cps+1
            left=loc.floor().long()-1
            ids=left[:,None]+torch.arange(4,device=device)[None]
            t=loc[:,None]-ids
            valid=(ids>=0)&(ids<ncp)
            weights=[basis(t,o)*valid*((native-1)/(nout-1)/cps)**o for o in range(3)]
            self.stencils.append((ids.clamp(0,ncp-1),weights))
        self.kernels={o:c.cubic_bspline1d(cps,derivative=o,dtype=dtype,device=device)/cps**o for o in range(3)} if self.full else {}

    def __call__(self, controls, orders):
        grid=controls.reshape(*self.control_shape,3)
        if self.full:
            field=grid.permute(3,0,1,2)[None]
            for axis,order in enumerate(orders):
                kernel=self.kernels[order]
                field=c.conv1d(field,kernel=kernel,dim=axis+2,stride=self.cps,
                               padding=(kernel.numel()-1)//2,transpose=True)
            k=self.cps; z,y,x=self.shape
            out=field[0,:,k:k+z,k:k+y,k:k+x].permute(1,2,3,0).contiguous()
        else:
            out=grid
            for axis,(ids,weights) in enumerate(self.stencils):
                moved=out.movedim(axis,0); result=None
                for j in range(4):
                    value=moved.index_select(0,ids[:,j])*weights[orders[axis]][:,j].reshape(-1,1,1,1)
                    result=value if result is None else result+value
                out=result.movedim(0,axis)
        return out*self.scale

    def jacobian(self,controls):
        return determinant(tuple(self(controls,o)[2:-2,2:-2,2:-2] for o in ((0,0,1),(0,1,0),(1,0,0))))

    def energy(self,controls,loss):
        if loss=="diffusion":
            return sum(self(controls,o).square().mean() for o in ((1,0,0),(0,1,0),(0,0,1)))/3
        if loss=="be":
            terms=(((2,0,0),1),((0,2,0),1),((0,0,2),1),((1,1,0),2),((1,0,1),2),((0,1,1),2))
            return sum(w*self(controls,o).square().mean() for o,w in terms)
        raise ValueError(loss)


class Objective(d.StageObjective):
    def __init__(self,fixed,moving,weight,stage,control_shape,arm):
        super().__init__(fixed,moving,weight,stage,control_shape)
        self.arm=arm
        self.derivatives=Derivatives(control_shape,self.image_shape,self.eval_shape,stage.cps,fixed.device) if arm.endswith("analytic") and weight else None

    def regularizer(self,parameter,q):
        if self.alpha==0: return q.new_zeros(())
        vox=q*self.scale
        if self.arm=="diffusion_fd": return c.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None)
        if self.arm=="be_fd": return c.forward_valid_be(vox.permute(3,0,1,2)[None],None)
        if self.arm=="kl_c1_fd": return kl_loss(fd_jacobian(vox))
        if self.arm=="kl_c1_analytic": return kl_loss(self.derivatives.jacobian(parameter))
        return self.derivatives.energy(parameter,self.arm.removesuffix("_analytic"))

    def __call__(self,parameter):
        q=self.field(parameter)
        warped=torch.nn.functional.grid_sample(self.moving[None,None],(self.identity+q)[None],
            mode="bilinear",padding_mode="border",align_corners=True)
        data=self.ncc(warped); reg=self.regularizer(parameter,q)
        return data+self.alpha*reg,data,reg
