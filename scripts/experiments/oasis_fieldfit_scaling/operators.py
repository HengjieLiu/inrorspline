"""Exact tensor-product and blocked separable operators; no tail truncation."""
from __future__ import annotations
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]/'oasis_polyaffine_fieldfit'))
from basis import _bspline_1d

def lattice_shape(cap, lengths, minimum):
    """Same P81 selection/tie rules, O(axis0*axis1) memory-free enumeration."""
    if cap < minimum**3: raise ValueError('Insufficient lattice budget')
    scale = (math.prod(lengths)/cap)**(1/3)
    upper = [max(minimum+2, math.ceil(2*l/scale)+4) for l in lengths]
    good = None; fallback = None
    # For each first two axes, largest feasible third axis satisfying the
    # isotropy window is optimal in occupied budget. Otherwise inspect all
    # third-axis possibilities for the exact fallback (only small budgets).
    offset = 3 if minimum == 4 else 1
    for a in range(minimum,upper[0]+1):
        for b in range(minimum,min(upper[1],cap//(a*minimum))+1):
            top = min(upper[2],cap//(a*b))
            s0,s1 = lengths[0]/(a-offset),lengths[1]/(b-offset)
            if max(s0,s1)/min(s0,s1) <= 1.6:
                low_c = max(minimum, math.ceil(lengths[2]/(1.6*min(s0,s1))+offset-1e-12))
                hi_c = min(top, math.floor(lengths[2]/(max(s0,s1)/1.6)+offset+1e-12))
                if hi_c >= low_c:
                    s2=lengths[2]/(hi_c-offset)
                    row=(-a*b*hi_c,max(s0,s1,s2)/min(s0,s1,s2),(a,b,hi_c))
                    if good is None or row < good: good=row
            # Closest third spacing to [min(s0,s1),max(s0,s1)] plus endpoints
            # suffices to minimize anisotropy; ties prefer larger product.
            cs={minimum,top}
            for s in (s0,s1):
                t=lengths[2]/s+offset
                cs.update((math.floor(t),math.ceil(t)))
            for c in cs:
                if minimum<=c<=top:
                    s2=lengths[2]/(c-offset)
                    row=(max(s0,s1,s2)/min(s0,s1,s2),-a*b*c,(a,b,c))
                    if fallback is None or row<fallback: fallback=row
    return list(good[2] if good is not None else fallback[2])

def mode_mul(x, a, axis):
    z=x.movedim(axis,0); shape=z.shape
    return (a @ z.reshape(shape[0],-1)).reshape(a.shape[0],*shape[1:]).movedim(0,axis)

def tensor_forward(c, factors):
    order=sorted(range(3),key=lambda i:factors[i].shape[0]/factors[i].shape[1])
    for axis in order: c=mode_mul(c,factors[axis],axis)
    return c.contiguous()

def tensor_adjoint(y, factors):
    order=sorted(range(3),key=lambda i:factors[i].shape[1]/factors[i].shape[0])
    for axis in order: y=mode_mul(y,factors[axis].T,axis)
    return y.contiguous()

def product_norm(factors):
    a,b,c=[f.square().sum(0) for f in factors]
    return (a[:,None,None]*b[None,:,None]*c[None,None,:]).sqrt()

class Grid:
    def __init__(self,family,shape,lo,hi,knots=None):
        self.family=family; self.shape=list(shape); self.lo=list(lo); self.hi=list(hi)
        self.knots=knots
    @property
    def m(self):
        k=math.prod(self.shape)
        return k+3 if self.family=='gaussian_translation' else k*4 if self.family=='gaussian_affine' else k
    @property
    def params(self): return self.m*3
    def metadata(self): return dict(type='grid',family=self.family,shape=self.shape,lo=self.lo,hi=self.hi,knots=self.knots)
    @classmethod
    def from_meta(cls,d): return cls(d['family'],d['shape'],d['lo'],d['hi'],d.get('knots'))
    def at_factor(self,f):
        if self.family=='bspline':
            knots=[]; shape=[]
            for n,lo,hi in zip(self.shape,self.lo,self.hi):
                boundaries=np.linspace(lo,hi,n-2)
                ids=sorted(set(range(0,len(boundaries),f))|{len(boundaries)-1})
                b=boundaries[ids].tolist()
                knots.append([b[0]]*3+b+[b[-1]]*3); shape.append(len(b)+2)
            return Grid(self.family,shape,self.lo,self.hi,knots)
        return Grid(self.family,[max(2,math.ceil((n-1)/f)+1) for n in self.shape],self.lo,self.hi)

class GridOperator:
    def __init__(self,model,shape,device='cpu',check=lambda:None):
        self.model=model; self.shape=tuple(shape); self.m=model.m; self.check=check
        self.device=torch.device(device); self.factors=[]; self.local=[]
        self.coords=[torch.linspace(a,b,n,device=device,dtype=torch.float64)
                     for a,b,n in zip(model.lo,model.hi,shape)]
        for axis,(x,n,lo,hi) in enumerate(zip(self.coords,model.shape,model.lo,model.hi)):
            check()
            if model.family=='bspline':
                k=model.knots[axis] if model.knots else [lo]*3+np.linspace(lo,hi,n-2).tolist()+[hi]*3
                base=_bspline_1d(x,torch.tensor(k,device=device,dtype=torch.float64),n)
            else:
                centers=torch.linspace(lo,hi,n,device=device,dtype=torch.float64)
                local=(x[:,None]-centers[None,:])/((hi-lo)/(n-1))
                base=torch.softmax(-.5*(local/.85).square(),1)
                self.local.append(base*local)
            self.factors.append(base)
        self.root=[torch.ones(shape,device=device,dtype=torch.float64)] if model.family=='gaussian_translation' else []
        if self.root:
            for axis,x in enumerate(self.coords):
                sh=[1,1,1]; sh[axis]=len(x)
                z=(x-(model.lo[axis]+model.hi[axis])/2)/((model.hi[axis]-model.lo[axis])/2)
                self.root.append(z.reshape(sh).expand(shape))
    def term(self,i):
        fs=list(self.factors)
        if i: fs[i-1]=self.local[i-1]
        return fs
    def forward(self,c):
        self.check(); channels=c.shape[-1]; fam=self.model.family
        if fam=='gaussian_translation':
            grid=torch.cat((c[4:],c.new_zeros((1,channels))),0).reshape(*self.model.shape,channels)
            out=tensor_forward(grid,self.factors)
            for f,v in zip(self.root,c[:4]): out.add_(f[...,None]*v)
            return out
        if fam=='gaussian_affine':
            grid=c.reshape(*self.model.shape,4,channels)
            out=tensor_forward(grid[...,0,:],self.factors)
            for i in range(1,4): self.check(); out.add_(tensor_forward(grid[...,i,:],self.term(i)))
            return out
        return tensor_forward(c.reshape(*self.model.shape,channels),self.factors)
    def adjoint(self,y):
        self.check(); channels=y.shape[-1]; fam=self.model.family
        if fam=='gaussian_translation':
            root=torch.stack([(f[...,None]*y).sum((0,1,2)) for f in self.root])
            return torch.cat((root,tensor_adjoint(y,self.factors).reshape(-1,channels)[:-1]),0)
        if fam=='gaussian_affine':
            return torch.stack([tensor_adjoint(y,self.term(i)) for i in range(4)],-2).reshape(-1,channels)
        return tensor_adjoint(y,self.factors).reshape(-1,channels)
    def norms(self):
        self.check(); fam=self.model.family
        if fam=='gaussian_translation':
            return torch.cat((torch.stack([f.square().sum().sqrt() for f in self.root]),product_norm(self.factors).reshape(-1)[:-1]))
        if fam=='gaussian_affine': return torch.stack([product_norm(self.term(i)) for i in range(4)],-1).reshape(-1)
        return product_norm(self.factors).reshape(-1)

def insert_knot(c,knots,u,axis):
    """Boehm insertion, degree three; exact continuous-field inheritance."""
    p=3; k=int(np.searchsorted(knots,u,side='right')-1); s=sum(v==u for v in knots)
    x=c.movedim(axis,0); n=x.shape[0]-1
    q=x.new_empty((n+2,*x.shape[1:])); q[:k-p+1]=x[:k-p+1]; q[k-s+1:]=x[k-s:]
    for i in range(k-p+1,k-s+1):
        a=(u-knots[i])/(knots[i+p]-knots[i]); q[i]=a*x[i]+(1-a)*x[i-1]
    return q.movedim(0,axis),knots[:k+1]+[u]+knots[k+1:]

def prolong_spline(old,new,c,check=lambda:None):
    grid=c.reshape(*old.shape,-1)
    for axis in range(3):
        k=list(old.knots[axis])
        for u in sorted(set(new.knots[axis][4:-4])-set(k)):
            check(); grid,k=insert_knot(grid,k,u,axis)
        if len(k)!=len(new.knots[axis]) or not np.allclose(k,new.knots[axis],rtol=0,atol=1e-12):
            raise ValueError('Non-nested knots')
    return grid.reshape(new.m,-1)
