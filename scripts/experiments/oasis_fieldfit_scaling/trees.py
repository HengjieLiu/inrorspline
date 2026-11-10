"""Compact binary trees and exact shared-axis/blocked rank contractions."""
from __future__ import annotations
import math
import numpy as np
import torch
from operators import tensor_forward,tensor_adjoint

DTYPE=np.dtype([('lo','f8',(3,)),('hi','f8',(3,)),('parent','i8'),('branch','i1'),
                ('depth','i2'),('axis','i1'),('mid','f8'),('tau','f8'),
                ('left','i8'),('right','i8'),('split','i8')])

class Tree:
    def __init__(self,family,lo,hi):
        self.family=family; self.lo=list(lo); self.hi=list(hi)
        self.data=np.zeros(16,dtype=DTYPE); self.count=1; self.k=1
        self.data['left']=-1; self.data['right']=-1; self.data['parent']=-1; self.data['split']=-1
        self.data[0]['lo']=lo; self.data[0]['hi']=hi
    @property
    def affine(self): return 'affine' in self.family
    @property
    def m(self): return 4+(self.k-1)*(4 if self.affine else 1)
    @property
    def params(self): return self.m*3
    def leaves(self): return np.flatnonzero(self.data[:self.count]['left']<0)
    def metadata(self): return dict(type='tree',family=self.family,lo=self.lo,hi=self.hi,count=self.count,k=self.k,
                                    gate_width=.12,min_child_size_mm=.25,max_depth=32)
    def clone(self):
        t=Tree(self.family,self.lo,self.hi); t.data=self.data[:self.count].copy(); t.count=self.count;t.k=self.k
        return t
    def split(self,goal,scores=None,check=lambda:None):
        leaves=self.leaves(); rows=self.data[leaves]; lengths=rows['hi']-rows['lo']
        scores=np.prod(lengths,1) if scores is None else np.asarray(scores)
        eligible=(rows['depth']<32)&(lengths.max(1)/2>=.25)
        ids=leaves[eligible]; values=scores[eligible]
        count=min(max(1,int(self.k*.25)),goal-self.k,len(ids))
        if count<=0: return 0
        chosen=ids[np.lexsort((ids,-values))[:count]]
        need=self.count+2*count
        if need>len(self.data):
            new=np.zeros(max(need,2*len(self.data)),dtype=DTYPE)
            for name in ('left','right','parent','split'):new[name]=-1
            new[:self.count]=self.data[:self.count]; self.data=new
        # Vectorized construction avoids millions of Python node dictionaries.
        p=self.data[chosen].copy(); axis=np.argmax(p['hi']-p['lo'],1); row=np.arange(count)
        middle=(p['lo'][row,axis]+p['hi'][row,axis])/2
        left=self.count+2*row;right=left+1
        self.data['axis'][chosen]=axis;self.data['mid'][chosen]=middle
        self.data['tau'][chosen]=.12*(p['hi'][row,axis]-p['lo'][row,axis])
        self.data['left'][chosen]=left;self.data['right'][chosen]=right
        self.data['split'][chosen]=np.arange(self.k-1,self.k-1+count)
        for branch,children in ((0,left),(1,right)):
            self.data['lo'][children]=p['lo'];self.data['hi'][children]=p['hi']
            self.data['parent'][children]=chosen;self.data['branch'][children]=branch
            self.data['depth'][children]=p['depth']+1
            self.data['left'][children]=-1;self.data['right'][children]=-1;self.data['split'][children]=-1
            self.data['lo' if branch else 'hi'][children,axis]=middle
        self.count=need;self.k+=count; check()
        return count

class TreeOperator:
    """Axis factors deduplicated by exact gate history, never by small weights.

    A bounded dense coefficient core is used when possible; otherwise rank
    blocks are contracted without constructing a voxel-by-feature design.
    """
    def __init__(self,model,shape,device='cpu',check=lambda:None,tile=128,core_bytes=256<<20,leaves=False):
        self.model=model;self.shape=tuple(shape);self.check=check;self.tile=tile
        self.device=torch.device(device);self.m=model.k if leaves else model.m
        coords=[np.linspace(a,b,n) for a,b,n in zip(model.lo,model.hi,shape)]
        paths=np.zeros((model.count,3),np.int32)
        factors=[[np.ones(n)] for n in shape];lookup=[{} for _ in shape]
        nodes=model.data[:model.count]
        def factor(axis,key,fn):
            found=lookup[axis].get(key)
            if found is None:
                found=len(factors[axis]); factors[axis].append(fn());lookup[axis][key]=found
            return found
        # Children always follow parents. Equal axis histories share columns.
        for i in range(1,model.count):
            if i%1024==0:check()
            node=nodes[i];p=int(node['parent']);parent=nodes[p];a=int(parent['axis'])
            paths[i]=paths[p];base=int(paths[p,a]);branch=int(node['branch'])
            mid=float(parent['mid']);tau=float(parent['tau'])
            def make(a=a,base=base,mid=mid,tau=tau,branch=branch):
                # Stable sigmoid, preserving tails in floating arithmetic.
                z=(coords[a]-mid)/tau
                e=np.exp(-np.abs(z));g=np.where(z>=0,1/(1+e),e/(1+e))
                return factors[a][base]*(g if branch else 1-g)
            paths[i,a]=factor(a,('gate',base,mid,tau,branch),make)
        if leaves:
            indices=paths[model.leaves()].copy()
        else:
            indices=np.zeros((model.m,3),np.int32)
            for a in range(3):
                center=(model.lo[a]+model.hi[a])/2;scale=(model.hi[a]-model.lo[a])/2
                indices[a+1,a]=factor(a,('linear',0,center,scale),lambda a=a,center=center,scale=scale:(coords[a]-center)/scale)
            for i in np.flatnonzero(nodes['left']>=0):
                if int(i)%1024==0:check()
                node=nodes[i];base=paths[int(node['right'])];split=int(node['split'])
                start=4+split*(4 if model.affine else 1);indices[start]=base
                if model.affine:
                    for a in range(3):
                        center=float((node['lo'][a]+node['hi'][a])/2);scale=float((node['hi'][a]-node['lo'][a])/2)
                        indices[start+a+1]=base;b=int(base[a])
                        indices[start+a+1,a]=factor(a,('linear',b,center,scale),
                            lambda a=a,b=b,center=center,scale=scale:factors[a][b]*(coords[a]-center)/scale)
        # Remove unused intermediate histories before forming any tensor core.
        self.factors=[];mapped=[]
        for a in range(3):
            check(); used,inv=np.unique(indices[:,a],return_inverse=True)
            self.factors.append(torch.as_tensor(np.stack([factors[a][int(j)] for j in used],1),device=device))
            mapped.append(torch.as_tensor(inv,device=device,dtype=torch.long))
        self.indices=torch.stack(mapped,1); self.counts=tuple(f.shape[1] for f in self.factors)
        self.flat=(self.indices[:,0]*self.counts[1]+self.indices[:,1])*self.counts[2]+self.indices[:,2]
        self.core=math.prod(self.counts)*3*8<=core_bytes
        check()
    def blocks(self):
        for start in range(0,self.m,self.tile):
            self.check(); stop=min(start+self.tile,self.m)
            yield start,stop,[f[:,self.indices[start:stop,a]] for a,f in enumerate(self.factors)]
    def forward(self,c):
        self.check();channels=c.shape[1]
        if self.core:
            core=c.new_zeros((math.prod(self.counts),channels))
            core.index_add_(0,self.flat,c)
            return tensor_forward(core.reshape(*self.counts,channels),self.factors)
        out=c.new_zeros((*self.shape,channels));flat=out.reshape(self.shape[0],-1,channels)
        for start,stop,(a,b,d) in self.blocks():
            yz=(b[:,None,:]*d[None,:,:]).reshape(-1,stop-start)
            for ch in range(channels):flat[:,:,ch].add_(a @ (yz*c[start:stop,ch]).T)
        return out
    def adjoint(self,y):
        self.check();channels=y.shape[-1]
        if self.core:return tensor_adjoint(y,self.factors).reshape(-1,channels)[self.flat]
        out=y.new_empty((self.m,channels));flat=y.reshape(self.shape[0],-1,channels)
        for start,stop,(a,b,d) in self.blocks():
            yz=(b[:,None,:]*d[None,:,:]).reshape(-1,stop-start)
            for ch in range(channels):out[start:stop,ch]=((a.T@flat[:,:,ch])*yz.T).sum(1)
        return out
    def norms(self):
        self.check();out=torch.ones(self.m,device=self.device,dtype=torch.float64)
        for a,f in enumerate(self.factors):out*=f.square().sum(0)[self.indices[:,a]]
        return out.sqrt()
