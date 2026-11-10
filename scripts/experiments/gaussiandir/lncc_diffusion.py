"""Uniform-window/edge estimates of the existing full-volume NCC9 + diffusion.

The stochastic objective has the same expectation as P43's native full-volume
loss. Entire warped 9-cubed neighborhoods are evaluated, never a neighborhood
translated only by its center's displacement. Every axis samples its own full
valid-edge domain. Dense final scoring calls the untouched P43 implementation.
"""
from functools import lru_cache
import importlib.util
import math
import sys

import torch
import torch.nn.functional as F

from scripts.experiments.oasis_alpha_sweep import common
from .data import flat_coordinates, false_to_true
from .model import REPO


@lru_cache(None)
def reference():
    name="_gaussiandir_p43_loss_reference"
    path=REPO/"scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    previous=sys.modules.get("common")
    sys.modules["common"]=common
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None: sys.modules.pop("common",None)
        else: sys.modules["common"]=previous
    return module


def unravel(ids, shape):
    d,h,w=shape
    return torch.stack((ids//(h*w),(ids//w)%h,ids%w),-1)


def window_indices(centers, shape, radius=4):
    axis=torch.arange(-radius,radius+1,device=centers.device)
    offsets=torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"),-1).reshape(-1,3)
    dhw=unravel(centers,shape)[:,None]+offsets
    valid=((dhw>=0)&(dhw<dhw.new_tensor(shape))).all(-1)
    safe=torch.minimum(dhw.clamp_min(0),dhw.new_tensor(shape)-1)
    return (safe[...,0]*shape[1]+safe[...,1])*shape[2]+safe[...,2],valid


def valid_edge_indices(shape, axis, device, count=None):
    edge_shape=list(shape)
    edge_shape[axis]-=1
    n=math.prod(edge_shape)
    ids=torch.arange(n,device=device) if count is None else torch.randint(n,(count,),device=device)
    dhw=unravel(ids,edge_shape)
    start=(dhw[:,0]*shape[1]+dhw[:,1])*shape[2]+dhw[:,2]
    return start,start+(shape[1]*shape[2],shape[2],1)[axis]


def window_ncc(fixed, moving):
    mu_f,mu_m=fixed.mean(-1),moving.mean(-1)
    var_f=fixed.square().mean(-1)-mu_f.square()
    var_m=moving.square().mean(-1)-mu_m.square()
    cov=(fixed*moving).mean(-1)-mu_f*mu_m
    return 1-(cov.square()/(var_f*var_m).clamp_min(torch.finfo(fixed.dtype).eps)).mean()


class LocalObjective:
    def __init__(self, fixed, moving, settings):
        self.fixed,self.moving=fixed,moving
        self.shape=tuple(fixed.shape)
        self.edge_count=settings["edges_per_axis"]
        self.voxel_scale=fixed.new_tensor(self.shape[::-1])/2
        self.axes=[torch.linspace(-1,1,n,device=fixed.device,dtype=fixed.dtype) for n in self.shape]

    def prepare(self, centers):
        indices,valid=window_indices(centers,self.shape)
        edges=[valid_edge_indices(self.shape,axis,centers.device,self.edge_count) for axis in range(3)]
        all_ids=torch.cat((indices.flatten(),torch.stack([torch.stack(edge) for edge in edges]).flatten()))
        return dict(indices=indices,valid=valid,field_indices=all_ids,coordinates=flat_coordinates(all_ids,self.shape),
                    window_queries=indices.numel())

    def __call__(self, flow, batch):
        n=batch["window_queries"]
        dhw=unravel(batch["indices"].flatten(),self.shape)
        # Use exactly the historical endpoint-aligned identity grid for warping.
        identity=torch.stack([self.axes[a][dhw[:,a]] for a in (2,1,0)],-1)
        query=identity+false_to_true(flow[:n],self.shape)
        warped=F.grid_sample(self.moving[None,None],query[None,None,None],
                             align_corners=True,padding_mode="border").reshape_as(batch["indices"])
        fixed=self.fixed.flatten()[batch["indices"]]*batch["valid"]
        warped=warped*batch["valid"]  # Zero padding of the LNCC window, not border replication.
        similarity=window_ncc(fixed,warped)
        edge_flow=flow[n:].reshape(3,2,self.edge_count,3)*self.voxel_scale
        regularization=(edge_flow[:,1]-edge_flow[:,0]).square().mean()
        return similarity,regularization


@torch.no_grad()
def dense_objective(field_false, fixed, moving, lam=1.):
    """Exact historical full-image loss, independently of stochastic training."""
    shape=fixed.shape
    grid=common.make_coordinate_grid(shape,fixed.device)
    warped=F.grid_sample(moving[None,None],(grid+false_to_true(field_false,shape))[None],
                         align_corners=True,padding_mode="border")
    similarity=reference().NCC9(fixed[None,None])(warped)
    vox=field_false*field_false.new_tensor(list(shape)[::-1])/2
    regularization=common.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None)
    return dict(loss=float(similarity+lam*regularization),lncc9_squared_loss=float(similarity),
                diffusion_voxel=float(regularization),lambda_diffusion=lam,
                scope="exact_full_native_volume_unmasked",image_padding="border",window_padding="zeros")
