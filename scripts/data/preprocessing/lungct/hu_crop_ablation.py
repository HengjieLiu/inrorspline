"""Explicit physical mask crop and common-scale intensity ablation primitives."""
from __future__ import annotations
import numpy as np
from .generate_4dct_ptvprep import img_thr

def intensity(raw, profile):
    windows = {'H0':(-944.,-124.),'H1':(-1000.,1000.),'H2':(-1000.,2000.)}
    if profile == 'H0': return img_thr(raw)
    lo,hi = windows[profile]
    value = np.asarray(raw,dtype=np.float32)-np.float32(1024.)
    value = np.clip(value,lo,hi)+np.float32(944.)
    return (value*np.float32(1./820.)).astype(np.float32,copy=False)

def mask_union_bounds(masks, spacing_zyx, margin_xyz_mm=(20.,20.,10.)):
    shape = np.asarray(masks[0].shape)
    lows=[];highs=[]
    for mask in masks:
        if tuple(mask.shape)!=tuple(shape): raise ValueError('Mask/image shape mismatch')
        lo=[];hi=[]
        for axis in range(3):
            indices=np.flatnonzero(np.any(mask>0,axis=tuple(i for i in range(3) if i!=axis)))
            if not len(indices): raise ValueError('Empty native R231 mask')
            lo.append(indices[0]);hi.append(indices[-1])
        lows.append(lo);highs.append(hi)
    pad=np.ceil(np.asarray(margin_xyz_mm)[::-1]/np.asarray(spacing_zyx)).astype(np.int64)
    low=np.maximum(0,np.min(lows,axis=0)-pad)
    high=np.minimum(shape-1,np.max(highs,axis=0)+pad)
    return np.stack((low,high),axis=1).astype(np.int64)
