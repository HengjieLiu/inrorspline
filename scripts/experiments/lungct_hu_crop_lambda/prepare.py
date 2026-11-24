"""Prepare all nine profiles without modifying accepted inputs or native data."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from .protocol import WINDOWS,CROPS,FIRST,REST,dump,read,sha,stamp
from scripts.data.common.lungct import resolve_4dct_case,resolve_copd_case,load_landmarks_xyz
from scripts.data.preprocessing.lungct.hu_crop_ablation import intensity,mask_union_bounds
from scripts.data.preprocessing.lungct.generate_4dct_ptvprep import (
    ptvreg_crop_bounds_zyx_from_landmarks_xyz,transform_landmarks_crop_to_registration)

def save(path, value):
    if path.exists(): raise FileExistsError(path)
    np.save(path,value,allow_pickle=False)

def link(source, destination):
    destination.symlink_to(source)

def prepare_case(raw_root, baseline_root, out_root, dataset, case):
    label='4DCT' if dataset=='4dct' else 'COPD'
    spec=(resolve_4dct_case if dataset=='4dct' else resolve_copd_case)(case,raw_root)
    if spec.status!='complete': raise ValueError(f'Incomplete raw case {dataset}/{case}')
    phases=('T00','T50') if dataset=='4dct' else ('iBH','eBH')
    image_paths=(spec.image_t00,spec.image_t50) if dataset=='4dct' else (spec.image_ibh,spec.image_ebh)
    point_paths=(spec.landmarks_t00,spec.landmarks_t50) if dataset=='4dct' else (spec.landmarks_ibh,spec.landmarks_ebh)
    xyz=[load_landmarks_xyz(p) for p in point_paths]
    native_points=[p[:,::-1].copy()-1 for p in xyz]
    shape=np.asarray(spec.shape_zyx);spacing=np.asarray(spec.spacing_zyx)
    raw=[np.memmap(p,mode='r',dtype=np.int16,shape=tuple(shape)) for p in image_paths]
    mask_dir=raw_root/'processed/lungmask_R231_native'/label/f'case{case:02d}'
    mask_paths=[mask_dir/f'case{case:02d}_{p}_R231_binary_zyx.npy' for p in phases]
    masks=[np.load(p,mmap_mode='r') for p in mask_paths]
    sources={str(p):sha(p) for p in [*image_paths,*point_paths,*mask_paths]}
    bounds={
        'C0':ptvreg_crop_bounds_zyx_from_landmarks_xyz(xyz[0],xyz[1],shape),
        'C1':mask_union_bounds(masks,spacing),
        'C2':np.stack((np.zeros(3,dtype=int),shape-1),axis=1),
    }
    records=[]
    for crop in CROPS:
        geo=out_root/'geometry'/crop/label/f'case{case:02d}'
        geo_receipt=geo/'accepted.json'
        if geo_receipt.exists():
            gr=read(geo_receipt)
            if gr['source_hashes']!=sources: raise ValueError('Source drift')
            if sha(geo/'preprocess_audit.json')!=gr['audit_sha256']:raise ValueError('Geometry audit drift')
            for name,digest in gr['array_hashes'].items():
                if sha(geo/name)!=digest:raise ValueError('Geometry array drift')
            audit=read(geo/'preprocess_audit.json')
        else:
            geo.mkdir(parents=True,exist_ok=False)
            box=bounds[crop];crop_shape=box[:,1]-box[:,0]+1
            reg=np.rint(crop_shape*spacing).astype(int)
            sl=tuple(slice(int(a),int(b)+1) for a,b in box)
            arrays={}
            for role,phase,pts,mask,mask_path in zip(('fixed','moving'),phases,native_points,masks,mask_paths):
                cp=pts-box[:,0]
                rp=transform_landmarks_crop_to_registration(cp,crop_shape,reg)
                if pts.shape!=(300,3) or np.any(cp<0) or np.any(cp>crop_shape-1):
                    raise ValueError(f'Landmarks outside requested {crop}; never recrop C1 from landmarks')
                np.testing.assert_allclose(rp*(crop_shape-1)/(reg-1)+box[:,0],pts,atol=1e-10,rtol=0)
                for suffix,array in [('native0_zyx',pts),('crop_zyx',cp),('reg1mm_zyx',rp)]:
                    name=f'landmarks_{role}_{phase}_{suffix}.npy';save(geo/name,array);arrays[name]=sha(geo/name)
                name=f'{role}_{phase}_lungmask_native_zyx.npy';link(mask_path,geo/name)
                resized=F.interpolate(torch.from_numpy(np.asarray(mask[sl],dtype=np.float32))[None,None],size=tuple(reg),mode='nearest')[0,0].numpy().astype(np.uint8)
                name=f'{role}_{phase}_lungmask_reg1mm_nearest_zyx.npy';save(geo/name,resized);arrays[name]=sha(geo/name)
            audit=dict(dataset=dataset,case_id=case,crop_convention='corrected',crop_profile=crop,
                raw_shape_zyx=shape.tolist(),native_spacing_zyx_mm=spacing.tolist(),
                crop_bounds_zyx_inclusive=box.tolist(),crop_shape_zyx=crop_shape.tolist(),
                registration_shape_zyx=reg.tolist(),registration_spacing_zyx_mm=[1.,1.,1.],
                registration_to_native_scale_zyx=((crop_shape-1)/(reg-1)).tolist(),
                image_paths=dict(zip(('fixed','moving'),map(str,image_paths))),
                landmark_paths=dict(zip(('fixed','moving'),map(str,point_paths))),
                source_hashes=sources,comparison_bounds_zyx_inclusive=bounds['C0'].tolist(),
                margin_xyz_mm=[20,20,10] if crop=='C1' else None,
                mask_crop_rule='union of both lungs in both phases; no optimization mask',
                verification=dict(pass_=True,landmarks=300,roundtrip_atol_mm=1e-10))
            audit['verification']['pass']=True
            dump(geo/'preprocess_audit.json',audit)
            dump(geo_receipt,dict(status='accepted',source_hashes=sources,array_hashes=arrays,audit_sha256=sha(geo/'preprocess_audit.json')))
        box=np.asarray(audit['crop_bounds_zyx_inclusive']);reg=tuple(audit['registration_shape_zyx'])
        sl=tuple(slice(int(a),int(b)+1) for a,b in box)
        for hu in WINDOWS:
            dest=out_root/f'{hu}_{crop}'/label/f'case{case:02d}'
            if (dest/'accepted.json').exists():
                receipt=read(dest/'accepted.json')
                if receipt['geometry_sha256']!=sha(geo/'accepted.json'):raise ValueError('Geometry drift')
                for name,digest in receipt['input_hashes'].items():
                    if sha(dest/name)!=digest:raise ValueError(f'Prepared input drift {dest/name}')
                records.append(str(dest));continue
            dest.mkdir(parents=True,exist_ok=False)
            for p in geo.glob('*.npy'):link(p,dest/p.name)
            parity=[]
            for role,phase,image in zip(('fixed','moving'),phases,raw):
                volume=intensity(image[sl],hu)
                value=F.interpolate(torch.from_numpy(volume)[None,None],size=reg,mode='trilinear',align_corners=True)[0,0].numpy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite preparation')
                name=f'{role}_{phase}_reg1mm_trilinear_zyx.npy';save(dest/name,value)
                if hu=='H0' and crop=='C0':
                    old=baseline_root/label/f'case{case:02d}'/name
                    if not np.array_equal(value,np.load(old,mmap_mode='r')):raise ValueError(f'H0C0 parity failure {old}')
                    parity.append(name)
            a=dict(audit,hu_profile=hu,hu_bounds=list(WINDOWS[hu]),intensity_rule='(clip(raw-1024,L,U)+944)/820; fixed gain; no postclip',geometry_path=str(geo))
            dump(dest/'preprocess_audit.json',a)
            hashes={p.name:sha(p) for p in dest.glob('*.npy')}
            hashes['preprocess_audit.json']=sha(dest/'preprocess_audit.json')
            dump(dest/'accepted.json',dict(status='accepted',geometry_sha256=sha(geo/'accepted.json'),input_hashes=hashes,historical_exact_parity=parity))
            records.append(str(dest))
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--batch',choices=['first','rest','all'],default='all');a=p.parse_args()
    torch.set_num_threads(2)
    pairs=FIRST if a.batch=='first' else REST if a.batch=='rest' else FIRST+REST
    receipt=a.out_root/f'preparation_{a.batch}.json'
    results=[]
    for ds,case in pairs:
        try:
            paths=prepare_case(a.raw_root,a.baseline_root,a.out_root,ds,case)
            r=dict(dataset=ds,case=case,status='accepted',conditions=len(paths),paths=paths)
        except Exception as e:
            r=dict(dataset=ds,case=case,status='failed',error=str(e),traceback=traceback.format_exc())
        results.append(r);dump(receipt,dict(updated=stamp(),cases=results))
        print(ds,case,r['status'],flush=True)
    if any(r['status']=='failed' for r in results):print('Preparation failures retained; controller skips affected conditions',flush=True)

if __name__=='__main__':main()
