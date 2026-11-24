"""CPU posthoc anatomy alignment on shared native coverage; never tunes lambda."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.ndimage import map_coordinates,binary_erosion,distance_transform_edt
from .protocol import read,dump,stamp
from .report import generate

def surface_metrics(fixed,moved,spacing):
    nf=int(fixed.sum());nm=int(moved.sum())
    if not nf and not nm:return dict(status='not_visible',dice=None,assd_mm=None,hd95_mm=None)
    dice=2*int(np.sum(fixed&moved))/(nf+nm)
    if not nf or not nm:return dict(status='one_mask_empty',dice=dice,assd_mm=None,hd95_mm=None)
    union=fixed|moved;bounds=[]
    for axis in range(3):
        ids=np.flatnonzero(np.any(union,axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(max(0,int(ids[0])-1),min(fixed.shape[axis],int(ids[-1])+2)))
    fixed=fixed[tuple(bounds)];moved=moved[tuple(bounds)]
    sf=fixed&~binary_erosion(fixed);sm=moved&~binary_erosion(moved)
    d1=distance_transform_edt(~sf,sampling=spacing)[sm]
    d2=distance_transform_edt(~sm,sampling=spacing)[sf]
    all_dist=np.concatenate((d1,d2))
    return dict(status='measured',dice=dice,assd_mm=float(all_dist.mean()),hd95_mm=float(np.percentile(all_dist,95)))

def bone_rigidity(field,low,high,origin,spacing,bone):
    count=0;strain_sum=0.;volume_sum=0.
    shape=high-low+1
    for start in range(0,shape[0],10):
        stop=min(start+10,shape[0]);z0=int(low[0]+start-origin[0]);z1=int(low[0]+stop-origin[0])
        begin=max(0,z0-1);end=min(field.shape[0],z1+1)
        block=np.asarray(field[begin:end],dtype=np.float64)
        sub=(slice(z0-begin,z1-begin),slice(int(low[1]-origin[1]),int(high[1]+1-origin[1])),slice(int(low[2]-origin[2]),int(high[2]+1-origin[2])))
        J=[]
        for component in range(3):
            grad=np.gradient(block[...,component]*spacing[component],*spacing,axis=(0,1,2),edge_order=1)
            J.append([grad[axis][sub]+(component==axis) for axis in range(3)])
        strain=np.zeros(J[0][0].shape)
        for i in range(3):
            for j in range(3):strain+=(sum(J[k][i]*J[k][j] for k in range(3))-(i==j))**2
        a,b,c=J[0];d,e,f=J[1];g,h,i=J[2]
        det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        mask=bone[start:stop];count+=int(mask.sum())
        strain_sum+=float(np.sqrt(strain)[mask].sum());volume_sum+=float(np.abs(det-1)[mask].sum())
    return dict(voxels=count,mean_frobenius_JtJ_minus_I=strain_sum/count if count else None,
                mean_abs_jdet_minus_one=volume_sum/count if count else None)

def evaluate(path,anatomy_root,prepared_root):
    evaluation=read(path/'evaluation.json');ds=evaluation['dataset'];case=evaluation['case']
    ann=anatomy_root/f'{ds}_{case:02d}'
    receipts=[ann/role/'outcome.json' for role in ('fixed','moving')]
    if not all(p.exists() for p in receipts):return None
    states=[read(p) for p in receipts]
    if any(s['status']!='accepted_automatic_labels' for s in states):
        return dict(status='annotation_failed',annotations=states,time=stamp())
    fixed=np.load(ann/'fixed/labels_native_zyx.npy',mmap_mode='r')
    moving=np.load(ann/'moving/labels_native_zyx.npy',mmap_mode='r')
    label='4DCT' if ds=='4dct' else 'COPD'
    audits=[read(prepared_root/'geometry'/crop/label/f'case{case:02d}'/'preprocess_audit.json') for crop in ('C0','C1','C2')]
    boxes=np.asarray([a['crop_bounds_zyx_inclusive'] for a in audits])
    low=boxes[:,:,0].max(axis=0);high=boxes[:,:,1].min(axis=0)
    shape=high-low+1;sl=tuple(slice(int(x),int(y)+1) for x,y in zip(low,high))
    target=np.asarray(fixed[sl]);warped=np.empty(tuple(shape),np.uint8)
    field=np.load(evaluation['native_field_path'],mmap_mode='r')
    origin=np.asarray(evaluation['geometry']['crop_bounds_zyx_inclusive'])[:,0]
    inside=0
    for start in range(0,shape[0],12):
        stop=min(start+12,shape[0]);indices=np.meshgrid(np.arange(low[0]+start,low[0]+stop),
            np.arange(low[1],high[1]+1),np.arange(low[2],high[2]+1),indexing='ij')
        fs=(slice(int(low[0]+start-origin[0]),int(low[0]+stop-origin[0])),
            slice(int(low[1]-origin[1]),int(high[1]+1-origin[1])),
            slice(int(low[2]-origin[2]),int(high[2]+1-origin[2])))
        u=np.asarray(field[fs]);coords=[indices[i]+u[...,i] for i in range(3)]
        valid=np.ones(u.shape[:3],bool)
        for i in range(3):valid&=(coords[i]>=0)&(coords[i]<=moving.shape[i]-1)
        inside+=int(valid.sum())
        warped[start:stop]=map_coordinates(moving,coords,order=0,mode='constant',cval=0,prefilter=False)
    mapping={int(k):v for k,v in states[0]['class_map'].items()}
    groups={'bone':[k for k,v in mapping.items() if v.startswith(('rib_','vertebrae_')) or v=='sternum'],
            'ribs':[k for k,v in mapping.items() if v.startswith('rib_')],
            'vertebrae':[k for k,v in mapping.items() if v.startswith('vertebrae_')],
            **{name:[k for k,v in mapping.items() if v==name] for name in ('sternum','heart','aorta','esophagus')}}
    spacing=np.asarray(evaluation['geometry']['native_spacing_zyx_mm'])
    metrics={name:surface_metrics(np.isin(target,ids),np.isin(warped,ids),spacing) for name,ids in groups.items()}
    # R231 lung contours are independent of the broader anatomy segmenter.
    audit=audits[0]
    source_case=Path(audit['geometry_path']) if 'geometry_path' in audit else prepared_root/'geometry'/'C0'/label/f'case{case:02d}'
    fx,mv=('T00','T50') if ds=='4dct' else ('iBH','eBH')
    lf=np.load(source_case/f'fixed_{fx}_lungmask_native_zyx.npy',mmap_mode='r')
    lm=np.load(source_case/f'moving_{mv}_lungmask_native_zyx.npy',mmap_mode='r')
    lung_moved=np.empty(tuple(shape),bool)
    for start in range(0,shape[0],12):
        stop=min(start+12,shape[0]);coords=np.meshgrid(np.arange(low[0]+start,low[0]+stop),np.arange(low[1],high[1]+1),np.arange(low[2],high[2]+1),indexing='ij')
        u=np.asarray(field[tuple(slice(int(low[i]-origin[i]+(start if i==0 else 0)),int((low[i]+stop if i==0 else high[i]+1)-origin[i])) for i in range(3))])
        lung_moved[start:stop]=map_coordinates(lm,[coords[i]+u[...,i] for i in range(3)],order=0,mode='constant',cval=0,prefilter=False)>0
    metrics['lung']=surface_metrics(np.asarray(lf[sl])>0,lung_moved,spacing)
    rigidity=bone_rigidity(field,low,high,origin,spacing,np.isin(target,groups['bone']))
    return dict(status='accepted_automatic_proxy',time=stamp(),common_bounds_zyx_inclusive=np.stack((low,high),axis=1).tolist(),
        source_inside_fraction=inside/int(np.prod(shape)),groups=metrics,bone_rigidity=rigidity,
        annotation_quality='independent automatic labels; visual QC pending; not manual ground truth',
        additional_fov_coverage=dict(fixed_bone_voxels_full=int(np.isin(fixed,groups['bone']).sum()),
            fixed_bone_voxels_common=int(np.isin(target,groups['bone']).sum())))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=read(a.config)
    run=Path(cfg['run_root']);review=Path(cfg['review_root']);prepared=Path(cfg['prepared_root']);anatomy=Path(cfg['anatomy_root'])
    while True:
        state_path=run/'controller_state.json'
        if not state_path.exists():time.sleep(10);continue
        state=read(state_path);pending=0
        for s in state['sweeps']:
            for r in s['results'].values():
                if r['status']!='accepted':continue
                path=Path(r['path']);dest=path/'anatomy_metrics.json'
                if dest.exists():continue
                try:result=evaluate(path,anatomy,prepared)
                except Exception as e:result=dict(status='failed',error=str(e),traceback=traceback.format_exc(),time=stamp())
                if result is None:pending+=1
                else:dump(dest,result);print('ANATOMY_METRICS',s['id'],result['status'],flush=True)
        annotation_done=(anatomy/'status.json').exists() and read(anatomy/'status.json').get('finished')
        # New supplements preserve the original sealed registration reports.
        first_done=(review/'first_batch/accepted.json').exists()
        if first_done and pending==0:generate(state,review,'anatomy_update',seal=False)
        if state['stage']=='complete' and annotation_done:
            if pending:
                # Every annotation has a terminal receipt, so missing geometry must be explicit.
                time.sleep(1);continue
            generate(state,review,'anatomy_final',seal=True)
            dump(run/'anatomy_posthoc_complete.json',dict(status='complete',time=stamp()));return
        time.sleep(15)

if __name__=='__main__':main()
