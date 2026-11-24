"""Independent evaluation-only anatomy labels using pinned cached TotalSegmentator."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import traceback
from .protocol import FIRST,REST,dump,read,sha,stamp

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True);a=p.parse_args()
    import importlib.metadata
    import numpy as np
    import nibabel as nib
    import torch
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    from scripts.data.common.lungct import resolve_4dct_case,resolve_copd_case
    torch.set_num_threads(4)
    assert importlib.metadata.version('TotalSegmentator')=='2.16.0'
    mapping={int(k):v for k,v in class_map['total'].items()}
    names=[v for v in mapping.values() if v.startswith(('rib_','vertebrae_')) or v in ('sternum','heart','aorta','esophagus')]
    a.out_root.mkdir(parents=True,exist_ok=True)
    dump(a.out_root/'protocol.json',dict(version='2.16.0',task='total',fast=False,device='cpu',roi_subset=names,
        class_map=mapping,anatomy_role='evaluation only; independent per phase; not manual ground truth',
        visual_qc='overlay images available; automatic labels remain provisional until visual review'))
    records=[]
    for ds,case in FIRST+REST:
        spec=(resolve_4dct_case if ds=='4dct' else resolve_copd_case)(case,a.raw_root)
        phases=('T00','T50') if ds=='4dct' else ('iBH','eBH')
        images=(spec.image_t00,spec.image_t50) if ds=='4dct' else (spec.image_ibh,spec.image_ebh)
        for role,phase,image in zip(('fixed','moving'),phases,images):
            dest=a.out_root/f'{ds}_{case:02d}'/role
            if (dest/'outcome.json').exists():records.append(read(dest/'outcome.json'));continue
            dest.mkdir(parents=True,exist_ok=True)
            try:
                raw=np.memmap(image,mode='r',dtype=np.int16,shape=spec.shape_zyx)
                hu=raw.astype(np.float32)-1024.
                sp=np.asarray(spec.spacing_zyx)[::-1]
                affine=np.diag([-sp[0],-sp[1],-sp[2],1.]);affine[2,3]=(hu.shape[0]-1)*sp[2]
                img=nib.Nifti1Image(hu.transpose(2,1,0),affine)
                result=totalsegmentator(img,output=None,ml=True,task='total',roi_subset=names,
                    fast=False,device='cpu',quiet=True,nr_thr_resamp=2,nr_thr_saving=1,skip_saving=True)
                if isinstance(result,tuple):result=result[0]
                labels=np.asanyarray(result.dataobj).transpose(2,1,0).astype(np.uint8)
                if labels.shape!=tuple(spec.shape_zyx):raise ValueError('Anatomy output geometry mismatch')
                if not np.allclose(result.affine,affine,atol=1e-4):raise ValueError('Anatomy affine mismatch')
                np.save(dest/'labels_native_zyx.npy',labels,allow_pickle=False)
                stats={mapping[int(k)]:dict(voxels=int(np.sum(labels==k)),mean_hu=float(hu[labels==k].mean()))
                       for k in np.unique(labels) if int(k) in mapping and k!=0}
                overlay(hu,labels,mapping,dest/'qc_overlay.png',spec.spacing_zyx)
                out=dict(status='accepted_automatic_labels',dataset=ds,case=case,role=role,phase=phase,
                    class_map=mapping,statistics=stats,shape_zyx=list(spec.shape_zyx),spacing_zyx=list(spec.spacing_zyx),
                    input_sha256=sha(image),labels_sha256=sha(dest/'labels_native_zyx.npy'),time=stamp(),
                    qc='geometry and nonempty anatomy checked; visual review pending; not manual ground truth')
                if not stats:raise ValueError('Empty anatomy output')
            except Exception as e:
                out=dict(status='failed',dataset=ds,case=case,role=role,error=str(e),traceback=traceback.format_exc(),time=stamp())
                traceback.print_exc()
            dump(dest/'outcome.json',out);records.append(out)
            dump(a.out_root/'status.json',dict(records=records,finished=False,updated=stamp()))
            print(ds,case,role,out['status'],flush=True)
    dump(a.out_root/'status.json',dict(records=records,finished=True,updated=stamp()))

def overlay(hu,labels,mapping,path,spacing):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bone=[k for k,v in mapping.items() if v.startswith(('rib_','vertebrae_')) or v=='sternum']
    soft=[k for k,v in mapping.items() if v in ('heart','aorta','esophagus')]
    fig,axes=plt.subplots(2,3,figsize=(12,9))
    for row,(lo,hi) in enumerate([(-1000,1000),(-200,1500)]):
        for axis in range(3):
            index=hu.shape[axis]//2
            image=np.take(hu,index,axis=axis);mask=np.take(labels,index,axis=axis)
            ax=axes[row,axis];ax.imshow(image,cmap='gray',vmin=lo,vmax=hi,origin='upper')
            for ids,color in [(bone,'cyan'),(soft,'yellow')]:
                binary=np.isin(mask,ids)
                if binary.any():ax.contour(binary,levels=[.5],colors=[color],linewidths=.5)
            ax.set_title(f'Raw array axis{axis}, slice{index}');ax.axis('off')
    fig.suptitle('Automatic evaluation labels: cyan bone / yellow heart+aorta+esophagus; visual QC required')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)

if __name__=='__main__':main()
