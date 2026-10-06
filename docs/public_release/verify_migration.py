"""Read-only private acceptance: compare public kernels and frozen geometry.

No optimizer is constructed or executed. Generated evidence goes to --out,
outside the source repository; only summarized milestone reports live in docs.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--research',type=Path,required=True)
    p.add_argument('--dirlab',type=Path,required=True);p.add_argument('--oasis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.public))
    from rightpriordir import oasis,lung,geometry as g,prepare_lung as prep
    old_oasis=module(a.research/'scripts/experiments/oasis_alpha_sweep/common.py','research_oasis')
    old_lung=module(a.research/'scripts/experiments/lungct_no_sweep_repro/common.py','research_lung')
    evidence={'optimization_run':False,'numerical_checks':[],'geometry_cases':[],'image_mask_cases':[]}
    # Initialization, values, coordinate gradients, analytic spline values/gradients.
    for public,old,label in [(oasis,old_oasis,'oasis'),(lung,old_lung,'lung')]:
        for name,omega in [('IDIR',32.),('SINR',50.)]:
            torch.manual_seed(1);x=old.build_siren(16,3,omega).double()
            torch.manual_seed(1);y=public.build_siren(16,3,omega,upstream=name).double()
            for u,v in zip(x.parameters(),y.parameters()):torch.testing.assert_close(u,v,rtol=0,atol=0)
            coords=torch.randn(11,3,dtype=torch.float64,requires_grad=True)
            torch.testing.assert_close(x(coords),y(coords),rtol=0,atol=0)
            gx=torch.autograd.grad(x(coords).square().sum(),coords)[0];gy=torch.autograd.grad(y(coords).square().sum(),coords)[0]
            torch.testing.assert_close(gx,gy,rtol=0,atol=0)
            evidence['numerical_checks'].append(f'{label}/{name}: initialization, values, gradients exact')
        shape=(13,15,17)
        for cps in [2,4]:
            _,control_shape=old.bspline_control_coords(shape,cps,torch.device('cpu'))
            control=torch.randn(int(np.prod(control_shape)),3,dtype=torch.float64,requires_grad=True)
            u=old.expand_bspline_controls(control,control_shape,shape,cps)
            v=public.expand_bspline_controls(control,control_shape,shape,cps)
            torch.testing.assert_close(u,v,rtol=0,atol=0)
            gu=torch.autograd.grad(u.square().sum(),control)[0];gv=torch.autograd.grad(v.square().sum(),control)[0]
            torch.testing.assert_close(gu,gv,rtol=0,atol=0)
            for mod in [old,public]:
                value=(mod.analytic_bspline_sampled_be(control.reshape(*control_shape,3),shape,cps)
                    if label=='oasis' else mod.analytic_bspline_sampled_be(control,control_shape,shape,cps))
                if mod is old:ref=value
                else:torch.testing.assert_close(ref,value,rtol=0,atol=0)
            evidence['numerical_checks'].append(f'{label}/cps{cps}: expansion, gradients, analytic BE exact')
    # All 20 real cases: crop bounds, shape, both sets of 300 transformed landmarks.
    prepared=a.dirlab/'processed/pTVReg_iso1mm'
    for dataset,mode,label in [('4dct','corrected','4DCT'),('copd','legacy','COPD')]:
        for i in range(1,11):
            case,phases,images,landmarks=prep.case_inputs(a.dirlab,dataset,i)
            points=[prep.raw.load_landmarks_xyz(p) for p in landmarks]
            bounds,shape,reg,corrected,legacy=prep.geometry(case,*points,mode)
            root=prepared/label/f'case{i:02d}';audit=json.loads((root/'preprocess_audit.json').read_text())
            np.testing.assert_array_equal(bounds,audit['crop_bounds_zyx_inclusive'])
            np.testing.assert_array_equal(shape,audit['crop_shape_zyx']);np.testing.assert_array_equal(reg,audit['registration_shape_zyx'])
            for role,phase,pts in zip(['fixed','moving'],phases,points):
                assert pts.shape==(300,3)
                native=g.landmarks_xyz_one_based_to_zyx_zero_based(pts)
                crop=g.transform_landmarks_native0_to_crop(native,bounds)
                registered=g.transform_landmarks_crop_to_registration(crop,shape,reg)
                saved=np.load(root/f'landmarks_{role}_{phase}_reg1mm_zyx.npy',allow_pickle=False)
                np.testing.assert_allclose(registered,saved,rtol=0,atol=1e-12)
                np.testing.assert_allclose(registered*(shape-1)/(reg-1)+bounds[:,0],native,rtol=0,atol=1e-12)
                assert g.points_inside_shape(registered,reg)['inside_count']==300
            evidence['geometry_cases'].append({'dataset':dataset,'case':i,'mode':mode,'legacy_minus_corrected':(legacy-corrected).tolist(),'landmarks_per_phase':300,'pass':True})
            # Pixel/mask parity on boundary case01, interior case02; both phases.
            if i in [1,2]:
                for role,phase,img_path in zip(['fixed','moving'],phases,images):
                    loader=prep.raw.load_raw_4dct_image if dataset=='4dct' else prep.raw.load_raw_copd_image
                    native=loader(img_path,case.shape_zyx)
                    image=g.resize_image_trilinear_zyx(g.crop_zyx(g.img_thr(native),bounds),reg,torch.device('cpu'))
                    saved=np.load(root/f'{role}_{phase}_reg1mm_trilinear_zyx.npy',mmap_mode='r',allow_pickle=False)
                    np.testing.assert_array_equal(image,saved)
                    mask_root=a.dirlab/'processed/lungmask_R231_native'/label/f'case{i:02d}'
                    mask=np.load(mask_root/f'case{i:02d}_{phase}_R231_binary_zyx.npy',allow_pickle=False)
                    mask_out=g.resize_mask_nearest_zyx(g.crop_zyx(mask,bounds),reg,torch.device('cpu'))
                    np.testing.assert_array_equal(mask_out,np.load(root/f'{role}_{phase}_lungmask_reg1mm_nearest_zyx.npy',mmap_mode='r',allow_pickle=False))
                    evidence['image_mask_cases'].append({'dataset':dataset,'case':i,'phase':phase,'image_and_mask_exact':True})
    # Verify source split and load a real OASIS pair without registration.
    assert oasis.parse_pair_file(a.oasis/'pair_test_200.txt')==oasis.parse_pair_file(a.public/'splits/pair_test_200.txt')
    batch=oasis.load_oasis_case(rank=0,device=torch.device('cpu'),oasis_root=a.oasis)
    evidence['oasis']={'rank':0,'fixed_id':batch['fixed_id'],'moving_id':batch['moving_id'],'shape':list(batch['fixed'].shape),
        'sorted_200_pair_order_verified':True,'mask_audit':batch['mask_audit']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'numerical_checks':len(evidence['numerical_checks']),'real_geometry_cases':len(evidence['geometry_cases']),
        'pixel_mask_checks':len(evidence['image_mask_cases']),'oasis_pair':evidence['oasis'],'optimization_run':False},indent=2))


if __name__=='__main__':main()
