import numpy as np
import pytest
from scripts.experiments.lungct_hu_crop_lambda.protocol import (
    weight,sweep_state,direction,combinations,FIRST,REST,WINDOWS,CAP)
from scripts.data.preprocessing.lungct.hu_crop_ablation import intensity,mask_union_bounds
from scripts.data.preprocessing.lungct.generate_4dct_ptvprep import img_thr

def accepted(value):return dict(status='accepted',score=value)

def test_windows_retain_common_hu_scale_and_exact_legacy():
    raw=np.array([-500,0,24,80,224,900,1024,2024,3024,5000],dtype=np.int16)
    np.testing.assert_array_equal(intensity(raw,'H0'),img_thr(raw))
    for profile,(lo,hi) in WINDOWS.items():
        expected=(np.clip(raw.astype(np.float32)-1024,lo,hi)+944)*np.float32(1/820)
        np.testing.assert_array_equal(intensity(raw,profile),expected)
    assert intensity(raw,'H1')[6]>1
    assert intensity(raw,'H1')[2]<0
    assert intensity(raw,'H2')[8]>intensity(raw,'H1')[8]

def test_physical_mask_union_margin_and_boundary():
    a=np.zeros((20,100,100),np.uint8);b=a.copy()
    a[1:7,40:50,30:45]=1;b[3:10,45:55,40:60]=1
    bounds=mask_union_bounds([a,b],[2.5,.625,.625])
    np.testing.assert_array_equal(bounds,[[0,13],[8,86],[0,91]])
    with pytest.raises(ValueError):mask_union_bounds([np.zeros_like(a)], [2.5,.625,.625])

def test_direction_is_independent_and_counts_successive_increases():
    r={'0':accepted(.9),'1':accepted(.85),'2':accepted(.87),'3':accepted(.89),'-1':accepted(.8)}
    s=sweep_state(r)
    assert s['upper']['status']=='two_increases' and s['lower']['next']==-2
    assert s['best']['k']==-1 and s['next']==[-2]
    # Two values worse than the best do not suffice when the last improves.
    r['3']=accepted(.86)
    assert direction(r,1)['next']==4

def test_ties_and_failures_reset_streak():
    r={'0':accepted(.9),'1':accepted(1.),'2':accepted(1.),'3':accepted(1.1)}
    assert direction(r,1)['streak']==1
    r['2']=dict(status='failed',score=None)
    assert direction(r,1)['streak']==0
    r['4']=accepted(1.2);r['5']=accepted(1.3)
    assert direction(r,1)['status']=='two_increases'

def test_cap_flat_curves_and_tie_selection():
    r={str(k):accepted(1.) for k in range(-8,9)}
    s=sweep_state(r)
    assert s['done'] and s['upper']['status']==s['lower']['status']=='cap_reached'
    assert s['best']['k']==0
    assert weight(-8)==.00021484375 and weight(8)==14.08
    with pytest.raises(ValueError):weight(9)
    assert len(combinations(FIRST))==36 and len(combinations(FIRST+REST))==180

def test_failed_center_does_not_block_remaining_candidates():
    r={'0':dict(status='failed',score=None),'1':accepted(1.),'2':accepted(1.1),'3':accepted(1.2)}
    assert sweep_state(r)['upper']['status']=='two_increases'
    assert sweep_state({},True)['done']

def test_native_tre_zero_and_known_pull_translation():
    from scripts.experiments.lungct_corrected_1mm.core import scores
    fixed=np.random.default_rng(123).integers(3,10,size=(300,3)).astype(float)
    translation=np.array([1.,-2.,3.]);moving=fixed+translation
    field=np.zeros((16,16,16,3),np.float32);field[:]=translation
    result,_=scores(field,fixed,moving,np.array([2.5,.75,1.25]))
    assert result['snap_mean']==result['subvoxel_mean']==0
    initial,_=scores(np.zeros_like(field),fixed,moving,np.array([2.5,.75,1.25]))
    assert initial['snap_mean']==pytest.approx(np.linalg.norm(translation*[2.5,.75,1.25]))

def test_anatomy_surface_and_rigid_vs_shear_metrics():
    from scripts.experiments.lungct_hu_crop_lambda.anatomy_metrics import surface_metrics,bone_rigidity
    mask=np.zeros((12,12,12),bool);mask[3:8,3:8,3:8]=True
    same=surface_metrics(mask,mask,[2.5,.75,1.25])
    assert same['dice']==1 and same['assd_mm']==same['hd95_mm']==0
    field=np.zeros((12,12,12,3),np.float32);field[:]=[1,2,3]
    result=bone_rigidity(field,np.array([0,0,0]),np.array([11,11,11]),np.zeros(3,int),np.ones(3),mask)
    assert result['mean_frobenius_JtJ_minus_I']==result['mean_abs_jdet_minus_one']==0
    field[:]=0;field[...,2]=.2*np.arange(12)[None,:,None]
    shear=bone_rigidity(field,np.array([0,0,0]),np.array([11,11,11]),np.zeros(3,int),np.ones(3),mask)
    assert shear['mean_frobenius_JtJ_minus_I']==pytest.approx(np.sqrt(2*.2**2+.2**4),rel=1e-5)
    assert shear['mean_abs_jdet_minus_one']==0

def test_controller_centers_first_and_report_checkpoint(tmp_path):
    import csv
    from scripts.experiments.lungct_hu_crop_lambda.protocol import dump
    from scripts.experiments.lungct_hu_crop_lambda.controller import Controller
    from scripts.experiments.lungct_hu_crop_lambda.report import generate
    cfg=dict(run_root=str(tmp_path),review_root=str(tmp_path/'review'),prepared_root=str(tmp_path/'prepared'),frozen_root=str(tmp_path/'source'))
    c=Controller(cfg);c.state['sweeps']=combinations(FIRST)
    for s in c.state['sweeps']:s.update(batch='first',results={})
    assert len(c.pending())==36 and all(k==0 for _,k in c.pending())
    for s in c.state['sweeps']:s['results']={'0':dict(accepted(1.),score_sd=.1)}
    assert len(c.pending())==72
    for s in c.state['sweeps']:
        for k in (-1,1):s['results'][str(k)]=dict(accepted(1.1),score_sd=.1)
        for k in (-2,2):s['results'][str(k)]=dict(accepted(1.2),score_sd=.1)
        path=tmp_path/f"labels_{s['dataset']}_{s['case']}"
        # Combined Dice is independently measured, not the average of components.
        values=(.9,.2,1.,.4) if s['case']==1 else (.6,.1,.8,None)
        dump(path/'anatomy_metrics.json',dict(groups={g:dict(dice=v) for g,v in zip(('bone','ribs','vertebrae','sternum'),values)}))
        s['results']['0']['path']=str(path)
    assert not c.pending()
    generate(c.state,tmp_path/'review','first_batch',seal=True)
    assert (tmp_path/'review/first_batch/accepted.json').exists()
    assert '2 / 2' in (tmp_path/'review/first_batch/report.md').read_text()
    cohorts=list(csv.DictReader((tmp_path/'review/first_batch/cohort_summaries.csv').open()))
    def metric(name):return next(r for r in cohorts if r['dataset']=='4dct' and r['hu']=='H0' and r['crop']=='C0' and r['endpoint']==name)
    assert float(metric('best_bone_dice')['mean'])==pytest.approx(.75)
    assert float(metric('best_ribs_dice')['mean'])==pytest.approx(.15)
    assert float(metric('best_vertebrae_dice')['mean'])==pytest.approx(.9)
    assert metric('best_sternum_dice')['n']=='1'
    assert metric('best_sternum_dice')['unit']=='fraction'
