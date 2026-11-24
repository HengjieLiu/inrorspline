"""Coordinate and protocol regression tests before GPU registrations."""
from pathlib import Path
import sys
import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/experiments/lungct_corrected_1mm'))
from core import corrfield_to_zyx,scores,numeric
sys.path.insert(0,str(REPO/'scripts/data/preprocessing/lungct'))
from generate_4dct_ptvprep import (ptvreg_crop_bounds_zyx_from_landmarks_xyz,
    task4_legacy_crop_bounds_zyx_from_landmarks_xyz,select_crop_bounds,require_fresh_case,
    transform_landmarks_crop_to_registration)


def test_interior_crop_difference_and_coordinate_roundtrip():
    pts=np.array([[40,50,20],[70,90,50]],dtype=float)
    corrected=ptvreg_crop_bounds_zyx_from_landmarks_xyz(pts,pts,(100,140,140))
    legacy=task4_legacy_crop_bounds_zyx_from_landmarks_xyz(pts,pts,(100,140,140))
    np.testing.assert_array_equal(legacy-corrected,np.ones((3,2)))
    np.testing.assert_array_equal(select_crop_bounds(corrected,legacy,'corrected'),corrected)
    native=pts[:,::-1]-1;shape=corrected[:,1]-corrected[:,0]+1;reg=np.rint(shape*[2.5,.625,.625])
    q=transform_landmarks_crop_to_registration(native-corrected[:,0],shape,reg)
    np.testing.assert_allclose(q*(shape-1)/(reg-1)+corrected[:,0],native,atol=1e-12)


def test_boundary_crop_changes_extent_and_preserves_rounding():
    pts=np.array([[20,20,1],[60,60,104]],dtype=float)
    corrected=ptvreg_crop_bounds_zyx_from_landmarks_xyz(pts,pts,(121,512,512))
    legacy=task4_legacy_crop_bounds_zyx_from_landmarks_xyz(pts,pts,(121,512,512))
    assert tuple(corrected[0])==(0,108) and tuple(legacy[0])==(0,109)
    assert np.rint((corrected[0,1]+1)*2.5)==272


def test_output_cannot_overwrite_existing_case(tmp_path):
    require_fresh_case(tmp_path/'new')
    with pytest.raises(FileExistsError):require_fresh_case(tmp_path)


def test_corrfield_adapter_keeps_axes_and_components():
    shape=(13,15,17);native=np.zeros((17,15,13,3),np.float32)
    x,y,z=np.meshgrid(np.arange(17),np.arange(15),np.arange(13),indexing='ij')
    native[...,0]=x;native[...,1]=2*y;native[...,2]=3*z
    field=corrfield_to_zyx(native,shape)
    np.testing.assert_array_equal(field[4,5,6],[12,10,6])


@pytest.mark.parametrize('shift',[(0.,0.,0.),(1.,-2.,3.)])
def test_known_field_sign_units_and_native_origin_invariance(shift):
    field=np.zeros((13,15,17,3),np.float32);field[:]=shift
    fixed=np.tile([6.,7.,8.],(300,1));moving=fixed+shift
    stats,v=scores(field,fixed,moving,np.array([2.5,.7,.8]))
    assert stats['snap_mean']==0. and stats['subvoxel_mean']==0.
    origin=np.array([5,71,115])
    np.testing.assert_array_equal(v[-2],np.linalg.norm((v[2]+origin-(moving+origin))*[2.5,.7,.8],axis=1))


def test_native_resize_converts_physical_displacements():
    field=np.zeros((7,9,11,3),np.float32);field[:]=[5.,1.4,2.4]
    native=numeric.resize_reg1mm_disp_to_native_crop_vox(field,(5,13,15),(2.5,.7,.8))
    np.testing.assert_allclose(native,np.broadcast_to([2.,2.,3.],native.shape),atol=5e-7)


def test_authorized_production_matrix_contains_exactly_30_distinct_cases():
    from campaign import production_jobs,key
    jobs=production_jobs()
    assert len(jobs)==len({key(j) for j in jobs})==30
    assert [j['case'] for j in jobs if j['method']=='mrdbscp']==list(range(1,11))
    assert all(j['dataset']=='copd' for j in jobs if j['method']=='mrdbscp')
    for ds in ['4dct','copd']:
        assert [j['case'] for j in jobs if j['method']=='corrfield' and j['dataset']==ds]==list(range(1,11))


def test_adapter_rejects_transposed_or_nonfinite_output():
    with pytest.raises(ValueError):corrfield_to_zyx(np.zeros((13,15,17,3),np.float32),(13,15,17))
    a=np.zeros((17,15,13,3),np.float32);a[0,0,0,0]=np.nan
    with pytest.raises(ValueError):corrfield_to_zyx(a,(13,15,17))
