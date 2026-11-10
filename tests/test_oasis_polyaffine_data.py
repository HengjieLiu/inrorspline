import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE=Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_polyaffine_fieldfit/data.py"
spec=importlib.util.spec_from_file_location("polyaffine_data",MODULE)
data=importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class DataTests(unittest.TestCase):
    def test_reorientation_physical_points(self):
        raw=np.arange(4*5*6).reshape(4,5,6)
        affine=np.array([[-2,0,0,4],[0,0,3,-8],[0,-4,0,9],[0,0,0,1]],float)
        arr,h=data.reorient(raw,affine)
        for d,w,y in [(0,0,0),(3,4,5),(1,2,3)]:
            self.assertEqual(arr[d,y,w],raw[3-d,4-w,y])
            np.testing.assert_allclose(h @ [d,y,w,1],affine @ [3-d,4-w,y,1])

    def test_different_affines_map_roundtrip(self):
        rng=np.random.default_rng(4)
        norm=rng.normal(size=(5,6,7,3)).astype(np.float32)*.1
        h_f=np.array([[1,0,0,-20],[0,2,0,3],[0,0,3,7],[0,0,0,1]],float)
        h_m=np.array([[0,-2,0,5],[1,0,0,-4],[0,0,1.5,3],[0,0,0,1]],float)
        mm=data.normalized_to_mm(norm,h_f,h_m)
        recovered=data.mm_to_normalized(mm,h_f,h_m)
        np.testing.assert_allclose(recovered,norm,atol=5e-7,rtol=0)
        index=np.array([2,3,4]);delta=(norm[2,3,4]*np.array([3,2.5,2]))[::-1]
        expected=(h_m @ np.r_[index+delta,1]-h_f @ np.r_[index,1])[:3]
        np.testing.assert_allclose(mm[2,3,4],expected,atol=2e-6)

    def test_xyz_grid_channel_order(self):
        norm=np.zeros((5,7,9,3),np.float32);norm[...,0]=1
        mm=data.normalized_to_mm(norm,np.eye(4),np.eye(4))
        np.testing.assert_array_equal(mm[...,0],0)
        np.testing.assert_array_equal(mm[...,1],0)
        np.testing.assert_array_equal(mm[...,2],4)

    def test_samples_disjoint_deterministic_foreground(self):
        mask=np.ones((17,19,23),bool);mask[:2]=False
        a=data.stratified_samples(mask,caps=(240,60,240));b=data.stratified_samples(mask,caps=(240,60,240))
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        ids=np.concatenate(list(a.values()))
        self.assertEqual(len(np.unique(ids)),540)
        self.assertTrue(mask.reshape(-1)[ids].all())
        small=data.stratified_samples(np.ones((3,4,5),bool),caps=(240,60,240))
        self.assertEqual(sum(map(len,small.values())),60)

    def test_selector_rejects_wrong_pair_and_duplicates(self):
        pairs=[("0001","0002")]*10
        row=dict(case_rank=0,fixed_id="0001",moving_id="0002",direction="moving_to_fixed",representation="SITReg")
        reports={"p43":dict(case_metrics=[row,row],candidates=[]),"p76":dict(candidates=[]),"p74":dict(candidates=[])}
        with self.assertRaises(ValueError):data.select_sources(reports,pairs)


if __name__=="__main__":unittest.main()
