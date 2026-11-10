"""Resource profile coverage and admission-reservation contracts."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_polyaffine_fieldfit"))
import preflight


class PreflightTests(unittest.TestCase):
    def test_exact_profile_coverage(self):
        specs=preflight.profiles()
        self.assertEqual(len(specs),17)
        self.assertEqual(len({s['key'] for s in specs}),17)
        for s in specs:
            self.assertEqual(s['budget_cap'],1536 if s['mode']=='d1' else 3072)

    def test_memory_headroom_and_rounding(self):
        self.assertEqual(preflight.required_memory_mib(0),2048)
        self.assertEqual(preflight.required_memory_mib(2049),4352)
        for bad in (-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):preflight.required_memory_mib(bad)

    def test_failure_is_retained_and_blocks_full_preflight(self):
        receipts=[dict(profile=s,status='passed',required_mib=4096) for s in preflight.profiles()]
        self.assertTrue(preflight.summarize(receipts)['passed'])
        receipts[3]['status']='failed';receipts[3]['error']='fixture failure'
        result=preflight.summarize(receipts)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_profiles'],[receipts[3]['profile']['key']])
        self.assertNotIn(receipts[3]['profile']['key'],result['memory_profiles_mib'])
        self.assertEqual(result['memory_profiles_mib']['d0__global_affine'],4096)

    def test_coarse_grid_preserves_physical_endpoints(self):
        shape=(160,224,192)
        affine=np.array([[0,0,-2,12],[0,3,0,-3],[1,0,0,9],[0,0,0,1]],dtype=float)
        coarse,h=preflight.coarse_geometry(shape,affine)
        np.testing.assert_allclose(affine[:3,:3]@np.array(shape-1 if isinstance(shape,np.ndarray) else [n-1 for n in shape])+affine[:3,3],
                                   h[:3,:3]@np.array([n-1 for n in coarse])+h[:3,3])


if __name__=='__main__':unittest.main()
