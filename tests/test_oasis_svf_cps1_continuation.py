"""Approved two-cell continuation: proof gates and unchanged numerical delegation."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
import svf_transition_validation_v2 as v
import resume_oasis_svf_cps1_v2 as runner


def good_proof():
    tiny={'max_abs_voxel':1e-12,'rms_voxel':1e-14}
    return {'velocity':tiny,'integrated_field':tiny,
            'rounded_controls_integrated_field':{'max_abs_voxel':2e-5,'rms_voxel':1e-7},'passed':True}


class ProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        cp=torch.zeros(6,6,6,3)
        cf=v.s.c.cubic_bspline_dyadic_knot_insert_3d(cp,(8,8,8))
        z=torch.zeros(40,40,40,3)
        near=z.clone();near[0,0,0,0]=1.1e-5
        return cp,cf,z,z,z,near,1,torch.ones(3)*100

    def test_approved_limits(self):
        v.check_physical({'max_abs_voxel':.002,'rms_voxel':1e-5})
        for bad in ({'max_abs_voxel':.002001,'rms_voxel':0.},
                    {'max_abs_voxel':.001,'rms_voxel':1.001e-5},
                    {'max_abs_voxel':float('nan'),'rms_voxel':0.}):
            with self.assertRaises(RuntimeError):v.check_physical(bad)

    def test_new_max_requires_fp64_and_actual_rounded_controls(self):
        with patch.object(v,'fp64_reference',return_value=good_proof()) as proof:
            result=v.verify_transition(*self.inputs())
        proof.assert_called_once()
        self.assertTrue(result['passed'])
        self.assertGreater(result['integrated_field']['max_abs_voxel'],.001)
        for key,field,value in [('velocity','max_abs_voxel',1.1e-9),
                                ('integrated_field','max_abs_voxel',1.1e-8),
                                ('rounded_controls_integrated_field','max_abs_voxel',1.1e-4),
                                ('rounded_controls_integrated_field','rms_voxel',1.1e-6)]:
            proof=good_proof();proof[key]={**proof[key],field:value}
            with patch.object(v,'fp64_reference',return_value=proof):
                with self.assertRaises(RuntimeError):v.verify_transition(*self.inputs())

    def test_proof_exception_cannot_be_bypassed(self):
        with patch.object(v,'fp64_reference',side_effect=RuntimeError('bad proof')):
            with self.assertRaises(RuntimeError):v.verify_transition(*self.inputs())

    def test_wrong_controls_nonfinite_and_velocity_change_fail(self):
        for index,edit in [(1,lambda x:x+.001),(1,lambda x:torch.full_like(x,float('nan'))),
                           (3,lambda x:x+.001),(5,lambda x:torch.full_like(x,float('inf')))]:
            args=list(self.inputs());args[index]=edit(args[index])
            with self.assertRaises(RuntimeError):v.verify_transition(*args)

    def test_prefix_dispatch_unchanged(self):
        args=list(self.inputs());args[6]=2
        with patch.object(v.old,'verify_transition',return_value={'legacy':True}) as old:
            self.assertEqual(v.verify_transition(*args),{'legacy':True})
        old.assert_called_once()

    def test_fp64_real_rounded_control_proof(self):
        shape=(16,24,16);shapes=v.s.c.build_nested_cp_shapes(shape,8,(8,4,2,1))
        torch.manual_seed(1);cp=torch.randn(*shapes[2],3)*.02
        cf=v.s.c.cubic_bspline_dyadic_knot_insert_3d(cp,shapes[1])
        proof=v.fp64_reference(cp,cf,1,shape,torch.tensor([7.5,11.5,7.5]))
        self.assertTrue(proof['passed'])
        self.assertLess(proof['integrated_field']['max_abs_voxel'],1e-8)
        self.assertLess(proof['rounded_controls_integrated_field']['max_abs_voxel'],1e-4)


class ContinuationTests(unittest.TestCase):
    def test_scope_and_cli(self):
        self.assertEqual(runner.spec_for(2).alpha,.01)
        with self.assertRaises(ValueError):runner.spec_for(0)
        args=runner.parse_args(['worker','--manifest','/tmp/m.json','--out-dir','/tmp/run',
                               '--case-rank','5','--gpu','3'])
        old=runner.worker_args(args)
        self.assertEqual((old.schedule,old.regularize,old.alpha,old.integration_steps),('8421','displacement',.01,7))
        self.assertFalse(old.smoke)

    def test_context_restores_original_numerical_objects_after_error(self):
        old=(runner.r.transition,runner.r.s.optimize,runner.r.setup,runner.r.s.Objective)
        with self.assertRaises(ValueError):
            with runner.worker_adapter({},{}):
                self.assertIs(runner.r.s.Objective,old[3])
                raise ValueError('test')
        self.assertEqual(old,(runner.r.transition,runner.r.s.optimize,runner.r.setup,runner.r.s.Objective))

    def test_only_stage3_calls_original_optimizer(self):
        initial=torch.zeros(2,2,2,3)
        with patch.object(runner,'replay_prefix',return_value=('prefix',{})) as replay, \
             patch.object(runner,'OPTIMIZE',return_value=('optimized',{})) as optimize:
            with runner.worker_adapter({},{}):
                for index in range(4):
                    objective=SimpleNamespace(stage=SimpleNamespace(index=index))
                    runner.r.s.optimize(initial,objective,Path('/tmp/unused'))
        self.assertEqual(replay.call_count,3)
        optimize.assert_called_once()
        self.assertEqual(optimize.call_args.args[1].stage.index,3)

    def test_adapter_keeps_legacy_policy_for_provenance_check(self):
        original=runner.r.transition
        for fixture in (False,True):
            with runner.worker_adapter({}, {},fixture=fixture):
                self.assertEqual(runner.r.transition.POLICY,original.POLICY)
                self.assertEqual(runner.v2.POLICY['version'],2)
            self.assertIs(runner.r.transition,original)

    def test_prefix_copy_has_no_adam_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'new'
            cp=torch.ones(2,2,2,3);initial=torch.zeros_like(cp)
            info={'best_loss':1.,'best_data_loss':.75,'best_reg_loss':.25}
            torch.save({'velocity_controls_norm_xyz':cp,'summary':info},source/'best.pt')
            for name in runner.COPY_NAMES[1:]: (source/name).write_text('preserved')
            field=source/'field.npy';np.save(field,cp.numpy())
            block={'optimizer':info,'initial_loss_data_reg':[1.,.75,.25],'dvf_path':str(field)}
            class Objective:
                stage=SimpleNamespace(index=2)
                def __call__(self,x):return 1.,.75,.25
                def field(self,x,native=False):return x
            with patch.object(runner,'check_prefix_block',return_value=(source,block)), \
                 patch.object(torch.optim,'Adam',side_effect=AssertionError('must not optimize prefix')):
                result,metadata=runner.replay_prefix(initial,Objective(),out,{}, {})
            self.assertTrue(torch.equal(result,cp));self.assertEqual(metadata,info)
            for name in runner.COPY_NAMES:self.assertEqual((source/name).read_bytes(),(out/name).read_bytes())

    def test_preservation_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);original=folder/'original';original.write_text('kept')
            runner.r.atomic_json(folder/'preserved_files.json',{str(original):runner.r.sha(original)})
            manifest={'continuation_v2':{'inventory_sha256':runner.r.sha(folder/'preserved_files.json')}}
            runner.verify_inventory(folder,manifest)
            original.write_text('changed')
            with self.assertRaises(RuntimeError):runner.verify_inventory(folder,manifest)


if __name__=='__main__':unittest.main()
