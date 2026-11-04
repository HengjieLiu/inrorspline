"""P79 scoped continuation, mandatory proof, immutable prefix and report checks."""
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
import svf_transition_validation_v3 as v
import resume_oasis_bspline842_first100_v3 as runner


def proof():
    tiny={'max_abs_voxel':1e-12,'rms_voxel':1e-14}
    return {'velocity':tiny,'integrated_field':tiny,
            'rounded_controls_integrated_field':{'max_abs_voxel':1.23e-4,'rms_voxel':1.8e-7},'passed':True}


class ProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def inputs(self):
        cp=torch.zeros(6,6,6,3);cf=v.s.c.cubic_bspline_dyadic_knot_insert_3d(cp,(8,8,8))
        z=torch.zeros(40,40,40,3);near=z.clone();near[0,0,0,0]=1.2e-5
        return [cp,cf,z,z,z,near,4,torch.ones(3)*100]

    def test_approved_limits(self):
        v.check_physical({'max_abs_voxel':.002,'rms_voxel':1e-5})
        for bad in ({'max_abs_voxel':.002001,'rms_voxel':0},
                    {'max_abs_voxel':.001,'rms_voxel':1.001e-5},
                    {'max_abs_voxel':float('nan'),'rms_voxel':0}):
            with self.assertRaises(RuntimeError):v.check_physical(bad)
        v.check_reference(proof())

    def test_fallback_requires_actual_control_proof(self):
        with patch.object(v,'fp64_reference',return_value=proof()) as check:
            result=v.verify_transition(*self.inputs())
        check.assert_called_once();self.assertFalse(result['original_field_gate_passed'])
        for key,field,value in [('velocity','max_abs_voxel',1.1e-9),
                                ('integrated_field','max_abs_voxel',1.1e-8),
                                ('rounded_controls_integrated_field','max_abs_voxel',2.01e-4),
                                ('rounded_controls_integrated_field','rms_voxel',1.01e-6)]:
            bad=proof();bad[key]={**bad[key],field:value}
            with patch.object(v,'fp64_reference',return_value=bad):
                with self.assertRaises(RuntimeError):v.verify_transition(*self.inputs())

    def test_physical_cap_even_when_relative_allclose_passes(self):
        args=self.inputs();args[4]=torch.ones_like(args[4]);args[5]=args[4]+3e-5
        self.assertTrue(torch.allclose(args[4],args[5],rtol=1e-4,atol=2e-6))
        with self.assertRaises(RuntimeError):v.verify_transition(*args)

    def test_wrong_controls_nonfinite_velocity_or_scope_fail(self):
        for index,edit in [(1,lambda x:x+.001),(1,lambda x:torch.full_like(x,float('nan'))),
                           (3,lambda x:x+.001),(5,lambda x:torch.full_like(x,float('inf')))]:
            args=self.inputs();args[index]=edit(args[index])
            with self.assertRaises(RuntimeError):v.verify_transition(*args)
        args=self.inputs();args[6]=1
        with self.assertRaises(ValueError):v.verify_transition(*args)
        with patch.object(v,'fp64_reference',side_effect=RuntimeError('failed proof')):
            with self.assertRaises(RuntimeError):v.verify_transition(*self.inputs())

    def test_real_fp64_proof_both_transitions(self):
        shape=(16,24,16);shapes=v.s.c.build_nested_cp_shapes(shape,8,(8,4,2))
        torch.manual_seed(1)
        for cps in (4,2):
            cp=torch.randn(*shapes[2*cps],3)*.02
            cf=v.s.c.cubic_bspline_dyadic_knot_insert_3d(cp,shapes[cps])
            result=v.fp64_reference(cp,cf,cps,shape,torch.tensor([7.5,11.5,7.5]))
            self.assertTrue(result['passed'])
            self.assertLess(result['integrated_field']['max_abs_voxel'],1e-8)


class ContinuationTests(unittest.TestCase):
    def test_exact_scope_cli(self):
        self.assertEqual(runner.spec_for(29).alpha,.01)
        self.assertEqual(runner.TARGETS,{29:{'resume_index':1,'gpu':'3'},44:{'resume_index':2,'gpu':'2'}})
        for bad in (0,28,45,True):
            with self.assertRaises(ValueError):runner.spec_for(bad)
        args=runner.parse_args(['worker','--manifest','/tmp/m','--out-dir','/tmp/job','--case-rank','29','--gpu','3','--fixture'])
        self.assertTrue(args.fixture)

    def test_only_unfinished_stages_call_original_optimizer(self):
        for rank,expected in ((29,[1,2]),(44,[2])):
            with patch.object(runner,'replay_prefix',return_value=('replayed',{})) as replay, \
                 patch.object(runner,'OPTIMIZE',return_value=('optimized',{})) as optimize:
                with runner.adapter(runner.TARGETS[rank],{}):
                    for index in range(3):
                        obj=SimpleNamespace(stage=SimpleNamespace(index=index))
                        runner.r.s.optimize(torch.zeros(1),obj,Path('/tmp/unused'))
            self.assertEqual(replay.call_count,runner.TARGETS[rank]['resume_index'])
            self.assertEqual([x.args[1].stage.index for x in optimize.call_args_list],expected)

    def test_old_transitions_retained_and_context_restored(self):
        old=runner.r.transition,runner.r.s.optimize,runner.r.s.stages,runner.r.s.Objective
        with patch.object(old[0],'verify_transition',return_value={'old':True}) as original, \
             patch.object(v,'verify_transition',return_value={'v3':True}) as new:
            with runner.adapter(runner.TARGETS[44],{}):
                self.assertEqual(runner.r.transition.POLICY,old[0].POLICY)
                self.assertEqual(runner.r.transition.verify_transition(*([None]*6+[4,None])),{'old':True})
                self.assertEqual(runner.r.transition.verify_transition(*([None]*6+[2,None])),{'v3':True})
            original.assert_called_once();new.assert_called_once()
        with self.assertRaises(ValueError):
            with runner.adapter(runner.TARGETS[29],{}):raise ValueError('test')
        self.assertEqual(old,(runner.r.transition,runner.r.s.optimize,runner.r.s.stages,runner.r.s.Objective))

    def test_smokes_change_only_missing_stage_budgets(self):
        from dataclasses import dataclass
        @dataclass
        class Stage:
            index:int
            max_epochs:int=400
            min_epochs:int=40
            lr:float=.03
        stages=[Stage(i) for i in range(3)]
        with patch.object(runner,'STAGES',return_value=stages):
            with runner.adapter(runner.TARGETS[29],{},True):got=runner.r.s.stages(None)
        self.assertEqual([x.max_epochs for x in got],[400,1,1])
        self.assertEqual([x.lr for x in got],[.03]*3)

    def test_prefix_replay_copies_exact_bytes_without_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'copy'
            cp=torch.ones(2,2,2,3);info={'best_loss':1.,'best_data_loss':.75,'best_reg_loss':.25}
            torch.save({'velocity_controls_norm_xyz':cp,'summary':info},source/'best.pt')
            for name in runner.COPY_NAMES[1:]:(source/name).write_text('preserved')
            field=source/'field.npy';np.save(field,cp.numpy())
            block={'optimizer':info,'initial_loss_data_reg':[1.,.75,.25],'dvf_path':str(field)}
            class Objective:
                stage=SimpleNamespace(index=0)
                def __call__(self,x):return 1.,.75,.25
                def field(self,x,native=False):return x
            with patch.object(runner,'check_prefix',return_value=(source,block)), \
                 patch.object(torch.optim,'Adam',side_effect=AssertionError('No prefix optimizer')):
                copied,metadata=runner.replay_prefix(torch.zeros_like(cp),Objective(),out,{}, {})
            self.assertTrue(torch.equal(cp,copied));self.assertEqual(metadata,info)
            for name in runner.COPY_NAMES:self.assertEqual((source/name).read_bytes(),(out/name).read_bytes())

    def test_preservation_detects_changed_failed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'failed.log';old.write_text('preserved')
            inventory=root/'inventory.json'
            runner.e.atomic_json(inventory,{'metadata_sha256':{},'receipt_paths':[],
                                           'failed_attempt_sha256':{str(old):runner.e.sha(old)}})
            m={'continuation_v3':{'inventory_path':str(inventory),'inventory_sha256':runner.e.sha(inventory)}}
            with patch.object(runner,'assert_manifest'):
                self.assertTrue(runner.verify_preserved(m)['passed'])
                old.write_text('changed')
                with self.assertRaises(RuntimeError):runner.verify_preserved(m)


if __name__=='__main__':unittest.main()
