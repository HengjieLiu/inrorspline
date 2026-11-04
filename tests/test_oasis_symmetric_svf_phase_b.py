from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
import symmetric_bspline_svf_phase_b as m
import run_oasis_symmetric_svf_phase_b as r
import queue_oasis_symmetric_svf_phase_b as queue


class PhaseBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_matrix_and_names(self):
        initial=m.initial_matrix()
        self.assertEqual(len(initial),15)
        self.assertEqual(len(m.smoke_matrix()),4)
        self.assertEqual(len({x.name() for x in initial+m.smoke_matrix()}),19)
        self.assertNotEqual(initial[0].name(),replace(initial[0],repeat=True).name())
        for spec in initial:
            self.assertEqual([x.lr for x in m.stages(spec)],[.03,.01,.003])
            self.assertEqual(spec.coefficient,2.)

    def test_invalid_specs(self):
        for spec in (m.Spec('bad'),m.Spec('base',1),m.Spec('amplitude',float('nan')),
                     m.Spec('amplitude',-1),m.Spec('balance',beta=100),
                     m.Spec('base',case_rank=1),m.Spec('base',swapped=True,repeat=True)):
            with self.assertRaises(ValueError):spec.validate()

    def test_cli(self):
        args=r.parse_args(['worker','--manifest','/tmp/m.json','--out-dir','/tmp/p78',
                          '--gpu','3','--arm','combined','--alpha','0.001','--beta','3','--swapped'])
        self.assertEqual((args.alpha,args.beta,args.swapped),(.001,3.,True))
        with self.assertRaises(SystemExit):
            r.parse_args(['worker','--manifest','/tmp/m','--out-dir','/tmp/p78','--gpu','0',
                          '--arm','amplitude','--alpha','-1'])

    def test_native_units_components_and_no_mask(self):
        scale=torch.tensor([20.,30.,40.],dtype=torch.float64)
        one=torch.ones(4,5,6,3,dtype=torch.float64)*torch.tensor([3.,4.,0.])/scale
        two=2*one
        amp,bal,(e1,e2,a1,a2)=m.penalties(one,two,scale)
        self.assertAlmostEqual(float(e1),25.)
        self.assertAlmostEqual(float(e2),100.)
        self.assertAlmostEqual(float(amp),125.)
        self.assertAlmostEqual(float(bal),25.,places=7)
        one[:2]=0
        self.assertAlmostEqual(float(m.penalties(one,two,scale)[2][0]),12.5)

    def test_zero_penalties_and_gradient(self):
        q=torch.zeros(2,3,4,3,dtype=torch.float64,requires_grad=True)
        amp,bal,_=m.penalties(q,q,torch.ones(3))
        self.assertEqual(float(amp+bal),0.)
        grad,=torch.autograd.grad(amp+bal,q)
        self.assertTrue(torch.equal(grad,torch.zeros_like(q)))

    def test_penalty_float64_gradcheck_and_exchange(self):
        torch.manual_seed(11)
        one=torch.randn(2,3,4,3,dtype=torch.float64,requires_grad=True)
        two=torch.randn_like(one,requires_grad=True)
        scale=torch.tensor([3.,5.,7.],dtype=torch.float64)
        fn=lambda a,b:sum(m.penalties(a,b,scale)[:2])
        self.assertTrue(torch.autograd.gradcheck(fn,(one,two),fast_mode=True))
        self.assertEqual(float(fn(one,two)),float(fn(two,one)))

    def test_common_opposing_identity(self):
        one,two=torch.randn(6,7,8,3,dtype=torch.float64),torch.randn(6,7,8,3,dtype=torch.float64)
        amp=m.penalties(one,two,torch.ones(3))[0]
        expected=2*((one+two)/2).square().sum(-1).mean()+2*((one-two)/2).square().sum(-1).mean()
        self.assertTrue(torch.allclose(amp,expected,atol=1e-12))
        self.assertEqual(float(m.penalties(one,one,torch.ones(3))[1]),0.)

    def test_objective_phase_a_parity_and_nonzero_swap(self):
        self.assertTrue(m.zero_parity('cpu')['passed'])

    def test_actual_velocity_half_not_parameter_energy(self):
        q=torch.ones(2,4,5,6,3)
        v1,v2=m.branch_controls(q)
        amp=m.penalties(v1,v2,torch.ones(3))[0]
        self.assertEqual(float(amp),1.5)

    def test_neighbors_boundaries_and_selection(self):
        self.assertEqual(m.neighbors(m.ALPHAS,m.ALPHAS[0]),m.ALPHAS[:3])
        self.assertEqual(m.neighbors(m.BETAS,m.BETAS[-1]),m.BETAS[-3:])
        rows=self.rows(m.initial_matrix())
        combined=m.combined_matrix(rows)
        self.assertEqual(len(combined),9)
        self.assertEqual({x.alpha for x in combined},set(m.ALPHAS[:3]))
        with self.assertRaises(ValueError):m.combined_matrix(rows[:-1])
        diag=m.diagnostic_matrix(rows+self.rows(combined))
        self.assertEqual(len(diag),4)
        self.assertTrue(all(x[0].swapped for x in diag[:3]))
        self.assertTrue(diag[-1][0].repeat)
        self.assertTrue(all(x[1]=='2' for x in diag))

    @staticmethod
    def rows(specs):
        return [{'spec':asdict(s),'dice':.8,'asd':.4,'hd95':2.,'gpu':'2','run_name':s.name()} for s in specs]

    def test_joint_history_and_selected_full_objective(self):
        torch.manual_seed(7)
        fixed,moving=torch.rand(12,16,20),torch.rand(12,16,20)
        spec=m.Spec('combined',.001,3.,smoke=True)
        stage=replace(m.stages(spec)[0],max_epochs=2,min_epochs=1)
        obj=m.Objective(fixed,moving,stage,spec)
        q=torch.zeros(2,*obj.control_shape,3)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'stage'
            best,info=m.optimize(q,obj,out)
            self.assertEqual(r.check_history(out/'history.csv',info,2.),3)
            self.assertEqual(info['amplitude_weight'],.001)
            self.assertEqual(info['balance_weight'],3.)
            self.assertLessEqual(info['best_loss'],info['initial_loss'])
            self.assertAlmostEqual(float(obj(best)[0]),info['best_loss'],places=6)

    def test_oom_classification_and_immutable_pinned_plan(self):
        self.assertTrue(queue.oom('torch.cuda.OutOfMemoryError: CUDA out of memory'))
        self.assertFalse(queue.oom('ValueError: loss replay mismatch'))
        with tempfile.TemporaryDirectory() as tmp:
            entries=[(m.Spec('amplitude',.001,swapped=True),'3','parent')]
            first=queue.plan(Path(tmp),'diagnostic',entries,[])
            self.assertEqual(first,queue.plan(Path(tmp),'diagnostic',entries,[]))
            self.assertEqual(first['entries'][0]['pinned_gpu'],'3')
            with self.assertRaises(RuntimeError):
                queue.plan(Path(tmp),'diagnostic',[(entries[0][0],'2','parent')],[])


if __name__=='__main__':unittest.main()
