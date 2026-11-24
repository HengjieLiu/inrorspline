"""Combined-loss accounting, zero-weight compatibility and rank0 scope guards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments.oasis_regularizer_sweep import losses as l,worker as w
from scripts.experiments.oasis_diffusion_kl_rank0.model import CombinedObjective,activate,optimize
from scripts.experiments.oasis_diffusion_kl_rank0.protocol import GAMMAS,setting


class CombinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        w.torch,w.np,w.l,w.c,w.d=torch,np,l,l.c,l.d

    def fixtures(self):
        torch.manual_seed(117)
        shape=(24,32,24); cs=l.c.build_nested_cp_shapes(shape,8,[8])[8]
        return shape,cs,torch.rand(shape),torch.rand(shape)

    def test_gamma_zero_value_gradient_each_pyramid_stage(self):
        shape,_,fixed,moving=self.fixtures()
        shapes=l.c.build_nested_cp_shapes(shape,8,[8,4,2])
        with patch.object(l.Derivatives,"jacobian",side_effect=AssertionError("Gamma0 must bypass KL")):
            for i,(cps,factor) in enumerate(((8,4),(4,2),(2,1))):
                stage=l.d.StageSpec(i,cps,factor,.003,2,1,25,0)
                cp=(torch.randn(*shapes[cps],3)*.001).requires_grad_()
                for backend in ("fd","analytic"):
                    arm="diffusion_"+backend
                    old=l.Objective(fixed,moving,1.,stage,shapes[cps],arm)
                    new=CombinedObjective(fixed,moving,1.,stage,shapes[cps],arm,gamma=0.)
                    for a,b in zip(old(cp),new(cp)): self.assertTrue(torch.equal(a,b))
                    a,=torch.autograd.grad(old(cp)[0],cp); b,=torch.autograd.grad(new(cp)[0],cp)
                    self.assertTrue(torch.equal(a,b))

    def test_positive_gamma_components_and_gradients(self):
        shape,cs,fixed,moving=self.fixtures(); stage=l.d.StageSpec(0,8,4,.03,2,1,25,0)
        cp=(torch.randn(*cs,3)*.01).requires_grad_()
        for backend in ("fd","analytic"):
            arm="diffusion_"+backend
            old=l.Objective(fixed,moving,1.,stage,cs,arm)
            for gamma in GAMMAS[1:]:
                obj=CombinedObjective(fixed,moving,1.,stage,cs,arm,gamma=gamma)
                total,data,reg=obj(cp); diff=obj.diffusion_component; kl=obj.kl_component
                expected_kl=l.kl_loss(l.Derivatives(cs,shape,obj.eval_shape,8,"cpu").jacobian(cp))
                self.assertTrue(torch.allclose(diff,old(cp)[2],atol=1e-9,rtol=1e-6))
                self.assertTrue(torch.equal(kl,expected_kl))
                self.assertTrue(torch.allclose(total,data+diff+gamma*kl,atol=1e-7,rtol=1e-6))
                actual,=torch.autograd.grad(total,cp,retain_graph=True)
                expected,=torch.autograd.grad(data+diff+gamma*kl,cp)
                self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-6))
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_zero_gamma_optimizer_parity_both_backends(self):
        shape,cs,fixed,moving=self.fixtures(); stage=l.d.StageSpec(0,8,1,.03,3,1,25,0)
        initial=torch.zeros(*cs,3)
        for backend in ("fd","analytic"):
            arm="diffusion_"+backend
            old=l.Objective(fixed,moving,1.,stage,cs,arm)
            new=CombinedObjective(fixed,moving,1.,stage,cs,arm,gamma=0.)
            with tempfile.TemporaryDirectory() as folder,patch.object(torch.cuda,"synchronize",return_value=None):
                a,am=w.optimize(initial,old,stage,Path(folder)/"old")
                b,bm=optimize(initial,new,stage,Path(folder)/"new")
                self.assertTrue(torch.equal(a,b))
                for key in ("best_loss","best_data_loss","best_reg_loss","actual_epochs","best_epoch","stopping_reason"):
                    self.assertEqual(am[key],bm[key])

    def test_adapter_restoration_and_rank_guard(self):
        original=(w.l.Objective,w.optimize,w.register,w.PROTOCOL)
        with activate(w,.01):
            self.assertEqual(w.PROTOCOL["gamma"],.01)
            with self.assertRaises(ValueError): w.register({},1,"warm",1,None,{})
        self.assertEqual((w.l.Objective,w.optimize,w.register,w.PROTOCOL),original)

    def test_requested_matrix_and_weight_guards(self):
        self.assertEqual(GAMMAS,(0.,.0001,.001,.01,.1,1.))
        self.assertEqual(len({setting(b,g) for b in ("fd","analytic") for g in GAMMAS}),12)
        with self.assertRaises(ValueError): setting("fd",.3)
        shape,cs,fixed,moving=self.fixtures(); stage=l.d.StageSpec(0,8,4,.03,1,1,25,0)
        with self.assertRaises(ValueError): CombinedObjective(fixed,moving,.3,stage,cs,"diffusion_fd",gamma=.01)


if __name__=="__main__": unittest.main()
