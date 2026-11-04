import csv
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import symmetric_bspline_svf as m
import symmetric_svf_diagnostics as diag
import run_oasis_symmetric_svf as r


class SymmetricSVFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1)

    def test_matrix_names_schedule_and_cli(self):
        specs = m.matrix()+m.matrix(True)
        self.assertEqual(len({s.name() for s in specs}), 8)
        for spec in specs:
            stages = m.stages(spec)
            self.assertEqual([s.cps for s in stages], [8,4,2])
            self.assertEqual([s.lr for s in stages], [.03,.01,.003])
            self.assertTrue(all(s.max_lr_reductions == 0 for s in stages))
        for spec in (m.Spec("bad"), m.Spec("A1", case_rank=1)):
            with self.assertRaises(ValueError): spec.validate()
        args = r.parse_args(["worker", "--arm", "A1_swap", "--manifest", "/tmp/m.json",
                             "--out-dir", "/tmp/p77", "--gpu", "3"])
        self.assertEqual(args.arm, "A1_swap")

    def test_ncc_values_both_gradients_swap_and_legacy(self):
        a = torch.rand(1,1,10,12,14, requires_grad=True)
        b = torch.rand_like(a, requires_grad=True)
        pair = m.NCC9Pair("cpu")
        value = pair(a,b)
        ga, gb = torch.autograd.grad(value, (a,b))
        oldb = m.d.NCC9(a)(b)
        olda = m.d.NCC9(b)(a)
        gbo, = torch.autograd.grad(oldb, b)
        gao, = torch.autograd.grad(olda, a)
        self.assertTrue(torch.equal(value, oldb))
        self.assertTrue(torch.equal(value, pair(b,a)))
        self.assertTrue(torch.allclose(ga, gao, rtol=1e-5, atol=2e-6))
        self.assertTrue(torch.allclose(gb, gbo, rtol=1e-5, atol=2e-6))
        self.assertGreater(float(ga.abs().max()), 0)
        self.assertGreater(float(gb.abs().max()), 0)
        self.assertFalse(m.d.NCC9(a).fixed.requires_grad)

    def test_ncc_float64_gradcheck(self):
        a = torch.rand(1,1,3,4,5,dtype=torch.float64,requires_grad=True)
        b = torch.rand_like(a,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(m.NCC9Pair("cpu",torch.float64), (a,b), fast_mode=True))

    def test_diffusion_factor_two_and_common_mode(self):
        v1 = torch.randn(8,9,10,3,dtype=torch.float64)
        v2 = torch.randn_like(v1)
        total = 2*(m.diffusion(v1)+m.diffusion(v2))
        expected = m.diffusion(v1-v2)+4*m.diffusion((v1+v2)/2)
        self.assertTrue(torch.allclose(total, expected, atol=1e-12))
        self.assertTrue(torch.allclose(2*(m.diffusion(v1/2)+m.diffusion(-v1/2)),m.diffusion(v1),atol=1e-12))
        self.assertEqual(float(m.diffusion(torch.ones_like(v1)*3)),0.)

    def test_magnitude_units_reduction_and_zero(self):
        v = torch.ones(4,5,6,3)*torch.tensor([3.,4.,0.])
        values = m.energies(v,-v)
        self.assertEqual(values["E1_mm2"],25.)
        self.assertAlmostEqual(values["a1_mm"],5.,places=6)
        self.assertEqual(values["common_mode_rms_mm"],0.)
        self.assertEqual(values["relative_rms_imbalance"],0.)
        zero = m.energies(torch.zeros_like(v),torch.zeros_like(v))
        self.assertTrue(zero["near_zero_both"])
        self.assertTrue(np.isfinite(zero["velocity_cosine"]))

    def test_branch_shapes_and_half_scale(self):
        q = torch.randn(2,4,5,6,3,requires_grad=True)
        a,b=m.branch_controls(q)
        self.assertTrue(torch.equal(a,q[0]/2))
        grad,=torch.autograd.grad((a+b).sum(),q)
        self.assertTrue(torch.equal(grad,torch.ones_like(q)/2))
        a,b=m.branch_controls(q[:1],True)
        self.assertTrue(torch.equal(a,-b))
        with self.assertRaises(ValueError):m.branch_controls(q[:1])

    def test_identity_translation_composition_and_boundary(self):
        shape=(9,11,13)
        grid=m.c.make_coordinate_grid(shape,"cpu")
        v1=torch.zeros_like(grid)+torch.tensor([.02,.03,-.01])
        v2=torch.zeros_like(grid)+torch.tensor([-.04,.01,.02])
        f=m.s.compose((m.s.exp_velocity(v1),m.s.exp_velocity(-v2)),grid)-grid
        self.assertTrue(torch.allclose(f,v1-v2,atol=3e-7))
        outside=torch.tensor([[1.4,-1.2,1.5]])
        self.assertTrue(torch.allclose(m.s.compose((v1,v2),outside),outside+v1[0,0,0]+v2[0,0,0]))

    def test_noncommuting_order_and_inverse(self):
        grid=m.c.make_coordinate_grid((13,15,17),"cpu")
        v1=torch.zeros_like(grid);v1[...,0]=.05*grid[...,1]
        v2=torch.zeros_like(grid);v2[...,1]=.04*grid[...,0]
        p1,p2=m.s.exp_velocity(v1),m.s.exp_velocity(v2)
        n1,n2=m.s.exp_velocity(-v1),m.s.exp_velocity(-v2)
        points=grid[3:-3,3:-3,3:-3].reshape(-1,3)
        right=m.s.compose((p1,n2),points)
        wrong=m.s.compose((n2,p1),points)
        self.assertGreater(float((right-wrong).abs().max()),1e-4)
        back=m.s.compose((p2,n1),right)
        self.assertLess(float((back-points).abs().max()),1e-5)

    def test_objective_both_controls_and_input_swap(self):
        a,b=torch.rand(16,20,24),torch.rand(16,20,24)
        spec=m.Spec("A1",True);stage=m.stages(spec)[0]
        obj=m.Objective(b,a,stage,spec)
        q=torch.zeros(2,*obj.control_shape,3,requires_grad=True)
        loss,data,reg=obj(q)
        grad,=torch.autograd.grad(loss,q)
        self.assertGreater(float(grad[0].abs().max()),0)
        self.assertGreater(float(grad[1].abs().max()),0)
        swapped=m.Objective(a,b,stage,m.Spec("A1_swap",True))
        q2=q.detach().flip(0).requires_grad_()
        ls=swapped(q2)[0];gs,=torch.autograd.grad(ls,q2)
        self.assertTrue(torch.equal(loss,ls))
        self.assertTrue(torch.allclose(grad,gs.flip(0),atol=1e-6,rtol=1e-5))
        self.assertEqual(float(reg),0.)

    def test_independent_prolongation(self):
        shape=(16,24,24)
        shapes=m.c.build_nested_cp_shapes(shape,8,(8,4,2))
        q=torch.randn(2,*shapes[8],3)*.0001
        for cps in (4,2):
            fine=m.prolong(q,shapes[cps])
            for old,new in zip(q,fine):
                a=m.d.AlignedBSpline(old.shape[:3],shape,shape,cps*2,"cpu")(old)
                b=m.d.AlignedBSpline(new.shape[:3],shape,shape,cps,"cpu")(new)
                self.assertTrue(torch.allclose(a,b,rtol=1e-5,atol=1e-9))
            q=fine

    def test_joint_optimizer_history_and_selection(self):
        a,b=torch.rand(12,16,20),torch.rand(12,16,20)
        spec=m.Spec("A1",True)
        stage=replace(m.stages(spec)[0],max_epochs=2,min_epochs=1)
        obj=m.Objective(b,a,stage,spec)
        initial=torch.zeros(2,*obj.control_shape,3)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"stage"
            best,info=m.optimize(initial,obj,out)
            self.assertEqual(info["actual_epochs"],2)
            self.assertEqual(r.check_history(out/"history.csv",info,2.),3)
            state=torch.load(out/"best.pt",weights_only=False)
            self.assertTrue(torch.equal(best,state["twice_branch_controls_norm_xyz"]))
            self.assertLessEqual(info["best_loss"],info["initial_loss"])
            self.assertTrue(torch.equal(initial,torch.zeros_like(initial)))

    def test_swap_batch_masks_labels_and_roundtrip(self):
        batch={"fixed":torch.ones(2,3,4),"moving":torch.zeros(2,3,4),
               "fixed_seg":torch.ones(2,3,4),"moving_seg":torch.zeros(2,3,4),
               "fixed_id":"0333","moving_id":"0115",
               "fixed_image_mask":torch.ones(2,3,4),"moving_image_mask":torch.zeros(2,3,4)}
        batch["fixed_mask"]=batch["fixed_image_mask"]
        result=diag.swapped_batch(batch)
        self.assertEqual(result["fixed_id"],"0115")
        self.assertTrue(torch.equal(result["fixed_mask"],batch["moving_image_mask"]))
        self.assertTrue(torch.equal(diag.swapped_batch(result)["fixed"],batch["fixed"]))


if __name__=="__main__":unittest.main()
