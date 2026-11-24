"""Mathematical and scheduling gates for the six-arm regularizer study."""
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments.oasis_regularizer_sweep import losses as l, protocol as p


class RegularizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_reference_values_and_gradients(self):
        j=torch.tensor([-.1,0,.01,.05,.5,1,2],dtype=torch.float64,requires_grad=True)
        expected=torch.tensor([7.495305501,4.095732274,3.821774951,2.845945660,.346573590,0,.693147181],dtype=j.dtype)
        actual=l.kl_density(j)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-9,rtol=0))
        grad,=torch.autograd.grad(actual.sum(),j)
        slopes=torch.tensor([-39.995732274,-27.995732274,-26.795732274,-21.995732274,-1.693147181,0,1.193147181],dtype=j.dtype)
        self.assertTrue(torch.allclose(grad,slopes,atol=1e-9,rtol=0))

    def test_exact_raw_and_c1_join(self):
        j=torch.tensor([.05,.1,.5,1.,20.],dtype=torch.float64)
        self.assertTrue(torch.equal(l.kl_density(j),(j-1)*j.log()))
        j=torch.tensor([.05-1e-9,.05,.05+1e-9],dtype=torch.float64,requires_grad=True)
        v=l.kl_density(j); g,=torch.autograd.grad(v.sum(),j)
        self.assertLess(float((v-v[1]).abs().max()),3e-8)
        self.assertLess(float((g-g[1]).abs().max()),5e-7)

    def test_gradcheck_and_negative_ordering(self):
        j=torch.tensor([-2.,-.1,0.,.02,.1,1.,2.],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(l.kl_loss,(j,)))
        g,=torch.autograd.grad(l.kl_density(j).sum(),j)
        self.assertTrue(bool((g[:4]<0).all()))
        self.assertTrue(bool((l.kl_density(j[:4]).diff()<0).all()))

    def test_mask_and_half(self):
        j=torch.tensor([0.,1.,2.]); mask=torch.tensor([1.,0.,3.])
        self.assertAlmostEqual(float(l.kl_loss(j,mask=mask)),float((l.kl_density(j)*mask).sum()/4),places=6)
        for invalid in (torch.zeros(3),torch.tensor([-1.,2.,1.]),torch.ones(2)):
            with self.assertRaises(ValueError): l.kl_loss(j,mask=invalid)
        for dtype in (torch.float16,torch.bfloat16):
            out=l.kl_density(j.to(dtype)); self.assertEqual(out.dtype,torch.float32)
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_fd_canonical_parity(self):
        torch.manual_seed(12)
        vox=torch.randn(9,10,11,3,dtype=torch.float64)*.1
        zyx=vox.numpy()[...,::-1].transpose(3,0,1,2)[None].copy()
        expected=l.c.jacobian_determinant_zyx(zyx)
        np.testing.assert_allclose(l.fd_jacobian(vox).numpy(),expected,rtol=1e-13,atol=1e-13)

    def test_known_fd_deformations(self):
        z,y,x=torch.meshgrid(*(torch.arange(n,dtype=torch.float64) for n in (8,9,10)),indexing="ij")
        xyz=torch.stack((x,y,z),-1)
        for matrix,det in ((torch.zeros(3,3),1.),(torch.diag(torch.tensor([.2,-.1,.4])),1.2*.9*1.4),
                           (torch.tensor([[0.,.4,0.],[0.,0.,0.],[0.,0.,0.]]),1.)):
            vox=xyz@matrix.double().T+torch.tensor([.5,.3,-.2])
            self.assertTrue(torch.allclose(l.fd_jacobian(vox),torch.full((4,5,6),det,dtype=vox.dtype),atol=2e-7,rtol=0))

    def control_affine(self,shape,cps,matrix):
        cs=l.c.build_nested_cp_shapes(shape,8,[8,4,2])[cps]
        axes=[(torch.arange(n,dtype=torch.float64)-1)*cps for n in cs]
        z,y,x=torch.meshgrid(*axes,indexing="ij")
        xyz=torch.stack((x,y,z),-1)
        native_scale=l.d.voxel_scale(shape,device="cpu",dtype=torch.float64)
        return ((xyz@matrix.T)/native_scale).requires_grad_(),cs

    def test_analytic_affine_all_pyramid_scales(self):
        shape=(24,32,24)
        matrix=torch.tensor([[.2,.3,0.],[0.,-.1,.2],[.1,0.,.15]],dtype=torch.float64)
        expected=torch.linalg.det(torch.eye(3)+matrix)
        for cps,factor in ((8,4),(4,2),(2,1)):
            cp,cs=self.control_affine(shape,cps,matrix)
            eval_shape=l.c.downsample_shape(shape,factor)
            op=l.Derivatives(cs,shape,eval_shape,cps,"cpu",torch.float64)
            j=op.jacobian(cp)
            self.assertTrue(torch.allclose(j,torch.full_like(j,expected),atol=2e-12,rtol=0))
            self.assertLess(float(op.energy(cp,"be")),1e-26)
            g,=torch.autograd.grad(l.kl_loss(j),cp)
            self.assertTrue(bool(torch.isfinite(g).all())); self.assertGreater(float(g.abs().max()),0)

    def test_analytic_identity_translation_shear(self):
        shape=(24,32,24)
        for matrix in (torch.zeros(3,3,dtype=torch.float64),torch.tensor([[0.,.4,0.],[0.,0.,0.],[0.,0.,0.]],dtype=torch.float64)):
            cp,cs=self.control_affine(shape,8,matrix); cp=cp+.02
            op=l.Derivatives(cs,shape,(6,8,6),8,"cpu",torch.float64)
            j=op.jacobian(cp)
            self.assertTrue(torch.allclose(j,torch.ones_like(j),atol=2e-14,rtol=0))
            self.assertLess(float(l.kl_loss(j)),1e-25)

    def test_analytic_native_oracle_and_gradients(self):
        torch.manual_seed(7)
        shape=(16,24,16); cs=l.c.build_nested_cp_shapes(shape,8,[8,4,2])[4]
        cp=(torch.randn(*cs,3)*.001).requires_grad_()
        op=l.Derivatives(cs,shape,shape,4,"cpu")
        for orders in ((0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0)):
            expected=l.c.expand_cubic_bspline_derivative(cp,shape,4,orders)[0].permute(1,2,3,0)*op.scale
            actual=op(cp,orders)
            self.assertTrue(torch.equal(actual,expected))
            a,=torch.autograd.grad(actual.square().mean(),cp,retain_graph=True)
            b,=torch.autograd.grad(expected.square().mean(),cp)
            self.assertTrue(torch.equal(a,b))

    def test_analytic_pyramid_independent_coordinate_differences(self):
        # Shift native evaluation coordinates analytically via a separate direct sum.
        torch.manual_seed(9); shape=(24,32,24); cs=l.c.build_nested_cp_shapes(shape,8,[8])[8]
        cp=torch.randn(*cs,3,dtype=torch.float64)*.01
        op=l.Derivatives(cs,shape,(6,8,6),8,"cpu",torch.float64)
        def point(pos):
            ws=[l.basis(torch.tensor(v,dtype=torch.float64)/8+1-torch.arange(n),0) for v,n in zip(pos,cs)]
            return torch.einsum("i,j,k,ijkc->c",*ws,cp)*op.scale
        idx=(2,3,2); pos=[i*(n-1)/(s-1) for i,n,s in zip(idx,shape,op.shape)]
        for axis in range(3):
            step=(shape[axis]-1)/(op.shape[axis]-1); h=1e-4
            low=list(pos); high=list(pos); low[axis]-=h*step; high[axis]+=h*step
            expected=(point(high)-point(low))/(2*h)
            orders=tuple(int(a==axis) for a in range(3))
            self.assertTrue(torch.allclose(op(cp,orders)[idx],expected,atol=1e-10,rtol=1e-7))

    def test_control_gradcheck_and_negative_j(self):
        shape=(24,24,24); matrix=torch.diag(torch.tensor([-1.2,.1,.2],dtype=torch.float64))
        cp,cs=self.control_affine(shape,8,matrix)
        op=l.Derivatives(cs,shape,(6,6,6),8,"cpu",torch.float64)
        self.assertTrue(bool((op.jacobian(cp)<0).all()))
        # Directional check traverses actual spline controls without 3*N probes.
        direction=torch.randn_like(cp)
        scalar=torch.tensor(.0001,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s:l.kl_loss(op.jacobian(cp.detach()+s*direction)),(scalar,)))

    def test_objective_fd_parity_and_zero_bypass(self):
        torch.manual_seed(8); shape=(16,24,16); cs=l.c.build_nested_cp_shapes(shape,8,[8])[8]
        a,b=torch.rand(shape),torch.rand(shape)
        stage=l.d.StageSpec(0,8,1,.03,3,1,25,0)
        cp=(torch.randn(*cs,3)*.001).requires_grad_()
        old=l.d.StageObjective(a,b,.3,stage,cs); new=l.Objective(a,b,.3,stage,cs,"diffusion_fd")
        for x,y in zip(old(cp),new(cp)): self.assertTrue(torch.equal(x,y))
        g1,=torch.autograd.grad(old(cp)[0],cp); g2,=torch.autograd.grad(new(cp)[0],cp)
        self.assertTrue(torch.equal(g1,g2))
        with patch.object(l,"fd_jacobian",side_effect=AssertionError("Zero must bypass")):
            for arm in p.ARMS:
                objective=l.Objective(a,b,0,stage,cs,arm)
                total,data,reg=objective(cp)
                self.assertTrue(torch.equal(total,data)); self.assertEqual(float(reg),0)

    def test_all_six_losses_have_control_gradients(self):
        torch.manual_seed(19); shape=(24,32,24); cs=l.c.build_nested_cp_shapes(shape,8,[8])[8]
        stage=l.d.StageSpec(0,8,4,.03,1,1,25,0)
        a,b=torch.rand(shape),torch.rand(shape)
        for arm in p.ARMS:
            cp=(torch.randn(*cs,3)*.005).requires_grad_()
            objective=l.Objective(a,b,.1,stage,cs,arm)
            loss,data,reg=objective(cp); grad,=torch.autograd.grad(reg,cp)
            self.assertTrue(bool(torch.isfinite(grad).all()),arm)
            self.assertGreater(float(grad.abs().max()),0,arm)

    def test_optimizer_best_state_and_trajectory_parity(self):
        from scripts.experiments.oasis_regularizer_sweep import worker as w
        w.torch,w.np,w.c,w.d=torch,np,l.c,l.d
        torch.manual_seed(44); shape=(16,24,16); cs=l.c.build_nested_cp_shapes(shape,8,[8])[8]
        stage=l.d.StageSpec(0,8,1,.03,3,1,25,0)
        fixed,moving=torch.rand(shape),torch.rand(shape)
        initial=torch.zeros(*cs,3)
        old=l.d.StageObjective(fixed,moving,.3,stage,cs); new=l.Objective(fixed,moving,.3,stage,cs,"diffusion_fd")
        with tempfile.TemporaryDirectory() as root,patch.object(torch.cuda,"synchronize",return_value=None):
            a,am=l.d.optimize_stage(initial,old,stage,Path(root)/"old")
            b,bm=w.optimize(initial,new,stage,Path(root)/"new")
            self.assertTrue(torch.equal(a,b))
            for key in ("best_epoch","best_loss","best_data_loss","best_reg_loss","actual_epochs","stopping_reason"):
                self.assertEqual(am[key],bm[key])

    def test_monitor_window_attribution_and_contamination(self):
        import json
        from scripts.experiments.oasis_regularizer_sweep.controller import Study
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p.dump(root/"block.json",dict(cases=[dict(case_load_complete_ns=100,dvf_file_ready_ns=200)]))
            samples=[dict(monotonic_ns=t,processes=[dict(pid=12,memory_mib=100+t)],rss_bytes=t*10) for t in (50,120,180,250)]
            (root/"register_monitor.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples))
            Study.summarize_monitor(None,root,"register",dict(pid=12))
            row=p.read(root/"block.json")["cases"][0]
            self.assertEqual(row["nvml_peak_mib"],280); self.assertEqual(row["host_rss_peak"],1800)
            samples[1]["processes"].append(dict(pid=99,memory_mib=1))
            (root/"register_monitor.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples))
            with self.assertRaises(RuntimeError): Study.summarize_monitor(None,root,"register",dict(pid=12))

    def test_report_no_silent_missing_case_averaging(self):
        from scripts.experiments.oasis_regularizer_sweep.report import stats
        self.assertIsNone(stats([1.,None,2.])["mean"])
        self.assertAlmostEqual(stats([1.,2.,3.])["std"],1.)

    def test_sequences_counts_and_refinement(self):
        self.assertEqual(p.sequence("pilot"),[(0,"cold"),(0,"warm")])
        self.assertEqual(p.sequence("cohort"),[(0,"cold")]+[(r,"warm") for r in range(10)])
        self.assertEqual(p.refined(1),[.01,.03,.1,.3,1,3,10,30,100])
        self.assertEqual((len(p.ARMS)*9+1)*11,605)

    def test_bracket_two_each_side_and_failure_exclusion(self):
        rows=[dict(weight=w,dice=1-.01*abs(math.log10(w)),asd=1,hd95=1) for w in p.COARSE]
        self.assertEqual(p.bracket(rows)["status"],"bracketed")
        edge=[dict(weight=w,dice=.8+.001*math.log10(w)) for w in p.COARSE]
        self.assertEqual(p.bracket(edge)["extensions"],[1e4])
        failed=[dict(weight=w,dice=None) for w in p.COARSE]
        self.assertEqual(p.bracket(failed)["status"],"no_valid_results")
        flat=[dict(weight=10.**k,dice=.8) for k in range(-8,7)]
        self.assertEqual(p.bracket(flat)["status"],"flat_or_unbracketed")

    def test_atomic_json_nulls_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"row.json"; p.dump(path,dict(value=float("nan")))
            self.assertIsNone(p.read(path)["value"])
        with self.assertRaises(ValueError): p.setting("diffusion_fd",0)
        with self.assertRaises(ValueError): p.setting("none",1)


if __name__=="__main__": unittest.main()
