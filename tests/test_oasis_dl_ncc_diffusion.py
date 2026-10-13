import importlib.util
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
sys.path.insert(0,str(ROOT/"scripts/reports"))
import common as c
import dl_ncc_diffusion as d
import run_oasis_dl_ncc_diffusion_one as worker
import launch_oasis_dl_ncc_diffusion_sweep as launch
import oasis_dl_ncc_diffusion_first10_arc as report


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1)

    def test_matrix_names_and_cache_identity(self):
        jobs=launch.specs()
        self.assertEqual(len(jobs),420)
        self.assertEqual(len({j.name() for j in jobs}),420)
        self.assertEqual(len({j.prefix().name() for j in jobs}),140)
        self.assertEqual(len(launch.specs(True)),6)
        self.assertFalse(set(j.name() for j in jobs)&set(j.name() for j in launch.specs(True)))

    def test_schedules_units_and_lambda(self):
        for mode in d.RESOLUTIONS:
            spec=d.FitSpec(0,mode,.3,"842_dense")
            stages=d.stages(spec)
            self.assertEqual([s.lr for s in stages],[.03,.01,.003,.001])
            self.assertEqual([s.factor for s in stages],[4,2,1,1] if mode=="pyramid" else [1]*4)
            self.assertEqual([s.patience for s in stages],[25,35,50,100])
            self.assertIsNone(stages[-1].cps)
            self.assertEqual(stages[-1].max_epochs,2000)
            self.assertEqual(stages[-1].max_lr_reductions,2)

    def test_invalid_specs_rejected(self):
        for spec in (d.FitSpec(10,"full",1.),d.FitSpec(0,"bad",1.),
                     d.FitSpec(0,"full",float("nan")),d.FitSpec(0,"full",2.),
                     d.FitSpec(0,"full",1.,"bad")):
            with self.assertRaises(ValueError): spec.validate()

    def test_cli_and_legacy_default_unchanged(self):
        argv=["--case-rank","0","--representation","842_dense","--resolution","full",
              "--alpha",".1","--gpu","3","--oasis-root","/tmp/input","--dl-code-root","/tmp/dl",
              "--vfa-root","/tmp/vfa","--out-dir","/tmp/output","--prefix-dir","/tmp/prefix",
              "--source-revision","test","--smoke"]
        args=worker.parse_args(argv)
        self.assertEqual(worker.spec_from_args(args),d.FitSpec(0,"full",.1,"842_dense",smoke=True))
        import run_one
        from unittest.mock import patch
        with patch.object(sys,"argv",["run_one.py","--method","direct_cp_8_4_2_be_bspline_analytic_multistage","--alpha","1000"]):
            old=run_one.parse_args()
        self.assertEqual(old.stage_alpha,"")
        self.assertEqual(old.dense_stage_lr,.001)

    def test_full_basis_matches_historical_values_and_gradients(self):
        shape=(16,24,16)
        shapes=c.build_nested_cp_shapes(shape,8,[8,4,2,1])
        for cps in (8,4,2,1):
            cp=(torch.randn(*shapes[cps],3)*.01).requires_grad_()
            actual=d.AlignedBSpline(shapes[cps],shape,shape,cps,"cpu")(cp)
            expected=c.expand_on_shape(cp,shapes[cps],cps,shape,1)
            self.assertTrue(torch.equal(actual,expected))
            g1,=torch.autograd.grad(actual.square().mean(),cp)
            g2,=torch.autograd.grad(expected.square().mean(),cp)
            self.assertTrue(torch.equal(g1,g2))

    def test_pyramid_exact_affine_positions_not_legacy_shortcut(self):
        shape=(16,24,16); cps=8; control_shape=c.build_nested_cp_shapes(shape,8,[8])[8]
        cp=torch.zeros(*control_shape,3)
        cp[...,0]=torch.arange(control_shape[2])[None,None,:]*.01
        eval_shape=c.downsample_shape(shape,4)
        out=d.AlignedBSpline(control_shape,shape,eval_shape,cps,"cpu")(cp)
        expected=(torch.linspace(0,shape[2]-1,eval_shape[2])/cps+1)*.01
        self.assertTrue(torch.allclose(out[...,0],expected[None,None,:].expand(*eval_shape),atol=1e-8))
        old=c.expand_on_shape(cp,control_shape,cps,eval_shape,4)
        self.assertGreater(float((old-out).abs().max()),.001)

    def test_knot_insertion_preserves_full_and_coarse_fields(self):
        shape=(16,24,16); shapes=c.build_nested_cp_shapes(shape,8,[8,4,2,1])
        cp=torch.randn(*shapes[8],3)*.01
        for coarse,fine in ((8,4),(4,2),(2,1)):
            refined=c.cubic_bspline_dyadic_knot_insert_3d(cp,shapes[fine])
            for eval_shape in (shape,(7,11,9)):
                a=d.AlignedBSpline(shapes[coarse],shape,eval_shape,coarse,"cpu")(cp)
                b=d.AlignedBSpline(shapes[fine],shape,eval_shape,fine,"cpu")(refined)
                self.assertTrue(torch.allclose(a,b,atol=1e-8,rtol=1e-5))
            cp=refined

    def test_native_loss_values_and_gradients(self):
        root=os.environ.get("DL_NCC_CODE_ROOT")
        vfa=os.environ.get("VFA_ROOT")
        if not root or not vfa: self.skipTest("Set native DL/VFA source roots for source parity")
        result=worker.native_parity(torch.device("cpu"),Path(root),Path(vfa))
        self.assertTrue(result["passed"])

    def test_ncc_zero_signed_and_intensity_behavior(self):
        zero=torch.zeros(1,1,10,12,14)
        self.assertEqual(float(d.NCC9(zero)(zero)),1.)
        a=torch.randn_like(zero)
        self.assertTrue(torch.allclose(d.NCC9(a)(a),d.NCC9(a)(-a)))
        low=torch.rand_like(zero)*1e-3
        self.assertGreater(float(d.NCC9(low)(low)),float(d.NCC9(low*100)(low*100)))

    def test_dense_objective_voxel_diffusion_unmasked_and_gradients(self):
        shape=(10,12,14)
        fixed,moving=torch.rand(shape),torch.rand(shape)
        stage=d.StageSpec(3,None,1,.001,1,1,100,2)
        objective=d.StageObjective(fixed,moving,.3,stage)
        q=(torch.randn(*shape,3)*.01).requires_grad_()
        total,data,reg=objective(q)
        vox=q*d.voxel_scale(shape,device="cpu")
        expected=c.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None)
        self.assertEqual(float(reg),float(expected))
        self.assertTrue(torch.equal(total,data+.3*reg))
        total.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()),0)
        translation=torch.ones_like(q)*.2
        self.assertEqual(float(c.forward_valid_diffusion((translation*objective.scale).permute(3,0,1,2)[None],None)),0.)

    def test_best_state_includes_initial_and_logs_match_states(self):
        class Objective:
            alpha=1.; eval_shape=(2,2,2); scale=torch.ones(3); blur_sigma=0.; blur_kernel_size=1
            def __call__(self,p):
                loss=p.square().mean()
                return loss,loss,loss*0
        stage=d.StageSpec(3,None,1,1.,1,1,100,2)
        with tempfile.TemporaryDirectory() as temp:
            best,summary=d.optimize_stage(torch.full((2,2,2,3),.01),Objective(),stage,Path(temp)/"stage")
            self.assertEqual(summary["best_epoch"],0)
            self.assertEqual(summary["actual_epochs"],1)
            saved=torch.load(Path(temp)/"stage/last.pt",weights_only=False)
            actual=float(saved["parameter_norm_xyz"].square().mean())
            self.assertAlmostEqual(summary["last_loss"],actual,places=7)
            self.assertTrue(torch.equal(best,torch.full_like(best,.01)))

    def test_stopping_minimum_and_windows(self):
        stage=d.stages(d.FitSpec(0,"full",1.,"842_dense"))[-1]
        rows=[{"epoch":i,"loss":1.,"data_loss":1.} for i in range(301)]
        self.assertTrue(d.plateau(rows,stage,0))
        self.assertFalse(d.plateau(rows,stage,225))
        self.assertFalse(d.plateau(rows[:276],stage,0))

    def test_zero_images_zero_controls_remain_zero(self):
        shape=(16,24,16); cp_shape=c.build_nested_cp_shapes(shape,8,[8])[8]
        stage=d.stages(d.FitSpec(0,"pyramid",1.))[0]
        objective=d.StageObjective(torch.zeros(shape),torch.zeros(shape),1.,stage,cp_shape)
        parameter=torch.zeros(*cp_shape,3,requires_grad=True)
        loss,data,reg=objective(parameter)
        self.assertEqual(float(loss),1.)
        self.assertEqual(float(reg),0.)
        loss.backward()
        self.assertTrue(torch.equal(parameter.grad,torch.zeros_like(parameter)))

    def test_border_translation_and_coordinate_order(self):
        shape=(5,6,7)
        image=torch.arange(np.prod(shape),dtype=torch.float32).reshape(shape)
        q=torch.zeros(*shape,3)
        q[...,0]=1/d.voxel_scale(shape,device="cpu")[0]
        grid=c.make_coordinate_grid(shape,torch.device("cpu"))+q
        actual=F.grid_sample(image[None,None],grid[None],padding_mode="border",align_corners=True)[0,0]
        self.assertTrue(torch.allclose(actual[...,:-1],image[...,1:],atol=3e-5))
        self.assertTrue(torch.allclose(actual[...,-1],image[...,-1],atol=3e-5))

    def test_stage_objective_never_uses_analytic_regularization(self):
        from unittest.mock import patch
        shape=(16,24,16); cp_shape=c.build_nested_cp_shapes(shape,8,[8])[8]
        stage=d.stages(d.FitSpec(0,"pyramid",1.))[0]
        with patch.object(c,"analytic_bspline_sampled_be",side_effect=AssertionError("Forbidden BE")), \
             patch.object(c,"forward_valid_be",side_effect=AssertionError("Forbidden BE")):
            objective=d.StageObjective(torch.rand(shape),torch.rand(shape),1.,stage,cp_shape)
            q=torch.zeros(*cp_shape,3,requires_grad=True)
            objective(q)[0].backward()
            self.assertGreater(float(q.grad.abs().sum()),0.)

    def test_reports_complete_cohort_and_sample_sd(self):
        rows=[]
        for spec in launch.specs():
            rows.append({"resolution":spec.resolution,"representation":spec.representation,"alpha":spec.alpha,
                         "case_rank":spec.case_rank,"group":"test",**{k:float(spec.case_rank) for k in report.METRICS}})
        for ref in ("VFA","SITReg"):
            rows += [{"resolution":"reference","representation":ref,"alpha":None,"case_rank":rank,
                      "group":ref,**{k:float(rank) for k in report.METRICS}} for rank in range(10)]
        means=report.summarize(rows)
        self.assertEqual(len(means),44)
        self.assertEqual(means[0]["dice_mean"],4.5)
        self.assertAlmostEqual(means[0]["dice_std"],np.std(np.arange(10),ddof=1))
        with self.assertRaises(RuntimeError): report.summarize(rows[:-1])
        with self.assertRaises(RuntimeError): report.summarize(rows+[rows[0]])
        with tempfile.TemporaryDirectory() as tmp:
            report.make_plot(means,Path(tmp))
            report.make_plot(means,Path(tmp),combined=True)
            report.make_plot(means,Path(tmp),folding=True)
            for stem in ("arc_sdlogj_faceted","arc_sdlogj_combined","tradeoff_folding"):
                for ext in ("png","svg","pdf"):
                    self.assertGreater((Path(tmp)/f"{stem}.{ext}").stat().st_size,1000)


if __name__=="__main__":
    unittest.main()
