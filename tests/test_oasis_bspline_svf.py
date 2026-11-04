import csv
from dataclasses import replace
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import bspline_svf as s
import run_oasis_bspline_svf as runner


class SVFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1)

    def test_matrix_cli_and_legacy(self):
        jobs = s.matrix()
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({x.name() for x in jobs+s.matrix(True)}), 16)
        for job in jobs:
            stages=s.stages(job)
            self.assertEqual([x.lr for x in stages], [.03,.01,.003,.001][:len(job.schedule)])
            self.assertEqual([x.max_lr_reductions for x in stages], [0,0,0,2][:len(job.schedule)])
        argv=["worker","--manifest","/tmp/input.json","--out-dir","/tmp/svf-test",
              "--gpu","0","--schedule","842","--formulation","refine","--regularize","velocity"]
        self.assertEqual(runner.parse_args(argv).integration_steps,7)
        with self.assertRaises(SystemExit):runner.parse_args(argv+["--integration-steps","8"])
        for invalid in (s.Spec("84","refine","velocity"),s.Spec("842","bad","velocity"),
                        s.Spec("842","refine","bad"),s.Spec("842","refine","velocity",case_rank=1)):
            with self.assertRaises(ValueError):invalid.validate()
        import run_one
        from unittest.mock import patch
        with patch.object(sys,"argv",["run_one.py","--method","direct_cp_8_4_2_be_bspline_analytic_multistage","--alpha","1000"]):
            old=run_one.parse_args()
        self.assertEqual(old.stage_alpha,"")
        self.assertEqual(old.dense_stage_lr,.001)

    def test_identity_translation_and_boundary(self):
        shape=(9,11,13)
        grid=s.c.make_coordinate_grid(shape,"cpu")
        for xyz in ((0.,0.,0.),(.08,0.,0.),(0.,-.12,0.),(0.,0.,.1)):
            v=torch.zeros(*shape,3)+torch.tensor(xyz)
            self.assertTrue(torch.allclose(s.exp_velocity(v),v,atol=2e-7))
        outside=torch.tensor([[1.3,-1.5,2.]])
        v=torch.zeros(*shape,3)+torch.tensor([.1,.2,.3])
        self.assertTrue(torch.allclose(s.compose([v],outside),outside+torch.tensor([.1,.2,.3])))
        self.assertTrue(torch.equal(s.exp_velocity(torch.zeros_like(grid)),torch.zeros_like(grid)))

    def test_scaling_and_zero_derivative(self):
        v=torch.randn(7,9,11,3,dtype=torch.float64)*.005
        self.assertTrue(torch.equal(s.exp_velocity(v,0),v))
        identity=s.c.make_coordinate_grid(v.shape[:3],"cpu").double()
        expected=v/2+s.sample_displacement(v/2,identity+v/2)
        self.assertTrue(torch.equal(s.exp_velocity(v,1,identity),expected))
        origin=torch.zeros_like(v,requires_grad=True)
        gradient=torch.autograd.grad(s.exp_velocity(origin).sum(),origin)[0]
        self.assertTrue(torch.allclose(gradient,torch.ones_like(gradient),atol=1e-12))

    def test_native_voxel_recurrence_values_and_gradients(self):
        root=Path(os.environ.get("DL_CODE_ROOT","/homebase/code_sync/Unsupervised-DL-DIR-Revisited/code"))
        native=runpy.run_path(str(root/"utils_warp.py"))
        shape=(9,11,13)
        velocity=(torch.randn(*shape,3,dtype=torch.float64)*.001).requires_grad_()
        scale=s.d.voxel_scale(shape,device="cpu",dtype=torch.float64)
        channel_first=(velocity*scale)[...,[2,1,0]].permute(3,0,1,2)[None]
        reference=native["VecInt"](shape,7).double()(channel_first)[0].permute(1,2,3,0)[...,[2,1,0]]/scale
        axes=[torch.arange(n,dtype=torch.float64)*2/(n-1)-1 for n in shape]
        grid=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1)[...,[2,1,0]]
        actual=s.exp_velocity(velocity,identity=grid)
        probe=torch.randn_like(actual)
        ga=torch.autograd.grad((actual*probe).sum(),velocity,retain_graph=True)[0]
        gb=torch.autograd.grad((reference*probe).sum(),velocity)[0]
        self.assertTrue(torch.allclose(actual,reference,atol=1e-12,rtol=1e-10))
        self.assertTrue(torch.allclose(ga,gb,atol=1e-10,rtol=1e-10))

    def test_knot_insertion_preserves_velocity_and_exponential(self):
        shape=(16,24,16)
        shapes=s.c.build_nested_cp_shapes(shape,8,(8,4,2,1))
        controls=torch.randn(*shapes[8],3)*.01
        previous=s.d.AlignedBSpline(shapes[8],shape,shape,8,"cpu")(controls)
        for cps in (4,2,1):
            controls=s.c.cubic_bspline_dyadic_knot_insert_3d(controls,shapes[cps])
            current=s.d.AlignedBSpline(shapes[cps],shape,shape,cps,"cpu")(controls)
            self.assertTrue(torch.allclose(previous,current,atol=1e-8,rtol=1e-5))
            self.assertTrue(torch.allclose(s.exp_velocity(previous),s.exp_velocity(current),atol=1e-7,rtol=1e-4))
            previous=current

    def test_noncommuting_composition_and_inverse_order(self):
        shape=(16,24,16)
        grid=s.c.make_coordinate_grid(shape,"cpu")
        v1=torch.zeros_like(grid);v1[...,1]=.2*grid[...,0]
        v2=torch.zeros_like(grid);v2[...,0]=.1
        f=[s.exp_velocity(v1),s.exp_velocity(v2)]
        inv=[s.exp_velocity(-v2),s.exp_velocity(-v1)]
        points=torch.tensor([[.1,.2,.3],[-.2,.1,-.3]])
        mapped=s.compose(f,points)
        self.assertGreater(float((mapped-s.compose(list(reversed(f)),points)).abs().max()),.019)
        self.assertTrue(torch.allclose(s.compose(inv,mapped),points,atol=1e-6))
        self.assertGreater(float((s.compose(list(reversed(inv)),mapped)-points).abs().max()),.019)

    def test_losses_and_gradients_through_frozen_maps(self):
        shape=(16,24,16);fixed=torch.rand(shape);moving=torch.rand(shape)
        stage=s.stages(s.Spec("8421","compose","displacement"))[-1]
        grid=s.c.make_coordinate_grid(shape,"cpu")
        earlier=(grid*.025).requires_grad_()
        for form in ("refine","compose"):
            for target in ("velocity","displacement"):
                spec=s.Spec("8421",form,target)
                objective=s.Objective(fixed,moving,stage,spec,[earlier] if form=="compose" else ())
                controls=(torch.randn(*objective.control_shape,3)*.003).requires_grad_()
                total,data,reg=objective(controls)
                field=objective.field(controls)
                q=objective.expand_eval(controls) if target=="velocity" else field
                manual=s.c.forward_valid_diffusion((q*objective.scale).permute(3,0,1,2)[None])
                self.assertTrue(torch.equal(manual,reg))
                warped=F.grid_sample(moving[None,None],(grid+field)[None],padding_mode="border",align_corners=True)
                self.assertTrue(torch.equal(data,s.d.NCC9(fixed[None,None])(warped)))
                grad=torch.autograd.grad(total,controls)[0]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().max()),0.)
                index=tuple(int(x) for x in torch.unravel_index(grad.abs().argmax(),grad.shape))
                plus=controls.detach().clone();minus=plus.clone();eps=2e-5
                plus[index]+=eps;minus[index]-=eps
                fd=float((objective(plus)[0]-objective(minus)[0])/(2*eps))
                self.assertAlmostEqual(fd,float(grad[index]),delta=max(.002,abs(fd)*.05))
                self.assertIsNone(earlier.grad)

    def test_checkpoint_state_and_initialization_candidate(self):
        shape=(8,8,8);zeros=torch.zeros(shape)
        spec=s.Spec("8421","refine","velocity",smoke=True)
        stage=s.stages(spec)[-1]
        objective=s.Objective(zeros,zeros,stage,spec)
        initial=torch.zeros(*objective.control_shape,3)
        with tempfile.TemporaryDirectory() as tmp:
            best,info=s.optimize(initial,objective,Path(tmp)/"stage")
            self.assertEqual(info["best_epoch"],0)
            self.assertEqual(info["actual_epochs"],1)
            self.assertEqual(info["representation"],"bspline_velocity_controls")
            state=torch.load(Path(tmp)/"stage/best.pt",weights_only=False)
            self.assertTrue(torch.equal(best,state["velocity_controls_norm_xyz"]))
            self.assertEqual(float(best.abs().max()),0.)
            self.assertTrue(torch.equal(objective.field(best),torch.zeros(*shape,3)))
            with (Path(tmp)/"stage/history.csv").open() as f:
                history=list(csv.DictReader(f))
            self.assertEqual([int(x["epoch"]) for x in history],[0,1])

    def test_reconstruct_and_inverse(self):
        shape=(16,24,16);shapes=s.c.build_nested_cp_shapes(shape,8,(8,4,2,1))
        states=[(8,torch.zeros(*shapes[8],3)+torch.tensor([.01,0.,0.])),
                (4,torch.zeros(*shapes[4],3)+torch.tensor([0.,.02,0.]))]
        for form in ("refine","compose"):
            field,parts,_=s.reconstruct(states,form,shape,"cpu")
            _,inverse,_=s.reconstruct(states,form,shape,"cpu",inverse=True)
            expected=torch.tensor([.01,.02,0.] if form=="compose" else [0.,.02,0.])
            self.assertTrue(torch.allclose(field,expected.expand_as(field),atol=2e-7))
            checks=s.inverse_consistency(parts,inverse,torch.ones(shape,dtype=torch.bool),count=100)
            self.assertLess(checks["inverse_after_forward"]["all_max_mm"],1e-5)


if __name__=="__main__": unittest.main()
