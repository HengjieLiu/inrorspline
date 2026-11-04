from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import bounded_composed_bspline as b


class BoundedCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sampling_bounds_and_adversarial_patterns(self):
        lookup = {1:.36, 2:.3854469526363808, 4:.39803114051999844, 8:.4007250760586097}
        for r in b.CPS:
            ref = b.compute_bound(r, "sitreg_half_voxel")
            self.assertAlmostEqual(ref["bound_control_units"], lookup[r], places=12)
            actual = b.compute_bound(r)
            witness = actual["witness"]
            derivatives = b.derivative_stencil(r, witness["phase_zyx"], witness["corner_zyx"])
            coeff = np.sign(np.einsum("i,ijkl->jkl", witness["signs_zyx"], derivatives))
            norm = np.abs((derivatives*coeff).sum(axis=(1,2,3))).sum()
            self.assertAlmostEqual(norm, actual["K_control_units"], places=12)
            # Independently instantiate the witness using the production expander.
            shape = (8*r+1,)*3
            cp_shape = b.c.build_nested_cp_shapes(shape, 8, [r])[r]
            controls = torch.zeros(*cp_shape,3)
            scale = b.d.voxel_scale(shape, device="cpu")
            controls[2:8,2:8,2:8,0] = torch.from_numpy(coeff).float()*actual["cap_native_voxel"]/scale[0]
            field = b.d.AlignedBSpline(cp_shape,shape,shape,r,"cpu")(controls)*scale
            start = np.asarray(witness["phase_zyx"])+3*r
            z,y,x = map(int,start); cz,cy,cx = witness["corner_zyx"]
            dx=field[z+cz,y+cy,x+1]-field[z+cz,y+cy,x]
            dy=field[z+cz,y+1,x+cx]-field[z+cz,y,x+cx]
            dz=field[z+1,y+cy,x+cx]-field[z,y+cy,x+cx]
            self.assertAlmostEqual(float((dx.abs()+dy.abs()+dz.abs()).max()),.99,places=5)
            audit=b.corner_diagnostics(field)
            self.assertLess(audit["max_displacement_jacobian_infinity_norm"],1.)
            self.assertEqual(audit["nonpositive_corner_count"],0)

    def test_tanh_units_cap_and_gradient(self):
        scale=b.d.voxel_scale((160,224,192),device="cpu")
        for r in b.CPS:
            cap=b.compute_bound(r)["cap_native_voxel"]/scale
            latent=torch.zeros(2,3,requires_grad=True)
            controls=b.capped_controls(latent,cap)
            self.assertTrue(torch.equal(controls,torch.zeros_like(controls)))
            self.assertTrue(torch.equal(torch.autograd.grad(controls.sum(),latent)[0],torch.ones_like(latent)))
            extreme=b.capped_controls(torch.tensor([[-1e4,1e4,-1e4]]),cap)
            self.assertTrue(torch.all(extreme.abs()<=cap))

    def test_identity_translation_noncommuting_and_boundaries(self):
        shape=(8,10,12); grid=b.c.make_coordinate_grid(shape,"cpu")
        translation=torch.zeros_like(grid); translation[...,0]=.1
        shear=torch.zeros_like(grid); shear[...,1]=.2*grid[...,0]
        points=torch.tensor([[0.,0.,0.],[.1,-.2,.3]],requires_grad=True)
        self.assertTrue(torch.equal(b.compose([],points),points))
        self.assertTrue(torch.equal(b.compose([torch.zeros_like(grid)],points),points))
        forward=b.compose([shear,translation],points)
        reverse=b.compose([translation,shear],points)
        self.assertTrue(torch.allclose(forward[:,1]-reverse[:,1],torch.full((2,),.02),atol=1e-7))
        outside=torch.tensor([[2.,-3.,4.]],requires_grad=True)
        mapped=b.compose([translation],outside)
        self.assertTrue(torch.allclose(mapped,outside+torch.tensor([[.1,0,0]])))
        self.assertTrue(torch.equal(torch.autograd.grad(mapped.sum(),outside)[0],torch.ones_like(outside)))

    def test_loss_parity_and_frozen_coordinate_gradients(self):
        torch.manual_seed(18); shape=(12,16,12)
        fixed,moving=torch.rand(shape),torch.rand(shape)
        stage=b.d.stages(b.d.FitSpec(0,"pyramid",1.,"8421"))[-1]
        frozen=(b.c.make_coordinate_grid(shape,"cpu")*.03).requires_grad_()
        obj=b.Objective(fixed,moving,stage,1.,[frozen],b.compute_bound(1))
        latent=(torch.randn(*obj.control_shape,3)*.0002).requires_grad_()
        q=obj.field(latent)
        dense=b.d.StageObjective(fixed,moving,1.,replace(stage,cps=None))
        for x,y in zip(obj(latent),dense(q)):
            self.assertTrue(torch.equal(x,y))
            gx=torch.autograd.grad(x,latent,retain_graph=True)[0]
            gy=torch.autograd.grad(y,latent,retain_graph=True)[0]
            self.assertTrue(torch.equal(gx,gy))
        loss=obj(latent)[0]; grad=torch.autograd.grad(loss,latent)[0]
        self.assertGreater(float(grad.abs().max()),0.)
        self.assertIsNone(frozen.grad)
        # Finite difference confirms gradients through previous-map coordinates.
        ix=np.unravel_index(int(grad.abs().argmax()),tuple(grad.shape))
        plus=latent.detach().clone(); minus=plus.clone(); eps=1e-4
        plus[ix]+=eps; minus[ix]-=eps
        fd=float((obj(plus)[0]-obj(minus)[0])/(2*eps))
        self.assertAlmostEqual(float(grad[ix]),fd,delta=max(.0005,abs(fd)*.03))

    def test_optimizer_initialization_and_roundtrip(self):
        fixed=torch.zeros(8,8,8)
        stage=replace(b.d.stages(b.d.FitSpec(0,"pyramid",1.,"8421",smoke=True))[-1],max_epochs=1)
        obj=b.Objective(fixed,fixed,stage,1.,[],b.compute_bound(1))
        with tempfile.TemporaryDirectory() as tmp:
            best,summary=b.optimize(obj,Path(tmp)/"stage")
            self.assertEqual(summary["best_epoch"],0)
            self.assertEqual(float(best.abs().max()),0.)
            state=torch.load(Path(tmp)/"stage/best.pt",weights_only=False)
            self.assertTrue(torch.equal(state["latent_norm_xyz"],best))
            self.assertTrue(torch.equal(state["controls_norm_xyz"],b.capped_controls(best,obj.cap_norm)))

    def test_offgrid_identity(self):
        field=torch.zeros(8,8,8,3)
        audit=b.offgrid_diagnostics([field,field],field,torch.ones(8,8,8,dtype=torch.bool),count=100)
        for key in ("retained_composition","interpolated_export"):
            self.assertEqual(audit[key]["minimum_jacobian"],1.)
            self.assertEqual(audit[key]["nonpositive_count"],0)

    def test_cli_names_and_legacy_schedule(self):
        import run_oasis_bounded_composed as runner
        args=runner.parse_args(["launch","--artifact-root","/tmp/bounded-artifacts","--sitreg-root","/tmp/sitreg"])
        self.assertEqual((args.case_rank,args.gpu,args.bound_multiplier,args.increments_per_stage),(0,"3",.99,1))
        self.assertEqual(args.stage_alpha,[1.,1.,5.,1.])
        self.assertNotEqual(runner.run_name(),runner.run_name(True))
        base=["launch","--artifact-root","/tmp/bounded-artifacts","--sitreg-root","/tmp/sitreg"]
        for extra in (["--case-rank","1"],["--gpu","0"],["--bound-multiplier","1"],
                      ["--increments-per-stage","2"],["--stage-alpha","1","1","1","1"]):
            with self.assertRaises(SystemExit): runner.parse_args(base+extra)
        stages=b.d.stages(b.d.FitSpec(0,"pyramid",1.,"8421"))
        self.assertEqual([s.lr for s in stages],[.03,.01,.003,.001])
        self.assertEqual([s.max_epochs for s in stages],[400,800,1300,2000])
        self.assertEqual([s.max_lr_reductions for s in stages],[0,0,0,2])


if __name__=="__main__": unittest.main()
