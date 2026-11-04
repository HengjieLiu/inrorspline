from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import bounded_composed_cps16 as b
import run_oasis_bounded_composed_cps16 as runner


class Cps16BoundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_and_geometry(self):
        stages = b.stages()
        self.assertEqual([s.cps for s in stages], [16,8,4,2,1])
        self.assertEqual([s.factor for s in stages], [8,4,2,1,1])
        self.assertEqual([s.lr for s in stages], [.03,.03,.01,.003,.001])
        self.assertEqual([s.max_lr_reductions for s in stages], [0,0,0,0,2])
        self.assertEqual([s.patience for s in stages], [25,25,35,50,100])
        self.assertEqual(b.control_shapes(), {16:(14,18,16),8:(24,32,28),
                                             4:(44,60,52),2:(84,116,100),1:(164,228,196)})
        for new, old in zip(stages[1:], b.d.stages(b.d.FitSpec(0,"pyramid",1.,"8421"))):
            self.assertEqual(asdict(replace(new,index=old.index)), asdict(old))
        self.assertTrue(all(s.max_epochs == s.min_epochs == 1 for s in b.stages(True)))
        self.assertEqual(b.original.CPS, (8,4,2,1))
        with self.assertRaises(ValueError): b.original.compute_bound(16)

    def test_exact_bound_and_actual_kernel_witness(self):
        actual = b.compute_bound(16)
        self.assertAlmostEqual(actual["bound_control_units"], .40190721937748236, places=12)
        self.assertAlmostEqual(actual["cap_native_voxel"], 6.366210354939321, places=12)
        self.assertAlmostEqual(b.compute_bound(16,"sitreg_half_voxel")["bound_control_units"],
                               .40175329749498856, places=12)
        w = actual["witness"]
        ds = b.derivative_stencil(16,w["phase_zyx"],w["corner_zyx"])
        coeff = np.sign(np.einsum("i,ijkl->jkl",w["signs_zyx"],ds))
        self.assertAlmostEqual(np.abs((ds*coeff).sum(axis=(1,2,3))).sum(),
                               actual["K_control_units"], places=12)
        shape=(129,)*3
        cp_shape=b.control_shapes(shape)[16]
        controls=torch.zeros(*cp_shape,3)
        scale=b.d.voxel_scale(shape,device="cpu")
        controls[2:8,2:8,2:8,0]=torch.from_numpy(coeff).float()*actual["cap_native_voxel"]/scale[0]
        field=b.d.AlignedBSpline(cp_shape,shape,shape,16,"cpu")(controls)*scale
        z,y,x=map(int,np.asarray(w["phase_zyx"])+48)
        cz,cy,cx=w["corner_zyx"]
        dx=field[z+cz,y+cy,x+1]-field[z+cz,y+cy,x]
        dy=field[z+cz,y+1,x+cx]-field[z+cz,y,x+cx]
        dz=field[z+1,y+cy,x+cx]-field[z,y+cy,x+cx]
        self.assertAlmostEqual(float((dx.abs()+dy.abs()+dz.abs()).max()),.99,places=5)
        audit=b.corner_diagnostics(field)
        self.assertTrue(audit["bound_below_one"])
        self.assertEqual(audit["nonpositive_corner_count"],0)
        for r in (8,4,2,1):
            self.assertEqual(b.compute_bound(r),b.original.compute_bound(r))

    def test_parameterization_and_five_map_order(self):
        scale=b.d.voxel_scale(b.c.IMAGE_SHAPE,device="cpu")
        cap=b.compute_bound(16)["cap_native_voxel"]/scale
        p=torch.zeros(2,3,requires_grad=True)
        controls=b.capped_controls(p,cap)
        self.assertTrue(torch.equal(torch.autograd.grad(controls.sum(),p)[0],torch.ones_like(p)))
        self.assertTrue(torch.all(b.capped_controls(torch.full((2,3),1e5),cap).abs()<=cap))
        grid=b.c.make_coordinate_grid((8,10,12),"cpu")
        shear=torch.zeros_like(grid); shear[...,1]=.2*grid[...,0]
        translation=torch.zeros_like(grid); translation[...,0]=.1
        zero=torch.zeros_like(grid)
        points=torch.tensor([[0.,0.,0.],[2.,-3.,4.]],requires_grad=True)
        mapped=b.compose([shear,translation,zero,zero,zero],points)
        self.assertTrue(torch.allclose(mapped[0],torch.tensor([.1,.02,0.]),atol=1e-7))
        self.assertTrue(torch.allclose(mapped[1],torch.tensor([2.1,-2.8,4.]),atol=1e-7))
        grad=torch.autograd.grad(mapped[1].sum(),points)[0][1]
        self.assertTrue(torch.equal(grad,torch.ones_like(grad)))

    def test_old_stage_and_same_field_loss_gradient_parity(self):
        torch.manual_seed(4)
        fixed,moving=torch.rand(32,48,32),torch.rand(32,48,32)
        frozen=(b.c.make_coordinate_grid(fixed.shape,"cpu")*.02).requires_grad_()
        for stage,alpha in zip(b.stages(),b.ALPHAS):
            obj=b.Objective(fixed,moving,stage,alpha,[frozen],b.compute_bound(stage.cps))
            p=(torch.randn(*obj.control_shape,3)*.0001).requires_grad_()
            q=obj.field(p)
            ref=b.d.StageObjective(obj.fixed,obj.moving,alpha,replace(stage,cps=None,factor=1))
            for actual, expected in zip(obj(p),ref(q)):
                self.assertTrue(torch.equal(actual,expected))
                self.assertTrue(torch.equal(torch.autograd.grad(actual,p,retain_graph=True)[0],
                                            torch.autograd.grad(expected,p,retain_graph=True)[0]))
            self.assertGreater(float(torch.autograd.grad(obj(p)[0],p)[0].abs().max()),0.)
            self.assertIsNone(frozen.grad)
            if stage.cps != 16:
                old=b.original.Objective(fixed,moving,stage,alpha,[frozen],b.compute_bound(stage.cps))
                self.assertTrue(torch.equal(obj.field(p),old.field(p)))
                self.assertTrue(all(torch.equal(a,e) for a,e in zip(obj(p),old(p))))

    def test_zero_roundtrip(self):
        fixed=torch.zeros(16,32,16)
        obj=b.Objective(fixed,fixed,b.stages(True)[0],1.,[],b.compute_bound(16))
        with tempfile.TemporaryDirectory() as tmp:
            best,summary=b.optimize(obj,Path(tmp)/"stage0")
            self.assertEqual(summary["best_epoch"],0)
            self.assertEqual(float(best.abs().max()),0.)
            saved=torch.load(Path(tmp)/"stage0/best.pt",weights_only=False)
            self.assertTrue(torch.equal(saved["controls_norm_xyz"],b.capped_controls(best,obj.cap_norm)))

    def test_cli_scope_and_names(self):
        base=["launch","--artifact-root","/tmp/cps16-artifacts","--sitreg-root","/tmp/sitreg"]
        args=runner.parse_args(base)
        self.assertEqual(args.stage_alpha,[1.,1.,1.,5.,1.])
        self.assertEqual((args.case_rank,args.gpu,args.increments_per_stage),(0,"3",1))
        self.assertNotEqual(runner.run_name(),runner.run_name(True))
        self.assertNotEqual(runner.run_name(),runner.previous_bounded.run_name())
        for extra in (["--stage-alpha","1","1","5","1"],
                      ["--stage-alpha","1","1","1","1","1"],
                      ["--stage-alpha","1","1","nan","5","1"],
                      ["--case-rank","4"],["--gpu","0"],
                      ["--increments-per-stage","2"],["--bound-multiplier","1"]):
            with self.assertRaises(SystemExit): runner.parse_args(base+extra)


if __name__=="__main__": unittest.main()
