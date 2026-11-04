from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import selective_bounded_composed as b
import run_oasis_selective_bounded as r


class SelectiveBoundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_masks_cli_and_names(self):
        self.assertEqual(b.stage_mask(1),(False,False,False,True))
        self.assertEqual(b.stage_mask(2),(False,False,True,True))
        self.assertEqual(b.stage_mask(3),(False,True,True,True))
        for n in (0,4,True,1.5):
            with self.assertRaises(ValueError): b.stage_mask(n)
        names=[r.run_name(n,smoke) for n in (1,2,3) for smoke in (False,True)]
        self.assertEqual(len(set(names)),6)
        self.assertNotIn(r.previous_bounded.run_name(),names)
        base=["launch","--artifact-root","/tmp/selective","--sitreg-root","/tmp/sitreg"]
        self.assertEqual(r.parse_args(base).bounded_last,[1,2,3])
        for extra in (["--bounded-last","0"],["--bounded-last","1","1"],
                      ["--stage-alpha","1","1","1","1"],["--gpu","0"],["--case-rank","1"]):
            with self.assertRaises(SystemExit): r.parse_args(base+extra)
        self.assertEqual([s.lr for s in b.stages()],[.03,.01,.003,.001])
        self.assertEqual([s.max_lr_reductions for s in b.stages()],[0,0,0,2])

    def test_transform_units_and_no_unbounded_clipping(self):
        fixed=torch.zeros(16,16,16)
        for bounded in (True,False):
            obj=b.Objective(fixed,fixed,b.stages()[0],1.,[],b.compute_bound(8),bounded)
            p=torch.zeros(*obj.control_shape,3,requires_grad=True)
            self.assertTrue(torch.equal(torch.autograd.grad(obj.controls(p).sum(),p)[0],torch.ones_like(p)))
            p=torch.ones_like(p)*10
            if bounded: self.assertTrue(torch.all(obj.controls(p).abs()<=obj.cap_norm))
            else: self.assertTrue(torch.equal(obj.controls(p),p))
            self.assertEqual(obj.bounded,bounded)

    def test_same_field_loss_and_bounded_legacy_parity(self):
        torch.manual_seed(7)
        fixed,moving=torch.rand(16,24,16),torch.rand(16,24,16)
        frozen=(b.c.make_coordinate_grid(fixed.shape,"cpu")*.04).requires_grad_()
        for stage,alpha in zip(b.stages(),b.ALPHAS):
            for bounded in (True,False):
                obj=b.Objective(fixed,moving,stage,alpha,[frozen],b.compute_bound(stage.cps),bounded)
                p=(torch.randn(*obj.control_shape,3)*.001).requires_grad_()
                ref=b.d.StageObjective(obj.fixed,obj.moving,alpha,replace(stage,cps=None,factor=1))
                for actual,expected in zip(obj(p),ref(obj.field(p))):
                    self.assertTrue(torch.equal(actual,expected))
                    self.assertTrue(torch.equal(torch.autograd.grad(actual,p,retain_graph=True)[0],
                                                torch.autograd.grad(expected,p,retain_graph=True)[0]))
                self.assertIsNone(frozen.grad)
                if bounded:
                    old=b.original.Objective(fixed,moving,stage,alpha,[frozen],b.compute_bound(stage.cps))
                    self.assertTrue(torch.equal(obj.atomic(p),old.atomic(p)))
                    self.assertTrue(all(torch.equal(a,e) for a,e in zip(obj(p),old(p))))

    def test_optimizer_and_bounded_checkpoint_parity(self):
        torch.manual_seed(12)
        fixed,moving=torch.rand(16,16,16),torch.rand(16,16,16)
        stage=replace(b.stages(True)[0],max_epochs=2,lr=.003)
        obj=b.Objective(fixed,moving,stage,1.,[],b.compute_bound(8),True)
        old=b.original.Objective(fixed,moving,stage,1.,[],b.compute_bound(8))
        with tempfile.TemporaryDirectory() as tmp:
            best,info=b.optimize(obj,Path(tmp)/"new")
            original,summary=b.original.optimize(old,Path(tmp)/"old")
            self.assertTrue(torch.equal(best,original))
            for k in ("best_loss","best_data_loss","best_reg_loss","best_epoch","actual_epochs"):
                self.assertEqual(info[k],summary[k])
            saved=torch.load(Path(tmp)/"new/best.pt",weights_only=False)
            self.assertTrue(torch.equal(saved["controls_norm_xyz"],obj.controls(best)))

    def test_unbounded_checkpoint_preserves_controls_above_cap(self):
        class Toy:
            stage=replace(b.stages(True)[0],lr=.2)
            alpha=1.; control_shape=(1,1,1); identity=torch.zeros(1,1,1,3)
            eval_shape=(1,1,1); scale=torch.ones(3); native_scale=torch.ones(3)
            cap_norm=torch.tensor([.01,.02,.03]); bound={}; bounded=False
            blur_sigma=0.; blur_kernel_size=1
            def controls(self,p): return p
            def __call__(self,p):
                data=(p-.2).square().mean(); reg=p.sum()*0
                return data,data,reg
        with tempfile.TemporaryDirectory() as tmp:
            best,info=b.optimize(Toy(),Path(tmp)/"run")
            state=torch.load(Path(tmp)/"run/best.pt",weights_only=False)
            self.assertEqual(info["best_epoch"],1)
            self.assertIsNone(info["bound"])
            self.assertIsNone(info["cap_norm_xyz"])
            self.assertIsNone(info["saturated_coefficient_fraction_gt_95pct"])
            self.assertTrue(torch.equal(state["controls_norm_xyz"],state["latent_norm_xyz"]))
            self.assertTrue(torch.all(best>Toy.cap_norm))

    def test_later_bounded_identity_does_not_remove_earlier_folding(self):
        grid=b.c.make_coordinate_grid((8,10,12),"cpu")
        unbounded=torch.zeros_like(grid); unbounded[...,0]=-2*grid[...,0]
        bounded=torch.zeros_like(grid)
        p=torch.tensor([[.1,.2,.3]],requires_grad=True)
        mapped=b.compose([unbounded,bounded],p)
        jac=torch.stack([torch.autograd.grad(mapped[:,i].sum(),p,retain_graph=i<2)[0]
                         for i in range(3)],dim=-2)
        self.assertLess(float(torch.linalg.det(jac)[0]),0.)
        self.assertTrue(torch.equal(b.compose([unbounded,bounded],grid),b.compose([unbounded],grid)))


if __name__=="__main__": unittest.main()
