from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import bounded_two_blocks as b
import run_oasis_bounded_two_blocks as r


class TwoBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_names_cli_schedule_and_indices(self):
        self.assertEqual(len({r.run_name(rank,mode,smoke) for rank in (0,4) for mode in b.MODES for smoke in (False,True)}),8)
        for mode in b.MODES:
            for step in range(8):
                self.assertEqual(b.retained_indices(mode,step),b.frozen_indices(mode,step)+[step])
        self.assertEqual(b.retained_indices("two_increments",7),list(range(8)))
        self.assertEqual(b.retained_indices("one_map_two_blocks",7),[1,3,5,7])
        self.assertEqual(b.frozen_indices("one_map_two_blocks",3),[1])
        self.assertEqual([s.max_lr_reductions for s in b.stages() for _ in range(2)],[0,0,0,0,0,0,2,2])
        self.assertEqual([s.lr for s in b.stages() for _ in range(2)],[.03,.03,.01,.01,.003,.003,.001,.001])
        for rank in (1,9,True):
            with self.assertRaises(ValueError): r.run_name(rank,"two_increments")
        base=["launch","--artifact-root","/tmp/two-block-test"]
        self.assertEqual(r.parse_args(base).blocks_per_level,2)
        for extra in (["--stage-alpha","1","1","1","1"],["--blocks-per-level","3"],["--bound-multiplier","1"]):
            with self.assertRaises(SystemExit): r.parse_args(base+extra)

    def test_initialization_contract(self):
        p=torch.ones(3,3,3,3,requires_grad=True)
        self.assertIsNone(b.initial_parameter("two_increments",1,p))
        self.assertIsNone(b.initial_parameter("one_map_two_blocks",2,p))
        q=b.initial_parameter("one_map_two_blocks",1,p)
        self.assertTrue(torch.equal(q,p)); self.assertFalse(q.requires_grad)
        self.assertNotEqual(q.data_ptr(),p.data_ptr())
        with self.assertRaises(ValueError): b.initial_parameter("one_map_two_blocks",1,None)

    def test_fresh_optimizer_parity_and_continuation_state(self):
        torch.manual_seed(11); fixed,moving=torch.rand(16,16,16),torch.rand(16,16,16)
        stage=replace(b.stages(True)[0],max_epochs=2)
        obj=b.Objective(fixed,moving,stage,1.,[],b.compute_bound(8),True)
        with tempfile.TemporaryDirectory() as tmp:
            a,info=b.optimize(obj,Path(tmp)/"new")
            old,prior=b.original.optimize(obj,Path(tmp)/"old")
            self.assertTrue(torch.equal(a,old))
            for key in ("best_loss","best_epoch","actual_epochs","initial_loss"):
                self.assertEqual(info[key],prior[key])
            continued,second=b.optimize(obj,Path(tmp)/"continuation",a)
            self.assertEqual(second["initial_latent_max_abs"],float(a.abs().max()))
            self.assertEqual(second["initial_loss"],info["best_loss"])
            self.assertLessEqual(second["best_loss"],info["best_loss"])
            state=torch.load(Path(tmp)/"continuation/last.pt",weights_only=False)
            self.assertTrue(all(float(x["step"])==2 for x in state["optimizer"]["state"].values()))
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"],stage.lr)
            saved=torch.load(Path(tmp)/"continuation/best.pt",weights_only=False)
            self.assertTrue(torch.equal(saved["controls_norm_xyz"],obj.controls(continued)))

    def test_two_modes_keep_initial_field_and_objective(self):
        torch.manual_seed(4); fixed,moving=torch.rand(16,16,16),torch.rand(16,16,16)
        stage=b.stages(True)[0]; bound=b.compute_bound(8)
        first=b.Objective(fixed,moving,stage,1.,[],bound,True)
        p=torch.randn(*first.control_shape,3)*.01
        atomic=first.atomic(p).detach()
        # Expanded samples are interpolated by BOTH old and new objectives.
        new=b.Objective(fixed,moving,stage,1.,[atomic],bound,True)
        zero=torch.zeros_like(p,requires_grad=True)
        self.assertTrue(torch.equal(first.field(p),new.field(zero)))
        self.assertTrue(all(torch.equal(a,e) for a,e in zip(first(p),new(zero))))
        grad=torch.autograd.grad(new(zero)[0],zero)[0]
        self.assertTrue(bool(torch.isfinite(grad).all())); self.assertGreater(float(grad.abs().max()),0.)
        self.assertIsNone(atomic.grad)
        self.assertTrue(torch.equal(b.capped_controls(zero,new.cap_norm),zero))
        self.assertTrue(torch.equal(torch.autograd.grad(b.capped_controls(zero,new.cap_norm).sum(),zero)[0],torch.ones_like(zero)))

    def test_retained_control_replacement_not_duplicate_map(self):
        grid=b.c.make_coordinate_grid((8,8,8),"cpu")
        atomics=[torch.ones_like(grid)*(.001*(i+1)) for i in range(8)]
        two=b.compose([atomics[i] for i in b.retained_indices("two_increments",7)],grid)-grid
        ctrl=b.compose([atomics[i] for i in b.retained_indices("one_map_two_blocks",7)],grid)-grid
        self.assertTrue(torch.allclose(two,torch.full_like(grid,.036),atol=1e-6))
        self.assertTrue(torch.allclose(ctrl,torch.full_like(grid,.020),atol=1e-6))


if __name__=="__main__": unittest.main()
