"""CPU contracts for detailed instrumentation and the existing Jacobian metric."""
import unittest
import numpy as np
import torch
from .backends import Backend,c
from .components import Recorder,instrument


class ComponentContracts(unittest.TestCase):
    def test_nested_host_accounting(self):
        r = Recorder()
        r.marks["case_load_complete"] = 0.
        with r.scope("parent"):
            with r.scope("child"):
                pass
        r.marks["dvf_file_ready"] = 1.
        result = r.payload()
        rows = {v["name"]:v for v in result["rows"]}
        self.assertAlmostEqual(rows["parent"]["wall_inclusive_sec"],
                               rows["parent"]["wall_exclusive_sec"]+rows["child"]["wall_inclusive_sec"])
        self.assertEqual(result["raw_event_count"],2)

    def test_instrumented_fields_be_gradients_and_cache(self):
        torch.manual_seed(7)
        shape,cs = (12,16,20),(10,12,14)
        initial = torch.randn((*cs,3))*.001
        for features in ((),("cpbe",),("cache","batch","cpbe")):
            results = []
            for diagnostic in (False,True):
                backend = Backend(features)
                with backend.activate():
                    from contextlib import nullcontext
                    with instrument(c) if diagnostic else nullcontext() as rec:
                        value = initial.clone().requires_grad_()
                        field = c.expand_bspline_controls(value,cs,shape,2)
                        reg = c.analytic_bspline_sampled_be(value,shape,2)
                        grad = torch.autograd.grad(field.square().mean()+1000*reg,value)[0]
                        if diagnostic:
                            before = dict(rec.counters)
                            c.analytic_bspline_sampled_be(value,shape,2)
                            for k,v in before.items():
                                if k.endswith("factor_build"):
                                    self.assertEqual(v,rec.counters[k])
                        results.append((field.detach(),reg.detach(),grad))
            for a,b in zip(*results):
                self.assertTrue(torch.equal(a,b),features)

    def test_training_step_preserves_adam_trajectory(self):
        torch.manual_seed(13)
        shape,cs = (12,16,20),(10,12,14)
        fixed,moving = torch.rand(shape),torch.rand(shape)
        identity = c.make_coordinate_grid(shape,torch.device("cpu"))
        for features in ((),("cpbe",),("cache","batch","cpbe")):
            outputs = []
            for diagnostic in (False,True):
                from contextlib import nullcontext
                with Backend(features).activate():
                    with instrument(c) if diagnostic else nullcontext() as rec:
                        control = torch.nn.Parameter(torch.zeros(int(np.prod(cs)),3))
                        opt = torch.optim.Adam([control],lr=.03)
                        for _ in range(5):
                            if diagnostic:
                                rec.training_step(c,control,cs,2,shape,1,moving,fixed,identity,2,1000,opt)
                            else:
                                grid = c.flat_to_grid(control,cs)
                                dense = c.expand_on_shape(control,cs,2,shape,1)
                                warped = c.warp_full(moving,identity+dense,mode="bilinear")
                                data = c.ncc_loss(fixed[None,None],warped[None,None])
                                reg = c.analytic_bspline_sampled_be(grid,shape,2,component_reduction="mean")
                                loss = data+1000.*reg
                                opt.zero_grad(set_to_none=True)
                                loss.backward()
                                opt.step()
                        outputs.append(control.detach().clone())
            self.assertTrue(torch.equal(*outputs),features)

    def test_sdlogj_identity_affine_and_nonpositive(self):
        shape = (12,16,20)
        mask = np.ones(shape,dtype=bool)
        zero = np.zeros((1,3,*shape),dtype=np.float64)
        identity = c.jacobian_metrics_voxel(zero,mask)
        self.assertEqual(identity["log_jdet_std_fg"],0.)
        self.assertEqual(identity["jdet_le0_pct_fg"],0.)
        z,y,x = np.meshgrid(*[np.arange(n) for n in shape],indexing="ij")
        affine = np.stack((.1*z,.2*y,.3*x))[None]
        metrics = c.jacobian_metrics_voxel(affine,mask)
        self.assertLess(metrics["log_jdet_std_fg"],1e-12)
        self.assertAlmostEqual(metrics["jdet_mean_fg"],1.1*1.2*1.3)
        self.assertAlmostEqual(c.log_std_positive(np.array([-1.,0.,1.,np.e**2])),1.)
        self.assertTrue(np.isnan(c.log_std_positive(np.array([-1.,0.]))))
        folded = np.stack((-2.*z,np.zeros(shape),np.zeros(shape)))[None]
        metrics = c.jacobian_metrics_voxel(folded,mask)
        self.assertEqual(metrics["jdet_le0_pct_fg"],100.)
        self.assertTrue(np.isnan(metrics["log_jdet_std_fg"]))


if __name__ == "__main__":
    unittest.main()
