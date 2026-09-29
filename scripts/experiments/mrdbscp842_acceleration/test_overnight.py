"""CPU contracts, source syntax and replay manifest tests; external test report."""
import argparse
import ast
import io
import json
from pathlib import Path
import unittest
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from .overnight_candidates import main_candidates, lr_multiplier
from .overnight_actions import scheduled_optimizer
from .backends import c


class Contracts(unittest.TestCase):
    def test_smoke_stage_arrays_match_geometry(self):
        from scripts.experiments.oasis_alpha_sweep import launch_oasis_unified_acceleration_sweep as unified
        for spec in main_candidates():
            candidate=replace(unified.candidate_map()["MR_A_fast_decay"],**spec["overrides"])
            effective=unified.phase_candidate(candidate,SimpleNamespace(epoch_override=1))
            stages=len(effective.stage_full_cps.split(","))
            self.assertEqual(effective.epochs,stages)
            self.assertEqual(c.parse_stage_ints(effective.stage_epochs,stages),[1]*stages)
            self.assertEqual(c.parse_stage_ints(effective.stage_check_interval,stages),[1]*stages)
            for key in ("stage_downsample","stage_lr","stage_min_epochs","stage_patience",
                        "stage_rel_delta_total","stage_rel_delta_data"):
                self.assertEqual(len(getattr(effective,key).split(",")),stages)

    def test_schedule_does_not_retain_optimizer(self):
        spec=next(s for s in main_candidates() if s["lr_schedule"]=="cosine")
        with scheduled_optimizer(spec):
            x=torch.nn.Parameter(torch.ones(2))
            opt=torch.optim.Adam([x],lr=.03)
            ref=weakref.ref(opt)
            opt.zero_grad();x.square().sum().backward();opt.step()
            del opt
            gc.collect()
            self.assertIsNone(ref())

    def test_manifest(self):
        specs=main_candidates()
        self.assertEqual(len(specs),12)
        self.assertEqual(len({s["name"] for s in specs}),12)
        self.assertTrue(all(s["features"]==["cpbe"] for s in specs))
        for spec in specs:
            o=spec["overrides"]
            cps=list(map(int,o["stage_full_cps"].split(",")))
            shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,base_spacing=cps[0],spacings=cps)
            c.assert_nested_schedule(shapes,cps)
            self.assertEqual(tuple(shapes[2]),(84,116,100))
            self.assertEqual(o["epochs"],sum(map(int,o["stage_epochs"].split(","))))

    def test_lr_endpoints(self):
        self.assertEqual(lr_multiplier("cosine",1,100),1.)
        self.assertAlmostEqual(lr_multiplier("cosine",100,100),.2)
        self.assertEqual(lr_multiplier("step",1,100),1.)
        self.assertEqual(lr_multiplier("step",100,100),.25)
        self.assertEqual(lr_multiplier("constant",10,100),1.)
        self.assertEqual(lr_multiplier("cosine",1,1),1.)

    def test_default_optimizer_not_patched(self):
        original=torch.optim.Adam
        with scheduled_optimizer(main_candidates()[0]):
            self.assertIs(torch.optim.Adam,original)
        self.assertIs(torch.optim.Adam,original)

    def test_scheduled_lr_before_update_and_restore(self):
        spec=next(s for s in main_candidates() if s["lr_schedule"]=="cosine")
        original=torch.optim.Adam
        try:
            with scheduled_optimizer(spec):
                x=torch.nn.Parameter(torch.ones(2))
                opt=torch.optim.Adam([x],lr=.03)
                for step in range(1,51):
                    opt.zero_grad(); x.square().sum().backward(); opt.step()
                    self.assertAlmostEqual(opt.param_groups[0]["lr"],.03*lr_multiplier("cosine",step,50))
                raise RuntimeError("restore test")
        except RuntimeError:
            pass
        self.assertIs(torch.optim.Adam,original)

    def test_four_stage_prolongation_shapes(self):
        shapes=c.build_nested_cp_shapes(c.IMAGE_SHAPE,base_spacing=16,spacings=[16,8,4,2])
        self.assertEqual(tuple(shapes[16]),(14,18,16))
        self.assertEqual(tuple(shapes[8]),(24,32,28))

    def test_jacobian_identity_and_negative_are_not_hidden(self):
        shape=(9,11,13)
        field=np.zeros((1,3,*shape))
        mask=np.ones(shape,dtype=bool)
        r=c.jacobian_metrics_voxel(field,mask)
        self.assertEqual(r["log_jdet_std_fg"],0.)
        self.assertEqual(r["jdet_le0_pct_fg"],0.)
        field[0,0]= -2*np.arange(shape[0])[:,None,None]
        r=c.jacobian_metrics_voxel(field,mask)
        self.assertEqual(r["jdet_le0_pct_fg"],100.)
        self.assertEqual(r["log_jdet_pos_frac_fg"],0.)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(6)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    for path in Path(__file__).parent.glob("*.py"):
        ast.parse(path.read_text(),filename=str(path))
    payload=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,log=stream.getvalue(),cuda_initialized=torch.cuda.is_initialized())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload))
    if not payload["passed"] or payload["cuda_initialized"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
