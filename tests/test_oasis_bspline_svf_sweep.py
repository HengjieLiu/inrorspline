import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import bspline_svf as pilot
import bspline_svf_sweep as s
import run_oasis_bspline_svf_sweep as runner
import report_oasis_bspline_svf_sweep as report


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self): torch.manual_seed(1)

    def test_matrix_names_and_frozen_pilot(self):
        jobs=s.matrix()
        self.assertEqual(len(jobs),280)
        self.assertEqual(len({x.name() for x in jobs+s.matrix(True)}),284)
        self.assertEqual(len(s.matrix(True)),4)
        self.assertEqual(sum(len(x.schedule) for x in jobs),980)
        self.assertEqual({x.case_rank for x in jobs},set(range(10)))
        self.assertEqual({x.alpha for x in jobs},set(s.LAMBDAS))
        self.assertEqual(len(pilot.matrix()),8)
        self.assertEqual(pilot.PROTOCOL["stage_alpha"],1.)
        for x in jobs:
            self.assertEqual([y.lr for y in s.stages(x)],[.03,.01,.003,.001][:len(x.schedule)])
        for change in ({"case_rank":10},{"alpha":0.},{"alpha":float("nan")},{"formulation":"compose"},{"integration_steps":8}):
            with self.assertRaises(ValueError): replace(jobs[0],**change).validate()

    def test_cli(self):
        a=runner.parse_args(["worker","--manifest","/tmp/m.json","--out-dir","/tmp/test_svf_p74",
                            "--gpu","3","--case-rank","9","--schedule","8421","--regularize","velocity","--alpha","100"])
        self.assertEqual((a.case_rank,a.alpha,a.gpu),(9,100.,"3"))
        with self.assertRaises(SystemExit):
            runner.parse_args(["worker","--manifest","/tmp/m.json","--out-dir","/tmp/t","--gpu","0",
                               "--case-rank","0","--schedule","842","--regularize","velocity","--alpha","2"])

    def test_lambda1_parity_all_stages_and_targets(self):
        shape=(16,24,16);fixed=torch.rand(shape);moving=torch.rand(shape)
        for reg in ("velocity","displacement"):
            spec=s.Spec(9,"8421",reg,1.)
            old=pilot.Spec("8421","refine",reg)
            for stage in s.stages(spec):
                newobj=s.Objective(fixed,moving,stage,spec)
                oldobj=pilot.Objective(fixed,moving,stage,old)
                controls=(torch.randn(*newobj.control_shape,3)*.001).requires_grad_()
                nv,ov=newobj(controls),oldobj(controls)
                for a,b in zip(nv,ov): self.assertTrue(torch.equal(a,b))
                ng=torch.autograd.grad(nv[0],controls)[0]
                og=torch.autograd.grad(ov[0],controls)[0]
                self.assertTrue(torch.equal(ng,og))

    def test_lambda_loss_gradient_and_both_regularizers(self):
        fixed=torch.rand(16,24,16);moving=torch.rand_like(fixed)
        for reg in ("velocity","displacement"):
            for alpha in s.LAMBDAS:
                spec=s.Spec(4,"8421",reg,alpha)
                obj=s.Objective(fixed,moving,s.stages(spec)[-1],spec)
                controls=(torch.randn(*obj.control_shape,3)*.003).requires_grad_()
                loss,data,penalty=obj(controls)
                self.assertTrue(torch.equal(loss,data+alpha*penalty))
                g=torch.autograd.grad(loss,controls,retain_graph=True)[0]
                gd=torch.autograd.grad(data,controls,retain_graph=True)[0]
                gr=torch.autograd.grad(penalty,controls)[0]
                self.assertTrue(torch.allclose(g,gd+alpha*gr,atol=2e-6,rtol=2e-5))
                self.assertGreater(float(gr.abs().max()),0.)
                q=obj.expand_eval(controls) if reg=="velocity" else obj.field(controls)
                self.assertTrue(torch.equal(penalty,s.c.forward_valid_diffusion((q*obj.scale).permute(3,0,1,2)[None])))

    def test_checkpoint_alpha_zero_initialization_and_history(self):
        spec=s.Spec(5,"8421","velocity",100.,True)
        obj=s.Objective(torch.zeros(8,8,8),torch.zeros(8,8,8),s.stages(spec)[-1],spec)
        initial=torch.zeros(*obj.control_shape,3)
        with tempfile.TemporaryDirectory() as tmp:
            best,info=s.optimize(initial,obj,Path(tmp)/"stage")
            self.assertEqual(info["alpha"],100.)
            self.assertEqual(info["best_epoch"],0)
            self.assertTrue(torch.equal(best,initial))
            self.assertEqual(info["representation"],"bspline_velocity_controls")
            with (Path(tmp)/"stage/history.csv").open() as stream: history=list(csv.DictReader(stream))
            self.assertEqual(len(history),2)
            for row in history:
                self.assertAlmostEqual(float(row["loss"]),float(row["data_loss"])+100*float(row["reg_loss"]))

    def test_complete_cohort_and_sample_sd(self):
        rows=[{"case_rank":i,**{k:float(i) for k in report.METRICS}} for i in range(10)]
        stats=report.summarize(rows)
        self.assertEqual(stats["dice_mean"],4.5)
        self.assertAlmostEqual(stats["dice_sd"],(82.5/9)**.5)
        self.assertIsNone(stats["optimizer_sec_mean"])
        with self.assertRaises(ValueError): report.summarize(rows[:-1])
        with self.assertRaises(ValueError): report.summarize(rows[:-1]+[rows[0]])

    def test_synthetic_full_report_and_immutable_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[]
            for spec in s.matrix():
                folder=root/spec.name();(folder/"final_metrics").mkdir(parents=True)
                metrics={k:(.8 if k=="dice" else .4)+spec.case_rank*.001 for k in report.METRICS}
                labelrow={"fixed_id":f"f{spec.case_rank}","moving_id":f"m{spec.case_rank}"}
                for label in s.c.OASIS_LABELS:
                    for key in ("dice","hd95","asd_symmetric"): labelrow[f"{key}_label_{label:02d}"]=.5
                s.c.write_csv(folder/"final_metrics/case_metrics.csv",[labelrow])
                blocks=[{"cps":x.cps,"optimizer":{},"velocity_diffusion_native":.1,"displacement_diffusion_native":.2,
                         "scaled_initial_map_corners":{"max_displacement_jacobian_infinity_norm":.3,"bound_below_one":True}} for x in s.stages(spec)]
                runs.append({"run_name":spec.name(),"spec":asdict(spec),"case_rank":spec.case_rank,
                             "fixed_id":labelrow["fixed_id"],"moving_id":labelrow["moving_id"],"run_dir":str(folder),
                             "gpu":"0","valid_label_count":35,**metrics,**{k:1. for k in report.RESOURCES},"blocks":blocks,
                             "integration_diagnostics":[{"integration_steps":n,"metrics":metrics,"delta_primary_metrics":{k:0. for k in report.METRICS},"inverse_consistency":{}} for n in (7,8,10)]})
            refs=[{"method":method,"case_rank":i,"fixed_id":f"f{i}","moving_id":f"m{i}",
                   **{k:.5 for k in report.METRICS},"optimizer_sec":"n/a"} for method in ("VFA","SITReg") for i in range(10)]
            old=[{**r,"spec":{**r["spec"],"formulation":"refine"}} for r in runs if r["case_rank"]==0 and r["spec"]["alpha"]==1.]
            report.generate(runs,{"references":refs,"pilot_rank0":old},root/"report")
            accepted=json.loads((root/"report/ACCEPTED.json").read_text())
            self.assertEqual(accepted["candidate_count"],280)
            self.assertTrue((root/"report/arc_lines.png").is_file())
            data=json.loads((root/"report/results.json").read_text())
            self.assertEqual(len(data["summary"]),30)
            self.assertEqual(len(data["per_label"]),9800)
            with self.assertRaises(RuntimeError): report.generate(runs,{"references":refs},root/"report")


if __name__=="__main__": unittest.main()
