import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import fixed_prefix_dense as f
import torch


class FixedPrefixTests(unittest.TestCase):
    def test_matrix_and_prefix(self):
        specs=[f.Spec(r,a) for r in range(10) for a in f.ALPHAS]
        self.assertEqual(len({s.name() for s in specs}),60)
        for spec in specs:
            self.assertEqual(spec.prefix().alpha,1.)
            self.assertEqual(spec.prefix().representation,"842")
            self.assertFalse(spec.prefix().smoke)
            self.assertEqual(f.stage_for(spec).lr,.001)
        self.assertFalse(f.Spec(0,5.,smoke=True).prefix().smoke)
        self.assertEqual(f.stage_for(f.Spec(0,5.,smoke=True)).max_epochs,1)

    def test_reject_invalid_and_legacy_unchanged(self):
        for spec in (f.Spec(10,.1),f.Spec(0,.3),f.Spec(0,1.,prefix_alpha=.1),f.Spec(0,float("nan")),f.Spec(0,1.,seed=42)):
            with self.assertRaises(ValueError): spec.validate()
        for a in (.2,.5,2.,5.):
            with self.assertRaises(ValueError): f.d.FitSpec(0,"pyramid",a,"842_dense").validate()

    def test_cli(self):
        args=f.parse_args(["worker","--manifest","/tmp/input.json","--out-dir","/tmp/fixed_prefix_test",
                           "--gpu","3","--case-rank","4","--prefix-alpha","1","--dense-alpha",".2"])
        self.assertEqual(args.dense_alpha,.2)
        self.assertEqual(args.prefix_alpha,1.)
        with self.assertRaises(SystemExit): f.parse_args(["launch","--artifact-root","/tmp/artifacts","--gpus","0,1,2"])

    def test_objective_parity_all_terms_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        fixed,moving=torch.rand(10,12,14),torch.rand(10,12,14)
        q=(torch.randn(10,12,14,3)*.01).requires_grad_()
        stage=f.stage_for(f.Spec(0,1.))
        previous=f.d.StageObjective(fixed,moving,1.,stage)
        current=f.objective_for(fixed,moving,f.Spec(0,1.))
        for old,new in zip(previous(q),current(q)):
            self.assertTrue(torch.equal(old,new))
            a,=torch.autograd.grad(old,q,retain_graph=True); b,=torch.autograd.grad(new,q,retain_graph=True)
            self.assertTrue(torch.equal(a,b))
        lo=f.objective_for(fixed,moving,f.Spec(0,.1))(q)
        hi=f.objective_for(fixed,moving,f.Spec(0,5.))(q)
        self.assertTrue(torch.equal(lo[1],hi[1]))
        self.assertTrue(torch.equal(lo[2],hi[2]))
        self.assertTrue(torch.allclose(hi[0]-lo[0],4.9*hi[2],atol=1e-6))

    def test_report_complete_and_paired_differences(self):
        prefixes={}; previous={}; results=[]; refs=[]
        for rank in range(10):
            base=dict(case_rank=rank,fixed_id=str(rank),moving_id=str(rank+1),
                      **{m:.2+rank*.001 for m in f.METRICS})
            prefixes[str(rank)]=base
            previous[str(rank)]={**base,"dvf_sha256":"old","gpu":"0","stages":[{"actual_epochs":300}]}
            for method in ("VFA (original)","SITReg λ=1 (original)","FeDDA-1MF lambda1 historical"):
                refs.append({**base,"rank":rank,"method":method})
            for alpha in f.ALPHAS:
                results.append({**base,"spec":asdict(f.Spec(rank,alpha)),"stages":[dict(best_epoch=100,actual_epochs=300,stopping_reason="test")],
                                "gpu":"1","training_elapsed_sec":10.,"total_elapsed_sec":15.,"dvf_sha256":"new"})
        manifest=dict(prefixes=prefixes,historical_dense_lambda1=previous,references=refs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(f,"assert_sources"),patch.object(f,"sources",return_value={}):
            f.report(results,manifest,Path(tmp))
            data=json.loads((Path(tmp)/"results.json").read_text())
            self.assertEqual(len(data["means"]),10)
            self.assertEqual(len(data["per_case"]),100)
            self.assertTrue((Path(tmp)/"ACCEPTED.json").exists())
            self.assertTrue(all(r["delta_dense1_dice"]==0. for r in data["per_case"]))
            with self.assertRaises(RuntimeError): f.report(results[:-1],manifest,Path(tmp))


if __name__=="__main__": unittest.main()
