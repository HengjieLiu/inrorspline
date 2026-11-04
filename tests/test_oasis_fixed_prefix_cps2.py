from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import fixed_prefix_cps2 as f
import torch


class Cps2Tests(unittest.TestCase):
    def test_matrix_stage_and_names(self):
        specs=[f.Spec(r,a) for r in range(10) for a in f.ALPHAS]
        self.assertEqual(len({s.name() for s in specs}),60)
        for spec in specs:
            stage=f.stage_for(spec)
            self.assertEqual((stage.index,stage.cps,stage.factor,stage.lr),(2,2,1,.003))
            self.assertEqual((stage.max_epochs,stage.min_epochs,stage.patience,stage.max_lr_reductions),(1300,130,50,0))
            self.assertEqual(spec.prefix().alpha,1.)
        self.assertEqual(f.stage_for(f.Spec(0,.1,smoke=True)).max_epochs,1)

    def test_invalid_and_cli(self):
        for spec in (f.Spec(10,1),f.Spec(0,.3),f.Spec(0,1,prefix_alpha=.5),f.Spec(0,float("nan")),f.Spec(0,1,seed=2)):
            with self.assertRaises(ValueError): spec.validate()
        args=f.parse_args(["worker","--manifest","/tmp/in.json","--out-dir","/tmp/cps2_test","--gpu","3","--case-rank","4","--prefix-alpha","1","--cps2-alpha",".2"])
        self.assertEqual(args.cps2_alpha,.2)
        for a in (.2,.5,2.,5.):
            with self.assertRaises(ValueError): f.d.FitSpec(0,"pyramid",a,"842").validate()

    def test_objective_and_gradient_parity(self):
        torch.set_num_threads(1); torch.manual_seed(71)
        fixed,moving=torch.rand(16,24,16),torch.rand(16,24,16)
        shapes=f.c.build_nested_cp_shapes(fixed.shape,8,[8,4,2])
        q=(torch.randn(*shapes[2],3)*.01).requires_grad_()
        old=f.d.StageObjective(fixed,moving,1.,f.stage_for(f.Spec(0,1.)),shapes[2])
        new=f.objective_for(fixed,moving,f.Spec(0,1.))
        for a,b in zip(old(q),new(q)):
            self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(torch.autograd.grad(a,q,retain_graph=True)[0],torch.autograd.grad(b,q,retain_graph=True)[0]))
        low=f.objective_for(fixed,moving,f.Spec(0,.1))(q)
        high=f.objective_for(fixed,moving,f.Spec(0,5.))(q)
        self.assertTrue(torch.equal(low[1],high[1])); self.assertTrue(torch.equal(low[2],high[2]))
        self.assertTrue(torch.allclose(high[0]-low[0],4.9*high[2],atol=1e-6))

    def test_report_complete(self):
        prefixes={}; results=[]; refs=[]
        for rank in range(10):
            base=dict(case_rank=rank,fixed_id=str(rank),moving_id=str(rank+1),**{m:.2+rank*.001 for m in f.METRICS})
            prefixes[str(rank)]={**base,"dvf_sha256":"old","gpu":"0","stages":[{"actual_epochs":200}]}
            for method in ("VFA (original)","SITReg λ=1 (original)","FeDDA-1MF lambda1 historical"):
                refs.append({**base,"rank":rank,"method":method})
            for alpha in f.ALPHAS:
                results.append({**base,"spec":asdict(f.Spec(rank,alpha)),"stages":[dict(best_epoch=180,actual_epochs=200,stopping_reason="test")],"gpu":"1","training_elapsed_sec":10.,"total_elapsed_sec":15.,"dvf_sha256":"new"})
        manifest=dict(prefixes=prefixes,historical_cps2_lambda1=prefixes,references=refs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(f,"assert_sources"),patch.object(f,"sources",return_value={}):
            f.report(results,manifest,Path(tmp))
            data=json.loads((Path(tmp)/"results.json").read_text())
            self.assertEqual(len(data["means"]),10); self.assertEqual(len(data["per_case"]),100)
            self.assertTrue((Path(tmp)/"ACCEPTED.json").exists())
            with self.assertRaises(RuntimeError): f.report(results[:-1],manifest,Path(tmp))

if __name__=="__main__": unittest.main()
