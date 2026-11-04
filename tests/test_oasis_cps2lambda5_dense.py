from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import cps2lambda5_dense as f
import torch


class DenseAfterCps2FiveTests(unittest.TestCase):
    def test_matrix_prefix_and_schedule(self):
        specs=[f.Spec(r,a) for r in range(10) for a in f.ALPHAS]
        self.assertEqual(len({s.name() for s in specs}),70)
        for s in specs:
            self.assertEqual(s.prefix().cps2_alpha,5.)
            self.assertEqual(s.prefix().prefix_alpha,1.)
            self.assertFalse(s.prefix().smoke)
            stage=f.stage_for(s)
            self.assertEqual((stage.cps,stage.lr,stage.max_epochs,stage.min_epochs),(None,.001,2000,300))
            self.assertEqual((stage.patience,stage.max_lr_reductions),(100,2))
        self.assertEqual(f.stage_for(f.Spec(0,10,smoke=True)).max_epochs,1)

    def test_cli_invalid_and_legacy_unchanged(self):
        for s in (f.Spec(10,1),f.Spec(0,.3),f.Spec(0,1,prefix_cps2_alpha=1),f.Spec(0,float("nan")),f.Spec(0,1,seed=2)):
            with self.assertRaises(ValueError): s.validate()
        args=f.parse_args(["worker","--manifest","/tmp/in.json","--out-dir","/tmp/dense5_test","--gpu","3","--case-rank","4","--prefix-cps2-alpha","5","--dense-alpha","10"])
        self.assertEqual(args.dense_alpha,10.); self.assertEqual(args.prefix_cps2_alpha,5.)
        for a in (.2,.5,2.,5.):
            with self.assertRaises(ValueError): f.d.FitSpec(0,"pyramid",a,"842_dense").validate()

    def test_loss_and_gradient_parity(self):
        torch.set_num_threads(1); torch.manual_seed(71)
        fixed,moving=torch.rand(10,12,14),torch.rand(10,12,14)
        q=(torch.randn(10,12,14,3)*.01).requires_grad_()
        for alpha in (1.,10.):
            old=f.d.StageObjective(fixed,moving,alpha,f.stage_for(f.Spec(0,alpha)))
            new=f.objective_for(fixed,moving,f.Spec(0,alpha))
            for a,b in zip(old(q),new(q)):
                self.assertTrue(torch.equal(a,b))
                self.assertTrue(torch.equal(torch.autograd.grad(a,q,retain_graph=True)[0],torch.autograd.grad(b,q,retain_graph=True)[0]))

    def test_report_seventy_and_matched_previous(self):
        prefixes={}; results=[]; refs=[]; previous=[]
        for rank in range(10):
            base=dict(case_rank=rank,fixed_id=str(rank),moving_id=str(rank+1),**{m:.2+rank*.001 for m in f.METRICS})
            prefixes[str(rank)]=base
            for method in ("VFA (original)","SITReg λ=1 (original)","FeDDA-1MF lambda1 historical"):
                refs.append({**base,"rank":rank,"method":method})
            for alpha in f.ALPHAS:
                row={**base,"spec":asdict(f.Spec(rank,alpha)),"stages":[dict(best_epoch=100,actual_epochs=300,stopping_reason="test")],"gpu":"1","training_elapsed_sec":10.,"total_elapsed_sec":15.,"dvf_sha256":"new"}
                results.append(row)
                if alpha!=10: previous.append(row)
        manifest=dict(prefixes=prefixes,previous_dense=previous,references=refs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(f,"assert_sources"),patch.object(f,"sources",return_value={}):
            f.report(results,manifest,Path(tmp))
            data=json.loads((Path(tmp)/"results.json").read_text())
            self.assertEqual(len(data["means"]),11); self.assertEqual(len(data["per_case"]),110)
            self.assertEqual(len(data["previous_prefix_comparison"]),60)
            self.assertTrue(all(r["dense_alpha"]!=10 and r["delta_dice"]==0 for r in data["previous_prefix_comparison"]))
            self.assertTrue((Path(tmp)/"ACCEPTED.json").exists())
            with self.assertRaises(RuntimeError): f.report(results[:-1],manifest,Path(tmp))

if __name__=="__main__": unittest.main()
