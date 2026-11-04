from dataclasses import replace
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import run_oasis_selective_bounded_first10 as r


class FirstTenBoundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_masks_unique_names_and_legacy_unchanged(self):
        for n in range(1,5):
            self.assertEqual(r.stage_mask(n),tuple(i>=4-n for i in range(4)))
        for n in (0,5,True,1.5):
            with self.assertRaises(ValueError): r.stage_mask(n)
        names={r.run_name(rank,n,smoke) for rank in range(10) for n in range(1,5) for smoke in (True,False)}
        self.assertEqual(len(names),80)
        self.assertNotIn(r.p66.run_name(1),names)
        with self.assertRaises(ValueError): r.p66.b.stage_mask(4)
        for rank in (-1,10,True):
            with self.assertRaises(ValueError): r.run_name(rank,1)

    def test_cli(self):
        base=["launch","--artifact-root","/tmp/first10-bound-test","--sitreg-root","/tmp/sitreg"]
        self.assertEqual(r.parse_args(base).gpus,["0","1","2","3"])
        for value in ("", "0,0", "4"):
            with self.assertRaises(SystemExit): r.parse_args(base+["--gpus",value])
        worker=["worker","--manifest","/tmp/manifest","--out-dir","/tmp/out","--gpu","2","--bounded-last","4","--case-rank","9"]
        args=r.parse_args(worker)
        self.assertEqual((args.case_rank,args.bounded_last),(9,4))
        with self.assertRaises(SystemExit): r.parse_args(worker+["--case-rank","10"])

    def test_all4_optimizer_parity(self):
        torch.manual_seed(7)
        fixed,moving=torch.rand(16,16,16),torch.rand(16,16,16)
        for stage in r.b.stages(True):
            stage=replace(stage,max_epochs=2)
            alpha=r.b.ALPHAS[stage.index]
            obj=r.b.Objective(fixed,moving,stage,alpha,[],r.b.compute_bound(stage.cps),True)
            old=r.b.original.Objective(fixed,moving,stage,alpha,[],r.b.compute_bound(stage.cps))
            with tempfile.TemporaryDirectory() as tmp:
                actual,info=r.b.optimize(obj,Path(tmp)/"new")
                expected,original=r.b.original.optimize(old,Path(tmp)/"old")
                self.assertTrue(torch.equal(actual,expected))
                for k in ("best_loss","best_data_loss","best_reg_loss","best_epoch","actual_epochs"):
                    self.assertEqual(info[k],original[k])

    def test_complete_cohort_and_sample_sd(self):
        rows=[{"case_rank":rank,"bounded_last":n,**{k:float(rank+n) for k in r.METRICS}}
              for rank in range(10) for n in range(1,5)]
        result=r.aggregate(rows)
        self.assertEqual(len(result),8)
        row=result[0]
        self.assertEqual(row["n"],10)
        self.assertEqual(row["dice_mean"],5.5)
        self.assertEqual(row["dice_sd"],statistics.stdev(range(1,11)))
        self.assertEqual(result[1]["n"],9)
        for invalid in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(RuntimeError): r.aggregate(invalid)
        rows[0]["dice"]=float("nan")
        with self.assertRaises(RuntimeError): r.aggregate(rows)


if __name__=="__main__": unittest.main()
