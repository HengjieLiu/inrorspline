from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import run_oasis_two_increments_first10 as r


class FirstTenTwoIncrementTests(unittest.TestCase):
    def test_weights_names_and_cli(self):
        names={r.run_name(rank,v,smoke) for rank in range(10) for v in r.VECTORS for smoke in (False,True)}
        self.assertEqual(len(names),40)
        for bad in ((1,1,5),(1,1,2,1),(1,1,float("nan"),1),(1,1,0,1)):
            with self.assertRaises(ValueError): r.weights(bad)
        for rank in (-1,10,True):
            with self.assertRaises(ValueError): r.run_name(rank,r.VECTORS[0])
        base=["worker","--manifest","/tmp/manifest.json","--out-dir","/tmp/p69test","--case-rank","9","--gpu","3"]
        for v in r.VECTORS:
            args=r.parse_args(base+["--stage-alpha",*[str(x) for x in v]])
            self.assertEqual(args.stage_alpha,v); self.assertEqual(args.mode,"two_increments")
        with self.assertRaises(SystemExit): r.parse_args(base+["--stage-alpha","1","1","3","1"])

    def test_context_restores_legacy_and_numerical_callables(self):
        old={k:getattr(r.pilot,k) for k in ("VARIANT","PROTOCOL","RANKS","PLACEMENT","sources","run_name","METHODS")}
        alpha=r.b.ALPHAS; worker=r.pilot.worker; objective=r.b.Objective; optimize=r.b.optimize
        for v in r.VECTORS:
            with r.configured(v):
                self.assertEqual(r.b.ALPHAS,v); self.assertEqual(r.pilot.RANKS,tuple(range(10)))
                self.assertEqual(r.pilot.run_name(9,r.MODE),r.run_name(9,v))
                self.assertEqual(r.pilot.PROTOCOL["stage_alpha"],list(v))
                self.assertIs(r.pilot.worker,worker); self.assertIs(r.b.Objective,objective); self.assertIs(r.b.optimize,optimize)
                self.assertEqual([r.b.ALPHAS[i//2] for i in range(8)],[v[0],v[0],v[1],v[1],v[2],v[2],v[3],v[3]])
            self.assertEqual(r.b.ALPHAS,alpha)
            for key,value in old.items(): self.assertIs(getattr(r.pilot,key),value)
        with self.assertRaises(RuntimeError):
            with r.configured(r.VECTORS[1]): raise RuntimeError("restore on failure")
        self.assertEqual(r.b.ALPHAS,alpha)
        self.assertEqual(r.pilot.parse_args(["launch","--artifact-root","/tmp/p68"]).stage_alpha,list(alpha))
        with self.assertRaises(ValueError): r.pilot.run_name(9,r.MODE)

    def test_complete_cohort_and_sample_sd(self):
        rows=[{"case_rank":rank,"stage_alpha":list(v),**{k:float(rank) for k in r.METRICS}}
              for rank in range(10) for v in r.VECTORS]
        r.require_cohort(rows)
        with self.assertRaises(RuntimeError): r.require_cohort(rows[:-1])
        with self.assertRaises(RuntimeError): r.require_cohort(rows[:-1]+[rows[0]])
        s=r.summarize(rows[::2],"test","first10",range(10))
        self.assertEqual(s["n"],10); self.assertEqual(s["dice_mean"],4.5)
        self.assertAlmostEqual(s["dice_sd"],(82.5/9)**.5)
        s=r.summarize(rows[::2],"test",*r.SUBSETS[1])
        self.assertEqual(s["n"],8)
        with self.assertRaises(RuntimeError): r.summarize(rows[:2],"bad","all",range(10))


if __name__=="__main__": unittest.main()
