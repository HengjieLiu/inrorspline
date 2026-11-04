"""Native off-grid autograd context regression and immutable optimizer delegation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import oasis_bspline8421_diagnostic_recovery as r


class DiagnosticRecovery(unittest.TestCase):
    def test_actual_offgrid_autograd_inside_no_grad(self):
        torch.set_num_threads(2)
        field=torch.zeros(12,12,12,3)
        mask=torch.ones(12,12,12,dtype=torch.bool)
        original=r.worker.b.offgrid_diagnostics
        def small(fields,exported,mask,count,seed):
            return original(fields,exported,mask,count=128,seed=seed)
        spec=r.e.Spec(0,"bounded2",1,True)
        with patch.object(r.worker.b,"offgrid_diagnostics",side_effect=small):
            with torch.no_grad():
                with self.assertRaises(RuntimeError):
                    r.BASE_DIAGNOSTICS(spec,field,None,[field],{"fixed_mask":mask},Path("/tmp/unused"))
                result=r.diagnostics(spec,field,None,[field],{"fixed_mask":mask},Path("/tmp/unused"))
                self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(result["offgrid_audit"]["retained_composition"]["nonpositive_count"],0)
        self.assertEqual(result["offgrid_audit"]["interpolated_export"]["nonpositive_count"],0)
        self.assertIsNone(field.grad)

    def test_optimization_objects_unchanged_and_sources_restorable(self):
        old=(r.e.sources,r.worker.diagnostics,r.queue.WORKER,r.queue.REPORT)
        numerical=(r.worker.d.optimize_stage,r.worker.b.optimize,r.worker.s.optimize,r.worker.transition.verify_transition)
        try:
            r.install()
            self.assertEqual(numerical,(r.worker.d.optimize_stage,r.worker.b.optimize,r.worker.s.optimize,r.worker.transition.verify_transition))
            self.assertIs(r.worker.diagnostics,r.diagnostics)
        finally:
            r.e.sources,r.worker.diagnostics,r.queue.WORKER,r.queue.REPORT=old

    def test_other_family_diagnostics_delegated(self):
        for family in ("displacement","svf_velocity"):
            with patch.object(r,"BASE_DIAGNOSTICS",return_value={"unchanged":True}) as old:
                self.assertEqual(r.diagnostics(r.e.Spec(0,family,1),None,None,[],{},None),{"unchanged":True})
                old.assert_called_once()


if __name__=="__main__":unittest.main()
