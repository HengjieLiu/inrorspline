import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/reports"))
import report_oasis_symmetric_svf_phase_a as report


class ReportTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for arm in report.LABELS:
            rows.append({"spec":{"arm":arm,"smoke":False},
                         "fixed_id":"0115" if arm=="A1_swap" else "0333",
                         "moving_id":"0333" if arm=="A1_swap" else "0115",
                         "evaluations":[{"integration_steps":n,
                             **{direction:{"metrics":{k:.1 for k in report.r.METRICS}}
                                for direction in ("forward","reverse")}} for n in (7,8,10)]})
        return rows

    def test_matrix_and_direction_validation(self):
        rows=self.fixture();report.validate_rows(rows)
        with self.assertRaises(RuntimeError):report.validate_rows(rows[:-1])
        bad=copy.deepcopy(rows);bad[-1]["fixed_id"]="0333"
        with self.assertRaises(RuntimeError):report.validate_rows(bad)
        bad=copy.deepcopy(rows);bad[-1]=bad[0]
        with self.assertRaises(RuntimeError):report.validate_rows(bad)

    def test_missing_or_nonfinite_diagnostics_rejected(self):
        rows=self.fixture();rows[0]["evaluations"].pop()
        with self.assertRaises(RuntimeError):report.validate_rows(rows)
        rows=self.fixture();rows[0]["evaluations"][0]["forward"]["metrics"]["dice"]=float("nan")
        with self.assertRaises(RuntimeError):report.validate_rows(rows)

    def test_physical_unit_difference(self):
        a=np.zeros((160,224,192,3),dtype=np.float32)
        b=np.zeros_like(a);b[...,0]=2/191
        value=report.difference(a,b)
        self.assertAlmostEqual(value["mean_endpoint_mm"],1.,places=6)
        self.assertAlmostEqual(value["component_rms_mm"],1/np.sqrt(3),places=6)
        with self.assertRaises(RuntimeError):report.difference(a[:2],b[:2])


if __name__=="__main__":unittest.main()
