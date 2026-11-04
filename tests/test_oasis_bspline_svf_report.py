import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_alpha_sweep"))
import aggregate_oasis_bspline_svf as a


class HistoricalTimingTests(unittest.TestCase):
    def test_mixed_historical_runtime_and_immutable_input(self):
        original=[{"method":"VFA","optimizer_sec":"n/a","worker_sec":"n/a","actual_adam_updates":"n/a","dice":.8},
                  {"method":"spline","pipeline_training_elapsed_sec":42.,"dice":.81},
                  {"method":"composed","optimizer_sec":"12.3","worker_sec":23.,"actual_adam_updates":450,"dice":.82}]
        out=a.normalized_references(original)
        self.assertEqual(original[0]["optimizer_sec"],"n/a")
        self.assertIsNone(out[0]["optimizer_sec"])
        self.assertEqual(out[1]["optimizer_sec"],42.)
        self.assertEqual(out[2]["optimizer_sec"],12.3)
        self.assertEqual([x["dice"] for x in out],[.8,.81,.82])
        with self.assertRaises(ValueError): a.optional_number(float("nan"))


if __name__=="__main__": unittest.main()
