from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_alpha_sweep"))
import dl_ncc_diffusion as d
import oasis_ncc9_first100 as cohort


class First100NCC9DenseContracts(unittest.TestCase):
    def test_scope_and_counts(self):
        specs = cohort.specs()
        self.assertEqual(len(specs), 800)
        self.assertEqual(sum(spec.representation == "842_dense" for spec in specs), 400)
        self.assertEqual(sum(spec.representation == "842" for spec in specs), 400)
        self.assertEqual(len({spec.name() for spec in specs}), 800)
        self.assertEqual(len(cohort.specs(smoke=True)), 2)
        self.assertEqual(cohort.LAMBDAS, (0.1, 0.3, 1.0, 3.0))

    def test_rank99_and_prefix_identity(self):
        spec = cohort.First100FitSpec(99, "pyramid", 0.3, "842_dense")
        spec.validate()
        self.assertEqual(spec.prefix().case_rank, 99)
        self.assertEqual(spec.prefix().representation, "842")
        self.assertEqual(asdict(spec), {
            "case_rank": 99, "resolution": "pyramid", "alpha": 0.3,
            "representation": "842_dense", "seed": 1, "smoke": False,
        })
        self.assertEqual([stage.factor for stage in d.stages(spec)], [4, 2, 1, 1])

    def test_legacy_worker_remains_first10(self):
        with self.assertRaises(ValueError):
            d.FitSpec(10, "pyramid", 0.3, "842_dense").validate()

    def test_gpu_slots_and_memory_reservation(self):
        self.assertEqual(cohort.validate_gpu_slots("0,2,3,3"), ("0", "2", "3", "3"))
        for value in ("0,1,2,3", "0,2,3", "0,2,3,3,3"):
            with self.assertRaises(ValueError):
                cohort.validate_gpu_slots(value)
        self.assertEqual(cohort.required_memory_mib(1.64), 4096)
        self.assertEqual(cohort.required_memory_mib(3.0), 5120)


if __name__ == "__main__":
    unittest.main()
