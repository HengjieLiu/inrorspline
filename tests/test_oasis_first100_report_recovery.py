from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/reports"))
import finalize_oasis_first100_ncc9_dense as recovery
report = recovery.report


class RecoveryContracts(unittest.TestCase):
    def test_provenance_paths(self):
        self.assertEqual(report.SWEEP, ROOT / "scripts/experiments/oasis_alpha_sweep")
        for name in ("common.py", "run_oasis_vfa_feature_dense_one.py"):
            self.assertTrue((report.SWEEP / name).is_file())

    def test_atomic_cache_resume_and_stale_identity(self):
        identity = dict(case_rank=10, fixed_id="a", moving_id="b", direction="moving_to_fixed",
                        source_path="field", source_sha256="hash", representation="VFA",
                        pair_sha256="split", evaluator_hashes={"code": "version"})
        row = {key: identity[key] for key in ("case_rank", "fixed_id", "moving_id", "direction",
                                              "source_path", "source_sha256", "representation")}
        row.update({key: 0.2 for key in report.METRICS})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            self.assertIsNone(recovery.load_cache(path, identity))
            recovery.atomic_json(path, {"identity": identity, "row": row})
            self.assertEqual(recovery.load_cache(path, identity), row)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            for key in ("source_sha256", "pair_sha256", "evaluator_hashes", "case_rank"):
                with self.assertRaises(ValueError):
                    recovery.load_cache(path, {**identity, key: "changed"})
            with self.assertRaises(ValueError):
                recovery.validate_reference({**row, "dice": float("nan")}, identity)

    def test_full_report_and_acceptance_gate(self):
        results, references = [], []
        for rank in range(100):
            base = dict(case_rank=rank, fixed_id=str(rank), moving_id=str(rank + 1))
            base.update({metric: 0.2 + rank / 1000 for metric in report.METRICS})
            for alpha in report.cohort.LAMBDAS:
                for rep in ("842", "842_dense"):
                    results.append({**base, "resolution": "pyramid", "representation": rep,
                                    "alpha": alpha, "group": str(alpha),
                                    "spec": {"representation": rep}})
            for method in ("VFA", "SITReg"):
                references.append({**base, "resolution": "reference", "representation": method,
                                   "alpha": None, "group": method})
        with tempfile.TemporaryDirectory() as tmp, patch.object(report.first10, "flat_row", side_effect=lambda r: {k:v for k,v in r.items() if k != "spec"}):
            root = Path(tmp)
            report.write_report(results, references, root, {"synthetic_unit_test": True}, mark_accepted=False)
            self.assertFalse((root / "ACCEPTED.json").exists())
            self.assertEqual(len(recovery.verify_report(root)), 5)
            data = json.loads((root / "results.json").read_text())
            self.assertEqual([r["n"] for r in data["leave_rank004_out_means"]], [99]*6)
            (root / "report.md").write_text("corrupted")
            with self.assertRaises(ValueError):
                recovery.verify_report(root)


if __name__ == "__main__":
    unittest.main()
