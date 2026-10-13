"""Contracts for the isolated first10 lambda-only extension."""
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_alpha_sweep"))
import common as c
import vfa_feature_refinement as f
import run_oasis_vfa_feature_dense_one as worker
import launch_oasis_raw_ncc_lambda_extension as ext


def control():
    return f.FitConfig(order="project_then_warp", loss="raw_ncc", support="full", alpha=30., lr=.3)


class ExtensionTests(unittest.TestCase):
    def test_grid_and_only_lambda_changes(self):
        cfg = control()
        specs = ext.specs_for(cfg)
        self.assertEqual(len(specs), 30)
        self.assertEqual(len({s["key"] for s in specs}), 30)
        self.assertEqual({(s["rank"], s["config"]["alpha"]) for s in specs},
                         {(r, a) for r in range(10) for a in (100., 300., 1000.)})
        for spec in specs:
            restored = dict(spec["config"], alpha=30.)
            self.assertEqual(restored, dataclasses.asdict(cfg))
            self.assertNotEqual(spec["key"], f.config_name(cfg, spec["rank"]))

    def test_reject_control_drift(self):
        self.assertEqual(ext.control_config({"control_config": dataclasses.asdict(control())}), control())
        with self.assertRaises(RuntimeError):
            ext.control_config({"control_config": dataclasses.asdict(dataclasses.replace(control(), lr=.1))})

    def test_cli_and_path_protection(self):
        self.assertEqual(ext.parse_args([]).gpu_list, ["0", "1", "2", "3"])
        with self.assertRaises(SystemExit):
            ext.parse_args(["--max-concurrent", "5"])
        with self.assertRaises(SystemExit):
            ext.parse_args(["--source-root", "/tmp/prior", "--out-root", "/tmp/prior/extension"])
        with self.assertRaises(SystemExit):
            ext.parse_args(["--source-root", "/tmp/prior", "--out-root", "/tmp"])
        with self.assertRaises(SystemExit):
            ext.parse_args(["--out-root", str(ROOT / "generated")])

    def test_legacy_worker_variant_default(self):
        argv = ["--oasis-root", "/tmp/o", "--dl-root", "/tmp/d", "--vfa-root", "/tmp/v",
                "--checkpoint", "/tmp/c", "--cache-dir", "/tmp/cache", "--out-dir", "/tmp/out"]
        self.assertEqual(worker.parse_args(argv).experiment_variant, f.VARIANT)
        self.assertEqual(worker.parse_args(argv + ["--experiment-variant", ext.VARIANT]).experiment_variant, ext.VARIANT)

    def test_full_report_and_mean_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ext.parse_args(["--review-root", directory])
            originals, old, new = [], [], []
            for rank in range(10):
                base = {"case_rank": rank, "fixed_id": f"f{rank}", "moving_id": f"m{rank}",
                        "dice": .8, "hd95": 2., "asd": .4, "matches_vfa": True}
                for method in ("VFA", "MR-D-BSCP 8-4-2", "Dense VFA features"):
                    originals.append({**base, "method": method})
                for alpha in (30., *ext.LAMBDAS):
                    cfg = dataclasses.replace(control(), alpha=alpha)
                    row = {**base, "config": dataclasses.asdict(cfg), "run_name": f.config_name(cfg, rank),
                           "dice": .81 if alpha == 300 else .8, "feature_loss_normalizer": .01,
                           "equivalent_raw_alpha": alpha * .01, "prefix_sha256": "prefix",
                           "cache_sha256": "cache", "regularity": {}, "deltas_vfa": {}}
                    (old if alpha == 30 else new).append(row)
            ext.write_report(args, new, old, originals)
            report = c.read_json(Path(directory) / "report.json")
            self.assertEqual(len(report["case_metrics"]), 70)
            self.assertEqual(len(report["means"]), 14)
            self.assertEqual(report["selected_by_first10_mean_dice"]["alpha"], 300.)
            self.assertTrue(report["all_first10_are_exploratory_tuning"])
            self.assertTrue((Path(directory) / "ACCEPTED.json").exists())
            selected = report["selected_by_first10_mean_dice"]
            self.assertAlmostEqual(selected["dice"], .81)
            self.assertAlmostEqual(selected["delta_vfa_dice"], .01)
            with self.assertRaises(RuntimeError):
                ext.write_report(args, new[:-1], old, originals)
            with self.assertRaises(RuntimeError):
                ext.write_report(args, new[:-1] + [new[0]], old, originals)

    def test_reject_incomplete_means(self):
        with self.assertRaises(RuntimeError):
            ext.cohort_means([{"method": "VFA", "case_rank": 0, "dice": .8, "hd95": 2.,
                               "asd": .4, "matches_vfa": True}])


if __name__ == "__main__":
    unittest.main()
