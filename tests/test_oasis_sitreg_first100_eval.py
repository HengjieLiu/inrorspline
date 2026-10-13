"""Detailed per-label reporting and non-invasive evaluation queue contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments/oasis_alpha_sweep'))
import common as c
import run_oasis_sitreg_first100_eval as e


class EvalContracts(unittest.TestCase):
    def test_cli_scope(self):
        a = e.parse_args(['--training-dir', '/tmp/train', '--out', '/tmp/eval', '--gpu', '3'])
        self.assertFalse(a.smoke)
        self.assertEqual(a.min_free_gib, 24)
        self.assertTrue(e.parse_args(['--training-dir', '/tmp/train', '--out', '/tmp/eval', '--gpu', '3', '--smoke']).smoke)

    def test_wait_for_complete_not_just_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'checkpoints').mkdir()
            (path / 'checkpoints/best_dice.pth.tar').touch()
            self.assertEqual(e.training_ready(path), (False, 'waiting_for_training_and_first10'))
            (path / 'TRAINING_COMPLETED').touch()
            (path / 'COMPLETED').touch()
            (path / 'first10').mkdir()
            (path / 'first10/per_case.json').write_text('[]')
            (path / 'training_summary.json').write_text(json.dumps({'epoch': 200, 'adam_updates': 10000}))
            self.assertEqual(e.training_ready(path), (True, 'ready'))
            (path / 'FAILED.txt').touch()
            self.assertEqual(e.training_ready(path), (False, 'training_failed'))

    def synthetic(self):
        fixed = torch.zeros((12, 12, 12))
        fixed[3:7, 3:7, 3:7] = 1
        fixed[8:10, 8:10, 8:10] = 2
        moving = fixed.clone()
        warped = fixed.clone()
        warped[warped == 2] = 0  # Label2 disappeared after warping.
        dice = c.dice_per_label(warped, fixed, moving)
        metrics = {'dice': dice, 'dice_mean': c.safe_nanmean(dice), 'warped_seg': warped,
                   **c.surface_metrics_per_label(warped, fixed, moving)}
        return fixed, moving, metrics

    def test_all_labels_saved_with_absence_reason_and_canonical_metrics(self):
        fixed, moving, metrics = self.synthetic()
        rows = e.label_rows(metrics, fixed, moving, {'rank': 0})
        self.assertEqual(len(rows), 35)
        self.assertEqual([r['label_id'] for r in rows], list(range(1, 36)))
        self.assertTrue(rows[0]['included_in_case_mean'])
        self.assertAlmostEqual(rows[0]['dice'], 1, places=6)
        self.assertEqual(rows[0]['hd95_mm'], 0)
        self.assertEqual(rows[0]['asd_symmetric_mm'], 0)
        self.assertEqual(rows[1]['exclusion_reason'], 'absent_in_warped')
        self.assertFalse(rows[1]['included_in_case_mean'])
        clean = e.json_safe(rows)
        self.assertIsNone(clean[1]['dice'])
        self.assertIsNone(clean[1]['hd95_mm'])
        self.assertIsNone(clean[1]['asd_symmetric_mm'])
        json.dumps(clean, allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            c.write_csv(path, clean)
            reread = c.read_csv(path)
            self.assertEqual(len(reread), 35)
            self.assertEqual(reread[1]['dice'], '')

    def test_nonfinite_present_label_fails(self):
        fixed, moving, metrics = self.synthetic()
        metrics['hd95'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite metric'):
            e.label_rows(metrics, fixed, moving, {})

    def test_vector_length_and_aggregate_drift_fail(self):
        fixed, moving, metrics = self.synthetic()
        metrics['dice_mean'] = .1
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            e.label_rows(metrics, fixed, moving, {})
        metrics['dice'].append(0.)
        with self.assertRaisesRegex(ValueError, 'vector length'):
            e.label_rows(metrics, fixed, moving, {})

    def test_summary_is_case_weighted_and_sample_sd(self):
        rows = [{key: x for key in e.SUMMARY_METRICS} for x in (1., 3.)]
        means, per_label = e.summary_tables(rows, [])
        self.assertEqual(means['dice_mean']['mean'], 2)
        self.assertAlmostEqual(means['dice_mean']['sample_sd'], np.sqrt(2))
        self.assertEqual(len(per_label), 35)
        self.assertEqual(e.stats([None, float('nan'), 3.]), {'n': 1, 'mean': 3., 'sample_sd': None})

    def test_case_order(self):
        pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
        self.assertEqual(len(pairs), 200)
        self.assertEqual(pairs[0], ('0333', '0115'))
        self.assertEqual(len(set(pairs[:100])), 100)

    def test_report_encoding_independent_of_container_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.md'
            e.write_report(path, ['Cases10\u201399; mean \u00b1 SD'])
            self.assertEqual(path.read_bytes().decode('utf-8'), 'Cases10\u201399; mean \u00b1 SD\n')

    def test_first10_parity_uses_tight_metric_specific_tolerances(self):
        reference = {key: .5 for key in e.SUMMARY_METRICS}
        row = dict(reference, diffusion_l2_full=.5 + 5e-10)
        e.validate_first10_metrics(row, reference)
        row['dice_mean'] += 1e-7
        with self.assertRaises(ValueError):
            e.validate_first10_metrics(row, reference)
        row = dict(reference, diffusion_l2_full=.5 + 1e-6)
        with self.assertRaises(ValueError):
            e.validate_first10_metrics(row, reference)


if __name__ == '__main__':
    unittest.main()
