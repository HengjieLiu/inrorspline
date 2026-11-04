"""Iterator/RNG recovery, squared SITReg semantics and all-four dependency gates."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
import oasis_resume_state as recovery
import sitreg_gncc2_training as squared
import sitreg_gncc_training as signed
import fedda_gncc2_training as fedda
import queue_oasis_fedda_recovery_then_sitreg_gncc2 as queue


class FakeDataset:
    def __init__(self, **kwargs):
        self.items = list(range(300))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        fixed = self.items.copy()
        del fixed[i]
        random.shuffle(fixed)
        return str(i), str(fixed[0])


class Tests(unittest.TestCase):
    def test_replay_at_partial_and_exhausted_sampler_boundaries(self):
        for epoch in (1, 6, 7, 12):
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            loader = torch.utils.data.DataLoader(FakeDataset(), batch_size=1, shuffle=True,
                                                num_workers=0, pin_memory=True)
            val = torch.utils.data.DataLoader(range(30), batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True)
            iterator, digest, expected = iter(loader), hashlib.sha256(), {}
            for e in range(1, epoch+1):
                for _ in range(50):
                    try: moving, fixed = next(iterator)
                    except StopIteration:
                        iterator = iter(loader); moving, fixed = next(iterator)
                    digest.update(f'{moving[0]}>{fixed[0]}\n'.encode())
                expected[str(e)] = digest.hexdigest()
                for _ in val: pass
            saved = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
            def more(it, loader):
                rows = []
                for _ in range(80):
                    try: pair = next(it)
                    except StopIteration: it = iter(loader); pair = next(it)
                    rows.append(pair)
                return rows
            after = more(iterator, loader)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            data = types.SimpleNamespace(myDataset_OASIS_v1_json=FakeDataset,
                                         myDataset_OASIS_v1_json_UnitTest=FakeDataset)
            restored, iterator, h = recovery.restore_iterator(data, Path('/unused'), epoch, expected, saved)
            self.assertEqual(h.hexdigest(), digest.hexdigest())
            self.assertEqual(more(iterator, restored), after)

    def test_sitreg_only_similarity_changes(self):
        differences = {k for k in squared.PROTOCOL if squared.PROTOCOL[k] != signed.PROTOCOL[k]}
        self.assertEqual(differences, {'loss_key', 'loss'})
        self.assertEqual(squared.LAMBDAS, signed.LAMBDAS)
        self.assertEqual(len({squared.run_name(k) for k in squared.LAMBDAS}), 4)
        self.assertTrue(all(squared.run_name(k) != signed.run_name(k) for k in squared.LAMBDAS))
        a = torch.randn(2, 1, 3, 4, 5, dtype=torch.double)
        b = torch.randn_like(a, requires_grad=True)
        x, y = squared.GlobalNCC()(a, b), fedda.GlobalNCCSquared()(a, b)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(x, b, retain_graph=True)[0], torch.autograd.grad(y, b)[0], rtol=0, atol=0)
        self.assertAlmostEqual(float(squared.GlobalNCC()(a, -a)), 0., places=12)
        with self.assertRaises(ValueError): squared.GlobalNCC()(a.long(), b.long())

    def test_barrier_requires_all_four_evaluations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); train, evaluated = root/'train', root/'eval'
            train.mkdir(); evaluated.mkdir(); (train/'COMPLETED').touch()
            for key in fedda.LAMBDAS:
                t, e = train/fedda.run_name(key), evaluated/fedda.run_name(key)
                t.mkdir(); e.mkdir(); (t/'COMPLETED').touch(); (e/'COMPLETED').touch()
                (t/'training_summary.json').write_text(json.dumps(dict(epoch=200, adam_updates=20000)))
                (e/'summary.json').write_text(json.dumps(dict(case_count=100, label_rows=3500)))
                (e/'per_case.json').write_text(json.dumps([dict(rank=r) for r in range(100)]))
            queue.require_fedda_barrier(train, evaluated)
            (e/'FAILED.txt').touch()
            with self.assertRaises(RuntimeError): queue.require_fedda_barrier(train, evaluated)

    def test_resume_source_uses_real_newlines_and_preserves_adam(self):
        source = (ROOT/'scripts/experiments/oasis_alpha_sweep/resume_oasis_fedda_gncc2.py').read_text()
        self.assertNotIn("json.dumps(r)+'\\\\n'", source)
        self.assertIn("optimizer.load_state_dict(saved['optimizer'])", source)
        self.assertIn('range(epoch0+1, 201)', source)
        self.assertIn("open('a')", source)

    def test_three_loss_report_wrapper(self):
        sys.path.insert(0, str(ROOT/'scripts/reports'))
        import oasis_sitreg_gncc2_report as report
        pairs = report.c.parse_pair_file(report.c.DEFAULT_PAIR_FILE)[:100]
        def records(losses):
            rows, means = [], []
            for loss in losses:
                keys = {**squared.LAMBDAS, **({'10': 10.} if loss == 'lncc9' else {})}
                for key, alpha in keys.items():
                    mean = dict(loss=loss, lambda_key=key, alpha=alpha, n=100,
                                historical=loss == 'lncc9' and key == '1')
                    for metric in report.METRICS:
                        mean[metric+'_mean'], mean[metric+'_std'] = .5, .1
                    means.append(mean)
                    for rank, (fixed, moving) in enumerate(pairs):
                        rows.append(dict(loss=loss, lambda_key=key, alpha=alpha, rank=rank,
                                         fixed_id=fixed, moving_id=moving,
                                         **{m: .5 for m in report.METRICS}))
            return dict(per_case=rows, means=means, training=[], validation_selected={})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); base = root/'repo_experiments/baseline'; base.mkdir(parents=True)
            signed_root = root/'repo_experiments/oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1'
            signed_root.mkdir(); (signed_root/'DONE').touch()
            (signed_root/'results.json').write_text(json.dumps(records(('lncc9', 'gncc'))))
            (signed_root/'ACCEPTED.json').write_text(json.dumps(dict(complete=True, outputs_sha256={
                'results.json': squared.sha256(signed_root/'results.json')})))
            out = root/'out'; out.mkdir()
            def component(train, evaluated, baseline, directory, refs):
                (directory/'results.json').write_text(json.dumps(records(('lncc9', 'gncc2'))))
            with patch.object(report, 'write_two_loss_report', component):
                report.write_report(root/'train', root/'eval', base, out, {})
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(len(result['per_case']), 1300)
            self.assertEqual(len(result['paired']), 800)
            self.assertTrue((out/'sitreg_three_ncc_stdlogj_arc_2x3.png').is_file())


if __name__ == '__main__':
    unittest.main()
