"""Fault injection for queue isolation, persistence, resource gates and reporting."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
import queue_oasis_svf_continue_on_error as q
import report_oasis_svf_queue_outcome as report


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/q.FOLDER).mkdir()
        self.manifest = {'cases': {str(i): {'fixed_id': f'f{i}', 'moving_id': f'm{i}'} for i in range(10)}}
        self.queue = q.Queue(self.root, self.manifest)

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, rank=0, status='pending', attempt=0):
        spec = q.r.s.Spec(rank, '842', 'velocity', 1.)
        job = {'spec': spec, 'status': status, 'attempt': attempt, 'gpu': '0', 'oom_retries': 0,
               'out': str(self.root/spec.name()/f'attempt{attempt}')}
        self.queue.jobs.append(job)
        return job

    def active(self, job, code, action='worker', pid=1):
        item = {'job': job, 'process': Mock(poll=Mock(return_value=code)), 'record': {'pid': pid},
                'action': action, 'last_activity': 0}
        self.queue.active[pid] = item
        return item

    def test_worker_failure_does_not_stop_next_configuration(self):
        failed, next_job = self.job(status='running'), self.job(rank=1)
        self.active(failed, 1)
        self.queue.poll()
        self.assertEqual(failed['status'], 'failed_deferred')
        self.assertIs(q.choose_job(self.queue.jobs, '0', {}), next_job)
        self.assertEqual(len(list((self.root/q.FOLDER/'failures').glob('*.json'))), 1)

    def test_auditor_parent_validation_exception_is_local(self):
        failed, next_job = self.job(status='auditing'), self.job(rank=1)
        self.active(failed, 0, 'audit')
        with patch.object(q.r, 'validate', side_effect=ValueError('bad hash')):
            self.queue.poll()
        self.assertEqual(failed['failure']['action'], 'parent_validation')
        self.assertIs(q.choose_job(self.queue.jobs, '0', {}), next_job)

    def test_exit_zero_requires_done(self):
        job = self.job(status='running')
        self.active(job, 0)
        self.queue.poll()
        self.assertEqual(job['status'], 'failed_deferred')

    def test_malformed_progress_does_not_change_job(self):
        job = self.job(status='running')
        path = Path(job['out']); path.mkdir(parents=True)
        (path/'progress.json').write_text('{')
        self.active(job, None)
        status = self.queue.progress({'rows': {'0': {}}, 'errors': []})
        self.assertEqual(job['status'], 'running')
        self.assertIn('warning', status['active_phases'][0]['detail'])

    def test_gpu_telemetry_partial_rows_and_timeout(self):
        snapshot = q.parse_gpus('0, 16000, 10000, 90\n1, N/A, 3, 4\n2, 20000, 3000, 1')
        self.assertEqual(set(snapshot['rows']), {'0', '2'})
        self.assertEqual(snapshot['missing_gpus'], ['1', '3'])
        with patch.object(q.subprocess, 'run', side_effect=q.subprocess.TimeoutExpired('nvidia-smi', 15)):
            self.assertEqual(q.gpu_snapshot()['rows'], {})
        self.job()
        progress = self.queue.progress({'rows': {}, 'errors': ['timeout']})
        self.assertEqual(progress['phase'], 'waiting_gpu_telemetry')
        self.assertIsNone(progress['eta_sec_observed_processed_throughput'])

    def test_persistent_conservative_memory_reservation_and_slots(self):
        job = self.job(status='running')
        self.active(job, None)
        self.assertFalse(q.capacity('0', self.queue.active, 15000, 8192))
        self.assertTrue(q.capacity('0', self.queue.active, 17000, 8192))
        self.active(job, None, pid=2)
        self.assertFalse(q.capacity('0', self.queue.active, 40000, 8192))
        active = {i: {'job': {'gpu': str(i//2)}} for i in range(8)}
        self.assertFalse(q.capacity('3', active, 40000, 8192))

    def test_oom_budget_persists_and_retry_waits_until_tail(self):
        job, other = self.job(status='running'), self.job(rank=1)
        self.queue.failure(job, 'worker', 'exit1', tail='CUDA out of memory', allow_oom=True)
        self.assertEqual(job['status'], 'oom_retry')
        self.assertEqual(job['oom_retries'], 1)
        self.assertIs(q.choose_job(self.queue.jobs, '0', {}), other)
        state = q.r.c.read_json(self.root/q.FOLDER/'state.json')
        self.assertEqual(state['jobs'][0]['oom_retries'], 1)
        other['status'] = 'failed_deferred'
        self.assertIs(q.choose_job(self.queue.jobs, '0', {}), job)
        self.queue.failure(job, 'worker', 'exit1', tail='CUDA out of memory', allow_oom=True)
        self.assertEqual(job['status'], 'failed_deferred')

    def test_failure_save_error_does_not_consume_retry_budget(self):
        job = self.job(status='running')
        with patch.object(self.queue, 'save', side_effect=OSError('NFS unavailable')):
            with self.assertRaises(OSError):
                self.queue.failure(job, 'worker', 'OOM', tail='CUDA out of memory', allow_oom=True)
        self.assertEqual(job['status'], 'running')
        self.assertEqual(job['oom_retries'], 0)

    def test_only_current_invocation_controls_oom_classification(self):
        job = self.job(status='running')
        path = Path(job['out']).parent; path.mkdir(parents=True)
        (path/'old.log').write_text('CUDA out of memory')
        current = path/'current.log'; current.write_text('Numerical check failed')
        job['log_path'] = str(current)
        self.active(job, 1)
        self.queue.poll()
        self.assertEqual(job['status'], 'failed_deferred')
        self.assertFalse(job['failure']['oom'])

    def test_fast_exit_before_proc_metadata_is_safe(self):
        job = self.job()
        proc = Mock(pid=123, poll=Mock(return_value=1))
        with patch.object(q.subprocess, 'Popen', return_value=proc), patch.object(q, 'process_info', return_value=None):
            self.queue.launch(job, '0', {'rows': {}})
        self.assertEqual(self.queue.active[123]['record']['pid'], 123)
        self.queue.progress({'rows': {}, 'errors': []})
        self.queue.poll()
        self.assertEqual(job['status'], 'failed_deferred')

    def test_spawn_failure_is_local(self):
        job, other = self.job(), self.job(rank=1)
        with patch.object(q.subprocess, 'Popen', side_effect=OSError('Resource temporarily unavailable')):
            self.queue.launch(job, '0', {'rows': {}})
        self.assertEqual(job['status'], 'failed_deferred')
        self.assertIs(q.choose_job(self.queue.jobs, '0', {}), other)

    def test_launch_intent_storage_failure_rolls_back(self):
        job = self.job()
        with patch.object(self.queue, 'save', side_effect=OSError('shared disk failed')):
            with self.assertRaises(OSError): self.queue.launch(job, '0', {})
        self.assertEqual(job['status'], 'pending')
        self.assertFalse(self.queue.active)

    def test_reconcile_success_overrides_old_failed_attempt(self):
        job = self.job(attempt=1)
        old = Path(job['out']).parent/'attempt0'; old.mkdir(parents=True)
        (old/'FAILED_worker.json').write_text('unchanged')
        good = Path(job['out']); good.mkdir(); (good/'independent_audit.json').write_text('{}')
        self.queue.jobs.clear()
        with patch.object(q.r.s, 'matrix', return_value=[job['spec']]), patch.object(q.r, 'validate', return_value={'gpu': '3'}):
            self.queue.reconcile([])
        self.assertEqual(self.queue.jobs[0]['status'], 'complete')
        self.assertEqual((old/'FAILED_worker.json').read_text(), 'unchanged')

    def test_reconcile_historical_failure_and_unattempted(self):
        failed, other = self.job(), self.job(rank=1)
        path = Path(failed['out']); path.mkdir(parents=True)
        (path/'FAILED_worker.json').write_text('{"error": "transition"}')
        self.queue.jobs.clear()
        with patch.object(q.r.s, 'matrix', return_value=[failed['spec'], other['spec']]):
            self.queue.reconcile([])
        self.assertEqual([j['status'] for j in self.queue.jobs], ['failed_deferred', 'pending'])
        # Durable deferral survives restart and does not produce a new numerical attempt.
        self.queue.jobs.clear()
        with patch.object(q.r.s, 'matrix', return_value=[failed['spec'], other['spec']]):
            self.queue.reconcile([])
        self.assertEqual(self.queue.jobs[0]['status'], 'failed_deferred')

    def test_adopt_child_even_before_directory_created(self):
        job = self.job()
        self.queue.jobs.clear()
        rec = {'pid': 88, 'start_ticks': '123', 'out': job['out'], 'action': 'worker', 'gpu': '2'}
        with patch.object(q.r.s, 'matrix', return_value=[job['spec']]):
            self.queue.reconcile([rec])
        self.assertEqual(self.queue.jobs[0]['status'], 'running')
        self.assertEqual(self.queue.active[88]['record'], rec)

    def test_watchdog_checks_identity_and_retains_slot(self):
        job = self.job(status='running')
        rec = {'pid': 999, 'start_ticks': '1', 'uid': 1009, 'argv': ['verified worker']}
        item = self.active(job, None); item.update(record=rec, last_activity=0, signature=['same'])
        with patch.object(q, 'activity', return_value=['same']), patch.object(q.time, 'time', return_value=3601), \
             patch.object(q, 'process_info', return_value=rec), patch.object(q, 'same_process', return_value=True), \
             patch.object(q.os, 'kill') as kill:
            self.queue.watchdog(item)
        kill.assert_called_once_with(999, q.signal.SIGTERM)
        self.assertTrue(self.queue.active)
        self.assertIn('terminated', item)

    def test_watchdog_never_kills_reused_pid(self):
        job = self.job(status='running'); item = self.active(job, None)
        item.update(record={'pid': 4, 'start_ticks': '1', 'argv': ['worker']}, signature=['same'], last_activity=0)
        with patch.object(q, 'activity', return_value=['same']), patch.object(q.time, 'time', return_value=4000), \
             patch.object(q, 'process_info', return_value={'uid': 999, 'argv': ['other']}), patch.object(q.os, 'kill') as kill:
            self.queue.watchdog(item)
        kill.assert_not_called()

    def test_end_to_end_fault_then_later_success(self):
        first, second = self.job(status='running'), self.job(rank=1)
        self.active(first, 1)
        self.queue.poll()
        proc = Mock(pid=789, poll=Mock(return_value=0))
        with patch.object(q.subprocess, 'Popen', return_value=proc), patch.object(q, 'process_info', return_value=None):
            self.queue.launch(q.choose_job(self.queue.jobs, '0', {}), '0', {})
        path = Path(second['out']); path.mkdir()
        (path/'DONE').write_text('done')
        self.queue.poll()
        self.assertEqual(second['status'], 'audit_pending')
        with patch.object(q.subprocess, 'Popen', return_value=proc), patch.object(q, 'process_info', return_value=None):
            self.queue.launch(second, '0', {})
        with patch.object(q.r, 'validate', return_value={'gpu': '0'}):
            self.queue.poll()
        self.assertEqual([j['status'] for j in self.queue.jobs], ['failed_deferred', 'complete'])

    def test_partial_report_no_incomplete_means_or_pair_keyerror(self):
        jobs, rows = [], []
        for rank in range(10):
            spec = q.r.s.Spec(rank, '842', 'velocity', 1.)
            jobs.append({'spec': asdict(spec), 'status': 'complete', 'attempt': 0, 'gpu': '0'})
            rows.append({'spec': asdict(spec), 'case_rank': rank, 'run_name': spec.name(), 'run_dir': 'test',
                         **{k: 1. for k in report.METRICS}})
        extra = q.r.s.Spec(0, '8421', 'velocity', 1.)
        jobs.append({'spec': asdict(extra), 'status': 'failed_deferred', 'attempt': 0, 'failure': 'check'})
        payload = report.generate(jobs, rows, self.manifest, self.root/'report')
        self.assertEqual(len(payload['summary']), 1)
        self.assertEqual(payload['group_coverage'][1]['accepted_n'], 0)
        self.assertFalse(payload['all_scientifically_accepted'])
        self.assertTrue(any(not x['pair_available'] for x in payload['paired_differences']))
        self.assertFalse((self.root/'report'/'ACCEPTED.json').exists())

    def test_cli_only_approved_concurrency(self):
        self.assertEqual(q.parse_args([]).max_concurrent, 8)
        with self.assertRaises(SystemExit): q.parse_args(['--workers-per-gpu', '3'])


if __name__ == '__main__':
    unittest.main()
