"""CPU regression gates for live multi-GPU continuation and automatic OOM retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]/'scripts/experiments/oasis_polyaffine_scheduler'
sys.path.insert(0, str(HERE))
import dispatcher as d
from support import cuda_oom, atomic_json, sha


class SchedulerTests(unittest.TestCase):
    def queue(self, jobs=None):
        temporary = tempfile.TemporaryDirectory(prefix='p81_multigpu_')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        jobs = jobs or [dict(id='one', mode='svf', family='bspline', budgets=[768],
                            profile_key='test', dependencies=[])]
        manifest = dict(jobs=jobs, queue_policy={}, memory_profiles_mib={'test': 4096})
        atomic_json(root/'launch_manifest.json', manifest)
        revision = dict(gpu_limits={'0': 16, '1': 16, '2': 4, '3': 4},
                        gpu_uuids={'0': 'GPU-zero', '1': 'GPU-one', '2': 'GPU-two', '3': 'GPU-three'},
                        gpu_total_mib={'0': 49140, '1': 49140, '2': 49140, '3': 49140},
                        queue_policy=dict(cpu_threads=3, headroom_mib=2048, min_disk_gib=0,
                                          min_host_gib=0, poll_sec=60, kill_grace_sec=60,
                                          job_timeout_sec=43200, stall_timeout_sec=3600))
        path = root/'revision.json'
        atomic_json(path, revision)
        return d.MultiGPUQueue(root, manifest, path)

    def attempt(self, q, job, gpu=0):
        folder = q.root/'jobs'/job['id']/f"attempt{len(job['attempts'])}"
        folder.mkdir(parents=True)
        attempt = dict(folder=str(folder), required_mib=4096, physical_gpu=gpu,
                       started_epoch=time.time(), identity=None)
        job['attempts'].append(attempt)
        job.update(status='running', active_attempt=attempt)
        return folder

    def test_tiers_16_8_4_2_1_and_zero(self):
        candidates = [(str(n), 1000) for n in range(30)]
        for fit, expected in [(30,16),(16,16),(15,8),(8,8),(7,4),(4,4),(3,2),(2,2),(1,1),(0,0)]:
            tier, batch = d.plan_batch(dict(free_mib=2048+fit*1000), [], candidates, 16)
            self.assertEqual((tier, len(batch)), (expected, expected))

    def test_gpu_two_and_three_never_exceed_four(self):
        for cap in (1, 2, 4):
            tier, batch = d.plan_batch(dict(free_mib=100000), [], [(str(i),1) for i in range(30)], cap)
            self.assertEqual((tier,len(batch)), (cap,cap))
        with self.assertRaises(ValueError):
            d.plan_batch(dict(free_mib=10000), [], [], 32)

    def test_reserve_peak_growth_and_external_usage(self):
        active = [dict(required_mib=8000, current_reserved_mib=3000)]*2
        # 10GB of unobserved own growth plus2GB device headroom.
        tier, batch = d.plan_batch(dict(free_mib=18048), active, [('a',3000),('b',3000)],4)
        self.assertEqual((tier,len(batch)), (4,2))
        tier, batch = d.plan_batch(dict(free_mib=18047), active, [('a',3000),('b',3000)],4)
        self.assertEqual((tier,len(batch)), (2,0))

    def test_busy_gpu_does_not_evict_existing_workers(self):
        active = [dict(required_mib=4000, current_reserved_mib=3000)]*9
        tier, batch = d.plan_batch(dict(free_mib=100), active, [('a',1000)],16)
        self.assertEqual(tier,8)
        self.assertEqual(batch,[])
        self.assertEqual(len(active),9)

    def test_small_job_backfills_behind_large_job(self):
        tier, batch = d.plan_batch(dict(free_mib=6144), [], [('large',9000),('small',4096)],4)
        self.assertEqual((tier,batch), (1,[('small',4096)]))

    def test_oom_requeues_preserves_attempt_and_increases_reservation(self):
        q=self.queue();j=q.state['jobs'][0];folder=self.attempt(q,j,2)
        (folder/'worker.log').write_text('original OOM log')
        q.failure(j,'OutOfMemoryError: CUDA out of memory')
        self.assertEqual(j['status'],'pending')
        self.assertEqual(j['oom_retries'],1)
        self.assertGreater(j['required_mib_override'],4096)
        self.assertGreater(j['retry_not_before'],time.time()+50)
        self.assertEqual(j['attempts'][0]['outcome'],'cuda_oom')
        self.assertEqual((folder/'worker.log').read_text(),'original OOM log')
        self.assertEqual(len(list((q.root/'failures').glob('*.json'))),1)

    def test_structured_oom_and_backoff_cap(self):
        q=self.queue();j=q.state['jobs'][0];folder=self.attempt(q,j)
        atomic_json(folder/'failure.json',dict(cuda_oom=True,error='allocator failed'))
        j['oom_retries']=20
        j['active_attempt']['required_mib']=90000
        q.failure(j,'allocator failed')
        self.assertEqual(j['status'],'pending')
        self.assertLessEqual(j['retry_not_before']-time.time(),900)
        self.assertLess(j['required_mib_override'],49140-2048)

    def test_non_oom_failure_remains_terminal(self):
        q=self.queue();j=q.state['jobs'][0];self.attempt(q,j)
        q.failure(j,'solver_diverged: invalid finite checkpoint')
        self.assertEqual(j['status'],'failed')
        self.assertNotIn('oom_retries',j)
        self.assertFalse(cuda_oom('CPU MemoryError'))
        self.assertTrue(cuda_oom('CUDA error: out of memory'))
        self.assertTrue(cuda_oom('CUBLAS_STATUS_ALLOC_FAILED'))

    def test_resume_adopts_live_original_worker_without_relaunch(self):
        q=self.queue();j=q.state['jobs'][0]
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        self.addCleanup(lambda: child.poll() is None and child.terminate())
        self.addCleanup(lambda: None)
        self.attempt(q,j)
        j['active_attempt']['identity']=d.process_identity(child.pid)
        q.save()
        resumed=d.MultiGPUQueue(q.root,q.manifest,q.revision_path)
        with mock.patch.object(resumed,'launch_on') as launch:
            resumed.reconcile()
        launch.assert_not_called()
        self.assertEqual(resumed.state['jobs'][0]['status'],'running')
        self.assertEqual(resumed.state['jobs'][0]['active_attempt']['physical_gpu'],0)
        child.terminate();child.wait(timeout=5)

    def test_resume_requeues_preexisting_cuda_oom(self):
        q=self.queue();j=q.state['jobs'][0];self.attempt(q,j)
        j.update(status='failed',failure={'reason':'CUDA out of memory'})
        q.save()
        resumed=d.MultiGPUQueue(q.root,q.manifest,q.revision_path)
        self.assertEqual(resumed.state['jobs'][0]['status'],'pending')
        self.assertEqual(resumed.state['jobs'][0]['oom_retries'],1)

    def test_reservations_are_independent_per_gpu(self):
        jobs=[dict(id=str(i),budgets=[768],profile_key='test',dependencies=[]) for i in range(3)]
        q=self.queue(jobs)
        for job,gpu in zip(q.state['jobs'],(0,2,3)):
            folder=self.attempt(q,job,gpu)
            atomic_json(folder/'progress.json',dict(current_reserved_mib=1024))
        for gpu in (0,2,3):
            self.assertEqual(q.reservations(gpu),[dict(required_mib=4096,current_reserved_mib=1024)])

    def test_busy_gpu_zero_does_not_block_two_and_three(self):
        jobs=[dict(id=str(i),budgets=[768],profile_key='test',dependencies=[]) for i in range(8)]
        q=self.queue(jobs)
        snaps={g:dict(gpu=g,uuid=q.revision['gpu_uuids'][str(g)],total_mib=49140,
                     free_mib=100 if g in (0,1) else 20000) for g in (0,1,2,3)}
        launched=[]
        def launch(job,required,gpu):
            launched.append(gpu);self.attempt(q,job,gpu)
        with mock.patch.object(d,'validate_revision'),mock.patch.object(d,'gpu_snapshots',return_value=snaps),mock.patch.object(q,'launch_on',side_effect=launch):
            q.admit_once()
        self.assertEqual(launched,[2]*4+[3]*4)
        self.assertFalse(any(j['status']=='resource_deferred' for j in q.state['jobs']))

    def test_memory_is_refreshed_before_spawn(self):
        q=self.queue()
        busy={g:dict(gpu=g,uuid=q.revision['gpu_uuids'][str(g)],total_mib=49140,free_mib=0) for g in (0,1,2,3)}
        first={**busy,0:{**busy[0],'free_mib':20000}}
        with mock.patch.object(d,'validate_revision'),mock.patch.object(d,'gpu_snapshots',side_effect=[first,busy,busy,busy,busy]),mock.patch.object(q,'launch_on') as launch:
            q.admit_once()
        launch.assert_not_called()

    def test_retry_cooldown_does_not_block_regular_job(self):
        jobs=[dict(id=str(i),budgets=[768],profile_key='test',dependencies=[]) for i in range(2)]
        q=self.queue(jobs);q.state['jobs'][0]['retry_not_before']=time.time()+100
        snaps={g:dict(gpu=g,uuid=q.revision['gpu_uuids'][str(g)],total_mib=49140,free_mib=20000) for g in (0,2,3)}
        launched=[]
        def launch(job,required,gpu):
            launched.append(job['id']);self.attempt(q,job,gpu)
        with mock.patch.object(d,'validate_revision'),mock.patch.object(d,'gpu_snapshots',return_value=snaps),mock.patch.object(q,'launch_on',side_effect=launch):
            q.admit_once()
        self.assertEqual(launched,['1'])

    def test_case_that_cannot_fit_gpu_zero_redistributes_to_gpu_one(self):
        q=self.queue()
        snaps={g:dict(gpu=g,uuid=q.revision['gpu_uuids'][str(g)],total_mib=49140,
                     free_mib=12000 if g==1 else 1000) for g in (0,1,2,3)}
        launched=[]
        def launch(job,required,gpu):
            launched.append(gpu);self.attempt(q,job,gpu)
        with mock.patch.object(d,'validate_revision'),mock.patch.object(d,'gpu_snapshots',return_value=snaps),mock.patch.object(q,'launch_on',side_effect=launch):
            q.admit_once()
        self.assertEqual(launched,[1])

    def test_spawn_masks_physical_gpu_uuid_and_records_durable_assignment(self):
        q=self.queue();j=q.state['jobs'][0]
        child=mock.Mock(pid=123456)
        with mock.patch.object(d.subprocess,'Popen',return_value=child) as spawn,mock.patch.object(d,'process_identity',return_value=dict(pid=123456,start_ticks='1',uid=os.getuid())):
            q.launch_on(j,4096,1)
        args,kwargs=spawn.call_args
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-one')
        self.assertEqual(args[0][-2:],['--gpu','1'])
        state=json.loads(q.path.read_text())
        self.assertEqual(state['jobs'][0]['active_attempt']['physical_gpu'],1)
        self.assertEqual(state['running_per_gpu'],{'1':1})

    def test_stale_oom_file_does_not_reclassify_unrelated_failure(self):
        q=self.queue();j=q.state['jobs'][0];folder=self.attempt(q,j)
        atomic_json(folder/'failure.json',dict(cuda_oom=True,error='CUDA out of memory'))
        q.failure(j,'preexisting_attempt_directory_preserved')
        self.assertEqual(j['status'],'failed')
        self.assertNotIn('oom_retries',j)

    def test_retry_checkpoint_preserves_pointer_and_restores_exact_coefficients(self):
        import numpy as np
        import torch
        from worker_entry import ContinuationWorker
        q=self.queue();j=q.state['jobs'][0];folder=self.attempt(q,j)
        basis=mock.Mock();basis.to_dict.return_value={'basis':'fixture'}
        atomic_json(folder/'basis.json',basis.to_dict())
        np.save(folder/'coeff.npy',np.array([[1.,2.,3.]]))
        (folder/'field.npy').write_bytes(b'fixture')
        endpoint={'status':'complete','job_id':j['id']}
        for kind,name in [('basis','basis.json'),('coefficients','coeff.npy'),('field','field.npy')]:
            endpoint[kind+'_path']=str(folder/name);endpoint[kind+'_sha256']=sha(folder/name)
        pointer=q.root/'endpoints'/j['id']/'cap768.json';atomic_json(pointer,endpoint)
        prior=pointer.read_bytes()
        worker=ContinuationWorker.__new__(ContinuationWorker)
        worker.root=q.root;worker.job=j;worker.endpoints=[];worker.reused_endpoints=[];worker.progress=mock.Mock()
        coeff=torch.zeros((1,3),dtype=torch.float64)
        worker.save_endpoint(basis,coeff,768)
        self.assertEqual(pointer.read_bytes(),prior)
        self.assertEqual(worker.endpoints,[str(pointer)])
        torch.testing.assert_close(coeff,torch.tensor([[1.,2.,3.]],dtype=torch.float64))
        basis.to_dict.return_value={'different':'geometry'}
        with self.assertRaisesRegex(RuntimeError,'Retry tree differs'):
            worker.save_endpoint(basis,coeff,768)


if __name__=='__main__':
    unittest.main()
