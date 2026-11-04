"""P79 coverage, frozen adapters, reporting and durable-queue fault injection."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import copy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
sys.path.insert(0,str(ROOT/'scripts/reports'))
import oasis_bspline842_first100 as e
import queue_oasis_bspline842_first100 as q
import run_oasis_bspline842_first100 as w
import report_oasis_bspline842_first100 as report


class ContractTests(unittest.TestCase):
    def test_full_and_missing_matrix(self):
        full=e.matrix();missing=e.matrix(missing=True)
        self.assertEqual(len(full),2100);self.assertEqual(len(missing),1530)
        self.assertEqual(len({s.name() for s in full}),2100)
        self.assertEqual({f:sum(s.family==f for s in missing) for f in e.FAMILIES},
                         {'displacement':270,'bounded2':630,'svf_velocity':630})
        self.assertTrue(all(s.case_rank>=10 for s in missing))
        self.assertEqual(len(e.matrix(smoke=True)),6)
        self.assertFalse({s.name() for s in full}&{s.name() for s in e.matrix(smoke=True)})

    def test_invalid_spec(self):
        for rank,family,alpha in [(-1,'displacement',1),(100,'bounded2',1),(True,'bounded2',1),
                                  (0,'dense',1),(0,'svf_velocity',float('nan')),(0,'bounded2',.2)]:
            with self.assertRaises(ValueError):e.Spec(rank,family,alpha).name()

    def test_cli(self):
        a=w.parse_args(['worker','--manifest','/tmp/m','--out-dir','/tmp/job','--gpu','3',
                       '--case-rank','99','--family','svf_velocity','--alpha','100'])
        self.assertEqual(a.case_rank,99)
        with self.assertRaises(SystemExit):w.parse_args(['worker'])

    def test_path_settings_filter_mixed_controller_metadata(self):
        paths={k:f'/tmp/{k}' for k in e.PATH_SETTINGS}
        mixed={**paths,'gpu_slots':['0','0','1','1'],'cpu_threads':4,
               'dry_run':False,'stop_after':'full','out_root':'/tmp/unrelated'}
        self.assertEqual(e.worker_settings(mixed),paths)
        self.assertEqual(mixed['gpu_slots'],['0','0','1','1'])
        for value in ([],None,True,4,''):
            with self.assertRaises(ValueError):
                e.worker_settings({**mixed,'pair_file':value})
        with self.assertRaises(KeyError):e.worker_settings({})

    def test_legacy_svf_setup_with_filtered_manifest(self):
        # Exercise the actual legacy Path loop that the previous tests missed,
        # without CUDA or image IO. The numerical setup function is unmodified.
        paths={k:f'/tmp/{k}' for k in e.PATH_SETTINGS}
        mixed={**paths,'gpu_slots':['0','1'],'cpu_threads':4,'dry_run':False}
        batch={'fixed':w.torch.tensor([0.,1.]),'fixed_mask':w.torch.tensor([False,True]),
               'fixed_id':'0275','moving_id':'0402'}
        case={'input_hashes':{},'fixed_id':'0275','moving_id':'0402'}
        manifest={'settings':mixed,'cases':{'10':case}}
        spec=Mock(case_rank=10)
        with patch.object(w.svf,'assert_manifest'), patch.object(w.svf.c,'set_cuda_device',return_value='cpu'), \
             patch.object(w.svf.legacy,'load_batch',return_value=(batch,{})), \
             patch.dict(w.os.environ,{},clear=False):
            with self.assertRaises(TypeError):w.svf.setup(SimpleNamespace(gpu='0'),manifest,spec)
            args=SimpleNamespace(gpu='0')
            device,loaded,identity=w.svf.setup(args,{**manifest,'settings':e.worker_settings(mixed)},spec)
        self.assertEqual(device,'cpu');self.assertIs(loaded,batch);self.assertEqual(identity,case)
        self.assertEqual(args.pair_file,Path(paths['pair_file']))
        self.assertFalse(hasattr(args,'gpu_slots'));self.assertEqual(args.gpu,'0')

    def test_recovery_root_is_distinct_and_validated(self):
        root=Path('/tmp/artifacts')
        old=e.run_root(root);new=e.run_root(root,'metadata_recovery_v2')
        self.assertEqual(new.parent,old);self.assertNotEqual(old,new)
        with self.assertRaises(ValueError):e.run_root(root,'../unsafe')
        args=q.parse_args(['--container-image','image','--container-image-id','digest',
                          '--execution','metadata_recovery_v2'])
        self.assertEqual(args.execution,'metadata_recovery_v2')

    def test_recovery_receipts_require_unchanged_numerics_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=e.run_root(root);origin.mkdir(parents=True)
            old={k:{} for k in ('variant','protocol','cases','reused','references','reference_hashes',
                               'pair_file_sha256','bounds','sitreg_root','sitreg_revision','recovery')}
            old['settings']={**{k:f'/tmp/{k}' for k in e.PATH_SETTINGS},'gpu_slots':['0','1']}
            old['source_hashes']={k:{'file':'digest'} for k in ('displacement','bounded2','svf','svf_recovery','extension')}
            path=origin/'launch_manifest.json';e.atomic_json(path,old)
            e.atomic_json(origin/'REUSE_VERIFIED.json',{'passed':True,'count':570,'reference_count':200,
                                                      'manifest_sha256':e.sha(path)})
            new=copy.deepcopy(old);new['settings']=e.worker_settings(old['settings'])
            new['source_hashes']['extension']={'file':'approved-new-adapter'}
            self.assertEqual(e.receipt_origin(root,new)[0],origin)
            bad=copy.deepcopy(new);bad['source_hashes']['svf']={'file':'changed'}
            with self.assertRaises(RuntimeError):e.receipt_origin(root,bad)
            bad=copy.deepcopy(new);bad['cases']={'99':'different'}
            with self.assertRaises(RuntimeError):e.receipt_origin(root,bad)
            e.atomic_json(path,{**old,'unapproved':'mutation'})
            with self.assertRaises(RuntimeError):e.receipt_origin(root,new)

    def test_rank99_adapter_and_unchanged_objectives(self):
        old=(w.d.FitSpec.validate,w.d.VARIANT,w.bounded.VARIANT,w.bounded.run_name,w.bounded.assert_manifest,
             w.s.Spec,w.s.VARIANT,w.s.PROTOCOL,w.svf.assert_manifest)
        functions=(w.d.StageObjective,w.d.optimize_stage,w.bounded.b.Objective,w.bounded.b.optimize,w.s.Objective,w.s.optimize)
        try:
            w.install()
            spec=w.s.Spec(99,'842','velocity',1.)
            spec.validate()
            stages=w.s.stages(spec)
            self.assertEqual([x.cps for x in stages],[8,4,2])
            self.assertEqual([x.lr for x in stages],[.03,.01,.003])
            self.assertEqual([x.max_epochs for x in stages],[400,800,1300])
            self.assertEqual([x.max_lr_reductions for x in stages],[0,0,0])
            self.assertEqual(functions,(w.d.StageObjective,w.d.optimize_stage,w.bounded.b.Objective,w.bounded.b.optimize,w.s.Objective,w.s.optimize))
            with self.assertRaises(ValueError):w.s.Spec(99,'842','displacement',1.).validate()
        finally:
            (w.d.FitSpec.validate,w.d.VARIANT,w.bounded.VARIANT,w.bounded.run_name,w.bounded.assert_manifest,
             w.s.Spec,w.s.VARIANT,w.s.PROTOCOL,w.svf.assert_manifest)=old
        with self.assertRaises(ValueError):w.d.FitSpec(99,'pyramid',1.).validate()

    def test_complete_only_sample_sd(self):
        rows=[{'case_rank':i,**{k:float(i) for k in e.METRICS}} for i in range(10)]
        a=report.summarize(rows,range(10))
        self.assertAlmostEqual(a['dice_mean'],4.5);self.assertAlmostEqual(a['dice_sd'],(82.5/9)**.5)
        self.assertEqual(a['peak_reserved_gib_n'],0)
        for bad in (rows[:-1],rows+rows[:1]):
            with self.assertRaises(ValueError):report.summarize(bad,range(10))

    def test_receipt_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a';p.write_text('first')
            r={'passed':True,'files':{str(p):{**e.fingerprint(p),'sha256':e.sha(p)}}}
            e.verify_receipt(r);p.write_text('changed length')
            with self.assertRaises(RuntimeError):e.verify_receipt(r)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.old=(q.base.FOLDER,q.base.POLICY,q.base.choose_job,q.base.r,q.base.sources)
        self.spec=e.Spec(10,'bounded2',.01)
        self.manifest={'cases':{'10':{'fixed_id':'0275','moving_id':'0402'}},'sitreg_root':'/tmp'}
        self.queue=q.Queue(self.root,self.manifest,[self.spec])

    def tearDown(self):
        q.base.FOLDER,q.base.POLICY,q.base.choose_job,q.base.r,q.base.sources=self.old
        self.tmp.cleanup()

    def job(self,status='pending',family='bounded2'):
        spec=e.Spec(10,family,.01)
        j={'spec':spec,'status':status,'attempt':0,'gpu':'0','oom_retries':0,
           'out':str(self.root/spec.name()/'attempt0')}
        self.queue.jobs.append(j);return j

    def test_failure_continues_queue(self):
        bad=self.job('running');next_job=self.job(family='displacement')
        self.queue.active[999]={'process':Mock(poll=Mock(return_value=1)),'job':bad,'action':'worker'}
        self.queue.poll()
        self.assertEqual(bad['status'],'failed_deferred')
        self.assertIs(q.choose_job(self.queue.jobs,'0',{}),next_job)

    def test_audit_gpu_is_mandatory_not_preference(self):
        audit=self.job('audit_pending')
        self.assertIsNone(q.choose_job([audit],'1',{}))
        self.assertIs(q.choose_job([audit],'0',{}),audit)

    def test_single_oom_retry_after_ordinary_jobs(self):
        job=self.job('running');other=self.job(family='displacement')
        self.queue.failure(job,'worker','oom',tail='CUDA out of memory',allow_oom=True)
        self.assertEqual(job['status'],'oom_retry')
        self.assertIs(q.choose_job(self.queue.jobs,'0',{}),other)
        other['status']='complete'
        self.assertIs(q.choose_job(self.queue.jobs,'0',{}),job)
        self.queue.failure(job,'worker','oom',tail='CUDA out of memory',allow_oom=True)
        self.assertEqual(job['status'],'failed_deferred')

    def test_memory_reservation(self):
        active={1:{'job':{'gpu':'0'}}}
        self.assertFalse(q.base.capacity('0',active,15000,8192))
        self.assertTrue(q.base.capacity('0',active,17000,8192))
        active[2]={'job':{'gpu':'0'}}
        self.assertFalse(q.base.capacity('0',active,48000,8192))

    def test_audit_validation_failure_is_local(self):
        bad=self.job('auditing');other=self.job(family='svf_velocity')
        self.queue.active[999]={'process':Mock(poll=Mock(return_value=0)),'job':bad,'action':'audit'}
        with patch.object(q.base.r,'validate',side_effect=ValueError('bad replay')):self.queue.poll()
        self.assertEqual(bad['status'],'failed_deferred')
        self.assertIs(q.choose_job(self.queue.jobs,'0',{}),other)

    def test_resume_adopts_live_worker(self):
        rec={'pid':123,'out':str(self.root/self.spec.name()/'attempt0'),'gpu':'2','action':'worker','argv':[],'start_ticks':'123'}
        self.queue.reconcile([rec])
        self.assertEqual(len(self.queue.active),1)
        self.assertEqual(self.queue.jobs[0]['status'],'running')
        self.assertEqual(self.queue.jobs[0]['gpu'],'2')


if __name__=='__main__':unittest.main()
