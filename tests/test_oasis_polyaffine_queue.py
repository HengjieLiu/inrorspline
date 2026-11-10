"""CPU-only failure/restart/admission tests for the GPU0 queue contract."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]/"scripts/experiments/oasis_polyaffine_fieldfit"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("polyaffine_controller",HERE/"controller.py")
controller=importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
launch_spec=importlib.util.spec_from_file_location("polyaffine_launch",HERE/"launch.py")
launch=importlib.util.module_from_spec(launch_spec)
launch_spec.loader.exec_module(launch)


def manifest(jobs):
    return {
        "gpu":0,"gpu0_uuid":"GPU-test-zero","code_hashes":{},
        "input_manifest_sha256":"unused",
        "memory_profiles_mib":{"tiny":1000},
        "queue_policy":dict(max_jobs=8,cpu_threads=1,kill_grace_sec=.05,
                            job_timeout_sec=10,stall_timeout_sec=10,
                            admission_timeout_sec=10,poll_sec=.01,
                            headroom_mib=2048,min_disk_gib=0,min_host_gib=0),
        "jobs":[dict(budgets=[768],**j) if "budgets" not in j else j for j in jobs],
    }


class QueueTests(unittest.TestCase):
    def make_queue(self,jobs):
        temp=tempfile.TemporaryDirectory(prefix="p81_queue_test_")
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        m=manifest(jobs)
        (root/"launch_manifest.json").write_text(json.dumps(m))
        return controller.Queue(root,m)

    def attempt(self,q,job,identity=None):
        folder=q.root/"jobs"/job["id"]/"attempt0"
        folder.mkdir(parents=True)
        a=dict(folder=str(folder),started_epoch=time.time(),required_mib=1000,
               last_activity=time.time(),identity=identity)
        job.update(status="running",active_attempt=a,attempts=[a])
        return folder

    def completion(self,q,job,folder,write_result=True):
        pointers=[]
        for cap in job["budgets"]:
            endpoint=dict(status="complete",job_id=job["id"],budget_cap=cap)
            for kind in ("basis","coefficients","field"):
                file=folder/f"{kind}_{cap}.fixture"
                file.write_bytes(f"serialized {kind} for cap {cap}".encode())
                endpoint[kind+"_path"]=str(file)
                endpoint[kind+"_sha256"]=controller.sha(file)
            pointer=q.root/"endpoints"/job["id"]/f"cap{cap}.json"
            pointer.parent.mkdir(parents=True,exist_ok=True)
            pointer.write_text(json.dumps(endpoint))
            pointers.append(str(pointer))
        result=dict(status="complete",job_id=job["id"],endpoints=pointers)
        if write_result:(folder/"result.json").write_text(json.dumps(result))
        return result

    def test_eight_job_hard_cap_even_if_caller_requests_more(self):
        active=[dict(required_mib=1000,current_reserved_mib=1000) for _ in range(8)]
        snap=dict(free_mib=40000)
        self.assertFalse(controller.can_admit(snap,active,1000,max_jobs=99))
        self.assertTrue(controller.can_admit(snap,active[:7],1000,max_jobs=99))
        self.assertFalse(controller.can_admit(snap,active[:2],1000,max_jobs=2))

    def test_full_svf_matrix_matches_dvf_for_both_initializations(self):
        targets=[dict(id=f"rank{rank:03d}_{source}",pair_rank=rank,source_method=source)
                 for rank in range(10) for source in ("sitreg","unbounded","bounded2","svf")]
        jobs=launch.build_jobs(targets)
        def endpoints(mode,initialization):
            return {(j["target_id"],j["family"],cap) for j in jobs
                    if j["mode"]==mode and j["initialization"]==initialization and j["kind"]=="main"
                    for cap in j["budgets"]}
        direct=endpoints("d0","zero")
        self.assertEqual(len(direct),840)
        self.assertEqual(endpoints("svf","zero"),direct)
        self.assertEqual(endpoints("svf","global_affine_ls"),direct)
        self.assertEqual({j["initialization"] for j in jobs if j["mode"]=="svf"},{"zero","global_affine_ls"})
        self.assertEqual({j["initialization"] for j in jobs if j["mode"]=="d1"},{"ls"})
        self.assertEqual(sum(j["kind"]=="anchor" for j in jobs),40)
        self.assertEqual(sum(j["kind"]=="bounded_followup" for j in jobs),24)
        self.assertEqual(sum(j["kind"]=="solver_diagnostic" for j in jobs),24)
        self.assertEqual(len(jobs),1680)
        ids={j["id"] for j in jobs}
        self.assertEqual(len(ids),len(jobs))
        self.assertTrue(all(d["job_id"] in ids for j in jobs for d in j["dependencies"]))

    def test_launch_binds_input_metric_adapter_profile_and_worker_receipts(self):
        for corruption in ("input","metric","adapter","profile_input","profile_source","worker_source"):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory(prefix="p81_launch_gate_") as temp:
                root=Path(temp)
                (root/"preflight").mkdir()
                (root/"input_manifest.json").write_text(json.dumps(dict(targets=[])))
                (root/"rank000_metric_replay.json").write_text("{}")
                accepted=dict(passed=True,input_manifest_sha256=controller.sha(root/"input_manifest.json"),
                              rank000_metric_replay_sha256=controller.sha(root/"rank000_metric_replay.json"),
                              data_source_sha256=controller.sha(HERE/"data.py"))
                profile=dict(passed=True,input_manifest_sha256=accepted["input_manifest_sha256"],
                             code_hashes={"data.py":controller.sha(HERE/"data.py")})
                smoke=dict(passed=True,source_hashes={str((HERE/name).relative_to(HERE.parents[2])):controller.sha(HERE/name)
                          for name in ("worker.py","metrics.py","diagnostics.py","basis.py","numerics.py")})
                if corruption=="input":(root/"input_manifest.json").write_text('{"targets":[],"changed":true}')
                if corruption=="metric":(root/"rank000_metric_replay.json").write_text('{"changed":true}')
                if corruption=="adapter":accepted["data_source_sha256"]="wrong"
                if corruption=="profile_input":profile["input_manifest_sha256"]="wrong"
                if corruption=="profile_source":profile["code_hashes"]["data.py"]="wrong"
                if corruption=="worker_source":smoke["source_hashes"][str((HERE/"worker.py").relative_to(HERE.parents[2]))]="wrong"
                (root/"INPUTS_ACCEPTED.json").write_text(json.dumps(accepted))
                (root/"preflight"/"summary.json").write_text(json.dumps(profile))
                (root/"WORKER_SMOKE.json").write_text(json.dumps(smoke))
                with mock.patch.object(sys,"argv",["launch.py","--root",str(root)]),mock.patch.object(launch,"gpu_snapshot") as telemetry:
                    with self.assertRaises(RuntimeError):launch.main()
                telemetry.assert_not_called()
                self.assertFalse((root/"launch_manifest.json").exists())

    def test_worker_callback_accepts_optimizer_phase_without_duplicate_keyword(self):
        worker_spec=importlib.util.spec_from_file_location("polyaffine_worker",HERE/"worker.py")
        worker=importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        instance=worker.Worker.__new__(worker.Worker)
        instance.progress=mock.Mock()
        instance.callback(dict(phase="adam",step=25,endpoint_mse_mm2=.1))
        instance.progress.assert_called_once()
        self.assertEqual(instance.progress.call_args.kwargs["step"],25)
        self.assertIn("phase",instance.progress.call_args.kwargs)

    def test_nonlinear_guard_records_runaway_or_nonfinite_row_before_failure(self):
        from worker import nonlinear_progress_guard
        for value in (201.,float("inf"),float("-inf"),float("nan")):
            with self.subTest(value=value):
                recorded=[];guard=nonlinear_progress_guard(recorded.append,{})
                guard(dict(step=0,loss_mm2=2.))
                row=dict(step=25,loss_mm2=value)
                with self.assertRaisesRegex(FloatingPointError,"solver_diverged"):guard(row)
                self.assertEqual(len(recorded),2)
                self.assertIs(recorded[-1],row)

    def test_nonlinear_guard_separate_baselines_and_restart_allow_normal_progress(self):
        from worker import nonlinear_progress_guard
        recorded=[];guard=nonlinear_progress_guard(recorded.append,{})
        # Loss limit is 100, validation limit is 2,000, from their own initial values.
        guard(dict(step=0,loss_mm2=1.,validation_mm2=20.))
        guard(dict(step=25,loss_mm2=50.,validation_mm2=1000.))
        guard(dict(step=50,loss_mm2=.5,validation_mm2=10.))
        with self.assertRaisesRegex(FloatingPointError,"loss_mm2"):guard(dict(step=75,loss_mm2=101.))
        # A new optimization level/restart receives new baselines.
        restarted=nonlinear_progress_guard(recorded.append,{})
        restarted(dict(step=0,loss_mm2=20.))
        restarted(dict(step=25,loss_mm2=101.))
        for value in (10.,1.,0.):restarted(dict(loss_mm2=value))
        self.assertEqual(len(recorded),9)

    def test_nonlinear_guard_ceiling_applies_at_initial_check_and_config_is_honored(self):
        from worker import nonlinear_progress_guard
        recorded=[];guard=nonlinear_progress_guard(recorded.append,{})
        with self.assertRaisesRegex(FloatingPointError,"limit=1e\\+06"):
            guard(dict(step=0,validation_mm2=1000001.))
        self.assertEqual(recorded[0]["step"],0)
        custom=nonlinear_progress_guard(recorded.append,dict(divergence_floor_mm2=5.,divergence_factor=2.,divergence_ceiling_mm2=100.))
        custom(dict(loss_mm2=2.))
        custom(dict(loss_mm2=5.))
        with self.assertRaisesRegex(FloatingPointError,"limit=5"):custom(dict(loss_mm2=6.))

    def test_launch_requires_stability_receipt_bound_to_inputs_config_and_sources(self):
        corruptions=(None,"passed","input","config","numerics.py","basis.py","data.py","worker.py","runtime.py","initialization.py","stability_pilot.py","stability_review.py")
        for corruption in corruptions:
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory(prefix="p81_stability_gate_") as temp:
                root=Path(temp)/"run";root.mkdir()
                repo=Path(temp)/"source";source=repo/"implementation";source.mkdir(parents=True)
                names=("worker.py","runtime.py","metrics.py","diagnostics.py","basis.py","numerics.py","data.py","initialization.py","stability_pilot.py","stability_review.py")
                for name in names:(source/name).write_text("# fixture "+name)
                hashes={str((source/name).relative_to(repo)):controller.sha(source/name) for name in names}
                config=repo/"protocol.json";config.write_text('{"version":"test"}')
                (root/"input_manifest.json").write_text('{"targets":[]}')
                (root/"rank000_metric_replay.json").write_text("{}")
                accepted=dict(passed=True,input_manifest_sha256=controller.sha(root/"input_manifest.json"),
                              rank000_metric_replay_sha256=controller.sha(root/"rank000_metric_replay.json"),
                              data_source_sha256=controller.sha(source/"data.py"))
                (root/"INPUTS_ACCEPTED.json").write_text(json.dumps(accepted))
                (root/"preflight").mkdir()
                profile=dict(passed=True,input_manifest_sha256=accepted["input_manifest_sha256"],
                             code_hashes={"data.py":controller.sha(source/"data.py")},
                             memory_profiles_mib={"d0__global_affine":100})
                (root/"preflight"/"summary.json").write_text(json.dumps(profile))
                (root/"WORKER_SMOKE.json").write_text(json.dumps(dict(passed=True,source_hashes=hashes)))
                stability=dict(passed=True,input_manifest_sha256=accepted["input_manifest_sha256"],
                               config_sha256=controller.sha(config),source_hashes=dict(hashes))
                if corruption=="passed":stability["passed"]=False
                if corruption=="input":stability["input_manifest_sha256"]="changed"
                if corruption=="config":stability["config_sha256"]="changed"
                if corruption in names:stability["source_hashes"]["implementation/"+corruption]="changed"
                (root/"STABILITY_PASSED.json").write_text(json.dumps(stability))
                (root/"validation").mkdir()
                (root/"validation"/"cpu_tests.json").write_text(json.dumps(dict(passed=True,exit_code=0,source_hashes=hashes)))
                with mock.patch.object(launch,"HERE",source),mock.patch.object(launch,"REPO",repo),mock.patch.object(launch,"code_hashes",return_value=hashes),mock.patch.object(sys,"argv",["launch.py","--root",str(root),"--config",str(config)]),mock.patch.object(launch,"gpu_snapshot",side_effect=RuntimeError("gpu sentinel")) as telemetry:
                    with self.assertRaises(RuntimeError) as raised:launch.main()
                if corruption is None:
                    self.assertEqual(str(raised.exception),"gpu sentinel")
                    telemetry.assert_called_once()
                else:
                    self.assertNotEqual(str(raised.exception),"gpu sentinel")
                    telemetry.assert_not_called()
                self.assertFalse((root/"launch_manifest.json").exists())

    def test_reserve_unobserved_peak_without_double_counting_current_memory(self):
        active=[dict(required_mib=8000,current_reserved_mib=4000)]*2
        self.assertTrue(controller.can_admit(dict(free_mib=18000),active,8000,headroom=2000))
        self.assertFalse(controller.can_admit(dict(free_mib=17999),active,8000,headroom=2000))
        # Missing progress conservatively reserves the entire peak.
        unknown=[dict(required_mib=8000)]
        self.assertFalse(controller.can_admit(dict(free_mib=17000),unknown,8000,headroom=2000))

    def test_gpu_snapshot_selects_physical_zero_only(self):
        output="1, GPU-one, 49140, 49000\n0, GPU-zero, 49140, 12345\n"
        with mock.patch.object(controller.subprocess,"run",return_value=mock.Mock(returncode=0,stdout=output)):
            got=controller.gpu_snapshot()
        self.assertEqual((got["gpu"],got["uuid"],got["free_mib"]),(0,"GPU-zero",12345))
        with mock.patch.object(controller.subprocess,"run",return_value=mock.Mock(returncode=0,stdout="1, GPU-one, 49140, 49000\n")):
            with self.assertRaises(RuntimeError):controller.gpu_snapshot()

    def test_malformed_progress_is_diagnostic_not_global_queue_failure(self):
        q=self.make_queue([dict(id="one")])
        path=q.root/"progress.json"
        for text in ("null","[]","42","broken json"):
            path.write_text(text)
            self.assertEqual(controller.safe_progress(path),{})
        path.write_text('{"phase":"running","current_reserved_mib":100}')
        self.assertEqual(controller.safe_progress(path)["current_reserved_mib"],100)

    def test_main_rejects_other_gpu_or_more_than_eight_jobs(self):
        for gpu,max_jobs in ((1,8),(0,9)):
            q=self.make_queue([dict(id="one")])
            m=dict(q.manifest,gpu=gpu)
            m["queue_policy"]=dict(m["queue_policy"],max_jobs=max_jobs)
            (q.root/"launch_manifest.json").write_text(json.dumps(m))
            with mock.patch.object(sys,"argv",["controller.py","--root",str(q.root)]):
                with self.assertRaises(ValueError):controller.main()

    def test_no_memory_admission_has_finite_deadline(self):
        q=self.make_queue([dict(id="waiting",profile_key="tiny")])
        q.last_guard=time.time()
        q.last_progress=time.time()-1
        q.policy["admission_timeout_sec"]=.01
        with mock.patch.object(controller,"gpu_snapshot",return_value=dict(gpu=0,uuid="GPU-test-zero",free_mib=0,total_mib=49140)),mock.patch.object(controller.subprocess,"run",return_value=mock.Mock(returncode=0)),mock.patch.object(q,"launch") as launch_job:
            q.run()
        launch_job.assert_not_called()
        self.assertEqual(q.state["status"],"finished")
        self.assertEqual(q.state["jobs"][0]["status"],"resource_deferred")
        self.assertEqual(q.state["jobs"][0]["failure"]["reason"],"admission_deadline_exceeded")

    def test_process_identity_detects_pid_reuse(self):
        identity=dict(pid=1234,start_ticks="old",uid=os.getuid())
        with mock.patch.object(controller,"process_identity",return_value={**identity,"start_ticks":"new"}):
            self.assertFalse(controller.same_process(identity))

    def test_worker_crash_records_failure_and_other_completion_survives(self):
        q=self.make_queue([dict(id="bad"),dict(id="good")])
        bad,good=q.state["jobs"]
        f_bad=self.attempt(q,bad);f_good=self.attempt(q,good)
        (f_bad/"failure.json").write_text(json.dumps(dict(error="synthetic solver failure")))
        self.completion(q,good,f_good)
        with mock.patch.object(controller,"same_process",return_value=False),mock.patch.object(controller,"recover_identity",return_value=None):
            q.reconcile()
        self.assertEqual(bad["status"],"failed")
        self.assertEqual(good["status"],"complete")
        self.assertEqual(len(list((q.root/"failures").glob("*.json"))),1)
        self.assertEqual(bad["failure"]["reason"],"synthetic solver failure")

    def test_completion_rejects_corrupted_field_despite_success_marker(self):
        q=self.make_queue([dict(id="bad")]);job=q.state["jobs"][0]
        folder=self.attempt(q,job)
        self.completion(q,job,folder)
        (folder/"field_768.fixture").write_bytes(b"corrupted after completion")
        with mock.patch.object(controller,"same_process",return_value=False),mock.patch.object(controller,"recover_identity",return_value=None):
            q.reconcile()
        self.assertEqual(job["status"],"failed")
        self.assertIn("field checksum mismatch",job["failure"]["reason"])

    def test_completion_rejects_missing_budget(self):
        q=self.make_queue([dict(id="bad",budgets=[768,1536])]);job=q.state["jobs"][0]
        folder=self.attempt(q,job)
        result=self.completion(q,job,folder)
        result["endpoints"].pop()
        (folder/"result.json").write_text(json.dumps(result))
        with mock.patch.object(controller,"same_process",return_value=False),mock.patch.object(controller,"recover_identity",return_value=None):
            q.reconcile()
        self.assertEqual(job["status"],"failed")
        self.assertIn("expected budgets",job["failure"]["reason"])

    def test_failed_dependency_only_blocks_dependent_job(self):
        jobs=[dict(id="source"),dict(id="dependent",dependencies=[dict(job_id="source",path="endpoints/source/cap1.json")]),dict(id="independent")]
        q=self.make_queue(jobs)
        source,dependent,independent=q.state["jobs"]
        source["status"]="failed"
        self.assertFalse(q.check_dependencies(dependent))
        self.assertEqual(dependent["status"],"dependency_failed")
        self.assertTrue(q.check_dependencies(independent))
        self.assertEqual(independent["status"],"pending")

    def test_restart_refuses_modified_manifest(self):
        q=self.make_queue([dict(id="one")]);q.save()
        (q.root/"launch_manifest.json").write_text("{}")
        with self.assertRaises(RuntimeError):controller.Queue(q.root,q.manifest)

    def test_superseded_controller_refuses_without_mutating_prior_receipts(self):
        q=self.make_queue([dict(id="old")]);q.save()
        (q.root/"SUPERSEDED.json").write_text('{"successor":"execution_v3"}')
        (q.root/"controller_identity.json").write_text('{"identity":"old controller"}')
        (q.root/"PREFLIGHT_PASSED.json").write_text('{"passed":true}')
        before={p.name:p.read_bytes() for p in q.root.iterdir() if p.is_file()}
        with mock.patch.object(sys,"argv",["controller.py","--root",str(q.root)]),mock.patch.object(controller.Queue,"run") as run:
            with self.assertRaisesRegex(RuntimeError,"superseded"):controller.main()
        run.assert_not_called()
        after={p.name:p.read_bytes() for p in q.root.iterdir() if p.is_file()}
        self.assertEqual(after,before)

    def test_preexisting_attempt_directory_is_preserved_and_isolated(self):
        q=self.make_queue([dict(id="orphan"),dict(id="independent")])
        orphan,independent=q.state["jobs"]
        folder=q.root/"jobs"/"orphan"/"attempt0"
        folder.mkdir(parents=True)
        (folder/"evidence.txt").write_text("retained")
        with mock.patch.object(controller,"recover_identity",return_value=None),mock.patch.object(controller.subprocess,"Popen") as popen:
            q.launch(orphan,1000)
        self.assertEqual(orphan["status"],"failed")
        self.assertEqual((folder/"evidence.txt").read_text(),"retained")
        self.assertEqual(independent["status"],"pending")
        popen.assert_not_called()

    def test_exit_race_during_watchdog_signal_does_not_crash_controller(self):
        q=self.make_queue([dict(id="one")]);job=q.state["jobs"][0]
        identity=dict(pid=1234,start_ticks="123",uid=os.getuid())
        self.attempt(q,job,identity)
        job["active_attempt"]["started_epoch"]=0
        with mock.patch.object(controller,"same_process",return_value=True),mock.patch.object(controller,"activity",return_value=["alive"]),mock.patch.object(controller.os,"killpg",side_effect=ProcessLookupError):
            q.reconcile()
            self.assertEqual(job["status"],"terminating")
            job["active_attempt"]["term_epoch"]=0
            q.reconcile()
        with mock.patch.object(controller,"same_process",return_value=False):q.reconcile()
        self.assertEqual(job["status"],"failed")
        self.assertEqual(job["failure"]["reason"],"wall_time_limit")

    def test_malformed_dependency_pointer_is_failed_instead_of_waited_forever(self):
        q=self.make_queue([dict(id="source"),dict(id="dependent",dependencies=[dict(job_id="source",path="endpoint.json")])])
        (q.root/"endpoint.json").write_text("{broken json")
        child=q.state["jobs"][1]
        self.assertFalse(q.check_dependencies(child))
        self.assertEqual(child["status"],"dependency_failed")

    def test_dependency_checkpoint_hash_corruption_is_detected(self):
        q=self.make_queue([dict(id="source"),dict(id="dependent",dependencies=[dict(job_id="source",path="endpoint.json")])])
        basis=q.root/"basis.json";coeff=q.root/"coeff.npy"
        basis.write_text("{}");coeff.write_bytes(b"coefficient fixture")
        pointer=dict(status="complete",job_id="source",basis_path=str(basis),basis_sha256=controller.sha(basis),
                     coefficients_path=str(coeff),coefficients_sha256=controller.sha(coeff))
        (q.root/"endpoint.json").write_text(json.dumps(pointer))
        child=q.state["jobs"][1]
        self.assertTrue(q.check_dependencies(child))
        coeff.write_bytes(b"changed coefficients")
        self.assertFalse(q.check_dependencies(child))
        self.assertEqual(child["status"],"dependency_failed")

    def test_durable_launch_identity_recovered_without_new_spawn(self):
        q=self.make_queue([dict(id="one")]);job=q.state["jobs"][0]
        self.attempt(q,job);job["status"]="launching"
        identity=dict(pid=1234,start_ticks="123",uid=os.getuid())
        with mock.patch.object(controller,"recover_identity",return_value=identity),mock.patch.object(controller,"same_process",return_value=True),mock.patch.object(controller,"activity",return_value=["alive"]),mock.patch.object(controller.subprocess,"Popen") as popen:
            q.reconcile()
        self.assertEqual(job["active_attempt"]["identity"],identity)
        self.assertEqual(job["status"],"running")
        popen.assert_not_called()

    def test_launch_masks_gpu_zero_and_persists_intent(self):
        q=self.make_queue([dict(id="one")]);job=q.state["jobs"][0]
        seen={}
        fake=mock.Mock(pid=1234)
        def spawn(argv,**kwargs):
            saved=json.loads(q.path.read_text())
            self.assertEqual(saved["jobs"][0]["status"],"launching")
            self.assertIsNone(saved["jobs"][0]["active_attempt"]["identity"])
            seen.update(kwargs)
            return fake
        identity=dict(pid=1234,start_ticks="123",uid=os.getuid())
        with mock.patch.object(controller.subprocess,"Popen",side_effect=spawn),mock.patch.object(controller,"process_identity",return_value=identity):
            q.launch(job,4000)
        self.assertEqual(seen["env"]["CUDA_VISIBLE_DEVICES"],"0")
        self.assertTrue(seen["start_new_session"])
        self.assertEqual(job["status"],"running")

    def test_real_cpu_failure_does_not_prevent_queue_finishing_other_job(self):
        q=self.make_queue([dict(id="bad",profile_key="tiny"),dict(id="good",profile_key="tiny")])
        q.last_guard=time.time()
        spawned=[]
        def launch(job,required):
            folder=self.attempt(q,job)
            if job["id"]=="bad":
                script="import sys;sys.exit(7)"
            else:
                result=self.completion(q,job,folder,write_result=False)
                script="import pathlib;pathlib.Path('result.json').write_text("+repr(json.dumps(result))+")"
            p=subprocess.Popen([sys.executable,"-c",script],cwd=folder,start_new_session=True)
            spawned.append(p)
            job["active_attempt"]["identity"]=controller.process_identity(p.pid)
            q.children[job["id"]]=p
        try:
            with mock.patch.object(q,"launch",side_effect=launch),mock.patch.object(controller,"gpu_snapshot",return_value=dict(gpu=0,uuid="GPU-test-zero",free_mib=48000,total_mib=49140)),mock.patch.object(controller.subprocess,"run",return_value=mock.Mock(returncode=0)):
                q.run()
        finally:
            for p in spawned:
                if p.poll() is None:p.kill()
                p.wait()
        self.assertEqual(q.state["status"],"finished")
        self.assertEqual([j["status"] for j in q.state["jobs"]],["failed","complete"])


if __name__=="__main__":unittest.main()
