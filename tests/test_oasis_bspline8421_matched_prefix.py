"""P80 contracts, real numerical parity, and case-local queue fault injection."""
from dataclasses import asdict,replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
sys.path.insert(0,str(ROOT/"scripts/reports"))
import bspline8421_matched_prefix as e
import run_oasis_bspline8421_matched_prefix as w
import queue_oasis_bspline8421_matched_prefix as q
import svf_transition_validation_v4 as v
import report_oasis_bspline8421_matched_prefix as report


class Contracts(unittest.TestCase):
    def test_matrix_names_and_no_prior_final_reuse(self):
        cells=e.matrix(); self.assertEqual(len(cells),2100)
        self.assertEqual(len({s.name() for s in cells}),2100)
        self.assertEqual(len(e.matrix(True)),9)
        self.assertFalse({s.name() for s in cells}&{s.name() for s in e.matrix(True)})
        self.assertEqual(sum(2 if s.family=="bounded2" else 1 for s in cells),2800)

    def test_invalid_and_cli(self):
        for rank,family,a in [(100,"bounded2",1),(True,"bounded2",1),(0,"dense",1),(0,"bounded2",.2),(0,"bounded2",float("nan"))]:
            with self.assertRaises(ValueError): e.Spec(rank,family,a).name()
        args=w.parse_args(["worker","--manifest","/tmp/m","--out-dir","/tmp/o","--gpu","0","--family","displacement","--alpha","1","--case-rank","99"])
        self.assertEqual(args.case_rank,99)

    def test_schedule_and_legacy_unchanged(self):
        for family in e.FAMILIES:
            s=w.stage_for(e.Spec(99,family,1))
            self.assertEqual((s.index,s.cps,s.factor,s.lr,s.max_epochs,s.min_epochs,s.patience,s.max_lr_reductions),(3,1,1,.001,2000,300,100,2))
            self.assertEqual(w.stage_for(e.Spec(0,family,1,True)).max_epochs,1)
        with self.assertRaises(ValueError): w.d.FitSpec(99,"pyramid",1,"8421").validate()
        w.SVFSpec(99,"8421","velocity",1).validate()

    def test_complete_groups_only_sample_sd(self):
        rows=[{"depth":"8421","family":"bounded2","alpha":1,"case_rank":i,**{k:float(i) for k in e.METRICS}} for i in range(100)]
        stats,_=report.grouped_summary(rows); self.assertEqual(stats[0]["dice_mean"],49.5)
        self.assertAlmostEqual(stats[0]["dice_sd"],torch.arange(100,dtype=torch.float64).std().item())
        self.assertEqual(report.grouped_summary(rows[:-1])[0],[])
        with self.assertRaises(RuntimeError): report.grouped_summary(rows+rows[:1])

    def test_folding_data_and_pair_signs(self):
        rows=[{"family":"svf_velocity","alpha":1,"case_rank":i,"dice":.001,"hd95":-.1,"asd":-.02,
               "log_jdet_std_fg":0.,"jdet_le0_pct_fg":-.001} for i in range(100)]
        r=report.paired_summary(rows)[0]
        self.assertEqual(r["dice_improved"],100); self.assertEqual(r["hd95_improved"],100)
        self.assertEqual(r["log_jdet_std_fg_tied"],100)


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_real_knot_preservation_and_velocity_integration(self):
        shape=(16,24,16); shapes=w.c.build_nested_cp_shapes(shape,8,(8,4,2,1))
        torch.manual_seed(1); cp=torch.randn(*shapes[2],3)*.005
        fine=w.c.cubic_bspline_dyadic_knot_insert_3d(cp,shapes[1])
        a=w.d.AlignedBSpline(shapes[2],shape,shape,2,"cpu")(cp)
        z=w.d.AlignedBSpline(shapes[1],shape,shape,1,"cpu")(fine)
        scale=w.d.voxel_scale(shape,device="cpu")
        w.legacy.verify_init(a,z)
        proof=v.verify_transition(cp,fine,a,z,w.s.exp_velocity(a),w.s.exp_velocity(z),1,scale)
        self.assertTrue(proof["passed"]); self.assertEqual(proof["policy"]["version"],4)
        with self.assertRaises(ValueError):v.verify_transition(cp,fine,a,z,a,z,2,scale)
        with self.assertRaises(RuntimeError):v.verify_transition(cp,fine+.001,a,z,a,z,1,scale)

    def test_broader_proof_gate_is_not_unconditional(self):
        v.v3.check_physical({"max_abs_voxel":.002,"rms_voxel":1e-5})
        for stats in ({"max_abs_voxel":.0021,"rms_voxel":0.},{"max_abs_voxel":0.,"rms_voxel":1.1e-5},
                      {"max_abs_voxel":float("nan"),"rms_voxel":0.}):
            with self.assertRaises(RuntimeError):v.v3.check_physical(stats)
        tiny={"max_abs_voxel":1e-12,"rms_voxel":1e-14}
        proof={"velocity":tiny,"integrated_field":tiny,"rounded_controls_integrated_field":{"max_abs_voxel":.000123,"rms_voxel":1e-7}}
        v.v3.check_reference(proof)
        proof["rounded_controls_integrated_field"]["max_abs_voxel"]=.000201
        with self.assertRaises(RuntimeError):v.v3.check_reference(proof)

    def test_failed_original_gate_mandates_actual_control_fp64_proof(self):
        cp=torch.zeros(6,6,6,3); fine=w.c.cubic_bspline_dyadic_knot_insert_3d(cp,(8,8,8))
        zero=torch.zeros(40,40,40,3); near=zero.clone();near[0,0,0,0]=1.1e-5
        args=(cp,fine,zero,zero,zero,near,1,torch.ones(3)*100)
        tiny={"max_abs_voxel":1e-12,"rms_voxel":1e-14}
        proof={"velocity":tiny,"integrated_field":tiny,"rounded_controls_integrated_field":{"max_abs_voxel":.000123,"rms_voxel":1e-7}}
        with patch.object(v,"fp64_reference",return_value=proof) as ref:
            self.assertTrue(v.verify_transition(*args)["passed"])
            ref.assert_called_once()
        proof["rounded_controls_integrated_field"]["max_abs_voxel"]=.00021
        with patch.object(v,"fp64_reference",return_value=proof):
            with self.assertRaises(RuntimeError):v.verify_transition(*args)
        with patch.object(v,"fp64_reference",side_effect=RuntimeError("failed proof")):
            with self.assertRaises(RuntimeError):v.verify_transition(*args)

    def test_bound_zero_unit_gradient_and_frozen_sampling_gradients(self):
        shape=(16,16,16); stage=w.stage_for(e.Spec(0,"bounded2",1,True))
        fixed=torch.rand(shape); moving=torch.rand(shape)
        frozen=torch.randn(*shape,3)*.001
        obj=w.b.Objective(fixed,moving,stage,1,[frozen],w.b.compute_bound(1),True)
        p=torch.zeros(*obj.control_shape,3,requires_grad=True)
        cp=obj.controls(p); grad,=torch.autograd.grad(cp.sum(),p)
        self.assertTrue(torch.equal(grad,torch.ones_like(grad)))
        field=obj.field(p)
        self.assertTrue(torch.isfinite(field).all())
        loss=obj(p)[0]; loss.backward()
        self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all()); self.assertIsNone(frozen.grad)
        direct=w.d.StageObjective(fixed,moving,1,replace(stage,cps=None))
        x=obj(p); y=direct(obj.field(p))
        self.assertTrue(all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(x,y)))

    def test_all_three_fresh_adam_paths_one_update(self):
        shape=(16,16,16); batch={"fixed":torch.rand(shape),"moving":torch.rand(shape)}
        for family in e.FAMILIES:
            spec=e.Spec(0,family,1,True); stage=w.stage_for(spec)
            if family=="bounded2": obj=w.b.Objective(**batch,stage=stage,alpha=1,frozen_fields=[],bound=w.b.compute_bound(1),bounded=True)
            elif family=="svf_velocity": obj=w.s.Objective(batch["fixed"],batch["moving"],stage,w.SVFSpec(0,"8421","velocity",1,True))
            else:
                shape_cp=w.c.build_nested_cp_shapes(shape,8,(8,4,2,1))[1]
                obj=w.d.StageObjective(batch["fixed"],batch["moving"],1,stage,shape_cp)
                obj.control_shape=shape_cp
            initial=torch.zeros(*obj.control_shape,3)
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp)/"fit"
                with patch.object(torch.optim,"Adam",wraps=torch.optim.Adam) as adam:
                    if family=="bounded2": _,info=w.b.optimize(obj,out,initial)
                    elif family=="svf_velocity": _,info=w.s.optimize(initial,obj,out)
                    else: _,info=w.d.optimize_stage(initial,obj,stage,out)
                self.assertEqual(adam.call_count,1); self.assertEqual(info["actual_epochs"],1)
                e.old.check_history(out/"history.csv",info,1)


class Failures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.queue=q.Queue.__new__(q.Queue); self.queue.folder=self.root; self.queue.jobs=[]; self.queue.active={}
        self.queue.m={}; self.queue.save=Mock()
    def tearDown(self):self.tmp.cleanup()
    def job(self,status="pending",family="bounded2"):
        s=e.Spec(0,family,1)
        j={"name":s.name(),"spec":asdict(s),"status":status,"gpu":"0","attempt":0,"oom_retries":0,"out":str(self.root/s.name()),"log":str(self.root/"missing.log")}
        self.queue.jobs.append(j); return j
    def test_failure_does_not_block_later_jobs(self):
        bad=self.job("running"); good=self.job(family="displacement")
        self.queue.active[1]={"job":bad,"action":"worker","process":Mock(poll=Mock(return_value=1)),"record":{"pid":1}}
        self.queue.poll(); self.assertEqual(bad["status"],"failed_deferred")
        self.assertIs(q.choose_job(self.queue.jobs,"0",{}),good)
    def test_parent_audit_exception_is_local(self):
        bad=self.job("auditing"); good=self.job(family="displacement")
        self.queue.active[1]={"job":bad,"action":"audit","process":Mock(poll=Mock(return_value=0)),"record":{"pid":1}}
        with patch.object(q.e,"validate",side_effect=RuntimeError("invalid metric")),patch.object(q.base,"same_process",return_value=False): self.queue.poll()
        self.assertEqual(bad["status"],"failed_deferred"); self.assertIs(q.choose_job(self.queue.jobs,"0",{}),good)
    def test_single_oom_retry_tail_same_gpu(self):
        bad=self.job("running"); good=self.job(family="displacement")
        self.queue.fail(bad,"worker","oom","CUDA out of memory",True)
        self.assertEqual(bad["status"],"oom_retry"); self.assertIs(q.choose_job(self.queue.jobs,"0",{}),good)
        good["status"]="complete"
        self.assertIsNone(q.choose_job(self.queue.jobs,"1",{})); self.assertIs(q.choose_job(self.queue.jobs,"0",{}),bad)
        self.assertIsNone(q.choose_job(self.queue.jobs,"0",{1:{"job":{"gpu":"0"}}}))
        self.queue.fail(bad,"worker","oom","CUDA out of memory",True); self.assertEqual(bad["status"],"failed_deferred")
    def test_reservations_and_malformed_status(self):
        self.assertFalse(q.capacity("0",{1:{"job":{"gpu":"0"},"required_mib":8192}},15000,8192))
        self.assertTrue(q.capacity("0",{1:{"job":{"gpu":"0"},"required_mib":8192}},20000,8192))
        p=self.root/"bad.json";p.write_text("{")
        self.assertEqual(q.base.safe_progress(p)["phase"],"progress_unavailable")
        self.assertNotIn("0",q.base.parse_gpus("0, bad, 1, 2")["rows"])
    def test_exit_zero_without_done(self):
        bad=self.job("running")
        self.queue.active[1]={"job":bad,"action":"worker","process":Mock(poll=Mock(return_value=0)),"record":{"pid":1}}
        self.queue.poll();self.assertEqual(bad["status"],"failed_deferred")

    def test_restart_done_only_schedules_audit(self):
        spec=e.Spec(0,"bounded2",1)
        out=self.root/"jobs"/spec.name()/"attempt0";out.mkdir(parents=True);(out/"DONE").write_text("done")
        m={"prefixes":{spec.key():{"gpu":"0"}}}
        with patch.object(q.e,"matrix",return_value=[spec]),patch.object(q.base,"process_info",return_value=None):
            queue=q.Queue(self.root,m)
        self.assertEqual(queue.jobs[0]["status"],"audit_pending")
        self.assertFalse(queue.active)

    def test_report_failure_preserves_completed_runs(self):
        out=self.root/"accepted_run";out.mkdir();marker=out/"DONE";marker.write_text("original")
        with patch.object(q.subprocess,"run",side_effect=RuntimeError("report failed")):
            q.finish_report(self.root,self.root)
        self.assertEqual(marker.read_text(),"original")
        self.assertEqual(len(list(self.root.glob("REPORT_FAILED_*.json"))),1)


if __name__=="__main__":unittest.main()
