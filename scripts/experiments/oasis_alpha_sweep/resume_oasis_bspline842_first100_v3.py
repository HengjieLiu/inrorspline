#!/usr/bin/env python
"""P79: preserve2098 results, continue only rank29/44 SVF lambda.01 stages."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import oasis_bspline842_first100 as e
import run_oasis_bspline842_first100 as w
import svf_transition_validation_v3 as v3
from queue_oasis_svf_continue_on_error import gpu_snapshot
from queue_oasis_svf_eight_workers import process_info, same_process

r = w.svf
FOLDER = "transition_continuation_v3"
TARGETS = {29: {"resume_index": 1, "gpu": "3"}, 44: {"resume_index": 2, "gpu": "2"}}
COPY_NAMES = ("best.pt", "last.pt", "history.csv", "history.jsonl", "summary.json")
OPTIMIZE, STAGES = r.s.optimize, r.s.stages
HERE = Path(__file__).resolve()


def folder_for(artifact):
    return e.run_root(artifact) / FOLDER


def spec_for(rank):
    if type(rank) is not int or rank not in TARGETS:
        raise ValueError("Only ranks29/44 refined842 velocity lambda.01 are approved")
    return e.Spec(rank, "svf_velocity", .01)


def sources():
    return {str(p.relative_to(e.c.REPO_ROOT)): e.sha(p) for p in
            (HERE, Path(v3.__file__), e.c.REPO_ROOT/"tests/test_oasis_bspline842_continuation_v3.py")}


def assert_manifest(m):
    e.assert_manifest(m)
    info = m["continuation_v3"]
    if info["policy"] != v3.POLICY or info["source_hashes"] != sources():
        raise RuntimeError("Continuation source/policy drift")
    if set(info["targets"]) != {str(k) for k in TARGETS}:
        raise RuntimeError("Wrong continuation targets")
    for rank, expected in TARGETS.items():
        target = info["targets"][str(rank)]
        if target["spec"] != asdict(spec_for(rank)) or any(target[k] != expected[k] for k in expected):
            raise RuntimeError("Changed approved continuation scope")
    for path, digest in info["preserved_metadata_sha256"].items():
        if e.sha(path) != digest:
            raise RuntimeError(f"Preserved metadata changed: {path}")


def verify_preserved(m):
    assert_manifest(m)
    info = m["continuation_v3"]
    inventory = e.c.read_json(Path(info["inventory_path"]))
    if e.sha(info["inventory_path"]) != info["inventory_sha256"]:
        raise RuntimeError("Preservation inventory changed")
    for path, digest in inventory["metadata_sha256"].items():
        if e.sha(path) != digest:
            raise RuntimeError(f"Preserved record changed: {path}")
    for receipt_path in inventory["receipt_paths"]:
        e.verify_receipt(e.c.read_json(Path(receipt_path)))
    for path, digest in inventory["failed_attempt_sha256"].items():
        if e.sha(path) != digest:
            raise RuntimeError(f"Failed-attempt file changed: {path}")
    return {"passed": True, "accepted_results": 2098, "receipt_count": len(inventory["receipt_paths"]),
            "failed_attempt_files": len(inventory["failed_attempt_sha256"]), "at": e.c.now_text()}


def prepare(artifact):
    folder = folder_for(artifact); folder.mkdir(parents=True, exist_ok=True)
    mp = folder/"launch_manifest.json"
    if mp.exists():
        m = e.c.read_json(mp); verify_preserved(m); return m
    parent = e.run_root(artifact, "metadata_recovery_v2")
    m = e.c.read_json(parent/"launch_manifest.json"); e.assert_manifest(m)
    state = e.c.read_json(parent/"controller/state.json")
    if Counter(j["status"] for j in state["jobs"]) != {"complete": 1528, "failed_deferred": 2}:
        raise RuntimeError("Require drained1528+2 queue")
    targets, receipts, failed_files, metadata = {}, [], {}, {}
    for name in m["reused"]:
        path = parent/"reuse_receipts"/f"{name}.json"
        e.verify_receipt(e.c.read_json(path)); receipts.append(str(path)); metadata[str(path)] = e.sha(path)
    print(json.dumps({"preservation_reused_checked":len(receipts)}),flush=True)
    for job in state["jobs"]:
        spec = e.Spec(**job["spec"]); out = Path(job["out"])
        if job["status"] == "complete":
            e.validate(out, m, spec)
            path = out/"extension_receipt.json"; receipts.append(str(path)); metadata[str(path)] = e.sha(path)
            if len(receipts)%200==0:print(json.dumps({"preservation_results_checked":len(receipts)}),flush=True)
            continue
        if spec != spec_for(spec.case_rank):
            raise RuntimeError("Unexpected failed cell")
        old = e.c.read_json(out/"manifest.json"); config = TARGETS[spec.case_rank]
        folders = sorted(out.glob("stage*_cps*"))
        if len(folders) != config["resume_index"] or old["gpu"] != config["gpu"]:
            raise RuntimeError("Unexpected saved stages or physical GPU")
        if (old["spec"]["schedule"], old["spec"]["regularize"], old["spec"]["alpha"]) != ("842", "velocity", .01):
            raise RuntimeError("Wrong original SVF variant")
        for key in ("case_rank", "fixed_id", "moving_id", "input_hashes"):
            if old[key] != m["cases"][str(spec.case_rank)][key]:
                raise RuntimeError("Prefix case identity differs")
        failure_paths = list(out.glob("FAILED_worker_*.json"))
        if len(failure_paths) != 1 or "Material integrated-field change" not in e.c.read_json(failure_paths[0])["traceback"]:
            raise RuntimeError("Unexpected failure condition")
        for path in out.rglob("*"):
            if path.is_file(): failed_files[str(path)] = e.sha(path)
        failed_files[job['log_path']] = e.sha(job['log_path'])
        targets[str(spec.case_rank)] = {**config, "spec": asdict(spec), "prefix_dir": str(out),
            "manifest_sha256": e.sha(out/"manifest.json"), "failure_path": str(failure_paths[0]),
            "failure_sha256": e.sha(failure_paths[0]),
            "prefix_stage_hashes": {str(i): e.sha(p/"validated.json") for i,p in enumerate(folders)}}
    if len(receipts) != 2098:
        raise RuntimeError("Incomplete preservation receipts")
    preserve = [parent/"launch_manifest.json", parent/"controller/state.json", parent/"progress.json",
                parent/"controller/QUEUE_FINISHED.json"]
    report_root = artifact/"repo_experiments"/e.VARIANT
    preserve += [p for p in report_root.rglob("*") if p.is_file() and not p.is_relative_to(report_root/FOLDER)]
    preserved = {str(p): e.sha(p) for p in preserve}
    metadata.update(preserved)
    inventory = folder/"preserved_files.json"
    e.atomic_json(inventory, {"metadata_sha256": metadata, "receipt_paths": receipts,
                              "failed_attempt_sha256": failed_files})
    m = {**m, "continuation_v3": {"policy": v3.POLICY, "source_hashes": sources(), "targets": targets,
         "parent_root": str(parent), "preserved_metadata_sha256": preserved,
         "inventory_path": str(inventory), "inventory_sha256": e.sha(inventory),
         "source_revision": subprocess.check_output(["git","rev-parse","HEAD"],cwd=e.c.REPO_ROOT,text=True).strip(),
         "optimization_changed": False, "created_at": e.c.now_text(), "initial_gpu_state": gpu_snapshot()}}
    e.atomic_json(mp, m)
    e.atomic_json(folder/"PRESERVATION_VERIFIED.json", verify_preserved(m))
    return m


def check_prefix(target, stage, m):
    prefix = Path(target["prefix_dir"])
    if (e.sha(prefix/"manifest.json") != target["manifest_sha256"] or
            e.sha(target["failure_path"]) != target["failure_sha256"]):
        raise RuntimeError("Original failure metadata changed")
    old = e.c.read_json(prefix/"manifest.json")
    if old["source_hashes"] != m["source_hashes"]:
        raise RuntimeError("Prefix numerical provenance mismatch")
    folder = prefix/f"stage{stage.index}_cps{stage.cps}"
    if e.sha(folder/"validated.json") != target["prefix_stage_hashes"][str(stage.index)]:
        raise RuntimeError("Prefix stage record changed")
    block = e.c.read_json(folder/"validated.json")
    for name,key in (("best.pt","model_sha256"),("last.pt","last_model_sha256"),("history.csv","history_sha256")):
        if e.sha(folder/name) != block[key]: raise RuntimeError("Prefix checkpoint/history changed")
    if e.sha(block["dvf_path"]) != block["dvf_sha256"] or block["optimizer"]["stage"] != asdict(stage) or block["alpha"] != .01:
        raise RuntimeError("Prefix field/schedule mismatch")
    e.check_history(folder/"history.csv", block["optimizer"], .01)
    return folder, block


def replay_prefix(initial, objective, out, target, m):
    folder, block = check_prefix(target, objective.stage, m)
    state = torch.load(folder/"best.pt", map_location=initial.device, weights_only=False)
    cp, info = state["velocity_controls_norm_xyz"], state["summary"]
    if info != block["optimizer"] or cp.shape != initial.shape:
        raise RuntimeError("Wrong selected prefix controls")
    with torch.no_grad():
        values = [float(x) for x in objective(cp)]
        init_values = [float(x) for x in objective(initial)]
        if any(not r.close(a,b) for a,b in zip(init_values,block["initial_loss_data_reg"])):
            raise RuntimeError("Prefix initial loss replay differs")
        if any(not r.close(v,info[k]) for v,k in zip(values,("best_loss","best_data_loss","best_reg_loss"))):
            raise RuntimeError("Prefix selected loss replay differs")
        if not np.array_equal(objective.field(cp,native=True).cpu().numpy(),np.load(block["dvf_path"])):
            raise RuntimeError("Prefix field replay is not exact")
    out.mkdir(parents=True,exist_ok=False)
    for name in COPY_NAMES:
        shutil.copy2(folder/name,out/name)
        if e.sha(folder/name) != e.sha(out/name): raise RuntimeError("Prefix copy changed")
    return cp.detach(), info


@contextmanager
def adapter(target, m, fixture=False):
    # Historical numerical functions, global provenance POLICY and source files stay intact.
    saved = r.transition, r.s.optimize, r.s.stages
    checks = []
    def stages(spec):
        result = STAGES(spec)
        return [replace(x,max_epochs=1,min_epochs=1) if fixture and x.index >= target["resume_index"] else x for x in result]
    def verify(*args, **kwargs):
        cps = args[6] if len(args)>6 else kwargs["cps"]
        index = {4:1,2:2}[cps]
        fn = v3.verify_transition if index >= target["resume_index"] else saved[0].verify_transition
        result = fn(*args,**kwargs); checks.append(result); return result
    def optimize(initial, objective, out):
        if objective.stage.index < target["resume_index"]:
            return replay_prefix(initial,objective,out,target,m)
        return OPTIMIZE(initial,objective,out)
    r.transition = SimpleNamespace(POLICY=saved[0].POLICY,verify_transition=verify)
    r.s.optimize, r.s.stages = optimize, stages
    try: yield checks
    finally: r.transition, r.s.optimize, r.s.stages = saved


def invoke(args):
    m = e.c.read_json(args.manifest); assert_manifest(m)
    target = m["continuation_v3"]["targets"][str(args.case_rank)]
    spec = spec_for(args.case_rank)
    if args.gpu != target["gpu"]: raise RuntimeError("Continuation/replay must use original physical GPU")
    fixture = args.fixture
    worker = SimpleNamespace(action=args.action,manifest=args.manifest,out_dir=args.out_dir,gpu=args.gpu,
                             case_rank=args.case_rank,family="svf_velocity",alpha=.01,smoke=False)
    if args.action == "audit":
        row = e.c.read_json(args.out_dir/"summary.json")
        done = e.c.read_json(args.out_dir/"CONTINUATION_DONE.json")
        if (done["summary_sha256"] != e.sha(args.out_dir/"summary.json") or
                row["continuation_v3"]["fixture"] != fixture or row["continuation_v3"]["prefix"] != target):
            raise RuntimeError("Incomplete or wrong continuation target")
    with adapter(target,m,fixture) as checks:
        w.main(worker)
    if args.action == "worker":
        row = e.c.read_json(args.out_dir/"summary.json")
        for block in row["blocks"][:target["resume_index"]]:
            old = e.c.read_json(Path(target["prefix_dir"])/f"stage{block['stage_index']}_cps{block['cps']}"/"validated.json")
            if block != {**old,"model_path":block["model_path"],"dvf_path":block["dvf_path"]}:
                raise RuntimeError("Retained prefix record differs beyond copied paths")
        prior = row["blocks"][:target["resume_index"]]; new = row["blocks"][target["resume_index"]:]
        if fixture and any(b["optimizer"]["actual_epochs"] != 1 for b in new):
            raise RuntimeError("Each new smoke stage must have exactly one update")
        row["continuation_v3"] = {"fixture": fixture,"policy":v3.POLICY,"source_hashes":sources(),"prefix":target,
            "new_optimizer_cps":[b["cps"] for b in new],
            "new_adam_updates":sum(b["optimizer"]["actual_epochs"] for b in new),
            "prefix_adam_updates_reused":sum(b["optimizer"]["actual_epochs"] for b in prior),
            "prefix_optimizer_sec_reused":sum(b["optimizer"]["training_elapsed_sec"] for b in prior),
            "new_optimizer_sec":sum(b["optimizer"]["training_elapsed_sec"] for b in new),
            "continuation_worker_sec":row["worker_sec"],"transition_checks":checks,
            "timing_note":"optimizer total includes retained prefix; worker time measures continuation only; no downtime"}
        e.atomic_json(args.out_dir/"summary.json",row)
        e.atomic_json(args.out_dir/"CONTINUATION_DONE.json",{"summary_sha256":e.sha(args.out_dir/"summary.json"),
                      "source_hashes":sources(),"at":e.c.now_text()})
    else:
        e.atomic_json(args.out_dir/"CONTINUATION_AUDIT.json",{"passed":True,"fixture":fixture,
            "policy":v3.POLICY,"source_hashes":sources(),"summary_sha256":e.sha(args.out_dir/"summary.json"),
            "audit_sha256":e.sha(args.out_dir/"independent_audit.json"),"at":e.c.now_text()})
        validate(args.out_dir,m,spec,fixture)
    assert_manifest(m)


def validate(out,m,spec,fixture=False):
    spec_for(spec.case_rank)
    if spec != spec_for(spec.case_rank): raise RuntimeError("Wrong continued configuration")
    row = e.validate(out,m,spec)
    check = e.c.read_json(out/"CONTINUATION_AUDIT.json")
    if (not check["passed"] or check["fixture"] != fixture or check["policy"] != v3.POLICY or
            check["source_hashes"] != sources() or check["summary_sha256"] != e.sha(out/"summary.json") or
            check["audit_sha256"] != e.sha(out/"independent_audit.json") or
            row["continuation_v3"]["source_hashes"] != sources() or row["continuation_v3"]["fixture"] != fixture):
        raise RuntimeError("Continuation validation/provenance mismatch")
    return row


def phase(folder,m,fixture):
    phase_name = "fixtures" if fixture else "production"
    root = folder/phase_name; root.mkdir(exist_ok=True)
    jobs = [{"rank":rank,"gpu":config["gpu"],"out":str(root/f"rank{rank:03d}"),"status":"pending"}
            for rank,config in TARGETS.items()]
    # A surviving child must not be duplicated after a controller interruption.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        rec = process_info(p.name)
        if rec and rec['state']!='Z' and str(HERE) in rec['argv'] and '--out-dir' in rec['argv']:
            out = Path(rec['argv'][rec['argv'].index('--out-dir')+1])
            if out.is_relative_to(root): raise RuntimeError(f'Existing continuation child needs reconciliation: {rec["pid"]}')
    for j in jobs:
        out = Path(j["out"])
        if (out/"CONTINUATION_AUDIT.json").exists():
            validate(out,m,spec_for(j["rank"]),fixture);j["status"]="complete"
        elif (out/"CONTINUATION_DONE.json").exists() and not list(out.glob('FAILED*')):
            j["status"]="audit_pending"
        elif out.exists(): raise RuntimeError(f"Preserve unfinished continuation: {out}")
    active = {}
    while any(j['status'] not in ('complete','failed') for j in jobs) or active:
        for pid,item in list(active.items()):
            code=item['process'].poll()
            if code is None: continue
            item['stream'].close();j=item['job'];out=Path(j['out'])
            try:
                if code: raise RuntimeError(f"{item['action']} exited{code}: {item['log']}")
                if item['action']=='worker':
                    if not (out/'CONTINUATION_DONE.json').exists(): raise RuntimeError('Missing continuation completion')
                    j['status']='audit_pending'
                else:
                    validate(out,m,spec_for(j['rank']),fixture);j['status']='complete'
            except Exception:
                j['status']='failed';j['failure']=traceback.format_exc()
                e.atomic_json(folder/f'failure_{time.time_ns()}.json',{'job':j,'log':str(item['log']),'at':e.c.now_text()})
            del active[pid]
        assert_manifest(m)
        snapshot=gpu_snapshot();required=8192
        for p in (folder/'fixtures').glob('rank*/extension_provenance.json'):
            required=max(required,math.ceil((e.c.read_json(p)['peak_reserved_gib']+2)*1024))
        for j in jobs:
            if j['status'] not in ('pending','audit_pending'): continue
            if snapshot['rows'].get(j['gpu'],{}).get('memory_free_mib',0)<required: continue
            action='audit' if j['status']=='audit_pending' else 'worker'
            log=folder/f'{phase_name}_{action}_rank{j["rank"]}_{time.time_ns()}.log';stream=log.open('x')
            cmd=[sys.executable,'-u','-B',str(HERE),action,'--manifest',str(folder/'launch_manifest.json'),
                 '--out-dir',j['out'],'--case-rank',str(j['rank']),'--gpu',j['gpu']]
            if fixture:cmd.append('--fixture')
            try:
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,
                    'CUDA_VISIBLE_DEVICES':j['gpu'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'4',
                    'MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'})
            except Exception:
                stream.close();raise
            j.update(status='running' if action=='worker' else 'auditing',pid=proc.pid)
            active[proc.pid]={'process':proc,'stream':stream,'job':j,'action':action,'log':log}
            event={'action':action,'fixture':fixture,'job':j.copy(),'command':cmd,'gpu_state':snapshot,
                   'required_free_mib':required,'at':e.c.now_text()}
            e.atomic_json(folder/f'launch_{time.time_ns()}.json',event);print(json.dumps(event),flush=True)
        e.atomic_json(folder/'progress.json',{'phase':phase_name,'jobs':jobs,'preserved':2098,
            'completed':sum(j['status']=='complete' for j in jobs),'failed':sum(j['status']=='failed' for j in jobs),
            'updated_at':e.c.now_text()})
        if active:time.sleep(5)
        elif any(j['status'] in ('pending','audit_pending') for j in jobs):time.sleep(10)
    if any(j['status']=='failed' for j in jobs):raise RuntimeError('Continuation failure preserved; no final acceptance')
    e.atomic_json(folder/f'{phase_name.upper()}_PASSED.json',{'passed':True,'count':2,'at':e.c.now_text()})


def report(artifact,m):
    # Reuse the frozen generator through a private read-only facade and a merged
    # state view. Original state, reports, acceptance markers and LATEST stay intact.
    sys.path.insert(0,str(e.c.REPO_ROOT/'scripts/reports'))
    import report_oasis_bspline842_first100 as base
    folder=folder_for(artifact);parent=Path(m['continuation_v3']['parent_root'])
    e.atomic_json(folder/'PRESERVATION_FINAL.json',verify_preserved(m))
    view=folder/'report_view';view.mkdir(exist_ok=False)
    e.atomic_json(view/'launch_manifest.json',m)
    shutil.copytree(parent/'reuse_receipts',view/'reuse_receipts')
    state=e.c.read_json(parent/'controller/state.json')
    for j in state['jobs']:
        if j['status']=='complete':continue
        spec=e.Spec(**j['spec']);out=folder/'production'/f'rank{spec.case_rank:03d}'
        validate(out,m,spec)
        j.update(status='complete',out=str(out),continuation_original_failure=j.pop('failure',None))
    e.atomic_json(view/'controller/state.json',state)
    proxy=SimpleNamespace(**vars(e));proxy.run_root=lambda *_:view
    proxy.VARIANT=e.VARIANT+'/'+FOLDER+'/building';proxy.assert_manifest=assert_manifest
    def validated(out,manifest,spec):
        if out.is_relative_to(folder/'production'):return validate(out,manifest,spec)
        return e.validate(out,manifest,spec)
    proxy.validate=validated
    old_e,old_flat=base.e,base.flat
    def flat(row,spec,kind):
        continued=row.get('continuation_v3')
        result,blocks=old_flat(row,spec,'checkpoint_continuation_v3' if continued else kind)
        if continued:
            for key in ('new_optimizer_sec','prefix_optimizer_sec_reused','new_adam_updates','prefix_adam_updates_reused','continuation_worker_sec'):
                result[key]=continued[key]
            result['worker_time_scope']='continuation only; original prefix worker total unavailable'
        return result,blocks
    base.e,base.flat=proxy,flat
    try:base.run(artifact)
    finally:base.e,base.flat=old_e,old_flat
    building=artifact/'repo_experiments'/e.VARIANT/FOLDER/'building'
    out=Path(e.c.read_json(building/'LATEST.json')['path'])
    accepted=e.c.read_json(out/'ACCEPTED.json')
    if accepted['candidate_count']!=2100 or not accepted['complete']:raise RuntimeError('Incomplete final report')
    path=out/'report.md'
    path.write_text(path.read_text()+
        '\n\n## Two scoped checkpoint continuations\n\n'
        '2098 accepted results are unchanged. Refined-SVF/velocity lambda0.01 ranks29/44 '
        'retained their own completed cps8 and cps8/4 checkpoints respectively, then optimized only '
        'the missing stages. FP32 optimization, seven-step integration and scientific settings are unchanged. '
        'Validation policyv3 permits maximum0.002/RMS1e-5 voxel only with exact prescribed controls and '
        'mandatory ideal/actual-rounded-control FP64 proof when the original field allclose fails. '
        'Ideal velocity/field limits remain1e-9/1e-8 voxel; rounded-control field limits are2e-4 maximum/1e-6 RMS. '
        'This is an explicit numerical-validation amendment, not a zero-folding guarantee.\n\n'
        'Resource rows tagged checkpoint_continuation_v3 are separate: optimizer totals include retained '
        'prefix work; worker time measures the continuation only. New/prefix optimizer seconds and Adam '
        'updates are explicit in per_case.csv. No prefix updates were rerun and idle downtime is excluded.\n')
    e.atomic_json(out/'continuation_provenance.json',m['continuation_v3'])
    accepted.update(continued_count=2,preserved_count=2098,validation_policy=v3.POLICY)
    accepted['output_sha256']['report.md']=e.sha(path)
    accepted['output_sha256']['continuation_provenance.json']=e.sha(out/'continuation_provenance.json')
    e.atomic_json(out/'ACCEPTED.json',accepted)
    for name,digest in accepted['output_sha256'].items():
        if e.sha(out/name)!=digest:raise RuntimeError('Final output hash mismatch')
    verify_preserved(m)
    final=building.parent/out.name;out.rename(final)
    e.atomic_json(final.parent/'LATEST.json',{'path':str(final),'complete':True,'at':e.c.now_text()})
    e.atomic_json(folder/'COMPLETE.json',{'passed':True,'count':2100,'preserved':2098,'continued':2,'report':str(final),'at':e.c.now_text()})
    e.atomic_json(folder/'progress.json',{'phase':'accepted','completed':2100,'total':2100,'remaining':0,'report':str(final),'updated_at':e.c.now_text()})


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('prepare','launch','report'):
        q=sub.add_parser(action);q.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    for action in ('worker','audit'):
        q=sub.add_parser(action);q.add_argument('--manifest',type=Path,required=True);q.add_argument('--out-dir',type=Path,required=True)
        q.add_argument('--case-rank',type=int,choices=TARGETS,required=True);q.add_argument('--gpu',choices=('2','3'),required=True)
        q.add_argument('--fixture',action='store_true')
    a=p.parse_args(argv);out=a.artifact_root if hasattr(a,'artifact_root') else a.out_dir
    if out.resolve().is_relative_to(e.c.REPO_ROOT):p.error('External artifacts only')
    return a


def main(args):
    if (os.getuid(),os.getgid())!=(1009,1020):raise RuntimeError('Require UID:GID1009:1020')
    if args.action in ('worker','audit'):invoke(args);return
    folder=folder_for(args.artifact_root);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=prepare(args.artifact_root)
        if args.action=='prepare':return
        if args.action=='launch':phase(folder,m,True);phase(folder,m,False)
        report(args.artifact_root,m)


if __name__=='__main__':
    args=parse_args()
    try:main(args)
    except Exception:
        folder=args.out_dir if hasattr(args,'out_dir') else folder_for(args.artifact_root)
        folder.mkdir(parents=True,exist_ok=True)
        e.atomic_json(folder/f'FAILED_{args.action}_{time.time_ns()}.json',{'traceback':traceback.format_exc(),'at':e.c.now_text()})
        raise
