"""Non-accepting diagnosis after reference parity failure; never launch training.

Completes the unchanged native audit and compares independently constructed
legacy/GNCC-adapter models in the SAME process/backend. No tolerance changes,
PASSED/GNCC_PASSED production markers or automatic retries are permitted.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import fedda_gncc_training as g
import run_oasis_fedda_training as legacy
import run_oasis_fedda_gncc_training as runner


def diagnose(args, modules, model, spatial, nearest, sim, reg):
    import numpy as np
    import torch
    started = time.monotonic()
    runner.NATIVE_PREFLIGHT(args, modules, model, spatial, nearest, sim, reg)
    native_audit = json.loads((args.out/'preflight.json').read_text())
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    if native_audit['initial_state_sha256'] != baseline['initial_state_sha256']:
        raise RuntimeError('Diagnostic initial-state mismatch')
    pair_hashes = g.baseline_pair_hashes(args.baseline_training_root, native_audit['initial_state_sha256'])
    checkpoint = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
    control_modules, control, control_spatial, control_nearest, control_sim, control_reg = g.native.setup_native(args.wavereg_root)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    control.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval(); control.eval()
    data = modules[1].myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root/'dataset.json'),
                                           stage='test_200', with_id=True, INFO=False)
    index = next(i for i, r in enumerate(data.items) if r['fixed'].endswith('img0333.nii.gz') and r['moving'].endswith('img0115.nii.gz'))
    moving, fixed, _, _ = data[index]
    moving, fixed = moving[None].cuda(), fixed[None].cuda()
    with torch.no_grad():
        actual = g.predict(modules[0], model, spatial, moving, fixed)[1].cpu().numpy().copy()
        reference = g.native.predict(control_modules[0], control, control_spatial, moving, fixed)[1].cpu().numpy().copy()
    diff = actual-reference
    comparison = dict(max_abs_voxel=float(np.abs(diff).max()), rmse_voxel=float(np.sqrt(np.mean(diff**2))),
                      bitwise_equal=bool(np.array_equal(actual, reference)),
                      identical_state_hash=g.state_hash(model.state_dict()) == g.state_hash(control.state_dict()),
                      native_similarity_class=type(control_sim).__name__, native_keep_sign=control_sim.keep_sign,
                      new_similarity_class=type(sim).__name__, native_parameters=sum(p.numel() for p in control.parameters()))
    del control, control_spatial, control_nearest, control_sim, control_reg, actual, reference, diff
    torch.cuda.empty_cache()
    model.train()
    optimizer = g.optimizer_for(model)
    directions = g.train_pair(modules[0], model, spatial, moving, fixed, sim, reg, optimizer, 1.)
    updates = [int(v['step']) for v in optimizer.state.values()]
    if not updates or set(updates) != {2}:
        raise RuntimeError('Diagnostic trained-checkpoint sequential-update failure')
    del optimizer, moving, fixed
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    del checkpoint
    smokes = runner.evaluation_smoke(args, modules, model, spatial)
    result = dict(status='diagnostic_only', production_qualified=False, historical_parity_issues=native_audit['issues'],
                  independent_native_vs_gncc_model=comparison, trained_checkpoint_directions=directions,
                  trained_checkpoint_adam_updates=2, evaluation_smokes=smokes,
                  expected_pair_hashes=pair_hashes, native_audit_sha256=g.sha256(args.out/'preflight.json'),
                  baseline_certificate_sha256=g.sha256(args.baseline_preflight/'preflight.json'),
                  source_sha256={**g.source_hashes(args.wavereg_root), str(Path(__file__).resolve()): g.sha256(__file__)},
                  elapsed_seconds=time.monotonic()-started)
    g.write_json(args.out/'diagnostic.json', result)
    print(json.dumps({'diagnostic': 'complete', 'production_qualified': False,
                      'historical_parity_issues': native_audit['issues'], 'native_adapter_comparison': comparison,
                      'warm_memory': native_audit['warm_memory']}), flush=True)


if __name__ == '__main__':
    args = runner.parse_args(['--mode', 'preflight', *sys.argv[1:]])
    args.mode = 'audit'
    g.require_identity()
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    g.verify_hashes(baseline['source_sha256'])
    g.verify_hashes(baseline['certification']['evidence_sha256'])
    legacy.s = g
    legacy.preflight = diagnose
    legacy.main(args)
