"""GNCC² adapter: preserve the audited native SITReg loop and read-only controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import sitreg_gncc2_training as g
import run_oasis_sitreg_training as legacy

NATIVE_PREFLIGHT = legacy.preflight


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight', 'train'), required=True)
    p.add_argument('--lambda-key', choices=tuple(g.LAMBDAS), default='0.1')
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    for name in ('wavereg-root', 'data-root', 'reference-checkpoint', 'reference-field', 'out', 'baseline-preflight'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight', type=Path)
    a = p.parse_args(argv)
    if a.mode == 'train' and not a.preflight:
        p.error('Require extended GNCC² preflight before training')
    protected = [a.wavereg_root, a.data_root, a.baseline_preflight.parent, Path(__file__).resolve().parents[3]]
    if any(a.out.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.out.resolve()) for b in protected):
        p.error('Output overlaps protected input/source')
    return a


def evaluation_smoke(args, modules, model, spatial):
    import common as c
    import nibabel as nib
    import numpy as np
    import torch
    from evaluate_saved_baselines_first10 import voxel_zyx_to_norm_dhw3
    from run_oasis_sitreg_first100_eval import label_rows, json_safe
    rows = []
    for rank in (0, 10):
        batch = c.load_oasis_case(rank=rank, device=torch.device('cpu'), oasis_root=args.data_root)
        with torch.no_grad():
            outputs = g.predict(modules[0], model, spatial, batch['moving'][None, None].cuda(), batch['fixed'][None, None].cuda())
            field = outputs[2].cpu().numpy().copy()
        del outputs
        case = args.out/f'evaluation_smoke_rank{rank:03d}'
        case.mkdir()
        path = case/f"disp_{batch['fixed_id']}_{batch['moving_id']}.nii.gz"
        nib.save(nib.Nifti1Image(field[0], np.eye(4)), str(path))
        restored = np.ascontiguousarray(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)[None])
        if field.shape != (1, 3, *g.SHAPE) or not np.isfinite(field).all() or not np.array_equal(restored, field):
            raise RuntimeError('Smoke DVF serialization/geometry failed')
        norm = torch.from_numpy(voxel_zyx_to_norm_dhw3(restored))
        metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(g.SHAPE, torch.device('cpu'))+norm, batch)
        metadata = {'rank': rank, 'fixed_id': batch['fixed_id'], 'moving_id': batch['moving_id'], 'direction': 'moving_to_fixed'}
        labels = label_rows(metrics, batch['fixed_seg'], batch['moving_seg'], metadata)
        row = {**metadata, **c.accuracy_row(metrics, batch['fixed_id'], batch['moving_id']),
               **c.jacobian_metrics_voxel(restored, batch['fixed_mask'].bool().numpy()),
               'diffusion_l2_full': c.diffusion_loss_voxel(restored), 'label_rows': len(labels),
               'field_path': str(path), 'field_sha256': g.sha256(path)}
        g.write_json(case/'metrics.json', json_safe({'summary': row, 'labels': labels}))
        c.write_csv(case/'per_label.csv', json_safe(labels))
        rows.append(row)
    return rows


def extended_preflight(args, modules, model, spatial, nearest, sim, reg):
    import torch
    started = time.monotonic()
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    # Native strict loading, saved-field/numerical metric gates, all30 validation
    # pairs, then four independent fresh GNCC² Adam/serialization smokes.
    NATIVE_PREFLIGHT(args, modules, model, spatial, nearest, sim, reg)
    gate = json.loads((args.out/'preflight.json').read_text())
    if gate['initial_state_sha256'] != baseline['initial_state_sha256']:
        raise RuntimeError('Initial weights no longer match the LNCC sweep')
    reference_train = args.baseline_preflight.parent/'sitreg_lambda_0.1_seed42'
    epochs = [json.loads(line) for line in (reference_train/'epochs.jsonl').read_text().splitlines()]
    if [e['epoch'] for e in epochs] != list(range(1, 201)) or epochs[-1]['adam_updates'] != 10000:
        raise RuntimeError('Incomplete baseline pair stream')
    pair_hashes = {str(e['epoch']): e['pair_stream_sha256'] for e in epochs}
    for key in ('0.1', '1over3', '3', '10'):
        folder = args.baseline_preflight.parent/f'sitreg_lambda_{key}_seed42'
        init = json.loads((folder/'initialization.json').read_text())
        rows = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        if init['state_sha256'] != gate['initial_state_sha256'] or {str(e['epoch']): e['pair_stream_sha256'] for e in rows} != pair_hashes:
            raise RuntimeError('Historical initial weights/pair sequences disagree')
    checkpoint = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.train()
    dataset = modules[1].myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root/'dataset.json'), stage='train', with_id=True, INFO=False)
    moving, fixed, moving_id, fixed_id = dataset[0]
    moving, fixed = moving[None].cuda(), fixed[None].cuda()
    optimizer = g.optimizer_for(model)
    outputs = g.predict(modules[0], model, spatial, moving, fixed)
    loss, similarity, diffusion = g.joint_loss(outputs, moving, fixed, sim, reg, 1.)
    optimizer.zero_grad(); loss.backward()
    if not torch.isfinite(loss) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Trained-checkpoint GNCC² gradient smoke failed')
    optimizer.step()
    trained_step = {'loss': loss.item(), 'similarity': similarity.item(), 'diffusion': diffusion.item(),
                    'moving_id': moving_id, 'fixed_id': fixed_id, 'finite_gradients': True, 'updates': 1}
    del optimizer, outputs, loss, similarity, diffusion, moving, fixed
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    del checkpoint
    torch.cuda.empty_cache()
    evaluated = evaluation_smoke(args, modules, model, spatial)
    gate.update(gncc2_gate='passed', trained_weight_step=trained_step, evaluation_smokes=evaluated,
                expected_pair_hashes=pair_hashes, baseline_sources_sha256=baseline['source_sha256'],
                baseline_preflight_sha256=g.sha256(args.baseline_preflight/'preflight.json'),
                source_sha256=g.source_hashes(args.wavereg_root),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                elapsed_seconds=time.monotonic()-started)
    g.verify_hashes(baseline['source_sha256'])
    g.write_json(args.out/'preflight.json', gate)
    (args.out/'GNCC2_PASSED').write_text('Extended GNCC²/native parity and detailed evaluation gates passed\n')
    print(json.dumps({'gncc_preflight': 'passed', 'peak_reserved_gib': gate['peak_reserved_gib']}), flush=True)


def main(args):
    g.require_identity()
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    if not (args.baseline_preflight/'PASSED').is_file() or baseline['status'] != 'passed':
        raise RuntimeError('Historical baseline gate missing')
    g.verify_hashes(baseline['source_sha256'])
    if args.mode == 'train':
        gate = json.loads((args.preflight/'preflight.json').read_text())
        g.require_protocol(gate)
        if not (args.preflight/'GNCC2_PASSED').is_file() or gate['gncc2_gate'] != 'passed':
            raise RuntimeError('Extended GNCC² gate missing')
        g.verify_hashes(gate['source_sha256'])
        g.EXPECTED_PAIRS = gate['expected_pair_hashes']
    # Bind dependencies in this worker only, not in a shared process or on disk.
    legacy.s = g
    legacy.preflight = extended_preflight
    legacy.main(args)


if __name__ == '__main__':
    main(parse_args())

