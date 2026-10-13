"""Approved FeDDA GNCC preflight/training, without changing native sources."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

import fedda_gncc_training as g
import run_oasis_fedda_training as legacy

NATIVE_PREFLIGHT = legacy.preflight


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight', 'train'), required=True)
    p.add_argument('--lambda-key', choices=tuple(g.LAMBDAS), default='0.1')
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    for name in ('wavereg-root', 'data-root', 'reference-checkpoint', 'reference-field', 'out',
                 'baseline-preflight', 'baseline-training-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--min-free-gib', type=float, default=24.)
    a = p.parse_args(argv)
    if a.mode == 'train' and not a.preflight:
        p.error('Require the extended FeDDA GNCC gate before training')
    if a.min_free_gib < 24:
        p.error('Cannot lower the historical24GiB minimum')
    protected = [a.wavereg_root, a.data_root, a.baseline_preflight, a.baseline_training_root,
                 a.reference_checkpoint.parent, a.reference_field.parent, Path(__file__).resolve().parents[3]]
    if any(a.out.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.out.resolve()) for b in protected):
        p.error('Output overlaps protected source/reference')
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
            outputs = g.predict(modules[0], model, spatial,
                                batch['moving'][None, None].cuda(), batch['fixed'][None, None].cuda())
            field = outputs[1].cpu().numpy().copy()
        del outputs
        directory = args.out/f'evaluation_smoke_rank{rank:03d}'
        directory.mkdir()
        path = directory/f"disp_{batch['fixed_id']}_{batch['moving_id']}.nii.gz"
        nib.save(nib.Nifti1Image(field[0], np.eye(4)), str(path))
        restored = np.ascontiguousarray(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)[None])
        if field.shape != (1, 3, *g.SHAPE) or not np.isfinite(field).all() or not np.array_equal(restored, field):
            raise RuntimeError('FeDDA DVF geometry/serialization smoke failed')
        norm = torch.from_numpy(voxel_zyx_to_norm_dhw3(restored))
        metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(g.SHAPE, torch.device('cpu'))+norm, batch)
        metadata = dict(rank=rank, fixed_id=batch['fixed_id'], moving_id=batch['moving_id'], direction='moving_to_fixed')
        details = label_rows(metrics, batch['fixed_seg'], batch['moving_seg'], metadata)
        row = {**metadata, **c.accuracy_row(metrics, batch['fixed_id'], batch['moving_id']),
               **c.jacobian_metrics_voxel(restored, batch['fixed_mask'].bool().numpy()),
               'label_rows': len(details), 'field_path': str(path), 'field_sha256': g.sha256(path)}
        if len(details) != 35 or not all(math.isfinite(row[k]) for k in
                ('dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg')):
            raise RuntimeError('Invalid detailed FeDDA evaluation smoke')
        g.write_json(directory/'metrics.json', json_safe({'summary': row, 'per_label': details}))
        c.write_csv(directory/'per_label.csv', json_safe(details))
        rows.append(row)
    return rows


def extended_preflight(args, modules, model, spatial, nearest, sim, reg):
    import torch
    started = time.monotonic()
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    # Retain strict native reference inference/30-pair Dice gates, four fresh
    # sequential forward/reverse smokes and repeated warm-state memory checks.
    NATIVE_PREFLIGHT(args, modules, model, spatial, nearest, sim, reg)
    gate = json.loads((args.out/'preflight.json').read_text())
    if gate['initial_state_sha256'] != baseline['initial_state_sha256']:
        raise RuntimeError('FeDDA initial weights differ from the certified LNCC controls')
    expected = g.baseline_pair_hashes(args.baseline_training_root, gate['initial_state_sha256'])
    checkpoint = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.train()
    dataset = modules[1].myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root/'dataset.json'),
                                               stage='train', with_id=True, INFO=False)
    moving, fixed, moving_id, fixed_id = dataset[0]
    optimizer = g.optimizer_for(model)
    torch.cuda.reset_peak_memory_stats()
    directions = g.train_pair(modules[0], model, spatial, moving[None].cuda(), fixed[None].cuda(), sim, reg, optimizer, 1.)
    if not all(int(v['step']) == 2 for v in optimizer.state.values()):
        raise RuntimeError('Trained GNCC smoke did not execute two sequential updates')
    trained = dict(directions=directions, moving_id=moving_id, fixed_id=fixed_id, adam_updates=2,
                   finite_gradients=True, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    del optimizer, moving, fixed
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    del checkpoint
    torch.cuda.empty_cache()
    evaluated = evaluation_smoke(args, modules, model, spatial)
    peak = max([r['peak_reserved_gib'] for r in gate['native_steps']]+
               [gate['warm_memory']['peak_reserved_gib'], trained['peak_reserved_gib']])
    gate.update(gncc_gate='passed', trained_weight_step=trained, evaluation_smokes=evaluated,
                expected_pair_hashes=expected, baseline_sources_sha256=baseline['source_sha256'],
                baseline_preflight_sha256=g.sha256(args.baseline_preflight/'preflight.json'),
                baseline_evidence_sha256=baseline.get('certification', {}).get('evidence_sha256', {}),
                source_sha256=g.source_hashes(args.wavereg_root), peak_reserved_gib=peak,
                min_train_free_gib=max(24, math.ceil(peak+2)), elapsed_seconds=time.monotonic()-started)
    g.verify_hashes(baseline['source_sha256'])
    g.write_json(args.out/'preflight.json', gate)
    (args.out/'GNCC_PASSED').write_text('FeDDA GNCC/native parity, sequential-update and canonical evaluation gates passed\n')
    print(json.dumps({'gncc_preflight': 'passed', 'peak_reserved_gib': peak,
                      'required_train_free_gib': gate['min_train_free_gib']}), flush=True)


def main(args):
    g.require_identity()
    baseline = json.loads((args.baseline_preflight/'preflight.json').read_text())
    if not (args.baseline_preflight/'PASSED').is_file() or baseline['status'] != 'passed':
        raise RuntimeError('Certified historical FeDDA baseline required')
    g.verify_hashes(baseline['source_sha256'])
    g.verify_hashes(baseline.get('certification', {}).get('evidence_sha256', {}))
    if args.mode == 'train':
        gate = json.loads((args.preflight/'preflight.json').read_text())
        g.require_protocol(gate)
        if not (args.preflight/'GNCC_PASSED').is_file() or gate['gncc_gate'] != 'passed':
            raise RuntimeError('Extended GNCC gate missing')
        g.verify_hashes(gate['source_sha256'])
        g.EXPECTED_PAIRS = gate['expected_pair_hashes']
    legacy.s = g
    legacy.preflight = extended_preflight
    legacy.main(args)


if __name__ == '__main__':
    main(parse_args())
