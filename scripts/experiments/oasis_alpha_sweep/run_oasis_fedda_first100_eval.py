"""Read-only checkpoint inference and detailed canonical first100 evaluation.

Separate from the hash-locked training sources. No training file is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fedda_training as s

VARIANT = 'oasis_seed3_first100_fedda_training_lambda_eval_v1'
from run_oasis_sitreg_first100_eval import (
    LABEL_METRICS, SUMMARY_METRICS, json_safe, write_report, stats,
    validate_first10_metrics, label_rows, summary_tables)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=range(3), required=True)
    p.add_argument('--smoke', action='store_true', help='Evaluate rank0 reused and rank10 fresh')
    p.add_argument('--min-free-gib', type=float, default=24)
    return p.parse_args(argv)


def training_ready(path):
    path = Path(path)
    if (path / 'FAILED.txt').exists():
        return False, 'training_failed'
    for name in ('TRAINING_COMPLETED', 'COMPLETED', 'training_summary.json',
                 'checkpoints/best_dice.pth.tar', 'first10/per_case.json'):
        if not (path / name).is_file():
            return False, 'waiting_for_training_and_first10'
    summary = json.loads((path / 'training_summary.json').read_text())
    if summary['epoch'] != 200 or summary['adam_updates'] != 20000:
        raise ValueError('Unexpected final training budget')
    return True, 'ready'


def evaluation_sources():
    here = Path(__file__).resolve()
    files = (here, here.with_name('launch_oasis_fedda_training.py'),
             here.with_name('fedda_training.py'), here.with_name('run_oasis_fedda_training.py'),
             here.with_name('run_oasis_sitreg_first100_eval.py'), here.with_name('sitreg_training.py'),
             here.with_name('smoke_dl_training_memory.py'),
             here.with_name('common.py'), here.with_name('evaluate_saved_baselines_first10.py'))
    return {str(p): s.sha256(p) for p in files}



def run(args):
    ready, reason = training_ready(args.training_dir)
    if not ready:
        raise RuntimeError(reason)
    train_manifest = json.loads((args.training_dir / 'manifest.json').read_text())
    train_args = train_manifest['arguments']
    key = train_args['lambda_key']
    if key not in s.LAMBDAS:
        raise ValueError('Unapproved lambda')
    source_hashes = evaluation_sources()
    s.verify_hashes(train_manifest['source_sha256'])
    data_root, wavereg_root = Path(train_args['data_root']), Path(train_args['wavereg_root'])
    s.check_dataset(data_root)
    checkpoint_path = args.training_dir / 'checkpoints/best_dice.pth.tar'
    checkpoint_sha = s.sha256(checkpoint_path)
    gpu_state = s.gpu_rows()
    device = next(r for r in gpu_state if r['index'] == args.gpu)
    if device['free_mib'] < args.min_free_gib * 1024:
        raise RuntimeError(f'Insufficient inference headroom: {device}')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = device['uuid']
    import numpy as np
    import nibabel as nib
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    fraction = (device['free_mib'] - 2048) / device['total_mib']
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    # Explicit historical protocol override; never silently mix inference backends.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    import common as c
    from evaluate_saved_baselines_first10 import voxel_zyx_to_norm_dhw3
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    if len(pairs) != 200 or pairs[0] != ('0333', '0115') or len(set(pairs[:100])) != 100:
        raise ValueError('Invalid canonical pair list')
    ranks = [0, 10] if args.smoke else list(range(100))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    training_summary = json.loads((args.training_dir / 'training_summary.json').read_text())
    if (checkpoint['alpha'] != s.LAMBDAS[key] or checkpoint['seed'] != 42
            or checkpoint['epoch'] != training_summary['best_epoch']
            or checkpoint['best_val_dice'] != training_summary['best_val_dice']):
        raise ValueError('Best-validation checkpoint metadata mismatch')
    s.verify_hashes(checkpoint['source_sha256'])
    modules, model, spatial, nearest, sim, reg = s.setup_native(wavereg_root)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint, nearest, sim, reg
    model.eval().requires_grad_(False)
    initial_state_sha = s.state_hash(model.state_dict())
    first10 = {int(r['rank']): r for r in json.loads((args.training_dir / 'first10/per_case.json').read_text())}
    if sorted(first10) != list(range(10)):
        raise ValueError('Incomplete original first10 cohort')
    manifest = {'variant': VARIANT, 'model_key': 'fedda_1mf', 'smoke': args.smoke, 'ranks': ranks,
                'lambda_key': key, 'alpha': s.LAMBDAS[key], 'training_dir': str(args.training_dir),
                'training_manifest_sha256': s.sha256(args.training_dir / 'manifest.json'),
                'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha,
                'best_epoch': training_summary['best_epoch'], 'best_val_dice': training_summary['best_val_dice'],
                'pair_file': str(c.DEFAULT_PAIR_FILE), 'pair_file_sha256': s.sha256(c.DEFAULT_PAIR_FILE),
                'dataset_manifest_sha256': s.sha256(data_root / 'dataset.json'),
                'pairs': [dict(rank=r, fixed_id=pairs[r][0], moving_id=pairs[r][1]) for r in ranks],
                'direction': 'moving_to_fixed', 'field_storage': 'channel_first_zyx_voxel',
                'shape': [3, *s.SHAPE], 'spacing_mm': [1, 1, 1],
                'inference_preprocessing': 'native_raw_stored_intensities_LIA_to_RAS',
                'labels': list(range(1, 36)), 'absent_label_rule': 'exclude_if_absent_in_fixed_moving_or_warped',
                'json_missing_value': None, 'csv_missing_value': 'empty_cell',
                'asd_definition': 'pooled_surfel_area_weighted_bidirectional_mean',
                'asd_backend': 'google_deepmind_surface_distance_area_weighted',
                'foreground': 'fixed_raw_image_positive', 'jacobian': 'canonical_LUMIR_crop2_positive_J_log_std',
                'reuse': 'first10 saved field bytes; recompute all metrics and label details',
                'first10_parity_atol': {m: 1e-8 if m == 'diffusion_l2_full' else 1e-10 for m in SUMMARY_METRICS},
                'physical_gpu': args.gpu, 'visible_cuda_device': 'cuda:0', 'gpu_uuid': device['uuid'],
                'gpu_before': gpu_state, 'allocator_fraction': fraction,
                'gpu_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True),
                'backend': {'benchmark': True, 'deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                'cuda_nondeterminism': 'historical benchmark mode; inference is not bitwise reproducible',
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'cpu_threads': torch.get_num_threads(), 'uid': os.getuid(), 'gid': os.getgid(), 'pid': os.getpid(),
                'source_sha256': source_hashes, 'native_source_sha256': train_manifest['source_sha256'],
                'repo_revision': subprocess.check_output(['git', '-C', str(c.REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'container_image_id': os.environ.get('INRORSPLINE_CONTAINER_IMAGE_ID'),
                'started_unix': time.time(), 'state_sha256': initial_state_sha}
    s.write_json(args.out / 'manifest.json', manifest)
    rows, labels = [], []
    started = time.monotonic()
    for rank in ranks:
        s.verify_hashes(source_hashes)
        s.verify_hashes(train_manifest['source_sha256'])
        fixed_id, moving_id = pairs[rank]
        case_dir = args.out / 'cases' / f'rank{rank:03d}_{fixed_id}_{moving_id}'
        case_dir.mkdir(parents=True, exist_ok=False)
        begin = time.monotonic()
        if rank < 10:
            reference = first10[rank]
            field_path = args.training_dir / 'first10' / f'disp_{fixed_id}_{moving_id}.nii.gz'
            if (reference['fixed_id'], reference['moving_id'], reference['direction']) != (fixed_id, moving_id, 'moving_to_fixed'):
                raise ValueError('First10 identity mismatch')
            if s.sha256(field_path) != reference['field_sha256'] or reference['checkpoint'] != str(checkpoint_path):
                raise ValueError('First10 provenance mismatch')
            field = np.asarray(nib.load(str(field_path)).dataobj, dtype=np.float32)[None]
            inference_seconds = 0.0
            reused = True
        else:
            moving = c.load_nifti_reoriented(data_root / 'test' / f'img{moving_id}.nii.gz', dtype=np.float32)
            fixed = c.load_nifti_reoriented(data_root / 'test' / f'img{fixed_id}.nii.gz', dtype=np.float32)
            if moving.shape != s.SHAPE or fixed.shape != s.SHAPE or not np.isfinite(moving).all() or not np.isfinite(fixed).all():
                raise ValueError('Invalid native inputs')
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.no_grad():
                outputs = s.predict(modules[0], model, spatial,
                                    torch.from_numpy(moving.copy())[None, None].cuda(),
                                    torch.from_numpy(fixed.copy())[None, None].cuda())
                field = outputs[1].cpu().numpy().copy()
            del outputs, moving, fixed
            torch.cuda.synchronize()
            inference_seconds = time.monotonic() - tick
            field_path = case_dir / f'disp_{fixed_id}_{moving_id}.nii.gz'
            nib.save(nib.Nifti1Image(field[0], np.eye(4)), str(field_path))
            restored = np.asarray(nib.load(str(field_path)).dataobj, dtype=np.float32)[None]
            if not np.array_equal(field, restored):
                raise ValueError('DVF serialization roundtrip mismatch')
            field = restored
            reused = False
        # Match the original inference buffer's C-order without changing voxel values.
        field = np.ascontiguousarray(field)
        if field.shape != (1, 3, *s.SHAPE) or not np.isfinite(field).all():
            raise ValueError('Invalid displacement geometry/values')
        batch = c.load_oasis_case(rank=rank, device=torch.device('cpu'), oasis_root=data_root)
        if (batch['fixed_id'], batch['moving_id']) != (fixed_id, moving_id):
            raise ValueError('Canonical case mismatch')
        dense = torch.from_numpy(voxel_zyx_to_norm_dhw3(field))
        metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(s.SHAPE, torch.device('cpu')) + dense, batch)
        metadata = dict(rank=rank, fixed_id=fixed_id, moving_id=moving_id, lambda_key=key, alpha=s.LAMBDAS[key],
                        direction='moving_to_fixed', best_epoch=training_summary['best_epoch'], checkpoint_sha256=checkpoint_sha)
        details = label_rows(metrics, batch['fixed_seg'], batch['moving_seg'], metadata)
        row = {**metadata, **c.accuracy_row(metrics, fixed_id, moving_id),
               **c.jacobian_metrics_voxel(field, batch['fixed_mask'].bool().numpy()),
               'diffusion_l2_full': c.diffusion_loss_voxel(field),
               'valid_label_count': sum(r['included_in_case_mean'] for r in details),
               'field_path': str(field_path), 'field_sha256': s.sha256(field_path), 'reused_first10': reused,
               'inference_seconds': inference_seconds, 'physical_gpu': args.gpu,
               'foreground': 'fixed_raw_image_positive'}
        for metric in SUMMARY_METRICS:
            if not math.isfinite(row[metric]):
                raise ValueError(f'Nonfinite case metric {metric}')
        if reused:
            differences = validate_first10_metrics(row, reference)
            row['first10_max_abs_metric_difference'] = max(abs(v) for v in differences.values())
        inputs = {str(path): s.sha256(path) for subject in (fixed_id, moving_id)
                  for path in (data_root / 'test' / f'img{subject}.nii.gz', data_root / 'test' / f'seg{subject}.nii.gz')}
        nib.save(nib.Nifti1Image(metrics['warped_seg'].numpy().astype(np.int16), np.eye(4)), str(case_dir / 'warped_seg.nii.gz'))
        row['case_seconds'] = time.monotonic() - begin
        s.write_json(case_dir / 'metrics.json', json_safe({'summary': row, 'per_label': details, 'input_sha256': inputs}))
        c.write_csv(case_dir / 'per_label.csv', json_safe(details))
        (case_dir / 'COMPLETED').write_text('35 labels evaluated; absent labels explicitly excluded\n')
        rows.append(row)
        labels.extend(details)
        s.write_json(args.out / 'per_case.json', json_safe(rows))
        s.write_json(args.out / 'per_label_per_case.json', json_safe(labels))
        c.write_csv(args.out / 'per_case.csv', json_safe(rows))
        c.write_csv(args.out / 'per_label_per_case.csv', json_safe(labels))
        progress = {'status': 'evaluating', 'lambda_key': key, 'completed_cases': len(rows),
                    'total_cases': len(ranks), 'last_rank': rank, 'elapsed_seconds': time.monotonic() - started,
                    'last_metrics': {m: row[m] for m in SUMMARY_METRICS}}
        s.write_json(args.out / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
        del batch, dense, metrics, field
    if [r['rank'] for r in rows] != ranks or len(labels) != 35 * len(ranks):
        raise ValueError('Incomplete cohort')
    if s.state_hash(model.state_dict()) != initial_state_sha or s.sha256(checkpoint_path) != checkpoint_sha:
        raise ValueError('Model/checkpoint changed during evaluation')
    s.verify_hashes(source_hashes)
    s.verify_hashes(train_manifest['source_sha256'])
    summary, per_label_summary = summary_tables(rows, labels)
    report = {'status': 'complete', 'lambda_key': key, 'case_count': len(rows), 'label_rows': len(labels),
              'metrics': summary, 'per_label': per_label_summary,
              'first10_metrics': summary_tables([r for r in rows if r['rank'] < 10], [r for r in labels if r['rank'] < 10])[0],
              'elapsed_seconds': time.monotonic() - started,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
              'checkpoint_sha256': checkpoint_sha, 'source_sha256': source_hashes}
    s.write_json(args.out / 'summary.json', report)
    c.write_csv(args.out / 'per_label_summary.csv', per_label_summary)
    c.write_csv(args.out / 'summary.csv', [{'metric': m, **v} for m, v in summary.items()])
    text = ['# FeDDA-1MF first100 detailed evaluation', '', f'Lambda: {key}; cases: {len(rows)}; best validation epoch: {training_summary["best_epoch"]}.', '',
            '| Metric | Mean | Sample SD |', '|---|---:|---:|']
    text += [f'| {m} | {v["mean"]:.8f} | {v["sample_sd"]:.8f} |' for m, v in summary.items()]
    text += ['', 'All35 label rows/case are in per_label_per_case.csv/json. Missing labels are null/blank with explicit exclusion reasons.',
             'HD95 and symmetric ASD use canonical area-weighted surfaces at1mm spacing. Summary means weight cases equally.',
             'First10 fields are reused byte-for-byte; ranks10–99 use the same best-validation checkpoint. No test-based checkpoint selection.']
    write_report(args.out / 'report.md', text)
    s.write_json(args.out / 'progress.json', {'status': 'complete', 'completed_cases': len(rows), 'total_cases': len(ranks)})
    (args.out / ('PASSED' if args.smoke else 'COMPLETED')).write_text('Detailed canonical evaluation passed\n')


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        (args.out / 'FAILED.txt').write_text(traceback.format_exc())
        s.write_json(args.out / 'progress.json', {'status': 'failed', 'time': time.time(),
                     'failure_kind': 'cuda_oom' if 'out of memory' in str(exc).lower() else 'other'})
        raise


if __name__ == '__main__':
    main()
