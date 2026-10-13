"""Approval-scoped SITReg lambda training, with mandatory reference parity gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import sitreg_training as s
from smoke_dl_training_memory import Tee, sha256


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['preflight', 'train'], required=True)
    p.add_argument('--lambda-key', choices=list(s.LAMBDAS), default='0.1')
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    p.add_argument('--wavereg-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--reference-checkpoint', type=Path, required=True)
    p.add_argument('--reference-field', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', type=Path)
    a = p.parse_args(argv)
    if a.mode == 'train' and not a.preflight:
        p.error('Training requires a successful --preflight directory')
    return a


def preflight(args, modules, model, spatial, nearest, sim, reg):
    import nibabel as nib
    import numpy as np
    import torch
    models, data, _, _, _, evaluation, _, _ = modules
    assert sha256(args.reference_checkpoint) == s.CHECKPOINT_SHA
    checkpoint = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
    fresh = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_hash = s.state_hash(fresh)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    test = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                       stage='test_200', with_id=True, INFO=False)
    index = next(i for i, pair in enumerate(test.items)
                 if pair['fixed'].endswith('img0333.nii.gz') and pair['moving'].endswith('img0115.nii.gz'))
    moving, fixed, moving_id, fixed_id = test[index]
    assert (fixed_id, moving_id) == ('0333', '0115')
    with torch.no_grad():
        outputs = s.predict(models, model, spatial, moving[None].cuda(), fixed[None].cuda())
        actual = outputs[2].cpu().numpy().copy()
    del outputs
    reference = nib.load(str(args.reference_field)).get_fdata(dtype=np.float32)[None]
    assert actual.shape == reference.shape == (1, 3, *s.SHAPE)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    difference = actual - reference
    field_error = float(np.abs(difference).max())
    # Historical benchmark-mode inference is measurably non-bitwise repeatable
    # (including identical weights/inputs in one process). Keep strict metric
    # gates and solver-scale field bounds; never claim exact field reproduction.
    parity = {'max_abs_voxel_error': field_error, 'rmse_voxel': float(np.sqrt(np.mean(difference ** 2))),
              'tolerance_max_abs_voxel': 0.25, 'tolerance_rmse_voxel': 0.005,
              'bitwise_field_reproduction': bool(np.array_equal(actual, reference)),
              'reproduction_kind': 'historical_benchmark_mode_numerical_and_metric',
              'fixed_id': fixed_id, 'moving_id': moving_id}
    s.write_json(args.out / 'field_parity.json', parity)
    if field_error > 0.25 or parity['rmse_voxel'] > 0.005:
        raise RuntimeError(f'Reference inference mismatch: {parity}')
    actual_metrics = s.canonical_case(args, actual, 0)
    reference_metrics = s.canonical_case(args, reference, 0)
    for metric, tolerance in [('dice_mean', 5e-5), ('hd95_mean', 0.001), ('asd_symmetric_mean', 0.0001)]:
        if abs(actual_metrics[metric] - reference_metrics[metric]) > tolerance:
            raise RuntimeError(f'Canonical metric parity failed: {metric}')
    parity.update(actual_metrics=actual_metrics, reference_metrics=reference_metrics)
    s.write_json(args.out / 'field_parity.json', parity)
    del actual, reference, difference
    val_data = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                           stage='validation', with_seg=True, INFO=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                            num_workers=0, pin_memory=True)
    reproduced_validation = validation(modules, model, spatial, nearest, sim, reg, val_loader, 1.0)
    expected_validation_dice = float(checkpoint['best_val_dice'])
    if abs(reproduced_validation['dice'] - expected_validation_dice) > 1e-4:
        raise RuntimeError(f'Native30-pair validation reproduction failed: {reproduced_validation}')
    print(json.dumps({'validation_parity': reproduced_validation, 'historical_dice': expected_validation_dice}), flush=True)
    model.load_state_dict(fresh, strict=True)
    model.train()
    train = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                       stage='train', with_id=True, INFO=False)
    moving, fixed, moving_id, fixed_id = train[0]
    moving, fixed = moving[None].cuda(), fixed[None].cuda()
    steps = []
    for key, alpha in s.LAMBDAS.items():
        model.load_state_dict(fresh, strict=True)
        optimizer = s.optimizer_for(model)
        outputs = s.predict(models, model, spatial, moving, fixed)
        loss, similarity, diffusion = s.joint_loss(outputs, moving, fixed, sim, reg, alpha)
        optimizer.zero_grad()
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert all(int(v['step'].item()) == 1 for v in optimizer.state.values())
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lambda': alpha}
        saved = args.out / f'smoke_lambda_{key}.pth'
        s.atomic_checkpoint(saved, state)
        reloaded = torch.load(saved, map_location='cpu', weights_only=False)
        assert s.state_hash(reloaded['state_dict']) == s.state_hash(model.state_dict())
        steps.append({'lambda_key': key, 'lambda': alpha, 'loss': loss.item(),
                      'similarity': similarity.item(), 'diffusion': diffusion.item(),
                      'finite_gradients': True, 'checkpoint_roundtrip_exact': True})
        del state, reloaded, optimizer
    result = {'status': 'passed', 'initial_state_sha256': initial_hash, 'parity': parity,
              'native_validation': reproduced_validation, 'historical_validation_dice': expected_validation_dice,
              'native_steps': steps, 'source_sha256': s.source_hashes(args.wavereg_root),
              'data_sha256': sha256(args.data_root / 'dataset.json'),
              'reference_checkpoint_sha256': sha256(args.reference_checkpoint),
              'reference_field_sha256': sha256(args.reference_field)}
    s.write_json(args.out / 'preflight.json', result)
    (args.out / 'PASSED').write_text('reference parity and all four native update smokes passed\n')
    print(json.dumps({'preflight': 'passed', 'field_error': field_error, 'metrics': actual_metrics}), flush=True)


def validation(modules, model, spatial, nearest, sim, reg, loader, alpha):
    import numpy as np
    import torch
    models, _, _, _, _, evaluation, _, _ = modules
    model.eval()
    rows = []
    with torch.no_grad():
        for moving, fixed, moving_seg, fixed_seg in loader:
            moving, fixed = moving.cuda(), fixed.cuda()
            outputs = s.predict(models, model, spatial, moving, fixed)
            total, similarity, diffusion = s.joint_loss(outputs, moving, fixed, sim, reg, alpha)
            warped_seg = nearest([moving_seg.cuda().float(), outputs[2]])
            dice = evaluation.dice_val_VOI(warped_seg.long(), fixed_seg.cuda().long(), n_labels=35)
            rows.append([total.item(), similarity.item(), diffusion.item(), float(dice)])
    means = np.mean(rows, axis=0)
    assert len(rows) == 30 and np.isfinite(means).all()
    return dict(zip(['loss', 'similarity', 'diffusion', 'dice'], map(float, means)))


def first10(args, modules, model, spatial, best):
    import common as c
    import nibabel as nib
    import numpy as np
    import torch
    model.load_state_dict(torch.load(best, map_location='cpu', weights_only=False)['state_dict'], strict=True)
    model.eval()
    models = modules[0]
    output = args.out / 'first10'
    output.mkdir()
    rows = []
    for rank, (fixed_id, moving_id) in enumerate(c.parse_pair_file(c.DEFAULT_PAIR_FILE)[:10]):
        moving = c.load_nifti_reoriented(args.data_root / 'test' / f'img{moving_id}.nii.gz', dtype=np.float32)
        fixed = c.load_nifti_reoriented(args.data_root / 'test' / f'img{fixed_id}.nii.gz', dtype=np.float32)
        with torch.no_grad():
            outputs = s.predict(models, model, spatial,
                                torch.from_numpy(moving.copy())[None, None].cuda(),
                                torch.from_numpy(fixed.copy())[None, None].cuda())
            field = outputs[2].cpu().numpy().copy()
        del outputs
        assert field.shape == (1, 3, *s.SHAPE) and np.isfinite(field).all()
        path = output / f'disp_{fixed_id}_{moving_id}.nii.gz'
        nib.save(nib.Nifti1Image(field[0], np.eye(4)), str(path))
        row = s.canonical_case(args, field, rank)
        row.update(lambda_key=args.lambda_key, alpha=s.LAMBDAS[args.lambda_key],
                   field_sha256=sha256(path), storage='channel_first_zyx_voxel', checkpoint=str(best))
        rows.append(row)
        s.write_json(output / 'per_case.json', rows)
        print(json.dumps({'test_rank': rank, **row}), flush=True)
    c.write_csv(output / 'per_case.csv', rows)
    keys = ['dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg', 'diffusion_l2_full']
    summary = {key: {'mean': float(np.mean([r[key] for r in rows])),
                     'sample_sd': float(np.std([r[key] for r in rows], ddof=1))} for key in keys}
    s.write_json(output / 'summary.json', summary)
    return summary


def train(args, modules, model, spatial, nearest, sim, reg):
    import numpy as np
    import torch
    models, data, _, _, _, _, _, _ = modules
    gate = json.loads((args.preflight / 'preflight.json').read_text())
    assert (args.preflight / 'PASSED').exists() and gate['status'] == 'passed'
    s.verify_hashes(gate['source_sha256'])
    assert sha256(args.reference_checkpoint) == gate['reference_checkpoint_sha256']
    assert sha256(args.reference_field) == gate['reference_field_sha256']
    initial_hash = s.state_hash(model.state_dict())
    assert initial_hash == gate['initial_state_sha256'], 'Seed42 model initialization mismatch'
    alpha = s.LAMBDAS[args.lambda_key]
    dataset = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                         stage='train', with_id=True, INFO=False)
    val = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                     stage='validation', with_seg=True, INFO=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    optimizer = s.optimizer_for(model)
    best_dice, best_epoch = -1.0, None
    pair_digest = hashlib.sha256()
    total_started = time.monotonic()
    checkpoint_dir = args.out / 'checkpoints'
    checkpoint_dir.mkdir()
    s.write_json(args.out / 'initialization.json', {'state_sha256': initial_hash, 'seed': 42,
                                                   'parameters': 15076110, 'alpha': alpha})
    with (args.out / 'pairs.jsonl').open('x') as pair_log, (args.out / 'epochs.jsonl').open('x') as epoch_log:
        for epoch in range(1, 201):
            s.verify_hashes(gate['source_sha256'])
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = []
            for pair_index in range(50):
                batch, iterator = s.next_pair(iterator, loader)
                moving, fixed, moving_id, fixed_id = batch
                moving, fixed = moving.cuda(), fixed.cuda()
                outputs = s.predict(models, model, spatial, moving, fixed)
                total, similarity, diffusion = s.joint_loss(outputs, moving, fixed, sim, reg, alpha)
                if not bool(torch.isfinite(total)):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad()
                total.backward()
                if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                    raise RuntimeError('Nonfinite training gradient')
                optimizer.step()
                numbers = [total.item(), similarity.item(), diffusion.item()]
                values.append(numbers)
                pair_digest.update(f'{moving_id[0]}>{fixed_id[0]}\n'.encode())
                record = dict(epoch=epoch, pair_index=pair_index, moving_id=moving_id[0], fixed_id=fixed_id[0],
                              loss=numbers[0], similarity=numbers[1], diffusion=numbers[2],
                              weighted_diffusion=alpha * numbers[2])
                pair_log.write(json.dumps(record) + '\n')
                pair_log.flush()
                if pair_index % 10 == 0:
                    s.write_json(args.out / 'progress.json', {'status': 'training', **record,
                                 'adam_updates': (epoch - 1) * 50 + pair_index + 1,
                                 'elapsed_sec': time.monotonic() - total_started, 'best_val_dice': best_dice})
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - started
            started = time.monotonic()
            val_metrics = validation(modules, model, spatial, nearest, sim, reg, val_loader, alpha)
            improved = val_metrics['dice'] > best_dice
            if improved:
                best_dice, best_epoch = val_metrics['dice'], epoch
            payload = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'best_val_dice': best_dice, 'best_epoch': best_epoch, 'val_metrics': val_metrics,
                       'alpha': alpha, 'seed': 42, 'initial_state_sha256': initial_hash,
                       'pair_stream_sha256': pair_digest.hexdigest(),
                       'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                     'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
                       'source_sha256': gate['source_sha256']}
            s.atomic_checkpoint(checkpoint_dir / 'latest.pth.tar', payload)
            if improved:
                s.atomic_checkpoint(checkpoint_dir / 'best_dice.pth.tar', payload)
            row = dict(epoch=epoch, alpha=alpha, train_loss=float(np.mean(values, axis=0)[0]),
                       train_similarity=float(np.mean(values, axis=0)[1]), train_diffusion=float(np.mean(values, axis=0)[2]),
                       validation=val_metrics, best_val_dice=best_dice, best_epoch=best_epoch,
                       train_seconds=train_seconds, validation_and_save_seconds=time.monotonic() - started,
                       elapsed_seconds=time.monotonic() - total_started, adam_updates=epoch * 50,
                       pair_stream_sha256=pair_digest.hexdigest(),
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
            epoch_log.write(json.dumps(row) + '\n')
            epoch_log.flush()
            s.write_json(args.out / 'progress.json', {'status': 'epoch_complete', **row})
            print(json.dumps(row), flush=True)
    assert all(int(v['step'].item()) == 10000 for v in optimizer.state.values())
    s.verify_hashes(gate['source_sha256'])
    s.write_json(args.out / 'training_summary.json', row)
    (args.out / 'TRAINING_COMPLETED').write_text('200 epochs;10000 joint Adam updates\n')
    del optimizer, payload, outputs, total, similarity, diffusion
    torch.cuda.empty_cache()
    summary = first10(args, modules, model, spatial, checkpoint_dir / 'best_dice.pth.tar')
    s.write_json(args.out / 'progress.json', {'status': 'complete', 'training': row, 'first10': summary})
    (args.out / 'COMPLETED').write_text('training and canonical first10 evaluation completed\n')


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    log = (args.out / 'stdout.log').open('x')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(json.dumps({'argv': sys.argv, 'pid': os.getpid(), 'started_unix': time.time()}), flush=True)
    try:
        if args.mode == 'train':
            gate = json.loads((args.preflight / 'preflight.json').read_text())
            assert (args.preflight / 'PASSED').exists() and gate['status'] == 'passed'
            s.verify_hashes(gate['source_sha256'])
        gpu_state, fraction = s.select_gpu(args.gpu)
        s.check_dataset(args.data_root)
        import torch
        setup = s.setup_native(args.wavereg_root)
        manifest = dict(arguments=vars(args), physical_gpu=args.gpu, visible_device='cuda:0',
                        gpu_start=gpu_state, allocator_fraction=fraction,
                        pid=os.getpid(), uid=os.getuid(), gid=os.getgid(),
                        pytorch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                        backend={'benchmark': True, 'deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
                        cpu_threads=torch.get_num_threads(), batch_size=1, epochs=200, pairs_per_epoch=50,
                        seed=42, alpha=s.LAMBDAS[args.lambda_key], lr=1e-4, optimizer='Adam_AMSGrad',
                        fresh_initialization=True, source_sha256=s.source_hashes(args.wavereg_root),
                        data_sha256=sha256(args.data_root / 'dataset.json'),
                        repo_revision=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[3]),
                                                              'rev-parse', 'HEAD'], text=True).strip())
        s.write_json(args.out / 'manifest.json', manifest)
        if args.mode == 'preflight':
            preflight(args, *setup)
        else:
            train(args, *setup)
    except BaseException:
        (args.out / 'FAILED.txt').write_text(traceback.format_exc())
        s.write_json(args.out / 'progress.json', {'status': 'failed', 'time': time.time(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main(parse_args())
