"""Checkpoint-faithful FeDDA GNCC² continuation into a new artifact directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import fedda_gncc2_training as s
import run_oasis_fedda_gncc2_training as adapter
import run_oasis_fedda_training as legacy
from smoke_dl_training_memory import sha256
validation = legacy.validation
first10 = legacy.first10

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
    assert args.min_free_gib >= gate['min_train_free_gib'], 'Memory gate below measured peak requirement'
    alpha = s.LAMBDAS[args.lambda_key]
    from oasis_resume_state import restore_iterator, accepted_logs
    source = args.resume_from
    saved = torch.load(source/'checkpoints/latest.pth.tar', map_location='cpu', weights_only=False)
    s.require_protocol(saved)
    s.verify_hashes(saved['source_sha256'])
    epoch0 = saved['epoch']
    if not 1 <= epoch0 < 200 or saved['alpha'] != alpha or saved['seed'] != 42 or saved['initial_state_sha256'] != initial_hash:
        raise RuntimeError('Wrong resume checkpoint metadata')
    old_epochs, old_pairs = accepted_logs(source, epoch0, saved, gate['expected_pair_hashes'])
    loader, iterator, pair_digest = restore_iterator(data, args.data_root, epoch0, gate['expected_pair_hashes'], saved['rng_state'])
    if pair_digest.hexdigest() != saved['pair_stream_sha256']:
        raise RuntimeError('Replayed iterator does not match checkpoint')
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise RuntimeError('Nonfinite checkpoint model')
    dataset = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                         stage='train', with_id=True, INFO=False)
    val = data.myDataset_OASIS_v1_json(str(args.data_root), str(args.data_root / 'dataset.json'),
                                     stage='validation', with_seg=True, INFO=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = s.optimizer_for(model)
    optimizer.load_state_dict(saved['optimizer'])
    if not all(int(v['step']) == epoch0*100 for v in optimizer.state.values()):
        raise RuntimeError('Adam step mismatch')
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v) and not bool(torch.isfinite(v).all()):
                raise RuntimeError('Nonfinite Adam state')
    best_dice, best_epoch = saved['best_val_dice'], saved['best_epoch']
    best = torch.load(source/'checkpoints/best_dice.pth.tar', map_location='cpu', weights_only=False)
    s.require_protocol(best)
    s.verify_hashes(best['source_sha256'])
    if best['epoch'] != best_epoch or best['best_val_dice'] != best_dice or best['alpha'] != alpha:
        raise RuntimeError('Best-checkpoint lineage mismatch')
    del best
    torch.set_rng_state(saved['rng_state']['torch'])
    torch.cuda.set_rng_state_all(saved['rng_state']['cuda'])
    random.setstate(saved['rng_state']['python'])
    np.random.set_state(saved['rng_state']['numpy'])
    evidence = dict(source=str(source), source_latest_sha256=sha256(source/'checkpoints/latest.pth.tar'),
                    source_best_sha256=sha256(source/'checkpoints/best_dice.pth.tar'),
                    resumed_epoch=epoch0, next_epoch=epoch0+1, adam_updates=epoch0*100,
                    pair_stream_sha256=pair_digest.hexdigest(), cpu_python_numpy_rng_exact=True,
                    cuda_rng_restored=True, old_elapsed_seconds=old_epochs[-1]['elapsed_seconds'],
                    bitwise_training_reproducibility=False)
    s.write_json(args.out/'recovery.json', evidence)
    if args.validate_only:
        (args.out/'RECOVERY_PASSED').write_text('Checkpoint, Adam, exact iterator and RNG recovery verified\n')
        return
    total_started = time.monotonic()-old_epochs[-1]['elapsed_seconds']
    checkpoint_dir = args.out / 'checkpoints'
    checkpoint_dir.mkdir()
    shutil.copyfile(source/'checkpoints/best_dice.pth.tar', checkpoint_dir/'best_dice.pth.tar')
    for name, records in [('pairs.jsonl', old_pairs), ('epochs.jsonl', old_epochs)]:
        (args.out/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
    del saved
    s.write_json(args.out / 'initialization.json', {'state_sha256': initial_hash, 'seed': 42,
                                                   'parameters': 2854079, 'alpha': alpha})
    with (args.out / 'pairs.jsonl').open('a') as pair_log, (args.out / 'epochs.jsonl').open('a') as epoch_log:
        for epoch in range(epoch0+1, 201):
            s.verify_hashes(gate['source_sha256'])
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = []
            for pair_index in range(50):
                batch, iterator = s.next_pair(iterator, loader)
                moving, fixed, moving_id, fixed_id = batch
                moving, fixed = moving.cuda(), fixed.cuda()
                direction_rows = s.train_pair(models, model, spatial, moving, fixed, sim, reg, optimizer, alpha)
                numbers = [float(np.mean([r[k] for r in direction_rows])) for k in ('loss', 'similarity', 'diffusion')]
                values.append(numbers)
                pair_digest.update(f'{moving_id[0]}>{fixed_id[0]}\n'.encode())
                record = dict(epoch=epoch, pair_index=pair_index, moving_id=moving_id[0], fixed_id=fixed_id[0],
                              loss=numbers[0], similarity=numbers[1], diffusion=numbers[2],
                              weighted_diffusion=alpha * numbers[2], directions=direction_rows)
                pair_log.write(json.dumps(record) + '\n')
                pair_log.flush()
                if pair_index % 10 == 0:
                    s.write_json(args.out / 'progress.json', {'status': 'training', **record,
                                 'adam_updates': (epoch - 1) * 100 + 2 * (pair_index + 1),
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
                       elapsed_seconds=time.monotonic() - total_started, adam_updates=epoch * 100,
                       pair_stream_sha256=pair_digest.hexdigest(),
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
            epoch_log.write(json.dumps(row) + '\n')
            epoch_log.flush()
            s.write_json(args.out / 'progress.json', {'status': 'epoch_complete', **row})
            print(json.dumps(row), flush=True)
    assert all(int(v['step'].item()) == 20000 for v in optimizer.state.values())
    s.verify_hashes(gate['source_sha256'])
    s.write_json(args.out / 'training_summary.json', row)
    (args.out / 'TRAINING_COMPLETED').write_text('200 epochs;20000 sequential directional Adam updates\n')
    del optimizer, payload
    torch.cuda.empty_cache()
    summary = first10(args, modules, model, spatial, checkpoint_dir / 'best_dice.pth.tar')
    s.write_json(args.out / 'progress.json', {'status': 'complete', 'training': row, 'first10': summary})
    (args.out / 'COMPLETED').write_text('training and canonical first10 evaluation completed\n')


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--resume-from', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    extra, rest = p.parse_known_args()
    args = adapter.parse_args(rest)
    args.resume_from, args.validate_only = extra.resume_from, extra.validate_only
    if args.mode != 'train' or not (extra.resume_from/'FAILED.txt').is_file():
        raise RuntimeError('Recovery requires the preserved failed training directory')
    if args.out.resolve().is_relative_to(extra.resume_from.resolve()):
        raise RuntimeError('Never overwrite failed source artifacts')
    writer = s.write_json
    def write(path, value):
        if Path(path).name == 'manifest.json':
            value = {**value, 'fresh_initialization': False, 'continuation_from': str(extra.resume_from),
                     'source_snapshot': __import__('os').environ['OASIS_SOURCE_SNAPSHOT']}
        writer(path, value)
    s.write_json = write
    legacy.train = train
    adapter.main(args)

if __name__ == '__main__':
    main()

