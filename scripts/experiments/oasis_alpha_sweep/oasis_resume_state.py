"""Reconstruct the native persistent shuffled iterator without image I/O."""
import hashlib
import json
from pathlib import Path
import random


class ReplayDataset:
    def __init__(self, native, lightweight):
        self.native, self.lightweight, self.replaying = native, lightweight, True

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        return (self.lightweight if self.replaying else self.native)[index]


def restore_iterator(data, data_root, epoch, expected, checkpoint_rng):
    """Call immediately after fresh seed42 native model construction.

    Replay native RandomSampler, Python-random fixed pairing and validation
    iterator seeds. The sampler generator remains alive at its exact position.
    Check both CPU and Python RNG equality before restoring CUDA RNG separately.
    """
    import numpy as np
    import torch
    kwargs = dict(base_dir=str(data_root), json_path=str(data_root/'dataset.json'),
                  stage='train', with_id=True, INFO=False)
    dataset = ReplayDataset(data.myDataset_OASIS_v1_json(**kwargs),
                            data.myDataset_OASIS_v1_json_UnitTest(**kwargs))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # The native validation loader consumes a base seed at each epoch, but its
    # non-random dataset and sequential sampler consume no other RNG.
    val_seed_loader = torch.utils.data.DataLoader(range(30), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    digest = hashlib.sha256()
    for e in range(1, epoch+1):
        for _ in range(50):
            try:
                moving, fixed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                moving, fixed = next(iterator)
            digest.update(f'{moving[0]}>{fixed[0]}\n'.encode())
        if digest.hexdigest() != expected[str(e)]:
            raise RuntimeError(f'Replayed historical pair sequence mismatch at epoch {e}')
        for _ in val_seed_loader:
            pass
    if not torch.equal(torch.get_rng_state(), checkpoint_rng['torch']):
        raise RuntimeError('Replayed CPU RNG differs from saved checkpoint')
    if random.getstate() != checkpoint_rng['python']:
        raise RuntimeError('Replayed Python pairing RNG differs from saved checkpoint')
    actual, saved = np.random.get_state(), checkpoint_rng['numpy']
    if actual[0] != saved[0] or not np.array_equal(actual[1], saved[1]) or actual[2:] != saved[2:]:
        raise RuntimeError('Replayed NumPy RNG differs from saved checkpoint')
    dataset.replaying = False
    return loader, iterator, digest


def accepted_logs(source, epoch, checkpoint, expected):
    """Exclude any partial post-checkpoint epoch, retaining original failed log."""
    epochs = [json.loads(x) for x in (source/'epochs.jsonl').read_text().splitlines() if x.strip()]
    pairs = [json.loads(x) for x in (source/'pairs.jsonl').read_text().splitlines() if x.strip()]
    epochs, pairs = [r for r in epochs if r['epoch'] <= epoch], [r for r in pairs if r['epoch'] <= epoch]
    if [r['epoch'] for r in epochs] != list(range(1, epoch+1)) or len(pairs) != 50*epoch:
        raise RuntimeError('Incomplete saved checkpoint history')
    digest = hashlib.sha256()
    for i, row in enumerate(pairs):
        if (row['epoch'], row['pair_index']) != (i//50+1, i%50):
            raise RuntimeError('Unexpected pair history ordering')
        digest.update(f"{row['moving_id']}>{row['fixed_id']}\n".encode())
        if (i+1) % 50 == 0 and digest.hexdigest() != expected[str(row['epoch'])]:
            raise RuntimeError('Saved pair history hash mismatch')
    if digest.hexdigest() != checkpoint['pair_stream_sha256'] or epochs[-1]['adam_updates'] != epoch*100:
        raise RuntimeError('Checkpoint and saved history disagree')
    return epochs, pairs
