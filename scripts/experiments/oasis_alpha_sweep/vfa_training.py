"""Native VFA contracts; shares only read-only utilities with the SITReg sweep."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from sitreg_training import (LAMBDAS, SHAPE, DATA_SHA, write_json, gpu_rows, native_modules,
                             optimizer_for, next_pair, state_hash, verify_hashes,
                             atomic_checkpoint, check_dataset, canonical_case, sha256)

CHECKPOINT_SHA = 'e5353f79a7e0ce0a42eb2c06f2e35bb40d18e0a538d9ae449f418182dc9138d7'
MIN_TRAIN_FREE_GIB = 42


def memory_fraction(row, *, allow_tight=False):
    if not allow_tight and row['free_mib'] < MIN_TRAIN_FREE_GIB * 1024:
        raise RuntimeError(f'Insufficient VFA training headroom: {row}')
    if row['free_mib'] <= 1024:
        raise RuntimeError('No memory remains after CUDA-context reservation')
    return (row['free_mib'] - 1024) / row['total_mib']


def select_gpu(index, *, allow_tight=False):
    rows = gpu_rows()
    row = next(r for r in rows if r['index'] == index)
    if allow_tight and index != 0:
        raise ValueError('Tight-memory override is restricted to GPU0')
    fraction = memory_fraction(row, allow_tight=allow_tight)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = row['uuid']
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return rows, fraction


def setup_native(root):
    modules = native_modules(root)
    models, _, losses, seed, warp, _, reg, tm_utils = modules
    seed.my_set_seed(42)
    model = models.get_models_v2(SHAPE, 'bs3_vfa_v0').cuda()
    assert sum(p.numel() for p in model.parameters()) == 2014827
    spatial = warp.SpatialTransformer(SHAPE).cuda()
    nearest = tm_utils.register_model(SHAPE, 'nearest').cuda()
    sim, regularizer = losses.NCC_vfa(window_size=9, keep_sign=False), reg.Grad3d(penalty='l2')
    return modules, model, spatial, nearest, sim, regularizer


def predict(models, model, spatial, moving, fixed):
    return models.model_predict_v2(moving, fixed, model, 'bs3_vfa_v0', spatial)


def directional_loss(outputs, fixed, sim, reg, alpha):
    warped, dvf = outputs
    similarity, diffusion = sim(fixed, warped), reg(dvf, 0)
    return similarity + alpha * diffusion, similarity, diffusion


def train_pair(models, model, spatial, moving, fixed, sim, reg, optimizer, alpha):
    """Two sequential Adam updates; reverse prediction uses the updated weights."""
    import torch
    records = []
    for direction, src, tgt in [('forward', moving, fixed), ('reverse', fixed, moving)]:
        outputs = predict(models, model, spatial, src, tgt)
        total, similarity, diffusion = directional_loss(outputs, tgt, sim, reg, alpha)
        if not bool(torch.isfinite(total)):
            raise RuntimeError('Nonfinite VFA training loss')
        optimizer.zero_grad()
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError('Nonfinite or missing VFA training gradients')
        optimizer.step()
        records.append({'direction': direction, 'loss': total.item(),
                        'similarity': similarity.item(), 'diffusion': diffusion.item(),
                        'weighted_diffusion': alpha * diffusion.item()})
        del outputs, total, similarity, diffusion, grads
    return records


def source_hashes(root):
    here = Path(__file__).resolve().parent
    files = {here / name for name in (
        'vfa_training.py', 'run_oasis_vfa_training.py', 'launch_oasis_vfa_training.py',
        'run_oasis_vfa_first100_eval.py', 'run_oasis_sitreg_first100_eval.py',
        'sitreg_training.py', 'smoke_dl_training_memory.py', 'common.py',
        'evaluate_saved_baselines_first10.py')}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(root) in name and Path(name).is_file():
            files.add(Path(name))
    files.update(root / name for name in (
        'vfa/vfa/params/vfa/vfa_ncc.json',
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/losses.py',
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/utils.py'))
    return {str(p.resolve()): sha256(p) for p in sorted(files)}
