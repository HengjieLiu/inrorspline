"""Native FeDDA contracts; shares only read-only utilities with the SITReg sweep."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from sitreg_training import (LAMBDAS, SHAPE, DATA_SHA, write_json, gpu_rows, native_modules,
                             optimizer_for, next_pair, state_hash, verify_hashes,
                             atomic_checkpoint, check_dataset, canonical_case, sha256)

CHECKPOINT_SHA = 'fa9ba93e5ea9296502f1e144accf715738fe9602cfeee32a41a81b1393a5c01e'
MODEL_CONFIG_REL = Path('scripts/250218_oasis_v1/myconfigs/exp_midl_0407/fedda_1mf')
MIN_TRAIN_FREE_GIB = 24
# FeDDA native benchmark-mode fresh processes varied up to0.05273voxel max /
# 0.0003621voxel RMS against the original saved field. Accuracy gates remain
# tight; documented diagnosis is in P51 technical_log.md, not a precision change.
FIELD_MAX_ATOL = 0.1
FIELD_RMS_ATOL = 0.001


def memory_fraction(row, *, min_free_gib=MIN_TRAIN_FREE_GIB):
    if min_free_gib < MIN_TRAIN_FREE_GIB:
        raise ValueError('Cannot lower the FeDDA memory gate')
    if row['free_mib'] < min_free_gib * 1024:
        raise RuntimeError(f'Insufficient FeDDA training headroom: {row}')
    if row['free_mib'] <= 1024:
        raise RuntimeError('No memory remains after CUDA-context reservation')
    return (row['free_mib'] - 1024) / row['total_mib']


def select_gpu(index, *, min_free_gib=MIN_TRAIN_FREE_GIB):
    rows = gpu_rows()
    row = next(r for r in rows if r['index'] == index)
    if index not in (0, 1, 2):
        raise ValueError('Only approved FeDDA GPU slots0/1/2')
    fraction = memory_fraction(row, min_free_gib=min_free_gib)
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
    import utils_models_v2v2_fedd as models
    modules = (models, *modules[1:])
    _, _, losses, seed, warp, _, reg, tm_utils = modules
    seed.my_set_seed(42)
    model = models.get_models_v2(SHAPE, str(root / MODEL_CONFIG_REL),
                                 model_type='fedd', bidir=False, use_batch_parallel=False).cuda()
    assert sum(p.numel() for p in model.parameters()) == 2854079
    spatial = warp.SpatialTransformer(SHAPE).cuda()
    nearest = tm_utils.register_model(SHAPE, 'nearest').cuda()
    sim, regularizer = losses.NCC_vfa(window_size=9, keep_sign=False), reg.Grad3d(penalty='l2')
    return modules, model, spatial, nearest, sim, regularizer


def predict(models, model, spatial, moving, fixed):
    return models.model_predict_v2(moving, fixed, model, bidir=False)


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
            raise RuntimeError('Nonfinite FeDDA training loss')
        optimizer.zero_grad()
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError('Nonfinite or missing FeDDA training gradients')
        optimizer.step()
        records.append({'direction': direction, 'loss': total.item(),
                        'similarity': similarity.item(), 'diffusion': diffusion.item(),
                        'weighted_diffusion': alpha * diffusion.item()})
        del outputs, total, similarity, diffusion, grads
    return records


def source_hashes(root):
    here = Path(__file__).resolve().parent
    files = {here / name for name in (
        'fedda_training.py', 'run_oasis_fedda_training.py', 'launch_oasis_fedda_training.py',
        'run_oasis_fedda_first100_eval.py', 'run_oasis_sitreg_first100_eval.py',
        'sitreg_training.py', 'smoke_dl_training_memory.py', 'common.py',
        'evaluate_saved_baselines_first10.py', 'report_oasis_fedda_training.py')}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(root) in name and Path(name).is_file():
            files.add(Path(name))
    files.update(root / name for name in (
        str(MODEL_CONFIG_REL / 'shared_encoder_config.yaml'),
        str(MODEL_CONFIG_REL / 'a_conv_1mf_decoder_config.yaml'),
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/losses.py',
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/utils.py'))
    return {str(p.resolve()): sha256(p) for p in sorted(files)}
