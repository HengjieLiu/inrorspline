"""Audited native SITReg training contracts; imports are GPU-selection safe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from smoke_dl_training_memory import sha256, load_module

LAMBDAS = {'0.1': 0.1, '1over3': 1 / 3, '3': 3.0, '10': 10.0}
SHAPE = (160, 224, 192)
DATA_SHA = '1e358ffc565c651e381a7eacb5b4e6d139b9a4a899c5819d26a58644b5c78d7a'
CHECKPOINT_SHA = '34ffbef1351f7f34645fed11cd4934a672b2781c564c8a1eccd870bca589001c'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def gpu_rows():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.strip().splitlines():
        i, uuid, name, total, used, free, util = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(i), uuid=uuid, name=name, total_mib=float(total),
                         used_mib=float(used), free_mib=float(free), utilization=float(util)))
    return rows


def memory_fraction(row):
    if row['free_mib'] < 37 * 1024:
        raise RuntimeError(f"Insufficient free GPU memory: {row}")
    return (row['free_mib'] - 1024) / row['total_mib']


def select_gpu(index):
    rows = gpu_rows()
    row = next(r for r in rows if r['index'] == index)
    fraction = memory_fraction(row)
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


def native_modules(root):
    sys.path.insert(0, str(root / 'code'))
    import utils_models_v2v2_baseline as models
    import utils_datasets_my as data
    import utils_losses_ncc as losses
    import utils_rand_seed as seed
    import utils_warp as warp
    import utils_eval as evaluation
    tm = root / 'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph'
    reg = load_module('sitreg_native_tm_losses', tm / 'losses.py')
    tm_utils = load_module('sitreg_native_tm_utils', tm / 'utils.py')
    return models, data, losses, seed, warp, evaluation, reg, tm_utils


def setup_native(root):
    import torch
    modules = native_modules(root)
    models, data, losses, seed, warp, evaluation, reg, tm_utils = modules
    seed.my_set_seed(42)
    model = models.get_models_v2(SHAPE, 'bs4_sitreg_v0').cuda()
    assert sum(p.numel() for p in model.parameters()) == 15076110
    spatial = warp.SpatialTransformer(SHAPE).cuda()
    nearest = tm_utils.register_model(SHAPE, 'nearest').cuda()
    sim, regularizer = losses.NCC_vfa(window_size=9, keep_sign=False), reg.Grad3d(penalty='l2')
    return modules, model, spatial, nearest, sim, regularizer


def joint_loss(outputs, moving, fixed, sim, reg, alpha):
    forward, backward, u, v = outputs
    similarity = (sim(fixed, forward) + sim(moving, backward)) * 0.5
    diffusion = (reg(u, 0) + reg(v, 0)) * 0.5
    return similarity + alpha * diffusion, similarity, diffusion


def predict(models, model, spatial, moving, fixed):
    return models.model_predict_v2(moving, fixed, model, 'bs4_sitreg_v0', spatial)


def optimizer_for(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)


def next_pair(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def source_hashes(root):
    paths = {Path(__file__), Path(__file__).with_name('run_oasis_sitreg_training.py'),
             Path(__file__).with_name('smoke_dl_training_memory.py'),
             Path(__file__).with_name('common.py'),
             Path(__file__).with_name('evaluate_saved_baselines_first10.py')}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(root) in name and Path(name).is_file():
            paths.add(Path(name))
    paths.update(root / rel for rel in [
        'SITReg/src/scripts/configs/sitreg/lumir/cc_grad_1.0_very_deep_heavy.json',
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/losses.py',
        'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/utils.py'])
    return {str(p.resolve()): sha256(p) for p in sorted(paths)}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f'Source/data drift: {path}')


def atomic_checkpoint(path, state):
    import torch
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def check_dataset(root):
    if sha256(root / 'dataset.json') != DATA_SHA:
        raise RuntimeError('OASIS manifest changed')
    data = json.loads((root / 'dataset.json').read_text())
    assert [len(data[k]) for k in ['training', 'validation', 'test_200']] == [300, 30, 200]
    for key in ['training', 'validation', 'test_200']:
        for item in data[key]:
            for value in item.values():
                if isinstance(value, str) and value.endswith('.nii.gz'):
                    if not (root / value).is_file():
                        raise FileNotFoundError(root / value)


def canonical_case(args, field, rank):
    import common as c
    import torch
    from evaluate_saved_baselines_first10 import voxel_zyx_to_norm_dhw3
    batch = c.load_oasis_case(rank=rank, device=torch.device('cpu'), oasis_root=args.data_root)
    dense = torch.from_numpy(voxel_zyx_to_norm_dhw3(field))
    metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(SHAPE, torch.device('cpu')) + dense, batch)
    row = {k: float(metrics[k]) for k in ['dice_mean', 'hd95_mean', 'asd_symmetric_mean']}
    row.update(c.jacobian_metrics_voxel(field, batch['fixed_mask'].bool().numpy()))
    row['diffusion_l2_full'] = c.diffusion_loss_voxel(field)
    row.update(rank=rank, fixed_id=batch['fixed_id'], moving_id=batch['moving_id'],
               direction='moving_to_fixed', foreground='fixed_raw_image_positive',
               asd_backend=metrics['asd_backend'])
    return row
