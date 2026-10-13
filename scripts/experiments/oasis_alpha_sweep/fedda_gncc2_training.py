"""Squared population GNCC with immutable native FeDDA sequential updates."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from collections import deque

import fedda_training as native
import sitreg_gncc_training as sitreg_gncc

VARIANT = 'oasis_fedda_gncc2_diffusion_training_lambda_sweep_v1'
EVAL_VARIANT = 'oasis_seed3_first100_fedda_gncc2_training_lambda_eval_v1'
REPORT_VARIANT = 'oasis_seed3_first100_fedda_three_ncc_v1'
LAMBDAS = {'0.1': .1, '1over3': 1/3, '1': 1., '3': 3.}

PROTOCOL = copy.deepcopy(sitreg_gncc.PROTOCOL)
PROTOCOL.update(loss_key='global_ncc_squared_population', loss='1-rho^2', model_key='fedda_1mf', parameters=2854079, use_batch_parallel=False,
                direction_reduction='two_sequential_updates_forward_then_reverse_with_updated_weights',
                updates_per_pair=2, adam_updates=20000, validation_direction='forward_only',
                nondeterminism='historical_cudnn_benchmark_mode_not_bitwise_reproducible')
EXPECTED_PAIRS = {}


class GlobalNCCSquared:
    """Per-image population correlation; bounded detached diagnostics only."""
    def __init__(self):
        self.recent_rho = deque(maxlen=2)

    def correlation(self, a, b):
        if (a.shape != b.shape or a.dtype != b.dtype or a.ndim != 5 or
                a.shape[1] != 1 or not a.is_floating_point() or not b.is_floating_point()):
            raise ValueError('Require equal floating [batch,1,D,H,W] tensors')
        axes = (1, 2, 3, 4)
        ac, bc = a-a.mean(axes, keepdim=True), b-b.mean(axes, keepdim=True)
        covariance = (ac*bc).mean(axes)
        denominator = (ac.square().mean(axes)*bc.square().mean(axes)).clamp_min(
            PROTOCOL['variance_product_epsilon']).sqrt()
        return covariance/denominator

    def __call__(self, a, b):
        rho = self.correlation(a, b)
        self.recent_rho.append(rho.detach().cpu().tolist())
        return (1-rho.square()).mean()


def train_pair(models, model, spatial, moving, fixed, sim, reg, optimizer, alpha):
    sim.recent_rho.clear()
    rows = native.train_pair(models, model, spatial, moving, fixed, sim, reg, optimizer, alpha)
    if len(sim.recent_rho) != 2:
        raise RuntimeError('Expected one correlation per sequential directional update')
    for row, rho in zip(rows, sim.recent_rho):
        row['signed_rho_per_image'] = rho
    return rows


def __getattr__(name):
    return getattr(native, name)


def require_identity():
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError('Require container UID:GID1009:1020')


def run_name(key):
    if key not in LAMBDAS:
        raise ValueError('Unapproved FeDDA GNCC² lambda')
    return f'fedda_1mf_gncc2_lambda_{key}_seed42'


def select_gpu(index, *, min_free_gib=24):
    if index not in range(4):
        raise ValueError('Approved physical GPUs are0/1/2/3')
    rows = native.gpu_rows()
    device = next(r for r in rows if r['index'] == index)
    fraction = native.memory_fraction(device, min_free_gib=min_free_gib)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = device['uuid']
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
    modules, model, spatial, nearest, _, reg = native.setup_native(root)
    return modules, model, spatial, nearest, GlobalNCCSquared(), reg


def adapter_sources():
    here = Path(__file__).resolve().parent
    files = [here/name for name in ('fedda_gncc2_training.py', 'run_oasis_fedda_gncc2_training.py',
             'run_oasis_fedda_gncc2_eval.py', 'launch_oasis_fedda_gncc2.py', 'fedda_training.py',
             'run_oasis_fedda_training.py', 'run_oasis_fedda_first100_eval.py',
             'sitreg_gncc_training.py', 'sitreg_training.py', 'run_oasis_sitreg_first100_eval.py',
             'common.py', 'smoke_dl_training_memory.py', 'evaluate_saved_baselines_first10.py')]
    files.extend(here.parents[1]/'reports'/name for name in
                 ('oasis_fedda_three_ncc_report.py', 'oasis_sitreg_ncc_arc_lines_top.py',
                  'oasis_ncc_three_regularity_arc.py'))
    for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if entry:
            root = Path(entry).resolve()
            for package in ('pystrum', 'monai', 'yaml'):
                if (root/package/'__init__.py').is_file():
                    files.append(root/package/'__init__.py')
            files.extend(root.glob('*.dist-info/METADATA'))
            # Hash actually imported isolated dependencies, including YAML's
            # extension module, without installing or modifying shared packages.
            for module in list(sys.modules.values()):
                source = getattr(module, '__file__', None)
                if source and Path(source).is_file() and Path(source).resolve().is_relative_to(root):
                    files.append(Path(source).resolve())
    return {str(p): native.sha256(p) for p in files}


def source_hashes(root):
    return {**native.source_hashes(root), **adapter_sources()}


def check_pair_hash(value):
    if EXPECTED_PAIRS and 'pair_stream_sha256' in value and 'epoch' in value:
        if EXPECTED_PAIRS[str(value['epoch'])] != value['pair_stream_sha256']:
            raise RuntimeError('FeDDA training-pair sequence differs from LNCC controls')


def write_json(path, value):
    if isinstance(value, dict):
        check_pair_hash(value)
        if Path(path).name in ('manifest.json', 'preflight.json', 'initialization.json',
                              'training_summary.json', 'summary.json'):
            value = {**value, 'gncc_variant': VARIANT, 'loss_protocol': PROTOCOL}
            if Path(path).name == 'manifest.json':
                value['runtime_pythonpath'] = os.environ.get('PYTHONPATH', '')
    native.write_json(path, value)


def atomic_checkpoint(path, state):
    check_pair_hash(state)
    return native.atomic_checkpoint(path, {**state, 'gncc_variant': VARIANT, 'loss_protocol': PROTOCOL})


def require_protocol(value):
    if value.get('gncc_variant') != VARIANT or value.get('loss_protocol') != PROTOCOL:
        raise RuntimeError('Wrong FeDDA GNCC² identity or loss/update protocol')


def baseline_pair_hashes(root, initial_hash):
    queue = json.loads((root/'queue.json').read_text())
    if queue['status'] != 'complete' or len(queue['jobs']) != 4:
        raise RuntimeError('Require four completed FeDDA LNCC controls')
    expected = None
    if {j['lambda_key'] for j in queue['jobs']} != {'0.1', '1over3', '3', '10'}:
        raise RuntimeError('Wrong baseline lambda matrix')
    for job in queue['jobs']:
        directory = Path(job['train_dir'])
        if job['status'] != 'complete' or not (directory/'COMPLETED').is_file():
            raise RuntimeError('Incomplete historical training')
        initialization = json.loads((directory/'initialization.json').read_text())
        if initialization['state_sha256'] != initial_hash:
            raise RuntimeError('Fresh FeDDA initialization no longer matches LNCC controls')
        epochs = [json.loads(x) for x in (directory/'epochs.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in epochs] != list(range(1, 201)) or epochs[-1]['adam_updates'] != 20000:
            raise RuntimeError('Historical FeDDA budget mismatch')
        hashes = {str(r['epoch']): r['pair_stream_sha256'] for r in epochs}
        if expected is not None and expected != hashes:
            raise RuntimeError('Historical FeDDA pair sequences disagree')
        expected = hashes
    return expected
