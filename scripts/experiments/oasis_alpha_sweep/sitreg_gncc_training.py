"""Process-local GNCC contracts for the immutable native SITReg loops.

No historical module/source is edited. Workers explicitly bind their private
legacy training/evaluation module's `s` dependency to this adapter.
"""
from __future__ import annotations

import os
from pathlib import Path
import sitreg_training as native

VARIANT = 'oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1'
EVAL_VARIANT = 'oasis_seed3_first100_sitreg_gncc_training_lambda_eval_v1'
LAMBDAS = {'0.1': .1, '1over3': 1 / 3, '1': 1., '3': 3.}
PROTOCOL = {
    'loss_key': 'global_ncc_signed_population', 'loss': '1-rho',
    'correlation_support': 'full_volume_per_image', 'similarity_scale': 1.,
    'correlation_moments': 'centered_population', 'backward': 'ordinary_autograd',
    'variance_product_epsilon': 1.1920928955078125e-7,
    'regularizer': 'native_Grad3d_l2_total_voxel_field_unmasked',
    'direction_reduction': 'mean_of_forward_and_backward_for_both_terms',
    'updates_per_pair': 1, 'seed': 42, 'epochs': 200, 'pairs_per_epoch': 50,
    'optimizer': 'Adam_AMSGrad', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
    'weight_decay': 0., 'scheduler': None, 'early_stopping': False,
    'checkpoint_selection': 'highest_native_30_pair_validation_dice',
    'batch_size': 1, 'precision': 'FP32_no_AMP',
    'backend': {'benchmark': True, 'deterministic': False, 'cudnn_tf32': True, 'matmul_tf32': False},
    'preprocessing': 'native_stored_intensities_LIA_to_RAS_no_extra_normalization',
    'image_sampling': 'native_border_align_corners_true',
    'nondeterminism': 'historical_backend_and_inversion_kernels_not_bitwise_reproducible',
}
EXPECTED_PAIRS = {}


def __getattr__(name):
    return getattr(native, name)


class GlobalNCC:
    """Stateless: neither image nor its moments survive across minibatches."""
    def __call__(self, a, b):
        if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 5 or a.shape[1] != 1:
            raise ValueError('Require equal [batch,1,D,H,W] floating tensors')
        axes = (1, 2, 3, 4)
        ac, bc = a-a.mean(axes, keepdim=True), b-b.mean(axes, keepdim=True)
        cov = (ac*bc).mean(axes)
        denom = (ac.square().mean(axes)*bc.square().mean(axes)).clamp_min(
            PROTOCOL['variance_product_epsilon']).sqrt()
        return (1-cov/denom).mean()


def run_name(key):
    if key not in LAMBDAS:
        raise ValueError('Unapproved GNCC lambda key')
    return f'sitreg_gncc_lambda_{key}_seed42'


def setup_native(root):
    modules, model, spatial, nearest, _, regularizer = native.setup_native(root)
    return modules, model, spatial, nearest, GlobalNCC(), regularizer


def adapter_sources():
    here = Path(__file__).resolve().parent
    paths = [here/name for name in ('sitreg_gncc_training.py', 'run_oasis_sitreg_gncc_training.py',
             'run_oasis_sitreg_gncc_eval.py', 'launch_oasis_sitreg_gncc.py',
             'run_oasis_sitreg_training.py', 'run_oasis_sitreg_first100_eval.py',
             'queue_oasis_sitreg_first100_eval.py', 'sitreg_training.py',
             'smoke_dl_training_memory.py', 'common.py', 'evaluate_saved_baselines_first10.py')]
    paths.append(here.parents[1]/'reports/oasis_sitreg_gncc_report.py')
    # Historical isolated pystrum remains read-only; fingerprint it when supplied.
    for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if entry and (Path(entry)/'pystrum').is_dir():
            paths.extend((Path(entry)/'pystrum').rglob('*.py'))
            paths.extend(Path(entry).glob('pystrum-*.dist-info/METADATA'))
    return {str(p): native.sha256(p) for p in paths}


def source_hashes(root):
    return {**native.source_hashes(root), **adapter_sources()}


def check_pair_hash(value):
    if EXPECTED_PAIRS and 'pair_stream_sha256' in value and 'epoch' in value:
        if EXPECTED_PAIRS[str(value['epoch'])] != value['pair_stream_sha256']:
            raise RuntimeError('Training pair sequence diverged from the LNCC control')


def write_json(path, value):
    if isinstance(value, dict):
        check_pair_hash(value)
        if Path(path).name in ('manifest.json', 'training_summary.json', 'summary.json',
                              'preflight.json', 'initialization.json'):
            value = {**value, 'gncc_variant': VARIANT, 'loss_protocol': PROTOCOL}
            if Path(path).name == 'manifest.json':
                value['runtime_pythonpath'] = os.environ.get('PYTHONPATH', '')
    native.write_json(path, value)


def atomic_checkpoint(path, state):
    check_pair_hash(state)
    return native.atomic_checkpoint(path, {**state, 'gncc_variant': VARIANT, 'loss_protocol': PROTOCOL})


def require_protocol(value):
    if value.get('gncc_variant') != VARIANT or value.get('loss_protocol') != PROTOCOL:
        raise RuntimeError('Wrong GNCC protocol/checkpoint identity')


def require_identity():
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError('Require container UID:GID1009:1020')
