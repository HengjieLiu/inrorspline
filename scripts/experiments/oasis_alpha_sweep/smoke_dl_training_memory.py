"""Bounded native VFA/SITReg training-memory probe, not a sweep launcher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gpu_snapshot():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['vfa', 'sitreg'], required=True)
    parser.add_argument('--wavereg-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=[3], default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / 'stdout.log').open('x')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(json.dumps({'argv': sys.argv, 'started_unix': time.time()}), flush=True)
    before = gpu_snapshot()
    gpu_row = next(r for r in before.splitlines() if int(r.split(',')[0]) == args.gpu)
    if float(gpu_row.split(',')[5]) < 43000:
        raise RuntimeError('GPU3 has less than43000MiB free; will not alter other jobs.')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_row.split(',')[1].strip()
    sys.path.insert(0, str(args.wavereg_root / 'code'))
    import torch
    import utils_models_v2v2_baseline as native_models
    import utils_datasets_my as native_data
    import utils_rand_seed
    import utils_losses_ncc
    import utils_warp

    tm_root = args.wavereg_root / 'TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph'
    tm_losses = load_module('smoke_tm_losses', tm_root / 'losses.py')
    tm_utils = load_module('smoke_tm_utils', tm_root / 'utils.py')
    utils_rand_seed.my_set_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    shape, gi = (160, 224, 192), 1024 ** 3
    checkpoint_hash = sha256(args.checkpoint)
    manifest = {
        'scope': 'memory_smoke_only_no_production_training', 'model': args.model,
        'physical_gpu': args.gpu, 'visible_device': 'cuda:0',
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'gpu_before': before, 'uid': os.getuid(), 'gid': os.getgid(),
        'seed': 42, 'dtype': 'float32', 'batch_size': 1,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'cudnn_benchmark': True,
        'cudnn_deterministic': False, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'cpu_threads': torch.get_num_threads(),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'dataset_manifest': str(args.data_root / 'dataset.json'),
        'dataset_sha256': sha256(args.data_root / 'dataset.json'),
        'optimizer': {'name': 'Adam', 'lr': 1e-4, 'amsgrad': True, 'weight_decay': 0,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'loss': 'NCC_vfa(window9)+1*Grad3d(l2)', 'pairs_per_phase': 3,
        'phases': ['fresh_seed42', 'trained_lambda1'], 'spatial_shape': list(shape),
        'git_revisions': {},
    }
    for label, root in [('inrorspline', Path(__file__).resolve().parents[3]), ('wavereg', args.wavereg_root)]:
        manifest['git_revisions'][label] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dump(args.output / 'manifest.json', manifest)
    samples, stop = [], threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                row = next(r for r in gpu_snapshot().splitlines() if int(r.split(',')[0]) == args.gpu)
                samples.append({'time': time.time(), 'used_mib': float(row.split(',')[4]),
                                'free_mib': float(row.split(',')[5]), 'utilization': float(row.split(',')[6])})
            except Exception as exc:
                samples.append({'time': time.time(), 'error': repr(exc)})
            stop.wait(0.2)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        model_key = {'vfa': 'bs3_vfa_v0', 'sitreg': 'bs4_sitreg_v0'}[args.model]
        model = native_models.get_models_v2(shape, model_key).cuda()
        fresh = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.load_state_dict(fresh, strict=True)
        manifest['strict_checkpoint_load'] = True
        manifest['parameters'] = sum(p.numel() for p in model.parameters())
        assert manifest['parameters'] == {'vfa': 2014827, 'sitreg': 15076110}[args.model]
        # Both grid buffers are kept resident by the native trainer for both models.
        spatial_trans = utils_warp.SpatialTransformer(shape).cuda()
        nearest_validation = tm_utils.register_model(shape, 'nearest').cuda()
        sim_fn, reg_fn = utils_losses_ncc.NCC_vfa(), tm_losses.Grad3d(penalty='l2')
        train_set = native_data.myDataset_OASIS_v1_json(
            str(args.data_root), str(args.data_root / 'dataset.json'),
            stage='train', with_seg=False, with_id=True, INFO=False)
        assert len(train_set) == 300
        loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                                            num_workers=0, pin_memory=True)
        iterator = iter(loader)
        history, summaries = [], []
        for phase in manifest['phases']:
            if phase == 'trained_lambda1':
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            phase_rows = []
            for pair_index in range(3):
                moving_cpu, fixed_cpu, moving_id, fixed_id = next(iterator)
                moving, fixed = moving_cpu.cuda(), fixed_cpu.cuda()
                assert tuple(moving.shape) == (1, 1, *shape)
                assert torch.isfinite(moving).all() and torch.isfinite(fixed).all()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                directions = [('forward', moving, fixed), ('reverse', fixed, moving)] if args.model == 'vfa' else [('joint', moving, fixed)]
                for direction, src, tgt in directions:
                    if args.model == 'vfa':
                        deformed, dvf = native_models.model_predict_v2(src, tgt, model, model_key)
                        sim, reg = sim_fn(tgt, deformed), reg_fn(dvf, 0)
                    else:
                        deformed1, deformed2, dvf12, dvf21 = native_models.model_predict_v2(
                            src, tgt, model, model_key, spatial_trans)
                        sim = 0.5 * (sim_fn(tgt, deformed1) + sim_fn(src, deformed2))
                        reg = 0.5 * (reg_fn(dvf12, 0) + reg_fn(dvf21, 0))
                    loss = sim + reg
                    optimizer.zero_grad()
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                    assert bool(torch.isfinite(loss))
                    optimizer.step()
                    history.append({'phase': phase, 'pair_index': pair_index, 'direction': direction,
                                    'moving': moving_id[0], 'fixed': fixed_id[0],
                                    'loss': loss.item(), 'similarity': sim.item(), 'diffusion': reg.item(),
                                    'finite_gradients': True, 'gradient_tensor_count': len(grads)})
                    del grads
                torch.cuda.synchronize()
                record = {'phase': phase, 'pair_index': pair_index,
                          'seconds': time.perf_counter() - tick,
                          'peak_allocated_gib': torch.cuda.max_memory_allocated() / gi,
                          'peak_reserved_gib': torch.cuda.max_memory_reserved() / gi}
                phase_rows.append(record)
                print(json.dumps(record), flush=True)
                dump(args.output / 'history.json', history)
            state_bytes = sum(v.numel() * v.element_size() for st in optimizer.state.values()
                              for v in st.values() if torch.is_tensor(v) and v.is_cuda)
            summaries.append({'phase': phase, 'pairs': phase_rows,
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / gi,
                              'peak_reserved_gib': torch.cuda.max_memory_reserved() / gi,
                              'warm_pair_seconds_mean': statistics.mean(r['seconds'] for r in phase_rows[1:]),
                              'optimizer_gpu_bytes': state_bytes,
                              'adam_updates': 6 if args.model == 'vfa' else 3})
            del optimizer
        assert sha256(args.checkpoint) == checkpoint_hash
        sources = {str(Path(__file__).resolve()): sha256(__file__)}
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file and str(args.wavereg_root) in file and Path(file).is_file():
                sources[file] = sha256(file)
        for rel in ['vfa/vfa/params/vfa/vfa_ncc.json', 'SITReg/src/scripts/configs/sitreg/lumir/cc_grad_1.0_very_deep_heavy.json']:
            sources[str(args.wavereg_root / rel)] = sha256(args.wavereg_root / rel)
        manifest['source_sha256'] = sources
        dump(args.output / 'manifest.json', manifest)
        result = {'model': args.model, 'status': 'complete', 'phases': summaries,
                  'overall_peak_allocated_gib': max(r['peak_allocated_gib'] for r in summaries),
                  'overall_peak_reserved_gib': max(r['peak_reserved_gib'] for r in summaries),
                  'sampled_gpu_peak_used_gib': max(s.get('used_mib', 0) for s in samples) / 1024,
                  'checkpoint_unchanged': True, 'gpu_after_with_worker': gpu_snapshot()}
        dump(args.output / 'result.json', result)
        (args.output / 'COMPLETED').write_text('bounded smoke completed; no production run\n')
        print(json.dumps(result), flush=True)
    except BaseException:
        (args.output / 'FAILED.txt').write_text(traceback.format_exc())
        raise
    finally:
        stop.set()
        watcher.join(timeout=2)
        dump(args.output / 'gpu_samples.json', samples)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            if output.is_dir() and not (output / 'COMPLETED').exists():
                failure = output / 'FAILED.txt'
                if not failure.exists():
                    failure.write_text(traceback.format_exc())
        raise
