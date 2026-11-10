"""Runtime-only continuation of the frozen P81 scientific implementation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent / 'oasis_polyaffine_fieldfit'
sys.path.insert(0, str(SCIENCE))
from runtime import atomic_json, read_json, sha, now, code_hashes, assert_external


def scheduler_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def validate_revision(root, revision):
    if sha(root/'launch_manifest.json') != revision['base_launch_sha256']:
        raise RuntimeError('Original launch manifest changed')
    if code_hashes() != revision['scientific_source_hashes']:
        raise RuntimeError('Frozen scientific source changed')
    if scheduler_hashes() != revision['scheduler_source_hashes']:
        raise RuntimeError('Scheduler source changed')
    if sha(root/'input_manifest.json') != revision['input_manifest_sha256']:
        raise RuntimeError('Input manifest changed')
    for receipt in revision.get('validation_receipts', []):
        if sha(receipt['path']) != receipt['sha256'] or read_json(receipt['path']).get('passed') is not True:
            raise RuntimeError('Scheduler validation receipt changed or failed')
    pointer = root/'ACTIVE_SCHEDULER.json'
    if pointer.exists():
        active = read_json(pointer)
        if sha(active['path']) != active['sha256'] or read_json(active['path']) != revision:
            raise RuntimeError('Active scheduler revision binding changed')


def cuda_oom(reason):
    message = str(reason).lower()
    return ('cuda' in message and ('out of memory' in message or 'outofmemoryerror' in message)) or 'cublas_status_alloc_failed' in message
