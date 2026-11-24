"""First10 scope adapter around unchanged pilot loss and optimization code."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import sys

from scripts.experiments.oasis_regularizer_sweep import worker as base
from scripts.experiments.oasis_regularizer_sweep.protocol import dump, read
from .protocol import PROTOCOL, validate_job


@contextmanager
def activate(worker, gamma):
    from scripts.experiments.oasis_diffusion_kl_rank0.model import CombinedObjective, optimize
    original = worker.l.Objective, worker.optimize, worker.register, worker.PROTOCOL
    original_register = worker.register

    def objective(*args, **kwargs):
        return CombinedObjective(*args, **kwargs, gamma=gamma)

    def register(job, rank, condition, index, device, backend):
        if rank not in range(10):
            raise ValueError("Only shuffled ranks0..9 are authorized")
        result = original_register(job, rank, condition, index, device, backend)
        result.update(gamma=gamma, lambda_diffusion=1., lambda_kl=gamma,
                      kl_backend="analytic", regularizer="diffusion_plus_analytic_kl_c1")
        dump(Path(result["dvf_path"]).parent / "result.json", result)
        return result

    worker.l.Objective = objective
    worker.optimize = optimize
    worker.register = register
    worker.PROTOCOL = dict(PROTOCOL, gamma=gamma)
    try:
        yield
    finally:
        worker.l.Objective, worker.optimize, worker.register, worker.PROTOCOL = original


def main():
    job = read(sys.argv[sys.argv.index("--job") + 1])
    validate_job(job)
    runtime = base.runtime
    with ExitStack() as stack:
        def setup(gpu):
            device = runtime(gpu)
            stack.enter_context(activate(base, job["gamma"]))
            return device
        base.runtime = setup
        try:
            base.main()
        finally:
            base.runtime = runtime


if __name__ == "__main__":
    main()
