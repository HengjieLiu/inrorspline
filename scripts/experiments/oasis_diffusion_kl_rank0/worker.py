"""Activate the combined loss after the established cold-worker boot handshake."""
from contextlib import ExitStack
import sys

from scripts.experiments.oasis_regularizer_sweep import worker as base
from scripts.experiments.oasis_regularizer_sweep.protocol import read


def main():
    job=read(sys.argv[sys.argv.index("--job")+1])
    if job["phase"] not in ("smoke","pilot") or job["weight"]!=1.:
        raise ValueError("Only rank0 combined-loss smoke/pilot blocks are permitted")
    runtime=base.runtime
    with ExitStack() as stack:
        def setup(gpu):
            device=runtime(gpu)
            from .model import activate
            stack.enter_context(activate(base,job["gamma"]))
            return device
        base.runtime=setup
        try: base.main()
        finally: base.runtime=runtime


if __name__=="__main__": main()
