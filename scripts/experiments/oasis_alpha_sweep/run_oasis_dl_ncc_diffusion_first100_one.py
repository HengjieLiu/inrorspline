#!/usr/bin/env python
"""First-100 adapter for the audited NCC9/diffusion worker."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oasis_ncc9_first100  # noqa: E402
oasis_ncc9_first100.install_first100_identity()

import run_oasis_dl_ncc_diffusion_one as worker  # noqa: E402


if __name__ == "__main__":
    worker.run(worker.parse_args())
