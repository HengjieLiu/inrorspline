"""Validate all11 observations of both historical gamma0 cohort blocks."""
import argparse
from pathlib import Path

import numpy as np

from scripts.experiments.oasis_regularizer_sweep.protocol import PROTOCOL as BASE
from scripts.experiments.oasis_regularizer_sweep.protocol import dump, read, sequence, sha256
from .protocol import BACKENDS

REPO = Path(__file__).resolve().parents[3]


def audit(prior, out, oasis):
    if not read(prior / "ACCEPTED.json")["complete"]:
        raise RuntimeError("Unaccepted historical execution")
    sources = read(prior / "manifest.json")["source_hashes"]
    for name, expected in sources.items():
        if sha256(REPO / name) != expected:
            raise RuntimeError(f"Historical source changed: {name}")
    # Use the same parser as the numerical worker, without loading any image.
    from scripts.experiments.oasis_alpha_sweep import common as c
    receipts = []
    for backend in BACKENDS:
        block_root = prior / "blocks" / "cohort" / f"diffusion_{backend}_w1" / "attempt0"
        for marker in ("register_ACCEPTED.json", "eval_ACCEPTED.json", "EVALUATED.json"):
            if not read(block_root / marker)["complete"]:
                raise RuntimeError("Incomplete historical block")
        block = read(block_root / "block.json")
        if [(r["case_rank"], r["condition"]) for r in block["cases"]] != sequence("cohort"):
            raise RuntimeError("Incorrect historical first10 sequence")
        records = []
        for row in block["cases"]:
            info = c.resolve_case(rank=row["case_rank"], pair_file=c.DEFAULT_PAIR_FILE, oasis_root=oasis)
            if (row["fixed_id"], row["moving_id"]) != (info.fixed_id, info.moving_id):
                raise RuntimeError("Historical pair differs from shuffled split")
            if row["protocol"] != BASE or row["weight"] != 1 or row["arm"] != f"diffusion_{backend}":
                raise RuntimeError("Historical numerical protocol differs")
            if row["source_hashes"] != sources:
                raise RuntimeError("Historical source receipt differs")
            for key in ("dvf", "model"):
                if sha256(row[key + "_path"]) != row[key + "_sha256"]:
                    raise RuntimeError("Historical field/model changed")
            for name, expected in row["input_hashes"].items():
                if sha256(oasis / "test" / name) != expected:
                    raise RuntimeError("Historical raw input changed")
            field = np.load(row["dvf_path"], mmap_mode="r")
            if field.shape != (160, 224, 192, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
                raise RuntimeError("Invalid historical field")
            path = Path(row["dvf_path"]).parent / "evaluation.json"
            metrics = read(path)
            if metrics["dvf_sha256"] != row["dvf_sha256"] or metrics["input_hashes"] != row["input_hashes"]:
                raise RuntimeError("Historical evaluation receipt differs")
            if row["backend"]["cudnn_benchmark"] or not row["backend"]["cudnn_deterministic"]:
                raise RuntimeError("Historical backend differs")
            records.append(dict(case=row, metrics=metrics, metric_path=str(path), metric_sha256=sha256(path)))
        receipt = dict(complete=True, gamma=0., lambda_diffusion=1., diffusion_backend=backend,
                       kl_backend="analytic", source_execution=str(prior), source_block=str(block_root),
                       cases=records, provenance="reused_accepted_P84_first10; historical timing/memory")
        dump(out / f"{backend}.json", receipt)
        receipts.append(receipt)
    dump(out / "ACCEPTED.json", dict(complete=True, reported_observations=22, primary_warm_cases=20,
                                     source_hashes=sources, backends=list(BACKENDS)))
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oasis-root", type=Path, required=True)
    args = parser.parse_args()
    print([(r["diffusion_backend"], len(r["cases"])) for r in audit(args.prior_root, args.out, args.oasis_root)])
