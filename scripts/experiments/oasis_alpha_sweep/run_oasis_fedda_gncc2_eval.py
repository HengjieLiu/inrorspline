"""GNCC FeDDA first100 evaluation with unchanged native canonical scoring."""
import argparse
import json
from pathlib import Path
import traceback

import fedda_gncc2_training as g
import run_oasis_fedda_first100_eval as legacy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--min-free-gib', type=float, default=24.)
    a = p.parse_args(argv)
    protected = [a.training_dir, Path(__file__).resolve().parents[3]]
    if any(a.out.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(a.out.resolve()) for p in protected):
        p.error('Evaluation output overlaps protected training/source')
    if a.min_free_gib < 24:
        p.error('Cannot lower the24GiB evaluation memory gate')
    return a


def main(args):
    g.require_identity()
    g.require_protocol(json.loads((args.training_dir/'manifest.json').read_text()))
    legacy.s = g
    legacy.VARIANT = g.EVAL_VARIANT
    legacy.evaluation_sources = g.adapter_sources
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        checkpoint = torch.load(args.training_dir/'checkpoints/best_dice.pth.tar', map_location='cpu', weights_only=False)
        g.require_protocol(checkpoint)
        del checkpoint
        legacy.run(args)
    except BaseException:
        (args.out/'FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main(parse_args())
