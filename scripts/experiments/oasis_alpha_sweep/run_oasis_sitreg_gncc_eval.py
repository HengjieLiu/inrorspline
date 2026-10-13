"""Detailed GNCC first100 inference using the immutable canonical evaluator."""
import json
from pathlib import Path
import traceback

import sitreg_gncc_training as g
import run_oasis_sitreg_first100_eval as legacy


def main(args):
    g.require_identity()
    g.require_protocol(json.loads((args.training_dir/'manifest.json').read_text()))
    # Dedicated process-local adapter; historical evaluation code is untouched.
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
    main(legacy.parse_args())
