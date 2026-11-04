#!/usr/bin/env python
"""P69 protocol binding for the immutable P68 independent eight-block replay."""
import argparse
from pathlib import Path
import audit_oasis_bounded_two_blocks as audit
import run_oasis_two_increments_first10 as cohort


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--gpu",choices=("0","1","2","3"),required=True)
    args=parser.parse_args()
    row=cohort.c.read_json(args.run_dir/"summary.json")
    if row["variant"]!=cohort.VARIANT or row["mode"]!=cohort.MODE: raise RuntimeError("Wrong audit target")
    with cohort.configured(row["stage_alpha"]): audit.main()


if __name__=="__main__": main()
