---
doc_type: technical_log
status: complete
---

# GPU-3 OASIS Table-1 Candidates cuDNN Timing Rerun

## Known limitations

- This is a first10 timing/quality diagnostic for the five Table-1 fast
  candidates, not a replacement for the paper's OASIS100 accuracy study.
- Cold-gap differences combine first-use backend work with case-specific
  runtime; each paired comparison uses the same candidate and first10
  protocol.
- The GPU-3 reference is paired within this rerun and is not bitwise
  comparable to historical GPU-2 timings.
- `cudnn_deterministic=True` records the cuDNN backend flag and does not claim
  global PyTorch deterministic-algorithm execution.

## Planned provenance

- Run ID: `oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1`.
- Experiment key: `2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3`.
- Container/image: `py310sitreg_v1_artifact_store` /
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Physical GPU: `3`.
- OASIS root:
  `/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
- Raw output root: `/database/inrorspline/runs/oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1`.
- Review root: `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3`.

## Execution checklist

- [x] Container, image, mounts, OASIS manifest, versions, artifact write
      access, and GPU-3 idleness verified.
- [x] Syntax checks and 165-job dry-run manifest passed.
- [x] All 15 one-step smoke jobs passed with requested/effective flags.
- [x] Full timing completed `165/165`.
- [x] All selected DVFs passed finite float32 normalized-grid XYZ shape checks.
- [x] Separate Dice evaluation completed `150/150`.
- [x] Report records timing boundaries, backend metadata, and Git provenance.

## Results

The full run completed on physical GPU 3 with Git commit
`0d16f8f4fdb6fa252b9a858971a8d2d545aed7b2` and dirty-tree state `true`.
Environment metadata was PyTorch `2.6.0+cu118`, CUDA `11.8`, cuDNN `90100`,
and NVIDIA RTX 6000 Ada. The timing endpoint was
`case_load_complete` through `dvf_file_ready`, excluding evaluation.

| Table-1 candidate | Backend | Cold s | Warm mean +/- SD s | Gap s | Actual steps cold/warm | Peak GB cold/warm | Dice mean +/- SD |
|---|---|---:|---:|---:|---:|---:|---:|
| MR-D-BSCP (8-4-2) | benchmark=False, deterministic=True | 23.80 | 9.13 +/- 0.30 | 14.67 | 450/437.5 | 1.93/1.93 | 0.8061 +/- 0.0417 |
| MR-D-BSCP (8-4-2) | benchmark=False, deterministic=False | 22.48 | 9.25 +/- 0.28 | 13.23 | 450/437.5 | 1.93/1.93 | 0.8061 +/- 0.0417 |
| MR-D-BSCP (8-4-2) | benchmark=True, deterministic=False | 102.52 | 9.38 +/- 0.85 | 93.14 | 450/437.5 | 3.86/1.94 | 0.8061 +/- 0.0417 |
| D-BSCP (cps2) | benchmark=False, deterministic=True | 23.19 | 13.16 +/- 1.30 | 10.03 | 375/400.0 | 1.84/1.84 | 0.8023 +/- 0.0370 |
| D-BSCP (cps2) | benchmark=False, deterministic=False | 23.13 | 13.23 +/- 1.30 | 9.89 | 375/400.0 | 1.84/1.84 | 0.8023 +/- 0.0370 |
| D-BSCP (cps2) | benchmark=True, deterministic=False | 72.95 | 17.58 +/- 2.62 | 55.36 | 375/400.0 | 2.09/1.84 | 0.8023 +/- 0.0370 |
| D-BSCP (cps4) | benchmark=False, deterministic=True | 30.92 | 20.86 +/- 2.20 | 10.07 | 510/476.9 | 1.50/1.50 | 0.7946 +/- 0.0401 |
| D-BSCP (cps4) | benchmark=False, deterministic=False | 32.70 | 20.85 +/- 2.12 | 11.85 | 510/476.9 | 1.50/1.50 | 0.7946 +/- 0.0401 |
| D-BSCP (cps4) | benchmark=True, deterministic=False | 72.80 | 20.61 +/- 2.28 | 52.19 | 510/476.9 | 1.75/1.50 | 0.7946 +/- 0.0401 |
| INR-BSCP (cps2) | benchmark=False, deterministic=True | 148.45 | 97.62 +/- 17.45 | 50.83 | 968/820.8 | 7.63/7.63 | 0.7920 +/- 0.0369 |
| INR-BSCP (cps2) | benchmark=False, deterministic=False | 123.79 | 91.37 +/- 10.88 | 32.42 | 968/820.8 | 7.63/7.63 | 0.7920 +/- 0.0369 |
| INR-BSCP (cps2) | benchmark=True, deterministic=False | 177.71 | 91.34 +/- 10.92 | 86.37 | 968/820.8 | 7.63/7.63 | 0.7920 +/- 0.0369 |
| INR-BSCP (cps4) | benchmark=False, deterministic=True | 41.07 | 32.00 +/- 5.85 | 9.07 | 605/671.2 | 2.25/2.25 | 0.7776 +/- 0.0464 |
| INR-BSCP (cps4) | benchmark=False, deterministic=False | 38.68 | 30.00 +/- 5.36 | 8.68 | 605/671.2 | 2.25/2.25 | 0.7776 +/- 0.0464 |
| INR-BSCP (cps4) | benchmark=True, deterministic=False | 98.67 | 32.29 +/- 5.92 | 66.38 | 605/671.2 | 2.50/2.25 | 0.7776 +/- 0.0464 |

Relative to each candidate's paired benchmark-enabled reference, the two
benchmark-disabled arms changed cold/warm/gap by, respectively:

- MR-D-BSCP (8-4-2): `-78.72 s (-76.8%)`, `-0.25 s (-2.7%)`, and
  `-78.47 s (-84.2%)` for deterministic=True; `-80.05 s (-78.1%)`,
  `-0.13 s (-1.4%)`, and `-79.91 s (-85.8%)` for deterministic=False.
- D-BSCP (cps2): `-49.76 s (-68.2%)`, `-4.43 s (-25.2%)`, and
  `-45.33 s (-81.9%)` for deterministic=True; `-49.82 s (-68.3%)`,
  `-4.35 s (-24.7%)`, and `-45.47 s (-82.1%)` for deterministic=False.
- D-BSCP (cps4): `-41.87 s (-57.5%)`, `+0.25 s (+1.2%)`, and
  `-42.12 s (-80.7%)` for deterministic=True; `-40.10 s (-55.1%)`,
  `+0.24 s (+1.2%)`, and `-40.34 s (-77.3%)` for deterministic=False.
- INR-BSCP (cps2): `-29.26 s (-16.5%)`, `+6.28 s (+6.9%)`, and
  `-35.54 s (-41.1%)` for deterministic=True; `-53.92 s (-30.3%)`,
  `+0.03 s (+0.0%)`, and `-53.95 s (-62.5%)` for deterministic=False.
- INR-BSCP (cps4): `-57.60 s (-58.4%)`, `-0.29 s (-0.9%)`, and
  `-57.31 s (-86.3%)` for deterministic=True; `-59.98 s (-60.8%)`,
  `-2.29 s (-7.1%)`, and `-57.69 s (-86.9%)` for deterministic=False.

The complete report and machine-readable outputs remain external at
`/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3`.
The camera-ready Table 1 was not modified.
