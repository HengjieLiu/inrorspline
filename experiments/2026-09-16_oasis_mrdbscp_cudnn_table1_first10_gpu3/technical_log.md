---
doc_type: technical_log
status: active
---

# GPU-3 MR-D-BSCP Table-1 cuDNN Timing Rerun

## Known limitations

- This is a first10 timing/quality diagnostic for one MR-D-BSCP fast schedule,
  not a replacement for the paper's OASIS-100 accuracy study.
- Cold rank 0 and warm rank 0/ranks 1-9 are distinct process/timing states;
  cold-minus-warm combines first-use overhead with case-specific runtime.
- The GPU-3 reference is paired within this rerun, while the accepted P2y
  reference remains a historical GPU-2 result.
- `cudnn_deterministic=True` refers to the cuDNN backend flag used by this
  runner, not a claim of global PyTorch deterministic-algorithm execution.

## Planned provenance

- Run ID: `oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1`.
- Experiment key: `2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3`.
- Container: `py310sitreg_v1_artifact_store`.
- Image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Physical GPU: `3`.
- Raw output root: `/database/inrorspline/runs/oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1`.
- Review root: `/database/inrorspline/repo_experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3`.

## Execution checklist

- [x] Container was recreated with the read-only shared-storage mount; image,
      artifact mount, OASIS root, and GPU 3 idle checks passed. The previous
      container is preserved as `py310sitreg_v1_artifact_store_rollback_20260916`.
- [x] Syntax checks passed on the controller, evaluator, and focused tests.
- [x] Dry-run contains exactly three configurations and 33 timing jobs.
- [x] One-step smoke passed for all three configurations; all three DVFs were
      finite float32 arrays with shape `(160, 224, 192, 3)` and effective flags
      matched the requested configurations.
- [x] Full timing completed 33/33 with finite endpoint values and valid DVFs.
- [x] Posthoc Dice completed 30/30 unique case/configuration rows.
- [x] Final report records backend metadata, Git hash, and dirty-tree state.

## Results

The OASIS data was mounted read-only from the shared-storage path at
`/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
The accepted timing summary was:

| Configuration | Cold (s) | Warm mean +/- SD (s) | Gap (s) | Expand gap (s) | Backward gap (s) | Dice mean |
|---|---:|---:|---:|---:|---:|---:|
| benchmark=False, deterministic=True | 26.49 | 9.31 +/- 0.31 | 17.19 | 0.08 | 2.47 | 0.8061 |
| benchmark=False, deterministic=False | 22.42 | 9.29 +/- 0.40 | 13.14 | 0.91 | 1.85 | 0.8061 |
| benchmark=True, deterministic=False | 110.20 | 9.23 +/- 0.50 | 100.97 | 48.60 | 35.16 | 0.8061 |

Relative to the paired reference, `benchmark=False, deterministic=True`
changed cold/warm/gap by `-83.70 s (-76.0%)`, `+0.07 s (+0.8%)`, and
`-83.78 s (-83.0%)`; `benchmark=False, deterministic=False` changed them by
`-87.77 s (-79.7%)`, `+0.05 s (+0.6%)`, and `-87.83 s (-87.0%)`.

Launch provenance: Git commit `0d16f8f4fdb6fa252b9a858971a8d2d545aed7b2`,
dirty tree `true`; PyTorch `2.6.0+cu118`, CUDA `11.8`, cuDNN `90100`, and
NVIDIA RTX 6000 Ada GPU 3. Generated timing, Dice, and DVF artifacts remain in
the external artifact store and are not copied into this source-only directory.
