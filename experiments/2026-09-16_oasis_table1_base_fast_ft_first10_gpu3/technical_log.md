---
doc_type: technical_log
status: in_progress
---

# OASIS Table-1 Base/Fast F/T Rerun Technical Log

## Known limitations

- This run uses seed-3 shuffled first10 only. Its Dice values are diagnostic
  and must not replace the camera-ready OASIS-100 Dice column.
- Timing is paired within this GPU-3 rerun and is not a claim of cross-GPU
  bitwise comparability with historical GPU-2 timings.
- `cudnn_deterministic=True` records the cuDNN backend flag; it does not claim
  global PyTorch deterministic-algorithm execution.
- Any contaminated attempt is excluded from the primary summary but retained
  with its GPU monitor log and retry provenance.

## Planned provenance

- Run ID: `oasis_table1_base_fast_ft_first10_gpu3_20260916_v1`.
- Experiment key: `2026-09-16_oasis_table1_base_fast_ft_first10_gpu3`.
- Physical GPU: `3`.
- Backend: `benchmark=False, deterministic=True`.
- Container/image: `py310sitreg_v1_artifact_store` /
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Split: seed-3 shuffled `pair_test_200`, ranks `0-9`.
- Timing candidates: five base and five Table-1 fast schedules.

## Execution checklist

- [ ] Container, image, mounts, OASIS manifest, versions, artifact write access,
      and GPU-3 idleness verified.
- [ ] Syntax checks and 110-job dry-run manifest passed.
- [ ] Ten one-step smoke jobs passed sequentially.
- [ ] Clean primary timing completed `110/110`.
- [ ] All accepted DVFs passed finite float32 normalized-grid XYZ shape checks.
- [ ] Separate Dice evaluation completed `100/100`.
- [ ] Report records monitor events, retry attempts, timing boundaries, backend
      metadata, and Git provenance.

## Results

To be filled after the external run completes. No camera-ready PDF or LaTeX
source is modified by this experiment.
