# OASIS first-100 pyramid 8→4→2→dense NCC9/diffusion sweep

## Goal

Check whether the first10 pyramid dense-refinement results, especially rank4,
generalize over seed-3 shuffled OASIS ranks0–99. Compare four diffusion weights
against the original best-Dice VFA and SITReg fields with the same canonical
metric code.

## Frozen setup

- Dataset/split: OASIS `pair_test_200`, seed-3 shuffled; first100 rows, ranks0–99.
- Direction: first pair member fixed, second moving; pull-sample moving using
  identity plus displacement.
- Method: cubic B-spline cps8→4→2 prefix, then direct full-resolution dense
  displacement refinement. No cps1 B-spline controls.
- Resolution mode: pyramid downsample factors4/2/1/1.
- Objective: native VFA NCC9, `1 - mean(correlation²)`, plus the same λ times
  unmasked total-field forward finite-difference L2 diffusion at every stage.
- λ:0.1,0.3,1,3. Identical λ at cps8, cps4, cps2 and dense stages.
- Normalized-parameter learning rates0.03/0.01/0.003/0.001; original P43
  adaptive stage schedules, fresh Adam per stage, seed1, FP32, cuDNN
  benchmark off/deterministic on, TF32 disabled.
- Canonical accuracy: Dice, HD95 and symmetric ASD, identical absent-label
  handling; deformation summaries include foreground SD(log J), folding,
  diffusion, stage stopping epochs, and runtime diagnostics.

## Reuse and execution

- Reuse accepted P43 results for ranks0–9, only for the exact pyramid `842`
  prefixes and `842_dense` candidates at λ0.1/0.3/1/3. Validate source hashes,
  protocol, pair identity, stage schedule, completion markers and field/model
  hashes before reuse. These are read-only inputs and remain under the original
  experiment root.
- Run ranks10–99 as 360 new dense candidates plus their 360 matching prefixes.
- Container identity UID:GID1009:1020, image
  `hengjieliu96/dl_dir:py310sitreg_v1` (image ID recorded in launch manifest).
- GPU slots: one worker on physical GPU0, one on GPU2, two on GPU3. Recheck
  current GPU memory before every launch; reserve at least4GiB per worker and
  8GiB for the two GPU3 slots. Do not stop, reprioritize or modify any existing
  process. Preserve failed attempts; retry a CUDA OOM candidate once only after
  a worker finishes and memory is sufficient.
- The prior 10-case dense pyramid run recorded1.638GiB maximum PyTorch tensor
  allocation at the dense stage (cps2 maximum1.464GiB). The new scheduler uses
  a4GiB per-worker free-memory reservation to include allocator/context headroom.

## References and outputs

- Evaluate original saved best-Dice VFA and SITReg fields for ranks0–99 from
  the unchanged inference directories. Reuse accepted canonical reference
  rows for ranks0–9 after source-hash checks; evaluate ranks10–99 with the same
  code and protocol.
- Run outputs:
  `${ARTIFACT_ROOT}/runs/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1/`
- Report outputs:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1/`
- Report 400 candidate rows, 200 VFA/SITReg reference rows, paired per-case
  accuracy differences, mean±sample-SD tables, rank4 results, and a sensitivity
  summary excluding rank4. Aggregate only after all fits, references, finite
  fields and identity checks pass.

## Limitations

This is a first100 OASIS test-cohort comparison after exploratory first10
tuning. The first10 fits are reused from the already-completed accepted sweep,
not rerun. Existing concurrent jobs make runtime unsuitable as a controlled
speed comparison. Deterministic cuDNN settings do not make CUDA grid-sample
backward bitwise deterministic.

## Report-only recovery

After all registrations are complete, resume reference scoring and finalization
without entering the GPU scheduler:

```bash
python -B scripts/reports/finalize_oasis_first100_ncc9_dense.py \
  --run-root "${ARTIFACT_ROOT}/runs/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1" \
  --review-root "${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1" \
  --workers 6
```

Run as the approved container UID:GID1009:1020. Stop any previous scorer for
this report in a coordinated handoff first; do not touch other experiments.
The finalizer's exclusive lock prevents duplicate new finalizers. It preserves
the launch manifest, records correction provenance separately, caches every
case atomically against source/split/evaluator hashes, and refuses stale cache
identities. Spawn workers initialize PyTorch threads exactly once. Serial/spawn
canonical metric parity is required for VFA and SITReg before bulk scoring.

`report_recovery/progress.json` reports newly scored throughput independently
of cache reuse. Full output validation has a separate phase and stat-bound hash
receipts so resumes do not reread unchanged large fields/checkpoints. Acceptance
is written only after complete case coverage, prefix pairing, finite/hash checks
and CSV/JSON/Markdown consistency. Original source hashes are never loosened.
