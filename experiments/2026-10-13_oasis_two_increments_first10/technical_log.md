# P69 technical log

## Known limitations

- This is exploratory test-case tuning. Ranks0/4 informed the design; report
  the remaining eight cases separately as well as the requested first10.
- Per-map contraction does not ensure zero canonical/exported-field folding.
- Shared-load runtime is diagnostic; CUDA sampling backward is not guaranteed
  bitwise deterministic even with deterministic cuDNN.
- Only cps2 lambda changes. Both increments retain the same lambda at a level.

## Preflight and scope

- User approved first10 at[1,1,5,1] and[1,1,1,1] for the two-map method.
- Reuse P68's two accepted1151 pilot outputs;18 fresh fits,20 total results.
- Immutable P68 worker/optimizer and audit are bound by a separate scoped
  adapter, so previous numerical files, accepted reports and fields stay intact.
- Live GPUs0/1/3 about47.4GiB free, GPU2 about43.2GiB with existing workload.
  One new worker per GPU, admission rechecked; use1009:1020.
- Full-size rank1 smokes at both lambda vectors and unit/regression gates
  precede production. Complete-only cohort plus independent checkpoint replay.

## Unit and regression verification

- New adapter suite:3 passed (weights/CLI/unique names, scoped binding and
  exception restoration, numerical callable identity, cohort completeness,
  nonpilot subset and sample SD). Historical bounded suites:28 passed.
- Native NCC/diffusion suite:15 passed/1 skipped. Total46 passed/1 skipped.
- Historical worker, objective, optimizer and audit files were not edited.
- Full-size rank1 one-step-per-increment smokes launched sequentially onGPU3;
  production remains gated on both smoke checks.

## Production launch

- Both full-size rank1 eight-increment smokes passed, including native loss
  parity, identity insertion, checkpoint reconstruction, DVF serialization,
  canonical metrics, elementary bounds and off-grid probes.
- Launched the18 fresh registrations acrossGPUs0/1/2/3, one per GPU and
  paired lambda vectors on the same device. Two unchanged pilot fits count
  toward the20-case/configuration cohort. Existing GPU2 workload retained.
- Production is followed automatically by18 independent eight-block replay
  processes; complete report acceptance requires all20 valid audits.
- All18 fresh fits completed successfully, giving20/20 results with the two
  reused pilots. No failures, retries, numerical changes or extra tuning.
- Reran the native-loss suite with DL_NCC_CODE_ROOT and VFA_ROOT explicitly
  configured:16/16 pass, resolving the earlier optional parity-test skip.
  Final unique test coverage is47 passed/0 skipped. Independent replay ongoing.

## Manifest interpretation audit

- As in P68, launch_manifest.settings carries the historical data-loader
  source context. Only oasis_root/pair_file/dl_code_root/vfa_root are consumed
  by this worker. Other inherited fields (old output roots, gpu_slots,
  cpu_threads and stop_after) are not this experiment's effective settings.
- Effective output paths and GPU/visible-device identities are in each new
  run manifest/summary; current GPU admission and P69 queue are recorded
  separately. Actual worker thread count is4, explicitly set by P68.worker
  and OMP_NUM_THREADS/MKL_NUM_THREADS. All lambda/loss/LR/stopping settings
  come from the new explicit protocol and per-block optimizer records, not
  from inherited settings. No numerical or source change was made mid-run.

## Accepted results

- Accepted2026-10-14 06:55:43UTC (Oct13 PDT). All20 results have successful
  independent audits:18 new replays plus2 pilot replays,160 block checkpoints
  total. All40 paired common-prefix fields are byte-identical, with equal
  stopping/best epochs and terminal loss values within the frozen tolerance.
- Final read-only check verified report acceptance hashes, current source
  hashes, immutable references and all20 audit-to-summary hashes. No failures,
  retries, historical output modifications or additional tuning.

| Stage weights | Dice | HD95 mm | Symmetric ASD mm | FG SDlogJ | FG folding % |
|---|---|---|---|---|---|
|1,1,5,1|0.821953 ± 0.037352|1.724650 ± 0.320593|0.358358 ± 0.104480|0.581739 ± 0.030237|0.000381 ± 0.000225|
|1,1,1,1|0.828756 ± 0.031502|1.678458 ± 0.316117|0.338309 ± 0.086565|0.650974 ± 0.037057|0.003143 ± 0.001225|

- All values are first10 case means±sampleSD. Lower cps2 lambda improves
  mean Dice+0.006803, HD95-0.046191mm, ASD-0.020049mm; wins9/10,9/10,10/10
  cases respectively. Exceptions: rank1 Dice and rank6 HD95. Both regularity
  endpoints rise10/10: mean SDlogJ+0.069235, folding+0.002762 percentage points.
- Lambda1111 improves all three mean accuracy metrics over original VFA/SITReg.
  Lambda1151 is more regular: folding0.000381% versus SITReg0.000423%, but
  SDlogJ0.581739 versus0.569473, HD951.724650 versus1.715898mm, ASD0.358358
  versus0.350896mm. Do not equate nearby means with proven equivalence.
- Rank4 remains an accuracy outlier (1111 Dice0.750952 versus SITReg0.768742).
  Overall accuracy SD remains larger than DL despite improved means. The
  separately reported nonpilot8 exclude both design-informing ranks0/4;
  they are diagnostic, not a substitute for the complete first10 comparison.
- All20 retained-chain100k off-grid sets have zero nonpositive Jacobians.
  Exported interpolation has nonzero probes: worst0.011% for1151 and0.045%
  for1111. These probe percentages are distinct from canonical FG folding
  above; bounded elementary maps do not establish a universal export guarantee.
- Mean optimization1151/1111:151.91/153.56s; complete worker230.65/225.40s;
  peak3.458GiB. Existing workloads remained active; times are diagnostic.

## Outputs and reproduction

External run/report key:
`oasis_seed3_first10_mrdbscp8421_tanh_two_increments_lambda1151_1111_v1`.
Reports include per_case.csv (60 comparison rows), summary.csv (12 summary
rows), paired_lambda.csv (10), blocks.csv (160), candidate_diagnostics.csv (20),
second_increment_changes.csv (80), results.json, report.md and ACCEPTED.json.
All model states, histories, serialized fields and independent audit logs are
retained externally; pilot outputs are referenced in place, not copied/modified.

Inside the recorded Docker image as1009:1020, with ARTIFACT_ROOT configured:

```bash
python -B -m unittest discover -s tests -p 'test_oasis_two_increments_first10.py'
python -B -m unittest discover -s tests -p 'test_oasis*bounded*.py'
# Set DL_NCC_CODE_ROOT and VFA_ROOT to the recorded native source roots first.
python -B -m unittest discover -s tests -p 'test_oasis_dl_ncc_diffusion.py'
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_two_increments_first10.py launch \
  --artifact-root "$ARTIFACT_ROOT" --gpus 0,1,2,3
```

The launcher recognizes its existing accepted report and will not overwrite
or rerun accepted outputs. Revisions, file hashes, Docker image identity and
effective per-run numerical settings are preserved in external manifests.
