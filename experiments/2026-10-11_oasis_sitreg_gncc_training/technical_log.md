# P56 technical log

## Limitations

- Changes locality and squaring together; not an isolated locality ablation.
- Same lambda need not mean equal similarity/regularization balance.
- Historical LNCC lambda1 is contextual, not freshly retrained with this sweep.
- Single seed and previously explored test cases do not establish significance
  or an untouched confirmatory benchmark.
- Historical CUDA/inversion backend is not bitwise deterministic; other GPU
  workloads make runtime diagnostic. GPU0 has limited startup headroom.

## 2026-10-11

- User approved exactGNCC weights0.1,1/3,1,3 and four-GPU training plus
  automatic detailedfirst100 evaluation.
- Audit verified unchanged historical source hashes/data manifest/checkpoint;
  four prior LNCC initial-state and full training-pair hashes match.
- Implementation and preflight in progress; no production training launched yet.
- All26 CPU tests passed:11 new GNCC contracts,6 unchanged native training
  contracts,and9 unchanged detailed canonical-evaluation contracts.
- Preflight_attempt0 stopped before model setup because the launch omitted
  the historical isolatedpystrum0.4 PYTHONPATH. Failed artifacts are retained.
  Restore the existing read-only P45 runtime_pydeps; no package installation,
  native source changes or training-protocol changes. Fingerprint dependency
  sources and record PYTHONPATH in new manifests.
- After restoring the historical dependency environment, all27 CPU tests pass
  (12 new,6 legacy training,9 legacy evaluation). The new suite also exercises
  complete900-row/14000-label final-report schemas using temporary synthetic
  fixtures; those fixtures are not experimental results.
- Preflight_attempt1 historical rank0 metric reproduction passed without any
  tolerance changes: Dice delta-0.0000096615,HD95delta0,ASDdelta-0.0000022309mm;
  fieldRMSE0.00181816voxel,max0.12027049voxel. Native30-pair validation Dice
  0.81446603 versus recorded0.81446372. This is numerical, not exact parity.
- Full preflight_attempt1 passed: all four fresh GNCC Adam/checkpoint smokes,
  trained-weight finite-gradient smoke,and ranks0/10 serialization/canonical
  per-label evaluation. Peak reserved36.4921875GiB; controller's measured
  startup guard is38GiB. All four physical GPUs meet it at launch.
- Detached four-GPU controller launched after GNCC_PASSED. Automatic detailed
  first100 evaluation is chained per released GPU, followed by final comparison.
- Verified launch2026-10-11 18:40:21UTC (11:40:21PDT), controllerPID93592;
  GPU0/lambda0.1/PID93606,GPU1/lambda1over3/PID93611,
  GPU2/lambda1/PID93616,GPU3/lambda3/PID93619. All four initialization hashes
  exactly match the historicalLNCC seed42 state. Other GPU jobs unchanged.
- Live startup verification: all four workers completed finite GNCC updates
  (13/18/11/25 pairs on GPUs0/1/2/3 at the check), no OOM/failure markers;
  every logged moving/fixed pair exactly matched the old LNCC sequence.
  Source hashes are now frozen for the full training/evaluation/report chain.

## 2026-10-12: completion and requested ARC presentation

- All four200-epoch/10000-update trainings and400 detailed case evaluations
  completed successfully. The final report passed its source/checkpoint/field
  hash, case/direction,3500-label-per-candidate and cohort-summary gates:
 400 new cases,14000 label rows and500 immutable LNCC reference cases.
- GNCC lambda0.1 was the last evaluation to finish. Its first100 mean
  Dice/HD95/symmetricASD is0.82066332/1.83814180/0.36918963; foreground
  SDlogJ0.65629388 and folding0.00993689%. No partial cohort was plotted.
- User requested the final2x3 figure with lines only on top and SD ellipses
  below. Added independent presentation-only source
  `scripts/reports/oasis_sitreg_ncc_arc_lines_top.py`; no hash-locked training,
  evaluation or original report source was changed.
- Export requires accepted DONE plus all source-output hashes and recomputes
  every plotted mean/sample SD from900 identity-matched case rows. Nine
  operating points:GNCC0.1/1over3/1/3 andLNCC0.1/1over3/1/3/10.
  Ellipses are axis-aligned marginal plus/minus1 sampleSD, not confidence
  regions or covariance ellipses. Axes match across rows; labels identify
  lambda; historicalLNCC1 uses a hollow diamond and dashed adjacent segments.
- Three standard-library test cases passed, including incomplete/corrupt-data
  rejection, exact ellipse dimensions/row ordering, PNG/PDF/SVG export,
  checksum enforcement and refusal to overwrite an existing export. Container
  pytest was unavailable; tests ran directly through unittest, without installs.
- PNG/PDF/SVG, plotted statistics and a checksummed plot manifest are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1/arc_lines_top_v1/`.
  The final PNG was visually inspected. The parent accepted report remains
  unchanged; its original figure retains the earlier row order.
