# P79 technical log

## Known limitations

- Historical and fresh timings reflect different competing workloads.
- Two-map bounded updates have twice the one-map schedule budget.
- Velocity and displacement penalties differ at equal lambda.
- Reuse is limited to exact accepted protocol/case/hash matches; no dense final
  output is counted as a pure cps842 displacement registration.
- Initial ten pairs informed tuning; the100-case cohort is descriptive.

## Implementation

2026-10-16: user approved1530 missing fits,570 accepted reuses, all four GPUs
with up to two processes each. Added isolated cohort/worker/controller/report
adapters and regression tests. Historical numerical sources and artifacts are
unchanged. Verification and launch status to be recorded after execution gates.

## Preflight

- 2026-10-16 07:56UTC:68 regression tests ran successfully (67 passed,
  one historical optional test skipped). New12 tests cover exact2100/1530
  matrix counts, rank99, invalid CLI values, unique smoke names, unchanged
  objective/optimizer objects, complete-only sample SD, artifact receipts,
  same-GPU audits, individual failure isolation, OOM budget and live-PID adoption.
- Read-only report preflight reconstructed all1710 accuracy statistics from
  per-label CSVs for570 reused cases. Runtime adapter installation preserves
  frozen source-manifest verification. No numerical historical source edited.
- Full external checksum/finite/history validation is running before the six
  native-volume smoke/replay jobs; production remains gated on both checks.
- A detached handoff supervisor was started after identifying the exact
  preflight PID/start-time. It waits for that controller to exit, checks both
  REUSE_VERIFIED and SMOKES_PASSED, rechecks frozen sources, then starts the
  durable full-production controller. Missing/failed gates produce an explicit
  AUTO_START_FAILED record; no launch is inferred from a missing marker.
- Latest startup inspection:465/570 reusable results validated, no integrity
  failures, preflight and supervisor alive. Production has not yet started.

## Metadata-only recovery, approved 2026-10-16

- Subsequent inspection established that preflight ended at08:27UTC, not in
  production. All570 reused outputs and200 reference fields passed validation.
  Four smoke/replays passed; both SVF smokes failed before their first update.
- Root cause: the P76 parent manifest contained controller settings alongside
  paths. The legacy SVF setup intentionally converts every worker setting to
  `Path`, so list-valued `gpu_slots` raised `TypeError`. The prior adapter tests
  did not exercise that setup loop. This is not a CUDA OOM or numerical error.
- Filter new worker manifests to `oasis_root`, `pair_file`, `dl_code_root` and
  `vfa_root`; reject invalid path values. Keep controller settings separate.
  Test the actual unchanged SVF setup with mixed and filtered settings.
- Use a distinct `metadata_recovery_v2` run subdirectory. Do not rewrite the
  original manifest, successes, failures, receipts or controller records.
  Prior reuse receipts are copied only after manifest-hash binding, unchanged
  scientific contracts/numerical sources and current file-stat verification.
  Current inputs and original DL fields are rehashed by preflight.
- Rerun all six full-size smoke/replays under the new source manifest, then
  automatically enter production in the same durable controller. Preserve
  individual failures and continue later cases; retain the existing shared
  source/storage safety gates and one OOM retry policy.
- Recovery checks:72 regression tests,71 passed/one skipped (including16
  adapter/queue tests); no live P79 workers, four idle GPUs with48518MiB free
  each, about18TiB artifact capacity. Launch status follows after gates pass.
- Recovery controller launched17:20:24UTC. Revalidated570 reuse receipts and
  all200 reference hashes; six native smoke/replays completed successfully in
  62.07s, no failures. Measured maximum allocated/reserved smoke peak3.371GiB;
  the8GiB admission rule remains sufficient.
- Production began17:24:12UTC in the same detached controller. At17:24:42UTC,
  eight registrations were active, exactly two on each GPU;1530 new fits in
  the queue, no failures or infrastructure errors. GPU used memory was about
  3.7–5.1GiB per device at that observation. This is a launch record, not an
  experiment completion claim; runtime ETA must use subsequent throughput.
- At17:27:25UTC, six new production runs passed independent replay: one
  displacement, four bounded-two, one refined-SVF. Seven optimizations had
  finished; eight worker/audit slots remained active, two/GPU. No failures,
  OOMs or infrastructure errors. The first192.7s yielded a naive remaining
  ETA13.60h, strongly affected by pipeline startup and only one case; not yet
  a stable whole-cohort forecast. Per-run worker/replay times are being saved.
- Post-launch preservation check passed for11 original preflight records and
  all570 reused result receipts; frozen numerical and adapter hashes matched.

## Scoped transition continuation v3

- Queue completed1528 new results plus570 reuses; ranks29/44 velocity-SVF at
  lambda.01 failed the field-transition max limit. Partial report preserved.
- Read-only audits on originalGPU3/2 exactly reproduced saved fields and
  both failures. Rank29 cps8->4 FP32 max/RMS .0010076873/9.6648432e-7;
  rank44 cps4->2 .0011883443/7.2416243e-7 voxel. Each exceeds.001 at one voxel.
- Ideal FP64 field preservation max2.7667e-12/7.9583e-13 voxel. Actual rounded
  controls evaluated in FP64 yield1.22946e-4/1.11568e-5 maximum; rank29 also
  exceeds the historical cps1-v2 rounded-control limit1e-4. A documented new
  scoped policy is required, not silent reuse of that earlier exception.
- Accuracy transition deltas: Dice-1.15193e-7/-9.59728e-8; HD95 unchanged;
  ASD+2.81408e-8/+9.28167e-8mm. Rank44 FGSDlogJ+3.27715e-5 and FGfold
  -4.83312e-5 percentage points (one voxel); rank29 folding unchanged. These
  are transition diagnostics, not final-registration accuracy claims.
- User approved policyv3 and missing-stage continuation. Added separate
  validator/runner/tests; historical numerical and extension sources remain
  unchanged. Preserve all accepted outputs using their existing hash receipts
  plus current stat checks; rehash failed-attempt bytes and metadata/reports.
  Only new stages construct Adam. Same-GPU independent replay is mandatory.
- Final prelaunch regression:89 tests in14.29s,88 passed/one optional test
  skipped. The added preservation test caught string-versus-Path handling in
  the new wrapper; corrected before any continuation manifest or run existed.
- Continuation controller launched2026-10-17 04:01:05UTC (Oct16 21:01PDT),
  UID:GID1009:1020. It first verifies2098 preserved results, then runs two
  full-size missing-stage smokes/replays on originalGPU3/2. Production and
  final aggregation are gated on those checks. Initial free memory18.3/19.5GiB
  onGPU3/2; other workloads remain active and untouched. This is a launch
  record, not acceptance of either remaining result.
- All2098 preservation checks passed. Two full-size smoke workers and their
  independent replays passed before production began04:04:40UTC. Rank29 ran
  exactly two new Adam updates (cps4/2), rank44 one (cps2); completed prefixes
  replayed exactly without optimizer work. Smoke worker times52.26/57.41s,
  audit39.48/35.18s, maximum reserved3.305GiB;8GiB admission retained.
  Both failing transitions reproduced their original FP32 differences and
  passed the mandatory ideal/rounded-control FP64 proofs. Production runs
  only these two unfinished registrations on originalGPU3/2.
- Both production continuations and independent replays passed. Rank29
  cps4/2 actual updates575/900, selected572/874; new optimizer431.39s,
  continuation worker477.27s, replay37.90s, reserved3.297GiB. Rank44 cps2
  actual1050, selected1047; new optimizer383.72s, worker424.18s, replay36.83s,
  reserved3.262GiB. Prefix optimizer26.10/36.79s was reused, not rerun.
  Rank29's newly encountered cps4->2 FP32 difference max6.68839e-4,
  RMS9.13104e-7 voxel passed without any further exception.
- Final rank29 Dice/HD95/ASD .636427/6.943286/1.617320, FGSDlogJ1.911788,
  folding4.618778%; rank44 .806451/3.244063/.502042, FGSDlogJ1.419806,
  folding2.413081%. Weak-lambda folding remains measured and included,
  not rejected or repaired. Seven-step outputs remain the primary results.
  Final2100-result report generation follows these successful replays.
- Final acceptance2026-10-17 04:15:43UTC (Oct16 21:15PDT):2100 cells,
  700/family,200 DL references,69 complete cohort groups and345 independently
  reproduced CSV mean/sample-SD checks. All13 report output hashes verified.
  Final preservation recheck passed for2098 receipts,29 failed-attempt files,
  original metadata and reports. No additional registration or tuning launched.
  Report release `transition_continuation_v3/release_1792210489765127750`.
- Visual inspection found the inherited long ASD subplot title clipped at
  the figure edge. Added a separate presentation-only renderer and external
  `readable_arc_v2` PNG/PDF/SVG companion, wrapping titles onto two lines.
  Data, styles, axes, SD ellipses and all13 accepted report files unchanged;
  companion provenance records its own source and output hashes.

## Five-family first100 ARC companions

- User requested two2x3 figures and explicitly selected full SITReg/VFA NCC9
  sweeps. Combine their five settings each with seven settings for each P79
  family (displacement842, bounded-two842, refined-SVF/velocity842):3100 rows,
  31 complete100-case groups. No dense, FeDDA, cps1, or symmetric arm included.
- Read-only source checks reproduce310 mean/sample-SD values and all original
  DLlambda1 reference metrics. Six dedicated tests passed (2.36s): cohort/
  identity/finite checks, sample SD, ratio units, zero handling, contour
  clipping and shared axes. No registration, metric reevaluation or GPU work.
- Top accuracy windows: Dice.76–.84, HD951.7–2.7mm, symmetricASD.30–.60mm;
  26/31 means are in view in each top panel. Bottom panels retain all31
  means and marginal sample-SD contours. Corresponding y limits are identical
  between figures; all six panels in each figure share x limits.
- Folding is a ratio, not percent. Exact-zero means use hollow markers at
  the documented display-only floor1e-10. Source values stay zero; no
  pseudocount or log-space statistics. Arithmetic SD contours are clipped at
  that floor before the axis transform; they are not confidence regions.
- PNG/PDF/SVG, plotted mean/SD CSV, per-case CSV/JSON, README and checksums
  stored under `oasis_seed3_first100_five_family_ncc9_arc_stdlogj_folding_v1`.
  Both figures visually checked; original accepted reports remain unchanged.
