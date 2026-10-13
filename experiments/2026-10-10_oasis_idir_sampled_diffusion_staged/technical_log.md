# Sampled IDIR staged-search technical log

## Known limitations

Exploratory first10test tuning; sampled signed global NCC, not localNCC9.
Analytic voxel-coordinate diffusion, not voxel forward differences. Old
effective CUDA flags were not recorded; current explicit F/T+TF32off is used.
Lambda will be selected atomega32 and frozen for the subsequent frequency test.
Existing sources/artifacts and other jobs remain read-only/unchanged.

## Historical audit

All60 historical omega32 BE runs exist with10,000 sampled coordinates,2500
updates, LR1e-4, seed1 and256×3SIREN. Old BE has a second division by10,000.
Mean Dice peaks atalpha1 (.772102), compared with.760961 at.1 and.743351 at10;
HD95/ASD atalpha1 are2.341493/.530032. These are reference diagnostics, not
new diffusion results. No oldomega32 output is reused as a diffusion candidate.

## Implementation status

User authorized two concurrent workers perGPU1/3, first a coarse diffusion
decade sweep atomega32, boundary extension if needed, then a fixed-lambda
omega sweep. Isolated sampled core/worker/calibrator/controller and tests are
implemented. Qualification passed and production was launched as recorded below.

## Qualification and scale calibration

- All 12 new sampled-protocol unit tests and 9 canonical-evaluator tests passed
  (21 total). `git diff --check` passed. The tests include exact 10,000-point
  budgets, global NCC equivalence, analytic diffusion value/gradient/unit scaling,
  constant-field behavior, incomplete-cohort rejection and boundary decisions.
- Read-only calibration used ranks 0, 4 and 9. Matching the historical best-BE
  regularization parameter-gradient norm suggested new diffusion coefficients
  0.108807/0.107508/0.107108 at initialization and 0.203023/0.207822/0.233742
  at the historical terminal BE fields. This is a scale diagnostic, not an
  accuracy-based choice: the requested 0.001–1000 grid remains unchanged.
- Four simultaneous 100-update smokes completed on slots 1/1/3/3, with finite
  gradients, serialized/reconstructed fields and canonical per-label metrics.
  Peak reserved training memory was 0.628906 GiB per worker. Warm median update
  times were 0.01307/0.02353/0.02017/0.02185 seconds under shared-GPU execution.
  These short smoke results are excluded from scientific sweep tables.
- The smoke certificate is `${ARTIFACT_ROOT}/runs/` followed by the artifact
  key in the execution spec and `/SMOKE_PASSED.json`. Calibration diagnostics
  are under the same root's `calibration/` directory.

## Production launch and monitoring

- Started 2026-10-10 21:54:45 UTC (14:54:45 PDT) as container UID:GID 1009:1020.
  Detached controller PID 73655; initial worker PIDs 73789/73790/73791/73795.
- Four production workers confirmed on GPUs 1/1/3/3; 70 initial omega-32 fits,
  then complete-cohort boundary decisions, then 40 additional frequency fits
  at the frozen selected lambda. Other GPU jobs were not modified.
- Persistent `controller_state.json`, `controller.log` and `events.jsonl`
  record progress and decisions. The controller extends boundary winners by
  one decade per complete ten-case cohort; it cannot select incomplete results.
  `LAMBDA_SELECTED.json` precedes the fixed-lambda frequency sweep, and
  `ACCEPTED.json`/`DONE` are written only after complete validation.
- Initial smoke-only compute estimates exclude startup, canonical CPU metrics,
  serialization and validation. Production timing must determine the user ETA;
  shared-workload timing is diagnostic, not a method speed comparison.
- First completed production fits (ranks 3, 0 and 1 at lambda 0.001) required
  56.85, 66.35 and 71.43 seconds per case, including evaluation but excluding
  process startup/controller checks. All performed exactly 2,500 updates and
  25,000,000 training-coordinate evaluations. No lambda winner is inferred from
  this partial cohort. Initial planning ETA: roughly 25–35 minutes for the
  initial 70-fit lambda sweep and 40–55 minutes for both stages, plus roughly
  3–5 minutes per extra ten-case boundary decade. Reassess under changing load.
