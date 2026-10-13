# IDIR full-resolution omega/diffusion sweep technical log

## Current status: cancelled and cleared at the user's request

The user clarified that IDIR must retain10,000 sampled coordinates per update.
The assistant's full-volume training-domain change exceeded the user's intent.
Controller72359 and its verified workers72493/72494 were terminated; a full
container process scan confirmed no matching controller/worker remained.
No other jobs were signaled. The exact run directory was moved to
`${ARTIFACT_ROOT}/trash/cancelled_oasis_seed3_first10_idir_siren_omega_ncc9_diffusion_2500_v1_20261010/`.
About411MiB is recoverable there; the active run path is absent. No matching
review directory had been generated. Source code/tests remain as an inactive
audit record, not an authorized replacement implementation. Do not resume it.

The execution specification now contains a sampled replan and explicitly asks
whether to retain sampled global1-NCC or native local NCC9, which needs spatial
neighborhood evaluations. No replacement run, smoke or training was launched.
All entries below are historical records of the cancelled attempt.

## Known limitations

First10 test-case tuning is exploratory. Omega changes native initialization
and conditioning. Fixed2500updates do not guarantee convergence. CUDA sampling
backward is not globally deterministic. Full-voxel loss is much more expensive
than the historical10000sample IDIR objective; old runtime is not transferable.
Discrete Jacobians do not establish continuous/off-grid invertibility.

## Implementation and launch gates

The intended scope was250 fresh registrations on GPU1/3, plus functional smoke checks,
and requested all35per-segmentation accuracy rows for every case. Implemented
isolated core/worker/controller/tests; existing numerical helpers are read-only.
No old omega32 outputs are reused. Primary checkpoint is terminal2500.

## Qualification — 2026-10-10 21:21 UTC

- Eleven new IDIR unit tests and25 related loss/metric tests passed (36total),
  including line-only plots and rejection of incomplete-cohort aggregation.
  One preliminary test used the artifact-method name instead of the legacy CLI
  method `idir_original`; the test fixture was corrected. No legacy code changed.
- Five rank0 GPU smokes passed:omega8/32 onGPU1 and16/64/128 onGPU3.
  Omega32 ran10updates; others1. Native NCC9/diffusion value and gradient
  differences were exactly0. Chunked GPU SIREN parameter-gradient maximum
  absolute difference was4.47035e-8; largest one-Adam-step parameter difference
  was4.47035e-8, within the predeclared numerical tolerances.
- All five finite160×224×192×3fields, checkpoint/field roundtrips, all175label
  records and canonical five-metric summaries passed. No failed GPU attempts.
- Maximum allocated2.43168GiB/reserved2.91406GiB. Conservative launch gate
  remains12GiB free; current GPU1/3 both exceed it. Other jobs untouched.
- Warm omega32 updates2–9 took0.819–0.845seconds; rough compute-only250fit
  projection isabout72hours on2GPUs, plus loading/metrics/checkpoint overhead.
  This short shared-workload estimate is provisional, not a timing claim.
- Original20 VFA/SITReg baseline cases and41 associated hashes were validated
  read-only for reference tables and line-only accuracy-versus-stdlogJ plots.
- Artifact-store free space exceeded12TiB; dense fields requireabout20GiB.

Production launch follows this completed qualification. Numerical sources are
hash-locked to the saved smoke certificate; generated logs/results remain in
the external artifact store.

## Production launch — 2026-10-10 21:22:27 UTC / 14:22 PDT

Detached controller container PID72359 started two production workers after
revalidating all five smoke results and the reference hashes. GPU1 worker
PID72493 starts rank0/omega8/lambda0.1; GPU3 worker PID72494 starts
rank0/omega8/lambda1over3. Remaining248fits are queued dynamically, one new
worker per approved GPU. At startup both workers were alive with finite Adam
updates; no production failure was recorded. Existing jobs/queues were not
stopped, reprioritized or altered.

Live state:`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_idir_siren_omega_ncc9_diffusion_2500_v1/controller_state.json`.
Each worker updates `progress.json` every25updates and streams every state's
loss to `history.jsonl`. Successful cases are exported to the corresponding
external review root incrementally. Final acceptance and line plots wait for
all250valid terminal fields and8750per-case/per-segmentation rows.

The controller/worker/numerical sources were frozen during this now-cancelled
run. Historical outputs and sources must not be reused as sampled-run evidence.
