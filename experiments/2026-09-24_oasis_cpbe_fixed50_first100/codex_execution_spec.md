# OASIS-100 fixed-50 CP-BE evaluation

## Objective

Run two selected fixed-budget CP-BE candidates on seed-3 shuffled OASIS ranks
0–99 and compare their quality with the existing OASIS-100 analytic-BE fast and
2500-step records.

The new candidates are:

- fixed 50/50/50, cache + batch + CP-BE, Adam;
- fixed 50/50/50, CP-BE, AMSGrad.

Both preserve spacing 8/4/2, image downsampling 4/2/1, stage learning rates
0.03/0.01/0.003, alpha 1000, seed 1, float32, and cuDNN benchmark false /
deterministic true.

## Runtime protocol

Only the two CP-BE candidates are newly registered. A fresh persistent worker
per candidate runs rank 0 cold, rank 0 warm, then ranks 1–99 warm. The endpoint
is case_load_complete through dvf_file_ready. Evaluation is excluded.

At launch, the controller selects one GPU that remains free of compute processes
for three checks and locks both timed arms to it. A one-Hz monitor retains and
retries contaminated attempts.

## Evaluation

After timing ends, the 200 warm fields are evaluated offline. Start with 16
persistent workers distributed across idle GPUs, then compare a 32-worker wave.
Keep 32 only when throughput improves by at least 10%, host available memory
remains at least 64 GiB, and every evaluation GPU remains below 40 GiB. Otherwise
finish with 16 workers. Evaluation timing is never reported as registration
runtime.

Metrics are 35-label Dice, area-weighted HD95 and ASD at 1 mm, full-resolution
NCC, foreground/full Jacobian statistics, folding, positive-J fractions,
finite-difference BE, and diffusion. Report first100 and ranks10–99 separately.

Historical quality anchors are resolved from the exact saved OASIS-100
MR_A_fast_decay and canonical alpha-1000 2500-step analytic-BE summaries. They
are not rerun or reevaluated. Historical timing is the controlled first10 GPU3
F/T reference and is explicitly not a matched full100 timing comparison.

## Outputs and acceptance

Generated material remains beneath:

- ${ARTIFACT_ROOT}/runs/oasis_cpbe_fixed50_first100_<timestamp>
- ${ARTIFACT_ROOT}/repo_experiments/2026-09-24_oasis_cpbe_fixed50_first100/<run-id>

Acceptance requires 202 clean timing rows, 200 valid new warm DVFs, 200 new
metric rows, and exactly 100 unique historical rows per baseline. Every new DVF
must be finite float32 normalized-grid XYZ with shape (160,224,192,3). Preserve
all commands, source hashes, dirty-tree provenance, environment/data manifests,
monitor logs, failed attempts, fields, CSV/JSON, and the Markdown report.
