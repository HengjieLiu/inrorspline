# Optional SINR CP-Kernel Follow-Up Plan

Source-linked adaptation:

- Primary source prompt:
  `docs/codex_prompts/260629_1_sinr_cp_kernel_experiment_plan.md`
- Prerequisite overnight run plan:
  `docs/codex_prompts/260629_3_from_260629_2_overnight_update_alignment_run_plan.md`
- Adaptation date: 2026-06-29
- User-approved defaults inherited from the overnight plan: use GPUs `0,1,2,3`
  and begin with the same first three diagnostic cases only.

## Purpose

Run this only after the update-alignment overnight plan has produced a clean
report or after that report identifies SINR-vs-DirectCP update similarity as the
main unresolved question.

This follow-up tests whether the SINR coefficient INR mainly rescales direct
B-spline control-point updates, and whether it introduces gradient amplification
or anisotropic conditioning.

Core source-prompt metrics:

```text
D_U = cos(B K s, B s)
R_U = ||B K s - alpha B s|| / ||B K s||
A_C = ||K s|| / ||s||
A_U = ||B K s|| / ||B s||
lambda_max_K = optional power-iteration estimate
kappa_eff_K = optional robust spectrum diagnostic
```

where:

```text
s = d image_loss / d C
K = J_theta J_theta^T
B = cubic B-spline decoder from control coefficients to dense displacement
```

## Trigger Conditions

Run this follow-up if one of these is true:

1. The overnight plan shows high SINR-vs-DirectCP sampled update similarity and
   the paper needs a lower-level mechanism for SINR learning-rate sensitivity.
2. The overnight plan shows low or ambiguous SINR-vs-DirectCP similarity and the
   paper needs to distinguish spline output bias from coefficient-INR kernel
   effects.
3. A reviewer-facing figure needs a compact mechanistic diagnostic beyond
   teacher-update alignment.

Do not run this follow-up if the overnight plan fails basic sign, unit, teacher,
or mask validation.

## Scope

Use the same first three cases:

```text
brain/oasis rank 9:
  method: sinr_cps4_be_bspline_analytic
  paired direct reference: direct_cp_cps4_be_bspline_analytic
  checkpoints: init, epoch 50, epoch 250

lungct case 01:
  method: sinr_cps4_ptv_lcc_isotv
  paired direct reference: direct_cp_cps4_ptv_lcc_isotv
  checkpoints: init, epoch 50, epoch 250

lungct case 08:
  method: sinr_cps4_ptv_lcc_isotv
  paired direct reference: direct_cp_cps4_ptv_lcc_isotv
  checkpoints: init, epoch 50, epoch 250
```

The multistage DirectCP reference is not part of the kernel calculation because
its control lattice, image pyramid, and stage transfer make it a capture-range
reference rather than a single-step direct CP counterpart.

## Output Roots

Use a separate root from the overnight update-alignment run:

```text
container run root:
  /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1

repo review root:
  experiments/2026-06-29_sinr_cp_kernel_followup
```

Expected review artifacts:

```text
experiments/2026-06-29_sinr_cp_kernel_followup/
  codex_execution_spec.md
  technical_log.md
  sinr_cp_kernel_review.ipynb
  sinr_cp_kernel_report.md
  tables/
    kernel_metrics.csv
    sampled_spectrum_metrics.csv
  plots/
    update_agreement_and_amplification.png
    kernel_spectrum_optional.png
```

## Implementation Strategy

Suggested script:

```text
scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py
```

Required features:

```text
--dataset oasis|lungct
--case-rank 9
--case-id 1|8
--checkpoint init|50|250
--gpus 0,1,2,3
--dry-run
--smoke
--run-full
--resume
--out-root /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1
--review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

Compute `K s = J J^T s` without forming `J`:

1. Flatten SINR coefficient output `C = h_theta(Q)`.
2. Compute `s = d image_loss / d C` by treating `C` as a leaf tensor and using
   the same B-spline decoder and image loss as the registration run.
3. Use VJP plus JVP or equivalent PyTorch functional APIs to compute
   `J J^T s`.
4. Decode `B s` and `B K s` to sampled or dense update fields.
5. Compute source-prompt metrics.

Use sampled dense outputs first:

```text
sample domain: same deterministic sample points as the overnight plan
sample units: OASIS voxel units; LungCT reg1mm voxel units
max sampled dense points: 100000
```

Only compute full dense `B s` and `B K s` if sampled diagnostics pass and memory
is acceptable.

## Spectrum Diagnostics

Default spectrum mode:

```text
lambda_max_K via power iteration
num_power_iterations: 10
```

Optional spectrum mode:

```text
sampled empirical kernel over at most 4096 coefficient entries
report percentile ratios instead of raw lambda_max/lambda_min
```

Do not report raw condition number if near-zero eigenvalues dominate. Prefer:

```text
p99_eigenvalue / p50_eigenvalue
p95_eigenvalue / p5_eigenvalue
lambda_max_K
A_C on the actual task gradient
A_U on the decoded dense update
```

## Self-Supervision Gates

Smoke gate:

- OASIS rank-9 SINR checkpoint `init` computes finite `s`, `K s`, `B s`, and
  `B K s` on sampled points.
- Positive-semidefinite sanity check passes:

```text
dot(s, K s) >= -numerical_tolerance
```

- Tiny-step loss sanity check passes when enabled:

```text
C_new = C - eps * K s
```

should not increase the image loss for sufficiently small `eps`, within
numerical noise.

Full gate:

- All three cases produce rows for `init`, `50`, and `250`, or the run stops
  with an explicit failed-checkpoint report.
- `D_U`, `R_U`, `A_C`, and `A_U` are finite.
- `D_U` lies in `[-1,1]` up to numerical epsilon.
- `A_C` and `A_U` are nonnegative.
- Any OOM is recorded with case, checkpoint, coefficient count, sample count,
  and attempted diagnostic mode.

Stop conditions:

- Repeated OOM after reducing dense samples.
- Nonfinite image loss or nonfinite gradient.
- pTVReg/OASIS teacher validation failure inherited from the overnight run.
- CP shape mismatch between SINR and direct CP counterpart.

## Suggested Commands After Implementation

Dry run:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --dry-run \
  --out-root /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --smoke \
  --resume \
  --out-root /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

Full:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --run-full \
  --resume \
  --out-root /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

## Report Language

Strong result language only if supported:

```text
On actual registration gradients for the selected cases, the SINR-induced
update B K s was close to a scalar-rescaled direct CP update B s, while the
coefficient INR amplified or anisotropically rescaled the CP-space gradient.
```

Avoid claiming:

```text
SINR is equivalent to DirectCP.
```

Preferred cautious claim:

```text
For the sampled checkpoints tested here, SINR and direct CP had similar decoded
update directions after scalar rescaling, but SINR added a coefficient-INR
optimization kernel that changed the effective step scale and conditioning.
```

## Expansion Gate

Only expand after the first three cases pass:

- Add more OASIS seed-3 ranks.
- Add LungCT cases `02-07,09,10`.
- Add checkpoint `1000` only if early checkpoint results are stable and useful.
- Add optional actual Adam-step diagnostics if vanilla `J J^T` metrics are
  insufficient for explaining observed LR behavior.
