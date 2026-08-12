---
doc_type: codex_execution_spec
status: full_complete
experiment_id: lungct_dual_inr_1mm_lungmask_author_schedule_v1
date_planned: 2026-08-07
---

# DUAL-INR PL Author-Schedule Execution Specification

## Scope

Run the controlled pTVReg 1 mm fixed-lung-mask protocol with the released
DUAL-INR original blur schedule instead of the controlled MR-style image
pyramid. The full cohort completed on 2026-08-07.

## Variant

- `PL1A`: protocol `PL`, fixed T00 lung-mask sampling and metric mask, released
  two-branch DUAL SIREN, native DUAL regularization, no semantic/lobe loss,
  original DUAL blur sigmas `[4,2,0]`, adaptive convergence transitions, total
  steps `3000`.

## Contract

- Dataset: DIR-Lab 4DCT cases `01-10`.
- Direction: T50 moving image to T00 fixed image.
- Image domain: canonical `pTVReg_iso1mm/4DCT/caseXX`.
- Sampling mask: fixed T00 1 mm lung mask only.
- Metric mask: fixed T00 1 mm lung mask only.
- Loss: sampled LNCC plus native DUAL Laplacian/negative-Jacobian
  regularization.
- Semantic/lobe loss: disabled with `mask_weight = 0.0`.
- Schedule: full-resolution images blurred by Gaussian sigmas `[4,2,0]`;
  transitions are triggered by the released DUAL convergence rule.

## Launch

```bash
python -m scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision \
  --stage full \
  --variants PL1A
```

Use the project Docker container for GPU execution. Keep large run artifacts
outside Git and outside dataset roots.

## Acceptance

Technical acceptance requires expected artifacts, finite history/metrics,
protocol `PL`, fixed T00 lung-mask sampling and metric rules, exact blur sigmas
`[4,2,0]`, all three blur stages visited, completed dense-field TRE
diagnostics, and completed common posthoc regularity.

TRE or regularity magnitude is not a completion gate.
