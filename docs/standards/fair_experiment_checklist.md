---
doc_type: fair_experiment_checklist
status: active
---

# Fair Experiment Checklist

Use this checklist before launching experiments and again before interpreting
tables or plots.

## Dataset And Split

- Dataset handle is explicit: `brain` or `lungct`.
- Split file is named, versioned, and stored or referenced by manifest.
- Pair order is fixed and documented.
- Fixed image and moving image direction are explicit.
- Subsets are selected by a reproducible protocol, not by sorted first rows unless
  the protocol specifically says so.

## Preprocessing And Domain

- Image crop, resize, spacing, orientation, and intensity scaling are documented.
- Foreground, body, lung, or label masks are named and tied to the correct image.
- Interpolation modes are explicit for images, labels, masks, and displacement
  fields.
- Physical units are explicit for displacement, TRE, regularity, and spacing.

## Method And Optimization

- Method family and variant are named with enough detail to reproduce the run.
- Optimizer, schedule, learning rate, iterations, random seed, and initialization
  are recorded.
- Regularization type and scale are documented in method-native units.
- Any multistage schedule records every stage's control spacing, resolution, and
  iteration count.

## Metrics And Claims

- Primary metric definitions are fixed before launch.
- Missing labels, invalid landmarks, masks, and fold handling are documented.
- Regularity metrics use the same posthoc pipeline where possible.
- Tables do not mix preprocessing protocols as method-only comparisons.
- Claims identify whether gains come from representation, local-control structure,
  optimization path, regularization, or deformation-regime fit.

## Provenance

- Git commit, branch, Docker image tag/digest, dataset manifest, split config,
  command, and output root are recorded in the experiment registry.
- Known limitations are written near the top of each run log before launch.
- Raw data, checkpoints, generated outputs, credentials, and private host details
  are not committed to Git.

