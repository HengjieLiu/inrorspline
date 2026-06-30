# SINR NTK Conditioning Execution Spec

## Purpose

Test whether the SINR coefficient INR induces a rank-deficient, anisotropic, or
amplifying CP-space tangent kernel compared with DirectCP's identity kernel in
the same cubic B-spline control space.

The experiment compares the action of:

```text
DirectCP dense update:  B s
SINR dense update:      B J_theta J_theta^T s
```

on actual OASIS rank-9 registration gradients, without materializing dense
`B B^T`, `B J_theta J_theta^T B^T`, `J_theta`, or `J_theta J_theta^T` matrices.

## Source Plan

- Theory and coding plan:
  `docs/codex_prompts/260629_5_from_260629_4_sinr_ntk_conditioning_plan.md`
- Prior proxy/exploration root:
  `experiments/2026-06-29_sinr_cp_kernel_followup/`

## Initial Scope

```text
dataset: OASIS BrainMR
split: seed-3 shuffled pair_test_200
case: rank 9, fixed/moving (0173, 0136)
method state: SINR cps4 alpha1000 lr1e-4
states: init and existing final checkpoint
gradients: data loss and total loss
spectrum probes: 32
power iterations: 10
sample voxels: 100000 for dense-action summaries
```

## Environment

Use the default Docker container:

```text
container: py310sitreg_v1_test1
image: hengjieliu96/dl_dir:py310sitreg_v1
repo mount: /homebase/code_sync/inrorspline
artifact root: /database/inrorspline/runs/sinr_ntk_conditioning_20260629_v1
```

## Planned Script

```text
scripts/experiments/update_alignment/run_sinr_ntk_conditioning.py
```

## Planned Commands

Dry run:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/update_alignment/run_sinr_ntk_conditioning.py --dry-run'
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/update_alignment/run_sinr_ntk_conditioning.py --smoke --gpu 0'
```

Full first pass:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/update_alignment/run_sinr_ntk_conditioning.py --run-full --gpu 0 --states init,final --gradients data,total --spectrum-probes 32 --power-iters 10'
```

## Acceptance Checks

- dry-run writes a plan JSON,
- smoke writes one finite OASIS rank-9 data-gradient row,
- PSD check passes for `s^T K_C s`,
- full first pass writes action, spectrum, and matrix-size tables,
- dense matrices are not materialized,
- report states limitations and local-NTK interpretation.
