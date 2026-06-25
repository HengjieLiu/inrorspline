---
doc_type: experiment_registry
status: active
---

# Experiment Registry

Every real run should get an entry before launch. Large artifacts belong under `SCRATCH_ROOT`, not Git.

## Required Fields

```text
Run ID:
Status:
Owner/agent:
Date planned:
Date started:
Date completed:
Server:
GPU model/count:
Git commit:
Branch:
Docker image tag/digest:
Dataset root:
Dataset manifest:
Split config:
Standards preflight:
Protocol tag:
Command:
Output root:
Primary metrics:
Known limitations:
Next decision:
```

## Planned Priority Runs

### P0: Scaffold Only

- Status: complete when root scaffold lands.
- Protocol tag: `repo_scaffold_no_experiments`.
- Notes: no data, Docker, rsync, or package/test scaffold.

### P1: OASIS Seed-3 First10 Main Pareto

- Status: planned.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Purpose: main brain Pareto result.
- Required before launch: `docs/standards/oasis_brainmr_protocol.md`,
  pair list manifest, metric script audit, alpha/LR grid lock.

### P1a: OASIS Seed-3 Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0`, original sorted index `158`,
  fixed/moving `(0333, 0115)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_seed3_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_seed3_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: user reviews pre-run Pareto/contact-sheet outputs before full
  overnight first10 launch.

### P1b: OASIS Original Sorted Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR original sorted `pair_test_200.txt` row `0`,
  fixed/moving `(0004, 0236)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_sorted_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_sorted_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: compare original sorted case-0 and seed-3 shuffled case-0
  curves before choosing first10 full-run alpha ranges.

### P2: OASIS Seed-3 First10 Deformation-Regime Stats

- Status: planned.
- Purpose: regenerate deformation-regime statistics on the same first10 protocol.

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.
