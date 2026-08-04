---
doc_type: experiment_protocol
status: completed
experiment_id: fit_modes_population_n10_v1
date: 2026-08-02
---

# Population Fit-Modes Experiment: OASIS N=10 and DIR-LAB N=10

## Purpose and reviewer motivation

A reviewer correctly noted that the existing representation-capacity figure
contains only one OASIS pair and two DIR-LAB cases. Those curves illustrate
optimization behavior, but three outcome-selected or difficulty-selected
examples cannot establish population consistency. This experiment repeats both
parts of the figure for a locked cohort of ten OASIS pairs and all ten DIR-LAB
4DCT cases.

The scientific question is narrow: at matched final control spacing and a fixed
optimization budget, how consistently can the four parameterizations optimize
image registration and reproduce an externally supplied displacement field?
Panel A studies the complete registration objective. Panel B isolates target
field representation and optimization using MSE only; it is not an independent
registration-accuracy experiment.

## Why the original three cases appeared

- OASIS `M0136→F0173` means moving subject `0136` registered to fixed subject
  `0173`. In the canonical pair file the fixed ID is written first, so the row
  is `(0173, 0136)`. It is seed-3 shuffled rank 9 and original sorted-pair index
  88.
- Rank 9 was selected post hoc because Greedy had the best Dice among the
  seed-3 shuffled first ten. It is therefore an illustrative favorable Greedy
  example, not an outcome-independent population sample by itself.
- DIR-LAB case01 was the initial/easy example used while developing the
  diagnostic. Case08 was added as the hardest prior single-stage example.

These reasons must be stated when describing the old three-case figure. The new
population analysis supersedes any implication that those examples were a
random evaluation cohort.

## Locked cohorts and provenance

### OASIS BrainMR

The cohort is the first ten rows after the repository's constrained seed-3
shuffle of `pair_test_200`, not the first ten rows of the sorted file:

`configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`

| Rank | Original sorted index | Fixed ID | Moving ID | Display direction |
|---:|---:|---:|---:|---|
| 0 | 158 | 0333 | 0115 | M0115→F0333 |
| 1 | 62 | 0113 | 0017 | M0017→F0113 |
| 2 | 14 | 0037 | 0394 | M0394→F0037 |
| 3 | 69 | 0117 | 0322 | M0322→F0117 |
| 4 | 47 | 0084 | 0108 | M0108→F0084 |
| 5 | 137 | 0288 | 0024 | M0024→F0288 |
| 6 | 110 | 0194 | 0184 | M0184→F0194 |
| 7 | 6 | 0014 | 0149 | M0149→F0014 |
| 8 | 116 | 0215 | 0326 | M0326→F0215 |
| 9 | 88 | 0173 | 0136 | M0136→F0173 |

Seed 3 was selected using subject-balance constraints rather than registration
outcomes. The correct paper description is **a prespecified, reproducibly
shuffled N=10 subset**. It is not sorted first10 and is not outcome-selected,
but it should not be called an unconstrained simple-random sample.

### DIR-LAB 4DCT

The cohort is every case, `01–10`. The direction is moving `T50` to fixed
`T00`. Both objectives use the canonical pTVReg-equivalent isotropic 1 mm crop;
the fixed-T00 lung mask defines the supervised fitting domain.

## Locked experiment matrix

| Dataset | Panel A: unsupervised registration | Panel B: supervised target-field fitting |
|---|---|---|
| OASIS N=10 | IDIR: `alpha=1`, `lr=1e-4`, 2500 epochs; INR-BSCP: `cps2`, `alpha=1000`, `lr=1e-4`, 2500 epochs; D-BSCP: `cps2`, `alpha=1000`, `lr=1e-3`, 2500 epochs; MR-D-BSCP: `8-4-2`, `lr=1e-3`, stages `400/800/1300` | SITReg target in normalized-grid displacement units; fixed `image>0` mask; MSE only; single-stage methods 2500 epochs; MR-D-BSCP `8-4-2`, stages `400/800/1300`; canonical LRs |
| DIR-LAB N=10 | IDIR: `alpha_BE=3`, `lr=1e-4`, 2500 epochs; INR-BSCP: `cps4`, `alpha=10000`, `lr=1e-4`, 2500 epochs; D-BSCP: `cps4`, `alpha=10000`, `lr=1e-3`, 2500 epochs; MR-D-BSCP: `32-16-8-4`, `lr=1e-3`, stages `200/400/800/1000` | pTVReg target reconstructed on the canonical 1 mm crop; fixed-T00 lung mask; MSE only; single-stage methods 2500 epochs; MR-D-BSCP `32-16-8-4`, stages `400/400/800/900`; canonical LRs |

The main comparison is final-spacing matched: OASIS single-stage `cps2` versus
multistage `8-4-2`, and DIR-LAB single-stage `cps4` versus multistage
`32-16-8-4`. No additional spacing ablation is part of this figure.

### Open camera-ready schedule issue

The accepted DIR-LAB MR-D-BSCP schedules are not identical across panels.
Panel A uses `200/400/800/1000` (2400 total epochs), inherited from the existing
Pareto-sweep registration protocol. Panel B uses `400/400/800/900` (2500 total
epochs), inherited from the pTVReg field-fitting protocol. Consequently,
DIR-LAB Panel A also gives MR-D-BSCP 100 fewer epochs than its three
single-stage comparators. OASIS does not have this issue: both panels use
`400/800/1300` (2500 total epochs).

Until corrected, the paper may claim matched final control spacing but must not
claim identical multistage schedules or equal DIR-LAB Panel A optimization
budgets. The preferred camera-ready fix is a new versioned DIR-LAB Panel A
MR-D-BSCP N=10 run using `400/400/800/900`, followed by reaggregation and
source-backed figure regeneration. The historical 2400-epoch results remain
accepted provenance and must not be overwritten. This is tracked in
`docs/todos/camera_ready_checklist.md`.

## Method definitions

| Display name | OASIS implementation ID | DIR-LAB Panel A ID | DIR-LAB Panel B ID | Learning rate |
|---|---|---|---|---:|
| INR-Dense | `idir_original` | `idir_be_default_lungmask` | `idir` | `1e-4` |
| INR-BSCP | `sinr_cps2_be_bspline_analytic` | `sinr_cps4_global_ncc_fd_be` | `sinr_cps4` | `1e-4` |
| D-BSCP | `direct_cp_cps2_be_bspline_analytic` | `direct_cp_cps4_global_ncc_fd_be` | `direct_cp_cps4` | `1e-3` |
| MR-D-BSCP | `direct_cp_8_4_2_be_bspline_analytic_multistage` | `direct_cp_32_16_8_4_global_ncc_fd_be` | `direct_cp_4stage` | `1e-3` |

All runs use Adam and seed 1. Panel A optimizes image similarity plus the
specified bending-energy regularizer. OASIS IDIR samples fixed-image-positive
coordinates; the OASIS spline methods use the full image for NCC. DIR-LAB uses
the audited method-specific global-NCC registration runners and the fixed-T00
mask convention. The full-image/global-NCC diagnostic is recorded every ten
epochs and is diagnostic only. Panel B has no image loss and no regularizer;
the recorded Panel A alpha is metadata only. DIR-LAB multistage Panel B records
the finest-domain MSE every epoch; OASIS multistage records it every ten epochs.

The Panel A endpoint is `1−NCC`. The Panel B endpoint is

`rRMSE (%) = 100 × sqrt(MSE / (target_power + 1e-12))`,

where both MSE and target power are component-and-masked-voxel means in
normalized-grid displacement units.

## Target-field conventions and validation

### SITReg

Each OASIS target is the saved SITReg dense displacement for the same fixed and
moving IDs. Stored voxel displacement is reoriented with the audited OASIS
loader and converted to PyTorch normalized-grid displacement before fitting.
The fitting and rRMSE domain is the fixed raw `image>0` mask. Validation must
lock case rank, original sorted index, fixed ID, moving ID, field shape,
component order, reorientation, and unit conversion. All `10/10` target files
must exist and load successfully before the full launch.

### pTVReg

Each DIR-LAB target is reconstructed from the saved pTVReg deformation for
`T50→T00` on the canonical isotropic 1 mm landmark crop. The reconstruction
must reproduce the recorded snap-to-voxel target TRE within the runner's
tolerance. The fixed-T00 lung mask is used for both fitting and rRMSE. All
`10/10` MAT targets must exist, load, pass orientation/TRE validation, and agree
with the case crop before the full launch.

## Reuse and versioning policy

An existing record may be reused only if all of the following match exactly:
dataset case and direction, target source and conversion, objective, units,
mask, implementation ID, final control spacing, multistage spacing/downsample/
epoch schedule, learning rate, alpha and its role, seed, epoch budget,
diagnostic interval, and output schema. The current source-backed figure's 24
selected records (OASIS rank 9 and DIR-LAB cases 01 and 08, two panels, four
methods) are the only intended reuse set. Compatibility evidence and source
paths are stored in `job_manifest.csv/json`.

Existing data are never overwritten. A protocol change requires a new external
output-root version. Partial compatible runs require `--resume`; the launcher
otherwise stops rather than guessing.

## Execution and storage

Public configuration uses environment variables rather than host-private
paths:

| Variable | Meaning |
|---|---|
| `REPO_ROOT` | Git checkout mounted in the container |
| `RUN_ROOT` | external root for large checkpoints, dense fields, and logs |
| `OASIS_ROOT` | organized OASIS split root |
| `SITREG_ROOT` | SITReg target-field root |
| `DIRLAB_PTVREG_ISO1MM_ROOT` | canonical DIR-LAB 1 mm crop root |
| `PTVREG_FIELD_ROOT` | pTVReg target MAT root |
| `INRORSPLINE_DOCKER_IMAGE` | portable image tag or digest recorded in provenance |

The launch environment is Docker container `py310sitreg_v1_test1`, image
`${DOCKER_IMAGE}`, with GPUs `0–3`. The runtime manifest
records the actual image/container context, Git commit and dirty paths, split
checksum, target checksums, and private resolved paths outside Git.

Large artifacts are written under the external `RUN_ROOT`. Compact review
artifacts live here:

- `cohort_manifest.csv`: 20 locked cases and target checksums.
- `job_manifest.csv/json`: 160 required records and compatibility provenance.
- `launcher_state.json`: recoverable queue state.
- `monitor_status.md/json`: counts, active GPUs, elapsed time, failures, ETA.
- `optimization_index.csv/json`: one accepted record per required job.
- `population_statistics.csv`: grouped endpoint statistics.
- `acceptance_report.md/json`: machine-readable and human-readable gates.
- `diagnostic_figures/`: one source-backed Panel A/B trajectory figure per case.
- `diagnostic_comparison.md`: OASIS and DIR-LAB case grids, five figures per
  row and two rows per dataset, plus endpoint tables.
- `diagnostic_curve_summary.csv` and `diagnostic_manifest.json`: curve-level
  source/endpoint validation and the locked dataset-shared axis definitions.

No population figure is generated until aggregation passes.

After the 160-record acceptance gate passed, the case-level diagnostic report
was generated. It uses log axes shared by all ten cases in a dataset: Panel A
`1e-3–5e-1` and Panel B `1e-1–1e3` rRMSE (%). These are diagnostic trajectory
grids, not a replacement for a compact population-statistics figure.

## Commands

Run these inside the declared container:

```bash
python scripts/experiments/fit_modes_population/launch_population.py \
  --dry-run --validate-targets --hash-targets

python scripts/experiments/fit_modes_population/launch_population.py \
  --smoke --gpus auto --poll-sec 15

python scripts/experiments/fit_modes_population/launch_population.py \
  --resume --gpus auto --poll-sec 15

python scripts/experiments/fit_modes_population/plot_case_diagnostics.py \
  --review-root experiments/2026-08-02_fit_modes_population_n10 \
  --run-root "${RUN_ROOT}" --data-root "${DATA_ROOT}"
```

The full manifest must resolve exactly `20 × 2 × 4 = 160` unique records. The
smoke cohort is OASIS rank 0 and DIR-LAB case02 across all methods/objectives,
for 16 isolated records.

## Completion status

| Dataset | Target validation | Required | Reused | New complete | Failed | Acceptance |
|---|---|---:|---:|---:|---:|---|
| OASIS | 10/10 passed | 80 | 8 | 72 | 0 | passed |
| DIR-LAB | 10/10 passed | 80 | 16 | 64 | 0 | passed |
| Total | 20/20 passed | 160 | 24 | 136 | 0 | passed |

The full four-GPU launch completed on 2026-08-02 at 22:43:58 UTC after
2 h 20 min 26 s of scheduler wall time. The final acceptance report contains
160/160 records, no unresolved problems, and all 16 dataset/panel/method
groups at N=10. The smoke run also passed 16/16 records and a subsequent
resume check reused all 16 without launching an optimizer.

## Runtime and endpoint results

| Dataset | Panel | Method | N | Runtime median (s) | Endpoint mean ± SD | Endpoint median [Q1, Q3] |
|---|---|---|---:|---:|---|---|
| OASIS | A | INR-Dense | 10 | 216.9 | 0.025414 ± 0.003872 | 0.024261 [0.023123, 0.028708] |
| OASIS | A | INR-BSCP | 10 | 304.0 | 0.008504 ± 0.001202 | 0.008136 [0.007932, 0.008997] |
| OASIS | A | D-BSCP | 10 | 108.5 | 0.006969 ± 0.000940 | 0.006914 [0.006541, 0.007500] |
| OASIS | A | MR-D-BSCP | 10 | 192.2 | **0.006016 ± 0.000827** | **0.005924 [0.005523, 0.006381]** |
| DIR-LAB | A | INR-Dense | 10 | 239.4 | 0.103922 ± 0.053683 | 0.090725 [0.062607, 0.128658] |
| DIR-LAB | A | INR-BSCP | 10 | 250.0 | 0.004149 ± 0.002509 | 0.003556 [0.002266, 0.005257] |
| DIR-LAB | A | D-BSCP | 10 | 215.1 | 0.005279 ± 0.003578 | 0.004681 [0.002408, 0.007385] |
| DIR-LAB | A | MR-D-BSCP | 10 | 308.9 | **0.003216 ± 0.001950** | **0.002862 [0.001826, 0.003793]** |
| OASIS | B | INR-Dense | 10 | 8.9 | 14.0356 ± 1.2449% | 13.7726% [13.0017, 14.9438] |
| OASIS | B | INR-BSCP | 10 | 261.6 | 9.8980 ± 0.8960% | 9.7505% [9.1651, 10.3883] |
| OASIS | B | D-BSCP | 10 | 68.0 | **4.0332 ± 0.3317%** | **4.0108% [3.8288, 4.2582]** |
| OASIS | B | MR-D-BSCP | 10 | 141.7 | 4.0528 ± 0.3374% | 4.0294% [3.8429, 4.2813] |
| DIR-LAB | B | INR-Dense | 10 | 9.6 | 2.7581 ± 1.5243% | 2.3400% [2.0729, 2.6267] |
| DIR-LAB | B | INR-BSCP | 10 | 114.9 | 1.5003 ± 0.2405% | 1.4269% [1.3243, 1.6942] |
| DIR-LAB | B | D-BSCP | 10 | 70.0 | 0.5368 ± 0.0892% | 0.5459% [0.4830, 0.6162] |
| DIR-LAB | B | MR-D-BSCP | 10 | 234.9 | **0.5249 ± 0.0930%** | **0.5347% [0.4709, 0.6094]** |

Panel A reports `1-NCC`; Panel B reports rRMSE. Lower is better in both
panels. Runtime is optimizer-internal wall time from the per-run summary, not
the launcher's process elapsed time. Bold marks the lowest cohort mean within
each dataset and panel; it is descriptive and does not imply statistical
significance.

## Final conclusions

The population experiment removes reliance on the original three illustrative
cases. At the locked final-spacing-matched settings, MR-D-BSCP has the lowest
mean registration endpoint on both OASIS and DIR-LAB. In supervised fitting,
the direct control-point methods are substantially lower than INR-Dense and
INR-BSCP: D-BSCP has the lowest OASIS mean by 0.0196 percentage points relative
to MR-D-BSCP, whereas MR-D-BSCP has the lowest DIR-LAB mean by 0.0118
percentage points relative to D-BSCP. These small D-BSCP/MR-D-BSCP differences
are descriptive; no significance or equivalence claim is made without a
prespecified paired analysis.

The result supports a limited representation/optimization claim for these
targets, masks, spacings, and budgets. It does not convert Panel B into a
registration-accuracy experiment, establish generalization, or answer the
reviewer's requested OASIS N=100 analysis.

## Paper-writing guidance

### Claims supported by the accepted experiment

- The supervised fitting result is reproduced across ten OASIS pairs and all
  ten DIR-LAB cases at the stated matched final spacing.
- Distributional summaries can describe consistency, variability, and failure
  cases within these two locked N=10 cohorts.
- Paired per-case comparisons are appropriate because all four methods see the
  same target and mask for a case.
- Panel B provides evidence about representation/optimization capacity for the
  stated target fields and budget.

### Claims not supported

- Do not call Panel B registration accuracy, generalization, or supervised
  model training.
- Do not claim the OASIS ten are an unconstrained simple-random sample or are
  equivalent to the prior N=100 evaluation cohort.
- Do not infer broad superiority from the original three illustrative cases.
- Do not compare unmatched LungCT preprocessing, masks, metrics, or target
  conventions as method-only differences.

### Methods and caption facts

The Methods and caption must state N, case direction, target source, normalized-
grid units, fixed-mask convention, MSE-only fitting, epoch budget, canonical
LRs, alpha for Panel A, and actual spacing. In particular, spell out OASIS
`cps2` and `8-4-2`, and DIR-LAB `cps4` and `32-16-8-4`; do not hide these in a
generic “same hyperparameters” phrase. Report the center/spread convention and
whether curves are casewise summaries or individual trajectories.

### Reviewer-response outline

1. Acknowledge that the old one-plus-two examples were illustrative and state
   how they were chosen.
2. Explain that the revised experiment uses a prespecified shuffled OASIS N=10
   and all DIR-LAB N=10, with paired methods and target conventions.
3. Report the population endpoint statistics and variability after acceptance.
4. State the final-spacing matching and all control-point schedules explicitly.
5. Keep the remaining OASIS limitation explicit: N=10 addresses the
   cherry-picked-single-case concern but does not literally repeat the prior
   N=100 testing cohort requested by the reviewer.

### Reviewer-response draft

> We thank the reviewer for identifying that the original supervised fitting
> figure relied on illustrative cases rather than population summaries. The
> OASIS example M0136→F0173 was seed-3 shuffled rank 9 and had been selected
> post hoc because Greedy achieved the highest Dice among the shuffled first
> ten; DIR-LAB case01 was the initial/easy development example and case08 the
> hardest prior single-stage example. We now state those selection criteria
> explicitly and replace the single-case implication with a paired population
> experiment comprising a prespecified, reproducibly shuffled OASIS N=10
> subset and all ten DIR-LAB cases. For supervised target-field fitting, mean
> rRMSE was 14.04%, 9.90%, 4.03%, and 4.05% on OASIS and 2.76%, 1.50%, 0.537%,
> and 0.525% on DIR-LAB for INR-Dense, INR-BSCP, D-BSCP, and MR-D-BSCP,
> respectively. We also now report the actual final-spacing-matched
> hyperparameters: OASIS single-stage cps2 versus multistage 8-4-2, and
> DIR-LAB single-stage cps4 versus multistage 32-16-8-4, together with the
> learning rates, masks, target units, schedules, and epoch budgets. The OASIS
> revision materially addresses the concern about reliance on one favorable
> example, but it remains N=10 rather than the N=100 cohort suggested by the
> reviewer; we retain this limitation explicitly.

This draft should be adapted to the final figure's display convention after
visualization is designed. Do not add a significance claim unless a paired
analysis is separately specified and reported.

### Limitation to retain

The revised OASIS experiment has N=10, not N=100. It materially improves the
evidence beyond one post hoc example and uses a locked outcome-independent
subset, but the smaller cohort limits precision and does not directly answer
whether identical conclusions hold across the full previous testing set.
