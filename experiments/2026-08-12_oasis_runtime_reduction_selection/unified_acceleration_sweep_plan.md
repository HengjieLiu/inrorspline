# Unified OASIS Acceleration Sweep Plan

Generated: 2026-08-12

## Problem

The completed acceleration screens were practical and method-specific. That is
reasonable engineering, but it creates a paper-writing problem: MR-D-BSCP,
D-BSCP, and SINR may appear to use different acceleration tricks rather than a
common acceleration protocol.

The current selected candidates are still defensible because each was selected
against its own full-budget counterpart using the same first10 OASIS protocol
and a common accuracy-degradation readout:

```text
primary grade: same-method Dice degradation
secondary readout: HD95 and ASD degradation
runtime readout: full-pipeline and DVF-only speedup
```

However, for a cleaner revision, the acceleration search should be described or
rerun as a unified schedule template applied to each method family.

## Paper-Ready Paragraph For Current Proposed Selection

> We first performed a method-specific runtime-reduction screen on the OASIS
> seed-3 first10 subset. For each method family, we varied learning rate, fixed
> epoch budget, and adaptive early stopping while keeping the registration
> objective and regularization setting fixed. Candidate accelerations were
> stratified by Dice degradation relative to the same-method full-budget
> baseline: accuracy-preserving (`<=0.02` Dice drop), moderate tradeoff
> (`0.02-0.05`), and failed (`>0.05`). HD95 and ASD changes were reported as
> boundary-distance robustness metrics rather than hard cutoffs. We then
> measured practical deployment runtime separately using a persistent-process
> DVF-only protocol, reporting both cold-start and warm-start time from loaded
> input images to the saved dense displacement field. This screen identified
> `A_fast_decay` for MR-D-BSCP `8-4-2`, `D2_A_lr1e-2` for D-BSCP `cps2`, and
> `S2_F1000_lr1e-3` for SINR `cps2` as the most balanced accelerated candidates
> for first100 validation.

This paragraph is accurate for the current evidence. Its weakness is that it
admits method-specific search choices. That is acceptable if the paper presents
runtime reduction as a pragmatic engineering screen, not as a perfectly
factorized ablation of learning-rate policy.

## Recommended Cleaner Scheme

Use a shared candidate template for every method, while preserving each
method's native baseline LR and representation.

Baseline LR:

| method family | baseline LR |
|---|---:|
| MR-D-BSCP / D-BSCP | `1e-3` |
| SINR | `1e-4` |

The paper should express accelerated LRs as multipliers of each method's own
baseline LR. This avoids making SINR look artificially different because its
absolute baseline LR is smaller.

Unified candidate grid:

| candidate | epoch budget | LR multiplier | early stop | intended role |
|---|---:|---:|---|---|
| `baseline_lr1x_full` | 100% | 1x | none | full-budget reference |
| `F60_lr3x` | 60% | 3x | none | conservative fixed short run |
| `F40_lr10x` | 40% | 10x | none | balanced fixed short run |
| `F25_lr30x` | 25% | 30x | none | aggressive fixed short run |
| `A_bal_lr10x` | max 100% | 10x | min 15%, patience 6%, `min_delta=1e-4` | balanced adaptive run |
| `A_fast_lr30x` | max 100% | 30x | min 8-10%, patience 4%, `min_delta=1e-4` | aggressive adaptive run |

For single-stage cps2/cps4 methods with a `2500` epoch baseline, this becomes:

| candidate | epochs / adaptive settings |
|---|---|
| `baseline_lr1x_full` | fixed `2500` |
| `F60_lr3x` | fixed `1500` |
| `F40_lr10x` | fixed `1000` |
| `F25_lr30x` | fixed `600` to `625`; use `600` if matching prior runs |
| `A_bal_lr10x` | max `2500`, min `375`, patience `150`, `min_delta=1e-4` |
| `A_fast_lr30x` | max `2500`, min `200-250`, patience `100`, `min_delta=1e-4` |

For MR-D-BSCP `8-4-2`, apply the same fractions stage-wise to the baseline
`400,800,1300` schedule:

| candidate | stage epochs or adaptive settings |
|---|---|
| `baseline_lr1x_full` | fixed `400,800,1300` |
| `F60_lr3x` | fixed approximately `240,480,780` |
| `F40_lr10x` | fixed approximately `160,320,520` |
| `F25_lr30x` | fixed approximately `100,200,300` |
| `A_bal_lr10x` | max `400,800,1300`; min about `60,120,195`; patience about `25,50,80` |
| `A_fast_lr30x` | max `400,800,1300`; min about `40,80,130`; patience about `20,35,50` |

For MR-D-BSCP, a decayed stage LR can be treated as a predefined method-native
implementation of the LR multiplier, but it should be described consistently:

```text
stage LR = multiplier-scaled coarse-to-fine LR schedule
```

For example, the already successful `A_fast_decay` uses
`3e-2,1e-2,3e-3`, which is roughly an aggressive coarse-to-fine LR schedule
around the `10x-30x` accelerated range.

## Selection Rule

Use the same selection rule for every method:

1. Compare each candidate only to its same-method full-budget baseline.
2. Assign an accuracy grade by Dice degradation:
   - accuracy-preserving: Dice drop `<= 0.02`
   - moderate tradeoff: Dice drop `0.02-0.05`
   - failed: Dice drop `> 0.05`
3. Report HD95 and ASD degradation for every candidate:
   - do not use HD95 `0.2 mm` as a hard cutoff,
   - use HD95/ASD as robustness readouts and tie-breakers within a Dice grade,
   - flag candidates with large boundary-distance degradation even if Dice is
     still acceptable.
4. Pass the runtime target if:
   - full-pipeline runtime saving `>= 25%`, or
   - clear epoch reduction with no Dice loss and no meaningful HD95/ASD
     degradation.
5. Rank candidates by:
   - Dice grade,
   - lower HD95/ASD degradation within the same Dice grade,
   - lower Dice loss,
   - larger warm DVF-only runtime saving,
   - lower peak memory,
   - simpler schedule for paper readability.

This rule gives a consistent basis for selecting one accelerated candidate per
method while still allowing each method to keep its natural baseline LR.

## Relationship To Current Results

The current results are close to this unified grid but not perfectly named that
way:

| current candidate | unified interpretation |
|---|---|
| MR-D-BSCP `A_fast_decay` | aggressive adaptive, coarse-to-fine LR schedule |
| MR-D-BSCP `F600_decay` | approximately `F25` fixed, coarse-to-fine LR schedule |
| D-BSCP cps2 `D2_A_lr1e-2` | `A_bal_lr10x` |
| D-BSCP cps2 `D2_A_lr3e-2` | `A_fast_lr30x` |
| SINR cps2 `S2_F1000_lr1e-3` | `F40_lr10x` |
| SINR cps2 `S2_A_lr1e-3` | `A_bal_lr10x` |
| SINR cps2 `S2_F1500_lr3e-4` | `F60_lr3x` |

The main mismatch is MR-D-BSCP, where the best candidate uses a decayed
stage-wise LR rather than a single LR multiplier. This is scientifically
reasonable because MR-D-BSCP is multistage, but the paper should state this
explicitly.

## Recommended Revision Strategy

There are two viable paths.

### Option A: Use Current Selections

Use the current first10 screen to choose candidates for first100:

```text
MR-D-BSCP 8-4-2: A_fast_decay
D-BSCP cps2:     D2_A_lr1e-2
SINR cps2:       S2_F1000_lr1e-3
```

Advantages:

- No additional screening needed before first100.
- Current choices are all accuracy-preserving by Dice degradation and have
  useful runtime reduction.
- The most important runtime claim can be made with controlled first100
  validation and DVF-only persistent timing.

Risk:

- The paper must explain that the initial search was method-specific.

### Option B: Rerun A Unified First10 Sweep

Run the unified candidate grid for all target methods before first100:

```text
baseline_lr1x_full
F60_lr3x
F40_lr10x
F25_lr30x
A_bal_lr10x
A_fast_lr30x
```

Advantages:

- Cleaner paper story.
- Reduces reviewer concern that different methods received unequal tuning.
- Produces a single acceleration-selection protocol.

Cost:

- Additional first10 full-evaluation screening before first100.
- MR-D-BSCP needs careful mapping of LR multipliers onto the multistage
  coarse-to-fine LR schedule.

Recommended practical choice:

- Use Option A for the immediate first100 validation if timing is urgent.
- Use Option B if the revision narrative will emphasize runtime optimization as
  a central claim rather than a supporting practical observation.

## First100 Validation After Selection

Once final candidates are selected:

1. Run full evaluation mode on OASIS seed-3 ranks `0-99` for:
   - each full-budget baseline,
   - each selected accelerated candidate.
2. Run persistent-process DVF-only timing on GPU `0`, one method at a time:
   - rank `0` as cold,
   - ranks `1-99` as warm.
3. Report:
   - Dice, HD95, ASD,
   - regularity metrics,
   - full-pipeline runtime,
   - loaded-images-to-DVF-file runtime,
   - cold and warm runtime separately,
   - peak GPU memory mean/std/max and worst-case case ID.
4. Do not mix cold and warm timing into a single headline without also showing
   the separate cold and warm values.

## Peak Memory Reporting

Peak memory should be included as a first-class practical metric. Report it
beside runtime:

```text
training_peak_mem_gb mean +/- std
training_peak_mem_gb max
case ID for max memory
```

Current first10 observed memory anchors:

| method | peak memory GB |
|---|---:|
| MR-D-BSCP `8-4-2` | 3.93 |
| D-BSCP cps4 | 1.75 |
| D-BSCP cps2 | 2.09 |
| SINR cps4 | 2.50 |
| SINR cps2 | 7.63 |

The first100 run should recompute these rather than only reusing first10
values.
