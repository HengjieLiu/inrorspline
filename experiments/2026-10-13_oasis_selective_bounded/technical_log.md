# P66 selectively bounded chain log

## Known limitations

Unbounded earlier maps can fold; bounding later increments does not establish
invertibility of the chain. Fixed-foreground folding may still change as
regions move. Compare the three variants and P64 as compositional ablations;
P63 historical unbounded and P62 dense use a different prolonged-prefix
representation and are contextual comparisons. One exploratory rank0 case,
no tuning or first-ten expansion; shared-load timing and CUDA backward
nondeterminism limitations remain.

## Preflight

- User requested last1/last2/last3 bounds for8/4/2/1. Retain all stage
  weights[1,1,5,1], schedules and retained-map composition; unbounded c=p.
- Current GPU3 free48515MiB, used4MiB, utilization0%. Other jobs untouched.
- Verified existing image hengjieliu96/dl_dir:py310sitreg_v1, ID
  sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8.
- All29 tests passed:6 new selective-bound tests,7 unchanged P64 tests and
  16 unchanged NCC9/diffusion tests including native-source gradient parity.
  Tests verify identity control transform when unbounded, no accidental tanh
  during serialization, bounded optimizer/checkpoint parity, correct stage
  masks/names and persistence of earlier folding through a bounded identity.
- All three full-size four-stage smoke tests passed, including identity
  initialization, checkpoint reload, native NCC9/diffusion parity, canonical
  metrics, bounded atomic corner checks and same-point100k off-grid diagnostics.
  Peak allocated memory2.678GiB; admission gate remains8GiB. Historical
  numerical files unchanged. Production cleared sequentially onGPU3.

## Production results

All three production fits completed without failures/retries. Report acceptance
requires all three and verifies independently run common unbounded prefixes.
Canonical rank0 results (Dice,HD95mm,ASDmm,FGSDlogJ,FGfold%):

| Bounded stages | Dice | HD95 | ASD | FG SDlogJ | FG folding % |
|---|---:|---:|---:|---:|---:|
| cps1 only | 0.789160087 | 2.113887896 | 0.466397090 | 0.610573312 | 1.021627579 |
| cps2/1 | 0.801810057 | 1.975920038 | 0.422405406 | 0.668009156 | 1.621929031 |
| cps4/2/1 | 0.792171765 | 2.023935376 | 0.448346296 | 0.562963885 | 0.378766055 |

- Last2 has the best three accuracy metrics among new variants, and better
  rank0 accuracy than both original VFA and SITReg. Folding1.621929% remains
  far above VFA0.255035% and SITReg0.000697%; not matched topology.
- Last3 improves all five metrics versus last1 on this case. It has the least
  folding of new variants but remains above both DL references. All three
  improve accuracy versus P64 all-bounded, which retains0% measured folding.
- All three share exactly the same unbounded cps8 result: Dice0.705031,
  folding0.543802%. Last1/last2 also share cps4: Dice0.776915,fold2.225994%.
  At cps2 lambda5, unbounded last1 reaches Dice0.768895/fold0.961737%, whereas
  bounded last2 reaches0.781435/1.566220%. The accuracy/regularity divergence
  is already present at cps2, before the shared bounded cps1 refinement.
- Increasing the number of bounded final stages does not monotonically reduce
  measured folding. Low positive-J SDlogJ also does not imply few nonpositive
  Jacobians: folding is a separate diagnostic, excluded from positive-J logs.
- All enforced atomic derivative checks passed. Retained/export off-grid
  nonpositive percentages are last1:1.516/1.183, last2:2.041/1.799,
  last3:0.407/0.387, at the same100k P64 locations. Canonical finite-difference
  folding is a different measurement, not replaced by these probe results.
- Optimization66.468/64.501/64.727s; total worker124.711/125.385/117.303s;
  each peak2.678GiB. All final stages stopped500 with two LR halvings, best
  epochs497/492/500. Timings are diagnostic under shared load.
- Comparison.csv,stages.csv,results.json,report.md and ACCEPTED.json saved
  externally; historical sources/outputs unchanged.

## Independent acceptance

- Independent replay passed2026-10-14 00:08:23UTC (2026-10-13 17:08PDT).
  All12 stage checkpoints reconstruct their saved controls and fields exactly,
  with selected objective terms, all five metrics and35 valid label results
  reproduced. Directional and symmetric ASD label values checked to1e-12.
- All12 full native-cell-corner audits reproduce; the bounds apply only to
  bounded stages. Folded unbounded maps and total compositions remain in the
  accepted data, with no filtering or topology-repair step.
- Off-grid100k diagnostics replay exactly at the same P64 points. Additional
  128-point FP64 coordinate finite differences agree with chain autograd;
  max errors2.202e-9/1.689e-9/2.789e-9 for last1/2/3.
- Common unbounded prefixes match EXACTLY: cps8 across all three and cps4
  across last1/last2, including field hashes, stopping/best epochs and loss.
  Source, historical-reference and accepted-report hashes are unchanged.
- External run root preserves independent_audit.json with audit-source hash,
  per-stage checks and common-prefix diagnostics. All requested work complete;
  no further lambda/LR/capacity variants or cohort expansion executed.

## Reproduction

Use the verified existing container as1009:1020; configure ARTIFACT_ROOT,
SITREG_ROOT, DL_NCC_CODE_ROOT and VFA_ROOT from private local paths. Export
INRORSPLINE_CONTAINER_IMAGE and INRORSPLINE_CONTAINER_IMAGE_ID with the
verified image, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=3,
MPLCONFIGDIR=/tmp, PYTORCH_KERNEL_CACHE_PATH=/tmp and disable bytecode writes.

```bash
python -B -m unittest discover -s tests -p test_oasis_selective_bounded.py -v
python -B -m unittest discover -s tests -p test_oasis_bounded_composed.py -v
python -B -m unittest discover -s tests -p test_oasis_dl_ncc_diffusion.py -v
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_selective_bounded.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 5 1 --bounded-last 1 2 3 --stop-after smoke
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_selective_bounded.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 5 1 --bounded-last 1 2 3
python -u -B scripts/experiments/oasis_alpha_sweep/audit_oasis_selective_bounded.py \
  --artifact-root "$ARTIFACT_ROOT"
```

All generated checkpoints, fields, histories, logs, metrics and reports live
under the distinct external P66 key. Accepted outputs are not overwritten;
incomplete worker directories require audit before any retry.
