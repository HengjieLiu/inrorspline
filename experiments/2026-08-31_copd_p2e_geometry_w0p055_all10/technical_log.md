---
doc_type: technical_log
status: completed_candidate_review
date_planned: 2026-08-31
---

# Technical log: COPD P2e-geometry `w=0.055` candidate

## Known limitations

- This is a COPD-specific regularization candidate, not a reproduction of the
  fixed P2e `w=0.11` reference.
- The `0.055` value was selected by a separate 4DCT screen.  No COPD transfer
  claim is made until all ten cases complete and are reviewed.
- The pTVReg comparison remains protocol-labelled: MATLAB pTVReg and this
  PyTorch direct cubic B-spline/Adam implementation are not a method-only
  ablation.
- GPU `0` is deliberately the only worker.  Times are sequential-run timings,
  not a multi-GPU throughput measurement.
- The aggregate helper reports population SD; any later cross-report aggregate
  must use a clearly labelled SD convention.

## Required checks

- [x] Confirm GPU `0` and the requested container are available: RTX 6000 Ada,
  49,140 MiB total, 5 MiB used before the smoke.
- [x] Compile the existing launcher, runner, and aggregator in the container.
- [x] Run the distinct case-01 five-stage one-epoch smoke with `w=0.055`.
- [x] Validate smoke provenance, final field contract, and finite TRE outputs.
- [x] Dry-run the full `01-10` sequential GPU-`0` launcher.
- [x] Run all ten cases without overwriting P2e outputs.
- [x] Verify all ten summaries and aggregate into this folder.
- [x] Await review before updating another report.

## Smoke result

The case-01 smoke completed in its own output root.  It records
`stage_full_cps=[64,32,16,8,4]`, `stage_downsample=[16,8,4,2,1]`,
`stage_epochs=[1,1,1,1,1]`, `stage_lr=[0.001]*5`, `ptv_isotv_weight=0.055`,
and the requested LCC/isoTV/border settings.  It produced finite native and
registration-grid TRE for 300 landmarks and a finite final dense field with
shape `[275,201,229,3]` (`z,y,x,component`).  The stage role flag was false
for the first four stages and true for the final stage.

Its native snap TRE is intentionally not a candidate result: five total
optimizer steps are only a technical smoke, not the 2,500-step protocol.

## Full candidate result

All ten requested cases completed sequentially on GPU `0`.  Every summary
records `w=0.055`, the frozen P2e control/image/epoch lists, 300 landmarks,
finite headline TREs, and a finite dense field with the expected final grid.

Using the consistent sample-SD (`ddof=1`) convention across ten case means,
the candidate's native snap TRE is `1.0501 +/- 0.7331 mm`.  The frozen P2e
`w=0.11` reference is `1.0505 +/- 0.8088 mm`; the mean paired candidate-minus-
reference change is `-0.0004 +/- 0.1325 mm`.  Three cases improved and seven
worsened.  This result is recorded for review only and has not updated the
cross-dataset comparison report.
