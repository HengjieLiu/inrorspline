# Technical Log

## Known Limitations

- This is a selected-case diagnostic: OASIS rank 9 and LungCT cases 01/08 only.
- LungCT final metrics from the current registration runner are cropped 1mm
  registration-grid TRE diagnostics. The pTVReg teacher fields are validated
  against saved original/native snap TRE, but this run does not yet reconstruct
  every method's original/native snap TRE for headline comparison.
- Checkpoint displacement logging uses deterministic sampled foreground points,
  not full dense checkpoint fields.
- The optional SINR CP-kernel follow-up is gated on small-plan results and
  requires a separate runner before it can produce the heavier `J J^T`
  diagnostics.

## 2026-06-29 Preflight

- Added opt-in sampled checkpoint logging to OASIS and LungCT registration
  runners.
- Added `scripts/experiments/update_alignment/launch_update_alignment_overnight.py`
  for dry-run, smoke, full launch, aggregation, plotting, report generation,
  and follow-up gating.
- Native compile passed for the modified runners and launcher.
