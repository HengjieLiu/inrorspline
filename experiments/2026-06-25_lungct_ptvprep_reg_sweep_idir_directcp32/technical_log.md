---
doc_type: technical_log
status: in_progress
date_started: 2026-06-25
---

# Technical Log

## Known Limitations

- This sweep isolates regularization weight only. Learning rates, epochs, stage
  schedule, similarity loss, and preprocessing remain fixed.
- IDIR bending and DirectCP isoTV weights are not on the same absolute scale.
  The sweep is therefore expressed as a multiplicative factor around each
  method's own default.
- The canonical regenerated pTVReg crop uses the corrected pTVReg/MATLAB-origin
  convention, not the old task4-compatible `+1` shifted crop.
- `original_snap` is the primary comparison metric. The other three TRE columns
  are retained for audit and metric-domain diagnostics.

## Verification Checklist

- Launcher dry-run: pending.
- Smoke jobs: pending.
- Full sweep launch: pending.
- Aggregation and Pareto plots: pending.

