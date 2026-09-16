# Technical log

## Known limitations

- Diagnostic-only pass; no optimization was rerun.
- The output covers only DIR-LAB case01 and case08.
- Panel A reports final masked NCC endpoints only, not full masked diagnostic curves over all epochs.
- Existing full-crop figures and `camera_ready_source_index.csv` are intentionally unchanged.

## Execution record

- Generated: `2026-08-14 08:54:37 +0000`
- Script: `scripts/experiments/fit_modes_population/recompute_dirlab_lungmask_case_diagnostics.py`
- GPU request: `0,1,2,3`
- Validation pass: `True`
- MR-D-BSCP Panel A schedule: `400,400,800,900` / total `2500`
