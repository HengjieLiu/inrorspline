# Full-resolution native DL-loss refinement

P44 adds MSE, MIND, Residual Complexity, NGF and frozen FEDD Feature-MSE to the
8→4→2→dense protocol. Native wavereg implementations are imported, source-hashed
and checked against adapter values/gradients. Geometry, adaptive Adam, image
preprocessing and canonical metrics reuse P43 unchanged.

See [execution specification](../../experiments/2026-10-08_oasis_fullres_dl_loss_sweep/execution_spec.md)
for weights, schedules, DL anchor limitations and acceptance criteria.

Implementation entry points:

- `scripts/experiments/oasis_alpha_sweep/dl_loss_sweep.py`: protocol, native adapters,
  full-resolution objective and logging around the unchanged P43 optimizer.
- `run_oasis_dl_loss_one.py`: fresh independent four-stage registration.
- `launch_oasis_dl_loss_sweep.py`: five-smoke gate, four-GPU scheduler, immutable
  reference validation and complete-only aggregation.
- `scripts/reports/oasis_dl_loss_first10_arc.py`: 350 new registrations plus70
  historical NCC and20 original DL rows, ARC/folding figures and tables.
- `scripts/reports/oasis_dl_loss_lines_only.py`: presentation-only three-panel
  Dice/HD95/symmetric-ASD versus foreground SD(log J) replot with six plain
  sweep lines and two reference points, no uncertainty glyphs. Reproduces all
  accepted cohort means/sample SDs from case rows; preserves existing reports.

Important: Feature-MSE encodes an already-warped raw image with the frozen FEDD-A
first-level encoder. It is not the historical VFA projected-feature experiment.
Effective lambda and actual regularizer coefficient differ for MIND and RC due
to native similarity multipliers. Optimization is unmasked at full resolution.
