# Technical Log

## 2026-08-13

### Scope

This run validates the five selected fast schedules on OASIS seed-3 shuffled first100. Ranks `0-9` are reused from the completed unified first10 acceleration sweep; ranks `10-99` are newly evaluated.

### Runtime Caveat

Runtime is collected for reference only. GPUs were allowed to remain shared with other jobs, so these times should not be used as the final paper benchmark. A separate isolated persistent-process timing pass is still required for final cold/warm deployment timing.

### Selected Schedules

- `MR_A_fast_decay`
- `D2_A_bal_lr10x`
- `D4_A_bal_lr10x`
- `S2_A_bal_lr10x`
- `S4_A_bal_lr10x`

### Status

- Implementation and registry entry prepared.
- Syntax, dry-run, smoke, full launch, and aggregation status are recorded in generated launch and aggregation JSON files.
