# OASIS rank0 parameter scaling

Seven direct-DVF families, two saved targets, six parameter caps, native and
joint multiresolution fitting:168 independent20-minute endpoints. Full image
box objective; final accuracy is reported in physical millimeters.

- [Execution specification](codex_execution_spec.md)
- [Configuration](config.json)
- [Technical log](technical_log.md)
- [Output-free notebook source](review.ipynb)
- [Implementation notes](../../docs/implementation/oasis_fieldfit_parameter_scaling.md)

Active execution:`execution_20261106T0644Z`, launched2026-11-05 22:43:11PST.
External root:
`${ARTIFACT_ROOT}/runs/oasis_rank000_fieldfit_parameter_scaling_v1/execution_20261106T0644Z/`.

Live status:`queue_state.json`. Live review:`reports/report.md`,
`reports/results.csv`, `reports/results.json`, `reports/full_box_curves.png`.
The final external `reports/review_executed.ipynb` contains inline figures and
tables. Interim files explicitly retain pending/missing/unresolved statuses.

Resume the same controller only after confirming no controller holds its
`queue.lock`; the launcher enforces an exclusive lock and verifies frozen source
receipts. Do not launch another host against this run. Historical outputs and
P81 sources remain unchanged. New experiments require a new execution directory.
