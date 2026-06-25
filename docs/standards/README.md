---
doc_type: experiment_standards_index
status: active
---

# Experiment Standards

This folder is the preflight review area for fair and reproducible experiments.
Read it before planning runs, launching experiments, comparing methods, or writing
claims from results.

## Required Preflight

Before a real run:

1. Review `fair_experiment_checklist.md`.
2. Confirm the dataset-specific protocol:
   - Brain/OASIS: `oasis_brainmr_protocol.md`
   - LungCT/4DCT/COPD: `lungct_protocol_tags.md`
3. Record the split, manifest, protocol tag, and known limitations in
   `docs/04_experiment_registry.md`.
4. Keep raw data and generated outputs outside Git.

## Standard Files

- `fair_experiment_checklist.md`: reusable fairness checks for all datasets.
- `oasis_brainmr_protocol.md`: seed-3 shuffled OASIS test protocol.
- `lungct_protocol_tags.md`: LungCT preprocessing/protocol tag standard and
  warning links.
- `lungct_tre_metric_conventions.md`: LungCT TRE metric domains, snap/subvoxel
  definitions, and pTVReg comparison convention.
