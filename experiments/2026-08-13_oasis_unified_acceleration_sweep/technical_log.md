# Technical Log

## 2026-08-13

### Initial Scope

This experiment screens unified acceleration schedules for OASIS seed-3 first10:

- MR-D-BSCP `8-4-2`
- D-BSCP `cps2`
- SINR `cps2`
- D-BSCP `cps4`
- SINR `cps4`

The primary phase is MR-D-BSCP `8-4-2`, D-BSCP `cps2`, and SINR `cps2`. The cps4 phase is run after the primary phase as context.

### Timing Policy

The launcher uses one persistent worker process per candidate and runs ranks `0-9` sequentially inside that worker. This is intended to expose cold versus warm runtime:

- rank `0`: cold process/GPU timing;
- ranks `1-9`: warm persistent-process timing.

This is not registration warm-starting. Each case still optimizes from its own initialization. Warm means the Python process and GPU library paths have already initialized.

Both timings are retained:

- full registration time;
- loaded-images-to-dense-DVF-file time.

### Known Limitations

- Existing GPU jobs were present when the user approved using all four GPUs. Runtime may be noisier than an isolated final timing run.
- This is still first10 screening. Final paper-facing runtime should be measured again on first100 after selecting the variants.
- Full registration time includes evaluation and output packaging; loaded-images-to-DVF-file time is the deployment-style endpoint.
- Dice degradation grades are selection aids, not proof of equivalence.

### Execution Status

- Plan and launcher prepared.
- Aggregator prepared.
- Syntax, manifest dry-runs, smoke runs, and full launches are tracked in the command outputs and generated `launch_summary` / `aggregation_status` files.
