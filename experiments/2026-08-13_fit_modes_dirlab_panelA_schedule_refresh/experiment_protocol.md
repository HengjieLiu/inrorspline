# DIR-LAB Fit-Modes Panel A Schedule Refresh Protocol

## Scientific purpose

This experiment replaces only the ten DIR-LAB Panel A MR-D-BSCP diagnostic
records whose historical stage schedule was `200/400/800/1000`. The
camera-ready schedule is `400/400/800/900`, matching the supervised Panel B
stage allocation and totaling 2500 optimization epochs. The experiment asks
whether this schedule refresh changes the registration trajectory or endpoint;
it is not a new method comparison or a control-spacing ablation.

The August 13 schedule-refresh outputs with `400/400/800/900` remain a useful
reproducibility cross-check, but they used
`global_ncc_diagnostic_interval=0`. They cannot source the diagnostic panels
and are not reused here.

## Locked cohort and protocol

| Field | Locked value |
|---|---|
| Dataset | DIR-LAB 4DCT cases `01-10` |
| Direction | moving `T50` to fixed `T00` |
| Image domain | canonical pTVReg landmark crop, isotropic 1 mm |
| Optimization mask | none; full-crop global NCC |
| Evaluation mask | fixed-T00 lung mask |
| Method | MR-D-BSCP |
| Implementation ID | `direct_cp_32_16_8_4_global_ncc_fd_be` |
| Similarity/regularizer | full-crop global NCC + dense forward-difference bending energy |
| Control spacing | `32/16/8/4` voxels on the full 1 mm domain |
| Image downsampling | `8/4/2/1` |
| Epoch schedule | `400/400/800/900` (2500 total) |
| Learning rate | `1e-3` at every stage |
| Alpha | `10000` |
| Seed | `1` |
| Diagnostic | full-crop global NCC at epoch 1 and every 10 epochs through 2500 |
| Diagnostic role | evaluation only; it does not affect training |
| GPUs | `0,1`, one active optimization per GPU with dynamic work stealing |

The expected diagnostic CSV therefore contains 251 finite records. Its final
record must be global epoch 2500, stage index 3, control spacing 4,
downsampling 1, and diagnostic domain `full_1mm_crop`. Panel A uses `1-NCC`,
computed from this diagnostic, as its curve value.

## Execution and storage

Large outputs are written outside Git under:

```text
${RUN_ROOT}/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1
```

The focused launcher is:

```text
scripts/experiments/fit_modes_population/launch_dirlab_panelA_schedule_refresh.py
```

It supports dry-run, isolated smoke, resume, explicit roots, a free-GPU
preflight, non-destructive output handling, and persistent manifests and ETA.
The isolated smoke covers case02 on GPU 0 and case08 on GPU 1 with stage epochs
`1/1/1/1` and diagnostic interval 1.

Compact review artifacts are stored in this experiment directory:

- `job_manifest.csv` and `job_manifest.json`: requested records and commands;
- `provenance.json`: Git revision, dirty paths, source hashes, container, data
  protocol, and launch-time GPU state;
- `monitor_status.md/json` and `launcher_state.json`: recoverable execution
  state;
- `execution_assignments.csv`: actual GPU/PID assignments, runner runtimes, and
  terminal states from the full launch;
- `replacement_index.csv/json`: the accepted ten-row replacement index;
- `case_schedule_comparison.csv/json`: paired old/new and cross-check metrics;
- `population_schedule_comparison.csv/json`: descriptive population summaries;
- `acceptance_report.md/json` and `comparison_report.md`: final validation and
  interpretation.

Paths in committed review artifacts use environment-variable placeholders; no
private host paths belong in paper-facing outputs.

## Compatibility and preservation policy

Resume is allowed only when case, implementation, spacing/downsampling,
schedule, LR, alpha, seed, and diagnostic interval match and both the completed
summary and diagnostic CSV exist. Partial or incompatible paths cause a hard
stop; the launcher never overwrites them. Existing 2400-epoch and August 13
2500-epoch diagnostic-disabled outputs remain immutable.

## Acceptance

Acceptance requires all ten records to pass all of the following:

- completed status and exact locked protocol metadata;
- 2500 stage-history rows and the exact stage schedule;
- 251 finite full-crop diagnostic points with the required epoch set and final
  stage identity;
- finite runtime, peak memory, native-crop snap-to-voxel TRE, foreground
  stdLogJ, and foreground nonpositive-Jacobian percentage;
- a valid paired comparison to the accepted `200/400/800/1000` record;
- protocol-compatible metadata for the August 13 diagnostic-disabled
  cross-check.

Acceptance is technical, not outcome-gated. The comparison reports differences
without rejecting a run merely because a metric improves or worsens. This step
does not modify the main population index and does not regenerate the 20
diagnostic figures; those actions remain camera-ready step 4.

## Paper-use boundary

After acceptance, the new ten records may replace the DIR-LAB MR-D-BSCP Panel A
curves and support a schedule-sensitivity statement. The preserved historical
cohort remains sensitivity evidence. Until step 4 regenerates and audits the
complete figures, neither the old paper raster nor the main population index
should be described as already refreshed.
