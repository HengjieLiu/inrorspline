# Codex Execution Specification

## Scope

Implement, smoke-test, launch, monitor, validate, and compare exactly ten
DIR-LAB Panel A MR-D-BSCP records at the camera-ready
`400/400/800/900` schedule. Do not run other methods, Panel B, OASIS, or a
spacing ablation. Do not update the main population index or figures.

## Required implementation

1. Use the existing LungCT `run_one.py` implementation ID
   `direct_cp_32_16_8_4_global_ncc_fd_be`.
2. Add a focused launcher under `scripts/experiments/fit_modes_population/`
   with `--dry-run`, `--smoke`, `--resume`, explicit roots, locked GPUs `0,1`,
   free-GPU preflight, one process per GPU, dynamic work stealing, portable
   manifests, persistent monitor state, and non-destructive reuse behavior.
3. Add a focused aggregator that enforces the protocol and curve contract,
   creates a ten-row replacement index, and compares against both the accepted
   2400-epoch population records and the August 13 2500-epoch
   diagnostic-disabled records.
4. Record the Git commit, dirty source paths and hashes, Docker image, dataset
   convention, command, GPU assignment, runtime, and peak memory.

## Ordered execution

1. Compile-check the launcher and aggregator in the project Docker container.
2. Run a full dry-run and require exactly ten unique pending or compatible
   records.
3. Confirm GPUs 0 and 1 are free immediately before execution.
4. Run isolated smoke jobs: case02/GPU0 and case08/GPU1, `1/1/1/1`, diagnostic
   interval 1. Inspect summaries, histories, curves, masks, units, and finite
   outputs.
5. Launch all ten full records and poll at intervals no greater than 60 seconds.
6. Require ten successful terminal records and no unresolved failures.
7. Aggregate and validate. Update registry/checklist only after acceptance.

## Commands

Inside `py310sitreg_v1_test1`, from `${REPO_ROOT}`:

```bash
python scripts/experiments/fit_modes_population/launch_dirlab_panelA_schedule_refresh.py --dry-run
python scripts/experiments/fit_modes_population/launch_dirlab_panelA_schedule_refresh.py --smoke
python scripts/experiments/fit_modes_population/launch_dirlab_panelA_schedule_refresh.py
python scripts/experiments/fit_modes_population/aggregate_dirlab_panelA_schedule_refresh.py
```

Use `--resume` only for exact-compatible completed outputs after an interrupted
launcher session. Use a new versioned root for any incompatible output.

## Stop conditions

Stop rather than overwrite if an output path exists but is incomplete or
incompatible. Stop the full launch if a smoke record is nonfinite, has incorrect
metadata, lacks its expected diagnostic curve, or shows an unexpected output
schema. Report any GPU contention, process failure, or protocol mismatch
immediately.
