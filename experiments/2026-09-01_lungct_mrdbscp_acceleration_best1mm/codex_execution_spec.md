# Codex execution specification

## Experiment

- Experiment ID: `2026-09-01_lungct_mrdbscp_acceleration_best1mm`
- Objective: compare six MR-D-BSCP acceleration schedules on the best 1-mm
  4DCT and canonical 1-mm COPD cohorts.
- Cases: LungCT cases `01-10` for each dataset.
- Method: direct cubic B-spline control-point optimization, pTV local CC
  (`sigma=2.1 mm`) plus physical isoTV (`weight=0.055`, `csqrt=0.005`),
  five-voxel border, seed `1`, Adam.
- Direction: 4DCT T50/exhale to T00/inspire; COPD eBHCT to iBHCT.
- Data root: `${LUNGCT_PTVREG_ISO1MM}`; raw data remain outside Git.

## Schedule

COPD uses the six frozen five-stage Learn2Breath schedules. 4DCT uses
`full_cps=32,16,8,4` and `downsample=8,4,2,1`. Its original variant is the
exact existing `[400,400,800,900]` 2500-epoch anchor. Accelerated 4DCT
variants collapse L2B stages 64 and 32 into the 4DCT 32 stage by summing caps
and minimum epochs, taking maximum patience, and retaining the finer-stage LR.
The resolved schedule is written into `job_manifest.json`.

## Timing contract

Each dataset/variant cohort is one fresh persistent worker. Case 1 is `cold`
and cases 2-10 are `warm`; each case still starts from zero controls with fresh
optimizer state and the same seed. The primary endpoint is
`loaded_images_to_dvf_file_sec`, from `case_load_complete` to the saved dense
1-mm DVF. CUDA synchronization is performed at in-memory and file-ready
boundaries. Worker queue and scheduling time are excluded.

For timing runs, all optional full-grid diagnostics are disabled. Per-epoch
stage-grid expansion remains required. The expected post-stage materialization
trace is `[false,false,false,false,true]` for COPD and
`[false,false,false,true]` for 4DCT.

## Execution

Use `py310sitreg_v1_test1` with GPUs `0,1,2,3`. Run syntax checks, schedule
unit checks, a dry-run, and one-case smoke checks before the full 120-case
campaign. Runtime outputs and logs belong under the external run root; only
small manifests and review tables belong in this experiment directory.
