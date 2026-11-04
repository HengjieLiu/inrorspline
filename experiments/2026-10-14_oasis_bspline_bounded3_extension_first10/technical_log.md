# P71 technical log

## Known limitations (read before interpreting results)

- The three-map arm adds representational capacity and receives three complete optimizer schedules at each active level. It is not compute-matched to P70's one- or two-map settings.
- The cohort is the seed-3 shuffled first10, which was used in prior exploratory tuning; it is not an unbiased held-out evaluation.
- Deterministic cuDNN does not eliminate all CUDA backward nondeterminism. Shared-prefix differences across independently restarted depths are diagnostic and must be disclosed.
- The P70 accepted outputs and report are read-only. P71 uses a new extension artifact key and combined report directory.
- GPU occupancy and timing must be sampled live. Existing jobs are not to be stopped, reprioritized, or otherwise modified.

## Execution record and acceptance

- Ran in `py310sitreg_v1_artifact_store` as UID:GID1009:1020, writing under
  `/database/inrorspline`. The host path is the configured external artifact
  store. GPUs0–3 were used one worker/GPU; all ten ranks retained their
  case-centric GPU assignment. GPUs1–2 had high existing utilization, but
  available memory remained sufficient; existing workloads were not altered.
- Targeted repository tests could not use `pytest` because it is absent from
  the container. The three P71 protocol checks were invoked directly and all
  passed. A first launch attempt caught a recursive manifest-builder adapter
  call; a second caught use of the P69 validator on a P70 manifest. Both
  stopped before starting GPU workers. The fixes were applied before manifest
  creation and smoke execution.
- Four full-grid, one-update smokes passed, one for each depth. A later
  production launcher invocation initially tried to recreate the already
  completed smokes and stopped before production. Those smoke artifacts were
  retained, revalidated, and the production queue resumed through the unchanged
  numerical runner; the actual resume command was recorded in the launch
  manifest.
- All40 production registrations completed with zero failures and zero OOMs.
  Maximum depth retains12 maps per registration; across the four depth groups,
 300 independent map checkpoints were produced. All40 independent audits
  passed, reproducing checkpoint/control fields, losses, canonical accuracy and
  regularity metrics, absent-label handling, bounded elementary-map checks,
  and off-grid diagnostics.
- Accepted combined report has28 summary rows,280 per-case rows,400 paired
  candidate/reference differences,900 stage rows, and180 new-arm shared-prefix
  diagnostics. It combines P70's accepted160 cells with40 fresh P71 cells and
  includes original VFA/SITReg references. P70 accepted output hashes were
  rechecked and remain unchanged. All P71 report hashes and row counts passed.
- Bounded-three-map mean±sample-SD Dice/HD95/ASD by schedule: cps8
  0.7169±0.0414 /2.4593±0.3407 /0.6667±0.1241mm; cps8→4
  0.7941±0.0292 /1.9280±0.2888 /0.4314±0.0833mm; cps8→4→2
  0.8244±0.0294 /1.7698±0.3035 /0.3542±0.0801mm; cps8→4→2→1
  0.8322±0.0288 /1.7013±0.3248 /0.3330±0.0816mm. Full-depth
  foreground SD(logJ)/folding is0.7047±0.0404 /0.0209±0.0078%, compared with
  bounded-two-map0.6509±0.0370 /0.0031±0.0013%. The new arm modestly improves
  full-depth accuracy but raises both regularity metrics; report the extra
  representational capacity and optimizer budget, not a bound-only effect.
- Production queue elapsed2429s; independent audit queue elapsed383s. Timing is
  diagnostic under competing GPU workloads. First10 remains exploratory.

## Artifact locations

- Runs and map checkpoints:
  `runs/oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1/`
- Combined tables and report:
  `repo_experiments/oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1/`
