# P44 technical log

## 2026-10-08: approved implementation

- Implemented a separate native-loss adapter, independent four-stage worker,
  four-GPU scheduler, strict report generator and unit contracts.
- Reused P43 numerical geometry/optimizer and canonical metrics without editing
  them. Source and reference drift checks protect existing outputs.
- Feature-MSE strict full-model checkpoint load precedes frozen projection+block0
  extraction. Native MIND/Feature-MSE fixed descriptors are cached only; moving
  descriptors follow image warp. Native value/gradient parity is a smoke gate.
- GPU preflight: GPU0/2/3 approximately47.4GiB free; GPU1 approximately37.4GiB
  free with99% existing utilization. No existing job altered.
- Container image identity confirmed: existing py310sitreg_v1 artifact-store
  image. Exact image identifier, GPU state and private paths belong in manifests.
- All49 OASIS regression tests passed, including native NCC/diffusion parity,
  complete-cohort/sample-SD validation and six ARC/folding export checks.
- Initial full-geometry MIND/RC smokes passed; NGF and Feature-MSE exposed missing
  MONAI/PyYAML imports before optimization. Failed attempts and successful initial
  smokes remain unchanged in the top-level run root. Production was not launched.
- Added isolated PyYAML6.0.2/MONAI1.3.2 under the experiment artifact directory,
  not the shared Python environment. Package versions/PYTHONPATH recorded.
- NGF constructor explicitly receives native-training mm_spacing=1; the class
  default None leaves its spacing attribute unset. Third-weight test tolerates
  floating-point multiplication versus division at1ulp, with no grid rounding.
- Corrected-environment smoke/production use distinct `verified_environment_v1`
  subdirectories under the same variant run and review roots.
- Corrected-environment gate: all five full-volume smokes passed. Native similarity
  value differences0, gradient maximum difference at most5.83e-11; diffusion parity
  passed. Exact cps2→dense clone differences0. Prolongation maximum difference
  1.67e-6voxel. Canonical metrics/finite DVF serialization/completion checks passed.
- Smoke peak allocated GiB: MSE1.43, MIND4.73, RC1.47, NGF1.77, Feature-MSE5.56.
  Feature-MSE strict checkpoint loaded; frozen first-level encoder has14256
  parameters and16 full-resolution output channels. No network parameter gradients.
- MIND constant-zero diagnostic returns nonfinite, matching native behavior;
  actual OASIS smoke is finite. Nonfinite production objectives will fail visibly.
- Released the production scheduler after the successful gate. It revalidates
  smoke/source/reference records before starting350 production fits. Status and
  attempt logs are under `verified_environment_v1/launcher_state.json` and `logs/`.
- Production started2026-10-09 00:12UTC (2026-10-08 17:12PDT). Scheduler PID31439;
  first four container worker PIDs31599/31607/31615/31623 on GPUs0/1/2/3,
  respectively, for MIND center λ.04 ranks0/1/2/3. Four added GPU workers verified
  active; pre-existing GPU1 workload remains present. Full350 aggregation pending.

## 2026-10-09: completion and line-only ARC export

- All350 new candidates completed; no production failures or OOM retries.
  Final candidate finished07:47:13UTC; full report accepted07:50:04UTC
  (00:50:04PDT). Combined report contains350 new fits,70 immutable NCC fits and
  20 original VFA/SITReg evaluations:44 ten-case cohorts.
- User requested the six loss-sweep curves without ellipses. Added a separate
  presentation-only script, `scripts/reports/oasis_dl_loss_lines_only.py`, leaving
  frozen numerical/report sources and existing outputs unchanged.
- Plot x-axis remains foreground positive-J SD(log J); y panels are Dice,
  HD95 and symmetric ASD. Six plain piecewise-linear curves connect the seven
  measured lambda settings; VFA/SITReg are original-field reference points.
  No ellipses, error bars, sweep markers, smoothing or casewise selection.
- Gate verifies accepted counts, exact rank/subject/direction matching, full-
  resolution dense NCC selection, finite case metrics and all44 mean/sample-SD
  records against440 case rows. Plot source/input/output hashes are recorded.
- New outputs are under the accepted review directory's `lines_only_v1/`.
  Scientific limitations remain: exploratory first10 tuning and positive-J-only
  SD(log J), which does not replace the separately reported folding statistics.
- Verification: three new unit tests passed, including rejection of incomplete,
  duplicate, nonfinite, wrong-case and wrong-NCC-branch inputs. Full accepted-data
  export passed and all44 mean/sample-SD records reproduced; source hashes
  remained unchanged. PNG/PDF/SVG and plotted-data CSV generated; PNG visually
  inspected for six lines, two baseline points per panel and no ellipses.
