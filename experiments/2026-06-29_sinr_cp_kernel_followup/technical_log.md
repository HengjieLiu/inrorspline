# Technical Log: SINR CP-Kernel Follow-Up

## Known Limitations

- The overnight continuation uses a sampled dense-update proxy rather than the
  full autograd CP-kernel diagnostic requested in the original source prompt.
- `D_U proxy` compares cumulative sampled SINR and DirectCP dense updates.
- `R_U proxy` is the residual after fitting one scalar from DirectCP to SINR.
- `A_U proxy` is the sampled dense-update norm ratio, SINR over DirectCP.
- The full `K = J_theta J_theta^T`, power iteration, and tiny-step image-loss
  sanity checks remain future work if this proxy suggests the lower-level
  mechanism is worth isolating.

## Execution Notes

- Uses the same first three diagnostic cases as the update-alignment run:
  OASIS rank 9 and LungCT cases 01/08.
- Reads source checkpoint samples from
  `/database/inrorspline/runs/update_alignment_overnight_20260629_v1`.
- Writes review artifacts under
  `experiments/2026-06-29_sinr_cp_kernel_followup`.

## Exploratory Role-Loop Extension

- The P5 full run completed, but the automatic promising gate did not pass.
- The user requested continuing the follow-up as exploration for at least four
  hours, with explicit Scientist and Reviewer roles.
- Communication rounds are recorded in `scientist_reviewer_log.md`.
- The exploration now emphasizes whether LungCT case08 failures are explained
  by representation capacity, image-registration optimization path, or
  multistage capture range.
- The first new schedule-ablation family removes or weakens the image-pyramid
  downsampling in DirectCP `32,16,8,4` while preserving control-grid staging.
  This is intended to separate control-stage prolongation from coarse-to-fine
  image-loss capture range.
- Smoke status, 2026-06-29: four `1,1,1,1` stage-epoch no-pyramid jobs
  completed with no failures. Case01 finished around 4 minutes/job; case08
  finished around 5.5-6 minutes/job. The reviewer decision is to avoid all10
  no-pyramid expansion for now and instead run a time-budgeted case01/case08
  diagnostic set with `25,25,25,25` stage epochs.
- A sidecar monitor was added after launch because the controller aggregates
  after the 10-job schedule completes. The monitor refreshes partial artifacts
  only after completed schedule rows appear and writes
  `exploration/live_monitor_status.json` for in-progress review.
- First-phase schedule-ablation result: all 10 targeted case01/case08 jobs
  completed with no failures. Case01 is competitive without the image pyramid
  (`1.076` mm best reg1mm snap TRE), but case08 remains poor under no-pyramid
  control staging (`6.478` mm best reg1mm snap TRE after the LR probe) and
  partial `4,2,1,1` image pyramid (`11.937` mm best reg1mm snap TRE).
- Second-phase direction: launch case08 standard `8,4,2,1` pyramid probes to
  test whether the missing ingredient is the full standard image-pyramid
  capture range, the objective, LR, or modestly longer schedule length.
- Second-phase result: the standard-pyramid probe completed `4/4` with no
  failures. The best short case08 row was pTV LCC + isoTV with standard
  `8,4,2,1`, `25,25,25,25` stage epochs, and stage LR `0.01`, reaching
  `2.270` mm reg1mm snap TRE. Standard pyramid with LR `0.001` remained poor
  (`11.496` mm pTV, `11.361` mm FD-BE), and pTV LR `0.001` with
  `50,50,50,50` stage epochs reached `8.896` mm. This supports a
  capture-range-plus-optimizer-scale interpretation: the full image pyramid is
  necessary for large recovery, but LR/runtime still matter.
- The case08 ablation plot now includes only metric-matched reg1mm rows. The
  prior full-schedule DirectCP row remains in the Markdown table with its
  native/crop TRE convention explicitly tagged.
- Active refinement launch: after the standard-pyramid LR `0.01` recovery, a
  case08 capture/refinement probe was launched under
  `/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1/capture_refinement_probe`.
  It tests standard `8,4,2,1` LR `0.01` with `100,100,100,100` and
  `200,200,200,200`, plus partial `4,2,1,1` LR `0.01`
  `100,100,100,100` and no-pyramid LR `0.01` `100,100,100,100`.
  This separates pyramid extent from longer runtime at the recovered LR.
- Refinement result: completed `4/4` with no failures. All LR `0.01`
  e100/e200 controls converged near `1.21` mm reg1mm snap TRE:
  standard `8,4,2,1` e100 `1.213`, standard e200 `1.217`, partial
  `4,2,1,1` e100 `1.219`, and no-pyramid e100 `1.218`. This revises the
  short-budget interpretation: the image pyramid improves early capture under
  small budgets, but the apparent hard-case pyramid requirement disappears
  under sufficient LR/runtime in this diagnostic case.
