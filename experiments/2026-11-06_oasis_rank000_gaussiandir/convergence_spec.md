# P83 paper-loss convergence, speed and GPU-memory study

Status: protocol executed; both convergence trajectories reached the time limit
without operational plateau. Original pilot artifacts remain immutable. Final
report, tables, figures and executed notebook are external under
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_convergence_v1/execution_20261106_v1/report_v1/`.
LNCC9²/diffusion and dense-gradient optimization are deferred.

## Scope

Seed-3 shuffled rank0, fixed0333/moving0115, native160×224×192 atlas grid.
Original historical SITReg lambda1 is the immutable accuracy/regularity reference.
No fitted parameters, labels or SITReg field initialize or guide registration.
Input verification is on CPU; only images and fixed-positive mask enter CUDA.
Use sampled signed NCC plus TV, journal lambda8. Released and corrected variants
retain their distinct reductions/adaptation, so this is not a representation-only
ablation. The joint polish is an extension of the explicit paper reconstruction.

The journal describes2,000 updates; the release has no automatic early stopping.
DIR-Lab2.5s and supplement5.1s on RTX4090 are not OASIS speed targets. The release
timer excludes setup, periodic validation and final save without CUDA synchronization.

## Matrix and schedules

`protocol_convergence_v1.json` layers over unchanged `protocol_v1.json`:
seed6, K20, nominal20,000 queries (5,000 centers plus three positive-axis
neighbors), raw fixed-positive center sampling, native resolution, existing
initialization/caps/thresholds/group LRs. Preserve short shuffled-epoch tail batches.

- Each variant: fresh200/500/2000-step schedules, three same-seed warm timing
  repeats sequentially on one empty GPU. Synthetic generic kernel warmup has no
  registration update or reusable parameters. Timing SD is not scientific SD.
- Each variant: additional fresh-process2000-step run with no generic warmup.
- Each variant: one fresh2000-step trajectory then refinement, maximum1200s.
- Short control: one stage. Corrected:100+100 or250+250, frozen coarse and zero
  fine residual. Warmup5/13, stage-local cosine. Control adaptation15%…60% every5%;
  corrected every10% of stage excluding final. Thresholds.002/1e-7 unchanged;
  invalid populations fail, with no protection added silently.
- Trajectory snapshots0/25/50/100/200/500/1000/2000, every500 refinement updates
  and LR/stopping events. Retain selected and terminal full-state checkpoints.

## Refinement and stopping

Preserve the2000-step endpoint. Freeze clone/prune; learn geometry/motion and
jointly unfreeze BOTH corrected scales. Fresh Adam at10% native LRs,50-step
warmup, then plateau reductions (no cosine).

Two fixed independent monitoring banks:16 batches each,20,000 queries per batch,
same center/positive-neighbor grouping and per-batch signed NCC/TV. Bank centers
use uniform replacement sampling; training remains shuffled sampling. Local RNGs
never consume training RNG or alter gradients/adaptation statistics. Probe8192
fixed foreground points. Evaluate at refinement0 and every100 updates.

Halve LR if primary best relative improvement over500 updates is below1e-4,
at least500 updates between reductions, maximum four. Denominator is
max(abs(previous score),1e-8), because NCC is negative. Operational plateau:
at least two reductions then another500 updates with both-bank relative score
change AND improvement below1e-4 and probe vector RMS change below.01mm.
Select lowest-primary-bank state; retain terminal. Labels do not select/stop.
All work establishing stopping is charged, even when the chosen state is earlier.

1200s includes initialization, optimization, transitions, required monitoring,
control/instrumentation and full DVF/warped-image generation. Exclude and measure
snapshot/serialization/posthoc work. Reserve1.25× measured anchor output time
before another update or monitor; report operation-level overshoot. If criteria
are unmet, `convergence_unresolved_at_time_limit`; fixed-budget completion is
separate. Lower sampled loss/plateau need not improve Dice or imply an optimum.

## Resource accounting and outputs

Synchronized wall and CUDA-event intervals for setup, initialization, optimization,
monitoring, full usable-output generation, analysis/storage. CUDA events may
include GPU idle gaps during CPU work; synchronized wall is primary. Warm latency
begins after inputs/provenance are ready, before model/sampler creation. Fresh
process latency additionally includes imports, verification and provenance I/O.
Nonphase control overhead and snapshot overhead are explicit.

Reset/aggregate allocated/reserved peaks per exclusive phase, absolute and above
input-loaded baseline. Sample process NVML memory every100ms including context
and non-Torch allocations; sampled peaks are lower bounds. Keep raw samples and
competing PID records. Record initial/final/peak node/parameter counts, GPU UUID,
image/runtime, requested/effective benchmarkFalse/deterministicTrue/TF32False.
An occupied GPU blocks the next timing run. SITReg saved-field loading is never
reported as inference; no unverified SITReg runtime row.

All35-label Dice/HD95/symmetric+directional ASD and exclusions; full/foreground
Jacobian/folding/positive-J SDlogJ, diffusion, BE, NDV, image/field fidelity from
the existing evaluator. Score snapshots after registration. Tables and figures
show accuracy versus optimization time and usable-output latency, loss/monitor/
LR/population/memory traces, repeats/stopping reasons and descriptive frontier.

Run root: `${ARTIFACT_ROOT}/runs/oasis_rank000_gaussiandir_convergence_v1/`.
Review root: `${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_convergence_v1/`.
Generated Markdown/figures/tables/executed notebook remain external. Source
notebook is output-free. Use isolated runtime/image in [runtime.md](runtime.md).
No rsync, base-package changes, new dataset, dense benchmark or lambda sweep.

Gates: numerical/native tests, joint checkpoint continuation, frozen populations,
exact transitions, independent bank RNG, shortened/anchor schedules, plateau and
improving-run timeout tests, phase memory accounting; native adaptation/two-scale/
joint/output smokes and frozen-source receipts before production.
