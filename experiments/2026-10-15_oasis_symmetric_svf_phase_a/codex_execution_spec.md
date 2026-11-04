# P77: Rank-0 symmetric refined-SVF Phase A

## Approved scope

Four fresh identity registrations only: A1 independent total velocities with
2*(R1+R2), A2 with R1+R2, A3 tied v1=w/2 and v2=-w/2 with R(w), and an
input-swapped A1 check. Seed-3 rank 0 is fixed 0333, moving 0115. Schedule
8->4->2; no cps1, dense field, extra increments, amplitude or balance sweeps.
Report branch energies before requesting permission for later phases.

## Numerical contract

Each branch retains one total cubic B-spline SVF, prolonged by exact knot
insertion. Do not compose pyramid stages. Independent parameters q1/q2 are
normalized XYZ coefficients of twice the actual branch velocities; v_i is
BSpline(q_i)/2. Tied control optimizes one normalized w coefficient tensor.
P_i=exp(v_i), integrated at native 160x224x192 with seven squarings, time 1,
border trilinear displacement sampling, align_corners=True, unclamped coords.
Moving A and fixed B are both warped from their original stage images to the
common grid. Forward pull is P1 o exp(-v2); reverse is P2 o exp(-v1).

Use a separate differentiable two-input implementation of existing NCC9:
1-mean(correlation^2), mean window kernels, zero statistical padding and
FP32-epsilon product clamp. Both image arguments receive gradients. No fixed
moment cache, intensity normalization, masks, endpoint loss or labels in loss.
Diffusion is unmasked forward-valid component/axis-mean in current loss-grid
voxel units. A1 coefficient 2 is equal-split-equivalent lambda 1, not a claim
of universal regularization equivalence. A2 literal coefficient 1 has equal-
split-equivalent lambda 0.5. Tied A3 equals R(w) algebraically.

| CPS | Downsample | Parameter LR | Max/min | Requested patience |
|---:|---:|---:|---:|---:|
|8|4|0.03|400/40|20|
|4|2|0.01|800/80|35|
|2|1|0.003|1300/130|50|

Joint Adam on the pair, fresh state per level; no alternating branch updates.
Median/check 25, effective patience at least 25, relative loss/data thresholds
1e-4/5e-5, no LR reductions. Select a joint lowest-total-loss checkpoint,
including initialization. Seed 1, FP32, TF32 off, cuDNN deterministic on and
benchmark off. Grid-sample backward may remain nondeterministic.

## Verification and execution

Unit-test NCC value/both-gradient parity and swapping, quadratic diffusion
identity, native-mm magnitude reductions, independent/tied shapes and unique
names, Adam checkpoint selection, knot insertion, translations, noncommuting
composition/inverse order, boundary sampling and gradients to both branches.
Four full-size smokes (one update per stage), with independent checkpoint,
history, serialization and canonical-metric replays before production.
Preserve historical sources and results. Use existing precision-aware SVF
transition validation on actual half-scaled coefficients, not a widened test.

One worker per available GPU 0-3, UID:GID 1009:1020. Admission at least 8 GiB
or smoke reserved/allocated peak +2 GiB. Existing jobs remain unchanged.
Record current GPU load, source/input/reference hashes and image revision.
Preserve failures; do not silently repair numerics or aggregate incomplete
groups. No later-phase optimization is authorized by this Phase-A approval.

## Reporting

Require all four registrations and independent replays. Seven-step final
forward and reverse DVFs are primary. Reintegrate 8/10 steps diagnostically;
do not replace primary outputs. Audit inverse residuals on 100000 seed-1
foreground points in each direction, in-bounds coverage, exported versus
retained composition, branch scaled-map derivatives and measured folding.
Report all five canonical metrics on full maps, per-label metrics, paired
forward changes against saved P76 bounded-two/P74 velocity-SVF lambda 1.
Single-case results have no across-case SD. Historical references are read-only.

Record per-stage and final E_i=mean_vox(sum_xyz(v_i_mm^2)), RMS magnitudes,
diffusion energies, relative imbalance, common-mode RMS and branch mean
translations. Use the same native full grid for both branch magnitudes;
foreground-union summaries are diagnostic only. Record midpoint foreground
coverage, out-of-bounds fractions and image moments to expose common drift.
Report joint optimizer updates, optimizer/worker/replay time, allocated and
reserved memory, 7/8/10-step differences and input-swap discrepancies.

External artifact key:
oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1/phase_a.
All generated checkpoints, logs, fields and CSV/JSON/Markdown are external.
