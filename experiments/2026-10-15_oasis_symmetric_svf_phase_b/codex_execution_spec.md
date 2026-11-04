# P78: Rank-0 symmetric B-spline SVF Phase B

Approved 28 fresh fits: baseline, seven amplitude-only, seven balance-only,
nine combined neighbors, three selected-family input swaps, one same-order
repeat of the overall winner. Four separate full-size smokes precede production.
Only seed-3 rank 0, fixed 0333 / moving 0115, cps8->4->2 from identity.

Loss = midpoint two-gradient NCC9 + 2*(R(v1)+R(v2)) + alpha*(E1+E2)
+ beta*(sqrt(E1+1e-8)-sqrt(E2+1e-8))^2. E uses actual dense branch velocities,
native full-volume voxel mean of squared XYZ vector norm in mm, not controls
or loss-grid voxel units. Both new weights are constant across stages.
No mask, endpoint similarity, segmentation loss, or initial-loss normalization.

Alpha: 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2. Beta: .01,.03,.1,.3,1,3,10.
Select each single-term positive weight by original-direction Dice, then ASD,
HD95, then smaller weights. Its centered three tested neighbors (closest three
at boundary) define the 3x3 combined grid. No grid expansion. Each family's
winner is swapped on its original GPU; overall winner is repeated on its GPU.
Selection uses final-field metrics, never segmentation-selected epochs.

Preserve Phase-A joint Adam, normalized twice-branch parameters, half-scaled
actual velocities, exact independent total-SVF knot refinement, native seven
squarings, border sampling and align_corners=True. No stage-map composition.
Cps8/4/2 downsampling 4/2/1, parameter LR .03/.01/.003, max updates 400/800/1300,
min 40/80/130, patience max(25,20/35/50), check/window25, total/data relative
thresholds 1e-4/5e-5. No LR reductions. Joint best total loss including identity.
Seed1, FP32, cuDNN deterministic, benchmark/TF32 off; CUDA sampling caveat stays.

Use isolated Phase-B sources, source/input/reference hashes, source revision,
UID:GID1009:1020. One new worker per available GPU0-3, minimum admission8GiB
or smoke peak+2GiB. Do not alter existing jobs. Preserve failures; one deferred
OOM retry on a sufficiently free slot, keep independent candidates running.
Shared source/storage failures pause admissions. Never accept incomplete sets.

Test native units/reductions, zero-field and nonzero-field gradients/swap,
zero-weight Phase-A parity, CLI/names, selection/neighbors/pinning, history and
checkpoint consistency. Four smokes cover zero, amplitude, balance, combined.
Every production field is independently reconstructed, loss/metrics replayed,
with 7/8/10-step and 100000-point inverse/export diagnostics retained.

Report both-direction canonical five metrics, per-label values, stage loss
components, actual updates/time/memory, shared/opposing RMS, energy balance,
cosines, midpoint coverage/OOB, swap versus repeat discrepancy, Pareto candidates.
Compare fresh/historical A1,A2,A3 and bounded-two/SVF lambda1. No across-case SD.
Artifacts: ${ARTIFACT_ROOT}/{runs,repo_experiments}/
oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1/phase_b_v1.
Phase-A sources/results immutable. No first-ten expansion or additional tuning.
