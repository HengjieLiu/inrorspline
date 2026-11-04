# Rank0 bounded compositional B-splines

Implement the approved single-case pilot: seed3 rank0 fixed0333/moving0115,
one fresh bounded increment per cps8/4/2/1, identity initialization throughout,
stage diffusion weights[1,1,5,1]. No inherited prefix, extra capacity, sweep,
network, bidirectional training, affine prediction, or supervision by labels.

Retain four elementary maps. Each is a cubic spline expanded to the native
grid followed by border-extended trilinear displacement interpolation.
Composition is Phi_previous o delta_new, not displacement addition or repeated
resampling of the accumulated map. Freeze earlier map values but retain
coordinate gradients. New representation is a chain, not a single cps1 spline.

Compute FP64 sampling-aware coefficient bounds for the actual endpoint-aligned
kernel across all phases, cell corners and derivative sign patterns. Test the
same construction against SITReg's half-voxel kernel lookup and native bound
algorithm. Margin0.99. With S=[(W-1)/2,(H-1)/2,(D-1)/2], a=0.99*cps*b/S,
c=a*tanh(p/a), p initially0. This gives derivative1 at zero for normalized
latent parameters; do not apply the old LR directly to dimensionless logits.

Native raw local squared NCC9 + unmasked total-composition evaluation-grid
voxel forward-valid diffusion; factors4/2/1/1, LR0.03/0.01/0.003/0.001,
max/min epochs400/40,800/80,1300/130,2000/300, requested patience20/35/50/100.
Fresh Adam each stage, seed1, FP32, cuDNN F/T, TF32off. Existing25-epoch
checks/median, relative total/data1e-4/5e-5; at cps1 only, two LR halvings
with100-epoch spacing then stop. Lowest stage total loss including epoch0.

Tests cover bound/worst-case patterns, zero/translation/noncommuting composition,
boundary displacement extension, current/frozen gradients, parameter units,
native loss parity, CLI/scope, best-checkpoint logic and output reconstruction.
Full-size one-step-each-stage smoke through canonical metrics gates production.
Require finite fields/checkpoints, exact case/direction, hashes, 35-label audit,
each saved atomic map's full-cell-corner derivative norm<1, and100k seed1
foreground off-grid probes comparing retained chain with interpolated export.

GPU3 sequential smoke/production, UID:GID1009:1020; recheck memory, gate8GiB
or smoke peak+2GiB. Preserve failures and stop for audit. Other jobs untouched.
Record Docker/source revisions, exact bounds/units, composition, saturation,
all histories, controls/latents, stage metrics and best/stopping epochs.
External runs/reports use key:
oasis_seed3_rank000_mrdbscp8421_tanh_composed_lambda1151_v1.

CSV/JSON/Markdown compare new bounded chain with accepted unbounded cps1,
dense[1,1,5,1], original VFA and SITReg: Dice,HD95,symmetric ASD, fixed-raw>0
foreground positive-J SDlogJ and J<=0 folding. Single-case paired changes;
no SD, tuning expansion, or automatic guarantee for the sampled final field.
