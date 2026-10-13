# Frozen VFA features for dense refinement

The standalone `vfa_feature_refinement.py` experiment starts from the unchanged
camera-ready MR-D-BSCP 8/4/2 prefix. It clones the expanded cps2 displacement
into an absolute, full-resolution FP32 xyz voxel parameter optimized by Adam.
There is no cps1 spline representation in this stage.

## Feature semantics

The historical VFA encoder has a 16-channel full-resolution output, and its
first decoder projection is a learned 3x3x3 convolution producing 8 channels.
Encoder and projection weights remain frozen. Native VFA input uses raw
float32 reoriented images, unlike the prefix's pair-minmax image normalization.

Native decoder ordering warps moving encoder features before replicate padding
and projection. The experiment's warp-then-project objective compares the
central projected moving features to the fixed projections. Project-then-warp
is a separately identified alternative, not an equivalent rearrangement.
Every evaluation samples original moving features using the current total
field; initialization already incorporates the prefix exactly once.

VFA uses local inner-product attention and displacement composition. Optimizing
projected-feature MSE/NCC is therefore not an exact reimplementation of VFA's
decoder and need not attain VFA's anatomical accuracy.

## Objective and selection

Primary similarity is full-volume MSE or signed per-channel global NCC loss.
Its value is divided by its fixed initial value (floor 1e-8). The regularizer
is unmasked forward-valid L2 diffusion of the total voxel-unit displacement:
component/spatial mean of squared differences, averaged across spatial axes.
The equivalent unnormalized lambda is `lambda * initial_loss_normalizer`.

Fallbacks use local signed NCC or channel-unit-normalized feature MSE, with
full-volume or fixed raw-image-positive support. Masks never affect diffusion;
segmentations are evaluation-only. The saved VFA field is diagnostic-only.

The optimizer retains the lowest-total-loss state, including initialization.
Rank0 labels choose hyperparameters, not epochs. The selected configuration is
then frozen for ranks1–9. Scientific matching and technical completion are
separate outcomes, and runtime is diagnostic under shared workloads.

The execution specification and bounded search are in
`experiments/2026-10-06_oasis_mrdbscp842_vfa_feature_dense/`.

## Raw-NCC lambda-only follow-up

The separate `launch_oasis_raw_ncc_lambda_extension.py` reuses accepted P40
prefix caches and the unchanged numerical worker at lambda100/300/1000 for all
ten cases, comparing with immutable lambda30 outputs. LR, normalized image NCC,
voxel-unit diffusion, border sampling and stopping/checkpoint selection stay
fixed. An optional worker `--experiment-variant` changes artifact metadata only;
its default preserves the original runner identity.

This explicitly requested follow-up is exploratory tuning across all first10,
including ranks1–9; it must not inherit P40's held-out-confirmation wording.
