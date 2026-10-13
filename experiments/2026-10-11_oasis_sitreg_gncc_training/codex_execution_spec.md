# P56: SITReg signed global NCC training

Approved2026-10-11: fresh lambda0.1,exact1/3,1,3 on GPUs0,1,2,3 respectively.
After each clean training exit automatically evaluate its best-validation
checkpoint on seed3 first100. Preserve every historical LNCC source/output and
unrelated GPU job. No new LNCC retraining or GNCC lambda10 is authorized.

Only similarity changes: full-volume per-image1-Pearson correlation, centered
population moments, sqrt(clamp(variance product,min=FP32epsilon)), ordinary
autograd, no initial-loss normalization or mask. Refresh both image statistics
for each pair and direction. Keep native total voxel-field Grad3d(l2), unmasked,
component/axis means. Average similarity and diffusion over both directions,
then one joint Adam update. No VFA features, squared correlation, StableStd,
sample-standard-deviation mixing, segmentation supervision or displacement fitting.

Reuse unchanged native SITReg model/preprocessing and old tested loops through
explicit process-local dependency injection, never on-disk modifications.
Seed42 fresh initialization; verify old initial-state hash and every epoch's
pair-stream hash. OASIS300 training subjects/30 validation pairs, native stored
intensities/LIA-to-RAS,160x224x192,batch1,persistent shuffled iterator,workers0.
Adam AMSGrad LR1e-4,betas.9/.999,eps1e-8,weight_decay0,200epochs,50pairs/epoch,
10000updates; no scheduler/early stopping/AMP. Preserve explicit historical
backend override:cuDNN benchmarkTrue/deterministicFalse,cuDNN TF32True,
matmulTF32False. Numerical, not bitwise reproducibility.

Preflight: source/data/reference hashes, strict historical checkpoint inference
and30-pair validation reproduction using previous documented numerical gates;
all four GNCC fresh-Adam/checkpoint smokes; trained-weight GNCC gradient smoke;
canonical serialization/all35label evaluation at ranks0/10. Test Pearson values
and gradients, both argument gradient paths, constants, batch isolation,
statelessness, native joint reduction/optimizer, parsing and legacy compatibility.
Require complete extendedGNCCgate, not merely historicalPASSED.

Train one process per physical GPU pinned byUUID asUID:GID1009:1020. Initial
memory guard at least37GiB and smoke-qualified headroom; allocator cap initial
free minus1GiB. Preserve failures and let other runs finish. No automatic OOM
protocol changes or restart. Freeze hashes for native and new source files.
Record full settings, inputs, backend, GPU state, losses, validation, RNG,
initial/pair hashes, best/latest checkpoints and completion markers.

Select each checkpoint by highest native30-pair validation Dice. Select an
overall GNCC lambda by highest best-validation Dice, never test metrics.
After training export first10 as before, then separate-process detailed100-case
evaluation, preserving first10 field bytes. Canonical nearest-label Dice,
area-weightedHD95/pooled-area symmetricASD at1mm, labels1..35 with explicit
absent-label exclusions, plus fixed-raw-positive stdlogJ/folding and diffusion.
Require400cases/14000newlabelrows and intact checkpoint/field hashes.

Compare existingLNCC0.1,1/3,1,3,10 on identical100pairs; lambda1 is historical,
not a new same-family control. Report per-case/paired deltas,mean/sampleSD,
first10 separately,validation curves and2x3ARC(stdlogJ vsDice/HD95/ASD;topSD
ellipses,bottomlines-only),CSV/JSON/Markdown/PNG/PDF/SVG. No test-selected
epochs or silent range expansion. Boundary validation optimum requires review.

Artifacts underARTIFACT_ROOT/runs/oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1,
runs/oasis_seed3_first100_sitreg_gncc_training_lambda_eval_v1,and
repo_experiments/oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1.
