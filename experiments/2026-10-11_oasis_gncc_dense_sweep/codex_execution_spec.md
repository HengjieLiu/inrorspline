# P55: signed/squared GNCC full-resolution dense sweep

Approved 2026-10-11. IDIR is deferred. Artifact key:
`oasis_seed3_first10_mrdbscp842_dense_fullres_gncc_signed_squared_diffusion_v1`.

140 registrations: two losses x lambda .01/.1/.3/1/3/10/100 x seed3 first10.
Fresh zero cps8 start, exact cps8->4->2 prolongation then exact detached dense
clone. Each candidate uses its own loss/lambda at every stage; no old prefix reuse.
Full-resolution raw images, border sampling, align_corners=True, unmasked
total voxel-unit forward-valid diffusion, component/axis means. Normalized XYZ
parameters and unchanged P43/P44 Adam LR .03/.01/.003/.001; budgets
400/800/1300/2000, minima40/80/130/300, effective patience25/35/50/100.
Check/window25, total/data relative thresholds1e-4/5e-5; dense only two LR
halvings; best-total-loss state (including initialization) at each stage.

Global correlation uses centered population moments over all voxels, covariance
divided by sqrt(clamp(variance product,min=FP32 epsilon)). Losses1-rho and
1-rho^2 use ordinary autograd, not StableStd or sample variance. No loss-scale
normalization, masks, features, BE, segmentation supervision or DVF supervision.
Seed1 FP32, cuDNN False/True, TF32off; CUDA sampling backward may be nondeterministic.

Preserve historical sources/results. Reuse and validate the70 accepted full-loss
LNCC9 dense outputs and20 original VFA/SITReg evaluations. Unit value/gradient
contracts, three one-step-per-stage smokes (two GNCC plus LNCC control), and a
rank0/lambda1 full LNCC reproduction gate precede production. LNCC reproduction
requires matching stopping/best epochs; max voxel field error<=.01, Dice<=1e-4,
HD95<=.01mm, ASD<=.001mm. Failures are retained; gates are not silently relaxed.

Read-only gradient calibration evaluates GNCC, GNCC2 and LNCC on common identity,
saved LNCC lambda1 cps2, and saved LNCC dense fields for all10 cases. Record raw
losses, voxel-field gradient RMS, local/global ratios and gradient cosines.
Calibration is diagnostic only and cannot change the prescribed raw lambda grid.

GPUs0-3 one added worker/GPU, UID:GID1009:1020, preserve other jobs; measure fresh
memory/utilization and qualify memory from smokes. Retain failures, one OOM retry
after a worker releases a GPU, choosing the most-free suitable GPU. No automatic
boundary expansion: report boundary winner and request approval.

Require140 finite native-shape dense fields,4900 label records, correct metadata,
source/input/output hashes, schedules and completion markers. Aggregate only
complete accepted cohorts. Save checkpoints/histories/stage fields and canonical
Dice/HD95/symmetric ASD, fixed-raw-positive stdlogJ/folding and diffusion.

Report three NCC sweeps grouped by loss and lambda and a selected-loss table;
highest mean Dice then lower ASD/HD95. Main ARC figure2x3: top marginal +/-1
sample-SD ellipses, bottom lines only; columns Dice/HD95/ASD, common foreground
stdlogJ x-axis, paired panel limits, VFA/SITReg references. Ellipses are axis-aligned
descriptive SDs, not confidence regions. PNG/PDF/SVG, CSV/JSON/Markdown. Include
folding companion and eight-loss context from read-only accepted P44 results.
All first10 remain exploratory test tuning; runtime is shared-workload diagnostic.

Generated material stays under ARTIFACT_ROOT/{runs,repo_experiments}/<key>.
