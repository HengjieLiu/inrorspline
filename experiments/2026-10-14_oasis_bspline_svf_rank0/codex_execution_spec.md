# P73: Rank-0 B-spline SVF eight-run pilot

Approved matrix: refined single SVF versus retained composition of one SVF per
stage; diffusion on velocity versus total displacement; schedules842 and8421.
All eight registrations start independently from identity on shuffled seed3
rank0, fixed0333/moving0115. NCC9 and lambda1 at every stage. No tanh bounds.

Native-grid scaling and squaring uses v/128 followed by seven self-compositions,
normalized xyz coordinates, trilinear border sampling and align_corners=True.
Refined velocity controls use exact dyadic knot insertion; composed increments
start at zero with earlier maps frozen but coordinate gradients retained.
Velocity regularization uses the active SVF for composition. Displacement
regularization always uses the total deformation. Both use unmasked valid
forward-difference component/axis-mean diffusion in the current loss-grid voxel
units. Image pyramid factors4/2/1/1, normalized LRs.03/.01/.003/.001,
max updates400/800/1300/2000, min40/80/130/300, patience20/35/50/100
(effective at least25), median/check25, relative total/data1e-4/5e-5.
Cps1 allows two halvings at least100 updates apart. Each stage starts fresh
Adam and selects lowest total loss including initialization. Seed1, FP32,
cuDNN benchmarkFalse/deterministicTrue and both TF32 flagsFalse.

Eight full-size one-update-per-stage smokes precede production. One family per
available GPU0–3, paired depths sequential on that GPU, UID:GID1009:1020.
Admission max(8GiB,smoke reserved peak+2GiB); existing jobs unchanged. One OOM
retry after a slot release on the GPU with most free memory; numerical failures
stop the affected run for audit. Histories and failed attempts are preserved.

All generated files are external under ARTIFACT_ROOT/runs or
ARTIFACT_ROOT/repo_experiments with key
oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1.
Every run saves controls, stage fields, provenance, histories, and completion
markers. Independently replay every stage/checkpoint and final canonical metric.
Reintegrate saved velocities at8/10 steps without optimization; report metrics,
inverse residuals and changes from the primary7-step export. Audit initial
scaled-map cell-corner derivatives. Composed arms retain the100k foreground
off-grid chain-versus-export Jacobian diagnostic. Report folding even if nonzero.

Two schedule tables compare four new arms with the historical prolonged
displacement, five P70/P71 composition settings and original VFA/SITReg.
Report five accuracy/regularity metrics, paired differences, update/runtime
diagnostics and integration sensitivity. Require eight complete replay-audited
fields before acceptance. Do not substitute higher-step diagnostic fields.

## Known limitations

Single exploratory case, no across-case SD/generalization claim. Equal lambda
does not match velocity/displacement regularization strength. Native integration
followed by trilinear loss-grid sampling differs from historical direct cubic
coarse-grid evaluation. FP32 CUDA grid_sample backward may be nondeterministic.
SVF exponentiation does not guarantee zero discrete or off-grid export folding.
Historical outputs are read-only; source hashes include dirty-tree source bytes.
