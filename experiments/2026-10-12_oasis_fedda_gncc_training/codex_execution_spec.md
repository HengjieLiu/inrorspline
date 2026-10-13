# P57: FeDDA-1MF signed global NCC training

Approved2026-10-12: four fresh FeDDA GNCC trainings,lambda0.1/exact1over3/1/3
on GPUs0/1/2/3. Each completes automatic best-validation-checkpoint first100
evaluation. Preserve historical sources,outputs,failed preflights and existing
GPU jobs. No new LNCC retraining or GNCC10/range extension.

Only replace similarity: full-volume per-image1-minus signed Pearson
correlation,centered population moments,ordinary autograd,FP32 variance-product
floor1.1920928955078125e-7 before square root. Reuse P56's stateless GlobalNCC.
No squaring,local windows,StableStd,masks,features,loss normalization or DVF
supervision. Retain native unmasked total voxel-field Grad3d(l2),axis/component
means. Training uses TWO sequential directional Adam updates per pair,forward
then reverse with updated weights; never SITReg's jointly averaged update.

Original exp_midl_0407/fedda_1mf,2854079parameters,shared encoder,pyramidal CNN
decoder,use_batch_parallel=False. Not the later101-epoch onepass feature model.
Seed42 fresh initialization,same300 subjects/30 validation pairs,persistent
shuffled loader iterator,batch1,workers0,pinned memory,native stored intensities,
LIA-to-RAS,160x224x192,1mm. Match original model-internal border image sampling,
align_cornersTrue. Adam AMSGrad LR1e-4,betas.9/.999,eps1e-8,wd0,200epochs,
50pairs/epoch,20000updates. No AMP,scheduler,early stopping. Native forward-only
validation; best full-precision validation Dice selects checkpoint and overall
GNCC lambda,never test metrics. Check initial weights and every epoch's pair
stream hashes against all four original LNCC controls.

Retain explicit historical backend override:cuDNNbenchmarkTrue,
deterministicFalse,cudnnTF32True,matmulTF32False. Record residual CUDA
nondeterminism and shared-load timing limitations. No bitwise training claim.

Preflight: verify old certificate/evidence/source/reference hashes; strict
original lambda1 checkpoint load; fresh rank0 field and five-metric parity
under unchanged historical tolerances; native30-pair validation Dice; four
fresh GNCC forward/reverse pairs and checkpoints; repeated warm optimizer
updates and trained-checkpoint gradient step; ranks0/10 canonical DVF
serialization and35 per-label metric rows each. Preserve every failed attempt;
diagnose gate failures rather than silently relaxing or retrying until passing.
Unit tests cover signed/stateless GNCC gradients,actual two-update sequencing,
exact lambdas,new naming,GPU3,legacy compatibility,pair/protocol guards,complete
900-case/14000-label report schemas and correct2x3plot layout.

Independent new adapters/controllers only; historical numerical sources are
read-only. Container UID:GID1009:1020,pin each physical GPU byUUID,one new worker
per GPU. Fresh startup memory gate=max(24GiB,ceil(measured peak reserved+2GiB));
record current GPU utilization/free memory and allocator headroom. Remove old
GPU3 exclusions and obsolete VFA dependencies only from the NEW interfaces.
No automatic retries,precision changes,job termination or reprioritization.

Save GNCC method/protocol metadata in manifests,checkpoints and summaries;
record source/config/data/checkpoint hashes,seed/backend,RNG,loss histories,
initial/pair hashes,best/latest checkpoints and completion/failure markers.
After training export first10,then reuse those field bytes in detailedfirst100
evaluation with fresh inference for ranks10-99. Canonical seed3 ranks0-99,
moving-to-fixed,DVF[3,160,224,192]channel-firstzyxvoxel. Save all35 per-label
Dice/area-weightedHD95/pooled-area symmetricASD and both directionalASDs with
explicit absent-label exclusions. Foreground=fixed raw image>0;canonicalcrop2,
positive-JstdlogJ and J<=0folding percent. Save diffusion/runtime diagnostics.

Report only after all400 new cases/14000 new label rows pass. Compare immutable
LNCC0.1/1over3/1/3/10 (500 reference cases). CSV/JSON/Markdown,mean/sampleSD,
paired equal-lambda differences,first10 separately,validation curves and ARC
PNG/PDF/SVG. ARC2x3:Dice/HD95/symmetricASD versus foregroundstdlogJ;TOP means
only,BOTTOM axis-aligned marginal±1sampleSD ellipses,not confidence/covariance
regions. HistoricalLNCC1 is marked hollow with dashed adjacent segments.

Reference-gate diagnosis record: the first full preflight encountered the known
cross-process benchmark-mode variation and failed tight accuracy parity. A
single bounded non-accepting diagnostic subsequently reproduced the original
field and all five metrics BITWISE/exactly and completed every original gate
with zero issues,including independent native-versus-adapter model equality.
An independent CPU certificate may qualify this saved complete evidence only
after rechecking the exact stored field,all checkpoints,all original tolerances,
native validation,initial/pair hashes,source/dependency hashes and detailed
serialization. It must link both earlier failed attempts,never alter the audit,
and reject merely approximate reproduction or any failed gate. No numerical
tolerance,training/backend setting or test-selected checkpoint is changed.
This establishes adapter correctness,not cross-process CUDA repeatability.

Limitations: equal lambda does not imply matched regularity; changes locality
and squaring together; single seed and previously explored test cohort; historical
LNCC1 is not a fresh matched-seed control; zero measured folding is not a proof
of continuous invertibility. Runtime remains diagnostic under shared load.

Artifacts under ARTIFACT_ROOT/runs/oasis_fedda_gncc_diffusion_training_lambda_sweep_v1,
runs/oasis_seed3_first100_fedda_gncc_training_lambda_eval_v1,and
repo_experiments/oasis_seed3_first100_fedda_gncc_vs_lncc_v1.
