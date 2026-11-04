# P80: Full100 matched-prefix cps8421 sweep

Approved matrix: ranks0–99 of seed3 OASIS, displacement / bounded2 / refined
SVF-velocity, lambda.01/.1/.3/1/3/10/100. All2100 accepted P79 prefixes retained;
2100 fresh cps1 outputs,2800 new optimizer blocks. No historical8421 final reused.

Use immutable P43/P68/P74 numerical objectives/optimizers. cps1 LR.001,
max/min2000/300, patience100, median/check25, relative thresholds1e-4/5e-5,
two halvings100 updates apart. Fresh Adam per new map. Seed1 FP32, deterministic
cuDNN, benchmark/TF32off. Fixed raw-positive metric foreground. Bounded2 adds
two bounded identity maps to six frozen maps; refinedSVF prolongs total velocity,
seven native integration steps. Same global lambda at all stages/maps.

New version4 cps2->1 validator: exact prescribed controls, original velocity
allclose; FP32 field max.002/RMS1e-5 voxel always. If original field allclose
fails, require idealFP64 velocity1e-9/field1e-8 and actual-rounded field
max2e-4/RMS1e-6. No FP64 optimization. Displacement retains max1e-4/RMS1e-5.

Nine rank0 fullsize smokes atlambda.01/1/100, independent replay, CPU contracts
and queue fault injection gate production. Two slots/GPU0–3, original prefixGPU
affinity,1009:1020, max(8GiB,smokepeak+2) admission, existing jobs untouched.
Cell failures deferred; oneOOM tail retry; shared source/storage failure pauses
admissions. Durable journal/process identity/restart adoption and inactivity /
hard-phase guards. Audit and reporting errors cannot restart accepted fits.

Every fit saves checkpoint/history/provenance/field/metrics and needs independent
replay. Bounded8-map corner/offgrid100k audit; SVF7/8/10 reintegration/inverse
diagnostics. Nonzero folding reported, not rejected as an outcome.

External artifact key:
`oasis_seed3_first100_bspline8421_matched_prefix_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1`.
Only complete100-case groups aggregate. Require2100 accepted for final acceptance.
CSV/JSON/Markdown, allfive metrics mean/sampleSD, matched depth differences,
resources, failure ledger, two2x3 ARC plus common-axis842 companions. Time for
newcps1 and retainedprefix separately; their sum is not fresh end-to-end time.
First10 exploratory tuning caveat retained. No auto-expansion or tolerance edits.
