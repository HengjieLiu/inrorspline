# Technical log

## Known limitations

All ten test cases are exploratory tuning. CUDA grid_sample backward can remain
nondeterministic despite deterministic cuDNN; no bitwise historical lambda1
reproduction is promised. Concurrent GPU jobs make timing diagnostic. Prefix
regularity is fixed; final-stage lambda is the only intended optimization change.

## Audit and implementation

All ten completed pyramid lambda1 prefixes and historical dense lambda1 outputs
were found through P48's state. Numerical source hashes still match the
historical launch. Historical dense optimization averaged58.6s/case, but new
weights can change stopping times. GPUs0/1/3 were idle at the preflight audit;
GPU2 had an unrelated active workload and is excluded.

The legacy FitSpec ties prefix/dense alpha and rejects four new weights. The
new isolated runner fixes this without modifying historical classes/functions:
the prefix always uses a legacy lambda1 spec; the dense objective receives a
separate validated alpha. Launch provenance records all frozen numerical and
new-runner hashes, original prefix stage records and input/output hashes.
All21 tests passed (five fixed-prefix contracts plus16 legacy numerical/native
parity tests). Initial test code incorrectly reused a freed autograd graph when
checking three loss terms; retaining the graph corrected the test, with no
implementation/math change. Two rank0 one-step full-volume smokes were launched
after the CPU gates, with production held pending their successful validation.

Both smokes passed2026-10-13 12:59PDT. Reconstructed cps2 field and detached
dense initialization were exact (max/RMS voxel difference0). Native NCC9 and
diffusion values/gradients all had zero discrepancy. Each used1.63784GiB peak
tensor allocation, retaining the4GiB free-memory admission gate. Canonical
Dice/HD95/ASD were0.804003/2.083035/0.415498 for one-step lambda0.1 and
0.800297/2.048343/0.425679 for one-step lambda5; these are smokes, not production
results. Production launched approximately13:00PDT onGPU0/1/3 after inspecting
both successful smokes. Existing jobs and historical sources remain unchanged.

## Completion and independent acceptance

All 60 production fits completed at 2026-10-13 13:23:32 PDT. The 62 total
attempts include two smokes; none failed or required an OOM retry. Production
stopping epochs ranged from 500 to 900. Mean dense optimization was 43.967 s;
mean total worker time was 61.354 s (about 17.388 s additional loading,
evaluation, and serialization). Timing remains diagnostic under shared load.

The controller wrote CSV/JSON/Markdown and ACCEPTED.json to the external
report directory with the experiment's artifact key. Independent read-only
checks then verified every report hash, 60 candidates, 100 comparison rows,
ten complete ten-case groups, all sample SDs, canonical per-label means and
absent-label consistency, and paired changes versus prefix/fresh lambda1.
All ten fresh lambda1 fields have the same SHA256 as their historical fields;
all five reported metric differences are zero. This is an observed result,
not a general CUDA reproducibility guarantee.

Dense lambda0.5 maximizes mean Dice (0.835772) and minimizes mean symmetric
ASD (0.334519 mm). Lambda1 has the lowest mean HD95 (1.734039 mm) among the
six weights. Stronger dense diffusion reduces regularity costs, but lambda5
still has 0.682494% foreground folding versus VFA 0.218560% and SITReg
0.000423%, while its Dice falls to 0.804058. Fixing the prefix therefore does
not remove the accuracy/regularity tradeoff. These ten cases remain exploratory
test-case tuning, not an unbiased benchmark.
