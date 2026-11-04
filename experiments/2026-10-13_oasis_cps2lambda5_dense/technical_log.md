# Technical log

## Known limitations

Exploratory first-ten test-case tuning, not an unbiased benchmark. Deterministic
cuDNN does not guarantee deterministic CUDA grid_sample backward. Runtime is
diagnostic under shared load. Lower prefix folding need not survive refinement
at lower diffusion weights. Prior [1,1,1] dense comparisons change initialization
and are not same-protocol reproducibility tests.

## Audit and implementation

P61 provides ten accepted [1,1,5] source fields with saved cps2 best controls.
The separate runner imports unchanged numerical functions and validates P61
directly; it does not misuse the legacy tied-global-lambda prefix loader.
Complete source acceptance/checksums, prefix stage records, pair/input hashes,
and source revisions are recorded. Dense cloning is exact and cps2 reconstruction
must meet established voxel tolerances. Historical sources and outputs remain
read-only. Preflight GPUs0/1/3 were idle with approximately48.5GiB free;GPU2
had an unrelated workload and is excluded.

All four new tests and16 historical numerical tests passed. Both rank0
one-step smokes (lambda0.1/10) passed, including canonical metrics. Cps2
reconstruction and detached cloning had zero voxel differences; native NCC9
and diffusion values/gradients also matched exactly. Peak tensor allocation
was1.63784GiB, retaining the4GiB free-memory gate. Production launched after
these checks on2026-10-13 approximately14:45PDT, GPUs0/1/3.

## Completion and independent acceptance

All70 production fits completed2026-10-13 15:15:01PDT. All72 attempts including
two smokes succeeded; no failures or OOM retries. Dense stopping epochs ranged
500–900. Mean optimization/total worker time was52.3798/68.9123s; the16.53s
difference covers loading, reconstruction, metrics, serialization and overhead.
Shared-load timing remains diagnostic.

Independent checks passed for accepted report hashes, all70 candidates,
110 comparison rows and11 complete ten-case groups, sample SD, canonical
per-label aggregation and absent-label consistency, frozen [1,1,5] stage
records and prefix hashes, paired deltas versus prefix/dense1, and all60
same-dense-weight differences against the prior [1,1,1] initialization.

Dense lambda0.5 wins all three mean accuracy metrics: Dice0.834955,
HD951.681926mm, symmetric ASD0.331114mm. Foreground SDlogJ/folding is
0.819045/3.879464%. Against its unrefined [1,1,5] prefix, changes are
+0.037233 Dice, -0.323666mm HD95, -0.119300mm ASD, +0.353611 SDlogJ and
+3.382283 folding percentage points. A more regular initialization therefore
does not preserve low folding under weakly regularized dense optimization.

At dense lambda0.5, changing the prefix from [1,1,1] to [1,1,5] gives
-0.000817 Dice, -0.056620mm HD95, -0.003405mm ASD, -0.011555 SDlogJ and
-0.583381 folding percentage points. All six shared dense weights reduce
folding, but other metrics have mixed changes. Lambda10 reaches0.024494%
folding with Dice0.773654, a substantial accuracy sacrifice. The selected
lambda0.5 point exceeds original VFA/SITReg mean accuracy but still has much
higher folding and larger across-case accuracy SDs. No significance or
unbiased-test claim is made.

Complete per-case CSV, mean/sample-SD CSV, JSON, Markdown, paired prior-prefix
CSV and checksummed ACCEPTED.json are under the external experiment artifact
key; historical reports remain unchanged.
