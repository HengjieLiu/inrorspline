# First100 three-family B-spline842 extension (P79)

This is a cohort/queue/reporting extension, not a new numerical objective.
`oasis_bspline842_first100.py` enumerates2100 cells and1530 missing fits,
checks accepted reports, and records immutable artifact hash/finite receipts.
`run_oasis_bspline842_first100.py` widens case guards process-locally while
retaining P43 displacement, P76 bounded-two and P74 refined-SVF velocity
objectives/optimizers. Source files and previous outputs are untouched.

The P48 100-case dense study contains separately evaluated pure842 prefixes;
400 of those plus30 additional P43 first10 cells provide430 accepted
displacement outputs. P76/P74 contribute70 each. No dense final fields,
fixed-stage-lambda or symmetric two-SVF results enter this comparison.

The new controller subclasses the audited continue-on-error queue: eight
slots, two per GPU, durable launch intents, PID identity reconciliation,
source/storage guards, same-GPU replay, one OOM tail retry and per-case failure
isolation. Numerical failures are never converted to accepted results.

All new fields require independent stage/metric replay. Historical fields
are rehashed and finite-checked without being overwritten. Reports aggregate
only complete case sets; full acceptance requires2100 outputs. Runtime and
memory are separated by historical/new provenance. SVF7 is primary;8/10-step
diagnostics remain secondary. No zero-folding or untouched-test-set claim.

## Worker metadata boundary and recovery

The legacy refined-SVF setup treats every `settings` entry as a filesystem
path. The first extension incorrectly copied P76 controller lists/flags into
that dictionary; both SVF smokes failed before optimization. New manifests use
an explicit four-path allowlist with type checks. Queue options remain outside
worker settings. A regression exercises the actual legacy setup, not only the
argument parser and objective identities.

`--execution metadata_recovery_v2` isolates the repaired run from the original
preflight. The570 previous hash receipts are admissible only with matching
scientific metadata, unchanged historical numerical-source hashes, a bound
original verification marker and unchanged artifact size/mtime. All six smokes
are repeated under the corrected manifest; old successes/failures remain intact.

## Two scoped SVF transition continuations

The completed queue left ranks29/44,velocity-SVF lambda.01 at cps8->4/cps4->2
respectively. Each had one FP32 displacement component exceeding.001 voxel;
ideal FP64 knot/exponential preservation was approximately1e-12 voxel.
Actual rounded-control FP64 field errors were1.23e-4/1.12e-5 voxel, motivating
the explicitly approved v3 limits, not a blanket acceptance of small errors.
New isolated modules retain completed stage checkpoints, preserve old gates
on old prefix transitions, and apply v3 only to unfinished stages in these
two cells. Full-size smokes and original independent replay cover all stages
and7/8/10-step diagnostics. No numerical objective/optimizer changes.

The accepted full report merges only these two independently replayed
continuations with the2098 unchanged rows through a new report-state view.
Historical manifests/state/reports remain unchanged. Resource provenance tags
distinguish retained-prefix optimizer work from continuation worker time.
The separate `render_oasis_bspline842_arc_readable.py` renderer reuses accepted
statistics and the frozen ARC function, wrapping titles only; it writes a
new figure companion and verifies that accepted report files are unchanged.

## Five-family dual-axis ARC presentation

`oasis_first100_five_family_dual_arc.py` combines the accepted P79 three-family
842 sweeps with the accepted full SITReg/VFA NCC9 sweeps:31 settings and3100
paired rows. It excludes the older dense MR-D-BSCP curve and FeDDA, verifies
source hashes, matches DLlambda1 against the P79 references, and reproduces
all310 source mean/sample-SD statistics. DL historical and newer checkpoints
share one family style; source provenance still identifies originallambda1.

Two2x3 exports use foreground SD(logJ) and foreground folding ratio on a
base10-log axis. Folding percent is divided by100 before plotting; there is
no logarithmic averaging or pseudocount. Exact-zero means have hollow markers
at an explicitly labeled display-only floor, one decade below the smallest
positive mean's decade. Arithmetic marginal±1sample-SD ellipses are intersected
with the positive display domain before the log transform, so their displayed
contours need not remain elliptical. They are not confidence/covariance regions.
Top rows zoom the accuracy axes; bottom rows retain all operating points.
Corresponding y limits match across figures and x limits match within each.
