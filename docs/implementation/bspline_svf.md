# B-spline stationary velocity fields

P73 optimizes normalized xyz cubic velocity controls, expands to the native
grid and integrates v/128 with seven displacement self-compositions. Sampling
is trilinear, border padded, align_corners=True; coordinates are not clipped.
The velocity exponential has unit derivative at zero, so normalized-coordinate
LRs retain their established local interpretation.

Refined single-SVF controls are prolonged by exact knot insertion. Stage-SVF
composition retains independent integrated maps in previous-o-current pull
order. Only current controls are optimized; frozen map sampling preserves
coordinate gradients. An inverse is approximated by exponentiating negative
velocities and reversing the retained map order.

NCC9 uses the existing image pyramid. Diffusion is explicitly on either
velocity (active increment for composition) or total displacement, in current
evaluation-grid voxel units. Both are forward-valid, component/axis-mean and
unmasked. Equal lambda does not imply matched regularity.

The output of scaling and squaring is a sampled displacement interpolant, not
an exact continuous B-spline deformation. Seven-step outputs remain primary;
eight/ten-step reintegrations, inverse residuals, cell-corner initial-map
derivatives and retained-chain versus exported-field probes are diagnostics.
No unconditional zero-folding guarantee is made.

## Pilot sources and acceptance

The isolated numerical implementation is
`scripts/experiments/oasis_alpha_sweep/bspline_svf.py`; the runner, resource
queue and independent replay are in `run_oasis_bspline_svf.py` alongside it.
P73 completed all eight fits and audits. The executed sources remain frozen
under their launch hashes. The final report required a report-only adapter,
`aggregate_oasis_bspline_svf.py`, to normalize historical `n/a` timing strings;
no optimizer or metric implementation changed. The adapter preserves partial
reports and refuses to overwrite an accepted report.

Tests: `python -B -m unittest discover -s tests -p 'test_oasis_bspline_svf*.py'`.
Artifact key: `oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1`.
The external report includes two12-row comparisons, integration sensitivity,
paired metric differences, and stage diagnostics. See the P73 technical log
for the measured accuracy/folding tradeoff and remaining limitations.
