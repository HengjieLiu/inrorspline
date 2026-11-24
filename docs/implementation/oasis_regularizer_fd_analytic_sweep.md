# OASIS FD/analytic diffusion, BE, and KL-C1

See [Jacobian SD(logJ), folding and NDV explained](jacobian_sdlogj_folding_and_ndv.md)
for the detailed metric definitions, central versus analytic distinction and
count-versus-volume interpretation.

The isolated `scripts/experiments/oasis_regularizer_sweep/` adapter reuses the
existing NCC9, continuous endpoint-aligned spline expansion, dyadic knot
insertion and stage-local plateau definition. Historical numerical modules
remain unchanged. See the source-only execution specification in
`experiments/2026-11-10_oasis_regularizer_fd_analytic_sweep/`.

`losses.py` implements the supplied Leow-style C1 continuation at epsilon0.05,
kappa120 with a separate loss weight. Its guarded raw branch has no inactive-log
NaNs, and the continued branch retains negative-J gradients. The FD determinant
matches the existing central-difference/crop2 evaluator while retaining autograd.
Analytic first/second derivatives use the continuous cubic basis; input-coordinate
chain factors and component scaling express derivatives in each loss grid's
voxel coordinates. Full-resolution derivatives reuse the native separable kernels.

The six arms differ only in active regularizer/derivative backend. Diffusion
retains component/axis means; BE doubles mixed terms. Analytic diffusion/BE use
voxel-center means; FD uses existing valid stencils. KL uses the same crop2 for
both backends and det(I+Du) of the total field. None is an invertibility guarantee.

`worker.py` starts every registration from identity. A fresh setting process runs
rank0 cold followed by ranks0-9 warm, without intervening metrics or cache clearing.
It records stage-local total-loss checkpoints, objective histories, time windows,
baseline/peak allocated/reserved memory and independent saved-field/loss replay.
`controller.py` admits one child per GPU, binds same-GPU evaluators, identifies
host/container PIDs and samples CPU-only NVML/RSS telemetry. `report.py` separates
paired rank0 cold/warm gaps, warm cohort statistics, all label/regularity metrics
and incomplete cohorts. Generated outputs and host mappings stay outside Git.

Validation commands (inside the configured Docker runtime):

```bash
python -B -m unittest discover -s tests -p test_oasis_regularizer_sweep.py -v
```

Full-volume six-arm smokes, independent replay, PID/memory telemetry, source
freeze and storage checks must pass before coarse production. Test/launch results
are recorded in the experiment technical log and external execution receipts.

## Jacobian-convention finding from the accepted first10 study

The posthoc paired review found a large discretization discrepancy for KL-FD:
at its selected weight0.03, mean foreground central folding is0.430% versus
analytic voxel-center folding4.021%. Analytic-KL at the same weight gives
0.507%/0.553%. Keep both conventions and NDV when interpreting Jacobian-trained
fields; the historical canonical metric remains unchanged. The discrepancy is
consistent with stencil sensitivity, not proof of a particular spatial mechanism
or continuous invertibility between the sampled locations.

The separate metadata-only paired analyzer is
`scripts/reports/analyze_oasis_regularizer_fd_analytic.py`. It snapshots external
tables, verifies frozen sources/input identity and complete cohorts, pairs cases
before bootstrapping, checks identical-label sensitivity, and separates endpoint
time from update counts and scoring memory. Its generated reports stay external.
