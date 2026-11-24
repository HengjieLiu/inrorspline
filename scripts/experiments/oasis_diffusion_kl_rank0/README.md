# Rank0 diffusion1 + analytic KL-C1

Use the execution specification in
`experiments/2026-11-10_oasis_rank000_diffusion_kl_c1/` and private runtime/data
configuration. The adapter preserves the original P84 numerical sources.

```bash
python3 -B -m scripts.experiments.oasis_diffusion_kl_rank0.controller \
  --container "$CONTAINER" --artifact-host "$ARTIFACT_ROOT" \
  --oasis-root "$OASIS_ROOT_CONTAINER" --prior-root "$P84_ROOT_CONTAINER" \
  --run-id "$RUN_ID" --phase all
```

The all phase validates unit checks and reused references, runs four full-volume
smokes, then10 new rank0 cold/warm blocks and separate same-GPU evaluations.
`--phase preflight` stops after validation; `--phase report` only rebuilds reports.
All available GPUs are used, one child perGPU. Scope is hard-limited to rank0;
there is no population queue or gamma-range extension. Source drift or failed
validation stops acceptance. Outputs and all attempts remain external.

After acceptance, run the independent complete-metric/resource review inside
the experiment container, using container-visible artifact paths:

```bash
python -B -m scripts.reports.analyze_oasis_diffusion_kl_rank0 \
  --root "$RUN_ROOT" --review "$REVIEW_ROOT" \
  --out "$REVIEW_ROOT/detailed_review_v1"
```

The review checks source/input/field identities, all per-step loss components,
label eligibility, cold/warm agreement and metric completeness. Its tables
include both foreground/whole-volume geometry, both derivative conventions,
both energy unit systems and all registration/scoring resource scopes.
