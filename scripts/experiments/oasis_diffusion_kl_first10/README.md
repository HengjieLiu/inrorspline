# First10 diffusion plus analytic KL-C1

This separately authorized cohort adapter preserves the P84/P85 numerical
sources. Use the execution specification in
`experiments/2026-11-10_oasis_first10_diffusion_kl_c1/` and private runtime map.

```bash
python3 -B -m scripts.experiments.oasis_diffusion_kl_first10.controller \
  --container "$CONTAINER" --artifact-host "$ARTIFACT_ROOT" \
  --oasis-root "$OASIS_ROOT_CONTAINER" --prior-root "$P84_ROOT_CONTAINER" \
  --run-id "$RUN_ID" --phase all
```

Preflight runs the26 existing/new checks, audits both historical gamma0 cohort
blocks and replays four native-volume endpoint-gamma smokes. Production uses
all available GPUs with one worker/GPU. Each positive-gamma setting starts in
a fresh worker: cold rank0, then independently initialized warm ranks0..9.
Scoring is a separate same-GPU wave. The fixed matrix has110 new production
registrations plus22 audited historical observations, yielding120 primary
warm endpoints in12 settings. No extra gamma or case-range search is enabled.

`--phase report` rebuilds the complete paired-case analysis without registration.
Generated artifacts, telemetry and reports remain outside the worktree.
