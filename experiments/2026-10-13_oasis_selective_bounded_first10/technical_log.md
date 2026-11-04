# P67 technical log

## Known limitations

- Rank0 was used for exploratory selection; report ranks1–9 separately.
- Later bounded maps cannot repair every fold from earlier unbounded maps.
- Retained-composition topology and exported-grid finite differences differ.
- Shared GPU load makes runtime diagnostic, not a clean timing benchmark.
- CUDA grid_sample backward can remain nondeterministic.

## Preflight

- Approved extension:36 fresh registrations plus4 immutable rank0 results.
- User approved one new worker per available GPU0/1/2/3.
- Checked current free GPU memory: approximately43–47GiB; existing jobs retained.
- New adapter calls P66 numerical objective/optimizer unchanged. All4 bounded
  uses the same transformed controls as P64; parity tests gate execution.
- No generated outputs in the Git worktree; external reports only.

## Verification and launch

- All23 bounded-composition unit/regression tests passed, including all4
  parity at every stage and exact sample-SD/complete-cohort contracts.
- Historical accepted rank0 reports, source guards, current raw inputs and
  all ten pair IDs passed preflight. Source revision and individual hashes
  are saved in the external launch manifest.
- Rank1 last1/all4 full-size one-step smokes gate the36-run four-GPU queue.
  Prior rank0 outputs remain read-only references, not new initialization.
- Both full-size smokes passed. Additional NCC9/diffusion regression suite:
  16 tests,15 passed/1 skipped; combined38 passed/1 skipped.
- Production started on GPU0/1/2/3 with ranks1/2/3/4 respectively. Each
  case's four variants stay on one GPU; dynamic next-case assignment follows
  released slots. Existing processes unchanged, smoke-based admission8GiB.
- A separate read-only audit will reload all160 stage checkpoints and replay
  all40 final per-label/accuracy/regularity results before handoff.

## Reproduction

Use the established Docker environment as1009:1020 with external
ARTIFACT_ROOT, the locally audited SITREG_ROOT, image tag/digest provenance,
MPLCONFIGDIR and PYTORCH_KERNEL_CACHE_PATH pointing to writable temporary
directories; disable bytecode writes. Data paths come from accepted manifests.

```bash
python -B -m unittest discover -s tests -p 'test_oasis*bounded*.py'
python -B -m unittest discover -s tests -p test_oasis_dl_ncc_diffusion.py
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_selective_bounded_first10.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpus 0,1,2,3
python -u -B scripts/experiments/oasis_alpha_sweep/audit_oasis_selective_bounded_first10.py \
  --artifact-root "$ARTIFACT_ROOT"
```

Accepted reports are immutable. Incomplete attempts require an explicit audit
before retry; a failed worker does not trigger modification of existing jobs.

## Completed cohort

- All36 fresh registrations and4 reused rank0 results passed. No failures
  or retries; all40 common-prefix comparisons have identical DVF hashes.
- Peak allocated memory2.67849GiB; mean fresh worker133.79s, including
  metrics/topology/serialization. Shared-load timing is diagnostic.
- Last2 has highest Dice in8/10 cases; last1 in ranks1/2. Last2 has the
  best mean Dice/HD95/ASD among the four variants but highest mean folding.
- All4 has0% measured foreground folding in every case; this does not
  establish an unconditional exported-interpolant topology guarantee.
- Maximum saved bounded-elementary corner row norm0.9897624<1.
- The CSV/JSON/Markdown report passed complete-cohort/source/reference/hash
  and roundtrip checks.

## Independent acceptance

- All160 saved stage checkpoints reconstructed their controls and dense
  fields exactly in separate processes; selected total/data/regularizer
  values agree within frozen tolerances.
- All40 final Dice/HD95/symmetric ASD and foreground regularity metrics
  independently reproduced to1e-12, including every per-label metric and
  absent-label result. No discrepancies, failures or retries.
- Audit used each result's original physical GPU, one audit worker per GPU;
  source/reference/accepted-report checksums unchanged. Machine-readable
  independent_audit.json and per-GPU logs are retained under the run root.
- Complete handoff: four variants on the same ten cases, mean/sample-SD and
  per-case tables, matched VFA/SITReg references, paired differences, stage
  histories/checkpoints, topology diagnostics and reproducible sources.
