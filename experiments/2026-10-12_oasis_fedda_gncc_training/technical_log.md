# P57 FeDDA GNCC training technical log

## Limitations

- Global signed versus local squared NCC changes both locality and squaring.
- Equal lambda does not ensure equal effective regularity.
- Historical LNCC1 is not a fresh matched-seed replication.
- Native autotuned CUDA backend is not bitwise reproducible; prior field and
  HD95 reproduction failures remain documented in P51. Do not erase or silently
  relax failures. Shared GPU load makes runtime diagnostic.
- Single seed,previously explored cohort; no statistical superiority claim.

## 2026-10-12

- User approved the audited four-GPU GNCC0.1/1over3/1/3 sweep,automatic detailed
  first100 evaluation and2x3ARC(top lines,bottom SD ellipses).
- Read-only audit verified complete historicalLNCC runs/report hashes,unchanged
  native sources and dataset manifest. All four baseline initial hashes and
  every epoch's pair-stream hashes agree. Original200-epoch FeDDA-1MF protocol
  retained;two sequential updates/pair,20000updates total.
- Initial GPU free memory38.6/47.4/47.4/47.4GiB;GPU0 has unrelated active work.
  Existing jobs remain untouched. Historical peak reserved21.22461GiB is
  contextual; fresh GNCC smoke will set the production memory gate.
- New process-local adapters,full-cohort report and tests implemented. Old
  sources and accepted outputs are untouched. Verification/preflight pending;
  no production launched yet.
- All30 CPU tests passed:6 new FeDDA GNCC contracts,12 legacy FeDDA contracts,
  and12 signed-GNCC/SITReg contracts. The full synthetic report test validates
  900 case rows/14000 label rows and top-lines/bottom-ellipses layout.
- Preflight_attempt1 failed before model construction: YAML was absent from
  the inherited SITReg-only PYTHONPATH. Preserve its logs/failure. Restore the
  existing P44 runtime_pydeps (PyYAML6.0.2/MONAI1.3.2) alongside P45 pystrum0.4;
  no installation or shared environment changes. Fingerprint the isolated
  package metadata and imported dependency files in new manifests/gates.
- Preflight_attempt2 reached strict reference inference but FAILED the unchanged
  metric gates. Fieldmax0.01064014/RMS0.00012475voxel passes field bounds; however
  Dice delta+0.00006081508 exceeds0.00005,HD95delta+0.00306322926mm exceeds0.001,
  and ASDdelta+0.00012961760mm exceeds0.0001. The earlier P51 autotuned-backend
  variability is relevant context,not permission to waive these gates. No
  production launch and no tolerance change. All failed artifacts retained.
- Added a separate non-accepting diagnostic to complete all four GNCC sequential
  update/memory/checkpoint smokes and native validation,then compare independent
  legacy/GNCC model instances in the same process and finish ranks0/10 detailed
  serialization. This diagnostic explicitly cannot create GNCC_PASSED or qualify
  production; its results will determine whether new approval is needed.
- The single bounded diagnostic completed all original gates with ZERO issues.
  It reproduced the historical rank0 field bitwise and all five canonical
  metrics exactly. Native30-pair validation Dice0.81410727649 differs from
  checkpoint0.81410958819 by-0.00000231170. Independent native/GNCC-adapter
  models with the same checkpoint also yielded bitwise-identical fields.
- All four fresh GNCC two-update/checkpoint smokes,6-update warm-state smoke,
  trained-checkpoint two-update smoke and ranks0/10 detailed canonical metrics
  completed. Warm peak reserved21.07421875GiB. No OOM or nonfinite gradients.
- As in the original P51 FeDDA qualification, a separate CPU certificate will
  verify this complete zero-issue evidence without another stochastic inference.
  It requires exact stored-field/native-adapter agreement and every original
  validation/update/metric criterion; merely close or failed audits are rejected.
  The diagnostic remains non-accepting/unchanged; failures remain linked. This
  satisfies the original numerical criteria without tolerance/backend changes
  and does not assert fresh-process repeatability. Certification is pending.
- Independent CPU certification completed successfully under
  `preflight_certified_v1/`. It revalidated all original numerical criteria,
  exact stored-reference field,exact independently constructed native/adapter
  fields,all four checkpoints and detailed serialized metric records. Both
  earlier failures remain linked and untouched. Maximum measured reserved
  memory21.216796875GiB; certified production gate24GiB.
- All32 CPU tests passed (8newGNCC/qualification,12legacyFeDDA,12signedGNCC).
  New qualification tests explicitly reject any failed gate,merely approximate
  field,wrong directional-update count or changed backend. git diff--check clean.

## Production launch — 2026-10-12 01:12:09PDT

- Detached controllerPID95152 launched all four approved workers concurrently:
  GPU0/lambda0.1/PID95160,GPU1/lambda1over3/PID95163,
  GPU2/lambda1/PID95166,GPU3/lambda3/PID95175.
- Launch memory:GPU0 free39513MiB with unrelated work;GPUs1/2/3 free48515MiB.
  No existing process/priority was changed. UID:GID1009:1020;UUID pinning and
  historical backend recorded. Both existing isolated dependency roots remain
  read-only in PYTHONPATH; no shared package installs.
- All four initialization hashes match the historicalFeDDA seed42 state:
  3d53895f2ddc0fcc339870c7635eac2328a567c6dbb78eaf5985a7c90729d6e4.
  At startup verification,workers completed17/41/43/36 pairs respectively
  (34/82/86/72Adamupdates),all finite and exactly matching originalLNCC pair IDs.
  No production failure/OOM markers; controller heartbeat is current.
- Each worker automatically exports its best-validation first10 fields,then
  performs detailedfirst100 evaluation. Final400-new/500-reference comparison
  and top-lines/bottom-SD-ellipsesARC runs only after all four clean completions.
  Numerical sources are now hash-locked for the training/evaluation/report chain.
