# Technical log

## Known limitations

- Rank0 is exploratory tuning on a test pair; ranks1–9 are frozen-configuration
  confirmation, not a replacement for a future independent validation study.
- MSE/NCC on learned projections does not reproduce VFA's attention objective.
- Runtime is shared-workload diagnostic only. CUDA grid-sampling backward may
  remain nondeterministic despite cuDNN deterministic mode.
- A failed search is retained and reported; the 96-fit ceiling is not expanded.

## Implementation

- Added a standalone frozen-feature objective/optimizer, preparation/fit worker,
  four-GPU staged scheduler, and CPU contracts. Existing experiment runners
  and outputs are not rewritten.
- Native VFA checkpoint architecture and key prefixes inspected; projection
  weights are [8,16,3,3,3], full-resolution projection enabled.
- Preparation caches identical prefix/feature tensors for all candidate fits;
  raw input normalization differs intentionally between native VFA extraction
  and the historical pair-minmax prefix/control.
- Actual execution checks and progress are recorded in external launcher and
  worker logs; completion is not implied by this implementation record.

## Preflight audit correction

- Eleven new CPU contracts and ten saved-field regression contracts passed.
- First preparation attempt stopped on a redundant FP32 voxel-to-normalized
  round-trip check (1.66148e-6 voxel), not an actual prefix-to-parameter change.
  Corrected the audit to compare the dense clone directly with the converted
  prefix in declared voxel units; retain round-trip quantization separately.
- The failed attempt and log remain immutable. A reviewed preflight retry uses
  a new numbered attempt; production retries remain OOM-only.

## Successful GPU preflight

- Prefix stopping epochs reproduced 125/150/175 and dense clone equality was
  exact. The separate redundant round-trip quantization remained diagnostic.
- Native VFA versus saved reference: voxel RMSE .0002936875, maximum component
  difference .0326843; Dice delta +.0000003090, HD95 delta 0, ASD delta
  +.0000518879. The predeclared RMSE/metric parity gates passed.
- Four one-step feature-order/loss combinations passed full-volume gradients,
  serialized finite DVFs, canonical metrics, and completion validation across
  GPUs0–3. These are smoke artifacts, not production search candidates.
- Eight additional legacy lambda-sweep contracts passed without changes to
  those runners. Pytest is absent in the container; tests use dependency-free
  unittest/function execution instead of changing the container environment.
- Expanded new contracts: 13 passed (31 total including 18 legacy contracts).
- The cached prefix's three terminal total/data/regularizer losses match P38's
  saved history exactly (zero scalar deltas). Its re-evaluated canonical Dice
  is .7587035; keep this actual initialization separate from the historical
  .7607446 MR-D-BSCP reference carried in previous comparison tables.

## Production launch

- Launched 2026-10-07 00:47 UTC in the existing approved container, UID:GID
  1009:1020, GPUs0–3, four concurrent workers. Container image and digest are
  recorded in the production invocation and per-worker manifests.
- Phase A starts with the four order/loss combinations at lambda0/LR.03.
  Active histories show finite decreasing objectives; other GPU jobs remain
  untouched. Launcher PID and current status live in the external artifact root.
- Added and passed an end-to-end synthetic report/holdout-mean test: 14 new
  contracts plus18 legacy contracts, 32 total. Notebook source-only audit and
  git diff whitespace checks passed.
- The detached launcher will perform bounded search, four controls, immutable
  selection, ranks1–9 preparation/refinement and CSV/JSON/Markdown reporting.
  No production accuracy conclusion is recorded before candidate completion.
