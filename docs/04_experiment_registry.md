---
doc_type: experiment_registry
status: active
---

# Experiment Registry

Every real run should get an entry before launch. Large artifacts belong under `SCRATCH_ROOT`, not Git.

## Required Fields

```text
Run ID:
Status:
Owner/agent:
Date planned:
Date started:
Date completed:
Server:
GPU model/count:
Git commit:
Branch:
Docker image tag/digest:
Dataset root:
Dataset manifest:
Split config:
Standards preflight:
Protocol tag:
Command:
Output root:
Primary metrics:
Known limitations:
Next decision:
```

## Planned Priority Runs

### P0: Scaffold Only

- Status: complete when root scaffold lands.
- Protocol tag: `repo_scaffold_no_experiments`.
- Notes: no data, Docker, rsync, or package/test scaffold.

### P1: OASIS Seed-3 First10 Main Pareto

- Status: planned.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Purpose: main brain Pareto result.
- Required before launch: `docs/standards/oasis_brainmr_protocol.md`,
  pair list manifest, metric script audit, alpha/LR grid lock.

### P2: OASIS Seed-3 First10 Deformation-Regime Stats

- Status: planned.
- Purpose: regenerate deformation-regime statistics on the same first10 protocol.

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.
