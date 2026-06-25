---
doc_type: workflow
status: active
---

# Anonymous Release Workflow

Build review-safe code and docs continuously, not only at submission time.

## Audit For

- private names, emails, usernames, hostnames
- institution identifiers
- absolute private paths
- W&B entities or private registry URLs
- API keys, tokens, secrets
- hidden metadata in generated files
- raw data, checkpoints, Docker image archives, and large generated outputs

## Public-Facing Targets

- `src/`
- `configs/public/`
- `scripts/experiments/`
- `scripts/reports/`
- `tests/`
- `public/`
- review-facing docs under `docs/`

## Private Or Ignored

- `private/`
- server-specific launch files
- credentials
- raw data and generated outputs

Update `public/ANONYMITY_AUDIT.md` and `public/release_manifest.md` before sharing.
