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

## Canonical Submission Archive

The exact two PDFs in `docs/offgrid_submission/submitted_files/` are
intentionally tracked archival records, not general generated-artifact
exceptions. The camera-ready PDF contains author-identifying information.
Exclude this entire directory from every anonymous-review or anonymous-release
payload, even though the initial-submission PDF itself is anonymized. See its
README before approving any replacement.

Update `public/ANONYMITY_AUDIT.md` and `public/release_manifest.md` before sharing.
