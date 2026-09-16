# Anonymity Audit

Status: scaffold.

## Current Notes

- Repo scaffold created with public-safe placeholders where possible.
- Carried-forward reports may contain private absolute paths from the previous workspace and need review before public release.
- `private/`, raw data, generated outputs, checkpoints, and Docker image archives are ignored.
- `docs/offgrid_submission/submitted_files/` is intentionally tracked as a
  canonical submission archive but must be excluded from anonymous-release
  bundles because its camera-ready PDF identifies authors.

## TODO

- Audit all review-facing docs for names, emails, usernames, hostnames, private paths, registry URLs, and W&B entities.
- Replace or quarantine absolute paths in carried-forward reports.
- Build a release manifest once implementation files exist.
