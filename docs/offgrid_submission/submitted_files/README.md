---
doc_type: submitted_paper_archive
status: active
---

# Canonical Submitted Paper Files

This directory is an intentionally narrow override to the repository's
external-only generated-PDF policy. It preserves the two canonical submitted
Off-Grid paper records as Git-tracked archival documents. It is not a location
for experiment output, manuscript builds, figures, or additional PDFs.

## Canonical Records

| Role | Exact file | SHA-256 | Handling |
| --- | --- | --- | --- |
| Initial submission | `submit_30_The_Right_Prior_for_the_Rig.pdf` | `377083adfdd96913c3fb2652b97bc6d47f65b4b112f5c85948c1c005d4227d44` | Anonymized archival record. |
| Camera-ready submission | `camera_ready_off-grid_30.pdf` | `1489fba1e412bb2e675c67fee6a0537d08e29d579a20702ea9acb0e1d41dd17d` | Named, identity-bearing archival record; exclude from anonymous-release bundles. |

Both files were validated as unencrypted PDFs on 2026-09-16. The initial
submission has 10 pages; the camera-ready submission has 12 pages.

## Retention and Replacement Rules

- Keep these two exact paths tracked in Git. They are not artifact-store
  migration or source-only-cleanup candidates.
- Do not overwrite, rename, delete, or replace either file without explicit
  user approval.
- For an approved replacement, validate that the PDF opens, record its
  SHA-256 and page count in this table, state the replacement reason in the
  commit, and stage only the intended archival/documentation paths.
- Do not add any other generated or submitted PDF to this directory without
  explicit user approval and a corresponding policy update.
- Exclude this entire directory from anonymous-review and anonymous-release
  exports. The camera-ready record includes identifying information even though
  the initial-submission record is anonymized.
