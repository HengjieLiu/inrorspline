---
doc_type: oasis_brainmr_protocol
status: active
---

# OASIS BrainMR Protocol

## Canonical Test Subset

The main Brain/OASIS short-run protocol is:

```text
split: pair_test_200
shuffle_seed: 3
primary_subset: first10 after seed-3 shuffle
repo split file: configs/splits/oasis/pair_test_200_shuffled_seed000003.txt
```

Do not select the first 10 rows directly from sorted `pair_test_200.txt`.
The sorted order can bias a quick subset toward a non-random slice of the test
pairs. The seed-3 shuffle makes the subset reproducible while avoiding a sorted
first-row shortcut.

## First10 Definition

The seed-3 first10 pairs are the first 10 pair rows in
`configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`:

```text
(0333, 0115)
(0113, 0017)
(0037, 0394)
(0117, 0322)
(0084, 0108)
(0288, 0024)
(0194, 0184)
(0014, 0149)
(0215, 0326)
(0173, 0136)
```

These correspond to original sorted `pair_test_200` indices:

```text
158, 62, 14, 69, 47, 137, 110, 6, 116, 88
```

## Foreground Mask Rule

Use the raw image-positive mask as the canonical OASIS/brainMR foreground:

```text
foreground_mask = image > 0
```

This applies to default foreground/domain masks for OASIS data inspection,
preprocessing, experiment sampling domains, and foreground-scoped deformation
or intensity summaries unless an experiment explicitly defines a different
mask rule.

Do not use `seg > 0` as the default OASIS foreground mask. The segmentation
foreground can be used for label-overlap metrics and visual/comparison audits,
but any use of `seg > 0` as a foreground/domain mask must be documented as an
explicit protocol override.

## Required Checks

- Use the repo split file as the source of pair order for first10/first20 runs.
- Record the subset length, pair list path, and seed in the experiment registry.
- Keep fixed/moving direction as written in the pair file: first element is fixed,
  second element is moving.
- Use raw `image > 0` as the default OASIS foreground mask; treat `seg > 0` as
  a label/comparison mask unless an override is explicitly documented.
- Use one metric pipeline across IDIR, SINR, DirectCP, learned baselines, and
  Greedy references whenever possible.
- Report deformation regularity beside overlap metrics, not as an afterthought.
