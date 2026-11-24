# MR-D-BSCP HU/crop/lambda ablation

Nine fixed preparations;180 independent bounded searches over20 cases. First
4DCT01/08,COPD01/08 are reported before automatic continuation of remaining cases.
Source/configuration are frozen externally before production; historical inputs
and results remain immutable. See the dated execution specification for science.

Run preparation, tests and GPU memory smokes in the pinned Docker environment.
All command arguments below are container paths; outputs must be outside Git.

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_lungct_hu_crop_lambda.py tests/test_lungct_corrected_1mm.py
python -B -m scripts.experiments.lungct_hu_crop_lambda.bootstrap \
  --artifact-root "$ARTIFACT_ROOT" --raw-root "$LUNGCT_DATA" \
  --baseline-root "$CORRECTED_PREPARED_ROOT" --execution-id "$EXECUTION_ID"
```

Launch the frozen source in a dedicated restartable container, with the pinned
image, source/raw RO and ARTIFACT_ROOT RW, UID/GID matching artifact ownership.
From its frozen source root:

```bash
python -B -m scripts.experiments.lungct_hu_crop_lambda.controller \
  --config "$ARTIFACT_ROOT/runs/$EXECUTION_ID/config.json"
```

The controller prepares inputs, profiles two iterations per stage before each
new shape, then starts fresh2500-step registrations with requested/effective
cuDNN F/T. It monitors free memory across all four GPUs. One campaign process
per GPU; other users' processes are not stopped. Profiles that cannot meet
1.2*peak+2GiB admission are explicit unavailable conditions, never downsampled.
Failed production candidates are retained, not retried, and break TRE adjacency.
Source/config/input checks, locks and atomic state allow safe container restart.

Live status and candidate logs are under the run root. Reports are external
under repo_experiments/EXECUTION_ID/{live,first_batch,final}. Both directions
stop on two successive TRE increases or k=+-8; ties reset. Missing scores are
never infinity. Per-case tuning and cap-limited best candidates are explicit.

Separate CPU services generate independent TotalSegmentator2.16.0 evaluation
labels from cached pinned weights, then measure anatomy alignment. They use
annotate and anatomy_metrics modules, the same frozen source, and fresh output
roots. Registration reports list pending anatomy; later supplements preserve
the original report receipts. Automatic-label overlays require visual review
before treating anatomical proxies as validated accuracy claims.

No staging, commits, publication, upstream numerical modifications or automatic
retries/parameter changes are part of this campaign.
