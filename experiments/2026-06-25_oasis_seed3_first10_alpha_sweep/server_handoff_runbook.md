# Server Handoff Runbook: OASIS Seed-3 First10 Alpha Sweep

## 0. Purpose

This file is the operational handoff for running the OASIS seed-3 first10 alpha
sweep on a different GPU server. It is written for a fresh Codex session or a
human operator on that server.

Do not launch the full run until the dry-run manifest has been inspected and
accepted.

## 1. Experiment Identity

Run ID:

```text
oasis_seed3_first10_alpha_sweep_be_2500e_v1
```

Scientific goal:

- Compare IDIR, SINR cps4, SINR cps2, and DirectCP cps4 as method families.
- Evaluate alignment-regularity tradeoffs in the ARC/Pareto style.
- Separate the dense coordinate-network prior from the spline/local-control
  prior by comparing SINR to DirectCP.

Dataset:

```text
OASIS BrainMR pair_test_200
seed-3 shuffled first10
repo split file: configs/splits/oasis/pair_test_200_shuffled_seed000003.txt
case ranks: 0-9
```

Expected case list:

| Rank | Original sorted index | Fixed | Moving |
|---:|---:|---:|---:|
| 0 | 158 | 0333 | 0115 |
| 1 | 62 | 0113 | 0017 |
| 2 | 14 | 0037 | 0394 |
| 3 | 69 | 0117 | 0322 |
| 4 | 47 | 0084 | 0108 |
| 5 | 137 | 0288 | 0024 |
| 6 | 110 | 0194 | 0184 |
| 7 | 6 | 0014 | 0149 |
| 8 | 116 | 0215 | 0326 |
| 9 | 88 | 0173 | 0136 |

Experiment matrix:

| Method | LR | Training regularizer | Alphas | Jobs |
|---|---:|---|---|---:|
| IDIR | `1e-4` | coordinate-autograd analytic BE | `0,0.01,0.1,1,10,100` | 60 |
| SINR cps4 | `1e-4` | cubic B-spline analytic BE | `0,10,100,1000,10000,100000` | 60 |
| SINR cps2 | `1e-4` | cubic B-spline analytic BE | `0,10,100,1000,10000,100000` | 60 |
| DirectCP cps4 | `1e-3` | cubic B-spline analytic BE | `0,10,100,1000,10000,100000` | 60 |

Total:

```text
10 cases x 4 methods x 6 alphas = 240 jobs
2500 epochs per job
eval_interval = 250
seed = 1
```

Important regularity convention:

- This is a BE-trained sweep.
- IDIR native analytic regularity means coordinate-autograd Hessian BE and
  coordinate-autograd Jacobian diffusion.
- SINR/DirectCP native analytic regularity means cubic B-spline analytic BE and
  cubic B-spline analytic diffusion.
- Diffusion is reported posthoc. This run does not train diffusion-regularized
  sweeps.

## 2. Expected Paths

The target server should mount host home and data into Docker as:

```text
host /home/$USER -> container /homebase
host /data/$USER -> container /database
```

Expected repo path inside Docker:

```text
/homebase/code_sync/inrorspline
```

Expected OASIS data path inside Docker:

```text
/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1
```

Expected large output root inside Docker:

```text
/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
```

Expected large output root on host:

```text
/data/$USER/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
```

Expected review folder inside the repo:

```text
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/
```

Expected Docker container name:

```text
py310sitreg_v1_test1
```

If the container name is different on the target server, replace
`py310sitreg_v1_test1` in all commands below with the actual name.

## 3. Required Files In The Repo

Before doing anything else, confirm these files exist on the target server:

```bash
cd /homebase/code_sync/inrorspline

test -f AGENTS.md
test -f docs/04_experiment_registry.md
test -f docs/standards/oasis_brainmr_protocol.md
test -f scripts/experiments/oasis_alpha_sweep/run_one.py
test -f scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py
test -f scripts/experiments/oasis_alpha_sweep/aggregate_first10_population.py
test -f experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/codex_execution_spec.md
test -f experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/technical_log.md
test -f experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/server_handoff_runbook.md
```

If any command fails, stop and fix the code transfer before continuing.

Recommended repo status check:

```bash
cd /homebase/code_sync/inrorspline
git branch --show-current
git rev-parse HEAD
git status --short
```

The worktree does not need to be perfectly clean if local docs are intentionally
uncommitted, but the first10 launcher, aggregator, and runbook must match the
intended version. If in doubt, stop and compare against the source server.

## 4. Docker And GPU Preflight

Confirm the container exists and is running:

```bash
docker ps --filter name=py310sitreg_v1_test1 --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Confirm GPUs are visible from inside Docker:

```bash
docker exec py310sitreg_v1_test1 nvidia-smi
```

Confirm Python dependencies:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
python - <<'"'"'PY'"'"'
import importlib.util
mods = ["torch", "nibabel", "scipy", "surface_distance", "matplotlib", "numpy"]
for mod in mods:
    print(mod, "OK" if importlib.util.find_spec(mod) else "MISSING")
PY
'
```

Stop if any dependency is missing.

Compile the sweep scripts inside Docker:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python -m py_compile scripts/experiments/oasis_alpha_sweep/*.py
'
```

Stop if compilation fails.

## 5. OASIS Data Preflight

Confirm the data root:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
OASIS=/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1
test -d "$OASIS"
test -f "$OASIS/pair_test_200.txt"
test -d "$OASIS/test"
echo "$OASIS"
'
```

Expected file counts:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
OASIS=/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1
echo "img count:"
find "$OASIS/test" -maxdepth 1 -type f -name "img*.nii.gz" | wc -l
echo "seg count:"
find "$OASIS/test" -maxdepth 1 -type f -name "seg*.nii.gz" | wc -l
'
```

Expected:

```text
img count: 84
seg count: 84
```

Confirm the 10 required fixed/moving IDs exist:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
OASIS=/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1/test
for id in 0333 0115 0113 0017 0037 0394 0117 0322 0084 0108 0288 0024 0194 0184 0014 0149 0215 0326 0173 0136; do
  test -f "$OASIS/img${id}.nii.gz" || { echo "missing img${id}"; exit 1; }
  test -f "$OASIS/seg${id}.nii.gz" || { echo "missing seg${id}"; exit 1; }
done
echo "required OASIS image/seg files present"
'
```

Stop if any file is missing.

## 6. Read The Local Protocol Before Launch

Inside the target server Codex session, read these before running anything:

```text
AGENTS.md
docs/04_experiment_registry.md
docs/standards/README.md
docs/standards/fair_experiment_checklist.md
docs/standards/oasis_brainmr_protocol.md
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/codex_execution_spec.md
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/technical_log.md
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/server_handoff_runbook.md
```

Non-negotiable protocol checks:

- Use seed-3 shuffled ranks `0-9`, not sorted `pair_test_200.txt` rows `0-9`.
- IDIR training samples from `fixed_image > 0`.
- SINR and DirectCP use full-image `sinr_ncc` without image-loss mask.
- Foreground regularity metrics use `fixed_image > 0`.
- Segmentation masks are for Dice, HD95, ASD, and visualization only.

## 7. Dry-Run Workflow

Run the dry-run first:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py --dry-run
'
```

The dry-run writes a manifest but should not launch training jobs.

Manifest paths:

```text
/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/job_manifest.csv
/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/job_manifest.json
```

Inspect the manifest:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
python - <<'"'"'PY'"'"'
import csv
from collections import Counter, defaultdict
root = "/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1"
path = f"{root}/job_manifest.csv"
rows = list(csv.DictReader(open(path)))
print("rows", len(rows))
print("case ranks", sorted({int(r["case_rank"]) for r in rows}))
print("methods", Counter(r["method_key"] for r in rows))
print("alphas by method")
by_method = defaultdict(set)
for r in rows:
    by_method[r["method_key"]].add(float(r["alpha"]))
for method, alphas in sorted(by_method.items()):
    print(method, sorted(alphas))
print("first row", rows[0])
print("last row", rows[-1])
PY
'
```

Expected dry-run manifest:

```text
rows = 240
case ranks = [0,1,2,3,4,5,6,7,8,9]
IDIR alphas = [0,0.01,0.1,1,10,100]
SINR/DirectCP alphas = [0,10,100,1000,10000,100000]
60 jobs per method family
```

Spot-check expected case IDs:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
python - <<'"'"'PY'"'"'
import csv
expected = {
    0: (158, "0333", "0115"),
    1: (62, "0113", "0017"),
    2: (14, "0037", "0394"),
    3: (69, "0117", "0322"),
    4: (47, "0084", "0108"),
    5: (137, "0288", "0024"),
    6: (110, "0194", "0184"),
    7: (6, "0014", "0149"),
    8: (116, "0215", "0326"),
    9: (88, "0173", "0136"),
}
rows = list(csv.DictReader(open("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/job_manifest.csv")))
for rank, (orig, fixed, moving) in expected.items():
    subset = [r for r in rows if int(r["case_rank"]) == rank]
    assert subset, rank
    for r in subset:
        assert int(r["original_case_index"]) == orig, r
        assert r["fixed_id"] == fixed, r
        assert r["moving_id"] == moving, r
print("case ID audit passed")
PY
'
```

Stop if any manifest check fails.

## 8. Full Launch

Expected runtime:

```text
best estimate: 6-7 hours on 4 GPUs
conservative estimate: 8-10 hours
```

Expected large output size:

```text
25-30 GB safe budget
```

Launch only after dry-run approval:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py
'
```

The launcher should:

- run max 4 jobs concurrently across GPUs `0,1,2,3`
- skip existing jobs that already have `DONE`
- write logs under the output root
- run population aggregation automatically after all jobs complete

Do not use `--overwrite` unless intentionally redoing completed jobs.

## 9. Monitoring During Launch

GPU status:

```bash
watch -n 10 nvidia-smi
```

Output root size:

```bash
du -sh /data/$USER/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
```

Inside Docker, count completed jobs:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
echo "DONE count:"
find "$ROOT" -mindepth 3 -maxdepth 3 -name DONE | wc -l
echo "summary count:"
find "$ROOT" -mindepth 3 -maxdepth 3 -name summary.json | wc -l
'
```

Expected final counts:

```text
DONE count = 240
summary count = 240
```

Recent logs:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
ls -lt "$ROOT/logs" | head -20
'
```

Tail a specific log:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
tail -100 "$ROOT/logs/LOG_NAME_HERE.log"
'
```

Look for failed Python tracebacks:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
grep -R "Traceback\\|RuntimeError\\|CUDA out of memory\\|failed" "$ROOT/logs" | tail -80 || true
'
```

Launcher summary after completion:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
ROOT=/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1
test -f "$ROOT/launch_summary.json" && cat "$ROOT/launch_summary.json" || echo "launch_summary.json not written yet"
'
```

## 10. Resume And Failure Handling

If the launcher is interrupted:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py
'
```

Default resume behavior:

- jobs with `DONE` are skipped
- jobs without `DONE` are attempted again
- partial run directories may be overwritten by the rerun job

Use `--overwrite` only when deliberately replacing completed results:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py --overwrite
```

If a job repeatedly fails:

1. Identify its log under `ROOT/logs`.
2. Record method, alpha, case rank, GPU, and traceback.
3. Check whether it is data missing, CUDA OOM, dependency missing, or script bug.
4. Do not change the scientific grid unless approved.
5. After fixing the issue, rerun the launcher without `--overwrite`.

## 11. Aggregation

The launcher runs aggregation automatically after all jobs complete. If needed,
rerun aggregation manually:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/aggregate_first10_population.py
'
```

The aggregator is intentionally strict by default. It should fail if any run is
missing, duplicated, has wrong case IDs, wrong epochs, wrong seed, wrong variant,
wrong mask convention, wrong native analytic formula, or missing artifacts.

Expected review outputs:

```text
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/summary_runs.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/summary_runs.json
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/population_summary.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/population_summary.json
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/population_label_metrics.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/case_metrics_all.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/regularity_metrics_all.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/be_metrics_all.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/diffusion_metrics_all.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/loss_history_all.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/loss_history_population.csv
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/population_arc_primary.pdf
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/population_pareto_contact_sheet.pdf
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/loss_curves_population.pdf
experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/review_report.md
```

## 12. Final Acceptance Audit

Run this after aggregation:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
python - <<'"'"'PY'"'"'
import json
from pathlib import Path
review = Path("experiments/2026-06-25_oasis_seed3_first10_alpha_sweep")
out = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
summary = json.loads((review / "summary_runs.json").read_text())
pop = json.loads((review / "population_summary.json").read_text())
print("summary rows", len(summary))
print("population rows", len(pop))
print("DONE count", len(list(out.glob("*/*/DONE"))))
print("review report exists", (review / "review_report.md").exists())
print("primary ARC exists", (review / "population_arc_primary.pdf").exists())
print("contact sheet exists", (review / "population_pareto_contact_sheet.pdf").exists())
assert len(summary) == 240
assert len(pop) == 24
assert len(list(out.glob("*/*/DONE"))) == 240
PY
'
```

Expected:

```text
summary rows = 240
population rows = 24
DONE count = 240
review report exists = True
primary ARC exists = True
contact sheet exists = True
```

Also inspect the report:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
cd /homebase/code_sync/inrorspline
sed -n "1,120p" experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/review_report.md
'
```

The report should say acceptance passed. If it reports failures, stop and report
the exact failure lines.

## 13. What To Report Back

After the run finishes, report:

- server name or short server label, if safe to share
- git branch and commit hash used
- Docker image tag or digest used
- launch start and completion time
- total launcher elapsed time
- output root path
- output root size
- review root path
- `DONE` count
- summary count
- population row count
- whether aggregation passed
- whether any jobs failed or were resumed
- best population alpha by method from `review_report.md`
- paths to:
  - `review_report.md`
  - `population_arc_primary.pdf`
  - `population_pareto_contact_sheet.pdf`
  - `population_summary.csv`
  - `summary_runs.csv`

Suggested completion summary template:

```text
OASIS seed-3 first10 sweep completed.

Git:
Docker:
Output root:
Review root:
Elapsed:
Output size:

Counts:
- DONE:
- summaries:
- population rows:

Aggregation:
- status:
- errors:

Best population alphas:
- IDIR:
- SINR cps4:
- SINR cps2:
- DirectCP cps4:

Review files:
- review_report.md:
- population_arc_primary.pdf:
- population_pareto_contact_sheet.pdf:
- population_summary.csv:
- summary_runs.csv:
```

## 14. Stop Conditions

Stop and ask before continuing if:

- Docker cannot see GPUs.
- Any required OASIS image/segmentation file is missing.
- Python dependencies are missing.
- `py_compile` fails.
- Dry-run manifest is not exactly 240 jobs.
- Dry-run manifest has wrong case ranks, IDs, alphas, or LRs.
- Full launch produces repeated failures for the same job.
- Aggregation fails acceptance checks.
- Output disk has less than 30 GB free before launch.
- Someone suggests changing alpha grid, LR, epochs, or mask convention mid-run.

