# OASIS Seed-3 First100 Extension

This experiment extends the OASIS seed-3 first20 paper-figure runs to first100.
Ranks `0-19` are reused from existing completed roots; ranks `20-99` are the
new extension jobs.

- Progress log: `progress_log.md`
- Live tracker figure: `tracker/oasis_first100_progress_tracker.png`
- Hourly watchdog log: `hourly_watchdog.md`
- Final figure package: `first100_artifacts/oasis100/`
- Large run outputs: `/database/inrorspline/runs/`

Launch the full matrix from inside the Docker container:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py --stage all --gpus auto --max-concurrent 0
```

Update the tracker without launching jobs:

```bash
python scripts/experiments/oasis_alpha_sweep/monitor_oasis_first100_extension.py
```

Run the hourly watchdog:

```bash
python scripts/experiments/oasis_alpha_sweep/watch_oasis_first100_hourly.py --watch --interval-sec 3600
```
