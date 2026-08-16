---
doc_type: technical_log
status: active
---

# OASIS Paper Cold/Warm DVF Timing

## Known Limitations

- Runtime is measured on OASIS only because OASIS image size is fixed across
  cases. DIR-LAB case size and motion difficulty vary and are not included in
  this compact timing table.
- The benchmark uses four GPUs for faster completion. Each method still runs
  its cases sequentially inside one persistent worker, but GPU-to-GPU variation
  may contribute noise.
- The paper endpoint is `loaded_images_to_dvf_file_sec`: loaded input images to
  saved dense DVF. This excludes evaluation metrics.
- Cold and warm timings are process/GPU warmup states, not registration
  warm-starting. Every case optimizes from its own initialization.

## Timing Definitions

- `cold`: first rank-0 run in a fresh method worker.
- `warm`: rank 0 rerun plus ranks 1-9 in the same already-initialized worker.
- `full_script_to_dvf_file_sec`: script start to saved dense DVF.
- `persistent_case_wall_sec`: worker wall time around one in-process
  `run_one.main()` call.
- `steps`: optimizer updates for one image pair.

## Execution Notes

- Fast schedules run first.
- 2500-step baselines run after all fast workers complete.
- `live_monitor.md` is updated during execution for easy progress tracking.
