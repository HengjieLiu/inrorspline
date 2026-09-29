"""Opt-in progress capture around the unchanged registration implementation."""
import json
from pathlib import Path
import time

import numpy as np

from .config import presentation


def registration(command, backend, args):
    from scripts.experiments.oasis_alpha_sweep import run_one as run
    from scripts.experiments.mrdbscp842_acceleration.worker import registration as original
    from scripts.experiments.mrdbscp842_acceleration.worker import write_json
    profile = presentation(command.get("presentation_profile", "adaptive_cpbe"))
    mode = command["progress_mode"]
    if mode not in ("baseline", "clock", "capture") or tuple(backend.features) != profile.features:
        raise ValueError("Progress capture backend/profile mismatch or invalid mode")
    out = Path(command["output"])
    events, snapshots, history, endpoints = [], {}, [], {}
    parse, mark, metadata = run.parse_args, run.TimingRecorder.mark, run.write_dvf_only_metadata

    def observed_mark(timer, name, **kwargs):
        result = mark(timer, name, **kwargs)
        if result is not None:
            endpoints[name] = timer.root_start + result["offset_sec"]
        return result

    def snapshot(stage, epoch, control, control_shape, eval_shape, domain_cps, final):
        now = time.perf_counter()
        events.append(dict(stage=int(stage), epoch=int(epoch), final=bool(final), perf_counter=now))
        if mode != "capture" or not (epoch == 0 or epoch % profile.capture_interval == 0 or final):
            return
        key = (int(stage), int(epoch))
        if key not in snapshots:
            path = out / "controls" / f"stage{stage}_step{epoch:04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = control.detach().cpu().numpy().reshape(*control_shape, 3).copy()
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Invalid control snapshot")
            np.savez(path, controls=values)
            snapshots[key] = dict(stage=int(stage), epoch=int(epoch), control_shape=list(control_shape),
                loss_shape=list(eval_shape), full_cps=profile.full_cps[stage], domain_cps=int(domain_cps),
                downsampling=profile.downsampling[stage], path=str(path), final=False,
                state="initialization" if epoch == 0 else "post_optimizer_step")
        snapshots[key]["final"] |= bool(final)

    def parsed():
        value = parse()
        if mode != "baseline":
            value._control_snapshot_callback = snapshot
        return value

    def observed_metadata(*a, **kw):
        # This function is invoked only AFTER dvf_file_ready, for all three modes.
        metrics = kw.get("metrics", a[2] if len(a) > 2 else None)
        history.extend(metrics["history"])
        return metadata(*a, **kw)

    run.parse_args, run.TimingRecorder.mark, run.write_dvf_only_metadata = parsed, observed_mark, observed_metadata
    try:
        result = original(command, backend, args)
    finally:
        run.parse_args, run.TimingRecorder.mark, run.write_dvf_only_metadata = parse, mark, metadata
    origin = endpoints["case_load_complete"]
    steps = result["steps"]
    for row in events:
        row["elapsed_sec"] = row.pop("perf_counter") - origin
        row["global_step"] = sum(steps[:row["stage"]]) + row["epoch"]
    for row in snapshots.values():
        row["global_step"] = sum(steps[:row["stage"]]) + row["epoch"]
    if len(history) != sum(steps):
        raise ValueError("Every training step must have a loss row")
    payload = dict(mode=mode, events=events, snapshots=list(snapshots.values()), history=history,
        endpoints={key: value - origin for key, value in endpoints.items()}, steps=steps,
        presentation=profile.to_dict(),
        loss_semantics="pre-update loss; snapshot is post-update; no extra loss evaluation",
        clock_semantics="host monotonic timestamps after existing scalar reads, no added step synchronization")
    write_json(out / "progress.json", payload)
    result.update(progress=str(out / "progress.json"), progress_mode=mode,
                  presentation=profile.to_dict())
    return result
