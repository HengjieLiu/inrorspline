"""Untimed trajectory parity, full-resolution reconstruction and 3-D Dice."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.oasis_alpha_sweep import common as c
from scripts.experiments.mrdbscp842_acceleration.be_combo_report import parity, field_error
from scripts.experiments.mrdbscp842_acceleration.worker import write_json
from .config import checkpoint_steps
from .geometry import display_plane, reference_grid


def load(path):
    return json.loads(Path(path).read_text())


def loss_event_key(row):
    # The legacy multistage collector rewrites epoch to the global epoch;
    # stage_epoch preserves the callback's local optimizer step.
    return int(row["stage_idx"]), int(row["stage_epoch"]), False


def csv_table(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_trajectories(rows):
    reference = rows[0]
    trace = load(reference["progress"])["history"]
    checks = []
    for row in rows[1:]:
        p = parity(row, reference)
        other = load(row["progress"])["history"]
        p["loss_rows_equal"] = trace == other
        if len(trace) == len(other):
            a = np.array([[r[k] for k in ("loss", "data_loss", "reg_loss")] for r in trace])
            b = np.array([[r[k] for k in ("loss", "data_loss", "reg_loss")] for r in other])
            p["loss_max_abs_delta"] = float(np.max(np.abs(a - b)))
            p["loss_within_tolerance"] = bool(np.allclose(a, b, rtol=1e-5, atol=1e-7))
        else:
            p["loss_within_tolerance"] = False
        p["passed"] &= p["loss_within_tolerance"]
        checks.append(p)
    return checks


def evaluate_progress(command, args):
    out = Path(command["output"])
    rows = command["rows"]
    profiles = [r.get("presentation") for r in rows]
    if not profiles or any(value != profiles[0] for value in profiles):
        raise ValueError("Registration presentation profiles do not match")
    presentation = profiles[0]
    checks = check_trajectories(rows)
    write_json(out / "trajectory_parity.json", checks)
    if not all(check["passed"] for check in checks):
        raise ValueError("Do not join unmatched clocks and snapshots")
    clocks = [r for r in rows if r["progress_mode"] == "clock"]
    bases = [r for r in rows if r["progress_mode"] == "baseline"]
    captures = [r for r in rows if r["progress_mode"] == "capture"]
    if len(captures) != 1 or len(bases) != len(clocks):
        raise ValueError("Expected paired baselines/clocks and one capture")
    ratios = [100. * (clock["elapsed_sec"] / base["elapsed_sec"] - 1.) for base, clock in zip(bases, clocks)]
    median_overhead = float(np.median(ratios))
    write_json(out / "overhead.json", dict(paired_percent=ratios, median_percent=median_overhead))
    if abs(median_overhead) > 5 and not command.get("smoke"):
        raise ValueError(f"Clock overhead gate exceeded: {median_overhead:.3f}%")
    chosen = sorted(range(len(bases)), key=lambda i: bases[i]["elapsed_sec"])[len(bases)//2]
    clock, capture = clocks[chosen], captures[0]
    timeline, saved = load(clock["progress"]), load(capture["progress"])
    lookup = {(r["stage"], r["epoch"], r["final"]): r for r in timeline["events"]}
    if any(b["elapsed_sec"] < a["elapsed_sec"] for a, b in zip(timeline["events"], timeline["events"][1:])):
        raise ValueError("Nonmonotonic measured clock")
    for stage, steps in enumerate(capture["steps"]):
        actual = sorted(r["epoch"] for r in saved["snapshots"] if r["stage"] == stage)
        if actual != checkpoint_steps(steps, int(presentation["capture_interval"])):
            raise ValueError("Missing or duplicated snapshot")
    device = c.set_cuda_device(str(args.gpu), cudnn_benchmark=False, cudnn_deterministic=True)
    batch = c.load_oasis_case(rank=command["rank"], device=device, oasis_root=Path(args.oasis_root), pair_file=Path(args.pair_file))
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, device)
    grid = reference_grid(c.IMAGE_SHAPE).to(device)
    output_rows = []
    (out / "slices").mkdir()
    fixed_slices = {key: display_plane(batch[key][:, :, 96].cpu().numpy())
                    for key in ("fixed", "moving", "fixed_seg", "moving_seg")}
    np.savez_compressed(out / "static.npz", **fixed_slices)
    final_match = None
    with torch.no_grad():
        for index, snapshot in enumerate(saved["snapshots"]):
            values = np.load(snapshot["path"])["controls"]
            control = torch.from_numpy(values).to(device).reshape(-1, 3)
            dense = c.expand_on_shape(control, snapshot["control_shape"], snapshot["full_cps"], c.IMAGE_SHAPE, 1)
            if dense.dtype != torch.float32 or tuple(dense.shape) != (*c.IMAGE_SHAPE, 3) or not torch.isfinite(dense).all():
                raise ValueError("Invalid reconstructed field")
            coords = id_grid + dense
            warped_seg = c.warp_full(batch["moving_seg"], coords, mode="nearest")
            dice = c.dice_per_label(warped_seg, batch["fixed_seg"], batch["moving_seg"])
            warped = c.warp_full(batch["moving"], coords)
            warped_grid = c.warp_full(grid, coords)
            path = out / "slices" / f"state{index:03d}.npz"
            np.savez_compressed(path,
                warped=display_plane(warped[:, :, 96].cpu().numpy()),
                warped_seg=display_plane(warped_seg[:, :, 96].cpu().numpy()),
                grid=display_plane(warped_grid[:, :, 96].cpu().numpy()))
            event = lookup[(snapshot["stage"], snapshot["epoch"], bool(snapshot["final"]))]
            row = dict(index=index, stage=snapshot["stage"], epoch=snapshot["epoch"],
                global_step=snapshot["global_step"], final=snapshot["final"],
                elapsed_sec=event["elapsed_sec"], dice=float(np.nanmean(dice)),
                valid_labels=int(np.isfinite(dice).sum()),
                per_label_dice=[float(v) if np.isfinite(v) else None for v in dice],
                slices=str(path), controls=snapshot["path"], full_cps=snapshot["full_cps"])
            output_rows.append(row)
            if snapshot["stage"] == 2 and snapshot["final"]:
                dest = out / "reconstructed_final.npy"
                np.save(dest, dense.cpu().numpy())
                final_match = field_error(str(dest), capture["dvf"])
                if final_match["vector_rms_voxel"] > .005 or final_match["vector_max_voxel"] > .05:
                    raise ValueError("Final reconstruction mismatch")
            print(json.dumps(dict(snapshot=index, stage=row["stage"], epoch=row["epoch"], dice=row["dice"])), flush=True)
            del control, dense, coords, warped, warped_seg, warped_grid
    history = timeline["history"]
    for row in history:
        row["elapsed_sec"] = lookup[loss_event_key(row)]["elapsed_sec"]
        row["weighted_be"] = float(row["alpha"]) * float(row["reg_loss"])
    csv_table(out / "losses.csv", history)
    csv_table(out / "snapshots.csv", output_rows)
    csv_table(out / "timeline.csv", timeline["events"])
    result = dict(rank=command["rank"], fixed_id=batch["fixed_id"], moving_id=batch["moving_id"],
        snapshots=output_rows, history=history, timeline=timeline, clock=clock, baseline=bases[chosen],
        capture=capture, static=str(out / "static.npz"), trajectory_parity=checks,
        presentation=presentation,
        final_reconstruction=final_match, median_clock_overhead_percent=median_overhead,
        paired_clock_overhead_percent=ratios, selected_pair_index=chosen,
        endpoint_sec=clock["elapsed_sec"], steps=clock["steps"],
        timing_source=command.get("timing_source", {"kind": "paired_current_run"}),
        interpretation="CP-BE numerical non-equivalence retained; single-case visualization, not cohort evidence")
    write_json(out / "animation_data.json", result)
    return dict(animation_data=str(out / "animation_data.json"), snapshot_count=len(output_rows),
                parity_pass=True, final_reconstruction=final_match, median_clock_overhead_percent=median_overhead)
