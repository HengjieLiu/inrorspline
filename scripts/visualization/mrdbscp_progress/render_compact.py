"""Render the fixed-50 presentation with compact images and one loss panel."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from scripts.experiments.oasis_alpha_sweep import common as c
from .offline import csv_table
from .preview import digest, show, validate_output, write_json
from .config import checkpoint_steps
from .render import stage_indices
from .render_revision import coordinate_planes, draw_grid


COLORS = {"similarity_display": "#0072B2", "weighted_be": "#D55E00",
          "total_display": "#6A3D9A"}
LABELS = {"similarity_display": "1 − NCC", "weighted_be": "1000 × BE",
          "total_display": "Total = (1 − NCC) + 1000 × BE"}


def display_losses(row):
    """A constant +1 display shift; the optimized objective is unchanged."""
    return dict(similarity_display=1. + float(row["data_loss"]),
                weighted_be=float(row["weighted_be"]),
                total_display=1. + float(row["loss"]))


def presentation_clock(data):
    """Build a stage-measured clock without reusing contaminated step fractions."""
    timing_source = data.get("timing_source", {})
    if timing_source.get("kind") != "historical_accepted":
        raise ValueError("Phase-aware compact replay requires accepted historical timing")
    timing_path = Path(timing_source["dvf"]).parent / "timing.json"
    timing = json.loads(timing_path.read_text())
    marks = {row["name"]: float(row["offset_sec"]) for row in timing["marks"]}
    endpoint = float(data["endpoint_sec"])
    memory_ready = marks["dvf_in_memory_ready"] - marks["case_load_complete"]
    file_ready = marks["dvf_file_ready"] - marks["case_load_complete"]
    stage_sec = list(map(float, timing_source["stage_sec"]))
    output_sec = file_ready - memory_ready
    preparation_total = memory_ready - sum(stage_sec)
    if (abs(file_ready - endpoint) > 1e-9 or output_sec <= 0
            or preparation_total < 0 or len(stage_sec) != 3):
        raise ValueError("Historical timing marks do not reconcile")
    # Endpoint-only timing cannot split this tiny residual among stage setup
    # calls. Equal allocation is explicit and affects only the displayed clock.
    preparation_sec = [preparation_total / 3.] * 3
    cursor = 0.
    stages = []
    for stage in range(3):
        prep_start, prep_end = cursor, cursor + preparation_sec[stage]
        opt_start, opt_end = prep_end, prep_end + stage_sec[stage]
        stages.append(dict(stage=stage, preparation_start_sec=prep_start,
                           preparation_end_sec=prep_end,
                           optimization_start_sec=opt_start,
                           optimization_end_sec=opt_end,
                           preparation_sec=preparation_sec[stage],
                           optimization_sec=stage_sec[stage]))
        cursor = opt_end
    if abs(cursor - memory_ready) > 1e-9:
        raise ValueError("Stage clock does not end at the in-memory DVF mark")
    return dict(endpoint_sec=endpoint, dvf_in_memory_ready_sec=memory_ready,
                output_sec=output_sec, preparation_total_sec=preparation_total,
                stages=stages,
                semantics=("Measured stage spans; endpoint residual split equally across "
                           "three preparation phases; steps evenly placed within each stage"))


def remap_display_clock(data, clock):
    """Copy numerical data and assign presentation timestamps from stage totals."""
    result = copy.deepcopy(data)
    for row in result["history"]:
        stage = int(row["stage_idx"])
        epoch = int(row["stage_epoch"])
        phase = clock["stages"][stage]
        row["elapsed_sec"] = (phase["optimization_start_sec"]
                              + phase["optimization_sec"] * epoch / result["steps"][stage])
    for row in result["snapshots"]:
        stage, epoch = int(row["stage"]), int(row["epoch"])
        phase = clock["stages"][stage]
        row["elapsed_sec"] = (phase["optimization_start_sec"]
                              + phase["optimization_sec"] * epoch / result["steps"][stage])
    result["endpoint_sec"] = clock["endpoint_sec"]
    result["presentation_clock"] = clock
    return result


def select_field_cadence(data, interval):
    """Select measured post-update fields without interpolating deformation."""
    if interval not in (1, 2, 5, 10):
        raise ValueError("Field interval must be one of 1, 2, 5 or 10")
    source_interval = int(data["presentation"]["capture_interval"])
    if source_interval > interval or interval % source_interval:
        raise ValueError("Requested field cadence is unavailable from this capture")
    result = copy.deepcopy(data)
    result["snapshots"] = [row for row in result["snapshots"]
        if int(row["epoch"]) == 0 or int(row["epoch"]) % interval == 0 or row["final"]]
    expected = sum(len(checkpoint_steps(steps, interval)) for steps in result["steps"])
    if len(result["snapshots"]) != expected:
        raise ValueError("Selected field cadence is incomplete")
    result["field_step_interval"] = interval
    return result


def compact_timeline(data, clock, duration=15., fps=30):
    """Show only optimization stages and the final ready state."""
    optimization_video = duration - 1.3
    stage_video = [optimization_video * row["optimization_sec"] /
                   sum(value["optimization_sec"] for value in clock["stages"])
                   for row in clock["stages"]]
    phases = []
    for stage, row in enumerate(clock["stages"]):
        phases.append(dict(kind="optimization", stage=stage, video_sec=stage_video[stage],
                           real_start=(0. if stage == 0 else
                                       clock["stages"][stage-1]["optimization_end_sec"]),
                           real_end=row["optimization_end_sec"],
                           status=f"Stage {stage+1}/3 · Optimizing", detail=""))
    phases += [dict(kind="final_hold", video_sec=1.3,
                    real_start=clock["endpoint_sec"], real_end=clock["endpoint_sec"],
                    status="Complete · DVF file ready", detail="")]
    if abs(sum(row["video_sec"] for row in phases) - duration) > 1e-9:
        raise ValueError("Compact phase allocation does not equal requested duration")
    history_times = np.array([row["elapsed_sec"] for row in data["history"]])
    snapshot_times = np.array([row["elapsed_sec"] for row in data["snapshots"]])
    ticks, video_cursor, frame_cursor = [], 0., 0
    for phase_index, phase in enumerate(phases):
        next_video = video_cursor + phase["video_sec"]
        end_frame = round(next_video * fps)
        count = end_frame - frame_cursor
        for local in range(count):
            fraction = local / count if count else 0.
            elapsed = phase["real_start"] + fraction * (phase["real_end"] - phase["real_start"])
            loss_count = int(np.searchsorted(history_times, elapsed, side="right"))
            loss_step = int(data["history"][loss_count-1]["global_epoch"]) if loss_count else 0
            snapshot_index = max(0, int(np.searchsorted(snapshot_times, elapsed, side="right")) - 1)
            state_index = snapshot_index + 1
            if elapsed < snapshot_times[0]:
                state_index = 0
            if phase["kind"] == "final_hold":
                state_index = len(data["snapshots"])
            status = phase["status"]
            if phase["kind"] == "optimization":
                stage = phase["stage"]
                local_step = max(0, min(data["steps"][stage],
                    loss_step - sum(data["steps"][:stage])))
                status += f" · Step {local_step}/{data['steps'][stage]}"
            ticks.append(dict(video_frame=len(ticks), video_sec=len(ticks)/fps,
                phase_index=phase_index, phase_kind=phase["kind"], state_index=state_index,
                snapshot_index=snapshot_index, elapsed_sec=float(elapsed),
                snapshot_elapsed_sec=float(data["snapshots"][snapshot_index]["elapsed_sec"]),
                loss_step=loss_step, loss_rows_revealed=loss_count,
                holding=phase["real_start"] == phase["real_end"],
                status=status, status_detail=phase["detail"]))
        frame_cursor, video_cursor = end_frame, next_video
    if len(ticks) != round(duration * fps):
        raise ValueError("Unexpected compact frame count")
    # Retain the established state contract for PNGs, HTML jumps and frozen-column checks.
    states = [dict(snapshot_index=0, elapsed_sec=0., initial=True)]
    states += [dict(snapshot_index=i, elapsed_sec=row["elapsed_sec"], initial=False)
               for i, row in enumerate(data["snapshots"])]
    states[-1]["elapsed_sec"] = clock["endpoint_sec"]
    states[-1]["output_ready"] = True
    for index, state in enumerate(states):
        used = [row["video_frame"] for row in ticks if row["state_index"] == index]
        if not used:
            raise ValueError(f"Presentation state {index} received no video frames")
        state["playback_start_sec"] = min(used) / fps
        state["playback_end_sec"] = (max(used) + 1) / fps
        state["video_frames"] = len(used)
        state["png"] = f"frames/frame{index:03d}.png"
        state["output_ready"] = state.get("output_ready", False)
    return states, ticks, phases


def _active_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#008C95")
        spine.set_linewidth(3.)


def compact_frame(data, static, slices, state, destination=None, keep_figure=False):
    index = state["snapshot_index"]
    current = data["snapshots"][index]
    chosen = stage_indices(data, index)
    profile = data["presentation"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    fig.text(.035, .958, "MR-D-BSCP · 50/50/50 + Cache/Batch + CP-BE",
             fontsize=31, weight="bold")
    field_caption = ("Initialization" if state["initial"] else
                     f"Stage {current['stage']+1}/3 · Step {current['epoch']}/{data['steps'][current['stage']]}")
    caption = f"Field snapshot: {field_caption} @ {current['elapsed_sec']:.2f} s"
    if state.get("output_ready"):
        caption += "  |  DVF file ready"
    fig.text(.035, .895, f"Case 04  |  Fixed {data['fixed_id']} / Moving {data['moving_id']}  |  {caption}",
             fontsize=22)
    fig._elapsed_text = fig.text(.972, .958, f"Warm elapsed: {state['elapsed_sec']:.2f} s",
                                 fontsize=28, ha="right", color="#006E82", weight="bold")

    # Portrait axial panels need narrow columns. This removes the broad white
    # column allocations in the historical top/bottom layout.
    grid = fig.add_gridspec(3, 5, left=.055, right=.615, bottom=.065, top=.815,
                            wspace=.015, hspace=.055)
    headers = ["Moving", "Fixed"] + [
        f"Stage {i+1}\ncps {profile['full_cps'][i]} · down {profile['downsampling'][i]}"
        for i in range(3)]
    for col, key in enumerate(("moving", "fixed")):
        ax = fig.add_subplot(grid[0, col])
        show(ax, static[key], title=headers[col], orientation=True)
        ax.title.set_fontsize(19)
        show(fig.add_subplot(grid[1, col]), static[key+"_seg"], "labels")
        note = fig.add_subplot(grid[2, col]); note.axis("off")
        note.text(.5, .55, "Input image" if col == 0 else "Target image",
                  ha="center", va="center", fontsize=18, color="#555555")
    for stage in range(3):
        active = stage == current["stage"] and not current["final"] and not state.get("output_ready")
        if stage not in chosen:
            for row in range(3):
                ax = fig.add_subplot(grid[row, stage+2]); ax.axis("off")
                ax.set_facecolor("#F1F2F3")
                if row == 0:
                    ax.set_title(headers[stage+2], fontsize=19, pad=8)
                    ax.text(.5, .5, "Not started", ha="center", va="center",
                            transform=ax.transAxes, color="#777777", fontsize=19)
            continue
        selected = chosen[stage]
        snap, planes = data["snapshots"][selected], slices[selected]
        for row, (key, kind) in enumerate((("warped", "image"), ("warped_seg", "labels"),
                                           ("grid", "grid"))):
            ax = fig.add_subplot(grid[row, stage+2])
            if kind == "grid":
                draw_grid(ax, planes)
            else:
                show(ax, planes[key], kind, headers[stage+2] if row == 0 else None,
                     orientation=row == 0)
            if row == 0:
                ax.title.set_fontsize(19)
            if active:
                _active_border(ax)
            if row == 2:
                label = f"3D Dice {snap['dice']:.4f}" + ("\nFrozen" if snap["final"] else "")
                ax.set_xlabel(label, fontsize=16, labelpad=5, linespacing=.9)
    for y, label in ((.690, "MRI"), (.442, "Segmentation"), (.190, "Warped grid")):
        fig.text(.018, y, label, rotation=90, ha="center", va="center",
                 fontsize=19, weight="bold")

    history = data["history"]
    end_step = state.get("loss_step", 0 if state["initial"] else current["global_step"])
    total_steps = sum(data["steps"])
    boundaries = np.cumsum(data["steps"])
    fig.text(.817, .835, "Optimization progress", ha="center", fontsize=27,
             weight="bold")
    fig._phase_text = fig.text(.817, .790, state.get("status", "Preparing"),
                               ha="center", fontsize=22, color="#006E82", weight="bold")
    fig._phase_detail = fig.text(.817, .752, state.get("status_detail", ""),
                                 ha="center", fontsize=17, color="#4D4D4D")
    fig._loss_value_text = fig.text(.817, .705, "Step 0/150", ha="center",
                                    va="center", fontsize=16, color="#252525",
                                    linespacing=1.25)
    ax = fig.add_axes((.665, .105, .305, .525))
    all_values = [value for row in history for value in display_losses(row).values()]
    lo, hi = min(all_values), max(all_values)
    margin = max((hi-lo)*.08, 1e-5)
    ax.set_ylim(lo-margin, hi+margin)
    ax.set_xlim(0, total_steps + 3)
    fig._loss_artists = []
    for key in ("similarity_display", "weighted_be", "total_display"):
        lines = []
        for stage in range(3):
            rows = [row for row in history if int(row["stage_idx"]) == stage
                    and int(row["global_epoch"]) <= end_step]
            lines.append(ax.plot([r["global_epoch"] for r in rows],
                                 [display_losses(r)[key] for r in rows],
                                 color=COLORS[key], lw=2.8, label=LABELS[key] if stage == 0 else None)[0])
        fig._loss_artists.append((key, lines))
    completed = []
    for stage, boundary in enumerate(boundaries):
        if stage in chosen and data["snapshots"][chosen[stage]]["final"]:
            ax.axvline(boundary, ls="--", color="#6F6F6F", lw=1.8)
            completed.append(int(boundary))
    fig._loss_cursor = ax.axvline(end_step, color="#008C95", lw=2.2, alpha=.75)
    fig._loss_cursor.set_visible(end_step not in completed)
    ax.set_xlabel("Global optimization step", fontsize=18, labelpad=8)
    ax.set_ylabel("Display loss", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=17, frameon=False)
    if destination is not None:
        fig.savefig(destination, dpi=100)
    if keep_figure:
        return fig
    plt.close(fig)


def update_frame(fig, data, tick):
    visible = data["history"][:tick["loss_rows_revealed"]]
    for key, lines in fig._loss_artists:
        for stage, line in enumerate(lines):
            rows = [r for r in visible if int(r["stage_idx"]) == stage]
            line.set_data([r["global_epoch"] for r in rows], [display_losses(r)[key] for r in rows])
    fig._loss_cursor.set_xdata([tick["loss_step"], tick["loss_step"]])
    fig._loss_cursor.set_visible(tick["loss_step"] not in (50, 100, 150))
    if visible:
        values = display_losses(visible[-1])
        fig._loss_value_text.set_text(
            f"Step {tick['loss_step']}/150\n"
            f"1−NCC {values['similarity_display']:.4f}   ·   BE×1000 {values['weighted_be']:.4f}"
            f"   ·   Total {values['total_display']:.4f}")
    else:
        fig._loss_value_text.set_text("Step 0/150\nLoss available after first update")
    fig._phase_text.set_text(tick["status"])
    fig._phase_detail.set_text(tick["status_detail"])
    fig._elapsed_text.set_text(f"Warm elapsed: {tick['elapsed_sec']:.2f} s")


def html_player(path, stage_times, video_name):
    path.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Case 04 · fixed-50 cache/batch replay</title><style>body{font:20px system-ui;max-width:1500px;margin:20px auto;background:#e9edf1;color:#132333}video{width:100%}button,select{font:inherit;margin:6px}</style>
<h1>Case 04 · 50/50/50 + Cache/Batch + CP-BE</h1><p>Measured warm clock · discrete measured fields · every-step loss · slowed replay.</p>
<video id="v" controls preload="metadata" src="__VIDEO__" poster="initial.png"></video><div><button onclick="v.paused?v.play():v.pause()">Play / pause</button><select onchange="v.playbackRate=Number(this.value)"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option></select>
<button onclick="v.currentTime=ENDS[0]">Stage 1 end</button><button onclick="v.currentTime=ENDS[1]">Stage 2 end</button><button onclick="v.currentTime=ENDS[2]">Stage 3 end</button></div>
<p><a href="__VIDEO__">Download MP4</a> · <a href="report.md">Report</a> · <a href="losses.csv">Losses</a> · <a href="frame_timeline.csv">Frame mapping</a></p><script>const v=document.getElementById('v'),ENDS=__ENDS__;</script></html>'''.replace("__ENDS__", json.dumps(stage_times)).replace("__VIDEO__", video_name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--field-step-interval", type=int, choices=[1, 2, 5, 10], default=10)
    parser.add_argument("--video-name", default=None)
    args = parser.parse_args()
    video_name = args.video_name or f"animation_step{args.field_step_interval}.mp4"
    if video_name != Path(video_name).name or not video_name.endswith(".mp4"):
        raise ValueError("Video name must be a local MP4 filename")
    output = validate_output(args.output, Path(os.environ["ARTIFACT_ROOT"]))
    data = json.loads(args.data.read_text())
    profile = data.get("presentation", {})
    if (not profile.get("profile", "").startswith("fixed50_cached_batch") or data["steps"] != [50, 50, 50]
            or profile.get("features") != ["cache", "batch", "cpbe"]):
        raise ValueError("Compact renderer requires the fixed50 cache/batch profile")
    data = select_field_cadence(data, args.field_step_interval)
    evaluation = json.loads((args.data.parent / "accepted_result.json").read_text())
    timing_source = data.get("timing_source", {"kind": "paired_current_run"})
    historical_clock = timing_source.get("kind") == "historical_accepted"
    timing_clean = data["baseline"]["clean"] and data["clock"]["clean"]
    capture_acceptable = data["capture"]["clean"] or historical_clock
    if not timing_clean or not capture_acceptable:
        raise ValueError("Timing source or numerical capture is not acceptable")
    if not all(row["passed"] for row in data["trajectory_parity"]) or abs(data["median_clock_overhead_percent"]) > 5:
        raise ValueError("Unmatched trajectories or clock overhead")
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    output.mkdir(parents=True); (output / "frames").mkdir(); (output / "coordinate_planes").mkdir()
    source = Path(__file__).parent
    with (output / "tests.txt").open("w") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromNames([
                "scripts.visualization.mrdbscp_progress.test_progress",
                "scripts.visualization.mrdbscp_progress.test_render_revision",
                "scripts.visualization.mrdbscp_progress.test_compact_render"]))
    if not tests.wasSuccessful():
        raise RuntimeError("Render contracts failed")
    static = dict(np.load(data["static"]))
    slices = [dict(np.load(row["slices"])) for row in data["snapshots"]]
    initial_error, final_error = None, None
    with torch.no_grad():
        for i, row in enumerate(data["snapshots"]):
            controls = np.load(row["controls"])["controls"]
            dense = c.expand_on_shape(torch.from_numpy(controls).reshape(-1, 3), controls.shape[:3],
                                      row["full_cps"], c.IMAGE_SHAPE, 1)
            planes = coordinate_planes(dense[:, :, 96].numpy())
            slices[i].update(planes)
            np.savez_compressed(output / "coordinate_planes" / f"state{i:03d}.npz", **planes)
            if i == 0:
                initial_error = float(dense.abs().max())
            if i == len(slices)-1:
                reference = np.load(data["capture"]["dvf"], mmap_mode="r")
                delta = (dense.numpy()-reference) * ((np.array(c.IMAGE_SHAPE)[::-1]-1)/2)
                squared = np.square(delta).sum(-1)
                final_error = dict(vector_rms_voxel=float(np.sqrt(squared.mean())),
                                   vector_max_voxel=float(np.sqrt(squared.max())))
            del dense
    if initial_error != 0 or final_error["vector_rms_voxel"] > .005 or final_error["vector_max_voxel"] > .05:
        raise ValueError("Control reconstruction failed")
    clock = presentation_clock(data)
    data = remap_display_clock(data, clock)
    states, ticks, phases = compact_timeline(data, clock)
    write_json(output / "presentation_timing.json", dict(clock=clock, phases=phases,
        video_duration_sec=15., optimization_video_sec=13.7, fps=30,
        field_step_interval=args.field_step_interval, video_name=video_name,
        timing_policy=("Only optimization stages and final DVF ready are displayed. Historical "
                       "stage spans set the proportional playback; nonoptimization intervals "
                       "are omitted visually while the accepted endpoint remains unchanged.")))
    write_json(output / "playback.json", states); csv_table(output / "playback.csv", states)
    write_json(output / "frame_timeline.json", ticks); csv_table(output / "frame_timeline.csv", ticks)
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.write_frames(str(output / video_name), (1920,1080), fps=30,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        output_params=["-movflags", "+faststart", "-threads", "6"], macro_block_size=1)
    encoder.send(None); fig=None; current=-1
    try:
        for tick in ticks:
            if tick["state_index"] != current:
                if fig is not None: plt.close(fig)
                state = dict(states[tick["state_index"]], loss_step=tick["loss_step"],
                             status=tick["status"], status_detail=tick["status_detail"])
                fig = compact_frame(data, static, slices, state, keep_figure=True)
                current = tick["state_index"]
                save = True
            else:
                save = False
            update_frame(fig, data, tick); fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[...,:3].copy(); encoder.send(rgb)
            if save: Image.fromarray(rgb).save(output / states[current]["png"])
            if tick["video_frame"] % 90 == 0:
                print(json.dumps(dict(encoded=tick["video_frame"], total=len(ticks))), flush=True)
    finally:
        encoder.close()
        if fig is not None: plt.close(fig)
    ends = [next(i for i,s in enumerate(states) if not s["initial"]
             and data["snapshots"][s["snapshot_index"]]["stage"] == stage
             and data["snapshots"][s["snapshot_index"]]["final"]) for stage in range(3)]
    shutil.copy2(output / states[0]["png"], output / "initial.png")
    shutil.copy2(output / states[-1]["png"], output / "final.png")
    for stage,index in enumerate(ends): shutil.copy2(output / states[index]["png"], output / f"stage{stage+1}_final.png")
    for name in ("losses.csv","snapshots.csv","timeline.csv","trajectory_parity.json","overhead.json"):
        shutil.copy2(args.data.parent/name, output/name)
    html_player(output/"index.html", [states[i]["playback_start_sec"] for i in ends], video_name)
    reader=imageio_ffmpeg.read_frames(str(output/video_name),pix_fmt="rgb24")
    metadata=next(reader); count=sum(1 for _ in reader); reader.close()
    checks=dict(tests=tests.wasSuccessful(), frames=count==450, dimensions=tuple(metadata["size"])==(1920,1080),
        fps=abs(metadata["fps"]-30)<.01, codec=metadata["codec"]=="h264" and "yuv420p" in metadata["pix_fmt"],
        initial_zero=initial_error==0, reconstruction=final_error["vector_max_voxel"]<=.05,
        endpoint=ticks[-1]["elapsed_sec"]==data["endpoint_sec"], all_losses=ticks[-1]["loss_step"]==150,
        loss_rows=len(data["history"])==150,
        snapshots=len(data["snapshots"])==sum(len(checkpoint_steps(v, args.field_step_interval)) for v in data["steps"]),
        duration_15_seconds=abs(count / metadata["fps"] - 15.) < .02,
        phase_statuses=all(row["status"] for row in ticks),
        optimization_or_ready_only=all(row["phase_kind"] in ("optimization", "final_hold") for row in ticks),
        cpu_only=not torch.cuda.is_initialized())
    acceptance=dict(passed=all(checks.values()),checks=checks,video_metadata=metadata,decoded_frames=count,
        endpoint_sec=data["endpoint_sec"],source_data=str(args.data),video_name=video_name,
        field_step_interval=args.field_step_interval,initial_field_max_abs=initial_error,
        final_reconstruction=final_error,tests_run=tests.testsRun,renderer_sha256=digest(__file__),
        ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(),ffmpeg_binary_sha256=digest(imageio_ffmpeg.get_ffmpeg_exe()))
    write_json(output/"acceptance.json",acceptance)
    provenance=dict(argv=sys.argv,utc=datetime.now(timezone.utc).isoformat(),git_head=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),git_status=subprocess.check_output(
        ["git","status","--porcelain"],text=True),source_sha256={p.name:digest(p) for p in source.glob("*.py")})
    write_json(output/"provenance.json",provenance)
    if historical_clock:
        timing_lines = [
            f"Accepted historical Case04 warm endpoint and displayed clock: {data['endpoint_sec']:.6f}s.",
            f"Separate concurrent capture: {data['capture']['elapsed_sec']:.6f}s; its wall time is excluded.",
        ]
    else:
        timing_lines = [
            f"Selected clean baseline: {data['baseline']['elapsed_sec']:.6f}s; displayed clock: {data['endpoint_sec']:.6f}s; separate capture: {data['capture']['elapsed_sec']:.6f}s.",
            f"Median paired clock overhead: {data['median_clock_overhead_percent']:.3f}%; selected median-baseline pair index: {data['selected_pair_index']}.",
        ]
    report=["# Case04 fixed-50 cache/batch progress animation","",
      "Single-case presentation artifact; not cohort evidence. CP-BE numerical-equivalence warning retained.","",
      "## Method and timing","",
      "Fixed50/50/50, cache+batch+CP-BE, spacing8/4/2, downsampling4/2/1, LR.03/.01/.003, alpha1000, seed1, cuDNN F/T.",
      *timing_lines,"",
      f"Timing provenance: {timing_source.get('kind')}. Capture clean-monitor verdict: {data['capture']['clean']}; capture wall time is not used.","",
      "## Display semantics","",
      "One combined panel shows raw1-NCC,1000*BE and their sum. The original optimized -NCC+1000*BE is shifted by +1 only for display; optimization and gradients are unchanged.",
      f"Losses are pre-update; fields are measured post-update snapshots every {args.field_step_interval} step(s). Curves are not smoothed or connected across stage-resolution changes.",
      "This 15-second presentation replay displays only the three optimization stages and final DVF-ready state. The 13.7 optimization seconds are allocated in proportion to measured stage spans. Setup, transitions, and output serialization are omitted visually; the accepted warm endpoint and displayed final time are unchanged.",
      "Within each stage, steps are evenly positioned because only the accepted stage total—not per-step historical timestamps—was measured.","",
      f"Encoding/structural acceptance: {acceptance['passed']}; 450 frames; {len(data['snapshots'])} field states;150 loss rows."]
    (output/"report.md").write_text("\n".join(report)+"\n")
    write_json(output/"output_sha256.json",{p.relative_to(output).as_posix():digest(p) for p in output.rglob("*") if p.is_file() and p.name!="output_sha256.json"})
    print(json.dumps(acceptance),flush=True)
    if not acceptance["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
