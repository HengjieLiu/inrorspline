"""CPU-only, non-overwriting revision: geometric grid + every-step loss playback.

Reuse accepted data and controls. No optimization, image warp or metric rerun.
Grid lines are level sets of the existing PULL coordinates, not pushed vertices.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from scripts.experiments.oasis_alpha_sweep import common as c
from .geometry import display_plane
from .offline import csv_table
from .preview import digest, validate_output, write_json
from .render import frame, schedule


def coordinate_planes(displacement):
    """Voxel source coordinates on axial index 96; normal component is not a grid axis.

    Input [D,H,3] contains the full 3-D field at the selected plane, normalized
    grid-sample XYZ. Display axes are flipped D and transposed H, as for MRI.
    The reference lattice extends beyond the image FOV; only viewport clipping
    applies. This is a visualization convention, not a change to image padding.
    """
    d, h = displacement.shape[:2]
    z, y = np.meshgrid(np.arange(d), np.arange(h), indexing="ij")
    return dict(phi_z=display_plane(z + displacement[..., 2] * ((d-1)/2)),
                phi_y=display_plane(y + displacement[..., 1] * ((h-1)/2)))


def draw_grid(ax, planes):
    """Rasterize white geometric strokes at final screen resolution, not a texture."""
    height, width = planes["phi_z"].shape
    ax.imshow(np.zeros((height, width)), cmap="gray", vmin=0, vmax=1,
              origin="lower", aspect="equal", interpolation="nearest")
    for key in ("phi_z", "phi_y"):
        values = planes[key]
        levels = np.arange(np.ceil(values.min()/8)*8, values.max()+1e-6, 8.)
        if len(levels):
            ax.contour(values, levels=levels, colors="white", linewidths=1.2,
                       antialiased=True)
    ax.set_xlim(-.5, width-.5)
    ax.set_ylim(-.5, height-.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def playback_frames(data, states, fps=30):
    """Uniform real-time replay with explicit holds; reveal ONLY observed loss rows.

    Each video tick maps to the recorded clock interval. The clock between
    callbacks is a playback coordinate, not an additional timing measurement.
    Losses are never smoothed/interpolated and never joined across stages.
    """
    times = np.array([h["elapsed_sec"] for h in data["history"]])
    if np.any(np.diff(times) < 0):
        raise ValueError("Nonmonotonic per-step timing")
    result = []
    for i, state in enumerate(states):
        row = data["snapshots"][state["snapshot_index"]]
        hold = 1. if i == 0 else 3. if i == len(states)-1 else 0.
        if i not in (0, len(states)-1) and row["final"] and row["stage"] < 2:
            hold += 1.
        start = state["elapsed_sec"]
        end = states[i+1]["elapsed_sec"] if i+1 < len(states) else start
        span = state["playback_end_sec"] - state["playback_start_sec"] - hold
        for _ in range(state["video_frames"]):
            video_sec = len(result)/fps
            progress = np.clip((video_sec-state["playback_start_sec"]-hold)/span, 0, 1) if span > 1e-10 else 0.
            clock = float(start + progress*(end-start))
            count = int(np.searchsorted(times, clock, side="right"))
            step = int(data["history"][count-1]["global_epoch"]) if count else 0
            result.append(dict(video_frame=len(result), video_sec=video_sec, state_index=i,
                snapshot_index=state["snapshot_index"], elapsed_sec=clock,
                snapshot_elapsed_sec=row["elapsed_sec"], loss_step=step,
                loss_rows_revealed=count, holding=bool(video_sec < state["playback_start_sec"]+hold)))
    return result


def update_losses(fig, data, tick):
    for key, lines, cursor, boundaries in fig._loss_artists:
        for stage, line in enumerate(lines):
            rows = [r for r in data["history"][:tick["loss_rows_revealed"]] if int(r["stage_idx"]) == stage]
            line.set_data([r["global_epoch"] for r in rows], [r[key] for r in rows])
        cursor.set_xdata([tick["loss_step"], tick["loss_step"]])
        cursor.set_visible(tick["loss_step"] not in boundaries)
    fig._elapsed_text.set_text(f"Warm elapsed: {tick['elapsed_sec']:.2f} s")


def html_player(path, ends):
    # Native local video playback keeps every-step loss updates identical to MP4.
    path.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Case 04 · revised grid and continuous loss</title>
<style>body{font:18px system-ui;max-width:1500px;margin:20px auto;background:#e9edf1;color:#132333}video{width:100%}button,select{font:inherit;margin:6px}input{width:95%}</style>
<h1>Case 04 · MR-D-BSCP + control-space BE</h1>
<p>Geometric white grid · every-step recorded loss · discrete real deformation snapshots. No interpolated fields.</p>
<video id="video" controls preload="metadata" src="animation.mp4" poster="initial.png"></video>
<div><button id="play">Play / pause</button><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<button onclick="jump(ENDS[0])">Stage 1 end</button><button onclick="jump(ENDS[1])">Stage 2 end</button><button onclick="jump(ENDS[2])">Stage 3 end</button></div>
<label>Snapshot <input id="snap" type="range" min="0" step="1"></label><p id="status"></p>
<p><a href="animation.mp4">Download MP4</a> · <a href="report.md">Report</a> · <a href="losses.csv">Losses</a> · <a href="frame_timeline.csv">Frame mapping</a></p>
<script>const ENDS=__ENDS__,TIMES=__TIMES__;const v=document.getElementById('video'),s=document.getElementById('snap');
s.max=TIMES.length-1;function jump(t){v.pause();v.currentTime=t;}
document.getElementById('play').onclick=()=>v.paused?v.play():v.pause();document.getElementById('speed').onchange=e=>v.playbackRate=Number(e.target.value);
s.oninput=()=>jump(TIMES[Number(s.value)]);v.ontimeupdate=()=>{document.getElementById('status').textContent='Playback '+v.currentTime.toFixed(2)+' / 30 s';let i=TIMES.findIndex(t=>t>v.currentTime);s.value=i<0?TIMES.length-1:Math.max(0,i-1);};</script></html>'''
        .replace("__ENDS__", json.dumps(ends["stages"]))
        .replace("__TIMES__", json.dumps(ends["states"])))


def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--previous-render", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = validate_output(args.output, Path(os.environ["ARTIFACT_ROOT"]))
    data = json.loads(args.data.read_text())
    previous = json.loads((args.previous_render / "delivery_validation.json").read_text())
    old_acceptance = json.loads((args.previous_render / "acceptance.json").read_text())
    if not previous["passed"] or not old_acceptance["passed"] or old_acceptance["source_data"] != str(args.data):
        raise ValueError("Require an accepted previous render of this exact source data")
    if not all(data[k]["clean"] for k in ("baseline", "clock", "capture")) or not all(r["passed"] for r in data["trajectory_parity"]):
        raise ValueError("Unaccepted numerical inputs")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    old_hashes = inventory(args.previous_render)
    inputs = [args.data, Path(data["static"]), Path(data["capture"]["dvf"])]
    inputs += [Path(r[key]) for r in data["snapshots"] for key in ("controls", "slices")]
    inputs += [args.data.parent / name for name in ("losses.csv", "snapshots.csv", "timeline.csv")]
    input_hashes = {str(p): digest(p) for p in inputs}
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "coordinate_planes").mkdir()
    source = Path(__file__).parent
    (output / "source").mkdir()
    source_hashes = {}
    for p in source.glob("*.py"):
        shutil.copy2(p, output / "source" / p.name)
        source_hashes[p.name] = digest(p)
    for name in ("report.md", "acceptance.json", "delivery_validation.json", "output_sha256.json"):
        shutil.copy2(args.previous_render / name, output / ("previous_"+name))
    write_json(output / "provenance.json", dict(argv=sys.argv, utc=datetime.now(timezone.utc).isoformat(),
        cwd=str(Path.cwd()), source_hashes=source_hashes, input_sha256=input_hashes,
        previous_render=str(args.previous_render), previous_sha256=old_hashes,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        torch_version=torch.__version__, numpy_version=np.__version__,
        environment={key: os.environ.get(key) for key in ("ARTIFACT_ROOT", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "MPLCONFIGDIR")}))
    with (output / "tests.txt").open("w") as stream:
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "scripts.visualization.mrdbscp_progress.test_progress",
            "scripts.visualization.mrdbscp_progress.test_render_revision"])
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise RuntimeError("Revision contracts failed; see tests.txt")
    slices = [dict(np.load(r["slices"])) for r in data["snapshots"]]
    static = dict(np.load(data["static"]))
    with torch.no_grad():
        for i, row in enumerate(data["snapshots"]):
            controls = np.load(row["controls"])["controls"]
            dense = c.expand_on_shape(torch.from_numpy(controls).reshape(-1, 3), controls.shape[:3],
                                      row["full_cps"], c.IMAGE_SHAPE, 1)
            if not torch.isfinite(dense).all() or dense.dtype != torch.float32:
                raise ValueError("Invalid reconstructed field")
            planes = coordinate_planes(dense[:, :, 96].numpy())
            slices[i].update(planes)
            np.savez_compressed(output / "coordinate_planes" / f"state{i:03d}.npz", **planes)
            if i == 0:
                initial_error = float(dense.abs().max())
            if i == len(slices)-1:
                reference = np.load(data["capture"]["dvf"], mmap_mode="r")
                diff = (dense.numpy() - reference) * ((np.array(c.IMAGE_SHAPE)[::-1]-1)/2)
                squared = np.square(diff).sum(-1)
                final_error = dict(vector_rms_voxel=float(np.sqrt(squared.mean())),
                                   vector_max_voxel=float(np.sqrt(squared.max())))
                del diff, squared
            print(json.dumps(dict(reconstructed_grid=i+1, total=len(slices))), flush=True)
            del dense
    if initial_error != 0 or final_error["vector_rms_voxel"] > .005 or final_error["vector_max_voxel"] > .05:
        raise ValueError("CPU reconstruction failed parity")
    states = schedule(data)
    ticks = playback_frames(data, states)
    write_json(output / "frame_timeline.json", ticks)
    csv_table(output / "frame_timeline.csv", ticks)
    write_json(output / "playback.json", states)
    csv_table(output / "playback.csv", states)
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.write_frames(str(output / "animation.mp4"), (1920,1080), fps=30,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        output_params=["-movflags", "+faststart", "-threads", "6"], macro_block_size=1)
    encoder.send(None)
    fig, current = None, -1
    try:
        for tick in ticks:
            index = tick["state_index"]
            if index != current:
                if fig is not None:
                    plt.close(fig)
                state = dict(states[index], loss_step=tick["loss_step"])
                fig = frame(data, static, slices, state, None, grid_drawer=draw_grid, keep_figure=True)
                current = index
                save_png = True
            else:
                save_png = False
            update_losses(fig, data, tick)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            encoder.send(rgb)
            if save_png:
                Image.fromarray(rgb).save(output / states[index]["png"])
            if tick["video_frame"] % 90 == 0:
                print(json.dumps(dict(encoded_frame=tick["video_frame"], total=len(ticks))), flush=True)
    finally:
        encoder.close()
        if fig is not None:
            plt.close(fig)
    ends = [next(i for i, state in enumerate(states) if not state["initial"]
            and data["snapshots"][state["snapshot_index"]]["stage"] == stage
            and data["snapshots"][state["snapshot_index"]]["final"]) for stage in range(3)]
    for stage, index in enumerate(ends):
        shutil.copy2(output / states[index]["png"], output / f"stage{stage+1}_final.png")
    shutil.copy2(output / states[0]["png"], output / "initial.png")
    shutil.copy2(output / states[-1]["png"], output / "final.png")
    html_player(output / "index.html", dict(stages=[states[i]["playback_start_sec"] for i in ends],
        states=[s["playback_start_sec"] for s in states]))
    for name in ("losses.csv", "snapshots.csv", "timeline.csv", "trajectory_parity.json", "overhead.json"):
        shutil.copy2(args.data.parent / name, output / name)
    for name in ("stage_timings.json", "final_quality.json"):
        shutil.copy2(args.previous_render / name, output / name)
    reader = imageio_ffmpeg.read_frames(str(output / "animation.mp4"), pix_fmt="rgb24")
    metadata = next(reader)
    count = sum(1 for _ in reader)
    reader.close()
    advancing = sum(b["loss_step"] > a["loss_step"] and b["snapshot_index"] == a["snapshot_index"]
                    for a, b in zip(ticks, ticks[1:]))
    checks = dict(tests_passed=tests.wasSuccessful(), video_frames=count == 900,
        video_dimensions=tuple(metadata["size"]) == (1920,1080), video_fps=abs(metadata["fps"]-30)<.01,
        codec=metadata["codec"] == "h264" and "yuv420p" in metadata["pix_fmt"],
        initial_zero_field=initial_error == 0, reconstruction_parity=final_error["vector_max_voxel"] <= .05,
        loss_advances_between_snapshots=advancing > 100,
        real_clock_monotonic=all(b["elapsed_sec"] >= a["elapsed_sec"] for a,b in zip(ticks,ticks[1:])),
        endpoint_preserved=ticks[-1]["elapsed_sec"] == data["endpoint_sec"],
        final_loss_visible=ticks[-1]["loss_step"] == sum(data["steps"]),
        cpu_only=not torch.cuda.is_initialized(), previous_render_unchanged=inventory(args.previous_render) == old_hashes,
        inputs_unchanged=all(digest(p) == h for p,h in input_hashes.items()),
        source_unchanged=all(digest(source / p) == h for p,h in source_hashes.items()))
    acceptance = dict(passed=all(checks.values()), checks=checks, tests_run=tests.testsRun,
        video_metadata=metadata, decoded_frames=count, initial_field_max_abs=initial_error,
        cpu_final_reconstruction=final_error, advancing_loss_frames_between_snapshots=advancing,
        endpoint_sec=data["endpoint_sec"], source_data=str(args.data),
        ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(), ffmpeg_binary_sha256=digest(imageio_ffmpeg.get_ffmpeg_exe()))
    write_json(output / "acceptance.json", acceptance)
    report = (args.previous_render / "report.md").read_text()
    report = "# Presentation revision: geometric grid and every-step loss\n\n" + (
        "This is an offline presentation-only revision of the accepted Case 04 run. No optimization, timing or metric rerun. "
        "Previous render directories and numerical inputs are checksum-verified unchanged.\n\n"
        "## Grid correction\n\n"
        "The old 1-voxel binary texture lost lines during nearest downsampling and faded during trilinear image sampling. "
        "This revision draws phi_z=8k and phi_y=8k contour curves of the full-resolution 3-D PULL map at axial index 96, "
        "in source voxel coordinates. Strokes are white, 1.2 points wide (1.67 pixels at 100 dpi), antialiased at the "
        "final panel resolution. Edge pixels may be gray; stroke opacity does not depend on subvoxel texture blending. "
        "The reference lattice is implicitly infinite and extruded through-plane; there is no black source-volume padding "
        "for grid curves. Only viewport clipping applies. MRI/label padding and interpolation are UNCHANGED. "
        "Curves follow coordinate-map level sets on the voxel-sampled field, not forward material trajectories or the control grid; "
        "folds/compression may legitimately bend or merge visible lines. No claim of invertibility is added.\n\n"
        "## Independent loss and field cadence\n\n"
        "All 450 actual per-step loss rows are retained without smoothing, fabricated values or connections across stages. "
        "At each 30fps frame, reveal losses whose recorded timestamps have been reached. Video sampling can skip displaying "
        "individual steps alone, but every reached point remains in the drawn trace. Images/labels/grid/Dice hold the latest "
        "one of 49 measured snapshots; completed stage columns freeze. Caption states the FIELD snapshot step/time. "
        "The main clock follows uniform playback mapping of the measured warm interval, not snapshot-write time or a new "
        "timestamp measurement between callbacks. Six seconds of explicit holds freeze both the clock and losses. "
        "Losses remain pre-update and controls post-update. `frame_timeline.csv/json` records every video frame's clock, "
        "loss step and snapshot identity. Offline HTML uses the same MP4, plus snapshot scrubbing and stage jumps.\n\n"
        f"CPU final-control reconstruction error: {final_error}. Loss advances on {advancing} frames while the field stays fixed. "
        f"Acceptance passed: {acceptance['passed']}; tests: {tests.testsRun}; decoded frames: {count}.\n\n"
        "`coordinate_planes/` permits grid-only rerendering; source snapshots, input/old-output hashes and exact command are "
        "in `source/` and `provenance.json`. PNGs are snapshot milestones; they are not the complete per-frame loss animation. "
        "The original numerical report is preserved below; its old frame-player description is superseded above.\n\n---\n\n") + report
    (output / "report.md").write_text(report)
    write_json(output / "output_sha256.json", inventory(output))
    print(json.dumps(acceptance), flush=True)
    if not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
