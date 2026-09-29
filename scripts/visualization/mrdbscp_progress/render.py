"""Render verified offline snapshots; never execute optimization or metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import DEFAULT
from .preview import show
from .offline import csv_table


def schedule(data, duration=30., fps=30):
    """Only measured states; playback holds never advance the true clock."""
    endpoint = data["endpoint_sec"]
    moving_duration = duration - 6.
    if moving_duration <= 0 or endpoint <= 0:
        raise ValueError("Duration must exceed the six seconds of explicit holds")
    states = [dict(snapshot_index=0, elapsed_sec=0., initial=True)]
    states += [dict(snapshot_index=i, elapsed_sec=row["elapsed_sec"], initial=False)
               for i, row in enumerate(data["snapshots"])]
    states += [dict(snapshot_index=len(data["snapshots"]) - 1, elapsed_sec=endpoint,
                    initial=False, output_ready=True)]
    if any(b["elapsed_sec"] < a["elapsed_sec"] for a, b in zip(states, states[1:])):
        raise ValueError("Nonmonotonic snapshot timeline")
    cumulative = 0.
    last_frame = 0
    for i, state in enumerate(states):
        next_time = states[i+1]["elapsed_sec"] if i+1 < len(states) else endpoint
        hold = 1. if i == 0 else 3. if i == len(states)-1 else 0.
        row = data["snapshots"][state["snapshot_index"]]
        if i not in (0, len(states)-1) and row["final"] and row["stage"] < 2:
            hold += 1.
        state["playback_start_sec"] = cumulative
        cumulative += moving_duration * (next_time - state["elapsed_sec"]) / endpoint + hold
        end_frame = round(cumulative * fps)
        state["video_frames"] = end_frame - last_frame
        state["playback_end_sec"] = cumulative
        state["png"] = f"frames/frame{i:03d}.png"
        state["output_ready"] = state.get("output_ready", False)
        last_frame = end_frame
    if last_frame != round(duration * fps):
        raise ValueError("Playback allocation does not equal requested duration")
    return states


def stage_indices(data, snapshot_index):
    """Latest reached state per stage; earlier final columns never change."""
    result = {}
    for i, row in enumerate(data["snapshots"][:snapshot_index+1]):
        result[row["stage"]] = i
    return result


def frame(data, static, slices, state, destination, *, grid_drawer=None, keep_figure=False):
    index = state["snapshot_index"]
    current = data["snapshots"][index]
    chosen = stage_indices(data, index)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    fig.text(.035, .959, "MR-D-BSCP fast 8-4-2 · Control-space BE", fontsize=23, weight="bold")
    caption = ("DVF file ready" if state["output_ready"] else "Initialization" if state["initial"] else
               f"Stage {current['stage']+1}/3 · Step {current['epoch']}/{data['steps'][current['stage']]}")
    if grid_drawer is not None:
        field_caption = ("Initialization" if state["initial"] else
                         f"Stage {current['stage']+1}/3 · Step {current['epoch']}/{data['steps'][current['stage']]}")
        caption = f"Field snapshot: {field_caption} @ {current['elapsed_sec']:.2f} s"
        if state["output_ready"]:
            caption += "  |  DVF file ready"
    fig.text(.035, .923, f"Case 04  |  Fixed {data['fixed_id']} / Moving {data['moving_id']}  |  {caption}", fontsize=16)
    fig._elapsed_text = fig.text(.97, .948, f"Warm elapsed: {state['elapsed_sec']:.2f} s", fontsize=24, ha="right", color="#006e82")
    gs = fig.add_gridspec(3, 5, left=.07, right=.965, bottom=.335, top=.875, wspace=.07, hspace=.10)
    for col, key in enumerate(("moving", "fixed")):
        show(fig.add_subplot(gs[0, col]), static[key], title=key.capitalize())
        show(fig.add_subplot(gs[1, col]), static[key + "_seg"], "labels")
    labels = fig.add_subplot(gs[2, :2]); labels.axis("off")
    labels.text(.025, .92, "Slowed replay · metrics computed offline\n"
                "3D Dice: 35-label whole-volume mean\n"
                "Grid: 8-voxel pull-warped reference\n"
                "Central axial slice · left R, top A\n" +
                ("Field: every 10 steps + stage end\nLoss: every recorded step" if grid_drawer is not None else
                 "Snapshots every 10 steps + stage end"),
                va="top", fontsize=12, linespacing=1.55)
    for stage in range(3):
        active = stage == current["stage"] and not current["final"] and not state["output_ready"]
        title = f"Stage {stage+1} · cps {DEFAULT.full_cps[stage]} / down {DEFAULT.downsampling[stage]}"
        if stage not in chosen:
            for row in range(3):
                ax = fig.add_subplot(gs[row, stage+2]); ax.axis("off")
                ax.set_facecolor("#eeeeee")
                if row == 0:
                    ax.set_title(title, fontsize=12)
                    ax.text(.5, .5, "Not started", ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=14)
            continue
        chosen_index = chosen[stage]
        snap, planes = data["snapshots"][chosen_index], slices[chosen_index]
        for row, (key, kind) in enumerate((("warped", "image"), ("warped_seg", "labels"), ("grid", "grid"))):
            ax = fig.add_subplot(gs[row, stage+2])
            if kind == "grid" and grid_drawer is not None:
                grid_drawer(ax, planes)
            else:
                show(ax, planes[key], kind, title if row == 0 else None)
            if row == 0:
                ax.title.set_fontsize(12)
            if active:
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_color("#009eae"); spine.set_linewidth(2)
            if row == 2:
                ax.set_xlabel(f"3D Dice {snap['dice']:.4f}" + ("  ·  Frozen" if snap["final"] else ""), fontsize=13)
    for y, text in ((.793, "MRI"), (.608, "Labels"), (.419, "Grid")):
        fig.text(.021, y, text, rotation=90, ha="center", va="center", fontsize=14)
    history = data["history"]
    end_step = state.get("loss_step", 0 if state["initial"] else current["global_step"])
    total_steps = sum(data["steps"])
    boundaries = np.cumsum(data["steps"])
    losses = fig.add_gridspec(3, 1, left=.145, right=.965, bottom=.075, top=.285, hspace=.18)
    fig._loss_artists = []
    for row, (key, title, color) in enumerate((("data_loss", "Similarity (-NCC)", "#2166ac"),
                 ("weighted_be", "Weighted BE", "#d78100"), ("loss", "Total loss", "#6e4d91"))):
        ax = fig.add_subplot(losses[row, 0])
        all_y = [h[key] for h in history]
        lo, hi = min(all_y), max(all_y)
        margin = max((hi - lo)*.1, 1e-7)
        ax.set_ylim(lo-margin, hi+margin)
        # Keep the final-stage dashed boundary visible inside the plotting area.
        ax.set_xlim(0, total_steps + max(5, .02 * total_steps))
        stage_lines = []
        for stage in range(3):
            segment = [h for h in history if int(h["stage_idx"]) == stage and int(h["global_epoch"]) <= end_step]
            stage_lines.append(ax.plot([h["global_epoch"] for h in segment], [h[key] for h in segment], color=color, lw=1.7)[0])
        completed_boundaries = []
        for stage, boundary in enumerate(boundaries):
            if stage in chosen and data["snapshots"][chosen[stage]]["final"]:
                ax.axvline(boundary, ls="--", color="#888888", lw=1)
                completed_boundaries.append(boundary)
        cursor = ax.axvline(end_step, color="#009eae", alpha=.6, lw=1)
        cursor.set_visible(end_step not in completed_boundaries)
        fig._loss_artists.append((key, stage_lines, cursor, completed_boundaries))
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=11, labelpad=14)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Global optimization step  |  Dashed lines: completed stage boundaries", fontsize=11)
    fig.text(.035, .021, "CP-BE exploratory implementation · real warm clock, slowed playback · loss before update; displayed field after update",
             fontsize=11, color="#555555")
    if destination is not None:
        fig.savefig(destination, dpi=100)
    if keep_figure:
        return fig
    plt.close(fig)


def html_player(path, states, stages):
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Case 04 optimization replay</title>
<style>body{font:18px system-ui;max-width:1500px;margin:20px auto;background:#e9edf1;color:#132333}img,video{width:100%}button,select{font:inherit;margin:6px}input{width:98%}</style>
<h1>Case 04 · MR-D-BSCP + control-space BE</h1><p>Measured warm clock; slowed replay. Metrics calculated offline. No interpolated deformation states.</p>
<img id="frame" alt="Optimization snapshot"><div><button onclick="playing=!playing;last=performance.now()">Play / pause</button>
<select id="speed"><option value="0.5">0.5× playback</option><option value="1" selected>1× playback</option><option value="2">2× playback</option></select>
<button onclick="jump(ENDS[0])">Stage 1 end</button><button onclick="jump(ENDS[1])">Stage 2 end</button><button onclick="jump(ENDS[2])">Stage 3 end</button></div>
<input id="scrub" type="range" min="0" step="1"><p id="status"></p>
<p><a href="animation.mp4">Download 1080p MP4</a> · <a href="report.md">Report</a> · <a href="snapshots.csv">Snapshots / Dice</a> · <a href="losses.csv">Every-step losses</a></p>
<script>const STATES=__STATES__,ENDS=__ENDS__;let t=0,playing=false,last=performance.now(),shown=-1;
const img=document.getElementById('frame'), slider=document.getElementById('scrub');slider.max=STATES.length-1;
function draw(i){if(shown===i)return;shown=i;img.src=STATES[i].png;slider.value=i;document.getElementById('status').textContent=`Snapshot ${i+1}/${STATES.length} · measured warm time ${STATES[i].elapsed_sec.toFixed(3)} s`;}
function jump(i){playing=false;t=STATES[i].playback_start_sec;draw(i);}slider.oninput=()=>jump(Number(slider.value));
function tick(now){if(playing){t+=(now-last)/1000*Number(document.getElementById('speed').value);if(t>=30){t=30;playing=false;}let i=STATES.findIndex(s=>t<s.playback_end_sec);draw(i<0?STATES.length-1:i);}last=now;requestAnimationFrame(tick);}draw(0);requestAnimationFrame(tick);</script></html>'''
    path.write_text(template.replace("__STATES__", json.dumps(states)).replace("__ENDS__", json.dumps(stages)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import os
    artifact = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    output = args.output.resolve()
    if output.exists() or output == artifact or not output.is_relative_to(artifact):
        raise ValueError("Use a NEW external render output directory")
    data = json.loads(args.data.read_text())
    evaluation = json.loads((args.data.parent / "accepted_result.json").read_text())
    if not evaluation["clean"] or not all(data[key]["clean"] for key in ("baseline", "clock", "capture")):
        raise ValueError("Contaminated registration or evaluation cannot enter the presentation")
    if not all(c["passed"] for c in data["trajectory_parity"]) or abs(data["median_clock_overhead_percent"]) > 5:
        raise ValueError("Only verified matched trajectories may be rendered")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "render_invocation.json").write_text(json.dumps(dict(
        argv=sys.argv, cwd=str(Path.cwd()),
        environment={key: os.environ.get(key) for key in
                     ("ARTIFACT_ROOT", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MPLCONFIGDIR")}), indent=2))
    static = dict(np.load(data["static"]))
    slices = [dict(np.load(r["slices"])) for r in data["snapshots"]]
    states = schedule(data)
    for i, state in enumerate(states):
        frame(data, static, slices, state, output / state["png"])
        print(json.dumps(dict(rendered=i+1, total=len(states))), flush=True)
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.write_frames(str(output / "animation.mp4"), (1920,1080), fps=30,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        output_params=["-movflags", "+faststart", "-threads", "6"], macro_block_size=1)
    encoder.send(None)
    try:
        for state in states:
            with Image.open(output / state["png"]) as picture:
                rgb = np.asarray(picture.convert("RGB"))
            for _ in range(state["video_frames"]):
                encoder.send(rgb)
    finally:
        encoder.close()
    ends = [next(i for i, state in enumerate(states) if not state["initial"]
                 and data["snapshots"][state["snapshot_index"]]["stage"] == stage
                 and data["snapshots"][state["snapshot_index"]]["final"]) for stage in range(3)]
    for stage, index in enumerate(ends):
        shutil.copy2(output / states[index]["png"], output / f"stage{stage+1}_final.png")
    shutil.copy2(output / states[0]["png"], output / "initial.png")
    shutil.copy2(output / states[-1]["png"], output / "final.png")
    for name in ("losses.csv", "snapshots.csv", "timeline.csv", "trajectory_parity.json", "overhead.json"):
        shutil.copy2(args.data.parent / name, output / name)
    html_player(output / "index.html", states, ends)
    (output / "playback.json").write_text(json.dumps(states, indent=2))
    csv_table(output / "playback.csv", states)
    reader = imageio_ffmpeg.read_frames(str(output / "animation.mp4"), pix_fmt="rgb24")
    metadata = next(reader)
    frame_count = sum(1 for _ in reader)
    reader.close()
    valid = (frame_count == 900 and tuple(metadata["size"]) == (1920,1080) and abs(metadata["fps"]-30)<.01
             and metadata["codec"] == "h264" and "yuv420p" in metadata["pix_fmt"])
    acceptance = dict(passed=valid, video_frames=frame_count, video_metadata=metadata,
        snapshot_count=len(data["snapshots"]), loss_rows=len(data["history"]),
        median_clock_overhead_percent=data["median_clock_overhead_percent"],
        parity=data["trajectory_parity"], final_reconstruction=data["final_reconstruction"],
        source_data=str(args.data), renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        encoder_version=imageio_ffmpeg.__version__, ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(),
        ffmpeg_binary_sha256=hashlib.sha256(Path(imageio_ffmpeg.get_ffmpeg_exe()).read_bytes()).hexdigest(),
        endpoint_sec=data["endpoint_sec"],
        baseline_sec=data["baseline"]["elapsed_sec"], capture_sec=data["capture"]["elapsed_sec"])
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    (output / "render_source.py").write_text(Path(__file__).read_text())
    report = ["# Case 04 MR-D-BSCP CP-BE progress animation", "",
        "Fixed 0117 / moving 0322; seed-3 rank 3. Single-case teaching visualization, not cohort evidence.", "",
        "CP-BE's prior numerical-equivalence failure is retained. No claim of equivalence to original BE.", "",
        "## Frozen method", "",
        "MR_A_fast_decay with CP-BE alone (no cache/batch/NCC cache/loss buffer). Alpha 1000, seed 1, float32, GPU 0, six CPU threads.",
        "Control spacing 8/4/2; image downsampling 4/2/1; stage learning rates .03/.01/.003; stage limits 400/800/1300.",
        "Original adaptive stopping and exact knot insertion retained. cuDNN benchmark=False, deterministic=True; this does not imply universal CUDA bitwise determinism.", "",
        "## Real timing and playback", "",
        f"- Selected clean baseline: {data['baseline']['elapsed_sec']:.6f} s.",
        f"- Displayed measured clock run: {data['endpoint_sec']:.6f} s.",
        f"- Separate capture run: {data['capture']['elapsed_sec']:.6f} s (NOT used as displayed time).",
        f"- Median paired clock overhead: {data['median_clock_overhead_percent']:.3f}%.",
        f"- Selected pair index: {data['selected_pair_index']} (median baseline of three, not fastest).",
        "- Endpoint: case_load_complete to dvf_file_ready. No Dice, rendering or post-DVF packaging time.",
        "- 30-second, 30fps slowed replay; six seconds of explicit holds. Only saved states are shown.", "",
        "## Stages", "", "| Stage | cps / down | Steps | End clock s | Final 3D Dice |", "|---|---|---:|---:|---:|"]
    for stage in range(3):
        row = next(r for r in data["snapshots"] if r["stage"] == stage and r["final"])
        report.append(f"| {stage+1} | {DEFAULT.full_cps[stage]} / {DEFAULT.downsampling[stage]} | {data['steps'][stage]} | {row['elapsed_sec']:.4f} | {row['dice']:.5f} |")
    report += ["", "## Files and semantics", "",
        "`index.html`: offline frame playback/scrubbing/speed and stage jumps; `animation.mp4`: slide-ready H.264/yuv420p.",
        "`losses.csv`: every pre-update loss; snapshots and displayed fields are post-update. Weighted BE = 1000 × raw BE.",
        "`snapshots.csv`: per-state whole-volume 35-label Dice with valid-label count and per-label values.",
        "`playback.csv/json`: measured time to video frame mapping, no invented deformation interpolation.",
        "All stages are reconstructed at full resolution; MRI is trilinear, labels nearest, align_corners=True, zero padding.",
        "Controls and full provenance remain in the source run. Final DVF reconstruction/parity evidence is in acceptance.json.",
        f"Verification: {valid}; {frame_count} decoded frames; {len(data['snapshots'])} snapshots; {len(data['history'])} loss rows."]
    (output / "report.md").write_text("\n".join(report)+"\n")
    if not valid:
        raise RuntimeError("Encoded video validation failed")
    print(json.dumps(dict(output=str(output), passed=True)), flush=True)


if __name__ == "__main__":
    main()
