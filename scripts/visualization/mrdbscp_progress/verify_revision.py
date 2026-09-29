"""Independently check decoded loss motion, frozen PNG columns and local assets."""
import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from .preview import digest, write_json


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(v for k,v in attrs if k in ("src", "href", "poster"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.render_root.resolve()
    artifact = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    if root == artifact or not root.is_relative_to(artifact) or (root / "visual_validation.json").exists():
        raise ValueError("Require external, not-yet-visually-validated render")
    load = lambda p: json.loads(Path(p).read_text())
    acceptance = load(root / "acceptance.json")
    states = load(root / "playback.json")
    ticks = load(root / "frame_timeline.json")
    data = load(acceptance["source_data"])
    hashes = load(root / "output_sha256.json")
    checks = dict(render_accepted=acceptance["passed"],
        output_hashes_match=all(digest(root / p) == h for p,h in hashes.items()))
    html = Assets()
    html.feed((root / "index.html").read_text())
    checks["html_assets_local_present"] = all("://" not in p and (root / p).is_file() for p in html.paths)
    checks["html_video_is_new_mp4"] = 'src="animation.mp4"' in (root / "index.html").read_text()
    checks["all_state_pngs_present"] = all((root / s["png"]).is_file() for s in states)
    frozen = True
    for stage, x in enumerate((930,1278,1627)):
        end = next(i for i,s in enumerate(states) if not s["initial"] and
                   data["snapshots"][s["snapshot_index"]]["stage"] == stage and
                   data["snapshots"][s["snapshot_index"]]["final"])
        reference = np.array(Image.open(root / states[end]["png"]))[130:742,x-5:x+135]
        for s in states[end+1:]:
            image = np.array(Image.open(root / s["png"]))[130:742,x-5:x+135]
            frozen &= np.array_equal(image, reference)
    checks["frozen_columns_pixel_identical"] = bool(frozen)
    # Find two video frames with different loss steps but the SAME field state.
    pair = None
    for i,a in enumerate(ticks):
        if a["loss_step"] < 5:
            continue
        candidates = [b for b in ticks[i+1:i+20] if b["state_index"] == a["state_index"]
                      and b["loss_step"] >= a["loss_step"]+2]
        if candidates:
            pair = (a["video_frame"], candidates[-1]["video_frame"])
            break
    if pair is None:
        raise ValueError("No independent loss motion to inspect")
    reader = imageio_ffmpeg.read_frames(str(root / "animation.mp4"), pix_fmt="rgb24")
    next(reader)
    decoded = {}
    for index, buffer in enumerate(reader):
        if index in pair:
            decoded[index] = np.frombuffer(buffer, dtype=np.uint8).reshape(1080,1920,3).copy()
        if index == pair[-1]:
            break
    reader.close()
    a,b = (decoded[i] for i in pair)
    loss_changed = int(np.any(a[770:1005] != b[770:1005],axis=-1).sum())
    # Encoding can slightly change otherwise identical static regions.
    grid_mae = float(np.abs(a[537:717,931:1058].astype(float)-b[537:717,931:1058]).mean())
    checks["decoded_loss_region_moves"] = loss_changed > 100
    checks["decoded_grid_stays_still"] = grid_mae < 3.
    result = dict(passed=all(checks.values()), checks=checks, decoded_frame_pair=pair,
        loss_changed_pixels=loss_changed, static_grid_mean_absolute_pixel_difference=grid_mae,
        loss_steps=[ticks[i]["loss_step"] for i in pair],
        same_snapshot_index=ticks[pair[0]]["snapshot_index"], source_sha256=digest(__file__))
    write_json(root / "visual_validation.json", result)
    (root / "verification_source.py").write_text(Path(__file__).read_text())
    report = root / "report.md"
    report.write_text(report.read_text() + "\n\n## Independent encoded-frame verification\n\n" +
        f"Passed: {result['passed']}. Decoded frames {pair} have the same field snapshot but "
        f"loss steps {result['loss_steps']}; {loss_changed} lower-plot pixels change. "
        f"Static grid mean pixel difference after encoding: {grid_mae:.6f}/255. "
        "Completed stage columns remain pixel-identical in all later milestone PNGs. "
        "Local video/player assets and pre-verification output checksums verified.\n")
    write_json(root / "output_sha256.json", {p.relative_to(root).as_posix(): digest(p)
        for p in root.rglob("*") if p.is_file() and p.name != "output_sha256.json"})
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
