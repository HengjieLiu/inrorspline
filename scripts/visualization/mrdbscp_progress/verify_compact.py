"""Independently verify the compact fixed-50 presentation render."""
import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from .preview import digest, write_json
from .render_compact import select_field_cadence


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(value for key, value in attrs
                          if key in ("src", "href", "poster"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.render_root.resolve()
    artifact = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    result_path = root / "visual_validation.json"
    if root == artifact or not root.is_relative_to(artifact) or result_path.exists():
        raise ValueError("Require external, not-yet-visually-validated render")

    load = lambda path: json.loads(Path(path).read_text())
    acceptance = load(root / "acceptance.json")
    video_name = acceptance.get("video_name", "animation.mp4")
    field_interval = int(acceptance.get("field_step_interval", 10))
    states = load(root / "playback.json")
    ticks = load(root / "frame_timeline.json")
    data = select_field_cadence(load(acceptance["source_data"]), field_interval)
    hashes = load(root / "output_sha256.json")
    checks = {
        "render_accepted": acceptance["passed"],
        "fixed50_profile": data["presentation"]["profile"].startswith("fixed50_cached_batch"),
        "exact_features": data["presentation"]["features"] == ["cache", "batch", "cpbe"],
        "exact_steps": data["steps"] == [50, 50, 50],
        "output_hashes_match": all(digest(root / path) == value
                                   for path, value in hashes.items()),
    }
    html_text = (root / "index.html").read_text()
    html = Assets()
    html.feed(html_text)
    checks["html_assets_local_present"] = all(
        "://" not in path and (root / path).is_file() for path in html.paths)
    checks["html_video_is_new_mp4"] = f'src="{video_name}"' in html_text
    checks["all_state_pngs_present"] = all((root / state["png"]).is_file()
                                            for state in states)

    # Compact layout stage columns occupy x ~= 537:1181. Once a stage is
    # complete, its image, label, grid, Dice and header must remain unchanged.
    frozen = True
    for stage, x_bounds in enumerate(((537, 750), (752, 966), (968, 1181))):
        end = next(index for index, state in enumerate(states)
                   if not state["initial"]
                   and data["snapshots"][state["snapshot_index"]]["stage"] == stage
                   and data["snapshots"][state["snapshot_index"]]["final"])
        reference = np.array(Image.open(root / states[end]["png"]))[
            145:1010, x_bounds[0]:x_bounds[1]]
        for state in states[end + 1:]:
            candidate = np.array(Image.open(root / state["png"]))[
                145:1010, x_bounds[0]:x_bounds[1]]
            frozen &= np.array_equal(candidate, reference)
    checks["frozen_columns_pixel_identical"] = bool(frozen)

    # For cadence >1, compare frames sharing a field while loss advances. For
    # cadence 1, compare adjacent measured fields and require visible motion.
    pair = None
    for index, first in enumerate(ticks):
        if first["loss_step"] < 5:
            continue
        candidates = [second for second in ticks[index + 1:index + 25]
                      if ((field_interval > 1 and second["state_index"] == first["state_index"]
                           and second["loss_step"] >= first["loss_step"] + 1)
                          or (field_interval == 1 and second["state_index"] > first["state_index"]
                              and second["loss_step"] > first["loss_step"]))]
        if candidates:
            pair = (first["video_frame"], candidates[-1]["video_frame"])
            break
    if pair is None:
        raise ValueError("No same-field/different-loss decoded-frame pair")
    reader = imageio_ffmpeg.read_frames(str(root / video_name), pix_fmt="rgb24")
    next(reader)
    decoded = {}
    for index, buffer in enumerate(reader):
        if index in pair:
            decoded[index] = np.frombuffer(buffer, dtype=np.uint8).reshape(
                1080, 1920, 3).copy()
        if index == pair[-1]:
            break
    reader.close()
    first, second = (decoded[index] for index in pair)
    loss_changed = int(np.any(first[225:900, 1275:1865] !=
                              second[225:900, 1275:1865], axis=-1).sum())
    image_mae = float(np.abs(first[150:1005, 105:1182].astype(float) -
                                  second[150:1005, 105:1182]).mean())
    checks["decoded_loss_region_moves"] = loss_changed > 100
    if field_interval > 1:
        checks["decoded_image_region_stays_still"] = image_mae < 3.0
    else:
        checks["decoded_every_step_field_moves"] = image_mae > 0.0001
    selected_epochs = [[int(row["epoch"]) for row in data["snapshots"]
                        if int(row["stage"]) == stage] for stage in range(3)]
    checks["field_cadence_exact"] = all(
        epochs == list(range(0, 51, field_interval)) for epochs in selected_epochs)

    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "decoded_frame_pair": pair,
        "loss_changed_pixels": loss_changed,
        "static_image_mean_absolute_pixel_difference": image_mae,
        "loss_steps": [ticks[index]["loss_step"] for index in pair],
        "field_step_interval": field_interval,
        "same_snapshot_index": (ticks[pair[0]]["snapshot_index"]
                                if field_interval > 1 else None),
        "source_sha256": digest(__file__),
    }
    write_json(result_path, result)
    (root / "verification_source.py").write_text(Path(__file__).read_text())
    report = root / "report.md"
    report.write_text(
        report.read_text()
        + "\n\n## Independent encoded-frame verification\n\n"
        + f"Passed: {result['passed']}. Decoded frames {pair} share one field "
          f"snapshot/cadence comparison while loss steps advance through {result['loss_steps']}; "
          f"{loss_changed} right-panel pixels change. The static left image "
          f"region has mean encoded difference {image_mae:.6f}/255. Completed "
          "stage columns remain pixel-identical in all later milestone PNGs. "
          "Local player assets and pre-verification checksums were verified.\n"
    )
    write_json(root / "output_sha256.json", {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "output_sha256.json"
    })
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
