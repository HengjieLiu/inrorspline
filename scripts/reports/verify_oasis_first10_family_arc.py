#!/usr/bin/env python
"""Independent saved-file checks; accept only after explicit visual review."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    read = lambda name: json.loads((root / name).read_text())
    manifest, rendered, result = read("manifest.json"), read("RENDERED.json"), read("results.json")
    for name, expected in rendered["output_sha256"].items():
        assert sha(root / name) == expected, name
    for path, expected in manifest["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    with (root / "per_case.csv").open() as f:
        cases = list(csv.DictReader(f))
    with (root / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert len(cases) == len(result["per_case"]) == 700
    assert len(summary) == len(result["summary"]) == 70
    group = lambda r: (r["panel"], r["series"], float(r["alpha"]))
    metrics = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
    seen, comparisons = set(), 0
    for record in summary:
        key = group(record)
        assert key not in seen
        seen.add(key)
        rows = [r for r in cases if group(r) == key]
        assert sorted(int(r["case_rank"]) for r in rows) == list(range(10))
        assert int(record["n"]) == 10
        for row in rows:
            assert [row["fixed_id"], row["moving_id"]] == manifest["pairs"][int(row["case_rank"])]
            assert row["direction"] == "moving_to_fixed"
        for metric in metrics:
            values = np.array([float(r[metric]) for r in rows], dtype=np.float64)
            assert np.isfinite(values).all()
            assert abs(values.mean() - float(record[metric + "_mean"])) < 1e-12
            assert abs(values.std(ddof=1) - float(record[metric + "_sd"])) < 1e-12
            comparisons += 2
    for csv_row, json_row in zip(cases, result["per_case"]):
        assert group(csv_row) == group(json_row)
        assert int(csv_row["case_rank"]) == json_row["case_rank"]
        for metric in metrics:
            assert float(csv_row[metric]) == json_row[metric]
    point_counts = dict(A=15, B=21, C=3, D=3, E=28)
    assert {p: sum(r["panel"] == p for r in summary) for p in point_counts} == point_counts
    assert len(manifest["rendered_axes"]) == 30
    axes_seen = set()
    for axis in manifest["rendered_axes"]:
        metric = axis["metric"]
        key = (axis["panel"], axis["row"], metric)
        assert key not in axes_seen
        axes_seen.add(key)
        for stem, spec in (("x", manifest["axes"]["x"]), ("y", manifest["axes"][metric])):
            assert axis[stem + "lim"] == spec["limits"]
            assert axis[stem + "ticks"] == spec["ticks"]
            assert axis[stem + "label"] == spec["label"]
        assert axis["point_count"] == point_counts[axis["panel"]]
        assert axis["ellipse_count"] == (point_counts[axis["panel"]] if axis["row"] else 0)
    png_sizes = {}
    for panel in "ABCDE":
        png = (root / f"panel_{panel}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        png_sizes[panel] = struct.unpack(">II", png[16:24])
        assert png_sizes[panel] == (4000, 2200)
        xml = ET.parse(root / f"panel_{panel}.svg")
        axes = [e for e in xml.iter() if re.fullmatch(r"axes_\d+", e.get("id", ""))]
        assert len(axes) == 6
        pdf = (root / f"panel_{panel}.pdf").read_bytes()
        assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == 1
        assert b"/FontFile2" in pdf  # Embedded TrueType rather than raster text.
    book = (root / "arc_panels_A_to_E.pdf").read_bytes()
    assert len(re.findall(rb"/Type\s*/Page\b", book)) == 5
    return {"complete": True, "case_count": 700, "point_count": 70,
            "independent_mean_sd_comparisons": comparisons, "axes_checked": 30,
            "points_by_panel": point_counts, "png_sizes": png_sizes,
            "five_page_pdf": True, "embedded_pdf_fonts": True,
            "source_hashes_unchanged": True, "csv_json_equal": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--confirm-visual-review", action="store_true")
    parser.add_argument("--container-image", default="")
    args = parser.parse_args()
    validation = verify(args.root)
    if args.confirm_visual_review:
        validation.update(visual_review_complete=True,
                          visual_review="All five rendered PNGs inspected: readable labels/legends, no clipping; shared broad axes intentional.",
                          accepted_at=datetime.now(timezone.utc).isoformat(),
                          container_image=args.container_image, verifier_sha256=sha(Path(__file__)),
                          output_sha256={p.name: sha(p) for p in sorted(args.root.iterdir()) if p.is_file()})
        with (args.root / "ACCEPTED.json").open("x", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "output_sha256"}, indent=2))
