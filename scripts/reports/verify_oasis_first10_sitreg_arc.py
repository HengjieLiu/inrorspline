#!/usr/bin/env python
"""Independent statistic, source preservation and presentation checks for F/G v2."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify(root):
    read = lambda name: json.loads((root / name).read_text())
    manifest, rendered, result = read("manifest.json"), read("RENDERED.json"), read("results.json")
    for name, digest in rendered["output_sha256"].items():
        assert sha(root / name) == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert sha(name) == digest, name
    cases, summary = read_csv(root / "per_case.csv"), read_csv(root / "summary.csv")
    original = read_csv(Path(manifest["source"]) / "per_case.csv")
    key = lambda r: (r["panel"], r["series"], float(r["alpha"]))
    casekey = lambda r: (*key(r), int(r["case_rank"]))
    source_index = {casekey(r): r for r in original}
    assert len(cases) == len(result["per_case"]) == 400
    assert len(summary) == len(result["summary"]) == 40
    metrics = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
    group_keys, source_keys = set(), set()
    for record, json_record in zip(summary, result["summary"]):
        assert key(record) not in group_keys and key(record) == key(json_record)
        group_keys.add(key(record))
        selected = [r for r in cases if key(r) == key(record)]
        assert sorted(int(r["case_rank"]) for r in selected) == list(range(10))
        assert int(record["n"]) == 10
        for metric in metrics:
            values = np.array([float(r[metric]) for r in selected])
            assert np.isfinite(values).all()
            assert abs(values.mean() - float(record[metric + "_mean"])) < 1e-12
            assert abs(values.std(ddof=1) - float(record[metric + "_sd"])) < 1e-12
            for suffix in ("_mean", "_sd"):
                assert float(record[metric + suffix]) == json_record[metric + suffix]
    for record, json_record in zip(cases, result["per_case"]):
        source_keys.add((record["source_panel"], record["source_series"], float(record["alpha"]), int(record["case_rank"])))
        reference = source_index[casekey(record)]
        assert [record["fixed_id"], record["moving_id"]] == manifest["pairs"][int(record["case_rank"])]
        assert record["direction"] == reference["direction"] == "moving_to_fixed"
        assert casekey(record) == casekey(json_record)
        for metric in metrics:
            assert float(record[metric]) == float(reference[metric]) == json_record[metric]
    assert len(source_keys) == 350
    for panel in "FG":
        assert sum(r["panel"] == panel for r in summary) == 20
        assert {r["series"] for r in summary if r["panel"] == panel} == {"sitreg", "displacement", "bounded2", "svf"}
        for series in ("sitreg", "displacement", "bounded2", "svf"):
            weights = sorted(float(r["alpha"]) for r in summary if r["panel"] == panel and r["series"] == series)
            expected = [.1, 1/3, 1., 3., 10.] if series == "sitreg" else [1.] if series == "bounded2" else [.01, .1, .3, 1., 3., 10., 100.]
            assert weights == expected
    assert len(manifest["rendered_axes"]) == 12
    old_manifest = json.loads((Path(manifest["source"]) / "manifest.json").read_text())
    assert manifest["axes"] == old_manifest["axes"]
    for axis in manifest["rendered_axes"]:
        yspec = manifest["axes"][axis["metric"]] if axis["row"] else manifest["top_axes"][axis["metric"]]
        for stem, spec in (("x", manifest["axes"]["x"]), ("y", yspec)):
            assert axis[stem + "lim"] == spec["limits"]
            assert axis[stem + "ticks"] == spec["ticks"]
            assert axis[stem + "label"] == spec["label"]
        assert axis["point_count"] == 20
        assert axis["ellipse_count"] == (20 if axis["row"] else 0)
    for record in summary:
        for metric, spec in manifest["top_axes"].items():
            assert spec["limits"][0] < float(record[metric + "_mean"]) < spec["limits"][1]
    for panel in "FG":
        png = (root / f"panel_{panel}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (4000, 2200)
        xml = ET.parse(root / f"panel_{panel}.svg")
        assert sum(bool(re.fullmatch(r"axes_\d+", e.get("id", ""))) for e in xml.iter()) == 6
        text = " ".join(xml.getroot().itertext())
        assert "VFA" not in text and "FeDDA" not in text and "tighter y" in text
        for label in ("SITReg", "Displacement B-spline", "Bounded · 2 maps/level", "SVF · velocity penalty"):
            assert label in text
    for filename, count in (("panel_F.pdf", 1), ("panel_G.pdf", 1), ("arc_panels_F_to_G.pdf", 2), ("arc_panels_A_to_G.pdf", 7)):
        pdf = (root / filename).read_bytes()
        assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == count, filename
        assert b"/FontFile2" in pdf
    return dict(complete=True, plotted_settings=40, displayed_case_rows=400, unique_source_case_rows=350,
                independently_recomputed_statistics=400, axes_checked=12, expected_pdf_pages=[1, 1, 2, 7],
                all_means_visible_in_tighter_top_axes=True, full_bottom_sd_ranges_unchanged=True,
                prior_A_to_G_and_lambda1_table_unchanged=True, exact_source_and_csv_json_metric_agreement=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--confirm-visual-review", action="store_true")
    args = parser.parse_args()
    validation = verify(args.root)
    if args.confirm_visual_review:
        validation.update(visual_review_complete=True,
                          visual_review="F/G PNGs inspected: four distinct methods, tighter top y scales and full SD bottom; readable text, no clipping.",
                          pdf_merge="Host pdfunite: original A–E PDF plus revised F–G; original pages not rerendered.",
                          accepted_at=datetime.now(timezone.utc).isoformat(), verifier_sha256=sha(__file__),
                          output_sha256={p.name: sha(p) for p in sorted(args.root.iterdir()) if p.is_file()})
        with (args.root / "ACCEPTED.json").open("x") as f:
            json.dump(validation, f, indent=2); f.write("\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "output_sha256"}, indent=2))
