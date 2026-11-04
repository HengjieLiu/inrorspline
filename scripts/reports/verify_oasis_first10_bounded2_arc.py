#!/usr/bin/env python
"""Independent saved-data and figure acceptance for P76 C/F/G update."""
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
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rendered = json.loads((root / "RENDERED.json").read_text())
    results = json.loads((root / "results.json").read_text())
    for name, digest in rendered["output_sha256"].items():
        assert sha(root / name) == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert sha(name) == digest, name
    original = Path(manifest["source_p75"])
    p76 = Path(manifest["source_sweep"])
    key = lambda r: (r["panel"], r["series"], float(r["alpha"]), int(r["case_rank"]))
    orig_index = {key(r): r for r in read_csv(original / "per_case.csv")}
    fg_index = {key(r): r for r in read_csv(original / "panels_FG_sitreg_v2/per_case.csv")}
    new_index = {(float(r["alpha"]), int(r["case_rank"])): r for r in read_csv(p76 / "per_case.csv")}
    cases, summary = read_csv(root / "per_case.csv"), read_csv(root / "summary.csv")
    assert len(cases) == len(results["per_case"]) == 550
    assert len(summary) == len(results["summary"]) == 55
    metrics = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
    unique = set()
    for row, jr in zip(cases, results["per_case"]):
        p, s, a, rank = key(row)
        assert key(row) == key(jr)
        assert [row["fixed_id"], row["moving_id"]] == manifest["pairs"][rank]
        assert row["direction"] == "moving_to_fixed"
        if row["plot_source"] == "P76":
            assert p in "CF" and s == "bounded2"
            source = new_index[(a, rank)]
            origin = ("P76", a, rank)
        elif row["plot_source"] == "P75":
            assert p == "C" and s in ("unbounded_1", "bounded_1") and a == 1.
            source = orig_index[key(row)]; origin = ("P75", *key(row))
        else:
            assert row["plot_source"] == "P75_FG_sitreg_v2" and p in "FG"
            source = fg_index[key(row)]
            origin = ("P75_SITReg", a, rank) if s == "sitreg" else ("P75_FG", *key(row))
        unique.add(origin)
        for metric in metrics:
            assert float(row[metric]) == float(source[metric]) == jr[metric]
    assert len(unique) == 430
    expected = set()
    for p, series in (("C", ("unbounded_1", "bounded_1", "bounded2")),
                      ("F", ("sitreg", "displacement", "bounded2", "svf")),
                      ("G", ("sitreg", "displacement", "bounded2", "svf"))):
        for s in series:
            weights = [.1, 1/3, 1., 3., 10.] if s == "sitreg" else [1.] if s in ("unbounded_1", "bounded_1") or (p == "G" and s == "bounded2") else [.01, .1, .3, 1., 3., 10., 100.]
            expected.update((p, s, a) for a in weights)
    assert {(r["panel"], r["series"], float(r["alpha"])) for r in summary} == expected
    for row, jr in zip(summary, results["summary"]):
        k = (row["panel"], row["series"], float(row["alpha"]))
        cohort = [r for r in cases if key(r)[:3] == k]
        assert sorted(int(r["case_rank"]) for r in cohort) == list(range(10))
        for metric in metrics:
            values = np.array([float(r[metric]) for r in cohort])
            assert np.isfinite(values).all()
            assert abs(values.mean() - float(row[metric + "_mean"])) < 1e-12
            assert abs(values.std(ddof=1) - float(row[metric + "_sd"])) < 1e-12
            for suffix in ("_mean", "_sd"):
                assert float(row[metric+suffix]) == jr[metric+suffix]
    assert len(manifest["rendered_axes"]) == 18
    for axis in manifest["rendered_axes"]:
        top = axis["row"] == 0 and axis["panel"] in "FG"
        yspec = manifest["top_axes"][axis["metric"]] if top else manifest["axes"][axis["metric"]]
        for stem, spec in (("x", manifest["axes"]["x"]), ("y", yspec)):
            assert axis[stem+"lim"] == spec["limits"]
            assert axis[stem+"ticks"] == spec["ticks"]
            assert axis[stem+"label"] == spec["label"]
        count = {"C": 9, "F": 26, "G": 20}[axis["panel"]]
        assert axis["point_count"] == count and axis["ellipse_count"] == (count if axis["row"] else 0)
        for row in [r for r in summary if r["panel"] == axis["panel"]]:
            x, sx = float(row["log_jdet_std_fg_mean"]), float(row["log_jdet_std_fg_sd"]) if axis["row"] else 0.
            y, sy = float(row[axis["metric"]+"_mean"]), float(row[axis["metric"]+"_sd"]) if axis["row"] else 0.
            assert axis["xlim"][0] <= x-sx <= x+sx <= axis["xlim"][1]
            assert axis["ylim"][0] <= y-sy <= y+sy <= axis["ylim"][1]
    for panel in "CFG":
        png = (root / f"panel_{panel}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (4000, 2200)
        svg = ET.parse(root / f"panel_{panel}.svg")
        assert sum(bool(re.fullmatch(r"axes_\d+", e.get("id", ""))) for e in svg.iter()) == 6
        text = " ".join(svg.getroot().itertext())
        assert "VFA" not in text and "FeDDA" not in text
    for name, count in (("panel_C.pdf", 1), ("panel_F.pdf", 1), ("panel_G.pdf", 1), ("arc_panels_F_to_G.pdf", 2), ("arc_panels_A_to_G.pdf", 7)):
        data = (root / name).read_bytes()
        assert len(re.findall(rb"/Type\s*/Page\b", data)) == count
        assert b"/FontFile2" in data
    return {"complete": True, "operating_points": 55, "displayed_case_rows": 550, "unique_source_rows": 430,
            "independent_statistic_checks": 550, "axes_checked": 18, "source_metrics_exact": True,
            "four_level_G_not_replaced_with_three_level_sweep": True, "all_points_and_ellipses_visible": True,
            "original_artifacts_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--confirm-visual-review", action="store_true")
    args = parser.parse_args()
    result = verify(args.root)
    if args.confirm_visual_review:
        result.update(visual_review_complete=True, accepted_at=datetime.now(timezone.utc).isoformat(),
                      verifier_sha256=sha(__file__),
                      output_sha256={p.name: sha(p) for p in args.root.iterdir() if p.is_file()})
        with (args.root / "ACCEPTED.json").open("x") as f:
            json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}, indent=2))
