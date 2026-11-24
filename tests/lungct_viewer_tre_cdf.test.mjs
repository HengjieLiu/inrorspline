import test from "node:test";
import assert from "node:assert/strict";
import { buildTreCdfs, cdfSteps, countAtOrBelow, TRE_KEYS } from "../scripts/visualization/lungct_viewer/web/tre_cdf.js";

const points = [0, 0, 1, 2].map((error, id) => ({
  id,
  side: id % 2 ? "right" : "left",
  tre_mm: Object.fromEntries(TRE_KEYS.map((key) => [key, error])),
  methods: { corrfield: { status: id === 0 ? "unresolved" : "resolved" } },
}));

test("right-continuous CDF includes zero errors and all ties", () => {
  const { series, count } = buildTreCdfs(points);
  assert.equal(count, 4);
  for (const values of Object.values(series)) {
    assert.equal(countAtOrBelow(values, -0.1), 0);
    assert.equal(countAtOrBelow(values, 0), 2);
    assert.equal(countAtOrBelow(values, 0.5), 2);
    assert.equal(countAtOrBelow(values, 1), 3);
    assert.equal(countAtOrBelow(values, 2), 4);
    assert.deepEqual(cdfSteps(values).map((p) => [p.tolerance, p.percent]), [[0, 50], [1, 75], [2, 100]]);
  }
});

test("lung subsets change denominator and retain inverse exceptions", () => {
  const before = JSON.stringify(points);
  const left = buildTreCdfs(points, "left"), right = buildTreCdfs(points, "right");
  assert.equal(left.count, 2);
  assert.equal(right.count, 2);
  assert.deepEqual(left.series.corrfield, [0, 1]);
  assert.deepEqual(right.series.corrfield, [0, 2]);
  assert.equal(JSON.stringify(points), before);
  assert.equal(buildTreCdfs([], "left").count, 0);
  assert.equal(countAtOrBelow([], 2), 0);
});

test("CDF thresholds retain full precision rather than two-decimal labels", () => {
  const values = [0.0049, 0.0051, 1.371, 1.374];
  assert.equal(countAtOrBelow(values, 0.005), 1);
  assert.equal(countAtOrBelow(values, 1.371), 3);
  assert.equal(countAtOrBelow(values, 1.374), 4);
  assert.equal((1.371).toFixed(2), (1.374).toFixed(2));
});
