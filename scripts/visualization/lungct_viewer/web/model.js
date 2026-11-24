export const METHOD_NAMES = {
  mrdbscp: "MR-D-BSCP",
  ptvreg: "pTVReg",
  corrfield: "CorrField",
};
export const DEFAULT_ORDER = ["mrdbscp", "ptvreg", "corrfield"];
export function orderedLandmarks(points, side, order, direction, seed = 1) {
  const p = points
    .filter((p) => side === "both" || p.side === side)
    .map((p) => p.id);
  if (order === "random") {
    let s = seed >>> 0;
    const random = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    return p;
  }
  const val = (id) =>
    order === "superior"
      ? points[id].fixed_lps_mm[2]
      : points[id].tre_mm[order];
  p.sort(
    (a, b) =>
      (direction === "descending" ? -1 : 1) * (val(a) - val(b)) || a - b,
  );
  return p;
}
export function planeGeometry(geometry, axis, point, extent = 0) {
  const lo = geometry.min_lps_mm,
    hi = geometry.max_lps_mm;
  const axes = axis === 0 ? [0, 2] : axis === 1 ? [1, 2] : [0, 1];
  const center = point.slice();
  if (!extent) for (const a of axes) center[a] = (lo[a] + hi[a]) / 2;
  return { center, span: axes.map((a) => extent || hi[a] - lo[a]), axes };
}
export function validMethodOrder(a) {
  return (
    Array.isArray(a) &&
    a.length === 3 &&
    new Set(a).size === 3 &&
    a.every((k) => DEFAULT_ORDER.includes(k))
  );
}
