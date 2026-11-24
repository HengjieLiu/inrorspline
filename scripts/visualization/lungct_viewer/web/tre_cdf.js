import { DEFAULT_ORDER, METHOD_NAMES } from "./model.js";

export const TRE_KEYS = ["initial", ...DEFAULT_ORDER];
const name = (key) => key === "initial" ? "Initial" : METHOD_NAMES[key];

// Same empirical distribution as the historical TRE notebooks, in percent.
// Never use rounded labels or inverse visualization coordinates as inputs.
export function buildTreCdfs(points, side = "both") {
  const selected = points.filter((p) => side === "both" || p.side === side);
  const series = Object.fromEntries(TRE_KEYS.map((key) => {
    const values = selected.map((p) => p.tre_mm[key]);
    if (values.some((v) => !Number.isFinite(v) || v < 0))
      throw new Error(`Invalid audited TRE for ${name(key)}`);
    return [key, values.sort((a, b) => a - b)];
  }));
  const maximum = Math.max(0, ...Object.values(series).flat());
  return { series, count: selected.length, fullMax: Math.max(1, Math.ceil(maximum * 1.03)) };
}

// Upper bound includes every tie, including multiple zero-error landmarks.
export function countAtOrBelow(sorted, tolerance) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= tolerance) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function cdfSteps(sorted) {
  const steps = [];
  for (let i = 0; i < sorted.length; i++) {
    if (i + 1 === sorted.length || sorted[i + 1] !== sorted[i])
      steps.push({ tolerance: sorted[i], count: i + 1, percent: 100 * (i + 1) / sorted.length });
  }
  return steps;
}

const NS = "http://www.w3.org/2000/svg";
const BOX = { left: 68, right: 620, top: 14, bottom: 198 };
function svg(tag, attrs = {}, content) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (content !== undefined) node.textContent = content;
  return node;
}

export class TreCdfChart {
  constructor(root) {
    this.root = root;
    this.plot = root.querySelector("svg");
    this.legend = root.querySelector(".cdf-legend");
    this.readout = root.querySelector("output");
    this.maximum = root.querySelector("input");
    this.fullButton = root.querySelector(".cdf-full");
    this.zoomButton = root.querySelector(".cdf-zoom");
    this.cache = null;
    this.active = null;
    this.order = [];
    this.pathBuilds = 0;
    const defs = svg("defs");
    const clip = svg("clipPath", { id: "tre-cdf-clip" });
    clip.append(svg("rect", { x: BOX.left, y: BOX.top, width: BOX.right - BOX.left, height: BOX.bottom - BOX.top }));
    defs.append(clip);
    this.axes = svg("g", { class: "cdf-axes" });
    this.curves = svg("g", { "clip-path": "url(#tre-cdf-clip)" });
    this.paths = Object.fromEntries(TRE_KEYS.map((key) => {
      const p = svg("path", { class: "cdf-curve", "data-method": key, fill: "none" });
      this.curves.append(p);
      return [key, p];
    }));
    this.cursor = svg("g", { class: "cdf-cursor", visibility: "hidden" });
    this.cursorLine = svg("line", { y1: BOX.top, y2: BOX.bottom });
    this.cursorPoint = svg("circle", { r: 5 });
    this.cursorArrow = svg("path", { d: `M ${BOX.right - 9} ${BOX.top + 7} l 8 6 l -8 6`, fill: "none" });
    this.cursor.append(this.cursorLine, this.cursorPoint, this.cursorArrow);
    this.plot.append(defs, this.axes, this.curves, this.cursor);
    this.fullButton.onclick = () => this.setRange("full");
    this.zoomButton.onclick = () => this.setRange("fixed", 6);
    this.maximum.onchange = () => {
      const value = Number(this.maximum.value);
      if (Number.isFinite(value) && value > 0) this.setRange("fixed", value);
      else this.maximum.value = this.xMax;
    };
    this.clearCase();
  }

  clearCase() {
    this.cache = this.active = this.point = null;
    this.side = this.ranking = null;
    this.rangeMode = "full";
    this.order = [];
    this.pathBuilds = 0;
    this.sweep = null;
    this.xMax = 1;
    this.axes.replaceChildren();
    this.legend.replaceChildren();
    for (const p of Object.values(this.paths)) p.removeAttribute("d");
    this.cursor.setAttribute("visibility", "hidden");
    this.cursor.setAttribute("display", "none");
    this.readout.textContent = "Load a case to see its TRE distribution.";
    this.readout.classList.remove("outside");
    for (const control of [this.fullButton, this.zoomButton, this.maximum]) control.disabled = true;
  }

  setCase(points) {
    this.clearCase();
    this.cache = Object.fromEntries(["both", "left", "right"].map((side) => [side, buildTreCdfs(points, side)]));
    for (const control of [this.fullButton, this.zoomButton, this.maximum]) control.disabled = false;
  }

  setSelection({ side, ranking, point, order }) {
    if (!this.cache) return;
    const changedSide = side !== this.side;
    this.side = side;
    this.active = this.cache[side];
    this.point = point;
    this.ranking = TRE_KEYS.includes(ranking) ? ranking : null;
    if (this.order.join() !== order.join()) {
      this.order = order.slice();
      this.legend.replaceChildren(...["initial", ...order].map((key) => {
        const item = document.createElement("span");
        item.dataset.method = key;
        item.append(Object.assign(document.createElement("i"), { ariaHidden: "true" }), name(key));
        return item;
      }));
    }
    if (changedSide) this.drawPaths();
    this.updateCursor();
  }

  setRange(mode, maximum) {
    if (!this.active) return;
    this.rangeMode = mode;
    if (mode === "fixed") this.xMax = maximum;
    this.drawPaths();
    this.updateCursor();
  }

  x(value) { return BOX.left + value / this.xMax * (BOX.right - BOX.left); }
  y(percent) { return BOX.bottom - percent / 100 * (BOX.bottom - BOX.top); }

  drawPaths() {
    if (this.rangeMode === "full") this.xMax = this.active.fullMax;
    this.maximum.value = this.xMax;
    this.fullButton.setAttribute("aria-pressed", String(this.rangeMode === "full"));
    this.zoomButton.setAttribute("aria-pressed", String(this.rangeMode === "fixed" && this.xMax === 6));
    const nodes = [];
    for (const value of [0, 25, 50, 75, 100]) {
      const y = this.y(value);
      nodes.push(svg("line", { x1: BOX.left, x2: BOX.right, y1: y, y2: y, class: "cdf-grid" }));
      nodes.push(svg("text", { x: BOX.left - 9, y: y + 5, "text-anchor": "end" }, `${value}`));
    }
    for (let i = 0; i <= 5; i++) {
      const value = this.xMax * i / 5, x = this.x(value);
      nodes.push(svg("line", { x1: x, x2: x, y1: BOX.top, y2: BOX.bottom, class: "cdf-grid" }));
      nodes.push(svg("text", { x, y: BOX.bottom + 21, "text-anchor": "middle" }, Number(value.toPrecision(3)).toString()));
    }
    nodes.push(svg("path", { d: `M ${BOX.left} ${BOX.top} V ${BOX.bottom} H ${BOX.right}`, class: "cdf-axis-line", fill: "none" }));
    nodes.push(svg("text", { x: (BOX.left + BOX.right) / 2, y: 244, "text-anchor": "middle" }, "Distance tolerance (mm)"));
    nodes.push(svg("text", { transform: "translate(19 106) rotate(-90)", "text-anchor": "middle" }, "Landmarks ≤ tolerance (%)"));
    this.axes.replaceChildren(...nodes);
    for (const key of TRE_KEYS) {
      let path = `M ${this.x(0)} ${this.y(0)}`;
      for (const step of cdfSteps(this.active.series[key])) {
        if (step.tolerance > this.xMax) break;
        path += ` H ${this.x(step.tolerance)} V ${this.y(step.percent)}`;
      }
      this.paths[key].setAttribute("d", `${path} H ${BOX.right}`);
    }
    this.pathBuilds++;
  }

  updateCursor() {
    this.root.dataset.side = this.side;
    this.root.dataset.count = this.active.count;
    this.root.dataset.ranking = this.ranking || "none";
    for (const [key, path] of Object.entries(this.paths)) path.classList.toggle("ranked", key === this.ranking);
    for (const item of this.legend.children) item.classList.toggle("ranked", item.dataset.method === this.ranking);
    if (!this.ranking || !this.point) {
      this.cursor.setAttribute("visibility", "hidden");
      this.cursor.setAttribute("display", "none");
      this.readout.classList.remove("outside");
      this.sweep = null;
      this.readout.textContent = `${this.active.count} landmarks · Choose a TRE ordering to show the sweep marker.`;
      return;
    }
    const tolerance = this.point.tre_mm[this.ranking];
    const count = countAtOrBelow(this.active.series[this.ranking], tolerance);
    const percent = this.active.count ? 100 * count / this.active.count : 0;
    const outside = tolerance > this.xMax;
    const x = this.x(Math.min(tolerance, this.xMax));
    this.cursor.setAttribute("visibility", "visible");
    this.cursor.setAttribute("display", "inline");
    this.cursor.dataset.method = this.ranking;
    this.cursorLine.setAttribute("x1", x);
    this.cursorLine.setAttribute("x2", x);
    this.cursorPoint.setAttribute("cx", x);
    this.cursorPoint.setAttribute("cy", this.y(percent));
    this.cursorPoint.setAttribute("visibility", outside ? "hidden" : "visible");
    this.cursorArrow.setAttribute("visibility", outside ? "visible" : "hidden");
    this.readout.classList.toggle("outside", outside);
    this.readout.textContent = `${name(this.ranking)} · ${tolerance.toFixed(2)} mm · ≤ tolerance: ${count}/${this.active.count} (${percent.toFixed(2)}%)${outside ? " · Outside displayed range →" : ""}`;
    this.sweep = { method: this.ranking, landmarkId: this.point.id, tolerance, count, percent, outside };
  }

  get diagnostics() {
    return { count: this.active?.count || 0, side: this.side, ranking: this.ranking, rangeMode: this.rangeMode, xMax: this.xMax, pathBuilds: this.pathBuilds, sweep: this.sweep ? { ...this.sweep } : null };
  }
}
