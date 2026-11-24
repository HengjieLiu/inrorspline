import * as THREE from "three";
import { DEFAULT_ORDER, METHOD_NAMES, validMethodOrder } from "../model.js";
import { checkedSpacing } from "./model.js";
import {
  screenVertex,
  displayFragment,
  buildGridCache,
  disposeCache,
} from "./cache.js";
const $ = (id) => document.getElementById(id),
  frame = () => new Promise((r) => requestAnimationFrame(r));
const renderer = new THREE.WebGLRenderer({
  canvas: $("renderer"),
  antialias: false,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
renderer.autoClear = false;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.debug.onShaderError = () => {
  throw new Error("Grid shader compilation failed on this graphics driver");
};
const gl = renderer.getContext(),
  plane = new THREE.PlaneGeometry(2, 2),
  camera = new THREE.Camera();
const dummy = new THREE.Data3DTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, 1);
dummy.format = THREE.RGBAFormat;
dummy.needsUpdate = true;
const dummyGrid = new THREE.DataArrayTexture(new Uint8Array([0]), 1, 1, 1);
dummyGrid.format = THREE.RedFormat;
dummyGrid.needsUpdate = true;
let saved;
try {
  saved = JSON.parse(localStorage.getItem("grid-compare-order"));
} catch {}
const state = {
  ready: false,
  busy: false,
  tab: "images",
  manifest: null,
  textures: {},
  views: [],
  cache: null,
  pending: null,
  cacheToken: 0,
  loadToken: 0,
  controller: null,
  catalog: [],
  slice: 0,
  order: validMethodOrder(saved) ? saved : DEFAULT_ORDER.slice(),
  renderPending: false,
  requestedAt: 0,
  drawTimes: [],
  latencies: [],
  cacheHistory: [],
  displayBytes: 0,
};
const settings = {
  level: -600,
  width: 1500,
  diffRange: 1000,
  spacing: 8,
  anatomy: true,
  target: true,
  theme: document.documentElement.dataset.theme === "light" ? "light" : "dark",
};
function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : state.ready ? "ready" : "";
}
function progress(text, value) {
  $("loading").hidden = false;
  $("load-stage").textContent = text;
  $("progress").value = value;
}
function busy(value) {
  state.busy = value;
  state.ready = !value && !!state.cache;
  $("slice").disabled = value || !state.manifest;
}
function requestRender() {
  if (!state.renderPending) {
    state.renderPending = true;
    state.requestedAt = performance.now();
    requestAnimationFrame(render);
  }
}
function textureFor(buffer, a) {
  if (buffer.byteLength !== a.bytes)
    throw new Error(`Incomplete asset ${a.file}`);
  const data = a.dtype.includes("u1")
    ? new Uint8Array(buffer)
    : a.dtype.includes("f2")
      ? new Uint16Array(buffer)
      : new Float32Array(buffer);
  const [z, y, x] = a.shape;
  if (Math.max(x, y, z) > gl.getParameter(gl.MAX_3D_TEXTURE_SIZE))
    throw new Error("Case exceeds GPU volume limit");
  const t = new THREE.Data3DTexture(data, x, y, z);
  t.format = a.channels === 4 ? THREE.RGBAFormat : THREE.RedFormat;
  t.type =
    data instanceof Uint8Array
      ? THREE.UnsignedByteType
      : data instanceof Uint16Array
        ? THREE.HalfFloatType
        : THREE.FloatType;
  t.minFilter = t.magFilter =
    data instanceof Uint8Array ? THREE.NearestFilter : THREE.LinearFilter;
  t.unpackAlignment = 1;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  renderer.initTexture(t);
  if (gl.getError() !== gl.NO_ERROR) {
    t.dispose();
    throw new Error("Volume upload failed; free graphics memory and reload");
  }
  return t;
}
function disposeCase() {
  state.ready = false;
  state.cacheToken++;
  disposeCache(state.cache);
  state.cache = null;
  for (const v of state.views) v.material.dispose();
  state.views = [];
  for (const t of Object.values(state.textures)) {
    t.dispose();
    t.image.data = null;
  }
  state.textures = {};
  state.manifest = null;
  state.displayBytes = 0;
  renderer.renderLists.dispose();
  $("comparison").replaceChildren();
  $("slice").disabled = true;
  requestRender();
}
async function loadCase(id) {
  const token = ++state.loadToken;
  state.controller?.abort();
  state.controller = new AbortController();
  const signal = state.controller.signal;
  disposeCase();
  // Unapplied text edits must not silently become another case's spacing.
  $("grid-spacing").value = settings.spacing;
  busy(true);
  status("Loading…");
  const owned = {};
  try {
    const entry = state.catalog.find((x) => x.id === id);
    if (!entry) throw new Error("Case is not prepared");
    $("dataset").value = entry.dataset;
    caseOptions(entry.case);
    progress("Reading case manifest", 0);
    const r = await fetch(entry.url, { signal });
    if (!r.ok) throw new Error(`Manifest HTTP ${r.status}`);
    const m = await r.json();
    if (m.schema !== "dirlab_viewer_v1" || m.landmarks.length !== 300)
      throw new Error("Unexpected case manifest");
    const assets = Object.entries(m.assets);
    let count = 0,
      bytes = 0;
    for (const [key, a] of assets) {
      progress(
        `Loading ${++count}/${assets.length} · ${key.replaceAll("_", " ")}`,
        (count / assets.length) * 0.72,
      );
      const res = await fetch(`/assets/cases/${id}/${a.file}?v=${a.sha256}`, {
        signal,
      });
      if (!res.ok) throw new Error(`Asset HTTP ${res.status}`);
      const b = await res.arrayBuffer();
      if (token !== state.loadToken || signal.aborted)
        throw new DOMException("Canceled", "AbortError");
      owned[key] = textureFor(b, a);
      bytes += b.byteLength;
      await frame();
    }
    if (token !== state.loadToken)
      throw new DOMException("Canceled", "AbortError");
    state.manifest = m;
    state.textures = owned;
    state.displayBytes = bytes;
    state.slice = Math.floor(m.geometry.shape_zyx[1] / 2);
    $("slice").max = m.geometry.shape_zyx[1] - 1;
    $("slice").value = state.slice;
    settings.diffRange = Math.min(1000, Math.max(100, m.difference_max_hu));
    $("diff-range").value = settings.diffRange;
    $("case-note").textContent =
      `${m.dataset.toUpperCase()} · Case ${String(m.case).padStart(2, "0")} · ${m.phases.moving} → ${m.phases.fixed} · ${m.geometry.shape_zyx.join(" × ")} display grid${m.methods.some((x) => x.variant.includes("forest")) ? " · CorrField: component-wise graph guard" : ""}. Common raw-HU display; original method protocols differ.`;
    buildViews();
    updateLabels();
    await rebuild(settings.spacing);
  } catch (e) {
    for (const t of Object.values(owned)) {
      t.dispose();
      t.image.data = null;
    }
    if (token !== state.loadToken) return;
    disposeCase();
    busy(false);
    if (e.name === "AbortError") return;
    progress(`Could not load: ${e.message}`, 0);
    status("Load failed", true);
    console.error(e);
  }
}
function rebuild(spacing) {
  spacing = checkedSpacing(spacing);
  $("grid-spacing").value = spacing;
  const token = ++state.cacheToken,
    loadToken = state.loadToken,
    m = state.manifest;
  if (!m) return Promise.resolve();
  const previous = state.pending;
  busy(true);
  status("Calculating grids…");
  progress(`Building all coronal grids · ${spacing} voxels`, 0.74);
  const current = () =>
    token === state.cacheToken && loadToken === state.loadToken;
  const task = (async () => {
    let next;
    try {
      // Wait for disposal of a canceled pending cache before allocating another.
      if (previous) await previous.catch(() => {});
      if (!current()) return;
      next = await buildGridCache(
        renderer,
        Object.fromEntries(
          DEFAULT_ORDER.map((k) => [k, state.textures[`${k}_maps`]]),
        ),
        m.geometry,
        spacing,
        {
          isCurrent: current,
          onProgress: (p) => {
            if (current())
              progress(
                `Caching ${spacing}-voxel grids · ${Math.round(p * 100)}%`,
                0.74 + 0.24 * p,
              );
          },
        },
      );
      if (!current()) {
        disposeCache(next);
        return;
      }
      const old = state.cache;
      state.cache = next;
      settings.spacing = spacing;
      disposeCache(old);
      // Compile every populated cell before Ready, including offscreen grid rows.
      renderer.setRenderTarget(null);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, 1, 1);
      for (const v of state.views) {
        configure(v, 1, 1);
        renderer.render(v.scene, camera);
      }
      gl.finish();
      if (gl.isContextLost()) throw new Error("Graphics context lost");
      state.cacheHistory.push({
        case: m.id,
        spacing,
        seconds: next.seconds,
        bytes: next.bytes,
        textures: renderer.info.memory.textures,
      });
      state.cacheHistory = state.cacheHistory.slice(-32);
      busy(false);
      $("loading").hidden = true;
      status("Ready");
      updateLabels();
      requestRender();
    } catch (e) {
      if (!current() || e.name === "AbortError") return;
      disposeCache(next);
      state.ready = false;
      $("slice").disabled = true;
      progress(
        `Grid calculation failed: ${e.message}. Apply grids to retry.`,
        0,
      );
      status("Grid calculation failed", true);
      console.error(e);
    }
  })();
  state.pending = task;
  return task.finally(() => {
    if (state.pending === task) state.pending = null;
  });
}
const titles = [
  ["CT images", ""],
  ["Difference from fixed", ""],
  ["Analytic pull grid", "Pull grid · fixed space"],
  ["Raster pull grid", "Pull grid · fixed space"],
  ["Projected mapped mesh", "Mapped fixed mesh · moving-space projection"],
];
function buildViews() {
  for (const v of state.views) v.material.dispose();
  state.views = [];
  const parent = $("comparison");
  parent.replaceChildren();
  parent.append(document.createElement("div"));
  const columns = ["moving", ...state.order, "fixed"];
  for (const key of columns) {
    const h = document.createElement("div");
    h.className = "column-head";
    h.textContent =
      METHOD_NAMES[key] || (key === "moving" ? "Moving" : "Fixed");
    if (
      state.manifest.methods
        .find((x) => x.id === key)
        ?.variant.includes("forest")
    ) {
      const s = document.createElement("small");
      s.textContent = "Graph guard";
      h.append(s);
    }
    parent.append(h);
  }
  titles.forEach(([title, subtitle], mode) => {
    const head = document.createElement("div");
    head.className = "row-head";
    head.textContent = title;
    const small = document.createElement("small");
    small.textContent = subtitle;
    head.append(small);
    parent.append(head);
    for (const key of columns) {
      const e = document.createElement("div");
      e.className = "view";
      e.dataset.method = key;
      e.dataset.mode = mode;
      parent.append(e);
      if (mode >= 2 && ["moving", "fixed"].includes(key)) {
        e.classList.add("blank");
        e.setAttribute("aria-hidden", "true");
        continue;
      }
      e.setAttribute("aria-label", `${METHOD_NAMES[key] || key} ${title}`);
      for (const [position, label] of [
        ["top", "S"],
        ["bottom", "I"],
        ["left", "R"],
        ["right", "L"],
      ]) {
        const s = document.createElement("span");
        s.className = `orientation ${position}`;
        s.textContent = label;
        e.append(s);
      }
      e.addEventListener(
        "wheel",
        (event) => {
          event.preventDefault();
          if (state.ready) setSlice(state.slice + (event.deltaY > 0 ? 1 : -1));
        },
        { passive: false },
      );
      const uniforms = {
        imageTex: { value: dummy },
        otherTex: { value: dummy },
        maskTex: { value: dummy },
        fixedMaskTex: { value: dummy },
        mapTex: { value: dummy },
        gridTex: { value: dummyGrid },
        origin: { value: new THREE.Vector3() },
        stepMM: { value: new THREE.Vector3() },
        shape: { value: new THREE.Vector3() },
        center: { value: new THREE.Vector3() },
        span: { value: new THREE.Vector2() },
        pixels: { value: new THREE.Vector2() },
        mode: { value: mode },
        layer: { value: 0 },
        level: { value: 0 },
        width: { value: 1 },
        diffRange: { value: 1000 },
        hasMap: { value: false },
        anatomyOn: { value: false },
        targetOn: { value: false },
        lightTheme: { value: false },
      };
      const material = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: screenVertex,
        fragmentShader: displayFragment,
        uniforms,
        depthTest: false,
        depthWrite: false,
      });
      const scene = new THREE.Scene();
      scene.add(new THREE.Mesh(plane, material));
      state.views.push({ element: e, key, mode, material, scene });
    }
  });
  renderer.renderLists.dispose();
  requestRender();
}
function configure(v, w, h) {
  const g = state.manifest.geometry,
    u = v.material.uniforms,
    key = v.key,
    hasMap = DEFAULT_ORDER.includes(key);
  u.imageTex.value = state.textures[`${key}_image`];
  u.otherTex.value = state.textures.fixed_image;
  u.maskTex.value = state.textures[`${key}_mask`];
  u.fixedMaskTex.value = state.textures.fixed_mask;
  u.mapTex.value = hasMap ? state.textures[`${key}_maps`] : dummy;
  u.gridTex.value =
    v.mode >= 2
      ? state.cache.targets[`${key}_${v.mode - 2}`].texture
      : dummyGrid;
  u.hasMap.value = hasMap;
  u.origin.value.fromArray(g.origin_lps_mm);
  u.stepMM.value.fromArray(g.step_lps_mm);
  u.shape.value.fromArray(g.shape_zyx.slice().reverse());
  u.center.value.set(
    (g.min_lps_mm[0] + g.max_lps_mm[0]) / 2,
    g.origin_lps_mm[1] + state.slice * g.step_lps_mm[1],
    (g.min_lps_mm[2] + g.max_lps_mm[2]) / 2,
  );
  u.span.value.set(
    g.max_lps_mm[0] - g.min_lps_mm[0],
    g.max_lps_mm[2] - g.min_lps_mm[2],
  );
  u.pixels.value.set(w, h);
  u.layer.value = state.slice;
  u.level.value = settings.level;
  u.width.value = Math.max(1, settings.width);
  u.diffRange.value = Math.max(1, settings.diffRange);
  u.anatomyOn.value = v.mode !== 4 && settings.anatomy;
  u.targetOn.value = v.mode !== 4 && settings.target;
  u.lightTheme.value = settings.theme === "light";
}
function render() {
  state.renderPending = false;
  const start = performance.now();
  renderer.setRenderTarget(null);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setClearColor(0, 0);
  renderer.clear();
  if (
    state.manifest &&
    state.cache &&
    !state.cache.disposed &&
    state.tab === "images"
  ) {
    renderer.setScissorTest(true);
    renderer.setClearColor(0x030507, 1);
    for (const v of state.views) {
      const r = v.element.getBoundingClientRect();
      if (
        r.bottom <= 0 ||
        r.top >= innerHeight ||
        r.right <= 0 ||
        r.left >= innerWidth ||
        !r.width ||
        !r.height
      )
        continue;
      renderer.setViewport(r.left, innerHeight - r.bottom, r.width, r.height);
      renderer.setScissor(
        Math.max(0, r.left),
        Math.max(0, innerHeight - r.bottom),
        Math.min(r.right, innerWidth) - Math.max(0, r.left),
        Math.min(r.bottom, innerHeight) - Math.max(0, r.top),
      );
      configure(v, r.width, r.height);
      renderer.clear();
      renderer.render(v.scene, camera);
    }
  }
  if (state.ready) {
    state.drawTimes.push(performance.now() - start);
    state.latencies.push(performance.now() - state.requestedAt);
    state.drawTimes = state.drawTimes.slice(-600);
    state.latencies = state.latencies.slice(-600);
  }
}
function setSlice(n) {
  if (!state.ready) return;
  state.slice = Math.max(
    0,
    Math.min(state.manifest.geometry.shape_zyx[1] - 1, Math.round(n)),
  );
  $("slice").value = state.slice;
  updateLabels();
  requestRender();
}
function updateLabels() {
  if (!state.manifest) return;
  const g = state.manifest.geometry;
  $("slice-value").textContent =
    `${state.slice + 1}/${g.shape_zyx[1]} · P ${(g.origin_lps_mm[1] + state.slice * g.step_lps_mm[1]).toFixed(1)} mm`;
  $("spacing-mm").textContent =
    `Applied ${settings.spacing} voxels = ${(settings.spacing * g.step_lps_mm[0]).toFixed(3)} mm LR × ${(settings.spacing * Math.abs(g.step_lps_mm[2])).toFixed(3)} mm SI`;
  $("diff-legend").textContent =
    `Difference: −${settings.diffRange} to +${settings.diffRange} HU`;
  $("cache-note").textContent = state.cache
    ? `Nine cached grids · ${(state.cache.bytes / 2 ** 20).toFixed(1)} MiB · ${state.cache.seconds.toFixed(2)} s build`
    : "";
  $("memory").textContent =
    `${(state.displayBytes / 2 ** 20).toFixed(0)} MiB display arrays · ${(state.cache?.bytes / 2 ** 20 || 0).toFixed(0)} MiB grid cache · LPS`;
}
function methodControls() {
  $("methods").replaceChildren();
  state.order.forEach((key, i) => {
    const c = document.createElement("span");
    c.className = "method-chip";
    c.append(METHOD_NAMES[key]);
    for (const [d, name] of [
      [-1, "Earlier"],
      [1, "Later"],
    ]) {
      const b = document.createElement("button");
      b.textContent = name;
      b.setAttribute(
        "aria-label",
        `${METHOD_NAMES[key]} ${name.toLowerCase()}`,
      );
      b.disabled = i + d < 0 || i + d > 2;
      b.onclick = () => {
        [state.order[i], state.order[i + d]] = [
          state.order[i + d],
          state.order[i],
        ];
        localStorage.setItem("grid-compare-order", JSON.stringify(state.order));
        methodControls();
        if (state.manifest) buildViews();
      };
      c.append(b);
    }
    $("methods").append(c);
  });
}
function caseOptions(selected) {
  $("case").replaceChildren();
  for (const c of state.catalog.filter(
    (c) => c.dataset === $("dataset").value,
  )) {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = `Case ${String(c.case).padStart(2, "0")}`;
    o.selected = c.case === selected;
    $("case").append(o);
  }
}
function setTab(tab) {
  state.tab = tab;
  $("images-panel").hidden = tab !== "images";
  $("landmarks-panel").hidden = tab !== "landmarks";
  $("tab-images").classList.toggle("active", tab === "images");
  $("tab-landmarks").classList.toggle("active", tab === "landmarks");
  requestRender();
}
function theme(value) {
  settings.theme = value;
  document.documentElement.dataset.theme = value;
  $("theme-toggle").setAttribute("aria-pressed", String(value === "light"));
  localStorage.setItem("grid-compare-theme", value);
  requestRender();
}
$("dataset").onchange = () => {
  caseOptions(1);
  loadCase($("case").value);
};
$("case").onchange = () => loadCase($("case").value);
$("reload-case").onclick = () => loadCase($("case").value);
$("slice").oninput = (e) => setSlice(Number(e.target.value));
$("apply-grids").onclick = () => {
  try {
    rebuild(checkedSpacing($("grid-spacing").value));
  } catch (e) {
    status(e.message, true);
  }
};
$("grid-spacing").onkeydown = (e) => {
  if (e.key === "Enter") $("apply-grids").click();
};
for (const [id, key] of [
  ["level", "level"],
  ["width", "width"],
  ["diff-range", "diffRange"],
])
  $(id).oninput = (e) => {
    if (e.target.value.trim() && Number.isFinite(Number(e.target.value))) {
      settings[key] =
        key === "level"
          ? Number(e.target.value)
          : Math.max(1, Number(e.target.value));
      updateLabels();
      requestRender();
    }
  };
for (const key of ["anatomy", "target"])
  $(key).onchange = (e) => {
    settings[key] = e.target.checked;
    requestRender();
  };
$("tab-images").onclick = () => setTab("images");
$("tab-landmarks").onclick = () => setTab("landmarks");
$("theme-toggle").onclick = () =>
  theme(settings.theme === "dark" ? "light" : "dark");
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  requestRender();
}
window.addEventListener("resize", resize);
window.addEventListener("scroll", requestRender, { passive: true });
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    setSlice(state.slice + (e.key === "ArrowRight" ? 1 : -1));
    e.preventDefault();
  }
});
$("renderer").addEventListener("webglcontextlost", (e) => {
  e.preventDefault();
  state.cacheToken++;
  state.ready = false;
  state.busy = false;
  $("slice").disabled = true;
  progress("Graphics context lost. Reload the case to rebuild grids.", 0);
  status("Graphics context lost", true);
});
window.addEventListener("error", (e) => {
  status("Viewer error", true);
  console.error(e.error);
});
window.gridViewer = {
  loadCase,
  setSlice,
  setTab,
  rebuild,
  requestRender,
  get ready() {
    return state.ready;
  },
  get manifest() {
    return state.manifest;
  },
  get settings() {
    return { ...settings };
  },
  get diagnostics() {
    return {
      ready: state.ready,
      busy: state.busy,
      case: state.manifest?.id,
      slice: state.slice,
      tab: state.tab,
      order: state.order.slice(),
      displayBytes: state.displayBytes,
      cacheBytes: state.cache?.bytes || 0,
      spacing: state.cache?.spacing,
      cacheVolumes: Object.keys(state.cache?.targets || {}).length,
      gpuTextures: renderer.info.memory.textures,
      geometries: renderer.info.memory.geometries,
      cacheHistory: state.cacheHistory.slice(),
      drawTimes: state.drawTimes.slice(),
      latencies: state.latencies.slice(),
      renderer: gl.getExtension("WEBGL_debug_renderer_info")
        ? gl.getParameter(
            gl.getExtension("WEBGL_debug_renderer_info")
              .UNMASKED_RENDERER_WEBGL,
          )
        : gl.getParameter(gl.RENDERER),
    };
  },
  readCache(method, kind, layer) {
    const t = state.cache.targets[`${method}_${kind}`];
    renderer.setRenderTarget(t, layer);
    const bytes = new Uint8Array(t.width * t.height * 4);
    gl.readPixels(0, 0, t.width, t.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    const error = gl.getError();
    renderer.setRenderTarget(null);
    if (error) throw new Error(`Readback ${error}`);
    return Array.from(bytes.filter((_, i) => i % 4 === 0));
  },
};
theme(settings.theme);
resize();
methodControls();
try {
  const r = await fetch("/api/catalog");
  if (!r.ok) throw new Error("Catalog unavailable");
  state.catalog = (await r.json()).cases;
  caseOptions(1);
  await loadCase($("case").value);
} catch (e) {
  status(e.message, true);
  console.error(e);
}
