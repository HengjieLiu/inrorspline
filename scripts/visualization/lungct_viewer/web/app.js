import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  DEFAULT_ORDER,
  METHOD_NAMES,
  orderedLandmarks,
  planeGeometry,
  validMethodOrder,
} from "./model.js";
import { vertex, fragment } from "./shaders.js";
import { TreCdfChart } from "./tre_cdf.js";

const $ = (id) => document.getElementById(id);
const treCdf = new TreCdfChart($("tre-cdf"));
const renderer = new THREE.WebGLRenderer({
  canvas: $("renderer"),
  antialias: false,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.autoClear = false;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.debug.onShaderError = () => {
  throw new Error("This graphics driver could not compile the viewer shader");
};
const gl = renderer.getContext();
const plane = new THREE.PlaneGeometry(2, 2),
  screenCamera = new THREE.Camera();
const dummy = new THREE.Data3DTexture(new Float32Array([0, 0, 0, 1]), 1, 1, 1);
dummy.format = THREE.RGBAFormat;
dummy.type = THREE.FloatType;
dummy.needsUpdate = true;
let saved;
try {
  saved = JSON.parse(localStorage.getItem("dirlab-method-order"));
} catch {}
const state = {
  ready: false,
  tab: "images",
  manifest: null,
  textures: {},
  views: [],
  treLabels: [],
  clouds: [],
  order: validMethodOrder(saved) ? saved : DEFAULT_ORDER.slice(),
  list: [],
  index: 0,
  seed: 1,
  slice: 0,
  loadToken: 0,
  controller: null,
  requestPending: false,
  requestedAt: 0,
  drawTimes: [],
  latencies: [],
  loadHistory: [],
  catalog: [],
  renderCount: 0,
  displayBytes: 0,
  anchorMaterial: null,
};
const settings = {
  level: -600,
  width: 1500,
  diffRange: 1000,
  mapRange: 50,
  gridSpacing: 10,
  anatomy: true,
  target: true,
  crosshair: true,
  targetPoint: true,
  theme: document.documentElement.dataset.theme === "light" ? "light" : "dark",
  extent: 0,
};

function requestRender() {
  if (!state.requestPending) {
    state.requestPending = true;
    state.requestedAt = performance.now();
    requestAnimationFrame(render);
  }
}
function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : state.ready ? "ready" : "";
}
function progress(stage, value) {
  $("loading").hidden = false;
  $("load-stage").textContent = stage;
  $("progress").value = value;
}
function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}
function columns() {
  return ["moving", ...state.order, "fixed"];
}
function label(method) {
  return METHOD_NAMES[method] || method[0].toUpperCase() + method.slice(1);
}
function normAngleCamera(c) {
  c.camera.up.set(0, 0, 1);
  c.camera.position.set(0, -600, 100);
  c.controls.target.set(0, 0, 0);
  c.camera.zoom = 1;
  c.camera.updateProjectionMatrix();
  c.controls.update();
}

function disposeCase() {
  state.ready = false;
  treCdf.clearCase();
  for (const material of new Set(
    [...state.views.map((v) => v.material), state.anchorMaterial].filter(
      Boolean,
    ),
  ))
    material.dispose();
  state.views = [];
  state.treLabels = [];
  state.anchorMaterial = null;
  for (const c of state.clouds) {
    c.controls.dispose();
    c.points.geometry.dispose();
    c.points.material.dispose();
    c.highlight.geometry.dispose();
    c.highlight.material.dispose();
    c.element.onpointerdown = null;
    c.element.onpointerup = null;
  }
  state.clouds = [];
  for (const texture of Object.values(state.textures)) {
    texture.dispose();
    texture.image.data = null;
  }
  state.textures = {};
  state.manifest = null;
  state.displayBytes = 0;
  renderer.renderLists.dispose();
  $("comparison").replaceChildren();
  $("mpr").replaceChildren();
  requestRender();
}

function textureFor(buffer, asset) {
  if (buffer.byteLength !== asset.bytes)
    throw new Error(`Incomplete asset ${asset.file}`);
  const data =
    asset.dtype.includes("u1") || asset.dtype === "|u1"
      ? new Uint8Array(buffer)
      : asset.dtype.includes("f2")
        ? new Uint16Array(buffer)
        : new Float32Array(buffer);
  const [z, y, x] = asset.shape;
  if (Math.max(x, y, z) > gl.getParameter(gl.MAX_3D_TEXTURE_SIZE))
    throw new Error("Case exceeds this browser’s 3D texture limit");
  const tex = new THREE.Data3DTexture(data, x, y, z);
  tex.format = asset.channels === 4 ? THREE.RGBAFormat : THREE.RedFormat;
  tex.type =
    data instanceof Uint8Array
      ? THREE.UnsignedByteType
      : data instanceof Uint16Array
        ? THREE.HalfFloatType
        : THREE.FloatType;
  tex.minFilter = tex.magFilter =
    data instanceof Uint8Array ? THREE.NearestFilter : THREE.LinearFilter;
  tex.unpackAlignment = 1;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  renderer.initTexture(tex);
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    tex.dispose();
    throw new Error(
      `Graphics upload failed (${error}); free browser graphics memory and reload the case`,
    );
  }
  return tex;
}

async function loadCase(id) {
  const token = ++state.loadToken;
  state.controller?.abort();
  state.controller = new AbortController();
  const signal = state.controller.signal;
  disposeCase();
  setStatus("Loading…");
  const owned = {};
  const started = performance.now();
  try {
    const entry = state.catalog.find((x) => x.id === id);
    if (!entry) throw new Error("Case is not prepared");
    $("dataset").value = entry.dataset;
    updateCaseOptions(entry.case);
    progress(
      `Reading ${entry.dataset.toUpperCase()} case ${String(entry.case).padStart(2, "0")}`,
      0,
    );
    const response = await fetch(entry.url, { signal });
    if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
    const m = await response.json();
    if (m.landmarks.length !== 300 || m.schema !== "dirlab_viewer_v1")
      throw new Error("Unexpected case manifest");
    const pairs = Object.entries(m.assets);
    let count = 0,
      bytes = 0;
    for (const [key, asset] of pairs) {
      progress(
        `Loading ${count + 1}/${pairs.length} · ${key.replaceAll("_", " ")}`,
        0.05 + (0.8 * count) / pairs.length,
      );
      const r = await fetch(
        `/assets/cases/${id}/${asset.file}?v=${asset.sha256}`,
        { signal },
      );
      if (!r.ok) throw new Error(`Asset HTTP ${r.status}`);
      const b = await r.arrayBuffer();
      if (signal.aborted || token !== state.loadToken)
        throw new DOMException("Canceled", "AbortError");
      owned[key] = textureFor(b, asset);
      bytes += b.byteLength;
      count++;
      await nextFrame();
    }
    if (token !== state.loadToken)
      throw new DOMException("Canceled", "AbortError");
    state.manifest = m;
    state.textures = owned;
    state.displayBytes = bytes;
    settings.mapRange = Math.max(0.1, m.scale_max_mm);
    settings.diffRange = Math.min(1000, Math.max(100, m.difference_max_hu));
    $("map-range").value = settings.mapRange;
    $("diff-range").value = settings.diffRange;
    $("slice").max = m.geometry.shape_zyx[1] - 1;
    state.slice = Math.round(Number($("slice").max) / 2);
    $("slice").value = state.slice;
    state.seed = 1;
    treCdf.setCase(m.landmarks);
    reorderLandmarks(false);
    buildMatrices();
    buildClouds();
    state.anchorMaterial = state.views[0].material;
    const guarded = m.methods
      .filter((x) => x.variant.includes("forest"))
      .map((x) => x.name);
    $("case-note").textContent =
      `${m.dataset.toUpperCase()} · Case ${String(m.case).padStart(2, "0")} · ${m.phases.moving} → ${m.phases.fixed} · ${m.geometry.shape_zyx.join(" × ")} grid · R ${m.lung_counts.right} / L ${m.lung_counts.left} landmarks${guarded.length ? " · CorrField: component-wise graph guard" : ""}. Common raw-HU display; method input protocols differ.`;
    progress("Initializing both panels and shaders…", 0.92);
    // Warm every material, including the hidden tab, before exposing Ready.
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, 1, 1);
    for (const v of state.views) {
      configure(v, 1, 1);
      renderer.render(v.scene, screenCamera);
    }
    for (const c of state.clouds) renderer.render(c.scene, c.camera);
    gl.finish();
    await nextFrame();
    if (token !== state.loadToken)
      throw new DOMException("Canceled", "AbortError");
    if (gl.isContextLost())
      throw new Error("Graphics context lost while loading");
    state.ready = true;
    state.loadHistory.push({
      id,
      seconds: (performance.now() - started) / 1000,
      displayBytes: bytes,
      textures: renderer.info.memory.textures,
    });
    $("loading").hidden = true;
    setStatus("Ready");
    updateLabels();
    $("memory").textContent =
      `${(bytes / 1024 ** 2).toFixed(0)} MiB display arrays · ${m.landmarks.length} landmarks · LPS`;
    requestRender();
  } catch (error) {
    for (const t of Object.values(owned)) {
      t.dispose();
      t.image.data = null;
    }
    if (token !== state.loadToken) return;
    disposeCase();
    if (error.name === "AbortError") return;
    progress(`Could not load case: ${error.message}`, 0);
    setStatus("Load failed", true);
    console.error(error);
  }
}

function viewElement(parent, method, mode, axis, tab) {
  const element = document.createElement("div");
  element.className = "view";
  element.dataset.method = method;
  element.dataset.mode = mode;
  element.dataset.axis = axis;
  element.setAttribute(
    "aria-label",
    `${label(method)} ${tab === "images" ? ["CT", "difference", "grid", "magnitude", "signed SI"][mode] : ["coronal", "sagittal", "axial"][axis]}`,
  );
  const labels =
    axis === 0
      ? ["S", "I", "R", "L"]
      : axis === 1
        ? ["S", "I", "A", "P"]
        : ["A", "P", "R", "L"];
  ["top", "bottom", "left", "right"].forEach((position, i) => {
    const a = document.createElement("span");
    a.className = `orientation ${position}`;
    a.textContent = labels[i];
    element.append(a);
  });
  const note = document.createElement("span");
  note.className = "view-note";
  element.append(note);
  parent.append(element);
  const material = new THREE.ShaderMaterial({
    vertexShader: vertex,
    fragmentShader: fragment,
    depthTest: false,
    depthWrite: false,
    uniforms: {
      imageTex: { value: null },
      otherTex: { value: null },
      maskTex: { value: null },
      fixedMaskTex: { value: null },
      mapTex: { value: dummy },
      origin: { value: new THREE.Vector3() },
      stepMM: { value: new THREE.Vector3() },
      shape: { value: new THREE.Vector3() },
      center: { value: new THREE.Vector3() },
      point: { value: new THREE.Vector3() },
      target: { value: new THREE.Vector3() },
      span: { value: new THREE.Vector2() },
      pixels: { value: new THREE.Vector2() },
      axis: { value: axis },
      mode: { value: mode },
      level: { value: settings.level },
      width: { value: settings.width },
      diffRange: { value: settings.diffRange },
      mapRange: { value: settings.mapRange },
      gridSpacing: { value: settings.gridSpacing },
      hasMap: { value: false },
      anatomyOn: { value: true },
      targetOn: { value: true },
      crossOn: { value: false },
      targetMarker: { value: false },
      lightTheme: { value: false },
    },
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(plane, material));
  const view = { element, note, method, mode, axis, tab, material, scene };
  state.views.push(view);
  if (tab === "images")
    element.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        if (state.ready) setSlice(state.slice + (event.deltaY > 0 ? 1 : -1));
      },
      { passive: false },
    );
}

function buildMatrices() {
  for (const v of state.views)
    if (v.material !== state.anchorMaterial) v.material.dispose();
  state.views = [];
  state.treLabels = [];
  const make = (id, tab, rows) => {
    const parent = $(id);
    parent.replaceChildren();
    parent.append(document.createElement("div"));
    for (const method of columns()) {
      const head = document.createElement("div");
      head.className = "column-head";
      head.dataset.method = method;
      head.textContent = label(method);
      if (tab === "landmarks" && method !== "fixed") {
        const value = document.createElement("span");
        value.className = "column-tre";
        const treMethod = method === "moving" ? "initial" : method;
        value.dataset.treMethod = treMethod;
        head.append(value);
        state.treLabels.push({ element: value, method: treMethod });
      }
      const info = state.manifest.methods.find((x) => x.id === method);
      if (info?.variant.includes("forest")) {
        const tag = document.createElement("small");
        tag.textContent = "Graph guard";
        head.append(tag);
      }
      parent.append(head);
    }
    rows.forEach((title, row) => {
      const h = document.createElement("div");
      h.className = "row-head";
      h.textContent = title;
      parent.append(h);
      for (const method of columns()) {
        const blank =
          tab === "images" && row >= 2 && ["moving", "fixed"].includes(method);
        if (blank) {
          const e = document.createElement("div");
          e.className = "view blank";
          e.setAttribute("aria-hidden", "true");
          parent.append(e);
        } else
          viewElement(
            parent,
            method,
            tab === "images" ? row : 0,
            tab === "images" ? 0 : row,
            tab,
          );
      }
    });
  };
  make("comparison", "images", [
    "CT images",
    "Difference from fixed",
    "Deformed grid",
    "Total magnitude · mm",
    "Signed SI · mm",
  ]);
  make("mpr", "landmarks", ["Coronal", "Sagittal", "Axial"]);
}

function currentPoint() {
  return state.manifest?.landmarks[state.list[state.index]];
}
function configure(view, w, h) {
  const m = state.manifest,
    g = m.geometry,
    u = view.material.uniforms,
    method = view.method;
  const hasMap = !["moving", "fixed"].includes(method);
  u.imageTex.value = state.textures[`${method}_image`];
  u.otherTex.value = state.textures.fixed_image;
  u.maskTex.value = state.textures[`${method}_mask`];
  u.fixedMaskTex.value = state.textures.fixed_mask;
  u.mapTex.value = hasMap ? state.textures[`${method}_maps`] : dummy;
  u.hasMap.value = hasMap;
  u.origin.value.fromArray(g.origin_lps_mm);
  u.stepMM.value.fromArray(g.step_lps_mm);
  u.shape.value.fromArray(g.shape_zyx.slice().reverse());
  u.pixels.value.set(w, h);
  u.level.value = settings.level;
  u.width.value = Math.max(1, settings.width);
  u.diffRange.value = Math.max(1, settings.diffRange);
  u.mapRange.value = Math.max(0.1, settings.mapRange);
  u.gridSpacing.value = Math.max(1, settings.gridSpacing);
  u.anatomyOn.value = settings.anatomy;
  u.targetOn.value = settings.target;
  u.lightTheme.value = settings.theme === "light";
  let point,
    extent = 0,
    cross = false,
    targetMarker = false,
    target;
  if (view.tab === "images") {
    point = g.min_lps_mm.map((v, i) => (v + g.max_lps_mm[i]) / 2);
    point[1] = g.origin_lps_mm[1] + state.slice * g.step_lps_mm[1];
    target = point;
    view.note.textContent = "";
  } else {
    const selected = currentPoint();
    target = selected.fixed_lps_mm;
    const inv = hasMap ? selected.methods[method] : null;
    point =
      method === "moving"
        ? selected.moving_lps_mm
        : method === "fixed"
          ? target
          : inv.point_lps_mm || target;
    extent = settings.extent;
    cross = settings.crosshair && (!inv || inv.status !== "unresolved");
    targetMarker = settings.targetPoint;
    const messages = [];
    if (inv && inv.status !== "resolved")
      messages.push(
        inv.status === "unresolved"
          ? "Inverse unresolved · fixed-target center"
          : `Ambiguous inverse · ${inv.roots} roots`,
      );
    const normal = [1, 0, 2][view.axis];
    const delta = target[normal] - point[normal];
    if (targetMarker && Math.abs(delta) > 0.5)
      messages.push(`Target ${Math.abs(delta).toFixed(1)} mm from plane`);
    const axes = view.axis === 0 ? [0, 2] : view.axis === 1 ? [1, 2] : [0, 1];
    if (
      targetMarker &&
      extent &&
      axes.some((a) => Math.abs(target[a] - point[a]) > extent / 2)
    )
      messages.push("Fixed target outside view");
    view.note.textContent = messages.join(" · ");
  }
  const pg = planeGeometry(g, view.axis, point, extent);
  u.center.value.fromArray(pg.center);
  u.point.value.fromArray(point);
  u.target.value.fromArray(target);
  u.span.value.fromArray(pg.span);
  u.crossOn.value = cross;
  u.targetMarker.value = targetMarker;
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  requestRender();
}
function viewport(element) {
  const r = element.getBoundingClientRect();
  if (
    r.bottom <= 0 ||
    r.top >= innerHeight ||
    r.right <= 0 ||
    r.left >= innerWidth ||
    !r.width ||
    !r.height
  )
    return null;
  renderer.setViewport(r.left, innerHeight - r.bottom, r.width, r.height);
  renderer.setScissor(
    Math.max(0, r.left),
    Math.max(0, innerHeight - r.bottom),
    Math.min(r.right, innerWidth) - Math.max(0, r.left),
    Math.min(r.bottom, innerHeight) - Math.max(0, r.top),
  );
  return r;
}
function render() {
  state.requestPending = false;
  const start = performance.now();
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  if (state.ready) {
    renderer.setScissorTest(true);
    renderer.setClearColor(0x030507, 1);
    for (const v of state.views) {
      if (v.tab !== state.tab) continue;
      const r = viewport(v.element);
      if (!r) continue;
      configure(v, r.width, r.height);
      renderer.clear();
      renderer.render(v.scene, screenCamera);
    }
    if (state.tab === "landmarks") {
      renderer.setClearColor(
        settings.theme === "light" ? 0xf1f5f9 : 0x030507,
        1,
      );
      for (const c of state.clouds) {
        const r = viewport(c.element);
        if (!r) continue;
        c.camera.left = (-180 * r.width) / r.height;
        c.camera.right = (180 * r.width) / r.height;
        c.camera.top = 180;
        c.camera.bottom = -180;
        c.camera.updateProjectionMatrix();
        renderer.clear();
        renderer.render(c.scene, c.camera);
      }
    }
    state.latencies.push(performance.now() - state.requestedAt);
    state.drawTimes.push(performance.now() - start);
    if (state.latencies.length > 1000) state.latencies.shift();
    if (state.drawTimes.length > 1000) state.drawTimes.shift();
    state.renderCount++;
  }
}

function buildClouds() {
  const m = state.manifest,
    center = m.geometry.min_lps_mm.map(
      (v, i) => (v + m.geometry.max_lps_mm[i]) / 2,
    );
  let syncing = false;
  for (const role of ["moving", "fixed"]) {
    const element = $(`cloud-${role}`),
      scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(
      -180,
      180,
      180,
      -180,
      0.1,
      3000,
    );
    const controls = new OrbitControls(camera, element);
    controls.enableDamping = false;
    controls.enablePan = true;
    const material = new THREE.PointsMaterial({
      size: 3.5,
      sizeAttenuation: false,
      vertexColors: true,
    });
    const points = new THREE.Points(new THREE.BufferGeometry(), material);
    scene.add(points);
    const highlight = new THREE.Mesh(
      new THREE.SphereGeometry(4.5, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffe65d, depthTest: false }),
    );
    highlight.renderOrder = 1000;
    scene.add(highlight);
    const c = {
      element,
      scene,
      camera,
      controls,
      points,
      highlight,
      role,
      center,
    };
    state.clouds.push(c);
    normAngleCamera(c);
    controls.addEventListener("change", () => {
      if ($("link-cameras").checked && !syncing) {
        syncing = true;
        for (const other of state.clouds)
          if (other !== c) {
            other.camera.position.copy(camera.position);
            other.camera.quaternion.copy(camera.quaternion);
            other.camera.zoom = camera.zoom;
            other.controls.target.copy(controls.target);
            other.camera.updateProjectionMatrix();
            other.controls.update();
          }
        syncing = false;
      }
      requestRender();
    });
    let down;
    element.onpointerdown = (e) => {
      down = [e.clientX, e.clientY];
    };
    element.onpointerup = (e) => {
      if (
        !state.ready ||
        !down ||
        Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5
      )
        return;
      const r = element.getBoundingClientRect(),
        mouse = new THREE.Vector2(
          ((e.clientX - r.left) / r.width) * 2 - 1,
          (-(e.clientY - r.top) / r.height) * 2 + 1,
        );
      const ray = new THREE.Raycaster();
      ray.params.Points.threshold = 3;
      ray.setFromCamera(mouse, camera);
      const hit = ray.intersectObject(points)[0];
      if (hit) {
        const id = points.userData.ids[hit.index],
          index = state.list.indexOf(id);
        if (index >= 0) setIndex(index);
      }
    };
  }
  updateClouds();
}
function updateClouds() {
  if (!state.manifest) return;
  const selected = currentPoint();
  for (const c of state.clouds) {
    const positions = [],
      colors = [];
    for (const id of state.list) {
      const p = state.manifest.landmarks[id],
        world = p[`${c.role}_lps_mm`];
      positions.push(...world.map((v, i) => v - c.center[i]));
      colors.push(
        ...(settings.theme === "light"
          ? p.side === "right"
            ? [0.02, 0.4, 0.5]
            : [0.6, 0.16, 0.42]
          : p.side === "right"
            ? [0.18, 0.78, 0.9]
            : [0.87, 0.49, 0.77]),
      );
    }
    c.points.geometry.dispose();
    c.points.geometry = new THREE.BufferGeometry();
    c.points.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    c.points.geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colors, 3),
    );
    c.points.userData.ids = state.list.slice();
    c.highlight.material.color.setHex(
      settings.theme === "light" ? 0xa66b00 : 0xffe65d,
    );
    c.highlight.position.fromArray(
      selected[`${c.role}_lps_mm`].map((v, i) => v - c.center[i]),
    );
    c.element.querySelector("strong").textContent =
      `${label(c.role)} · ${state.list.length} correspondences · L/P/S frame`;
  }
}
function updateHighlight() {
  const p = currentPoint();
  if (!p) return;
  for (const c of state.clouds)
    c.highlight.position.fromArray(
      p[`${c.role}_lps_mm`].map((v, i) => v - c.center[i]),
    );
}

function reorderLandmarks(update = true) {
  if (!state.manifest) return;
  $("direction").disabled = $("order").value === "random";
  state.list = orderedLandmarks(
    state.manifest.landmarks,
    $("lung").value,
    $("order").value,
    $("direction").value,
    state.seed,
  );
  state.index = 0;
  for (const id of ["landmark-slider", "landmark-slider-bottom"]) {
    $(id).max = state.list.length - 1;
    $(id).value = 0;
  }
  updateTreCdf();
  if (update) {
    updateClouds();
    updateLabels();
    requestRender();
  }
}
function setIndex(index) {
  if (!state.ready) return;
  state.index = (Number(index) + state.list.length) % state.list.length;
  for (const id of ["landmark-slider", "landmark-slider-bottom"])
    $(id).value = state.index;
  updateHighlight();
  updateLabels();
  requestRender();
}
function setSlice(index) {
  if (!state.ready) return;
  state.slice = Math.max(
    0,
    Math.min(Number($("slice").max), Math.round(index)),
  );
  $("slice").value = state.slice;
  updateLabels();
  requestRender();
}
function updateLabels() {
  const p = currentPoint();
  if (!p) return;
  updateTreCdf();
  $("slice-value").textContent =
    `P ${(state.manifest.geometry.origin_lps_mm[1] + state.slice * state.manifest.geometry.step_lps_mm[1]).toFixed(1)} mm`;
  for (const id of ["landmark-label", "landmark-label-bottom"])
    $(id).textContent =
      `${state.index + 1} / ${state.list.length} · ID ${p.id + 1} · ${p.side}`;
  for (const { element, method } of state.treLabels)
    element.textContent = `${method === "initial" ? "Initial TRE" : "TRE"}: ${p.tre_mm[method].toFixed(2)} mm`;
  $("diff-legend").textContent =
    `Difference: −${settings.diffRange} to +${settings.diffRange} HU`;
  $("map-legend").textContent =
    `Magnitude / signed SI: −${settings.mapRange} to +${settings.mapRange} mm`;
}
function updateTreCdf() {
  treCdf.setSelection({
    side: $("lung").value,
    ranking: $("order").value,
    point: currentPoint(),
    order: state.order,
  });
}
function setTab(tab) {
  state.tab = tab;
  $("images-panel").hidden = tab !== "images";
  $("landmarks-panel").hidden = tab !== "landmarks";
  $("tab-images").classList.toggle("active", tab === "images");
  $("tab-landmarks").classList.toggle("active", tab === "landmarks");
  requestRender();
}
function setTheme(theme, persist = true) {
  settings.theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = settings.theme;
  $("theme-toggle").setAttribute(
    "aria-pressed",
    String(settings.theme === "light"),
  );
  if (persist) localStorage.setItem("dirlab-theme", settings.theme);
  if (state.clouds.length) updateClouds();
  requestRender();
}
function methodControls() {
  $("methods").replaceChildren();
  state.order.forEach((method, index) => {
    const chip = document.createElement("span");
    chip.className = "method-chip";
    const text = document.createElement("span");
    text.textContent = label(method);
    chip.append(text);
    for (const [delta, name] of [
      [-1, "Earlier"],
      [1, "Later"],
    ]) {
      const b = document.createElement("button");
      b.textContent = name;
      b.setAttribute("aria-label", `${label(method)} ${name.toLowerCase()}`);
      b.disabled = index + delta < 0 || index + delta >= 3;
      b.onclick = () => {
        [state.order[index], state.order[index + delta]] = [
          state.order[index + delta],
          state.order[index],
        ];
        localStorage.setItem(
          "dirlab-method-order",
          JSON.stringify(state.order),
        );
        methodControls();
        if (state.manifest) {
          buildMatrices();
          updateLabels();
          requestRender();
        }
      };
      chip.append(b);
    }
    $("methods").append(chip);
  });
}
function updateCaseOptions(selected) {
  const cases = state.catalog.filter((c) => c.dataset === $("dataset").value);
  $("case").replaceChildren();
  for (const c of cases) {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = `Case ${String(c.case).padStart(2, "0")}`;
    if (c.case === selected) o.selected = true;
    $("case").append(o);
  }
}

$("dataset").onchange = () => {
  updateCaseOptions(1);
  loadCase($("case").value);
};
$("case").onchange = () => loadCase($("case").value);
$("reload-case").onclick = () => loadCase($("case").value);
$("theme-toggle").onclick = () =>
  setTheme(settings.theme === "dark" ? "light" : "dark");
$("tab-images").onclick = () => setTab("images");
$("tab-landmarks").onclick = () => setTab("landmarks");
$("slice").oninput = (e) => setSlice(Number(e.target.value));
for (const suffix of ["", "-bottom"]) {
  $("landmark-slider" + suffix).oninput = (e) =>
    setIndex(Number(e.target.value));
  $("previous" + suffix).onclick = () => setIndex(state.index - 1);
  $("next" + suffix).onclick = () => setIndex(state.index + 1);
}
for (const id of ["lung", "order", "direction"])
  $(id).onchange = () => reorderLandmarks();
$("shuffle").onclick = () => {
  state.seed++;
  $("order").value = "random";
  reorderLandmarks();
};
for (const [id, key] of [
  ["level", "level"],
  ["width", "width"],
  ["diff-range", "diffRange"],
  ["map-range", "mapRange"],
  ["grid-spacing", "gridSpacing"],
])
  $(id).oninput = (e) => {
    if (!e.target.value.trim()) return;
    const v = Number(e.target.value);
    if (Number.isFinite(v)) {
      settings[key] =
        key === "level" ? v : Math.max(key === "mapRange" ? 0.1 : 1, v);
      updateLabels();
      requestRender();
    }
  };
for (const key of ["anatomy", "target", "crosshair", "targetPoint"])
  $(key).onchange = (e) => {
    settings[key] = e.target.checked;
    requestRender();
  };
$("zoom").onchange = (e) => {
  settings.extent = Number(e.target.value);
  requestRender();
};
$("reset-cameras").onclick = () => {
  for (const c of state.clouds) normAngleCamera(c);
  requestRender();
};
window.addEventListener("resize", resize);
window.addEventListener("scroll", requestRender, { passive: true });
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const d = e.key === "ArrowRight" ? 1 : -1;
    state.tab === "landmarks"
      ? setIndex(state.index + d)
      : setSlice(state.slice + d);
    e.preventDefault();
  }
});
window.addEventListener("error", (e) => {
  setStatus("Viewer error", true);
  console.error(e.error);
});
$("renderer").addEventListener("webglcontextlost", (e) => {
  e.preventDefault();
  state.ready = false;
  setStatus("Graphics context lost · reload case", true);
  progress("Graphics context was lost. Select the case again to reload.", 0);
});
window.viewer = {
  loadCase,
  setTab,
  setIndex,
  setSlice,
  requestRender,
  get ready() {
    return state.ready;
  },
  get diagnostics() {
    return {
      ready: state.ready,
      case: state.manifest?.id,
      tab: state.tab,
      order: state.order.slice(),
      landmarkId: currentPoint()?.id,
      landmarkCount: state.list.length,
      treCdf: treCdf.diagnostics,
      slice: state.slice,
      displayBytes: state.displayBytes,
      gpuTextures: renderer.info.memory.textures,
      geometries: renderer.info.memory.geometries,
      renderCount: state.renderCount,
      drawTimes: state.drawTimes.slice(),
      latencies: state.latencies.slice(),
      loadHistory: state.loadHistory.slice(),
      renderer: gl.getParameter(gl.RENDERER),
      unmaskedRenderer: gl.getExtension("WEBGL_debug_renderer_info")
        ? gl.getParameter(
            gl.getExtension("WEBGL_debug_renderer_info")
              .UNMASKED_RENDERER_WEBGL,
          )
        : null,
    };
  },
  get manifest() {
    return state.manifest;
  },
  get settings() {
    return { ...settings };
  },
};
Object.defineProperty(window.viewer, "cameraState", {
  get: () =>
    state.clouds.map((c) => ({
      role: c.role,
      position: c.camera.position.toArray(),
      target: c.controls.target.toArray(),
      zoom: c.camera.zoom,
    })),
});
window.viewer.projectLandmark = (role, id) => {
  const c = state.clouds.find((c) => c.role === role),
    p = state.manifest.landmarks[id][`${role}_lps_mm`];
  const point = new THREE.Vector3(...p.map((v, i) => v - c.center[i])).project(
      c.camera,
    ),
    r = c.element.getBoundingClientRect();
  return {
    x: r.left + ((point.x + 1) * r.width) / 2,
    y: r.top + ((1 - point.y) * r.height) / 2,
  };
};
setTheme(settings.theme, false);
resize();
methodControls();
try {
  const r = await fetch("/api/catalog");
  if (!r.ok) throw new Error("Catalog unavailable");
  const catalog = await r.json();
  state.catalog = catalog.cases;
  if (!state.catalog.length) throw new Error("No prepared cases");
  updateCaseOptions(1);
  await loadCase($("case").value);
} catch (e) {
  setStatus(e.message, true);
  console.error(e);
}
