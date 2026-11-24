// Browser checks write only to the caller's external validation directory.
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { checkTreCdf } from "./browser_cdf_check.mjs";
import {
  orderedLandmarks,
  planeGeometry,
  validMethodOrder,
} from "./web/model.js";

const out = process.env.VIEWER_VALIDATION_ROOT;
if (!out || !process.env.VIEWER_PLAYWRIGHT_MODULE)
  throw new Error(
    "Set external VIEWER_VALIDATION_ROOT and VIEWER_PLAYWRIGHT_MODULE",
  );
await fs.mkdir(out, { recursive: true });
const { chromium } = await import(
  pathToFileURL(process.env.VIEWER_PLAYWRIGHT_MODULE)
);
const hardware =
  process.env.VIEWER_ANGLE && process.env.VIEWER_ANGLE !== "swiftshader";
const browser = await chromium.launch({
  headless: true,
  args: [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    `--use-angle=${process.env.VIEWER_ANGLE || "swiftshader"}`,
    ...(hardware
      ? [
          "--enable-gpu",
          "--ignore-gpu-blocklist",
          "--disable-software-rasterizer",
        ]
      : ["--enable-unsafe-swiftshader"]),
    "--js-flags=--max-old-space-size=4096",
  ],
});
const page = await browser.newPage({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
});
const cdp = await page.context().newCDPSession(page);
const errors = [],
  requests = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.on("request", (r) => requests.push(r.url()));
const report = {
  browser: browser.version(),
  viewport: { width: 1920, height: 1080 },
  renderer_scope: hardware
    ? "Headless hardware rendering requested; actual renderer recorded per case"
    : "Headless SwiftShader software renderer; desktop hardware gate not certified",
  cases: [],
  errors,
};
async function checkTreHeaders() {
  const snapshot = await page.evaluate(() => {
    const point = viewer.manifest.landmarks[viewer.diagnostics.landmarkId];
    return {
      errors: point.tre_mm,
      order: viewer.diagnostics.order,
      methods: viewer.manifest.methods,
      heads: [...document.querySelectorAll("#mpr .column-head")].map((head) => ({
        method: head.dataset.method,
        tre: head.querySelector(".column-tre")?.textContent ?? null,
        firstSubtitleIsTre: head.firstElementChild?.classList.contains("column-tre"),
        guard: head.querySelector("small")?.textContent ?? null,
      })),
    };
  });
  assert.equal(await page.locator("#tre-values, #tre-values-bottom").count(), 0);
  assert.equal(await page.locator("#comparison .column-tre").count(), 0);
  assert.deepEqual(snapshot.heads.map((h) => h.method), ["moving", ...snapshot.order, "fixed"]);
  for (const h of snapshot.heads) {
    if (h.method === "fixed") {
      assert.equal(h.tre, null);
      continue;
    }
    const key = h.method === "moving" ? "initial" : h.method;
    assert.equal(h.tre, `${key === "initial" ? "Initial TRE" : "TRE"}: ${snapshot.errors[key].toFixed(2)} mm`);
    assert.equal(h.firstSubtitleIsTre, true);
    const info = snapshot.methods.find((m) => m.id === h.method);
    assert.equal(h.guard, info?.variant.includes("forest") ? "Graph guard" : null);
  }
}
try {
  await page.goto(process.env.VIEWER_URL || "http://127.0.0.1:1101");
  await page.waitForFunction(
    () => window.viewer?.ready,
    {},
    { timeout: 180000 },
  );
  const ids = (
    process.env.VIEWER_CASES || "4dct_01,4dct_02,copd_01,copd_10"
  ).split(",");
  for (const id of ids) {
    if ((await page.evaluate(() => viewer.diagnostics.case)) !== id) {
      await page.evaluate((id) => {
        viewer.loadCase(id);
      }, id);
      await page.waitForFunction(
        (id) => viewer.ready && viewer.diagnostics.case === id,
        id,
        { timeout: 180000 },
      );
    }
    const start = requests.length;
    const m = await page.evaluate(() => viewer.manifest);
    assert.equal(await page.evaluate(() => viewer.diagnostics.treCdf.rangeMode), "full", "Each case starts with the full CDF range");
    const cdf = await checkTreCdf(page, m, out);
    await page.evaluate(() => viewer.setTab("images"));
    assert.equal(
      await page.locator('#comparison .view[data-mode="1"]').count(),
      5,
    );
    assert.match(
      await page.locator("#comparison .row-head").nth(1).textContent(),
      /from fixed/,
    );
    await page.locator("#anatomy").uncheck();
    await page.locator("#target").uncheck();
    const readFixedPixels = async () =>
      page.evaluate(
        () =>
          new Promise((resolve) => {
            viewer.requestRender();
            requestAnimationFrame(() => {
              const canvas = document.getElementById("renderer"),
                gl = canvas.getContext("webgl2");
              const ratio = canvas.width / innerWidth;
              const pixel = (mode) => {
                const r = document
                  .querySelector(
                    `#comparison .view[data-mode="${mode}"][data-method="fixed"]`,
                  )
                  .getBoundingClientRect();
                const p = new Uint8Array(4);
                gl.readPixels(
                  Math.floor((r.left + r.width / 2) * ratio),
                  Math.floor((innerHeight - r.top - r.height / 2) * ratio),
                  1,
                  1,
                  gl.RGBA,
                  gl.UNSIGNED_BYTE,
                  p,
                );
                return Array.from(p);
              };
              resolve({ difference: pixel(1), ct: pixel(0) });
            });
          }),
      );
    const darkPixels = await readFixedPixels();
    [237, 233, 224].forEach((value, i) =>
      assert.ok(
        Math.abs(darkPixels.difference[i] - value) <= 2,
        "Fixed minus Fixed is exactly the zero color",
      ),
    );
    await page.locator("#theme-toggle").click();
    assert.equal(
      await page.evaluate(() => document.documentElement.dataset.theme),
      "light",
    );
    const lightPixels = await readFixedPixels();
    assert.deepEqual(
      lightPixels,
      darkPixels,
      "Theme preserves CT intensities and scientific colormaps",
    );
    await page.screenshot({
      path: path.join(out, `${id}_light_comparison.png`),
    });
    await page.locator("#theme-toggle").click();
    assert.equal(
      await page.evaluate(() => document.documentElement.dataset.theme),
      "dark",
    );
    await page.locator("#anatomy").check();
    await page.locator("#target").check();
    assert.equal(m.landmarks.length, 300);
    assert.equal(m.lung_counts.left + m.lung_counts.right, 300);
    if (id === "copd_01") assert.equal(m.geometry.shape_zyx[0], 272);
    for (const key of [
      "superior",
      "initial",
      "mrdbscp",
      "ptvreg",
      "corrfield",
    ]) {
      const desc = orderedLandmarks(m.landmarks, "both", key, "descending");
      const asc = orderedLandmarks(m.landmarks, "both", key, "ascending");
      const value = (i) =>
        key === "superior"
          ? m.landmarks[i].fixed_lps_mm[2]
          : m.landmarks[i].tre_mm[key];
      assert.ok(desc.every((v, i) => !i || value(desc[i - 1]) >= value(v)));
      assert.ok(asc.every((v, i) => !i || value(asc[i - 1]) <= value(v)));
    }
    assert.deepEqual(
      orderedLandmarks(m.landmarks, "left", "random", "descending", 1),
      orderedLandmarks(m.landmarks, "left", "random", "descending", 1),
    );
    const g = planeGeometry(m.geometry, 0, m.landmarks[0].fixed_lps_mm, 10);
    assert.deepEqual(g.span, [10, 10]);
    await page.locator("#tab-landmarks").click();
    const treLabels = await page.locator("#mpr .column-tre").allTextContents();
    assert.equal(treLabels.length, 4);
    assert.ok(
      treLabels.every((t) => /^(Initial TRE|TRE): \d+\.\d{2} mm$/.test(t)),
      "All TRE labels use two decimal places",
    );
    await checkTreHeaders();
    assert.match(
      (await page.locator("#marker-legend").textContent()).replace(/\s+/g, " "),
      /Yellow ring: fixed target projected/,
    );
    await page.locator("#targetPoint").uncheck();
    assert.equal(await page.evaluate(() => viewer.settings.targetPoint), false);
    assert.equal(
      await page.locator("#target").isChecked(),
      true,
      "Marker toggle is independent of fixed contour",
    );
    await page.locator("#targetPoint").check();
    await page.locator("#lung").selectOption("left");
    assert.equal(
      await page.evaluate(() => viewer.diagnostics.landmarkCount),
      m.lung_counts.left,
    );
    await page.locator("#lung").selectOption("right");
    assert.equal(
      await page.evaluate(() => viewer.diagnostics.landmarkCount),
      m.lung_counts.right,
    );
    await page.locator("#lung").selectOption("both");
    await page.locator("#order").selectOption("corrfield");
    await page.locator("#direction").selectOption("descending");
    const expected = orderedLandmarks(
      m.landmarks,
      "both",
      "corrfield",
      "descending",
    )[0];
    assert.equal(
      await page.evaluate(() => viewer.diagnostics.landmarkId),
      expected,
    );
    await page.locator("#previous").click();
    assert.match(
      await page.locator("#landmark-label").textContent(),
      /^300 \/ 300/,
    );
    await page.locator("#next").click();
    await page.locator("#next-bottom").click();
    assert.equal(
      await page.locator("#landmark-slider-bottom").inputValue(),
      await page.locator("#landmark-slider").inputValue(),
    );
    assert.equal(
      await page.locator("#landmark-label-bottom").textContent(),
      await page.locator("#landmark-label").textContent(),
    );
    await checkTreHeaders();
    await page.locator("#previous-bottom").click();
    await page.locator("#previous").scrollIntoViewIfNeeded();
    const beforePick = await page.evaluate(() => viewer.diagnostics.landmarkId);
    let picked = false;
    for (let attempt = 1; attempt <= 5 && !picked; attempt++) {
      const xy = await page.evaluate(
        (id) => viewer.projectLandmark("moving", id),
        (beforePick + attempt * 37) % 300,
      );
      await page.mouse.click(xy.x, xy.y);
      picked = await page.evaluate(
        (id) => viewer.diagnostics.landmarkId !== id,
        beforePick,
      );
    }
    assert.ok(picked, "3D point picking updates the correspondence");
    await checkTreHeaders();
    assert.equal(await page.evaluate(() => viewer.diagnostics.treCdf.sweep.landmarkId), await page.evaluate(() => viewer.diagnostics.landmarkId));
    // Genuine pointer interaction in the 3D views must update the linked camera.
    const beforeCamera = await page.evaluate(() => viewer.cameraState);
    const cloud = await page.locator("#cloud-moving").boundingBox();
    await page.mouse.move(
      cloud.x + cloud.width * 0.5,
      cloud.y + cloud.height * 0.5,
    );
    await page.mouse.down();
    await page.mouse.move(
      cloud.x + cloud.width * 0.55,
      cloud.y + cloud.height * 0.6,
      { steps: 5 },
    );
    await page.mouse.up();
    const camera = await page.evaluate(() => viewer.cameraState);
    assert.notDeepEqual(camera[0].position, beforeCamera[0].position);
    assert.ok(
      Math.hypot(...camera[0].position.map((v, i) => v - camera[1].position[i])) < 1e-9,
      "Linked cameras agree within floating-point roundoff",
    );
    await page.locator("#reset-cameras").click();
    await page.locator("#shuffle").click();
    await page.locator("#zoom").selectOption("10");
    assert.equal(await page.evaluate(() => viewer.settings.extent), 10);
    await page.locator("#crosshair").uncheck();
    assert.equal(await page.evaluate(() => viewer.settings.crosshair), false);
    await page.locator("#crosshair").check();
    await page.screenshot({ path: path.join(out, `${id}_landmarks.png`) });
    await page.locator("#theme-toggle").click();
    await page.screenshot({
      path: path.join(out, `${id}_light_landmarks.png`),
    });
    await page.locator("#theme-toggle").click();
    const beforeOrder = await page.evaluate(() => viewer.diagnostics.order);
    const selected = await page.evaluate(() => viewer.diagnostics.landmarkId);
    await page
      .getByRole("button", {
        name: `${{ mrdbscp: "MR-D-BSCP", ptvreg: "pTVReg", corrfield: "CorrField" }[beforeOrder[1]]} earlier`,
        exact: true,
      })
      .click();
    assert.equal(
      await page.evaluate(() => viewer.diagnostics.landmarkId),
      selected,
    );
    assert.ok(
      validMethodOrder(await page.evaluate(() => viewer.diagnostics.order)),
    );
    await checkTreHeaders();
    await page.locator("#tab-images").click();
    await page.locator("#anatomy").uncheck();
    await page.locator("#anatomy").check();
    await page.locator("#target").uncheck();
    await page.locator("#target").check();
    const s = await page.evaluate(() => viewer.diagnostics.slice);
    await page.locator("#comparison .view:not(.blank)").first().hover();
    await page.mouse.wheel(0, 100);
    await page.waitForFunction((s) => viewer.diagnostics.slice === s + 1, s);
    await page.screenshot({ path: path.join(out, `${id}_comparison.png`) });
    // All five rows, including the signed-SI row, must render after page scroll.
    await page.locator("#comparison .row-head").last().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, `${id}_deformation.png`) });
    await page.evaluate(() => window.scrollTo(0, 0));
    // Request one coronal change per animation frame; record actual frame pacing.
    const perf = await page.evaluate(async () => {
      const times = [];
      let last = performance.now();
      for (let i = 0; i < 90; i++) {
        viewer.setSlice(40 + (i % 30));
        await new Promise(requestAnimationFrame);
        const now = performance.now();
        times.push(now - last);
        last = now;
      }
      const sorted = times.slice(5).sort((a, b) => a - b);
      return {
        fps:
          1000 /
          (times.slice(5).reduce((a, b) => a + b, 0) / (times.length - 5)),
        p95_frame_ms: sorted[Math.floor(sorted.length * 0.95)],
      };
    });
    await page.evaluate(() => viewer.setTab("landmarks"));
    await page.locator("#zoom").selectOption("50");
    await page.locator("#zoom").selectOption("0");
    await page.locator("#mpr .row-head").last().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, `${id}_mpr.png`) });
    await page.evaluate(() => window.scrollTo(0, 0));
    assert.equal(requests.length, start, "No post-Ready network requests");
    const diagnostics = await page.evaluate(() => viewer.diagnostics);
    await cdp.send("HeapProfiler.collectGarbage");
    const heap = await cdp.send("Runtime.getHeapUsage");
    if (heap.backingStorageSize !== undefined) {
      assert.ok(
        heap.backingStorageSize <
          diagnostics.displayBytes * 1.25 + 256 * 1024 ** 2,
        "Previous-case ArrayBuffers released",
      );
    }
    const latency = diagnostics.latencies.slice().sort((a, b) => a - b);
    report.cases.push({
      id,
      ...perf,
      tre_cdf: cdf,
      header_tre: "passed",
      p95_interaction_ms: latency[Math.floor(latency.length * 0.95)],
      request_free: true,
      heap,
      diagnostics,
    });
    console.log(
      id,
      JSON.stringify({
        fps: perf.fps,
        p95_frame_ms: perf.p95_frame_ms,
        textures: diagnostics.gpuTextures,
      }),
    );
  }
  // A canceled selection cannot replace the latest case or leak its textures.
  const finalId = ids[ids.length - 1];
  await page.evaluate(
    async ({ first, finalId }) => {
      viewer.loadCase(first);
      await new Promise((r) => setTimeout(r, 20));
      viewer.loadCase(finalId);
    },
    { first: ids[0], finalId },
  );
  await page.waitForFunction(
    (id) => viewer.ready && viewer.diagnostics.case === id,
    finalId,
    { timeout: 180000 },
  );
  const final = await page.evaluate(() => viewer.diagnostics);
  assert.ok(final.gpuTextures <= 15, "Previous-case GPU textures released");
  report.cancellation_and_disposal = { status: "passed", final };
  const persistedOrder = final.order;
  await page.locator("#theme-toggle").click();
  await page.reload();
  await page.waitForFunction(
    () => window.viewer?.ready,
    {},
    { timeout: 180000 },
  );
  assert.deepEqual(
    await page.evaluate(() => viewer.diagnostics.order),
    persistedOrder,
  );
  report.order_persistence = "passed";
  assert.equal(
    await page.evaluate(() => document.documentElement.dataset.theme),
    "light",
  );
  report.theme_persistence = "passed";
  assert.deepEqual(errors, []);
  report.status = "accepted_functional";
  report.performance_gate = report.cases.every(
    (c) => c.fps >= 30 && c.tre_cdf.performance.fps >= 30 && c.p95_interaction_ms < 100,
  )
    ? "passed"
    : "not_met_on_recorded_renderer";
} catch (error) {
  report.status = "failed";
  report.failure = error.stack;
  console.error(error);
  process.exitCode = 1;
} finally {
  await fs.writeFile(
    path.join(out, "browser_report.json"),
    JSON.stringify(report, null, 2),
  );
  await browser.close();
}
