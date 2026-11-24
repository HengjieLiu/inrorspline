import assert from "node:assert/strict";
import path from "node:path";

export async function checkTreCdf(page, manifest, out) {
  const keys = ["initial", "mrdbscp", "ptvreg", "corrfield"];
  const chart = () => page.evaluate(() => viewer.diagnostics.treCdf);
  const verify = async () => {
    const d = await page.evaluate(() => viewer.diagnostics);
    const c = d.treCdf;
    const points = manifest.landmarks.filter((p) => c.side === "both" || p.side === c.side);
    assert.equal(c.count, points.length);
    assert.equal(c.count, d.landmarkCount);
    assert.equal(await page.locator(".cdf-curve").count(), 4);
    if (!c.ranking) {
      assert.equal(c.sweep, null);
      assert.equal(await page.locator(".cdf-cursor").getAttribute("visibility"), "hidden");
      assert.equal(await page.locator(".cdf-cursor circle").isVisible(), false);
      assert.equal(await page.locator(".cdf-cursor path").isVisible(), false);
      return c;
    }
    const error = manifest.landmarks[d.landmarkId].tre_mm[c.ranking];
    const count = points.filter((p) => p.tre_mm[c.ranking] <= error).length;
    assert.equal(c.sweep.landmarkId, d.landmarkId);
    assert.equal(c.sweep.tolerance, error);
    assert.equal(c.sweep.count, count);
    assert.equal(c.sweep.percent, 100 * count / points.length);
    assert.equal(c.sweep.outside, error > c.xMax);
    assert.match(await page.locator(".cdf-readout").textContent(), new RegExp(`${error.toFixed(2).replace(".", "\\.")} mm`));
    assert.equal(await page.locator(".cdf-curve.ranked").getAttribute("data-method"), c.ranking);
    assert.equal(await page.locator(".cdf-cursor circle").getAttribute("visibility"), c.sweep.outside ? "hidden" : "visible");
    return c;
  };

  await page.locator("#tab-landmarks").click();
  let checks = 0;
  for (const side of ["both", "left", "right"]) {
    await page.locator("#lung").selectOption(side);
    for (const metric of keys) {
      await page.locator("#order").selectOption(metric);
      for (const direction of ["descending", "ascending"]) {
        await page.locator("#direction").selectOption(direction);
        const before = await verify();
        const first = before.sweep.tolerance;
        await page.locator("#next").click();
        const next = await verify();
        assert.ok(direction === "ascending" ? next.sweep.tolerance >= first : next.sweep.tolerance <= first);
        assert.equal(next.pathBuilds, before.pathBuilds, "Navigation must not rebuild CDF paths");
        assert.equal(next.xMax, before.xMax, "Navigation must not change the tolerance axis");
        checks++;
      }
    }
  }
  await page.locator("#lung").selectOption("both");
  await page.locator("#order").selectOption("corrfield");
  await page.locator("#direction").selectOption("descending");
  await page.locator(".cdf-full").click();
  const full = await verify();
  assert.equal(full.rangeMode, "full");
  assert.ok(full.xMax >= Math.max(...manifest.landmarks.flatMap((p) => keys.map((k) => p.tre_mm[k]))));
  await page.locator(".clouds").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(out, `${manifest.id}_tre_cdf_full_dark.png`) });
  await page.locator(".cdf-zoom").click();
  assert.equal((await verify()).xMax, 6);
  await page.locator("#theme-toggle").click();
  await page.locator(".clouds").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(out, `${manifest.id}_tre_cdf_zoom_light.png`) });
  await page.locator("#theme-toggle").click();
  await page.locator("#tre-cdf input").fill("0.01");
  await page.locator("#tre-cdf input").press("Tab");
  assert.equal((await verify()).sweep.outside, true);
  assert.match(await page.locator(".cdf-readout").textContent(), /Outside displayed range/);
  await page.locator("#lung").selectOption("left");
  assert.equal((await verify()).xMax, 0.01, "Explicit zoom survives lung filtering");
  await page.locator("#lung").selectOption("both");
  await page.locator(".cdf-full").click();

  // Both slider bindings and the duplicated buttons must select the same marker.
  for (const suffix of ["", "-bottom"]) {
    await page.locator(`#landmark-slider${suffix}`).evaluate((node) => {
      node.value = "74";
      node.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await verify();
    await page.locator(`#next${suffix}`).click();
    await verify();
    await page.locator(`#previous${suffix}`).click();
    await verify();
  }
  const before = await page.evaluate(() => viewer.diagnostics);
  const names = { mrdbscp: "MR-D-BSCP", ptvreg: "pTVReg", corrfield: "CorrField" };
  await page.getByRole("button", { name: `${names[before.order[1]]} earlier`, exact: true }).click();
  const after = await verify();
  assert.equal(after.sweep.landmarkId, before.landmarkId);
  assert.equal(after.ranking, before.treCdf.ranking);
  assert.deepEqual(await page.locator(".cdf-legend span").evaluateAll((nodes) => nodes.map((n) => n.dataset.method)), ["initial", ...(await page.evaluate(() => viewer.diagnostics.order))]);
  await page.getByRole("button", { name: `${names[before.order[1]]} later`, exact: true }).click();
  assert.equal((await chart()).pathBuilds, before.treCdf.pathBuilds, "Reordering legend/columns preserves CDF paths");

  for (const order of ["superior", "random"]) {
    await page.locator("#order").selectOption(order);
    await verify();
  }
  await page.locator("#shuffle").click();
  await verify();
  await page.locator("#order").selectOption("initial");
  await page.locator("#direction").selectOption("descending");
  await page.locator(".clouds").scrollIntoViewIfNeeded();
  const builds = (await chart()).pathBuilds;
  const performance = await page.evaluate(async () => {
    const times = [];
    let last = window.performance.now();
    for (let i = 0; i < 90; i++) {
      viewer.setIndex(i);
      await new Promise(requestAnimationFrame);
      const now = window.performance.now();
      times.push(now - last);
      last = now;
    }
    const sample = times.slice(5), sorted = sample.slice().sort((a, b) => a - b);
    return { fps: 1000 * sample.length / sample.reduce((a, b) => a + b), p95_frame_ms: sorted[Math.floor(sorted.length * 0.95)] };
  });
  await verify();
  assert.equal((await chart()).pathBuilds, builds);
  await page.locator("#order").selectOption("superior");
  await page.locator("#tab-images").click();
  await page.evaluate(() => window.scrollTo(0, 0));
  return { checks, performance, status: "passed" };
}
