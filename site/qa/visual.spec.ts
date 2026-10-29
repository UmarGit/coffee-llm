import { test, expect } from "@playwright/test";

const SECTIONS = ["order", "spell", "rank", "silent", "race", "again", "cafe", "end"];

for (const [w, h] of [[1440, 900], [390, 844]] as const) {
  test(`sections at ${w}px`, async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.setViewportSize({ width: w, height: h });
    await page.goto("./");
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `qa/out/${info.project.name}-${w}-hero.png` });
    for (const id of SECTIONS) {
      await page.locator(`#${id}`).scrollIntoViewIfNeeded();
      await page.waitForTimeout(1500);
      await page.screenshot({ path: `qa/out/${info.project.name}-${w}-${id}.png` });
    }
    expect(errors).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(w);
  });
}
