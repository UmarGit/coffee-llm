/**
 * The brief requires a production build to fail if any placeholder data is
 * present. This runs before `astro build`.
 *
 * Every file in data/ (the committed snapshot the page is built from) must
 * carry a schema version and the run ids it came from, so no number reaches
 * the page untraceable. Site text must not contain em dashes.
 */
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const DATA = new URL("../data/", import.meta.url).pathname;
const PLACEHOLDER = /\b(lorem ipsum|TODO|FIXME|PLACEHOLDER|FAKE_|DUMMY_DATA)\b/i;

const files = (await readdir(DATA)).filter((f) => f.endsWith(".json"));
if (!files.length) {
  console.error("no data in data/. Run tools/export_site_data.py, then npm run snapshot");
  process.exit(1);
}

let failed = false;
const fail = (m) => {
  console.error(`  FAIL ${m}`);
  failed = true;
};

for (const f of files) {
  const text = await readFile(join(DATA, f), "utf8");
  if (PLACEHOLDER.test(text)) fail(`${f} contains placeholder text`);
  let d;
  try {
    d = JSON.parse(text);
  } catch {
    fail(`${f} is not valid JSON`);
    continue;
  }
  if (typeof d.schema !== "number") fail(`${f} has no schema version`);
  if (!Array.isArray(d.sources)) fail(`${f} has no sources list`);
  if (Array.isArray(d.sources) && d.sources.length === 0) {
    fail(`${f} records no source run ids`);
  }
}

// No em dashes in site text.
const srcDir = new URL("../src/", import.meta.url).pathname;
async function walk(dir) {
  const out = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await walk(full)));
    else if (/\.(astro|ts|css)$/.test(e.name)) out.push(full);
  }
  return out;
}
for (const f of await walk(srcDir)) {
  if ((await readFile(f, "utf8")).includes("\u2014")) {
    fail(`${f.replace(srcDir, "src/")} contains an em dash`);
  }
}

console.log(`data check: ${files.length} files${failed ? "" : " ok"}`);
process.exit(failed ? 1 : 0);
