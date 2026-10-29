/**
 * Reduce the exported run data to the small snapshot the site is built from.
 *
 *   uv run --extra pizza python tools/export_site_data.py   # runs/ -> src/data/, raw/
 *   npm run snapshot                                        # -> data/ (committed)
 *
 * data/ is what GitHub Actions builds the page from, because runs/ never leaves
 * the machine it was made on. It holds aggregate numbers and the few dozen
 * example orders the page actually shows, nothing else. Those rows come from
 * the PIZZA test set and are CC-BY-NC-4.0, like the page itself.
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

const SITE = new URL("..", import.meta.url).pathname;
const SRC = join(SITE, "src", "data");
const RAW = join(SITE, "raw");
const OUT = join(SITE, "data");

const read = async (dir, f) => JSON.parse(await readFile(join(dir, f), "utf8"));

/** A fixed, seeded pick, so the snapshot is identical across runs. */
function pick(xs, n, seed) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  return xs.map((x) => [rnd(), x]).sort((a, b) => a[0] - b[0]).slice(0, n).map((p) => p[1]);
}

// A fixed rhythm, so a silent error shows up within the first few hero tickets.
const RHYTHM = [0, 2, 0, 1, 0, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0];
function interleave(ok, loud, silent) {
  const by = [ok, loud, silent].map((xs) => [...xs]);
  return RHYTHM.map((o) => by[o].shift()).filter(Boolean);
}

async function rounds() {
  const tickets = (await read(RAW, "tickets.json")).tickets;
  const arm = (await read(RAW, "outcomes.qwen.json")).arms.words;
  const all = tickets.map((t, i) => ({
    src: t.src, model: arm.raw[i], truth: t.enc.words.t, o: arm.outcome[i],
  }));
  // Short orders only, so a ticket reads in a couple of seconds.
  const rs = all.filter((r) => r.src.length <= 84 && r.model.split(";").length <= 2);
  const by = (o) => rs.filter((r) => r.o === o);
  return {
    // Silent errors are rare (6%), so the game over-samples them on purpose.
    game: { silent: pick(by(2), 30, 11), correct: pick(by(0), 40, 13) },
    hero: interleave(pick(by(0), 9, 5), pick(by(1), 3, 6), pick(by(2), 4, 7)),
  };
}

const KEEP = ["formats.json", "hero.json", "examples.json", "counts.qwen.json"];

// The live model: its export manifest, plus where the published site downloads
// it from. GitHub cannot hold an 834 MB file, so it lives on Hugging Face.
const HF_REPO = "umarsheikh303/pizza-words-gguf";
const MODEL = "pizza-words-q8_0";

await mkdir(OUT, { recursive: true });
for (const f of KEEP) await writeFile(join(OUT, f), JSON.stringify(await read(SRC, f)) + "\n");
const tickets = await read(RAW, "tickets.json");
await writeFile(join(OUT, "rounds.json"), JSON.stringify({
  schema: 1,
  generated: tickets.generated,
  sources: (await read(RAW, "outcomes.qwen.json")).sources,
  ...(await rounds()),
}) + "\n");
const manifest = await read(join(SITE, "public", "models"), `${MODEL}.json`).catch(() => null);
if (manifest) {
  const file = Object.values(manifest.files)[0];
  await writeFile(join(OUT, "model.json"), JSON.stringify({
    schema: 1,
    sources: [manifest.run_id],
    url: `https://huggingface.co/${HF_REPO}/resolve/main/${MODEL}.gguf`,
    page: `https://huggingface.co/${HF_REPO}`,
    ...manifest,
    bytes: file.bytes,
    sha256: file.sha256,
  }, null, 2) + "\n");
} else {
  console.log("no public/models manifest; data/model.json left as it is");
}
console.log(`snapshot: data/ updated`);
