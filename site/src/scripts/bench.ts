/**
 * The on-device benchmark behind src/pages/bench.astro.
 *
 * Runs the deployed model through the same engine as the main page
 * (src/lib/model.ts: same prompt, stop string, token cap, greedy decoding) on a
 * fixed, seeded sample of PIZZA test orders (data/bench_orders.json), scores
 * each ticket with the in-tab matcher, and times it from handing the text to the
 * model until the ticket is painted. Nothing is uploaded: the result is offered
 * as a CSV or JSON download, or copied to the clipboard.
 *
 * URL parameters: ?n=20 (fewer timed orders), &label=Pixel%208, &auto=1 (load
 * and run without clicks; used by www2027_demo/bench/latency.spec.ts).
 * window.__bench exposes the same functions to Playwright, so the accuracy run
 * on the full test set uses this exact code path.
 */
import { engine } from "../lib/model";
import { readTicket } from "../lib/ticket";
import { wordsToExr } from "../lib/words";
import { isUnorderedExactMatch } from "../lib/matcher";
import catalog from "../../data/catalog.json";
import bench from "../../data/bench_orders.json";

export type Outcome = "correct" | "detectable" | "silent";
export interface Order { id: number; src: string; gold: string }
export interface Row {
  i: number; id: number; src: string; gold: string; raw: string; parsed: string | null;
  outcome: Outcome; refused: boolean;
  /** submit -> ticket painted (the number the paper reports) */
  ms: number;
  /** submit -> engine done */
  modelMs: number;
  /** submit -> first streamed token */
  firstMs: number;
  tokens: number;
  /** The worker died on this order and was reloaded; ms includes the reload. */
  recovered: boolean;
}

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
const frame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
const q = (xs: number[], p: number) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const k = (s.length - 1) * p, lo = Math.floor(k), hi = Math.ceil(k);
  return s[lo]! + (s[hi]! - s[lo]!) * (k - lo);
};
const share = (xs: number[], f: (x: number) => boolean) => (xs.length ? xs.filter(f).length / xs.length : NaN);
const r1 = (x: number) => Math.round(x * 10) / 10;
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** Same rule as the paper: silent = decodes to a valid tree but is not an exact match. */
export function score(raw: string, gold: string) {
  const parsed = wordsToExr(raw, catalog);
  const refused = raw.split(/[\s;]+/).includes("?");
  const outcome: Outcome = parsed === null ? "detectable" : isUnorderedExactMatch(parsed, gold) ? "correct" : "silent";
  return { parsed, refused, outcome };
}

/** One order, timed from submit to the painted ticket. */
async function runOne(o: Order, i: number, box: HTMLElement): Promise<Row> {
  const t0 = performance.now();
  const res = await engine.parse(o.src);
  const t1 = performance.now();
  const raw = res?.text ?? "";
  box.textContent = readTicket(raw).map((l) => [l.head, ...l.chips].join(" · ")).join("\n") || "(empty)";
  await frame();
  await frame();
  const t2 = performance.now();
  return {
    i, id: o.id, src: o.src, gold: o.gold, raw, ...score(raw, o.gold),
    ms: r1(t2 - t0), modelMs: r1(t1 - t0), firstMs: r1(res?.firstMs ?? NaN), tokens: res?.tokens ?? 0,
    recovered: res?.recovered ?? false,
  };
}

export async function runOrders(orders: Order[], onRow?: (r: Row) => void) {
  const box = $("#b-ticket");
  const out: Row[] = [];
  for (let i = 0; i < orders.length; i++) {
    const r = await runOne(orders[i]!, i, box);
    out.push(r);
    onRow?.(r);
  }
  return out;
}

export function summarise(rows: Row[]) {
  // A recovered order's time includes reloading the model, so it is not a parse time.
  const ms = rows.filter((r) => !r.recovered).map((r) => r.ms);
  const n = rows.length || NaN;
  const count = (o: Outcome) => rows.filter((r) => r.outcome === o).length;
  return {
    n: rows.length,
    medianMs: r1(q(ms, 0.5)),
    p90Ms: r1(q(ms, 0.9)),
    meanMs: r1(ms.reduce((a, b) => a + b, 0) / n),
    under100ms: r4(share(ms, (x) => x < 100)),
    under1s: r4(share(ms, (x) => x < 1000)),
    medianModelMs: r1(q(rows.map((r) => r.modelMs), 0.5)),
    medianFirstTokenMs: r1(q(rows.map((r) => r.firstMs), 0.5)),
    meanTokens: r1(rows.reduce((a, r) => a + r.tokens, 0) / n),
    exactMatch: r4(count("correct") / n),
    parseValidity: r4((count("correct") + count("silent")) / n),
    silentRate: r4(count("silent") / n),
    refusedRate: r4(rows.filter((r) => r.refused).length / n),
    recovered: rows.filter((r) => r.recovered).length,
  };
}

async function device() {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    userAgentData?: { getHighEntropyValues(h: string[]): Promise<Record<string, unknown>> };
  };
  let uaData: Record<string, unknown> | null = null;
  try {
    uaData = (await nav.userAgentData?.getHighEntropyValues(["platform", "platformVersion", "model", "architecture", "fullVersionList", "mobile"])) ?? null;
  } catch { /* not Chromium */ }
  return {
    userAgent: navigator.userAgent,
    uaData,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemoryGB: nav.deviceMemory ?? null,
    crossOriginIsolated: self.crossOriginIsolated,
    secureContext: self.isSecureContext,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
}

/* ---------------------------------------------------------------- the page */

const params = new URLSearchParams(location.search);
const timedAll = (bench.orders as Order[]).slice(bench.warmup);
const warm = (bench.orders as Order[]).slice(0, bench.warmup);
let result: Record<string, unknown> | null = null;
let hiddenDuringRun = false;

function setText(sel: string, t: string) { $(sel).textContent = t; }
const fmtMs = (x: number) => (Number.isFinite(x) ? (x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`) : "n/a");
const fmtPct = (x: number) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : "n/a");
const slug = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "device";

function save(name: string, type: string, text: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function csv(rows: Row[], label: string) {
  const cols = ["label", "i", "id", "ms", "modelMs", "firstMs", "tokens", "outcome", "refused", "recovered", "src", "raw"] as const;
  const esc = (v: unknown) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(c === "label" ? label : r[c])).join(","))].join("\n") + "\n";
}

async function run() {
  const label = ($<HTMLInputElement>("#b-label").value || "").trim() || "unnamed device";
  try { localStorage.setItem("bench-label", label); } catch { /* private mode */ }
  const n = Math.min(Number(params.get("n")) || timedAll.length, timedAll.length);
  const timed = timedAll.slice(0, n);
  const go = $<HTMLButtonElement>("#b-run");
  go.disabled = true;
  $("#b-result").hidden = true;
  hiddenDuringRun = document.hidden;

  let lock: { release(): Promise<void> } | null = null;
  try { lock = await (navigator as any).wakeLock?.request("screen"); } catch { /* not allowed */ }

  if (engine.state !== "ready") {
    setText("#b-status", "Loading the model…");
    await engine.load();
    if ((engine.state as string) !== "ready") {
      setText("#b-status", `Could not load the model: ${engine.error}`);
      go.disabled = false;
      return;
    }
  }
  const s = engine.stats!;
  setText("#b-load", `${s.cached ? "warm cache" : s.cached === false ? "cold cache" : "cache unknown"} · ` +
    `download ${fmtMs(s.downloadMs)} · init ${fmtMs(s.initMs)} · ${s.threads} thread${s.threads > 1 ? "s" : ""}`);

  const bar = $("#b-bar");
  const t0 = performance.now();
  let rows: Row[];
  let at = "warm-up";
  try {
    setText("#b-status", `Warming up (${warm.length} orders, not counted)…`);
    await runOrders(warm);
    rows = await runOrders(timed, (r) => {
      at = `order ${r.i + 2}`;
      bar.style.setProperty("--p", String((r.i + 1) / timed.length));
      setText("#b-status", `Order ${r.i + 1} of ${timed.length} · last ${fmtMs(r.ms)}` +
        (engine.restarts ? ` · ${engine.restarts} model reload${engine.restarts > 1 ? "s" : ""}` : ""));
    });
  } catch (e) {
    await lock?.release().catch(() => {});
    setText("#b-status", `Stopped at ${at}: ${e instanceof Error ? e.message : String(e)}. Reload the page and run again.`);
    go.disabled = false;
    return;
  }
  const wallS = (performance.now() - t0) / 1000;
  await lock?.release().catch(() => {});

  const m = engine.manifest;
  const sum = summarise(rows);
  result = {
    schema: 1,
    kind: "latency",
    label,
    date: new Date().toISOString(),
    page: location.origin + location.pathname,
    model: { run_id: m.run_id, url: m.url, sha256: m.sha256, bytes: m.bytes },
    device: await device(),
    load: { ...s, bytes: m.bytes },
    sample: { seed: bench.seed, warmup: warm.length, timed: rows.length, of: bench.of },
    hiddenDuringRun,
    modelRestarts: engine.restarts,
    wallSeconds: r1(wallS),
    summary: sum,
    rows,
  };
  (window as any).__bench.result = result;

  setText("#r-median", fmtMs(sum.medianMs));
  setText("#r-p90", fmtMs(sum.p90Ms));
  setText("#r-100", fmtPct(sum.under100ms));
  setText("#r-1s", fmtPct(sum.under1s));
  setText("#r-load", s.cached ? `${fmtMs(s.totalMs)} (warm)` : fmtMs(s.totalMs));
  setText("#r-ttft", fmtMs(sum.medianFirstTokenMs));
  setText("#r-acc", `${fmtPct(sum.exactMatch)} exact · ${fmtPct(sum.silentRate)} silent on ${rows.length} orders` +
    (sum.recovered ? ` · ${sum.recovered} order(s) needed a model reload and are left out of the times` : ""));
  setText("#r-dev", `${label} · ${navigator.hardwareConcurrency ?? "?"} cores · ${s.threads} threads used`);
  $("#b-warn").hidden = !hiddenDuringRun;
  $("#b-result").hidden = false;
  setText("#b-status", `Done in ${wallS.toFixed(0)} s.`);
  go.disabled = false;
  go.textContent = "Run again";
}

function boot() {
  const input = $<HTMLInputElement>("#b-label");
  try { input.value = params.get("label") ?? localStorage.getItem("bench-label") ?? ""; } catch { /* private mode */ }
  setText("#b-count", String(Math.min(Number(params.get("n")) || timedAll.length, timedAll.length)));
  setText("#b-dev", `${navigator.hardwareConcurrency ?? "?"} cores · ${self.crossOriginIsolated ? "cross-origin isolated (multithreaded)" : "NOT isolated: single thread"}`);
  if (!self.isSecureContext) {
    // Plain http on a LAN address: no model cache, no service worker, no threads.
    $("#b-insecure").hidden = false;
  }
  document.addEventListener("visibilitychange", () => { if (document.hidden) hiddenDuringRun = true; });

  engine.on(() => {
    if (engine.state === "loading") setText("#b-status", `Downloading the model… ${Math.round(engine.progress * 100)}%`);
    $("#b-bar").style.setProperty("--p", String(engine.state === "loading" ? engine.progress : 0));
  });

  $("#b-run").addEventListener("click", run);
  $("#b-clear").addEventListener("click", async () => {
    try {
      const { CacheManager } = await import("@wllama/wllama");
      await new CacheManager().clear();
    } catch { /* no OPFS here: nothing is cached */ }
    location.reload();
  });
  const label = () => slug(($<HTMLInputElement>("#b-label").value || "device"));
  $("#b-json").addEventListener("click", () => result && save(`latency_${label()}.json`, "application/json", JSON.stringify(result, null, 1)));
  $("#b-csv").addEventListener("click", () => result && save(`latency_${label()}.csv`, "text/csv", csv(result.rows as Row[], String(result.label))));
  $("#b-copy").addEventListener("click", async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result));
      setText("#b-copy", "Copied");
    } catch {
      setText("#b-copy", "Copy blocked: use Download");
    }
  });

  (window as any).__bench = { engine, score, runOrders, summarise, device, run, result: null };
  if (params.get("auto") === "1") run();
}

boot();
