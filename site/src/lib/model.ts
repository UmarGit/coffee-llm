/**
 * The real model, running in this tab through wllama (llama.cpp compiled to
 * WebAssembly). One instance per page, shared by the hero and the game.
 *
 * The prompt, stop string and token cap come from the export manifest, the same
 * contract the GGUF was verified under on the server, so the browser asks the
 * model exactly what evaluation asked it.
 */
import type { Wllama } from "@wllama/wllama";
import exported from "../../data/model.json";

export type Manifest = typeof exported;
export type State = "idle" | "loading" | "ready" | "error";

/**
 * The published site downloads the model from Hugging Face. For local work set
 * PUBLIC_MODEL_URL (see .env.development) to a copy under public/; a relative
 * value resolves against the site's base path.
 */
function modelUrl() {
  // wllama names its OPFS cache entry after the URL, and a new model is published
  // under the same file name. The sha256 in the query makes each model its own
  // cache entry, so a browser can never run an old cached model by mistake.
  // Hugging Face and static servers ignore the parameter.
  const v = `?v=${exported.sha256.slice(0, 16)}`;
  const local = import.meta.env.PUBLIC_MODEL_URL as string | undefined;
  if (!local) return exported.url + v;
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");
  return new URL(local, new URL(base, location.origin)).href + v;
}

type Listener = () => void;

/**
 * wllama 3 runs llama-server inside the worker, by default with 4 parallel slots
 * and a prompt cache for idle slots. The page only ever runs one parse at a time,
 * so one slot is enough, and no state is kept between parses (see cache_prompt
 * in parse). Greedy decoding of one request does not depend on these settings.
 */
const CTX = { n_ctx: 512, n_parallel: 1, cache_idle_slots: false };
const NO_PROMPT_CACHE = { cache_prompt: false } as {};

/** First-load cost, read by the bench page (src/pages/bench.astro). */
export interface LoadStats {
  /** The model file was already in the OPFS cache before this load. */
  cached: boolean | null;
  /** ms from load() to the last downloaded byte (0 when cached). */
  downloadMs: number;
  /** ms from the last byte to the model being ready. */
  initMs: number;
  totalMs: number;
  threads: number;
  multithread: boolean;
  /** Where the model file is kept: the OPFS cache, or memory for this visit only. */
  storage: "opfs" | "memory";
}

/**
 * Stand-in for wllama's OPFS cache where OPFS does not exist: pages that are not
 * a secure context (plain http on a LAN address) and some private windows.
 * The model then lives in memory for this visit and is downloaded again next time.
 */
class MemoryBackend {
  private files = new Map<string, Blob>();
  isSupported() { return true; }
  async read(key: string) { return this.files.get(key) ?? null; }
  async write(key: string, stream: ReadableStream) { this.files.set(key, await new Response(stream).blob()); }
  async getSize(key: string) { return this.files.get(key)?.size ?? -1; }
  async list() { return [...this.files].map(([key, b]) => ({ key, size: b.size })); }
  async delete(key: string) { this.files.delete(key); }
}

class Engine {
  state: State = "idle";
  progress = 0;
  error = "";
  manifest: Manifest = exported;
  runs = 0;
  /** Times the worker died and was reloaded from the cache (see parse). */
  restarts = 0;
  stats: LoadStats | null = null;
  private w: Wllama | null = null;
  private listeners = new Set<Listener>();
  private queue: Promise<unknown> = Promise.resolve();

  on(fn: Listener) {
    this.listeners.add(fn);
    fn();
    return () => this.listeners.delete(fn);
  }
  private emit() {
    this.listeners.forEach((fn) => fn());
  }
  private set(state: State, extra: Partial<Engine> = {}) {
    Object.assign(this, extra, { state });
    this.emit();
  }

  get bytes() {
    return this.manifest.bytes;
  }

  async load() {
    if (this.state === "ready" || this.state === "loading") return;
    this.set("loading", { progress: 0, error: "" });
    const t0 = performance.now();
    let tBytes = 0;
    try {
      const [{ Wllama, LoggerWithoutDebug, CacheManager }, { default: wasm }] = await Promise.all([
        import("@wllama/wllama"),
        import("@wllama/wllama/esm/wasm/wllama.wasm?url"),
      ]);
      let storage: LoadStats["storage"] = "opfs";
      let w: Wllama;
      try {
        w = new Wllama({ default: wasm }, { logger: LoggerWithoutDebug });
      } catch (e) {
        // wllama builds its OPFS cache in the constructor and throws without it.
        if (!/storage backend/i.test(String(e))) throw e;
        storage = "memory";
        w = new Wllama({ default: wasm }, {
          logger: LoggerWithoutDebug,
          cacheManager: new CacheManager([new MemoryBackend()]),
        });
      }
      const tick = (loaded: number, total: number) => {
        this.progress = total ? loaded / total : 0;
        if (total && loaded >= total && !tBytes) tBytes = performance.now();
        this.emit();
      };
      const url = modelUrl();
      let cached: boolean | null = null;
      try {
        cached = (await w.cacheManager.getSize(await w.cacheManager.getNameFromURL(url))) > 0;
      } catch { /* no OPFS: unknown */ }
      try {
        // Cached, so a second visit loads from disk instead of the network.
        await w.loadModelFromUrl(url, {
          ...CTX,
          progressCallback: ({ loaded, total }: { loaded: number; total: number }) => tick(loaded, total),
        });
      } catch (e) {
        // A full disk or a private window refuses the cache, and wllama's own
        // uncached path still writes to it, so download into memory instead.
        if (!/quota|storage/i.test(String(e))) throw e;
        await w.loadModel([await fetchBlob(url, this.bytes, tick)], CTX);
      }
      this.w = w;
      const t1 = performance.now();
      const bytesAt = cached || !tBytes ? t0 : tBytes;
      // Keep the first load's numbers: a recovery reload is not a first visit.
      this.stats ??= {
        cached,
        downloadMs: bytesAt - t0,
        initMs: t1 - bytesAt,
        totalMs: t1 - t0,
        threads: w.getNumThreads(),
        multithread: w.isMultithread(),
        storage,
      };
      this.set("ready", { progress: 1 });
    } catch (e) {
      this.set("error", { error: e instanceof Error ? e.message : String(e) });
    }
  }

  /** Throw away a dead worker and load the model again (from the OPFS cache). */
  private async recover() {
    const old = this.w;
    this.w = null;
    this.restarts++;
    try { await old?.exit(); } catch { /* already dead */ }
    this.state = "idle";
    await this.load();
    if ((this.state as State) !== "ready") throw new Error(this.error || "model reload failed");
  }

  /**
   * Greedy, stopping at the manifest's stop string, exactly as verified.
   * Calls run one at a time; an aborted call resolves to null. If the worker
   * dies mid-parse, it is reloaded and the parse retried once (recovered: true).
   */
  parse(src: string, signal?: AbortSignal, onText?: (t: string) => void) {
    const once = async () => {
      if (!this.w || signal?.aborted) return null;
      const m = this.manifest;
      const t0 = performance.now();
      let text = "";
      let firstMs = 0, tokens = 0;
      try {
        await this.w.createCompletion({
          prompt: m.prompt.replace("{}", src.trim().toLowerCase()),
          max_tokens: m.cap,
          temperature: 0,
          top_k: 1,
          stop: [m.stop],
          // Every parse starts from an empty cache, like the server evaluation.
          // Reusing the previous prompt's prefix makes llama-server roll back
          // this hybrid model's recurrent state, which crashed the worker at a
          // fixed point in a fixed sequence of orders ("Invalid magic number").
          // llama-server reads cache_prompt on every endpoint; wllama's typings
          // list it only for chat, hence the spread.
          ...NO_PROMPT_CACHE,
          abortSignal: signal,
          stream: true,
          onData: (chunk) => {
            if (!tokens++) firstMs = performance.now() - t0;
            text += chunk.choices[0]?.text ?? "";
            onText?.(text.split(m.stop)[0]!.trim());
          },
        });
      } catch (e) {
        if (signal?.aborted) return null;
        throw e;
      }
      if (signal?.aborted) return null;
      this.runs++;
      this.emit();
      // tokens counts streamed chunks; wllama sends one per generated token.
      return { text: text.split(m.stop)[0]!.trim(), ms: performance.now() - t0, firstMs, tokens, recovered: false };
    };
    const run = async () => {
      try {
        return await once();
      } catch (e) {
        if (signal?.aborted) return null;
        console.warn("model worker failed, reloading:", e);
        await this.recover();
        const r = await once();
        return r && { ...r, recovered: true };
      }
    };
    const p = this.queue.then(run, run);
    this.queue = p.catch(() => {});
    return p;
  }
}

async function fetchBlob(url: string, size: number, tick: (n: number, total: number) => void) {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`model download failed: ${r.status}`);
  const total = Number(r.headers.get("content-length")) || size;
  const reader = r.body.getReader();
  const parts: BlobPart[] = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    n += value.length;
    tick(n, total);
  }
  return new Blob(parts);
}

export const engine = new Engine();
