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
  const local = import.meta.env.PUBLIC_MODEL_URL as string | undefined;
  if (!local) return exported.url;
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");
  return new URL(local, new URL(base, location.origin)).href;
}

type Listener = () => void;

class Engine {
  state: State = "idle";
  progress = 0;
  error = "";
  manifest: Manifest = exported;
  runs = 0;
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
    try {
      const [{ Wllama, LoggerWithoutDebug }, { default: wasm }] = await Promise.all([
        import("@wllama/wllama"),
        import("@wllama/wllama/esm/wasm/wllama.wasm?url"),
      ]);
      const w = new Wllama({ default: wasm }, { logger: LoggerWithoutDebug });
      const tick = (loaded: number, total: number) => {
        this.progress = total ? loaded / total : 0;
        this.emit();
      };
      const url = modelUrl();
      try {
        // Cached, so a second visit loads from disk instead of the network.
        await w.loadModelFromUrl(url, {
          n_ctx: 512,
          progressCallback: ({ loaded, total }: { loaded: number; total: number }) => tick(loaded, total),
        });
      } catch (e) {
        // A full disk or a private window refuses the cache, and wllama's own
        // uncached path still writes to it, so download into memory instead.
        if (!/quota|storage/i.test(String(e))) throw e;
        await w.loadModel([await fetchBlob(url, this.bytes, tick)], { n_ctx: 512 });
      }
      this.w = w;
      this.set("ready", { progress: 1 });
    } catch (e) {
      this.set("error", { error: e instanceof Error ? e.message : String(e) });
    }
  }

  /**
   * Greedy, stopping at the manifest's stop string, exactly as verified.
   * Calls run one at a time; an aborted call resolves to null.
   */
  parse(src: string, signal?: AbortSignal, onText?: (t: string) => void) {
    const run = async () => {
      if (!this.w || signal?.aborted) return null;
      const m = this.manifest;
      const t0 = performance.now();
      let text = "";
      try {
        await this.w.createCompletion({
          prompt: m.prompt.replace("{}", src.trim().toLowerCase()),
          max_tokens: m.cap,
          temperature: 0,
          top_k: 1,
          stop: [m.stop],
          abortSignal: signal,
          stream: true,
          onData: (chunk) => {
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
      return { text: text.split(m.stop)[0]!.trim(), ms: performance.now() - t0 };
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
