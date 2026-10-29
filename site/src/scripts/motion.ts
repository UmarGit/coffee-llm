/**
 * All motion on the page. No libraries: scroll position drives the sticky
 * scenes, IntersectionObserver drives one-shot reveals, and the pointer drives
 * the variable-font letters.
 */
import { boot as play } from "./play";

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => r.querySelector<T>(s)!;
const $$ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => [...r.querySelectorAll<T>(s)];
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ---------- splitting ---------- */

function splitLetters(root: HTMLElement) {
  let i = 0;
  for (const line of $$("[data-split]", root)) {
    const text = line.textContent ?? "";
    line.textContent = "";
    for (const c of text) {
      const s = document.createElement("span");
      s.className = "ch";
      s.textContent = c === " " ? " " : c;
      s.style.setProperty("--i", String(i++));
      line.append(s);
    }
  }
}

function splitWords(el: HTMLElement) {
  const words = (el.textContent ?? "").trim().split(/\s+/);
  el.textContent = "";
  words.forEach((w, i) => {
    const m = document.createElement("span");
    m.className = "m";
    const s = document.createElement("span");
    s.className = "w";
    s.textContent = w;
    s.style.setProperty("--i", String(i));
    m.append(s);
    el.append(m, " ");
  });
}

/* ---------- text effects ---------- */

const GLYPHS = "abcdefghijklmnopqrstuvwxyz0123456789(){}[]+!=:;_";

/** Scrambles el from its current text into `to`, left to right. */
function scramble(el: HTMLElement, to: string, ms = 650) {
  const from = el.textContent ?? "";
  if (reduced) { el.textContent = to; return; }
  const len = Math.max(from.length, to.length);
  const t0 = performance.now();
  const token = Symbol();
  (el as any)._scr = token;
  const tick = (now: number) => {
    if ((el as any)._scr !== token) return;
    const p = clamp((now - t0) / ms, 0, 1);
    let out = "";
    for (let i = 0; i < len; i++) {
      const settle = i / len;
      if (p >= settle + 0.25) out += to[i] ?? "";
      else if (p >= settle) out += to[i] === " " ? " " : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      else out += from[i] ?? "";
    }
    el.textContent = out;
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = to;
  };
  requestAnimationFrame(tick);
}

function tween(el: HTMLElement, to: number, dec = 0, ms = 900) {
  const from = parseFloat(el.textContent ?? "0") || 0;
  if (reduced) { el.textContent = to.toFixed(dec); return; }
  const t0 = performance.now();
  const token = Symbol();
  (el as any)._tw = token;
  const tick = (now: number) => {
    if ((el as any)._tw !== token) return;
    const p = clamp((now - t0) / ms, 0, 1);
    const e = 1 - Math.pow(1 - p, 4);
    el.textContent = (from + (to - from) * e).toFixed(dec);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function typeOut(el: HTMLElement, text: string) {
  if (reduced) { el.textContent = text; return; }
  el.textContent = "";
  for (const c of text) {
    el.textContent += c;
    await wait(c === " " ? 55 : 28 + Math.random() * 40);
  }
}

/* ---------- kinetic letters ---------- */

function kinetic() {
  const blocks = $$(".kinetic");
  for (const b of blocks) splitLetters(b);

  const play = (b: HTMLElement) => {
    b.classList.add("intro");
    const n = $$(".ch", b).length;
    setTimeout(() => b.classList.remove("intro"), n * 45 + 1200);
  };
  play(blocks[0]!);
  // The closing line assembles on scroll instead (see assemble()).
  once(blocks.slice(1).filter((b) => !b.classList.contains("outro")), play);
  if (reduced) return;

  const wrong = document.querySelector<HTMLElement>(".wrong");
  if (wrong) {
    setInterval(() => {
      wrong.classList.add("glitch");
      setTimeout(() => wrong.classList.remove("glitch"), 300);
    }, 3200);
  }

  // The hero title never sits still: every few seconds the letters roll over
  // again and the strike redraws, as long as the title is on screen.
  const heroTitle = blocks[0]!;
  let heroVisible = true;
  new IntersectionObserver(([e]) => { heroVisible = e!.isIntersecting; }).observe(heroTitle);
  setInterval(() => {
    if (!heroVisible || document.hidden || heroTitle.classList.contains("intro")) return;
    heroTitle.classList.remove("roll");
    void heroTitle.offsetWidth;
    heroTitle.classList.add("roll");
  }, 7000);

  // A slow wave always runs through the letters; letters near the pointer
  // swell on top of it.
  let px = -1e4, py = -1e4;
  addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse") return;
    px = e.clientX; py = e.clientY;
  }, { passive: true });

  const frame = (now: number) => {
    for (const b of blocks) {
      const r = b.getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight || b.classList.contains("intro")) continue;
      for (const ch of $$(".ch", b)) {
        const c = ch.getBoundingClientRect();
        const i = Number(ch.style.getPropertyValue("--i"));
        const wave = 0.5 + 0.5 * Math.sin(now / 900 - i * 0.35);
        const d = Math.hypot(c.left + c.width / 2 - px, c.top + c.height / 2 - py);
        const f = Math.max(wave * 0.8, clamp(1 - d / 420, 0, 1));
        ch.style.setProperty("--w", String(Math.round(320 + 480 * f)));
        ch.style.setProperty("--wd", String(Math.round(75 + 25 * f)));
      }
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/* ---------- reveals ---------- */

function once(els: Element[], fn: (el: HTMLElement) => void, margin = "0px 0px -15% 0px") {
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { fn(e.target as HTMLElement); io.unobserve(e.target); }
  }, { rootMargin: margin, threshold: 0.1 });
  els.forEach((e) => io.observe(e));
}

function reveals() {
  $$("[data-reveal]").forEach(splitWords);
  once($$("[data-reveal], .fade-up, .faceoff"), (el) => el.classList.add("in"));
  once($$("[data-count]"), (el) => {
    const dec = el.dataset.dec ? Number(el.dataset.dec) : 1;
    tween(el, Number(el.dataset.count), dec, 1400);
  });
}

/* ---------- scenes ---------- */

type Handler = (step: number, prev: number) => void;
const handlers: Record<string, Handler> = {};

function scenes() {
  const list = $$(".scene");
  for (const s of list) {
    s.style.setProperty("--steps", s.dataset.steps!);
    $$(".cap", s).forEach(splitWords);
  }
  const update = () => {
    for (const s of list) {
      const r = s.getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight) continue;
      const steps = Number(s.dataset.steps);
      const p = clamp(-r.top / Math.max(1, r.height - innerHeight), 0, 0.9999);
      const step = Math.floor(p * steps);
      const prev = s.dataset.step === undefined ? -1 : Number(s.dataset.step);
      if (step === prev) continue;
      s.dataset.step = String(step);
      for (const c of $$(".cap", s)) {
        const on = Number(c.dataset.at) === step;
        c.classList.toggle("on", on);
        c.classList.toggle("off", !on && c.classList.contains("on"));
      }
      handlers[s.id]?.(step, prev);
    }
  };
  addEventListener("scroll", () => requestAnimationFrame(update), { passive: true });
  addEventListener("resize", update);
  update();
}

/* ---------- 1 · the order ---------- */

function order() {
  const said = $("#order .said");
  let typed = false;
  handlers.order = () => {
    if (typed) return;
    typed = true;
    typeOut(said, said.dataset.type!);
  };
}

/* ---------- 2 · six spellings ---------- */

interface Spell { f: string; name: string; text: string; tokens: string[] }

function spell() {
  const data: Spell[] = JSON.parse($("#spell-data").textContent!);
  const root = $("#spell .spell");
  const name = $(".spell-name", root);
  const count = $(".spell-count b", root);
  const text = $(".spell-text", root);
  const chips = $(".spell-chips", root);
  const pips = $$("#spell .pips li");
  const max = Math.max(...data.map((d) => d.tokens.length));

  const show = (step: number) => {
    const d = data[step]!;
    root.style.setProperty("--fc", `var(--c-${d.f})`);
    root.style.setProperty("--sw", String(Math.round(75 + 25 * (d.tokens.length / max))));
    scramble(name, d.name, 450);
    scramble(text, d.text, 700);
    tween(count, d.tokens.length, 0, 600);
    chips.textContent = "";
    d.tokens.forEach((t, i) => {
      const c = document.createElement("span");
      c.className = "chip" + (t.startsWith("·") ? " sp" : "");
      c.textContent = t.replace(/^·/, "") || "␣";
      c.style.setProperty("--i", String(i));
      chips.append(c);
    });
    pips.forEach((p, i) => p.classList.toggle("on", i === step));
  };
  handlers.spell = (step) => show(step);
}

/* ---------- 3 · ranking ---------- */

function ranks() {
  const box = $("#rank .ranks");
  const rows = $$(".rk", box);
  const buttons = $$<HTMLButtonElement>("#rank .toggle button");
  const sort = (by: "len" | "acc") => {
    box.dataset.sort = by;
    const order = box.dataset[by]!.split(" ");
    for (const r of rows) {
      const n = order.indexOf(r.dataset.f!);
      r.style.setProperty("--r", String(n));
      r.classList.toggle("first", by === "acc" && n === 0);
    }
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sort === by)));
  };
  sort("len");
  buttons.forEach((b) => b.addEventListener("click", () => sort(b.dataset.sort as "len" | "acc")));
  once([box], () => setTimeout(() => sort("acc"), 1100), "0px 0px -35% 0px");
}

/* ---------- 4 · a hundred orders ---------- */

function hundred() {
  const sec = $("#silent");
  const cells = $$(".hundred i", sec);
  // Fixed shuffle, so the same cell always means the same order across steps
  // and switching format only recolours the cells whose fate changes.
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const perm = cells.map((_, i) => i).sort(() => rnd() - 0.5);
  const delay = cells.map(() => Math.round(rnd() * 500));
  const tally = Object.fromEntries($$("[data-k]", sec).map((b) => [b.dataset.k!, b]));

  const paint = (key: "exr" | "words" | null) => {
    if (!key) {
      cells.forEach((c) => c.removeAttribute("data-o"));
      for (const b of Object.values(tally)) tween(b, 0);
      return;
    }
    const [ok, loud, silent] = sec.dataset[key]!.split(" ").map(Number) as [number, number, number];
    perm.forEach((cell, rank) => {
      const o = rank < ok ? "ok" : rank < ok + loud ? "loud" : "silent";
      const c = cells[cell]!;
      c.style.setProperty("--d", String(delay[cell]));
      c.dataset.o = o;
    });
    tween(tally.ok!, ok);
    tween(tally.loud!, loud);
    tween(tally.silent!, silent);
  };
  handlers.silent = (step) => paint(step === 0 ? null : step === 1 ? "exr" : "words");
}

/* ---------- 5 · race ---------- */

function race() {
  const sec = $("#race");
  const data = JSON.parse(sec.dataset.race!) as Record<string, { ms: number; tokens: string[] }>;
  const SLOW = 2;
  let run = 0;

  const go = async () => {
    const id = ++run;
    const lanes = $$(".lane", sec);
    const fastest = Math.min(...Object.values(data).map((d) => d.ms));
    await Promise.all(lanes.map(async (lane) => {
      const d = data[lane.dataset.lane!]!;
      const out = $(".lane-out", lane);
      const ms = $(".lane-ms b", lane);
      lane.classList.remove("done", "win");
      out.textContent = "";
      ms.textContent = "0";
      const per = (d.ms * SLOW) / d.tokens.length;
      const t0 = performance.now();
      const clock = () => {
        if (id !== run) return;
        const t = Math.min(d.ms, (performance.now() - t0) / SLOW);
        ms.textContent = String(Math.round(t));
        if (t < d.ms) requestAnimationFrame(clock);
      };
      requestAnimationFrame(clock);
      for (const [i, t] of d.tokens.entries()) {
        if (id !== run) return;
        const c = document.createElement("span");
        c.className = "chip" + (t.startsWith("·") ? " sp" : "");
        c.textContent = t.replace(/^·/, "") || "␣";
        out.append(c);
        await wait(reduced ? 0 : per);
        if (i === d.tokens.length - 1) {
          lane.classList.add("done");
          if (d.ms === fastest) lane.classList.add("win");
        }
      }
    }));
  };
  once([$(".lanes", sec)], () => go(), "0px 0px -25% 0px");
  $(".replay", sec).addEventListener("click", go);
}

/* ---------- chrome ---------- */

function chrome() {
  const bar = $(".progress i");
  const where = $(".where");
  const track = $(".band .track");
  const labelled = $$("[data-label]");
  let lastY = scrollY, skew = 0, x = 0;

  const frame = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.setProperty("--p", String(clamp(scrollY / Math.max(1, max), 0, 1)));

    const mid = innerHeight / 2;
    const cur = labelled.find((s) => { const r = s.getBoundingClientRect(); return r.top <= mid && r.bottom >= mid; });
    const label = cur?.dataset.label ?? "";
    if (where.textContent !== label) where.textContent = label;

    // The band drifts on its own and is pushed by scroll velocity.
    const v = scrollY - lastY;
    lastY = scrollY;
    skew += (clamp(v * 0.3, -14, 14) - skew) * 0.12;
    x -= 0.6 + Math.abs(v) * 0.4;
    const w = track.scrollWidth / 2;
    if (-x > w) x += w;
    track.style.setProperty("--x", `${x}px`);
    track.style.setProperty("--sk", `${-skew}deg`);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/* ---------- fluid extras ---------- */

/**
 * Wheel scrolling glides instead of jumping. Native scroll stays the source of
 * truth (sticky scenes, the scrollbar, keys and touch all work as usual); only
 * wheel deltas are eased toward their target.
 */
function smooth() {
  if (reduced || !matchMedia("(pointer: fine)").matches) return;
  let target = scrollY, current = scrollY, running = false;
  const max = () => document.documentElement.scrollHeight - innerHeight;
  const step = () => {
    current += (target - current) * 0.12;
    if (Math.abs(target - current) < 0.5) current = target;
    scrollTo(0, current);
    if (current !== target) requestAnimationFrame(step);
    else running = false;
  };
  addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.deltaX && Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    if ((e.target as Element).closest("textarea, [data-native-scroll]")) return;
    e.preventDefault();
    if (!running) current = target = scrollY;
    const px = e.deltaMode === 1 ? e.deltaY * 32 : e.deltaMode === 2 ? e.deltaY * innerHeight : e.deltaY;
    target = clamp(target + px, 0, max());
    if (!running) { running = true; requestAnimationFrame(step); }
  }, { passive: false });
  // Anything else that scrolls (keys, scrollbar, anchor links) resets the glide.
  addEventListener("scroll", () => { if (!running) current = target = scrollY; }, { passive: true });
}

/** The closing line flies in from scattered letters as the reader scrolls to it. */
function assemble() {
  const box = document.querySelector<HTMLElement>(".assemble");
  const title = box?.querySelector<HTMLElement>(".outro");
  if (!box || !title) return;
  let seed = 3;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  for (const ch of $$(".ch", title)) {
    ch.style.setProperty("--sx", `${(rnd() * 60).toFixed(1)}vw`);
    ch.style.setProperty("--sy", `${(rnd() * 70).toFixed(1)}vh`);
    ch.style.setProperty("--sr", `${(rnd() * 160).toFixed(0)}deg`);
  }
  const frame = () => {
    const r = box.getBoundingClientRect();
    if (r.bottom > 0 && r.top < innerHeight) {
      const p = clamp(-r.top / Math.max(1, r.height - innerHeight) * 1.25, 0, 1);
      title.style.setProperty("--p", (1 - Math.pow(1 - p, 3)).toFixed(4));
    }
    requestAnimationFrame(frame);
  };
  if (reduced) title.style.setProperty("--p", "1");
  else requestAnimationFrame(frame);
}

/** Buttons lean toward the pointer when it comes close. */
function magnets() {
  if (reduced || !matchMedia("(pointer: fine)").matches) return;
  addEventListener("pointermove", (e) => {
    for (const b of $$("[data-magnet], .replay, .toggle button")) {
      const r = b.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      const near = Math.hypot(dx, dy) < Math.max(r.width, r.height) * 0.9;
      b.style.transform = near ? `translate(${dx * 0.25}px, ${dy * 0.35}px)` : "";
    }
  }, { passive: true });
}

/** The background warms toward each section's colour as it takes the screen. */
function tints() {
  const secs = $$("[data-tint]");
  const io = new IntersectionObserver((es) => {
    for (const e of es) {
      if (e.isIntersecting) document.body.dataset.tint = (e.target as HTMLElement).dataset.tint!;
    }
  }, { rootMargin: "-45% 0px -45% 0px" });
  secs.forEach((s) => io.observe(s));
  new IntersectionObserver(([e]) => { if (e!.isIntersecting) delete document.body.dataset.tint; },
    { rootMargin: "-45% 0px -45% 0px" }).observe($("#hero"));
  new IntersectionObserver(([e]) => { if (e!.isIntersecting) delete document.body.dataset.tint; },
    { rootMargin: "-45% 0px -45% 0px" }).observe($("#order"));
}

export function boot() {
  kinetic();
  assemble();
  smooth();
  magnets();
  tints();
  play();
  reveals();
  order();
  spell();
  ranks();
  hundred();
  race();
  scenes();
  chrome();
  // The hero sub line reveals with the intro, not on scroll.
  requestAnimationFrame(() => $$(".hero .fade-up").forEach((e) => e.classList.add("in")));
}
