/**
 * Everything on the page that reads orders: the hero's live box, the
 * "spot the silent error" game and the "break the real model" mode. Saved
 * outputs drive them until the reader loads the real model, which then runs
 * in this tab through src/lib/model.ts.
 */
import { readTicket, diff, type Line } from "../lib/ticket";
import { engine } from "../lib/model";

interface Round { src: string; model: string; truth: string; o: 0 | 1 | 2 }

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const el = (tag: string, cls = "", text = "") => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
};
const mb = (n: number) => `${Math.round(n / 1e6)} MB`;

/**
 * Fills `box` with ticket lines. Chips listed in `wrong` are struck through;
 * `missing` chips are appended in green, so a reveal shows the fix in place.
 */
function paint(box: Element, lines: Line[], wrong: string[] = [], missing: string[] = []) {
  const left = [...wrong];
  const take = (t: string) => {
    const i = left.indexOf(t);
    if (i < 0) return false;
    left.splice(i, 1);
    return true;
  };
  box.textContent = "";
  lines.forEach((l, n) => {
    const row = el("div", "tl");
    row.style.setProperty("--n", String(n));
    row.append(el("span", "th" + (take(l.head) ? " bad" : ""), l.head));
    l.chips.forEach((c, i) => {
      const chip = el("span", "tc" + (take(c) ? " bad" : ""), c);
      chip.style.setProperty("--i", String(i));
      row.append(chip);
    });
    box.append(row);
  });
  if (missing.length) {
    const row = el("div", "tl fixrow");
    row.append(el("span", "th", "should have"));
    missing.forEach((m) => row.append(el("span", "tc add", m)));
    box.append(row);
  }
}

/** Every load button and progress bar on the page follows the one engine. */
function loaders() {
  const buttons = [...document.querySelectorAll<HTMLButtonElement>(".live-load")];
  const bars = [...document.querySelectorAll<HTMLElement>(".live-bar")];
  buttons.forEach((b) => b.addEventListener("click", () => engine.load()));
  engine.on(() => {
    const s = engine.state;
    document.body.dataset.model = s;
    for (const b of buttons) {
      b.hidden = s === "ready";
      b.disabled = s === "loading";
      b.textContent =
        s === "loading" ? `Loading… ${Math.round(engine.progress * 100)}%`
        : s === "error" ? "Couldn't load it. Try again"
        : `Run the real model here · ${mb(engine.bytes)}`;
    }
    for (const bar of bars) {
      bar.hidden = s !== "loading";
      bar.style.setProperty("--p", String(engine.progress));
    }
  });
}

/* ---------- hero: live box ---------- */

export function liveHero() {
  const root = document.querySelector<HTMLElement>(".live");
  if (!root) return;
  const demo: Round[] = JSON.parse(document.getElementById("hero-data")!.textContent!);
  const input = root.querySelector<HTMLTextAreaElement>(".live-input")!;
  const tix = root.querySelector<HTMLElement>(".live-ticket .tix")!;
  const note = root.querySelector<HTMLElement>(".live-note")!;
  const mode = root.querySelector<HTMLElement>(".live-mode")!;
  const ms = root.querySelector<HTMLElement>(".live-ms")!;
  const tryBox = root.querySelector<HTMLElement>(".live-try")!;
  let live = false;
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e!.isIntersecting; }).observe(root);

  // Until the model loads: type a real order, print the real saved output,
  // and when that output was wrong, say so on the ticket.
  const replay = async () => {
    for (let n = 0; !live; n++) {
      if (!visible || document.hidden) { await wait(500); n--; continue; }
      const r = demo[n % demo.length]!;
      input.value = "";
      paint(tix, []);
      note.textContent = "";
      root.dataset.o = "";
      for (const c of r.src) {
        if (live) return;
        input.value += c;
        await wait(reduced ? 0 : 22 + Math.random() * 30);
      }
      await wait(350);
      if (live) return;
      const model = readTicket(r.model);
      paint(tix, model);
      root.dataset.o = String(r.o);
      if (r.o === 1) note.textContent = "Not a valid ticket, so the system asks again. A loud error.";
      if (r.o === 2) {
        await wait(1400);
        if (live) return;
        const d = diff(model, readTicket(r.truth));
        paint(tix, model, d.extra, d.missing);
        note.textContent = "Valid ticket, wrong order. Nothing flags it. A silent error.";
      }
      await wait(r.o === 0 ? 2200 : 3400);
    }
  };
  replay();

  // With the model: parse when the reader pauses, streaming the ticket in.
  let ctrl: AbortController | null = null;
  let timer = 0;
  const run = async () => {
    const text = input.value.trim();
    ctrl?.abort();
    if (!text) { paint(tix, []); ms.textContent = ""; return; }
    const c = (ctrl = new AbortController());
    root.dataset.busy = "1";
    const res = await engine.parse(text, c.signal, (partial) => paint(tix, readTicket(partial)));
    if (c !== ctrl) return;
    delete root.dataset.busy;
    if (!res) return;
    paint(tix, readTicket(res.text));
    ms.textContent = `${(res.ms / 1000).toFixed(1)} s on your device`;
  };
  input.addEventListener("input", () => {
    if (!live) return;
    clearTimeout(timer);
    timer = window.setTimeout(run, 650);
  });
  input.addEventListener("keydown", (e) => {
    if (live && e.key === "Enter" && !e.shiftKey) { e.preventDefault(); clearTimeout(timer); run(); }
  });
  tryBox.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { input.value = b.textContent!; run(); }));

  engine.on(() => {
    if (engine.state !== "ready" || live) return;
    live = true;
    root.dataset.mode = "live";
    root.dataset.o = "";
    mode.textContent = "live · the real model";
    input.readOnly = false;
    input.value = "";
    input.placeholder = "type a pizza order…";
    note.textContent = "";
    paint(tix, []);
    tryBox.hidden = false;
    input.focus({ preventScroll: true });
  });
}

/* ---------- game: spot the silent error ---------- */

const ROUNDS = 10, SILENT_PER_GAME = 4;

function spot(root: HTMLElement) {
  const pool: { silent: Round[]; correct: Round[] } =
    JSON.parse(document.getElementById("game-data")!.textContent!);
  const pane = root.querySelector<HTMLElement>('[data-pane="spot"]')!;
  const q = pane.querySelector<HTMLElement>(".t-q")!;
  const rcpt = pane.querySelector<HTMLElement>(".rcpt")!;
  const tix = rcpt.querySelector<HTMLElement>(".tix")!;
  const stamp = rcpt.querySelector<HTMLElement>(".rcpt-stamp")!;
  const $round = pane.querySelector<HTMLElement>(".g-round")!;
  const pips = pane.querySelector<HTMLElement>(".g-pips")!;
  const verdict = pane.querySelector<HTMLElement>(".verdict")!;
  const choose = [...pane.querySelectorAll<HTMLButtonElement>("[data-choice]")];
  const next = pane.querySelector<HTMLButtonElement>(".next")!;
  const table = pane.querySelector<HTMLElement>(".table")!;
  pane.querySelector(".g-total")!.textContent = String(ROUNDS);

  const shuffle = <T>(xs: T[]) => xs.map((x) => [Math.random(), x] as const).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
  let hand: Round[] = [];
  let i = 0, score = 0, caught = 0, answered = false;

  const show = () => {
    const r = hand[i]!;
    answered = false;
    $round.textContent = String(i + 1);
    table.classList.remove("in", "good", "bad");
    void table.offsetWidth;
    table.classList.add("in");
    q.textContent = `“${r.src}”`;
    paint(tix, readTicket(r.model));
    stamp.textContent = "accepted";
    stamp.className = "rcpt-stamp ok";
    verdict.textContent = "";
    choose.forEach((b) => { b.hidden = false; b.disabled = false; });
    next.hidden = true;
    [...pips.children].forEach((p, k) => p.classList.toggle("now", k === i));
  };

  const start = () => {
    hand = shuffle([
      ...shuffle(pool.silent).slice(0, SILENT_PER_GAME),
      ...shuffle(pool.correct).slice(0, ROUNDS - SILENT_PER_GAME),
    ]);
    i = score = caught = 0;
    pips.textContent = "";
    for (let k = 0; k < ROUNDS; k++) pips.append(el("li"));
    pane.classList.remove("over");
    show();
  };

  const answer = (choice: "back" | "serve") => {
    if (answered || i >= ROUNDS) return;
    answered = true;
    const r = hand[i]!;
    const wrong = r.o === 2;
    const right = (choice === "back") === wrong;
    if (right) score++;
    if (right && wrong) caught++;
    table.classList.add(right ? "good" : "bad");
    pips.children[i]!.className = right ? "hit" : "miss";

    if (wrong) {
      const d = diff(readTicket(r.model), readTicket(r.truth));
      paint(tix, readTicket(r.model), d.extra, d.missing);
      stamp.textContent = "wrong";
      stamp.className = "rcpt-stamp bad";
    }
    verdict.textContent = wrong
      ? right ? "Caught it. The ticket was valid and still wrong." : "Served. The kitchen made the wrong order."
      : right ? "Right call. That ticket was correct." : "That one was correct. The customer waited for nothing.";
    verdict.className = `verdict ${right ? "good" : "bad"}`;
    choose.forEach((b) => (b.hidden = true));
    next.hidden = false;
    next.textContent = "";
    next.append(i + 1 < ROUNDS ? "Next order " : "See how you did ", el("kbd", "", "↵"));
    next.focus({ preventScroll: true });
  };

  const advance = () => {
    if (!answered) return;
    i++;
    if (i < ROUNDS) show();
    else end();
  };

  const end = () => {
    pane.classList.add("over");
    table.classList.remove("good", "bad");
    q.textContent = `You caught ${caught} of ${SILENT_PER_GAME} wrong tickets.`;
    stamp.textContent = "";
    tix.textContent = "";
    const p = el("p", "end-line",
      `The parser caught none of them: every one was a valid ticket. ${score}/${ROUNDS} calls right overall.`);
    tix.append(p);
    verdict.textContent = "";
    next.hidden = false;
    next.textContent = "Play again ↻";
    [...pips.children].forEach((x) => x.classList.remove("now"));
  };

  choose.forEach((b) => b.addEventListener("click", () => answer(b.dataset.choice as "back" | "serve")));
  next.addEventListener("click", () => (pane.classList.contains("over") ? start() : advance()));
  return {
    start,
    key(e: KeyboardEvent) {
      if (e.key === "ArrowLeft") { e.preventDefault(); answer("back"); }
      else if (e.key === "ArrowRight") { e.preventDefault(); answer("serve"); }
      else if (e.key === "Enter" && answered) { e.preventDefault(); pane.classList.contains("over") ? start() : advance(); }
    },
  };
}

/* ---------- game: break the real model ---------- */

function breakIt(root: HTMLElement) {
  const pane = root.querySelector<HTMLElement>('[data-pane="break"]')!;
  const input = pane.querySelector<HTMLTextAreaElement>(".b-input")!;
  const go = pane.querySelector<HTMLButtonElement>(".b-go")!;
  const tix = pane.querySelector<HTMLElement>(".rcpt .tix")!;
  const empty = pane.querySelector<HTMLElement>(".b-empty")!;
  const ms = pane.querySelector<HTMLElement>(".b-ms")!;
  const judge = pane.querySelector<HTMLElement>(".b-judge")!;
  const tally = pane.querySelector<HTMLElement>(".b-tally")!;
  const meta = pane.querySelector<HTMLElement>(".b-load .live-meta")!;
  let tries = 0, broke = 0, busy = false;

  const sync = () => {
    const ready = engine.state === "ready";
    go.disabled = !ready || busy || !input.value.trim();
    meta.textContent = ready ? "" : "Downloads once, then runs entirely in your browser. Nothing is sent anywhere.";
  };
  engine.on(sync);
  input.addEventListener("input", sync);

  const send = async () => {
    if (go.disabled) return;
    busy = true;
    sync();
    judge.hidden = true;
    empty.hidden = true;
    pane.classList.add("thinking");
    const res = await engine.parse(input.value, undefined, (t) => paint(tix, readTicket(t)));
    pane.classList.remove("thinking");
    busy = false;
    sync();
    if (!res) return;
    paint(tix, readTicket(res.text));
    empty.textContent = "The model wrote nothing it could print.";
    empty.hidden = !!res.text;
    ms.textContent = `${(res.ms / 1000).toFixed(1)} s`;
    judge.hidden = false;
  };
  go.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  judge.querySelectorAll<HTMLButtonElement>("[data-judge]").forEach((b) =>
    b.addEventListener("click", () => {
      tries++;
      if (b.dataset.judge === "wrong") broke++;
      judge.hidden = true;
      tally.textContent = broke
        ? `You broke it ${broke} time${broke > 1 ? "s" : ""} in ${tries}. Each one was a valid ticket the kitchen would have made.`
        : `It held up ${tries} time${tries > 1 ? "s" : ""}. Try negations, two pizzas at once, or drinks with sizes.`;
      input.select();
    }));
}

/* ---------- wiring ---------- */

export function game() {
  const root = document.getElementById("game");
  if (!root) return;
  const s = spot(root);
  breakIt(root);
  const tabs = [...root.querySelectorAll<HTMLButtonElement>("[data-mode]")];
  const panes = [...root.querySelectorAll<HTMLElement>("[data-pane]")];
  let current = "spot";
  tabs.forEach((t) => t.addEventListener("click", () => {
    current = t.dataset.mode!;
    tabs.forEach((x) => x.setAttribute("aria-selected", String(x === t)));
    panes.forEach((p) => (p.hidden = p.dataset.pane !== current));
  }));
  let inView = false;
  new IntersectionObserver(([e]) => { inView = e!.isIntersecting; }, { threshold: 0.3 }).observe(root);
  addEventListener("keydown", (e) => {
    if (!inView || current !== "spot" || (e.target as Element).closest("textarea, input")) return;
    s.key(e);
  });
  s.start();
}

export function boot() {
  loaders();
  liveHero();
  game();
}
