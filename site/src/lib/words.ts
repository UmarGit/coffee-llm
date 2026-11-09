/**
 * WORDS ticket -> EXR string, a port of from_words + to_exr + parse_exr in
 * src/bench/tasks/pizza/{ladder,formats}.py. The browser needs it to score the
 * live model with the benchmark's matcher (src/lib/matcher.ts), which compares
 * EXR trees.
 *
 * Like the matcher, this port is trusted only because it is checked:
 * www2027_demo/bench/check_words_decoder.mjs feeds every saved WORDS output
 * through it and compares with the `parsed` field the Python side wrote.
 * Returns null exactly where the Python side returns None (a detectable error).
 */

export interface Catalog { by_word: Record<string, string> }

type Tree = string | [string, Tree[]];

const QUANTITY: Record<string, string> = { extra: "EXTRA", light: "LIGHT" };
const UNITS = new Set(["ml", "oz", "floz", "liter"]);

function attr(toks: string[], i: number, cat: Catalog): [Tree, number] {
  const t = toks[i];
  if (t === undefined) throw new Error("attribute past end");
  const next = () => {
    const n = toks[i + 1];
    if (n === undefined) throw new Error("missing value");
    return n;
  };
  if (t === "no") {
    const [inner, j] = attr(toks, i + 1, cat);
    return [["NOT", [inner]], j];
  }
  if (t in QUANTITY) {
    return [["COMPLEX_TOPPING", [["QUANTITY", [QUANTITY[t]!]], ["TOPPING", [next().toUpperCase()]]]], i + 2];
  }
  if (t === "style") return [["STYLE", [next().toUpperCase()]], i + 2];
  if (/^[0-9]/.test(t) && i + 1 < toks.length && UNITS.has(toks[i + 1]!)) {
    return [["VOLUME", [t, toks[i + 1]!.toUpperCase()]], i + 2];
  }
  if (/^[0-9]+$/.test(t)) return [["NUMBER", [t]], i + 1];
  const ctor = Object.hasOwn(cat.by_word, t) ? cat.by_word[t] : undefined;
  if (!ctor) throw new Error(`unknown entity ${t}`);
  return [[ctor, [t.toUpperCase()]], i + 1];
}

function fromWords(text: string, cat: Catalog): Tree {
  const subs: Tree[] = [];
  for (const part of text.trim().split(";")) {
    const toks = part.trim() ? part.trim().split(/\s+/) : [];
    if (!toks.length || (toks[0] !== "pizza" && toks[0] !== "drink")) throw new Error("bad suborder head");
    const ctor = toks[0] === "pizza" ? "PIZZAORDER" : "DRINKORDER";
    const attrs: Tree[] = [];
    for (let i = 1; i < toks.length; ) {
      const [a, j] = attr(toks, i, cat);
      attrs.push(a);
      i = j;
    }
    subs.push([ctor, attrs]);
  }
  return ["ORDER", subs];
}

function toExr(n: Tree): string {
  if (typeof n === "string") return n;
  return `(${n[0]} ` + n[1].map(toExr).join(" ") + " )";
}

/** parse_exr's checks: balanced, no empty constructor, nothing trailing, ORDER root. */
function validExr(exr: string): boolean {
  const toks = exr.split(/([^a-zA-Z0-9._-])/).filter((t) => t && t.trim() && t !== "," && t !== "\n");
  let i = 0;
  const node = (): string => {
    if (toks[i] !== "(") throw new Error("expected (");
    const ctor = toks[i + 1];
    i += 2;
    let kids = 0;
    for (;;) {
      if (i >= toks.length) throw new Error("unbalanced");
      if (toks[i] === ")") break;
      if (toks[i] === "(") node();
      else i++;
      kids++;
    }
    if (!kids) throw new Error("empty constructor");
    i++;
    return ctor!;
  };
  try {
    const root = node();
    return i === toks.length && root === "ORDER";
  } catch {
    return false;
  }
}

/** The EXR for a WORDS output, or null when it does not decode (same as metrics.to_exr_or_none). */
export function wordsToExr(text: string, cat: Catalog): string | null {
  try {
    const exr = toExr(fromWords(text, cat));
    return validExr(exr) ? exr : null;
  } catch {
    return null;
  }
}
