/**
 * Turns a WORDS-format string ("pizza 1 small pepperoni no olives ; drink 2
 * coke") into ticket lines a person can read, and diffs two tickets.
 * Nothing is dropped: every word in the input lands in exactly one chip.
 */
export interface Line { head: string; chips: string[] }

const SIZES = new Set([
  "small", "medium", "large", "extra_large", "personal_size", "regular",
  "lunch_size", "party_size", "high_rise",
]);
const QUANT = new Set(["extra", "more", "light", "less", "little", "lot_of", "a_lot_of", "a_little"]);
const human = (w: string) => w.replace(/_/g, " ");

export function readTicket(text: string): Line[] {
  return text.split(";").map((part) => part.trim()).filter(Boolean).map((part) => {
    const t = part.split(/\s+/);
    const kind = t[0] === "drink" || t[0] === "pizza" ? t.shift()! : "";
    let qty = "", size = "";
    const chips: string[] = [];
    for (let i = 0; i < t.length; i++) {
      const w = t[i]!;
      if (/^\d+$/.test(w) && !qty) qty = w;
      else if ((SIZES.has(w) || /ounce|liter/.test(w)) && !size) size = human(w);
      else if (w === "style" && t[i + 1]) chips.push(`${human(t[++i]!)} style`);
      else if (w === "no" && t[i + 1]) chips.push(`no ${human(t[++i]!)}`);
      else if (QUANT.has(w) && t[i + 1]) chips.push(`${human(w)} ${human(t[++i]!)}`);
      else chips.push(human(w));
    }
    const head = [qty ? `${qty}×` : "", size, kind].filter(Boolean).join(" ");
    return { head, chips };
  });
}

/** Every chip, with the line head as a chip too, for a multiset diff. */
function bag(lines: Line[]) {
  const m = new Map<string, number>();
  for (const l of lines) for (const c of [l.head, ...l.chips]) m.set(c, (m.get(c) ?? 0) + 1);
  return m;
}

/** Chips the model wrote that the truth lacks, and chips the truth has that the model missed. */
export function diff(model: Line[], truth: Line[]) {
  const a = bag(model), b = bag(truth);
  const extra: string[] = [], missing: string[] = [];
  for (const [k, n] of a) for (let i = (b.get(k) ?? 0); i < n; i++) extra.push(k);
  for (const [k, n] of b) for (let i = (a.get(k) ?? 0); i < n; i++) missing.push(k);
  return { extra, missing };
}
