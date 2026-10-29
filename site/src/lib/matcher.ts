/**
 * Port of the PIZZA benchmark's official unordered exact-match, from
 * data/external/pizza/utils/{semantic_matchers,trees,sexp_reader}.py.
 *
 * The Python side never reimplements this matcher, on purpose. The browser
 * cannot call Python, so this port exists instead, and is only trustworthy
 * because tools/check_matcher_parity.py proves it agrees with the official
 * matcher on every saved prediction. Any behaviour below that looks wrong is
 * reproduced deliberately: parity with the reference is the contract, not
 * correctness in the abstract.
 */

const ROOT_SYMBOL = "DUMMY-ROOT";

export interface Node {
  id: string;
  children: Node[];
}

/** re.split('([^a-zA-Z0-9._-])', s), dropping empties, whitespace, ',' and newline. */
export function tokenize(flat: string): string[] {
  const out: string[] = [];
  let buf = "";
  for (const ch of flat) {
    if (/[a-zA-Z0-9._-]/.test(ch)) {
      buf += ch;
      continue;
    }
    if (buf) {
      out.push(buf);
      buf = "";
    }
    if (ch.trim() === "" || ch === ",") continue;
    out.push(ch);
  }
  if (buf) out.push(buf);
  return out;
}

function parentGroups(toks: string[]): Map<number, number> {
  const stack: number[] = [];
  const map = new Map<number, number>();
  for (let i = 0; i < toks.length; i++) {
    if (toks[i] === "(") stack.push(i);
    if (toks[i] === ")") {
      const open = stack.pop();
      if (open === undefined) throw new Error("unbalanced");
      map.set(open, i);
    }
  }
  return map;
}

function parseSexp(
  toks: string[],
  start: number,
  end: number,
  groups: Map<number, number>,
): Node {
  const head = toks[start];
  if (head === undefined) throw new Error("index past end");
  if (head !== "(") return { id: head, children: [] };
  const root = toks[start + 1];
  if (root === undefined) throw new Error("empty group");
  const args: Node[] = [];
  let i = start + 2;
  while (i < end - 1) {
    let j: number;
    if (toks[i] === "(") {
      const close = groups.get(i);
      if (close === undefined) throw new Error("unbalanced");
      j = close + 1;
    } else {
      j = i + 1;
    }
    args.push(parseSexp(toks, i, j, groups));
    i = j;
  }
  return { id: root, children: args };
}

/** The tree for an EXR string, or null when one cannot be built. */
export function parseExr(flat: string): Node | null {
  try {
    const toks = tokenize(`(${ROOT_SYMBOL} ${flat})`);
    if (!toks.length) return null;
    return parseSexp(toks, 0, toks.length, parentGroups(toks));
  } catch {
    return null;
  }
}

/**
 * Greedy first-fit pairing of children, which is what the reference does. It is
 * not a maximum bipartite matching and can miss a pairing that exists; that
 * behaviour is part of the published metric and is kept.
 */
export function treesMatch(a: Node, b: Node): boolean {
  if (a.id !== b.id) return false;
  if (a.children.length !== b.children.length) return false;
  const usedA = new Set<number>();
  const usedB = new Set<number>();
  for (let i = 0; i < a.children.length; i++) {
    for (let j = 0; j < b.children.length; j++) {
      if (usedA.has(i) || usedB.has(j)) continue;
      if (treesMatch(a.children[i]!, b.children[j]!)) {
        usedA.add(i);
        usedB.add(j);
      }
    }
  }
  return usedA.size === a.children.length;
}

/** Unordered exact match between two EXR strings. False if either fails to parse. */
export function isUnorderedExactMatch(exr1: string, exr2: string): boolean {
  const t1 = parseExr(exr1);
  const t2 = parseExr(exr2);
  if (!t1 || !t2) return false;
  return treesMatch(t1, t2);
}
