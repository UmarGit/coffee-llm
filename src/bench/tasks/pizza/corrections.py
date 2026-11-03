"""Self-corrections for PIZZA: a later phrase overrides an earlier one.

PIZZA has no self-corrections: its training grammar puts every mentioned item on
the ticket, so "a pizza with cheese, wait no cheese, only tomato sauce" comes out
as `pizza 1 cheese no cheese tomato`. This generator takes real training orders,
edits their EXR tree and appends the spoken correction, so the target is an
ordinary order and every arm encodes it with no new spelling.

    swap      "... with ham, actually make that bacon"            bacon, no ham
    retract   "... with cheese and olives, wait no olives"        olives disappear
              "... wait, no cheese"   (cheese never mentioned)    no cheese
    only      "... ham, peppers and onions, actually only ham"    ham only
    size      "a large ..., no wait a medium"                     medium
    number    "two cokes, make that three"                        3
    drink     "a coke, sorry, a sprite"                           sprite
    cancel    "two pizzas and a coke, never mind the coke"        coke line removed
    restart   "a small pizza, scratch that, a large pepperoni"    only the new order
    decoy     "actually, a large ham pizza", "..., wait, also two cokes"   as said

Retracting an item the customer added removes it: they are undoing their own
words, not asking for it to be taken off a default. Retracting something never
mentioned stays a negation, as in PIZZA. Decoys keep the cue words from becoming
a signal to delete something.

Every edit is chosen so that it is unambiguous: an item named by a correction
appears once in the order, and a correction that names nothing ("make it a
medium") is only used when the order has a single line it can apply to.
A quarter of the cue phrases are held out for evaluation (`eval_rows`).
"""

import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .formats import PIZZA_DIR, parse_exr, to_exr
from .negatives import _split

KINDS = {
    "swap": 0.16, "retract": 0.16, "only": 0.10, "size": 0.10, "number": 0.08,
    "drink": 0.08, "cancel": 0.10, "restart": 0.08, "decoy": 0.14,
}

CUES = [
    "actually", "wait", "sorry", "no wait", "i mean", "oh wait", "hmm actually",
    "on second thought", "um", "oops", "hold on", "actually no", "no sorry",
    "wait a second", "let me change that", "my bad", "uh actually", "oh no",
    "come to think of it", "correction", "wait wait", "sorry i meant", "hang on",
    "um no", "on second thoughts", "oh and actually", "erm", "no hold on",
]
RESTART = [
    "scratch that", "start over", "forget all that", "let me start again",
    "never mind all that", "forget that", "cancel all that", "let me start over",
    "redo", "do over", "wipe that", "ignore that", "cancel everything",
    "start from scratch", "throw that out", "change of plans",
]

SWAP = ["{cue} {new} instead of {old}", "{cue} not {old}, {new}", "{cue} swap the {old} for {new}",
        "{cue} replace the {old} with {new}", "{cue} change the {old} to {new}",
        "{cue} {new} not {old}"]
SWAP_BARE = ["{cue} make that {new}", "{cue} {new}", "{cue} change that to {new}",
             "{cue} make it {new}"]
RETRACT = ["{cue} no {old}", "{cue} no {old} after all", "{cue} without the {old}",
           "{cue} skip the {old}", "{cue} take off the {old}", "{cue} leave out the {old}",
           "{cue} forget the {old}", "{cue} remove the {old}", "{cue} not {old}"]
NEGATE = ["{cue} no {new}", "{cue} no {new} please", "{cue} without {new}",
          "{cue} and no {new}"]
ONLY = ["{cue} only {new}", "{cue} just {new}", "{cue} only {new} on it",
        "{cue} just {new} nothing else", "{cue} make it just {new}", "{cue} only {new} please"]
ONLY_OLD = ["{cue} no {old}, only {new}", "{cue} no {old} just {new}",
            "{cue} not {old}, only {new}"]
SIZE = ["{cue} {new} not {old}", "{cue} {new} instead of {old}", "{cue} change the {old} to {new}"]
SIZE_BARE = ["{cue} make it {new}", "{cue} make that a {new}", "{cue} a {new} one", "{cue} {new}",
             "{cue} make it a {new}"]
NUMBER = ["{cue} make that {n} {thing}", "{cue} {n} {thing} not {old}",
          "{cue} {n} {thing} instead of {old}"]
NUMBER_BARE = ["{cue} make it {n}", "{cue} make that {n}", "{cue} {n} of those"]
DRINK = ["{cue} {new} instead of {old}", "{cue} not {old}, {new}", "{cue} change the {old} to {new}",
         "{cue} swap the {old} for {new}"]
DRINK_BARE = ["{cue} make that {new}", "{cue} {new}", "{cue} make it {new}"]
CANCEL = ["{cue} never mind the {thing}", "{cue} forget the {thing}", "{cue} cancel the {thing}",
          "{cue} drop the {thing}", "{cue} remove the {thing}", "{cue} skip the {thing}"]
ADD = ["{cue} also {b}", "{cue} and {b}", "{cue} and also {b}", "{cue} i forgot {b}",
       "{cue} one more thing {b}"]
TEMPLATES = (SWAP, SWAP_BARE, RETRACT, NEGATE, ONLY, ONLY_OLD, SIZE, SIZE_BARE, NUMBER,
             NUMBER_BARE, DRINK, DRINK_BARE, CANCEL, ADD)


SPLIT = {"cue": _split(CUES), "restart": _split(RESTART)}


@lru_cache(maxsize=None)
def cue_words():
    """Every word a correction can add that a plain PIZZA order never uses. The
    negatives generator must never insert these into an order to teach a refusal,
    or the two generators would teach opposite answers for the same words."""
    from .negatives import pizza_vocab

    words = {w.strip(",") for c in CUES + RESTART for w in c.split()}
    for group in TEMPLATES:
        for t in group:
            words |= {w.strip(",") for w in t.split() if "{" not in w}
    return frozenset(words - pizza_vocab() - {""})


# ---------- surface forms, from the dataset's own catalogs ----------

_FILES = {"topping.txt": "TOPPING", "style.txt": "STYLE", "drinks.txt": "DRINKTYPE",
          "size.txt": "SIZE", "number.txt": "NUMBER"}


@lru_cache(maxsize=None)
def _phrases(pizza_dir=None):
    root = Path(pizza_dir or PIZZA_DIR) / "utils/catalogs"
    out = {}
    for name, ctor in _FILES.items():
        for line in (root / name).read_text().splitlines():
            phrase, _, ref = line.partition("\t")
            phrase = phrase.strip().lower()
            if not ref or "-" in phrase.split():
                continue
            entity = ref[ref.index("(") + 1: ref.rindex(")")].strip()
            out.setdefault((ctor, entity), []).append(phrase)
    return out


def _say(ctor, entity, rng, plural=None):
    forms = _phrases()[(ctor, entity)]
    if ctor == "DRINKTYPE" and plural is not None:
        pick = [f for f in forms if f.endswith("s") == plural]
        forms = pick or forms
    return rng.choice(forms)


def _number(n, rng):
    forms = [f for f in _phrases().get(("NUMBER", str(n)), [str(n)])
             if f not in ("a", "an", "just one", "only one")]
    return rng.choice(forms or [str(n)])


# ---------- the tree ----------


def _copy(node):
    return ("ORDER", [(c, [_attr_copy(a) for a in attrs]) for c, attrs in node[1]])


def _attr_copy(a):
    ctor, kids = a
    return (ctor, [k if isinstance(k, str) else _attr_copy(k) for k in kids])


def _topping(a):
    """The topping a positive TOPPING or COMPLEX_TOPPING attribute names, else None."""
    if a[0] == "TOPPING":
        return a[1][0]
    if a[0] == "COMPLEX_TOPPING":
        return next(k[1][0] for k in a[1] if k[0] == "TOPPING")
    return None


def _value(sub, ctor):
    return next((a[1][0] for a in sub[1] if a[0] == ctor), None)


def facts(node):
    """What a ticket says, flattened per line kind: "p TOPPING HAM",
    "p NOT TOPPING HAM", "d DRINKTYPE COKE", "p" (a pizza line exists)."""
    out = set()

    def attr(prefix, a):
        ctor, kids = a
        if ctor == "NOT":
            for f in attr("", kids[0]):
                out.add(f"{prefix}NOT {f}")
            return []
        if ctor == "COMPLEX_TOPPING":
            q = next(k[1][0] for k in kids if k[0] == "QUANTITY")
            t = _topping(a)
            fs = [f"TOPPING {t}", f"{q} {t}"]
        else:
            fs = [f"{ctor} {' '.join(kids)}"]
        if prefix:
            out.update(prefix + f for f in fs)
        return fs

    for sub, attrs in node[1]:
        p = "p" if sub == "PIZZAORDER" else "d"
        out.add(p)
        for a in attrs:
            attr(p + " ", a)
    return out


def _mentions(node):
    """How often each topping, size and drink is named anywhere, negated or not."""
    c = Counter()
    for _, attrs in node[1]:
        for a in attrs:
            inner = a[1][0] if a[0] == "NOT" else a
            t = _topping(inner)
            if t:
                c["TOPPING", t] += 1
            elif inner[0] in ("SIZE", "DRINKTYPE", "STYLE"):
                c[inner[0], inner[1][0]] += 1
    return c


def _lines(node, ctor):
    return [i for i, (c, _) in enumerate(node[1]) if c == ctor]


@lru_cache(maxsize=None)
def _menu(ctor):
    return sorted(e for (c, e) in _phrases() if c == ctor)


def _fresh(ctor, node, rng):
    """A menu entity the order does not mention at all."""
    seen = _mentions(node)
    return rng.choice([e for e in _menu(ctor) if not seen[ctor, e]])


# ---------- edits: each returns (tree, spoken correction, must, must_not) or None ----------


def _swap(node, rng, cue):
    seen = _mentions(node)
    cands = [(i, j) for i in _lines(node, "PIZZAORDER") for j, a in enumerate(node[1][i][1])
             if a[0] == "TOPPING" and seen["TOPPING", a[1][0]] == 1]
    if not cands:
        return None
    i, j = rng.choice(cands)
    old = node[1][i][1][j][1][0]
    new = _fresh("TOPPING", node, rng)
    alone = sum(seen[k] for k in seen if k[0] == "TOPPING") == 1
    tpl = rng.choice(SWAP + (SWAP_BARE if alone else []))
    out = _copy(node)
    out[1][i][1][j] = ("TOPPING", [new])
    said = tpl.format(cue=cue, old=_say("TOPPING", old, rng), new=_say("TOPPING", new, rng))
    return out, said, [f"p TOPPING {new}"], [f"p TOPPING {old}", f"p NOT TOPPING {old}"]


def _retract(node, rng, cue):
    seen = _mentions(node)
    pizzas = _lines(node, "PIZZAORDER")
    cands = [(i, j) for i in pizzas for j, a in enumerate(node[1][i][1])
             if _topping(a) and seen["TOPPING", _topping(a)] == 1]
    if len(pizzas) == 1 and (not cands or rng.random() < 0.3):
        # Never mentioned, so it stays a negation of the default.
        new = _fresh("TOPPING", node, rng)
        out = _copy(node)
        out[1][pizzas[0]][1].append(("NOT", [("TOPPING", [new])]))
        said = rng.choice(NEGATE).format(cue=cue, new=_say("TOPPING", new, rng))
        return out, said, [f"p NOT TOPPING {new}"], [f"p TOPPING {new}"]
    if not cands:
        return None
    i, j = rng.choice(cands)
    if len(node[1][i][1]) == 1:
        return None
    old = _topping(node[1][i][1][j])
    out = _copy(node)
    del out[1][i][1][j]
    said = rng.choice(RETRACT).format(cue=cue, old=_say("TOPPING", old, rng))
    return out, said, [], [f"p TOPPING {old}", f"p NOT TOPPING {old}"]


def _only(node, rng, cue):
    pizzas = _lines(node, "PIZZAORDER")
    if len(pizzas) != 1:
        return None
    i = pizzas[0]
    attrs = node[1][i][1]
    if any(a[0] == "NOT" for a in attrs):
        return None
    tops = [j for j, a in enumerate(attrs) if _topping(a)]
    if not tops:
        return None
    plain = [j for j in tops if attrs[j][0] == "TOPPING"]
    keep = None
    if len(tops) >= 2 and plain and rng.random() < 0.6:
        keep = attrs[rng.choice(plain)]
        new = keep[1][0]
        tpl = rng.choice(ONLY)
        old = None
    else:
        new = _fresh("TOPPING", node, rng)
        keep = ("TOPPING", [new])
        old = _topping(attrs[tops[0]]) if len(tops) == 1 else None
        tpl = rng.choice(ONLY + (ONLY_OLD if old else []))
    gone = {_topping(attrs[j]) for j in tops} - {new}
    out = _copy(node)
    rest = [a for a in out[1][i][1] if not _topping(a)]
    rest.insert(min(tops[0], len(rest)), keep)
    out[1][i] = (out[1][i][0], rest)
    said = tpl.format(cue=cue, new=_say("TOPPING", new, rng),
                      old=_say("TOPPING", old, rng) if old else "")
    return out, said, [f"p TOPPING {new}"], sorted(f"p TOPPING {g}" for g in gone)


def _size(node, rng, cue):
    seen = _mentions(node)
    pizzas = _lines(node, "PIZZAORDER")
    cands = [i for i in pizzas if _value(node[1][i], "SIZE")]
    if not cands:
        return None
    i = rng.choice(cands)
    old = _value(node[1][i], "SIZE")
    if seen["SIZE", old] != 1:
        return None
    new = _fresh("SIZE", node, rng)
    tpl = rng.choice(SIZE + (SIZE_BARE if len(node[1]) == 1 else []))
    out = _copy(node)
    out[1][i] = (out[1][i][0], [("SIZE", [new]) if a[0] == "SIZE" else a for a in out[1][i][1]])
    said = tpl.format(cue=cue, old=_say("SIZE", old, rng), new=_say("SIZE", new, rng))
    return out, said, [f"p SIZE {new}"], [f"p SIZE {old}"]


def _thing(sub, n, rng):
    if sub[0] == "PIZZAORDER":
        return rng.choice(["pizzas", "pies"] if n > 1 else ["pizza", "pie"])
    return _say("DRINKTYPE", _value(sub, "DRINKTYPE"), rng, plural=n > 1)


def _renumber(node, rng, cue):
    seen = _mentions(node)
    cands = []
    for i, sub in enumerate(node[1]):
        if not _value(sub, "NUMBER"):
            continue
        same = _lines(node, sub[0])
        if sub[0] == "PIZZAORDER" and len(same) == 1:
            cands.append(i)
        elif sub[0] == "DRINKORDER" and _value(sub, "DRINKTYPE") and \
                seen["DRINKTYPE", _value(sub, "DRINKTYPE")] == 1:
            cands.append(i)
    if not cands:
        return None
    i = rng.choice(cands)
    sub = node[1][i]
    old = _value(sub, "NUMBER")
    new = rng.choice([n for n in range(1, 6) if str(n) != old])
    tpl = rng.choice(NUMBER + (NUMBER_BARE if len(node[1]) == 1 else []))
    out = _copy(node)
    out[1][i] = (sub[0], [("NUMBER", [str(new)]) if a[0] == "NUMBER" else a for a in out[1][i][1]])
    said = tpl.format(cue=cue, n=_number(new, rng), old=_number(int(old), rng)
                      if old.isdigit() else old, thing=_thing(sub, new, rng))
    p = "p" if sub[0] == "PIZZAORDER" else "d"
    return out, said, [f"{p} NUMBER {new}"], []


def _drink(node, rng, cue):
    seen = _mentions(node)
    cands = [i for i in _lines(node, "DRINKORDER") if _value(node[1][i], "DRINKTYPE")
             and seen["DRINKTYPE", _value(node[1][i], "DRINKTYPE")] == 1]
    if not cands:
        return None
    i = rng.choice(cands)
    sub = node[1][i]
    old = _value(sub, "DRINKTYPE")
    new = _fresh("DRINKTYPE", node, rng)
    n = _value(sub, "NUMBER")
    plural = bool(n and n.isdigit() and int(n) > 1)
    tpl = rng.choice(DRINK + (DRINK_BARE if len(node[1]) == 1 else []))
    new_said = _say("DRINKTYPE", new, rng, plural=plural)
    if not plural and "{new}" in tpl and rng.random() < 0.5:
        new_said = "a " + new_said
    out = _copy(node)
    out[1][i] = (sub[0], [("DRINKTYPE", [new]) if a[0] == "DRINKTYPE" else a for a in sub[1]])
    said = tpl.format(cue=cue, old=_say("DRINKTYPE", old, rng, plural=plural), new=new_said)
    return out, said, [f"d DRINKTYPE {new}"], [f"d DRINKTYPE {old}"]


def _cancel(node, rng, cue):
    if len(node[1]) < 2:
        return None
    seen = _mentions(node)
    pizzas = _lines(node, "PIZZAORDER")
    cands = [i for i in _lines(node, "DRINKORDER") if _value(node[1][i], "DRINKTYPE")
             and seen["DRINKTYPE", _value(node[1][i], "DRINKTYPE")] == 1]
    if len(pizzas) == 1:
        cands.append(pizzas[0])
    if not cands:
        return None
    i = rng.choice(cands)
    sub = node[1][i]
    n = _value(sub, "NUMBER")
    thing = _thing(sub, int(n) if n and n.isdigit() else 1, rng)
    out = ("ORDER", [s for k, s in enumerate(_copy(node)[1]) if k != i])
    said = rng.choice(CANCEL).format(cue=cue, thing=thing)
    if sub[0] == "PIZZAORDER":
        return out, said, [], ["p"]
    return out, said, [], [f"d DRINKTYPE {_value(sub, 'DRINKTYPE')}"]


def _restart(node, rng, cue, other):
    new = parse_exr(other["exr"])
    if to_exr(new) == to_exr(node):
        return None
    before, after = facts(node), facts(new)
    return new, f"{cue} {other['src']}", sorted(after - before), sorted(before - after)


def _decoy(node, rng, cue, other):
    """A cue that changes nothing: it leads the order, or adds to it."""
    if rng.random() < 0.5:
        return node, None, sorted(facts(node)), []
    new = parse_exr(other["exr"])
    if len(new[1]) != 1 or len(node[1]) > 2:
        return node, None, sorted(facts(node)), []
    said = rng.choice(ADD).format(cue=cue, b=other["src"])
    out = ("ORDER", _copy(node)[1] + _copy(new)[1])
    return out, said, sorted(facts(out)), []


EDITS = {"swap": _swap, "retract": _retract, "only": _only, "size": _size,
         "number": _renumber, "drink": _drink, "cancel": _cancel}


def cue(kind, rng, held=False):
    return rng.choice(SPLIT["restart" if kind == "restart" else "cue"][1 if held else 0])


def edit(node, kind, rng, cue, other=None):
    """(new tree, spoken correction, must, must_not), or None when this order
    cannot take this kind of correction unambiguously. A decoy that leads the
    order returns None for the spoken part: the cue goes in front instead."""
    if kind == "restart":
        return _restart(node, rng, cue, other)
    if kind == "decoy":
        return _decoy(node, rng, cue, other)
    return EDITS[kind](node, rng, cue)


def _join(src, said, rng):
    said = " ".join(said.split())
    return src.rstrip() + rng.choice([", ", " ", ", ", " , "]) + said


def _row(kind, rng, rows, held):
    for _ in range(200):
        base = rng.choice(rows)
        other = rng.choice(rows)
        node = parse_exr(base["exr"])
        said_cue = cue(kind, rng, held)
        if kind == "cancel" and rng.random() < 0.3:
            said_cue = ""
        got = edit(node, kind, rng, said_cue, other)
        if got is None:
            continue
        tree, said, must, must_not = got
        if said is None:
            text = said_cue + rng.choice([", ", " "]) + base["src"]
        else:
            text = _join(base["src"], said, rng)
        return {"src": text, "exr": to_exr(tree), "fix": kind, "cue": said_cue,
                "must": must, "must_not": must_not}
    raise RuntimeError(f"no order in the pool can take a {kind} correction")


def _kinds(n, rng):
    names, weights = zip(*KINDS.items())
    return rng.choices(names, weights, k=n)


def mix(rows, frac, seed):
    """`rows` plus corrections numbering `frac` of the real orders among them,
    built from those orders with training cues only. Deterministic."""
    if not 0 < frac < 1:
        raise ValueError(f"corrections fraction must be in (0, 1), got {frac}")
    rng = random.Random(40_000 + seed)
    real = [r for r in rows if not r.get("neg")]
    out = list(rows)
    out += [_row(k, rng, real, held=False) for k in _kinds(round(len(real) * frac), rng)]
    rng.shuffle(out)
    return out


def dev_rows(dev, n=120, seed=0):
    """For model selection during training: training cues, dev orders."""
    rng = random.Random(45_000 + seed)
    return [_row(k, rng, dev, held=False) for k in _kinds(n, rng)]


def eval_rows(test_rows, n=600, seed=0):
    """Held-out cues never trained on, applied to test orders."""
    rng = random.Random(50_000 + seed)
    return [_row(k, rng, test_rows, held=True) for k in _kinds(n, rng)]


# Hand-written, in the words the site receives (lowercased). Expected tickets are
# in the words format. The first is the failure reported from the live demo.
PROBES = [
    ("i want a pizza with cheese, wait no cheese, only tomato sauce", "pizza 1 tomato_sauce", "only"),
    ("i want a pizza with only tomato sauce", "pizza 1 tomato_sauce", "control"),
    ("a pizza with ham, actually make that bacon", "pizza 1 bacon", "swap"),
    ("a large pizza with mushrooms, sorry, i mean onions", "pizza 1 large onions", "swap"),
    ("a medium pizza with pepperoni and olives, actually sausage instead of olives",
     "pizza 1 medium pepperoni sausage", "swap"),
    ("i'd like a pizza with chicken and spinach, wait, swap the chicken for tuna",
     "pizza 1 tuna spinach", "swap"),
    ("a pizza with cheese and olives, wait no olives", "pizza 1 cheese", "retract"),
    ("a large pizza with ham and pineapple, actually no pineapple", "pizza 1 large ham", "retract"),
    ("two pizzas with extra cheese and peppers, oh wait, skip the peppers",
     "pizza 2 extra cheese", "retract"),
    ("a small pizza with bacon, wait, no onions", "pizza 1 small bacon no onions", "retract"),
    ("a pizza with tuna and mushrooms, sorry, no mushrooms after all", "pizza 1 tuna", "retract"),
    ("a pizza with ham, peppers and onions, actually only ham", "pizza 1 ham", "only"),
    ("a medium pizza with sausage, bacon and chicken, wait just chicken",
     "pizza 1 medium chicken", "only"),
    ("a large pizza with pepperoni, actually just cheese", "pizza 1 large cheese", "only"),
    ("a large pizza with ham, no wait a medium", "pizza 1 medium ham", "size"),
    ("a small pizza with mushrooms, actually make it large", "pizza 1 large mushrooms", "size"),
    ("i want a party size pizza with chicken, sorry, a personal size",
     "pizza 1 personal_size chicken", "size"),
    ("a medium vegetarian pizza, wait, make that a small", "pizza 1 small style vegetarian", "size"),
    ("two cokes, make that three", "drink 3 coke", "number"),
    ("one pizza with ham, actually two of those", "pizza 2 ham", "number"),
    ("three large pizzas with bacon, sorry, make that two pizzas", "pizza 2 large bacon", "number"),
    ("four sprites, wait, two sprites", "drink 2 sprite", "number"),
    ("a coke, sorry, a sprite", "drink 1 sprite", "drink"),
    ("two pepsis, actually make that fantas", "drink 2 fanta", "drink"),
    ("a large pizza with ham and a coke, wait, a diet coke instead of the coke",
     "pizza 1 large ham ; drink 1 diet_coke", "drink"),
    ("a ginger ale, i mean an iced tea", "drink 1 ice_tea", "drink"),
    ("two pizzas with ham and a coke, never mind the coke", "pizza 2 ham", "cancel"),
    ("a large pizza with bacon and two sprites, actually forget the sprites",
     "pizza 1 large bacon", "cancel"),
    ("a medium pizza with tuna and a pepsi, wait, cancel the pizza", "drink 1 pepsi", "cancel"),
    ("a small pizza, scratch that, a large pepperoni pizza", "pizza 1 large pepperoni", "restart"),
    ("two pizzas with ham, let me start over, one pizza with chicken", "pizza 1 chicken", "restart"),
    ("a coke and a sprite, forget all that, three pepsis", "drink 3 pepsi", "restart"),
    ("a large pizza with olives, never mind all that, a medium pizza with tuna",
     "pizza 1 medium tuna", "restart"),
    ("actually, a large ham pizza", "pizza 1 large ham", "decoy"),
    ("wait, i want a medium pizza with bacon", "pizza 1 medium bacon", "decoy"),
    ("a pizza with chicken, wait, i also want a coke", "pizza 1 chicken ; drink 1 coke", "decoy"),
    ("a small pizza with mushrooms, oh and actually also two sprites",
     "pizza 1 small mushrooms ; drink 2 sprite", "decoy"),
    ("sorry, can i get two large pizzas with pepperoni", "pizza 2 large pepperoni", "decoy"),
    ("a pizza with ham, actually, with extra ham", "pizza 1 extra ham", "swap"),
    ("a large pizza with no onions, wait, actually onions are fine, with onions",
     "pizza 1 large onions", "swap"),
]


def probe_rows():
    from .ladder import from_words

    return [{"src": src, "exr": to_exr(from_words(ticket)), "fix": kind, "must": [], "must_not": []}
            for src, ticket, kind in PROBES]


# ---------- scoring ----------


def applied(pred_exr, row):
    """The correction itself landed, even if something else on the ticket is wrong."""
    if pred_exr is None:
        return False
    got = facts(parse_exr(pred_exr))
    return all(f in got for f in row["must"]) and not any(f in got for f in row["must_not"])


def score(outputs, rows, arm):
    """Exact match overall and per kind, plus how often the correction itself was
    applied, and how often the model refused instead of answering."""
    from . import metrics
    from .negatives import is_reject

    _, preds, em = metrics.evaluate(outputs, [r["exr"] for r in rows], arm)
    by, app = {}, {}
    for r, p, e in zip(rows, preds, em):
        by.setdefault(r["fix"], []).append(e)
        app.setdefault(r["fix"], []).append(applied(p, r))
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "em": sum(em) / n,
        "applied": sum(sum(v) for v in app.values()) / n,
        "parse_validity": sum(p is not None for p in preds) / n,
        "refused": sum(is_reject(o, arm) for o in outputs) / n,
        **{f"em_{k}": sum(v) / len(v) for k, v in sorted(by.items())},
        **{f"applied_{k}": sum(v) / len(v) for k, v in sorted(app.items())},
    }, preds, em


def outcome(output, pred_exr, em, arm):
    """correct, silent (a wrong ticket), refused, or broken (does not parse)."""
    from .negatives import is_reject

    if em:
        return "correct"
    if is_reject(output, arm):
        return "refused"
    return "silent" if pred_exr is not None else "broken"


def outcomes(outputs, preds, em, arm):
    c = Counter(outcome(o, p, e, arm) for o, p, e in zip(outputs, preds, em))
    return {k: c[k] for k in ("correct", "silent", "refused", "broken")}
