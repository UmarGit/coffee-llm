"""Negative training data for PIZZA: inputs that are not a pizza order.

PIZZA has no way to say "this is not an order", so a model trained on it always
prints a ticket, even for "hello" or "two apples". The café task fixed the same
problem by teaching the model a reject symbol; this does the same here.

Three kinds of negative, as in the café generator:

    non-order     "hello", "where is the restroom"   ->  ?
    off-menu      "can i get two apples"             ->  ?
    mixed         <a real PIZZA order> + "and some fries"  ->  <its ticket> ; ?

`?` is a suborder of its own, so it composes with the ";"-separated formats
(words, dsl, tagged). The bracketed formats have no such slot and are refused
rather than given an invented encoding.

Nothing here touches the official PIZZA metric: dev and test stay pure PIZZA, so
early stopping and every paper number mean what they meant before. The negatives
get their own held-out evaluation (`evaluate`), whose items the model never saw.
"""

import random
import re

from .formats import PIZZA_DIR, decode, encode, parse_exr, to_exr
from .metrics import exact_match

REJECT = "?"
ARMS = ("words", "dsl", "tagged")

FILLERS = ["", "", "", "hi ", "hello ", "um ", "hey ", "excuse me ", "so ", "ok "]
CLOSERS = ["", "", "", " please", " thanks", " thank you"]

NON_ORDERS = [
    "hello", "hi there", "good morning", "hey how are you", "thank you",
    "thanks so much", "have a nice day", "bye", "goodbye", "see you later",
    "where is the restroom", "what's the wifi password", "can i pay by card",
    "do you take cash", "what time do you close", "are you open on sunday",
    "do you deliver", "how long is the wait", "what do you recommend",
    "what's good here", "how much is a large pizza", "what toppings do you have",
    "do you have a menu", "is this gluten free", "can i speak to the manager",
    "i'm still deciding", "give me a minute", "never mind", "nothing for me",
    "i already ordered", "where is my order", "cancel that", "the usual",
    "something tasty", "surprise me", "i'm just looking", "can you repeat that",
    "is the parking free", "test test", "asdf", "hmm", "yes", "no", "okay",
    "what's your name", "are you a robot", "tell me a joke", "what's the weather like",
    "how was your day", "can i get a receipt", "where do i pick up",
]

OFF_MENU = [
    "apples", "oranges", "bananas", "grapes", "a fruit salad", "a burger",
    "french fries", "a hot dog", "tacos", "a burrito", "sushi", "ramen",
    "fried rice", "a salad", "a sandwich", "a sub", "pasta", "spaghetti",
    "lasagna", "soup", "nuggets", "a donut", "a muffin", "a croissant",
    "a bagel", "pancakes", "waffles", "ice cream", "a milkshake", "a cake",
    "cookies", "a brownie", "cheesecake", "a beer", "wine", "whiskey",
    "orange juice", "apple juice", "lemonade", "a smoothie", "hot chocolate",
    "a latte", "a cappuccino", "an espresso", "green tea", "milk", "kombucha",
    "a phone charger", "a newspaper", "cigarettes", "a gift card", "a taxi",
]

ORDER_VERBS = [
    "i want {x}", "can i get {x}", "could i have {x}", "i'd like {x}",
    "give me {x}", "let me get {x}", "i'll have {x}", "{x}", "i need {x}",
    "do you have {x}", "one order of {x}",
]
AND = [" and ", " and also ", " plus ", " with a side of ", " and some "]


# Catalogs naming things a customer can order. number.txt and quant_qualifier.txt
# are left out: they map words like "a" and "extra", which every phrase contains.
ENTITY_CATALOGS = ("topping.txt", "style.txt", "drinks.txt", "container.txt",
                   "size.txt", "drink_volume.txt")


def _surface_forms(pizza_dir=None):
    """Every phrase the PIZZA entity catalogs map to an entity, lowercased."""
    from pathlib import Path

    root = Path(pizza_dir or PIZZA_DIR) / "utils/catalogs"
    forms = set()
    for f in (root / name for name in ENTITY_CATALOGS):
        for line in f.read_text().splitlines():
            phrase = line.split("\t")[0].strip().lower()
            if phrase:
                forms.add(phrase)
    return forms


def collisions(pizza_dir=None):
    """Off-menu items that contain a real menu phrase. Must be empty: an item
    like "iced tea" or "pineapple soda" is ON the PIZZA menu, and teaching the
    model to reject it would teach it to reject real orders."""
    forms = _surface_forms(pizza_dir)
    bad = []
    for item in OFF_MENU:
        for f in forms:
            if re.search(rf"\b{re.escape(f)}\b", item):
                bad.append((item, f))
    return bad


def _split(items):
    """Every fourth item is held out of training, so evaluation measures whether
    rejection generalises to inputs the model never saw."""
    return [x for i, x in enumerate(items) if i % 4 != 3], [x for i, x in enumerate(items) if i % 4 == 3]


NON_TRAIN, NON_EVAL = _split(NON_ORDERS)
OFF_TRAIN, OFF_EVAL = _split(OFF_MENU)


def _non_order(rng, pool):
    return (rng.choice(FILLERS) + rng.choice(pool)).strip()


def _off_menu(rng, pool):
    text = rng.choice(ORDER_VERBS).format(x=rng.choice(pool))
    return (rng.choice(FILLERS) + text + rng.choice(CLOSERS)).strip()


def _mixed(rng, row, pool):
    return row["src"].rstrip() + rng.choice(AND) + rng.choice(pool)


def target(row, arm):
    """The completion for a negative row. Ordinary rows never reach here."""
    if arm not in ARMS:
        raise ValueError(f"negatives need a ';'-separated arm {ARMS}, not {arm!r}")
    if row["neg"] == "mixed":
        return encode(parse_exr(row["exr"]), arm) + " ; " + REJECT
    return REJECT


def mix(rows, frac, seed, arm):
    """`rows` plus enough negatives that they make up `frac` of the result.

    A third each of non-orders, off-menu requests and mixed orders, the mixed
    ones built from training rows so no evaluation utterance is touched.
    Deterministic in (rows, frac, seed).
    """
    if not 0 < frac < 1:
        raise ValueError(f"negatives fraction must be in (0, 1), got {frac}")
    if arm not in ARMS:
        raise ValueError(f"negatives need a ';'-separated arm {ARMS}, not {arm!r}")
    rng = random.Random(20_000 + seed)
    n = round(len(rows) * frac / (1 - frac))
    out = list(rows)
    for i in range(n):
        kind = ("non", "off", "mixed")[i % 3]
        if kind == "non":
            out.append({"src": _non_order(rng, NON_TRAIN), "exr": None, "neg": kind})
        elif kind == "off":
            out.append({"src": _off_menu(rng, OFF_TRAIN), "exr": None, "neg": kind})
        else:
            base = rng.choice(rows)
            out.append({"src": _mixed(rng, base, OFF_TRAIN), "exr": base["exr"], "neg": kind})
    rng.shuffle(out)
    return out


def eval_rows(test_rows, n=300, seed=0):
    """Held-out negatives: unseen phrases only, mixed ones built on test orders."""
    rng = random.Random(30_000 + seed)
    rows = []
    for i in range(n):
        kind = ("non", "off", "mixed")[i % 3]
        if kind == "non":
            rows.append({"src": _non_order(rng, NON_EVAL), "exr": None, "neg": kind})
        elif kind == "off":
            rows.append({"src": _off_menu(rng, OFF_EVAL), "exr": None, "neg": kind})
        else:
            base = rng.choice(test_rows)
            rows.append({"src": _mixed(rng, base, OFF_EVAL), "exr": base["exr"], "neg": kind})
    return rows


def split_reject(text):
    """(ticket text without reject parts, whether any part was the reject symbol)."""
    parts = [p.strip() for p in text.split(";")]
    keep = [p for p in parts if p and p != REJECT]
    return " ; ".join(keep), len(keep) < len([p for p in parts if p])


def correct(output, row, arm):
    """A negative is handled right when the reject flag is set and, for a mixed
    order, the rest of the ticket matches the real order exactly."""
    rest, flagged = split_reject(output)
    if not flagged:
        return False
    if row["neg"] != "mixed":
        return rest == ""
    try:
        return exact_match(to_exr(decode(rest, arm)), row["exr"])
    except Exception:
        return False


def score(outputs, rows, arm, test_outputs=()):
    """Rejection accuracy on held-out negatives, per kind, plus how often the
    model wrongly flags a real test order (the cost side of the trade)."""
    by = {}
    for o, r in zip(outputs, rows):
        by.setdefault(r["neg"], []).append(correct(o, r, arm))
    res = {
        "n": len(rows),
        "handled": sum(sum(v) for v in by.values()) / len(rows),
        **{f"handled_{k}": sum(v) / len(v) for k, v in by.items()},
    }
    if test_outputs:
        res["false_reject_on_test"] = sum(split_reject(o)[1] for o in test_outputs) / len(test_outputs)
    return res
