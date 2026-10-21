"""Lossless converters between the PIZZA EXR representation, a compact DSL and JSON.

All three keep the dataset's original sibling order, so EXR -> X -> EXR reproduces the
input string character for character.

DSL grammar:
    order    := suborder (" ; " suborder)*
    suborder := ("p" | "d") (" " attr)*
    attr     := NUMBER | volume | entity | "^" style | "+" topping | "~" topping | "!" attr
    volume   := digits unit            e.g. 20floz, 500ml, 2liter
    entity   := lowercased EXR symbol, resolved by catalog
"""

import json
import os
import re
from pathlib import Path

# src/bench/tasks/pizza/formats.py -> repo root. Counted from this file's depth,
# so moving the module without updating this is what breaks the data path.
REPO = Path(__file__).resolve().parents[4]
PIZZA_DIR = Path(os.environ.get("PIZZA_DIR", REPO / "data/external/pizza"))

QUANTITY_SIGIL = {"EXTRA": "+", "LIGHT": "~"}
SIGIL_QUANTITY = {v: k for k, v in QUANTITY_SIGIL.items()}
SUBORDER_TAG = {"PIZZAORDER": "p", "DRINKORDER": "d"}
TAG_SUBORDER = {v: k for k, v in SUBORDER_TAG.items()}
BARE = ("TOPPING", "SIZE", "STYLE", "DRINKTYPE", "CONTAINERTYPE")

_TOKEN = re.compile("([^a-zA-Z0-9._-])")
_CATALOG_ENTRY = re.compile(r"\w+\(([^)]*)\)")
_VOLUME = re.compile(r"^(\d+(?:\.\d+)?)(ml|oz|floz|liter)$")

_CATALOG_FILES = {
    "topping.txt": "TOPPING",
    "style.txt": "STYLE",
    "drinks.txt": "DRINKTYPE",
    "size.txt": "SIZE",
    "container.txt": "CONTAINERTYPE",
    "quant_qualifier.txt": "QUANTITY",
    "number.txt": "NUMBER",
    "drink_volume.txt": "VOLUME",
}


class Catalog:
    """Entity ids per constructor, read from the dataset's own catalogs/."""

    def __init__(self, pizza_dir=None):
        root = Path(pizza_dir or PIZZA_DIR) / "utils/catalogs"
        if not root.is_dir():
            raise FileNotFoundError(
                f"PIZZA catalogs not found at {root}. Clone the dataset into "
                f"{PIZZA_DIR} or set PIZZA_DIR."
            )
        self.ids = {}
        for filename, ctor in _CATALOG_FILES.items():
            found = set()
            for line in (root / filename).read_text().splitlines():
                m = _CATALOG_ENTRY.search(line)
                if m:
                    found.add(m.group(1).strip())
            self.ids[ctor] = found
        self.volumes = {
            tuple(p.strip() for p in v.split(",")) for v in self.ids["VOLUME"]
        }
        self.by_word = {}
        for ctor in BARE:
            for entity in self.ids[ctor]:
                word = entity.lower()
                if word in self.by_word:
                    raise ValueError(
                        f"ambiguous bare word {word!r}: {self.by_word[word]} and {ctor}"
                    )
                self.by_word[word] = ctor

    def constructor_of(self, word):
        try:
            return self.by_word[word]
        except KeyError:
            raise KeyError(f"unknown entity {word!r}") from None


_default_catalog = None


def catalog():
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = Catalog()
    return _default_catalog


# ---------- EXR ----------


def tokenize(exr):
    return [
        t for t in _TOKEN.split(exr) if t and not t.isspace() and t not in (",", "\n")
    ]


def _parse(toks, i):
    if toks[i] != "(":
        raise ValueError(f"expected '(' at token {i}")
    ctor = toks[i + 1]
    i += 2
    kids = []
    while True:
        if i >= len(toks):
            raise ValueError("unbalanced EXR")
        if toks[i] == ")":
            break
        if toks[i] == "(":
            kid, i = _parse(toks, i)
            kids.append(kid)
        else:
            kids.append(toks[i])
            i += 1
    if not kids:
        raise ValueError(f"constructor {ctor} has no children")
    return (ctor, kids), i + 1


def parse_exr(exr):
    toks = tokenize(exr)
    node, end = _parse(toks, 0)
    if end != len(toks):
        raise ValueError("trailing tokens after EXR root")
    if node[0] != "ORDER":
        raise ValueError(f"root is {node[0]}, expected ORDER")
    return node


def to_exr(node):
    if isinstance(node, str):
        return node
    ctor, kids = node
    return f"({ctor} " + " ".join(to_exr(k) for k in kids) + " )"


# ---------- DSL ----------


def _attr_to_dsl(node):
    ctor, kids = node
    if ctor == "NUMBER":
        return kids[0]
    if ctor == "VOLUME":
        return (kids[0] + kids[1]).lower()
    if ctor == "STYLE":
        return "^" + kids[0].lower()
    if ctor == "NOT":
        return "!" + _attr_to_dsl(kids[0])
    if ctor == "COMPLEX_TOPPING":
        quantity = next(k[1][0] for k in kids if k[0] == "QUANTITY")
        topping = next(k[1][0] for k in kids if k[0] == "TOPPING")
        return QUANTITY_SIGIL[quantity] + topping.lower()
    return kids[0].lower()


def to_dsl(node):
    return " ; ".join(
        SUBORDER_TAG[sub[0]] + "".join(" " + _attr_to_dsl(a) for a in sub[1])
        for sub in node[1]
    )


def _attr_from_dsl(token, cat):
    if token.startswith("!"):
        return ("NOT", [_attr_from_dsl(token[1:], cat)])
    if token[:1] in SIGIL_QUANTITY:
        return (
            "COMPLEX_TOPPING",
            [
                ("QUANTITY", [SIGIL_QUANTITY[token[0]]]),
                ("TOPPING", [token[1:].upper()]),
            ],
        )
    if token.startswith("^"):
        return ("STYLE", [token[1:].upper()])
    if token.isdigit():
        return ("NUMBER", [token])
    m = _VOLUME.match(token)
    if m:
        return ("VOLUME", [m.group(1), m.group(2).upper()])
    entity = token.upper()
    return (cat.constructor_of(token), [entity])


def from_dsl(dsl, cat=None):
    cat = cat or catalog()
    subs = []
    for part in dsl.strip().split(";"):
        toks = part.split()
        if not toks:
            raise ValueError("empty suborder")
        if toks[0] not in TAG_SUBORDER:
            raise ValueError(f"unknown suborder tag {toks[0]!r}")
        subs.append((TAG_SUBORDER[toks[0]], [_attr_from_dsl(t, cat) for t in toks[1:]]))
    if not subs:
        raise ValueError("empty order")
    return ("ORDER", subs)


# ---------- JSON ----------


def _attr_to_json(node):
    ctor, kids = node
    if ctor == "NOT":
        return {"not": _attr_to_json(kids[0])}
    if ctor == "VOLUME":
        return {"volume": {"amount": kids[0], "unit": kids[1]}}
    if ctor == "COMPLEX_TOPPING":
        quantity = next(k[1][0] for k in kids if k[0] == "QUANTITY")
        topping = next(k[1][0] for k in kids if k[0] == "TOPPING")
        return {"topping": topping, "quantity": quantity}
    return {ctor.lower(): kids[0]}


def to_json(node):
    return json.dumps(
        {
            "orders": [
                {
                    "type": SUBORDER_TAG[sub[0]],
                    "attrs": [_attr_to_json(a) for a in sub[1]],
                }
                for sub in node[1]
            ]
        },
        separators=(",", ":"),
    )


def _attr_from_json(obj):
    if "not" in obj:
        return ("NOT", [_attr_from_json(obj["not"])])
    if "volume" in obj:
        return ("VOLUME", [obj["volume"]["amount"], obj["volume"]["unit"]])
    if "quantity" in obj:
        return (
            "COMPLEX_TOPPING",
            [("QUANTITY", [obj["quantity"]]), ("TOPPING", [obj["topping"]])],
        )
    ((key, value),) = obj.items()
    return (key.upper(), [value])


def from_json(text):
    data = json.loads(text)
    subs = []
    for sub in data["orders"]:
        subs.append(
            (TAG_SUBORDER[sub["type"]], [_attr_from_json(a) for a in sub["attrs"]])
        )
    if not subs:
        raise ValueError("empty order")
    return ("ORDER", subs)


# ---------- dispatch ----------


def _ladder():
    from . import ladder

    return ladder


ARMS = ("exr", "json", "exr_short", "tagged", "dsl", "words")
LADDER_ARMS = ("exr", "exr_short", "tagged", "dsl")
MATCHED_PAIRS = (("exr", "json"), ("dsl", "words"))

ENCODERS = {
    "exr": to_exr,
    "json": to_json,
    "dsl": to_dsl,
    "exr_short": lambda n: _ladder().to_exr_short(n),
    "tagged": lambda n: _ladder().to_tagged(n),
    "words": lambda n: _ladder().to_words(n),
}
DECODERS = {
    "exr": parse_exr,
    "json": from_json,
    "dsl": from_dsl,
    "exr_short": lambda t: _ladder().from_exr_short(t),
    "tagged": lambda t: _ladder().from_tagged(t),
    "words": lambda t: _ladder().from_words(t),
}


def encode(node, fmt):
    return ENCODERS[fmt](node)


def decode(text, fmt):
    return DECODERS[fmt](text)


def to_exr_from(text, fmt):
    """Decode a model output in `fmt` and re-serialise as EXR, for the official matcher."""
    return to_exr(decode(text, fmt))
