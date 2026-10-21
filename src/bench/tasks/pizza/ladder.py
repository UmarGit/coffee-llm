"""The three additional encodings that complete the output-length ladder.

Every encoder here is lossless: decoding reproduces the EXR tree byte for byte, including
the dataset's original sibling order. See PROTOCOL.md section 3 for why these exist.

    exr        45.0 tokens   (ORDER (PIZZAORDER (NUMBER 2 ) (SIZE MEDIUM ) ... ) )
    exr_short  33.4          (O (P (N 2) (S MEDIUM) ...))
    tagged     21.9          pizza n=2 size=medium extra=cheese topp=tuna
    dsl        10.8          p 2 medium +cheese tuna
    words      10.3          pizza 2 medium extra cheese tuna      (matched control for dsl)
    json       45.6                                                 (matched control for exr)
"""

import re

from .formats import catalog

CTOR_SHORT = {
    "ORDER": "O",
    "PIZZAORDER": "P",
    "DRINKORDER": "D",
    "NUMBER": "N",
    "SIZE": "S",
    "TOPPING": "T",
    "STYLE": "Y",
    "NOT": "X",
    "COMPLEX_TOPPING": "C",
    "QUANTITY": "Q",
    "DRINKTYPE": "K",
    "CONTAINERTYPE": "R",
    "VOLUME": "V",
}
SHORT_CTOR = {v: k for k, v in CTOR_SHORT.items()}

TAG_OF = {
    "TOPPING": "topp",
    "SIZE": "size",
    "STYLE": "style",
    "DRINKTYPE": "drink",
    "CONTAINERTYPE": "cont",
    "NUMBER": "n",
    "VOLUME": "vol",
}
CTOR_OF_TAG = {v: k for k, v in TAG_OF.items()}
QUANTITY_WORD = {"EXTRA": "extra", "LIGHT": "light"}
WORD_QUANTITY = {v: k for k, v in QUANTITY_WORD.items()}

_VOL_JOINED = re.compile(r"^(\d+(?:\.\d+)?)(ml|oz|floz|liter)$")
_TOK_SHORT = re.compile("([()])")


# ---------- exr_short: same s-expression, one-letter constructors, no padding ----------


def to_exr_short(node):
    if isinstance(node, str):
        return node
    ctor, kids = node
    return f"({CTOR_SHORT[ctor]} " + " ".join(to_exr_short(k) for k in kids) + ")"


def _tokenize_short(text):
    return [t for t in _TOK_SHORT.split(text) if t and not t.isspace()]


def _parse_short(toks, i):
    if toks[i] != "(":
        raise ValueError("expected '('")
    head = toks[i + 1].split()
    ctor = SHORT_CTOR[head[0]]
    kids = list(head[1:])
    i += 2
    while toks[i] != ")":
        if toks[i] == "(":
            kid, i = _parse_short(toks, i)
            kids.append(kid)
        else:
            kids.extend(toks[i].split())
            i += 1
    if not kids:
        raise ValueError(f"{ctor} has no children")
    return (ctor, kids), i + 1


def from_exr_short(text):
    toks = _tokenize_short(text)
    node, end = _parse_short(toks, 0)
    if end != len(toks) or node[0] != "ORDER":
        raise ValueError("malformed exr_short")
    return node


# ---------- tagged: flat key=value attributes ----------


def _tagged_attr(node):
    ctor, kids = node
    if ctor == "NUMBER":
        return "n=" + kids[0]
    if ctor == "VOLUME":
        return "vol=" + (kids[0] + kids[1]).lower()
    if ctor == "NOT":
        return "no:" + _tagged_attr(kids[0])
    if ctor == "COMPLEX_TOPPING":
        q = next(k[1][0] for k in kids if k[0] == "QUANTITY")
        t = next(k[1][0] for k in kids if k[0] == "TOPPING")
        return f"{QUANTITY_WORD[q]}={t.lower()}"
    return f"{TAG_OF[ctor]}=" + kids[0].lower()


def to_tagged(node):
    return " ; ".join(
        ("pizza" if sub[0] == "PIZZAORDER" else "drink")
        + "".join(" " + _tagged_attr(a) for a in sub[1])
        for sub in node[1]
    )


def _tagged_attr_back(token):
    if token.startswith("no:"):
        return ("NOT", [_tagged_attr_back(token[3:])])
    key, _, value = token.partition("=")
    if key in WORD_QUANTITY:
        return (
            "COMPLEX_TOPPING",
            [("QUANTITY", [WORD_QUANTITY[key]]), ("TOPPING", [value.upper()])],
        )
    if key == "vol":
        m = _VOL_JOINED.match(value)
        if not m:
            raise ValueError(f"bad volume {value!r}")
        return ("VOLUME", [m.group(1), m.group(2).upper()])
    ctor = CTOR_OF_TAG.get(key)
    if ctor is None:
        raise ValueError(f"unknown tag {key!r}")
    return (ctor, [value if ctor == "NUMBER" else value.upper()])


def from_tagged(text):
    subs = []
    for part in text.strip().split(";"):
        toks = part.split()
        if not toks or toks[0] not in ("pizza", "drink"):
            raise ValueError("bad suborder head")
        ctor = "PIZZAORDER" if toks[0] == "pizza" else "DRINKORDER"
        subs.append((ctor, [_tagged_attr_back(t) for t in toks[1:]]))
    if not subs:
        raise ValueError("empty order")
    return ("ORDER", subs)


# ---------- words: dsl structure, spelled out, no sigils ----------


def _words_attr(node):
    ctor, kids = node
    if ctor == "NUMBER":
        return kids[0]
    if ctor == "VOLUME":
        return f"{kids[0]} {kids[1].lower()}"
    if ctor == "STYLE":
        return "style " + kids[0].lower()
    if ctor == "NOT":
        return "no " + _words_attr(kids[0])
    if ctor == "COMPLEX_TOPPING":
        q = next(k[1][0] for k in kids if k[0] == "QUANTITY")
        t = next(k[1][0] for k in kids if k[0] == "TOPPING")
        return f"{QUANTITY_WORD[q]} {t.lower()}"
    return kids[0].lower()


def to_words(node):
    return " ; ".join(
        ("pizza" if sub[0] == "PIZZAORDER" else "drink")
        + "".join(" " + _words_attr(a) for a in sub[1])
        for sub in node[1]
    )


_VOL_UNITS = {"ml", "oz", "floz", "liter"}


def _words_attr_back(toks, i, cat):
    """Attributes span several whitespace-separated words, so return the next index too."""
    t = toks[i]
    if t == "no":
        inner, j = _words_attr_back(toks, i + 1, cat)
        return ("NOT", [inner]), j
    if t in WORD_QUANTITY:
        return (
            "COMPLEX_TOPPING",
            [("QUANTITY", [WORD_QUANTITY[t]]), ("TOPPING", [toks[i + 1].upper()])],
        ), i + 2
    if t == "style":
        return ("STYLE", [toks[i + 1].upper()]), i + 2
    if t[0].isdigit() and i + 1 < len(toks) and toks[i + 1] in _VOL_UNITS:
        return ("VOLUME", [t, toks[i + 1].upper()]), i + 2
    if t.isdigit():
        return ("NUMBER", [t]), i + 1
    return (cat.constructor_of(t), [t.upper()]), i + 1


def from_words(text, cat=None):
    cat = cat or catalog()
    subs = []
    for part in text.strip().split(";"):
        toks = part.split()
        if not toks or toks[0] not in ("pizza", "drink"):
            raise ValueError("bad suborder head")
        ctor = "PIZZAORDER" if toks[0] == "pizza" else "DRINKORDER"
        attrs, i = [], 1
        while i < len(toks):
            attr, i = _words_attr_back(toks, i, cat)
            attrs.append(attr)
        subs.append((ctor, attrs))
    if not subs:
        raise ValueError("empty order")
    return ("ORDER", subs)
