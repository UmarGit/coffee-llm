"""Turn model output into structured orders.

Compact format:  `cap L oat;lat x2`
  items separated by `;`, each `drink [size] [milk] [syrup] [xN]`
  `?` = not an order, off-menu, or too vague
"""

from collections import Counter

from .menu import DRINK_CODES, MILKS, SIZES, SYRUPS, UNCLEAR


def _item_key(drink, size=None, milk=None, syrup=None):
    return (drink, size, milk, syrup)


def parse_compact(text):
    """Parse model output into a canonical, comparable form.

    Returns (frozenset of ((drink, size, milk, syrup), qty), unclear_flag),
    or None if the text is not valid compact output.
    """
    items, unclear = Counter(), False
    for part in text.strip().split(";"):
        toks = part.split()
        if not toks:
            continue
        if toks == [UNCLEAR]:
            unclear = True
            continue
        if toks[0] not in DRINK_CODES:
            return None
        drink, size, milk, syrup, qty = DRINK_CODES[toks[0]], None, None, None, 1
        for t in toks[1:]:
            if t in SIZES:
                size = t
            elif t in MILKS:
                milk = t
            elif t in SYRUPS:
                syrup = t
            elif t.startswith("x") and t[1:].isdigit():
                qty = int(t[1:])
            else:
                return None
        items[_item_key(drink, size, milk, syrup)] += qty
    return (frozenset(items.items()), unclear)


def to_json(text):
    """Expand model output into a JSON-serialisable order, for downstream use.

    >>> to_json("cap L oat;lat x2")
    {'items': [{'drink': 'cappuccino', 'qty': 1, 'size': 'L', 'milk': 'oat'},
               {'drink': 'latte', 'qty': 2}], 'unclear': False}

    Raises ValueError if the text is not valid compact output.
    """
    items, unclear = [], False
    for part in text.strip().split(";"):
        toks = part.split()
        if not toks:
            continue
        if toks == [UNCLEAR]:
            unclear = True
            continue
        if toks[0] not in DRINK_CODES:
            raise ValueError(f"unknown drink code: {toks[0]!r}")
        item = {"drink": DRINK_CODES[toks[0]], "qty": 1}
        for t in toks[1:]:
            if t in SIZES:
                item["size"] = t
            elif t in MILKS:
                item["milk"] = t
            elif t in SYRUPS:
                item["syrup"] = t
            elif t.startswith("x") and t[1:].isdigit():
                item["qty"] = int(t[1:])
            else:
                raise ValueError(f"unknown token: {t!r}")
        items.append(item)
    return {"items": items, "unclear": unclear}
