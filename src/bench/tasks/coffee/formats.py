"""The two cafe output encodings, and the round trip between them.

`compact` is the shipped format. `words` spells the same fields with natural
words. Field order is fixed and the four vocabularies are disjoint, so the
mapping is unambiguous in both directions; `decode` returns None rather than
guessing when it meets anything outside the menu, which is what makes parse
validity a real measurement instead of an assumption.
"""

from .menu import DRINK_CODES, MILKS, SIZES, SYRUPS, UNCLEAR

ARMS = ("compact", "words")

SIZE_WORDS = {"S": "small", "M": "medium", "L": "large"}
MILK_WORDS = {"lactosefree": "lactose free", "nomilk": "no milk"}

# _DRINK_BACK = {v: k for k, v in DRINK_CODES.items()}
_SIZE_BACK = {v: k for k, v in SIZE_WORDS.items()}
_MILK_BACK = {v: k for k, v in MILK_WORDS.items()}
_UNCLEAR_WORD = "unclear"

# A drink name may contain an underscore (flat_white); words spells it out.
_DRINK_WORDS = {code: name.replace("_", " ") for code, name in DRINK_CODES.items()}
_DRINK_WORDS_BACK = {v: k for k, v in _DRINK_WORDS.items()}


def encode(compact, arm):
    """A compact string to the chosen arm. `compact` is the gold format."""
    if arm == "compact":
        return compact
    if arm != "words":
        raise ValueError(f"unknown arm {arm!r}")
    return ";".join(_item_to_words(i) for i in compact.split(";"))


def decode(text, arm):
    """Back to the canonical compact string, or None if it is not well formed."""
    text = (text or "").strip()
    if not text:
        return None
    if arm == "compact":
        return text if all(_valid_compact(i) for i in text.split(";")) else None
    if arm != "words":
        raise ValueError(f"unknown arm {arm!r}")
    out = []
    for item in text.split(";"):
        back = _item_from_words(item)
        if back is None:
            return None
        out.append(back)
    return ";".join(out)


def _item_to_words(item):
    item = item.strip()
    if item == UNCLEAR:
        return _UNCLEAR_WORD
    parts = item.split()
    if not parts:
        return _UNCLEAR_WORD
    out = [_DRINK_WORDS.get(parts[0], parts[0])]
    for p in parts[1:]:
        if p in SIZE_WORDS:
            out.append(SIZE_WORDS[p])
        elif p in MILKS:
            out.append(MILK_WORDS.get(p, p))
        elif p in SYRUPS:
            out.append(p)
        elif p.startswith("x") and p[1:].isdigit():
            out.append(f"times {p[1:]}")
        else:
            out.append(p)
    return " ".join(out)


def _item_from_words(item):
    item = " ".join((item or "").split())
    if item == _UNCLEAR_WORD:
        return UNCLEAR
    # Longest multi-word names first, so "flat white" and "lactose free" win over
    # a greedy single-word match.
    for phrase, code in sorted(_MILK_BACK.items(), key=lambda kv: -len(kv[0])):
        item = item.replace(phrase, code)
    for phrase, code in sorted(_DRINK_WORDS_BACK.items(), key=lambda kv: -len(kv[0])):
        if item == phrase or item.startswith(phrase + " "):
            item = code + item[len(phrase) :]
            break
    else:
        return None
    parts = item.split()
    out, i = [parts[0]], 1
    while i < len(parts):
        p = parts[i]
        if p in _SIZE_BACK:
            out.append(_SIZE_BACK[p])
        elif p in MILKS or p in SYRUPS:
            out.append(p)
        elif p == "times" and i + 1 < len(parts) and parts[i + 1].isdigit():
            out.append(f"x{parts[i + 1]}")
            i += 1
        else:
            return None
        i += 1
    return " ".join(out)


def _valid_compact(item):
    parts = (item or "").strip().split()
    if not parts:
        return False
    if parts[0] == UNCLEAR:
        return len(parts) == 1
    if parts[0] not in DRINK_CODES:
        return False
    for p in parts[1:]:
        if not (
            p in SIZES
            or p in MILKS
            or p in SYRUPS
            or (p.startswith("x") and p[1:].isdigit())
        ):
            return False
    return True
