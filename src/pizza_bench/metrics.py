"""Metrics. Exact match always goes through the dataset's own semantic matcher.

The official utils use flat imports and print to stdout when a tree fails to build, so
they are loaded with utils/ on sys.path and their output is captured rather than leaked.
"""

import contextlib
import io
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .formats import PIZZA_DIR, parse_exr, to_exr_from


@lru_cache(maxsize=1)
def _official():
    utils = Path(PIZZA_DIR) / "utils"
    if not utils.is_dir():
        raise FileNotFoundError(f"PIZZA utils not found at {utils}")
    if str(utils) not in sys.path:
        sys.path.insert(0, str(utils))
    import semantic_matchers

    return semantic_matchers


def exact_match(pred_exr, gold_exr):
    """Unordered EM via the official matcher. Never reimplemented here."""
    m = _official()
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(m.is_unordered_exact_match(pred_exr, gold_exr, "EXR"))


def to_exr_or_none(text, fmt):
    try:
        exr = to_exr_from(text, fmt)
    except Exception:
        return None
    try:
        parse_exr(exr)
    except Exception:
        return None
    return exr


def _slots(node, out, path=()):
    ctor, kids = node
    leaves = [k for k in kids if isinstance(k, str)]
    children = [k for k in kids if not isinstance(k, str)]
    if leaves:
        out[(ctor,) + path].append(" ".join(leaves))
    for c in children:
        _slots(c, out, path + (ctor,) if ctor in ("NOT", "COMPLEX_TOPPING") else path)


def slot_bags(exr):
    """Multiset of (slot, value) per utterance, with NOT/COMPLEX_TOPPING kept in the key."""
    from collections import defaultdict

    bags = defaultdict(list)
    for sub in parse_exr(exr)[1]:
        _slots(sub, bags)
    return {k: Counter(v) for k, v in bags.items()}


SLOT_GROUPS = {
    "topping": lambda k: k[0] == "TOPPING",
    "topping_negated": lambda k: k[0] == "TOPPING" and "NOT" in k[1:],
    "topping_quantified": lambda k: k[0] == "TOPPING" and "COMPLEX_TOPPING" in k[1:],
    "size": lambda k: k[0] == "SIZE",
    "number": lambda k: k[0] == "NUMBER",
    "style": lambda k: k[0] == "STYLE",
    "drink": lambda k: k[0] == "DRINKTYPE",
    "volume": lambda k: k[0] == "VOLUME",
    "container": lambda k: k[0] == "CONTAINERTYPE",
}


def prf(preds_exr, golds_exr):
    """Micro P/R/F1 per slot group over multisets of (slot, value)."""
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred, gold in zip(preds_exr, golds_exr):
        pbags = slot_bags(pred) if pred else {}
        gbags = slot_bags(gold)
        for name, keep in SLOT_GROUPS.items():
            p = Counter()
            g = Counter()
            for k, c in pbags.items():
                if keep(k):
                    p.update({(k, v): n for v, n in c.items()})
            for k, c in gbags.items():
                if keep(k):
                    g.update({(k, v): n for v, n in c.items()})
            inter = sum((p & g).values())
            tp[name] += inter
            fp[name] += sum(p.values()) - inter
            fn[name] += sum(g.values()) - inter
    out = {}
    for name in SLOT_GROUPS:
        precision = tp[name] / (tp[name] + fp[name]) if tp[name] + fp[name] else None
        recall = tp[name] / (tp[name] + fn[name]) if tp[name] + fn[name] else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if precision is not None else None)
        )
        out[name] = {
            "p": precision,
            "r": recall,
            "f1": f1,
            "tp": tp[name],
            "fp": fp[name],
            "fn": fn[name],
        }
    return out


def _suborders(exr):
    return Counter(sub[0] for sub in parse_exr(exr)[1])


def structural_errors(preds_exr, golds_exr):
    """Suborder count errors and leaf entities that are not in the gold tree."""
    extra = missing = 0
    halluc_entity = halluc_number = 0
    total_pred_leaves = total_gold_numbers = 0
    utt_with_halluc = 0
    for pred, gold in zip(preds_exr, golds_exr):
        if pred is None:
            continue
        p, g = _suborders(pred), _suborders(gold)
        for ctor in set(p) | set(g):
            diff = p[ctor] - g[ctor]
            extra += max(diff, 0)
            missing += max(-diff, 0)
        pbags, gbags = slot_bags(pred), slot_bags(gold)
        flat = lambda bags: Counter(
            {(k, v): n for k, c in bags.items() for v, n in c.items()}
        )
        pf, gf = flat(pbags), flat(gbags)
        unmatched = pf - gf
        total_pred_leaves += sum(pf.values())
        numbers = sum(n for (k, _), n in unmatched.items() if k[0] == "NUMBER")
        halluc_number += numbers
        halluc_entity += sum(unmatched.values()) - numbers
        total_gold_numbers += sum(n for (k, _), n in gf.items() if k[0] == "NUMBER")
        if unmatched:
            utt_with_halluc += 1
    n = len(golds_exr)
    return {
        "extra_suborders": extra / n,
        "missing_suborders": missing / n,
        "hallucinated_entities_per_utt": halluc_entity / n,
        "hallucinated_numbers_per_utt": halluc_number / n,
        "utts_with_hallucination": utt_with_halluc / n,
        "hallucination_rate_of_pred_leaves": (
            (halluc_entity + halluc_number) / total_pred_leaves
            if total_pred_leaves
            else None
        ),
    }


def evaluate(outputs, golds_exr, fmt, pcfg_err_flags=None):
    preds = [to_exr_or_none(o, fmt) for o in outputs]
    em = [bool(p) and exact_match(p, g) for p, g in zip(preds, golds_exr)]
    n = len(golds_exr)
    result = {
        "n": n,
        "em": sum(em) / n,
        "parse_validity": sum(p is not None for p in preds) / n,
        "prf": prf(preds, golds_exr),
        **structural_errors(preds, golds_exr),
    }
    if pcfg_err_flags is not None:
        idx = [i for i, f in enumerate(pcfg_err_flags) if f]
        result["em_pcfg_err"] = sum(em[i] for i in idx) / len(idx) if idx else None
        result["n_pcfg_err"] = len(idx)
    return result, preds
