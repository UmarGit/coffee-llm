"""Metrics. Exact match always goes through the dataset's own matcher.

Error definitions, all per utterance and all computed on the predicted tree against the
gold tree after converting the arm's output back to EXR:

  parse_validity        the output converts to a well-formed EXR tree
  hallucinated_entity   a leaf (constructor, value) present in the prediction and absent
                        from gold, excluding NUMBER. Multiset difference, so a topping
                        predicted twice but present once counts as one hallucination.
  wrong_number          a NUMBER leaf present in the prediction and absent from gold.
                        Separated from hallucinated_entity because an invented count is a
                        different failure from an invented topping.
  missing_entity        a gold leaf absent from the prediction (multiset difference).
  extra_suborder        predicted PIZZAORDER + DRINKORDER count above gold.
  missing_suborder      predicted suborder count below gold.
  merged_suborder       predicted suborder count below gold while every non-NUMBER gold leaf
                        is still present, indicating two orders were collapsed into one
                        rather than one being dropped. NUMBER is excluded because collapsing
                        two suborders necessarily loses one of their counts, so requiring it
                        to survive would make the category unreachable.

The official utils use flat imports and print to stdout when a tree fails to build, so they
are loaded with utils/ on sys.path and their output is captured.
"""
import contextlib
import io
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .formats import PIZZA_DIR, decode, parse_exr, to_exr



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
    """Unordered EM from the official matcher. Never reimplemented here."""
    m = _official()
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(m.is_unordered_exact_match(pred_exr, gold_exr, "EXR"))


def to_exr_or_none(text, arm):
    try:
        exr = to_exr(decode(text, arm))
        parse_exr(exr)
        return exr
    except Exception:
        return None


def _leaves(node, path=(), out=None):
    out = [] if out is None else out
    ctor, kids = node
    nxt = path + (ctor,) if ctor in ("NOT", "COMPLEX_TOPPING") else path
    leaf = [k for k in kids if isinstance(k, str)]
    if leaf:
        out.append(((ctor,) + path, " ".join(leaf)))
    for k in kids:
        if not isinstance(k, str):
            _leaves(k, nxt, out)
    return out


def leaf_bag(exr):
    return Counter(_leaves(parse_exr(exr)))


def suborder_count(exr):
    return len(parse_exr(exr)[1])


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
    tp, fp, fn = Counter(), Counter(), Counter()
    for pred, gold in zip(preds_exr, golds_exr):
        pb = leaf_bag(pred) if pred else Counter()
        gb = leaf_bag(gold)
        for name, keep in SLOT_GROUPS.items():
            p = Counter({k: v for k, v in pb.items() if keep(k[0])})
            g = Counter({k: v for k, v in gb.items() if keep(k[0])})
            inter = sum((p & g).values())
            tp[name] += inter
            fp[name] += sum(p.values()) - inter
            fn[name] += sum(g.values()) - inter
    out = {}
    for name in SLOT_GROUPS:
        P = tp[name] / (tp[name] + fp[name]) if tp[name] + fp[name] else None
        R = tp[name] / (tp[name] + fn[name]) if tp[name] + fn[name] else None
        F = 2 * P * R / (P + R) if P and R else (0.0 if P is not None else None)
        out[name] = {"p": P, "r": R, "f1": F, "tp": tp[name], "fp": fp[name], "fn": fn[name]}
    return out


ERROR_KEYS = ("invalid", "wrong_number", "hallucinated_entity", "missing_entity",
              "utts_with_hallucination", "extra_suborder", "missing_suborder",
              "merged_suborder")


def error_counts(preds_exr, golds_exr):
    c = Counter({k: 0 for k in ERROR_KEYS})
    n = len(golds_exr)
    for pred, gold in zip(preds_exr, golds_exr):
        if pred is None:
            c["invalid"] += 1
            continue
        pb, gb = leaf_bag(pred), leaf_bag(gold)
        extra, missing = pb - gb, gb - pb
        c["wrong_number"] += sum(v for k, v in extra.items() if k[0][0] == "NUMBER")
        c["hallucinated_entity"] += sum(v for k, v in extra.items() if k[0][0] != "NUMBER")
        c["missing_entity"] += sum(missing.values())
        c["utts_with_hallucination"] += bool(extra)
        ps, gs = suborder_count(pred), suborder_count(gold)
        c["extra_suborder"] += max(ps - gs, 0)
        c["missing_suborder"] += max(gs - ps, 0)
        gold_non_number = Counter({k: v for k, v in gb.items() if k[0][0] != "NUMBER"})
        if ps < gs and not (gold_non_number - pb):
            c["merged_suborder"] += 1
    return {k: v / n for k, v in c.items()} | {f"{k}_total": v for k, v in c.items()}


def evaluate(outputs, golds_exr, arm, pcfg_err=None, n_tokens=None, cap=None):
    preds = [to_exr_or_none(o, arm) for o in outputs]
    em = [bool(p) and exact_match(p, g) for p, g in zip(preds, golds_exr)]
    n = len(golds_exr)
    n_valid = sum(p is not None for p in preds)
    res = {
        "n": n,
        "em": sum(em) / n,
        "parse_validity": n_valid / n,
        # Accuracy among outputs that parsed at all. Separates "cannot emit a well-formed
        # tree" from "misunderstood the utterance", which a pilot showed is the whole story.
        "em_given_valid": (sum(em) / n_valid) if n_valid else None,
        "prf": prf(preds, golds_exr),
        **error_counts(preds, golds_exr),
    }
    if n_tokens is not None:
        res["mean_generated_tokens"] = sum(n_tokens) / n
        res["max_generated_tokens"] = max(n_tokens)
        if cap:
            res["hit_cap_rate"] = sum(t >= cap for t in n_tokens) / n
    if pcfg_err is not None:
        idx = [i for i, f in enumerate(pcfg_err) if f]
        res["em_pcfg_err"] = sum(em[i] for i in idx) / len(idx) if idx else None
        res["n_pcfg_err"] = len(idx)
    return res, preds, em
