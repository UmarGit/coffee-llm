"""The cafe order-parsing task: Russian and English transcripts to a compact string.

Not encumbered. This is the production task, so a model trained here may ship.
Everything it needs is here and uses nothing but the standard library:

    menu.py       the vocabulary, which the shipped model's output is defined by
    generator.py  the training dataset
    testset.py    the 49 hand-written evaluation cases
    parsing.py    compact string to JSON, used by callers of the shipped model
    ollama.py     the client that runs the shipped GGUF
    formats.py    the two output arms
    data.py       the splits

Importing this package never imports the PIZZA task, because the registry loads
tasks lazily. That is what keeps the CC-BY-NC boundary one-directional, and
`tests/test_registries.py` asserts it rather than trusting it.

Two arms, deliberately a matched pair rather than a ladder:

    compact   cap L oat;ame S                      the shipped format
    words     cappuccino large oat;americano small  the same content in words

They encode identical semantics and round-trip to a byte-identical compact string,
so the pair varies surface form while holding content fixed. That is the same
comparison the PIZZA study's `dsl` against `words` pair makes, on a second task
and in a second language, which is the cheapest way to test whether the finding
travels.
"""

from .. import Task
from . import formats, generator, parsing
from .data import load_splits, pool_size, train_rows
from .formats import ARMS, decode, encode
from .menu import DRINK_CODES, MILKS, SIZES, SYRUPS, UNCLEAR
from .ollama import PROMPT, STOPS, clean, generate
from .parsing import parse_compact, to_json
from .testset import TEST_SET, category


def _score(outputs, rows, arm, n_tokens=None, cap=None):
    """Exact match on the canonical compact form, plus parse validity.

    Key names match the PIZZA task's exactly, so the report, the early-stopping
    rule and the silent error rate (parse_validity minus em) mean the same thing
    whichever task produced them.
    """
    golds = [r["output"] for r in rows]
    preds = [decode(o, arm) for o in outputs]
    em = [p is not None and p == g for p, g in zip(preds, golds)]
    n = max(1, len(rows))
    n_valid = sum(p is not None for p in preds)
    res = {
        "n": len(rows),
        "em": sum(em) / n,
        "parse_validity": n_valid / n,
        "em_given_valid": (sum(em) / n_valid) if n_valid else None,
        # Two slices the cafe actually cares about: refusing correctly, and not
        # collapsing a multi-drink order into one.
        "em_unclear": _subset_em(rows, preds, lambda r: r["output"] == "?"),
        "em_multi": _subset_em(rows, preds, lambda r: ";" in r["output"]),
    }
    if n_tokens is not None:
        res["mean_generated_tokens"] = sum(n_tokens) / n
        res["max_generated_tokens"] = max(n_tokens)
        if cap:
            res["hit_cap_rate"] = sum(t >= cap for t in n_tokens) / n
    return res, preds, em


def _subset_em(rows, preds, pred):
    idx = [i for i, r in enumerate(rows) if pred(r)]
    if not idx:
        return None
    return sum(preds[i] == rows[i]["output"] for i in idx) / len(idx)


def _rows_from_predictions(pred_rows):
    return [{"input": r["src"], "output": r["gold"]} for r in pred_rows]


def _subset_em(rows, decoded, pred):
    idx = [i for i, r in enumerate(rows) if pred(r)]
    if not idx:
        return None
    return sum(decoded[i] == rows[i]["output"] for i in idx) / len(idx)


def _checks(cfg, splits):
    """Round trip on both arms, and zero overlap between training and evaluation.

    Overlap is zero by construction here, because `train_rows` excludes every dev
    and test input, so this asserts the construction rather than discovering a
    problem. It is still run: the generator is the kind of thing that changes.
    """
    from . import data as coffee_data

    failures = []
    for name, rows in (("dev", splits["dev"]), ("test", splits["test"])):
        for arm in ARMS:
            bad = sum(
                1 for r in rows if decode(encode(r["output"], arm), arm) != r["output"]
            )
            print(f"  round-trip {name:5} {arm:8} failures {bad}")
            if bad:
                failures.append(f"round-trip {name}/{arm}: {bad}")

    for size in cfg.sizes:
        for seed in cfg.seeds:
            sub = train_rows(size, seed)
            for split in ("dev", "test"):
                shared = coffee_data.overlap(sub, splits[split])
                print(f"  overlap n={size:<7} seed={seed} vs {split:4} {len(shared)}")
                if shared:
                    failures.append(f"overlap n={size} seed={seed} {split}")
            dup = len(sub) - len({r["input"] for r in sub})
            print(f"  duplicate sources n={size:<7} seed={seed}: {dup}")
            if dup:
                failures.append(f"duplicate sources n={size} seed={seed}: {dup}")
    return failures


TASK = Task(
    name="coffee",
    label="Cafe orders",
    arms=ARMS,
    ladder_arms=(),
    matched_pairs=(("compact", "words"),),
    load_splits=load_splits,
    train_rows=train_rows,
    pool_size=pool_size,
    source=lambda row: row["input"],
    target=lambda row, arm: encode(row["output"], arm),
    gold=lambda row: row["output"],
    score=_score,
    rows_from_predictions=_rows_from_predictions,
    default_arm="compact",
    # 4000 is what the shipped model was trained on, so it is the cell that
    # reproduces production; the others bracket it.
    default_sizes=(1_000, 4_000, 20_000, 100_000),
    lr_select_size=4_000,
    licence="",
    checks=_checks,
)

__all__ = [
    # the task
    "TASK", "ARMS", "encode", "decode", "formats",
    "load_splits", "train_rows", "pool_size",
    # the shipped parser, as `coffee_parser` used to expose it
    "DRINK_CODES", "MILKS", "SIZES", "SYRUPS", "UNCLEAR",
    "PROMPT", "STOPS", "clean", "generate",
    "parse_compact", "to_json",
    "TEST_SET", "category",
    "generator", "parsing",
]
