"""The PIZZA semantic-parsing task.

CC-BY-NC-4.0. The data, any prediction derived from it and any adapter trained on
it are non-commercial, must never reach the cafe parser, and must never be
committed. The matcher is the dataset's own `is_unordered_exact_match` and is
never reimplemented.
"""

from .. import Task
from . import data, formats, grammar, ladder, metrics
from .formats import ARMS, LADDER_ARMS, MATCHED_PAIRS, decode, encode, parse_exr, to_exr


def _target(row, arm):
    return encode(parse_exr(row["exr"]), arm)


def _score(outputs, rows, arm, n_tokens=None, cap=None):
    """Returns (metrics, preds_exr, correct). The matcher is the dataset's own."""
    return metrics.evaluate(
        outputs,
        [r["exr"] for r in rows],
        arm,
        pcfg_err=[r.get("pcfg_err") for r in rows],
        n_tokens=n_tokens,
        cap=cap,
    )


def _rows_from_predictions(pred_rows):
    return [
        {"src": r["src"], "exr": r["gold"], "pcfg_err": r.get("pcfg_err")}
        for r in pred_rows
    ]


PIZZA_URL = "https://github.com/amazon-science/pizza-semantic-parsing-dataset.git"


def _prepare():
    import subprocess
    from pathlib import Path

    target = Path(formats.PIZZA_DIR)
    if not (target / "utils/catalogs").is_dir():
        print(f"cloning PIZZA into {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", PIZZA_URL, str(target)])


def _checks(cfg, splits):
    """Round trip on every arm, and zero contamination at every size and seed."""
    failures = []
    sample = data.train_subset(min(50_000, data.pool_size(cfg.pool)), 0, cfg.pool)
    for name, rows in (
        ("dev", splits["dev"]),
        ("test", splits["test"]),
        ("train50k", sample),
    ):
        for arm in cfg.arms:
            bad = 0
            for r in rows:
                node = parse_exr(r["exr"])
                try:
                    if to_exr(decode(encode(node, arm), arm)) != to_exr(node):
                        bad += 1
                except Exception:
                    bad += 1
            print(f"  round-trip {name:9} {arm:11} failures {bad}")
            if bad:
                failures.append(f"round-trip {name}/{arm}: {bad}")

    for size in cfg.sizes:
        for seed in cfg.seeds:
            sub = data.train_subset(size, seed, cfg.pool)
            for split in ("dev", "test"):
                ov = data.overlap(sub, splits[split])
                print(
                    f"  overlap n={size:<8} seed={seed} vs {split:4} "
                    f"exact {ov['exact']} normalized {ov['normalized']}"
                )
                if ov["exact"] or ov["normalized"]:
                    failures.append(f"overlap n={size} seed={seed} {split}")
            print(
                f"  duplicate sources n={size:<8} seed={seed}: "
                f"{data.duplicate_sources(sub)}"
            )
    return failures


TASK = Task(
    name="pizza",
    label="PIZZA",
    arms=ARMS,
    ladder_arms=LADDER_ARMS,
    matched_pairs=MATCHED_PAIRS,
    load_splits=lambda: {s: data.load(s) for s in ("dev", "test")},
    train_rows=lambda size, seed: data.train_subset(size, seed),
    pool_size=lambda: data.pool_size(),
    source=lambda row: row["src"],
    target=_target,
    gold=lambda row: row["exr"],
    score=_score,
    rows_from_predictions=_rows_from_predictions,
    pred_extra=lambda row: {"pcfg_err": row.get("pcfg_err")},
    default_arm="dsl",
    default_sizes=(1_000, 10_000, 100_000, 1_000_000, 2_456_443),
    lr_select_size=100_000,
    licence="CC-BY-NC-4.0",
    prepare=_prepare,
    checks=_checks,
    write_grammar=lambda path: grammar.write_gbnf(path),
)

__all__ = [
    "ARMS",
    "LADDER_ARMS",
    "MATCHED_PAIRS",
    "TASK",
    "data",
    "decode",
    "encode",
    "formats",
    "grammar",
    "ladder",
    "metrics",
    "parse_exr",
    "to_exr",
]
