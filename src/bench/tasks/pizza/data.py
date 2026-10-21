"""PIZZA splits and nested training subsets.

Training subsets are drawn from the full 2.46M utterance pool, as the prefix of a single
seeded permutation. So for a given seed the subsets are nested: the 1k subset is the first
1000 rows of the 10k subset, and so on. One pool and one permutation means the data size
axis varies only in how much of the same distribution is drawn.

The full pool is read straight from its zip, never extracted: 2.46M lines stream in about
five seconds using 17 MiB, against 1.44 GB on disk if unpacked.
"""

import json
import random
import re
import zipfile
from pathlib import Path

from .formats import PIZZA_DIR

EVAL_FILES = {"dev": "PIZZA_dev.json", "test": "PIZZA_test.json"}
POOLS = {
    "full": ("PIZZA_train_unshuffled.json.zip", "PIZZA_train.json"),
    "10pct": ("PIZZA_train.10_percent.json.zip", "PIZZA_train.10_percent.json"),
}
DEFAULT_POOL = "full"
PROMPT = (
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def data_dir():
    return Path(PIZZA_DIR) / "data"


def _row(d, prefix):
    return {
        "src": d[f"{prefix}.SRC"],
        "exr": d[f"{prefix}.EXR"],
        "pcfg_err": d.get(f"{prefix}.PCFG_ERR") == "True"
        if prefix != "train"
        else None,
    }


def _open_pool(pool):
    """Yield raw lines of a training pool, preferring an already extracted file."""
    archive, member = POOLS[pool]
    plain = data_dir() / member
    if plain.exists():
        with open(plain, "rb") as f:
            yield from f
        return
    zpath = data_dir() / archive
    if not zpath.exists():
        raise FileNotFoundError(
            f"neither {plain} nor {zpath} exists; run `python -m bench prepare`"
        )
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        yield from f


def _cache_path(pool, kind):
    return data_dir() / f".{pool}.{kind}"


def blocked_lines(pool=DEFAULT_POOL):
    """Line numbers to drop from a pool because they duplicate a dev or test utterance.

    The full 2.46M pool contains three rows whose source and target are identical to an
    evaluation utterance (two in test, one in dev). The 10 percent pool contains none.
    They are removed from training rather than from the evaluation splits, so the splits
    stay exactly as published and comparable with prior work.
    """
    cache = _cache_path(pool, "blocked")
    if cache.exists():
        try:
            return {int(x) for x in cache.read_text().split() if x}
        except ValueError:
            pass
    evaluation = load("dev") + load("test")
    exact = {r["src"] for r in evaluation}
    normalised = {_norm(r["src"]) for r in evaluation}
    blocked = set()
    for line_no, line in enumerate(_open_pool(pool)):
        if not line.strip():
            continue
        src = json.loads(line)["train.SRC"]
        if src in exact or _norm(src) in normalised:
            blocked.add(line_no)
    try:
        cache.write_text(" ".join(str(x) for x in sorted(blocked)))
    except OSError:
        pass
    return blocked


def eligible_lines(pool=DEFAULT_POOL):
    """Line numbers a training subset may draw from, in file order."""
    cache = _cache_path(pool, "eligible")
    if cache.exists():
        try:
            return int(cache.read_text().strip())
        except ValueError:
            pass
    total = sum(1 for line in _open_pool(pool) if line.strip())
    n = total - len(blocked_lines(pool))
    try:
        cache.write_text(str(n))
    except OSError:
        pass
    return n


def pool_size(pool=DEFAULT_POOL):
    """Utterances available for training, after removing evaluation duplicates."""
    return eligible_lines(pool)


def load(split):
    """Load dev or test. For training data use train_subset, which streams."""
    if split in EVAL_FILES:
        rows = []
        with open(data_dir() / EVAL_FILES[split]) as f:
            for line in f:
                if line.strip():
                    rows.append(_row(json.loads(line), split))
        return rows
    if split == "train":
        return train_subset(pool_size(DEFAULT_POOL), seed=0)
    raise ValueError(f"unknown split {split!r}")


def permutation(seed, pool=DEFAULT_POOL):
    """The single seeded ordering that every subset for this seed is a prefix of.

    Indexes eligible rows only, so a blocked row can never enter any subset.
    """
    blocked = blocked_lines(pool)
    idx = [i for i in range(len(blocked) + pool_size(pool)) if i not in blocked]
    random.Random(10_000 + seed).shuffle(idx)
    return idx


def train_subset(size, seed, pool=DEFAULT_POOL):
    """The first `size` rows of this seed's permutation, in permutation order."""
    total = pool_size(pool)
    if size > total:
        raise ValueError(f"asked for {size:,} of {total:,} rows in the {pool} pool")
    wanted = permutation(seed, pool)[:size]
    rank = {line_no: position for position, line_no in enumerate(wanted)}
    out = [None] * size
    for line_no, line in enumerate(_open_pool(pool)):
        position = rank.get(line_no)
        if position is not None:
            out[position] = _row(json.loads(line), "train")
    missing = sum(r is None for r in out)
    if missing:
        raise RuntimeError(f"{missing} rows not found; the pool changed under us")
    return out


def head(size, pool=DEFAULT_POOL):
    """The first `size` eligible rows in file order.

    One partial pass, so this is far cheaper than a seeded subset, which must scan the
    whole pool. Use it for format checks, never for training: it is not a random sample.
    """
    blocked = blocked_lines(pool)
    out = []
    for line_no, line in enumerate(_open_pool(pool)):
        if line_no in blocked or not line.strip():
            continue
        out.append(_row(json.loads(line), "train"))
        if len(out) == size:
            break
    return out


def subset(rows, size, seed):
    """Deterministic subset of an in-memory list, used for dev slices and few-shot."""
    idx = list(range(len(rows)))
    random.Random(10_000 + seed).shuffle(idx)
    if size > len(idx):
        raise ValueError(f"asked for {size} of {len(idx)} rows")
    return [rows[i] for i in idx[:size]]


def pcfg_error_subset(rows):
    return [r for r in rows if r["pcfg_err"]]


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


def overlap(train_rows, eval_rows):
    """Exact and normalised source overlap between a training subset and an eval split."""
    te, tn = {r["src"] for r in train_rows}, {_norm(r["src"]) for r in train_rows}
    ee, en = {r["src"] for r in eval_rows}, {_norm(r["src"]) for r in eval_rows}
    return {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "exact": len(te & ee),
        "normalized": len(tn & en),
    }


def duplicate_sources(rows):
    from collections import Counter

    c = Counter(r["src"] for r in rows)
    return sum(v - 1 for v in c.values() if v > 1)
