"""Cafe splits and training subsets.

Test is the 49 hand-written cases in `testset.py`, which were written
separately from the generator and contain slang it never produces. Dev is
generated with a seed disjoint from any training seed, because tuning on the
hand-written set would spend the only independent measurement this task has.

Training rows exclude every dev and test input by construction, so overlap is
zero by the same argument the PIZZA pool uses rather than by a later check.
"""

from .generator import generate_unique
from .testset import TEST_SET

DEV_SIZE = 500
DEV_SEED = 90_001
"""Far from any training seed, which are small integers."""

POOL = 200_000
"""The generator is unbounded; this is the figure reported as the pool, and the
largest subset the harness will ask for."""


def _test_rows():
    return [{"input": src, "output": gold} for src, gold in TEST_SET]


def _dev_rows():
    excluded = {src for src, _ in TEST_SET}
    return generate_unique(DEV_SIZE, DEV_SEED, exclude=excluded)


def load_splits():
    return {"dev": _dev_rows(), "test": _test_rows()}


def pool_size():
    return POOL


def train_rows(size, seed):
    """`size` unique examples, none of them a dev or test input.

    Nested within a seed only in the weak sense that the generator is seeded, so
    unlike PIZZA these are not prefixes of one permutation. The generator draws
    with replacement from an unbounded space, so a nested prefix would require
    materialising the whole pool first for no benefit.
    """
    if size > POOL:
        raise ValueError(f"size {size} exceeds the declared pool of {POOL}")
    held_out = {src for src, _ in TEST_SET} | {r["input"] for r in _dev_rows()}
    return generate_unique(size, seed, exclude=held_out)


def overlap(train, evaluation):
    """Shared inputs. Should always be zero; `check` asserts it anyway."""
    a = {r["input"] for r in train}
    return a & {r["input"] for r in evaluation}
