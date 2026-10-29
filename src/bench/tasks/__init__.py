"""Every task-specific fact, in one place.

A task supplies its splits, its arms, how to turn a row into a target string, and
how to score a prediction. Nothing else in the harness knows which task is running.

Tasks are imported lazily, so a coffee run never imports the PIZZA modules and a
PIZZA run never imports the cafe generator. That keeps the CC-BY-NC boundary a
property of the code rather than of someone remembering.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    label: str

    arms: tuple
    """Output encodings to compare. A task with one arm simply has no ladder."""

    ladder_arms: tuple
    """The subset forming a length ladder, for the slope. Empty is allowed."""

    matched_pairs: tuple
    """Pairs holding length roughly fixed while varying surface form."""

    load_splits: Callable[[], dict]
    """() -> {"dev": [row, ...], "test": [row, ...]}"""

    train_rows: Callable[[int, int], list]
    """(size, seed) -> rows, drawn so that subsets are nested within a seed."""

    pool_size: Callable[[], int]

    source: Callable[[dict], str]
    """row -> the model's input text."""

    target: Callable[[dict, str], str]
    """(row, arm) -> the completion the model is trained to emit."""

    gold: Callable[[dict], str]
    """row -> the canonical gold string a prediction is compared against."""

    score: Callable[..., tuple]
    """(outputs, rows, arm, n_tokens=None, cap=None) -> (metrics, preds, correct).

    `metrics` must include "em" and "parse_validity" so the report means the same
    across tasks. `preds` is each output canonicalised, or None where it did not
    parse. Each task reads whatever else it needs straight off `rows`."""

    rows_from_predictions: Callable[[list], list]
    """Saved prediction rows -> rows `score` accepts, so every number can be
    recomputed from disk without a GPU and without the dataset."""

    default_arm: str
    """Used where a single arm is needed, such as the determinism replicate."""

    default_sizes: tuple = ()
    """Training pool sizes to sweep. Task specific, because a generator with an
    unbounded space and a fixed 2.46M corpus do not have comparable ladders."""

    lr_select_size: int = 0
    """The size the learning rate is tuned at, so it is tuned where the headline
    number is produced rather than extrapolated from a smaller run."""

    licence: str = ""
    """Non-empty means the results are encumbered and must stay out of git."""

    pred_extra: Callable[[dict], dict] = lambda row: {}
    """Extra fields to persist per prediction, for tasks that need them."""

    prepare: Callable[[], None] | None = None
    """Fetch whatever the task needs before a run. None means nothing to fetch."""

    checks: Callable[..., list] | None = None
    """(cfg, splits) -> list of failure strings. The task's own gates: round trip,
    contamination, anything else it must prove before GPU time is spent. Prints
    its own detail. The generic gates (token caps, EOS id) live in the CLI."""

    write_grammar: Callable[..., None] | None = None
    """(path) -> None, for a task with a constrained-decoding grammar."""

    published: tuple = ()
    """((system, test_em), ...) from the literature, for context only. Empty for a
    task with no published numbers, which is most of them; the report then omits
    the section rather than quoting another task's baselines."""

    published_train_n: int = 0
    """Training set size those published numbers used, so the report can say how
    far from like-for-like the comparison is."""

    negatives: object = None
    """A module with mix/target/eval_rows/score for training on inputs that are
    not orders (see tasks/pizza/negatives.py). None means the task has none."""


_LOADERS = {
    "pizza": lambda: __import__("bench.tasks.pizza", fromlist=["TASK"]).TASK,
    "coffee": lambda: __import__("bench.tasks.coffee", fromlist=["TASK"]).TASK,
}

NAMES = tuple(_LOADERS)
_cache = {}


def get(name):
    if name not in _LOADERS:
        raise KeyError(f"unknown task {name!r}; have {sorted(_LOADERS)}")
    if name not in _cache:
        _cache[name] = _LOADERS[name]()
    return _cache[name]
