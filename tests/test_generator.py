import json
from pathlib import Path

import pytest

from bench.tasks.coffee import generator as g
from bench.tasks.coffee.parsing import parse_compact

REPO = Path(__file__).resolve().parents[1]


def test_menu_consistent():
    g.assert_menu_consistent()


def test_deterministic():
    assert g.generate(200, seed=0) == g.generate(200, seed=0)
    assert g.generate(200, seed=0) != g.generate(200, seed=1)


def test_difficulty_changes_distribution():
    easy = g.generate(500, seed=0, cfg=g.DEFAULT)
    hard = g.generate(500, seed=0, cfg=g.HARD)
    assert easy != hard
    items = lambda d: sum(len(e["output"].split(";")) for e in d) / len(d)
    assert items(hard) > items(easy)


def test_every_label_parses():
    for ex in g.generate(1000, seed=7, cfg=g.HARD):
        assert parse_compact(ex["output"]) is not None, ex


def test_generate_unique_is_unique_and_excludes():
    a = g.generate_unique(300, seed=1)
    inputs = [e["input"] for e in a]
    assert len(set(inputs)) == 300
    b = g.generate_unique(200, seed=2, exclude=set(inputs))
    assert not set(e["input"] for e in b) & set(inputs)


def test_splits_are_disjoint():
    from bench.tasks.coffee.testset import TEST_SET
    hand = {t for t, _ in TEST_SET}
    train = g.generate_unique(800, seed=101, exclude=hand)
    seen = hand | {e["input"] for e in train}
    val = g.generate_unique(300, seed=202, exclude=seen)
    seen |= {e["input"] for e in val}
    test = g.generate_unique(500, seed=303, exclude=seen)
    seen |= {e["input"] for e in test}
    hard = g.generate_unique(500, seed=404, cfg=g.HARD, exclude=seen)
    groups = [hand, *({e["input"] for e in s} for s in (train, val, test, hard))]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not a & b


def test_production_dataset_matches_frozen_reference():
    """The moved generator must reproduce the production training data exactly."""
    ref = REPO / "tests/data/production_generator_v2.py"
    ns = {}
    exec(compile(ref.read_text(), str(ref), "exec"), ns)
    train, val = g.production_dataset()
    assert ns["train_data"] == train
    assert ns["val_data"] == val
