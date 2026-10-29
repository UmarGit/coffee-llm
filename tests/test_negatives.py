"""Negative training data for PIZZA: only menu orders are parsed, all else is "?"."""

import pytest

from bench.tasks.pizza import formats, negatives

pytestmark = pytest.mark.skipif(
    not (formats.PIZZA_DIR / "utils/catalogs").is_dir(),
    reason="PIZZA catalogs not downloaded",
)

ROWS = [
    {"src": "i want a large pizza with ham",
     "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE LARGE ) (TOPPING HAM ) ) )"},
    {"src": "two cokes please",
     "exr": "(ORDER (DRINKORDER (NUMBER 2 ) (DRINKTYPE COKE ) ) )"},
] * 200

# The real vocabulary needs the training pool; a stand-in keeps the unit tests fast.
VOCAB = frozenset("i want a large pizza with ham two cokes please".split())


@pytest.fixture(autouse=True)
def small_vocab(monkeypatch):
    monkeypatch.setattr(negatives, "pizza_vocab", lambda: VOCAB)


def test_nothing_on_the_menu_is_taught_as_a_refusal():
    # e.g. "cheeseburger" is a PIZZA topping and "iced tea" a PIZZA drink.
    assert negatives.collisions() == []


def test_eval_vocabulary_is_never_trained_on():
    for name, (train, held) in negatives.VOCAB.items():
        assert held and not set(train) & set(held), name


def test_mix_fraction_kinds_and_determinism():
    a = negatives.mix(ROWS, 0.3, 0, "words")
    assert a == negatives.mix(ROWS, 0.3, 0, "words")
    neg = [r for r in a if r.get("neg")]
    assert len(neg) == round(len(ROWS) * 0.3 / 0.7)
    assert {r["neg"] for r in neg} == set(negatives.KINDS)


def test_every_negative_is_refused_whole():
    for r in negatives.mix(ROWS, 0.5, 1, "words"):
        if r.get("neg"):
            assert negatives.target(r, "words") == "?"


def test_added_words_are_outside_real_order_language():
    for r in negatives.mix(ROWS, 0.5, 2, "words"):
        if r.get("neg") == "unknown":
            assert set(r["src"].split()) - VOCAB, r["src"]


def test_whole_negatives_never_name_a_menu_item():
    for r in negatives.eval_rows(ROWS, n=300):
        if r["neg"] in ("non", "off", "junk", "long"):
            assert negatives.on_menu(r["src"]) is None, r["src"]


def test_bracketed_arms_are_refused():
    with pytest.raises(ValueError):
        negatives.mix(ROWS, 0.2, 0, "exr")


def test_scoring_is_strict():
    rows = [{"src": "hehe hehe", "neg": "junk"}, {"src": "a hehe pizza", "neg": "unknown"},
            {"src": "two cats", "neg": "off"}]
    res = negatives.score(["?", "pizza 1 ; ?", "pizza 1 style cats"], rows, "words",
                          test_outputs=["pizza 1 ham", "?", "pizza 1 ham ; ?", "drink 2 coke"])
    # Only "?" alone counts: a half-parsed ticket or an invented one does not.
    assert res["handled_junk"] == 1 and res["handled_unknown"] == 0 and res["handled_off"] == 0
    assert res["false_reject_on_test"] == 0.5


def test_probes_include_the_reported_failures():
    assert "two cats and dog" in negatives.PROBES
    assert all(r["neg"] == "probe" for r in negatives.probe_rows())
