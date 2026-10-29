"""Negative training data for PIZZA: inputs that are not an order."""

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
] * 50


def test_off_menu_items_are_not_on_the_menu():
    # "iced tea", "coffee" or "cheeseburger" are PIZZA entities; rejecting them
    # would teach the model to refuse real orders.
    assert negatives.collisions() == []


def test_eval_phrases_are_never_trained_on():
    assert not set(negatives.NON_EVAL) & set(negatives.NON_TRAIN)
    assert not set(negatives.OFF_EVAL) & set(negatives.OFF_TRAIN)
    assert negatives.NON_EVAL and negatives.OFF_EVAL


def test_mix_hits_the_fraction_and_is_deterministic():
    a = negatives.mix(ROWS, 0.2, 0, "words")
    b = negatives.mix(ROWS, 0.2, 0, "words")
    assert a == b
    neg = [r for r in a if r.get("neg")]
    assert len(neg) == 25 and len(a) == 125
    assert {r["neg"] for r in neg} == {"non", "off", "mixed"}


@pytest.mark.parametrize("arm", negatives.ARMS)
def test_targets_round_trip_through_the_scorer(arm):
    for r in negatives.mix(ROWS, 0.5, 1, arm):
        if not r.get("neg"):
            continue
        out = negatives.target(r, arm)
        assert negatives.correct(out, r, arm), (r, out)


def test_bracketed_arms_are_refused():
    with pytest.raises(ValueError):
        negatives.mix(ROWS, 0.2, 0, "exr")


def test_scoring():
    rows = negatives.eval_rows(ROWS, n=3)
    kinds = [r["neg"] for r in rows]
    assert kinds == ["non", "off", "mixed"]
    mixed_ok = negatives.target(rows[2], "words")
    res = negatives.score(["?", "pizza 1 ham", mixed_ok], rows, "words",
                          test_outputs=["pizza 1 ham", "pizza 1 ham ; ?"])
    assert res["handled_non"] == 1 and res["handled_off"] == 0 and res["handled_mixed"] == 1
    assert res["false_reject_on_test"] == 0.5
    # A mixed order without the flag, or with a wrong ticket, is not handled.
    base = mixed_ok.rsplit(" ; ?", 1)[0]
    assert not negatives.correct(base, rows[2], "words")
    assert not negatives.correct("pizza 9 ham ; ?", rows[2], "words")
