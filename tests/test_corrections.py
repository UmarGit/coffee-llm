"""Self-corrections for PIZZA: a later phrase overrides an earlier one."""

import random

import pytest

from bench.tasks.pizza import corrections, formats, negatives
from bench.tasks.pizza.formats import decode, encode, parse_exr, to_exr

pytestmark = pytest.mark.skipif(
    not (formats.PIZZA_DIR / "utils/catalogs").is_dir(),
    reason="PIZZA catalogs not downloaded",
)

ROWS = [
    {"src": "i want a large pizza with ham and olives",
     "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE LARGE ) (TOPPING HAM ) (TOPPING OLIVES ) ) )"},
    {"src": "two cokes please",
     "exr": "(ORDER (DRINKORDER (NUMBER 2 ) (DRINKTYPE COKE ) ) )"},
    {"src": "a medium pizza with bacon and a sprite",
     "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE MEDIUM ) (TOPPING BACON ) ) "
            "(DRINKORDER (NUMBER 1 ) (DRINKTYPE SPRITE ) ) )"},
    {"src": "a small pizza with tuna and no onions",
     "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE SMALL ) (TOPPING TUNA ) "
            "(NOT (TOPPING ONIONS ) ) ) )"},
] * 50

VOCAB = frozenset(w for r in ROWS for w in r["src"].split())


@pytest.fixture(autouse=True)
def small_vocab(monkeypatch):
    monkeypatch.setattr(negatives, "pizza_vocab", lambda: VOCAB)
    corrections.cue_words.cache_clear()
    yield
    corrections.cue_words.cache_clear()


def test_every_target_round_trips_through_every_arm():
    for r in corrections.mix(ROWS, 0.5, 0):
        node = parse_exr(r["exr"])
        for arm in formats.ARMS:
            assert to_exr(decode(encode(node, arm), arm)) == to_exr(node), (arm, r["src"])


def test_mix_size_kinds_and_determinism():
    a = corrections.mix(ROWS, 0.5, 0)
    assert a == corrections.mix(ROWS, 0.5, 0)
    fix = [r for r in a if r.get("fix")]
    assert len(fix) == round(len(ROWS) * 0.5)
    assert {r["fix"] for r in fix} == set(corrections.KINDS)


def test_counted_against_real_orders_only():
    rows = negatives.mix(ROWS, 0.3, 0, "words")
    out = corrections.mix(rows, 0.1, 0)
    assert sum(1 for r in out if r.get("fix")) == round(len(ROWS) * 0.1)
    assert all(not r.get("neg") for r in out if r.get("fix"))


def test_every_edit_but_a_decoy_changes_the_ticket():
    rng = random.Random(0)
    for kind in corrections.KINDS:
        changed = 0
        for base in ROWS[:4] * 10:
            node = parse_exr(base["exr"])
            got = corrections.edit(node, kind, rng, "wait", other=rng.choice(ROWS[:4]))
            if got is None:
                continue
            tree, said, must, must_not = got
            if kind == "decoy":
                assert set(corrections.facts(node)) <= set(corrections.facts(tree))
                continue
            changed += 1
            assert to_exr(tree) != to_exr(node), (kind, said)
            assert corrections.applied(to_exr(tree), {"must": must, "must_not": must_not})
            assert not corrections.applied(to_exr(node), {"must": must, "must_not": must_not}) \
                or kind == "number"
        assert kind == "decoy" or changed, kind


def test_retracting_what_was_said_removes_it():
    node = parse_exr(ROWS[0]["exr"])
    for seed in range(40):
        got = corrections._retract(node, random.Random(seed), "wait")
        tree, said, must, must_not = got
        words = encode(tree, "words")
        if must:
            # Never mentioned: stays a negation, as PIZZA already does.
            assert " no " in words and "no ham" not in words and "no olives" not in words
        else:
            assert ("ham" in words) != ("olives" in words), (said, words)
            assert " no " not in words


def test_held_out_cues_are_never_trained_on():
    for train, held in corrections.SPLIT.values():
        assert held and not set(train) & set(held)
    held = set(corrections.SPLIT["cue"][1] + corrections.SPLIT["restart"][1])
    trained = {r["cue"] for r in corrections.mix(ROWS, 0.5, 3) if r.get("fix")}
    evaluated = {r["cue"] for r in corrections.eval_rows(ROWS, n=300)} - {""}
    assert not trained & held and evaluated <= held


def test_no_cue_word_is_taught_as_a_refusal():
    assert negatives.cue_collisions() == []
    cues = corrections.cue_words()
    for r in negatives.mix(ROWS, 0.5, 2, "words"):
        if r.get("neg") == "unknown":
            assert not set(r["src"].split()) & cues, r["src"]


def test_the_reported_failure_is_a_probe():
    probes = {src: ticket for src, ticket, _ in corrections.PROBES}
    assert probes["i want a pizza with cheese, wait no cheese, only tomato sauce"] \
        == "pizza 1 tomato_sauce"
    assert len(corrections.probe_rows()) == len(corrections.PROBES) >= 40


def test_scoring_separates_exact_and_applied():
    rows = [
        {"src": "a", "fix": "swap", "must": ["p TOPPING BACON"], "must_not": ["p TOPPING HAM"],
         "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING BACON ) ) )"},
        {"src": "b", "fix": "swap", "must": ["p TOPPING BACON"], "must_not": ["p TOPPING HAM"],
         "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING BACON ) ) )"},
        {"src": "c", "fix": "retract", "must": [], "must_not": ["p TOPPING HAM"],
         "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) ) )"},
    ]
    res, _, em = corrections.score(["pizza 1 bacon", "pizza 2 bacon", "?"], rows, "words")
    assert em == [True, False, False]
    assert res["em_swap"] == 0.5 and res["applied_swap"] == 1.0
    assert res["refused"] == pytest.approx(1 / 3)


def test_applied_survives_a_tree_that_breaks_the_grammar():
    # Seen mid-training: it parses, but a COMPLEX_TOPPING has no QUANTITY.
    pred = "(ORDER (PIZZAORDER (NUMBER 1 ) (COMPLEX_TOPPING (TOPPING HAM ) ) ) )"
    assert corrections.applied(pred, {"must": ["p TOPPING HAM"], "must_not": []}) is False
