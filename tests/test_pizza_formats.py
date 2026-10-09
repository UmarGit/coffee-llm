import random

import pytest

pizza = pytest.importorskip("pizza_bench")
from pizza_bench import data, formats, grammar, metrics

pytestmark = pytest.mark.skipif(
    not (formats.PIZZA_DIR / "utils/catalogs").is_dir(),
    reason="PIZZA dataset not cloned (see README: Benchmark: PIZZA)",
)

TRAIN_SAMPLE = 50_000


@pytest.fixture(scope="module")
def cat():
    return formats.Catalog()


@pytest.fixture(scope="module")
def splits():
    return {s: data.load(s) for s in ("dev", "test")}


@pytest.fixture(scope="module")
def train_sample():
    rows = data.load("train")
    return data.subset(rows, min(TRAIN_SAMPLE, len(rows)), seed=0)


def _roundtrip(rows, fmt, cat):
    bad = []
    for r in rows:
        node = formats.parse_exr(r["exr"])
        canonical = formats.to_exr(node)
        text = formats.encode(node, fmt)
        back = formats.to_exr(formats.decode(text, fmt))
        if back != canonical:
            bad.append((r["exr"], text, back))
    return bad


@pytest.mark.parametrize("fmt", ["dsl", "json"])
@pytest.mark.parametrize("split", ["dev", "test"])
def test_roundtrip_is_identical(splits, cat, fmt, split):
    bad = _roundtrip(splits[split], fmt, cat)
    assert not bad, f"{len(bad)}/{len(splits[split])} failed, first: {bad[0]}"


@pytest.mark.parametrize("fmt", ["dsl", "json"])
def test_roundtrip_train_sample(train_sample, cat, fmt):
    bad = _roundtrip(train_sample, fmt, cat)
    assert not bad, f"{len(bad)}/{len(train_sample)} failed, first: {bad[0]}"


@pytest.mark.parametrize("split", ["dev", "test"])
def test_roundtrip_matches_official_matcher(splits, split):
    for r in splits[split][:300]:
        node = formats.parse_exr(r["exr"])
        assert metrics.exact_match(
            formats.to_exr(formats.from_dsl(formats.to_dsl(node))), r["exr"]
        )


def test_exr_parse_rejects_malformed():
    for bad in [
        "(ORDER (PIZZAORDER (SIZE LARGE ) ) ) )",
        "(ORDER (PIZZAORDER (NOT ) ) )",
        "(PIZZAORDER (NUMBER 1 ) )",
        "(ORDER (PIZZAORDER (NUMBER 1 ) )",
    ]:
        with pytest.raises(Exception):
            formats.parse_exr(bad)


def test_dsl_decode_rejects_unknown_entity(cat):
    with pytest.raises(KeyError):
        formats.from_dsl("p 1 unicorn_meat", cat)
    with pytest.raises(ValueError):
        formats.from_dsl("x 1 ham", cat)


def test_catalog_is_unambiguous(cat):
    words = [e.lower() for ctor in formats.BARE for e in cat.ids[ctor]]
    assert len(words) == len(set(words))
    assert not {"p", "d"} & set(words)
    assert not [w for w in words if w[0] in "!+~^" or w[0].isdigit()]


def test_catalog_covers_every_entity_in_the_data(splits, cat):
    seen = set()
    for rows in splits.values():
        for r in rows:
            node = formats.parse_exr(r["exr"])
            stack = [node]
            while stack:
                ctor, kids = stack.pop()
                for k in kids:
                    if isinstance(k, str):
                        if ctor in formats.BARE:
                            seen.add((ctor, k))
                    else:
                        stack.append(k)
    missing = [(c, e) for c, e in seen if e not in cat.ids[c]]
    assert not missing, f"entities absent from catalogs: {missing[:10]}"


def test_nested_not_over_complex_topping(cat):
    exr = (
        "(ORDER (PIZZAORDER (NOT (COMPLEX_TOPPING (QUANTITY EXTRA ) (TOPPING CHEESE ) ) ) "
        "(NUMBER 1 ) ) )"
    )
    node = formats.parse_exr(exr)
    assert formats.to_dsl(node) == "p !+cheese 1"
    assert formats.to_exr(formats.from_dsl("p !+cheese 1", cat)) == formats.to_exr(node)


def test_volume_roundtrip(cat):
    exr = "(ORDER (DRINKORDER (NUMBER 1 ) (VOLUME 20 FLOZ ) (DRINKTYPE COKE ) ) )"
    node = formats.parse_exr(exr)
    assert formats.to_dsl(node) == "d 1 20floz coke"
    assert formats.to_exr(formats.from_dsl("d 1 20floz coke", cat)) == formats.to_exr(
        node
    )


def test_subsets_are_deterministic_and_nested_by_seed():
    rows = data.load("dev")
    assert data.subset(rows, 50, 0) == data.subset(rows, 50, 0)
    assert data.subset(rows, 50, 0) != data.subset(rows, 50, 1)
    assert data.subset(rows, 20, 0) == data.subset(rows, 50, 0)[:20]


def test_pcfg_error_subset_size(splits):
    assert len(data.pcfg_error_subset(splits["test"])) == 434
    assert len(data.pcfg_error_subset(splits["dev"])) == 106


def test_metrics_on_perfect_and_broken_predictions(splits):
    rows = splits["dev"][:40]
    golds = [r["exr"] for r in rows]
    perfect = [formats.to_dsl(formats.parse_exr(g)) for g in golds]
    res, preds = metrics.evaluate(perfect, golds, "dsl", [r["pcfg_err"] for r in rows])
    assert res["em"] == 1.0 and res["parse_validity"] == 1.0
    assert res["hallucinated_entities_per_utt"] == 0.0
    assert all(p is not None for p in preds)

    broken = ["p 1 (((" for _ in rows]
    res2, preds2 = metrics.evaluate(broken, golds, "dsl")
    assert res2["parse_validity"] == 0.0 and res2["em"] == 0.0


def test_metrics_detect_a_wrong_number(splits):
    gold = "(ORDER (PIZZAORDER (NUMBER 2 ) (TOPPING HAM ) ) )"
    pred = formats.to_dsl(
        formats.parse_exr("(ORDER (PIZZAORDER (NUMBER 3 ) (TOPPING HAM ) ) )")
    )
    res, _ = metrics.evaluate([pred], [gold], "dsl")
    assert res["em"] == 0.0
    assert res["hallucinated_numbers_per_utt"] == 1.0
    assert res["prf"]["number"]["p"] == 0.0
    assert res["prf"]["topping"]["p"] == 1.0


def test_grammar_builds_and_covers_the_sigils(cat):
    g = grammar.dsl_gbnf(cat)
    assert "root" in g and '"!"' in g and '"^"' in g
    for word in ("ham", "thin_crust", "coke", "large", "20floz"):
        assert f'"{word}"' in g
