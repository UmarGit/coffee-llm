import json

import pytest

pizza = pytest.importorskip("pizza_bench")
from pizza_bench import data, evaluate, formats, grammar, ladder, metrics
from pizza_bench.config import EOS, PROMPT

pytestmark = pytest.mark.skipif(
    not (formats.PIZZA_DIR / "utils/catalogs").is_dir(),
    reason="PIZZA dataset not cloned; run python -m pizza_bench prepare")

TRAIN_SAMPLE = 50_000


@pytest.fixture(scope="module")
def splits():
    return {s: data.load(s) for s in ("dev", "test")}


@pytest.fixture(scope="module")
def train_sample():
    # head() is a single partial pass; a seeded subset would scan all 2.46M rows and
    # this gate only cares that the converters handle real training targets.
    return data.head(TRAIN_SAMPLE)


def _roundtrip(rows, arm):
    bad = []
    for r in rows:
        node = formats.parse_exr(r["exr"])
        canon = formats.to_exr(node)
        text = formats.encode(node, arm)
        try:
            back = formats.to_exr(formats.decode(text, arm))
        except Exception as e:
            bad.append((r["exr"], text, repr(e)))
            continue
        if back != canon:
            bad.append((r["exr"], text, back))
    return bad


@pytest.mark.parametrize("arm", formats.ARMS)
@pytest.mark.parametrize("split", ["dev", "test"])
def test_roundtrip_identical(splits, arm, split):
    bad = _roundtrip(splits[split], arm)
    assert not bad, f"{len(bad)}/{len(splits[split])} failed, first: {bad[0]}"


@pytest.mark.parametrize("arm", formats.ARMS)
def test_roundtrip_train_sample(train_sample, arm):
    bad = _roundtrip(train_sample, arm)
    assert not bad, f"{len(bad)}/{len(train_sample)} failed, first: {bad[0]}"


@pytest.mark.parametrize("arm", formats.ARMS)
def test_roundtrip_agrees_with_official_matcher(splits, arm):
    for r in splits["dev"][:200]:
        node = formats.parse_exr(r["exr"])
        back = formats.to_exr(formats.decode(formats.encode(node, arm), arm))
        assert metrics.exact_match(back, r["exr"])


def test_ladder_is_ordered_by_length(splits):
    means = {}
    for arm in formats.ARMS:
        texts = [formats.encode(formats.parse_exr(r["exr"]), arm) for r in splits["test"]]
        means[arm] = sum(len(t) for t in texts) / len(texts)
    assert means["exr"] > means["exr_short"] > means["dsl"]
    assert means["tagged"] < means["exr_short"]


def test_nested_not_over_complex_topping():
    exr = ("(ORDER (PIZZAORDER (NOT (COMPLEX_TOPPING (QUANTITY EXTRA ) (TOPPING CHEESE ) ) ) "
           "(NUMBER 1 ) ) )")
    node = formats.parse_exr(exr)
    for arm in formats.ARMS:
        assert formats.to_exr(formats.decode(formats.encode(node, arm), arm)) \
            == formats.to_exr(node)
    assert formats.to_dsl(node) == "p !+cheese 1"


def test_volume_two_tokens():
    exr = "(ORDER (DRINKORDER (NUMBER 1 ) (VOLUME 20 FLOZ ) (DRINKTYPE COKE ) ) )"
    node = formats.parse_exr(exr)
    assert formats.to_dsl(node) == "d 1 20floz coke"
    assert ladder.to_words(node) == "drink 1 20 floz coke"
    for arm in formats.ARMS:
        assert formats.to_exr(formats.decode(formats.encode(node, arm), arm)) \
            == formats.to_exr(node)


def test_decoders_reject_garbage():
    for arm in formats.ARMS:
        with pytest.raises(Exception):
            formats.decode("!!! not a tree !!!", arm)


def test_catalog_unambiguous():
    cat = formats.catalog()
    words = [e.lower() for ctor in formats.BARE for e in cat.ids[ctor]]
    assert len(words) == len(set(words))
    assert not {"p", "d", "pizza", "drink", "no", "style", "extra", "light"} & set(words)


@pytest.mark.slow
def test_subsets_deterministic_and_nested():
    """Subsets are prefixes of one seeded permutation, so sizes nest for a given seed."""
    assert data.train_subset(200, 0) == data.train_subset(200, 0)
    assert data.train_subset(200, 0) != data.train_subset(200, 1)
    assert data.train_subset(200, 0) == data.train_subset(1000, 0)[:200]


@pytest.mark.slow
def test_evaluation_duplicates_are_excluded_from_the_pool():
    """The full pool contains three rows identical to an eval utterance; they must never
    enter a subset, and the eval splits must be left untouched."""
    blocked = data.blocked_lines()
    assert blocked == {491345, 516460, 681985}
    assert data.pool_size() == 2_456_443
    assert len(data.load("dev")) == 348 and len(data.load("test")) == 1357


@pytest.mark.slow
def test_no_overlap_between_train_subsets_and_eval(splits):
    for size in (1000, 10000):
        for seed in (0, 1, 2):
            sub = data.train_subset(size, seed)
            for split in ("dev", "test"):
                ov = data.overlap(sub, splits[split])
                assert ov["exact"] == 0 and ov["normalized"] == 0, (size, seed, split, ov)


def test_pcfg_error_subset_size(splits):
    assert len(data.pcfg_error_subset(splits["test"])) == 434
    assert len(data.pcfg_error_subset(splits["dev"])) == 106


def test_metric_definitions():
    gold = "(ORDER (PIZZAORDER (NUMBER 2 ) (TOPPING HAM ) ) )"
    same = formats.to_dsl(formats.parse_exr(gold))
    res, _, _ = metrics.evaluate([same], [gold], "dsl")
    assert res["em"] == 1.0 and res["hallucinated_entity"] == 0.0

    wrong_n = formats.to_dsl(formats.parse_exr(
        "(ORDER (PIZZAORDER (NUMBER 3 ) (TOPPING HAM ) ) )"))
    res, _, _ = metrics.evaluate([wrong_n], [gold], "dsl")
    assert res["wrong_number"] == 1.0 and res["hallucinated_entity"] == 0.0

    invented = formats.to_dsl(formats.parse_exr(
        "(ORDER (PIZZAORDER (NUMBER 2 ) (TOPPING HAM ) (TOPPING TUNA ) ) )"))
    res, _, _ = metrics.evaluate([invented], [gold], "dsl")
    assert res["hallucinated_entity"] == 1.0 and res["wrong_number"] == 0.0

    res, _, _ = metrics.evaluate(["p 1 (((", ], [gold], "dsl")
    assert res["parse_validity"] == 0.0 and res["em"] == 0.0


def test_missing_and_merged_suborders():
    gold = ("(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING HAM ) ) "
            "(PIZZAORDER (NUMBER 1 ) (TOPPING TUNA ) ) )")
    merged = "p 1 ham tuna"
    res, _, _ = metrics.evaluate([merged], [gold], "dsl")
    assert res["missing_suborder"] == 1.0
    assert res["merged_suborder"] == 1.0


def test_eos_is_a_real_token():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from pizza_bench.config import BASE_MODEL
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    eos_id = tok.convert_tokens_to_ids(EOS)
    assert eos_id is not None and eos_id != tok.unk_token_id
    assert len(tok(EOS, add_special_tokens=False)["input_ids"]) == 1


def test_caps_cover_every_gold_target():
    """max_new_tokens must never truncate a gold-length output."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from pizza_bench.config import BASE_MODEL, load_caps
    try:
        caps = load_caps()
    except FileNotFoundError:
        pytest.skip("run python -m pizza_bench prepare first")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = data.load("dev") + data.load("test")
    for arm in formats.ARMS:
        texts = [formats.encode(formats.parse_exr(r["exr"]), arm) + EOS for r in rows]
        longest = max(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
        assert caps[arm] >= longest, f"{arm}: cap {caps[arm]} < longest {longest}"


def test_strip_eos():
    assert evaluate.strip_eos("p 1 ham<|im_end|>\nxyz") == "p 1 ham"
    assert evaluate.strip_eos("  p 1 ham  ") == "p 1 ham"


def test_score_file_recomputes_from_disk(tmp_path):
    rows = data.load("dev")[:20]
    outputs = [formats.to_dsl(formats.parse_exr(r["exr"])) for r in rows]
    res, preds, em = metrics.evaluate(outputs, [r["exr"] for r in rows], "dsl",
                                      n_tokens=[5] * len(rows))
    path = tmp_path / "dev_predictions.jsonl"
    evaluate.write_predictions(path, rows, outputs, preds, em, [5] * len(rows))
    again, back = evaluate.score_file(path, "dsl")
    assert again["em"] == res["em"] == 1.0
    assert len(back) == len(rows)
    assert back[0]["n_generated_tokens"] == 5


def test_grammar_covers_the_dsl():
    g = grammar.dsl_gbnf()
    assert '"!"' in g and '"^"' in g
    for word in ("ham", "thin_crust", "coke", "large"):
        assert f'"{word}"' in g


class _FakeTok:
    """Minimal stand-in for a real tokenizer."""
    padding_side = "right"
    pad_token = None
    eos_token = "<|endoftext|>"
    unk_token_id = 99999

    def convert_tokens_to_ids(self, t):
        return 248046 if t == EOS else self.unk_token_id

    def decode(self, *a, **k):
        return ""

    def __call__(self, text, **kw):
        return {"input_ids": [248046] if text == EOS else [1, 2, 3]}


class _FakeProcessor:
    """What Unsloth hands back for some models: no tokenizer API of its own."""
    def __init__(self):
        self.tokenizer = _FakeTok()


def test_normalize_tokenizer_unwraps_a_processor():
    train = pytest.importorskip("pizza_bench.train")
    tok = train.normalize_tokenizer(_FakeProcessor())
    assert isinstance(tok, _FakeTok)
    assert tok.padding_side == "left"
    assert tok.pad_token == tok.eos_token


def test_normalize_tokenizer_rejects_a_bad_eos():
    train = pytest.importorskip("pizza_bench.train")

    class NoEos(_FakeTok):
        def convert_tokens_to_ids(self, t):
            return self.unk_token_id

    with pytest.raises(RuntimeError, match="never stop"):
        train.normalize_tokenizer(NoEos())

    class SplitEos(_FakeTok):
        def __call__(self, text, **kw):
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="expected 1"):
        train.normalize_tokenizer(SplitEos())


def test_normalize_tokenizer_rejects_a_non_tokenizer():
    train = pytest.importorskip("pizza_bench.train")

    class Useless:
        pass

    with pytest.raises(TypeError, match="not a usable tokenizer"):
        train.normalize_tokenizer(Useless())


def test_fewshot_prompt_is_chat_turns():
    """Shots must be complete user/assistant turns, not one stuffed user turn.

    A single user turn makes a chat-tuned base model reply in prose instead of the target
    format, which produced 0% parse validity across all six arms.
    """
    from pizza_bench.__main__ import fewshot_prompt
    shots = [{"src": "two cokes", "exr": "(ORDER (DRINKORDER (NUMBER 2 ) (DRINKTYPE COKE ) ) )"},
             {"src": "one ham pizza", "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING HAM ) ) )"}]
    p = fewshot_prompt(shots, "dsl", "three sprites")

    assert p.count("<|im_start|>user") == 3
    assert p.count("<|im_start|>assistant") == 3
    # each shot's answer is inside an assistant turn and is terminated
    assert "d 2 coke" + EOS in p
    assert "p 1 ham" + EOS in p
    # the final turn is left open for the model to complete
    assert p.endswith("<think>\n\n</think>\n\n")
    assert p.count(EOS) == 2 + 3   # two answers plus three user-turn terminators


def test_fewshot_prompt_uses_the_arm_format():
    from pizza_bench.__main__ import fewshot_prompt
    shots = [{"src": "one ham pizza", "exr": "(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING HAM ) ) )"}]
    assert "p 1 ham" in fewshot_prompt(shots, "dsl", "x")
    assert "pizza 1 ham" in fewshot_prompt(shots, "words", "x")
    assert "(ORDER (PIZZAORDER" in fewshot_prompt(shots, "exr", "x")


def test_latency_resume_skips_measured_runs(tmp_path, monkeypatch):
    from pizza_bench import config, latency
    monkeypatch.setattr(config, "RUNS", tmp_path)
    monkeypatch.setattr(latency, "RUNS", tmp_path)
    assert latency.done_run_ids() == set()
    latency.append_result({"run_id": "dsl-n20000-s0-lr0.0001", "median_ms": 12.0})
    latency.append_result({"run_id": "exr-n20000-s0-lr0.0001", "median_ms": 45.0})
    assert latency.done_run_ids() == {"dsl-n20000-s0-lr0.0001", "exr-n20000-s0-lr0.0001"}
    # a truncated line from a crash must not lose the rest
    with open(latency.results_path(), "a") as f:
        f.write('{"run_id": "trunc"')
    latency.append_result({"run_id": "words-n20000-s0-lr0.0001", "median_ms": 11.0})
    assert "words-n20000-s0-lr0.0001" in latency.done_run_ids()


def test_assert_adapter_active_rejects_an_unapplied_adapter():
    torch = pytest.importorskip("torch")
    from pizza_bench import latency

    class Fake:
        def __init__(self, vals):
            self._p = [(f"base.layers.0.lora_B.default.weight", torch.tensor(v))
                       for v in vals]

        def named_parameters(self):
            return iter(self._p)

    with pytest.raises(RuntimeError, match="every lora_B is zero"):
        latency.assert_adapter_active(Fake([[0.0, 0.0]]), "somewhere")

    class NoLora:
        def named_parameters(self):
            return iter([("base.weight", torch.tensor([1.0]))])

    with pytest.raises(RuntimeError, match="no LoRA layers"):
        latency.assert_adapter_active(NoLora(), "somewhere")

    stats = latency.assert_adapter_active(Fake([[0.0], [0.3]]), "somewhere")
    assert stats == {"lora_layers": 2, "nonzero_lora_B": 1}
