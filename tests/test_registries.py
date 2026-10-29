"""The model and task registries, and the coffee task's round trip.

These are the invariants the harness relies on but that no single run would
catch: a model whose prompt has no substitution point, a task whose arms do not
round-trip, two models sharing a run directory.
"""

import pytest

bench = pytest.importorskip("bench")
from bench import config, models, tasks


def test_every_model_is_self_consistent():
    for name, m in models.MODELS.items():
        assert m.name == name, "registry key must match the model's own name"
        assert m.hf_id and m.label
        assert m.eos
        assert isinstance(m.target_modules, (tuple, str))
        if m.prompt is not None:
            assert m.prompt.count("{}") == 1, f"{name}: prompt needs one slot"


def test_pinned_prompts_survive_formatting():
    """A prompt with a stray brace would raise only at the first training step."""
    for m in models.MODELS.values():
        if m.prompt is not None:
            assert "SRC" in m.prompt.format("SRC")


def test_every_task_is_self_consistent():
    for name in tasks.NAMES:
        t = tasks.get(name)
        assert t.name == name
        assert t.arms, f"{name}: needs at least one arm"
        assert t.default_arm in t.arms
        assert set(t.ladder_arms) <= set(t.arms)
        for a, b in t.matched_pairs:
            assert a in t.arms and b in t.arms
        assert t.default_sizes, f"{name}: needs a size ladder"
        assert t.lr_select_size in t.default_sizes, (
            f"{name}: the LR is tuned at a size that is not in the grid"
        )


def test_run_directories_are_isolated_per_model_and_task():
    seen = {}
    for m in models.MODELS:
        for t in tasks.NAMES:
            config.select(m, t)
            d = config.model_task_dir()
            assert d not in seen, f"{(m, t)} collides with {seen.get(d)}"
            seen[d] = (m, t)
            # caps depend on the tokenizer, so they must never be shared
            assert config.caps_path().parent == d
            assert config.lrs_path().parent == d
    assert len(seen) == len(models.MODELS) * len(tasks.NAMES)


def test_config_defaults_come_from_the_selected_task():
    for name in tasks.NAMES:
        config.select("qwen", name)
        t, cfg = tasks.get(name), config.Config.for_active()
        assert cfg.arms == t.arms
        assert cfg.sizes == t.default_sizes
        assert cfg.determinism_arm == t.default_arm


# ---------- the coffee task ----------


def test_coffee_arms_round_trip_losslessly():
    config.select("qwen", "coffee")
    t = tasks.get("coffee")
    from bench.tasks.coffee import formats

    splits = t.load_splits()
    rows = splits["dev"] + splits["test"] + t.train_rows(500, 0)
    for r in rows:
        for arm in t.arms:
            encoded = formats.encode(r["output"], arm)
            assert formats.decode(encoded, arm) == r["output"], (
                f"{arm}: {r['output']!r} -> {encoded!r} did not come back"
            )


def test_coffee_words_arm_actually_differs_from_compact():
    """If the two arms produced the same string the matched pair would be vacuous."""
    from bench.tasks.coffee import formats

    differs = 0
    for compact in ("cap L oat", "fw M x3", "?", "raf S coconut caramel"):
        if formats.encode(compact, "words") != compact:
            differs += 1
    assert differs == 4


def test_coffee_rejects_output_outside_the_menu():
    """Parse validity has to be a real measurement, so junk must not decode."""
    from bench.tasks.coffee import formats

    for junk in ("", "zzz", "cappuccino enormous", "cap XL", "unicorn latte"):
        assert (
            formats.decode(junk, "words") is None
            or formats.decode(junk, "compact") is None
        )


def test_coffee_training_rows_never_touch_evaluation():
    config.select("qwen", "coffee")
    t = tasks.get("coffee")
    splits = t.load_splits()
    train = t.train_rows(300, 0)
    held = {r["input"] for r in splits["dev"]} | {r["input"] for r in splits["test"]}
    assert not ({r["input"] for r in train} & held)


def test_coffee_scoring_reports_the_shared_keys():
    """The report and the early-stopping rule read these names for every task."""
    config.select("qwen", "coffee")
    t = tasks.get("coffee")
    rows = t.load_splits()["test"][:6]
    outputs = [t.target(r, "compact") for r in rows]
    res, preds, em = t.score(outputs, rows, "compact", n_tokens=[2] * 6, cap=8)
    for key in ("em", "parse_validity", "em_given_valid", "mean_generated_tokens"):
        assert key in res, key
    assert res["em"] == 1.0 and all(em)
    assert len(preds) == len(rows)


def test_resolve_prompt_rejects_an_unclosed_think_block():
    """The failure that returned empty output from the cafe model: a prompt that
    opens a think block the stop string then fires inside."""

    class FakeTok:
        def __init__(self, rendered):
            self._r = rendered

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return self._r.replace("SRC", msgs[0]["content"])

    m = models.Model(
        name="x",
        hf_id="x/y",
        label="X",
        prompt=None,
        eos="<|im_end|>",
        target_modules="all-linear",
        supports_unsloth=False,
    )

    with pytest.raises(RuntimeError, match="think block"):
        models.resolve_prompt(
            m,
            FakeTok(
                "<|im_start|>user\nSRC<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ),
        )

    # a closed block, or none at all, is fine
    for rendered in (
        "<|im_start|>user\nSRC<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n\nSRC",
    ):
        assert "{}" in models.resolve_prompt(m, FakeTok(rendered))


def test_resolve_prompt_requires_exactly_one_substitution_point():
    class Twice:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            c = msgs[0]["content"]
            return f"{c} and again {c}"

    m = models.Model(
        name="x",
        hf_id="x/y",
        label="X",
        prompt=None,
        eos="<|im_end|>",
        target_modules="all-linear",
        supports_unsloth=False,
    )
    with pytest.raises(RuntimeError, match="expected once"):
        models.resolve_prompt(m, Twice())


def test_importing_the_cafe_task_never_imports_pizza_code():
    """The CC-BY-NC boundary, enforced rather than documented.

    The cafe parser ships. If importing it pulled in the PIZZA task, a dependency
    on non-commercial code would reach production without anyone noticing, so
    this runs in a clean interpreter and fails if any pizza module loads.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import bench.tasks.coffee as c;"
        "assert c.TASK.name == 'coffee';"
        "assert c.parse_compact('cap L oat');"
        "leaked = [m for m in sys.modules if 'pizza' in m];"
        "print('LEAKED:' + repr(leaked) if leaked else 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "CLEAN", out.stdout


def test_the_cafe_task_uses_only_the_standard_library():
    """It ships, so a third-party import here would become a deployment dependency."""
    import pathlib
    import re

    third_party = {"torch", "transformers", "peft", "trl", "datasets", "unsloth",
                   "matplotlib", "anytree", "numpy", "requests", "scipy"}
    root = pathlib.Path(__file__).resolve().parents[1] / "src/bench/tasks/coffee"
    offenders = []
    for path in sorted(root.glob("*.py")):
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w.]*)", line)
            if m and m.group(1).split(".")[0] in third_party:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, offenders


def test_config_for_active_is_what_the_report_filters_on():
    """`Config()` has empty defaults now; the real sizes come from the task.

    report.primary_runs() filtered on `Config().sizes`, which silently became the
    empty set after the registry refactor, so every run was rejected and the
    report said "no completed grid runs" with 90 complete runs on disk.
    """
    config.select("qwen", "pizza")
    assert config.Config().sizes == (), "bare defaults are intentionally empty"
    assert config.Config.for_active().sizes == tasks.get("pizza").default_sizes

    from bench import report
    import inspect
    src = inspect.getsource(report.primary_runs)
    assert "Config.for_active()" in src, \
        "primary_runs must take sizes from the selected task, not the bare dataclass"
