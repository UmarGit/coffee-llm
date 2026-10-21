"""Every model-specific fact, in one place.

Adding a model means adding one entry here and nothing else. The prompt, the stop
token, the adapter targets and whether Unsloth can load it are all properties of
the model, not of the training code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    """Short key. Appears in run paths, so keep it filesystem safe."""

    hf_id: str
    label: str
    """How the paper names it."""

    prompt: str | None
    """Format string with one `{}` for the source utterance. None means derive it
    from the tokenizer's own chat template at load time, which is the safe default
    for a model whose template this project has not frozen."""

    eos: str
    """Generation stops here, and `check` refuses a model whose tokenizer does not
    map it to exactly one id."""

    target_modules: tuple | str
    """LoRA targets. The string "all-linear" lets PEFT resolve them, which is the
    only honest option for an architecture whose module names differ from a plain
    transformer's."""

    supports_unsloth: bool
    vocab_size: int | None = None
    """Recorded for the tokenizer-floor comparison. None means read it at load."""

    notes: str = ""


QWEN = Model(
    name="qwen",
    hf_id="Qwen/Qwen3.5-0.8B",
    label="Qwen3.5-0.8B",
    # Frozen. The trailing already-closed think block is what the café model was
    # trained on, and every evaluation in the paper uses it byte for byte.
    prompt="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    eos="<|im_end|>",
    target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    supports_unsloth=True,
    vocab_size=248_320,
    notes="Attention / linear-attention hybrid, 0.773B parameters.",
)

LFM2 = Model(
    name="lfm2",
    hf_id="LiquidAI/LFM2.5-350M",
    label="LFM2.5-350M",
    # Not frozen by this project, so it is derived from the tokenizer's chat
    # template rather than guessed, and `check` prints what it resolved to.
    prompt=None,
    eos="<|im_end|>",
    # 10 of its 16 blocks are convolution, not attention, so the Qwen target list
    # does not apply. "all-linear" is the nearest equivalent intervention and the
    # difference is declared rather than hidden.
    target_modules="all-linear",
    supports_unsloth=False,
    vocab_size=65_536,
    notes="10 convolution + 6 grouped-query-attention blocks, 350M parameters. "
    "Vocabulary is 3.8x smaller than Qwen's, which is the point of the "
    "comparison: it tests whether the tokenizer floor transfers. "
    "Officially nine languages, Russian not among them, so it is a study "
    "model only and never a candidate for the café parser.",
)

MODELS = {m.name: m for m in (QWEN, LFM2)}


def get(name):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    return MODELS[name]


def resolve_prompt(m, tok):
    """The prompt format string for this model, from the tokenizer when the model
    does not pin one. Asserts the template leaves exactly one substitution point."""
    if m.prompt is not None:
        return m.prompt
    sentinel = "\u0001SRC\u0001"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(sentinel) != 1:
        raise RuntimeError(
            f"{m.hf_id}: chat template put the source {rendered.count(sentinel)} times, "
            "expected once; pin Model.prompt explicitly instead"
        )
    # An opened-but-unclosed think block is the one prompt bug this project has
    # already paid for: the cafe model returned empty output for exactly this
    # reason, because the stop string fired before the answer. Refuse it here
    # rather than discover it as a run of zero parse validity.
    if "<think>" in rendered and "</think>" not in rendered:
        raise RuntimeError(
            f"{m.hf_id}: the generation prompt opens a think block it never closes "
            f"({rendered!r}). Generation would stop inside it. Pin Model.prompt with "
            "an already-closed block, as the Qwen entry does."
        )
    return rendered.replace("{", "{{").replace("}", "}}").replace(sentinel, "{}")
