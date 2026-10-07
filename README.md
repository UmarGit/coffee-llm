# coffee-llm

A fine-tune of **Qwen3.5-0.8B** that turns a spoken coffee order — a Russian or English
speech-to-text transcript — into a compact machine-readable string.

```
добрый день можно капучинку побольше          →  cap L
2 американо и 1 эспрессо                      →  ame x2;esp
can i get a large oat latte                   →  lat L oat
где у вас туалет                              →  ?
```

It runs locally through Ollama at ~80 ms per order and scores 48/49 on the eval set.

## Output format

Items separated by `;`, each one `drink [size] [milk] [syrup] [xN]`:

| Field | Values |
|---|---|
| drink | `cap` `lat` `ame` `esp` `fw` `raf` `fil` `coc` `mat` `ice` |
| size | `S` `M` `L` (omitted = unspecified) |
| milk | `oat` `almond` `coconut` `lactosefree` `nomilk` |
| syrup | `vanilla` `caramel` `hazelnut` |
| quantity | `x2`, `x3`, … (omitted = 1) |

`?` means *not an order, off-menu, or too vague*. It can appear on its own (`?`) or
alongside recognised items (`капучино и круассан` → `cap;?`).

The full vocabulary lives in [menu.py](src/coffee_parser/menu.py) and must stay in sync
with the dataset generator in the training notebook.

## Setup

Python is pinned to **3.12**. The parser itself needs no third-party packages.

```bash
uv sync
```

Download the weights from Hugging Face into `models/` (git-ignored) and build the
Ollama model:

```bash
mkdir -p models
huggingface-cli download umarsheikh303/coffee-parser-gguf coffee-q8_0.gguf --local-dir models
ollama create coffee -f ollama/Modelfile
```

> Don't add the `hf` CLI package as a project dependency — it pulls huggingface-hub 2.x,
> which conflicts with transformers.

## Calling the model

**Thinking must be off, or the model returns empty output.** Qwen3.5 is a thinking model,
so Ollama enables thinking by default; see [Why thinking breaks it](#why-thinking-breaks-it)
below. Two ways to avoid it:

- `"raw": true` with the exact training prompt — what
  [ollama.py](src/coffee_parser/ollama.py) does.
- `"think": false` on a normal (templated) call — identical output and token counts.
  This is also what `ollama run coffee --think=false "…"` does.

Use the client rather than rolling your own call:

```python
from coffee_parser import generate, to_json

out, ms, tokens = generate("латте на овсяном и маленький американо")
# 'lat oat;ame S'
to_json(out)
# {'items': [{'drink': 'latte', 'qty': 1, 'milk': 'oat'},
#            {'drink': 'americano', 'qty': 1, 'size': 'S'}], 'unclear': False}
```

The prompt is `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
with stops `<|im_end|>`, `<|endoftext|>`, `</think>`. Don't change it without retraining.

### Why thinking breaks it

The training prompt ends with an already-closed, empty think block, so the model expects
to start generating the answer immediately. With thinking enabled, Ollama renders the
prompt *without* that prefilled block (15 prompt tokens instead of 17), ending at
`<|im_start|>assistant\n` and expecting the model to open its own think block. The model
duly emits `<think>\n\n</think>\n\ncap` — but `PARAMETER stop </think>` halts generation
before the answer, and the `<think>\n\n` that survives is routed to the thinking channel,
so `response` comes back empty.

Measured on `ollama 0.34.4`, prompt `ну давайте капучино`:

| Call | Prompt tokens | Generated | `response` |
|---|---|---|---|
| `raw: true` | 17 | 2 | `cap` |
| `think: false` | 17 | 2 | `cap` |
| default (thinking on) | 15 | 1 | *(empty)* |

Removing the `</think>` stop is *not* a fix: thinking mode then returns
`</think>\n\ncap`, with the stray closer in `response` rather than parsed out.

## Running the eval and the benchmark

```bash
uv run python scripts/coffee_eval.py     # accuracy, one pass, prints every failure
uv run python scripts/benchmark.py       # latency, three passes, median + p90
```

The eval set (49 hand-written cases in [testset.py](src/coffee_parser/testset.py)) was
written separately from the training generator, and includes slang the generator never
produces (`флэтик`, `капучиночку`) to test generalisation.

Current: **48/49 (98%)**, median ~80 ms, ~3 output tokens per order.

## Training

Training runs in Colab (T4) with Unsloth + LoRA on a synthetic dataset, 1 epoch. See
[coffee_order_finetune.ipynb](notebooks/coffee_order_finetune.ipynb); it exports a Q8_0
GGUF and uploads it to `umarsheikh303/coffee-parser-gguf`. Strip cell outputs before
committing.

The dataset generator lives in [generator.py](src/coffee_parser/generator.py), shared with the
research notebook. `production_dataset()` reproduces exactly the 4000 examples this model was
trained on, pinned by a test against a frozen copy of the original inline code.

## Research: NoRA vs LoRA

[nora_vs_lora.ipynb](notebooks/nora_vs_lora.ipynb) compares **LoRA**, **NoRA-init** and **NoRA**
on this task, for a paper. The method is from Jiale Kang, Ziyin Yue, Zheng Zhan, Yangyi Huang and
Weiyang Liu, *Normalized Low-Rank Adaptation*, [arXiv:2608.31036](https://arxiv.org/abs/2608.31036)
(2026) — the full title matters, since other unrelated papers also use the name "NoRA". It
constrains each column of `lora_A.weight` (shape `r × in_features`) to unit L2 norm along the rank
dimension, so a column is a direction and all magnitude sits in `lora_B`.

- **NoRA-init** — normalize once after initialization, then train as plain LoRA.
- **NoRA** — keep the constraint throughout training.

```bash
pip install -e ".[research,dev]"
pip install "git+https://github.com/Joluck/NoRA.git#subdirectory=peft"   # install LAST
pytest
```

The official [PEFT fork](https://github.com/Joluck/NoRA) is used (`use_nora=True` /
`use_nora="init"`), verified working with transformers 5.19 + TRL 1.14 + peft 0.19.2.dev0.
Install it last: anything that resolves `peft` from PyPI afterwards silently replaces it, which
the notebook asserts against. [nora.py](src/coffee_parser/nora.py) also carries an independent
`torch.nn.utils.parametrize` implementation that the tests cross-check against the fork and
against the closed form `ΔW = α/r · B · Norm(A)`.

No Unsloth in this study: transformers + TRL + PEFT for every method, identical stack and
settings except the `LoraConfig`. Loss is on the answer only, via TRL's prompt/completion
format — which also avoids the `<think>` masking bug above.

Four things found while implementing it, all of which affect how results should be read:

- **Continuous NoRA does not constrain the stored parameter.** The fork normalizes inside
  `forward()` and `get_delta_weight()`, so `lora_A.weight` itself drifts during training.
  Invariant checks must test the effective matrix `A/‖A‖`; a check on the raw parameter fails
  against the reference implementation.
- **The epsilon is additive, not a clamp:** `A / (‖A‖ + 1e-6)`. Column norms land at
  `1 − O(1e-5)` in fp32, never exactly 1.
- **Under bf16, NoRA-init is only approximately normalized.** The fork normalizes `lora_A`
  and then calls `_move_adapter_to_device_of_base_layer`, which round-trips the weights
  through the base dtype. With a bf16 base the columns come out unit-norm to about `2e-3`,
  stored in an fp32 tensor — so `lora_A.dtype` reports fp32 while the values carry bf16
  rounding, and dtype-blind tolerances fail every bf16 run. `norm_atol()` derives the
  tolerance from the base dtype as well as the adapter dtype. The rounding (≈2e-3) is three
  orders of magnitude larger than the epsilon (1e-6), so on GPU it is the only term that
  matters.
- **BIMI is not implemented in the published fork.** `bimi_init()` is
  `with torch.no_grad(): pass` with the block-identity code commented out, so
  `init_lora_weights="bimi"` silently falls back to default LoRA init. There is no BIMI arm.
- **The paper's LoRA baseline uses `alpha = 2r` while NoRA uses `alpha = r`**, so scaling is
  confounded with normalization. The grid adds a `lora_alpha_r` arm (LoRA at `alpha = r`) to
  separate the two.

The notebook's final cell lists the remaining caveats; the headline one is that every split comes
from the same generator, so the `hard` split raises difficulty but not realism.

## Known issues

- **Training marks the `<think>` block as part of the answer.** `train_on_responses_only`
  is configured with `response_part="<|im_start|>assistant\n"`, so loss is computed over the
  empty `<think>\n\n</think>\n\n` block as well as the real answer. Next run should use
  `response_part="</think>\n\n"`. This is directly observable: prompted with a bare
  `<|im_start|>assistant\n` and no stop strings, the model reproduces the whole block —
  `<think>\n\n</think>\n\ncap` — which is what makes Ollama's thinking mode misfire.
- **"known item + off-menu item" is sometimes refused outright.** `лавандовый раф` should be
  `raf;?` but comes back as just `?` — the model generalises the off-menu modifier to the
  whole order instead of splitting it. This is the one remaining eval failure.
- **The eval set and the training generator share one author's assumptions.** Both were
  written by the same person imagining how people order, so 98% likely overstates real
  performance. Real speech-to-text transcripts are needed before trusting the number.

## Layout

```
src/coffee_parser/   menu.py, parsing.py, ollama.py, testset.py   production
                     generator.py                                 shared data generator
                     nora.py, experiment.py                        research only (needs torch)
scripts/             coffee_eval.py (accuracy), benchmark.py (latency)
notebooks/           coffee_order_finetune.ipynb (production), nora_vs_lora.ipynb (research)
tests/               pytest suite; data/ holds the frozen generator reference
ollama/              Modelfile
models/              weights, git-ignored
runs/                research results, curves, figures (git-ignored)
```
