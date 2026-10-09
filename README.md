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

The dataset generator lives in [generator.py](src/coffee_parser/generator.py).
`production_dataset()` reproduces exactly the 4000 examples this model was trained on, pinned by
a test against a frozen copy of the original inline code.

```bash
uv run --extra dev pytest
```

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
src/coffee_parser/   menu.py, parsing.py, ollama.py, testset.py, generator.py
scripts/             coffee_eval.py (accuracy), benchmark.py (latency)
notebooks/           coffee_order_finetune.ipynb
tests/               pytest suite; data/ holds the frozen generator reference
ollama/              Modelfile
models/              weights, git-ignored
```
