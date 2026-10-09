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

## Benchmark: PIZZA

[pizza_experiments.ipynb](notebooks/pizza_experiments.ipynb) evaluates this project's approach —
Qwen3.5-0.8B + LoRA SFT + a compact output format — on the public
[PIZZA](https://github.com/amazon-science/pizza-semantic-parsing-dataset) semantic-parsing
benchmark (Arkoudas et al., 2022).

> **Licensing.** PIZZA is **CC-BY-NC-4.0**. The dataset, anything derived from it, and any model
> trained on it are non-commercial: they must not be used by the production café parser and must
> not be committed. `data/external/`, `data/pizza/` and `pizza-runs/` are git-ignored.

### Setup

```bash
git clone --depth 1 https://github.com/amazon-science/pizza-semantic-parsing-dataset.git \
  data/external/pizza
uv run --extra dev --extra pizza pytest tests/test_pizza_formats.py
```

The round-trip tests are a gate: they must pass before any experiment is meaningful. They check
`EXR → DSL → EXR` and `EXR → JSON → EXR` on all of dev (348), all of test (1357) and a 50k train
sample, requiring the reconstruction to be *character-identical* to the input.

### The compact DSL

| token | meaning |
|---|---|
| `p` / `d` | PIZZAORDER / DRINKORDER, joined by ` ; ` |
| digits | `NUMBER` |
| `20floz`, `500ml`, `2liter` | `VOLUME amount unit` |
| bare word | `TOPPING` \| `SIZE` \| `STYLE` \| `DRINKTYPE` \| `CONTAINERTYPE`, resolved by catalog |
| `^x` | `STYLE` |
| `+x` / `~x` | `COMPLEX_TOPPING` with `QUANTITY EXTRA` / `LIGHT` |
| `!x` | `NOT(x)`, composes — `!+cheese` is NOT over a complex topping |

```
EXR  (ORDER (PIZZAORDER (NOT (TOPPING PESTO ) ) (NUMBER 2 ) (SIZE MEDIUM )
             (COMPLEX_TOPPING (QUANTITY EXTRA ) (TOPPING CHEESE ) ) (TOPPING TUNA ) ) )
DSL  p !pesto 2 medium +cheese tuna
```

Entity ids are the lowercased EXR symbols (a verified bijection) and the vocabulary comes from
the dataset's own `utils/catalogs/`, which covers every entity occurring in all three splits.
Mean output length on test, in Qwen3.5 tokens: **EXR 45.0, JSON 46.1, DSL 10.8** — the DSL is
4.15× shorter than EXR.

### Reproducing each table

| Table / figure | How |
|---|---|
| Output tokens per format | notebook §3 → `tables/output_tokens.csv` |
| Few-shot baseline (condition A) | notebook §4 |
| LR selection (dev only) | notebook §5 |
| Data-efficiency curve (condition B) | notebook §6, §9 → `figures/data_efficiency_dev.pdf` |
| Main test table vs published baselines | notebook §7, §9 → `tables/main.csv` / `.tex` |
| Slot P/R/F1 and hallucination rates | notebook §9 → `tables/slot_prf.csv`, `hallucination.csv` |
| Grammar-constrained decoding (condition C) | notebook §8, then `scripts/pizza_latency.py --grammar` |
| On-device latency | `uv run python scripts/pizza_latency.py --backend ollama --model <m> --format dsl` |
| Error taxonomy | notebook §10 → `tables/error_taxonomy.csv` |

Exact match always uses the dataset's own `semantic_matchers.is_unordered_exact_match`; it is
never reimplemented. Dev drives every selection decision and test is evaluated once per final
configuration. Published baselines (PCFG 68.02, BERT 64.08, BART-Large EXR 78.56, BART on the
PCFG-error subset 59.49) are cited, not re-run — and they used the full 2.46M-utterance train
set, while these runs use 1k/5k/20k from the 10% split.

## Layout

```
src/coffee_parser/   production parser: menu, parsing, ollama client, testset, generator
src/pizza_bench/     PIZZA benchmark: data, formats, metrics, grammar, experiment, report
scripts/             coffee_eval.py, benchmark.py, pizza_latency.py
notebooks/           coffee_order_finetune.ipynb, pizza_experiments.ipynb
tests/               pytest suite; data/ holds the frozen generator reference
ollama/              Modelfile
models/              weights, git-ignored
data/external/       PIZZA dataset, git-ignored (CC-BY-NC-4.0)
pizza-runs/          PIZZA results, git-ignored
```
