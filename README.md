# Fewer Tokens, Fewer Silent Errors

**Output format design for small language models in real time order parsing.**

Umar Ahmed, Maxim Duanev. Paper source in [paper/](paper/), pre-registered protocol in
[PROTOCOL.md](PROTOCOL.md).

A small language model is cheap to fine tune and fast to serve, but it hallucinates, and in
a structured output setting a hallucination is not an obviously broken string. It is a well
formed tree describing an order nobody placed. This repository asks whether the **output
format** can be designed to reduce that risk, and ships the café ordering system that came
out of the answer.

## The finding

We fine tune Qwen3.5-0.8B with LoRA on the public
[PIZZA](https://github.com/amazon-science/pizza-semantic-parsing-dataset) semantic parsing
benchmark using six lossless encodings of the same target tree, measured over 36 runs on
1,357 test utterances.

**Shortening the output does not by itself improve accuracy.** At 1000 training examples
every ten tokens removed costs about four points of exact match. The ladder is not monotone:
`exr_short` at 34.0 tokens scores 0.591, well below `exr` at 45.4 tokens and 0.833.

**What matters is whether the surface form is made of natural words.** The two matched token
pairs isolate this. `dsl` and `words` sit at effectively the same length, 12.5 against 11.9
tokens, yet differ by 0.075 exact match, which is above the measured rerun noise floor of
0.043. Length is held constant; only the surface form changes.

| arm | tokens | exact match | parse validity | hallucinated | silent error | latency |
|---|---|---|---|---|---|---|
| `exr` (native) | 45.4 | **0.833** | 0.996 | 0.158 | 0.163 | 1213 ms |
| `json` | 44.5 | 0.759 | 0.904 | 0.145 | 0.146 | 1222 ms |
| `exr_short` | 34.0 | 0.591 | 0.764 | 0.169 | 0.174 | 927 ms |
| `tagged` | 22.7 | 0.807 | 0.947 | 0.132 | 0.140 | 651 ms |
| `dsl` | 12.5 | 0.755 | 0.894 | 0.109 | 0.139 | 346 ms |
| `words` | **11.9** | **0.830** | 0.900 | **0.036** | **0.070** | **322 ms** |

A word based encoding matches the native s-expression on accuracy (0.830 against 0.833,
inside the noise floor) while using **3.8x fewer tokens**, running **3.8x faster** at batch
size one, producing **4.4x fewer hallucinated entities**, and **halving** the rate of errors
that parse cleanly but are wrong.

### Silent error rate

The quantity we argue is the right objective. An output that fails to parse is caught by the
consuming system and can be retried. An output that parses cleanly and is wrong is not
caught by anything. Silent error rate is parse validity minus exact match, and it is
reported next to both everywhere rather than inferred.

`words` cuts it from 0.163 to 0.070 against the native format. Note that `exr` has the best
parse validity of any arm, 0.996, and still the second worst silent error rate: validity and
safety are not the same thing.

### Why the ladder stops at about 10 tokens

A seventh encoding using short opaque entity codes (`p !p5 2 m3 +c2 t2`) has 16.4 characters
per utterance against the DSL's 34.9, less than half, and yet costs **more** tokens, not
fewer. Codes outside the BPE vocabulary fragment into several tokens each. The achievable
floor is set by the tokenizer, not by string length, which is why no arm sits below about ten
tokens, and it is part of why natural words win: they are what the vocabulary is built from.
The encoding was measured during design and dropped, so it is not one of the six arms.

---

## The deployed system

The café parser is the finding applied: a fine-tune of Qwen3.5-0.8B that turns a spoken
order, a Russian or English speech-to-text transcript, into a compact string at about three
output tokens per order.

```
добрый день можно капучинку побольше          →  cap L
2 американо и 1 эспрессо                      →  ame x2;esp
can i get a large oat latte                   →  lat L oat
где у вас туалет                              →  ?
```

It scores 48/49 on a held-out eval set at a median of about 80 ms per order through Ollama
on an Apple M1, and runs entirely on an iPad with no network call.

### Output format

Items separated by `;`, each one `drink [size] [milk] [syrup] [xN]`:

| Field | Values |
|---|---|
| drink | `cap` `lat` `ame` `esp` `fw` `raf` `fil` `coc` `mat` `ice` |
| size | `S` `M` `L` (omitted = unspecified) |
| milk | `oat` `almond` `coconut` `lactosefree` `nomilk` |
| syrup | `vanilla` `caramel` `hazelnut` |
| quantity | `x2`, `x3`, … (omitted = 1) |

`?` means *not an order, off-menu, or too vague*. It can appear on its own (`?`) or alongside
recognised items (`капучино и круассан` → `cap;?`).

The full vocabulary lives in [menu.py](src/bench/tasks/coffee/menu.py) and must stay in sync with
the dataset generator.

### Setup

Python is pinned to **3.12**. The parser itself needs no third-party packages.

```bash
uv sync
mkdir -p models
huggingface-cli download umarsheikh303/coffee-parser-gguf coffee-q8_0.gguf --local-dir models
ollama create coffee -f ollama/Modelfile
```

> Do not add the `hf` CLI package as a project dependency. It pulls huggingface-hub 2.x,
> which conflicts with transformers.

### Calling the model

**Thinking must be off, or the model returns empty output.** Qwen3.5 is a thinking model, so
Ollama enables thinking by default. Two ways to avoid it:

- `"raw": true` with the exact training prompt, which is what
  [ollama.py](src/bench/tasks/coffee/ollama.py) does.
- `"think": false` on a normal templated call. Identical output and token counts, and it is
  what `ollama run coffee --think=false "…"` does.

Use the client rather than rolling your own call:

```python
from bench.tasks.coffee import generate, to_json

out, ms, tokens = generate("латте на овсяном и маленький американо")
# 'lat oat;ame S'
to_json(out)
# {'items': [{'drink': 'latte', 'qty': 1, 'milk': 'oat'},
#            {'drink': 'americano', 'qty': 1, 'size': 'S'}], 'unclear': False}
```

The prompt is `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
with stops `<|im_end|>`, `<|endoftext|>`, `</think>`. Do not change it without retraining.
The client and the paper's evaluations use byte-identical templates and stop strings, so the
deployed model sees the input distribution it was fitted on.

#### Why thinking breaks it

The training prompt ends with an already-closed, empty think block, so the model expects to
start generating the answer immediately. With thinking enabled, Ollama renders the prompt
*without* that prefilled block (15 prompt tokens instead of 17), ending at
`<|im_start|>assistant\n` and expecting the model to open its own think block. The model duly
emits `<think>\n\n</think>\n\ncap`, but `PARAMETER stop </think>` halts generation before the
answer, and the `<think>\n\n` that survives is routed to the thinking channel, so `response`
comes back empty.

Measured on `ollama 0.34.4`, prompt `ну давайте капучино`:

| Call | Prompt tokens | Generated | `response` |
|---|---|---|---|
| `raw: true` | 17 | 2 | `cap` |
| `think: false` | 17 | 2 | `cap` |
| default (thinking on) | 15 | 1 | *(empty)* |

Removing the `</think>` stop is **not** a fix: thinking mode then returns `</think>\n\ncap`,
with the stray closer in `response` rather than parsed out.

### Eval and benchmark

The eval set is 49 hand-written cases in
[testset.py](src/bench/tasks/coffee/testset.py), written separately from the training
generator and including slang the generator never produces (`флэтик`, `капучиночку`) to
test generalisation. Current: **48/49 (98%)**.

Two ways to measure the model, each on the path it actually describes:

```bash
python -m bench test --task coffee     # the training path, on a GPU
```

and the **Бенчмарк** button in the iPad app for the shipped GGUF, which is the only
measurement that describes what a customer waits for. See
[On-device deployment](#on-device-deployment).

### Training

Training produces a Q8_0 GGUF, published to `umarsheikh303/coffee-parser-gguf`. The dataset
generator is [generator.py](src/bench/tasks/coffee/generator.py); `production_dataset()`
reproduces exactly the 4000 examples this model was trained on, pinned by a test against a
frozen copy of the original inline code.

### On-device deployment

The booth client is a SwiftUI iPad app at `../loop-coffee-app`, linking llama.cpp as an
`xcframework` and offloading every layer to the Metal GPU. The whole parser is a 795 MiB
model file beside an 11 MiB runtime, with a 512 token context that the compact format fits
comfortably and the equivalent JSON encoding would not at the longest orders. A format chosen
for token count at training time therefore also decides whether the model fits a small
on-device context at serving time.

The app measures itself. Triple-tap the wordmark in the sidebar to reveal the debug
affordances, then use **Бенчмарк** in the Orders toolbar. It runs ten representative orders
three times through the same `OrderParser` the booth uses, so the numbers describe the
shipped path rather than a parallel engine loaded for measuring, and one warm-up pass is
discarded first. Staff never see the button.

The sheet reports device and OS, model file and size, context window, cold load time, peak
resident memory, median, p90 and mean latency, tokens per order, tokens per second, and every
model output so you can see what it actually answered. **Копировать** puts all of it on the
clipboard, together with ready-made `\renewcommand{\pad*}{…}` lines and
`\renewcommand{\estnote}{}` to paste into [paper.tex](paper/paper.tex), which is the only
place in the paper still holding projections rather than measurements.

Cold load time is recorded at launch by `prepareParser()`, not by the benchmark: by the time
the button is tapped the model is already resident. If the app was already warm the row is
omitted rather than reported wrongly.

### Known issues

- **Training marks the `<think>` block as part of the answer.** `train_on_responses_only` is
  configured with `response_part="<|im_start|>assistant\n"`, so loss is computed over the
  empty `<think>\n\n</think>\n\n` block as well as the real answer. The next run should use
  `response_part="</think>\n\n"`. This is directly observable: prompted with a bare
  `<|im_start|>assistant\n` and no stop strings, the model reproduces the whole block, which
  is what makes Ollama's thinking mode misfire.
- **"known item + off-menu item" is sometimes refused outright.** `лавандовый раф` should be
  `raf;?` but comes back as just `?`; the model generalises the off-menu modifier to the whole
  order instead of splitting it. This is the one remaining eval failure.
- **The eval set and the training generator share one author's assumptions.** Both were
  written by the same person imagining how people order, so 98% likely overstates real
  performance. Real speech-to-text transcripts are needed before trusting the number.

---

## Reproducing the study

> **Licensing.** PIZZA is CC-BY-NC-4.0. The data, any predictions derived from it and any
> adapter trained on it are non-commercial, must not be used by the production café parser,
> and must not be committed. `data/external/`, `runs/` and adapters are git-ignored.

### Design

Six lossless encodings of the identical semantic tree. Four form a length ladder; two are
matched-token controls that hold length roughly constant while changing format family.

| arm | role | example |
|---|---|---|
| `exr` | ladder, dataset native | `(ORDER (PIZZAORDER (NUMBER 2 ) (SIZE MEDIUM ) ... ) )` |
| `json` | control for `exr` | `{"orders":[{"type":"p","attrs":[...]}]}` |
| `exr_short` | ladder | `(O (P (N 2) (S MEDIUM) ...))` |
| `tagged` | ladder | `pizza n=2 size=medium extra=cheese topp=tuna` |
| `dsl` | ladder | `p 2 medium +cheese tuna` |
| `words` | control for `dsl` | `pizza 2 medium extra cheese tuna` |

Comparing three unrelated formats would confound output length with format family, so the
ladder makes length a continuous regressor and the two matched pairs isolate the confound.
That isolation is what produced the result: the matched pair, not the ladder, is what shows
surface form rather than length is driving it.

All six round-trip to byte-identical EXR on all of dev, all of test and a 50,000 row train
sample, checked by `python -m bench check` and by the test suite.

### What the axes mean

**Pool size** is *not* amount of data trained on. The step budget is fixed at 4000 steps of
batch 16, so every run sees at most 64,000 samples regardless of pool size; at 1M the model
sees 6% of the pool and at 2.46M about 3%. The axis is **pool diversity at fixed compute**,
and it answers whether 64,000 samples drawn from a large pool beat many epochs over a small
one. Runs record both the pool size and the samples actually seen, so the distinction is
never implicit.

**Exact match** always comes from the dataset's own `is_unordered_exact_match`; it is never
reimplemented. Exact match and parse validity move together closely, and accuracy conditional
on a well-formed tree is nearly constant across arms, learning rates and seeds (0.75 to
0.86). Differences between arms are therefore mostly differences in how reliably the model
emits well-formed output. Parse validity sits next to exact match everywhere and conditional
accuracy is reported as a third column, so this is visible rather than hidden.

### Running it

```bash
make push                  # send code to the server (tar over ssh, no rsync needed)

# on the server
uv sync --group gpu --extra pizza
make prepare && make check && make smoke
# every stage for one model and task, in order, in one session
./scripts/tmux.sh all qwen pizza 3
./scripts/tmux.sh all lfm2 pizza 3     # the tokenizer replication
./scripts/tmux.sh all qwen coffee 3    # the cafe task

make fetch                 # back on the Mac: pull runs/, adapters skipped
make paper                 # rebuild figures, tables and paper.pdf
```

`make help` lists everything and `make help-server` lists the server side in order. Copy
`server.conf.example` to `server.conf` (git-ignored) and set `HOST`, `PORT`, `KEY`, `REMOTE`
and `GPUS`; every script and tmux session reads it, and each value can be overridden by an
environment variable of the same name.

Every stage is **resumable**: runs with a complete `metrics.json` are skipped, so Ctrl-C and
restart is always safe. `./scripts/tmux.sh status` shows live sessions and progress, `tail`
follows every shard log, and `stop` kills them all.

The stages, individually:

Every stage takes `--model qwen|lfm2` and `--task pizza|coffee`, defaulting to
`qwen pizza`, and writes to `runs/<model>/<task>/`:

```bash
python -m bench prepare      # fetch data, compute max_new_tokens per arm
python -m bench check        # round-trip, overlap, cap coverage, EOS id, environment
python -m bench smoke        # n=200, 100 steps, full pipeline, under 10 minutes
python -m bench sweep        # LR per arm at n=100k, seeds 0, 1, 2
python -m bench grid         # 6 arms x 5 pool sizes x 3 seeds
python -m bench determinism  # one cell retrained, to size the noise floor
python -m bench test         # aggregation only, nothing is retrained
python -m bench grammar      # constrained decoding comparison
python -m bench fewshot      # no-finetune reference
python -m bench latency      # batch size 1, sharded across GPUs
python -m bench report       # tables, figures, summary.md
```

`max_new_tokens` is computed per arm as the longest target across train, dev and test times
1.25, so a gold-length output can never be truncated: EXR 173, JSON 183, `exr_short` 132,
`tagged` 90, DSL and `words` 53. Every run logs the share of generations that reach the cap
and warns above 1%.

The server has no `rsync`, so both transfer scripts use tar over ssh, needing only `ssh`,
`tar` and `gzip`. `push_code.sh` sends an explicit allow list (`src`, `scripts`, `tests`,
`Makefile`, `PROTOCOL.md`, `TASK.md`, `pyproject.toml`, `README.md`, `uv.lock`) and never
touches `runs/` or `data/` on the server. Both take `--list` to show what would move
without moving it.

### Experimental design

- **Pool sizes**: 1k, 10k, 100k, 1M and 2,456,443, drawn from the full pool as prefixes of
  one seeded permutation, so for a given seed the subsets are nested.
- **Seeds**: 0, 1, 2, controlling adapter init, data order and the train subset. Every seed
  is reported, including failures. No seed is dropped for any reason.
- **LoRA**: r=16, alpha=16, dropout 0, on q,k,v,o,gate,up,down. Identical across arms.
- **Learning rate**: swept per arm over {1e-4, 2e-4, 5e-4} at n=100k across all three seeds,
  chosen by mean dev EM, then frozen for that arm and used at every pool size. Applying it
  unchanged at n=1000 is a limitation, since the optimal LR can depend on data size, and it
  is noted in the results.
- **Budget**: max 4000 steps, batch 16, cosine schedule, 50 warmup steps.
- **Early stopping**: on dev EM every 100 steps, patience 10, and no stopping before step
  1000. These values were set from a pilot sweep that was then discarded rather than mixed
  with later runs; see the Deviations log in [PROTOCOL.md](PROTOCOL.md).
- **Generation**: greedy, `eos_token_id` set explicitly, stopping at `<|im_end|>`.

### Data discipline

Dev drives every decision: learning rate, early stopping, checkpoint selection. Test is
evaluated once per run, saved, and never read by any selection step. Reported test numbers
are the mean and standard deviation over the three seeds of the configuration chosen on dev;
there are no separate retrains for test.

**Contamination** is re-verified for every subset and seed by the `check` stage. The full
2.46M pool contains three rows whose source utterance and target tree are both identical to
an evaluation utterance (lines 491345 and 516460 match test, line 681985 matches dev). The 10
percent pool contains none, which is why earlier checks were clean. Those three rows are
excluded from the training pool, leaving **2,456,443** eligible rows. The evaluation splits
are left exactly as published so results stay comparable with prior work.

**Determinism.** One configuration is trained a second time with an identical configuration
and a different run tag. The absolute difference in test EM between replicates is the floor
on any difference this study can resolve, measured at **0.043**. Differences between arms
smaller than it are reported as indistinguishable from rerun noise rather than as findings,
which is why the 0.003 gap between `words` and `exr` is reported as a match rather than a win.

### What this study does not claim

Stated in advance in [PROTOCOL.md](PROTOCOL.md) so they are not argued about afterwards. In
short: no state-of-the-art claim, since the published BART-Large EXR result used the full
2.46M training set under a different architecture and training regime; no causal claim beyond
the two matched pairs; single task, single model, single tokenizer, so the tokenizer floor
finding is specific to the Qwen BPE vocabulary; and latency is a deployment figure at batch
size one on specific hardware, not a throughput benchmark.

### Reproducibility

Each run writes a self-contained `runs/<run_id>/` with `config.json`, `env.json`,
`metrics.json`, `curves.json`, `dev_predictions.jsonl`, `test_predictions.jsonl` and
`adapter/`. Prediction rows carry the source, gold, raw model output, parsed EXR, a
correctness flag and the generated token count, so **every reported number is recomputable
from disk with `python -m bench report`, without a GPU and without regenerating**.

`make paper` rebuilds every figure and table from `runs/`. No number in the paper is typed by
hand, apart from the on-device table, which is marked as projected until a device run
replaces it.

---

## Layout

```
Makefile             one entry point; `make help` and `make help-server`
PROTOCOL.md          pre-registered protocol, written before any run
server.conf.example  GPU server connection settings; copy to server.conf
paper/               LaTeX source, figure and table generators, palette
src/bench/           one harness, two registries, nothing model- or task-specific
  models.py          MODELS: every model-specific fact, one entry per model
  tasks/__init__.py  TASKS: every task-specific fact, loaded lazily
  tasks/pizza/       6 arms, EXR trees, the dataset's own matcher (CC-BY-NC)
  tasks/coffee/      the shipped parser and the cafe task together:
                     menu, generator, testset, parsing, ollama client, 2 arms.
                     Standard library only, and importing it never imports a
                     line of PIZZA code, both asserted by the test suite.
  config.py          selection, run layout, token caps, chosen learning rates
  train.py evaluate.py latency.py report.py __main__.py
scripts/             tmux.sh, run_all.sh (the runner), push_code.sh, fetch_runs.sh
tests/               pytest suite; data/ holds the frozen generator reference
ollama/              Modelfile
models/              weights, git-ignored
data/external/       PIZZA dataset, git-ignored (CC-BY-NC-4.0)
runs/<model>/<task>/  run artifacts, git-ignored
reports/<model>/<task>/  generated tables, git-ignored
```
