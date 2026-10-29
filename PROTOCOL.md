# PIZZA output-length study: pre-registered protocol

Written before any training run. Any deviation is recorded in "Deviations" at the bottom,
with the date and the reason.

## 1. Research question

Does reducing the number of output tokens make a small LLM more accurate, less prone to
hallucination, more reliably parseable and faster, on one specific task?

Task: PIZZA semantic parsing (natural language food order to a formal tree).
Model: Qwen3.5-0.8B, LoRA fine-tuned, identical hyperparameters across all conditions.

## 2. Why the obvious design does not work

Comparing three unrelated formats (EXR, JSON, a compact DSL) confounds **output length**
with **format family**: they differ in syntax, vocabulary and nesting at the same time as
they differ in length. A difference between them cannot be attributed to length.

This study instead uses a **ladder**: several lossless encodings of the identical semantic
tree that vary mainly in surface verbosity, plus **matched-token control pairs** that hold
length roughly constant while changing format family.

## 3. Conditions

All six encode exactly the same tree and round-trip to byte-identical EXR. Token counts are
measured with the Qwen3.5 tokenizer on the test split.

### Ladder (primary, spans the achievable length range)

| arm         | mean tokens | example                                                 |
| ----------- | ----------- | ------------------------------------------------------- |
| `exr`       | 45.0        | `(ORDER (PIZZAORDER (NUMBER 2 ) (SIZE MEDIUM ) ... ) )` |
| `exr_short` | 33.4        | `(O (P (N 2) (S MEDIUM) ...))`                          |
| `tagged`    | 21.9        | `pizza n=2 size=medium extra=cheese topp=tuna`          |
| `dsl`       | 10.8        | `p 2 medium +cheese tuna`                               |

### Matched-token controls (secondary, test the confound)

| pair             | tokens       | holds constant | varies          |
| ---------------- | ------------ | -------------- | --------------- |
| `exr` vs `json`  | 45.0 vs 45.6 | length         | syntax family   |
| `dsl` vs `words` | 10.8 vs 10.3 | length         | sigils vs words |

If a matched pair differs significantly, format family has an effect independent of length,
and any length-based claim must be qualified accordingly.

### Why the ladder stops at about 10 tokens

A fifth encoding using short opaque entity codes (`p !p5 2 m3 +c2 t2`) was measured during
design. It has 16.4 characters per utterance against the DSL's 34.9, less than half, but
**11.9 tokens against 10.8, more**. Codes outside the BPE vocabulary fragment into several
tokens each. The achievable floor is set by the tokenizer, not by string length. This is
reported as a finding, and it is the reason no arm below about 10 tokens exists.

## 4. Hypotheses and endpoints

Primary endpoint: **unordered exact match on test**, computed by the dataset's own
`semantic_matchers.is_unordered_exact_match`, which is never reimplemented.

- **H1 (accuracy)**: test EM increases as mean output tokens decrease.
- **H2 (hallucination)**: hallucinated entities per utterance decrease as tokens decrease.
- **H3 (speed)**: batch-size-1 median latency decreases as tokens decrease.
- **H4 (usability)**: parse validity increases as tokens decrease.

H3 is expected to hold almost trivially, since latency is close to linear in generated
tokens. It is reported with tokens per second alongside, so that a latency difference caused
purely by generating fewer tokens is distinguishable from a throughput difference.

## 5. Analysis plan

1. **Primary**: ordinary least squares of the endpoint on mean output tokens across the four
   ladder arms, at each train size, with the slope and its 95 percent bootstrap confidence
   interval over seeds. Reported per train size, not pooled.
2. **Matched pairs**: paired difference within `exr` vs `json` and `dsl` vs `words`, using
   the same seed, with a 95 percent paired bootstrap CI over test utterances.
3. **Pairwise arm comparisons**: all pairs reported with paired bootstrap CIs over test
   utterances. These are secondary and exploratory. No significance stars. With six arms
   there are fifteen pairs, so Holm correction is applied to any pairwise claim made in the
   abstract or conclusions.
4. **Seeds**: three per cell. Every seed is reported, including failures. No seed is dropped
   for any reason. Per-seed values appear in the tables, not only means.
5. **Uncertainty**: confidence intervals over three seeds are wide and are labelled as such.
   The paired bootstrap over the 1,357 test utterances is the better-powered comparison and
   is the one used for arm-vs-arm statements.

## 6. Experimental design

- Train pool sizes: n in {1k, 10k, 100k, 1M, 2{,}456{,}443}, drawn from the full pool as
  the prefix of one seeded permutation, so for a given seed the subsets are nested.
- Seeds: 0, 1, 2, controlling adapter init, data order and the train subset.
- LR: swept per arm over {1e-4, 2e-4, 5e-4} at **n=20000**, seeds 0, 1 and 2, chosen by mean
  dev EM, then frozen for that arm and used at both train sizes. The sweep runs at the chosen
  LR are the n=20000 grid cells for all three seeds; they are not repeated, so the entire
  n=20000 row of the grid comes from the sweep. Selecting at the
  larger size means the LR is tuned where the headline number is produced rather than
  extrapolated from a smaller run. Applying it unchanged at n=1000 is a limitation: the
  optimal LR can depend on data size, and this is noted in the results.
- LoRA: r=16, alpha=16, dropout 0, on q,k,v,o,gate,up,down. Identical across arms.
- Max 4000 steps, batch 16, cosine schedule, 50 warmup steps.
- Early stopping on dev EM, evaluated every 100 steps, patience 10, and **no stopping before
  step 1000**. These values were set from a pilot sweep, see Deviations.
- `max_new_tokens` per arm: the maximum target length over train union dev union test, times
  1.25, rounded up. Computed by the `prepare` stage and persisted; never hardcoded. This
  guarantees no gold-length output can be truncated.
- Generation: greedy, `eos_token_id` set explicitly, stopping at `<|im_end|>`. The share of
  generations reaching `max_new_tokens` is logged per run and warns above 1 percent.
- Prompt: `<|im_start|>user\n{src}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
  completion `{target}<|im_end|>`, loss on the completion only.

## 7. Data discipline

- Dev drives every decision: LR choice, early stopping, checkpoint selection.
- Test is evaluated once per run, saved, and never read by any selection step.
- Reported test numbers are the mean and standard deviation over the three seeds of the
  configuration chosen on dev. There are no separate retrains for test.
- Contamination was checked before any run: zero overlap, exact and normalised, between the
  245,645 pool and dev or test, at every n and seed. The pool contains no duplicate sources.

## 8. Metric definitions

Counted on the predicted tree against the gold tree, after converting the arm's output back
to EXR. All are per utterance unless stated.

- **Parse validity**: the output converts to a well-formed EXR tree.
- **Hallucinated entity**: a leaf (constructor, value) pair present in the prediction and
  absent from the gold tree, excluding NUMBER. Multiset difference, so duplicates count.
- **Wrong NUMBER**: a NUMBER leaf present in the prediction and absent from gold.
  Reported separately because a wrong count is a different failure from an invented topping.
- **Missing entity**: a gold leaf absent from the prediction.
- **Extra / missing suborder**: difference in the count of PIZZAORDER and DRINKORDER children.
- **Merged suborder**: predicted suborder count lower than gold while the union of leaves is
  approximately preserved, indicating two orders collapsed into one.

## 9. What this study will not claim

These are stated in advance so they are not argued about afterwards.

1. **No state-of-the-art claim.** The published BART-Large EXR result (78.56 unordered EM)
   used the full 2.46M-utterance training set. This rule was written when the grid topped out
   at 20,000 examples from the 10 percent pool, and it required a full-data run before any
   like-for-like statement. That run has since happened (see Deviations, 2026-10-28): the
   headline grid trains on 2,456,443 utterances, the published pool less the 3 rows that
   duplicate an evaluation utterance and are dropped as contamination.

   The data-budget objection is therefore discharged and the comparison may be described as
   like-for-like **on data**. The state-of-the-art claim is still declined, for the narrower
   reason in item 4: the architectures and training regimes are not matched. The published
   number stays a reference line, now labelled with both training set sizes so the 3-row
   difference is visible rather than rounded away.
2. **No causal claim beyond the matched pairs.** The ladder shows association between length
   and the endpoints. Only the two matched-token pairs isolate format family from length, and
   they do so at only two points on the curve.
3. **Two models, two tasks, and no claim beyond them.** The primary result is PIZZA
   semantic parsing with Qwen3.5-0.8B. LFM2.5-350M is a replication, added after the
   primary grid (see Deviations, 2026-10-27). The cafe task was intended as a second
   replication on a different language and a production system; its grid came back
   saturated and separates nothing, so it is reported as a resolution limit and not as
   support for the finding (see Deviations, 2026-10-28). Absolute scores are **not** compared
   between models: they differ in capacity (0.773B against 0.35B), architecture
   (attention/linear-attention against 10 convolution plus 6 attention blocks), adapter
   targets (an explicit seven against PEFT's `all-linear`, which also reaches the
   convolution blocks), and prompt conditioning (Qwen prefills a closed think block,
   LFM2.5 does not). Only the **ordering of arms** and the **matched-pair differences**
   are compared across models, because those are what the design isolates.
4. **Architecture is not matched to the baseline.** BART-Large is an encoder-decoder trained
   fully; this is a decoder-only model with LoRA adapters. Parameter counts and training
   regimes differ.
5. **Latency is measured on specific hardware** at batch size 1, and is a deployment figure,
   not a throughput benchmark.

## 10. Falsification

The hypothesis is not supported if any of the following hold:

- The slope of test EM on mean output tokens has a 95 percent CI containing zero at both
  train sizes.
- The matched pairs (`exr` vs `json`, `dsl` vs `words`) differ by as much as the ladder arms
  do, which would indicate format family rather than length is driving the result.
- The shortest arm is not the most accurate, which would indicate an optimum rather than a
  monotone benefit.

Any of these will be reported as the headline result if observed.

## 11. Reproducibility

Every run writes `runs/<run_id>/` containing `config.json`, `env.json`, `metrics.json`,
`curves.json`, `dev_predictions.jsonl`, `test_predictions.jsonl` and `adapter/`. Every number
in the report is recomputable from the saved predictions with
`python -m bench report`, without a GPU and without rerunning generation.

## 12. Determinism

One configuration (`dsl`, n=20000, seed 0, chosen LR) is trained a second time with an
identical configuration and a different run tag. The absolute difference in test EM between
the two replicates is the floor on any difference this study can resolve. Differences between
arms smaller than that floor are reported as indistinguishable from rerun noise. The number
is produced by `python -m bench determinism` and written to `runs/determinism.json`.

## 13. Note on what the metric is measuring

A pilot sweep showed that exact match and parse validity move together almost exactly, and
that accuracy conditional on producing a well-formed tree is close to constant across arms,
learning rates and seeds (0.75 to 0.86). Differences between arms are therefore mostly
differences in how reliably the model emits well-formed output, not in how well it
understands the utterance. Parse validity is reported next to exact match everywhere, and
conditional accuracy (EM divided by validity) is reported as a third column so this is
visible rather than hidden.

## 14. What the data size axis measures

The step budget is fixed at 4000 steps of batch 16, so every run sees at most 64{,}000
samples regardless of pool size. Above that the extra data is never visited: at 1M the model
sees 6 percent of the pool and at 2.46M about 3 percent. The axis is therefore **pool
diversity at fixed compute**, not amount of data trained on, and it is labelled "pool size"
rather than "training examples" everywhere. The question it answers is whether 64{,}000
samples drawn from a large pool beat 64 epochs over a small one. Runs record both the pool
size and the samples actually seen so the distinction is never implicit.

## 15. Evaluation duplicates in the full pool

The full 2.46M pool contains three rows whose source utterance and target tree are both
identical to an evaluation utterance: lines 491345 and 516460 match test, line 681985
matches dev ("three large pizzas with ham", "...with onions", "...with mushrooms"). The 10
percent pool contains none, which is why earlier contamination checks were clean. These three
rows are excluded from the training pool, leaving 2{,}456{,}443 eligible rows. The evaluation
splits are left exactly as published so results stay comparable with prior work. Overlap is
re-verified for every subset and seed by the `check` stage.

## Deviations

- 2026-10-19: the LR sweep moved from n=5000 to n=20000 so that sweep runs double as grid
  cells, and tunes the LR at the size used for the headline result. The sweep keeps all three
  seeds. No other part of the protocol changed.
- 2026-10-21: the training pool moved from the 10 percent file to the full 2.46M file so
  that pool sizes up to 2.46M are reachable, and subsets became nested prefixes of one
  seeded permutation rather than independent samples. Sizes changed from {1k, 20k} to
  {1k, 10k, 100k, 1M, 2.46M}, and the learning rate is now selected at n=100k. Three
  evaluation duplicates present only in the full pool are excluded, see section 15. No
  result from the earlier pool is carried forward.
- 2026-10-19: stopping rules changed after a pilot sweep of 44 runs, and the pilot was
  discarded rather than mixed with later runs. Evidence: (a) dev EM oscillates by up to 0.4
  between consecutive evaluations, so patience 5 stopped runs on noise, for example a dsl run
  with the sequence 0.32 0.51 0.47 0.07 0.49 0.38 0.51 stopped at step 700 with its best at
  200 although steps 500 and 700 matched it; (b) three runs had their best step at or within
  one evaluation of the 2000 step budget and were still improving, so those learning rates
  were being judged on truncated training. New values: max_steps 2000 to 4000, patience 5 to
  10, min_steps 500 to 1000. No other part of the protocol changed, and no pilot number is
  carried forward.
- 2026-10-19: the determinism replicate uses n=20000 rather than n=5000, because n=5000 is no
  longer one of the train sizes and the first replicate therefore already exists at n=20000.
  This makes the check cost one run instead of two.

- 2026-10-27: a second model, LFM2.5-350M, was added as a replication after the primary
  Qwen grid was complete. It was not pre-registered. The motivation is section 9 item 3:
  the tokenizer-floor claim was declared untransferable, and LFM2.5's vocabulary is 65,536
  against Qwen's 248,320, which makes it a direct test. Its grid is identical in arms,
  sizes, seeds and stopping rules; it has its own learning-rate sweep, its own token caps
  (a cap is tokenizer-dependent) and its own determinism replicate, and its results live in
  a separate `runs/lfm2/pizza/`. Because it was added afterwards it is reported as a
  replication of the ordering and of the matched pairs, never as a model comparison, and no
  Qwen conclusion depends on it.
- 2026-10-27: a second task, the cafe order parser, was added for the same reason. It has
  two arms rather than six (`compact`, the shipped format, and `words`, the same content in
  natural words), so it contributes no ladder and no slope. What it contributes is one
  matched pair, in Russian and English rather than English alone, on the system actually in
  production. Its test split is the 49 hand-written cases, which is small enough that it can
  support a directional statement and not a precise one, and that limit is reported with
  the result.
- 2026-10-28: the grid reached the full 2.46M pool, so the comparison with the published
  BART-Large EXR baseline is now matched on training set size. Item 1 of section 9 was
  written on the assumption that it never would be and is amended above rather than
  deleted: the data-budget objection is discharged, the architecture objection (item 4)
  stands, and no state-of-the-art claim is made. Our pool is 2,456,443 against the
  published 2,456,446; the 3 missing rows duplicate an evaluation utterance exactly and
  are dropped from training, not from the evaluation splits, so the splits stay as
  published.
- 2026-10-28: the cafe grid completed and did **not** replicate the format effect. Every
  cell, across both arms and all four pool sizes, falls between 0.939 and 0.980 exact
  match; the headline gap is -0.007, against a 49 case test split where one utterance is
  worth 0.020 and the determinism replicate reads 0.020 exactly. All 12 matched pairs fall
  inside the paired bootstrap interval. The task is saturated, so it is reported as a
  negative result that bounds the claim rather than as a replication, and no arm is marked
  best in its table. No LFM2.5 cafe grid was run: a second saturated grid would add no
  information. Settling the question on cafe orders requires a larger and harder held-out
  set drawn from real transcripts, which does not exist yet.
- 2026-10-28: published reference points moved from `report.py` onto the task that owns
  them, because the cafe report was quoting PIZZA's BART-Large, PCFG and BERT baselines
  underneath cafe numbers. The report also titles itself from the task label now.
- 2026-10-27: generation provenance is now written into every `metrics.json`
  (`padding_side`, EOS token and id, vocabulary size, the resolved prompt). An LFM2.5 grid
  was first run with right padding, which places pad tokens between the prompt and the
  continuation, and scored 0.13 to 0.20 lower exact match on every arm with nothing on disk
  to explain the difference. The affected runs were discarded and re-run with left padding;
  the discarded set is kept out of the reported results. The arm ordering was unchanged by
  the bug, which is reported as a robustness observation rather than a result.
