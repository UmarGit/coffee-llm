---
license: cc-by-nc-4.0
base_model: LiquidAI/LFM2.5-350M
library_name: gguf
pipeline_tag: text-generation
language: [en]
tags: [gguf, llama.cpp, semantic-parsing, pizza, lora, guardrails]
---

# pizza-lfm2-words-gguf

LFM2.5-350M fine-tuned with LoRA to turn a spoken-style pizza order into a
structured ticket, written in the **WORDS** output format (`pizza 2 large
pepperoni no olives`). Like its larger sibling
[pizza-words-gguf](https://huggingface.co/umarsheikh303/pizza-words-gguf), it
parses orders made only of menu items, answers `?` to everything else, and
applies spoken self-corrections. Merged and exported as a single Q8_0 GGUF
(379 MB) for llama.cpp.

It accompanies the paper *Parsed, Plausible, Wrong: Output Format Controls
Silent Errors in Small-Model Semantic Parsing* (in preparation), from the IDM
(Intelligent Decision Making) Lab, ITMO University:

- Umar Ahmed, ITMO University · ORCID [0009-0003-6040-6328](https://orcid.org/0009-0003-6040-6328)
- Maxim Dunaev, ITMO University · ORCID [0009-0009-2988-9839](https://orcid.org/0009-0009-2988-9839)
- Mohammad Nour Al Awad, ITMO University · ORCID [0009-0008-5078-9800](https://orcid.org/0009-0008-5078-9800)
- Sergey Ivanov, ITMO University · ORCID [0000-0002-1128-2942](https://orcid.org/0000-0002-1128-2942)
- Ivan Khodnenko, ITMO University · ORCID [0009-0003-7787-7126](https://orcid.org/0009-0003-7787-7126)

Code: [UmarGit/coffee-llm](https://github.com/UmarGit/coffee-llm).

## Files

| file | quantization | size |
|---|---|---|
| `pizza-lfm2-words-q8_0.gguf` | Q8_0 | 379 MB |
| `pizza-lfm2-words-q8_0.json` | export manifest: run id, prompt, stop string, sha256, verification | |

## Prompt

Trained and evaluated with exactly this prompt (LFM2's own chat template, no
think block), greedy decoding, stopping at `<|im_end|>`, at most 57 new tokens.
Inputs in the benchmark are lowercase.

```
<|startoftext|><|im_start|>user
{order}<|im_end|>
<|im_start|>assistant
```

```bash
llama-server -m pizza-lfm2-words-q8_0.gguf --port 8080 -c 1024
curl -s localhost:8080/completion -d '{
  "prompt": "<|startoftext|><|im_start|>user\ni want two medium pizzas with ham and no olives<|im_end|>\n<|im_start|>assistant\n",
  "n_predict": 57, "temperature": 0, "top_k": 1, "stop": ["<|im_end|>"]
}' | jq -r .content
```

## Results

This file is seed 0 of the paper's full-pool WORDS cell for LFM2.5-350M (run
`words-n2456443-s0-lr0.0002`): LoRA r=16, trained on the full 2,456,443-order
PIZZA pool plus 30% negatives and 10% self-corrections. On the 1,357
human-written PIZZA test orders:

| | this run | mean of 5 seeds |
|---|---|---|
| exact match | 79.3% | 77.4% ± 3.2 |
| silent errors (parsed, but wrong) | 10.2% | 10.0% |
| real orders refused with `?` | 5.5% | 6.7% |

For comparison, the same model writing the dataset's native s-expression
averages 74.0% exact match with 17.9% silent errors.

**Refusals.** On 614 held-out negatives (vocabulary never trained on) it refuses
98.4%: gibberish, long text, chatter and off-menu requests 100%, orders mixed
with off-menu items 99%, orders with an unknown word 93.2%; 14 of 14 demo-style
prompts. The price: 5.5% of real test orders are refused too, mostly polite or
chatty ones.

**Self-corrections.** On 600 test orders with correction phrases never seen in
training, 61.0% come out exactly right and the correction is applied in 76.5%.
Of 40 hand-written correction probes, 37 are right and 3 are silently wrong.

## Verification: read this before relying on the GGUF

The GGUF was run greedily through llama-server on the first 200 PIZZA test orders
and compared with the training run's own saved predictions: **93% identical**
(the Qwen sibling: 99%). The differences are close calls on long multi-line
orders. On those 200 orders the GGUF's exact match is 0.910 against 0.870
unquantized, so the drift is not one-sided, but this file does not reproduce
the trained run exactly. No number in the paper comes from a GGUF.

## Licence

Trained on the [PIZZA benchmark](https://github.com/amazon-science/pizza-semantic-parsing-dataset),
which is CC-BY-NC-4.0, so this model is for **non-commercial research use only**.
The base model, LFM2.5-350M, is under Liquid AI's own licence, whose terms also
apply.
