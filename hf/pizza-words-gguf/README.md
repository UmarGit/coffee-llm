---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.5-0.8B
library_name: gguf
pipeline_tag: text-generation
language: [en]
tags: [gguf, llama.cpp, semantic-parsing, pizza, lora, wllama, guardrails]
---

# pizza-words-gguf

Qwen3.5-0.8B fine-tuned with LoRA to turn a spoken-style pizza order into a
structured ticket, written in the **WORDS** output format (`pizza 2 large
pepperoni no olives`). It also answers `?` when the input is not an order or
asks for something off the menu, instead of inventing a ticket. Merged and exported as a single Q8_0 GGUF for llama.cpp,
and the model that runs live, in the browser, on
[Parsed, Plausible, Wrong](https://umargit.github.io/coffee-llm/).

It accompanies the paper *Fewer Tokens, Fewer Silent Errors: Output Format
Design for Small Language Models in Real Time Order Parsing*, by Umar Ahmed and
Maxim Duanev, ITMO University, IDM (Intelligent Decision Making) Lab.
Code: [UmarGit/coffee-llm](https://github.com/UmarGit/coffee-llm).

## Files

| file | quantization | size |
|---|---|---|
| `pizza-words-q8_0.gguf` | Q8_0 | 834 MB |
| `pizza-words-q8_0.json` | export manifest: run id, prompt, stop string, sha256, verification | |

## Prompt

The model was trained and evaluated with exactly this prompt (note the
already-closed think block), greedy decoding, stopping at `<|im_end|>`, at most
50 new tokens. Inputs in the benchmark are lowercase.

```
<|im_start|>user
{order}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

```bash
llama-server -m pizza-words-q8_0.gguf --port 8080 -c 1024
curl -s localhost:8080/completion -d '{
  "prompt": "<|im_start|>user\ni want two medium pizzas with ham and no olives<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "n_predict": 50, "temperature": 0, "top_k": 1, "stop": ["<|im_end|>"]
}' | jq -r .content
```

## Not an order: `?`

15% of the training rows were negatives, built the way the café parser's were:

| input | output |
|---|---|
| a non-order: `hello`, `where is the restroom` | `?` |
| an off-menu request: `can i get two apples` | `?` |
| an order plus something off the menu: `a large ham pizza and some fries` | `pizza 1 large ham ; ?` |

`?` is a suborder of its own, so a consumer can split on `;`, serve the real
items and ask again about the rest. Off-menu items were checked against the PIZZA
catalogs, so nothing the dataset can order is ever taught as a refusal. A quarter
of the phrases were held out of training and only used to evaluate rejection.

## Verification

Before release the GGUF was run greedily through llama-server on the first 200
PIZZA test orders and compared against the training run's own saved
predictions: 199 of 200 identical, the one difference listing the same three
drinks in another order, which the official matcher scores as the same order
(exact match 0.965 on those 200, the same as the unquantized model). Run
`words-n2456443-s0-lr0.0001-neg15-fresh-20261029`, trained on the full
2,456,443-example PIZZA pool plus negatives, best checkpoint by dev exact match
on PIZZA alone. For reference, the paper's run of the same configuration without
negatives scores 0.881 exact match on the full 1,357-order test set.

## Licence

Trained on the [PIZZA benchmark](https://github.com/amazon-science/pizza-semantic-parsing-dataset),
which is CC-BY-NC-4.0, so this model is for **non-commercial research use only**.
The base model, Qwen3.5-0.8B, is under its own licence.
