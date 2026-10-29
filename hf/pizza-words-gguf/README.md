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
pepperoni no olives`). It is deliberately strict: it parses orders made only of
menu items, and answers `?` to everything else, instead of inventing a ticket. Merged and exported as a single Q8_0 GGUF for llama.cpp,
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

## Only the menu: everything else is `?`

30% of the training rows were negatives, generated the way the café parser's
were, and every one of them is answered with `?` alone:

| kind | example |
|---|---|
| chatter and questions | `hello`, `where is the restroom` |
| off-menu requests | `two cats and a dog`, `can i get a burger` |
| gibberish, any language | `hehe hehe`, `asdkjh`, `1 hello world cola`, `привет` |
| long text | `lorem ipsum dolor sit amet ...` |
| an order with something off the menu | `a large ham pizza and some fries` |
| an order with an unknown word | `a party sized zebra with extra cheese` |

An order with anything extra in it is refused whole, never half-parsed. Nothing
the PIZZA catalogs can order is ever used as a refusal, and words inserted into
real orders come from outside PIZZA's 211-word order vocabulary.

Held-out results (vocabulary never trained on, 614 inputs): 96.7% refused
correctly; gibberish, long text and chatter 100%, off-menu 99%, orders mixed
with off-menu items 94%, orders with an unknown word 90%. A fixed set of 14
demo-style prompts: 14 of 14 refused. The price of strictness: 4.7% of real
PIZZA test orders are refused too. Those are loud errors, so the customer is
asked again rather than served the wrong pizza.

## Verification

Before release the GGUF was run greedily through llama-server on the first 200
PIZZA test orders and compared against the training run's own saved
predictions: 98.5% identical. The three differences are quantization breaking
near-ties, two of them in the GGUF's favour, so its exact match on those 200 is
0.955 against 0.940 unquantized. Run `words-n2456443-s0-lr0.0001-neg30-fresh-20261029`,
trained on the full 2,456,443-example PIZZA pool plus negatives, checkpoint chosen
on dev exact match and dev refusals together. For reference, the paper's run of
the same configuration without negatives scores 0.881 exact match on the full
1,357-order test set.

## Licence

Trained on the [PIZZA benchmark](https://github.com/amazon-science/pizza-semantic-parsing-dataset),
which is CC-BY-NC-4.0, so this model is for **non-commercial research use only**.
The base model, Qwen3.5-0.8B, is under its own licence.
