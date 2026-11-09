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
menu items, and answers `?` to everything else, instead of inventing a ticket.
It also applies spoken self-corrections (`a pizza with cheese, wait no cheese,
only tomato sauce` gives `pizza 1 tomato_sauce`). Merged and exported as a
single Q8_0 GGUF for llama.cpp,
and the model that runs live, in the browser, on
[Parsed, Plausible, Wrong](https://umargit.github.io/coffee-llm/).

It accompanies the paper *Parsed, Plausible, Wrong: Output Format Controls
Silent Errors in Small-Model Semantic Parsing* (in preparation), from the IDM
(Intelligent Decision Making) Lab, ITMO University:

- Umar Ahmed, ITMO University · ORCID [0009-0003-6040-6328](https://orcid.org/0009-0003-6040-6328)
- Maxim Duanev, ITMO University · ORCID [0009-0009-2988-9839](https://orcid.org/0009-0009-2988-9839)
- Mohammad Nour Al Awad, ITMO University · ORCID [0009-0008-5078-9800](https://orcid.org/0009-0008-5078-9800)
- Sergey Ivanov, ITMO University · ORCID [0000-0002-1128-2942](https://orcid.org/0000-0002-1128-2942)
- Ivan V. Khodnenko, ITMO University · ORCID [0009-0003-7787-7126](https://orcid.org/0009-0003-7787-7126)

Code: [UmarGit/coffee-llm](https://github.com/UmarGit/coffee-llm). A smaller
sibling trained the same way: [pizza-lfm2-words-gguf](https://huggingface.co/umarsheikh303/pizza-lfm2-words-gguf).

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

## Results

This file is seed 0 of the paper's full-pool WORDS cell (run
`words-n2456443-s0-lr0.0002`): Qwen3.5-0.8B, LoRA r=16, trained on the full
2,456,443-order PIZZA pool plus 30% negatives and 10% self-corrections. On the
1,357 human-written PIZZA test orders:

| | this run | mean of 5 seeds |
|---|---|---|
| exact match | 88.2% | 87.5% ± 0.8 |
| silent errors (parsed, but wrong) | 3.8% | 4.5% |
| real orders refused with `?` | 2.1% | 3.4% |

For comparison, the same model writing the dataset's native s-expression
averages 85.8% exact match with 10.3% silent errors, while emitting about four
times as many tokens.

## Only the menu: everything else is `?`

30% of the training rows are negatives, and every one is answered with `?` alone:

| kind | example |
|---|---|
| chatter and questions | `hello`, `where is the restroom` |
| off-menu requests | `two cats and a dog`, `can i get a burger` |
| gibberish, any language | `hehe hehe`, `asdkjh`, `1 hello world cola` |
| long text | `lorem ipsum dolor sit amet ...` |
| an order with something off the menu | `a large ham pizza and some fries` |
| an order with an unknown word | `a party sized zebra with extra cheese` |

An order with anything extra in it is refused whole, never half-parsed. Nothing
the PIZZA catalogs can order is ever used as a refusal.

Held-out results (vocabulary never trained on, 614 inputs): 97.7% refused
correctly; gibberish, long text and off-menu requests 100%, chatter 98.6%,
orders with an unknown word 95.5%, orders mixed with off-menu items 93%. A fixed
set of 14 demo-style prompts: 14 of 14 refused.

**The price of strictness:** 2.1% of real test orders are refused too, mostly
polite or chatty ones (`hello my order is for one pie... thank you so much`),
because the synthetic training grammar contains no greetings. A refusal is a
loud error: the customer is asked again rather than served the wrong pizza.

## Self-corrections

10% of the training orders get a spoken correction appended (swap a topping,
retract one, change the size or count, cancel a line, start over, and decoys
where the cue word changes nothing). On 600 test orders with correction phrases
never seen in training, 70.0% come out exactly right and the correction itself
is applied in 84.0%. Of 40 hand-written correction probes, 38 are right and the
other 2 are refused, none silently wrong.

## Verification

Before release the GGUF was run greedily through llama-server on the first 200
PIZZA test orders and compared with the training run's own saved predictions:
99.0% identical, exact match 0.940 for the GGUF against 0.935 unquantized on
those 200 orders.

## Licence

Trained on the [PIZZA benchmark](https://github.com/amazon-science/pizza-semantic-parsing-dataset),
which is CC-BY-NC-4.0, so this model is for **non-commercial research use only**.
The base model, Qwen3.5-0.8B, is under its own licence.
