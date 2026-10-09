"""
On-device latency for a PIZZA-trained model, per output format.

Raw mode with the exact training prompt, matching scripts/benchmark.py. Ollama cannot do
grammar-constrained decoding (its API takes JSON schemas only), so --backend llamacpp is
required for --grammar.

  python scripts/pizza_latency.py --backend ollama --model pizza-dsl --format dsl
  python scripts/pizza_latency.py --backend llamacpp --model models/pizza-dsl.gguf \
      --format dsl --grammar build/pizza_dsl.gbnf
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from pizza_bench import data, metrics
from pizza_bench.experiment import MAX_NEW

STOPS = ["<|im_end|>", "<|endoftext|>", "</think>"]


def clean(text):
    for s in STOPS:
        text = text.split(s)[0]
    return text.strip()


def ollama_runner(model, host, fmt):
    def run(src):
        body = json.dumps(
            {
                "model": model,
                "prompt": data.PROMPT.format(src),
                "raw": True,
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0,
                    "num_predict": MAX_NEW[fmt],
                    "stop": STOPS,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
        return (
            clean(resp["response"]),
            (time.perf_counter() - t0) * 1000,
            resp.get("eval_count", 0),
        )

    return run


def llamacpp_runner(path, fmt, grammar_path, n_ctx):
    from llama_cpp import Llama, LlamaGrammar

    llm = Llama(model_path=path, n_ctx=n_ctx, logits_all=False, verbose=False)
    grammar = LlamaGrammar.from_file(grammar_path) if grammar_path else None

    def run(src):
        t0 = time.perf_counter()
        out = llm(
            data.PROMPT.format(src),
            max_tokens=MAX_NEW[fmt],
            temperature=0.0,
            stop=STOPS,
            grammar=grammar,
        )
        ms = (time.perf_counter() - t0) * 1000
        return clean(out["choices"][0]["text"]), ms, out["usage"]["completion_tokens"]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "llamacpp"], default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--format", dest="fmt", choices=["exr", "json", "dsl"], required=True
    )
    ap.add_argument("--grammar", default=None, help="GBNF file (llamacpp only)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.grammar and args.backend != "llamacpp":
        sys.exit(
            "--grammar needs --backend llamacpp; Ollama's API only accepts JSON schemas"
        )

    rows = data.load(args.split)[: args.limit]
    run = (
        ollama_runner(args.model, args.host, args.fmt)
        if args.backend == "ollama"
        else llamacpp_runner(args.model, args.fmt, args.grammar, args.n_ctx)
    )

    for r in rows[: args.warmup]:
        run(r["src"])

    times, toks, outputs = [], [], []
    for r in rows:
        out, ms, n = run(r["src"])
        outputs.append(out)
        times.append(ms)
        toks.append(n)

    res, _ = metrics.evaluate(
        outputs, [r["exr"] for r in rows], args.fmt, [r["pcfg_err"] for r in rows]
    )
    times.sort()
    summary = {
        "backend": args.backend,
        "model": args.model,
        "format": args.fmt,
        "grammar": bool(args.grammar),
        "split": args.split,
        "n": len(rows),
        "median_ms": statistics.median(times),
        "p90_ms": times[int(len(times) * 0.9)],
        "mean_output_tokens": statistics.fmean(toks),
        "em": res["em"],
        "parse_validity": res["parse_validity"],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
