#!/usr/bin/env python3
"""
Batch-size-1 latency on an M-series Mac, via llama.cpp on a merged GGUF.

The GPU figure from `python -m pizza_bench latency` is a server number. This is the
deployment number. Both use batch size 1 so they are comparable in kind.

  python scripts/pizza_latency_mac.py --model models/pizza-dsl-q8_0.gguf --arm dsl
  python scripts/pizza_latency_mac.py --model models/pizza-dsl-q8_0.gguf --arm dsl \
      --grammar runs/pizza_dsl.gbnf
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pizza_bench import data, formats, metrics
from pizza_bench.config import EOS, PROMPT, load_caps

STOPS = [EOS, "<|endoftext|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the GGUF")
    ap.add_argument("--arm", choices=formats.ARMS, required=True)
    ap.add_argument("--grammar", default=None, help="GBNF file; dsl only")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.grammar and args.arm != "dsl":
        sys.exit("the GBNF grammar describes the DSL only")

    from llama_cpp import Llama, LlamaGrammar

    cap = load_caps()[args.arm]
    rows = data.load(args.split)[:args.limit]
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, verbose=False)
    grammar = LlamaGrammar.from_file(args.grammar) if args.grammar else None

    def run(src):
        t0 = time.perf_counter()
        out = llm(PROMPT.format(src), max_tokens=cap, temperature=0.0,
                  stop=STOPS, grammar=grammar)
        ms = (time.perf_counter() - t0) * 1000
        return out["choices"][0]["text"].strip(), ms, out["usage"]["completion_tokens"]

    for r in rows[:args.warmup]:
        run(r["src"])

    times, toks, outs = [], [], []
    for r in rows:
        text, ms, n = run(r["src"])
        outs.append(text)
        times.append(ms)
        toks.append(n)

    res, _, _ = metrics.evaluate(outs, [r["exr"] for r in rows], args.arm,
                                 pcfg_err=[r["pcfg_err"] for r in rows],
                                 n_tokens=toks, cap=cap)
    s = sorted(times)
    summary = {
        "device": "mac-llama.cpp", "model": args.model, "arm": args.arm,
        "grammar": bool(args.grammar), "split": args.split, "n": len(rows),
        "batch_size": 1, "max_new_tokens": cap,
        "median_ms": statistics.median(s), "p90_ms": s[int(len(s) * 0.9)],
        "mean_generated_tokens": statistics.fmean(toks),
        "tokens_per_second": statistics.fmean(toks) / (statistics.fmean(s) / 1000),
        "em": res["em"], "parse_validity": res["parse_validity"],
        "hit_cap_rate": res.get("hit_cap_rate"),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
