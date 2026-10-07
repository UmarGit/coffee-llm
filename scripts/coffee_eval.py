#!/usr/bin/env python3
"""
Accuracy test for the coffee order parser, run against Ollama.

Usage:
  uv run python scripts/coffee_eval.py
"""
import argparse
from collections import Counter

from coffee_parser import TEST_SET, category, generate, parse_compact
from coffee_parser.ollama import DEFAULT_HOST, DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    args = ap.parse_args()

    generate("эспрессо", args.model, args.host)  # warm-up
    stats, fails, times, toks = Counter(), [], [], []
    for text, expected in TEST_SET:
        out, ms, n = generate(text, args.model, args.host)
        times.append(ms)
        toks.append(n)
        cat = category(expected)
        stats[cat + "_total"] += 1
        pred = parse_compact(out)
        if pred is not None and pred == parse_compact(expected):
            stats[cat + "_ok"] += 1
        else:
            fails.append((text, expected, out))

    print(f"\nModel: {args.model}\n" + "-" * 60)
    for text, exp, got in fails:
        print(f"✗ {text}\n    want: {exp}\n    got:  {got}")
    print("-" * 60)
    total_ok = sum(v for k, v in stats.items() if k.endswith("_ok"))
    print(f"Overall:    {total_ok}/{len(TEST_SET)}  ({100*total_ok/len(TEST_SET):.0f}%)")
    for cat in ["order", "partial", "not-order"]:
        t = stats[cat + "_total"]
        if t:
            print(f"{cat:11} {stats[cat+'_ok']}/{t}")
    times.sort()
    print(f"Latency:    median {times[len(times)//2]:.0f} ms | worst {times[-1]:.0f} ms "
          f"| avg {sum(toks)/len(toks):.0f} output tokens")


if __name__ == "__main__":
    main()
