"""
Latency benchmark: several passes over the eval set, end-to-end through Ollama.

Usage:
  uv run python scripts/benchmark.py
  uv run python scripts/benchmark.py --runs 5
"""

import argparse
import statistics

from coffee_parser import TEST_SET, generate, parse_compact
from coffee_parser.ollama import DEFAULT_HOST, DEFAULT_MODEL


def bench(run, runs):
    for _ in range(3):
        run("эспрессо")  # warm-up (load, compile, cache)
    times, toks, correct = [], [], 0
    for run_idx in range(runs):
        for text, expected in TEST_SET:
            out, ms, n = run(text)
            times.append(ms)
            toks.append(n)
            if run_idx == 0:  # decoding is greedy, so score the first pass only
                pred = parse_compact(out)
                correct += pred is not None and pred == parse_compact(expected)
    times.sort()
    return {
        "median": statistics.median(times),
        "p90": times[int(len(times) * 0.9)],
        "tokens": statistics.mean(toks),
        "acc": correct / len(TEST_SET),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--runs", type=int, default=3, help="passes over the eval set")
    args = ap.parse_args()

    r = bench(lambda text: generate(text, args.model, args.host), args.runs)

    print(f"\n{len(TEST_SET)} orders × {args.runs} runs (end-to-end, per order)\n")
    print(f"{'median':>9} {'p90':>9} {'tokens':>8} {'accuracy':>9}")
    print(f"{r['median']:7.0f}ms {r['p90']:7.0f}ms {r['tokens']:8.1f} {r['acc']:8.0%}")


if __name__ == "__main__":
    main()
