"""Run the correction probes (and optionally held-out corrections on test orders)
through a GGUF, in process, and count correct / silent / refused / broken.

    CMAKE_ARGS="-DGGML_METAL=on" uv run --extra pizza --with llama-cpp-python \\
        python tools/probe_corrections.py site/public/models/pizza-words-q8_0.gguf [--eval 200]

The prompt, stop string and token cap come from the manifest next to the GGUF
(`<name>.json`, as `bench export` writes it), else the PIZZA defaults.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from bench.tasks.pizza import corrections, data


def manifest(gguf):
    p = gguf.with_suffix(".json")
    if p.exists():
        m = json.loads(p.read_text())
        return m["prompt"], m["stop"], m["cap"], m["arm"]
    return data.PROMPT, "<|im_end|>", 50, "words"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--eval", type=int, default=0, help="also this many held-out test corrections")
    ap.add_argument("--out", type=Path, help="write every output here as jsonl")
    args = ap.parse_args()

    from llama_cpp import Llama

    prompt, stop, cap, arm = manifest(args.gguf)
    llm = Llama(model_path=str(args.gguf), n_ctx=1024, n_gpu_layers=-1, verbose=False)
    rows = corrections.probe_rows()
    if args.eval:
        rows += corrections.eval_rows(data.load("test"), n=args.eval)

    outs = []
    for r in rows:
        res = llm(prompt.format(r["src"]), max_tokens=cap, temperature=0, top_k=1, stop=[stop])
        outs.append(res["choices"][0]["text"].split(stop)[0].strip())

    n_probe = len(corrections.PROBES)
    for name, sl in (("probes", slice(0, n_probe)), ("held-out", slice(n_probe, None))):
        part, got = rows[sl], outs[sl]
        if not part:
            continue
        res, preds, em = corrections.score(got, part, arm)
        kinds = {}
        for r, o, p, e in zip(part, got, preds, em):
            kinds.setdefault(r["fix"], Counter())[corrections.outcome(o, p, e, arm)] += 1
        applied = f"  applied {res['applied']:.3f}" if name != "probes" else ""
        print(f"\n{name}: n {res['n']}  em {res['em']:.3f}{applied}  refused {res['refused']:.3f}")
        print("  " + "  ".join(f"{k} {v}" for k, v in corrections.outcomes(got, preds, em, arm).items()))
        for k, c in sorted(kinds.items()):
            print(f"  {k:8} " + "  ".join(f"{o} {c[o]}" for o in ("correct", "silent", "refused", "broken")))
        if name == "probes":
            print()
            for r, o, e in zip(part, got, em):
                if not e:
                    print(f"  x [{r['fix']}] {r['src']}\n      got {o!r}")

    if args.out:
        with open(args.out, "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"src": r["src"], "kind": r["fix"], "gold": r["exr"], "raw": o}) + "\n")


if __name__ == "__main__":
    main()
