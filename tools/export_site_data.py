#!/usr/bin/env python3
"""Build the site's static JSON from saved runs. No GPU, no dataset download.

    uv run --extra pizza python tools/export_site_data.py

Everything the site shows about measured behaviour comes from here, so every
file carries the run ids it was built from and a schema version. If a number on
the site cannot be traced to a run id in meta.json, it does not belong there.

Sections 3 to 6 of the brief need per-token probabilities, layer internals and
training checkpoints. None of that is saved in runs/, so this script does not
invent it: it writes nothing for those sections and records them as absent, and
the site build fails rather than shipping a placeholder.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper"))

from bench import config, report
from bench.tasks.pizza import formats

OUT = REPO / "site" / "src" / "data"
# Big per-order files are build inputs only, never published: site/scripts/snapshot.mjs
# reduces them to the few rows the page shows.
PUBLIC = REPO / "site" / "raw"
BIG = {"tickets.json", "orders.json", "outcomes.qwen.json", "outcomes.lfm2.json"}
SCHEMA = 1
ARMS = ("exr", "json", "exr_short", "tagged", "dsl", "words")
MODELS = ("qwen", "lfm2")
# correct / detectable (did not parse) / silent (parsed, wrong). The site never
# shows a bare "wrong": the whole argument is that these two differ.
CORRECT, DETECTABLE, SILENT = 0, 1, 2


def outcome(row):
    if row["correct"]:
        return CORRECT
    return DETECTABLE if row["parsed"] is None else SILENT


def headline_runs(model):
    """One run per arm: the headline size, seed 0, at the arm's chosen lr."""
    config.select(model, "pizza")
    sizes = config.Config.for_active().sizes
    head = max(sizes)
    lrs = report.chosen_lrs()
    out = {}
    for r in report.primary_runs():
        if r["size"] != head or r["seed"] != 0:
            continue
        if abs(r["lr"] - lrs.get(r["arm"], -1)) > 1e-12:
            continue
        out[r["arm"]] = r
    return out, head


def read_preds(run):
    rows = []
    for line in (Path(run["_dir"]) / "test_predictions.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write(name, payload, sources):
    payload = {"schema": SCHEMA, "generated": date.today().isoformat(),
               "sources": sources, **payload}
    p = (PUBLIC if name in BIG else OUT) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    kb = p.stat().st_size / 1024
    print(f"  {name:28} {kb:8.1f} KB")
    return kb


def export_formats():
    """Per model and arm: the headline numbers the essay quotes."""
    out, sources, total = {}, [], 0.0
    for model in MODELS:
        runs, head = headline_runs(model)
        if len(runs) != len(ARMS):
            print(f"  (skipping {model}: {len(runs)}/{len(ARMS)} arms)")
            continue
        config.select(model, "pizza")
        lat = {d["arm"]: d for d in report.latency_table()}
        # Mean over seeds for the quoted figures, so the site matches the paper
        # rather than the single seed it can run live.
        allruns = defaultdict(list)
        for r in report.primary_runs():
            if r["size"] == head:
                allruns[r["arm"]].append(r)
        arms = {}
        for a in ARMS:
            rs = allruns[a]
            em = statistics.fmean(r["test"]["em"] for r in rs)
            val = statistics.fmean(r["test"]["parse_validity"] for r in rs)
            arms[a] = {
                "tokens": round(statistics.fmean(
                    r["test"]["mean_generated_tokens"] for r in rs), 2),
                "em": round(em, 4),
                "validity": round(val, 4),
                "silent": round(val - em, 4),
                "hallucinated": round(statistics.fmean(
                    r["test"]["hallucinated_entity"] for r in rs), 4),
                "missing": round(statistics.fmean(
                    r["test"]["missing_entity"] for r in rs), 4),
                "seeds": len(rs),
                "latencyMs": round(lat[a]["median_ms"], 1) if a in lat else None,
                "liveRunId": runs[a]["run_id"],
            }
            sources.append(runs[a]["run_id"])
        det = json.loads((config.model_task_dir() / "determinism.json").read_text())
        out[model] = {
            "label": config.model().label,
            "vocabSize": config.model().vocab_size,
            "trainSize": head,
            "noiseFloor": round(det["test_abs_diff"], 4),
            "arms": arms,
        }
    total += write("formats.json", {"models": out}, sorted(set(sources)))
    return out, total


def export_orders(runs):
    """The test utterances, once. Outcomes reference these by index."""
    rows = read_preds(runs["words"])
    orders = [{"i": r["id"], "src": r["src"], "gold": r["gold"]} for r in rows]
    return orders, write("orders.json", {"orders": orders},
                         [runs["words"]["run_id"]])


def export_outcomes(model, runs):
    """Per arm: one outcome code and the raw output per test order.

    The dots in section 2 need only the code; clicking one needs the output, so
    both live here. Outputs are kept verbatim, including the malformed ones,
    because the failure gallery is built from exactly what the model emitted.
    """
    per_arm, sources, counts = {}, [], {}
    for a in ARMS:
        rows = read_preds(runs[a])
        per_arm[a] = {
            "outcome": [outcome(r) for r in rows],
            "raw": [r["raw"] for r in rows],
            "tokens": [r["n_generated_tokens"] for r in rows],
        }
        counts[a] = Counter(per_arm[a]["outcome"])
        sources.append(runs[a]["run_id"])
    kb = write(f"outcomes.{model}.json", {"arms": per_arm}, sorted(set(sources)))
    # A counts-only summary stays inlined, so a page that needs nothing more
    # than the three totals does not fetch a megabyte to get them.
    kb += write(
        f"counts.{model}.json",
        {"total": len(per_arm[ARMS[0]]["outcome"]),
         "arms": {a: [counts[a][CORRECT], counts[a][DETECTABLE], counts[a][SILENT]]
                  for a in ARMS}},
        sorted(set(sources)),
    )
    return counts, kb


def export_example():
    """One real order in all six encodings, with real subword boundaries."""
    from bpe import BPE, tokenizer_file

    tok = {m: BPE(tokenizer_file(hf)) for m, hf in (
        ("qwen", "Qwen/Qwen3.5-0.8B"),
    )}
    runs, _ = headline_runs("qwen")
    rows = read_preds(runs["words"])
    picks = [r for r in rows if "EXTRA" in r["gold"] and len(r["gold"]) < 150][:6]
    examples = []
    for r in picks:
        node = formats.parse_exr(r["gold"])
        enc = {}
        for a in ARMS:
            s = formats.encode(node, a)
            enc[a] = {"text": s, "chars": len(s),
                      "tokens": {m: b.readable(s) for m, b in tok.items()}}
        examples.append({"src": r["src"], "gold": r["gold"], "encodings": enc})
    return write("examples.json", {"examples": examples},
                 [runs["words"]["run_id"]])


HERO_TITLE = "Parsed, Plausible, Wrong"


def _fields(exr):
    import re
    return re.findall(r"\((\w+) ([A-Z_0-9]+) \)", exr)


def export_hero():
    """The title in real tokens, and one real silent error to reveal.

    The hero claims the model parsed something plausible and got it wrong, so
    the example has to be an actual saved prediction rather than a staged one.
    This picks the smallest one-field disagreement, which is the clearest to
    show and the hardest to dismiss.
    """
    from bpe import BPE, tokenizer_file

    b = BPE(tokenizer_file())
    runs, _ = headline_runs("qwen")
    best = None
    for arm in ("words", "dsl"):
        for r in read_preds(runs[arm]):
            if r["correct"] or r["parsed"] is None:
                continue
            g, p = set(_fields(r["gold"])), set(_fields(r["parsed"]))
            only_g, only_p = sorted(g - p), sorted(p - g)
            # One field each way: the model swapped a value rather than losing
            # or inventing structure, which is the cleanest thing to point at.
            if len(only_g) == 1 and len(only_p) == 1 and len(r["src"]) < 80:
                score = len(r["src"])
                if best is None or score < best[0]:
                    best = (score, arm, r, only_g[0], only_p[0])
    if best is None:
        print("  (no one-field silent error found; hero reveal unavailable)")
        return 0.0

    _, arm, r, gold_field, pred_field = best
    payload = {
        "title": {
            "text": HERO_TITLE,
            "tokens": b.readable(HERO_TITLE),
            "ids": b.ids(HERO_TITLE),
        },
        "silentError": {
            "arm": arm,
            "src": r["src"],
            "raw": r["raw"],
            "gold": r["gold"],
            "parsed": r["parsed"],
            "goldFields": _fields(r["gold"]),
            "predFields": _fields(r["parsed"]),
            "diff": {
                "field": pred_field[0],
                "got": pred_field[1],
                "expected": gold_field[1],
            },
            "tokens": b.readable(r["raw"]),
        },
    }
    return write("hero.json", payload, [runs[arm]["run_id"]])


def export_idle():
    """Real orders for the Model Window to cycle through when nothing is happening."""
    from bpe import BPE, tokenizer_file

    b = BPE(tokenizer_file())
    runs, _ = headline_runs("qwen")
    rows = read_preds(runs["words"])
    picked, seen = [], {CORRECT: 0, DETECTABLE: 0, SILENT: 0}
    for r in rows:
        o = outcome(r)
        # A spread of outcomes, so the idle loop is not all green.
        if seen[o] >= 4 or len(r["src"]) > 64 or not r["raw"].strip():
            continue
        seen[o] += 1
        picked.append({
            "src": r["src"],
            "raw": r["raw"],
            "outcome": o,
            "tokens": b.readable(r["raw"]),
            "inputTokens": b.readable(r["src"]),
            "fields": _fields(r["parsed"]) if r["parsed"] else [],
            "goldFields": _fields(r["gold"]),
        })
        if len(picked) >= 12:
            break
    return write("idle.json", {"orders": picked}, [runs["words"]["run_id"]])


# The DSL markers a reader can respell, and what each becomes. Cycling these is
# how the visitor builds a seventh format, so each variant has to be tokenized
# for real rather than estimated.
RESPELL = {
    "+": ["+", "extra"],
    "!": ["!", "no"],
    "p": ["p", "pizza"],
    "d": ["d", "drink"],
}


def _variants(dsl):
    """Every combination of marker spellings for one DSL string."""
    import itertools
    import re

    keys = [k for k in RESPELL if re.search(re.escape(k), dsl)]
    out = []
    for combo in itertools.product(*[RESPELL[k] for k in keys]):
        text = dsl
        for k, v in zip(keys, combo):
            if k in ("+", "!"):
                text = text.replace(k, v + " " if v not in "+!" else v)
            else:
                text = re.sub(rf"\b{re.escape(k)}\b", v, text)
        out.append({"spell": dict(zip(keys, combo)), "text": " ".join(text.split())})
    return keys, out


def export_tickets():
    """Per test order: the six gold spellings, their real token counts, and the
    token boundaries needed to draw the splits without inserting spaces.

    Boundaries are character offsets rather than token strings, which is what
    keeps this file small enough to ship.
    """
    from bpe import BPE, tokenizer_file

    toks = {
        "qwen": BPE(tokenizer_file("Qwen/Qwen3.5-0.8B")),
        "lfm2": BPE(tokenizer_file("LiquidAI/LFM2.5-350M")),
    }
    runs, _ = headline_runs("qwen")
    rows = read_preds(runs["words"])

    def bounds(b, text):
        out, n = [], 0
        for t in b.tokens(text):
            n += len(t)
            out.append(n)
        return out

    tickets = []
    for r in rows:
        node = formats.parse_exr(r["gold"])
        enc = {}
        for a in ARMS:
            text = formats.encode(node, a)
            enc[a] = {
                "t": text,
                "n": {m: len(b.tokens(text)) for m, b in toks.items()},
                "b": {m: bounds(b, text) for m, b in toks.items()},
            }
        tickets.append({"i": r["id"], "src": r["src"], "enc": enc})
    return write("tickets.json", {"tickets": tickets}, [runs["words"]["run_id"]])


def export_respell():
    """Token counts for every marker spelling of a few real orders, so the
    visitor's own format is measured rather than guessed."""
    from bpe import BPE, tokenizer_file

    toks = {
        "qwen": BPE(tokenizer_file("Qwen/Qwen3.5-0.8B")),
        "lfm2": BPE(tokenizer_file("LiquidAI/LFM2.5-350M")),
    }
    runs, _ = headline_runs("qwen")
    rows = read_preds(runs["words"])[:60]
    out = []
    for r in rows:
        node = formats.parse_exr(r["gold"])
        dsl = formats.encode(node, "dsl")
        keys, vs = _variants(dsl)
        if not keys:
            continue
        for v in vs:
            v["n"] = {m: len(b.tokens(v["text"])) for m, b in toks.items()}
        out.append({"i": r["id"], "keys": keys, "variants": vs})
        if len(out) >= 40:
            break
    return write("respell.json", {"orders": out}, [runs["words"]["run_id"]])


def export_architecture():
    """The Stack is a picture of the real model, so it reads the real config.

    Qwen3.5 interleaves linear-attention and full-attention blocks; the site
    lights different regions per section, which is only honest if the regions
    correspond to actual layers. Read from the tokenizer cache rather than
    hardcoded, so a model swap cannot leave a lying diagram behind.
    """
    from bpe import HUB

    out = {}
    for name, repo in (("qwen", "Qwen/Qwen3.5-0.8B"),):
        root = HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
        hits = sorted(root.glob("*/config.json")) if root.exists() else []
        if not hits:
            print(f"  (no cached config for {repo}; skipping architecture)")
            continue
        c = json.loads(hits[0].read_text())
        t = c.get("text_config", c)
        out[name] = {
            "repo": repo,
            "layers": t["num_hidden_layers"],
            "layerTypes": t.get("layer_types", []),
            "hiddenSize": t["hidden_size"],
            "vocabSize": t["vocab_size"],
            "heads": t["num_attention_heads"],
        }
    if out:
        return write("architecture.json", {"models": out},
                     [v["repo"] for v in out.values()])
    return 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing", OUT)
    total = 0.0
    fmts, kb = export_formats()
    total += kb

    runs, _ = headline_runs("qwen")
    _, kb = export_orders(runs)
    total += kb

    for model in MODELS:
        r, _ = headline_runs(model)
        if len(r) == len(ARMS):
            counts, kb = export_outcomes(model, r)
            total += kb
            for a in ARMS:
                c = counts[a]
                print(f"      {a:10} correct {c[CORRECT]:5}  "
                      f"detectable {c[DETECTABLE]:4}  silent {c[SILENT]:4}")

    total += export_example()
    total += export_architecture()
    total += export_hero()
    total += export_idle()
    total += export_tickets()
    total += export_respell()

    # Named so the site can fail loudly instead of rendering an empty section.
    absent = {
        "confidence": "per-token log probabilities are not saved in runs/",
        "internals": "layer activations are not saved in runs/",
        "checkpoints": "train.py saves only the best step, not a checkpoint series",
        "largeModelBaseline": "no large-model run exists; only published numbers",
    }
    write("missing.json", {"absent": absent}, [])
    print(f"\ntotal {total / 1024:.2f} MB")
    print("sections without data:", ", ".join(absent))


if __name__ == "__main__":
    main()
