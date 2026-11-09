"""Section 1 of analysis.md: integrity checks, before any result is read.

    uv run --extra pizza python tools/analysis/integrity.py

Writes analysis/out/integrity.txt (pass/fail per check) and
analysis/out/integrity/*.json (the detail behind each line).
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as C  # noqa: E402

lines = []


def check(name, ok, detail, warn=False):
    """FAIL invalidates results until explained; WARN is a finding to disclose."""
    lines.append(f"{'PASS' if ok else 'WARN' if warn else 'FAIL'}  {name}: {detail}")
    print(lines[-1], flush=True)


def save(name, data):
    C.write(f"integrity/{name}.json", json.dumps(data, indent=1, default=str))


# ---------- I1 completeness ----------


def i1():
    m = C.manifest()
    missing, bad = [], []
    for model, arm, size, seed in C.cells():
        p = C.run_dir(model, arm, size, seed) / "metrics.json"
        if not p.exists():
            missing.append(str(p))
            continue
        r = C.metrics(model, arm, size, seed)
        if not r.get("complete") or r.get("tag") or r["stage"] not in ("grid", "sweep") \
                or (r["arm"], r["size"], r["seed"]) != (arm, size, seed):
            bad.append(r["run_id"])
    others = {}
    for model in m["models"]:
        chosen = {C.run_dir(model, *c[1:]).name for c in C.cells() if c[0] == model}
        dirs = [d.name for d in C.base(model).iterdir() if (d / "metrics.json").exists()]
        others[model] = sorted(set(dirs) - chosen)
    save("i1_completeness", {"missing": missing, "bad": bad, "not_in_grid": others})
    n = len(C.cells())
    check("I1 completeness", not missing and not bad,
          f"{n - len(missing) - len(bad)}/{n} grid cells complete; outside the grid "
          + ", ".join(f"{k} {len(v)} (sweep at other LRs, rep2)" for k, v in others.items()))


# ---------- I2 recomputed exact match ----------


def _recompute(cell):
    from bench.tasks.pizza import metrics

    model, arm, size, seed = cell
    out = {}
    for split in ("dev", "test"):
        rows = C.predictions(model, arm, size, seed, split)
        res, _, em = metrics.evaluate([r["raw"] for r in rows], [r["gold"] for r in rows], arm)
        stored = C.metrics(model, arm, size, seed)[split]
        flips = sum(bool(e) != bool(r["correct"]) for e, r in zip(em, rows))
        out[split] = {"em": res["em"], "stored_em": stored["em"],
                      "validity": res["parse_validity"], "stored_validity": stored["parse_validity"],
                      "row_flips": flips}
    return cell, out


def i2():
    with Pool() as pool:
        res = dict(pool.map(_recompute, C.cells(), chunksize=4))
    worst = 0.0
    flips = 0
    for cell, out in res.items():
        for split, v in out.items():
            worst = max(worst, abs(v["em"] - v["stored_em"]), abs(v["validity"] - v["stored_validity"]))
            flips += v["row_flips"]
    save("i2_recompute", {"/".join(map(str, k)): v for k, v in res.items()})
    check("I2 recomputed EM and validity equal stored", worst < 1e-12 and flips == 0,
          f"largest difference {worst:.2e}, rows whose verdict changed {flips}, "
          f"over {len(res)} runs x dev+test")


# ---------- I5 generation contract ----------


def i5():
    pads = Counter(C.metrics(*c).get("tokenizer", {}).get("padding_side") for c in C.cells())
    diffs = []
    for model in C.MODELS:
        for line in open(C.base(model) / "latency_gpu.jsonl"):
            r = json.loads(line)
            rows = [json.loads(x) for x in open(C.base(model) / r["run_id"] / "test_predictions.jsonl")]
            saved = sum(bool(x["correct"]) for x in rows[: r["n"]]) / r["n"]
            diffs.append({"model": model, "run": r["run_id"], "batch1": r["em"], "batched": saved,
                          "diff_utterances": round((r["em"] - saved) * r["n"])})
    save("i5_generation", {"padding_side": pads, "batch1_vs_batched": diffs})
    worst = max(abs(d["diff_utterances"]) for d in diffs)
    net = sum(d["diff_utterances"] for d in diffs)
    check("I5 generation contract", set(pads) == {"left"} and worst <= 2,
          f"padding {dict(pads)}; batch-1 vs batched on {len(diffs)} runs: at most {worst} "
          f"utterances of 200 differ, net {net:+d}")


# ---------- I6 token cap hits ----------


def _loop(raw):
    toks = raw.split()
    for n in range(1, 8):
        if len(toks) >= 4 * n and toks[-n:] * 3 == toks[-3 * n:]:
            return True
    return False


def i6():
    out = []
    for c in C.cells():
        m = C.metrics(*c)
        for split in ("dev", "test"):
            if not m[split].get("hit_cap_rate"):
                continue
            for r in C.predictions(*c, split):
                if r["n_generated_tokens"] >= m["cap"]:
                    out.append({"run": m["run_id"], "model": c[0], "split": split, "id": r["id"],
                                "tokens": r["n_generated_tokens"], "cap": m["cap"],
                                "loop": _loop(r["raw"]), "starts_with_refusal": r["raw"].lstrip().startswith(("(REJECT", "(R)", "?", '{"reject')),
                                "tail": r["raw"][-80:]})
    runs = len({(o["model"], o["run"]) for o in out})
    total = sum(C.metrics(*c)[s]["n"] for c in C.cells() for s in ("dev", "test"))
    save("i6_cap_hits", out)
    # The cap is 1.25x the longest gold target of its arm, so a generation that
    # reaches it is longer than every correct ticket: it is wrong at any cap, and
    # the cap never decided a verdict. Disclosed because the paper says zero.
    check("I6 token cap never truncated a possible correct answer", True,
          f"{len(out)} of {total} generations reached the cap ({runs} runs), all longer than any "
          f"gold ticket; {sum(o['loop'] for o in out)} are exact repetition loops, the rest invent "
          f"extra lines. The paper's 'zero for every run' must change")


# ---------- I7, I8 learning rate ----------


def i7_i8():
    m = C.manifest()
    out = {}
    edge, flips = [], []
    for model in m["models"]:
        rows = defaultdict(lambda: defaultdict(list))
        for d in C.base(model).iterdir():
            p = d / "metrics.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            if r.get("stage") != "sweep" and not (r.get("stage") == "grid" and r["size"] == 100_000
                                                   and r["seed"] in (0, 1, 2)):
                continue
            if r["size"] != 100_000 or r["seed"] not in (0, 1, 2) or r.get("tag"):
                continue
            rows[r["arm"]][r["lr"]].append((r["dev"]["em"], r["best_selection_score"]))
        out[model] = {}
        for arm, by in rows.items():
            em = {lr: statistics.fmean(x[0] for x in v) for lr, v in by.items()}
            comb = {lr: statistics.fmean(x[1] for x in v) for lr, v in by.items()}
            chosen = C.lrs(model)[arm]
            out[model][arm] = {"chosen": chosen, "dev_em": em, "combined": comb,
                               "n": {lr: len(v) for lr, v in by.items()}}
            if chosen in (min(em), max(em)):
                edge.append(f"{model}/{arm}={chosen:g}")
            if max(comb, key=comb.get) != chosen:
                flips.append(f"{model}/{arm}: dev EM picks {chosen:g}, combined picks {max(comb, key=comb.get):g}")
    save("i7_i8_learning_rate", out)
    check("I7 chosen learning rate inside the candidate grid", not edge,
          f"at an edge of {{1e-4, 2e-4, 5e-4}}: {', '.join(edge) or 'none'}", warn=True)
    check("I8 LR choice unchanged under the checkpoint criterion", not flips,
          "; ".join(flips) or "same LR under dev EM and the combined score for every arm", warn=True)


# ---------- I9, I10 training length ----------


def i9_i10():
    out = defaultdict(list)
    for c in C.cells():
        r = C.metrics(*c)
        out[(c[0], c[1], c[2])].append((r["best_step"], r["epochs_at_best"]))
    table, at_max, at_min = [], 0, 0
    for (model, arm, size), v in sorted(out.items()):
        steps = [s for s, _ in v]
        at_max += sum(s >= 3900 for s in steps)
        at_min += sum(s <= 1000 for s in steps)
        table.append({"model": model, "arm": arm, "size": size, "best_steps": steps,
                      "epochs_at_best": [round(e, 3) for _, e in v]})
    save("i9_i10_training_length", table)
    n = len(C.cells())
    check("I9 early stopping found its best inside the budget", at_max == 0,
          f"best checkpoint at >= 3900 of 4000 steps in {at_max}/{n} runs (may be under-trained); "
          f"at <= 1000 in {at_min}/{n}", warn=True)
    ep = defaultdict(list)
    for t in table:
        ep[t["size"]] += t["epochs_at_best"]
    detail = ", ".join(f"n={s}: {min(v):.2f}-{max(v):.2f}" for s, v in sorted(ep.items()))
    check("I10 examples seen (epochs at best checkpoint, by size)", True, detail)


# ---------- I11 noise floor ----------


def i11():
    out = {}
    for model in C.MODELS:
        d = json.loads((C.base(model) / "determinism.json").read_text())
        seeds = [C.metrics(model, d["arm"], d["size"], s)["test"]["em"] for s in C.manifest()["seeds"]]
        out[model] = {"rerun_diff": d["test_abs_diff"], "arm": d["arm"], "size": d["size"],
                      "seed_sd": statistics.stdev(seeds), "seed_range": max(seeds) - min(seeds)}
    save("i11_noise", out)
    check("I11 rerun gap vs seed spread (same cell)", True,
          "; ".join(f"{k}: rerun {v['rerun_diff']:.3f}, seed SD {v['seed_sd']:.3f}, "
                    f"seed range {v['seed_range']:.3f} ({v['arm']} n={v['size']})" for k, v in out.items()))


# ---------- I12 exports ----------


def i12():
    out = []
    for p in sorted((C.ROOT / "models/pizza").glob("*.json")):
        m = json.loads(p.read_text())
        v = m["files"]["q8_0"]["verify"]
        out.append({"model": m["model"], "run": m["run_id"], "agreement": v["agreement_with_saved"],
                    "em_gguf": v["em_gguf"], "em_saved": v["em_saved_same_rows"],
                    "disagreements": v["first_disagreements"]})
    save("i12_exports", out)
    check("I12 GGUF exports agree >= 95% with their runs", all(o["agreement"] >= 0.95 for o in out),
          "; ".join(f"{o['model']} {o['agreement']:.0%} (gguf em {o['em_gguf']:.3f}, saved {o['em_saved']:.3f})"
                    for o in out) + "; deployment only, no paper number comes from a GGUF", warn=True)


# ---------- I13 identical evaluation inputs ----------


def i13():
    bad = []
    for kind in ("negatives", "corrections"):
        ref = {}
        for model, arm, size, seed in C.cells():
            p = C.run_dir(model, arm, size, seed) / f"{kind}_predictions.jsonl"
            srcs = tuple(json.loads(x)["src"] for x in open(p))
            if seed in ref and ref[seed] != srcs:
                bad.append(f"{kind} {model}/{arm}/{size}/{seed}")
            ref.setdefault(seed, srcs)
    check("I13 negatives and corrections inputs identical across models, formats and sizes per seed",
          not bad, f"mismatches {len(bad)}" + (f": {bad[:3]}" if bad else ""))


# ---------- I14 contamination ----------


def i14():
    log = (C.ROOT.parent / "server-runs/logs/prepare-check.log")
    found = []
    for model in C.MODELS:
        p = C.base(model) / "logs/check.log"
        text = p.read_text() if p.exists() else ""
        hits = re.findall(r"overlap n=(\d+)\s+seed=(\d+) vs (\w+)\s+exact (\d+) normalized (\d+)", text)
        found.append((model, len(hits), sum(int(h[3]) + int(h[4]) for h in hits)))
    ok = all(n == 25 * 2 and c == 0 for _, n, c in found)
    check("I14 no dev/test utterance in any training subset (from each model's check.log)", ok,
          "; ".join(f"{m}: {n} size/seed/split checks, {c} overlapping" for m, n, c in found)
          + ("" if log.exists() else ""))

    from bench.tasks.pizza import corrections, data

    # Every generated correction contains a lead cue (CUES, RESTART) or a cancel
    # phrase, so it can only equal a dev/test utterance that contains one too.
    import re as _re

    frags = {f.strip(" ,") for group in corrections.TEMPLATES for t in group
             for f in _re.split(r"\{\w+\}", t)}
    phrases = sorted(set(corrections.CUES + corrections.RESTART)
                     | {f for f in frags if len(f.split()) >= 2})
    # Read row by row: in PIZZA these keep our meaning ("skip the ham" = no ham,
    # "and no" = negation) or are plain connectors, so they teach nothing opposite.
    same_meaning = {"and no", "without the", "skip the", "on it", "and also", "no {new}"}
    phrases = [p for p in phrases if p not in same_meaning]
    held = {r["src"] for s in ("dev", "test") for r in data.load(s)}
    hit = sorted({(p, r) for r in held for p in phrases if f" {p} " in f" {r} "})
    test = {r["src"] for r in data.load("test")}
    on_test = sorted({r for _, r in hit if r in test})
    save("i14b_cue_phrases_in_eval", {"all": hit, "test": on_test})
    # Equality is impossible (a generated correction is a training utterance plus a
    # spoken correction); what can happen is a cue phrase meaning something else in
    # human PIZZA text: "but hold on chicken" there means "no chicken".
    per = Counter(p for p, r in hit if r in test)
    check("I14b correction cue phrases that also occur in human dev/test text", not hit,
          f"{len({r for _, r in hit})} utterances ({len(on_test)} in test, at most "
          f"{len(on_test) / len(test):.4f} test EM) use these phrases with PIZZA's own meaning; "
          f"test rows per phrase {dict(per.most_common())}",
          warn=True)


def main():
    for f in (i1, i2, i5, i6, i7_i8, i9_i10, i11, i12, i13, i14):
        f()
    lines.insert(0, f"integrity checks, {C.ROOT}\n")
    print("\nwrote", C.write("integrity.txt", "\n".join(lines) + "\n"))


if __name__ == "__main__":
    main()
