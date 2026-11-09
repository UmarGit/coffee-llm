"""Sections 2 to 4 of analysis.md: cell tables, each paper claim, the new results.

    uv run --extra pizza python tools/analysis/claims.py

Writes analysis/out/tables/*.csv and analysis/out/claims.md. Verdicts follow the
keep / soften / drop rules written in analysis.md before these numbers existed.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import data as D  # noqa: E402
import stats as S  # noqa: E402

M = C.manifest()
SIZES, HEAD = M["sizes"], max(M["sizes"])
md = []


def say(s=""):
    md.append(s)


def csv_out(name, rows):
    p = C.OUT / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def seed_means(model, arm, size, name):
    return D.get(model, arm, size, name).mean(axis=1)


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(r, pct=False):
    k = 100 if pct else 1
    return f"{r['est'] * k:+.3f} [{r['lo'] * k:+.3f}, {r['hi'] * k:+.3f}]"


def sep(r, adj):
    return adj is not None and adj < 0.05


# ---------- section 2: every cell ----------


def cell_table():
    rows = []
    for model in M["models"]:
        for arm in M["arms"]:
            for size in SIZES:
                row = {"model": model, "arm": arm, "size": size}
                for k in ("correct", "silent", "d_parse", "d_refuse", "tokens", "hallucinated", "missing",
                          "neg_handled", "fix_em", "fix_applied", "probe_em"):
                    v = seed_means(model, arm, size, k)
                    row[f"{k}_mean"] = round(float(v.mean()), 4)
                    row[f"{k}_sd"] = round(float(v.std(ddof=1)), 4)
                    row[f"{k}_min"] = round(float(v.min()), 4)
                    row[f"{k}_max"] = round(float(v.max()), 4)
                rows.append(row)
    csv_out("cells.csv", rows)

    say("## Headline cells (full pool, mean ± seed SD over 5 seeds)")
    say()
    for model in M["models"]:
        say(f"**{C.LABEL[model]}**")
        say()
        say("| format | A (EM) | S silent | D-parse | D-refuse | tokens | halluc./utt | negatives handled | corrections EM | probes EM |")
        say("|---|---|---|---|---|---|---|---|---|---|")
        for arm in M["arms"]:
            r = next(x for x in rows if (x["model"], x["arm"], x["size"]) == (model, arm, HEAD))
            say(f"| {C.LABEL[arm]} | {r['correct_mean']:.3f} ± {r['correct_sd']:.3f} | {r['silent_mean']:.3f} ± {r['silent_sd']:.3f} "
                f"| {r['d_parse_mean']:.3f} | {r['d_refuse_mean']:.3f} | {r['tokens_mean']:.1f} | {r['hallucinated_mean']:.3f} "
                f"| {r['neg_handled_mean']:.3f} | {r['fix_em_mean']:.3f} | {r['probe_em_mean']:.3f} |")
        say()
    say("EM by size (mean over seeds):")
    say()
    say("| model | format | " + " | ".join(f"n={s}" for s in SIZES) + " |")
    say("|---|---|" + "---|" * len(SIZES))
    for model in M["models"]:
        for arm in M["arms"]:
            v = [next(x for x in rows if (x["model"], x["arm"], x["size"]) == (model, arm, s)) for s in SIZES]
            say(f"| {model} | {C.LABEL[arm]} | " + " | ".join(f"{x['correct_mean']:.3f}" for x in v) + " |")
    say()
    return rows


# ---------- section 3: claims ----------

verdicts = []


def verdict(cid, text, v, why):
    verdicts.append({"claim": cid, "text": text, "verdict": v, "why": why})


def pair_family(model, a, b, metric="correct"):
    res = [S.diff(D.get(model, a, s, metric), D.get(model, b, s, metric)) for s in SIZES]
    adj = S.holm([r["p"] for r in res])
    for r, p in zip(res, adj):
        r["p_holm"] = p
    return res


def family_table(title, fam, label, pct=False):
    say(f"**{title}**")
    say()
    say("| model | size | estimate [95% CI] | Holm p | separates |")
    say("|---|---|---|---|---|")
    for model, res in fam.items():
        for s, r in zip(SIZES, res):
            say(f"| {model} | {s} | {ci(r)} | {f3(r['p_holm'])} | {'yes' if sep(r, r['p_holm']) else 'no'} |")
    say()
    csv_out(label, [{"model": m, "size": s, **r} for m, res in fam.items() for s, r in zip(SIZES, res)])


def counts(res):
    pos = sum(r["est"] > 0 for r in res)
    neg = sum(r["est"] < 0 for r in res)
    sp = sum(sep(r, r["p_holm"]) and r["est"] > 0 for r in res)
    sn = sum(sep(r, r["p_holm"]) and r["est"] < 0 for r in res)
    return pos, neg, sp, sn


def c1():
    fam = {}
    for model in M["models"]:
        res = []
        for s in SIZES:
            res.append(S.slope([D.get(model, a, s, "correct") for a in C.LADDER],
                               [D.get(model, a, s, "tokens") for a in C.LADDER]))
        adj = S.holm([r["p"] for r in res])
        for r, p in zip(res, adj):
            r["p_holm"] = p
        fam[model] = res
    say("### C1. Ladder slope (EM per 10 tokens, ladder formats EXR, EXR-short, TAGGED, DSL)")
    say()
    say("Positive = removing tokens lowers EM.")
    say()
    family_table("slope", fam, "c1_slope.csv")
    small = {m: fam[m][0] for m in fam}
    both = all(r["lo"] > 0 for r in small.values())
    one = any(r["lo"] > 0 for r in small.values())
    allsizes_pos = {m: sum(r["est"] > 0 for r in fam[m]) for m in fam}
    neg_sep = any(r["hi"] < 0 for res in fam.values() for r in res)
    if both:
        v = "keep"
    elif one or any(n >= 3 for n in allsizes_pos.values()):
        v = "soften"
    else:
        v = "drop"
    if neg_sep and not both:
        v = "drop" if not one else v
    verdict("C1", "Shortening alone does not improve accuracy (positive ladder slope at n=1000)", v,
            "; ".join(f"{m}: n=1000 slope {ci(small[m])}, positive at {allsizes_pos[m]}/5 sizes" for m in fam))


def simple_pair(cid, text, a, b, keep_rule):
    fam = {m: pair_family(m, a, b) for m in M["models"]}
    say(f"### {cid}. {C.LABEL[a]} minus {C.LABEL[b]}, test EM")
    say()
    family_table(f"{C.LABEL[a]} − {C.LABEL[b]}", fam, f"{cid.lower()}_{a}_{b}.csv")
    v, why = keep_rule(fam)
    verdict(cid, text, v, why)
    return fam


def c2(fam):
    head = {m: fam[m][-1] for m in fam}
    seps = [m for m, r in head.items() if sep(r, r["p_holm"]) and r["est"] > 0]
    v = "keep" if len(seps) == 2 else "soften" if seps else "drop"
    return v, "; ".join(f"{m} full pool {ci(r)}, Holm p {f3(r['p_holm'])}" for m, r in head.items())


def c3(fam):
    allres = [r for res in fam.values() for r in res]
    pos, neg, sp, sn = counts(allres)
    if sn:
        v = "drop"
    elif neg == 0 and sp >= 6:
        v = "keep"
    elif neg == 0:
        v = "soften"
    else:
        v = "soften" if sp > 0 else "drop"
    return v, f"WORDS ahead in {pos}/10 cells, behind in {neg}; separates in favour {sp}, against {sn}"


def c4(fam):
    allres = [r for res in fam.values() for r in res]
    pos, neg, sp, sn = counts(allres)
    v = "drop" if sn else "keep" if pos >= neg else "soften"
    return v, f"EXR ahead in {pos}/10 cells, behind in {neg}; separates in favour {sp}, against {sn}"


def c5():
    fam = {m: pair_family(m, "words", "exr") for m in M["models"]}
    say("### C5. WORDS minus EXR, test EM")
    say()
    family_table("WORDS − EXR", fam, "c5_words_exr.csv")
    rows = []
    for m in M["models"]:
        r = fam[m][-1]
        sd = statistics.stdev(seed_means(m, "words", HEAD, "correct"))
        ratio = seed_means(m, "exr", HEAD, "tokens").mean() / seed_means(m, "words", HEAD, "tokens").mean()
        if r["lo"] <= 0 <= r["hi"] or r["lo"] > 0:
            v = "keep"
        elif not sep(r, r["p_holm"]) or -r["est"] <= sd:
            v = "soften"
        else:
            v = "drop"
        rows.append((m, v, f"{m}: {ci(r)}, WORDS seed SD {sd:.3f}, EXR/WORDS tokens {ratio:.2f}x"))
    v = min((x[1] for x in rows), key=["drop", "soften", "keep"].index)
    verdict("C5", "WORDS matches EXR on accuracy with far fewer tokens", v, "; ".join(x[2] for x in rows))


def c6():
    say("### C6. Silent rate, EXR minus WORDS (positive = WORDS silent less)")
    say()
    fam = {m: pair_family(m, "exr", "words", "silent") for m in M["models"]}
    family_table("S(EXR) − S(WORDS)", fam, "c6_silent.csv")
    out = []
    say("| model | size | S(EXR)/S(WORDS) [95% CI] |")
    say("|---|---|---|")
    for m in M["models"]:
        for s in SIZES:
            r = S.ratio(D.get(m, "exr", s, "silent"), D.get(m, "words", s, "silent"))
            say(f"| {m} | {s} | {r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] |")
            if s == HEAD:
                out.append((m, r, fam[m][-1]))
    say()
    vs = []
    for m, r, d in out:
        if sep(d, d["p_holm"]) and d["est"] > 0:
            vs.append("keep" if r["est"] >= 1.5 else "soften")
        else:
            vs.append("drop")
    v = min(vs, key=["drop", "soften", "keep"].index)
    verdict("C6", "WORDS halves the silent rate of EXR", v,
            "; ".join(f"{m}: full pool ratio {r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}], diff {ci(d)}" for m, r, d in out))


def c7():
    fam = {m: pair_family(m, "exr", "words", "hallucinated") for m in M["models"]}
    say("### C7. Hallucinated entities per utterance, EXR minus WORDS")
    say()
    family_table("halluc(EXR) − halluc(WORDS)", fam, "c7_hallucinated.csv")
    rows = []
    for m in M["models"]:
        d = fam[m][-1]
        r = S.ratio(D.get(m, "exr", HEAD, "hallucinated"), D.get(m, "words", HEAD, "hallucinated"))
        rows.append((m, d, r))
    ok = [sep(d, d["p_holm"]) and d["est"] > 0 for _, d, _ in rows]
    v = "keep" if all(ok) else "soften" if any(ok) else "drop"
    verdict("C7", "WORDS invents fewer entities than EXR (paper: 2.5x)", v,
            "; ".join(f"{m}: ratio {r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}], diff {ci(d)}" for m, d, r in rows))


def c8():
    say("### C8. Latency at batch size 1 (full pool, 5 seeds, 200 test orders each)")
    say()
    say("| model | format | median ms (mean ± SD over seeds) | tokens | tokens/s | own GPU MiB |")
    say("|---|---|---|---|---|---|")
    why = []
    for m in M["models"]:
        lat = {}
        for line in open(C.base(m) / "latency_gpu.jsonl"):
            r = json.loads(line)
            lat.setdefault(r["arm"], []).append(r)
        xs, ys = [], []
        for arm in M["arms"]:
            v = lat[arm]
            ms = [x["median_ms"] for x in v]
            say(f"| {m} | {C.LABEL[arm]} | {statistics.fmean(ms):.0f} ± {statistics.stdev(ms):.0f} "
                f"| {statistics.fmean(x['mean_generated_tokens'] for x in v):.1f} "
                f"| {statistics.fmean(x['tokens_per_second'] for x in v):.1f} "
                f"| {min(x['gpu_memory_used_by_this_process_mib'] for x in v)}-{max(x['gpu_memory_used_by_this_process_mib'] for x in v)} |")
            for x in v:
                xs.append(x["mean_generated_tokens"])
                ys.append(x["median_ms"])
        a, b = np.polyfit(xs, ys, 1)
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2
        sp = statistics.fmean(x["median_ms"] for x in lat["exr"]) / statistics.fmean(x["median_ms"] for x in lat["words"])
        mem = [x["gpu_memory_used_by_this_process_mib"] for v in lat.values() for x in v]
        why.append(f"{m}: EXR/WORDS {sp:.2f}x, ms = {b:.0f} + {a:.1f}/token (R² {r2:.2f}), own memory {min(mem)}-{max(mem)} MiB")
    say()
    verdict("C8", "Latency ~linear in tokens; WORDS ~3.7x faster than EXR; memory barely moves", "see numbers", "; ".join(why))


def c9(c3fam, c5fam):
    say("### C9. Does the gap shrink with size?")
    say()
    why = []
    for name, fam in (("WORDS−DSL", c3fam), ("WORDS−EXR", c5fam)):
        for m, res in fam.items():
            est = [r["est"] for r in res]
            why.append(f"{name} {m}: " + " → ".join(f"{e:+.3f}" for e in est))
    for w in why:
        say(f"- {w}")
    say()
    trends = []
    for fam in (c3fam,):
        for m, res in fam.items():
            est = [r["est"] for r in res]
            trends.append(est[0] == max(est) and est[0] > est[-1])
    v = "keep" if all(trends) else "soften" if any(trends) else "drop"
    verdict("C9", "The format penalty is largest at small data and narrows", v, "; ".join(why))


def c10(c3fam):
    tw = {m: seed_means(m, "words", HEAD, "tokens").mean() for m in M["models"]}
    td = {m: seed_means(m, "dsl", HEAD, "tokens").mean() for m in M["models"]}
    q, l = c3fam["qwen"], c3fam["lfm2"]
    same_sign = all((a["est"] > 0) == (b["est"] > 0) for a, b in zip(q, l))
    lfm_sep = sum(sep(r, r["p_holm"]) for r in l)
    lfm_against = sum(sep(r, r["p_holm"]) and r["est"] < 0 for r in l)
    inverts = tw["lfm2"] > td["lfm2"]
    if lfm_against:
        v = "drop"
    elif same_sign and lfm_sep >= 3 and inverts:
        v = "keep"
    else:
        v = "soften"
    verdict("C10", "Replicates on LFM2, where WORDS costs more tokens than DSL and still wins", v,
            f"pair sign same at all sizes: {same_sign}; LFM2 separates {lfm_sep}/5 (against {lfm_against}); "
            f"tokens WORDS/DSL: qwen {tw['qwen']:.1f}/{td['qwen']:.1f}, lfm2 {tw['lfm2']:.1f}/{td['lfm2']:.1f}")


def c11():
    say("### C11. Ranking at the full pool (difference to the best format; ties = not separating)")
    say()
    why, vs = [], []
    for m in M["models"]:
        for metric, better in (("correct", max), ("silent", min)):
            means = {a: seed_means(m, a, HEAD, metric).mean() for a in M["arms"]}
            best = better(means, key=means.get)
            res = {a: S.diff(D.get(m, a, HEAD, metric), D.get(m, best, HEAD, metric))
                   for a in M["arms"] if a != best}
            adj = dict(zip(res, S.holm([r["p"] for r in res.values()])))
            tied = [a for a in res if adj[a] >= 0.05]
            order = sorted(means, key=means.get, reverse=better is max)
            say(f"- {m} by {'EM' if metric == 'correct' else 'silent rate'}: "
                + ", ".join(f"{C.LABEL[a]} {means[a]:.3f}" for a in order)
                + f" — best {C.LABEL[best]}; not separable from it: {', '.join(C.LABEL[a] for a in tied) or 'none'}")
            words_top = best == "words" or "words" in tied
            vs.append(words_top)
            why.append(f"{m}/{'EM' if metric == 'correct' else 'S'}: best {best}, words tied-or-best {words_top}")
    say()
    v = "keep" if all(vs) else "soften" if any(vs) else "drop"
    verdict("C11", "WORDS is the best format", v, "; ".join(why))


# ---------- section 4: new results ----------


def kind_table(block, prefix, title):
    say(f"### {title}")
    say()
    kinds = sorted({k[len(prefix):] for c in C.cells() for k in C.metrics(*c)[block]
                    if k.startswith(prefix) and not isinstance(C.metrics(*c)[block][k], dict)})
    say("| model | format | " + " | ".join(kinds) + " |")
    say("|---|---|" + "---|" * len(kinds))
    rows = []
    for m in M["models"]:
        for arm in M["arms"]:
            vals = {k: [C.metrics(m, arm, HEAD, s)[block][prefix + k] for s in M["seeds"]] for k in kinds}
            say(f"| {m} | {C.LABEL[arm]} | " + " | ".join(f"{statistics.fmean(v):.3f}" for v in vals.values()) + " |")
            rows.append({"model": m, "arm": arm, **{k: round(statistics.fmean(v), 4) for k, v in vals.items()}})
    say()
    csv_out(f"{block}_{prefix.strip('_')}_by_kind.csv", rows)


def n_section():
    say("## New results (section 4)")
    say()
    kind_table("negatives", "handled_", "N1. Held-out negatives handled, by kind (full pool, mean of 5 seeds)")

    say("### N2. False rejection of real test orders (D-refuse), by size")
    say()
    say("| model | format | " + " | ".join(f"n={s}" for s in SIZES) + " |")
    say("|---|---|" + "---|" * len(SIZES))
    for m in M["models"]:
        for arm in M["arms"]:
            say(f"| {m} | {C.LABEL[arm]} | " + " | ".join(
                f"{seed_means(m, arm, s, 'd_refuse').mean():.3f}" for s in SIZES) + " |")
    say()

    kind_table("corrections", "em_", "N4. Held-out corrections, exact match by kind (full pool)")
    kind_table("corrections", "applied_", "N4b. Held-out corrections, correction applied by kind (full pool)")

    say("### N5. Hand-written correction probes (40), full pool, mean over 5 seeds")
    say()
    say("| model | format | correct | silent | refused | broken | reported failure fixed (of 5 seeds) |")
    say("|---|---|---|---|---|---|---|")
    for m in M["models"]:
        for arm in M["arms"]:
            p = [C.metrics(m, arm, HEAD, s)["corrections"]["probes"] for s in M["seeds"]]
            first = sum(int(D.get(m, arm, HEAD, "probe_em")[s][0]) for s in range(len(M["seeds"])))
            say(f"| {m} | {C.LABEL[arm]} | " + " | ".join(
                f"{statistics.fmean(x[k] for x in p):.1f}" for k in ("correct", "silent", "refused", "broken"))
                + f" | {first}/5 |")
    say()

    say("### N6. Do formats differ on the guardrail? Difference to WORDS at the full pool")
    say()
    say("| model | metric | " + " | ".join(C.LABEL[a] for a in M["arms"] if a != "words") + " |")
    say("|---|---|" + "---|" * (len(M["arms"]) - 1))
    for m in M["models"]:
        for metric in ("neg_handled", "fix_em", "d_refuse"):
            res = {a: S.diff(D.get(m, a, HEAD, metric), D.get(m, "words", HEAD, metric))
                   for a in M["arms"] if a != "words"}
            adj = dict(zip(res, S.holm([r["p"] for r in res.values()])))
            say(f"| {m} | {metric} | " + " | ".join(
                f"{r['est']:+.3f}{'*' if adj[a] < 0.05 else ''}" for a, r in res.items()) + " |")
    say()
    say("`*` = separates (Holm p < 0.05 within the row).")
    say()

    say("### N7. Safest format: silent rate, and silent plus wrongly refused (full pool)")
    say()
    for m in M["models"]:
        s = {a: seed_means(m, a, HEAD, "silent").mean() for a in M["arms"]}
        sr = {a: s[a] + seed_means(m, a, HEAD, "d_refuse").mean() for a in M["arms"]}
        say(f"- {m} by S: " + ", ".join(f"{C.LABEL[a]} {s[a]:.3f}" for a in sorted(s, key=s.get)))
        say(f"- {m} by S + D-refuse: " + ", ".join(f"{C.LABEL[a]} {sr[a]:.3f}" for a in sorted(sr, key=sr.get)))
    say()


def main():
    say("# neg-v1: claims and new results")
    say()
    say("Generated by `tools/analysis/claims.py`. Statistics as fixed in analysis.md section 2: "
        f"5 seeds, hierarchical bootstrap ({S.B} resamples), Holm within each family of 5 sizes.")
    say()
    cell_table()
    say("## Claims (section 3)")
    say()
    c1()
    simple_pair("C2", "EXR-short loses accuracy against EXR", "exr", "exr_short", c2)
    c3fam = simple_pair("C3", "WORDS beats DSL at equal length in every cell", "words", "dsl", c3)
    simple_pair("C4", "EXR is ahead of JSON (consistent, rarely separating)", "exr", "json", c4)
    c5()
    c5fam = {m: json.loads(json.dumps(pair_family(m, "words", "exr"))) for m in M["models"]}
    c6()
    c7()
    c8()
    c9(c3fam, c5fam)
    c10(c3fam)
    c11()
    n_section()
    say("## Verdicts")
    say()
    say("| claim | statement | verdict | evidence |")
    say("|---|---|---|---|")
    for v in verdicts:
        say(f"| {v['claim']} | {v['text']} | **{v['verdict']}** | {v['why']} |")
    csv_out("verdicts.csv", verdicts)
    p = C.write("claims.md", "\n".join(md) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
