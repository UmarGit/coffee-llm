#!/usr/bin/env python3
"""Every LaTeX table and macro in the paper, from the neg-v1 campaign. Nothing is hand typed.

    uv run --extra pizza python paper/make_tables.py

Reads the campaign through tools/analysis (same cache, same bootstrap draws, same
Holm families as analysis/out/claims.md). Rates are printed in percentage points.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "analysis"))
sys.path.insert(0, str(HERE))

import common as C
import errors as E
import stats as S
from palette import MACRO, latex_colors

import data as D

M = C.manifest()
ARMS, MODELS, SIZES, SEEDS = M["arms"], M["models"], M["sizes"], M["seeds"]
HEAD = max(SIZES)
OUT = HERE / "generated"
P = {"qwen": "Q", "lfm2": "L"}
FMT = {a: f"\\fmt{{{C.LABEL[a]}}}" for a in ARMS}
TAB = C.OUT / "tables"
DAG = "$^\\dagger$"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def spct(x, d=1):
    return f"{100 * x:+.{d}f}"


def pretty(n):
    return f"{n:,}".replace(",", "{,}")


def tick(n):
    return f"{n / 1e6:.1f}M".replace(".0M", "M") if n >= 1e6 else f"{n // 1000}k"


def sm(model, arm, size, name):
    return D.get(model, arm, size, name).mean(axis=1)


def mean(model, arm, name, size=HEAD):
    return float(sm(model, arm, size, name).mean())


def sd(model, arm, name, size=HEAD):
    return float(sm(model, arm, size, name).std(ddof=1))


def csv_rows(name):
    with open(TAB / name) as f:
        return list(csv.DictReader(f))


def latency(model):
    out = {}
    for line in open(C.base(model) / "latency_gpu.jsonl"):
        r = json.loads(line)
        out.setdefault(r["arm"], []).append(r)
    return out


LAT = {m: latency(m) for m in MODELS}


def lat_ms(model, arm):
    return statistics.fmean(x["median_ms"] for x in LAT[model][arm])


def family(model, a, b, metric="correct"):
    res = [S.diff(D.get(model, a, s, metric), D.get(model, b, s, metric)) for s in SIZES]
    for r, p in zip(res, S.holm([r["p"] for r in res])):
        r["p_holm"] = p
        r["sep"] = p < 0.05
    return res


def ranking(model, metric, better):
    """Best arm, and arms not separable from it (Holm across the five others)."""
    means = {a: mean(model, a, metric) for a in ARMS}
    best = better(means, key=means.get)
    res = {a: S.diff(D.get(model, a, HEAD, metric), D.get(model, best, HEAD, metric)) for a in ARMS if a != best}
    adj = dict(zip(res, S.holm([r["p"] for r in res.values()])))
    return best, {a for a in res if adj[a] >= 0.05}


def mark(text, arm, best, tied):
    if arm == best:
        return f"\\textbf{{{text}}}"
    return f"\\underline{{{text}}}" if arm in tied else text


def tabular(cols, head, body, foot=""):
    return "\n".join([f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head, "\\midrule", body,
                      "\\bottomrule", "\\end{tabular}"]) + foot + "\n"


# ---------- T1: main results ----------

def t1():
    lines = []
    for i, m in enumerate(MODELS):
        ba, ta = ranking(m, "correct", max)
        bs, ts = ranking(m, "silent", min)
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a in ARMS:
            em = mark(f"{pct(mean(m, a, 'correct'))}", a, ba, ta) + f"\\,{{\\scriptsize$\\pm${pct(sd(m, a, 'correct'))}}}"
            s = mark(f"{pct(mean(m, a, 'silent'))}", a, bs, ts) + f"\\,{{\\scriptsize$\\pm${pct(sd(m, a, 'silent'))}}}"
            lines.append(" & ".join([
                FMT[a], f"{mean(m, a, 'tokens'):.1f}", em, s,
                f"{pct(mean(m, a, 'd_parse'))}\\,{{\\scriptsize$\\pm${pct(sd(m, a, 'd_parse'))}}}",
                f"{pct(mean(m, a, 'd_refuse'))}\\,{{\\scriptsize$\\pm${pct(sd(m, a, 'd_refuse'))}}}",
                f"{mean(m, a, 'hallucinated'):.3f}",
                f"{lat_ms(m, a):.0f}"]) + " \\\\")
    head = ("Format & Tok. & $A$ (EM) $\\uparrow$ & $S$ (silent) $\\downarrow$ & $D_\\text{parse}$ & "
            "$D_\\text{refuse}$ & Inv./utt. & ms \\\\")
    return tabular("lrrrrrrr", head, "\n".join(lines))


# ---------- T2: paired differences across sizes ----------

PAIRS = [("words", "dsl", "correct", "matched length"), ("exr", "json", "correct", "matched length"),
         ("exr", "exr_short", "correct", "same structure, shorter names"), ("words", "exr", "correct", "headline"),
         ("exr", "words", "silent", "silent rate")]


def pair_label(a, b, metric):
    if metric == "silent":
        return f"$S$: {FMT[a]}$-${FMT[b]}"
    return f"{FMT[a]}$-${FMT[b]}"


def t2():
    lines, ci_lines = [], []
    fams = {}
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
            ci_lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        ci_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a, b, metric, _ in PAIRS:
            res = family(m, a, b, metric)
            fams[(m, a, b, metric)] = res
            cells = [f"{spct(r['est'])}{DAG if r['sep'] else ''}" for r in res]
            lines.append(pair_label(a, b, metric) + " & " + " & ".join(cells) + " \\\\")
            cells = [f"{spct(r['est'])} [{spct(r['lo'])}, {spct(r['hi'])}]{DAG if r['sep'] else ''}" for r in res]
            ci_lines.append(pair_label(a, b, metric) + " & " + " & ".join(cells) + " \\\\")
    head = "Difference (points) & " + " & ".join(tick(s) for s in SIZES) + " \\\\"
    return (tabular("l" + "r" * len(SIZES), head, "\n".join(lines)),
            tabular("l" + "r" * len(SIZES), head, "\n".join(ci_lines)), fams)


# ---------- T3: guardrail ----------

def first_probe_fixed(m, a):
    return int(D.get(m, a, HEAD, "probe_em")[:, 0].sum())


def t3():
    lines = []
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        bf, tf = ranking(m, "fix_em", max)
        for a in ARMS:
            fix = mark(pct(mean(m, a, "fix_em")), a, bf, tf)
            lines.append(" & ".join([
                FMT[a], pct(mean(m, a, "neg_handled")), pct(mean(m, a, "d_refuse")),
                fix + f"\\,{{\\scriptsize$\\pm${pct(sd(m, a, 'fix_em'))}}}",
                pct(mean(m, a, "fix_applied")),
                f"{40 * mean(m, a, 'probe_em'):.1f}"]) + " \\\\")
    head = ("Format & Neg.\\ handled $\\uparrow$ & Real refused $\\downarrow$ & Corr.\\ EM $\\uparrow$ & "
            "Corr.\\ applied & Probes /40 \\\\")
    return tabular("lrrrrr", head, "\n".join(lines))


# ---------- appendix tables ----------

def a_sizes():
    lines = []
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{1 + 2 * len(SIZES)}}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a in ARMS:
            cells = []
            for s in SIZES:
                cells += [f"{pct(mean(m, a, 'correct', s))}{{\\tiny$\\pm${pct(sd(m, a, 'correct', s))}}}",
                          f"{pct(mean(m, a, 'silent', s))}"]
            lines.append(FMT[a] + " & " + " & ".join(cells) + " \\\\")
    head = (" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{tick(s)}}}" for s in SIZES) + " \\\\\n"
            + " & " + " & ".join(["$A$", "$S$"] * len(SIZES)) + " \\\\")
    return tabular("l" + "rr" * len(SIZES), head, "\n".join(lines))


def a_kinds(block, prefix, kinds, names):
    lines = []
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{1 + len(kinds)}}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a in ARMS:
            vals = [statistics.fmean(C.metrics(m, a, HEAD, s)[block][prefix + k] for s in SEEDS) for k in kinds]
            lines.append(FMT[a] + " & " + " & ".join(pct(v) for v in vals) + " \\\\")
    return tabular("l" + "r" * len(kinds), "Format & " + " & ".join(names) + " \\\\", "\n".join(lines))


SWEEP_SEEDS = (0, 1, 2)


def sweep():
    """Dev EM per learning rate at the selection size, mean over the sweep seeds."""
    size = 100_000
    rows = {}
    for m in MODELS:
        for a in ARMS:
            for d in C.base(m).glob(f"{a}-n{size}-s*-lr*"):
                if d.name.endswith("rep2") or not (d / "metrics.json").exists():
                    continue
                k = json.loads((d / "metrics.json").read_text())
                if k.get("tag") or k["arm"] != a or k["seed"] not in SWEEP_SEEDS:
                    continue
                rows.setdefault((m, a, k["lr"]), []).append(k["dev"]["em"])
    return rows


def a_lr():
    rows = sweep()
    lrs = sorted({k[2] for k in rows})
    lines = []
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{2 + len(lrs)}}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a in ARMS:
            chosen = C.lrs(m)[a]
            cells = []
            for lr in lrs:
                v = rows.get((m, a, lr))
                t = f"{pct(statistics.fmean(v), 2)}" if v else "--"
                cells.append(f"\\textbf{{{t}}}" if lr == chosen else t)
            edge = chosen in (min(lrs), max(lrs))
            lines.append(FMT[a] + " & " + " & ".join(cells) + f" & {'edge' if edge else ''} \\\\")
    head = "Format & " + " & ".join(f"{lr:.0e}".replace("e-0", "e-") for lr in lrs) + " & \\\\"
    n_seeds = max(len(v) for v in rows.values())
    return tabular("l" + "r" * len(lrs) + "l", head, "\n".join(lines)), n_seeds


def a_errors():
    rows = E.load()
    kinds = [k for k in E.KINDS]
    lines = []
    for i, m in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{2 + len(kinds)}}}{{l}}{{\\textit{{{C.LABEL[m]}}}}} \\\\")
        for a in ARMS:
            r = next(x for x in rows if (x["model"], x["arm"]) == (m, a))
            lines.append(FMT[a] + f" & {int(r['n'])} & " + " & ".join(f"{100 * r[k]:.0f}" for k in kinds) + " \\\\")
    head = "Format & $n$ & " + " & ".join(k.replace(" ", "\\ ") for k in kinds) + " \\\\"
    return tabular("lr" + "r" * len(kinds), head, "\n".join(lines))


# ---------- macros ----------

def integrity():
    t = (C.OUT / "integrity.txt").read_text()
    cap = re.search(r"(\d+) of (\d+) generations reached the cap", t)
    late = re.search(r">= (\d+) of (\d+) steps in (\d+)/(\d+) runs", t)
    cue = re.search(r"\((\d+) in test, at most ([\d.]+) test EM\)", t)
    rerun = re.findall(r"(\w+): rerun ([\d.]+), seed SD ([\d.]+)", t)
    gguf = re.search(r"lfm2 (\d+)% \(gguf", t)
    return {
        "capHits": cap.group(1), "nGenerations": pretty(int(cap.group(2))),
        "lateRuns": late.group(3), "maxSteps": pretty(int(late.group(2))),
        "cueRows": cue.group(1), "cueBound": pct(float(cue.group(2))),
        **{f"{P[m]}rerun": pct(float(r)) for m, r, _ in rerun},
        **{f"{P[m]}rerunSeedSd": pct(float(s)) for m, _, s in rerun},
        "lfmGgufAgree": gguf.group(1),
    }


def macros(fams):
    d = {"nSeeds": str(len(SEEDS)), "nSizes": str(len(SIZES)), "nHead": pretty(HEAD), "nSmall": pretty(min(SIZES)),
         "nArms": str(len(ARMS)), "nGrid": str(len(C.cells())), "nBoot": pretty(S.B),
         "negFrac": f"{100 * M['negatives']:.0f}", "corFrac": f"{100 * M['corrections']:.0f}"}
    first = C.metrics(MODELS[0], "words", HEAD, 0)
    d["nTest"] = pretty(first["test"]["n"])
    d["nDev"] = pretty(first["dev"]["n"])
    d["nNeg"] = pretty(first["negatives"]["n"])
    d["nCor"] = pretty(first["corrections"]["n"])
    d["samplesSeen"] = pretty(first["samples_seen_max"])
    d["poolCoverage"] = pct(first["pool_coverage"])
    d.update(integrity())

    for m in MODELS:
        p = P[m]
        for a in ARMS:
            k = p + MACRO[a]
            for name, key in (("Em", "correct"), ("S", "silent"), ("Dp", "d_parse"), ("Dr", "d_refuse"),
                              ("Neg", "neg_handled"), ("Fix", "fix_em"), ("Applied", "fix_applied")):
                d[k + name] = pct(mean(m, a, key))
            d[k + "EmSd"] = pct(sd(m, a, "correct"))
            d[k + "Tok"] = f"{mean(m, a, 'tokens'):.1f}"
            d[k + "Hall"] = f"{mean(m, a, 'hallucinated'):.3f}"
            d[k + "Lat"] = f"{lat_ms(m, a):.0f}"
            d[k + "EmSmall"] = pct(mean(m, a, "correct", min(SIZES)))
        r = S.ratio(D.get(m, "exr", HEAD, "silent"), D.get(m, "words", HEAD, "silent"))
        d[p + "silentRatio"], d[p + "silentRatioLo"], d[p + "silentRatioHi"] = (f"{r[k]:.1f}" for k in ("est", "lo", "hi"))
        r = S.ratio(D.get(m, "exr", HEAD, "hallucinated"), D.get(m, "words", HEAD, "hallucinated"))
        d[p + "hallRatio"], d[p + "hallRatioLo"], d[p + "hallRatioHi"] = (f"{r[k]:.1f}" for k in ("est", "lo", "hi"))
        d[p + "tokRatio"] = f"{mean(m, 'exr', 'tokens') / mean(m, 'words', 'tokens'):.1f}"
        d[p + "speedup"] = f"{lat_ms(m, 'exr') / lat_ms(m, 'words'):.1f}"
        xs = [x["mean_generated_tokens"] for v in LAT[m].values() for x in v]
        ys = [x["median_ms"] for v in LAT[m].values() for x in v]
        slope, icpt = statistics.linear_regression(xs, ys)
        d[p + "msPerTok"] = f"{slope:.1f}"
        d[p + "msFixed"] = f"{icpt:.0f}"
        d[p + "latRsq"] = f"{statistics.correlation(xs, ys) ** 2:.2f}"
        mem = [x["gpu_memory_used_by_this_process_mib"] for v in LAT[m].values() for x in v]
        d[p + "memMin"], d[p + "memMax"] = pretty(min(mem)), pretty(max(mem))
        d[p + "tpsMin"] = f"{min(statistics.fmean(x['tokens_per_second'] for x in v) for v in LAT[m].values()):.0f}"
        d[p + "tpsMax"] = f"{max(statistics.fmean(x['tokens_per_second'] for x in v) for v in LAT[m].values()):.0f}"
        for a, b, metric, _ in PAIRS:
            res = fams[(m, a, b, metric)]
            k = p + MACRO[a] + MACRO[b][0].upper() + MACRO[b][1:] + ("S" if metric == "silent" else "")
            h = res[-1]
            d[k + "Gap"], d[k + "Lo"], d[k + "Hi"] = spct(h["est"]), spct(h["lo"]), spct(h["hi"])
            d[k + "GapSmall"] = spct(res[0]["est"])
            d[k + "Ahead"] = str(sum(r["est"] > 0 for r in res))
            d[k + "Sep"] = str(sum(r["sep"] and r["est"] > 0 for r in res))
        refuse = [mean(m, a, "d_refuse") for a in ARMS]
        d[p + "refuseMin"], d[p + "refuseMax"] = pct(min(refuse)), pct(max(refuse))
        fix = [mean(m, a, "fix_em") for a in ARMS]
        d[p + "fixMin"], d[p + "fixMax"] = pct(min(fix)), pct(max(fix))
        app = [mean(m, a, "fix_applied") for a in ARMS]
        d[p + "appliedMin"], d[p + "appliedMax"] = pct(min(app), 0), pct(max(app), 0)
        neg = [mean(m, a, "neg_handled") for a in ARMS]
        d[p + "negMin"], d[p + "negMax"] = pct(min(neg)), pct(max(neg))
        unk = [statistics.fmean(C.metrics(m, a, HEAD, s)["negatives"]["handled_unknown"] for s in SEEDS) for a in ARMS]
        d[p + "unkMin"], d[p + "unkMax"] = pct(min(unk), 0), pct(max(unk), 0)
        probes = [40 * mean(m, a, "probe_em") for a in ARMS]
        d[p + "probeMin"], d[p + "probeMax"] = f"{min(probes):.0f}", f"{max(probes):.0f}"
        d[p + "bestEm"] = C.LABEL[ranking(m, "correct", max)[0]]
        d[p + "bestS"] = C.LABEL[ranking(m, "silent", min)[0]]
        d[p + "tiedEm"] = ", ".join(f"\\fmt{{{C.LABEL[a]}}}" for a in sorted(ranking(m, "correct", max)[1], key=ARMS.index)) or "none"
        d[p + "tiedS"] = ", ".join(f"\\fmt{{{C.LABEL[a]}}}" for a in sorted(ranking(m, "silent", min)[1], key=ARMS.index)) or "none"
        d[p + "worstArm"] = C.LABEL[min(ARMS, key=lambda a: mean(m, a, "correct"))]

        ref = {r["arm"]: r for r in csv_rows("refused.csv") if r["model"] == m}
        d[p + "wordsPolite"] = pct(float(ref["words"]["refuse_polite"]))
        d[p + "wordsNotPolite"] = pct(float(ref["words"]["refuse_other"]))
        oov = [float(r["oov_refused"]) for r in ref.values()]
        d[p + "oovRefMin"], d[p + "oovRefMax"] = pct(min(oov), 0), pct(max(oov), 0)
        d[p + "oovAns"] = pct(statistics.fmean(float(r["oov_answered"]) for r in ref.values()), 0)
        gap = [float(r["dev_training_cues"]) - float(r["test_heldout_cues"]) for r in csv_rows("cue_gap.csv") if r["model"] == m]
        d[p + "cueGapMin"], d[p + "cueGapMax"] = pct(min(gap), 0), pct(max(gap), 0)

        kinds = {r["arm"]: r for r in E.load() if r["model"] == m}
        d[p + "wordsNegErr"] = f"{100 * kinds['words']['negation']:.0f}"
        others = [kinds[a]["negation"] for a in ARMS if a != "words"]
        d[p + "otherNegErrMin"], d[p + "otherNegErrMax"] = f"{100 * min(others):.0f}", f"{100 * max(others):.0f}"
        d[p + "wordsLineErr"] = f"{100 * kinds['words']['line count']:.0f}"
        d[p + "exrNegRate"] = pct(kinds["exr"]["negation"] * mean(m, "exr", "silent"))
        d[p + "wordsNegRate"] = pct(kinds["words"]["negation"] * mean(m, "words", "silent"))

        lrs = C.lrs(m)
        d[p + "lrs"] = ", ".join(f"{C.LABEL[a]} {lrs[a]:.0e}".replace("e-0", "e-") for a in ARMS)

    slopes = {m: [S.slope([D.get(m, a, s, "correct") for a in C.LADDER], [D.get(m, a, s, "tokens") for a in C.LADDER])
                  for s in SIZES] for m in MODELS}
    for m, res in slopes.items():
        for r, p in zip(res, S.holm([r["p"] for r in res])):
            r["sep"] = p < 0.05
    allslopes = [r for res in slopes.values() for r in res]
    d["slopePos"] = str(sum(r["est"] > 0 for r in allslopes))
    d["slopeSep"] = str(sum(r["sep"] for r in allslopes))
    d["slopeCells"] = str(len(allslopes))
    d["slopeMax"] = pct(max(r["est"] for r in allslopes))

    fixed = sum(first_probe_fixed(m, a) for m in MODELS for a in ARMS)
    d["probeFixed"], d["probeRuns"] = str(fixed), str(len(MODELS) * len(ARMS) * len(SEEDS))
    d["nPolite"] = str(sum(1 for r in C.predictions("qwen", "words", HEAD, 0)
                           if re.search(r"\b(hi|hello|hey|please|thanks|thank you|good (morning|afternoon|evening))\b", r["src"])))
    d["pairCells"] = str(len(MODELS) * len(SIZES))
    from bench import config
    config.select("qwen", "pizza")
    t = config.task()
    name, em = t.published[0]
    d["publishedSystem"], d["publishedEm"] = name, pct(em)
    d["publishedTrainN"] = pretty(t.published_train_n)
    d["publishedGap"] = str(t.published_train_n - HEAD)
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items())) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("main_table.tex", "matched_pairs.tex", "replication.tex", "replication_macros.tex",
                  "slopes.tex", "coffee.tex", "coffee_macros.tex"):
        (OUT / stale).unlink(missing_ok=True)
    t2_short, t2_ci, fams = t2()
    lr_table, lr_seeds = a_lr()
    files = {
        "t1_main.tex": t1(),
        "t2_pairs.tex": t2_short,
        "t3_guardrail.tex": t3(),
        "a_pairs_ci.tex": t2_ci,
        "a_sizes.tex": a_sizes(),
        "a_negatives.tex": a_kinds("negatives", "handled_", ["junk", "long", "non", "off", "mixed", "unknown", "probe"],
                                   ["junk", "long", "non-order", "off-menu", "mixed", "unknown word", "probe"]),
        "a_corrections.tex": a_kinds("corrections", "em_",
                                     ["swap", "retract", "only", "size", "number", "drink", "cancel", "restart", "decoy"],
                                     ["swap", "retract", "only", "size", "number", "drink", "cancel", "restart", "decoy"]),
        "a_lr.tex": lr_table,
        "a_errors.tex": a_errors(),
        "macros.tex": macros(fams) + f"\\newcommand{{\\nSweepSeeds}}{{{lr_seeds}}}\n",
        "colors.tex": latex_colors() + "\n",
    }
    for name, text in files.items():
        (OUT / name).write_text(text)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
