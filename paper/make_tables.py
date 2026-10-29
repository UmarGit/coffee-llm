#!/usr/bin/env python3
"""Emit every LaTeX table and macro in the paper from runs/. Nothing is hand typed.

    uv run --extra pizza python paper/make_tables.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bench import config, evaluate, report

SIZES = config.Config.for_active().sizes
SMALL, HEADLINE = SIZES[0], SIZES[-1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import LABEL, MACRO, latex_colors

OUT = REPO / "paper" / "generated"
ARMS = ["exr", "json", "exr_short", "tagged", "dsl", "words"]


def load(size):
    by = defaultdict(list)
    for r in report.primary_runs():
        if r["size"] == size:
            by[r["arm"]].append(r)
    return by


def m(rs, key, split="test"):
    return statistics.fmean(r[split][key] for r in rs)


def s(rs, key, split="test"):
    v = [r[split][key] for r in rs]
    return statistics.stdev(v) if len(v) > 1 else 0.0


HEAD_MAIN = (r"\begin{tabular}{lrrrrrrr}" "\n" r"\toprule" "\n"
             r"Format & Tokens & Exact match & Parse validity & "
             r"\textbf{Silent error} & Halluc./utt.\ & Latency (ms) & Mem.\ (MiB) \\" "\n"
             r"\midrule")
HEAD_PAIRS = (r"\begin{tabular}{llrrrl}" "\n" r"\toprule" "\n"
              r"Pair & Seed & Tokens & $\Delta$ EM & 95\% CI & Verdict \\" "\n"
              r"\midrule")
HEAD_SLOPE = (r"\begin{tabular}{lrr}" "\n" r"\toprule" "\n"
              r"Train size & Endpoint & Slope / 10 tokens \\" "\n" r"\midrule")
FOOT = "\n" r"\bottomrule" "\n" r"\end{tabular}"


def wrap(head, body):
    r"""Emit a complete tabular. \input inside an alignment breaks \noalign."""
    return head + "\n" + body + FOOT


def main_table(by):
    lat = {d["arm"]: d for d in report.latency_table()}
    lines = []
    for a in ARMS:
        rs = by[a]
        em, valid = m(rs, "em"), m(rs, "parse_validity")
        silent = valid - em
        L = lat.get(a, {})
        lines.append(
            f"{LABEL[a]} & {m(rs, 'mean_generated_tokens'):.1f} & "
            f"{em:.3f} {{\\scriptsize $\\pm$ {s(rs, 'em'):.3f}}} & "
            f"{valid:.3f} & {silent:.3f} & "
            f"{m(rs, 'hallucinated_entity'):.3f} & "
            f"{L.get('median_ms', float('nan')):.0f} & "
            f"{L.get('proc_mem_mib', float('nan')):.0f} \\\\")
    return "\n".join(lines)


def matched_pairs_table():
    rows = report.matched_pairs(report.primary_runs(), "test")
    out = []
    for r in rows:
        if r["size"] != HEADLINE:
            continue
        out.append(
            f"{LABEL[r['a']]}/{LABEL[r['b']]} & {r['seed']} & "
            f"{r['tokens_a']:.1f}/{r['tokens_b']:.1f} & "
            f"{r['mean_diff']:+.3f} & [{r['ci_lo']:+.3f},\\,{r['ci_hi']:+.3f}] & "
            f"{'noise' if r['within_noise'] else 'sep.'} \\\\")
    return "\n".join(out)


def slope_table():
    runs = report.primary_runs()
    out = []
    for size in (SMALL, HEADLINE):
        for metric, name in (("em", "exact match"),
                             ("parse_validity", "parse validity"),
                             ("hallucinated_entity", "hallucinated entities")):
            sl = report.slope_on_tokens(runs, size, "test", metric)
            if sl:
                out.append(f"{size:,} & {name} & {sl['slope_per_10_tokens']:+.4f} \\\\"
                           .replace(",", "{,}"))
    return "\n".join(out)


def macros(by, by_small):
    lat = {d["arm"]: d for d in report.latency_table()}
    # runs/<model>/<task>/, not the old flat runs/, so two models cannot share a
    # noise floor measured on only one of them.
    det = json.loads((config.model_task_dir() / "determinism.json").read_text())
    d = {}
    for a in ARMS:
        rs = by[a]
        d[f"{MACRO[a]}Em"] = f"{m(rs, 'em'):.3f}"
        d[f"{MACRO[a]}Tok"] = f"{m(rs, 'mean_generated_tokens'):.1f}"
        d[f"{MACRO[a]}Valid"] = f"{m(rs, 'parse_validity'):.3f}"
        d[f"{MACRO[a]}Silent"] = f"{m(rs, 'parse_validity') - m(rs, 'em'):.3f}"
        d[f"{MACRO[a]}Hall"] = f"{m(rs, 'hallucinated_entity'):.3f}"
        d[f"{MACRO[a]}Miss"] = f"{m(rs, 'missing_entity'):.3f}"
        d[f"{MACRO[a]}Lat"] = f"{lat[a]['median_ms']:.0f}" if a in lat else "n/a"
        d[f"{MACRO[a]}EmSmall"] = f"{m(by_small[a], 'em'):.3f}"
    # Sizes and slopes as macros too, so prose cannot keep quoting a grid that
    # has changed underneath it. This is exactly what went stale last time.
    def pretty(n):
        return f"{n:,}".replace(",", "{,}")

    d["nSmall"] = pretty(SMALL)
    d["nHeadline"] = pretty(HEADLINE)
    d["nLrSelect"] = pretty(config.Config.for_active().lr_select_size)
    d["nSizes"] = str(len(SIZES))
    runs = report.primary_runs()
    for key, size in (("Small", SMALL), ("Head", HEADLINE)):
        sl = report.slope_on_tokens(runs, size, "test", "em")
        if sl:
            # Reported as points of exact match per ten tokens, the way the prose
            # talks about it, and signed so "costs" reads correctly.
            d[f"slopeEm{key}"] = f"{abs(sl['slope_per_10_tokens']) * 100:.1f}"
    d["noiseFloor"] = f"{det['test_abs_diff']:.3f}"
    d["nTest"] = str(by["exr"][0]["test"]["n"])
    d["nPcfg"] = str(by["exr"][0]["test"]["n_pcfg_err"])
    d["nRuns"] = str(len(report.primary_runs()))
    d["speedup"] = f"{lat['exr']['median_ms'] / lat['words']['median_ms']:.1f}"
    d["tokRatio"] = f"{m(by['exr'], 'mean_generated_tokens') / m(by['words'], 'mean_generated_tokens'):.1f}"
    d["hallRatio"] = f"{m(by['exr'], 'hallucinated_entity') / m(by['words'], 'hallucinated_entity'):.1f}"
    exr_silent = m(by['exr'], 'parse_validity') - m(by['exr'], 'em')
    words_silent = m(by['words'], 'parse_validity') - m(by['words'], 'em')
    d["silentRatio"] = f"{exr_silent / words_silent:.1f}"
    mems = [d2["proc_mem_mib"] for d2 in report.latency_table()
            if d2.get("proc_mem_mib")]
    if mems:
        d["memMin"] = f"{min(mems):.0f}"
        d["memMax"] = f"{max(mems):.0f}"
    # The published reference points, from the task registry rather than typed,
    # together with how far our pool is from the published training set. The gap
    # is the three rows we drop for duplicating an evaluation utterance, so the
    # prose can state the comparison exactly instead of rounding it.
    task = config.task()
    if task.published:
        name, em = task.published[0]
        d["publishedSystem"] = name
        d["publishedEm"] = f"{em:.3f}"
        d["publishedTrainN"] = f"{task.published_train_n:,}".replace(",", "{,}")
        d["publishedGap"] = str(task.published_train_n - HEADLINE)
        d["headlineBeatsPublished"] = f"{max(m(by[a], 'em') for a in ARMS) - em:.3f}"
    d["pcfgWords"] = f"{m(by['words'], 'em_pcfg_err'):.3f}"
    d["pcfgExr"] = f"{m(by['exr'], 'em_pcfg_err'):.3f}"

    # The worst ladder arm, generated rather than typed. The prose used to name
    # exr_short here and exr_short is not the worst; the two are 0.013 apart and
    # the ordering is not stable enough to hand type.
    ladder = config.task().ladder_arms
    worst = min(ladder, key=lambda a: m(by[a], "em"))
    d["worstArm"] = LABEL[worst]
    d["worstArmEm"] = f"{m(by[worst], 'em'):.3f}"
    # What the most aggressive structural compression costs against the format it
    # compresses, which is the claim the paragraph is actually making.
    d["exrShortDrop"] = f"{m(by['exr'], 'em') - m(by['exr_short'], 'em'):.3f}"

    # Matched-pair separation counts over the WHOLE grid, not just the headline
    # size the table shows. The prose claimed twelve comparisons all separating;
    # the real picture is that the symbolic-versus-words pair separates almost
    # everywhere and the s-expression-versus-JSON pair is near chance, which is a
    # different and better argument. Counted here so it cannot be mistyped.
    pairs = report.matched_pairs(report.primary_runs(), "test")
    for a, b in config.task().matched_pairs:
        rows = [r for r in pairs if (r["a"], r["b"]) == (a, b)]
        key = f"{MACRO[a]}{MACRO[b][0].upper()}{MACRO[b][1:]}"
        d[f"pair{key}Total"] = str(len(rows))
        d[f"pair{key}Sep"] = str(sum(1 for r in rows if not r["within_noise"]))
        # How many favour each side, so a sign-consistency claim is checkable
        # without the prose doing arithmetic on the totals.
        d[f"pair{key}FavB"] = str(sum(1 for r in rows if r["mean_diff"] < 0))
        d[f"pair{key}FavA"] = str(sum(1 for r in rows if r["mean_diff"] > 0))
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items()))


# ---------------------------------------------------------------------------
# Replication across models.
#
# Absolute scores are not comparable between models: they differ in capacity,
# architecture, adapter targets and prompt conditioning. What is comparable is
# the ordering of arms and the matched-pair difference, because those are what
# the ladder design isolates. So this emits exactly those, and nothing that
# would invite a model-versus-model reading.
# ---------------------------------------------------------------------------

REPLICATION_MODELS = ("qwen", "lfm2")


def _model_summary(model_name):
    """Per-arm tokens and exact match at the headline size, plus the noise floor."""
    config.select(model_name, "pizza")
    by = load(HEADLINE)
    if not by or not all(by.get(a) for a in ARMS):
        return None
    det = json.loads((config.model_task_dir() / "determinism.json").read_text())
    return {
        "label": config.model().label,
        "vocab": config.model().vocab_size,
        "noise": det["test_abs_diff"],
        "arms": {a: {"tok": m(by[a], "mean_generated_tokens"),
                     "em": m(by[a], "em")} for a in ARMS},
    }


def replication_table(summaries):
    """Tokens and exact match per arm, one block per model."""
    out = ["\\begin{tabular}{l" + "rr" * len(summaries) + "}", "\\toprule"]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{s['label']}}}" for s in summaries)
    out.append(f" & {head} \\\\")
    out.append(" & " + " & ".join(["tok.", "EM"] * len(summaries)) + " \\\\")
    out.append("\\midrule")
    for a in ARMS:
        cells = []
        for s in summaries:
            v = s["arms"][a]
            best = max(s["arms"].values(), key=lambda x: x["em"])["em"]
            em = f"\\textbf{{{v['em']:.3f}}}" if v["em"] == best else f"{v['em']:.3f}"
            cells += [f"{v['tok']:.1f}", em]
        out.append(f"\\fmt{{{LABEL[a]}}} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def replication_macros(summaries):
    d = {}
    for s, prefix in zip(summaries, ("primary", "repl")):
        arms = s["arms"]
        gap = arms["dsl"]["em"] - arms["words"]["em"]
        d[f"{prefix}Label"] = s["label"]
        d[f"{prefix}Vocab"] = f"{s['vocab']:,}".replace(",", "{,}")
        d[f"{prefix}Noise"] = f"{s['noise']:.3f}"
        d[f"{prefix}WordsEm"] = f"{arms['words']['em']:.3f}"
        d[f"{prefix}DslEm"] = f"{arms['dsl']['em']:.3f}"
        d[f"{prefix}ExrEm"] = f"{arms['exr']['em']:.3f}"
        # Signed so a positive number always means words ahead of dsl.
        d[f"{prefix}PairGap"] = f"{-gap:.3f}"
        d[f"{prefix}WordsTok"] = f"{arms['words']['tok']:.1f}"
        d[f"{prefix}DslTok"] = f"{arms['dsl']['tok']:.1f}"
        d[f"{prefix}ExrTok"] = f"{arms['exr']['tok']:.1f}"
        d[f"{prefix}BestArm"] = LABEL[max(arms, key=lambda a: arms[a]["em"])]
        # Does words cost more tokens than dsl under this tokenizer? Under LFM2 it
        # does, which is the strongest form of the argument: more tokens, better.
        d[f"{prefix}WordsLonger"] = ("yes" if arms["words"]["tok"] > arms["dsl"]["tok"]
                                     else "no")
        for a in ARMS:
            d[f"{prefix}{MACRO[a]}Tokens"] = f"{arms[a]['tok']:.1f}"

    # How much more the smaller vocabulary costs each arm. This is the measured
    # form of the tokenizer-floor claim: it should hit the punctuation-heavy
    # formats far harder than the word-based ones.
    pri, rep = summaries[0]["arms"], summaries[1]["arms"]
    for a in ARMS:
        d[f"infl{MACRO[a]}"] = f"{rep[a]['tok'] / pri[a]['tok']:.2f}"
    d["inflMax"] = f"{max(rep[a]['tok'] / pri[a]['tok'] for a in ARMS):.2f}"
    d["inflMin"] = f"{min(rep[a]['tok'] / pri[a]['tok'] for a in ARMS):.2f}"

    # How much the format choice moves accuracy, against how much the model choice
    # does. Generated rather than typed because it is easy to get the direction of
    # this comparison wrong by hand.
    for prefix, arms in (("primary", pri), ("repl", rep)):
        ems = [v["em"] for v in arms.values()]
        d[f"{prefix}ArmSpread"] = f"{max(ems) - min(ems):.3f}"
    d["modelGapBest"] = f"{abs(max(v['em'] for v in pri.values()) - max(v['em'] for v in rep.values())):.3f}"
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items()))


# ---------------------------------------------------------------------------
# The cafe task, written only once its grid exists.
#
# Two arms, so no ladder and no slope. It was added as one more matched pair, in
# Russian and English rather than English alone, on the system actually in
# production. What it in fact produced is a saturated grid: both arms sit within
# one held-out utterance of each other at every pool size. So this also emits the
# facts that establish the saturation, because the honest reading of the cafe
# numbers is a resolution limit and the prose has to be able to say so from
# measured values rather than from a hopeful adjective.
#
# Nothing is emitted while the grid is incomplete, so the paper never contains a
# placeholder that could be mistaken for a measurement.
# ---------------------------------------------------------------------------

def coffee_section():
    config.select("qwen", "coffee")
    task = config.task()
    sizes = config.Config.for_active().sizes
    runs = report.primary_runs(cfg_sizes=sizes)
    head = max(sizes)
    by = {}
    for r in runs:
        if r["size"] == head:
            by.setdefault(r["arm"], []).append(r)
    if not all(by.get(a) for a in task.arms):
        return None

    rows = []
    for a in task.arms:
        rs = by[a]
        rows.append((a, m(rs, "mean_generated_tokens"), m(rs, "em"),
                     m(rs, "parse_validity")))
    # No arm is bolded. Nothing here separates by more than a third of one
    # held-out utterance, and bolding a winner would read as a result.
    table = ["\\begin{tabular}{lrrr}", "\\toprule",
             "arm & tokens & EM & validity \\\\", "\\midrule"]
    for a, tok, em, val in rows:
        table.append(f"\\fmt{{{a}}} & {tok:.1f} & {em:.3f} & {val:.3f} \\\\")
    table += ["\\bottomrule", "\\end{tabular}"]

    d = {"coffeeN": str(by[task.arms[0]][0]["test"]["n"]),
         "coffeeHead": f"{head:,}".replace(",", "{,}")}
    for a, tok, em, val in rows:
        key = "coffeeCompact" if a == "compact" else "coffeeWords"
        d[key + "Em"] = f"{em:.3f}"
        d[key + "Tok"] = f"{tok:.1f}"
    gap = float(d["coffeeWordsEm"]) - float(d["coffeeCompactEm"])
    d["coffeePairGap"] = f"{gap:+.3f}"

    # One held-out utterance is worth this much exact match. With 49 cases it is
    # 2.0 points, which is larger than any difference the grid shows.
    n = int(d["coffeeN"])
    d["coffeeResolution"] = f"{100 / n:.1f}"

    # Every cell, not just the headline, and how many matched pairs separate.
    cells = [r["test"]["em"] for r in runs]
    d["coffeeEmMin"] = f"{min(cells):.3f}"
    d["coffeeEmMax"] = f"{max(cells):.3f}"
    pairs = report.matched_pairs(runs, "test")
    d["coffeePairsTotal"] = str(len(pairs))
    d["coffeePairsSep"] = str(sum(1 for r in pairs if not r["within_noise"]))
    d["coffeeNoise"] = f"{json.loads((config.model_task_dir() / 'determinism.json').read_text())['test_abs_diff']:.3f}"

    # The production numbers the task does support, which is why it is in the
    # paper at all now that it resolves nothing about format.
    lat = {x["arm"]: x for x in report.latency_table()}
    for a in task.arms:
        key = "coffeeCompact" if a == "compact" else "coffeeWords"
        if a in lat:
            d[key + "Lat"] = f"{lat[a]['median_ms']:.0f}"

    macros = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items()))
    return "\n".join(table), macros


if __name__ == "__main__":
    by, by_small = load(HEADLINE), load(SMALL)
    if not by:
        sys.exit("no completed runs")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main_table.tex").write_text(wrap(HEAD_MAIN, main_table(by)) + "\n")
    (OUT / "matched_pairs.tex").write_text(wrap(HEAD_PAIRS, matched_pairs_table()) + "\n")
    (OUT / "slopes.tex").write_text(wrap(HEAD_SLOPE, slope_table()) + "\n")
    (OUT / "macros.tex").write_text(macros(by, by_small) + "\n")
    (OUT / "colors.tex").write_text(latex_colors() + "\n")
    for f in sorted(OUT.iterdir()):
        print("wrote", f)

    # Replication last, because it re-selects the model and leaves that selection
    # behind; everything above is the primary model's.
    summaries = [x for x in (_model_summary(n) for n in REPLICATION_MODELS) if x]
    if len(summaries) > 1:
        (OUT / "replication.tex").write_text(replication_table(summaries) + "\n")
        (OUT / "replication_macros.tex").write_text(replication_macros(summaries) + "\n")
        print(f"wrote {OUT / 'replication.tex'} ({len(summaries)} models)")
    else:
        print("only one model has a complete grid; skipping the replication table")

    coffee = coffee_section()
    if coffee:
        (OUT / "coffee.tex").write_text(coffee[0] + "\n")
        (OUT / "coffee_macros.tex").write_text(coffee[1] + "\n")
        print(f"wrote {OUT / 'coffee.tex'}")
    else:
        print("cafe grid incomplete; its section stays out of the paper")
