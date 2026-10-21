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

from bench import evaluate, report
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
        if r["size"] != 20000:
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
    for size in (1000, 20000):
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
    det = json.loads((REPO / "runs" / "determinism.json").read_text())
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
    d["pcfgWords"] = f"{m(by['words'], 'em_pcfg_err'):.3f}"
    d["pcfgExr"] = f"{m(by['exr'], 'em_pcfg_err'):.3f}"
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items()))


if __name__ == "__main__":
    by, by_small = load(20000), load(1000)
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
