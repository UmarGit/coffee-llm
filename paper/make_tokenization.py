#!/usr/bin/env python3
"""Characters against tokens: the tokenization figure and the test-set measurement.

    uv run --extra pizza python paper/make_tokenization.py

Every box is a token from the model's own vocabulary and merge table (paper/bpe.py).
Before anything is drawn, bpe.py is checked against the token counts each model
recorded for its own outputs during the campaign. The test-set table encodes every
gold tree in all six formats and counts characters and tokens under both tokenizers.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "analysis"))
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import common as C
import matplotlib.pyplot as plt
from bpe import BPE, tokenizer_file, verify
from matplotlib.patches import FancyBboxPatch
from palette import ARMS, COLOR, LABEL, MACRO

from bench.tasks.pizza import formats

M = C.manifest()
HEAD = max(M["sizes"])
OUT = HERE / "figures"
GEN = HERE / "generated"
COL_W = 3.03
REPOS = {"qwen": "Qwen/Qwen3.5-0.8B", "lfm2": "LiquidAI/LFM2.5-350M"}
EXAMPLE = "i want a pie with pepperoni and extra cheese"
TOLERANCE = 0.002

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "figure.dpi": 200,
})


def check(model, bpe):
    total, bad = 0, 0
    for arm in ARMS:
        d = C.run_dir(model, arm, HEAD, 0)
        cap = json.loads((d / "config.json").read_text())["cap"]
        rows = [(r["raw"], r["n_generated_tokens"]) for r in C.predictions(model, arm, HEAD, 0)]
        n, miss = verify(bpe, rows, cap=cap)
        total += n
        bad += len(miss)
    print(f"{model} tokenizer check: {total - bad}/{total} exact")
    if bad / total > TOLERANCE:
        sys.exit(f"{model}: tokenizer disagrees on {bad}/{total} rows; the figure would not be a measurement")


def test_trees():
    return [formats.parse_exr(r["gold"]) for r in C.predictions("qwen", "words", HEAD, 0)]


def measure(bpes):
    trees = test_trees()
    out = {}
    for a in ARMS:
        strings = [formats.encode(t, a) for t in trees]
        out[a] = {"chars": statistics.fmean(len(s) for s in strings)}
        for m, b in bpes.items():
            out[a][m] = statistics.fmean(len(b.tokens(s)) for s in strings)
    return out, len(trees)


def example(bpe):
    gold = next(r["gold"] for r in C.predictions("qwen", "words", HEAD, 0) if r["src"] == EXAMPLE)
    node = formats.parse_exr(gold)
    return [(a, formats.encode(node, a), bpe.readable(formats.encode(node, a))) for a in ARMS]


def _boxes(ax, y, widths, x0, h, color, labels=None, fs=5.0):
    x = x0
    for i, w in enumerate(widths):
        ax.add_patch(FancyBboxPatch((x + w * 0.05, y - h / 2), w * 0.9, h, boxstyle="round,pad=0,rounding_size=0.008",
                                    lw=0, facecolor=color))
        if labels:
            ax.text(x + w / 2, y, labels[i], ha="center", va="center", fontsize=fs, family="monospace", color="white")
        x += w


def build(rows):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(COL_W, 3.1), gridspec_kw={"height_ratios": [6, 1.9]})
    max_tok = max(len(t) for _, _, t in rows)
    max_ch = max(len(s) for _, s, _ in rows)
    for i, (a, s, toks) in enumerate(rows):
        y = -i * 1.0
        top.text(-0.03, y - 0.17, LABEL[a], ha="right", va="center", fontsize=6.6, color=COLOR[a], fontweight="bold")
        _boxes(top, y, [1.0 / max_tok] * len(toks), 0.0, 0.3, COLOR[a])
        top.text(len(toks) / max_tok + 0.015, y, f"{len(toks)} tokens", ha="left", va="center", fontsize=5.8, color="#444")
        _boxes(top, y - 0.36, [len(t) / max_ch for t in toks], 0.0, 0.24, COLOR[a])
        top.text(len(s) / max_ch + 0.015, y - 0.36, f"{len(s)} characters", ha="left", va="center", fontsize=5.8,
                 color="#444")
    top.text(0, 0.45, "upper bar: one box per token; lower bar: the same tokens, each as wide as its characters",
             fontsize=5.3, style="italic", color="#7A7A7A")
    top.set_xlim(-0.26, 1.3)
    top.set_ylim(-len(rows) + 0.4, 0.62)
    top.axis("off")

    pair = {a: (s, t) for a, s, t in rows if a in ("dsl", "words")}
    widest = max(len(s) for s, _ in pair.values())
    for i, a in enumerate(("dsl", "words")):
        s, toks = pair[a]
        y = -i * 0.8
        bot.text(-0.03, y, LABEL[a], ha="right", va="center", fontsize=6.6, color=COLOR[a], fontweight="bold")
        _boxes(bot, y, [len(t) / widest for t in toks], 0.0, 0.5, COLOR[a], labels=[t.replace("·", "") for t in toks],
               fs=5.4)
        bot.text(len(s) / widest + 0.015, y, f"{len(toks)} tok, {len(s)} ch", ha="left", va="center", fontsize=5.8,
                 color="#444")
    bot.text(0, 0.5, "the matched pair, box width by characters", fontsize=5.3, style="italic", color="#7A7A7A")
    bot.set_xlim(-0.26, 1.3)
    bot.set_ylim(-1.15, 0.7)
    bot.axis("off")
    fig.subplots_adjust(hspace=0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tokenization.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / "tokenization.pdf")


def tex(s):
    out = []
    for ch in s:
        out.append({"{": "\\{", "}": "\\}", "_": "\\_", "#": "\\#", "&": "\\&", "%": "\\%", "$": "\\$",
                    "~": "\\textasciitilde{}", "^": "\\^{}", "\\": "\\textbackslash{}",
                    '"': "\\textquotedbl{}"}.get(ch, ch))
    return "".join(out)


def write(rows, table, n):
    d = {"exUtterance": EXAMPLE, "nTokTest": f"{n:,}".replace(",", "{,}")}
    for a, s, toks in rows:
        k = MACRO[a][0].upper() + MACRO[a][1:]
        d[f"ex{k}Tok"], d[f"ex{k}Ch"] = str(len(toks)), str(len(s))
    for a, v in table.items():
        k = MACRO[a][0].upper() + MACRO[a][1:]
        d[f"ch{k}"] = f"{v['chars']:.1f}"
        d[f"tq{k}"] = f"{v['qwen']:.1f}"
        d[f"tl{k}"] = f"{v['lfm2']:.1f}"
        d[f"cpt{k}"] = f"{v['chars'] / v['qwen']:.1f}"
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tokenization_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(d.items())) + "\n")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Format & Chars & \\multicolumn{2}{c}{Qwen3.5} & \\multicolumn{2}{c}{LFM2.5} \\\\",
             " & & tok. & ch./tok. & tok. & ch./tok. \\\\", "\\midrule"]
    for a, v in table.items():
        lines.append(f"\\fmt{{{LABEL[a]}}} & {v['chars']:.1f} & {v['qwen']:.1f} & {v['chars'] / v['qwen']:.1f} "
                     f"& {v['lfm2']:.1f} & {v['chars'] / v['lfm2']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "a_chars.tex").write_text("\n".join(lines) + "\n")
    ex = ["\\begin{tabular}{@{}lrl@{}}", "\\toprule", "Format & Tok. & Target \\\\", "\\midrule"]
    for a, s, toks in rows:
        shown = s if len(s) <= 38 else s[:35] + "..."
        ex.append(f"\\fmt{{{LABEL[a]}}} & {len(toks)} & \\texttt{{{tex(shown)}}} \\\\")
    ex += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "formats_example.tex").write_text("\n".join(ex) + "\n")
    print("wrote", GEN / "tokenization_macros.tex", GEN / "a_chars.tex")


if __name__ == "__main__":
    bpes = {m: BPE(tokenizer_file(r)) for m, r in REPOS.items()}
    for m, b in bpes.items():
        check(m, b)
    rows = example(bpes["qwen"])
    for a, s, toks in rows:
        print(f"  {a:10} {len(s):3}ch {len(toks):3}tok  {s[:70]}")
    table, n = measure(bpes)
    for a, v in table.items():
        print(f"  {a:10} chars {v['chars']:.1f}  qwen {v['qwen']:.1f}  lfm2 {v['lfm2']:.1f}")
    build(rows)
    write(rows, table, n)
