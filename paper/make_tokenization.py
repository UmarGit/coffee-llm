#!/usr/bin/env python3
"""The tokenizer-floor figure: what each format costs in real subwords.

    uv run --extra pizza python paper/make_tokenization.py

Every box is a token produced by the model's own vocabulary and merge table, read
out of tokenizer.json by paper/bpe.py. Nothing here is drawn by hand. The example
order is a real test utterance and the six strings come from the project's own
encoders, so the figure cannot drift from what was trained on.

The script refuses to draw anything until bpe.py has been checked against the
token counts the real tokenizer recorded during the runs.
"""
import json
import sys
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bench.tasks.pizza import formats
from bpe import BPE, tokenizer_file, verify
from palette import ARMS, COLOR, LABEL

OUT = REPO / "paper" / "figures"
COL_W = 3.35
# The order drawn. A real test utterance, chosen because it exercises the one
# field where the two matched arms differ: an EXTRA quantity on a topping.
EXAMPLE = "i want a pie with pepperoni and extra cheese"
# Fraction of the verification rows allowed to disagree. The three that do are
# all a single hallucinated misspelling ("peppersoni") where greedy merging
# takes a different path from the reference implementation; no canonical target
# is affected. Kept tight so a real regression still trips it.
TOLERANCE = 0.001

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "figure.dpi": 200,
})


def check(bpe):
    """Re-tokenise every saved prediction and compare against the recorded count."""
    total, bad = 0, 0
    for arm in ARMS:
        hits = sorted(glob(str(REPO / f"runs/qwen/pizza/{arm}-n2456443-s0-*")))
        if not hits:
            sys.exit(f"no headline run for {arm}; cannot verify the tokenizer")
        d = Path(hits[0])
        cap = json.loads((d / "config.json").read_text())["cap"]
        rows = []
        for line in (d / "test_predictions.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append((r["raw"], r["n_generated_tokens"]))
        n, miss = verify(bpe, rows, cap=cap)
        total += n
        bad += len(miss)
    rate = bad / total
    print(f"tokenizer check: {total - bad}/{total} exact ({100 * (1 - rate):.2f}%)")
    if rate > TOLERANCE:
        sys.exit(f"tokenizer disagrees on {bad}/{total} rows, over the {TOLERANCE:.1%} "
                 f"tolerance; the figure would not be a measurement")
    return total, bad


def encodings(bpe):
    """(arm, string, chars, tokens) for the example order, in ladder order."""
    hits = sorted(glob(str(REPO / "runs/qwen/pizza/words-n2456443-s0-*")))
    gold = None
    for line in (Path(hits[0]) / "test_predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["src"] == EXAMPLE:
            gold = r["gold"]
            break
    if gold is None:
        sys.exit(f"example utterance not in the test split: {EXAMPLE!r}")
    node = formats.parse_exr(gold)
    out = []
    for a in ARMS:
        s = formats.encode(node, a)
        out.append((a, s, len(s), bpe.readable(s)))
    return out


def _boxrow(ax, y, toks, x0, width, color, h=0.62):
    """One row of equal boxes spanning x0..x0+width: each box is one token."""
    w = width / len(toks)
    for i, _t in enumerate(toks):
        ax.add_patch(FancyBboxPatch(
            (x0 + i * w + w * 0.06, y - h / 2), w * 0.88, h,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0.5, edgecolor="white", facecolor=color))


def _charrow(ax, y, toks, x0, width, color, h=0.52):
    """One row where each box is as wide as the characters it holds.

    The equal-width row above answers "how many tokens"; this one answers "where
    do the cuts fall", so the string reads left to right as it is actually split.
    """
    total = sum(len(t) for t in toks)
    x = x0
    for t in toks:
        w = width * len(t) / total
        ax.add_patch(FancyBboxPatch(
            (x + w * 0.04, y - h / 2), w * 0.92, h,
            boxstyle="round,pad=0,rounding_size=0.015",
            linewidth=0.5, edgecolor="white", facecolor=color))
        ax.text(x + w / 2, y, t, ha="center", va="center",
                fontsize=5.4, family="monospace", color="white")
        x += w


def build(rows):
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(COL_W, 3.05), gridspec_kw={"height_ratios": [6, 2.2]})

    # --- overview: one block per token, every arm to the same scale ---------
    widest = max(len(t) for _, _, _, t in rows)
    for i, (arm, _s, nch, toks) in enumerate(rows):
        y = -i
        top.text(-0.02, y, LABEL[arm], ha="right", va="center",
                 fontsize=6.6, color=COLOR[arm], fontweight="bold")
        _boxrow(top, y, toks, 0.0, len(toks) / widest, COLOR[arm])
        top.text(len(toks) / widest + 0.015, y,
                 f"{len(toks)} tok   {nch} ch", ha="left", va="center",
                 fontsize=6, color="#55606d")
    top.set_xlim(-0.30, 1.42)
    top.set_ylim(-len(rows) + 0.4, 0.95)
    top.text(0, 0.72, "one token per box, same scale on every row",
             fontsize=6, style="italic", color="#8A94A0")
    top.axis("off")

    # --- zoom: the matched pair, tokens labelled ---------------------------
    pair = {a: (s, nch, t) for a, s, nch, t in rows if a in ("dsl", "words")}
    for i, arm in enumerate(("dsl", "words")):
        _s, nch, toks = pair[arm]
        y = -i
        bot.text(-0.02, y, LABEL[arm], ha="right", va="center",
                 fontsize=6.6, color=COLOR[arm], fontweight="bold")
        _charrow(bot, y, toks, 0.0, 1.0, COLOR[arm])
        bot.text(1.02, y, f"{len(toks)} tok   {nch} ch", ha="left", va="center",
                 fontsize=6, color="#55606d")
    bot.set_xlim(-0.30, 1.42)
    bot.set_ylim(-2.35, 0.95)
    bot.text(0, 0.62, "the matched pair, box width by characters held",
             fontsize=6, style="italic", color="#8A94A0")
    bot.text(0, -1.72,
             "longer string, fewer tokens: \"+cheese\" splits in three,",
             fontsize=6, color="#22303F")
    bot.text(0, -2.10, "\"extra cheese\" stays whole",
             fontsize=6, color="#22303F")
    bot.axis("off")

    fig.subplots_adjust(hspace=0.12, left=0.01, right=0.99, top=0.99, bottom=0.01)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tokenization.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / "tokenization.pdf")


def macros(rows):
    """So the prose quotes this example's real numbers rather than retyping them."""
    d = {}
    for arm, _s, nch, toks in rows:
        key = {"exr": "Exr", "json": "Json", "exr_short": "ExrShort",
               "tagged": "Tagged", "dsl": "Dsl", "words": "Words"}[arm]
        d[f"ex{key}Tok"] = str(len(toks))
        d[f"ex{key}Ch"] = str(nch)
    d["exUtterance"] = EXAMPLE
    out = REPO / "paper" / "generated" / "tokenization_macros.tex"
    out.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                             for k, v in sorted(d.items())) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    bpe = BPE(tokenizer_file())
    check(bpe)
    rows = encodings(bpe)
    for arm, s, nch, toks in rows:
        print(f"  {arm:10} {nch:3}ch {len(toks):3}tok  {s[:70]}")
    build(rows)
    macros(rows)
