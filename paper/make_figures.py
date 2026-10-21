#!/usr/bin/env python3
"""Build every figure in the paper from runs/. No hardcoded numbers.

    uv run --extra pizza python paper/make_figures.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pizza_bench import report

OUT = REPO / "paper" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import ARMS, COLOR, LABEL, SEMANTIC

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.dpi": 200,
})

COL_W = 3.35   # IEEEtran single column, inches
FULL_W = 7.0


def load(size=20000):
    runs = [r for r in report.primary_runs() if r["size"] == size]
    by = defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    return by


def stat(rs, key, split="test"):
    return statistics.fmean(r[split][key] for r in rs)


def sd(rs, key, split="test"):
    v = [r[split][key] for r in rs]
    return statistics.stdev(v) if len(v) > 1 else 0.0


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_accuracy_vs_tokens(by):
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for a in ARMS:
        rs = by[a]
        x, y = stat(rs, "mean_generated_tokens"), stat(rs, "em")
        ax.errorbar(x, y, yerr=sd(rs, "em"), fmt="o", ms=6, color=COLOR[a],
                    capsize=2.5, elinewidth=0.8, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3)
        dy = 0.016 if a not in ("json", "dsl") else -0.030
        ax.annotate(LABEL[a], (x, y), textcoords="offset points",
                    xytext=(0, 9 if dy > 0 else -14), ha="center",
                    fontsize=7, color=COLOR[a])
    # the two matched-token pairs
    for pair in (("exr", "json"), ("dsl", "words")):
        xs = [stat(by[p], "mean_generated_tokens") for p in pair]
        ys = [stat(by[p], "em") for p in pair]
        ax.plot(xs, ys, color="#999", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("mean generated tokens per utterance")
    ax.set_ylabel("exact match (test)")
    ax.set_xlim(0, 52)
    ax.set_ylim(0.52, 0.90)
    save(fig, "accuracy_vs_tokens")


def fig_error_composition(by):
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    order = sorted(ARMS, key=lambda a: stat(by[a], "mean_generated_tokens"), reverse=True)
    ys = range(len(order))
    for i, a in enumerate(order):
        rs = by[a]
        em = stat(rs, "em")
        valid = stat(rs, "parse_validity")
        silent = valid - em
        detect = 1 - valid
        ax.barh(i, em, color=SEMANTIC["correct"], edgecolor="white", lw=0.6)
        ax.barh(i, silent, left=em, color="#c4513b", edgecolor="white", lw=0.6)
        ax.barh(i, detect, left=em + silent, color=SEMANTIC["detectable"], edgecolor="white", lw=0.6)
        ax.text(em + silent / 2, i, f"{silent:.2f}", ha="center", va="center",
                fontsize=6.2, color="white", fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{LABEL[a]}\n{stat(by[a], 'mean_generated_tokens'):.0f} tok"
                        for a in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of test utterances")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white") for c in
               (SEMANTIC["correct"], SEMANTIC["silent"], SEMANTIC["detectable"])]
    ax.legend(handles, ["correct", "silent error", "detectable (unparseable)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.1, columnspacing=1.0)
    save(fig, "error_composition")


def fig_latency(by):
    lat = report.latency_table()
    if not lat:
        print("no latency results, skipping")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    lat = sorted(lat, key=lambda d: d["generated_tokens"])
    for d in lat:
        a = d["arm"]
        ax.scatter(d["generated_tokens"], d["median_ms"], s=42, color=COLOR[a],
                   edgecolor="white", lw=0.6, zorder=3)
        ax.annotate(LABEL[a], (d["generated_tokens"], d["median_ms"]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7, color=COLOR[a])
    xs = [d["generated_tokens"] for d in lat]
    ys = [d["median_ms"] for d in lat]
    k = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    span = [0, max(xs) * 1.12]
    ax.plot(span, [k * x for x in span], color="#999", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("mean generated tokens per utterance")
    ax.set_ylabel("median latency, batch 1 (ms)")
    ax.set_xlim(0, max(xs) * 1.12)
    ax.set_ylim(0, max(ys) * 1.2)
    save(fig, "latency_vs_tokens")


def fig_data_efficiency():
    small, big = load(1000), load(20000)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for a in ARMS:
        xs = [1000, 20000]
        ys = [stat(small[a], "em"), stat(big[a], "em")]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.2, color=COLOR[a], label=LABEL[a])
        ax.fill_between(xs,
                        [ys[0] - sd(small[a], "em"), ys[1] - sd(big[a], "em")],
                        [ys[0] + sd(small[a], "em"), ys[1] + sd(big[a], "em")],
                        color=COLOR[a], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([1000, 20000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("training examples")
    ax.set_ylabel("exact match (test)")
    ax.legend(ncol=3, loc="lower right", handlelength=1.3, columnspacing=0.9)
    save(fig, "data_efficiency")


if __name__ == "__main__":
    by = load(20000)
    missing = [a for a in ARMS if not by.get(a)]
    if missing:
        sys.exit(f"no completed runs for {missing}; run the grid first")
    fig_accuracy_vs_tokens(by)
    fig_error_composition(by)
    fig_latency(by)
    fig_data_efficiency()
    print("done")
