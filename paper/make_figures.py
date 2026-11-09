#!/usr/bin/env python3
"""Every figure in the paper, from the neg-v1 campaign. No hardcoded numbers.

    uv run --extra pizza python paper/make_figures.py

Intervals are 95% hierarchical bootstrap intervals (tools/analysis/stats.py).
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
import errors as E
import matplotlib.pyplot as plt
import numpy as np
import stats as S
from palette import ARMS, COLOR, LABEL, LINE, MARKER, OUTCOME

import data as D

M = C.manifest()
MODELS, SIZES = M["models"], M["sizes"]
HEAD = max(SIZES)
OUT = HERE / "figures"
COL_W = 3.03
FULL_W = 6.3

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.dpi": 200, "hatch.linewidth": 0.5,
})


def tick(n):
    return f"{n / 1e6:.1f}M".replace(".0M", "M") if n >= 1e6 else f"{n // 1000}k"


def ci_mean(model, arm, size, name):
    x = D.get(model, arm, size, name)
    r = S.bootstrap(lambda m: m[0].mean(axis=1), [x])
    return r["est"], r["lo"], r["hi"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def title(ax, model):
    ax.set_title(C.LABEL[model], loc="left", fontsize=7.5, pad=3)


def fig_accuracy_vs_tokens():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
    for ax, m in zip(axes, MODELS):
        pts = {}
        for a in ARMS:
            x = D.get(m, a, HEAD, "tokens").mean()
            y, lo, hi = ci_mean(m, a, HEAD, "correct")
            pts[a] = (x, y)
            ax.errorbar(x, 100 * y, yerr=[[100 * (y - lo)], [100 * (hi - y)]], fmt=MARKER[a],
                        ms=6 if a == "words" else 4, color=COLOR[a], mec="white", mew=0.4,
                        capsize=1.6, elinewidth=0.7, zorder=3)
        for p, q in (("exr", "json"), ("dsl", "words")):
            ax.plot([pts[p][0], pts[q][0]], [100 * pts[p][1], 100 * pts[q][1]],
                    color="#888", lw=0.7, ls=(0, (3, 2)), zorder=1)
        title(ax, m)
        ax.set_xlim(0, 64)
        ax.set_xlabel("output tokens")
    axes[0].set_ylabel("exact match (%)")
    handles = [plt.Line2D([], [], marker=MARKER[a], color=COLOR[a], ls="", ms=5 if a == "words" else 4,
                          mec="white", mew=0.4) for a in ARMS]
    fig.legend(handles, [LABEL[a] for a in ARMS], loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.07), handletextpad=0.1, columnspacing=0.6)
    fig.subplots_adjust(wspace=0.1, bottom=0.27)
    save(fig, "accuracy_vs_tokens")


def fig_error_composition():
    keys = ("correct", "silent", "d_parse", "d_refuse")
    names = {"correct": "correct", "silent": "silent error",
             "d_parse": "unparseable", "d_refuse": "real order refused"}
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    order = list(reversed(ARMS))
    for ax, m in zip(axes, MODELS):
        for i, a in enumerate(order):
            left = 0.0
            for k in keys:
                v = D.get(m, a, HEAD, k).mean()
                fc, hatch = OUTCOME[k]
                ax.barh(i, 100 * v, left=100 * left, color=fc, hatch=hatch, edgecolor="#444" if hatch else "white",
                        lw=0.3, height=0.72)
                if k == "silent":
                    ax.text(100 * (left + v / 2), i, f"{100 * v:.1f}", va="center", ha="center", fontsize=5.6,
                            color="white", zorder=4)
                left += v
        ax.set_xlim(60, 100)
        ax.set_xticks([60, 70, 80, 90, 100])
        ax.set_xticklabels(["60", "70", "80", "90", ""])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL[a] for a in order])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of test orders (%)")
        title(ax, m)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=OUTCOME[k][0], hatch=OUTCOME[k][1], ec="#444", lw=0.3) for k in keys]
    fig.legend(handles, [names[k] for k in keys], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.13), handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.08, bottom=0.3)
    save(fig, "error_composition")


def fig_data_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.1), sharey=True)
    for ax, m in zip(axes, MODELS):
        for a in ARMS:
            v = np.array([ci_mean(m, a, s, "correct") for s in SIZES]) * 100
            ax.fill_between(SIZES, v[:, 1], v[:, 2], color=COLOR[a], alpha=0.10, lw=0)
            ax.plot(SIZES, v[:, 0], color=COLOR[a], ls=LINE[a], lw=1.5 if a == "words" else 0.9,
                    marker=MARKER[a], ms=5 if a == "words" else 3.2, mec="white", mew=0.3)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([tick(s) for s in SIZES], rotation=40, ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlabel("training pool size")
        title(ax, m)
    axes[0].set_ylabel("exact match (%)")
    handles = [plt.Line2D([], [], color=COLOR[a], ls=LINE[a], marker=MARKER[a], lw=1.5 if a == "words" else 0.9,
                          ms=5 if a == "words" else 3.2, mec="white", mew=0.3) for a in ARMS]
    fig.legend(handles, [LABEL[a] for a in ARMS], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.14), handlelength=2.4, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.08, bottom=0.33)
    save(fig, "data_efficiency")


def fig_silent_kinds():
    rows = {(r["model"], r["arm"]): r for r in E.load()}
    groups = [("negation", ("negation",)), ("line count", ("line count",)),
              ("wrong value", ("topping swapped", "value swapped")),
              ("invented / dropped", ("invented", "dropped")), ("other", ("other structure",))]
    style = {"negation": ("#D55E00", ""), "line count": ("#0072B2", ""), "wrong value": ("#E69F00", "////"),
             "invented / dropped": ("#56B4E9", "...."), "other": ("#BBBBBB", "")}
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        for i, a in enumerate(ARMS):
            s = D.get(m, a, HEAD, "silent").mean()
            bottom = 0.0
            for g, ks in groups:
                v = s * sum(rows[(m, a)][k] for k in ks)
                fc, hatch = style[g]
                ax.bar(i, 100 * v, bottom=100 * bottom, color=fc, hatch=hatch, width=0.72,
                       edgecolor="#444" if hatch else "white", lw=0.3)
                bottom += v
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([LABEL[a] for a in ARMS], rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(axis="x", visible=False)
        title(ax, m)
    axes[0].set_ylabel("silent errors (% of test orders)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=style[g][0], hatch=style[g][1], ec="#444", lw=0.3) for g, _ in groups]
    fig.legend(handles, [g for g, _ in groups], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=0.7, handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08, top=0.84)
    save(fig, "silent_kinds")


def fig_latency():
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for m, fill in zip(MODELS, (True, False)):
        rows = [json.loads(x) for x in open(C.base(m) / "latency_gpu.jsonl")]
        xs, ys = [], []
        for a in ARMS:
            v = [r for r in rows if r["arm"] == a]
            x = statistics.fmean(r["mean_generated_tokens"] for r in v)
            y = statistics.fmean(r["median_ms"] for r in v)
            xs.append(x)
            ys.append(y)
            ax.plot(x, y, marker=MARKER[a], ms=7 if a == "words" else 5, color=COLOR[a], mfc=COLOR[a] if fill else "white", mew=0.9, ls="")
        k, b = np.polyfit(xs, ys, 1)
        span = np.array([0, 64])
        ax.plot(span, k * span + b, color="#888", lw=0.7, ls="-" if fill else (0, (3, 2)), zorder=1)
        ax.text(62, k * 62 + b, C.LABEL[m], ha="right", va="bottom", fontsize=6.3, color="#444")
    ax.set_xlim(0, 64)
    ax.set_ylim(0)
    ax.set_xlabel("output tokens")
    ax.set_ylabel("median latency, batch 1 (ms)")
    handles = [plt.Line2D([], [], marker=MARKER[a], color=COLOR[a], ls="", ms=4) for a in ARMS]
    ax.legend(handles, [LABEL[a] for a in ARMS], ncol=2, loc="upper left", frameon=False, handletextpad=0.1)
    save(fig, "latency_vs_tokens")


if __name__ == "__main__":
    fig_accuracy_vs_tokens()
    fig_error_composition()
    fig_data_efficiency()
    fig_silent_kinds()
    fig_latency()
