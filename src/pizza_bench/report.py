"""Tables, bootstrap CIs and figures. Torch-free so results can be analysed anywhere."""

import csv
import json
import os
import statistics
from collections import defaultdict

FORMAT_ORDER = ["exr", "json", "dsl"]
COLORS = {"exr": "#4e79a7", "json": "#f28e2b", "dsl": "#e15759"}


def load_results(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"skipping malformed line: {line[:80]}...")
    return rows


def fmt_key(f):
    return FORMAT_ORDER.index(f) if f in FORMAT_ORDER else len(FORMAT_ORDER)


def bootstrap_ci(values, iters=10000, alpha=0.05, seed=0):
    import random

    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iters)
    )
    return (means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2))])


def summarize(rows, split="dev", metric="em", condition="B_sft"):
    groups = defaultdict(list)
    for r in rows:
        if r.get("condition") != condition or split not in r:
            continue
        value = r[split].get(metric)
        if value is not None:
            groups[(r["format"], r["size"])].append(value)
    out = []
    for (fmt, size), vals in groups.items():
        lo, hi = bootstrap_ci(vals)
        out.append(
            {
                "format": fmt,
                "size": size,
                "n_seeds": len(vals),
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return sorted(out, key=lambda d: (fmt_key(d["format"]), d["size"]))


def paired(rows, split="dev", a="dsl", b="exr", metric="em"):
    index = {
        (r["format"], r["size"], r["seed"]): r[split][metric]
        for r in rows
        if r.get("condition") == "B_sft" and split in r
    }
    sizes = sorted({k[1] for k in index})
    seeds = sorted({k[2] for k in index})
    out = []
    for size in sizes:
        diffs = [
            index[(a, size, s)] - index[(b, size, s)]
            for s in seeds
            if (a, size, s) in index and (b, size, s) in index
        ]
        if not diffs:
            continue
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        lo, hi = bootstrap_ci(diffs)
        out.append(
            {
                "a": a,
                "b": b,
                "size": size,
                "n_pairs": len(diffs),
                "mean_diff": statistics.fmean(diffs),
                "sd_diff": sd,
                "cohens_dz": (statistics.fmean(diffs) / sd) if sd else None,
                "ci_lo": lo,
                "ci_hi": hi,
                "within_noise": lo is None or lo <= 0 <= hi,
            }
        )
    return out


def render(table, cols=None, floatfmt="{:.4f}"):
    if not table:
        return "(empty)"
    cols = cols or list(table[0].keys())

    def cell(v):
        if v is None:
            return "-"
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    w = {c: max(len(c), *(len(cell(r.get(c))) for r in table)) for c in cols}
    lines = [
        "  ".join(c.rjust(w[c]) for c in cols),
        "  ".join("-" * w[c] for c in cols),
    ]
    for r in table:
        lines.append("  ".join(cell(r.get(c)).rjust(w[c]) for c in cols))
    return "\n".join(lines)


def to_csv(table, path):
    if not table:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def to_latex(table, path, caption, label, floatfmt="{:.4f}"):
    if not table:
        return
    cols = list(table[0].keys())

    def cell(v):
        if v is None:
            return "--"
        return (
            floatfmt.format(v) if isinstance(v, float) else str(v).replace("_", r"\_")
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "\\begin{table}[t]\n\\centering\n\\begin{tabular}{"
            + "l" * len(cols)
            + "}\n"
        )
        f.write(
            "\\toprule\n"
            + " & ".join(c.replace("_", r"\_") for c in cols)
            + " \\\\\n\\midrule\n"
        )
        f.writelines(
            " & ".join(cell(r.get(c)) for c in cols) + " \\\\\n" for r in table
        )
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


PUBLISHED = [
    {"system": "PCFG", "em_test": 68.02, "source": "Arkoudas et al. 2022"},
    {"system": "BERT pipeline", "em_test": 64.08, "source": "Arkoudas et al. 2022"},
    {"system": "BART-Large EXR", "em_test": 78.56, "source": "Arkoudas et al. 2022"},
    {
        "system": "BART-Large on PCFG-error subset",
        "em_test": 59.49,
        "source": "Arkoudas et al. 2022",
    },
]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def fig_data_efficiency(rows, split="dev", metric="em", baseline=78.56):
    plt = _style()
    table = summarize(rows, split, metric)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for fmt in sorted({d["format"] for d in table}, key=fmt_key):
        pts = sorted([d for d in table if d["format"] == fmt], key=lambda d: d["size"])
        xs = [d["size"] for d in pts]
        ax.plot(
            xs,
            [d["mean"] for d in pts],
            marker="o",
            label=fmt.upper(),
            color=COLORS.get(fmt),
        )
        if all(d["ci_lo"] is not None for d in pts):
            ax.fill_between(
                xs,
                [d["ci_lo"] for d in pts],
                [d["ci_hi"] for d in pts],
                alpha=0.15,
                color=COLORS.get(fmt),
            )
    if baseline:
        ax.axhline(baseline / 100, ls="--", lw=1, color="#666")
        ax.text(
            ax.get_xlim()[0],
            baseline / 100 + 0.005,
            "BART-Large EXR (published)",
            fontsize=7,
            color="#666",
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted({d["size"] for d in table}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("training examples")
    ax.set_ylabel(f"unordered exact match ({split})")
    ax.legend()
    fig.tight_layout()
    return fig


ERROR_CATEGORIES = (
    "quantity",
    "missing_topping",
    "negation_scope",
    "merged_or_split_suborder",
    "unseen_entity",
    "other",
)


def categorize(pred_exr, gold_exr, catalog_ids=None):
    """Coarse error taxonomy for the dev error analysis."""
    from .formats import BARE, parse_exr

    if pred_exr is None:
        return "invalid_output"
    pnode, gnode = parse_exr(pred_exr), parse_exr(gold_exr)
    if len(pnode[1]) != len(gnode[1]):
        return "merged_or_split_suborder"

    def leaves(node, negated=False, out=None, quant=False):
        out = [] if out is None else out
        ctor, kids = node
        neg = negated or ctor == "NOT"
        q = quant or ctor == "COMPLEX_TOPPING"
        for k in kids:
            if isinstance(k, str):
                out.append((ctor, k, neg, q))
            else:
                leaves(k, neg, out, q)
        return out

    pl, gl = leaves(pnode), leaves(gnode)
    ps, gs = set(pl), set(gl)
    if catalog_ids is not None:
        for ctor, val, _, _ in ps - gs:
            if ctor in BARE and val not in catalog_ids.get(ctor, set()):
                return "unseen_entity"
    if {x for x in ps if x[0] == "NUMBER"} != {x for x in gs if x[0] == "NUMBER"}:
        return "quantity"
    if {(c, v) for c, v, _, _ in gs} - {(c, v) for c, v, _, _ in ps}:
        return "missing_topping"
    if {(c, v, n) for c, v, n, _ in ps} != {(c, v, n) for c, v, n, _ in gs}:
        return "negation_scope"
    return "other"
