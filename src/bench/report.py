"""Tables, figures and summary.md, rebuilt from runs/ with no GPU and no regeneration."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import evaluate
from .config import model_task_dir, reports_dir
from .config import task as active_task

COLORS = {
    "exr": "#4e79a7",
    "json": "#76b7b2",
    "exr_short": "#59a14f",
    "tagged": "#f28e2b",
    "dsl": "#e15759",
    "words": "#b07aa1",
}


def chosen_lrs():
    p = model_task_dir() / "lrs.json"
    return json.loads(p.read_text()) if p.exists() else {}


TRAINING_STAGES = ("sweep", "grid")
REQUIRED_KEYS = ("arm", "size", "seed", "lr", "stage")


def primary_runs(cfg_sizes=None):
    """The runs the report is built from.

    Only sweep and grid rows at each arm's chosen learning rate. Everything else in runs/
    is excluded by construction: the fewshot summary (no lr), the smoke run (stage smoke,
    n=200), sweep runs at rejected learning rates, and the determinism replicate (tagged).
    """
    from .config import Config

    lrs = chosen_lrs()
    # for_active(), not Config(): the bare dataclass defaults are empty now and the
    # real sizes come from the selected task, so Config().sizes would reject every run.
    sizes = set(cfg_sizes or Config.for_active().sizes)
    out = []
    for r in load_runs():
        if any(k not in r for k in REQUIRED_KEYS):
            continue
        if r["stage"] not in TRAINING_STAGES or r.get("tag"):
            continue
        if r["size"] not in sizes:
            continue
        if lrs and abs(r["lr"] - lrs.get(r["arm"], -1)) > 1e-12:
            continue
        out.append(r)
    return out


def load_runs(stage=None):
    out = []
    for d in sorted(model_task_dir().glob("*/")):
        m = d / "metrics.json"
        if not m.exists():
            continue
        try:
            r = json.loads(m.read_text())
        except Exception:
            continue
        if r.get("complete") and (stage is None or r.get("stage") == stage):
            r["_dir"] = str(d)
            out.append(r)
    return out


def arm_key(a):
    arms = active_task().arms
    return arms.index(a) if a in arms else len(arms)


def bootstrap_ci(values, iters=10000, alpha=0.05, seed=0):
    import random

    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iters)
    )
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2))]


def paired_bootstrap(a_correct, b_correct, iters=10000, alpha=0.05, seed=0):
    """Paired over utterances, which is far better powered than three seeds."""
    import random

    rng = random.Random(seed)
    n = len(a_correct)
    diffs = [a - b for a, b in zip(a_correct, b_correct)]
    obs = statistics.fmean(diffs)
    boot = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(iters))
    lo, hi = boot[int(iters * alpha / 2)], boot[int(iters * (1 - alpha / 2))]
    return {
        "mean_diff": obs,
        "ci_lo": lo,
        "ci_hi": hi,
        "within_noise": lo <= 0 <= hi,
        "n_utterances": n,
    }


def summarize(runs, split="test", metric="em"):
    groups = defaultdict(list)
    for r in runs:
        if split in r and r[split].get(metric) is not None:
            groups[(r["arm"], r["size"])].append((r["seed"], r[split][metric]))
    rows = []
    for (arm, size), vals in groups.items():
        v = [x for _, x in sorted(vals)]
        lo, hi = bootstrap_ci(v)
        rows.append(
            {
                "arm": arm,
                "size": size,
                "n_seeds": len(v),
                "mean": statistics.fmean(v),
                "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "per_seed": ";".join(f"{x:.4f}" for x in v),
            }
        )
    return sorted(rows, key=lambda d: (arm_key(d["arm"]), d["size"]))


def token_table(runs, split="test"):
    groups = defaultdict(list)
    for r in runs:
        if split in r and r[split].get("mean_generated_tokens") is not None:
            groups[r["arm"]].append(r[split]["mean_generated_tokens"])
    return sorted(
        (
            {"arm": a, "mean_generated_tokens": statistics.fmean(v), "n_runs": len(v)}
            for a, v in groups.items()
        ),
        key=lambda d: arm_key(d["arm"]),
    )


def slope_on_tokens(runs, size, split="test", metric="em"):
    """OLS of the endpoint on mean generated tokens across the ladder arms only."""
    pts = []
    for r in runs:
        if (
            r["size"] != size
            or r["arm"] not in active_task().ladder_arms
            or split not in r
        ):
            continue
        x = r[split].get("mean_generated_tokens")
        y = r[split].get(metric)
        if x is not None and y is not None:
            pts.append((x, y))
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / denom
    return {
        "size": size,
        "metric": metric,
        "n_points": len(pts),
        "slope_per_token": slope,
        "slope_per_10_tokens": slope * 10,
        "intercept": my - slope * mx,
    }


def matched_pairs(runs, split="test"):
    out = []
    by = {(r["arm"], r["size"], r["seed"]): r for r in runs}
    for a, b in active_task().matched_pairs:
        for size in sorted({r["size"] for r in runs}):
            for seed in sorted({r["seed"] for r in runs}):
                ra, rb = by.get((a, size, seed)), by.get((b, size, seed))
                if not ra or not rb:
                    continue
                pa = evaluate.read_predictions(
                    Path(ra["_dir"]) / f"{split}_predictions.jsonl"
                )
                pb = evaluate.read_predictions(
                    Path(rb["_dir"]) / f"{split}_predictions.jsonl"
                )
                res = paired_bootstrap(
                    [int(x["correct"]) for x in pa], [int(x["correct"]) for x in pb]
                )
                out.append(
                    {
                        "a": a,
                        "b": b,
                        "size": size,
                        "seed": seed,
                        "tokens_a": ra[split]["mean_generated_tokens"],
                        "tokens_b": rb[split]["mean_generated_tokens"],
                        **res,
                    }
                )
    return out


def render(table, cols=None, floatfmt="{:.4f}"):
    if not table:
        return "(empty)"
    cols = cols or list(table[0].keys())
    cell = lambda v: (
        "-" if v is None else (floatfmt.format(v) if isinstance(v, float) else str(v))
    )
    w = {c: max(len(c), *(len(cell(r.get(c))) for r in table)) for c in cols}
    lines = [
        "| " + " | ".join(c.ljust(w[c]) for c in cols) + " |",
        "|" + "|".join("-" * (w[c] + 2) for c in cols) + "|",
    ]
    for r in table:
        lines.append("| " + " | ".join(cell(r.get(c)).ljust(w[c]) for c in cols) + " |")
    return "\n".join(lines)


def to_csv(table, path):
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def _plt():
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


def fig_em_vs_tokens(runs, split="test", out=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sizes = sorted({r["size"] for r in runs})
    markers = {sizes[0]: "o", sizes[-1]: "s"} if sizes else {}
    for size in sizes:
        for arm in active_task().arms:
            pts = [
                (r[split]["mean_generated_tokens"], r[split]["em"])
                for r in runs
                if r["arm"] == arm and r["size"] == size and split in r
            ]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.errorbar(
                statistics.fmean(xs),
                statistics.fmean(ys),
                yerr=(statistics.stdev(ys) if len(ys) > 1 else 0),
                marker=markers.get(size, "o"),
                color=COLORS.get(arm),
                capsize=3,
                ms=7,
                ls="none",
                label=f"{arm} n={size}",
            )
    ax.set_xlabel("mean generated tokens per utterance")
    ax.set_ylabel(f"unordered exact match ({split})")
    ax.set_title("Accuracy against output length")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    if out:
        _save(fig, out, "em_vs_tokens")
    return fig


def _save(fig, out, name):
    out = Path(out)
    (out).mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=150)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def latency_table():
    from .latency import results_path

    p = results_path()
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    out = []
    for arm, rs in by.items():
        out.append(
            {
                "arm": arm,
                "n_runs": len(rs),
                "n_utterances": rs[0].get("n"),
                "median_ms": statistics.fmean(r["median_ms"] for r in rs),
                "p90_ms": statistics.fmean(r["p90_ms"] for r in rs),
                "generated_tokens": statistics.fmean(
                    r["mean_generated_tokens"] for r in rs
                ),
                "tokens_per_second": statistics.fmean(
                    r["tokens_per_second"] for r in rs
                ),
                "em": statistics.fmean(r["em"] for r in rs if "em" in r)
                if any("em" in r for r in rs)
                else None,
                "proc_mem_mib": statistics.fmean(
                    r["gpu_memory_used_by_this_process_mib"]
                    for r in rs
                    if "gpu_memory_used_by_this_process_mib" in r
                )
                if any("gpu_memory_used_by_this_process_mib" in r for r in rs)
                else None,
                "device_mem_used_mib": statistics.fmean(
                    r["gpu_memory_used_mib"] for r in rs if "gpu_memory_used_mib" in r
                )
                if any("gpu_memory_used_mib" in r for r in rs)
                else None,
            }
        )
    return sorted(out, key=lambda d: arm_key(d["arm"]))


def fewshot_table():
    p = model_task_dir() / "fewshot" / "metrics.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return sorted(
        (
            {"arm": a, **{k: v for k, v in m.items()}}
            for a, m in d.get("arms", {}).items()
        ),
        key=lambda x: arm_key(x["arm"]),
    )


def build(out_dir=None, split="test"):
    out = Path(out_dir or reports_dir())
    tables, figs = out / "tables", out / "figures"
    runs = primary_runs()
    if not runs:
        return {"error": "no completed grid runs in runs/"}

    em = summarize(runs, split)
    toks = token_table(runs, split)
    halluc = summarize(runs, split, "hallucinated_entity")
    validity = summarize(runs, split, "parse_validity")
    cond = summarize(runs, split, "em_given_valid")
    to_csv(cond, tables / "em_given_valid.csv")
    to_csv(em, tables / f"em_{split}.csv")
    to_csv(summarize(runs, "dev"), tables / "em_dev.csv")
    to_csv(toks, tables / "generated_tokens.csv")
    to_csv(halluc, tables / "hallucination.csv")
    to_csv(validity, tables / "validity.csv")

    slopes = [
        s
        for size in sorted({r["size"] for r in runs})
        for m in ("em", "hallucinated_entity", "parse_validity")
        for s in [slope_on_tokens(runs, size, split, m)]
        if s
    ]
    to_csv(slopes, tables / "slopes.csv")
    pairs = matched_pairs(runs, split)
    to_csv(pairs, tables / "matched_pairs.csv")

    lat = latency_table()
    few = fewshot_table()
    to_csv(lat, tables / "latency_gpu.csv")
    to_csv(few, tables / "fewshot.csv")

    fig_em_vs_tokens(runs, split, figs)

    task = active_task()
    lines = [
        f"# {task.label} output-length study: results",
        "",
        f"Built from {len(runs)} completed grid runs in `{model_task_dir()}`. Every number here is",
        "recomputable with `python -m bench report`. Protocol: `PROTOCOL.md`.",
        "",
        "## Accuracy by arm and train size",
        "",
        render(
            em, ["arm", "size", "n_seeds", "mean", "std", "ci_lo", "ci_hi", "per_seed"]
        ),
        "",
        "## Parse validity",
        "",
        render(validity, ["arm", "size", "n_seeds", "mean", "std"]),
        "",
        "## Accuracy among outputs that parsed",
        "",
        "If this is flat across arms while exact match is not, the arms differ in how",
        "reliably they emit a well-formed tree, not in how well they parse the utterance.",
        "",
        render(cond, ["arm", "size", "n_seeds", "mean", "std"]),
        "",
        "## Mean generated tokens",
        "",
        render(toks, ["arm", "mean_generated_tokens", "n_runs"]),
        "",
        "## Slope of each endpoint on output length (ladder arms only)",
        "",
        render(slopes, ["size", "metric", "n_points", "slope_per_10_tokens"]),
        "",
        "## Matched-token controls",
        "",
        "Same token budget, different format family. A difference here means format family",
        "matters independently of length.",
        "",
        render(
            pairs,
            [
                "a",
                "b",
                "size",
                "seed",
                "tokens_a",
                "tokens_b",
                "mean_diff",
                "ci_lo",
                "ci_hi",
                "within_noise",
            ],
        ),
        "",
        "## Batch-size-1 latency on the training GPU",
        "",
        "Measured on a shared GPU, so absolute milliseconds include other tenants'",
        "contention. gpu_mem_used_mib records how busy the device was during each run.",
        "",
        render(
            lat,
            [
                "arm",
                "n_runs",
                "median_ms",
                "p90_ms",
                "generated_tokens",
                "tokens_per_second",
                "em",
                "proc_mem_mib",
            ],
            "{:.1f}",
        ),
        "",
        "## Few-shot baseline, no fine-tuning",
        "",
        render(
            few,
            ["arm", "em", "parse_validity", "em_given_valid", "mean_generated_tokens"],
        ),
        "",
    ]
    # Only for a task that has published numbers. Quoting another task's baselines
    # here is how a cafe report ends up carrying PIZZA's.
    if task.published:
        lines += [
            "## Published reference points",
            "",
            f"Trained on {task.published_train_n:,} utterances, against at most "
            f"{max(r['size'] for r in runs):,} here. Not a like-for-like comparison, "
            "and not",
            "claimed as one. See PROTOCOL.md section 9.",
            "",
            render(
                [{"system": k, "test_em": v} for k, v in task.published],
                ["system", "test_em"],
            ),
            "",
        ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines))
    return {"runs": len(runs), "out": str(out)}
