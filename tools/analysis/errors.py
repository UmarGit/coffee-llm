"""What kind of mistake each silent error is (analysis.md, error analysis).

    uv run --extra pizza python tools/analysis/errors.py

Writes analysis/out/tables/silent_kinds.csv: per model and format at the full
pool, the share of silent errors (all 5 seeds pooled) in each kind.
"""

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common as C

KINDS = ("line count", "negation", "topping swapped", "value swapped", "invented", "dropped", "other structure")


def kind(pred, gold):
    from bench.tasks.pizza import metrics

    pb, gb = metrics.leaf_bag(pred), metrics.leaf_bag(gold)
    if metrics.suborder_count(pred) != metrics.suborder_count(gold):
        return "line count"
    extra, miss = pb - gb, gb - pb
    if extra and miss:
        if {k[0][0] for k in extra} == {k[0][0] for k in miss} == {"TOPPING"}:
            return "topping swapped"
        if "NOT" in str(list(extra) + list(miss)):
            return "negation"
        return "value swapped"
    if extra:
        return "invented"
    if miss:
        return "dropped"
    return "other structure"


def table():
    m = C.manifest()
    head = max(m["sizes"])
    rows = []
    for model in m["models"]:
        for arm in m["arms"]:
            c = collections.Counter()
            for s in m["seeds"]:
                for r in C.predictions(model, arm, head, s):
                    if r["parsed"] and not r["correct"]:
                        c[kind(r["parsed"], r["gold"])] += 1
            n = sum(c.values())
            rows.append({"model": model, "arm": arm, "n": n, **{k: round(c[k] / n, 4) for k in KINDS}})
    return rows


def load():
    p = C.OUT / "tables" / "silent_kinds.csv"
    if not p.exists():
        main()
    with open(p) as f:
        return [{k: (v if k in ("model", "arm") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    rows = table()
    p = C.OUT / "tables" / "silent_kinds.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:5} {r['arm']:10} n={int(r['n']):4} " + "  ".join(f"{k} {r[k]:.2f}" for k in KINDS))
    print("wrote", p)


if __name__ == "__main__":
    main()
