"""analysis.md N3 (why real orders are refused) and N9 (training vs held-out cues).

    uv run --extra pizza python tools/analysis/extra.py
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common as C

import data as D

M = C.manifest()
HEAD = max(M["sizes"])
POLITE = re.compile(r"\b(hi|hello|hey|please|thanks|thank you|good (morning|afternoon|evening))\b")


def csv_out(name, rows):
    p = C.OUT / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def n3():
    from bench.tasks.pizza import negatives

    vocab = negatives.pizza_vocab() | {w for r in __import__("bench.tasks.pizza.data", fromlist=["x"]).train_subset(200_000, 0) for w in r["src"].split()}
    src = [r["src"] for r in C.predictions("qwen", "words", HEAD, 0)]
    oov = [any(w not in vocab for w in s.split()) for s in src]
    lines = [f"N3. Test orders with a word outside the training vocabulary ({len(vocab)} words, "
             f"from the first 500k pool rows plus a 200k random training subset): "
             f"{sum(oov)}/{len(oov)} = {sum(oov) / len(oov):.3f}", ""]
    lines.append("| model | format | P(OOV word given refused) | P(OOV word given answered) | refused rows (5 seeds) |")
    lines.append("|---|---|---|---|---|")
    polite = [bool(POLITE.search(s)) for s in src]
    rows = []
    for m in M["models"]:
        for arm in M["arms"]:
            ref = D.get(m, arm, HEAD, "d_refuse")
            p_ref = statistics.fmean(ref[:, i].mean() for i in range(len(src)) if polite[i])
            o_ref = statistics.fmean(ref[:, i].mean() for i in range(len(src)) if not polite[i])
            r_oov = [oov[i] for s in range(ref.shape[0]) for i in range(ref.shape[1]) if ref[s, i]]
            a_oov = [oov[i] for s in range(ref.shape[0]) for i in range(ref.shape[1]) if not ref[s, i]]
            lines.append(f"| {m} | {C.LABEL[arm]} | {statistics.fmean(r_oov):.3f} | "
                         f"{statistics.fmean(a_oov):.3f} | {len(r_oov)} |")
            rows.append({"model": m, "arm": arm, "oov_refused": round(statistics.fmean(r_oov), 4),
                         "oov_answered": round(statistics.fmean(a_oov), 4), "refused_rows": len(r_oov),
                         "refuse_polite": round(p_ref, 4), "refuse_other": round(o_ref, 4)})
    lines += ["", f"Orders with a greeting, please or thanks: {sum(polite)}/{len(polite)}; refusal rate polite vs other:"]
    lines += [f"- {r['model']} {C.LABEL[r['arm']]}: {r['refuse_polite']:.3f} vs {r['refuse_other']:.3f}" for r in rows]
    csv_out("refused.csv", rows)
    return lines


def n9():
    lines = ["", "N9. Corrections EM: dev (training cues, dev orders) at the selected checkpoint "
             "vs held-out cues on test orders, full pool, mean of 5 seeds", "",
             "| model | format | dev, training cues | test, held-out cues | gap |", "|---|---|---|---|---|"]
    rows = []
    for m in M["models"]:
        for arm in M["arms"]:
            dev, held = [], []
            for s in M["seeds"]:
                d = C.run_dir(m, arm, HEAD, s)
                curve = json.loads((d / "curves.json").read_text())["dev"]
                best = C.metrics(m, arm, HEAD, s)["best_step"]
                dev.append(next(c["dev_corrections_em"] for c in curve if c["step"] == best))
                held.append(C.metrics(m, arm, HEAD, s)["corrections"]["em"])
            a, b = statistics.fmean(dev), statistics.fmean(held)
            rows.append({"model": m, "arm": arm, "dev_training_cues": round(a, 4), "test_heldout_cues": round(b, 4)})
            lines.append(f"| {m} | {C.LABEL[arm]} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    csv_out("cue_gap.csv", rows)
    return lines


if __name__ == "__main__":
    out = n3() + n9()
    print(C.write("extra.md", "\n".join(out) + "\n"))
    print("\n".join(out))
