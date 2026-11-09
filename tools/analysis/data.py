"""Per-utterance outcome arrays for every cell, cached in analysis/out/cache.npz.

For (model, arm, size) each array is (seeds, rows). Test rows:
correct, silent, d_parse, d_refuse (they sum to 1 per row), tokens,
hallucinated (non-number entities invented), missing (entities left out).
Negatives rows: handled. Corrections rows: em, applied (probes kept apart).
"""

import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

TEST_KEYS = ("correct", "silent", "d_parse", "d_refuse", "tokens", "hallucinated", "missing")
CACHE = C.OUT / "cache.npz"


def _test_rows(model, arm, size, seed):
    from bench.tasks.pizza import metrics, negatives

    out = {k: [] for k in TEST_KEYS}
    for r in C.predictions(model, arm, size, seed):
        pred, ok = r["parsed"], bool(r["correct"])
        refuse = negatives.has_reject(r["raw"], arm) and pred is None
        out["correct"].append(ok)
        out["silent"].append(pred is not None and not ok)
        out["d_refuse"].append(refuse)
        out["d_parse"].append(pred is None and not refuse)
        out["tokens"].append(r["n_generated_tokens"])
        if pred is None:
            out["hallucinated"].append(0)
            out["missing"].append(0)
            continue
        pb, gb = metrics.leaf_bag(pred), metrics.leaf_bag(r["gold"])
        out["hallucinated"].append(sum(v for k, v in (pb - gb).items() if k[0][0] != "NUMBER"))
        out["missing"].append(sum((gb - pb).values()))
    return out


def _cell(cell):
    model, arm, size = cell
    res = {}
    for seed in C.manifest()["seeds"]:
        t = _test_rows(model, arm, size, seed)
        d = C.run_dir(model, arm, size, seed)
        neg = [json.loads(x) for x in open(d / "negatives_predictions.jsonl")]
        fix = [json.loads(x) for x in open(d / "corrections_predictions.jsonl")]
        for k, v in t.items():
            res.setdefault(k, []).append(v)
        res.setdefault("neg_handled", []).append([bool(x["correct"]) for x in neg])
        res.setdefault("fix_em", []).append([bool(x["correct"]) for x in fix if not x["probe"]])
        res.setdefault("fix_applied", []).append([bool(x["applied"]) for x in fix if not x["probe"]])
        res.setdefault("probe_em", []).append([bool(x["correct"]) for x in fix if x["probe"]])
    return cell, {k: np.array(v, dtype=float) for k, v in res.items()}


def _key(cell, name):
    return "|".join(map(str, cell)) + "|" + name


def build():
    m = C.manifest()
    cells = [(model, arm, size) for model in m["models"] for arm in m["arms"] for size in m["sizes"]]
    with Pool() as pool:
        done = pool.map(_cell, cells, chunksize=2)
    flat = {_key(c, k): v for c, arrs in done for k, v in arrs.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **flat)
    return flat


@lru_cache(maxsize=None)
def _all():
    if not CACHE.exists():
        build()
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def get(model, arm, size, name):
    return _all()[_key((model, arm, size), name)]


@lru_cache(maxsize=None)
def labels(kind, seed):
    """Row labels (kind of negative or correction) for one seed. The rows depend on
    the seed but are identical across models, formats and sizes (I13)."""
    d = C.run_dir("qwen", "words", C.manifest()["sizes"][-1], seed)
    rows = [json.loads(x) for x in open(d / f"{kind}_predictions.jsonl")]
    if kind == "corrections":
        return [r["kind"] for r in rows if not r["probe"]], [r["src"] for r in rows if r["probe"]]
    return [r["kind"] for r in rows], None


if __name__ == "__main__":
    flat = build()
    print(f"wrote {CACHE}: {len(flat)} arrays")
