"""Hierarchical bootstrap and Holm correction, as fixed in analysis.md section 2.

A statistic is a function of per-seed means. Each resample draws seeds with
replacement (paired by seed index across formats: they share training data)
and utterances with replacement (paired: every format answers the same rows),
then evaluates the statistic on the resampled per-seed means.
"""

import numpy as np

B = 10_000
_rng = np.random.default_rng(20261109)
_W = {}


def weights(n):
    """(B, n) multinomial utterance counts, one draw per row count, shared by
    every comparison so that all statistics see the same resamples."""
    if n not in _W:
        _W[n] = _rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)
    return _W[n]


_S = {}


def seed_draws(k):
    if k not in _S:
        _S[k] = _rng.integers(0, k, size=(B, k))
    return _S[k]


def resampled_means(x):
    """x: (seeds, rows) -> (B, seeds) per-seed means under resampled rows and seeds."""
    k, n = x.shape
    m = (weights(n) @ x.T.astype(np.float32)) / n          # (B, seeds)
    return np.take_along_axis(m, seed_draws(k), axis=1)


def bootstrap(stat, arrays):
    """stat(list of (*, seeds) per-seed means) -> (*,). Returns point, lo, hi, p."""
    point = float(stat([a.mean(axis=1)[None, :] for a in arrays])[0])
    draws = stat([resampled_means(a) for a in arrays])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"est": point, "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


def diff(a, b):
    """Mean over seeds of a minus b."""
    return bootstrap(lambda m: (m[0] - m[1]).mean(axis=1), [a, b])


def ratio(a, b):
    """Ratio of seed-mean a to seed-mean b (e.g. silent rate EXR / WORDS)."""
    out = bootstrap(lambda m: m[0].mean(axis=1) / np.maximum(m[1].mean(axis=1), 1e-12), [a, b])
    out["p"] = None
    return out


def slope(ems, toks):
    """Per seed, least-squares slope of EM on mean tokens across formats; mean over seeds.
    Reported per 10 tokens: positive means removing tokens lowers EM."""
    k = len(ems)

    def stat(m):
        e, t = np.stack(m[:k]), np.stack(m[k:])            # (formats, *, seeds)
        tc = t - t.mean(axis=0)
        s = (tc * (e - e.mean(axis=0))).sum(axis=0) / (tc ** 2).sum(axis=0)
        return 10 * s.mean(axis=1)

    return bootstrap(stat, list(ems) + list(toks))


def holm(ps):
    """Holm-adjusted p-values, same order as given; None entries are skipped."""
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - rank) * ps[i]))
        adj[i] = run
    return adj
