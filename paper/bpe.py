#!/usr/bin/env python3
"""A byte-level BPE encoder read straight from a model's tokenizer.json.

Why this exists: the tokenization figure has to show real subword boundaries, and
`transformers` is deliberately a GPU-group dependency so the Mac never installs
it. Rather than draw the boundaries by hand, this reads the vocabulary and merge
list out of the tokenizer file and applies them, which needs nothing but
`regex`.

It is only trustworthy because it is checked: `verify()` re-tokenizes the saved
`raw` output of every prediction row and compares the length against the
`n_generated_tokens` the real tokenizer recorded at run time. If that check does
not pass exactly, the figure does not get built.
"""

import json
from functools import lru_cache
from pathlib import Path

import regex

HUB = Path.home() / ".cache" / "huggingface" / "hub"


def tokenizer_file(repo="Qwen/Qwen3.5-0.8B"):
    root = HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    hits = sorted(root.glob("*/tokenizer.json")) if root.exists() else []
    if not hits:
        raise FileNotFoundError(
            f"no tokenizer.json for {repo} under {root}. It is in the HF cache "
            f"only if the model has been downloaded on this machine."
        )
    return hits[0]


@lru_cache(maxsize=4)
def _byte_encoder():
    """GPT-2's byte to printable-unicode map, so every byte has a glyph."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        m = d["model"]
        if m.get("type") != "BPE":
            raise ValueError(f"expected a BPE model, got {m.get('type')}")
        self.vocab = m["vocab"]
        merges = m["merges"]
        # tokenizer.json stores merges either as "a b" or as ["a", "b"].
        pairs = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in merges]
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.split = regex.compile(d["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])
        self.bmap = _byte_encoder()
        self.specials = {a["content"] for a in d.get("added_tokens", [])}

    def _merge_word(self, word):
        syms = list(word)
        while len(syms) > 1:
            best, at = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at < 0:
                break
            syms[at : at + 2] = [syms[at] + syms[at + 1]]
        return syms

    def tokens(self, text):
        """The text as a list of subword strings, in order."""
        out = []
        for piece in self.split.findall(text):
            word = "".join(self.bmap[b] for b in piece.encode("utf-8"))
            out += self._merge_word(word)
        return out

    def ids(self, text):
        """Vocabulary ids for the tokens of `text`."""
        return [self.vocab.get(t, -1) for t in self.tokens(text)]

    def readable(self, text):
        """Same, but with the byte glyphs turned back into visible characters.

        A leading space shows as a middle dot so an empty-looking box is not
        mistaken for a bug.
        """
        return [t.replace("Ġ", "·") for t in self.tokens(text)]


def verify(bpe, rows, offset=1, cap=None):
    """(checked, mismatches) over (raw string, recorded token count) pairs.

    `offset` is 1 because a recorded count includes the stop token. Two kinds of
    disagreement are accepted, both artefacts of the saved file rather than of
    the encoding. A prediction is stripped before it is written, so an output
    that ended in a space is one token short on disk; the trailing-space variant
    is tried before a row is called a mismatch. And a row that reached the token
    cap was cut mid-token, so its string cannot round trip at all.
    """
    bad = []
    for raw, n in rows:
        if cap is not None and n >= cap:
            continue
        if len(bpe.tokens(raw)) + offset == n:
            continue
        if len(bpe.tokens(raw + " ")) + offset == n:
            continue
        bad.append((raw, n, len(bpe.tokens(raw)) + offset))
    return len(rows), bad
