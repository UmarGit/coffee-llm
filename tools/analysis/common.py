"""Shared loading for the neg-v1 analysis (analysis.md). Reads run directories only."""

import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CAMPAIGN", REPO / "neg-v1-results/campaign"))
OUT = REPO / "analysis/out"
MODELS = ("qwen", "lfm2")
ARMS = ("exr", "json", "exr_short", "tagged", "dsl", "words")
LADDER = ("exr", "exr_short", "tagged", "dsl")
LABEL = {"exr": "EXR", "json": "JSON", "exr_short": "EXR-short", "tagged": "TAGGED",
         "dsl": "DSL", "words": "WORDS", "qwen": "Qwen3.5-0.8B", "lfm2": "LFM2.5-350M"}


def manifest():
    return json.loads((ROOT / "campaign.json").read_text())


def base(model):
    return ROOT / model / "pizza"


@lru_cache(maxsize=None)
def lrs(model):
    return json.loads((base(model) / "lrs.json").read_text())


def run_id(arm, size, seed, lr, tag=""):
    return f"{arm}-n{size}-s{seed}-lr{lr:g}" + (f"-{tag}" if tag else "")


def cells():
    m = manifest()
    return [(model, arm, size, seed) for model in m["models"] for arm in m["arms"]
            for size in m["sizes"] for seed in m["seeds"]]


def run_dir(model, arm, size, seed):
    return base(model) / run_id(arm, size, seed, lrs(model)[arm])


@lru_cache(maxsize=None)
def metrics(model, arm, size, seed):
    return json.loads((run_dir(model, arm, size, seed) / "metrics.json").read_text())


def predictions(model, arm, size, seed, split="test"):
    with open(run_dir(model, arm, size, seed) / f"{split}_predictions.jsonl") as f:
        return [json.loads(line) for line in f]


def write(name, text):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
