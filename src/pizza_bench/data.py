"""Loading the PIZZA splits and taking deterministic subsets of train.

The shipped train file is unshuffled, so subsets always shuffle before slicing.
"""

import json
import random
import zipfile
from pathlib import Path

from .formats import PIZZA_DIR

SPLIT_FILES = {
    "train": "PIZZA_train.10_percent.json",
    "dev": "PIZZA_dev.json",
    "test": "PIZZA_test.json",
}
PROMPT = (
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def data_dir():
    return Path(PIZZA_DIR) / "data"


def _ensure_unzipped(path):
    if path.exists():
        return path
    archive = path.with_suffix(".json.zip")
    if not archive.exists():
        raise FileNotFoundError(
            f"{path} and {archive} both missing; clone the dataset first"
        )
    with zipfile.ZipFile(archive) as z:
        z.extract(path.name, path.parent)
    return path


def load(split):
    """-> list of {src, exr, pcfg_err}. pcfg_err is None for train."""
    path = _ensure_unzipped(data_dir() / SPLIT_FILES[split])
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                {
                    "src": d[f"{split}.SRC"],
                    "exr": d[f"{split}.EXR"],
                    "pcfg_err": d.get(f"{split}.PCFG_ERR") == "True"
                    if split != "train"
                    else None,
                }
            )
    return rows


def subset(rows, size, seed):
    idx = list(range(len(rows)))
    random.Random(10_000 + seed).shuffle(idx)
    if size > len(idx):
        raise ValueError(f"asked for {size} of {len(idx)} rows")
    return [rows[i] for i in idx[:size]]


def pcfg_error_subset(rows):
    return [r for r in rows if r["pcfg_err"]]


def as_prompt_completion(rows, fmt, encode_fn):
    return [
        {
            "prompt": PROMPT.format(r["src"]),
            "completion": encode_fn(r, fmt) + "<|im_end|>",
        }
        for r in rows
    ]
