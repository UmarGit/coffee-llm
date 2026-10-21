"""Run configuration and the on-disk layout.

A run is identified by (model, task, arm, size, seed, lr). The model and the task
are chosen once, by `select`, and everything else reads them back through `model()`
and `task()` rather than being handed them through every signature. Results live
under runs/<model>/<task>/, so two models or two tasks can never overwrite each
other's token caps, chosen learning rates or predictions.

See PROTOCOL.md for the frozen protocol.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import models as _models
from . import tasks as _tasks

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("BENCH_RUNS", os.environ.get("PIZZA_RUNS", REPO / "runs")))
REPORTS = Path(
    os.environ.get("BENCH_REPORTS", os.environ.get("PIZZA_REPORTS", REPO / "reports"))
)

DEFAULT_MODEL = "qwen"
DEFAULT_TASK = "pizza"

_active_model = None
_active_task = None


def select(model_name=DEFAULT_MODEL, task_name=DEFAULT_TASK):
    """Pick the model and task for this process. Called once, by the CLI."""
    global _active_model, _active_task
    _active_model = _models.get(model_name)
    _active_task = _tasks.get(task_name)
    return _active_model, _active_task


def model():
    if _active_model is None:
        select()
    return _active_model


def task():
    if _active_task is None:
        select()
    return _active_task


@dataclass(frozen=True)
class Config:
    arms: tuple = ()
    sizes: tuple = ()
    determinism_arm: str = ""
    pool: str = "full"
    seeds: tuple = (0, 1, 2)
    shard: tuple = (0, 1)
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    max_steps: int = 4000
    min_steps: int = 1000
    batch_size: int = 16
    warmup_steps: int = 50
    max_seq_length: int = 640
    eval_every: int = 100
    patience: int = 10
    gen_batch_size: int = 32
    lr_candidates: tuple = (1e-4, 2e-4, 5e-4)
    lr_select_size: int = 0
    lr_select_seeds: tuple = (0, 1, 2)
    truncation_warn: float = 0.01
    caps: dict = field(default_factory=dict)
    lrs: dict = field(default_factory=dict)

    @classmethod
    def for_active(cls, **kw):
        """Defaults taken from the selected task, so adding a task does not mean
        editing this file."""
        t = task()
        base = cls(
            arms=t.arms,
            sizes=t.default_sizes,
            determinism_arm=t.default_arm,
            lr_select_size=t.lr_select_size,
        )
        return base.with_(**kw) if kw else base

    def with_(self, **kw):
        return replace(self, **kw)

    def to_json(self):
        d = asdict(self)
        d["model"] = model().name
        d["model_hf_id"] = model().hf_id
        d["task"] = task().name
        return json.dumps(d, indent=2, sort_keys=True, default=list)


def model_task_dir():
    return RUNS / model().name / task().name


def run_id(arm, size, seed, lr, tag=""):
    """Identity within a (model, task) is (arm, size, seed, lr). The stage is
    recorded inside metrics.json, not in the id, so a sweep run at the chosen LR
    IS the grid cell and is never repeated."""
    base = f"{arm}-n{size}-s{seed}-lr{lr:g}"
    return base + (f"-{tag}" if tag else "")


def run_dir(rid):
    return model_task_dir() / rid


def is_complete(rid):
    m = run_dir(rid) / "metrics.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text()).get("complete") is True
    except Exception:
        return False


def caps_path():
    return model_task_dir() / "caps.json"


def lrs_path():
    return model_task_dir() / "lrs.json"


def reports_dir():
    return REPORTS / model().name / task().name


def load_caps():
    p = caps_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m bench prepare` first")
    return json.loads(p.read_text())


def load_lrs():
    p = lrs_path()
    return json.loads(p.read_text()) if p.exists() else {}


def git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return os.environ.get("BENCH_COMMIT", "unknown")


def describe_env():
    import platform

    env = {
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "model": model().name,
        "model_hf_id": model().hf_id,
        "task": task().name,
    }
    try:
        import torch

        env.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            n_gpus=torch.cuda.device_count(),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        )
    except Exception:
        env["torch"] = None
    env["tokenizer_class"] = None
    for mod in ("transformers", "trl", "peft", "datasets", "unsloth", "xgrammar"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env
