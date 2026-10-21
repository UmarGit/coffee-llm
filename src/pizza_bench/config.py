"""Run configuration and the on-disk layout. See PROTOCOL.md for the frozen protocol."""
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("PIZZA_RUNS", REPO / "runs"))
REPORTS = Path(os.environ.get("PIZZA_REPORTS", REPO / "reports/pizza"))
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EOS = "<|im_end|>"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class Config:
    arms: tuple = ("exr", "json", "exr_short", "tagged", "dsl", "words")
    sizes: tuple = (1_000, 10_000, 100_000, 1_000_000, 2_456_443)
    determinism_arm: str = "dsl"
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
    lr_select_size: int = 100_000
    lr_select_seeds: tuple = (0, 1, 2)
    truncation_warn: float = 0.01
    caps: dict = field(default_factory=dict)
    lrs: dict = field(default_factory=dict)

    def with_(self, **kw):
        return replace(self, **kw)

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=list)


def run_id(arm, size, seed, lr, tag=""):
    """Identity is (arm, size, seed, lr). The stage is recorded inside metrics.json, not
    in the id, so a sweep run at the chosen LR IS the grid cell and is never repeated."""
    base = f"{arm}-n{size}-s{seed}-lr{lr:g}"
    return base + (f"-{tag}" if tag else "")


def run_dir(rid):
    return RUNS / rid


def is_complete(rid):
    m = run_dir(rid) / "metrics.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text())
        return d.get("complete") is True
    except Exception:
        return False


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.environ.get("PIZZA_COMMIT", "unknown")


def describe_env():
    import platform
    env = {"python": platform.python_version(), "git_commit": git_commit(),
           "platform": platform.platform()}
    try:
        import torch
        env.update(torch=torch.__version__,
                   cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   n_gpus=torch.cuda.device_count(),
                   bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
    except Exception:
        env["torch"] = None
    env["tokenizer_class"] = None
    for mod in ("transformers", "trl", "peft", "datasets", "unsloth", "xgrammar"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def caps_path():
    return RUNS / "caps.json"


def load_caps():
    p = caps_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m pizza_bench prepare` first")
    return json.loads(p.read_text())
