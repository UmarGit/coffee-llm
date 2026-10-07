"""Training/eval harness for the NoRA vs LoRA study.

transformers + TRL + PEFT only. The stack is identical across methods; only the
LoraConfig differs. Requires torch, so it is not imported by coffee_parser/__init__.
"""
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed

from . import generator as gen
from .menu import UNCLEAR
from .nora import check_invariants, normalize_lora_A_
from .ollama import PROMPT, STOPS
from .parsing import parse_compact
from .testset import TEST_SET

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# method -> (use_nora value or None, alpha multiplier on r)
METHODS = {
    "lora": (None, 2),
    "lora_alpha_r": (None, 1),
    "nora_init": ("init", 1),
    "nora": (True, 1),
}

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303, "hard": 404}


@dataclass
class Grid:
    methods: tuple = ("lora", "lora_alpha_r", "nora_init", "nora")
    sizes: tuple = (100, 300, 1000)
    ranks: tuple = (16,)
    seeds: tuple = (0, 1, 2)
    max_steps: int = 250
    batch_size: int = 16
    warmup_steps: int = 10
    max_seq_length: int = 192
    eval_every: int = 25
    curve_val_n: int = 150
    gen_batch_size: int = 64
    max_new_tokens: int = 32
    lr_candidates: tuple = (1e-4, 2e-4, 5e-4)
    lr_select_size: int = 1000
    lr_select_rank: int = 16
    lr_select_seed: int = 0
    lrs: dict = field(default_factory=dict)


# ---------- environment ----------

def describe_env():
    import datasets, peft, transformers, trl
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = os.environ.get("COFFEE_LLM_COMMIT", "unknown")
    return {
        "gpu": gpu,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
        "nora_fork": "use_nora" in LoraConfig.__dataclass_fields__,
        "python": platform.python_version(),
        "git_commit": commit,
    }


def precision(env):
    """bf16 where supported, else fp32. Qwen3.5 is unstable in fp16 on a T4."""
    if env["bf16"]:
        return {"dtype": torch.bfloat16, "bf16": True, "fp16": False}
    return {"dtype": torch.float32, "bf16": False, "fp16": False}


# ---------- data ----------

def build_splits(train_pool=3800, val_n=300, test_n=1000, hard_n=1000):
    hand = [{"input": t, "output": o} for t, o in TEST_SET]
    seen = {e["input"] for e in hand}
    out = {"hand": hand}
    for name, n, cfg in [("train", train_pool, gen.DEFAULT), ("val", val_n, gen.DEFAULT),
                         ("test", test_n, gen.DEFAULT), ("hard", hard_n, gen.HARD)]:
        s = gen.generate_unique(n, seed=SPLIT_SEEDS[name], cfg=cfg, exclude=seen)
        seen |= {e["input"] for e in s}
        out[name] = s
    return out


def subset(train, size, seed):
    import random
    idx = list(range(len(train)))
    random.Random(1000 + seed).shuffle(idx)
    return [train[i] for i in idx[:size]]


def to_trl(examples):
    return Dataset.from_list([
        {"prompt": PROMPT.format(e["input"]), "completion": e["output"] + "<|im_end|>"}
        for e in examples
    ])


# ---------- model ----------

def build_model(method, rank, seed, env):
    use_nora, alpha_mult = METHODS[method]
    set_seed(seed)
    prec = precision(env)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=prec["dtype"])
    model.config.use_cache = False
    cfg = dict(r=rank, lora_alpha=alpha_mult * rank, lora_dropout=0.0,
               target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    if use_nora is not None:
        cfg["use_nora"] = use_nora
    peft_model = get_peft_model(model, LoraConfig(**cfg))
    if use_nora == "init" and not env["nora_fork"]:
        normalize_lora_A_(peft_model)
    return peft_model


# ---------- evaluation ----------

def _clean(text):
    for s in STOPS:
        text = text.split(s)[0]
    return text.strip()


@torch.no_grad()
def generate_batch(model, tok, inputs, grid, max_new_tokens=None):
    model.eval()
    outs = []
    n = max_new_tokens or grid.max_new_tokens
    for i in range(0, len(inputs), grid.gen_batch_size):
        chunk = [PROMPT.format(x) for x in inputs[i:i + grid.gen_batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen_ids = model.generate(**enc, max_new_tokens=n, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for row, prompt_ids in zip(gen_ids, enc["input_ids"]):
            outs.append(_clean(tok.decode(row[len(prompt_ids):], skip_special_tokens=True)))
    model.train()
    return outs


def score(preds, examples):
    """Exact match, parse validity, and refusal precision/recall for '?'."""
    em = valid = 0
    tp = fp = fn = 0
    for p, ex in zip(preds, examples):
        parsed = parse_compact(p)
        valid += parsed is not None
        em += parsed is not None and parsed == parse_compact(ex["output"])
        gold_ref = ex["output"].strip() == UNCLEAR
        pred_ref = p.strip() == UNCLEAR
        tp += gold_ref and pred_ref
        fp += pred_ref and not gold_ref
        fn += gold_ref and not pred_ref
    n = len(examples)
    return {
        "em": em / n,
        "parse_validity": valid / n,
        "refusal_precision": tp / (tp + fp) if tp + fp else None,
        "refusal_recall": tp / (tp + fn) if tp + fn else None,
        "n": n,
    }


def evaluate(model, tok, examples, grid):
    preds = generate_batch(model, tok, [e["input"] for e in examples], grid)
    return score(preds, examples), preds


# ---------- curves ----------

class CurveCallback(TrainerCallback):
    """Per-step loss and grad norm; val exact match every `eval_every` steps."""

    def __init__(self, model, tok, val_examples, grid):
        self.model, self.tok, self.val, self.grid = model, tok, val_examples, grid
        self.loss, self.grad_norm, self.val_em = [], [], []

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        if "loss" in logs:
            self.loss.append([state.global_step, logs["loss"]])
        if "grad_norm" in logs:
            self.grad_norm.append([state.global_step, logs["grad_norm"]])

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.grid.eval_every == 0 or state.global_step == 1:
            m, _ = evaluate(self.model, self.tok, self.val, self.grid)
            self.val_em.append([state.global_step, m["em"]])

    def steps_to(self, threshold=0.9):
        for step, em in self.val_em:
            if em >= threshold:
                return step
        return None


# ---------- one run ----------

def run_id(method, rank, size, seed, lr):
    return f"{method}_r{rank}_n{size}_s{seed}_lr{lr:g}"


def train_one(method, rank, size, seed, lr, splits, grid, env, tok, out_dir, save_adapter=False):
    from trl import SFTConfig, SFTTrainer

    rid = run_id(method, rank, size, seed, lr)
    t0 = time.perf_counter()
    model = build_model(method, rank, seed, env)
    invariants_start = check_invariants(model, method if method.startswith("nora") else "lora")

    train = to_trl(subset(splits["train"], size, seed))
    curve_val = splits["val"][:grid.curve_val_n]
    prec = precision(env)

    args = SFTConfig(
        output_dir=os.path.join(out_dir, "tmp", rid),
        max_steps=grid.max_steps,
        per_device_train_batch_size=grid.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=grid.warmup_steps,
        logging_steps=1,
        max_length=grid.max_seq_length,
        completion_only_loss=True,
        packing=False,
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        seed=seed,
        data_seed=seed,
        save_strategy="no",
        report_to="none",
        dataloader_drop_last=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, processing_class=tok)
    curves = CurveCallback(model, tok, curve_val, grid)
    trainer.add_callback(curves)
    trainer.train()

    invariants_end = check_invariants(model, method if method.startswith("nora") else "lora")
    metrics = {}
    for split in ["val", "test", "hard", "hand"]:
        metrics[split], _ = evaluate(model, tok, splits[split], grid)

    # Save first: check_merge_equivalence unloads the adapter from `model`.
    if save_adapter:
        model.save_pretrained(os.path.join(out_dir, "adapters", rid))
    merged_ok = check_merge_equivalence(model, tok, splits["val"][:20], grid)

    use_nora, alpha_mult = METHODS[method]
    row = {
        "run_id": rid, "method": method, "rank": rank, "alpha": alpha_mult * rank,
        "size": size, "seed": seed, "lr": lr,
        "implementation": "official_fork" if env["nora_fork"] else "parametrize_fallback",
        "use_nora": use_nora, "max_steps": grid.max_steps, "batch_size": grid.batch_size,
        "metrics": metrics,
        "steps_to_90_val_em": curves.steps_to(0.9),
        "grad_norm_first_50": curves.grad_norm[:50],
        "invariants_start": invariants_start, "invariants_end": invariants_end,
        "merge_equivalent": merged_ok,
        "wall_time_s": round(time.perf_counter() - t0, 1),
        **{f"env_{k}": v for k, v in env.items()},
    }
    with open(os.path.join(out_dir, "curves", f"{rid}.json"), "w") as f:
        json.dump({"loss": curves.loss, "grad_norm": curves.grad_norm, "val_em": curves.val_em}, f)
    del model, trainer
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def check_merge_equivalence(model, tok, examples, grid):
    before = generate_batch(model, tok, [e["input"] for e in examples], grid)
    merged = model.merge_and_unload()
    after = generate_batch(merged, tok, [e["input"] for e in examples], grid)
    return before == after


# ---------- grid ----------

def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_result(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def select_lrs(grid, splits, env, tok, out_dir, results_path):
    """Per-method LR, chosen on VAL exact match only."""
    done = {(r["method"], r["lr"]): r for r in load_results(results_path) if r.get("phase") == "lr_sweep"}
    chosen = {}
    for method in grid.methods:
        scores = {}
        for lr in grid.lr_candidates:
            key = (method, lr)
            if key in done:
                scores[lr] = done[key]["metrics"]["val"]["em"]
                continue
            row = train_one(method, grid.lr_select_rank, grid.lr_select_size,
                            grid.lr_select_seed, lr, splits, grid, env, tok, out_dir)
            row["phase"] = "lr_sweep"
            append_result(results_path, row)
            scores[lr] = row["metrics"]["val"]["em"]
        chosen[method] = max(scores, key=scores.get)
        print(f"{method}: lr={chosen[method]:g} from {dict(sorted(scores.items()))}")
    return chosen


def run_grid(grid, splits, env, tok, out_dir, results_path, dry_run=False):
    done = {r["run_id"] for r in load_results(results_path) if r.get("phase") == "grid"}
    todo = [(m, r, n, s) for m in grid.methods for r in grid.ranks
            for n in grid.sizes for s in grid.seeds
            if run_id(m, r, n, s, grid.lrs[m]) not in done]
    print(f"{len(todo)} runs to go, {len(done)} already done")
    if dry_run:
        return todo
    for method, rank, size, seed in todo:
        row = train_one(method, rank, size, seed, grid.lrs[method], splits, grid, env, tok, out_dir)
        row["phase"] = "grid"
        append_result(results_path, row)
        m = row["metrics"]
        print(f"{row['run_id']}: val {m['val']['em']:.3f} test {m['test']['em']:.3f} "
              f"hard {m['hard']['em']:.3f} hand {m['hand']['em']:.3f} ({row['wall_time_s']:.0f}s)")
    return todo


# ---------- analysis ----------

def bootstrap_ci(values, iters=10000, alpha=0.05, seed=0):
    import random
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iters))
    return (means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2))])


def summarize(rows, split="hard"):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r.get("phase") != "grid":
            continue
        groups[(r["method"], r["rank"], r["size"])].append(r["metrics"][split]["em"])
    out = []
    for (method, rank, size), vals in sorted(groups.items()):
        lo, hi = bootstrap_ci(vals)
        out.append({"method": method, "rank": rank, "size": size, "n_seeds": len(vals),
                    "mean": statistics.fmean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "ci_lo": lo, "ci_hi": hi})
    return out


def paired(rows, split="hard", a="nora", b="lora"):
    """Same (rank, size, seed): mean paired difference, Cohen's dz, bootstrap CI."""
    index = {(r["method"], r["rank"], r["size"], r["seed"]): r["metrics"][split]["em"]
             for r in rows if r.get("phase") == "grid"}
    out = []
    keys = sorted({(k[1], k[2]) for k in index})
    for rank, size in keys:
        diffs = [index[(a, rank, size, s)] - index[(b, rank, size, s)]
                 for s in sorted({k[3] for k in index})
                 if (a, rank, size, s) in index and (b, rank, size, s) in index]
        if not diffs:
            continue
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        lo, hi = bootstrap_ci(diffs)
        out.append({"a": a, "b": b, "rank": rank, "size": size, "n_pairs": len(diffs),
                    "mean_diff": statistics.fmean(diffs), "sd_diff": sd,
                    "cohens_dz": (statistics.fmean(diffs) / sd) if sd else None,
                    "ci_lo": lo, "ci_hi": hi,
                    "within_noise": lo is not None and lo <= 0 <= hi})
    return out
