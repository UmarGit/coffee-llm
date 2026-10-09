"""Training and evaluation harness for the PIZZA conditions.

One process per GPU. Needs torch, so it is not imported by pizza_bench/__init__.
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field

import torch

from . import data, formats, metrics

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
FORMATS = ("exr", "json", "dsl")
MAX_NEW = {"exr": 128, "json": 128, "dsl": 48}
ALLOW_CPU = os.environ.get("PIZZA_ALLOW_CPU") == "1"


@dataclass
class Config:
    formats: tuple = FORMATS
    sizes: tuple = (1000, 5000, 20000)
    seeds: tuple = (0, 1, 2)
    shard: tuple = (0, 1)
    rank: int = 16
    alpha: int = 16
    max_steps: int = 2000
    batch_size: int = 16
    warmup_steps: int = 50
    max_seq_length: int = 512
    eval_every: int = 100
    early_stop_patience: int = 5
    dev_curve_n: int = 348
    gen_batch_size: int = 64
    lr_candidates: tuple = (1e-4, 2e-4, 5e-4)
    lr_select_size: int = 5000
    lr_select_seed: int = 0
    lrs: dict = field(default_factory=dict)


def assert_single_gpu():
    n = torch.cuda.device_count()
    if n == 0 and not ALLOW_CPU:
        raise RuntimeError(
            "No CUDA device visible, so this would train on CPU. CUDA_VISIBLE_DEVICES may point\n"
            "at a GPU that does not exist (an out-of-range index yields zero devices, not an\n"
            "error). Check `nvidia-smi -L`. Set PIZZA_ALLOW_CPU=1 to override."
        )
    if n > 1:
        raise RuntimeError(
            f"{n} GPUs visible; Trainer would use DataParallel and split the batch.\n"
            "Set CUDA_VISIBLE_DEVICES to one GPU before torch initialises CUDA, and shard with\n"
            f"Config(shard=(i, {n}))."
        )


def describe_env():
    import datasets, peft, transformers, trl

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = os.environ.get("COFFEE_LLM_COMMIT", "unknown")
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_visible_gpus": torch.cuda.device_count(),
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
        "python": platform.python_version(),
        "git_commit": commit,
    }


def encode_row(row, fmt):
    return formats.encode(formats.parse_exr(row["exr"]), fmt)


def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg, seed, use_unsloth=True):
    from transformers import set_seed

    set_seed(seed)
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model, tok
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    return model, tokenizer()


@torch.no_grad()
def generate(model, tok, sources, fmt, cfg):
    model.eval()
    out = []
    for i in range(0, len(sources), cfg.gen_batch_size):
        prompts = [data.PROMPT.format(s) for s in sources[i : i + cfg.gen_batch_size]]
        enc = tok(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW[fmt],
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        for row, prompt_ids in zip(ids, enc["input_ids"]):
            text = tok.decode(row[len(prompt_ids) :], skip_special_tokens=True)
            out.append(text.split("<|im_end|>")[0].strip())
    model.train()
    return out


def evaluate_split(model, tok, rows, fmt, cfg):
    outputs = generate(model, tok, [r["src"] for r in rows], fmt, cfg)
    result, preds = metrics.evaluate(
        outputs, [r["exr"] for r in rows], fmt, [r["pcfg_err"] for r in rows]
    )
    truncated = sum(
        len(tok(o, add_special_tokens=False)["input_ids"]) >= MAX_NEW[fmt] - 1
        for o in outputs
    )
    result["truncated"] = truncated / len(rows)
    return result, outputs, preds


class DevSelect:
    """Evaluate dev every eval_every steps, keep the best adapter in memory."""

    def __init__(self, model, tok, rows, fmt, cfg):
        self.model, self.tok, self.rows, self.fmt, self.cfg = model, tok, rows, fmt, cfg
        self.curve = []
        self.best = (-1.0, 0)
        self.best_state = None
        self.since_best = 0

    def maybe_eval(self, step):
        if step % self.cfg.eval_every and step != self.cfg.max_steps:
            return False
        res, _, _ = evaluate_split(self.model, self.tok, self.rows, self.fmt, self.cfg)
        self.curve.append([step, res["em"], res["parse_validity"]])
        if res["em"] > self.best[0]:
            self.best = (res["em"], step)
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
                if "lora_" in k
            }
            self.since_best = 0
        else:
            self.since_best += 1
        return self.since_best >= self.cfg.early_stop_patience

    def restore_best(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state, strict=False)


def run_id(fmt, size, seed, lr):
    return f"{fmt}_n{size}_s{seed}_lr{lr:g}"


def load_results(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"skipping malformed results line: {line[:80]}...")
    return rows


def append_result(path, row):
    import fcntl

    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell():
                f.seek(f.tell() - 1)
                if f.read(1) != "\n":
                    f.write("\n")
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def train_one(
    fmt,
    size,
    seed,
    lr,
    splits,
    cfg,
    env,
    out_dir,
    eval_test=False,
    use_unsloth=True,
    save_adapter=False,
):
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    assert_single_gpu()
    rid = run_id(fmt, size, seed, lr)
    t0 = time.perf_counter()
    model, tok = build_model(cfg, seed, use_unsloth)

    train_rows = data.subset(splits["train"], size, seed)
    train_ds = Dataset.from_list(
        [
            {
                "prompt": data.PROMPT.format(r["src"]),
                "completion": encode_row(r, fmt) + "<|im_end|>",
            }
            for r in train_rows
        ]
    )

    selector = DevSelect(model, tok, splits["dev"], fmt, cfg)

    class Hook(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if selector.maybe_eval(state.global_step):
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append([state.global_step, logs["loss"]])

    losses = []
    prec = {"bf16": env["bf16"], "fp16": False}
    args = SFTConfig(
        output_dir=os.path.join(out_dir, "tmp", rid),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        logging_steps=10,
        max_length=cfg.max_seq_length,
        completion_only_loss=True,
        packing=False,
        seed=seed,
        data_seed=seed,
        save_strategy="no",
        report_to="none",
        **prec,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds, processing_class=tok
    )
    trainer.add_callback(Hook())
    trainer.train()
    selector.restore_best()

    row = {
        "run_id": rid,
        "condition": "B_sft",
        "format": fmt,
        "size": size,
        "seed": seed,
        "lr": lr,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "max_steps": cfg.max_steps,
        "best_dev_em": selector.best[0],
        "best_step": selector.best[1],
        "dev_curve": selector.curve,
        "loss_curve": losses[:200],
        "wall_time_s": round(time.perf_counter() - t0, 1),
        **{f"env_{k}": v for k, v in env.items()},
    }

    dev_res, dev_out, _ = evaluate_split(model, tok, splits["dev"], fmt, cfg)
    row["dev"] = dev_res
    _dump(out_dir, rid, "dev", splits["dev"], dev_out)
    if eval_test:
        test_res, test_out, _ = evaluate_split(model, tok, splits["test"], fmt, cfg)
        row["test"] = test_res
        _dump(out_dir, rid, "test", splits["test"], test_out)
    if save_adapter:
        model.save_pretrained(os.path.join(out_dir, "adapters", rid))

    del model, trainer
    torch.cuda.empty_cache()
    return row


def _dump(out_dir, rid, split, rows, outputs):
    path = os.path.join(out_dir, "predictions", f"{rid}.{split}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r, o in zip(rows, outputs):
            f.write(
                json.dumps(
                    {
                        "src": r["src"],
                        "gold": r["exr"],
                        "pred": o,
                        "pcfg_err": r["pcfg_err"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def select_lrs(cfg, splits, env, out_dir, results_path, use_unsloth=True):
    done = {
        (r["format"], r["lr"]): r
        for r in load_results(results_path)
        if r.get("condition") == "A_lr_sweep"
    }
    chosen = {}
    for fmt in cfg.formats:
        scores = {}
        for lr in cfg.lr_candidates:
            if (fmt, lr) in done:
                scores[lr] = done[(fmt, lr)]["best_dev_em"]
                continue
            row = train_one(
                fmt,
                cfg.lr_select_size,
                cfg.lr_select_seed,
                lr,
                splits,
                cfg,
                env,
                out_dir,
                use_unsloth=use_unsloth,
            )
            row["condition"] = "A_lr_sweep"
            append_result(results_path, row)
            scores[lr] = row["best_dev_em"]
        chosen[fmt] = max(scores, key=scores.get)
        print(
            f"{fmt}: lr={chosen[fmt]:g} from {dict(sorted(scores.items()))}", flush=True
        )
    return chosen


def run_grid(cfg, splits, env, out_dir, results_path, dry_run=False, use_unsloth=True):
    done = {
        r["run_id"] for r in load_results(results_path) if r.get("condition") == "B_sft"
    }
    jobs = [(f, n, s) for f in cfg.formats for n in cfg.sizes for s in cfg.seeds]
    i, total = cfg.shard
    todo = [
        j
        for k, j in enumerate(jobs)
        if k % total == i and run_id(j[0], j[1], j[2], cfg.lrs[j[0]]) not in done
    ]
    label = f" (shard {i + 1}/{total})" if total > 1 else ""
    print(f"{len(todo)} runs to go{label}, {len(done)} already done", flush=True)
    if dry_run:
        return todo
    for fmt, size, seed in todo:
        row = train_one(
            fmt,
            size,
            seed,
            cfg.lrs[fmt],
            splits,
            cfg,
            env,
            out_dir,
            use_unsloth=use_unsloth,
        )
        row["condition"] = "B_sft"
        append_result(results_path, row)
        print(
            f"{row['run_id']}: dev EM {row['dev']['em']:.4f} "
            f"valid {row['dev']['parse_validity']:.4f} ({row['wall_time_s']:.0f}s)",
            flush=True,
        )
    return todo


FEWSHOT_CRITERIA = (
    "multi-suborder pizza+drink",
    "negated topping",
    "complex topping",
    "style",
    "drink with volume and container",
)


def pick_fewshot(train_rows, n=5, seed=0):
    """Five fixed train examples covering the phenomena; identical across formats."""
    wanted = [
        lambda e: "(DRINKORDER" in e and "(PIZZAORDER" in e,
        lambda e: "(NOT (TOPPING" in e,
        lambda e: "(COMPLEX_TOPPING" in e,
        lambda e: "(STYLE" in e,
        lambda e: "(VOLUME" in e and "(CONTAINERTYPE" in e,
    ][:n]
    pool = data.subset(train_rows, min(20000, len(train_rows)), seed)
    picked, used = [], set()
    for want in wanted:
        for i, r in enumerate(pool):
            if i not in used and want(r["exr"]):
                picked.append(r)
                used.add(i)
                break
    return picked


def fewshot_prompt(shots, fmt, src):
    parts = []
    for r in shots:
        parts.append(f"{r['src']}\n{encode_row(r, fmt)}")
    parts.append(f"{src}\n")
    return data.PROMPT.format("\n\n".join(parts))


def run_fewshot(
    cfg, splits, env, out_dir, results_path, fmt, split="dev", use_unsloth=False
):
    assert_single_gpu()
    t0 = time.perf_counter()
    model, tok = build_model(cfg, 0, use_unsloth)
    shots = pick_fewshot(splits["train"])
    rows = splits[split]
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), cfg.gen_batch_size):
            chunk = rows[i : i + cfg.gen_batch_size]
            prompts = [fewshot_prompt(shots, fmt, r["src"]) for r in chunk]
            enc = tok(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW[fmt],
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            for r, p in zip(ids, enc["input_ids"]):
                text = tok.decode(r[len(p) :], skip_special_tokens=True)
                outputs.append(text.split("<|im_end|>")[0].split("\n\n")[0].strip())
    res, _ = metrics.evaluate(
        outputs, [r["exr"] for r in rows], fmt, [r["pcfg_err"] for r in rows]
    )
    rid = f"fewshot_{fmt}_{split}"
    _dump(out_dir, rid, split, rows, outputs)
    row = {
        "run_id": rid,
        "condition": "A_fewshot",
        "format": fmt,
        "split": split,
        "n_shots": len(shots),
        split: res,
        "wall_time_s": round(time.perf_counter() - t0, 1),
        **{f"env_{k}": v for k, v in env.items()},
    }
    append_result(results_path, row)
    del model
    torch.cuda.empty_cache()
    return row
