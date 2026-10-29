"""LoRA SFT with dev-based early stopping. One run writes one self-contained runs/<id>/."""

import json
import os
import time

from . import evaluate
from .config import describe_env, is_complete, run_dir, run_id
from .config import model as active_model
from .config import task as active_task

ALLOW_CPU = (
    os.environ.get("BENCH_ALLOW_CPU", os.environ.get("PIZZA_ALLOW_CPU", "")) == "1"
)


def assert_single_gpu():
    import torch

    n = torch.cuda.device_count()
    if n == 0 and not ALLOW_CPU:
        raise RuntimeError(
            "No CUDA device visible. CUDA_VISIBLE_DEVICES may point at a GPU that does not\n"
            "exist: an out-of-range index yields zero devices rather than an error, and\n"
            "training then silently falls back to CPU. Check `nvidia-smi -L`.\n"
            "Set BENCH_ALLOW_CPU=1 only if you mean it."
        )
    if n > 1:
        raise RuntimeError(
            f"{n} GPUs visible. Trainer would wrap the model in DataParallel and split\n"
            f"the batch, which breaks the fixed batch size. Pin one GPU before torch\n"
            f"initialises CUDA:\n\n"
            f"    CUDA_VISIBLE_DEVICES=0 python -m bench <stage> ...\n"
            f"    make smoke GPU=0\n\n"
            f"or let the runner pin one per shard and use all {n}:\n\n"
            f"    ./scripts/tmux.sh all <model> <task> {n}\n"
        )


def normalize_tokenizer(tok):
    """Unsloth returns a Processor for some models, which has no tokenizer API.

    A processor wraps the real tokenizer at `.tokenizer`. Unwrap it, then assert the
    stopping contract up front: EOS must resolve to exactly one id, or generation would
    never stop and every run would silently hit max_new_tokens.
    """
    eos = active_model().eos
    tok = getattr(tok, "tokenizer", tok)
    missing = [
        m
        for m in ("convert_tokens_to_ids", "decode", "__call__")
        if not hasattr(tok, m)
    ]
    if missing:
        raise TypeError(
            f"{type(tok).__name__} is not a usable tokenizer, missing {missing}"
        )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.convert_tokens_to_ids(eos)
    if eos_id is None or eos_id == getattr(tok, "unk_token_id", object()):
        raise RuntimeError(
            f"tokenizer has no id for {eos}; generation would never stop"
        )
    ids = tok(eos, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{eos} tokenizes to {len(ids)} tokens, expected 1")
    return tok


def build(cfg, seed, use_unsloth=True):
    """Loads the selected model. A model the registry marks as unsupported by
    Unsloth is loaded through plain transformers and PEFT regardless of the flag,
    because silently loading the wrong thing is worse than being slower."""
    import torch
    from transformers import AutoTokenizer, set_seed

    m = active_model()
    set_seed(seed)
    if use_unsloth and m.supports_unsloth:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            model_name=m.hf_id,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias="none",
            target_modules=list(m.target_modules),
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(m.hf_id, dtype=dtype)
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.rank,
                lora_alpha=cfg.alpha,
                lora_dropout=cfg.dropout,
                target_modules=(
                    m.target_modules
                    if isinstance(m.target_modules, str)
                    else list(m.target_modules)
                ),
                task_type="CAUSAL_LM",
            ),
        )
        tok = AutoTokenizer.from_pretrained(m.hf_id)
    return model, normalize_tokenizer(tok)


class EarlyStop:
    def __init__(self, model, tok, rows, arm, cap, cfg, neg_rows=None):
        self.model, self.tok, self.rows = model, tok, rows
        self.arm, self.cap, self.cfg = arm, cap, cfg
        self.neg_rows = neg_rows
        self.curve, self.best, self.state, self.stale = [], (-1.0, 0), None, 0

    def step(self, step):
        if step % self.cfg.eval_every and step != self.cfg.max_steps:
            return False
        res, *_ = evaluate.evaluate_split(
            self.model, self.tok, self.rows, self.arm, self.cap, self.cfg.gen_batch_size
        )
        point = {"step": step, "dev_em": res["em"], "dev_validity": res["parse_validity"]}
        score = res["em"]
        if self.neg_rows:
            # With negatives, a checkpoint must parse orders AND refuse everything
            # else; selecting on dev EM alone stops before refusal is learned.
            t = active_task()
            outs, _ = evaluate.generate(
                self.model, self.tok, [t.source(r) for r in self.neg_rows], self.cap,
                self.cfg.gen_batch_size,
            )
            handled = t.negatives.score(outs, self.neg_rows, self.arm)["handled"]
            point["dev_neg_handled"] = handled
            score = (res["em"] + handled) / 2
        point["selection_score"] = score
        self.curve.append(point)
        if score > self.best[0]:
            self.best = (score, step)
            self.state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
                if "lora_" in k
            }
            self.stale = 0
        else:
            self.stale += 1
        return step >= self.cfg.min_steps and self.stale >= self.cfg.patience

    def restore(self):
        if self.state:
            self.model.load_state_dict(self.state, strict=False)


def train_one(
    stage, arm, size, seed, lr, splits, cfg, use_unsloth=True, save_adapter=True, tag=""
):
    rid = run_id(arm, size, seed, lr, tag)
    out = run_dir(rid)
    if is_complete(rid):
        return json.loads((out / "metrics.json").read_text())

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    assert_single_gpu()
    cap = cfg.caps[arm]
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "stage": stage,
                "tag": tag,
                "arm": arm,
                "size": size,
                "seed": seed,
                "lr": lr,
                "cap": cap,
                **json.loads(cfg.to_json()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    env = describe_env()
    (out / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True))

    model, tok = build(cfg, seed, use_unsloth)
    t = active_task()
    eos = active_model().eos
    fmt = evaluate.prompt_for(tok)
    train_rows = t.train_rows(size, seed)
    if cfg.negatives:
        if t.negatives is None:
            raise ValueError(f"task {t.name} has no negative data")
        # Added on top of the task's rows, so `size` still counts real orders.
        train_rows = t.negatives.mix(train_rows, cfg.negatives, seed, arm)
    ds = Dataset.from_list(
        [
            {"prompt": fmt.format(t.source(r)), "completion": t.target(r, arm) + eos}
            for r in train_rows
        ]
    )

    neg_dev = t.negatives.dev_rows(splits["dev"], seed=seed) if cfg.negatives else None
    stopper = EarlyStop(model, tok, splits["dev"], arm, cap, cfg, neg_dev)
    losses = []

    class Hook(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stopper.step(state.global_step):
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append({"step": state.global_step, "loss": logs["loss"]})

    args = SFTConfig(
        output_dir=str(out / "tmp"),
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
        bf16=bool(env.get("bf16")),
        fp16=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.add_callback(Hook())
    trainer.train()
    stopper.restore()

    metrics_out = {
        "run_id": rid,
        "stage": stage,
        "tag": tag,
        "arm": arm,
        "size": size,
        "seed": seed,
        "lr": lr,
        "cap": cap,
        "best_step": stopper.best[1],
        # The selection score is dev EM, or with negatives the mean of dev EM and
        # negatives handled; dev EM itself is recorded either way.
        "best_dev_em_during_training": next(
            (c["dev_em"] for c in stopper.curve if c["step"] == stopper.best[1]), stopper.best[0]
        ),
        "best_selection_score": stopper.best[0],
        "pool_size": size,
        "examples_seen": stopper.best[1] * cfg.batch_size,
        "samples_seen_max": cfg.max_steps * cfg.batch_size,
        "epochs_at_best": stopper.best[1] * cfg.batch_size / size,
        "pool_coverage": min(1.0, cfg.max_steps * cfg.batch_size / size),
    }
    test_outputs = []
    for split in ("dev", "test"):
        res, outputs, preds, em, counts = evaluate.evaluate_split(
            model, tok, splits[split], arm, cap, cfg.gen_batch_size
        )
        metrics_out[split] = res
        if split == "test":
            test_outputs = outputs
        evaluate.write_predictions(
            out / f"{split}_predictions.jsonl",
            splits[split],
            outputs,
            preds,
            em,
            counts,
        )
        if res.get("hit_cap_rate", 0) > cfg.truncation_warn:
            print(
                f"  WARNING {rid} {split}: {res['hit_cap_rate']:.1%} of generations hit "
                f"the {cap}-token cap",
                flush=True,
            )

    if cfg.negatives:
        # Held-out vocabulary only, so this measures whether refusal generalises,
        # plus the fixed challenge set of things people type into a demo.
        neg = t.negatives
        neg_rows = neg.eval_rows(splits["test"], seed=seed) + neg.probe_rows()
        neg_out, _ = evaluate.generate(
            model, tok, [t.source(r) for r in neg_rows], cap, cfg.gen_batch_size
        )
        metrics_out["negatives"] = neg.score(neg_out, neg_rows, arm, test_outputs)
        with open(out / "negatives_predictions.jsonl", "w") as f:
            for r, o in zip(neg_rows, neg_out):
                f.write(json.dumps({"src": r["src"], "kind": r["neg"], "raw": o,
                                    "correct": neg.correct(o, r, arm)}) + "\n")
        print(f"  negatives {rid}: {json.dumps(metrics_out['negatives'])}", flush=True)

    (out / "curves.json").write_text(
        json.dumps({"dev": stopper.curve, "loss": losses}, indent=2)
    )
    if save_adapter:
        model.save_pretrained(str(out / "adapter"))
    metrics_out["wall_time_s"] = round(time.perf_counter() - t0, 1)
    metrics_out["complete"] = True
    (out / "metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True))

    del model, trainer
    import torch

    torch.cuda.empty_cache()
    return metrics_out
