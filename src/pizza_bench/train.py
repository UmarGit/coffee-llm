"""LoRA SFT with dev-based early stopping. One run writes one self-contained runs/<id>/."""
import json
import os
import time

from . import data, evaluate, formats
from .config import (
    BASE_MODEL, EOS, PROMPT, TARGET_MODULES, describe_env, is_complete, run_dir, run_id,
)

ALLOW_CPU = os.environ.get("PIZZA_ALLOW_CPU") == "1"


def assert_single_gpu():
    import torch
    n = torch.cuda.device_count()
    if n == 0 and not ALLOW_CPU:
        raise RuntimeError(
            "No CUDA device visible. CUDA_VISIBLE_DEVICES may point at a GPU that does not\n"
            "exist: an out-of-range index yields zero devices rather than an error, and\n"
            "training then silently falls back to CPU. Check `nvidia-smi -L`.\n"
            "Set PIZZA_ALLOW_CPU=1 only if you mean it.")
    if n > 1:
        raise RuntimeError(
            f"{n} GPUs visible. Trainer would wrap the model in DataParallel and split the\n"
            f"batch, which breaks the fixed batch size. Pin one GPU before torch initialises\n"
            f"CUDA and shard with Config(shard=(i, {n})).")


def normalize_tokenizer(tok):
    """Unsloth returns a Processor for some models, which has no tokenizer API.

    A processor wraps the real tokenizer at `.tokenizer`. Unwrap it, then assert the
    stopping contract up front: EOS must resolve to exactly one id, or generation would
    never stop and every run would silently hit max_new_tokens.
    """
    tok = getattr(tok, "tokenizer", tok)
    missing = [m for m in ("convert_tokens_to_ids", "decode", "__call__")
               if not hasattr(tok, m)]
    if missing:
        raise TypeError(f"{type(tok).__name__} is not a usable tokenizer, missing {missing}")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.convert_tokens_to_ids(EOS)
    if eos_id is None or eos_id == getattr(tok, "unk_token_id", object()):
        raise RuntimeError(f"tokenizer has no id for {EOS}; generation would never stop")
    ids = tok(EOS, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{EOS} tokenizes to {len(ids)} tokens, expected 1")
    return tok


def build(cfg, seed, use_unsloth=True):
    import torch
    from transformers import AutoTokenizer, set_seed
    set_seed(seed)
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=cfg.max_seq_length,
            load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, bias="none",
            target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth",
            random_state=seed)
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
            target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    return model, normalize_tokenizer(tok)


class EarlyStop:
    def __init__(self, model, tok, rows, arm, cap, cfg):
        self.model, self.tok, self.rows = model, tok, rows
        self.arm, self.cap, self.cfg = arm, cap, cfg
        self.curve, self.best, self.state, self.stale = [], (-1.0, 0), None, 0

    def step(self, step):
        if step % self.cfg.eval_every and step != self.cfg.max_steps:
            return False
        res, *_ = evaluate.evaluate_split(self.model, self.tok, self.rows, self.arm,
                                          self.cap, self.cfg.gen_batch_size)
        self.curve.append({"step": step, "dev_em": res["em"],
                           "dev_validity": res["parse_validity"]})
        if res["em"] > self.best[0]:
            self.best = (res["em"], step)
            self.state = {k: v.detach().cpu().clone()
                          for k, v in self.model.state_dict().items() if "lora_" in k}
            self.stale = 0
        else:
            self.stale += 1
        return step >= self.cfg.min_steps and self.stale >= self.cfg.patience

    def restore(self):
        if self.state:
            self.model.load_state_dict(self.state, strict=False)


def train_one(stage, arm, size, seed, lr, splits, cfg, use_unsloth=True,
              save_adapter=True, tag=""):
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
    (out / "config.json").write_text(json.dumps(
        {"run_id": rid, "stage": stage, "tag": tag, "arm": arm, "size": size, "seed": seed,
         "lr": lr, "cap": cap, **json.loads(cfg.to_json())}, indent=2, sort_keys=True))
    env = describe_env()
    (out / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True))

    model, tok = build(cfg, seed, use_unsloth)
    train_rows = data.train_subset(size, seed, pool=cfg.pool)
    ds = Dataset.from_list([
        {"prompt": PROMPT.format(r["src"]),
         "completion": formats.encode(formats.parse_exr(r["exr"]), arm) + EOS}
        for r in train_rows])

    stopper = EarlyStop(model, tok, splits["dev"], arm, cap, cfg)
    losses = []

    class Hook(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stopper.step(state.global_step):
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append({"step": state.global_step, "loss": logs["loss"]})

    args = SFTConfig(
        output_dir=str(out / "tmp"), max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=1,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=cfg.warmup_steps,
        logging_steps=10, max_length=cfg.max_seq_length, completion_only_loss=True,
        packing=False, seed=seed, data_seed=seed, save_strategy="no", report_to="none",
        bf16=bool(env.get("bf16")), fp16=False)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.add_callback(Hook())
    trainer.train()
    stopper.restore()

    metrics_out = {"run_id": rid, "stage": stage, "tag": tag, "arm": arm, "size": size, "seed": seed,
                   "lr": lr, "cap": cap, "best_step": stopper.best[1],
                   "best_dev_em_during_training": stopper.best[0],
                   "pool_size": size,
                   "examples_seen": stopper.best[1] * cfg.batch_size,
                   "samples_seen_max": cfg.max_steps * cfg.batch_size,
                   "epochs_at_best": stopper.best[1] * cfg.batch_size / size,
                   "pool_coverage": min(1.0, cfg.max_steps * cfg.batch_size / size)}
    for split in ("dev", "test"):
        res, outputs, preds, em, counts = evaluate.evaluate_split(
            model, tok, splits[split], arm, cap, cfg.gen_batch_size)
        metrics_out[split] = res
        evaluate.write_predictions(out / f"{split}_predictions.jsonl", splits[split],
                                   outputs, preds, em, counts)
        if res.get("hit_cap_rate", 0) > cfg.truncation_warn:
            print(f"  WARNING {rid} {split}: {res['hit_cap_rate']:.1%} of generations hit "
                  f"the {cap}-token cap", flush=True)

    (out / "curves.json").write_text(json.dumps(
        {"dev": stopper.curve, "loss": losses}, indent=2))
    if save_adapter:
        model.save_pretrained(str(out / "adapter"))
    metrics_out["wall_time_s"] = round(time.perf_counter() - t0, 1)
    metrics_out["complete"] = True
    (out / "metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True))

    del model, trainer
    import torch
    torch.cuda.empty_cache()
    return metrics_out
