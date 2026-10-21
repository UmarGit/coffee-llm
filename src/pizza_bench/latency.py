"""Batch-size-1 latency on the training GPU.

Resumable: each run appends to runs/latency_gpu.jsonl as it finishes, and run ids already
present are skipped. The Mac measurement lives in scripts/pizza_latency_mac.py.
"""
import json
import os
import statistics
import time
from pathlib import Path

from . import evaluate
from .config import BASE_MODEL, RUNS, run_dir


def results_path():
    return RUNS / "latency_gpu.jsonl"


def done_run_ids():
    p = results_path()
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["run_id"])
            except Exception:
                pass
    return out


def append_result(row):
    """Append one result. Several shards write here, and a crash can leave a partial line,
    so take an exclusive lock and repair a missing newline before writing."""
    import fcntl
    import os
    p = results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with open(p, "a+") as f:
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


def load_adapter(adapter_dir, max_seq_length=640, use_unsloth=True):
    """Load a trained adapter for inference.

    The adapter is saved by Unsloth, whose module layout for this hybrid architecture does
    not match what plain PEFT reconstructs: loading it with AutoModelForCausalLM plus
    PeftModel reports missing keys for mlp.* on most layers and self_attn.* on every fourth
    layer, nothing binds, every lora_B stays zero, and the measurement silently becomes the
    base model. So load it the same way it was saved.
    """
    import torch

    from .train import normalize_tokenizer

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir), max_seq_length=max_seq_length,
                load_in_4bit=False, load_in_16bit=True)
            if hasattr(FastLanguageModel, "for_inference"):
                FastLanguageModel.for_inference(model)
            assert_adapter_active(model, adapter_dir)
            return model, normalize_tokenizer(tok)
        except ImportError:
            pass

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert_adapter_active(model, adapter_dir)
    return model, normalize_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL))


def assert_adapter_active(model, adapter_dir):
    """A trained adapter has non-zero lora_B. An untrained or unapplied one is all zeros."""
    seen = nonzero = 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            seen += 1
            if p.detach().abs().sum().item() > 0:
                nonzero += 1
    if seen == 0:
        raise RuntimeError(f"no LoRA layers found after loading {adapter_dir}")
    if nonzero == 0:
        raise RuntimeError(
            f"adapter at {adapter_dir} loaded but all {seen} lora_B tensors are zero, so "
            f"the trained weights are not active and this would time the base model. If "
            f"PEFT printed 'missing adapter keys' above, the saver and loader disagree "
            f"about module naming; load with the same framework that saved it.")
    return {"lora_layers": seen, "nonzero_lora_B": nonzero}


def gpu_context():
    """Record the device and how much of it someone else is using.

    This box is shared. Absolute latency depends on contention, so it is recorded with
    every measurement rather than assumed away.
    """
    import subprocess
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    out = {"device": torch.cuda.get_device_name(0),
           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        used, total = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()[0].split(",")
        out["gpu_memory_used_mib"] = int(used)
        out["gpu_memory_total_mib"] = int(total)
        out["gpu_memory_used_by_this_process_mib"] = round(
            torch.cuda.memory_reserved() / 1024 / 1024)
    except Exception:
        pass
    return out


def measure(model, tok, rows, arm, cap, warmup=5):
    """Batch size 1, one utterance at a time, so this is a deployment latency."""
    import torch
    for r in rows[:warmup]:
        evaluate.generate(model, tok, [r["src"]], cap, batch_size=1)
    times, counts, outs = [], [], []
    for r in rows:
        t0 = time.perf_counter()
        texts, n = evaluate.generate(model, tok, [r["src"]], cap, batch_size=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        counts.append(n[0])
        outs.append(texts[0])
    s = sorted(times)
    hit_cap = sum(c >= cap for c in counts) / len(counts)
    return {
        "arm": arm, "n": len(rows), "batch_size": 1, "max_new_tokens": cap,
        "median_ms": statistics.median(s), "p90_ms": s[int(len(s) * 0.9)],
        "mean_ms": statistics.fmean(s),
        "mean_generated_tokens": statistics.fmean(counts),
        "hit_cap_rate": hit_cap,
        "tokens_per_second": statistics.fmean(counts) / (statistics.fmean(s) / 1000),
    }, outs, counts


def run(cfg, splits, arm, run_ids, limit=200, split="test", use_unsloth=True):
    import torch
    from . import metrics

    rows = splits[split][:limit]
    cap = cfg.caps[arm]
    already = done_run_ids()
    out = []
    for rid in run_ids:
        if rid in already:
            print(f"  skip {rid}: already measured")
            continue
        adapter = run_dir(rid) / "adapter"
        if not adapter.is_dir():
            print(f"  skip {rid}: no adapter saved")
            continue
        model, tok = load_adapter(adapter, max_seq_length=cfg.max_seq_length,
                                  use_unsloth=use_unsloth)
        res, outs, _ = measure(model, tok, rows, arm, cap)
        acc, _, _ = metrics.evaluate(outs, [r["exr"] for r in rows], arm)
        res.update(run_id=rid, split=split, em=acc["em"],
                   parse_validity=acc["parse_validity"], **gpu_context())
        if res["hit_cap_rate"] > 0.5:
            raise RuntimeError(
                f"{rid}: {res['hit_cap_rate']:.0%} of generations reached the {cap}-token "
                f"cap and validity is {acc['parse_validity']:.2f}. That is the signature of "
                f"an inactive adapter, not a slow model. Refusing to record this.")
        append_result(res)
        out.append(res)
        print(f"  {rid}: median {res['median_ms']:.0f} ms, "
              f"{res['mean_generated_tokens']:.1f} tokens, EM {acc['em']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return out
