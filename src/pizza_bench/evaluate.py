"""Generation and scoring. Scoring is separable from generation so every reported number
can be recomputed from saved predictions without a GPU."""
import json

from . import metrics
from .config import EOS, PROMPT


def strip_eos(text):
    return text.split(EOS)[0].strip()


def generate(model, tok, sources, cap, batch_size=32):
    """Greedy, stopping at EOS. Returns (texts, n_generated_tokens)."""
    return generate_from_prompts(model, tok, [PROMPT.format(s) for s in sources],
                                 cap, batch_size)


def generate_from_prompts(model, tok, prompts, cap, batch_size=32):
    """Same, but the caller supplies fully rendered prompts (used by the few-shot stage)."""
    import torch

    eos_id = tok.convert_tokens_to_ids(EOS)
    if eos_id is None or eos_id == tok.unk_token_id:
        raise RuntimeError(f"tokenizer has no id for {EOS}; stopping would never fire")
    texts, counts = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=cap, do_sample=False,
                                 eos_token_id=eos_id, pad_token_id=tok.pad_token_id)
            for row, prompt_ids in zip(out, enc["input_ids"]):
                gen = row[len(prompt_ids):]
                keep = [t for t in gen.tolist() if t != tok.pad_token_id]
                counts.append(len(keep))
                texts.append(strip_eos(tok.decode(gen, skip_special_tokens=True)))
    model.train()
    return texts, counts


def write_predictions(path, rows, outputs, preds, correct, n_tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, (r, o, p, c, t) in enumerate(zip(rows, outputs, preds, correct, n_tokens)):
            f.write(json.dumps({
                "id": i, "src": r["src"], "gold": r["exr"], "raw": o,
                "parsed_exr": p, "correct": bool(c), "n_generated_tokens": t,
                "pcfg_err": r["pcfg_err"],
            }, ensure_ascii=False) + "\n")


def read_predictions(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def score_file(path, arm):
    """Recompute every metric from a saved prediction file. No GPU, no model."""
    rows = read_predictions(path)
    res, _, _ = metrics.evaluate(
        [r["raw"] for r in rows], [r["gold"] for r in rows], arm,
        pcfg_err=[r["pcfg_err"] for r in rows],
        n_tokens=[r["n_generated_tokens"] for r in rows])
    return res, rows


def evaluate_split(model, tok, rows, arm, cap, batch_size=32):
    outputs, counts = generate(model, tok, [r["src"] for r in rows], cap, batch_size)
    res, preds, em = metrics.evaluate(outputs, [r["exr"] for r in rows], arm,
                                      pcfg_err=[r["pcfg_err"] for r in rows],
                                      n_tokens=counts, cap=cap)
    return res, outputs, preds, em, counts
