"""Generation and scoring.

Scoring is separable from generation so every reported number can be recomputed
from saved predictions without a GPU, without the dataset and without the model.
Neither half knows which task is running: both go through the task registry.
"""

import json

from . import models
from .config import model as active_model
from .config import task as active_task

_prompt_cache = {}


def prompt_for(tok):
    """The active model's prompt format string, resolved once per tokenizer.

    A model that pins its prompt returns it verbatim; one that does not has it
    derived from the tokenizer's own chat template, so it can never drift from
    what the tokenizer would actually produce.
    """
    m = active_model()
    key = (m.name, id(tok))
    if key not in _prompt_cache:
        _prompt_cache[key] = models.resolve_prompt(m, tok)
    return _prompt_cache[key]


def strip_eos(text):
    return text.split(active_model().eos)[0].strip()


def generate(net, tok, sources, cap, batch_size=32):
    """Greedy, stopping at the model's EOS. Returns (texts, n_generated_tokens)."""
    fmt = prompt_for(tok)
    return generate_from_prompts(
        net, tok, [fmt.format(s) for s in sources], cap, batch_size
    )


def generate_from_prompts(net, tok, prompts, cap, batch_size=32):
    """Same, but the caller supplies fully rendered prompts (the few-shot stage)."""
    import torch

    eos = active_model().eos
    eos_id = tok.convert_tokens_to_ids(eos)
    if eos_id is None or eos_id == tok.unk_token_id:
        raise RuntimeError(f"tokenizer has no id for {eos}; stopping would never fire")
    texts, counts = [], []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(net.device)
            out = net.generate(
                **enc,
                max_new_tokens=cap,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=tok.pad_token_id,
            )
            for row, prompt_ids in zip(out, enc["input_ids"]):
                gen = row[len(prompt_ids) :]
                keep = [t for t in gen.tolist() if t != tok.pad_token_id]
                counts.append(len(keep))
                texts.append(strip_eos(tok.decode(gen, skip_special_tokens=True)))
    net.train()
    return texts, counts


def write_predictions(path, rows, outputs, preds, correct, n_tokens):
    t = active_task()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, (r, o, p, c, n) in enumerate(
            zip(rows, outputs, preds, correct, n_tokens)
        ):
            rec = {
                "id": i,
                "src": t.source(r),
                "gold": t.gold(r),
                "raw": o,
                "parsed": p,
                "correct": bool(c),
                "n_generated_tokens": n,
            }
            rec.update(t.pred_extra(r))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_predictions(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def score_file(path, arm):
    """Recompute every metric from a saved prediction file. No GPU, no model."""
    t = active_task()
    pred_rows = read_predictions(path)
    rows = t.rows_from_predictions(pred_rows)
    res, _, _ = t.score(
        [r["raw"] for r in pred_rows],
        rows,
        arm,
        n_tokens=[r["n_generated_tokens"] for r in pred_rows],
    )
    return res, pred_rows


def evaluate_split(net, tok, rows, arm, cap, batch_size=32):
    t = active_task()
    outputs, counts = generate(net, tok, [t.source(r) for r in rows], cap, batch_size)
    res, preds, correct = t.score(outputs, rows, arm, n_tokens=counts, cap=cap)
    return res, outputs, preds, correct, counts
